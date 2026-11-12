"""Replay of loop speculative decoding with suffix-tree drafts and shallow-depth readouts.

A looped LM runs one shared block T times per token. A draft node that starts at
block call s has its depth-d readout at s + d and its full-depth readout at s + T.
One decoding step starts from the pending token (the last committed token, not yet
run through the loop):

1. Suffix-tree drafts from the committed context start at call 0.
2. When a node reaches depth d, its shallow top-1 is added as a child if absent
   (a fill, starting at s + d), and the suffix tree continues from the fill. With
   pruning on, children outside the node's shallow top-m are removed at s + d.
3. Full-depth verification accepts the longest path whose tokens equal the greedy
   output, plus one bonus token. The step ends when the last accepted node reaches
   depth T (or, at the end of the output, when its last token was verified).

Only nodes on the greedy path change acceptance or latency, so teacher-forced
readouts along the greedy output give exact block-call counts. Row counts are
approximate: off-path suffix nodes run until pruned by an on-path parent or until
the step ends, and fills under off-path nodes are not modelled.

Arms are switches of one procedure: AR (no suffix, no fill), shallow chain (fill
only), suffix only (the original SuffixDecoding with full-depth verification) and
suffix + shallow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ArmConfig:
    name: str
    depth_total: int  # T: block calls for a full-depth readout
    draft_depth: int  # d: depth of the shallow readout
    max_draft: int  # K: maximum draft depth below the pending token
    use_suffix: bool
    use_fill: bool
    prune_top: int | None = None  # m: keep children inside the shallow top-m
    spec_factor: float = 1.0  # alpha in MAX_SPEC = alpha * match_length
    min_token_prob: float = 0.1
    tree_spec: bool = True


class SuffixCache(Protocol):
    def start_request(self, req_id, prompt_token_ids) -> None: ...

    def add_active_response(self, req_id, token_ids) -> None: ...

    def stop_request(self, req_id) -> None: ...

    def speculate(
        self,
        req_id,
        context,
        max_spec_tokens=None,
        max_spec_factor=1.0,
        max_spec_offset=0.0,
        min_token_prob=0.1,
        use_tree_spec=False,
    ): ...


@dataclass
class _Node:
    token: int
    start: int
    depth: int
    source: str
    children: dict[int, _Node] = field(default_factory=dict)
    killed: int | None = None


def _kill(node: _Node, time: int) -> None:
    stack = [node]
    while stack:
        n = stack.pop()
        n.killed = time if n.killed is None else min(n.killed, time)
        stack.extend(n.children.values())


def _attach(parent: _Node, draft, start: int, max_draft: int, nodes: list[_Node]) -> None:
    created: list[_Node | None] = []
    for i, (tok, par) in enumerate(zip(draft.token_ids, draft.parents, strict=True)):
        if par >= i:
            raise ValueError("suffix draft parents must precede children")
        base = parent if par < 0 else created[par]
        if base is None or base.depth >= max_draft:
            created.append(None)
            continue
        child = base.children.get(int(tok))
        if child is None:
            child = _Node(int(tok), start, base.depth + 1, "suffix")
            base.children[int(tok)] = child
            nodes.append(child)
        created.append(child)


def replay_request(
    req_id,
    prompt: Sequence[int],
    output: Sequence[int],
    readouts: Mapping[int, np.ndarray],
    arm: ArmConfig,
    cache: SuffixCache | None,
) -> dict:
    """Replay one request. output[0] comes from prefill and is not counted.

    readouts[t][i] lists the depth-t top-k tokens predicted for output[i] (row i
    is read at the position just before output[i]).
    """
    T, d, K = arm.depth_total, arm.draft_depth, arm.max_draft
    if arm.use_fill and d >= T:
        raise ValueError("draft depth must be below full depth")
    out = [int(t) for t in output]
    L = len(out)
    shallow = readouts[d] if arm.use_fill or arm.prune_top else None
    use_suffix = arm.use_suffix and cache is not None
    if use_suffix:
        cache.start_request(req_id, np.asarray(prompt, dtype=np.int32))
        cache.add_active_response(req_id, np.asarray(out[:1], dtype=np.int32))
    base_ctx = [int(t) for t in prompt]

    def draft(ctx: list[int], room: int):
        return cache.speculate(
            req_id,
            np.asarray(ctx, dtype=np.int32),
            max_spec_tokens=room,
            max_spec_factor=arm.spec_factor,
            max_spec_offset=0.0,
            min_token_prob=arm.min_token_prob,
            use_tree_spec=arm.tree_spec,
        )

    stats = {"tokens": 0, "steps": 0, "calls": 0, "rows": 0, "from_suffix": 0, "from_fill": 0}
    committed = 1
    while committed < L:
        ctx = base_ctx + out[:committed]
        root = _Node(out[committed - 1], 0, 0, "root")
        nodes = [root]
        if use_suffix and K > 0:
            _attach(root, draft(ctx, K), 0, K, nodes)
        u, j, path = root, committed - 1, []
        new: list[int] = []
        verify_time = 0
        while True:
            if j + 1 >= L:
                end = verify_time
                break
            target = out[j + 1]
            if shallow is not None:
                row = shallow[j + 1]
                if arm.use_fill and u.depth < K and int(row[0]) not in u.children:
                    f = _Node(int(row[0]), u.start + d, u.depth + 1, "fill")
                    u.children[f.token] = f
                    nodes.append(f)
                    if use_suffix and f.depth < K:
                        _attach(f, draft(ctx + path + [f.token], K - f.depth), f.start, K, nodes)
                if arm.prune_top:
                    keep = {int(t) for t in row[: arm.prune_top]}
                    for tok in [t for t in u.children if t not in keep]:
                        _kill(u.children.pop(tok), u.start + d)
            nxt = u.children.get(target)
            new.append(target)
            if nxt is None:
                end = u.start + T
                break
            stats["from_suffix" if nxt.source == "suffix" else "from_fill"] += 1
            verify_time = u.start + T
            path.append(target)
            u, j = nxt, j + 1
        for n in nodes:
            stop = end if n.killed is None else min(end, n.killed)
            stats["rows"] += max(0, min(stop, n.start + T) - n.start)
        if use_suffix:
            cache.add_active_response(req_id, np.asarray(new, dtype=np.int32))
        committed += len(new)
        stats["tokens"] += len(new)
        stats["steps"] += 1
        stats["calls"] += end
    if use_suffix:
        cache.stop_request(req_id)
    return stats


def summarize(stats: Sequence[Mapping[str, int]], depth_total: int) -> dict:
    tot = {k: sum(s[k] for s in stats) for k in stats[0]} if stats else {}
    tokens = max(1, tot.get("tokens", 0))
    calls = max(1, tot.get("calls", 0))
    accepted = tot.get("from_suffix", 0) + tot.get("from_fill", 0)
    return tot | {
        "calls_per_token": calls / tokens,
        "speedup_vs_ar": depth_total * tokens / calls,
        "tokens_per_step": tokens / max(1, tot.get("steps", 0)),
        "rows_per_token": tot.get("rows", 0) / tokens,
        "suffix_share_of_accepted": tot.get("from_suffix", 0) / max(1, accepted),
    }
