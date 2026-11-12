"""Greedy dual-proposal LoopSpec on the shared LoopFill execution backend.

Schedule reference: kaist-flexml-lab/loopspec @ 41aa710ed7c9ce908cffa03597be5d1df19d7b56,
sglang_recurrent/loopspec/{pipeline,decisions}.py. This is a same-engine
reimplementation, not the authors' SGLang/CUDA-graph performance result.
"""

from dataclasses import dataclass
import time

import torch

from .loop_budget_decode import RecyclingTree
from .loop_decode_steps import ModelCall, run_single


@dataclass(frozen=True)
class LoopSpecConfig:
    name: str
    depth_total: int
    draft_depth: int
    second_depth: int
    use_suffix: bool = False
    draft_node_budget: int | None = None

    def __post_init__(self):
        if not 0 < self.draft_depth < self.second_depth < self.depth_total:
            raise ValueError("LoopSpec requires 0 < d1 < d2 < T")
        if self.depth_total % self.draft_depth or self.second_depth % self.draft_depth:
            raise ValueError("LoopSpec requires T and d2 divisible by d1")
        if self.use_suffix:
            raise ValueError("LoopSpec does not use suffix proposals")
        if self.draft_node_budget is not None and (
            type(self.draft_node_budget) is not int or self.draft_node_budget < 0
        ):
            raise ValueError("draft_node_budget must be a nonnegative integer or None")

    @property
    def live_bound(self):
        second = self.second_depth // self.draft_depth
        births = [1] * second
        for cycle in range(second, self.depth_total // self.draft_depth):
            births.append(births[cycle - 1] + births[cycle - second])
        return sum(births)

    @property
    def physical_nodes(self):
        native = 2 * self.live_bound + 1
        return native if self.draft_node_budget is None else min(native, self.draft_node_budget + 1)


@torch.no_grad()
def loopspec_decode(backbone, *args, **kwargs):
    return run_single(backbone, loopspec_decode_steps(backbone, *args, **kwargs))


def loopspec_decode_steps(backbone, prompt, cfg, max_new_tokens, stop, cache=None,
                          req_id=None, *, max_nodes=1152, flush_every=16, observer=None):
    """Keep q1 and distinct q2 branches until full-depth verification selects one.

All live branches advance d1 blocks per cycle, exactly as the reference schedule.
Confirmed nodes commit before their physical slots are recycled. No top-N
selection, suffix proposals, score pruning, or online controller is used.
"""
    if not prompt or max_new_tokens < 1:
        raise ValueError("nonempty prompt and positive generation budget required")
    T, d1, d2 = cfg.depth_total, cfg.draft_depth, cfg.second_depth
    # The official full-tree upper bound (sum of births at ages 0..T/d1-1).
    live_bound = cfg.live_bound
    # New proposals are temporarily present before final branch retention.
    capacity = cfg.physical_nodes
    budget = cfg.draft_node_budget
    if max_nodes < capacity:
        raise ValueError(f"LoopSpec needs {capacity} tree slots; got {max_nodes}")
    tree = RecyclingTree(capacity)
    stats = dict(tokens=0, steps=0, calls=0, rows=0, max_rows=0, nodes=0,
                 readout_rows=0, cap_skips=0, first_proposals=0, second_proposals=0,
                 second_agreements=0, accepted_first=0, accepted_second=0,
                 fallback_tokens=0, draft_hits=0, wasted_rows=0, peak_live_nodes=0,
                 readout_depths=[d1, d2], tree_policy="loopspec_dual" if budget is None else "loopspec_capped",
                 draft_node_budget=budget, budget_rejections=0, peak_draft_nodes=0,
                 schedule="selective_second_proposal", implementation="same_engine")
    yield ModelCall("synchronize")
    start = time.perf_counter()
    out = [int((yield ModelCall("prefill", (list(prompt), d1))))]
    prefilled = time.perf_counter()
    tree.base_pos = backbone.length
    root_path = tuple(out)
    pool = {}

    def add(path, parent, source):
        # Capped ablation: retain existing paths, admit proposals in the native
        # d1-then-d2 order while space remains. Never replace the confirmed root.
        if parent is not None and budget is not None and len(pool) - 1 >= budget:
            stats["budget_rejections"] += 1
            return None
        n = tree.add(path[-1], parent, stats["calls"], source)
        if n is None:
            raise RuntimeError("LoopSpec exceeded its preregistered physical bound")
        pool[path] = n
        stats["nodes"] += 1
        stats["peak_draft_nodes"] = max(stats["peak_draft_nodes"], len(pool) - 1)
        return n

    if len(out) < max_new_tokens and out[-1] not in stop:
        add(root_path, None, "root")
    while len(out) < max_new_tokens and out[-1] not in stop:
        fresh = tree.take_fresh()
        if fresh:
            yield ModelCall("embed", (tree, fresh))
        live = sorted(pool.values(), key=lambda n: (n.step, n.depth, n.idx))
        stats["peak_live_nodes"] = max(stats["peak_live_nodes"], len(live))
        if len(live) > live_bound:
            raise RuntimeError("LoopSpec exceeded the official live-lane bound")
        if observer is not None:
            observer("cycle", lanes=sorted((p, n.step) for p, n in pool.items()))
        for _ in range(d1):
            yield ModelCall("advance", (tree, live))
            for n in live:
                n.step += 1
            stats["calls"] += 1
            stats["rows"] += len(live)
            stats["max_rows"] = max(stats["max_rows"], len(live))
        groups = [(d, [n for n in live if n.step == d]) for d in (d1, d2)]
        full = [n for n in live if n.step == T]
        # m=0 is grouped argmax; no full-vocabulary softmax is needed by LoopSpec.
        reads, tops = yield ModelCall("readout_top", (tree, groups, full, 0))
        stats["readout_rows"] += sum(len(g) for _, g in groups) + len(full)
        paths = {n.idx: p for p, n in pool.items()}
        for (depth, group), predictions in zip(groups, reads, strict=True):
            for n, prediction in zip(group, predictions, strict=True):
                token = int(prediction)
                path = paths[n.idx]
                if depth == d1:
                    n.shallow_top = token
                    stats["first_proposals"] += 1
                elif token == n.shallow_top:
                    stats["second_agreements"] += 1
                    continue
                else:
                    n.second_top = token
                    stats["second_proposals"] += 1
                # Do not compute the final output token as another input lane.
                if len(path) + 1 < max_new_tokens:
                    add(path + (token,), n, "first" if depth == d1 else "second")
        if full:
            if len(full) != 1 or full[0] is not pool[root_path]:
                raise RuntimeError("LoopSpec attempted out-of-order full verification")
            root = full[0]
            root.full_top = gold = int(tops[0])
            next_path = root_path + (gold,)
            selected = pool.get(next_path)
            stats["draft_hits"] += int(selected is not None)
            stats["tokens"] += 1
            stats["steps"] += 1
            accepted = ("accepted_first" if gold == root.shallow_top else
                        "accepted_second" if gold == getattr(root, "second_top", None)
                        else "fallback_tokens")
            stats[accepted] += 1
            out.append(gold)
            # Commit before releasing the slot, including at EOS/output exhaustion.
            backbone.commit(tree, [root])
            for path, n in list(pool.items()):
                if path[:len(next_path)] != next_path:
                    if n is not root:
                        stats["wasted_rows"] += n.step
                    tree.release(n)
                    del pool[path]
            tree.base_pos += 1
            for n in pool.values():
                n.depth -= 1
            if selected is not None:
                selected.parent = -1
            root_path = next_path
            if len(out) < max_new_tokens and gold not in stop and selected is None:
                add(root_path, None, "root")
    if budget is not None:
        # Match the budget decoder's accounting of unfinished terminal drafts.
        stats['wasted_rows'] += sum(n.step for path, n in pool.items() if path != root_path)
    yield ModelCall("synchronize")
    stats.update(prefill_s=prefilled - start, decode_s=time.perf_counter() - prefilled)
    return out, stats
