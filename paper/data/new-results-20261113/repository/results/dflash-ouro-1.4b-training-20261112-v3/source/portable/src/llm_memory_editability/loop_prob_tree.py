"""Replay of loop speculative decoding that places draft candidates by their chance of being right.

Extends llm_memory_editability.loop_suffix_spec. A looped LM runs one shared block T
times per token; a node that starts at block call s reads the depth-r prediction for
its next position at s + r and its full-depth prediction at s + T. One decoding step
starts from the pending token (root, p = 1):

1. Suffix-tree drafts are offered under the root at call 0 and under every node
   placed later, starting with that node.
2. A node reaching a readout depth r offers its depth-r top-m tokens as children.
   Tokens already in the tree are rescored instead.
3. Every offered candidate c gets p(c) = p(parent) * e(c), where e is the calibrated
   chance that the edge is right given its parent is right. c is placed iff
   p(c) > tau. When more than B rows are live in a call, the lowest-p nodes are
   removed with their subtrees.
4. At s + T a node's full-depth output removes its wrong children. The step ends when
   the greedy token is missing from the children of the last accepted node.
5. Every accepted token goes into the suffix tree at once, and the suffix tree drafts
   again from the newly accepted node (commit_on_accept). Without this an uncapped
   step can run for hundreds of tokens while the suffix tree never sees them.

Block calls are exact along the greedy output, as in loop_suffix_spec. Rows are
approximate: readouts of off-path nodes were not stored, so the readout at the same
output position on the greedy path stands in for them (offpath_readouts=True).

legacy=True reproduces the `suffix+shallow` arm of loop_suffix_spec.replay_request:
the depth-d top-1 is always added when absent, suffix drafts keep tokens with
probability >= 0.1, and the draft depth is capped at K.
"""

from __future__ import annotations

import bisect
import heapq
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

PROB_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
MATCH_EDGES = (0, 2, 3, 4, 6, 8, 12, 16, 24, 32)
ABSENT = 8  # rank used for a suffix candidate outside the stored top-8
MIN_COUNT = 50
CTX_WINDOW = 256  # suffix queries see only the context tail; the cache itself keeps 64
LEAD_EDGES = (0, 1, 2, 4, 8)  # draft positions ahead of the last committed token


def _bin(x: float, edges: Sequence[float]) -> int:
    return max(0, bisect.bisect_right(edges, x) - 1)


class Calibration:
    """Calibrated edge probabilities with coarser fallbacks for sparse cells.

    shallow keys: (depth, rank, prob bin, proposed by the suffix tree)
    suffix keys: (conditional prob bin, match-length bin), from full-context queries
    insitu keys: (conditional prob bin, match-length bin, rival child present, lead bin),
        from the suffix queries a decoder actually makes
    joint keys: (deep depth, rank at d, rank at the deep depth, prob bin there, suffix flag)
    Counts are [hits, n].
    """

    def __init__(self, shallow=None, suffix=None, insitu=None, joint=None):
        self.shallow = defaultdict(lambda: [0, 0], shallow or {})
        self.suffix = defaultdict(lambda: [0, 0], suffix or {})
        self.insitu = defaultdict(lambda: [0, 0], insitu or {})
        self.joint = defaultdict(lambda: [0, 0], joint or {})
        self._cache: dict = {}

    def add_shallow(self, depth: int, rank: int, prob: float, in_suffix: bool, hit: bool) -> None:
        c = self.shallow[(depth, rank, _bin(prob, PROB_EDGES), bool(in_suffix))]
        c[0] += int(hit)
        c[1] += 1

    def add_suffix(self, q: float, match_len: int, hit: bool) -> None:
        c = self.suffix[(_bin(q, PROB_EDGES), _bin(match_len, MATCH_EDGES))]
        c[0] += int(hit)
        c[1] += 1

    def add_insitu(self, q: float, match_len: int, rival: bool, lead: int, hit: bool) -> None:
        key = (
            _bin(q, PROB_EDGES),
            _bin(match_len, MATCH_EDGES),
            bool(rival),
            _bin(lead, LEAD_EDGES),
        )
        c = self.insitu[key]
        c[0] += int(hit)
        c[1] += 1

    def add_joint(self, deep: int, rank_d: int, rank: int, prob: float, flag: bool, hit: bool):
        c = self.joint[(deep, rank_d, rank, _bin(prob, PROB_EDGES), bool(flag))]
        c[0] += int(hit)
        c[1] += 1

    @staticmethod
    def _lookup(table: Mapping, key: tuple, levels: Sequence, default=0.0):
        for keep in levels:
            hits = n = 0
            for k, (h, m) in table.items():
                if keep(k, key):
                    hits += h
                    n += m
            if n >= MIN_COUNT:
                return hits / n
        return default

    def shallow_prob(self, depth: int, rank: int, prob: float, in_suffix: bool) -> float:
        key = (depth, rank, _bin(prob, PROB_EDGES), bool(in_suffix))
        if ("s", key) not in self._cache:
            levels = (
                lambda k, q: k == q,
                lambda k, q: k[:3] == q[:3],
                lambda k, q: k[:2] == q[:2],
                lambda k, q: k[0] == q[0] and (k[1] == ABSENT) == (q[1] == ABSENT),
            )
            self._cache[("s", key)] = self._lookup(self.shallow, key, levels)
        return self._cache[("s", key)]

    def suffix_prob(self, q: float, match_len: int) -> float:
        key = (_bin(q, PROB_EDGES), _bin(match_len, MATCH_EDGES))
        if ("x", key) not in self._cache:
            levels = (lambda k, q: k == q, lambda k, q: k[0] == q[0], lambda k, q: True)
            self._cache[("x", key)] = self._lookup(self.suffix, key, levels)
        return self._cache[("x", key)]

    def suffix_prob_insitu(self, q: float, match_len: int, rival: bool, lead: int) -> float:
        key = (
            _bin(q, PROB_EDGES),
            _bin(match_len, MATCH_EDGES),
            bool(rival),
            _bin(lead, LEAD_EDGES),
        )
        if ("i", key) not in self._cache:
            levels = (
                lambda k, q: k == q,
                lambda k, q: (k[0], k[2], k[3]) == (q[0], q[2], q[3]),
                lambda k, q: (k[0], k[2]) == (q[0], q[2]),
            )
            got = self._lookup(self.insitu, key, levels, default=None)
            self._cache[("i", key)] = self.suffix_prob(q, match_len) if got is None else got
        return self._cache[("i", key)]

    def joint_prob(self, deep: int, rank_d: int, rank: int, prob: float, flag: bool) -> float:
        key = (deep, rank_d, rank, _bin(prob, PROB_EDGES), bool(flag))
        if ("j", key) not in self._cache:
            levels = (lambda k, q: k == q, lambda k, q: k[:4] == q[:4], lambda k, q: k[:3] == q[:3])
            got = self._lookup(self.joint, key, levels, default=None)
            self._cache[("j", key)] = (
                self.shallow_prob(deep, rank, prob, flag) if got is None else got
            )
        return self._cache[("j", key)]

    def to_json(self) -> dict:
        return {
            "prob_edges": PROB_EDGES,
            "match_edges": MATCH_EDGES,
            "lead_edges": LEAD_EDGES,
            "shallow": [[*k, *v] for k, v in sorted(self.shallow.items())],
            "suffix": [[*k, *v] for k, v in sorted(self.suffix.items())],
            "insitu": [[*k, *v] for k, v in sorted(self.insitu.items())],
            "joint": [[*k, *v] for k, v in sorted(self.joint.items())],
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Calibration:
        if tuple(d["prob_edges"]) != PROB_EDGES or tuple(d["match_edges"]) != MATCH_EDGES:
            raise ValueError("calibration bins differ from this module")
        if tuple(d.get("lead_edges", LEAD_EDGES)) != LEAD_EDGES:
            raise ValueError("calibration bins differ from this module")
        shallow = {(a, b, c, bool(f)): [h, n] for a, b, c, f, h, n in d["shallow"]}
        suffix = {(a, b): [h, n] for a, b, h, n in d["suffix"]}
        insitu = {(a, b, bool(c), e): [h, n] for a, b, c, e, h, n in d.get("insitu", [])}
        joint = {(a, b, c, e, bool(f)): [h, n] for a, b, c, e, f, h, n in d.get("joint", [])}
        return cls(shallow, suffix, insitu, joint)


def _speculate(cache, req_id, ctx: list[int], room: int, spec_factor: float, min_prob: float):
    return cache.speculate(
        req_id,
        np.asarray(ctx, dtype=np.int32),
        max_spec_tokens=room,
        max_spec_factor=spec_factor,
        max_spec_offset=0.0,
        min_token_prob=min_prob,
        use_tree_spec=True,
    )


def _edge_probs(draft) -> list[float]:
    """Conditional probability of each draft token given its draft parent."""
    probs = list(getattr(draft, "probs", None) or [1.0] * len(draft.token_ids))
    return [
        p if par < 0 else p / max(probs[par], 1e-12)
        for p, par in zip(probs, draft.parents, strict=True)
    ]


def collect_calibration(
    req_id,
    prompt: Sequence[int],
    output: Sequence[int],
    ids: Mapping[int, np.ndarray],
    probs: Mapping[int, np.ndarray],
    depths: Sequence[int],
    cache,
    calib: Calibration,
    spec_factor: float = 1.0,
    room: int = 64,
    min_prob: float = 0.001,
    joint_pairs: Sequence[tuple[int, int]] = (),
) -> None:
    """Add one request's greedy-path edges to calib.

    At every output position i the suffix tree drafts from prompt + output[:i]. Each
    draft token whose draft ancestors all match the output is a sample of a suffix
    edge with a right parent. Each stored depth-r top-8 token is a shallow sample,
    flagged when the suffix tree also drafted it as the next token. For each (d, deep)
    in joint_pairs, every token in either top-8 is a joint sample of its two ranks.
    """
    out = [int(t) for t in output]
    L = len(out)
    base = [int(t) for t in prompt]
    cache.start_request(req_id, np.asarray(prompt, dtype=np.int32))
    cache.add_active_response(req_id, np.asarray(out[:1], dtype=np.int32))
    for i in range(1, L):
        draft = _speculate(cache, req_id, base + out[:i], room, spec_factor, min_prob)
        edge = _edge_probs(draft)
        level: list[int] = []
        right: list[bool] = []
        first = set()
        for k, (tok, par) in enumerate(zip(draft.token_ids, draft.parents, strict=True)):
            lv = 0 if par < 0 else level[par] + 1
            level.append(lv)
            pos = i + lv
            ok_parent = par < 0 or right[par]
            hit = pos < L and int(tok) == out[pos]
            right.append(ok_parent and hit)
            if par < 0:
                first.add(int(tok))
            if ok_parent and pos < L:
                calib.add_suffix(edge[k], draft.match_len, hit)
        for r in depths:
            row_ids = [int(t) for t in ids[r][i]]
            row_p = probs[r][i]
            for k, tok in enumerate(row_ids):
                calib.add_shallow(r, k, float(row_p[k]), tok in first, tok == out[i])
            for tok in first - set(row_ids):
                calib.add_shallow(r, ABSENT, 0.0, True, tok == out[i])
        for d, deep in joint_pairs:
            rank_d = {int(t): j for j, t in enumerate(ids[d][i])}
            deep_ids = [int(t) for t in ids[deep][i]]
            for k, tok in enumerate(deep_ids):
                flag, hit = tok in first, tok == out[i]
                calib.add_joint(
                    deep, rank_d.get(tok, ABSENT), k, float(probs[deep][i][k]), flag, hit
                )
            for tok in (set(rank_d) | first) - set(deep_ids):
                flag, hit = tok in first, tok == out[i]
                calib.add_joint(deep, rank_d.get(tok, ABSENT), ABSENT, 0.0, flag, hit)
        cache.add_active_response(req_id, np.asarray(out[i : i + 1], dtype=np.int32))
    cache.stop_request(req_id)


@dataclass(frozen=True)
class ProbConfig:
    name: str
    depth_total: int  # T
    draft_depth: int  # d
    readout_depths: tuple[int, ...]  # depths at which a node offers its top-m; includes d
    tau: float = 0.01
    tau_suffix: float | None = None  # threshold for suffix drafts (placed at r = 0)
    tau_deep: float | None = None  # threshold for readouts deeper than d
    top_m: int = 8
    row_cap: int | None = None  # B
    max_draft: int | None = None  # K; None removes the per-step depth cap
    spec_factor: float = 1.0
    suffix_min_prob: float = 0.001
    suffix_room: int = 64
    use_suffix: bool = True
    offpath_readouts: bool = True
    commit_on_accept: bool = True  # feed accepted tokens to the suffix tree and redraft at once
    rescore: bool = True  # readouts rescore children already in the tree
    suffix_calibration: str = "offline"  # "insitu": table fitted on the decoder's own queries
    joint: bool = False  # deeper readouts score tokens jointly with the depth-d readout
    deep_expand: bool = True  # deeper readouts may place new candidates
    deep_raise: bool = True  # deeper readouts may raise a candidate's p (else only lower it)
    prune: bool = False  # remove a candidate and its subtree once its rescored p <= tau
    legacy: bool = False

    @property
    def eager_commit(self) -> bool:
        return self.commit_on_accept and self.use_suffix and not self.legacy


class _Node:
    __slots__ = (
        "token", "start", "pos", "depth", "p", "e", "onpath",
        "children", "parent", "killed", "src", "suffix", "locked",
    )  # fmt: skip

    def __init__(self, token, start, pos, depth, p, e, onpath, parent, src, suffix):
        self.token = token
        self.start = start
        self.pos = pos
        self.depth = depth
        self.p = p
        self.e = e
        self.onpath = onpath
        self.children: dict[int, _Node] = {}
        self.parent = parent
        self.killed: int | None = None
        self.src = src
        self.suffix = suffix
        self.locked = parent is None  # root and accepted nodes are never removed


class _Step:
    """One decoding step: block calls advance until the greedy token is missing."""

    def __init__(self, sim: ProbReplay, committed: int):
        self.sim = sim
        cfg = sim.cfg
        self.T = cfg.depth_total
        self.K = cfg.max_draft
        self.events: dict[int, list] = defaultdict(list)
        self.times: list[int] = []
        self.nodes: list[_Node] = []
        root = _Node(
            sim.out[committed - 1], 0, committed - 1, 0, 1.0, 1.0, True, None, "root", False
        )
        self._add(root)
        self.root = root
        self.ctx = sim.base_ctx + sim.out[:committed]
        self.accepted: list[int] = []
        self.sources: list[str] = []
        self.miss: tuple[_Node, int] | None = None
        self.end: int | None = None
        self._place_suffix(root, 0)

    # tree edits -----------------------------------------------------------
    def _add(self, n: _Node) -> None:
        self.nodes.append(n)
        for r in self.sim.cfg.readout_depths:
            self._schedule(n.start + r, (1, n.depth, len(self.nodes), "read", n, r))
        self._schedule(n.start + self.T, (0, n.depth, len(self.nodes), "verify", n, 0))

    def _schedule(self, t: int, ev: tuple) -> None:
        if t not in self.events:
            heapq.heappush(self.times, t)
        self.events[t].append(ev)

    def _child(self, parent, tok, t, e, src, suffix, anchor) -> _Node:
        """anchor: the node whose readout or suffix draft offered this candidate."""
        pos = parent.pos + 1
        out = self.sim.out
        onpath = parent.onpath and pos < len(out) and out[pos] == tok
        c = _Node(tok, t, pos, parent.depth + 1, parent.p * e, e, onpath, parent, src, suffix)
        parent.children[tok] = c
        self._add(c)
        trace = self.sim.trace
        if trace is not None and anchor.onpath and anchor.p > 0:
            rivals = len(parent.children) - 1  # live candidates already offered here
            trace.append((c.p / anchor.p, c.onpath, src, c.depth - anchor.depth, rivals))
        return c

    def _kill(self, n: _Node, t: int) -> int:
        """Removes n and its subtree at call t; returns how many of them were live at t."""
        if n.parent is not None and n.parent.children.get(n.token) is n:
            del n.parent.children[n.token]
        live = 0
        stack = [n]
        while stack:
            m = stack.pop()
            if m.killed is None:
                m.killed = t
                live += m.start <= t < m.start + self.T
            stack.extend(m.children.values())
        return live

    def _reprob(self, n: _Node) -> None:
        stack = [n]
        while stack:
            m = stack.pop()
            m.p = m.parent.p * m.e
            stack.extend(m.children.values())

    def _depth_ok(self, parent: _Node) -> bool:
        return self.K is None or parent.depth < self.K

    def _path_tokens(self, n: _Node) -> list[int]:
        toks = []
        while n.parent is not None:
            toks.append(n.token)
            n = n.parent
        return toks[::-1]

    # drafters ---------------------------------------------------------------
    def _place_suffix(self, n: _Node, t: int) -> None:
        sim, cfg = self.sim, self.sim.cfg
        if not cfg.use_suffix or sim.cache is None:
            return
        room = cfg.suffix_room if self.K is None else self.K - n.depth
        if room <= 0:
            return
        min_prob = 0.1 if cfg.legacy else cfg.suffix_min_prob
        ctx = (self.ctx[-CTX_WINDOW:] + self._path_tokens(n))[-CTX_WINDOW:]
        draft = _speculate(sim.cache, sim.req_id, ctx, room, cfg.spec_factor, min_prob)
        sim.stats["suffix_queries"] += 1
        edge = _edge_probs(draft)
        tau = cfg.tau if cfg.tau_suffix is None else cfg.tau_suffix
        frontier = self.root.pos + (len(self.accepted) if cfg.eager_commit else 0)
        out = sim.out
        created: list[_Node | None] = []
        for k, (tok, par) in enumerate(zip(draft.token_ids, draft.parents, strict=True)):
            base = n if par < 0 else created[par]
            if base is None or not self._depth_ok(base) or base.killed is not None:
                created.append(None)
                continue
            tok = int(tok)
            child = base.children.get(tok)
            if child is not None:
                child.suffix = True
                created.append(child)
                continue
            rival = bool(base.children)  # a different token is already offered here
            lead = base.pos - frontier
            if sim.edge_log is not None and base.onpath and base.pos + 1 < len(out):
                hit = tok == out[base.pos + 1]
                sim.edge_log.append((edge[k], draft.match_len, rival, lead, hit))
            if cfg.legacy:
                created.append(self._child(base, tok, t, 1.0, "suffix", True, n))
                continue
            if cfg.suffix_calibration == "insitu":
                e = sim.calib.suffix_prob_insitu(edge[k], draft.match_len, rival, lead)
            else:
                e = sim.calib.suffix_prob(edge[k], draft.match_len)
            if base.p * e > tau:
                created.append(self._child(base, tok, t, e, "suffix", True, n))
            else:
                created.append(None)

    def _readout(self, n: _Node, r: int, t: int) -> None:
        sim, cfg = self.sim, self.sim.cfg
        i = n.pos + 1
        if i >= len(sim.out) or (not n.onpath and not cfg.offpath_readouts):
            return
        row_ids = sim.ids[r][i]
        row_p = sim.probs[r][i]
        if cfg.legacy:
            tok = int(row_ids[0])
            if r == cfg.draft_depth and self._depth_ok(n) and tok not in n.children:
                f = self._child(n, tok, t, 1.0, "fill", False, n)
                self._place_suffix(f, t)
            return
        tau = cfg.tau if (r <= cfg.draft_depth or cfg.tau_deep is None) else cfg.tau_deep
        deep = r > cfg.draft_depth
        joint = deep and cfg.joint
        rank_d = {int(x): j for j, x in enumerate(sim.ids[cfg.draft_depth][i])} if joint else {}

        def edge_prob(tok: int, k: int, q: float, flag: bool) -> float:
            if joint:
                return sim.calib.joint_prob(r, rank_d.get(tok, ABSENT), k, q, flag)
            return sim.calib.shallow_prob(r, k, q, flag)

        seen = set()
        for k in range(len(row_ids)):
            tok, q = int(row_ids[k]), float(row_p[k])
            seen.add(tok)
            child = n.children.get(tok)
            if child is not None:
                if cfg.rescore:
                    self._rescore(child, edge_prob(tok, k, q, child.suffix), deep, t)
            elif k < cfg.top_m and self._depth_ok(n) and (cfg.deep_expand or not deep):
                e = edge_prob(tok, k, q, False)
                if n.p * e > tau:
                    src = "deep" if deep else ("fill" if k == 0 else "sibling")
                    c = self._child(n, tok, t, e, src, False, n)
                    self._place_suffix(c, t)
        for child in list(n.children.values()):
            if not cfg.rescore or child.token in seen or child.killed is not None:
                continue
            if joint:
                e = sim.calib.joint_prob(
                    r, rank_d.get(child.token, ABSENT), ABSENT, 0.0, child.suffix
                )
            elif child.suffix:
                e = sim.calib.shallow_prob(r, ABSENT, 0.0, True)
            else:
                continue
            self._rescore(child, e, deep, t)

    def _rescore(self, child: _Node, e: float, deep: bool, t: int) -> None:
        cfg = self.sim.cfg
        if child.locked or (deep and not cfg.deep_raise and e > child.e):
            return
        child.e = e
        self._reprob(child)
        if cfg.prune and child.p <= cfg.tau:
            self._kill(child, t)
            self.sim.stats["prune_kills"] += 1

    def _verify(self, n: _Node, t: int) -> None:
        out = self.sim.out
        if not n.onpath:
            for c in list(n.children.values()):
                self._kill(c, t)
            return
        target = out[n.pos + 1]
        for c in [c for c in n.children.values() if c.token != target]:
            self._kill(c, t)
        nxt = n.children.get(target)
        self.accepted.append(target)
        sim = self.sim
        if sim.cfg.eager_commit and sim.cache is not None:
            sim.cache.add_active_response(sim.req_id, np.asarray([target], dtype=np.int32))
        if nxt is None:
            self.miss = (n, target)
            self.end = t
        else:
            self.sources.append(nxt.src)
            nxt.locked = True
            nxt.e = 1.0
            self._reprob(nxt)
            if nxt.pos + 1 >= len(out):
                self.end = t
            elif sim.cfg.eager_commit:
                self._place_suffix(nxt, t)  # the tree now holds every committed token

    def _enforce_cap(self, t: int) -> None:
        cap = self.sim.cfg.row_cap
        if cap is None:
            return
        live = [n for n in self.nodes if n.killed is None and n.start <= t < n.start + self.T]
        if len(live) <= cap:
            return
        excess = len(live) - cap
        for n in sorted(live, key=lambda m: (m.p, -m.start, -m.depth)):
            if excess <= 0:
                break
            if n.killed is not None or n.locked:
                continue
            excess -= self._kill(n, t)
            self.sim.stats["cap_kills"] += 1

    # run --------------------------------------------------------------------
    def run(self) -> None:
        while self.end is None:
            t = heapq.heappop(self.times)
            for ev in sorted(self.events.pop(t), key=lambda e: e[:3]):
                _, _, _, kind, n, r = ev
                if n.killed is not None:
                    continue
                if kind == "verify":
                    self._verify(n, t)
                    if self.end is not None:
                        break
                else:
                    self._readout(n, r, t)
            if self.end is None:
                self._enforce_cap(t)

    def tally(self) -> dict:
        end, T = self.end, self.T
        per_call = np.zeros(end + 1, dtype=np.int64)
        reads = np.zeros(end + 1, dtype=np.int64)
        for n in self.nodes:
            stop = min(end, n.start + T, n.killed if n.killed is not None else end)
            if stop > n.start:
                per_call[n.start] += 1
                per_call[stop] -= 1
            for r in self.sim.cfg.readout_depths:
                c = n.start + r - 1
                if c < stop:
                    reads[c] += 1
        rows = np.cumsum(per_call)[:end]
        return {"rows_per_call": rows, "reads_per_call": reads[:end]}


class ProbReplay:
    """Replays one request under a ProbConfig."""

    def __init__(
        self, req_id, prompt, output, ids, probs, cfg, cache, calib, trace=None, edge_log=None
    ):
        if cfg.draft_depth not in cfg.readout_depths:
            raise ValueError("readout_depths must include draft_depth")
        if max(cfg.readout_depths) >= cfg.depth_total:
            raise ValueError("readouts must be shallower than T")
        self.req_id = req_id
        self.base_ctx = [int(t) for t in prompt]
        self.out = [int(t) for t in output]
        self.ids = ids
        self.probs = probs
        self.cfg = cfg
        self.cache = cache if cfg.use_suffix else None
        self.calib = calib
        self.stats: dict = defaultdict(int)
        # (p relative to the offering on-path node, placed node on the greedy path, source,
        # draft depth below that node, siblings already present) for every placement
        # offered by an on-path node
        self.trace: list | None = trace
        # (edge prob, match length, rival present, lead, right) for every suffix token
        # offered under an on-path node: the in-situ calibration samples
        self.edge_log: list | None = edge_log

    def run(self, on_miss=None) -> dict:
        """Returns step totals and the histogram of rows per block call.

        on_miss(node, target) is called at every step that ends because the greedy
        token was not among the children of the last accepted node.
        """
        out = self.out
        L = len(out)
        if self.cache is not None:
            self.cache.start_request(self.req_id, np.asarray(self.base_ctx, dtype=np.int32))
            self.cache.add_active_response(self.req_id, np.asarray(out[:1], dtype=np.int32))
        hist: dict[tuple[int, int], int] = defaultdict(int)
        committed = 1
        while committed < L:
            step = _Step(self, committed)
            step.run()
            if step.miss is not None and on_miss is not None:
                on_miss(*step.miss)
            tal = step.tally()
            for a, b in zip(
                tal["rows_per_call"].tolist(), tal["reads_per_call"].tolist(), strict=True
            ):
                hist[(a, b)] += 1
            s = self.stats
            s["tokens"] += len(step.accepted)
            s["steps"] += 1
            s["calls"] += step.end
            s["rows"] += int(tal["rows_per_call"].sum())
            s["reads"] += int(tal["reads_per_call"].sum())
            s["nodes"] += len(step.nodes)
            for src in step.sources:
                s[f"from_{src}"] += 1
            if self.cache is not None and not self.cfg.eager_commit:
                self.cache.add_active_response(
                    self.req_id, np.asarray(step.accepted, dtype=np.int32)
                )
            committed += len(step.accepted)
        if self.cache is not None:
            self.cache.stop_request(self.req_id)
        return dict(self.stats) | {"hist": dict(hist)}


def predicted_ms(hist: Mapping[tuple[int, int], int], cost) -> float:
    """Sum of cost(rows, reads) over block calls; cost maps one call to milliseconds."""
    return float(sum(n * cost(rows, reads) for (rows, reads), n in hist.items()))
