"""Fixed per-request draft budget with optional online suffix integral feedback.

No calibration tables or labels are consumed. Logical output paths identify feedback;
recyclable tree slots identify only model state. The legacy threshold decoder remains
in loop_prob_decode. See docs/loopfill-online-suffix-control.md for the contract.
"""

from __future__ import annotations

import heapq
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

from .loop_decode_steps import ModelCall, run_single
from .loop_prob_tree import _edge_probs
from .loop_tree_decode import DraftTree, TreeNode

Path = tuple[int, ...]


@dataclass(frozen=True)
class BudgetConfig:
    name: str
    depth_total: int
    draft_depth: int
    readout_depths: tuple[int, ...]
    draft_node_budget: int = 64
    scoring: str = "online_suffix_i"
    eta: float = 0.1
    c_min: float = 0.25
    c_max: float = 2.0
    use_suffix: bool = True
    use_model: bool = True
    top_m: int = 8
    suffix_room: int = 64
    spec_factor: float = 1.0
    suffix_min_prob: float = 0.001
    trace_controller: bool = False
    tree_policy: str = "node_budget"
    child_suffix_policy: str = "all"
    positive_only: bool = False
    child_suffix_caps: tuple[int, ...] | None = None
    c_initial: float = 1.0

    def __post_init__(self):
        if not isinstance(self.draft_node_budget, int) or self.draft_node_budget < 0:
            raise ValueError("draft_node_budget must be a nonnegative integer")
        if self.scoring not in ("raw", "fixed_suffix", "online_suffix_i") or self.tree_policy != "node_budget":
            raise ValueError("budget decoder supports raw, fixed_suffix or online_suffix_i scoring")
        if not (math.isfinite(self.eta) and self.eta > 0):
            raise ValueError("eta must be finite and positive")
        if not (0 < self.c_min < 1 < self.c_max < math.inf):
            raise ValueError("coefficient bounds must satisfy 0 < c_min < 1 < c_max < inf")
        if not math.isfinite(self.c_initial) or not self.c_min <= self.c_initial <= self.c_max:
            raise ValueError("initial coefficient must be finite and within coefficient bounds")
        if self.scoring == "raw" and self.c_initial != 1.0:
            raise ValueError("raw scoring requires c_initial=1; use fixed_suffix for a fixed weight")
        if not 0 < self.draft_depth < self.depth_total:
            raise ValueError("draft depth must be between zero and total depth")
        if (
            len(set(self.readout_depths)) != len(self.readout_depths)
            or any(not 0 < r < self.depth_total for r in self.readout_depths)
            or (self.use_model and self.draft_depth not in self.readout_depths)
        ):
            raise ValueError("readout depths must be unique, below T, and include draft depth")
        if self.top_m not in (1, 2, 8) or self.suffix_room < 1:
            raise ValueError("model top-k must be 1, 2 or 8; suffix query cap must be positive")
        if self.child_suffix_policy not in ("all", "best1", "none"):
            raise ValueError("child_suffix_policy must be all, best1 or none")
        if self.child_suffix_caps is not None and (
            len(self.child_suffix_caps) != self.top_m
            or any(type(q) is not int or q < 0 for q in self.child_suffix_caps)
        ):
            raise ValueError("child suffix caps must contain one nonnegative integer per model rank")
        if not math.isfinite(self.spec_factor) or self.spec_factor <= 0:
            raise ValueError("spec_factor must be finite and positive")
        if not 0 <= self.suffix_min_prob <= 1:
            raise ValueError("suffix_min_prob must be in [0, 1]")


def _score(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1 + 1e-6:
        raise ValueError(f"edge score must be finite and in [0, 1], got {value}")
    return min(1.0, value)


@dataclass(frozen=True)
class SuffixPrediction:
    q: float
    coefficient: float
    prediction: float


class SuffixController:
    """Request-local immutable prediction snapshots, independent of GPU residency."""

    def __init__(self, cfg: BudgetConfig):
        self.cfg = cfg
        self.u = math.log(cfg.c_initial)
        self.records: dict[Path, dict[int, SuffixPrediction]] = {}
        # Includes unconfirmed parents whose full result is already available.
        self.known: set[Path] = set()
        self.pending = self.peak_records = self.peak_parents = 0
        self.created = self.discarded = self.feedback_records = self.updates = 0
        self.c_low = self.c_high = cfg.c_initial
        self.trace: list[dict] = []

    @property
    def c(self):
        return math.exp(self.u)

    @property
    def enabled(self):
        return self.cfg.scoring == "online_suffix_i" and self.cfg.draft_node_budget > 0

    def observe(self, parent: Path, token: int, q: float):
        if not self.enabled or parent in self.known:
            return
        records = self.records.setdefault(parent, {})
        if token not in records:
            q = _score(q)
            records[token] = SuffixPrediction(q, self.c, min(1.0, self.c * q))
            self.pending += 1
            self.created += 1
            self.peak_records = max(self.peak_records, self.pending)
            self.peak_parents = max(self.peak_parents, len(self.records))

    def mark_known(self, parent: Path):
        if self.enabled:
            self.known.add(parent)

    def verify(self, parent: Path, gold: int):
        """Called only when this parent is on the confirmed prefix; at most once."""
        self.mark_known(parent)
        records = self.records.pop(parent, {})
        self.pending -= len(records)
        if not records:
            return None
        delta = math.fsum((token == gold) - r.prediction for token, r in records.items())
        delta /= len(records)
        before = self.c
        self.u = min(
            math.log(self.cfg.c_max), max(math.log(self.cfg.c_min), self.u + self.cfg.eta * delta)
        )
        self.updates += 1
        self.feedback_records += len(records)
        self.c_low, self.c_high = min(self.c_low, self.c), max(self.c_high, self.c)
        event = dict(
            output_position=len(parent),
            records=len(records),
            delta=delta,
            c_before=before,
            c_after=self.c,
        )
        if self.cfg.trace_controller:
            self.trace.append(event)
        return event

    def retain_prefix(self, prefix: Path):
        """Only a confirmed token (never budget eviction) rules out parent paths."""
        for path in list(self.records):
            if path[: len(prefix)] != prefix:
                n = len(self.records.pop(path))
                self.pending -= n
                self.discarded += n
        self.known = {p for p in self.known if p[: len(prefix)] == prefix}

    def metadata_bytes(self):
        """Current Python allocation estimate, excluding allocator/GPU overhead."""
        return (
            sys.getsizeof(self.records)
            + sys.getsizeof(self.known)
            + sum(
                sys.getsizeof(p)
                + sys.getsizeof(rs)
                + sum(
                    sys.getsizeof(r)
                    + sys.getsizeof(r.__dict__)
                    + sum(sys.getsizeof(v) for v in r.__dict__.values())
                    + sys.getsizeof(t)
                    for t, r in rs.items()
                )
                for p, rs in self.records.items()
            )
            + sum(sys.getsizeof(p) for p in self.known)
        )


@dataclass
class Candidate:
    path: Path
    raw_score: float
    score_source: str
    origin: str
    suffix: bool = False
    node: TreeNode | None = None
    p: float = 0.0
    e: float = 0.0
    model_rank: int | None = None


def select_candidates(pool: dict[Path, Candidate], root: Path, budget: int, coefficient: float,
                      *, positive_only=False):
    """Select an ancestor-closed tree before allocating state; legacy keeps zeros."""
    pool[root].p = pool[root].e = 1.0
    ranked = []
    for path in sorted((p for p in pool if p != root), key=lambda p: (len(p), p)):
        c = pool[path]
        c.e = min(1.0, coefficient * c.raw_score) if c.score_source == "suffix" else c.raw_score
        c.p = pool[path[:-1]].p * c.e
        ranked.append(c)
    ranked.sort(key=lambda c: (-c.p, len(c.path), c.path))
    chosen = [c for c in ranked if not positive_only or c.p > 0][:budget]
    return {root: pool[root], **{c.path: c for c in chosen}}


class RecyclingTree(DraftTree):
    """Recycle dead slots, clearing every ancestry reference before reuse.

    Fresh nodes always start at step zero and are embedded again. Each layer writes
    its K/V before it can be attended to, so stale slots never supply model state.
    Slot order need not be topological; attention uses the explicit ancestor mask.
    """

    def __init__(self, max_nodes):
        super().__init__(max_nodes, 1 << 30)
        self.free: list[int] = []

    def add(self, token, parent, start, source):
        if not self.free:
            return super().add(token, parent, start, source)
        i = heapq.heappop(self.free)
        n = TreeNode(
            i,
            int(token),
            -1 if parent is None else parent.idx,
            0 if parent is None else parent.depth + 1,
            start,
            source,
        )
        if parent is not None:
            self.anc[i] = self.anc[parent.idx]
            parent.children[n.token] = i
        self.anc[i, i] = True
        self.nodes[i] = n
        self.fresh.append(n)
        return n

    def release(self, n):
        if not n.alive:
            return
        if n.parent >= 0:
            parent = self.nodes[n.parent]
            if parent.children.get(n.token) == n.idx:
                del parent.children[n.token]
        n.alive = False
        self.anc[:, n.idx] = False
        self.anc[n.idx] = False
        heapq.heappush(self.free, n.idx)

    def take_fresh(self):
        return [n for n in super().take_fresh() if n.alive]


@torch.no_grad()
def budget_tree_decode(backbone, *args, **kwargs):
    return run_single(backbone, budget_tree_decode_steps(backbone, *args, **kwargs))


def budget_tree_decode_steps(
    backbone,
    prompt,
    cfg: BudgetConfig,
    max_new_tokens: int,
    stop: set[int],
    cache=None,
    req_id=None,
    *,
    max_nodes=None,
    flush_every=16,
    observer=None,
):
    """Shared single/batch state machine; confirmed chains are flushed before reuse.

    flush_every is accepted for scheduler compatibility. This policy commits each
    ready chain immediately, requiring only N+1 tree slots even for long requests.
    """
    if not len(prompt) or max_new_tokens < 1:
        raise ValueError("prompt must be nonempty and max_new_tokens must be positive")
    N, T = cfg.draft_node_budget, cfg.depth_total
    kv = getattr(backbone, "kv", None)
    capacity = kv.TK.shape[1] if kv is not None else (max_nodes or N + 1)
    if max_nodes is None:
        max_nodes = capacity
    if max_nodes < N + 1 or max_nodes > capacity:
        raise ValueError("physical tree capacity must cover N+1 and fit the backbone buffers")
    if getattr(backbone, "T", T) != T:
        raise ValueError("model recurrence depth differs from decoder configuration")
    max_pos = getattr(kv, "max_seq", None)
    if max_pos is not None and len(prompt) + max_new_tokens > max_pos:
        raise ValueError("request exceeds max_seq; no truncation applied")
    if cfg.use_suffix and N and cache is None:
        raise ValueError("suffix decoding requires a suffix cache")
    use_suffix = cfg.use_suffix and cache is not None and N > 0
    depths = sorted(cfg.readout_depths) if cfg.use_model and N else []
    controller = SuffixController(cfg)
    # Extra physical buffers, if supplied for other arms, do not enlarge this tree.
    tree = RecyclingTree(N + 1)
    prompt = [int(t) for t in prompt]
    stats = dict(
        tokens=0,
        steps=0,
        calls=0,
        rows=0,
        max_rows=0,
        nodes=0,
        readout_rows=0,
        speculate_calls=0,
        speculate_s=0.0,
        selection_s=0.0,
        controller_s=0.0,
        peak_draft_nodes=0,
        budget_evictions=0,
        budget_rejections=0,
        candidate_replacements=0,
        wasted_rows=0,
        wasted_budget_rows=0,
        wasted_rejected_rows=0,
        wasted_end_rows=0,
        selection_events=0,
        selected_draft_nodes_sum=0,
        selected_zero_nodes_sum=0,
        child_suffix_queries=0,
        cap_skips=0,
        verified_parents=0,
        covered_tokens=0,
        accepted_lengths=[],
        peak_feedback_metadata_bytes_estimate=0,
    )
    stats.update({f"from_{s}": 0 for s in ("suffix", "fill", "sibling", "deep")})
    stats.update({f"child_suffix_rank{rank}_{metric}": 0
                  for rank in range(1, cfg.top_m + 1)
                  for metric in ("queries", "returned_nodes", "skipped")})
    pool: dict[Path, Candidate] = {}
    tick = 0

    def emit(event, **data):
        if observer is not None:
            observer(event, **data)

    def release(c, wasted=True, reason="rejected"):
        if c.node is not None:
            if wasted:
                stats["wasted_rows"] += c.node.step
                stats[f"wasted_{reason}_rows"] += c.node.step
            tree.release(c.node)

    def suffix(parent, cap=None):
        cap = cfg.suffix_room if cap is None else cap
        if cap <= 0:
            raise ValueError("zero-cap child queries must be skipped before calling the provider")
        start = time.perf_counter()
        ctx = prompt + list(parent.path)
        limit = getattr(cache, "max_tree_depth", None)
        draft = cache.speculate(
            req_id,
            np.asarray(ctx[-limit:] if limit else ctx, dtype=np.int32),
            max_spec_tokens=cap,
            max_spec_factor=cfg.spec_factor,
            max_spec_offset=0.0,
            min_token_prob=cfg.suffix_min_prob,
            use_tree_spec=True,
        )
        stats["speculate_calls"] += 1
        stats["speculate_s"] += time.perf_counter() - start
        if len(draft.token_ids) > cap:
            raise ValueError("suffix provider exceeded max_spec_tokens")
        created = []
        # Validate topology before deriving conditional scores.
        for k, par in enumerate(draft.parents):
            if par < -1 or par >= k:
                raise ValueError("suffix parents must precede children")
        edges = _edge_probs(draft)
        for tok, par, q in zip(draft.token_ids, draft.parents, edges, strict=True):
            base = parent if par < 0 else created[par]
            if base is None or (max_pos is not None and len(prompt) + len(base.path) >= max_pos):
                created.append(None)
                continue
            tok, q = int(tok), _score(q)
            controller.observe(base.path, tok, q)
            path = base.path + (tok,)
            c = pool.get(path)
            if c is None:
                c = Candidate(path, q, "suffix", "suffix", suffix=True)
                pool[path] = c
            else:
                c.suffix = True  # never override a model score (nor the first snapshot)
            created.append(c)
        return len(draft.token_ids)

    def model_candidates(c, depth, top):
        if max_pos is not None and len(prompt) + len(c.path) >= max_pos:
            return
        ids, probs = top
        seen = set()
        for rank, (token, probability) in enumerate(zip(ids, probs, strict=True)):
            if rank >= cfg.top_m:
                break
            token, probability = int(token), _score(probability)
            seen.add(token)
            path = c.path + (token,)
            if path in pool:
                pool[path].raw_score = probability
                pool[path].score_source = "model"
            else:
                origin = "deep" if depth > cfg.draft_depth else ("fill" if rank == 0 else "sibling")
                pool[path] = Candidate(path, probability, "model", origin)
            # Rank belongs to this parent's latest model readout, not admission order
            # or the candidate's original source (which may have been suffix).
            pool[path].model_rank = rank
        for path, child in pool.items():
            if path[:-1] == c.path and path[-1] not in seen:
                # Preserve the documented legacy ABSENT=0 heuristic; not true softmax.
                if depth > cfg.draft_depth or child.suffix:
                    child.raw_score, child.score_source = 0.0, "model"

    def choose(root):
        nonlocal pool
        start = time.perf_counter()
        selected = select_candidates(pool, root, N, controller.c, positive_only=cfg.positive_only)
        evicted = rejected = 0
        for path, c in pool.items():
            if path not in selected:
                if c.node is None:
                    rejected += 1
                else:
                    evicted += 1
                    release(c, reason="budget")
        new = sum(c.node is None for p, c in selected.items() if p != root)
        stats["budget_evictions"] += evicted
        stats["budget_rejections"] += rejected
        stats["candidate_replacements"] += min(evicted, new)
        pool = selected
        stats["selection_events"] += 1
        stats["selected_draft_nodes_sum"] += len(pool) - 1
        stats["selected_zero_nodes_sum"] += sum(c.p == 0 for p, c in pool.items() if p != root)
        stats["peak_draft_nodes"] = max(stats["peak_draft_nodes"], len(pool) - 1)
        stats["selection_s"] += time.perf_counter() - start

    def allocate(root):
        for path in sorted(pool, key=lambda p: (len(p), p)):
            c = pool[path]
            if c.node is None:
                parent = None if path == root else pool[path[:-1]].node
                c.node = tree.add(path[-1], parent, tick, c.origin)
                if c.node is None:
                    raise AssertionError("N+1 recyclable slots must fit the selected tree")
                stats["nodes"] += 1
        if observer is not None:
            emit(
                "selection",
                root=root,
                coefficient=controller.c,
                paths=tuple(pool),
                slots={p: c.node.idx for p, c in pool.items()},
                scores={p: (c.score_source, c.raw_score, c.e, c.p) for p, c in pool.items()},
            )

    def boundary(root, due, query_root):
        # One coefficient is used throughout this entire proposal/selection boundary.
        for c, depth, top in due:
            if c.path in pool:
                model_candidates(c, depth, top)
        if query_root and use_suffix:
            suffix(pool[root])
        choose(root)
        # Only newly selected model children trigger immediate suffix continuation.
        if use_suffix and cfg.child_suffix_policy != "none":
            children = [c for c in pool.values()
                        if c.path != root and c.node is None and c.score_source == "model"]
            if cfg.child_suffix_policy == "best1":
                children = sorted(children, key=lambda c: (-c.p, len(c.path), c.path))[:1]
            for c in children:
                rank = c.model_rank
                # Legacy ABSENT=0 entries can have model scores without ever
                # appearing in top-k. Preserve their existing query behavior.
                cap = cfg.suffix_room if cfg.child_suffix_caps is None or rank is None else cfg.child_suffix_caps[rank]
                if cap == 0:
                    stats[f"child_suffix_rank{rank + 1}_skipped"] += 1
                    emit("child_suffix_query", path=c.path, rank=rank, cap=0, returned_nodes=0)
                    continue
                returned = suffix(c, cap)
                stats["child_suffix_queries"] += 1
                if rank is not None:
                    stats[f"child_suffix_rank{rank + 1}_queries"] += 1
                    stats[f"child_suffix_rank{rank + 1}_returned_nodes"] += returned
                emit("child_suffix_query", path=c.path, rank=rank, cap=cap, returned_nodes=returned)
            choose(root)
        allocate(root)
        if cfg.trace_controller:
            stats["peak_feedback_metadata_bytes_estimate"] = max(
                stats["peak_feedback_metadata_bytes_estimate"], controller.metadata_bytes()
            )

    yield ModelCall("synchronize")
    t0 = time.perf_counter()
    out = [int((yield ModelCall("prefill", (prompt, cfg.draft_depth))))]
    t1 = time.perf_counter()
    started = False
    try:
        if use_suffix:
            cache.start_request(req_id, np.asarray(prompt, dtype=np.int32))
            started = True
            cache.add_active_response(req_id, np.asarray(out, dtype=np.int32))
        root = tuple(out)
        tree.base_pos = backbone.length
        pool[root] = Candidate(root, 1.0, "model", "root")
        if len(out) < max_new_tokens and out[-1] not in stop:
            boundary(root, [], True)
        while len(out) < max_new_tokens and out[-1] not in stop:
            fresh = tree.take_fresh()
            if fresh:
                yield ModelCall("embed", (tree, fresh))
            live = sorted(
                (c.node for c in pool.values() if c.node.step < T),
                key=lambda n: (n.step, n.depth, n.idx),
            )
            assert len(pool) - 1 <= N and len(live) <= N + 1
            yield ModelCall("advance", (tree, live))
            tick += 1
            for n in live:
                n.step += 1
            stats["calls"] += 1
            stats["rows"] += len(live)
            stats["max_rows"] = max(stats["max_rows"], len(live))
            groups = [(d, [n for n in live if n.step == d]) for d in depths]
            full = [n for n in live if n.step == T]
            reads, tops = yield ModelCall("readout_top", (tree, groups, full, cfg.top_m))
            stats["readout_rows"] += sum(len(g) for _, g in groups) + len(full)
            by_slot = {c.node.idx: c for c in pool.values()}
            due = [
                (by_slot[n.idx], d, top)
                for (d, group), group_tops in zip(groups, reads, strict=True)
                for n, top in zip(group, group_tops, strict=True)
            ]
            for n, gold in zip(full, tops, strict=True):
                n.full_top = int(gold)
                controller.mark_known(by_slot[n.idx].path)

            chain, old_roots, burst = [], [], 0
            # Consume every ready correct node before any rescoring can evict it.
            while pool[root].node is not None and pool[root].node.full_top is not None:
                c = pool[root]
                gold = c.node.full_top
                start = time.perf_counter()
                feedback = controller.verify(root, gold)
                stats["controller_s"] += time.perf_counter() - start
                emit("verify", parent=root, gold=gold, output_position=len(out), feedback=feedback)
                stats["verified_parents"] += 1
                stats["tokens"] += 1
                burst += 1
                out.append(gold)
                if use_suffix:
                    cache.add_active_response(req_id, np.asarray([gold], dtype=np.int32))
                nxt = root + (gold,)
                if nxt in pool:
                    stats["covered_tokens"] += 1
                    stats[f"from_{pool[nxt].origin}"] += 1
                chain.append(c.node)
                old_roots.append(c)
                kept = {}
                for path, candidate in pool.items():
                    if path[: len(nxt)] == nxt:
                        kept[path] = candidate
                    elif path != root:
                        release(candidate)
                pool = kept
                root = nxt
                controller.retain_prefix(root)
                if root not in pool:
                    pool[root] = Candidate(root, 1.0, "model", "root")
                if out[-1] in stop or len(out) >= max_new_tokens:
                    break
            if chain:
                backbone.commit(tree, chain)
                for c in old_roots:
                    release(c, wasted=False)
                # All survivors descend from the new root; physical positions stay fixed.
                shift = len(chain)
                tree.base_pos += shift
                for c in pool.values():
                    if c.node is not None:
                        c.node.depth -= shift
                if pool[root].node is not None:
                    pool[root].node.parent = -1
                stats["steps"] += 1
                stats["accepted_lengths"].append(burst)
            if out[-1] in stop or len(out) >= max_new_tokens:
                break
            if due or chain:
                boundary(
                    root, sorted(due, key=lambda x: (len(x[0].path), x[0].path, x[1])), bool(chain)
                )

        for path, c in pool.items():
            if path != root:
                release(c, reason="end")
        yield ModelCall("synchronize")
        t2 = time.perf_counter()
        if started:
            cache.stop_request(req_id)
            started = False
        stats.update(
            prefill_s=t1 - t0,
            decode_s=t2 - t1,
            tree_policy="node_budget",
            scoring=cfg.scoring,
            draft_node_budget=N,
            controller_scope="request",
            update_event="verified_parent",
            c_initial=cfg.c_initial,
            c_final=controller.c,
            c_observed_min=controller.c_low,
            c_observed_max=controller.c_high,
            eta=cfg.eta,
            c_min=cfg.c_min,
            c_max=cfg.c_max,
            controller_updates=controller.updates,
            feedback_records=controller.feedback_records,
            feedback_created=controller.created,
            feedback_discarded=controller.discarded,
            feedback_unresolved=controller.pending,
            peak_feedback_records=controller.peak_records,
            peak_feedback_parents=controller.peak_parents,
            controller_trace=controller.trace,
            commit_policy="ready_chain",
            metadata_profiled=cfg.trace_controller,
        )
        return out, stats
    finally:
        if started and hasattr(cache, "abort_request"):
            cache.abort_request(req_id)
        controller.records.clear()
        controller.known.clear()
