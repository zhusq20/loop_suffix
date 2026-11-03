"""Shared-weight loop inference with fixed or continuously refilled slots.

Projection/MLP, embedding and readout computations concatenate request rows.
FusedLoopExecutor is the eager per-request attention reference; the optional
PackedLoopExecutor combines attention and replays warmed CUDA graphs. The scheduler
can interleave prefill chunks and decode; all measured prefill work is included in E2E.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .loop_prob_decode import prob_tree_decode_steps
from .loop_budget_decode import BudgetConfig, budget_tree_decode_steps
from .loop_loopspec_decode import LoopSpecConfig, loopspec_decode_steps
from .loop_prob_tree import ProbConfig
from .loop_tree_decode import (
    HuginnLoop,
    OuroLoop,
    _split,
    _top,
    _top_probs,
    make_rows,
    tree_attention,
    tree_decode_steps,
)


class FusedLoopExecutor:
    """Execute one operation across distinct request states sharing one model."""

    def __init__(self, backbones):
        if not backbones:
            raise ValueError("at least one backbone is required")
        self.first = backbones[0]
        if type(self.first) not in (OuroLoop, HuginnLoop):
            raise TypeError("fused executor supports OuroLoop and HuginnLoop")
        model = getattr(self.first, "lm", getattr(self.first, "model", None))
        for b in backbones:
            if (
                type(b) is not type(self.first)
                or b.T != self.first.T
                or getattr(b, "lm", getattr(b, "model", None)) is not model
            ):
                raise ValueError("backbones must share the same model and recurrence depth")
        if len({id(b.kv) for b in backbones}) != len(backbones):
            raise ValueError("each request slot needs its own KV cache")
        self.core_calls = 0
        self.core_rows = 0
        self.max_core_rows = 0

    def execute(self, name, jobs):
        # jobs: [(backbone, ModelCall)]. Results are returned in input order.
        if name == "synchronize":
            self.first.synchronize()
            return [None] * len(jobs)
        if name == "prefill":
            return [b.prefill(*c.args) for b, c in jobs]
        if name == "embed" and isinstance(self.first, OuroLoop):
            sizes = [len(c.args[1]) for _, c in jobs]
            ids = self.first.kv.upload(
                np.asarray([n.token for _, c in jobs for n in c.args[1]], dtype=np.int64)
            )
            values = self.first.inner.embed_tokens(ids)
            for (b, c), x in zip(jobs, values.split(sizes), strict=True):
                idx = b.kv.upload(np.asarray([n.idx for n in c.args[1]], dtype=np.int64))
                b.hidden[idx] = x
            return [None] * len(jobs)
        return self._rows(name, jobs)

    def _rows(self, name, jobs):
        plans = []
        for j, (b, call) in enumerate(jobs):
            tree = call.args[0]
            if name in ("advance", "embed"):
                nodes = call.args[1]
                if name == "embed":
                    bases = [0] * len(nodes)
                elif isinstance(b, OuroLoop):
                    bases = [n.step * b.n_layers for n in nodes]
                else:
                    bases = [b.n_pre + n.step * b.n_core for n in nodes]
                sizes = None
            elif name == "readout_top":
                _, groups, full, m = call.args
                sizes = [len(g) for _, g in groups]
                nodes = [n for _, g in groups for n in g] + full
                bases = (
                    (
                        [b.coda_extra.get(r, b.coda_shallow) for r, g in groups for _ in g]
                        + [b.coda_full] * len(full)
                    )
                    if isinstance(b, HuginnLoop)
                    else [0] * len(nodes)
                )
            elif name == "readout":
                _, shallow, full, m, shared = call.args
                sizes = [len(shallow)]
                nodes = shallow + full
                bases = (
                    (
                        [b.coda_full if shared else b.coda_shallow] * len(shallow)
                        + [b.coda_full] * len(full)
                    )
                    if isinstance(b, HuginnLoop)
                    else [0] * len(nodes)
                )
            else:
                raise ValueError(name)
            if nodes:
                rows = make_rows(b.kv, tree, nodes, bases)
                plans.append((j, b, rows, nodes, sizes))
        results = [None] * len(jobs)
        if name == "readout_top":
            results = [([[] for _ in c.args[1]], []) for _, c in jobs]
        elif name == "readout":
            results = [([], []) for _ in jobs]
        if not plans:
            return results
        sizes = [len(rows.nodes) for _, _, rows, _, _ in plans]
        pos = torch.cat([rows.pos for _, _, rows, _, _ in plans])
        b0 = self.first

        def attend(layer):
            def run(q, k, v):
                qs, ks, vs = (t.split(sizes) for t in (q, k, v))
                return torch.cat(
                    [
                        tree_attention(qi, ki, vi, b.kv, rows, layer)
                        for (_, b, rows, _, _), qi, ki, vi in zip(plans, qs, ks, vs, strict=True)
                    ]
                )

            return run

        if name == "embed":  # Huginn prelude, including its request-specific tree mask
            ids = b0.kv.upload(
                np.asarray(
                    [n.token for _, _, rows, _, _ in plans for n in rows.nodes], dtype=np.int64
                )
            )
            x = b0._embed_tokens(ids[None])
            freqs = b0.model.freqs_cis[:, pos]
            for li, block in enumerate(b0.prelude):
                x = b0._block(block, x, freqs, attend(li))
            for (_, b, rows, _, _), part in zip(plans, x[0].split(sizes), strict=True):
                b.static[rows.idx] = part
                b.hidden[rows.idx] = b.init[rows.pos]
            return results

        x = torch.cat([b.hidden[rows.idx] for _, b, rows, _, _ in plans])[None]
        if name == "advance":
            self.core_calls += 1
            self.core_rows += sum(sizes)
            self.max_core_rows = max(self.max_core_rows, sum(sizes))
            if isinstance(b0, OuroLoop):
                cos, sin = b0.inner.rotary_emb(x, pos[None])
                for li, layer in enumerate(b0.layers):
                    x = b0._layer(layer, x, cos, sin, attend(li))
                x = b0.inner.norm(x)
            else:
                static = torch.cat([b.static[r.idx] for _, b, r, _, _ in plans])[None]
                x = b0.model.transformer.adapter(torch.cat([x, static], dim=-1))
                for li, block in enumerate(b0.core):
                    x = b0._block(block, x, b0.model.freqs_cis[:, pos], attend(li))
            for (_, b, rows, _, _), part in zip(plans, x[0].split(sizes), strict=True):
                b.hidden[rows.idx] = part
            return results

        if isinstance(b0, HuginnLoop):
            x = b0._coda(x, b0.model.freqs_cis[:, pos], attend)
            logits = b0.model.lm_head(b0.model.transformer.ln_f(x[0])).float()
        else:
            logits = b0.lm.lm_head(x[0]).float()
        # Reorder from attention slot order back to each decoder's readout order.
        for (j, b, rows, nodes, group_sizes), part in zip(plans, logits.split(sizes), strict=True):
            at = {n.idx: i for i, n in enumerate(rows.nodes)}
            order = b.kv.upload(np.asarray([at[n.idx] for n in nodes], dtype=np.int64))
            part = part[order]
            k = sum(group_sizes)
            m = jobs[j][1].args[3]
            if name == "readout_top":
                results[j] = (
                    _split(part[:k].argmax(-1).tolist() if m == 0 else
                           _top_probs(part[:k], m), group_sizes),
                    part[k:].argmax(-1).tolist(),
                )
            else:
                tops = _top(part, m)
                results[j] = (tops[:k], tops[k:])
        return results


@dataclass
class BatchRequest:
    request_id: str | int
    prompt: list[int]
    max_new_tokens: int
    stop: set[int] = field(default_factory=set)


@torch.no_grad()
def batch_decode(
    requests,
    backbones,
    config,
    calibration=None,
    cache=None,
    *,
    max_nodes=None,
    flush_every=16,
    executor=None,
    schedule="fixed",
    prefill_chunk_size=0,
):
    """Run a finite stream with fixed batches or continuous slot replenishment.

    Continuous mode admits queued requests into free slots at the next scheduler
    iteration. With positive prefill_chunk_size, round-robin prompt chunks alternate
    with shared decode advances. Otherwise each prompt is prefetched in full.
    The finite input queue is available from time zero.

    Model weights and slot buffers must be allocated before this call. Timing starts
    before admission and includes prefills, CPU scheduling, suffix updates
    and GPU work. Requests' latency starts at slot admission, not queue arrival.
    No offline answers are injected. The supplied cache must isolate active outputs.
    """
    if schedule not in ("fixed", "continuous"):
        raise ValueError("schedule must be fixed or continuous")
    if prefill_chunk_size < 0:
        raise ValueError("prefill_chunk_size must be nonnegative")
    requests = list(requests)
    if not backbones or not requests:
        raise ValueError("requests and slots must be nonempty")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    caches = [b.kv for b in backbones if getattr(b, "kv", None) is not None]
    if max_nodes is None and caches:
        max_nodes = min(kv.TK.shape[1] for kv in caches)
    for b in backbones:
        kv = getattr(b, "kv", None)
        if kv is not None and max_nodes is not None and max_nodes > kv.TK.shape[1]:
            raise ValueError("max_nodes exceeds a slot tree buffer")
        if getattr(b, "T", config.depth_total) != config.depth_total:
            raise ValueError("model recurrence depth differs from decoder configuration")
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    for r in requests:
        if not len(r.prompt) or r.max_new_tokens < 1:
            raise ValueError("nonempty prompts and positive output budgets required")
        if any(len(r.prompt) + r.max_new_tokens > kv.max_seq for kv in caches):
            raise ValueError(f"request {r.request_id} exceeds max_seq (no silent truncation)")
    if isinstance(config, ProbConfig) and calibration is None:
        raise ValueError("probability decoder requires calibration")
    if isinstance(config, BudgetConfig) and max_nodes is not None:
        if max_nodes < config.draft_node_budget + 1:
            raise ValueError("budget decoder requires at least N+1 physical tree slots")
    needs_suffix = config.use_suffix and not (
        isinstance(config, BudgetConfig) and config.draft_node_budget == 0
    )
    if needs_suffix and cache is None:
        raise ValueError("suffix decoding requires an isolated suffix cache")
    engine = executor if executor is not None else FusedLoopExecutor(backbones)
    if hasattr(engine, "core_calls"):
        engine.core_calls = engine.core_rows = engine.max_core_rows = 0
    if hasattr(engine, "reset_metrics"):
        engine.reset_metrics()
    prefill_chunks = prefill_tokens = 0
    last_prefill_slot = -1
    chunk_since_advance = False
    active, completed = {}, []
    next_request = 0
    max_active_requests = 0
    refilled_requests = 0
    backbones[0].synchronize()
    start = time.perf_counter()

    def resume(slot, value=None):
        state = active[slot]
        try:
            state["call"] = state["steps"].send(value)
        except StopIteration as done:
            end = time.perf_counter()
            tokens, stats = done.value
            completed.append(
                dict(
                    request_id=state["req"].request_id,
                    slot=slot,
                    tokens=tokens,
                    n_tokens=len(tokens),
                    stats=stats,
                    admitted_s=state["start"] - start,
                    completed_s=end - start,
                    latency_s=end - state["start"],
                    ttft_s=state["first"] - state["start"],
                )
            )
            del active[slot]

    def admit():
        nonlocal next_request, max_active_requests, refilled_requests
        if active and schedule == "fixed":
            return
        free = [i for i in range(len(backbones)) if i not in active]
        replenishing = bool(active)
        admitted = time.perf_counter()
        for slot in free:
            if next_request == len(requests):
                break
            req = requests[next_request]
            next_request += 1
            refilled_requests += int(replenishing)
            b = backbones[slot]
            common = dict(cache=cache, req_id=req.request_id, flush_every=flush_every)
            if max_nodes is not None:
                common["max_nodes"] = max_nodes
            if isinstance(config, LoopSpecConfig):
                steps = loopspec_decode_steps(
                    b, req.prompt, config, req.max_new_tokens, req.stop, **common
                )
            elif isinstance(config, BudgetConfig):
                steps = budget_tree_decode_steps(
                    b, req.prompt, config, req.max_new_tokens, req.stop, **common
                )
            elif isinstance(config, ProbConfig):
                steps = prob_tree_decode_steps(
                    b, req.prompt, config, calibration, req.max_new_tokens, req.stop, **common
                )
            else:
                steps = tree_decode_steps(
                    b,
                    req.prompt,
                    config,
                    req.max_new_tokens,
                    req.stop,
                    shared_coda=config.name == "wavefront",
                    **common,
                )
            active[slot] = dict(req=req, steps=steps, start=admitted, first=None, prefill_offset=0)
            resume(slot)
        max_active_requests = max(max_active_requests, len(active))

    try:
        while active or next_request < len(requests):
            admit()
            if not active:
                continue
            # Drain per-request bookkeeping/readouts until all survivors are ready
            # for the next shared recurrent block. This prevents a fast request
            # from running ahead and turning the batch into serial inference.
            pending = [s for s in active if active[s]["call"].name != "advance"]
            name = active[pending[0]]["call"].name if pending else "advance"
            if prefill_chunk_size:
                # Drain bookkeeping, then alternate one prefill chunk and one
                # shared decode advance. Rotate prefilling slots fairly.
                bookkeeping = [s for s in active if active[s]["call"].name not in ("advance", "prefill")]
                prefilling = sorted(s for s in active if active[s]["call"].name == "prefill")
                advancing = any(active[s]["call"].name == "advance" for s in active)
                if bookkeeping:
                    name = active[bookkeeping[0]]["call"].name
                elif prefilling and (not chunk_since_advance or not advancing):
                    from .loop_optimized_decode import prefill_chunk
                    slot = next((s for s in prefilling if s > last_prefill_slot), prefilling[0])
                    state = active[slot]
                    offset = state["prefill_offset"]
                    end, token = prefill_chunk(backbones[slot], *state["call"].args, offset, prefill_chunk_size)
                    state["prefill_offset"] = end
                    last_prefill_slot = slot
                    chunk_since_advance = True
                    prefill_chunks += 1
                    prefill_tokens += end - offset
                    if token is not None:
                        state["first"] = time.perf_counter()
                        resume(slot, token)
                    continue
                else:
                    name = "advance"
            slots = [s for s in active if active[s]["call"].name == name]
            if name == "prefill":
                slots = slots[:1]  # timestamp each sequential prefill immediately
            jobs = [(backbones[s], active[s]["call"]) for s in slots]
            if not jobs:
                continue
            values = engine.execute(name, jobs)
            if name == "advance":
                chunk_since_advance = False
            for slot, value in zip(slots, values, strict=True):
                if name == "prefill":
                    active[slot]["first"] = time.perf_counter()
                resume(slot, value)
    finally:
        for state in active.values():
            state["steps"].close()
            if cache is not None and hasattr(cache, "abort_request"):
                cache.abort_request(state["req"].request_id)
    backbones[0].synchronize()
    wall = time.perf_counter() - start
    order = {r.request_id: i for i, r in enumerate(requests)}
    completed.sort(key=lambda r: order[r["request_id"]])
    total = sum(r["n_tokens"] for r in completed)
    summary = dict(
        schedule=schedule,
        concurrency=len(backbones),
        max_active_requests=max_active_requests,
        requests=len(completed),
        output_tokens=total,
        wall_s=wall,
        output_tput=total / wall,
        request_tput=len(completed) / wall,
        prefill_policy="chunked_interleaved" if prefill_chunk_size else "sequential",
        prefill_chunk_size=prefill_chunk_size,
        prefill_chunks=prefill_chunks,
        prefill_chunk_tokens=prefill_tokens,
        attention_policy="per_request_tree",
        latency_origin="slot_admission",
        refilled_requests=refilled_requests,
        average_active_requests=sum(r["latency_s"] for r in completed) / wall,
        slot_utilization=sum(r["latency_s"] for r in completed) / (wall * len(backbones)),
    )
    for metric in ("core_calls", "core_rows", "max_core_rows"):
        summary[metric] = getattr(engine, metric, None)
    if hasattr(engine, "metrics"):
        summary.update(engine.metrics())
    for metric in ("latency_s", "ttft_s"):
        vals = [r[metric] for r in completed]
        summary[metric] = {
            f"p{p}": float(np.percentile(vals, p)) if vals else None for p in (50, 95)
        }
    return completed, summary
