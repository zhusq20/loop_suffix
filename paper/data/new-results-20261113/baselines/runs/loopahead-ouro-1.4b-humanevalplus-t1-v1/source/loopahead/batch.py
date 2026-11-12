"""Continuous LoopAhead batching on the frozen Table 2 packed target executor.

Each slot owns its prefix states, vocabulary, tree and KV views. The existing
LoopAhead decode generator is unchanged. Target rows and LAD transformer calls
are batched; each request retains its own restricted output vocabulary.
"""
from collections import defaultdict
from types import SimpleNamespace
import time

import torch
from torch.nn import functional as F

from llm_memory_editability.loop_decode_steps import ModelCall
from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, prefill_chunk
from llm_memory_editability.loop_tree_decode import HuginnLoop
from .decode import decode_steps
from .lad import RequestVocabulary


class BatchedLoopAheadExecutor(PackedLoopExecutor):
    def __init__(self, backbones, lad, frequent_ids):
        super().__init__(backbones, cuda_graph=False)
        self.lad, self.frequent_ids = lad, frequent_ids
        self.head = self.first.model.lm_head if isinstance(self.first, HuginnLoop) else self.first.lm.lm_head
        self.embedding = self.first.model.transformer.wte if isinstance(self.first, HuginnLoop) else self.first.inner.embed_tokens
        self.requests = {}
        self.lad_batches = self.lad_rows = self.max_lad_batch = 0
        self.max_advance_requests = 0

    def start_request(self, b, req, cfg):
        if cfg.temperature != 0:
            raise ValueError('This Table 2 runner implements greedy decoding only')
        state = SimpleNamespace(prefix_states=None, vocab=None, depth=cfg.first,
            stops=req.stop, generator=torch.Generator(device=b.hidden.device).manual_seed(0))
        self.requests[id(b)] = state
        return state

    def reset_metrics(self):
        super().reset_metrics()
        self.lad_batches = self.lad_rows = self.max_lad_batch = 0
        self.max_advance_requests = 0

    def metrics(self):
        return dict(super().metrics(), lad_batches=self.lad_batches, lad_rows=self.lad_rows,
            max_lad_batch=self.max_lad_batch, max_advance_requests=self.max_advance_requests)

    def prefill(self, b, prompt, depth, offset, size):
        state, captured = self.requests[id(b)], []
        if isinstance(b, HuginnLoop):
            original = b._coda
            def coda(*args, **kwargs):
                result = original(*args, **kwargs)
                if not captured:
                    captured.append(b.model.transformer.ln_f(result)[0].detach().clone())
                return result
            b._coda = coda
            hook = None
        else:
            hook = b.inner.norm.register_forward_hook(
                lambda module, inputs, output: captured.append(output[0].detach().clone()))
        try:
            end, token = prefill_chunk(b, prompt, depth, offset, size)
        finally:
            if hook is not None:
                hook.remove()
            else:
                b._coda = original
        shallow = captured[0] if isinstance(b, HuginnLoop) else captured[depth - 1]
        state.prefix_states = (shallow if offset == 0 else
            torch.cat([state.prefix_states, shallow], 0))[-64:].clone()
        if token is not None:
            state.vocab = RequestVocabulary(self.head, self.frequent_ids, prompt, state.stops)
            state.vocab.commit([token])
        return end, token

    def context(self, b, tree, node):
        prefix = self.requests[id(b)].prefix_states
        branch = []
        while node is not None:
            state = getattr(node, 'shallow_state', None)
            if state is None:
                return None
            branch.append(state)
            node = tree.nodes[node.parent] if node.parent >= 0 else None
        if len(prefix) + len(branch) < 64:
            return None
        return torch.cat([prefix, torch.stack(branch[::-1])], 0)[-64:]

    def execute(self, name, jobs):
        if name == 'lad_commit':
            for b, call in jobs:
                tree, node, token = call.args
                state = self.requests[id(b)]
                if getattr(node, 'shallow_state', None) is None:
                    raise RuntimeError('Committed node lacks a real shallow state')
                state.prefix_states = torch.cat([state.prefix_states, node.shallow_state[None]], 0)[-64:]
                state.vocab.commit([token])
                b.commit(tree, [node])
            return [None] * len(jobs)
        if name == 'lad_propose':
            groups, results = defaultdict(list), [([], []) for _ in jobs]
            for i, (b, call) in enumerate(jobs):
                tree, parent, anchor, count = call.args
                context = self.context(b, tree, parent)
                if context is not None:
                    groups[count].append((i, b, anchor.token, context))
            for count, group in groups.items():
                anchors = torch.tensor([x[2] for x in group], device=self.first.hidden.device)
                hidden = self.lad(torch.stack([x[3] for x in group]), self.embedding(anchors), count)
                self.lad_batches += 1
                self.lad_rows += len(group)
                self.max_lad_batch = max(self.max_lad_batch, len(group))
                for (i, b, _, _), values in zip(group, hidden, strict=True):
                    vocab = self.requests[id(b)].vocab
                    logits = F.linear(values, vocab.weight, vocab.bias).float()
                    indices = logits.argmax(-1)
                    confidence = logits.softmax(-1).gather(-1, indices[:, None])[:, 0]
                    results[i] = (vocab.ids[indices].tolist(), confidence.tolist())
            return results
        if name == 'lad_readout':
            plans, target_jobs, total = [], [], 0
            for b, call in jobs:
                tree, groups, full = call.args
                nodes = [n for _, group in groups for n in group] + full
                bases = ([b.coda_extra.get(d, b.coda_shallow) for d, group in groups for _ in group]
                         + [b.coda_full] * len(full)) if isinstance(b, HuginnLoop) else [0] * len(nodes)
                ordered = [nodes[i] for i in sorted(range(len(nodes)), key=lambda i: bases[i])]
                plans.append((b, groups, full, ordered))
                target_jobs.append((b, ModelCall('readout_top', (tree, groups, full, 0))))
                total += len(nodes)
            if not total:
                return [({}, {}) for _ in jobs]
            captured = []
            hook = self.head.register_forward_hook(
                lambda module, inputs, output: captured.append((inputs[0].detach(), output.detach().float())))
            try:
                super().execute('readout_top', target_jobs)
            finally:
                hook.remove()
            if len(captured) != 1 or len(captured[0][0]) != total:
                raise RuntimeError('Packed target readout layout changed')
            states, logits = captured[0]
            results, offset = [], 0
            for b, groups, full, nodes in plans:
                mapped = {n.idx: (states[offset+i], logits[offset+i]) for i, n in enumerate(nodes)}
                offset += len(nodes)
                reads = {}
                for depth, group in groups:
                    for node in group:
                        hidden, logit = mapped[node.idx]
                        reads[node.idx] = (depth, logit)
                        if depth == self.requests[id(b)].depth:
                            node.shallow_state = hidden.clone()
                results.append((reads, {n.idx: mapped[n.idx][1] for n in full}))
            return results
        if name == 'advance':
            self.max_advance_requests = max(self.max_advance_requests, len(jobs))
        return super().execute(name, jobs)


@torch.inference_mode()
def batch_decode(requests, backbones, cfg, engine, *, chunk_size=128, progress_callback=None):
    """Table 2 timing and refill policy; all requests are queued at time zero."""
    requests = list(requests)
    if not requests or not backbones or chunk_size < 1:
        raise ValueError('Nonempty requests/slots and positive prefill chunk required')
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError('Request IDs must be unique')
    for req in requests:
        if not req.prompt or req.max_new_tokens < 1:
            raise ValueError('Invalid prompt or output budget')
        if any(len(req.prompt) + req.max_new_tokens > b.kv.max_seq for b in backbones):
            raise ValueError('Request exceeds capacity; truncation is not allowed')
    engine.reset_metrics()
    active, completed = {}, []
    next_request = max_active = refilled = chunks = chunk_tokens = 0
    last_prefill, chunk_since_advance = -1, False
    backbones[0].synchronize()
    started = time.perf_counter()

    def resume(slot, value=None):
        state = active[slot]
        try:
            state['call'] = state['steps'].send(value)
        except StopIteration as done:
            end = time.perf_counter()
            tokens, stats = done.value
            completed.append(dict(request_id=state['req'].request_id, slot=slot, tokens=tokens,
                n_tokens=len(tokens), stats=stats, admitted_s=state['start']-started,
                completed_s=end-started, latency_s=end-state['start'], ttft_s=state['first']-state['start']))
            del active[slot]
            if progress_callback is not None:
                progress_callback(completed, end-started)

    try:
        while active or next_request < len(requests):
            replenishing = bool(active)
            admitted = time.perf_counter()
            for slot, b in enumerate(backbones):
                if slot in active or next_request == len(requests):
                    continue
                req = requests[next_request]
                next_request += 1
                refilled += int(replenishing)
                state = engine.start_request(b, req, cfg)
                active[slot] = dict(req=req, start=admitted, first=None, offset=0,
                    steps=decode_steps(b, req.prompt, cfg, req.max_new_tokens, req.stop, generator=state.generator))
                resume(slot)
            max_active = max(max_active, len(active))
            bookkeeping = [s for s in active if active[s]['call'].name not in ('advance', 'prefill')]
            prefilling = sorted(s for s in active if active[s]['call'].name == 'prefill')
            advancing = any(active[s]['call'].name == 'advance' for s in active)
            if bookkeeping:
                name = active[bookkeeping[0]]['call'].name
            elif prefilling and (not chunk_since_advance or not advancing):
                slot = next((s for s in prefilling if s > last_prefill), prefilling[0])
                state = active[slot]
                offset = state['offset']
                end, token = engine.prefill(backbones[slot], *state['call'].args, offset, chunk_size)
                state['offset'], last_prefill, chunk_since_advance = end, slot, True
                chunks += 1
                chunk_tokens += end-offset
                if token is not None:
                    state['first'] = time.perf_counter()
                    resume(slot, token)
                continue
            else:
                name = 'advance'
            slots = [s for s in active if active[s]['call'].name == name]
            values = engine.execute(name, [(backbones[s], active[s]['call']) for s in slots])
            if name == 'advance':
                chunk_since_advance = False
            for slot, value in zip(slots, values, strict=True):
                resume(slot, value)
    finally:
        for state in active.values():
            state['steps'].close()
    backbones[0].synchronize()
    wall = time.perf_counter()-started
    order = {r.request_id:i for i,r in enumerate(requests)}
    completed.sort(key=lambda r: order[r['request_id']])
    output_tokens = sum(r['n_tokens'] for r in completed)
    calls = sum(r['stats']['calls'] for r in completed)
    summary = dict(requests=len(completed), concurrency=len(backbones), schedule='continuous',
        output_tokens=output_tokens, input_tokens=sum(len(r.prompt) for r in requests), wall_s=wall,
        output_tput=output_tokens/wall, total_tput=(output_tokens+sum(len(r.prompt) for r in requests))/wall,
        gamma=cfg.total*sum(r['stats']['tokens'] for r in completed)/calls if calls else None,
        max_active_requests=max_active, refilled_requests=refilled, prefill_chunk_size=chunk_size,
        prefill_chunks=chunks, prefill_chunk_tokens=chunk_tokens,
        core_calls=engine.core_calls, core_rows=engine.core_rows, max_core_rows=engine.max_core_rows,
        **engine.metrics())
    return completed, summary
