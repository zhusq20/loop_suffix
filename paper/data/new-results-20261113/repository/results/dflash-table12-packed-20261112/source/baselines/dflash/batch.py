"""Continuous DFlash with packed target verification and batched draft SDPA.

Target KV is a disjoint view per slot. Drafter contexts use explicit masked
padding; noise positions are attended but never retained in the drafter cache.
No target execution-row padding, graphs, compilation, or cross-request history.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import time

import torch
from transformers import DynamicCache

from llm_memory_editability.loop_decode_steps import ModelCall
from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, SharedKVPool
from .decode import probabilities, rejection_sample, sample_logits, sample_probabilities, sync_time
from .packed_target import PackedDFlashTarget, PackedTargetOutput


class BatchedDFlashTarget:
    def __init__(self, target, batch_size, block_size=16):
        if batch_size < 1:
            raise ValueError('Positive batch size required')
        pool = SharedKVPool(batch_size, pad_decode_rows=False)
        self.slots = [PackedDFlashTarget(target, block_size, pool=pool) for _ in range(batch_size)]
        self.engine = PackedLoopExecutor([s.backbone for s in self.slots], cuda_graph=False)
        self.max_verify_batch = 0

    def close(self):
        for slot in self.slots:
            slot.close()

    @torch.inference_mode()
    def verify(self, jobs, *, features=True):
        """jobs = [(slot index, input IDs, current owned cache)], in output order."""
        if not jobs or len({i for i, _, _ in jobs}) != len(jobs):
            raise ValueError('Nonempty distinct request slots required')
        sizes, selected = [], []
        for i, ids, cache in jobs:
            slot = self.slots[i]
            slot._check_cache(cache)
            if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= ids.shape[1] <= slot.block_size:
                raise ValueError('Invalid verification block')
            start = cache.get_seq_length()
            if not start or start + ids.shape[1] > slot.max_seq:
                raise ValueError('Verification needs a prefilled prefix and available context capacity')
            selected.append(slot)
            sizes.append(ids.shape[1])
        for slot, (_, ids, _) in zip(selected, jobs):
            slot.tree.reset(slot.backbone.kv.length)
            parent = None
            for token in ids[0].tolist():
                parent = slot.tree.add(token, parent, 0, 'dflash')
        observer = self.slots[0]  # PackedLoopExecutor applies physical layers via its first backbone.
        observer._counts, observer._features, observer._logits = {}, {}, None
        observer._active, observer._capture = True, features
        try:
            calls = [(s.backbone, ModelCall('embed', (s.tree, s.tree.nodes))) for s in selected]
            self.engine.execute('embed', calls)
            calls = [(s.backbone, ModelCall('advance', (s.tree, s.tree.nodes))) for s in selected]
            for _ in range(observer.depth):
                self.engine.execute('advance', calls)
                for s in selected:
                    for node in s.tree.nodes:
                        node.step += 1
            self.engine.execute('readout_top', [(s.backbone, ModelCall(
                'readout_top', (s.tree, [], s.tree.nodes, 0))) for s in selected])
            if features and observer._counts != observer.capture_at:
                raise RuntimeError('Incomplete batched target features')
            logits = observer._logits.reshape(1, sum(sizes), -1).split(sizes, dim=1)
            hidden = (torch.cat([observer._features[i] for i in observer.physical_ids], -1)
                      .split(sizes, dim=1)) if features else [None] * len(jobs)
            for s in selected:
                s.backbone.commit(s.tree, s.tree.nodes)
            self.max_verify_batch = max(self.max_verify_batch, len(jobs))
            return [PackedTargetOutput(l, h, c) for l, h, (_, _, c) in zip(logits, hidden, jobs)]
        except Exception:
            for s in selected:
                s._cache = None
            raise
        finally:
            observer._active = observer._capture = False
            observer._features.clear()
            observer._logits = None


class DraftBatchCache:
    """Join independent DynamicCaches for one padded attention call per layer.

Only real new target features are appended to each owned cache. Temporary
noise KV follows the padded prefix in the attention tensor and is discarded.
"""
    def __init__(self, caches, context_lengths, starts, padded_context, width):
        self.caches, self.context_lengths, self.starts = caches, context_lengths, starts
        self.padded_context, self.width = padded_context, width

    def update(self, key, value, layer_idx, cache_kwargs=None):
        batch, heads, _, dim = key.shape
        prefix = max(self.starts)
        keys = key.new_zeros(batch, heads, prefix + self.width, dim)
        values = value.new_zeros(keys.shape)
        for i, (cache, length, start) in enumerate(zip(self.caches, self.context_lengths, self.starts)):
            k, v = cache.update(key[i:i+1, :, :length], value[i:i+1, :, :length], layer_idx)
            if k.shape[-2] != start:
                raise RuntimeError('Drafter cache lost request alignment')
            keys[i, :, :start], values[i, :, :start] = k[0], v[0]
            keys[i, :, prefix:] = key[i, :, self.padded_context:]
            values[i, :, prefix:] = value[i, :, self.padded_context:]
        return keys, values


@torch.inference_mode()
def draft_batch(target, draft, features, blocks, caches, starts):
    """Same-width blocks, arbitrary prompt/accepted-context lengths; returns logits."""
    lengths = [x.shape[1] for x in features]
    width, context, batch = blocks[0].shape[1], max(lengths), len(blocks)
    if len({b.shape[1] for b in blocks}) != 1:
        raise ValueError('Group draft calls by block width')
    hidden = features[0].new_zeros(batch, context, features[0].shape[-1])
    positions = torch.zeros(batch, context + width, dtype=torch.long, device=target.device)
    for i, (x, cache, start) in enumerate(zip(features, caches, starts)):
        previous = cache.get_seq_length()
        if previous + lengths[i] != start:
            raise ValueError('Features must extend exactly the retained drafter prefix')
        hidden[i, :lengths[i]] = x[0]
        positions[i, :lengths[i]] = torch.arange(previous, start, device=target.device)
        positions[i, context:] = torch.arange(start, start + width, device=target.device)
    prefix = max(starts)
    valid = torch.arange(prefix + width, device=target.device)[None, :] < torch.tensor(
        starts, device=target.device)[:, None]
    valid[:, prefix:] = True
    cache = DraftBatchCache(caches, lengths, starts, context, width)
    output = draft(target_hidden=hidden, noise_embedding=target.embedding(torch.cat(blocks)),
        position_ids=positions, attention_mask=valid[:, None, None, :],
        past_key_values=cache, use_cache=True, is_causal=False)
    return target.head(output[:, 1:])


@dataclass
class RequestState:
    request: object
    cache: object
    generator: object
    admitted: float
    draft_cache: object = field(default_factory=DynamicCache)
    features: object = None
    parts: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    emitted: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    prefilled: int = 0
    first: float = 0.


@torch.inference_mode()
def batch_decode(requests, target, *, draft=None, temperature=0., seeds=None,
                 prefill_chunk_size=128, progress_callback=None):
    """Finite stream wall time includes admission, chunked prefill and callback IO."""
    requests = list(requests)
    slots = target.slots
    if not requests or prefill_chunk_size < 1 or temperature not in (0., 1.):
        raise ValueError('Nonempty requests, positive chunk size and temperature 0/1 required')
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError('Duplicate request IDs')
    for r in requests:
        if not r.prompt or r.max_new_tokens < 1 or len(r.prompt) + r.max_new_tokens > slots[0].max_seq:
            raise ValueError('Invalid request or context capacity exceeded; no truncation')
    seeds = seeds or {r.request_id: 0 for r in requests}
    active, completed = {}, []
    next_request = chunks = chunk_tokens = refills = max_active = max_draft = 0
    last_prefill = -1
    target.max_verify_batch = 0
    target.engine.reset_metrics()
    start = sync_time(slots[0].device)

    def finish(i):
        state = active[i]
        if not state.tokens or (len(state.tokens) < state.request.max_new_tokens and
                                state.tokens[-1] not in state.request.stop):
            return
        end = time.perf_counter()
        count, calls = len(state.tokens) - 1, len(state.emitted)
        stats = dict(decode_tokens=count, target_verifications=calls,
            loop_steps=calls * slots[i].depth, gamma=count / calls if calls else None,
            emitted_per_verification=state.emitted, accepted_draft_tokens=state.accepted)
        completed.append(dict(request_id=state.request.request_id, tokens=state.tokens,
            n_tokens=len(state.tokens), stats=stats, slot=i,
            stop_reason='eos' if state.tokens[-1] in state.request.stop else 'length',
            admitted_s=state.admitted - start, completed_s=end - start,
            latency_s=end - state.admitted, ttft_s=state.first - state.admitted))
        del active[i]
        if progress_callback:
            progress_callback(completed, end - start)

    while next_request < len(requests) or active:
        for i, slot in enumerate(slots):
            if i not in active and next_request < len(requests):
                r = requests[next_request]
                refills += int(next_request >= len(slots))
                next_request += 1
                active[i] = RequestState(r, slot.new_cache(), torch.Generator(
                    device=slot.device).manual_seed(seeds[r.request_id]), time.perf_counter())
        max_active = max(max_active, len(active))
        waiting = sorted(i for i, s in active.items() if not s.tokens)
        if waiting:
            i = next((i for i in waiting if i > last_prefill), waiting[0])
            last_prefill = i
            state, slot = active[i], slots[i]
            ids = state.request.prompt[state.prefilled:state.prefilled + prefill_chunk_size]
            result = slot.prefill_chunk(torch.tensor([ids], device=slot.device), state.cache,
                                        features=draft is not None)
            state.prefilled += len(ids)
            chunks += 1
            chunk_tokens += len(ids)
            if draft is not None:
                state.parts.append(result.features)
            if state.prefilled == len(state.request.prompt):
                state.tokens = [int(sample_logits(result.logits[:, -1], temperature, state.generator).item())]
                state.first = time.perf_counter()
                if draft is not None:
                    state.features = torch.cat(state.parts, dim=1)
                    state.parts.clear()
                finish(i)
        ready = [i for i, s in active.items() if s.tokens]
        if not ready:
            continue
        blocks, proposals, groups = {}, {}, defaultdict(list)
        for i in ready:
            state = active[i]
            remaining = state.request.max_new_tokens - len(state.tokens)
            width = min(draft.block_size, remaining) if draft is not None else 1
            block = torch.full((1, width), draft.mask_token_id if draft is not None else 0,
                               dtype=torch.long, device=slots[i].device)
            block[:, 0] = state.tokens[-1]
            blocks[i] = block
            if draft is not None and width > 1:
                groups[width].append(i)
        for group in groups.values():
            states = [active[i] for i in group]
            logits = draft_batch(slots[0], draft, [s.features for s in states],
                [blocks[i] for i in group], [s.draft_cache for s in states],
                [s.cache.get_seq_length() for s in states])
            max_draft = max(max_draft, len(group))
            for offset, i in enumerate(group):
                q = probabilities(logits[offset:offset+1], temperature) if temperature else None
                blocks[i][:, 1:] = (sample_probabilities(q, active[i].generator) if temperature
                                    else logits[offset:offset+1].argmax(-1))
                proposals[i] = q
        results = target.verify([(i, blocks[i], active[i].cache) for i in ready], features=draft is not None)
        for i, result in zip(ready, results):
            state, block = active[i], blocks[i]
            if temperature == 0:
                choices = result.logits.argmax(-1)
                accepted = int((block[:, 1:] == choices[:, :-1]).int().cumprod(-1).sum().item())
                bonus = int(choices[0, accepted].item())
            elif block.shape[1] == 1:
                accepted = 0
                bonus = int(sample_logits(result.logits[:, 0], temperature, state.generator).item())
            else:
                accepted, bonus = rejection_sample(block[:, 1:], probabilities(result.logits, temperature),
                                                   proposals[i], state.generator)
            emitted = block[0, 1:accepted+1].tolist() + [bonus]
            for j, token in enumerate(emitted):
                if token in state.request.stop:
                    emitted = emitted[:j+1]
                    break
            state.tokens.extend(emitted)
            state.accepted.append(accepted)
            state.emitted.append(len(emitted))
            slots[i].crop(state.cache, len(state.request.prompt) + len(state.tokens) - 1)
            if draft is not None:
                state.features = result.features[:, :len(emitted)]
            finish(i)
    wall = sync_time(slots[0].device) - start
    output_tokens = sum(r['n_tokens'] for r in completed)
    decode_tokens = sum(r['stats']['decode_tokens'] for r in completed)
    calls = sum(r['stats']['target_verifications'] for r in completed)
    input_tokens = sum(len(r.prompt) for r in requests)
    order = {r.request_id: i for i, r in enumerate(requests)}
    completed.sort(key=lambda r: order[r['request_id']])
    return completed, dict(requests=len(completed), concurrency=len(slots), schedule='continuous',
        wall_s=wall, output_tokens=output_tokens, decode_tokens=decode_tokens, input_tokens=input_tokens,
        output_tput=output_tokens / wall, total_tput=(input_tokens + output_tokens) / wall,
        gamma=decode_tokens / calls if calls else None, target_verifications=calls,
        max_active_requests=max_active, max_verify_batch=target.max_verify_batch,
        max_draft_batch=max_draft, refilled_requests=refills,
        prefill_chunks=chunks, prefill_chunk_tokens=chunk_tokens, **target.engine.metrics())
