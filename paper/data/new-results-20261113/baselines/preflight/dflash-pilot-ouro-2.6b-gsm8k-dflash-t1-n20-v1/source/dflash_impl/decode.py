"""Adapt the official DFlash v1 generation loop to loop caches and frozen timing.

Drafter: z-lab/dflash@94e4abc5, unchanged.
Generation structure: dflash/model.py at that revision. Stochastic acceptance uses
the p/q and positive-residual rule also implemented by upstream@07ebd93d,
with an explicit generator. Graphs/compilation are disabled.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from transformers import DynamicCache


def probabilities(logits, temperature):
    if temperature <= 0:
        raise ValueError('Probability conversion requires positive temperature')
    return (logits.float() / temperature).softmax(dim=-1)


def sample_probabilities(probs, generator):
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1,
                             generator=generator).reshape(probs.shape[:-1])


def sample_logits(logits, temperature, generator):
    if temperature == 0:
        return logits.argmax(dim=-1)
    return sample_probabilities(probabilities(logits, temperature), generator)


def rejection_sample(draft_tokens, target_probs, draft_probs, generator):
    """Batch1 sequential rejection. q is exactly the distribution used to draft."""
    count = draft_tokens.shape[1]
    p = target_probs[:, :count].gather(-1, draft_tokens[..., None])[..., 0]
    q = draft_probs.gather(-1, draft_tokens[..., None])[..., 0]
    uniforms = torch.rand(q.shape, device=q.device, dtype=q.dtype, generator=generator)
    accepted = int((uniforms * q < p).to(torch.int32).cumprod(-1).sum().item())
    if accepted == count:
        return accepted, sample_probabilities(target_probs[:, -1], generator).item()
    residual = (target_probs[0, accepted] - draft_probs[0, accepted]).clamp_min(0)
    total = residual.sum()
    # Equal p=q has zero rejection probability in exact arithmetic. Guard roundoff.
    residual = torch.where(total > 0, residual / total.clamp_min(torch.finfo(residual.dtype).tiny),
                           target_probs[0, accepted])
    return accepted, sample_probabilities(residual, generator).item()


@dataclass
class DecodeResult:
    tokens: list[int]
    stats: dict


def sync_time(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def generate(target, input_ids, max_new_tokens, stop_ids, *, draft=None, temperature=0., seed=0,
             profile_draft=False, target_sampler=None, trace=None, diagnostic_stop_after=None):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError('This aligned latency decoder supports one nonempty request')
    if max_new_tokens < 1 or temperature not in (0., 1.):
        raise ValueError('Positive output budget and frozen temperatures 0/1 required')
    if input_ids.shape[1] + max_new_tokens > target.max_seq:
        raise ValueError('Prompt plus generation exceeds context capacity')
    generator = torch.Generator(device=target.device).manual_seed(seed)
    target_cache = target.new_cache()
    draft_cache = DynamicCache() if draft is not None else None
    begin = sync_time(target.device)
    prefill = target.forward(input_ids, target_cache, features=draft is not None)
    out = [int(sample_logits(prefill.logits[:, -1], temperature, generator).item()) if target_sampler is None else
           int(target_sampler(prefill.logits[:, -1], [input_ids.shape[1]])[0])]
    features = prefill.features
    prefilled = sync_time(target.device)
    prefill_s = prefilled - begin
    start = input_ids.shape[1]
    verifies, raw_accepted, emitted_lengths, events = 0, [], [], []
    stop = set(stop_ids)
    while len(out) < max_new_tokens and out[-1] not in stop:
        remaining = max_new_tokens - len(out)
        width = min(draft.block_size, remaining) if draft is not None else 1
        block = torch.full((1, width), draft.mask_token_id if draft is not None else 0,
                           device=target.device, dtype=torch.long)
        block[:, 0] = out[-1]
        if draft is not None and width > 1:
            if profile_draft:
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                first.record()
            hidden = draft(target_hidden=features, noise_embedding=target.embedding(block),
                position_ids=torch.arange(draft_cache.get_seq_length(), start + width,
                                          device=target.device).unsqueeze(0),
                past_key_values=draft_cache, use_cache=True, is_causal=False)
            draft_logits = target.head(hidden[:, 1:])
            draft_cache.crop(start)
            if temperature == 0:
                block[:, 1:] = draft_logits.argmax(-1)
                q = None
            else:
                q = probabilities(draft_logits, temperature)
                block[:, 1:] = sample_probabilities(q, generator)
            if profile_draft:
                last.record()
                events.append((first, last))
        else:
            q = None
        verified = target.forward(block, target_cache, features=draft is not None)
        verifies += 1
        if temperature == 0:
            choices = verified.logits.argmax(-1)
            accepted = int((block[:, 1:] == choices[:, :-1]).to(torch.int32).cumprod(-1).sum().item())
            bonus = int(choices[0, accepted].item())
        elif width == 1:
            accepted = 0
            bonus = (int(target_sampler(verified.logits[:, 0], [start + 1])[0])
                     if draft is None and target_sampler is not None else
                     int(sample_logits(verified.logits[:, 0], temperature, generator).item()))
        else:
            accepted, bonus = rejection_sample(block[:, 1:], probabilities(verified.logits, temperature), q, generator)
        new = block[0, 1:accepted + 1].tolist() + [bonus]
        for i, token in enumerate(new):
            if token in stop:
                new = new[:i + 1]
                break
        assert 1 <= len(new) <= remaining
        if trace is not None:
            values, indices = verified.logits.float().topk(3, dim=-1)
            trace.append(dict(anchor_output_index=len(out)-1, cache_start=start,
                block=block[0].tolist(), accepted=accepted, emitted=new,
                top_ids=indices[0].tolist(), top_logits=values[0].tolist()))
        out.extend(new)
        emitted_lengths.append(len(new))
        raw_accepted.append(accepted)
        start += len(new)
        target.crop(target_cache, start)
        if draft is not None:
            features = verified.features[:, :len(new)]
        if diagnostic_stop_after is not None and len(out) >= diagnostic_stop_after:
            break
    elapsed = sync_time(target.device) - prefilled
    emitted = len(out) - 1  # Prefill produces the first token; match frozen drive().
    assert sum(emitted_lengths) == emitted
    stats = dict(decode_tokens=emitted, output_tokens=len(out), decode_s=elapsed,
                 decode_tok_s=emitted / elapsed if elapsed else None,
                 target_verifications=verifies, loop_steps=verifies * target.depth,
                 gamma=emitted / verifies if verifies else None, prefill_s=prefill_s,
                 accepted_draft_tokens=raw_accepted, emitted_per_verification=emitted_lengths,
                 stop_reason='eos' if out[-1] in stop else 'length', temperature=temperature,
                 top_p=1., top_k=0, seed=seed, graph=False,
                 instrumented_draft_profile=profile_draft,
                 diagnostic_trace=trace is not None,
                 diagnostic_stop_after=diagnostic_stop_after,
                 draft_gpu_s=sum(a.elapsed_time(b) for a, b in events) / 1000 if profile_draft else None,
                 timing='synchronized decode only; includes first draft; excludes prefill first token')
    return DecodeResult(out, stats)
