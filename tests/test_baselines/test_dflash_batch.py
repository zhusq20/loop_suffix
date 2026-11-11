"""Batched feature parity, masked draft cache, rollback, EOS and slot refill."""
import pytest
import torch
from transformers import DynamicCache

from baselines.dflash.batch import BatchedDFlashTarget, batch_decode, draft_batch
from baselines.dflash.decode import generate
from baselines.dflash.projected_draft import inference_draft_class
from baselines.vendor.dflash_inference import DFlashDraftModel
from llm_memory_editability.loop_batch_decode import BatchRequest
from .test_portable_training import TinyTarget, tiny_config


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@torch.inference_mode()
def test_ragged_target_features_rollback_and_slot_ownership(family):
    native = TinyTarget(family)
    batch = BatchedDFlashTarget(native, 3)
    prompts = [[1, 2, 3, 4, 5], [7, 8], [9, 10, 11]]
    caches = [s.new_cache() for s in batch.slots]
    for slot, cache, prompt in zip(batch.slots, caches, prompts):
        features, logits = [], []
        for start in range(0, len(prompt), 2):
            out = slot.prefill_chunk(torch.tensor([prompt[start:start+2]]), cache)
            features.append(out.features)
            logits.append(out.logits)
        expected = native.packed.forward(torch.tensor([prompt]))
        torch.testing.assert_close(torch.cat(features, 1), expected.features, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(torch.cat(logits, 1), expected.logits, atol=3e-5, rtol=3e-5)
    # Jobs deliberately omit observer slot zero and reverse the remaining order.
    for order, blocks, retains in [([2, 1], [[4]*16, [5, 6, 7]], [2, 0]),
                                  ([1, 0, 2], [[8], [9]*7, [10, 11]], [1, 3, 1])]:
        outputs = batch.verify([(i, torch.tensor([b]), caches[i]) for i, b in zip(order, blocks)])
        for i, block, keep, actual in zip(order, blocks, retains, outputs):
            expected = native.packed.forward(torch.tensor([prompts[i]+block]))
            torch.testing.assert_close(actual.logits, expected.logits[:, -len(block):], atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(actual.features, expected.features[:, -len(block):], atol=3e-5, rtol=3e-5)
            prompts[i] += block[:keep]
            batch.slots[i].crop(caches[i], len(prompts[i]))
    with pytest.raises(ValueError, match='foreign'):
        batch.verify([(0, torch.tensor([[1]]), caches[1])])
    batch.slots[1].new_cache()
    with pytest.raises(ValueError, match='Stale'):
        batch.verify([(1, torch.tensor([[1]]), caches[1])])
    batch.close()
    native.close()


@pytest.mark.parametrize('projected', [False, True])
@torch.inference_mode()
def test_padded_drafter_matches_separate_calls_across_different_acceptances(projected):
    native = TinyTarget()
    cfg = tiny_config(native, projected)
    cfg._attn_implementation = 'sdpa'
    draft = (inference_draft_class() if projected else DFlashDraftModel)(cfg).eval()
    caches, references = [DynamicCache() for _ in range(3)], [DynamicCache() for _ in range(3)]
    starts = [0, 0, 0]
    for lengths in [[3, 7, 2], [1, 5, 2], [3, 1, 7]]:
        hidden = [torch.randn(1, n, len(native.physical_ids)*native.hidden_size) for n in lengths]
        blocks = [torch.tensor([[1, 0, 0, 0]]) for _ in lengths]
        starts = [s+n for s, n in zip(starts, lengths)]
        expected = []
        for features, block, cache, start in zip(hidden, blocks, references, starts):
            result = draft(target_hidden=features, noise_embedding=native.embedding(block),
                position_ids=torch.arange(cache.get_seq_length(), start+4)[None],
                past_key_values=cache, use_cache=True, is_causal=False)
            expected.append(native.head(result[:, 1:]))
            cache.crop(start)
        actual = draft_batch(native, draft, hidden, blocks, caches, starts)
        torch.testing.assert_close(actual, torch.cat(expected), atol=3e-5, rtol=3e-5)
        for got, ref, start in zip(caches, references, starts):
            assert got.get_seq_length() == start
            for a, b in zip(got.layers, ref.layers):
                torch.testing.assert_close(a.keys, b.keys, atol=3e-5, rtol=3e-5)
                torch.testing.assert_close(a.values, b.values, atol=3e-5, rtol=3e-5)
    native.close()


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('temperature', [0., 1.])
@torch.inference_mode()
def test_continuous_matches_serial_and_refills(family, temperature):
    native = TinyTarget(family)
    cfg = tiny_config(native, family == 'huginn')
    cfg._attn_implementation = 'sdpa'
    draft = (inference_draft_class() if family == 'huginn' else DFlashDraftModel)(cfg).eval()
    requests = [BatchRequest(str(i), [1, 2, 3][:1+i%3], limit, set())
                for i, limit in enumerate([33, 18, 2, 1, 17, 9, 5])]
    seeds = {r.request_id: 42+i for i, r in enumerate(requests)}
    expected = [generate(native.packed, torch.tensor([r.prompt]), r.max_new_tokens, r.stop,
        draft=draft, temperature=temperature, seed=seeds[r.request_id]) for r in requests]
    # A later request stops inside its actual serial continuation.
    requests[-1].stop = {expected[-1].tokens[2]}
    expected[-1] = generate(native.packed, torch.tensor([requests[-1].prompt]), 5,
        requests[-1].stop, draft=draft, temperature=temperature, seed=seeds[requests[-1].request_id])
    for size in [1, 2, 4]:
        batch = BatchedDFlashTarget(native, size)
        published = []
        actual, summary = batch_decode(requests, batch, draft=draft, temperature=temperature,
            seeds=seeds, prefill_chunk_size=2,
            progress_callback=lambda rows, elapsed: published.append(rows[-1]['request_id']))
        assert [r['tokens'] for r in actual] == [r.tokens for r in expected]
        assert [r['stats']['emitted_per_verification'] for r in actual] == [r.stats['emitted_per_verification'] for r in expected]
        assert len(published) == len(set(published)) == len(requests)
        assert summary['prefill_chunk_tokens'] == sum(len(r.prompt) for r in requests)
        assert summary['refilled_requests'] == len(requests)-size
        if size > 1:
            assert summary['max_verify_batch'] > 1 and summary['max_draft_batch'] > 1
        assert summary['core_padding_rows'] == 0
        assert summary['output_tput'] == sum(len(r['tokens']) for r in actual)/summary['wall_s']
        batch.close()
    native.close()


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('batch_size', [8, 16])
@torch.inference_mode()
def test_large_batch_has_disjoint_prefixes_and_features(family, batch_size):
    native = TinyTarget(family)
    batch = BatchedDFlashTarget(native, batch_size)
    jobs, expected = [], []
    for i, slot in enumerate(batch.slots):
        prompt = [1, 2+i%15, 3][:1+i%3]
        block = [4+i%14]*(1+i%4)
        cache = slot.new_cache()
        slot.prefill_chunk(torch.tensor([prompt]), cache)
        jobs.append((i, torch.tensor([block]), cache))
        ref = native.packed.forward(torch.tensor([prompt+block]))
        expected.append((ref.logits[:, -len(block):], ref.features[:, -len(block):]))
    for got, (logits, features) in zip(batch.verify(jobs[::-1]), expected[::-1]):
        torch.testing.assert_close(got.logits, logits, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(got.features, features, atol=3e-5, rtol=3e-5)
    assert batch.max_verify_batch == batch_size
    batch.close()
    native.close()
