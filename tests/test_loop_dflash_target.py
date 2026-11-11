"""Packed DFlash causal blocks, feature rows, and rollback vs dense replay."""
from types import SimpleNamespace

import pytest
import torch

from baselines.dflash.packed_target import PackedDFlashTarget
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop
from test_loop_batch_decode import TinyHuginn, TinyOuro


@pytest.fixture(params=['ouro', 'huginn'])
def target(request):
    torch.manual_seed(41)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    family = request.param
    model = (TinyOuro() if family == 'ouro' else TinyHuginn()).eval()
    depth, capacity = 4, 64
    if family == 'ouro':
        b = OuroLoop(model, depth, capacity, 16)
        layers = b.layers
        capture_at = {0: depth, 1: depth}
        embedding, head = model.model.embed_tokens, model.lm_head
        noise = None
    else:
        b = HuginnLoop(model, depth, capacity, 16, 0, capacity)
        layers = b.prelude + b.core + b.coda
        capture_at = {0: 1, 1: depth, 2: depth, 3: 1}
        embedding, head = model.transformer.wte, model.lm_head
        noise = b.init
    spec = SimpleNamespace(model=model, family=family, depth=depth, max_seq=capacity,
        device=torch.device('cpu'), dtype=torch.float32, embedding=embedding, head=head,
        hidden_size=16, physical_ids=list(capture_at), expanded_ids=[],
        capture_at=capture_at, noise=noise)
    packed = PackedDFlashTarget(spec)
    # Dense reference always recomputes the complete causal prefix; no packed
    # attention, speculative tree storage, or crop operation participates.
    operation = '_layer' if family == 'ouro' else '_block'
    original = getattr(b, operation)
    indices = {id(layer): i for i, layer in enumerate(layers)}
    states, counts = {}, {}

    def observed(layer, *args):
        out = original(layer, *args)
        index = indices[id(layer)]
        counts[index] = counts.get(index, 0) + 1
        if counts[index] == capture_at[index]:
            states[index] = out.clone()
        return out

    setattr(b, operation, observed)

    def replay(tokens):
        states.clear()
        counts.clear()
        b.prefill(tokens, None)
        hidden = states[len(layers) - 1]
        if family == 'ouro':
            hidden = b.inner.norm(hidden)
        else:
            hidden = model.transformer.ln_f(hidden)
        return head(hidden), torch.cat([states[i] for i in spec.physical_ids], dim=-1)

    yield packed, replay
    packed.close()
    torch.set_num_threads(old_threads)


@torch.inference_mode()
def test_features_logits_and_rejected_suffix_are_causal(target):
    packed, replay = target
    tokens = [1, 4, 3, 2]
    cache = packed.new_cache()
    got = packed.forward(torch.tensor([tokens]), cache)
    logits, features = replay(tokens)
    torch.testing.assert_close(got.logits, logits, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(got.features, features, atol=3e-5, rtol=3e-5)
    # Include non-power-of-two widths, full 16-position blocks, full rejection,
    # partial acceptance, and a one-position final verification.
    for block, accepted in [([5, 6, 7], 1), ([8] * 16, 5), ([9, 10], 0), ([11], 1)]:
        got = packed.forward(torch.tensor([block]), cache)
        logits, features = replay(tokens + block)
        torch.testing.assert_close(got.logits, logits[:, -len(block):], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(got.features, features[:, -len(block):], atol=3e-5, rtol=3e-5)
        tokens += block[:accepted]
        packed.crop(cache, len(tokens))
        assert cache.get_seq_length() == len(tokens)
        assert set(packed.cache_lengths(cache)) == {len(tokens)}
    no_features = packed.forward(torch.tensor([[12]]), cache, features=False)
    assert no_features.features is None
    logits, _ = replay(tokens + [12])
    torch.testing.assert_close(no_features.logits, logits[:, -1:], atol=3e-5, rtol=3e-5)


@torch.inference_mode()
def test_cache_reuse_and_invalid_inputs(target):
    packed, replay = target
    first = packed.new_cache()
    packed.forward(torch.tensor([[1, 2, 3]]), first)
    second = packed.new_cache()
    with pytest.raises(ValueError, match='Stale'):
        packed.forward(torch.tensor([[4]]), first)
    got = packed.forward(torch.tensor([[7, 6]]), second)
    logits, features = replay([7, 6])
    torch.testing.assert_close(got.logits, logits)
    torch.testing.assert_close(got.features, features)
    with pytest.raises(ValueError, match='capacity'):
        packed.forward(torch.ones(1, 17, dtype=torch.long), second)
    with pytest.raises(ValueError, match='Crop'):
        packed.crop(second, 3)
    with pytest.raises(ValueError, match='batch-one'):
        packed.forward(torch.ones(2, 1, dtype=torch.long), second)
    with pytest.raises(ValueError, match='unpadded'):
        packed.forward(torch.ones(1, 1, dtype=torch.long), second, use_cache=False)
