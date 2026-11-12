"""Check real packed target/LAD batching against the single-request executor."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'results/loopfill-table2-20261107-v2/source/src'), str(ROOT/'tests')]
from test_loop_batch_decode import TinyHuginn, TinyOuro
from llm_memory_editability.loop_batch_decode import BatchRequest
from llm_memory_editability.loop_optimized_decode import SharedKVPool
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop
from .batch import BatchedLoopAheadExecutor, batch_decode
from .decode import LoopAheadConfig, decode_steps
from .engine import LoopAheadExecutor
from .lad import LADConfig, LookaheadDrafter


def setup(family, batch):
    torch.manual_seed(71)
    model = (TinyHuginn() if family == 'huginn' else TinyOuro()).eval()
    lad = LookaheadDrafter(LADConfig(16, width=16, layers=1, heads=2, intermediate=32)).eval()
    pool = SharedKVPool(batch)
    slots = [(HuginnLoop(model, 4, 128, 32, 0, 128, extra_depths=(2,), kv_factory=pool)
              if family == 'huginn' else OuroLoop(model, 4, 128, 32, kv_factory=pool)) for _ in range(batch)]
    cfg = LoopAheadConfig(family=family, future=2 if family == 'huginn' else 3)
    return slots, lad, cfg


def drive(b, engine, req, cfg):
    engine.request(0, req.stop)
    steps = decode_steps(b, req.prompt, cfg, req.max_new_tokens, req.stop, generator=engine.generator)
    value = None
    while True:
        try:
            call = steps.send(value)
        except StopIteration as done:
            return done.value
        value = engine.execute(call.name, [(b, call)])[0]


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('batch', [1, 2, 4, 16])
@torch.inference_mode()
def test_packed_lad_matches_serial_with_refill(family, batch):
    slots, lad, cfg = setup(family, batch)
    # Long distinct prompts exercise LAD context; short/EOS/cap requests force slot reuse.
    reqs = [BatchRequest(i, [(j*3+i)%19 for j in range(65+i%4)], 1 if i%5 == 0 else 10)
            for i in range(batch+3)]
    reference_slots, reference_lad, _ = setup(family, 1)
    b = reference_slots[0]
    reference = LoopAheadExecutor(b, reference_lad, [0, 1, 2])
    expected = [drive(b, reference, req, cfg) for req in reqs]
    engine = BatchedLoopAheadExecutor(slots, lad, [0, 1, 2])
    observed, summary = batch_decode(reqs, slots, cfg, engine, chunk_size=17)
    assert [r['tokens'] for r in observed] == [r[0] for r in expected]
    for actual, (_, stats) in zip(observed, expected, strict=True):
        for key in ['tokens', 'calls', 'lad_calls', 'lad_tokens', 'first_proposals', 'second_proposals']:
            assert actual['stats'][key] == stats[key], (actual['request_id'], key)
    assert summary['prefill_chunk_tokens'] == sum(len(r.prompt) for r in reqs)
    assert summary['max_active_requests'] == batch
    assert summary['lad_batches'] > 0
    assert summary['output_tokens'] == sum(len(r['tokens']) for r in observed)
    if batch > 1:
        assert summary['max_advance_requests'] > 1
        assert summary['refilled_requests'] > 0
    if batch >= 4:
        assert summary['max_lad_batch'] > 1


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@torch.inference_mode()
def test_chunked_shallow_features_and_slot_reset(family):
    slots, lad, cfg = setup(family, 1)
    refs, ref_lad, _ = setup(family, 1)
    engine = BatchedLoopAheadExecutor(slots, lad, [0, 1, 2])
    ref = LoopAheadExecutor(refs[0], ref_lad, [0, 1, 2])
    for prompt in [[i%19 for i in range(79)], [8, 3, 1]]:
        req = BatchRequest('x', prompt, 3, {18})
        engine.start_request(slots[0], req, cfg)
        ref.request(0, req.stop)
        expected = ref._prefill(refs[0], prompt, cfg.first)
        end, actual = 0, None
        while end < len(prompt):
            end, actual = engine.prefill(slots[0], prompt, cfg.first, end, 17)
        assert expected == actual
        state = engine.requests[id(slots[0])]
        torch.testing.assert_close(state.prefix_states, ref.prefix_states, atol=2e-6, rtol=2e-5)
        assert state.vocab.tokens == ref.vocab.tokens
        torch.testing.assert_close(slots[0].kv.K[:, :len(prompt)], refs[0].kv.K[:, :len(prompt)], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@torch.inference_mode()
def test_prefill_eos_and_post_decode_stop(family):
    refs, ref_lad, cfg = setup(family, 1)
    ref = LoopAheadExecutor(refs[0], ref_lad, [0, 1, 2])
    req = BatchRequest('probe', list(range(19))*4, 10)
    tokens, _ = drive(refs[0], ref, req, cfg)
    requests = [BatchRequest('first', req.prompt, 10, {tokens[0]}),
                BatchRequest('later', req.prompt, 10, {tokens[-1]}),
                BatchRequest('budget', [2, 1, 3], 1)]
    slots, lad, _ = setup(family, 2)
    observed, _ = batch_decode(requests, slots, cfg, BatchedLoopAheadExecutor(slots, lad, [0, 1, 2]), chunk_size=17)
    assert observed[0]['tokens'] == tokens[:1]
    assert observed[1]['tokens'] == tokens[:tokens.index(tokens[-1])+1]
    assert len(observed[2]['tokens']) == 1
