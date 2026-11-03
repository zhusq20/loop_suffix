"""Execution optimizations must preserve KV isolation and graph metadata updates."""
from dataclasses import replace
import pytest
import torch

from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor, prefill_chunk
from llm_memory_editability.loop_tree_decode import OuroLoop, HuginnLoop
from scripts.loop_batch_bench import arm_config
from scripts.loop_batch_bench import cuda_oom_cause
from test_loop_batch_decode import TinyOuro, TinyHuginn

ARMS = ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-raw', 'budget-online']


def test_graph_capture_failure_preserves_only_typed_cuda_oom():
    oom = torch.OutOfMemoryError('allocation failed')
    end = RuntimeError('capture ended after earlier error')
    end.__context__ = oom
    assert cuda_oom_cause(end) is oom
    assert cuda_oom_cause(RuntimeError('CUDA out of memory text alone')) is None
    assert cuda_oom_cause(MemoryError('host memory')) is None

@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup(family, device='cpu', dtype=torch.float32):
    torch.manual_seed(71)
    model = (TinyOuro() if family == 'ouro' else TinyHuginn()).to(device, dtype).eval()
    def make(pool=None):
        return (OuroLoop(model, 4, 128, 128, kv_factory=pool) if family == 'ouro' else
                HuginnLoop(model, 4, 128, 128, 0, 128, extra_depths=(2,), kv_factory=pool))
    return make


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('chunk_size', [1, 3, 64])
@torch.no_grad()
def test_chunked_prefill_matches_whole_prompt_cache(family, chunk_size):
    make = setup(family)
    original, chunked = make(), make()
    # Second prompt exercises reused storage and shorter valid prefix.
    for prompt in [[1, 2, 3, 6, 4, 2, 7], [8, 1]]:
        expected = original.prefill(prompt, 1)
        offset, token = 0, None
        while offset < len(prompt):
            offset, token = prefill_chunk(chunked, prompt, 1, offset, chunk_size)
        assert token == expected
        assert chunked.kv.length == len(prompt)
        torch.testing.assert_close(original.kv.K[:, :len(prompt)], chunked.kv.K[:, :len(prompt)], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(original.kv.V[:, :len(prompt)], chunked.kv.V[:, :len(prompt)], atol=2e-6, rtol=2e-5)


def requests():
    return [BatchRequest(i, p, n) for i, (p, n) in enumerate([
        ([1, 2, 3], 1), ([6, 7, 8, 9], 16), ([4, 2, 3, 1, 8, 2, 7], 2),
        ([1, 5, 3], 3), ([7], 5), ([6, 1, 7, 2], 1), ([2, 7], 8)])]


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ARMS)
def test_packed_and_chunked_preserve_tokens_and_refill(family, arm):
    make = setup(family)
    cfg = arm_config(arm, 4, 1, 2, 64, draft_node_budget=8)
    def cache():
        return CompletionSuffixCache.arctic() if cfg.use_suffix else None
    refs, _ = batch_decode(requests(), [make(), make()], cfg, cache=cache(), schedule='continuous')
    pool = SharedKVPool(2)
    slots = [make(pool), make(pool)]
    engine = PackedLoopExecutor(slots)
    rows, stats = batch_decode(requests(), slots, cfg, cache=cache(), schedule='continuous',
                               executor=engine, prefill_chunk_size=2)
    assert [r['tokens'] for r in rows] == [r['tokens'] for r in refs]
    assert stats['prefill_chunks'] == 14
    assert stats['prefill_chunk_tokens'] == sum(len(r.prompt) for r in requests())
    assert stats['attention_batches'] > 0
    assert stats['refilled_requests'] > 0
    assert stats['attention_policy'] == 'packed_prefix_and_tree'
    assert rows[2]['admitted_s'] < rows[1]['completed_s']
    assert slots[0].kv.K.untyped_storage().data_ptr() == slots[1].kv.K.untyped_storage().data_ptr()
    assert slots[0].kv.K.data_ptr() != slots[1].kv.K.data_ptr()


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_graph_replay_matches_eager_with_slot_reuse(family, arm):
    make = setup(family, 'cuda', torch.bfloat16)
    cfg = arm_config(arm, 4, 1, 2, 64, draft_node_budget=8)
    def cache():
        return CompletionSuffixCache.arctic() if cfg.use_suffix else None
    outputs = []
    for enabled in [False, True]:
        pool = SharedKVPool(2)
        slots = [make(pool), make(pool)]
        engine = PackedLoopExecutor(slots, cuda_graph=enabled, max_graphs=32)
        batch_decode(requests(), slots, cfg, cache=cache(), schedule='continuous',
                     executor=engine, prefill_chunk_size=2)
        engine.graphs.freeze()
        before = len(engine.graphs.entries)
        # Changed tokens and prompt lengths under existing shapes must use current metadata.
        changed = [replace(r, prompt=[(t + 1) % 13 for t in r.prompt] + ([3] if r.request_id % 2 else []))
                   for r in requests()]
        rows, stats = batch_decode(changed, slots, cfg, cache=cache(), schedule='continuous',
                                  executor=engine, prefill_chunk_size=2)
        assert len(engine.graphs.entries) == before
        if enabled:
            assert before > 0
            assert stats['cuda_graph_replays'] > 0
        outputs.append([r['tokens'] for r in rows])
    assert outputs[0] == outputs[1]


def test_prefill_chunks_interleave_active_decode(monkeypatch):
    import llm_memory_editability.loop_optimized_decode as optimized
    make = setup('ouro')
    pool = SharedKVPool(2)
    slots = [make(pool), make(pool)]
    events = []
    original = optimized.prefill_chunk
    def traced(backbone, prompt, depth, offset, size):
        events.append(('chunk', backbone.kv.pool_index, offset))
        return original(backbone, prompt, depth, offset, size)
    monkeypatch.setattr(optimized, 'prefill_chunk', traced)
    class Traced(PackedLoopExecutor):
        def execute(self, name, jobs):
            if name == 'advance':
                events.append(('advance', tuple(b.kv.pool_index for b, _ in jobs)))
            return super().execute(name, jobs)
    cfg = arm_config('ar', 4, 1, 2, 64)
    reqs = [BatchRequest(0, [1], 25), BatchRequest(1, [2,3,4,5,6,7,8,9,1], 2)]
    batch_decode(reqs, slots, cfg, executor=Traced(slots), schedule='continuous', prefill_chunk_size=2)
    chunks = [i for i, e in enumerate(events) if e[0] == 'chunk' and e[1] == 1]
    assert len(chunks) == 5
    for a, b in zip(chunks, chunks[1:]):
        assert any(e[0] == 'advance' and 0 in e[1] for e in events[a+1:b])
