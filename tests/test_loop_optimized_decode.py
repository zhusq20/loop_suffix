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


def setup(family, device='cpu', dtype=torch.float32, max_nodes=128):
    torch.manual_seed(71)
    model = (TinyOuro() if family == 'ouro' else TinyHuginn()).to(device, dtype).eval()
    def make(pool=None):
        return (OuroLoop(model, 4, 128, max_nodes, kv_factory=pool) if family == 'ouro' else
                HuginnLoop(model, 4, 128, max_nodes, 0, 128, extra_depths=(2,), kv_factory=pool))
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


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ['ar', 'budget-suffix', 'budget-raw'])
@pytest.mark.parametrize('cuda', [False, True])
@pytest.mark.parametrize('max_nodes', [9, 128])
def test_padded_shapes_preserve_outputs_accounting_and_kv(family, arm, cuda, max_nodes):
    if cuda and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    make = setup(family, 'cuda' if cuda else 'cpu', torch.bfloat16 if cuda else torch.float32,
                 max_nodes=max_nodes)
    cfg = arm_config(arm, 4, 1, 2, 4, draft_node_budget=8)
    def cache():
        return CompletionSuffixCache.arctic() if cfg.use_suffix else None
    runs = []
    for padding in (False, True):
        pool = SharedKVPool(2, pad_decode_rows=padding)
        slots = [make(pool), make(pool)]
        engine = PackedLoopExecutor(slots, cuda_graph=cuda, max_graphs=32)
        batch_decode(requests(), slots, cfg, cache=cache(), schedule='continuous',
                     executor=engine, prefill_chunk_size=2)
        pure = arm in ('ar', 'budget-suffix')
        if padding and pure and max_nodes <= 65:
            engine.prime_verifier_shapes(1 if arm == 'ar' else 5)
        engine.graphs.freeze()
        if cuda and padding and pure and max_nodes <= 65:
            original_run = engine.graphs.run
            def checked_run(key, inputs, fn):
                if key[0] == 'advance':
                    assert key in engine.graphs.entries
                return original_run(key, inputs, fn)
            engine.graphs.run = checked_run
        # Poison the private sink: real outputs must never read its contents.
        pool.tree_k_storage[pool.TK.numel() // pool.TK.shape[-1] // pool.TK.shape[-2]:].fill_(float('nan'))
        pool.tree_v_storage[pool.TV.numel() // pool.TV.shape[-1] // pool.TV.shape[-2]:].fill_(float('nan'))
        # Reused request slots, changed prompts, EOS, and short output budgets.
        changed = [replace(r, prompt=[(t + 1) % 13 for t in r.prompt],
                           stop={18} if r.request_id % 2 else set()) for r in requests()]
        rows, summary = batch_decode(changed, slots, cfg, cache=cache(), schedule='continuous',
                                     executor=engine, prefill_chunk_size=2)
        runs.append((rows, summary, [(b.kv.K.clone(), b.kv.V.clone(), b.kv.length) for b in slots]))
    rows, summary, kv = runs[1]
    refs, original, ref_kv = runs[0]
    assert [r['tokens'] for r in rows] == [r['tokens'] for r in refs]
    for r, ref in zip(rows, refs, strict=True):
        for metric in ('calls', 'rows', 'nodes', 'readout_rows', 'speculate_calls'):
            assert r['stats'][metric] == ref['stats'][metric]
    assert summary['core_calls'] == original['core_calls']
    assert summary['core_rows'] == original['core_rows']
    assert summary['core_execution_rows'] >= summary['core_rows']
    assert summary['padding_sink_bytes'] > 0
    for (k, v, n), (rk, rv, rn) in zip(kv, ref_kv, strict=True):
        assert n == rn
        # Padding may change GEMM/reduction order, especially under BF16.
        tol = 0.015625 if cuda else 2e-6
        torch.testing.assert_close(k[:, :n], rk[:, :n], atol=tol, rtol=tol)
        torch.testing.assert_close(v[:, :n], rv[:, :n], atol=tol, rtol=tol)
