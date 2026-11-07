"""Continuous admission must refill early and preserve per-request decoder state."""
from dataclasses import replace

import pytest
import torch

from llm_memory_editability.loop_batch_decode import BatchRequest, FusedLoopExecutor, batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop, tree_decode
from llm_memory_editability.loop_suffix_spec import ArmConfig
from scripts.loop_batch_bench import arm_config
from test_loop_batch_decode import TinyHuginn, TinyOuro

ARMS = ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-raw', 'budget-online']


class AuditedCache(CompletionSuffixCache):
    def __init__(self):
        base = CompletionSuffixCache.arctic()
        super().__init__(base.local, base.history)
        self.admissions = {}

    def start_request(self, rid, prompt):
        self.admissions[rid] = dict(active=set(self.responses),
                                   published=[p['request_id'] for p in self.publications])
        super().start_request(rid, prompt)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup(family, device='cpu', dtype=torch.float32):
    torch.manual_seed(71)
    model = (TinyOuro() if family == 'ouro' else TinyHuginn()).to(device, dtype).eval()
    def backbone():
        return OuroLoop(model, 4, 128, 128) if family == 'ouro' else HuginnLoop(model, 4, 128, 128, 0, 128, extra_depths=(2,))
    requests = [BatchRequest(i, p, n) for i, (p, n) in enumerate([
        ([1, 2, 3], 1), ([6, 7, 8, 9], 25), ([4, 2], 2),
        ([1, 5, 3], 3), ([7], 5), ([6, 1, 7, 2], 1), ([2, 7], 8)])]
    return backbone, requests


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ARMS)
def test_all_six_arms_refill_and_preserve_serial_tokens_and_history(family, arm):
    backbone, requests = setup(family)
    cfg = arm_config(arm, 4, 1, 2, 64, draft_node_budget=8)
    ar = ArmConfig('ar', 4, 1, 0, False, False)
    refs = [tree_decode(backbone(), r.prompt, ar, r.max_new_tokens, r.stop, max_nodes=128)[0]
            for r in requests]
    # EOS on the first token, with a much larger requested output budget.
    requests[2] = replace(requests[2], max_new_tokens=15, stop={refs[2][0]})
    refs[2] = refs[2][:1]
    cache = AuditedCache() if cfg.use_suffix else None
    rows, summary = batch_decode(requests, [backbone(), backbone()], cfg, cache=cache,
                                 schedule='continuous', max_nodes=128, flush_every=3)
    assert [r['tokens'] for r in rows] == refs
    assert [r['request_id'] for r in rows] == list(range(7))
    assert rows[0]['completed_s'] <= rows[2]['admitted_s'] < rows[1]['completed_s']
    assert rows[2]['slot'] == rows[0]['slot']
    assert rows[2]['completed_s'] <= rows[3]['admitted_s'] < rows[1]['completed_s']
    assert summary['schedule'] == 'continuous'
    assert summary['refilled_requests'] >= 2
    assert summary['max_active_requests'] == 2
    assert 0 < summary['average_active_requests'] <= 2
    assert 0 < summary['slot_utilization'] <= 1
    assert all(0 <= r['ttft_s'] <= r['latency_s'] for r in rows)
    assert summary['output_tokens'] == sum(map(len, refs))
    if cache:
        assert cache.admissions[2] == dict(active={1}, published=[0])
        assert cache.responses == {}
        assert {p['request_id'] for p in cache.publications} == set(range(7))
        assert len(cache.publications) == 7
    if arm.startswith('budget-'):
        assert all(r['stats']['c_initial'] == 1.0 for r in rows)


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ARMS)
def test_batch_one_fixed_and_continuous_have_identical_tokens_and_work(family, arm):
    backbone, requests = setup(family)
    cfg = arm_config(arm, 4, 1, 2, 64, draft_node_budget=8)
    results = []
    for schedule in ['fixed', 'continuous']:
        cache = AuditedCache() if cfg.use_suffix else None
        rows, summary = batch_decode(requests, [backbone()], cfg, cache=cache,
                                     schedule=schedule, max_nodes=128, flush_every=3)
        results.append(([r['tokens'] for r in rows], summary['core_calls'], summary['core_rows']))
        assert summary['refilled_requests'] == 0
    assert results[0] == results[1]


def test_failure_cleans_active_suffix_requests_after_refill():
    backbone, requests = setup('ouro')
    cfg = arm_config('budget-online', 4, 1, 2, 64, draft_node_budget=8)
    cache = AuditedCache()
    slots = [backbone(), backbone()]
    class FailingExecutor(FusedLoopExecutor):
        def execute(self, name, jobs):
            if name == 'advance' and 2 in cache.responses:
                raise RuntimeError('injected decode failure after refill')
            return super().execute(name, jobs)
    with pytest.raises(RuntimeError, match='after refill'):
        batch_decode(requests, slots, cfg, cache=cache, schedule='continuous',
                     executor=FailingExecutor(slots))
    assert 2 in cache.admissions
    assert cache.responses == {}
    assert [p['request_id'] for p in cache.publications] == [0]
    assert set(cache.history.cached_requests) == {0}


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_bf16_continuous_slot_reuse(family, arm):
    backbone, requests = setup(family, 'cuda', torch.bfloat16)
    cfg = arm_config(arm, 4, 1, 2, 64, draft_node_budget=8)
    # Two continuous repetitions must agree; BF16 fused-vs-serial differences
    # are recorded in the full benchmark, not suppressed with a parity claim.
    outputs = []
    for _ in range(2):
        cache = AuditedCache() if cfg.use_suffix else None
        rows, summary = batch_decode(requests, [backbone(), backbone()], cfg,
                                     schedule='continuous', max_nodes=128, cache=cache)
        outputs.append([r['tokens'] for r in rows])
        assert rows[2]['admitted_s'] < rows[1]['completed_s']
        assert summary['refilled_requests'] >= 2
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('arm', ARMS)
def test_all_requests_finish_on_first_token_and_tail_is_drained(arm):
    backbone, requests = setup('ouro')
    requests = [replace(r, max_new_tokens=1) for r in requests]
    cfg = arm_config(arm, 4, 1, 2, 64, draft_node_budget=8)
    cache = AuditedCache() if cfg.use_suffix else None
    rows, summary = batch_decode(requests, [backbone() for _ in range(4)], cfg,
                                 schedule='continuous', cache=cache)
    assert len(rows) == len(requests)
    assert all(r['n_tokens'] == 1 for r in rows)
    assert summary['core_calls'] == 0
    if cache:
        assert cache.responses == {}
