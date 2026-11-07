"""Sampling distribution, speculative RNG isolation, and full decoder integration."""
import pytest
import torch
from llm_memory_editability.loop_sampling import TargetSampler, request_seed
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
from llm_memory_editability.loop_tree_decode import tree_decode_steps
from llm_memory_editability.loop_budget_decode import BudgetConfig, budget_tree_decode_steps
from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig, loopspec_decode_steps
from llm_memory_editability.loop_decode_steps import run_single
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from scripts.loop_batch_bench import arm_config
from scripts.loop_latency_worker import drive, summarize
from test_loop_optimized_decode import setup


def test_target_sampler_distribution_and_temperature():
    torch.set_num_threads(1)
    for temperature in [0.5, 1., 2.]:
        sampler = TargetSampler(temperature, 37, 40000, 'cpu')
        logits = torch.tensor([0., 1., 2.]).expand(40000, -1)
        samples = torch.tensor(sampler(logits, list(range(40000))))
        observed = samples.bincount(minlength=3)/len(samples)
        torch.testing.assert_close(observed, (logits[0]/temperature).softmax(-1), atol=.01, rtol=0)


def test_off_path_and_row_reordering_do_not_consume_rng():
    s = TargetSampler(1., 123, 200, 'cpu')
    logits = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    expected = s(logits, [5, 9])
    s(torch.randn(50, 3), list(range(50)))  # irrelevant speculative branches
    assert s(logits.flip(0), [9, 5]) == expected[::-1]
    assert s(logits, [5, 9]) == expected
    assert TargetSampler(1.,123,200,'cpu')(logits,[5,9]) == expected
    assert request_seed(0,'m','d','id') != request_seed(0,'m','d/warmup','id')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('family', ['ouro','huginn'])
@pytest.mark.parametrize('arm', ['ar','wavefront','loopspec-dual','budget-suffix','budget-online'])
@torch.no_grad()
def test_sampling_packed_matches_serial_and_ar(family, arm, device):
    torch.set_num_threads(1)
    make = setup(family, device=device, max_nodes=128)
    prompt = [1, 2, 3, 6, 4]
    def execute(method, packed):
        cfg = arm_config(method,4,1,2,64,draft_node_budget=8,model_top_k=2)
        b = make(SharedKVPool(1,pad_decode_rows=False)) if packed else make()
        b.target_sampler = TargetSampler(1.,971,200,device)
        cache = CompletionSuffixCache.arctic() if cfg.use_suffix else None
        fn = budget_tree_decode_steps if isinstance(cfg,BudgetConfig) else (
            loopspec_decode_steps if isinstance(cfg,LoopSpecConfig) else tree_decode_steps)
        steps = fn(b,prompt,cfg,24,set(),cache=cache,req_id=0,max_nodes=128)
        if packed:
            return drive(b,PackedLoopExecutor([b],cuda_graph=False),steps)
        return run_single(b,steps)
    target,_=execute('ar',False)
    serial,_=execute(arm,False)
    packed,stats=execute(arm,True)
    assert serial == packed == target
    assert stats['tokens']==23
    if arm=='ar': assert stats['calls']==4*23


def test_acceptance_metrics_distinguish_request_mean_from_pooled():
    rows=[dict(tokens=[1,2],stats=dict(tokens=1,calls=4,decode_s=1,prefill_s=1,covered_tokens=1),stop_reason='eos'),
          dict(tokens=[1,2,3,4],stats=dict(tokens=3,calls=12,decode_s=3,prefill_s=1,from_fill=0),stop_reason='length')]
    s=summarize(rows,4)
    assert s['gamma']==1 and s['draft_hits']==1
    assert s['mean_acceptance_rate']==.5 and s['pooled_acceptance_rate']==.25
