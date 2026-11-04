"""Fixed weights and online initial states preserve decoding and request isolation."""
from dataclasses import replace
import pytest
import torch

from scripts.loop_batch_bench import arm_config, parser, ARMS
from llm_memory_editability.loop_budget_decode import SuffixController, budget_tree_decode
from llm_memory_editability.loop_batch_decode import batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
from test_loop_budget_decode import BudgetFakeLoop, BoundedSuffix
from test_loop_tree_decode import greedy
from test_loop_optimized_decode import setup, requests

SETTINGS = [('budget-fixed', c, .1) for c in (.5, .7, .9)] + [
    ('budget-online', 1., .1), ('budget-online', .7, .1), ('budget-online', .7, .03)]


def config(arm, initial, eta):
    return arm_config(arm, 4, 1, 2, 4, draft_node_budget=16, model_top_k=2,
                      child_suffix_caps=(4, 4), suffix_c_initial=initial, suffix_eta=eta)


@pytest.mark.parametrize('arm,initial,eta', SETTINGS)
def test_greedy_feedback_and_initial_state_reset(arm, initial, eta):
    cfg = replace(config(arm, initial, eta), trace_controller=True)
    cache = BoundedSuffix()
    outputs = []
    for _ in range(2):
        tokens, s = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3, 1, 2], cfg,
                                      60, set(), cache, 'same-request')
        assert tokens == greedy([1, 2, 3, 1, 2], 60, set())
        assert s['c_initial'] == initial
        assert cfg.c_min <= s['c_observed_min'] <= s['c_final'] <= s['c_observed_max'] <= cfg.c_max
        assert s['peak_draft_nodes'] <= 16
        if arm == 'budget-fixed':
            assert s['controller_updates'] == s['feedback_created'] == 0
            assert s['c_final'] == pytest.approx(initial)
        else:
            assert s['controller_updates'] > 0
            assert s['controller_trace'][0]['c_before'] == pytest.approx(initial)
        outputs.append((tokens, s['controller_trace'], s['rows']))
    assert outputs[0] == outputs[1]


def test_fixed_one_matches_raw_work_and_output():
    results = []
    for arm in ('budget-raw', 'budget-fixed'):
        tokens, s = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3, 1, 2],
                                      config(arm, 1., .1), 60, set(), BoundedSuffix())
        results.append((tokens, [s[k] for k in ('rows', 'calls', 'nodes', 'speculate_calls',
                                               'wasted_rows', 'readout_rows')]))
    assert results[0] == results[1]


def test_initial_prediction_is_saved_and_updates_from_initial_log_weight():
    ctl = SuffixController(config('budget-online', .7, .1))
    ctl.observe((1,), 2, .8)
    assert ctl.records[(1,)][2].prediction == pytest.approx(.56)
    event = ctl.verify((1,), 2)
    assert event['c_before'] == pytest.approx(.7)
    assert event['delta'] == pytest.approx(.44)
    import math
    assert ctl.c == pytest.approx(.7 * math.exp(.044))


@pytest.mark.parametrize('initial', [0., .1, 3., float('nan'), float('inf')])
def test_invalid_initial_rejected(initial):
    with pytest.raises(ValueError, match='initial coefficient'):
        config('budget-online', initial, .1)


def test_raw_name_cannot_hide_fixed_reweighting_and_cli_is_wired():
    with pytest.raises(ValueError, match='raw scoring'):
        config('budget-raw', .7, .1)
    args = parser().parse_args(['--model','huginn-3.5b','--model-path','weights',
                               '--questions','input','--out','output','--suffix-c-initial','.7'])
    assert args.suffix_c_initial == .7
    assert 'budget-fixed' in ARMS


@pytest.mark.parametrize('arm,initial,eta', SETTINGS)
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_tuning_packed_graphs_and_recycled_slots(arm, initial, eta, device):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        make = setup('huginn', device, torch.bfloat16 if device == 'cuda' else torch.float32)
        cfg = config(arm, initial, eta)
        refs, _ = batch_decode(requests(), [make(), make()], cfg,
                               cache=CompletionSuffixCache.arctic(), schedule='continuous')
        pool = SharedKVPool(2)
        slots = [make(pool), make(pool)]
        engine = PackedLoopExecutor(slots, cuda_graph=device == 'cuda', max_graphs=32)
        batch_decode(requests(), slots, cfg, cache=CompletionSuffixCache.arctic(),
                     schedule='continuous', executor=engine, prefill_chunk_size=2)
        engine.graphs.freeze()
        rows, summary = batch_decode(requests(), slots, cfg, cache=CompletionSuffixCache.arctic(),
                                    schedule='continuous', executor=engine, prefill_chunk_size=2)
        assert [r['tokens'] for r in rows] == [r['tokens'] for r in refs]
        assert all(r['stats']['peak_draft_nodes'] <= 16 and r['stats']['cap_skips'] == 0 for r in rows)
        if arm == 'budget-fixed':
            assert all(r['stats']['controller_updates'] == 0 and
                       abs(r['stats']['c_final'] - initial) < 1e-12 for r in rows)
        if device == 'cuda':
            assert summary['cuda_graph_replays'] > 0
    finally:
        torch.set_num_threads(previous)
