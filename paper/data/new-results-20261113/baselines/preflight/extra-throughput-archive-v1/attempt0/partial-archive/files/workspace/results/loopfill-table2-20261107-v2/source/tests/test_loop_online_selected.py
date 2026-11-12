"""Online feedback with the selected small budgets and ranked suffix caps."""
import pytest
import torch

from scripts.loop_batch_bench import arm_config
from llm_memory_editability.loop_budget_decode import budget_tree_decode
from llm_memory_editability.loop_batch_decode import batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
from test_loop_budget_decode import BudgetFakeLoop, BoundedSuffix
from test_loop_tree_decode import greedy
from test_loop_optimized_decode import setup, requests

SETTINGS = [(16, 4, (4, 4)), (16, 4, (4, 0)), (32, 8, (8, 8))]


def configuration(n, q, caps):
    return arm_config('budget-online', 4, 1, 2, q, draft_node_budget=n,
                      model_top_k=2, child_suffix_caps=caps)


@pytest.mark.parametrize('n,q,caps', SETTINGS)
def test_selected_feedback_preserves_greedy_and_rank_query_policy(n, q, caps):
    cfg = configuration(n, q, caps)
    prompt = [1, 2, 3, 1, 2]
    tokens, stats = budget_tree_decode(BudgetFakeLoop(4, 1), prompt, cfg, 60, set(), BoundedSuffix())
    assert tokens == greedy(prompt, 60, set())
    assert stats['controller_updates'] > 0
    assert stats['scoring'] == 'online_suffix_i'
    assert .25 <= stats['c_observed_min'] <= stats['c_final'] <= stats['c_observed_max'] <= 2
    assert stats['peak_draft_nodes'] <= n
    if caps[1] == 0:
        assert stats['child_suffix_rank2_queries'] == 0
        assert stats['child_suffix_rank2_skipped'] > 0


@pytest.mark.parametrize('n,q,caps', SETTINGS)
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_selected_online_packed_graphs_and_recycling(n, q, caps, device):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        make = setup('huginn', device, torch.bfloat16 if device == 'cuda' else torch.float32)
        cfg = configuration(n, q, caps)
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
        assert all(r['stats']['peak_draft_nodes'] <= n and r['stats']['cap_skips'] == 0 for r in rows)
        if device == 'cuda':
            assert summary['cuda_graph_replays'] > 0
    finally:
        torch.set_num_threads(previous)
