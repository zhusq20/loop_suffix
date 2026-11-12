from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_suffix_union_probe import merge_drafts, union_speculate
from llm_memory_editability.loop_prob_tree import _edge_probs


def draft(tokens, parents, probs, score):
    return SimpleNamespace(token_ids=tokens, parents=parents, probs=probs, score=score)


def test_shared_prefix_keeps_winner_edge_and_loser_conditional_extension():
    local = draft([1, 2], [-1, 0], [.8, .4], 2)
    history = draft([1, 3, 4], [-1, 0, -1], [.4, .3, .2], 1)
    merged = merge_drafts(local, history)
    assert merged.token_ids == [1, 2, 3, 4]
    assert merged.parents == [-1, 0, 0, -1]
    assert _edge_probs(merged) == pytest.approx([.8, .5, .75, .2])
    assert merged.match_len is None


@pytest.mark.parametrize('scores,expected', [((1, 1), [.8, .2]), ((1, 2), [.2, .8])])
def test_winner_order_and_each_source_retains_original_query_cap(scores, expected):
    calls = []
    def source(token, prob, score):
        def speculate(rid, context, **options):
            calls.append((rid, context, options))
            return draft([token], [-1], [prob], score)
        return SimpleNamespace(speculate=speculate)
    cache = SimpleNamespace(local=source(1, .8, scores[0]), history=source(2, .2, scores[1]))
    merged = union_speculate(cache, 7, [9], max_spec_tokens=8, use_tree_spec=True)
    assert merged.probs == expected
    assert calls == [(7, [9], dict(max_spec_tokens=4, use_tree_spec=True))] * 2


def test_empty_sources():
    merged = merge_drafts(draft([], [], [], 0), draft([], [], [], 0))
    assert merged.token_ids == merged.parents == merged.probs == []


def test_real_cache_union_preserves_greedy_tokens_and_budget():
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    from llm_memory_editability.loop_budget_decode import budget_tree_decode
    from test_loop_budget_decode import BudgetFakeLoop, config
    baseline, _ = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3, 1, 2],
        config(8, scoring='raw', top_m=2, suffix_room=4), 32, set(), CompletionSuffixCache.arctic(), 7)
    cache = CompletionSuffixCache.arctic()
    cache.speculate = lambda *a, **kw: union_speculate(cache, *a, **kw)
    actual, stats = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3, 1, 2],
        config(8, scoring='raw', top_m=2, suffix_room=8), 32, set(), cache, 7)
    assert actual == baseline
    assert stats['rows'] <= 9 * stats['calls']
