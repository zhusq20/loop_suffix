"""Per-parent model-rank caps constrain queries without changing verification."""
from types import SimpleNamespace

import pytest

from scripts.loop_batch_bench import arm_config, parser
from llm_memory_editability.loop_budget_decode import budget_tree_decode
from test_loop_budget_decode import BudgetFakeLoop, BoundedSuffix
from test_loop_tree_decode import greedy


class RecordingSuffix(BoundedSuffix):
    def __init__(self, empty=False):
        super().__init__()
        self.empty = empty
        self.caps = []

    def speculate(self, *args, **kwargs):
        self.caps.append(kwargs['max_spec_tokens'])
        assert kwargs['max_spec_tokens'] > 0
        if self.empty:
            return SimpleNamespace(token_ids=[], parents=[], probs=[], match_len=0)
        return super().speculate(*args, **kwargs)


def cfg(caps):
    return arm_config('budget-raw', 4, 1, 2, 4, model_top_k=2,
                      draft_node_budget=16, child_suffix_caps=caps)


@pytest.mark.parametrize('caps', [(4, 4), (4, 2), (4, 0)])
@pytest.mark.parametrize('empty', [False, True])
def test_caps_preserve_greedy_and_zero_never_reaches_provider(caps, empty):
    cache = RecordingSuffix(empty)
    events = []
    prompt = [1, 2, 3, 1, 2]
    tokens, stats = budget_tree_decode(BudgetFakeLoop(4, 1), prompt, cfg(caps), 60, set(), cache,
        observer=lambda kind, **data: events.append((kind, data)))
    assert tokens == greedy(prompt, 60, set())
    queries = [e for kind, e in events if kind == 'child_suffix_query']
    assert {e['rank'] for e in queries if e['rank'] is not None} == {0, 1}
    for e in queries:
        assert e['cap'] == (4 if e['rank'] is None else caps[e['rank']])
        assert e['returned_nodes'] <= e['cap']
    assert stats['peak_draft_nodes'] <= 16
    assert stats['speculate_calls'] == len(cache.caps)
    assert stats['child_suffix_rank1_queries'] > 0
    if caps[1] == 0:
        assert stats['child_suffix_rank2_queries'] == 0
        assert stats['child_suffix_rank2_returned_nodes'] == 0
        assert stats['child_suffix_rank2_skipped'] > 0
        assert set(cache.caps) == {4}  # Includes unchanged root queries.
    elif caps[1] == 2:
        assert set(cache.caps) == {2, 4}
    for kind, e in events:
        if kind == 'selection':
            paths = set(e['paths'])
            assert all(p == e['root'] or p[:-1] in paths for p in paths)


def test_symmetric_caps_preserve_legacy_selection_and_work():
    outputs = []
    for caps in (None, (4, 4)):
        events = []
        tokens, stats = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3, 1, 2], cfg(caps),
            80, set(), BoundedSuffix(), observer=lambda kind, **data: events.append((kind, data)))
        outputs.append((tokens, {k: stats[k] for k in ('calls', 'rows', 'readout_rows',
            'speculate_calls', 'child_suffix_queries', 'wasted_rows')}, events))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('caps', [(4,), (4, -1), (4, 2, 1), (4, 1.5)])
def test_invalid_caps_rejected(caps):
    with pytest.raises(ValueError, match='child suffix caps'):
        cfg(caps)


def test_cli_explicit_zero_and_default():
    args = parser().parse_args(['--model', 'huginn-3.5b', '--model-path', 'weights',
                               '--questions', 'input', '--out', 'output',
                               '--child-suffix-caps', '4', '0'])
    assert args.child_suffix_caps == [4, 0]
    assert cfg(None).child_suffix_caps is None
