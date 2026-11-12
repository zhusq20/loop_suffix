from dataclasses import replace

import pytest

from scripts.loop_draft_trace import DraftTrace
from llm_memory_editability.loop_budget_decode import budget_tree_decode
from test_loop_budget_decode import BudgetFakeLoop, BoundedSuffix, config


@pytest.mark.parametrize('scoring', ['raw', 'online_suffix_i', 'fixed_suffix_length', 'online_suffix_length'])
@pytest.mark.parametrize('budget', [0, 1, 4, 16])
def test_observer_reconstructs_work_and_does_not_change_decode(scoring, budget):
    cfg = config(budget, scoring=scoring, top_m=2, suffix_room=4)
    args = ([1, 2, 3, 1, 2], cfg, 32, set())
    expected, baseline = budget_tree_decode(BudgetFakeLoop(4, 1), *args, BoundedSuffix())
    trace = DraftTrace(4)
    got, stats = budget_tree_decode(BudgetFakeLoop(4, 1), *args, BoundedSuffix(), observer=trace.observe)
    assert got == expected
    for key in ('calls', 'rows', 'nodes', 'wasted_rows', 'covered_tokens', 'controller_updates'):
        assert stats[key] == baseline[key]
    counts, nodes, entries = trace.finish(got, stats)
    assert counts['initial_root_wait'] + counts['remaining_root_wait'] == stats['calls']
    assert len(entries) == len(got)-1
    assert len(nodes) >= len(got)-1


def test_no_decode_has_no_root_wait():
    trace = DraftTrace(4)
    out, stats = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2], config(), 1, set(),
                                   BoundedSuffix(), observer=trace.observe)
    counts, _, _ = trace.finish(out, stats)
    assert counts['initial_root_wait'] == counts.get('calls', 0) == 0
