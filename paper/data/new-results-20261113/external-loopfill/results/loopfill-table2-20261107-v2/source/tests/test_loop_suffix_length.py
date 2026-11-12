"""Length evidence changes allocation without labels or task-specific thresholds."""
import math
from dataclasses import replace

import pytest
import torch

from scripts.loop_batch_bench import arm_config
from llm_memory_editability.loop_budget_decode import (
    Candidate, SuffixController, budget_tree_decode, select_candidates, suffix_score,
)
from llm_memory_editability.loop_batch_decode import batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, SharedKVPool
from test_loop_budget_decode import BoundedSuffix, BudgetFakeLoop, CorrectChain
from test_loop_optimized_decode import requests, setup
from test_loop_tree_decode import greedy


def config(arm='budget-online-length', **kw):
    return replace(arm_config(arm, 4, 1, 2, 4, draft_node_budget=8,
                             model_top_k=2, child_suffix_caps=(4, 4)), **kw)


def test_monotone_bounded_scores_and_non_saturating_unit_confidence():
    for c in (.25, 1., 2.):
        for q in (0., .01, .5, 1.):
            scores = [suffix_score(q, c, length) for length in (0, 1, 5, 20, 64)]
            assert scores == sorted(scores)
            assert all(0 <= p < 1 for p in scores)
        assert suffix_score(1., c, 1) < suffix_score(1., c, 20)
    # Long, moderate evidence can outrank short, certain historical frequency.
    assert suffix_score(.5, 1, 20) > suffix_score(1., 1, 1)
    assert suffix_score(0., 1, 20) == 0
    assert suffix_score(.5, 1, 20) > .5 > suffix_score(.5, 1, 0)


@pytest.mark.parametrize('gold', [2, 9])
def test_update_matches_log_loss_gradient_and_uses_immutable_snapshot(gold):
    ctl = SuffixController(config(trace_controller=True))
    ctl.observe((1,), 2, 1., 1)
    saved = ctl.records[(1,)][2]
    # Independent finite difference of log loss at prediction time.
    def loss(u):
        p = suffix_score(1., math.exp(u), 1)
        return -math.log(p if gold == 2 else 1-p)
    eps = 1e-5
    derivative = (loss(eps)-loss(-eps))/(2*eps)
    ctl.u = math.log(.7)
    ctl.observe((1,), 2, .1, 64)
    assert ctl.records[(1,)][2] is saved
    event = ctl.verify((1,), gold)
    assert event['delta'] == pytest.approx(-derivative)
    assert ctl.c == pytest.approx(.7*math.exp(-ctl.cfg.eta*derivative))
    assert event['predictions'][0]['match_len'] == 1
    assert event['predictions'][0]['prediction'] == pytest.approx(2/3)
    assert ctl.verify((1,), gold) is None
    ctl.observe((1,), 3, .9, 10)
    assert not ctl.records  # cannot create feedback after knowing the answer


def test_zero_evidence_is_not_trainable_and_offpath_records_are_discarded():
    ctl = SuffixController(config())
    ctl.observe((1,), 2, 0., 3)
    assert ctl.verify((1,), 2) is None
    assert ctl.updates == 0
    ctl.observe((1, 2), 3, .9, 1)
    ctl.observe((1, 9), 3, .9, 20)
    ctl.retain_prefix((1, 2))
    assert ctl.discarded == 1 and ctl.pending == 1
    ctl.verify((1, 2), 3)
    assert ctl.updates == 1 and ctl.pending == 0


def test_length_reorders_competing_paths_and_preserves_ancestor_closure():
    root = (7,)
    short, long, child, model = (7, 1), (7, 2), (7, 2, 3), (7, 4)
    pool = {root: Candidate(root, 1, 'model', 'root'),
            short: Candidate(short, 1., 'suffix', 'suffix', suffix_match_len=1),
            long: Candidate(long, .5, 'suffix', 'suffix', suffix_match_len=20),
            child: Candidate(child, .5, 'suffix', 'suffix', suffix_match_len=20),
            model: Candidate(model, .8, 'model', 'fill')}
    assert list(select_candidates(pool, root, 1, 1)) == [root, short]
    chosen = select_candidates(pool, root, 2, 1, length_conditioned=True)
    assert list(chosen) == [root, long, child]
    assert pool[model].e == .8
    for budget in range(5):
        chosen = select_candidates(pool, root, budget, .25, length_conditioned=True)
        assert all(p == root or p[:-1] in chosen for p in chosen)
    # A model readout becomes authoritative; old suffix length cannot boost it.
    pool[long].score_source, pool[long].raw_score = 'model', .1
    select_candidates(pool, root, 4, 2., length_conditioned=True)
    assert pool[long].e == .1


@pytest.mark.parametrize('length', [None, -1, 2.5, True])
def test_missing_or_invalid_length_is_not_silently_replaced(length):
    with pytest.raises(ValueError, match='match_len'):
        SuffixController(config()).observe((1,), 2, .5, length)


@pytest.mark.parametrize('arm', ['budget-online-length', 'budget-fixed-length'])
def test_draft_length_is_shared_by_edges_not_inflated_by_speculative_depth(arm):
    class MatchedChain(CorrectChain):
        def speculate(self, *args, **kw):
            draft = super().speculate(*args, **kw)
            draft.match_len = 2
            return draft
    events = []
    cfg = config(arm, use_model=False, trace_controller=True)
    out, stats = budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3], cfg, 12, set(),
                                   MatchedChain(), observer=lambda e, **d: events.append((e, d)))
    assert out == greedy([1, 2, 3], 12, set())
    selections = [d for e, d in events if e == 'selection']
    assert all(length == 2 for d in selections for p, length in d['suffix_match_lengths'].items()
               if p != d['root'] and d['scores'][p][0] == 'suffix')
    first = selections[0]
    assert all(score[2] == pytest.approx(suffix_score(score[1], 1, 2))
               for p, score in first['scores'].items() if p != first['root'])
    assert max(stats['accepted_lengths']) > 1
    assert (stats['controller_updates'] > 0) == (arm == 'budget-online-length')
    if arm == 'budget-fixed-length':
        assert stats['c_final'] == 1
    with pytest.raises(ValueError, match='match_len'):
        budget_tree_decode(BudgetFakeLoop(4, 1), [1, 2, 3], cfg, 12, set(), CorrectChain())


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('arm', ['budget-online-length', 'budget-fixed-length'])
def test_packed_continuous_real_suffix_provider_preserves_tokens(family, arm):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        make = setup(family, 'cpu', torch.float32)
        cfg = config(arm, trace_controller=True)
        refs, _ = batch_decode(requests(), [make(), make()], cfg,
                              cache=CompletionSuffixCache.arctic(), schedule='continuous')
        pool = SharedKVPool(2)
        slots = [make(pool), make(pool)]
        rows, summary = batch_decode(requests(), slots, cfg, cache=CompletionSuffixCache.arctic(),
                                     schedule='continuous', executor=PackedLoopExecutor(slots),
                                     prefill_chunk_size=2)
        assert [r['tokens'] for r in rows] == [r['tokens'] for r in refs]
        assert all(r['stats']['peak_draft_nodes'] <= 8 for r in rows)
        assert summary['refilled_requests'] > 0
    finally:
        torch.set_num_threads(old_threads)
