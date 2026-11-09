"""Sampling contracts that matter for state reuse and experimental interpretation."""
import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from llm_memory_editability.loop_budget_decode import Candidate, budget_tree_decode
from llm_memory_editability.loop_tree_decode import _top_probs
from llm_memory_editability.loop_tree_sampling import DraftTreeSampler, sample_candidates
from test_loop_budget_decode import BoundedSuffix, config
from test_loop_existing_probs import ExactFakeLoop
from test_loop_tree_decode import greedy


def pool_for(edges):
    root = (9,)
    pool = {root: Candidate(root, 1., "model", "root")}
    for tail, score in edges:
        path = root + tail
        pool[path] = Candidate(path, score, "model", "fill")
    return root, pool


class Draws:
    def __init__(self, indices):
        self.indices = indices
        self.calls = 0

    def choice(self, count, size, replace, p):
        self.calls += 1
        assert size == len(self.indices) and replace
        assert p.sum() == pytest.approx(1.)
        return self.indices


def test_duplicates_and_whole_path_budget_skip_never_refill():
    root, pool = pool_for([((1,), .8), ((2,), .7), ((1, 1), .9), ((2, 1), .8)])
    rng = Draws([0, 2, 0, 2])
    selected, counts = sample_candidates(pool, root, 4, 1., rng)
    assert set(selected) == {root, root+(1,), root+(1, 1)}
    assert counts["draws"] == 4 and counts["duplicates"] == 2 and rng.calls == 1
    rng = Draws([2, 3, 3])
    selected, counts = sample_candidates(pool, root, 3, 1., rng)
    assert set(selected) == {root, root+(1,), root+(1, 1)}
    assert counts["budget_skips"] == 2 and counts["ancestor_additions"] == 1
    assert counts["closure_nodes"] == 4


def test_distribution_is_normalized_path_product_not_softmax_of_scores():
    root, pool = pool_for([((1,), .8), ((2,), .7), ((1, 1), .9), ((2, 1), .8)])

    class Inspect(Draws):
        def choice(self, count, size, replace, p):
            expected = np.asarray([.8, .7, .72, .56])
            assert p == pytest.approx(expected / expected.sum())
            return super().choice(count, size, replace, p)

    sample_candidates(pool, root, 4, 1., Inspect([0, 1, 2, 3]))


def test_empty_zero_invalid_and_zero_exclusion():
    root, pool = pool_for([])
    assert sample_candidates(pool, root, 16, 1., Draws([]))[0] == pool
    root, pool = pool_for([((1,), 0.), ((2,), 0.)])
    selected, counts = sample_candidates(pool, root, 1, 1., Draws([]))
    assert set(selected) == {root, root+(1,)} and counts["all_zero_fallbacks"] == 1
    for bad in [None, math.nan, -1., math.inf]:
        pool[root+(1,)].raw_score = bad
        with pytest.raises((ValueError, TypeError)):
            sample_candidates(pool, root, 1, 1., Draws([]))
    pool[root+(1,)].raw_score = 1.
    for seed in range(20):
        selected, _ = sample_candidates(pool, root, 16, 1., np.random.default_rng(seed))
        assert root+(2,) not in selected


def test_long_paths_and_underflow_use_log_weights():
    root, pool = pool_for([((1,)*i, .001) for i in range(1, 130)])
    selected, counts = sample_candidates(pool, root, 16, 1., np.random.default_rng(0))
    assert pool[root+(1,)*129].p == 0 and counts["all_zero_fallbacks"] == 0
    logits = torch.tensor([[0., -1000., -1001.]])
    ids, ps, logs = _top_probs(logits, 1, [[1, 2]], True)[0]
    assert ids == [0, 1, 2] and ps[1:] == [0., 0.]
    assert logs == pytest.approx([0., -1000., -1001.])
    root, pool = pool_for([((1,), 0.), ((2,), 0.)])
    for i in (1, 2):
        pool[root+(i,)].raw_log_score = logs[i]
    _, counts = sample_candidates(pool, root, 16, 1., np.random.default_rng(0))
    assert counts["all_zero_fallbacks"] == 0 and counts["draws"] == 16


def test_stream_identity_event_recovery_and_global_rng_independence():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    a = DraftTreeSampler(0, "ouro", "dataset:request-7")
    values = [a.generator().random(16) for _ in range(4)]
    resumed = DraftTreeSampler(0, "ouro", "dataset:request-7", event=3)
    assert np.array_equal(values[3], resumed.generator().random(16))
    assert not np.array_equal(values[0], DraftTreeSampler(1, "ouro", "dataset:request-7").generator().random(16))
    assert before[0] == random.getstate()
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert torch.equal(before[2], torch.get_rng_state())
    root, pool = pool_for([((1,), .5), ((2,), .4), ((1, 1), .5)])
    selected = []
    for p in (pool, dict(reversed(list(pool.items())))):
        selected.append(sample_candidates(p, root, 2, 1., DraftTreeSampler(0, "m", 1).generator())[0])
    assert list(selected[0]) == list(selected[1])


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("continuation", ["all", "none"])
def test_initial_boundary_once_per_event_replay_and_greedy(exact, continuation):
    prompt = [1, 2, 3, 4, 5, 6, 1, 2]
    cfg = config(16, scoring="raw", top_m=2, suffix_room=4,
                 child_suffix_policy=continuation, existing_child_probs=exact,
                 selection_policy="sample", draft_model_id="fake")
    events = []
    out, stats = budget_tree_decode(ExactFakeLoop(4, 1), prompt, cfg, 80, set(), BoundedSuffix(),
                                   "q1", observer=lambda e, **d: events.append((e, d)))
    assert out == greedy(prompt, 80, set())
    sampling = [d for e, d in events if e == "sampling"]
    assert len(sampling) == len({d["tick"] for d in sampling}) == stats["sampling_events"]
    assert all(d["tick"] > 0 for d in sampling)
    initial = [d for e, d in events if e == "proposal" and d["tick"] == 0]
    first = next(d for e, d in events if e == "selection")
    assert set(d["path"] for d in initial) <= set(first["paths"])
    for event, data in events:
        if event == "selection":
            paths = set(data["paths"])
            assert len(paths) <= 17
            assert all(p == data["root"] or p[:-1] in paths for p in paths)
    again, other = budget_tree_decode(ExactFakeLoop(4, 1), prompt, cfg, 80, set(), BoundedSuffix(), "q1")
    assert out == again and stats["sampling_counts"] == other["sampling_counts"]
    assert stats["tree_histograms"] == other["tree_histograms"]
    assert sum(int(k)*v for k, v in stats["tree_histograms"]["execution_rows"].items()) == stats["rows"]
    counts = stats["sampling_counts"]
    assert counts["draws"] == 16 * (stats["sampling_events"] - counts["all_zero_fallbacks"] - counts["empty_events"])
    with pytest.raises(ValueError):
        replace(cfg, positive_only=True)
