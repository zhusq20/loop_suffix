import random
from types import SimpleNamespace

import numpy as np
import pytest

from llm_memory_editability.loop_prob_tree import (
    ABSENT,
    Calibration,
    ProbConfig,
    ProbReplay,
    predicted_ms,
)
from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request

PROMPT = [100, 101, 102]


class NoisyCache:
    """Drafts the true continuation with a few deterministic corruptions."""

    def __init__(self, out, seed):
        self.out = out
        self.seed = seed
        self.added: list[int] = []

    def start_request(self, req_id, prompt_token_ids):
        self.added = []

    def add_active_response(self, req_id, token_ids):
        self.added.extend(int(t) for t in token_ids)

    def stop_request(self, req_id):
        pass

    def speculate(self, req_id, context, max_spec_tokens=None, **kwargs):
        n = len(context) - len(PROMPT)
        rng = random.Random(self.seed * 1000 + n + 7 * int(context[-1]))
        chain = []
        for t in self.out[n : n + rng.randint(0, 5)]:
            chain.append(t if rng.random() < 0.75 else 900 + rng.randint(0, 9))
        chain = chain[:max_spec_tokens]
        probs = [0.9**k for k in range(1, len(chain) + 1)]
        return SimpleNamespace(
            token_ids=chain,
            parents=list(range(-1, len(chain) - 1)),
            probs=probs,
            match_len=3,
        )


def random_case(seed, T, L=40):
    rng = np.random.default_rng(seed)
    out = [int(t) for t in rng.integers(1, 20, size=L)]
    ids, probs = {}, {}
    for r in range(1, T):
        rows = rng.integers(20, 60, size=(L + 1, 8))
        hit = rng.random(L + 1) < 0.5 + 0.4 * r / T
        second = rng.random(L + 1) < 0.5
        for i in range(L):
            if hit[i]:
                rows[i, 0] = out[i]
            elif second[i]:
                rows[i, 1] = out[i]
        ids[r] = rows
        probs[r] = np.sort(rng.random((L + 1, 8)), axis=1)[:, ::-1].astype(np.float32)
    return out, ids, probs


def flat_calibration(depths, e_top=0.6, e_rest=0.2, e_suffix=0.7):
    c = Calibration()
    for r in depths:
        for k in list(range(8)) + [ABSENT]:
            for flag in (False, True):
                for pb in (0.0, 0.5, 0.95):
                    e = e_top if k == 0 else e_rest
                    for j in range(100):
                        c.add_shallow(r, k, pb, flag, j < 100 * e)
    for j in range(100):
        c.add_suffix(0.9, 3, j < 100 * e_suffix)
    return c


@pytest.mark.parametrize("T,d,K", [(4, 1, 16), (4, 2, 3), (8, 2, 16), (8, 3, 4)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_legacy_matches_replay_request(T, d, K, seed):
    out, ids, probs = random_case(seed, T)
    arm = ArmConfig("x", T, d, K, use_suffix=True, use_fill=True)
    old = replay_request(0, PROMPT, out, ids, arm, NoisyCache(out, seed))
    cfg = ProbConfig("x", T, d, (d,), max_draft=K, legacy=True)
    cache = NoisyCache(out, seed)
    new = ProbReplay(0, PROMPT, out, ids, probs, cfg, cache, None).run()
    for key in ("tokens", "steps", "calls"):
        assert new[key] == old[key], key
    assert new.get("from_suffix", 0) == old["from_suffix"]
    assert new.get("from_fill", 0) == old["from_fill"]
    assert cache.added == out
    assert sum(new["hist"].values()) == new["calls"]


def test_nothing_placed_is_autoregressive():
    out, ids, probs = random_case(3, 8)
    cfg = ProbConfig("x", 8, 2, (2,), tau=2.0, use_suffix=False)
    s = ProbReplay(0, PROMPT, out, ids, probs, cfg, None, flat_calibration([2])).run()
    assert s["tokens"] == len(out) - 1 and s["calls"] == 8 * (len(out) - 1)
    assert s["rows"] == s["calls"]


def test_top1_rule_with_sure_edges_equals_shallow_chain():
    out, ids, probs = random_case(4, 8)
    arm = ArmConfig("x", 8, 2, 5, use_suffix=False, use_fill=True)
    old = replay_request(0, PROMPT, out, ids, arm, None)
    calib = flat_calibration([2], e_top=1.0)
    cfg = ProbConfig("x", 8, 2, (2,), tau=0.5, top_m=1, max_draft=5, use_suffix=False)
    new = ProbReplay(0, PROMPT, out, ids, probs, cfg, None, calib).run()
    assert (new["tokens"], new["steps"], new["calls"]) == (
        old["tokens"],
        old["steps"],
        old["calls"],
    )


def test_siblings_save_calls_but_add_rows():
    out, ids, probs = random_case(5, 8)
    calib = flat_calibration([2])
    kw = dict(tau=0.05, use_suffix=False, max_draft=None)
    one = ProbReplay(
        0, PROMPT, out, ids, probs, ProbConfig("x", 8, 2, (2,), top_m=1, **kw), None, calib
    ).run()
    two = ProbReplay(
        0, PROMPT, out, ids, probs, ProbConfig("x", 8, 2, (2,), top_m=2, **kw), None, calib
    ).run()
    assert two["calls"] < one["calls"] and two["tokens"] == one["tokens"]
    assert two["rows"] / two["calls"] > one["rows"] / one["calls"]
    assert two.get("from_sibling", 0) > 0


def test_row_cap_of_one_falls_back_to_autoregressive():
    out, ids, probs = random_case(6, 8)
    calib = flat_calibration([2, 4])
    cfg = ProbConfig("x", 8, 2, (2, 4), tau=0.0, row_cap=1)
    s = ProbReplay(0, PROMPT, out, ids, probs, cfg, NoisyCache(out, 6), calib).run()
    assert s["calls"] == 8 * (len(out) - 1) and s["cap_kills"] > 0


def test_more_candidates_never_cost_calls_without_cap():
    out, ids, probs = random_case(7, 8)
    calib = flat_calibration([2, 4])
    calls = []
    for tau in (0.5, 0.1, 0.01, 0.0):
        cfg = ProbConfig("x", 8, 2, (2, 4), tau=tau)
        calls.append(
            ProbReplay(0, PROMPT, out, ids, probs, cfg, NoisyCache(out, 7), calib).run()["calls"]
        )
    assert calls == sorted(calls, reverse=True)


def test_calibration_round_trip_and_fallback():
    c = flat_calibration([2], e_top=0.8)
    d = Calibration.from_json(c.to_json())
    assert d.shallow_prob(2, 0, 0.5, False) == pytest.approx(0.8)
    assert d.suffix_prob(0.9, 3) == pytest.approx(0.7)
    # an unseen probability bin falls back to the same depth and rank
    assert d.shallow_prob(2, 0, 0.3, False) == pytest.approx(0.8)


def test_predicted_ms_sums_cost_over_calls():
    assert predicted_ms({(1, 0): 3, (5, 2): 2}, lambda rows, reads: rows + 10 * reads) == 3 + 2 * 25


@pytest.mark.parametrize("eager", [False, True])
def test_suffix_tree_receives_every_token_once(eager):
    out, ids, probs = random_case(8, 8)
    cache = NoisyCache(out, 8)
    cfg = ProbConfig("x", 8, 2, (2, 4), tau=0.01, commit_on_accept=eager)
    ProbReplay(0, PROMPT, out, ids, probs, cfg, cache, flat_calibration([2, 4])).run()
    assert cache.added == out


class LoopCache(NoisyCache):
    """Copies a periodic output from what the tree has seen, as a suffix tree would."""

    period = 5

    def speculate(self, req_id, context, max_spec_tokens=None, **kwargs):
        n = len(context) - len(PROMPT)
        known = len(self.added) + self.period  # out[i] == out[i - period]
        chain = self.out[n : min(n + max_spec_tokens, known)]
        return SimpleNamespace(
            token_ids=chain,
            parents=list(range(-1, len(chain) - 1)),
            probs=[0.99**k for k in range(1, len(chain) + 1)],
            match_len=8,
        )


def test_eager_commit_keeps_suffix_drafts_current_in_long_steps():
    out, ids, probs = random_case(9, 8, L=120)
    out = [out[i % LoopCache.period] for i in range(len(out))]
    ids[2][: len(out), 0] = out  # fills are always right, so the step never ends
    calib = flat_calibration([2], e_top=0.9, e_suffix=0.95)
    calls = {}
    for eager in (False, True):
        cfg = ProbConfig("x", 8, 2, (2,), tau=0.01, top_m=1, commit_on_accept=eager)
        s = ProbReplay(0, PROMPT, out, ids, probs, cfg, LoopCache(out, 9), calib).run()
        calls[eager] = (s["calls"], s["steps"])
    assert calls[True][1] == 1 and calls[True][0] < calls[False][0]


def test_insitu_and_joint_tables_fall_back_and_round_trip():
    c = flat_calibration([2, 4], e_top=0.8, e_suffix=0.7)
    assert c.suffix_prob_insitu(0.9, 3, True, 2) == pytest.approx(0.7)  # empty: offline table
    assert c.joint_prob(4, 0, 0, 0.5, False) == pytest.approx(0.8)  # empty: shallow table
    for j in range(100):
        c.add_insitu(0.9, 3, True, 2, j < 10)
        c.add_joint(4, ABSENT, 0, 0.5, False, j < 30)
    d = Calibration.from_json(c.to_json())
    assert d.suffix_prob_insitu(0.9, 3, True, 2) == pytest.approx(0.1)
    assert d.suffix_prob_insitu(0.9, 3, False, 2) == pytest.approx(0.7)
    assert d.joint_prob(4, ABSENT, 0, 0.5, False) == pytest.approx(0.3)


def test_deep_readouts_without_expansion_place_nothing():
    out, ids, probs = random_case(10, 8)
    calib = flat_calibration([2, 4])
    kw = dict(tau=0.01, joint=True)
    cfg = ProbConfig("x", 8, 2, (2, 4), deep_expand=False, deep_raise=False, **kw)
    s = ProbReplay(0, PROMPT, out, ids, probs, cfg, NoisyCache(out, 10), calib).run()
    assert s.get("from_deep", 0) == 0 and s["tokens"] == len(out) - 1
    full = ProbReplay(
        0, PROMPT, out, ids, probs, ProbConfig("x", 8, 2, (2, 4), **kw), NoisyCache(out, 10), calib
    ).run()
    assert full.get("from_deep", 0) > 0


def test_pruning_removes_low_candidates_and_thins_calls():
    out, ids, probs = random_case(11, 8)
    calib = flat_calibration([2, 4], e_rest=0.05)
    got = {}
    for prune in (False, True):
        cfg = ProbConfig("x", 8, 2, (2, 4), tau=0.04, joint=True, prune=prune)
        s = ProbReplay(0, PROMPT, out, ids, probs, cfg, NoisyCache(out, 11), calib).run()
        got[prune] = (s["rows"] / s["calls"], s.get("prune_kills", 0), s["tokens"])
    assert got[True][1] > 0 and got[False][1] == 0
    assert got[True][0] < got[False][0]  # fewer live rows per block call
    assert got[True][2] == got[False][2] == len(out) - 1


def test_edge_log_holds_only_offers_under_the_greedy_path():
    out, ids, probs = random_case(12, 8)
    log: list = []
    cfg = ProbConfig("x", 8, 2, (2,), tau=0.01)
    ProbReplay(
        0, PROMPT, out, ids, probs, cfg, NoisyCache(out, 12), flat_calibration([2]), edge_log=log
    ).run()
    assert log and all(lead >= 0 for _, _, _, lead, _ in log)
    assert 0 < sum(hit for *_, hit in log) < len(log)
