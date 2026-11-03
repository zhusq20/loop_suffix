"""The probability-threshold decoder equals greedy decoding and the replay's block calls."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_loop_tree_decode import (  # noqa: E402
    HUGINN,
    OURO,
    FakeLoop,
    FakeSuffix,
    greedy,
    greedy_next,
    shallow_next,
    tiny_huginn,
    tiny_ouro,
    top_list,
)

from llm_memory_editability.loop_prob_decode import prob_tree_decode  # noqa: E402
from llm_memory_editability.loop_prob_tree import (  # noqa: E402
    ABSENT,
    Calibration,
    ProbConfig,
    ProbReplay,
)

PROBS = [0.55, 0.2, 0.1, 0.06, 0.04, 0.03, 0.01, 0.01]


class ProbFakeLoop(FakeLoop):
    def readout_top(self, tree, groups, full, m):
        reads = [
            [(top_list(shallow_next(self.context(tree, n), r), m), PROBS[:m]) for n in g]
            for r, g in groups
        ]
        return reads, [greedy_next(self.context(tree, n)) for n in full]


class ProbFakeSuffix(FakeSuffix):
    def speculate(self, req_id, context, max_spec_tokens=None, **kwargs):
        d = super().speculate(req_id, context, max_spec_tokens)
        n = len(d.token_ids)
        return SimpleNamespace(
            token_ids=d.token_ids, parents=d.parents, probs=[0.8] * n, match_len=3
        )


def fake_reads(prompt, out, depths):
    rows = [list(prompt) + out[:i] for i in range(len(out) + 1)]
    ids = {r: np.array([top_list(shallow_next(c, r), 8) for c in rows]) for r in depths}
    probs = {r: np.tile(np.array(PROBS), (len(rows), 1)) for r in depths}
    return ids, probs


def calibration(depths):
    c = Calibration()
    for r in depths:
        for k in [*range(8), ABSENT]:
            for flag in (False, True):
                e = 0.7 if k == 0 else (0.2 if k == 1 else 0.02)
                for j in range(100):
                    c.add_shallow(r, k, PROBS[min(k, 7)], flag, j < 100 * e)
    for j in range(100):
        c.add_suffix(0.8, 3, j < 60)
        c.add_insitu(0.8, 3, True, 1, j < 10)
    return c


def configs(T, d, deep):
    two = (d, deep)
    return [
        ProbConfig("A", T, d, two, tau=0.01),
        ProbConfig("A t0.1", T, d, two, tau=0.1),
        ProbConfig("C", T, d, two, tau=0.03, joint=True, prune=True, suffix_calibration="insitu"),
        ProbConfig(
            "D", T, d, two, tau=0.03, joint=True, prune=True, deep_expand=False, deep_raise=False
        ),  # fmt: skip
        ProbConfig("E", T, d, (d,), tau=0.01, prune=True),
        ProbConfig("stale", T, d, two, tau=0.01, commit_on_accept=False),
    ]


@pytest.mark.parametrize("T,d,deep", [(4, 1, 2), (8, 2, 4)])
@pytest.mark.parametrize("stop", [set(), {5}])
def test_calls_match_replay_and_output_is_greedy(T, d, deep, stop):
    prompt = [1, 2, 3, 4, 5, 6, 1, 2]
    max_new = 60
    out = greedy(prompt, max_new, stop)
    ids, probs = fake_reads(prompt, out, (d, deep))
    calib = calibration((d, deep))
    for cfg in configs(T, d, deep):
        ref = ProbReplay(0, prompt, out, ids, probs, cfg, ProbFakeSuffix(), calib).run()
        got, st = prob_tree_decode(
            ProbFakeLoop(T, d), prompt, cfg, calib, max_new, stop, ProbFakeSuffix(), 0,
            flush_every=3,
        )  # fmt: skip
        assert got == out, cfg.name
        assert st["tokens"] == ref["tokens"] == len(out) - 1, cfg.name
        assert st["calls"] == ref["calls"], (cfg.name, st["calls"], ref["calls"])
        for src in ("suffix", "fill", "sibling", "deep"):
            assert st[f"from_{src}"] == ref.get(f"from_{src}", 0), (cfg.name, src)


def test_node_cap_skips_but_stays_lossless():
    prompt, T, d = [1, 2, 3, 4, 5, 6, 1, 2], 6, 2
    out = greedy(prompt, 40, set())
    cfg = ProbConfig("A", T, d, (d, 4), tau=0.0)
    got, st = prob_tree_decode(
        ProbFakeLoop(T, d), prompt, cfg, calibration((d, 4)), 40, set(), ProbFakeSuffix(), 0,
        max_nodes=5,
    )  # fmt: skip
    assert got == out and st["cap_skips"] > 0


def check_against_hf(backbone_for, runner, T, d, deep):
    prompt = [5, 9, 13, 5, 9, 2, 40, 41, 5, 9, 13, 7]
    max_new = 40
    ids_t = torch.tensor([prompt])
    out = runner.generate(ids_t, max_new, set())
    seq = torch.cat([ids_t, torch.tensor([out])], dim=1)
    reads = runner.readouts(seq, len(prompt) - 1, [d, deep], 8)
    ids = {r: x[0] for r, x in zip((d, deep), reads, strict=True)}
    probs = {r: x[1] for r, x in zip((d, deep), reads, strict=True)}
    calib = calibration((d, deep))
    for cfg in configs(T, d, deep):
        ref = ProbReplay(0, prompt, out, ids, probs, cfg, ProbFakeSuffix(), calib).run()
        got, st = prob_tree_decode(
            backbone_for(), prompt, cfg, calib, max_new, set(), ProbFakeSuffix(), 0, flush_every=3
        )
        assert got == out, cfg.name
        assert st["cap_skips"] == 0, (cfg.name, st["cap_skips"])
        assert st["calls"] == ref["calls"], (cfg.name, st["calls"], ref["calls"])


@pytest.mark.skipif(not OURO.exists(), reason="Ouro snapshot not mounted")
def test_ouro_prob_decoder_matches_hf_greedy_and_replay():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from loop_suffix_spec_readouts import OuroRunner

    from llm_memory_editability.loop_tree_decode import OuroLoop

    model = tiny_ouro()
    with torch.no_grad():
        check_against_hf(lambda: OuroLoop(model, 4, 256, 4096), OuroRunner(model, 4), 4, 1, 2)


@pytest.mark.skipif(not HUGINN.exists(), reason="Huginn snapshot not mounted")
def test_huginn_prob_decoder_matches_hf_greedy_and_replay():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from loop_suffix_spec_readouts import HuginnRunner

    from llm_memory_editability.loop_tree_decode import HuginnLoop

    T, d, deep = 6, 2, 4
    model = tiny_huginn(T)
    init_len = 64
    runner = HuginnRunner(model, T, 0, init_len, "cpu")
    with torch.no_grad():
        check_against_hf(
            lambda: HuginnLoop(model, T, 256, 4096, 0, init_len, extra_depths=(deep,)),
            runner, T, d, deep,
        )  # fmt: skip
