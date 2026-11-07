"""Budget/control contracts, recycled KV correctness, and calibration-free CLI."""

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode
from llm_memory_editability.loop_budget_decode import (
    BudgetConfig,
    Candidate,
    SuffixController,
    budget_tree_decode,
    select_candidates,
)
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop, tree_decode
from llm_memory_editability.loop_suffix_spec import ArmConfig
from test_loop_batch_decode import FakeExecutor, TinyHuginn, TinyOuro
from test_loop_prob_decode import ProbFakeLoop, ProbFakeSuffix
from test_loop_tree_decode import greedy, greedy_next


def config(N=8, **kw):
    return BudgetConfig("test", 4, 1, (1, 2), draft_node_budget=N, **kw)


class BudgetFakeLoop(ProbFakeLoop):
    def context(self, tree, node):
        path = tree.path_tokens(node)
        while node.parent >= 0:
            node = tree.nodes[node.parent]
        return self.seq[: tree.base_pos] + [node.token] + path


class BoundedSuffix(ProbFakeSuffix):
    def speculate(self, *args, **kw):
        d = super().speculate(*args, **kw)
        limit = kw["max_spec_tokens"]
        d.token_ids, d.parents, d.probs = d.token_ids[:limit], d.parents[:limit], d.probs[:limit]
        return d


def test_feedback_snapshot_dedup_and_parent_mean():
    ctl = SuffixController(config(trace_controller=True))
    ctl.observe((3,), 1, 0.7)
    ctl.observe((3,), 2, 0.2)
    ctl.u = math.log(0.5)  # later feedback must use the original predictions
    ctl.observe((3,), 1, 0.1)
    ctl.mark_known((3,))
    ctl.observe((3,), 9, 0.9)  # full result already available: never backfill
    assert ctl.records[(3,)][1].prediction == 0.7
    result = ctl.verify((3,), 1)
    assert result["records"] == 2
    assert result["delta"] == pytest.approx(0.05)
    assert ctl.c == pytest.approx(0.5 * math.exp(0.005))
    assert ctl.verify((3,), 1) is None
    ctl.observe((3,), 4, 0.8)
    assert ctl.records == {}
    assert ctl.updates == 1
    assert ctl.verify((99,), 1) is None


def test_feedback_is_bounded_and_budget_eviction_does_not_drop_records():
    ctl = SuffixController(config(eta=100))
    ctl.observe((1, 2), 3, 0.9)
    ctl.observe((1, 8), 9, 0.8)
    ctl.mark_known((1, 2))
    assert ctl.updates == 0  # computed off path is not a verification event
    ctl.retain_prefix((1, 2))
    assert set(ctl.records) == {(1, 2)}
    ctl.verify((1, 2), 0)
    assert ctl.c == pytest.approx(0.25)
    # No hidden integral windup: a positive event immediately raises c to the upper bound.
    ctl.observe((1, 2, 0), 4, 0.1)
    ctl.verify((1, 2, 0), 4)
    assert ctl.c == pytest.approx(2)
    assert ctl.discarded == 1


def test_top_n_ancestor_closure_ties_zero_scores_and_current_coefficient():
    root = (7,)
    pool = {root: Candidate(root, 1, "model", "root")}
    for path, q, source in [
        ((7, 9), 0.8, "suffix"),
        ((7, 1), 0.6, "model"),
        ((7, 9, 0), 1, "suffix"),
        ((7, 1, 0), 0, "model"),
    ]:
        pool[path] = Candidate(path, q, source, source)
    assert list(select_candidates(pool, root, 2, 1)) == [root, (7, 9), (7, 9, 0)]
    chosen = select_candidates(pool, root, 2, 0.5)
    assert list(chosen) == [root, (7, 1), (7, 9)]
    assert pool[(7, 9, 0)].p == pytest.approx(0.2)  # c applies to each suffix edge
    assert pool[(7, 1)].e == 0.6  # never multiply model edges
    assert len(select_candidates(pool, root, 10, 0.5)) == 5  # retain zero if room
    pool[(7, 9)].raw_score = 0.6
    assert list(select_candidates(pool, root, 1, 1)) == [root, (7, 1)]


@pytest.mark.parametrize("N", [0, 1, 2, 8, 32])
@pytest.mark.parametrize("scoring", ["raw", "online_suffix_i", "online_suffix_length", "fixed_suffix_length"])
@pytest.mark.parametrize("stop", [set(), {5}])
def test_budget_decoder_greedy_recycling_and_request_reset(N, scoring, stop):
    prompt = [1, 2, 3, 4, 5, 6, 1, 2]
    cfg = config(N, scoring=scoring, trace_controller=True)
    cache = BoundedSuffix()
    events = []
    got, stats = budget_tree_decode(
        BudgetFakeLoop(4, 1),
        prompt,
        cfg,
        75,
        stop,
        cache,
        "a",
        max_nodes=N + 1,
        observer=lambda event, **data: events.append((event, data)),
    )
    assert got == greedy(prompt, 75, stop)
    assert stats["peak_draft_nodes"] <= N
    assert stats["max_rows"] <= N + 1
    assert stats["cap_skips"] == 0
    selections = [e for kind, e in events if kind == "selection"]
    for event in selections:
        paths = set(event["paths"])
        assert len(paths) - 1 <= N
        assert all(p == event["root"] or p[:-1] in paths for p in paths)
        assert len(set(event["slots"].values())) == len(paths)
    verified = [e for kind, e in events if kind == "verify"]
    assert len({e["parent"] for e in verified}) == len(verified)
    assert all(tuple(got[: e["output_position"]]) == e["parent"] for e in verified)
    if N == 0:
        assert stats["speculate_calls"] == stats["controller_updates"] == 0
        assert stats["readout_rows"] == len(got) - 1
        assert stats["calls"] == (len(got) - 1) * 4
    elif scoring in ("online_suffix_i", "online_suffix_length"):
        assert stats["controller_updates"] > 0
        assert 0.25 <= stats["c_final"] <= 2
    else:
        assert stats["c_final"] == 1 and stats["controller_updates"] == 0
    again, other = budget_tree_decode(BudgetFakeLoop(4, 1), prompt, cfg, 75, stop, cache, "a")
    assert again == got and other["controller_trace"] == stats["controller_trace"]
    assert cache.text == {}
    if N and not stop:
        assert stats["candidate_replacements"] > 0
        assert stats["nodes"] > N + 1


class CorrectChain:
    """All nodes finish together; N=1 retains only the first child of a long proposal."""

    def start_request(self, *args):
        pass

    def add_active_response(self, *args):
        pass

    def stop_request(self, *args):
        pass

    def speculate(self, rid, context, **kwargs):
        ctx = list(context)
        toks = []
        for _ in range(3):
            toks.append(greedy_next(ctx))
            ctx.append(toks[-1])
        return SimpleNamespace(token_ids=toks, parents=[-1, 0, 1], probs=[0.9, 0.81, 0.729])


def test_confirmed_root_is_queried_once_and_eos_does_not_expand():
    prompt = [1, 2, 3]
    expected = greedy(prompt, 2, set())
    got, stats = budget_tree_decode(
        BudgetFakeLoop(4, 1),
        prompt,
        config(3, use_model=False),
        20,
        {expected[-1]},
        CorrectChain(),
    )
    assert got == expected
    assert stats["speculate_calls"] == 1
    assert stats["peak_draft_nodes"] == 3


def test_evicted_parent_records_survive_and_ready_chain_accepts_before_reweighting():
    prompt = [1, 2, 3]
    for N in [1, 3]:
        events = []
        got, stats = budget_tree_decode(
            BudgetFakeLoop(4, 1),
            prompt,
            config(N, use_model=False, eta=20),
            12,
            set(),
            CorrectChain(),
            observer=lambda e, **d: events.append((e, d)),
        )
        assert got == greedy(prompt, 12, set())
        assert max(stats["accepted_lengths"]) >= N + 1
        assert 0 < stats["controller_updates"] <= stats["verified_parents"]
        if N == 1:
            assert stats["controller_updates"] == stats["verified_parents"]
            assert stats["budget_rejections"] > 0
            # Parent at output position 3 was originally metadata outside the budget.
            event = next(d for e, d in events if e == "verify" and d["output_position"] == 3)
            assert event["feedback"] is not None


def test_low_suffix_weight_cannot_evict_ready_correct_chain():
    class BroadChain(CorrectChain):
        def speculate(self, rid, context, **kwargs):
            d = super().speculate(rid, context, **kwargs)
            # Each parent gets eight high-confidence wrong siblings, driving c down.
            # Correct tokens sort first only by lexical tie-breaking below, so give
            # their edges score 1 and competitors .99 for an unambiguous first tree.
            d.probs = [1.0, 1.0, 1.0]
            for parent in [-1, 0, 1]:
                gold = d.token_ids[parent + 1]
                for tok in range(100, 108):
                    assert tok != gold
                    d.token_ids.append(tok)
                    d.parents.append(parent)
                    d.probs.append(0.99)
            return d

    events = []
    out, stats = budget_tree_decode(
        BudgetFakeLoop(4, 1),
        [1, 2, 3],
        config(3, use_model=False, eta=10),
        5,
        set(),
        BroadChain(),
        observer=lambda event, **data: events.append((event, data)),
    )
    assert out == greedy([1, 2, 3], 5, set())
    assert stats["accepted_lengths"] == [4]
    assert stats["c_final"] == pytest.approx(0.25)
    assert stats["controller_updates"] == 3


def test_model_overwrite_preserves_first_suffix_snapshot():
    class SureLoop(BudgetFakeLoop):
        def readout_top(self, tree, groups, full, m):
            reads = [
                [
                    (
                        [greedy_next(self.context(tree, n))] + list(range(20, 27)),
                        [0.55] + [0.01] * 7,
                    )
                    for n in nodes
                ]
                for _, nodes in groups
            ]
            return reads, [greedy_next(self.context(tree, n)) for n in full]

    events = []
    _, stats = budget_tree_decode(
        SureLoop(4, 1),
        [1, 2, 3],
        config(1, trace_controller=True),
        9,
        set(),
        CorrectChain(),
        observer=lambda e, **d: events.append((e, d)),
    )
    selections = [d for e, d in events if e == "selection"]
    root = selections[0]["root"]
    child = next(p for p in selections[0]["paths"] if p != root)
    assert selections[0]["scores"][child][:2] == ("suffix", 0.9)
    overwritten = [d for d in selections if d["root"] == root and d["scores"][child][0] == "model"]
    assert overwritten and all(d["scores"][child][1:3] == (0.55, 0.55) for d in overwritten)
    assert stats["controller_trace"][0]["delta"] == pytest.approx(0.1)


@pytest.fixture
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("family", ["ouro", "huginn"])
@pytest.mark.parametrize("scoring", ["raw", "online_suffix_i", "online_suffix_length", "fixed_suffix_length"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_real_backbones_recycled_states_match_ar_and_batch(family, scoring, device, one_thread):
    torch.manual_seed(71)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (TinyOuro() if family == "ouro" else TinyHuginn()).to(device, dtype).eval()
    N = 4

    def backbone():
        return (
            OuroLoop(model, 4, 128, N + 1)
            if family == "ouro"
            else HuginnLoop(model, 4, 128, N + 1, 0, 128, extra_depths=(2,))
        )

    requests = [
        BatchRequest(i, p, n)
        for i, (p, n) in enumerate(
            [
                ([1, 2, 3, 1, 2], 20),
                ([6, 7, 8], 13),
                ([4, 2, 1, 4], 17),
            ]
        )
    ]
    cfg = config(N, scoring=scoring, trace_controller=True)
    references, serial = [], []
    for r in requests:
        refs, _ = tree_decode(
            backbone(), r.prompt, ArmConfig("ar", 4, 1, 0, False, False), r.max_new_tokens, r.stop
        )
        references.append(refs)
        serial.append(
            budget_tree_decode(
                backbone(), r.prompt, cfg, r.max_new_tokens, r.stop, BoundedSuffix(), r.request_id
            )
        )
    records, summary = batch_decode(requests, [backbone(), backbone()], cfg, cache=BoundedSuffix())
    for row, ref, (seq, stats) in zip(records, references, serial, strict=True):
        assert row["tokens"] == seq == ref
        assert row["stats"]["controller_trace"] == stats["controller_trace"]
        assert row["stats"]["rows"] == stats["rows"]
        assert stats["nodes"] > N + 1  # slots actually reused, not a large append-only buffer
    assert summary["core_calls"] < sum(r["stats"]["calls"] for r in records)


def test_invalid_budget_fails_before_prefill():
    for kw in (
        {"draft_node_budget": -1},
        {"eta": 0},
        {"eta": float("nan")},
        {"c_min": 1},
        {"c_max": 1},
        {"readout_depths": (4,)},
    ):
        with pytest.raises(ValueError):
            replace(config(), **kw)
    with pytest.raises(ValueError, match="N\\+1"):
        budget_tree_decode(
            BudgetFakeLoop(4, 1), [1, 2], config(), 10, set(), BoundedSuffix(), max_nodes=8
        )
    req = BatchRequest(0, [1, 2], 5)
    records, _ = batch_decode([req], [BudgetFakeLoop(4, 1)], config(0), executor=FakeExecutor())
    assert records[0]["stats"]["speculate_calls"] == 0


def test_budget_cli_runs_without_calibration(tmp_path, monkeypatch, one_thread):
    import json
    import sys
    import transformers
    from scripts import loop_batch_bench as bench

    requests = tmp_path / "input.jsonl"
    requests.write_text(
        "\n".join(
            json.dumps(dict(question_id=i, prompt_token_ids=p, max_new_tokens=6))
            for i, p in enumerate([[1, 2, 1, 2], [3, 7, 8]])
        )
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: TinyOuro().eval()
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(eos_token_id=None, get_vocab=lambda: {}),
    )

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **kw: TinyOuro().config)

    def cache():
        c = BoundedSuffix()
        c.publications = []
        return c

    monkeypatch.setattr(bench.CompletionSuffixCache, "arctic", cache)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--model",
            "ouro-2.6b",
            "--model-path",
            str(tmp_path),
            "--questions",
            str(requests),
            "--out",
            str(out),
            "--arms",
            "ar,budget-suffix,budget-raw,budget-online,budget-online-length,budget-fixed-length",
            "--scoring",
            "raw",
            "--draft-node-budget",
            "4",
            "--max-nodes",
            "5",
            "--batch-sizes",
            "1,2",
            "--order-seeds",
            "0",
            "--repeat-id",
            "1",
            "--max-seq",
            "128",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--warmup-tokens",
            "2",
            "--check",
        ],
    )
    bench.main()
    summaries = [json.loads(p.read_text()) for p in out.glob("*-summary.json")]
    assert len(summaries) == 12
    for s in summaries:
        assert s["same_as_serial_ar"] == s["requests"] == 2
        assert s["repeat_id"] == 1
        if s["arm"].startswith("budget-"):
            assert s["tree_policy"] == "node_budget"
            assert s["config"]["draft_node_budget"] == 4
    assert json.loads((out / "metadata.json").read_text())["calibration_sha256"] is None
