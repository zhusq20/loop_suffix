"""Oracle reranking, handover-wait decomposition, subset/job planning and analysis."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import llm_memory_editability.loop_budget_decode as budget_module  # noqa: E402
from llm_memory_editability.loop_budget_decode import (  # noqa: E402
    Candidate, budget_tree_decode, select_candidates,
)
from loop_batch_bench import arm_config  # noqa: E402
from loop_score_headroom import (  # noqa: E402
    OracleRanking, budget_config, parse_variant, ranking, traced_decode, variant_names,
    wait_decomposition,
)
import loop_score_headroom_suite as suite  # noqa: E402
from test_loop_budget_decode import BoundedSuffix, BudgetFakeLoop, config  # noqa: E402

SPEC = SimpleNamespace(depth_total=4, loopfill_readout_depths=(1, 2), loopspec_depths=(1, 2),
                       wavefront_depth=1)
PROTOCOL = dict(suffix_max_draft=4, model_top_k=2, child_suffix_caps=[4, 4], suffix_eta=0.1,
                online_c_min=0.25, online_c_max=2.0)


def pool_of(items):
    root = (7,)
    pool = {root: Candidate(root, 1, "model", "root")}
    for path, q, source in items:
        pool[path] = Candidate(path, q, source, source)
    return root, pool


ITEMS = [((7, 9), 0.8, "suffix"), ((7, 1), 0.6, "model"), ((7, 9, 0), 1, "suffix"),
         ((7, 1, 3), 0.1, "model")]


def test_oracle_puts_reference_chain_first_and_fills_by_score():
    root, pool = pool_of(ITEMS)
    oracle = OracleRanking([7, 1, 3, 5])
    assert list(oracle(pool, root, 2, 1.0)) == [root, (7, 1), (7, 1, 3)]
    assert list(oracle(pool, root, 3, 1.0)) == [root, (7, 1), (7, 1, 3), (7, 9)]
    assert pool[(7, 9, 0)].p == pytest.approx(0.8)  # decoder scoring still applied
    assert list(oracle(pool, root, 1, 1.0)) == [root, (7, 1)]
    assert oracle.counts["on_path_truncated"] == 1
    only = OracleRanking([7, 1, 3], only=True)
    assert list(only(pool, root, 4, 1.0)) == [root, (7, 1), (7, 1, 3)]


def test_oracle_off_reference_is_the_score_order():
    root, pool = pool_of(ITEMS)
    for budget in range(5):
        for c in (0.25, 1.0, 2.0):
            want = list(select_candidates(pool, root, budget, c))
            assert list(OracleRanking([7, 4])(pool, root, budget, c)) == want
    off = OracleRanking([8])
    off(pool, root, 2, 1.0)
    assert off.counts["root_on_path"] == 0 and off.counts["on_path_candidates"] == 0


def test_ranking_context_restores_decoder_rule():
    original = budget_module.select_candidates
    with pytest.raises(RuntimeError):
        with ranking(OracleRanking([1])):
            assert budget_module.select_candidates is not original
            raise RuntimeError
    assert budget_module.select_candidates is original
    with ranking(None):
        assert budget_module.select_candidates is original


def test_wait_decomposition_by_hand():
    entries = [dict(tick=5, step=4, position=1, terminal=False),
               dict(tick=6, step=1, position=2, terminal=False),
               dict(tick=7, step=0, position=3, terminal=False),
               dict(tick=8, step=0, position=4, terminal=True)]
    gold = [dict(position=1, first_proposal=0, allocations=[]),
            dict(position=2, first_proposal=2, allocations=[
                dict(tick=2, reason="budget", release_tick=3, spent=1),
                dict(tick=5, reason=None, release_tick=None, spent=1)])]
    totals, per_entry = wait_decomposition(4, entries, gold)
    assert totals["entries"] == 3 and totals["wait"] == 7
    assert totals["proposal_wait"] == 4 and totals["ranking_wait"] == 3
    assert totals["evicted_entries"] == 1 and totals["ranking_entries"] == 1
    assert per_entry == [[1, 4, 0, 0], [2, 1, 0, 3], [3, 0, 4, 0]]
    assert totals.get("reach_violations", 0) == 0


def runner(prompt, cfg, max_new=40):
    return lambda observer: budget_tree_decode(BudgetFakeLoop(4, 1), prompt, cfg, max_new, set(),
                                               BoundedSuffix(), observer=observer)


@pytest.mark.parametrize("budget", [1, 2, 4, 16])
@pytest.mark.parametrize("prompt", [[1, 2, 3, 1, 2], [1, 2, 3, 4, 5, 6, 1, 2]])
def test_oracle_decode_is_exact_and_removes_ranking_wait(budget, prompt):
    cfg = config(budget, scoring="raw", top_m=2, suffix_room=4)
    ref, ref_stats, ref_payload = traced_decode(runner(prompt, cfg), 4)
    first = OracleRanking(ref)
    got, stats, payload = traced_decode(runner(prompt, cfg), 4, first)
    only = OracleRanking(ref, only=True)
    got_only, stats_only, payload_only = traced_decode(runner(prompt, cfg), 4, only)
    assert got == got_only == ref
    for s, p in [(ref_stats, ref_payload), (stats, payload), (stats_only, payload_only)]:
        w = p["waits"]
        assert w["proposal_wait"] + w["ranking_wait"] + 4 == s["calls"]
        assert w.get("reach_violations", 0) == 0
        assert p["trace"]["positive_contested_selections"] <= p["trace"]["contested_selections"]
    for s, p in [(stats, payload), (stats_only, payload_only)]:
        assert p["oracle"]["root_on_path"] == p["oracle"]["selections"]
        if p["oracle"]["on_path_truncated"] == 0:
            assert p["waits"]["ranking_wait"] == 0
    # Only reference-path nodes are computed, so oracle-only spends no wasted rows.
    assert stats_only["calls"] == stats["calls"]
    assert stats_only["rows"] <= stats["rows"] and stats_only["wasted_rows"] <= stats["wasted_rows"]


def test_variants_jobs_and_configs():
    names = variant_names([0.5, 2], 0.7)
    assert names == ["fixed-c0.5", "fixed-c1", "fixed-c2", "online-c0.7", "oracle-first", "oracle-only"]
    assert parse_variant("fixed-c0.001") == dict(kind="fixed", c=0.001)
    with pytest.raises(ValueError):
        parse_variant("oracle")
    jobs = suite.make_jobs(["m"], [8, 16], [0.5], 0.7)
    oracle = [j for j in jobs if j["variant"].startswith("oracle")]
    assert len(jobs) == 2 * 5 and len(oracle) == 4
    assert {j["reference"] for j in oracle} == {"m-n8-fixed-c1", "m-n16-fixed-c1"}
    for variant, scoring, c in [("fixed-c0.001", "fixed_suffix", 0.001), ("fixed-c1000", "fixed_suffix", 1000),
                                ("online-c0.7", "online_suffix_i", 0.7), ("oracle-first", "fixed_suffix", 1)]:
        cfg = budget_config(SPEC, variant, 16, PROTOCOL, arm_config)
        assert cfg.scoring == scoring and cfg.c_initial == c and cfg.draft_node_budget == 16
        assert cfg.top_m == 2 and cfg.child_suffix_caps == (4, 4) and cfg.suffix_room == 4
    online = budget_config(SPEC, "online-c0.7", 16, PROTOCOL, arm_config)
    assert (online.c_min, online.c_max, online.eta) == (0.25, 2.0, 0.1)


def test_stratified_subset_is_deterministic_and_balanced():
    rows = [dict(source_id=f"a/{i}", category="a") for i in range(5)]
    rows += [dict(source_id=f"b/{i}", category="b") for i in range(2)]
    rows = rows[::2] + rows[1::2]
    chosen = suite.stratified_subset(rows, 4)
    assert chosen == suite.stratified_subset(rows, 4)
    assert sorted(r["category"] for r in chosen) == ["a", "a", "b", "b"]
    assert [rows.index(r) for r in chosen] == sorted(rows.index(r) for r in chosen)
    assert suite.stratified_subset(rows, 99) == rows


def test_prepare_and_analyze_end_to_end(tmp_path):
    inputs = tmp_path / "in"
    inputs.mkdir()
    prompts = {"d1": [[1, 2, 3, 1, 2], [3, 1, 2, 3]], "d2": [[1, 2, 3, 4, 5, 6, 1, 2]]}
    for dataset, items in prompts.items():
        (inputs / f"{dataset}.jsonl").write_text("".join(json.dumps(dict(
            source_id=f"{dataset}/{i}", category="x", prompt="unused", ids=ids)) + "\n"
            for i, ids in enumerate(items)))
    out = tmp_path / "suite"
    protocol = suite.prepare(out, tmp_path / "assets", sys.executable, ["fake"], inputs, ["d1", "d2"],
                             5, [4], [0.05, 4], 0.7, 0.0, 0, 40, model_specs={"fake": SPEC})
    assert (out / "source/scripts/loop_score_headroom.py").exists()
    assert set(protocol["source_sha256"]) >= {"scripts/loop_score_headroom.py",
                                             "src/llm_memory_editability/loop_budget_decode.py"}
    rows = [json.loads(line) for d in protocol["datasets"]
            for line in (out / "inputs" / f"{d}.jsonl").read_text().splitlines()]
    jobs = sorted((json.loads(p.read_text()) for p in (out / "jobs").glob("*.json")),
                  key=lambda j: j["variant"] in ("oracle-first", "oracle-only"))
    reference = {}
    for job in jobs:
        cfg = budget_config(SPEC, job["variant"], job["budget"], protocol, arm_config)
        target = out / "runs" / job["key"]
        target.mkdir(parents=True)
        records = []
        for row in rows:
            oracle = None
            if job["reference"]:
                oracle = OracleRanking(reference[row["source_id"]], only=job["variant"] == "oracle-only")
            tokens, stats, payload = traced_decode(runner(row["ids"], cfg), 4, oracle)
            records.append(dict(source_id=row["source_id"], dataset=row["source_id"].split("/")[0],
                                tokens=tokens, stats=stats, **payload))
        if job["variant"] == "fixed-c1":
            reference = {r["source_id"]: r["tokens"] for r in records}
        (target / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (target / "complete.json").write_text("{}")
    report = suite.analyze(out)
    (head,) = report["headlines"]
    assert head["requests"] == head["common_requests"] == 3
    assert head["oracle_first_same_output"] == 3
    assert head["best_fixed_c"] in protocol["fixed_cs"]
    assert not report["missing_jobs"]
    by_variant = {r["variant"]: r for r in report["table"]}
    assert set(by_variant) == set(protocol["variants"])
    base = by_variant["fixed-c1"]
    assert base["gamma"] == pytest.approx(4 * base["decode_tokens"] / base["loop_steps"])
    assert by_variant["online-c0.7"].get("c_final_median") is not None
    text = (out / "report.md").read_text()
    assert "oracle-first" in text and "## Headline" in text
    assert (out / "table.csv").read_text().startswith("model,budget,variant")
