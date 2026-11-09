"""Validate completed headroom measurements and report output-matched comparisons.

The oracle is an optimistic diagnostic, not a proof of a global optimum: changing
admission also changes subsequent proposal timing and the candidate pools.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_latency_worker import digest, read, save  # noqa: E402
from loop_score_headroom_suite import analyze, job_key, metrics  # noqa: E402


def audit(out):
    out = Path(out).resolve()
    protocol = read(out / "protocol.json")
    inputs = []
    for dataset, info in protocol["inputs"].items():
        path = out / "inputs" / f"{dataset}.jsonl"
        assert digest(path) == info["sha256"], path
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == info["count"], dataset
        inputs.extend(rows)
    ids = [r["source_id"] for r in inputs]
    assert len(ids) == len(set(ids)), "request IDs must be globally unique"
    for rel, sha in protocol["source_sha256"].items():
        assert digest(Path(protocol["source"]) / rel) == sha, rel
    checked, pairs = [], []
    for model, spec in protocol["models"].items():
        depth = spec["depth_total"]
        for budget in protocol["budgets"]:
            reference = None
            variants = ["fixed-c1"] + [v for v in protocol["variants"] if v != "fixed-c1"]
            for variant in variants:
                key = job_key(model, budget, variant)
                target = out / "runs" / key
                job = read(out / "jobs" / f"{key}.json")
                assert job["outcome"] == "complete", key
                done = read(target / "complete.json")
                journal = target / "requests.jsonl"
                assert digest(journal) == done["requests_sha256"], key
                rows = [json.loads(line) for line in journal.read_text().splitlines()]
                assert [r["source_id"] for r in rows] == ids, key
                assert len(rows) == done["requests"] == len(ids), key
                if variant == "fixed-c1":
                    reference = rows
                for r, ref in zip(rows, reference, strict=True):
                    s, t, w = r["stats"], r["trace"], r["waits"]
                    label = (key, r["source_id"])
                    assert s["tokens"] == len(r["tokens"]) - 1, label
                    assert s["calls"] == t.get("calls", 0), label
                    assert s["rows"] == t.get("rows", 0), label
                    assert s["peak_draft_nodes"] <= budget, label
                    assert w.get("reach_violations", 0) == 0, label
                    assert s["calls"] == (depth if s["tokens"] else 0) + w.get("wait", 0), label
                    assert w.get("wait", 0) == w.get("proposal_wait", 0) + w.get("ranking_wait", 0), label
                    assert sum(e[2] for e in r["wait_entries"]) == w.get("proposal_wait", 0), label
                    assert sum(e[3] for e in r["wait_entries"]) == w.get("ranking_wait", 0), label
                    assert all(min(e[1:]) >= 0 and sum(e[1:]) == depth
                               for e in r["wait_entries"]), label
                    if r["oracle"] is not None:
                        assert r["same_as_reference"] == (r["tokens"] == ref["tokens"]), label
                matched = [(r, ref) for r, ref in zip(rows, reference, strict=True)
                           if r["tokens"] == ref["tokens"]]
                a = metrics([r for r, _ in matched], depth)
                b = metrics([r for _, r in matched], depth)
                pairs.append(dict(
                    model=model, budget=budget, variant=variant, requests=len(rows),
                    matched_requests=len(matched), gamma_variant=a["gamma"], gamma_c1=b["gamma"],
                    gamma_gain_pct=100 * (a["gamma"] / b["gamma"] - 1) if b["gamma"] else None,
                    rows_variant=a["rows_per_token"], rows_c1=b["rows_per_token"],
                    rows_change_pct=100 * (a["rows_per_token"] / b["rows_per_token"] - 1)
                    if b["rows_per_token"] else None))
                checked.append(dict(job=key, requests=len(rows), sha256=digest(journal),
                                    failures=job["failures"]))
    result = dict(passed=True, jobs=len(checked), requests=sum(c["requests"] for c in checked),
                  checks=checked, paired=pairs, audit_sha256=digest(Path(__file__)),
                  caveats=["Matched-output subsets are selected subsets, not unbiased full-population effects.",
                           "Each variant builds its own suffix history; equal current outputs need not imply equal history.",
                           "Oracle admission changes later candidate pools; this is not a strict global bound.",
                           "No timing or task-quality conclusion follows from loop-step measurements."])
    save(out / "audit.json", result)
    with (out / "paired.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    return result


def plot(out, analysis):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    models = list(read(out / "protocol.json")["models"])
    budgets = read(out / "protocol.json")["budgets"]
    fig, axes = plt.subplots(2 * len(budgets), len(models),
                             figsize=(5 * len(models), 6.6 * len(budgets)),
                             squeeze=False, constrained_layout=True)
    for row, budget in enumerate(budgets):
        for col, model in enumerate(models):
            records = [r for r in analysis["table"] if r["model"] == model and r["budget"] == budget]
            fixed = sorted((float(r["variant"][7:]), r) for r in records
                           if r["variant"].startswith("fixed-c"))
            for offset, (metric, population, count_key) in enumerate([
                ("gamma", "All requests", "requests"),
                ("gamma_common", "Common outputs", "common_requests"),
            ]):
                ax = axes[2 * row + offset, col]
                ax.plot([c for c, _ in fixed], [r[metric] for _, r in fixed], "o-", label="Fixed c")
                for variant, color, label in [("oracle-first", "#c44e52", "Oracle-first"),
                                               ("online-c0.7", "#55a868", "Online")]:
                    value = next((r[metric] for r in records if r["variant"] == variant), None)
                    if value is not None:
                        ax.axhline(value, color=color, linestyle=":", label=label)
                count = records[0][count_key]
                ax.set(xscale="log", xlabel="Suffix coefficient c", ylabel="Gamma (T × tokens / loop steps)",
                       title=f"{model}, N={budget}\n{population} (n={count})")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"score-headroom.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    result = audit(out)
    analysis = analyze(out)
    plot(out, analysis)
    print(json.dumps({k: result[k] for k in ("passed", "jobs", "requests")}, indent=2))


if __name__ == "__main__":
    main()
