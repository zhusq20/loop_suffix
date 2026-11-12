"""Paired wall-clock comparison of the probability-threshold arms (S4).

For each model: decode time per decoded token of every arm relative to a base arm
(both-path by default), as the ratio of total decode time over the same questions,
with a 95% interval from a bootstrap over questions stratified by category; the same
on the questions where all compared arms produced the same text; the per-question
geometric mean; and engine block calls against the replay, rows per token, ms per call.

Protocol: results/loop-spec-prob-tree-20261031/protocol.md (S4)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def per_question(rec: dict, arm: str) -> tuple[float, int]:
    """Mean decode seconds and decoded tokens over the repeats of one arm."""
    runs = rec["runs"][arm]
    tok = runs[0]["new_tokens"] if "new_tokens" in runs[0] else runs[0]["n_tokens"] - 1
    sec = sum(r["decode_s"] for r in runs) / len(runs)
    return sec, max(1, tok)


def ratio(recs: list[dict], arm: str, base: str) -> float:
    """Time per token of `base` over time per token of `arm`: > 1 means `arm` is faster."""
    a = sum(per_question(r, arm)[0] for r in recs) / sum(per_question(r, arm)[1] for r in recs)
    b = sum(per_question(r, base)[0] for r in recs) / sum(per_question(r, base)[1] for r in recs)
    return b / a


def bootstrap(recs: list[dict], arm: str, base: str, n: int = 4000, seed: int = 0):
    by = defaultdict(list)
    for r in recs:
        by[r["category"]].append(r)
    rng = random.Random(seed)
    vals = sorted(
        ratio([rng.choice(v) for v in by.values() for _ in v], arm, base) for _ in range(n)
    )
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bench", nargs="+", help="bench-bfloat16.jsonl per model")
    p.add_argument("--base", default="both-path")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = {}
    for path in args.bench:
        recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x]
        arms = list(recs[0]["runs"])
        model = Path(path).parent.name
        same = [r for r in recs if all(r["runs"][a][0]["same_as_ar"] for a in arms)]
        rows = {}
        for arm in arms:
            out = {"questions": len(recs)}
            if arm != args.base:
                out["speedup_vs_base"] = ratio(recs, arm, args.base)
                out["ci95"] = bootstrap(recs, arm, args.base)
                if same:
                    out["same_text_questions"] = len(same)
                    out["speedup_vs_base_same_text"] = ratio(same, arm, args.base)
                    out["ci95_same_text"] = bootstrap(same, arm, args.base)
                logs = []
                for r in recs:
                    sa, ta = per_question(r, arm)
                    sb, tb = per_question(r, args.base)
                    logs.append(math.log((sb / tb) / (sa / ta)))
                out["per_question_geomean"] = math.exp(sum(logs) / len(logs))
            first = [r["runs"][arm][0] for r in recs]
            calls = sum(f.get("calls", 0) for f in first)
            toks = sum(f.get("new_tokens", f["n_tokens"] - 1) for f in first)
            if calls:
                out["calls_per_token"] = calls / toks
                out["rows_per_token"] = sum(f.get("rows", 0) for f in first) / toks
                out["rows_per_call"] = sum(f.get("rows", 0) for f in first) / calls
                dec = sum(sum(x["decode_s"] for x in r["runs"][arm]) for r in recs)
                n_calls = sum(sum(x.get("calls", 0) for x in r["runs"][arm]) for r in recs)
                out["ms_per_call"] = 1000 * dec / n_calls
                rep = [f for f in first if f.get("replay_calls") is not None]
                if rep:
                    out["calls_vs_replay"] = sum(f["calls"] for f in rep) / sum(
                        f["replay_calls"] for f in rep
                    )
                    out["calls_eq_replay"] = sum(
                        f["calls"] == f["replay_calls"] for f in rep
                    ) / len(rep)
            out["same_as_ar"] = sum(f["same_as_ar"] for f in first) / len(first)
            rows[arm] = out
        report[model] = rows
        print(f"== {model}: {len(recs)} questions, {len(same)} with identical text in all arms")
        for arm, o in rows.items():
            sp = (
                f"x{o['speedup_vs_base']:.3f} [{o['ci95'][0]:.3f}, {o['ci95'][1]:.3f}]"
                f" same-text x{o.get('speedup_vs_base_same_text', float('nan')):.3f}"
                f" geo x{o['per_question_geomean']:.3f}"
                if "speedup_vs_base" in o
                else "(base)"
            )
            cpt = o.get("calls_per_token")
            extra = (
                f" calls/tok {cpt:.3f} rows/tok {o['rows_per_token']:.1f}"
                f" ms/call {o['ms_per_call']:.2f} calls/replay {o.get('calls_vs_replay', 0):.3f}"
                if cpt
                else ""
            )
            print(f"   {arm:10s} vs {args.base}: {sp}{extra} same-as-ar {o['same_as_ar']:.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
