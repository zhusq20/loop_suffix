"""Tables for the probability-threshold draft tree replay.

Reads sweep JSONs from loop_prob_tree_replay.py and prints, per model and split:
block calls per token against the current decoder (legacy K16), rows and readouts per
call, predicted decode time under a per-row cost relative to a one-row call, and where
the greedy token sat in the shallow top-8 at the steps that ended because it was
missing from the tree.

Protocol: results/loop-spec-prob-tree-20261031/protocol.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROW_COSTS = (0.0005, 0.002, 0.01)  # c_row / t_call; the wall-clock arms bracket 0.0003-0.002


def predicted(hist: list[list[int]], c: float) -> float:
    return sum(n * (1 + c * (rows - 1)) for rows, _, n in hist)


def coverage(miss: dict, depth: int) -> dict:
    total = sum(miss.get("n", {}).values())
    ranks = {int(k): v for k, v in miss.get(str(depth), {}).items()}
    cum, out = 0, {}
    for k in range(9):
        cum += ranks.get(k, 0)
        if k in (0, 1, 3, 7):
            out[f"top{k + 1}"] = cum / max(1, total)
    return out | {"misses": total}


def table(sweep: dict, split: str) -> str:
    res = sweep["results"]
    base = next(r for r in res if r["config"]["name"] == "legacy K16")
    b = base["splits"][split]
    bt = {c: predicted(base["hist"][split], c) for c in ROW_COSTS}
    lines = [
        "| setting | calls/token | speedup | vs legacy calls | rows/call | reads/call"
        + " | tokens/step "
        + " ".join(f"| time c={c:g}" for c in ROW_COSTS)
        + " |",
        "| --- " * (7 + len(ROW_COSTS)) + "|",
    ]
    for r in res:
        o = r["splits"][split]
        times = " ".join(f"| {predicted(r['hist'][split], c) / bt[c]:.3f}" for c in ROW_COSTS)
        lines.append(
            f"| {r['config']['name']} | {o['calls_per_token']:.3f} | {o['speedup_vs_ar']:.2f}"
            f" | {o['calls'] / b['calls'] * b['tokens'] / o['tokens']:.3f}"
            f" | {o['rows_per_call']:.1f} | {o['reads_per_call']:.2f}"
            f" | {o['tokens_per_step']:.2f} {times} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweeps", nargs="+")
    p.add_argument("--split", default="dev")
    args = p.parse_args()
    for path in args.sweeps:
        sweep = json.loads(Path(path).read_text())
        res = sweep["results"]
        print(f"\n## {sweep['model']} — {Path(path).name} — split {args.split}\n")
        for r in res:
            chk = r.get("legacy_check")
            if chk is not None:
                n = len(chk["mismatched_questions"])
                print(f"legacy check: {n} questions differ from replay_request")
        print(table(sweep, args.split))
        print("\ntime = predicted decode time relative to legacy K16, rows-only cost model")
        for r in res:
            if r["config"]["name"] in ("legacy K16", "rule top8 noK deep t0.01"):
                depths = sorted(int(k) for k in r["miss_rank"][args.split] if k != "n")
                cov = {dd: coverage(r["miss_rank"][args.split], dd) for dd in depths}
                print(
                    f"\nmisses of {r['config']['name']}: "
                    + "; ".join(
                        f"depth {dd}: "
                        + ", ".join(
                            f"{k} {v:.3f}" if k != "misses" else f"n={v}" for k, v in c.items()
                        )
                        for dd, c in cov.items()
                    )
                )
                print(
                    "accepted share: "
                    + ", ".join(
                        f"{k} {v:.3f}" for k, v in r["splits"][args.split]["accepted_share"].items()
                    )
                )


if __name__ == "__main__":
    main()
