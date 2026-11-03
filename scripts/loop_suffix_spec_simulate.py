"""Replay suffix-tree and shallow-readout speculation over a looped LM's greedy outputs.

Reads the per-question files written by loop_suffix_spec_readouts.py and replays
every arm over all questions in question_id order. Each arm gets a fresh
ArcticInference SuffixDecodingCache (commit aca5d9a): a per-request tree over the
prompt and the output so far, and a global tree over earlier outputs that fills
online, as in the official simulator. Block-call accounting is in
llm_memory_editability.loop_suffix_spec.

Protocol: results/loop-suffix-spec-20261030/protocol.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request, summarize

MAIN = {
    "ouro-2.6b": {"T": 4, "d": [1, 2], "d_main": 1},
    "huginn-3.5b": {"T": 32, "d": [2, 4, 8], "d_main": 4},
}
K_GRID = [4, 8, 16, 32]
K_MAIN = 16
ALPHA_GRID = [1.0, 2.0]
PRUNE_GRID = [None, 4]


def arms_for(model: str) -> list[ArmConfig]:
    spec = MAIN[model]
    T, d0 = spec["T"], spec["d_main"]
    arms = [ArmConfig("ar", T, d0, 0, use_suffix=False, use_fill=False)]
    for d, k in itertools.product(spec["d"], K_GRID):
        arms.append(ArmConfig(f"shallow d{d} K{k}", T, d, k, use_suffix=False, use_fill=True))
    for k, a in itertools.product(K_GRID, ALPHA_GRID):
        arms.append(ArmConfig(f"suffix K{k} a{a:g}", T, d0, k, True, False, spec_factor=a))
    for d, k, a, m in itertools.product(spec["d"], K_GRID, ALPHA_GRID, PRUNE_GRID):
        name = f"suffix+shallow d{d} K{k} a{a:g}" + (f" prune{m}" if m else "")
        arms.append(ArmConfig(name, T, d, k, True, True, prune_top=m, spec_factor=a))
    return arms


REPEAT_BINS = [0.0, 0.25, 0.5, 0.75, 1.01]
SELF_BINS = [0.0, 0.1, 0.25, 0.5, 1.01]
NO_LOOP_MAX = 0.25  # self-repeat below this counts as output without degenerate loops


def repeat_rates(prompt: np.ndarray, output: np.ndarray, n: int = 4) -> dict[str, float]:
    """Shares of output tokens whose n-gram (ending at that token) occurred earlier.

    any: earlier in prompt or output; self: earlier in the output itself (loops);
    prompt_copy: in the prompt but not earlier in the output.
    """
    seq = [int(t) for t in prompt] + [int(t) for t in output]
    start = len(prompt)
    prompt_grams = {tuple(seq[i - n + 1 : i + 1]) for i in range(n - 1, start)}
    out_grams: set[tuple[int, ...]] = set()
    self_hits = copy_hits = 0
    for i in range(max(start, n - 1), len(seq)):
        gram = tuple(seq[i - n + 1 : i + 1])
        if gram in out_grams:
            self_hits += 1
        elif gram in prompt_grams:
            copy_hits += 1
        out_grams.add(gram)
    L = max(1, len(output))
    return {
        "any": (self_hits + copy_hits) / L,
        "self": self_hits / L,
        "prompt_copy": copy_hits / L,
    }


def to_bin(rate: float, edges: list[float]) -> str:
    for lo, hi in zip(edges, edges[1:], strict=False):
        if lo <= rate < hi:
            return f"{lo:.2f}-{min(hi, 1.0):.2f}"
    raise ValueError(rate)


def load_requests(runs_dir: Path) -> list[dict]:
    meta = {}
    for path in sorted(runs_dir.glob("index-shard*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                rec = json.loads(line)
                meta[rec["question_id"]] = rec
    reqs = []
    for qid in sorted(meta):
        z = np.load(runs_dir / "readouts" / f"{qid}.npz")
        depths = [int(t) for t in z["depths"]]
        reqs.append(
            {
                "question_id": qid,
                "category": meta[qid]["category"],
                "prompt": z["prompt"],
                "output": z["output"],
                "readouts": {t: z["ids"][i] for i, t in enumerate(depths)},
                "agreement": meta[qid]["full_depth_teacher_forced_agreement"],
                "repeat": repeat_rates(z["prompt"], z["output"]),
            }
        )
    return reqs


_REQS: list[dict] = []
_CACHE_CLS = None


def _init_worker(runs_dir: str, suffix_lib: str) -> None:
    global _REQS, _CACHE_CLS
    sys.path.insert(0, suffix_lib)
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    _CACHE_CLS = SuffixDecodingCache
    _REQS = load_requests(Path(runs_dir))


def run_arm(arm: ArmConfig) -> dict:
    cache = _CACHE_CLS(max_tree_depth=64, max_cached_requests=100000)
    per_cat, per_rep, per_self = defaultdict(list), defaultdict(list), defaultdict(list)
    all_stats, no_loop = [], []
    for req in _REQS:
        s = replay_request(
            req["question_id"], req["prompt"], req["output"], req["readouts"], arm, cache
        )
        per_cat[req["category"]].append(s)
        per_rep[to_bin(req["repeat"]["any"], REPEAT_BINS)].append(s)
        per_self[to_bin(req["repeat"]["self"], SELF_BINS)].append(s)
        if req["repeat"]["self"] < NO_LOOP_MAX:
            no_loop.append(s)
        all_stats.append(s)
    T = arm.depth_total
    return {
        "arm": arm.__dict__,
        "overall": summarize(all_stats, T),
        "by_category": {c: summarize(v, T) for c, v in sorted(per_cat.items())},
        "by_repeat_rate": {b: summarize(v, T) for b, v in sorted(per_rep.items())},
        "by_self_repeat": {b: summarize(v, T) for b, v in sorted(per_self.items())},
        "no_loop": summarize(no_loop, T) if no_loop else {},
    }


def fmt_table(results: list[dict]) -> str:
    lines = [
        "| arm | calls/token | speedup vs AR | tokens/step | rows/token | suffix share |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        o = r["overall"]
        lines.append(
            f"| {r['arm']['name']} | {o['calls_per_token']:.2f} | {o['speedup_vs_ar']:.2f}"
            f" | {o['tokens_per_step']:.2f} | {o['rows_per_token']:.1f}"
            f" | {o['suffix_share_of_accepted']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MAIN))
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--suffix-lib", required=True, help="directory holding arctic_inference/")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    reqs = load_requests(Path(args.runs_dir))
    arms = arms_for(args.model)
    with Pool(
        args.workers, initializer=_init_worker, initargs=(args.runs_dir, args.suffix_lib)
    ) as pool:
        results = pool.map(run_arm, arms, chunksize=1)
    agreement = float(np.mean([r["agreement"] for r in reqs]))
    out = {
        "model": args.model,
        "n_requests": len(reqs),
        "n_tokens_counted": int(sum(len(r["output"]) - 1 for r in reqs)),
        "mean_full_depth_teacher_forced_agreement": agreement,
        "requests": [
            {
                "question_id": r["question_id"],
                "category": r["category"],
                "output_len": len(r["output"]),
                "repeat": r["repeat"],
                "agreement": r["agreement"],
            }
            for r in reqs
        ],
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{args.model}: {len(reqs)} requests, agreement {agreement:.4f}")
    print(fmt_table(results))


if __name__ == "__main__":
    main()
