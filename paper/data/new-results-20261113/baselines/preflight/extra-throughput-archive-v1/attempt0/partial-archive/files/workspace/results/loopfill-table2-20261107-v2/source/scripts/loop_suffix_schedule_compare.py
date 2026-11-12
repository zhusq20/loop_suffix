"""The same suffix tree under two schedules: one verification per step, or a looped pipeline.

Answers "what does the loop give the suffix tree" on the stored greedy outputs and
per-depth readouts, CPU only (protocol: results/loop-suffix-spec-20261030/protocol.md §10).

Standard schedule ("std"): every draft enters at the start of a step and is checked by
one full verification T block calls later; a missing word ends the step and comes from
the full-depth output. This is SuffixDecoding as a non-looped model runs it, counted in
block calls. It is loop_prob_tree with no readouts: top_m=0 and no rescoring, so the
suffix tree is the only drafter and its candidates are placed by the same
probability threshold tau as in the loop schedule.

Loop schedule: loop_prob_tree as the prob-tree batch runs it (v3 B: joint d x deep
readouts; v3 C: B plus pruning), and the legacy suffix+shallow decoder (fill top-1, K=16).

For every setting the replay records, besides calls and rows per word:
- where each committed word came from and how many block calls after the previous
  word it started computing (suffix tree, depth-d fill or sibling, deeper readout,
  or the full-depth output);
- for every row that was not on the committed path, how many loop steps it ran before
  it was removed or the step ended. A committed word's own rows are its verification,
  so only these rows are wasted compute.

Usage:
  python scripts/loop_suffix_schedule_compare.py run --model ouro-2.6b --set sb-chat \
      --runs-dir <persistent>/loop-suffix-spec-20261030/runs/sb-chat/ouro-2.6b \
      --suffix-lib <persistent>/loop-suffix-spec-20261030/arctic-suffix/pkg \
      --calibration <persistent>/loop-spec-prob-tree-20261031/calib-v3/sb-chat-ouro-2.6b.json \
      --out <dir>/sb-chat-ouro-2.6b.json --workers 4
  python scripts/loop_suffix_schedule_compare.py check ...   (same args, --n 20)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loop_prob_tree_replay import MODELS, load_requests, new_cache  # noqa: E402

from llm_memory_editability.loop_prob_tree import (  # noqa: E402
    Calibration,
    ProbConfig,
    ProbReplay,
    _Step,
)
from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request  # noqa: E402

TAUS_STD = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
TAUS_LOOP = (0.3, 0.1, 0.03, 0.01)


def configs(model: str) -> list[ProbConfig]:
    s = MODELS[model]
    T, d, deep = s["T"], s["d"], s["deep"]
    ins = dict(suffix_calibration="insitu")
    cfgs = [
        ProbConfig(f"std t{t:g}", T, d, (d,), tau=t, top_m=0, rescore=False, **ins)
        for t in TAUS_STD
    ]
    cfgs.append(ProbConfig("legacy K16", T, d, (d,), max_draft=16, legacy=True))
    for t in TAUS_LOOP:
        cfgs.append(ProbConfig(f"v3 B t{t:g}", T, d, (d, deep), tau=t, joint=True, **ins))
        cfgs.append(
            ProbConfig(f"v3 C t{t:g}", T, d, (d, deep), tau=t, joint=True, prune=True, **ins)
        )
    return cfgs


def is_std(cfg: ProbConfig) -> bool:
    return cfg.name.startswith("std")


def run_request(sim: ProbReplay, std: bool) -> dict:
    """ProbReplay.run with the word ledger and the wasted-row histogram added."""
    out = sim.out
    L = len(out)
    T = sim.cfg.depth_total
    acc: dict = defaultdict(float)
    waste = defaultdict(lambda: np.zeros(T + 1, dtype=np.int64))  # source -> steps run
    if sim.cache is not None:
        sim.cache.start_request(sim.req_id, np.asarray(sim.base_ctx, dtype=np.int32))
        sim.cache.add_active_response(sim.req_id, np.asarray(out[:1], dtype=np.int32))
    committed = 1
    while committed < L:
        step = _Step(sim, committed)
        step.run()
        tal = step.tally()
        end = step.end
        acc["tokens"] += len(step.accepted)
        acc["steps"] += 1
        acc["calls"] += end
        acc["rows"] += int(tal["rows_per_call"].sum())
        acc["reads"] += 0 if std else int(tal["reads_per_call"].sum())
        chain = {id(step.root)}
        n = step.root
        for tok in step.accepted:
            c = n.children.get(tok)
            if c is None:  # the step's last word, read from the full-depth output
                src, delay = "full", end - n.start
            else:
                src, delay = c.src, c.start - n.start
                chain.add(id(c))
                n = c
            acc[f"n_{src}"] += 1
            acc[f"delay_{src}"] += delay
        for m in step.nodes:
            stop = min(end, m.start + T, m.killed if m.killed is not None else end)
            ran = max(0, stop - m.start)
            if id(m) in chain:
                acc["rows_committed"] += ran
            else:
                waste[m.src][ran] += 1
        if sim.cache is not None and not sim.cfg.eager_commit:
            sim.cache.add_active_response(sim.req_id, np.asarray(step.accepted, dtype=np.int32))
        committed += len(step.accepted)
    if sim.cache is not None:
        sim.cache.stop_request(sim.req_id)
    return {"acc": dict(acc), "waste": {k: v.tolist() for k, v in waste.items()}}


def run_original(req: dict, T: int, d: int, cache) -> dict:
    """SuffixDecoding's own settings (min_token_prob 0.1, K=16), standard schedule."""
    arm = ArmConfig("sd-original", T, d, 16, True, False)
    s = replay_request(req["question_id"], req["prompt"], req["output"], req["ids"], arm, cache)
    return {
        "acc": {
            "tokens": s["tokens"],
            "steps": s["steps"],
            "calls": s["calls"],
            "rows": s["rows"],
            "reads": 0,
            "n_suffix": s["from_suffix"],
            "n_full": s["tokens"] - s["from_suffix"],
        },
        "waste": {},
    }


def summarize(parts: list[dict], T: int) -> dict:
    acc: dict = defaultdict(float)
    waste: dict = defaultdict(lambda: np.zeros(T + 1, dtype=np.int64))
    for p in parts:
        for k, v in p["acc"].items():
            acc[k] += v
        for k, v in p["waste"].items():
            waste[k] += np.asarray(v, dtype=np.int64)
    n = max(1.0, acc["tokens"])
    srcs = sorted(k[2:] for k in acc if k.startswith("n_"))
    wasted_rows = {k: int((np.arange(T + 1) * v).sum()) for k, v in waste.items()}
    return {
        "tokens": int(acc["tokens"]),
        "calls_per_token": acc["calls"] / n,
        "speedup_vs_ar": T * n / max(1.0, acc["calls"]),
        "rows_per_token": acc["rows"] / n,
        "reads_per_token": acc["reads"] / n,
        "committed_rows_per_token": acc.get("rows_committed", 0.0) / n,
        "wasted_rows_per_token": sum(wasted_rows.values()) / n,
        "share": {s: acc[f"n_{s}"] / n for s in srcs},
        "calls_per_token_by_source": {s: acc.get(f"delay_{s}", 0.0) / n for s in srcs},
        "mean_delay": {s: acc.get(f"delay_{s}", 0.0) / max(1.0, acc[f"n_{s}"]) for s in srcs},
        "ledger_gap": sum(acc.get(f"delay_{s}", 0.0) for s in srcs) / max(1.0, acc["calls"]) - 1,
        "wasted_rows_per_token_by_source": {k: v / n for k, v in wasted_rows.items()},
        "wasted_nodes_by_steps_run": {k: v.tolist() for k, v in waste.items()},
    }


_REQS: list[dict] = []
_CALIB: Calibration | None = None
_LIB = ""


def _init(runs_dir: str, suffix_lib: str, calib_path: str, T: int) -> None:
    global _REQS, _CALIB, _LIB
    _REQS = load_requests(Path(runs_dir), T)
    _CALIB = Calibration.from_json(json.loads(Path(calib_path).read_text())["dev"])
    _LIB = suffix_lib


def run_config(cfg: ProbConfig | str) -> dict:
    t0 = time.time()
    cache = new_cache(_LIB)
    by_split: dict = defaultdict(list)
    by_cat: dict = defaultdict(list)
    for req in _REQS:
        if isinstance(cfg, str):  # SuffixDecoding's own settings
            T = MODELS[cfg]["T"]
            part = run_original(req, T, MODELS[cfg]["d"], cache)
        else:
            T = cfg.depth_total
            sim = ProbReplay(
                req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
                cfg, cache, _CALIB,
            )  # fmt: skip
            part = run_request(sim, is_std(cfg))
        by_split[req["split"]].append(part)
        by_cat[req["category"]].append(part)
    allp = [p for v in by_split.values() for p in v]
    name = "sd-original K16" if isinstance(cfg, str) else cfg.name
    return {
        "name": name,
        "config": None if isinstance(cfg, str) else cfg.__dict__,
        "seconds": time.time() - t0,
        "overall": summarize(allp, T),
        "splits": {k: summarize(v, T) for k, v in by_split.items()},
        "by_category": {k: summarize(v, T) for k, v in sorted(by_cat.items())},
    }


def sources_hash() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = [
        "scripts/loop_suffix_schedule_compare.py",
        "scripts/loop_prob_tree_replay.py",
        "src/llm_memory_editability/loop_prob_tree.py",
        "src/llm_memory_editability/loop_suffix_spec.py",
    ]
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16] for f in files}


def cmd_run(args) -> None:
    spec = MODELS[args.model]
    cfgs: list = [args.model, *configs(args.model)]  # the model name stands for sd-original
    if args.only:
        keep = set(args.only.split(","))
        cfgs = [c for c in cfgs if ("sd-original K16" if isinstance(c, str) else c.name) in keep]
    with Pool(
        args.workers, initializer=_init,
        initargs=(args.runs_dir, args.suffix_lib, args.calibration, spec["T"]),
    ) as pool:  # fmt: skip
        results = pool.map(run_config, cfgs, chunksize=1)
    out = {
        "model": args.model,
        "set": args.set,
        "runs_dir": args.runs_dir,
        "calibration": args.calibration,
        "sources": sources_hash(),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print("| setting | calls/word | speedup | rows/word | wasted rows/word | reads/word | s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        o = r["overall"]
        print(
            f"| {r['name']} | {o['calls_per_token']:.3f} | {o['speedup_vs_ar']:.2f} |"
            f" {o['rows_per_token']:.1f} | {o['wasted_rows_per_token']:.1f} |"
            f" {o['reads_per_token']:.1f} | {r['seconds']:.0f} |"
        )


def cmd_check(args) -> None:
    """Instrumented replay must count the same as ProbReplay.run on the first n questions."""
    spec = MODELS[args.model]
    reqs = load_requests(Path(args.runs_dir), spec["T"])[: args.n]
    calib = Calibration.from_json(json.loads(Path(args.calibration).read_text())["dev"])
    bad = 0
    for cfg in configs(args.model):
        c1, c2 = new_cache(args.suffix_lib), new_cache(args.suffix_lib)
        for req in reqs:
            a = (req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"], cfg)
            ref = ProbReplay(*a, c1, calib).run()
            got = run_request(ProbReplay(*a, c2, calib), is_std(cfg))["acc"]
            for k in ("tokens", "steps", "calls", "rows"):
                if int(ref.get(k, 0)) != int(got.get(k, 0)):
                    bad += 1
                    q = req["question_id"]
                    print(f"[check] {cfg.name} q{q} {k}: {ref.get(k)} vs {got.get(k)}")
            if is_std(cfg) and any(got.get(f"n_{s}", 0) for s in ("fill", "sibling", "deep")):
                bad += 1
                print(f"[check] {cfg.name} q{req['question_id']}: readout word in std schedule")
    print(f"[check] {args.model}: {len(reqs)} questions, {bad} mismatches")
    if bad:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        q = sub.add_parser(name)
        q.add_argument("--model", required=True, choices=sorted(MODELS))
        q.add_argument("--set", default="sb-chat")
        q.add_argument("--runs-dir", required=True)
        q.add_argument("--suffix-lib", required=True)
        q.add_argument("--calibration", required=True)
    r = sub.choices["run"]
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--only", default="")
    sub.choices["check"].add_argument("--n", type=int, default=20)
    args = p.parse_args()
    {"run": cmd_run, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
