"""Calibrate edge probabilities and replay the probability-threshold draft tree.

calibrate: one pass over the greedy outputs in question_id order, counting how often
each suffix-tree and shallow-readout candidate equals the full-depth output. Counts
from the selection split form the calibration; counts from the confirmation split
check it.

sweep: replays the current decoder (legacy) and the probability rule under several
settings, in parallel over settings. Every setting sees a fresh ArcticInference
SuffixDecodingCache whose global tree fills online, as in loop_suffix_spec_simulate.
The legacy setting is also checked against loop_suffix_spec.replay_request per
question.

Protocol: results/loop-spec-prob-tree-20261031/protocol.md
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from llm_memory_editability.loop_prob_tree import (
    Calibration,
    ProbConfig,
    ProbReplay,
    collect_calibration,
)
from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request

MODELS = {
    "ouro-2.6b": {"T": 4, "d": 1, "deep": 2},
    "huginn-3.5b": {"T": 32, "d": 4, "deep": 8},
}
# The wall-clock questions (10 per category); never used to choose settings.
CONFIRM = frozenset(range(81, 561, 8))
TAUS = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
CAPS = (256, 128, 64, 32)


def load_requests(runs_dir: Path, T: int) -> list[dict]:
    meta = {}
    for path in sorted(runs_dir.glob("index-shard*.jsonl")):
        for line in path.read_text().splitlines():
            if line:
                rec = json.loads(line)
                meta[rec["question_id"]] = rec
    reqs = []
    for qid in sorted(meta):
        z = np.load(runs_dir / "readouts" / f"{qid}.npz")
        depths = [int(t) for t in z["depths"] if int(t) < T]
        index = {int(t): i for i, t in enumerate(z["depths"])}
        reqs.append(
            {
                "question_id": qid,
                "category": meta[qid]["category"],
                "split": "confirm" if qid in CONFIRM else "dev",
                "prompt": z["prompt"],
                "output": z["output"],
                "depths": depths,
                "ids": {t: z["ids"][index[t]] for t in depths},
                "probs": {t: z["probs"][index[t]].astype(np.float32) for t in depths},
            }
        )
    return reqs


def new_cache(suffix_lib: str):
    if suffix_lib not in sys.path:
        sys.path.insert(0, suffix_lib)
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    return SuffixDecodingCache(max_tree_depth=64, max_cached_requests=100000)


def calibrate(args) -> None:
    spec = MODELS[args.model]
    reqs = load_requests(Path(args.runs_dir), spec["T"])
    cache = new_cache(args.suffix_lib)
    calib = {"dev": Calibration(), "confirm": Calibration()}
    t0 = time.time()
    for req in reqs:
        collect_calibration(
            req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
            req["depths"], cache, calib[req["split"]], joint_pairs=[(spec["d"], spec["deep"])],
        )  # fmt: skip
    out = {
        "model": args.model,
        "runs_dir": args.runs_dir,
        "n_requests": {s: sum(r["split"] == s for r in reqs) for s in calib},
        "seconds": time.time() - t0,
        "dev": calib["dev"].to_json(),
        "confirm": calib["confirm"].to_json(),
    }
    Path(args.out).write_text(json.dumps(out))
    print(f"[calibrate] {args.model}: {out['n_requests']} in {out['seconds']:.0f}s -> {args.out}")


def calibrate_insitu(args) -> None:
    """Fit the suffix-edge table on the suffix queries a decoder makes.

    Replays one setting with the offline calibration and logs every suffix token
    offered under an on-path node, with whether a rival child was already there and
    how far the node sits ahead of the last committed token. The fitted table is added
    to a copy of the offline calibration file.
    """
    spec = MODELS[args.model]
    cfg = next(c for c in configs_for(args.model, "v3") if c.name == args.config)
    base = json.loads(Path(args.calibration).read_text())
    reqs = load_requests(Path(args.runs_dir), spec["T"])
    calib = Calibration.from_json(base["dev"])
    cache = new_cache(args.suffix_lib)
    logs: dict[str, list] = defaultdict(list)
    for req in reqs:
        sim = ProbReplay(
            req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
            cfg, cache, calib, edge_log=logs[req["split"]],
        )  # fmt: skip
        sim.run()
    out = dict(base) | {"insitu_from": cfg.__dict__}
    for split, log in logs.items():
        c = Calibration.from_json(base[split])
        for q, m, rival, lead, hit in log:
            c.add_insitu(q, m, rival, lead, hit)
        out[split] = c.to_json()
        right = sum(x[-1] for x in log) / max(1, len(log))
        print(f"[insitu] {args.model} {split}: {len(log)} suffix offers, right {right:.3f}")
    Path(args.out).write_text(json.dumps(out))


def configs_for(model: str, stage: str) -> list[ProbConfig]:
    s = MODELS[model]
    T, d, deep = s["T"], s["d"], s["deep"]
    one, two = (d,), (d, deep)
    if stage == "v3":
        return v3_configs(T, d, one, two)
    cfgs = [
        ProbConfig("legacy K16", T, d, one, max_draft=16, legacy=True),
        ProbConfig("rule top1 K16 t0.01", T, d, one, tau=0.01, top_m=1, max_draft=16),
        ProbConfig("rule top8 K16 t0.01", T, d, one, tau=0.01, max_draft=16),
        ProbConfig("rule top8 noK t0.01", T, d, one, tau=0.01),
        ProbConfig("rule top8 noK deep t0.01", T, d, two, tau=0.01),
    ]
    if stage == "main":
        return cfgs
    for tau in TAUS:
        if tau != 0.01:
            cfgs.append(ProbConfig(f"rule top8 noK t{tau:g}", T, d, one, tau=tau))
            cfgs.append(ProbConfig(f"rule top8 noK deep t{tau:g}", T, d, two, tau=tau))
    for tau in (0.01, 0.003):
        for rd, tag in ((one, ""), (two, " deep")):
            for cap in CAPS:
                cfgs.append(
                    ProbConfig(
                        f"rule top8 noK{tag} t{tau:g} B{cap}", T, d, rd, tau=tau, row_cap=cap
                    )
                )
    for ts in (0.1, 0.001):
        cfgs.append(
            ProbConfig(f"rule top8 noK deep t0.01 suffix{ts:g}", T, d, two, tau=0.01, tau_suffix=ts)
        )
    for tdp in (0.1, 0.001):
        cfgs.append(
            ProbConfig(f"rule top8 noK deep t0.01 deep{tdp:g}", T, d, two, tau=0.01, tau_deep=tdp)
        )
    cfgs.append(
        ProbConfig(
            "rule top8 noK deep t0.01 onpath-reads", T, d, two, tau=0.01, offpath_readouts=False
        )
    )
    cfgs.append(
        ProbConfig(
            "legacy K16 onpath-reads", T, d, one, max_draft=16, legacy=True, offpath_readouts=False
        )
    )
    for rd, tag in ((one, ""), (two, " deep")):
        name = f"rule top8 noK{tag} t0.01"
        cfgs.append(ProbConfig(f"{name} stale-suffix", T, d, rd, tau=0.01, commit_on_accept=False))
        cfgs.append(ProbConfig(f"{name} no-rescore", T, d, rd, tau=0.01, rescore=False))
    return cfgs


def v3_configs(T, d, one, two) -> list[ProbConfig]:
    """Ablation of the two precision changes, at several thresholds.

    A: deeper readout places and rescores (v2). B: A with the joint d x deep table.
    C: B and pruning. D: deeper readout only lowers scores and prunes, never places.
    E: no deeper readout, with pruning; E0 without. All use the in-situ suffix table,
    except the two `offline-suffix` rows that measure its effect.
    """
    ins = dict(suffix_calibration="insitu")
    cfgs = [
        ProbConfig("legacy K16", T, d, one, max_draft=16, legacy=True),
        ProbConfig("v3 rule top1 K16 t0.01", T, d, one, tau=0.01, top_m=1, max_draft=16, **ins),
    ]
    for tau in (0.01, 0.03, 0.1, 0.3):
        t = f"t{tau:g}"
        cfgs += [
            ProbConfig(f"v3 A {t}", T, d, two, tau=tau, **ins),
            ProbConfig(f"v3 B {t}", T, d, two, tau=tau, joint=True, **ins),
            ProbConfig(f"v3 C {t}", T, d, two, tau=tau, joint=True, prune=True, **ins),
            ProbConfig(
                f"v3 D {t}",
                T,
                d,
                two,
                tau=tau,
                joint=True,
                prune=True,
                deep_expand=False,
                deep_raise=False,
                **ins,
            ),  # fmt: skip
            ProbConfig(f"v3 E {t}", T, d, one, tau=tau, prune=True, **ins),
            ProbConfig(f"v3 E0 {t}", T, d, one, tau=tau, **ins),
        ]
    for tau in (0.01, 0.1):
        cfgs.append(ProbConfig(f"v3 A t{tau:g} offline-suffix", T, d, two, tau=tau))
    return cfgs


_REQS: list[dict] = []
_CALIB: Calibration | None = None
_LIB = ""


def _init(runs_dir: str, suffix_lib: str, calib_path: str, T: int) -> None:
    global _REQS, _CALIB, _LIB
    _REQS = load_requests(Path(runs_dir), T)
    _CALIB = Calibration.from_json(json.loads(Path(calib_path).read_text())["dev"])
    _LIB = suffix_lib


def _summ(parts: list[dict], T: int) -> dict:
    keys = {k for p in parts for k in p if k != "hist"}
    tot = {k: sum(p.get(k, 0) for p in parts) for k in keys}
    tokens, calls = max(1, tot.get("tokens", 0)), max(1, tot.get("calls", 0))
    accepted = sum(v for k, v in tot.items() if k.startswith("from_"))
    return tot | {
        "calls_per_token": calls / tokens,
        "speedup_vs_ar": T * tokens / calls,
        "tokens_per_step": tokens / max(1, tot.get("steps", 0)),
        "rows_per_token": tot.get("rows", 0) / tokens,
        "rows_per_call": tot.get("rows", 0) / calls,
        "reads_per_call": tot.get("reads", 0) / calls,
        "accepted_share": {
            k[5:]: v / max(1, accepted) for k, v in sorted(tot.items()) if k.startswith("from_")
        },
    }


def run_config(cfg: ProbConfig) -> dict:
    t0 = time.time()
    T = cfg.depth_total
    cache = new_cache(_LIB)
    check = new_cache(_LIB) if cfg.legacy and cfg.offpath_readouts else None
    per_req, by_split, by_cat = {}, defaultdict(list), defaultdict(list)
    hist = {"dev": defaultdict(int), "confirm": defaultdict(int)}
    miss_rank = {
        "dev": defaultdict(lambda: defaultdict(int)),
        "confirm": defaultdict(lambda: defaultdict(int)),
    }
    mismatches = []
    for req in _REQS:
        split = req["split"]

        def on_miss(node, target, req=req, split=split):
            i = node.pos + 1
            miss_rank[split]["n"][0] += 1
            for r in req["depths"]:
                row = [int(t) for t in req["ids"][r][i]]
                rank = row.index(target) if target in row else 8
                miss_rank[split][r][rank] += 1

        sim = ProbReplay(
            req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
            cfg, cache, _CALIB,
        )  # fmt: skip
        s = sim.run(on_miss=on_miss)
        for key, n in s.pop("hist").items():
            hist[split][key] += n
        per_req[req["question_id"]] = {k: s.get(k, 0) for k in ("tokens", "calls", "rows", "reads")}
        by_split[split].append(s)
        by_cat[req["category"]].append(s)
        if check is not None:
            arm = ArmConfig("legacy", T, cfg.draft_depth, cfg.max_draft, True, True)
            old = replay_request(
                req["question_id"], req["prompt"], req["output"], req["ids"], arm, check
            )
            if (old["calls"], old["tokens"]) != (s.get("calls", 0), s.get("tokens", 0)):
                mismatches.append(req["question_id"])
    return {
        "config": cfg.__dict__,
        "seconds": time.time() - t0,
        "legacy_check": None if check is None else {"mismatched_questions": mismatches},
        "overall": _summ(by_split["dev"] + by_split["confirm"], T),
        "splits": {k: _summ(v, T) for k, v in by_split.items()},
        "by_category": {k: _summ(v, T) for k, v in sorted(by_cat.items())},
        "hist": {k: [[a, b, n] for (a, b), n in sorted(v.items())] for k, v in hist.items()},
        "miss_rank": {
            sp: {str(r): dict(sorted(v.items())) for r, v in m.items()}
            for sp, m in miss_rank.items()
        },
        "per_request": per_req,
    }


def sweep(args) -> None:
    spec = MODELS[args.model]
    cfgs = configs_for(args.model, args.stage)
    if args.only:
        cfgs = [c for c in cfgs if c.name in set(args.only.split(","))]
    with Pool(
        args.workers, initializer=_init,
        initargs=(args.runs_dir, args.suffix_lib, args.calibration, spec["T"]),
    ) as pool:  # fmt: skip
        results = pool.map(run_config, cfgs, chunksize=1)
    Path(args.out).write_text(
        json.dumps({"model": args.model, "runs_dir": args.runs_dir, "results": results})
    )
    print(
        "| setting | calls/token (dev) | speedup (dev) | rows/call | reads/call | tokens/step | s |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        o = r["splits"]["dev"]
        chk = r["legacy_check"]
        flag = "" if chk is None else f" (check: {len(chk['mismatched_questions'])} mismatched)"
        print(
            f"| {r['config']['name']}{flag} | {o['calls_per_token']:.3f} | {o['speedup_vs_ar']:.2f}"
            f" | {o['rows_per_call']:.1f} | {o['reads_per_call']:.1f} | {o['tokens_per_step']:.2f}"
            f" | {r['seconds']:.0f} |"
        )


P_EDGES = (0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0001)


def trace(args) -> None:
    """Placement-time p against whether the node lies on the greedy path.

    Only placements offered by an on-path node count, and p is taken relative to that
    node, so the outcome does not depend on the stand-in readouts of off-path nodes.
    Suffix chains are split by draft depth to check the product of edge probabilities.
    """
    spec = MODELS[args.model]
    cfg = next(c for c in configs_for(args.model, args.stage) if c.name == args.config)
    reqs = load_requests(Path(args.runs_dir), spec["T"])
    calib = Calibration.from_json(json.loads(Path(args.calibration).read_text())["dev"])
    cache = new_cache(args.suffix_lib)
    rows: dict[str, list] = defaultdict(list)
    for req in reqs:
        buf: list = []
        sim = ProbReplay(
            req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
            cfg, cache, calib, trace=buf,
        )  # fmt: skip
        sim.run()
        rows[req["split"]].extend(buf)

    def group(src: str, depth: int) -> str:
        if src != "suffix":
            return src
        return (
            "suffix depth 1"
            if depth == 1
            else ("suffix depth 2-4" if depth <= 4 else "suffix depth 5+")
        )

    table = {}
    for split, items in rows.items():
        cells = defaultdict(lambda: [0, 0.0, 0])  # n, sum of p, hits
        for p_rel, hit, src, depth, rivals in items:
            b = max(0, bisect.bisect_right(P_EDGES, p_rel) - 1)
            g = group(src, depth)
            keys = (f"all|{b}", f"{g}|{b}", f"{g} {'with' if rivals else 'no'} sibling|{b}")
            for key in keys:
                c = cells[key]
                c[0] += 1
                c[1] += p_rel
                c[2] += int(hit)
        table[split] = {
            k: {"n": n, "mean_p": sp / n, "observed": h / n}
            for k, (n, sp, h) in sorted(cells.items())
        }
    out = {"model": args.model, "config": cfg.__dict__, "p_edges": P_EDGES, "table": table}
    Path(args.out).write_text(json.dumps(out, indent=1))
    for split in ("dev", "confirm"):
        print(f"[trace] {args.model} {cfg.name} {split}")
        for k, v in table.get(split, {}).items():
            if v["n"] >= 30:
                print(f"   {k:22s} n={v['n']:7d} p={v['mean_p']:.3f} observed={v['observed']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "sweep", "trace", "calibrate-insitu"):
        q = sub.add_parser(name)
        q.add_argument("--model", required=True, choices=sorted(MODELS))
        q.add_argument("--runs-dir", required=True)
        q.add_argument("--suffix-lib", required=True)
        q.add_argument("--out", required=True)
    s = sub.choices["sweep"]
    s.add_argument("--calibration", required=True)
    s.add_argument("--stage", choices=["main", "full", "v3"], default="full")
    s.add_argument("--only", default="")
    s.add_argument("--workers", type=int, default=16)
    t = sub.choices["trace"]
    t.add_argument("--calibration", required=True)
    t.add_argument("--config", default="rule top8 noK deep t0.01")
    t.add_argument("--stage", default="full")
    c = sub.choices["calibrate-insitu"]
    c.add_argument("--calibration", required=True)
    c.add_argument("--config", default="v3 A t0.01 offline-suffix")
    args = p.parse_args()
    run = {
        "calibrate": calibrate,
        "sweep": sweep,
        "trace": trace,
        "calibrate-insitu": calibrate_insitu,
    }
    run[args.cmd](args)


if __name__ == "__main__":
    main()
