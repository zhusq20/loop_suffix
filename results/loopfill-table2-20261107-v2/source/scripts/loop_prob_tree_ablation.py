"""Ablation of the frozen probability-threshold decoder (v3 C), laid out as ExpWeaver Fig. 4.

Every panel starts from v3 C at tau=0.01 and changes one thing, with the rest fixed
(protocol: results/loop-suffix-spec-20261030/protocol.md §11). Bars are grouped by the
six Spec-Bench categories plus all questions. Top row: speedup over token-by-token
decoding in block calls. Bottom row: rows per word.

(a) threshold tau and (c) use of the deeper readout come from the prob-tree v3 sweep
    (results/loop-spec-prob-tree-20261031/sim/summary-v3-<set>-<model>.json).
(b) placement and suffix-tree update: top_m=1, K=16 cap, offline suffix table, suffix
    tree written only at step end. Replayed here with the frozen calib-v3 table.
(d) readout depth: (d, deeper) pairs, each with its own joint table and in-situ suffix
    table, fitted on the selection split exactly as for v3.

Subcommands:
  calibrate  offline pass with every joint pair, then one in-situ fit per d
  run        replays panels (b) and (d); raw results to --out, compact summary to --summary
  plot       draws figures/ablation-<model>.png from the two summaries

Replays reuse loop_prob_tree_replay.run_config, so counting is the same as the v3 sweep.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loop_prob_tree_replay as lpr  # noqa: E402

from llm_memory_editability.loop_prob_tree import (  # noqa: E402
    Calibration,
    ProbConfig,
    ProbReplay,
    collect_calibration,
)

REPO = Path(__file__).resolve().parents[1]
# (d, deeper readout or None); the deeper one is the deepest stored depth < T and <= 2d
PAIRS = {
    "ouro-2.6b": [(1, 2), (2, 3), (3, None)],
    "huginn-3.5b": [(2, 4), (4, 8), (8, 16), (16, 24)],
}
CATS = ["math_reasoning", "mt_bench", "qa", "rag", "summarization", "translation"]
SOURCES = [
    "scripts/loop_prob_tree_ablation.py",
    "scripts/loop_prob_tree_replay.py",
    "src/llm_memory_editability/loop_prob_tree.py",
]


def _readouts(d: int, deep: int | None) -> tuple[int, ...]:
    return (d,) if deep is None else (d, deep)


def _c(name: str, T: int, d: int, deep: int | None, tau: float, **kw) -> ProbConfig:
    """v3 C: joint d x deeper scoring, pruning, in-situ suffix table; kw overrides one thing."""
    base = dict(tau=tau, joint=True, prune=True, suffix_calibration="insitu")
    return ProbConfig(name, T, d, _readouts(d, deep), **(base | kw))


def panel_b(model: str) -> list[ProbConfig]:
    s = lpr.MODELS[model]
    T, d, deep = s["T"], s["d"], s["deep"]
    out = [ProbConfig("legacy K16", T, d, (d,), max_draft=16, legacy=True)]
    for tau in (0.01, 0.1):
        t = f"t{tau:g}"
        out += [
            _c(f"C {t}", T, d, deep, tau),
            _c(f"C {t} top1", T, d, deep, tau, top_m=1),
            _c(f"C {t} K16", T, d, deep, tau, max_draft=16),
            _c(f"C {t} offline-suffix", T, d, deep, tau, suffix_calibration="offline"),
            _c(f"C {t} step-end", T, d, deep, tau, commit_on_accept=False),
        ]
    return out


def panel_d(model: str) -> list[ProbConfig]:
    T = lpr.MODELS[model]["T"]
    return [
        _c(f"d{d} C t{tau:g}", T, d, deep, tau) for d, deep in PAIRS[model] for tau in (0.01, 0.1)
    ]


def _calib_path(out_dir: Path, setname: str, model: str, d: int) -> Path:
    return out_dir / "calib" / f"{setname}-{model}-d{d}.json"


def cmd_calibrate(args) -> None:
    spec = lpr.MODELS[args.model]
    T = spec["T"]
    out_dir = Path(args.out_dir)
    (out_dir / "calib").mkdir(parents=True, exist_ok=True)
    reqs = lpr.load_requests(Path(args.runs_dir), T)
    pairs = [(d, deep) for d, deep in PAIRS[args.model] if deep is not None]
    # 1. offline pass, as lpr.calibrate, with every joint pair (keys differ by deeper depth)
    t0 = time.time()
    cache = lpr.new_cache(args.suffix_lib)
    calib = {"dev": Calibration(), "confirm": Calibration()}
    for req in reqs:
        collect_calibration(
            req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
            req["depths"], cache, calib[req["split"]], joint_pairs=pairs,
        )  # fmt: skip
    offline = {
        "model": args.model,
        "runs_dir": args.runs_dir,
        "joint_pairs": pairs,
        "n_requests": {s: sum(r["split"] == s for r in reqs) for s in calib},
        "dev": calib["dev"].to_json(),
        "confirm": calib["confirm"].to_json(),
    }
    off_path = out_dir / "calib" / f"offline-{args.set}-{args.model}.json"
    off_path.write_text(json.dumps(offline))
    print(f"[calibrate] offline {args.model} pairs {pairs} in {time.time() - t0:.0f}s")
    # 2. in-situ suffix table per d, as lpr.calibrate_insitu with "v3 A t0.01 offline-suffix"
    for d, deep in PAIRS[args.model]:
        t0 = time.time()
        cfg = ProbConfig("A t0.01 offline-suffix", T, d, _readouts(d, deep), tau=0.01)
        base_dev = Calibration.from_json(offline["dev"])
        cache = lpr.new_cache(args.suffix_lib)
        logs: dict[str, list] = defaultdict(list)
        for req in reqs:
            ProbReplay(
                req["question_id"], req["prompt"], req["output"], req["ids"], req["probs"],
                cfg, cache, base_dev, edge_log=logs[req["split"]],
            ).run()  # fmt: skip
        out = dict(offline) | {"insitu_from": cfg.__dict__}
        for split, log in logs.items():
            c = Calibration.from_json(offline[split])
            for q, m, rival, lead, hit in log:
                c.add_insitu(q, m, rival, lead, hit)
            out[split] = c.to_json()
        _calib_path(out_dir, args.set, args.model, d).write_text(json.dumps(out))
        n = len(logs["dev"])
        right = sum(x[-1] for x in logs["dev"]) / max(1, n)
        print(f"[calibrate] insitu d={d}: {n} dev suffix offers, right {right:.3f}, "
              f"{time.time() - t0:.0f}s")  # fmt: skip


def cmd_check_calib(args) -> None:
    """The recomputed main-d table must equal the frozen calib-v3 table."""
    d = lpr.MODELS[args.model]["d"]
    mine = json.loads(_calib_path(Path(args.out_dir), args.set, args.model, d).read_text())
    frozen = json.loads(Path(args.calibration).read_text())
    deep = lpr.MODELS[args.model]["deep"]
    bad = []
    for split in ("dev", "confirm"):
        for table in ("shallow", "suffix", "insitu"):
            if mine[split].get(table) != frozen[split].get(table):
                bad.append(f"{split}.{table}")
        mj = [r for r in mine[split]["joint"] if r[0] == deep]
        if mj != frozen[split]["joint"]:
            bad.append(f"{split}.joint")
    print(f"[check-calib] {args.model} d={d}: " + ("identical" if not bad else f"DIFFER {bad}"))
    if bad:
        sys.exit(1)


_CALIBS: dict[str, Calibration] = {}


def _init(runs_dir: str, suffix_lib: str, calib_path: str, T: int) -> None:
    lpr._init(runs_dir, suffix_lib, calib_path, T)


def _run(task: tuple[ProbConfig, str]) -> dict:
    cfg, calib_path = task
    if calib_path not in _CALIBS:
        _CALIBS[calib_path] = Calibration.from_json(json.loads(Path(calib_path).read_text())["dev"])
    lpr._CALIB = _CALIBS[calib_path]
    r = lpr.run_config(cfg)
    r["calibration"] = calib_path
    return r


def _compact(r: dict) -> dict:
    keep = ("tokens", "calls", "rows", "calls_per_token", "speedup_vs_ar", "rows_per_token")
    pick = lambda s: {k: s[k] for k in keep}  # noqa: E731
    return {
        "config": r["config"],
        "calibration": r.get("calibration"),
        "overall": pick(r["overall"]),
        "splits": {k: pick(v) for k, v in r["splits"].items()},
        "by_category": {k: pick(v) for k, v in r["by_category"].items()},
    }


def cmd_run(args) -> None:
    spec = lpr.MODELS[args.model]
    out_dir = Path(args.out_dir)
    tasks = [(c, args.calibration) for c in panel_b(args.model)]
    for c in panel_d(args.model):
        tasks.append((c, str(_calib_path(out_dir, args.set, args.model, c.draft_depth))))
    t0 = time.time()
    with Pool(
        args.workers, initializer=_init,
        initargs=(args.runs_dir, args.suffix_lib, args.calibration, spec["T"]),
    ) as pool:  # fmt: skip
        results = pool.map(_run, tasks, chunksize=1)
    hashes = {s: hashlib.sha256((REPO / s).read_bytes()).hexdigest() for s in SOURCES}
    meta = {"model": args.model, "set": args.set, "runs_dir": args.runs_dir, "sources": hashes,
            "seconds": time.time() - t0}  # fmt: skip
    Path(args.out).write_text(json.dumps(meta | {"results": results}))
    Path(args.summary).write_text(
        json.dumps(meta | {"results": [_compact(r) for r in results]}, indent=1)
    )
    print("| setting | speedup | rows/word | speedup (confirm) | s |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for r in results:
        o, c = r["overall"], r["splits"]["confirm"]
        print(f"| {r['config']['name']} | {o['speedup_vs_ar']:.2f} | {o['rows_per_token']:.1f}"
              f" | {c['speedup_vs_ar']:.2f} | {r['seconds']:.0f} |")  # fmt: skip


# ---------------------------------------------------------------- plot

NAMES = {"ouro-2.6b": "Ouro-2.6B", "huginn-3.5b": "Huginn-0125"}
CAT_LABEL = ["数学", "多轮对话", "问答", "检索问答", "摘要", "翻译", "全部"]
INK, MUTED, RULE, SURFACE = "#18202B", "#5A6573", "#D8DDE4", "#FFFFFF"
FULL = "#2a78d6"  # the frozen v3 C in every panel
ALT = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 2-5, fixed order
GREYS = ["#E3E6EA", "#C3C9D1", "#9AA3AE"]  # ordered values other than the frozen one


def _vals(entry: dict, key: str) -> list[float]:
    return [entry["by_category"][c][key] for c in CATS] + [entry["overall"][key]]


def panels(model: str, v3: dict, ab: dict) -> list[dict]:
    """[(title, [(label, entry, colour, frozen)])] for (a)-(d)."""
    s = lpr.MODELS[model]
    d0, deep0 = s["d"], s["deep"]
    V = {r["config"]["name"]: r for r in v3["results"]}
    A = {r["config"]["name"]: r for r in ab["results"]}
    taus = ("0.3", "0.1", "0.03")
    a = [(f"τ = {t}", V[f"v3 C t{t}"], g, False) for t, g in zip(taus, GREYS, strict=True)]
    a.append(("τ = 0.01", V["v3 C t0.01"], FULL, True))
    b = [("v3 C", A["C t0.01"], FULL, True)]
    for (suffix, label), col in zip(
        (("top1", "每个读出只放第1名"), ("K16", "每步最多16层"),
         ("offline-suffix", "后缀树用离线校准"), ("step-end", "后缀树步末才写入")),
        ALT, strict=True,
    ):  # fmt: skip
        b.append((label, A[f"C t0.01 {suffix}"], col, False))
    c = [
        ("不用更深读出（E）", V["v3 E t0.01"], ALT[0], False),
        ("只剪枝（D）", V["v3 D t0.01"], ALT[1], False),
        ("只放候选（A）", V["v3 A t0.01"], ALT[2], False),
        ("放候选＋剪枝（C）", V["v3 C t0.01"], FULL, True),
    ]
    greys = iter(GREYS)
    dd = []
    for d, deep in PAIRS[model]:
        lab = f"d = {d}" + (f"，更深 {deep}" if deep else "，无更深")
        frozen = (d, deep) == (d0, deep0)
        dd.append((lab, A[f"d{d} C t0.01"], FULL if frozen else next(greys), frozen))
    return [
        {"title": "(a) 阈值 τ", "bars": a},
        {"title": "(b) 放候选与后缀树写入", "bars": b},
        {"title": "(c) 更深读出的用法", "bars": c},
        {"title": "(d) 读出深度", "bars": dd},
    ]


def cmd_plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    v3 = json.loads(Path(args.v3_summary).read_text())
    ab = json.loads(Path(args.summary).read_text())
    P = panels(args.model, v3, ab)
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.2), sharex=True)
    x = list(range(len(CAT_LABEL)))
    for j, p in enumerate(P):
        n = len(p["bars"])
        w = 0.82 / n
        for row, key in enumerate(("speedup_vs_ar", "rows_per_token")):
            ax = axes[row][j]
            for k, (lab, e, col, frozen) in enumerate(p["bars"]):
                xs = [i - 0.41 + w * (k + 0.5) for i in x]
                edge, lw = (INK, 0.9) if frozen else (SURFACE, 0.5)
                ax.bar(xs, _vals(e, key), width=w * 0.9, color=col, label=lab,
                       edgecolor=edge, linewidth=lw)  # fmt: skip
            ax.axvspan(len(CATS) - 0.5, len(CATS) + 0.5, color="#F2F4F7", zorder=0)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=RULE, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(RULE)
            ax.tick_params(colors=MUTED, labelsize=8.5)
            ax.set_xlim(-0.55, len(CAT_LABEL) - 0.45)
            if row == 0:
                ax.set_title(p["title"], loc="left", fontsize=11, color=INK)
                ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=1,
                          handlelength=1.0, borderaxespad=0.2)  # fmt: skip
            else:
                ax.set_xticks(x, CAT_LABEL, fontsize=8.5, color=INK)
        axes[0][0].set_ylabel("相对逐词的加速（块调用）", fontsize=9, color=MUTED)
        axes[1][0].set_ylabel("每词行数（算力）", fontsize=9, color=MUTED)
    for row in range(2):
        top = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(0, top * (1.6 if row == 0 else 1.0))
    s = lpr.MODELS[args.model]
    fig.suptitle(
        f"{NAMES[args.model]}（T = {s['T']}）：从冻结的 v3 C（τ = 0.01，d = {s['d']}，"
        f"更深 {s['deep']}）出发，每栏只改一处",
        x=0.01, ha="left", fontsize=12, color=INK,
    )  # fmt: skip
    fig.legend(handles=[Patch(facecolor=FULL, edgecolor=INK, label="冻结的 v3 C")],
               loc="upper right", frameon=False, fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.png, dpi=150, facecolor=SURFACE)
    print(f"[plot] {args.png}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "check-calib", "run", "plot"):
        q = sub.add_parser(name)
        q.add_argument("--model", required=True, choices=sorted(lpr.MODELS))
        q.add_argument("--set", default="sb-chat")
    for name in ("calibrate", "run"):
        q = sub.choices[name]
        q.add_argument("--runs-dir", required=True)
        q.add_argument("--suffix-lib", required=True)
    for name in ("calibrate", "check-calib", "run"):
        sub.choices[name].add_argument("--out-dir", required=True)
    for name in ("check-calib", "run"):
        sub.choices[name].add_argument("--calibration", required=True, help="frozen calib-v3")
    r = sub.choices["run"]
    r.add_argument("--out", required=True)
    r.add_argument("--summary", required=True)
    r.add_argument("--workers", type=int, default=4)
    q = sub.choices["plot"]
    q.add_argument("--v3-summary", required=True)
    q.add_argument("--summary", required=True)
    q.add_argument("--png", required=True)
    args = p.parse_args()
    run = {"calibrate": cmd_calibrate, "check-calib": cmd_check_calib, "run": cmd_run,
           "plot": cmd_plot}  # fmt: skip
    run[args.cmd](args)


if __name__ == "__main__":
    main()
