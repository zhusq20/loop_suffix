"""Tables and figures for protocol §10 comparisons 2 and 3 (loop-suffix-spec-20261030).

Reads the outputs of loop_suffix_schedule_compare.py, copies a compact summary per
(set, model) into results/loop-suffix-spec-20261030/schedule/, and draws:
- frontier.png: block calls per word against rows per word, standard schedule (suffix
  tree only, one verification per step) and loop schedule (v3 B, v3 C), per model;
- waste.png: rows per word split into the committed words' own rows and the wasted
  rows, the latter by how many loop steps they ran before removal.

Usage: python scripts/loop_suffix_schedule_report.py --in-dir <persistent>/.../schedule
       --out-dir results/loop-suffix-spec-20261030/schedule
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NAMES = {"ouro-2.6b": "Ouro-2.6B", "huginn-3.5b": "Huginn-0125"}
DEPTHS = {"ouro-2.6b": (4, 1, 2), "huginn-3.5b": (32, 4, 8)}
SETS = {"sb-chat": "sb1024 chat", "full": "raw-256"}
INK, MUTED, RULE = "#18202B", "#5A6573", "#D8DDE4"
SUFFIX, SHALLOW, BAD = "#0D6F7E", "#A35F0F", "#B33838"


def compact(d: dict) -> dict:
    T = DEPTHS[d["model"]][0]
    keep = []
    for r in d["results"]:
        o = r["overall"]
        if r["name"] == "sd-original K16":  # replay_request does not track rows per node
            o = dict(o, wasted_rows_per_token=None, committed_rows_per_token=None)
        keep.append(
            {
                "name": r["name"],
                "overall": o,
                "splits": {
                    k: {x: v[x] for x in ("calls_per_token", "rows_per_token", "tokens")}
                    for k, v in r["splits"].items()
                },
            }
        )
    return {k: d[k] for k in ("model", "set", "runs_dir", "calibration", "sources")} | {
        "T": T,
        "results": keep,
    }


def waste_buckets(o: dict, model: str) -> dict:
    """Wasted rows per word by loop steps run before removal: <=d, <=deep, <T, =T."""
    T, d, deep = DEPTHS[model]
    edges = [(1, d), (d + 1, deep), (deep + 1, T - 1), (T, T)]
    n = o["tokens"]
    out = {}
    for lo, hi in edges:
        rows = 0
        for hist in o["wasted_nodes_by_steps_run"].values():
            h = np.asarray(hist)
            k = np.arange(len(h))
            rows += int((h * k)[lo : hi + 1].sum())
        out[f"{lo}-{hi}"] = rows / n
    return out


def table_rows(c: dict) -> list[str]:
    T = c["T"]
    lines = []
    for r in c["results"]:
        o = r["overall"]
        w = o["wasted_rows_per_token"]
        lines.append(
            f"| {r['name']} | {o['calls_per_token']:.3f} | {o['speedup_vs_ar']:.2f} | "
            f"{o['rows_per_token']:.1f} | {o['rows_per_token'] / T:.2f} | "
            f"{'—' if w is None else f'{w:.1f}'} | {o['reads_per_token']:.1f} |"
        )
    return lines


def frontier(data: dict, setting: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, model in zip(axes, ("ouro-2.6b", "huginn-3.5b"), strict=True):
        c = data[(setting, model)]
        T = c["T"]
        res = {r["name"]: r["overall"] for r in c["results"]}
        for prefix, color, ls, label in (
            ("std ", SUFFIX, "-", "普通调度：只有后缀树，一步核对一次"),
            ("v3 B ", INK, "--", "loop 调度，不剪枝（v3 B）"),
            ("v3 C ", INK, "-", "loop 调度，读出时剪枝（v3 C）"),
        ):
            pts = sorted(
                (o["rows_per_token"], o["calls_per_token"] / T, k)
                for k, o in res.items()
                if k.startswith(prefix)
            )
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ax.plot(xs, ys, ls, color=color, lw=2, label=label, zorder=2)
            ax.scatter(xs, ys, s=28, color=color, edgecolor="white", lw=1.5, zorder=3)
            for x, y, k in pts:
                tau = k.split("t")[-1]
                if prefix != "v3 B " and tau in ("0.3", "0.1", "0.01"):
                    ax.annotate(f"τ={tau}", (x, y), textcoords="offset points",
                                xytext=(5, 5 if prefix == "std " else -12), fontsize=8,
                                color=MUTED)  # fmt: skip
        o = res["sd-original K16"]
        ax.scatter([o["rows_per_token"]], [o["calls_per_token"] / T], s=50, facecolor="white",
                   edgecolor=SUFFIX, lw=2, zorder=4, label="SuffixDecoding 原设置")  # fmt: skip
        ax.scatter([T], [1.0], s=40, color=MUTED, zorder=4, label="逐词")
        ax.set_xscale("log", base=2)
        xs_all = [o["rows_per_token"] for k, o in res.items() if k != "legacy K16"] + [T]
        ax.set_xlim(T * 0.85, max(xs_all) * 1.25)
        ax.set_ylim(0, 1.08)
        ticks = [t for t in (4, 8, 16, 32, 64, 128, 256) if T * 0.85 <= t <= max(xs_all) * 1.25]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.set_title(f"{NAMES[model]}（T = {T}）", loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("每词行数（算力，对数轴）", fontsize=9, color=MUTED)
        ax.grid(color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[0].set_ylabel("每词块调用数 ÷ T（逐词为 1）", fontsize=9, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=180, facecolor="white")
    print(f"[report] {path}")


def waste_fig(data: dict, setting: str, tau: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6))
    shades = ["#F6D5D5", "#E9A9A9", "#D27474", BAD]
    for ax, model in zip(axes, ("ouro-2.6b", "huginn-3.5b"), strict=True):
        T, d, deep = DEPTHS[model]
        full = json.loads(Path(data[(setting, model)]["_path"]).read_text())
        res = {r["name"]: r["overall"] for r in full["results"]}
        rows = [
            (f"std t{tau}", "普通调度"),
            (f"v3 B t{tau}", "loop，不剪枝"),
            (f"v3 C t{tau}", "loop，读出时剪枝"),
        ]
        for i, (k, _lab) in enumerate(rows[::-1]):
            o = res[k]
            x = 0.0
            used = o["committed_rows_per_token"] / T
            ax.barh(i, used, color="#E6EAEF", edgecolor=MUTED, lw=1, height=0.55)
            x = used
            for j, (_, v) in enumerate(waste_buckets(o, model).items()):
                ax.barh(i, v / T, left=x, color=shades[j], edgecolor="white", lw=1, height=0.55)
                x += v / T
            note = f"{o['rows_per_token']:.1f} 行/词，白跑 {o['wasted_rows_per_token']:.1f}"
            ax.text(x + 0.02, i, note, va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(3), [r[1] for r in rows[::-1]], fontsize=9, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"{NAMES[model]}（T = {T}，d = {d}，更深读出 {deep}），τ = {tau}",
                     loc="left", fontsize=10, color=INK)  # fmt: skip
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=RULE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
        ax.set_xlim(0, 2.6)
    axes[1].set_xlabel("每词行数 ÷ T（逐词为 1）", fontsize=9, color=MUTED)
    handles = [Patch(facecolor="#E6EAEF", edgecolor=MUTED, label="提交词自己的行（就是它的核对）")]
    for j, lab in enumerate(("白跑：跑到深度 d 内被删", "跑到更深读出内", "跑过更深读出、未满 T",
                             "跑满 T 层")):  # fmt: skip
        handles.append(Patch(facecolor=shades[j], label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=180, facecolor="white")
    print(f"[report] {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--waste-tau", default="0.1")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for path in sorted(Path(args.in_dir).glob("*.json")):
        d = json.loads(path.read_text())
        c = compact(d)
        for r_full, r in zip(d["results"], c["results"], strict=True):
            if r_full["name"] != "sd-original K16":
                r["waste_by_steps_run"] = waste_buckets(r_full["overall"], d["model"])
        (out_dir / f"summary-{d['set']}-{d['model']}.json").write_text(json.dumps(c, indent=1))
        data[(d["set"], d["model"])] = c | {"_path": str(path)}
    for setting, sname in SETS.items():
        for model in NAMES:
            if (setting, model) not in data:
                continue
            c = data[(setting, model)]
            print(f"\n### {sname} · {NAMES[model]}\n")
            print("| 设置 | 每词调用 | 倍数 | 每词行数 | 行数÷T | 白跑行/词 | 读出/词 |")
            print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            print("\n".join(table_rows(c)))
        if all((setting, m) in data for m in NAMES):
            frontier(data, setting, out_dir / f"frontier-{setting}.png")
            waste_fig(data, setting, args.waste_tau, out_dir / f"waste-{setting}.png")
    # equal-compute comparison: loop points with rows/word at most std's best
    print("\n### 同等或更少算力下的调用数\n")
    for (setting, model), c in sorted(data.items()):
        res = {r["name"]: r["overall"] for r in c["results"]}
        best = min(
            (o for k, o in res.items() if k.startswith("std ")), key=lambda o: o["calls_per_token"]
        )
        for k, o in res.items():
            if k.startswith("v3 C ") and o["rows_per_token"] <= best["rows_per_token"] * 1.02:
                c1, r1 = o["calls_per_token"], o["rows_per_token"]
                c0, r0 = best["calls_per_token"], best["rows_per_token"]
                print(
                    f"{setting} {model} {k}: {c1:.3f} calls, {r1:.1f} rows vs std best"
                    f" {c0:.3f} calls, {r0:.1f} rows -> {c0 / c1:.2f}x fewer calls"
                )


if __name__ == "__main__":
    main()
