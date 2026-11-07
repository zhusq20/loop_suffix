"""Where each output word comes from, and what it costs, in the suffix-spec replays.

Reads the replay summaries of loop_suffix_spec_simulate.py (sim/sim-<setting>-<model>.json)
and splits the committed words of three arms (suffix tree only, shallow readout only,
suffix tree + shallow readout; K=16, main d) by where the word came from:

- suffix: the suffix tree had it in the draft tree. It started computing in the same
  block call as the word before it, so it adds no block call.
- fill: the suffix tree missed it and the depth-d readout of the word before it supplied
  it. It started d calls after the word before it.
- full depth: no draft had it. It is read from the full-depth output of the word before
  it, so it started T calls after that word. Each such wait ends one decoding step.

Block calls then follow exactly: calls = T * steps + d * fills. The number of steps is the
number of full-depth waits, one more per question than the full-depth words, because
the last step of a question ends at the end of the output instead of at a missing word.

Counterfactual: if every fill cost T, as filling a gap does without depth-d readouts
(the next full verification), the combined arm would cost T * (steps + fills). The
script reports it next to the suffix-only arm.

Usage: python scripts/loop_suffix_spec_ledger.py --sim-dir results/loop-suffix-spec-20261030/sim
       --out results/loop-suffix-spec-20261030/sim/ledger.json
       --figure results/loop-suffix-spec-20261030/figures/ledger.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {"ouro-2.6b": (4, 1), "huginn-3.5b": (32, 4)}
SETTINGS = {"raw": "raw-256（主比较）", "chat": "chat模板", "sb": "sb1024 chat"}
ARMS = {(True, False): "suffix", (False, True): "shallow", (True, True): "both"}
SUBSETS = ("overall", "no_loop")


def sim_path(sim_dir: Path, setting: str, model: str) -> Path | None:
    for name in (f"sim-{setting}-{model}.json", f"sim-{setting}-{model}-partial477.json"):
        p = sim_dir / name
        if p.exists():
            return p
    return None


def ledger(o: dict, T: int, d: int) -> dict:
    n, S, F, X, C = o["tokens"], o["steps"], o["from_fill"], o["from_suffix"], o["calls"]
    full = n - F - X
    return {
        "tokens": n,
        "steps": S,
        "full_depth_words": full,
        "fill_words": F,
        "suffix_words": X,
        "share_full_depth": full / n,
        "share_fill": F / n,
        "share_suffix": X / n,
        "calls": C,
        "calls_identity": T * S + d * F,
        "identity_gap": (T * S + d * F) / C - 1,
        "calls_if_fill_cost_T": T * (S + F),
        "calls_per_token": C / n,
        "calls_per_token_waits": T * S / n,
        "calls_per_token_fills": d * F / n,
        "speedup_vs_ar": T * n / C,
        "rows_per_token": o["rows_per_token"],
    }


def build(sim_dir: Path) -> dict:
    out: dict = {}
    for setting in SETTINGS:
        for model, (T, d) in MODELS.items():
            path = sim_path(sim_dir, setting, model)
            if path is None:
                continue
            sim = json.loads(path.read_text())
            arms = {}
            for r in sim["results"]:
                a = r["arm"]
                key = (a["use_suffix"], a["use_fill"])
                if key not in ARMS or a["max_draft"] != 16 or a.get("prune_top"):
                    continue
                if a["spec_factor"] != 1.0 or (a["use_fill"] and a["draft_depth"] != d):
                    continue
                arms[ARMS[key]] = {s: ledger(r[s], T, d) for s in SUBSETS if s in r}
            for s in SUBSETS:
                both, suffix = arms["both"][s], arms["suffix"][s]
                both["counterfactual_vs_suffix_only"] = (
                    both["calls_if_fill_cost_T"] / suffix["calls"] - 1
                )
                both["suffix_words_vs_suffix_only"] = (
                    both["suffix_words"] / suffix["suffix_words"] - 1
                )
            out[f"{setting}/{model}"] = {
                "setting": setting,
                "model": model,
                "T": T,
                "d": d,
                "source": str(path),
                "n_requests": sim["n_requests"],
                "arms": arms,
            }
    return out


def figure(data: dict, path: Path, setting: str = "raw") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch, Rectangle

    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.exists():
        font_manager.fontManager.addfont(str(cjk))
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    ink, muted, rule = "#18202B", "#5A6573", "#D8DDE4"
    fill = {"full": "#E6EAEF", "fill": "#FBEBD3", "suffix": "#D3EDF0"}
    edge = {"full": "#5A6573", "fill": "#A35F0F", "suffix": "#0D6F7E"}
    rows = [
        ("suffix", "只用后缀树\n（普通模型的 SuffixDecoding 调度）"),
        ("shallow", "只用浅层读出\n（LoopSpec / WFD 调度）"),
        ("both", "后缀树 + 浅层读出"),
        ("cf", "后缀树 + 浅层读出\n若每次补位都按 T 计价"),
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 6.4), gridspec_kw={"width_ratios": [1, 1.25]}, sharey="row"
    )
    for i, model in enumerate(MODELS):
        rec = data[f"{setting}/{model}"]
        T, d = rec["T"], rec["d"]
        arms = rec["arms"]
        axl, axr = axes[i]
        ys = list(range(len(rows)))[::-1]
        for y, (key, _) in zip(ys, rows, strict=True):
            a = arms["both" if key == "cf" else key]["overall"]
            h = 0.56
            # left: where words come from (shares of committed words)
            if key == "cf":
                axl.text(0.01, y, "词的来源同上一行，只改补位的计价", va="center", fontsize=9,
                         color=muted)  # fmt: skip
            else:
                x = 0.0
                for part, share in (
                    ("full", a["share_full_depth"]),
                    ("fill", a["share_fill"]),
                    ("suffix", a["share_suffix"]),
                ):
                    if share <= 0:
                        continue
                    axl.add_patch(
                        Rectangle((x, y - h / 2), share, h, facecolor=fill[part],
                                  edgecolor=edge[part], linewidth=1.1)
                    )  # fmt: skip
                    if share >= 0.08:
                        axl.text(x + share / 2, y, f"{share:.0%}", ha="center", va="center",
                                 fontsize=9, color=ink)  # fmt: skip
                    x += share
            # right: block calls per word, as a fraction of word-by-word decoding (T per word).
            # Segments follow calls = T * steps + d * fills, scaled to the measured calls.
            waits = a["calls_per_token_waits"] / T
            fills = a["calls_per_token_fills"] / T
            if key == "cf":
                fills = a["fill_words"] / a["tokens"]
                total = waits + fills
            else:
                total = a["calls_per_token"] / T
                scale = total / (waits + fills)
                waits, fills = waits * scale, fills * scale
            axr.add_patch(
                Rectangle((0, y - h / 2), waits, h, facecolor=fill["full"], edgecolor=edge["full"],
                          linewidth=1.1)
            )  # fmt: skip
            if fills > 0:
                axr.add_patch(
                    Rectangle((waits, y - h / 2), fills, h, facecolor=fill["fill"],
                              edgecolor=edge["fill"], linewidth=1.1,
                              linestyle="--" if key == "cf" else "-")
                )  # fmt: skip
            suffix_cpt = arms["suffix"]["overall"]["calls_per_token"]
            label = (
                f"{a['calls_per_token']:.2f} 次/词 · {a['speedup_vs_ar']:.2f}×"
                if key != "cf"
                else f"{total * T:.2f} 次/词（只用后缀树 {suffix_cpt:.2f}）"
            )
            axr.text(total + 0.012, y, label, va="center", fontsize=9, color=ink)
        for ax in (axl, axr):
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_color(rule)
            ax.spines["bottom"].set_color(muted)
            ax.tick_params(colors=muted, labelsize=9)
            ax.tick_params(axis="y", length=0)
            ax.grid(axis="x", color=rule, linewidth=0.6)
            ax.set_axisbelow(True)
        axl.set_yticks(ys, [r[1] for r in rows], fontsize=9, color=ink)
        axl.set_xlim(0, 1)
        axl.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        axr.set_xlim(0, 0.75)
        name = "Ouro-2.6B" if model.startswith("ouro") else "Huginn-0125"
        axl.set_title(f"{name}（T = {T}，d = {d}）· 每个词从哪里来", loc="left", fontsize=10.5,
                      color=ink)  # fmt: skip
        axr.set_title("每词块调用数，以逐词解码为 1", loc="left", fontsize=10.5, color=ink)
    axes[1][0].set_xlabel("占提交词的比例", fontsize=9, color=muted)
    axes[1][1].set_xlabel("每词块调用数 ÷ T", fontsize=9, color=muted)
    handles = [
        Patch(facecolor=fill["full"], edgecolor=edge["full"],
              label="全深度给出：草稿树里没有，等前一个词跑满 T 层（每个计 T 次）"),
        Patch(facecolor=fill["fill"], edgecolor=edge["fill"],
              label="浅层补上：前一个词跑到深度 d 时读出（每个计 d 次）"),
        Patch(facecolor=fill["suffix"], edgecolor=edge["suffix"],
              label="后缀树给出：与前一个词同一次调用开始（计 0 次）"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, facecolor="white")
    print(f"[ledger] figure -> {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--figure", default="")
    args = p.parse_args()
    data = build(Path(args.sim_dir))
    Path(args.out).write_text(json.dumps(data, indent=1))
    head = ["设置", "模型", "臂", "全深度", "浅层补上", "后缀树", "次/词", "倍数", "恒等式误差"]
    print("| " + " | ".join([*head, "补位按T计 vs 只用后缀树"]) + " |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rec in data.values():
        for arm, v in rec["arms"].items():
            o = v["overall"]
            cf = o.get("counterfactual_vs_suffix_only")
            print(
                f"| {rec['setting']} | {rec['model']} | {arm} | {o['share_full_depth']:.1%}"
                f" | {o['share_fill']:.1%} | {o['share_suffix']:.1%} | {o['calls_per_token']:.3f}"
                f" | {o['speedup_vs_ar']:.2f} | {o['identity_gap']:+.2%}"
                f" | {'' if cf is None else f'{cf:+.1%}'} |"
            )
    if args.figure:
        figure(data, Path(args.figure))


if __name__ == "__main__":
    main()
