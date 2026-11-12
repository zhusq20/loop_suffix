"""Analyse WFD event logs for the rejected-suffix reuse audit (metrics M1-M4).

Protocol: results/loop-spec-reuse-20261030/protocol.md

M1  per-offset agreement of the discarded in-flight suffix with the final output.
M2  implied R-call saving under the cross-check reuse rule.
M3  prompt-lookup hit lengths; M4 recycled-candidate hit lengths (top-1 chain, top-8 tree).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MODELS = {"ouro-2.6b": {"T": 4, "d": 1}, "huginn-3.5b": {"T": 32, "d": 4}}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_drafts_after(drafts: list[dict], tick: int) -> dict[int, int]:
    """Position -> token of the first draft created strictly after `tick`."""
    out: dict[int, int] = {}
    for d in drafts:
        if d["tick"] > tick and d["pos"] not in out:
            out[d["pos"]] = d["tok"]
    return out


def suffix_stats(rec: dict, d: int) -> tuple[list[dict], int]:
    """Per-rejection M1/M2 records and the total saved ticks for one prompt."""
    base = rec["prefix_len"]
    y = rec["tokens"]
    end = base + len(y)  # first absolute position past the generated text
    rejects = rec["rejects"]
    events = []
    saved = 0
    for idx, e in enumerate(rejects):
        q = e["pos"]
        q_next = rejects[idx + 1]["pos"] if idx + 1 < len(rejects) else end
        later = first_drafts_after(rec["drafts"], e["tick"])
        suffix = []
        for s in e["suffix"]:
            if s["pos"] != q + 1 + len(suffix):
                break
            suffix.append(s)
        gaps = len(e["suffix"]) - len(suffix)
        match, survive, k_reuse = [], 0, 0
        prefix_alive, reuse_alive = True, True
        fixes = 0
        for s in suffix:
            pos = s["pos"]
            if pos >= end:
                match.append(None)
                prefix_alive = False
                reuse_alive = False
                continue
            ok = s["tok"] == y[pos - base]
            match.append(ok)
            if prefix_alive and ok:
                survive += 1
            else:
                prefix_alive = False
            wfd = later.get(pos) if pos <= q_next else None
            if reuse_alive and wfd is not None and wfd == s["tok"]:
                k_reuse += 1
            else:
                reuse_alive = False
            if ok and wfd is not None and wfd != s["tok"]:
                fixes += 1
        gain = min(k_reuse, q_next - q - 1)
        saved += d * gain
        events.append(
            {
                "m": len(suffix),
                "match": match,
                "L": survive,
                "K": k_reuse,
                "gain": gain,
                "wasted_rows": len(suffix) - k_reuse,
                "reuse_right_draft_wrong": fixes,
                "noncontiguous": gaps,
            }
        )
    return events, saved


def lookup_lengths(rec: dict, k_max: int) -> list[int]:
    """M3: prompt-lookup match length at every generated position (n-gram 3 -> 1)."""
    seq = rec["prompt_ids"] + rec["tokens"]
    base = rec["prefix_len"]
    last: dict[tuple, int] = {}
    out = []
    for a in range(len(seq)):
        if a >= base and a + 1 < len(seq):
            proposal: list[int] = []
            for n in (3, 2, 1):
                if a - n + 1 < 0:
                    continue
                s = last.get(tuple(seq[a - n + 1 : a + 1]))
                if s is not None:
                    proposal = seq[s + n : min(s + n + k_max, a + 1)]
                    break
            length = 0
            for k, tok in enumerate(proposal, start=1):
                if a + k < len(seq) and seq[a + k] == tok:
                    length += 1
                else:
                    break
            out.append(length)
        for n in (1, 2, 3):
            if a - n + 1 >= 0:
                last[tuple(seq[a - n + 1 : a + 1])] = a - n + 1
    return out


def recycle_lengths(rec: dict, k_max: int) -> tuple[list[int], list[int]]:
    """M4: recycled-candidate match lengths (top-1 chain, top-8 tree) per generated position."""
    base = rec["prefix_len"]
    y = rec["tokens"]
    drafts = sorted(rec["drafts"], key=lambda d: d["pos"])
    table: dict[int, list[int]] = {}
    di = 0
    chain_out, tree_out = [], []
    for j in range(len(y) - 1):
        a = base + j
        while di < len(drafts) and drafts[di]["pos"] <= a:
            table[drafts[di]["src"]] = list(drafts[di]["top"])
            di += 1
        if j >= 1:
            row = table.setdefault(y[j - 1], [])
            if y[j] in row:
                row.remove(y[j])
            row.insert(0, y[j])
        cur, chain = y[j], 0
        for k in range(1, k_max + 1):
            if j + k >= len(y):
                break
            cand = table.get(cur)
            if not cand or cand[0] != y[j + k]:
                break
            chain += 1
            cur = cand[0]
        cur, tree = y[j], 0
        for k in range(1, k_max + 1):
            if j + k >= len(y) or y[j + k] not in table.get(cur, [])[:8]:
                break
            tree += 1
            cur = y[j + k]
        chain_out.append(chain)
        tree_out.append(tree)
    return chain_out, tree_out


def hit_summary(lengths: list[int]) -> dict:
    n = len(lengths)
    return {
        "positions": n,
        "p_ge1": sum(v >= 1 for v in lengths) / n if n else None,
        "p_ge2": sum(v >= 2 for v in lengths) / n if n else None,
        "mean": sum(lengths) / n if n else None,
    }


def analyse(records: list[dict], T: int, d: int) -> dict:
    w = T // d
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["category"]].append(rec)
        groups["all"].append(rec)
    out = {}
    for cat, recs in sorted(groups.items()):
        events, saved, r_calls, tokens = [], 0, 0, 0
        proposed = accepted = 0
        m3, m4c, m4t = [], [], []
        for rec in recs:
            ev, sv = suffix_stats(rec, d)
            events += ev
            saved += sv
            c = rec["counters"]
            r_calls += c["n_r_calls"]
            proposed += c["proposed"]
            accepted += c["accepted"]
            tokens += len(rec["tokens"])
            m3 += lookup_lengths(rec, w - 1)
            ch, tr = recycle_lengths(rec, w - 1)
            m4c += ch
            m4t += tr
        per_offset = []
        for k in range(w - 1):
            vals = [e["match"][k] for e in events if k < len(e["match"])]
            known = [v for v in vals if v is not None]
            cond = [
                e["match"][k]
                for e in events
                if k < len(e["match"]) and e["match"][k] is not None and e["L"] >= k
            ]
            per_offset.append(
                {
                    "offset": k + 1,
                    "available": len(known),
                    "beyond_end": len(vals) - len(known),
                    "agree": sum(known) / len(known) if known else None,
                    "agree_given_prefix": sum(cond) / len(cond) if cond else None,
                }
            )
        n_ev = len(events)
        out[cat] = {
            "prompts": len(recs),
            "tokens": tokens,
            "acceptance_rate": accepted / proposed if proposed else None,
            "rejections": n_ev,
            "rejections_per_100_tokens": 100 * n_ev / tokens if tokens else None,
            "r_calls_per_token": r_calls / tokens if tokens else None,
            "suffix_len_hist": {m: sum(e["m"] == m for e in events) for m in range(w + 1)},
            "m1_per_offset": per_offset,
            "m1_mean_L": sum(e["L"] for e in events) / n_ev if n_ev else None,
            "m1_p_L_ge1": sum(e["L"] >= 1 for e in events) / n_ev if n_ev else None,
            "m2_mean_K": sum(e["K"] for e in events) / n_ev if n_ev else None,
            "m2_saved_ticks": saved,
            "m2_wfd_r_calls": r_calls,
            "m2_saved_fraction": saved / r_calls if r_calls else None,
            "m2_implied_r_call_ratio": 1 - saved / r_calls if r_calls else None,
            "m2_wasted_reused_rows": sum(e["wasted_rows"] for e in events),
            "reuse_right_draft_wrong": sum(e["reuse_right_draft_wrong"] for e in events),
            "noncontiguous_suffix_tokens": sum(e["noncontiguous"] for e in events),
            "m3_lookup": hit_summary(m3),
            "m4_recycle_chain": hit_summary(m4c),
            "m4_recycle_tree": hit_summary(m4t),
        }
    return out


def fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{100 * v:.1f}" if pct else f"{v:.2f}"


def report(summary: dict) -> str:
    lines = ["# Loop投机解码：拒绝后复用旧后缀的零训练统计（结果）", ""]
    lines.append("由 `scripts/loop_spec_reuse_analyze.py` 生成；定义见[protocol.md](protocol.md)。")
    for model, res in summary.items():
        cfg = MODELS[model]
        lines += ["", f"## {model}（T={cfg['T']}, d={cfg['d']}）", ""]
        lines.append(
            "| 类别 | 题数 | 接受率% | 每百词拒绝 | 旧后缀k=1一致% | k=2 | k=3 | 平均L | "
            "隐含R调用节省% | 上文查找≥2% | 回收树≥2% |"
        )
        lines.append("| --- |" + " ---: |" * 10)
        for cat, r in res.items():
            po = r["m1_per_offset"] + [{"agree": None}] * 3
            lines.append(
                f"| {cat} | {r['prompts']} | {fmt(r['acceptance_rate'], True)} | "
                f"{fmt(r['rejections_per_100_tokens'])} | {fmt(po[0]['agree'], True)} | "
                f"{fmt(po[1]['agree'], True)} | {fmt(po[2]['agree'], True)} | "
                f"{fmt(r['m1_mean_L'])} | {fmt(r['m2_saved_fraction'], True)} | "
                f"{fmt(r['m3_lookup']['p_ge2'], True)} | "
                f"{fmt(r['m4_recycle_tree']['p_ge2'], True)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    summary = {}
    for model, cfg in MODELS.items():
        path = Path(args.runs_dir) / model / "events.jsonl"
        if path.exists():
            summary[model] = analyse(load(path), cfg["T"], cfg["d"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
