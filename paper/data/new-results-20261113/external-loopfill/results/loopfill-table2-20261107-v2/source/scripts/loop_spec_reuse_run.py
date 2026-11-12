"""Log WaveFront Decoding (WFD) events for the rejected-suffix reuse audit.

Runs inside the LM Docker image with the WFD repository mounted read-only. The
logged scheduler is a copy of ``generate_wavefront_dynamic`` (WFD commit 4345209)
whose commit/rollback logic is unchanged; it only records drafts, rejections and
the in-flight tokens that a rejection discards. A sanity pass first checks that
the copy reproduces the original scheduler token-for-token and counter-for-counter.

Protocol: results/loop-spec-reuse-20261030/protocol.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

MT_BENCH = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}
TOPK = 8


def load_questions(path: Path, index_start: int, index_end: int) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        qid = obj["question_id"]
        if not index_start <= qid <= index_end or not obj.get("turns"):
            continue
        text = obj["turns"][0].strip()
        if not text:
            continue
        cat = "mt_bench" if obj["category"] in MT_BENCH else obj["category"]
        rows.append({"question_id": qid, "category": cat, "text": text})
    return rows


def sanity_subset(rows: list[dict], per_category: int) -> list[dict]:
    seen: dict[str, int] = {}
    out = []
    for row in sorted(rows, key=lambda r: r["question_id"]):
        if seen.get(row["category"], 0) < per_category:
            out.append(row)
            seen[row["category"]] = seen.get(row["category"], 0) + 1
    return out


def inflight(state, position: int) -> list[dict]:
    """In-flight tokens past `position`, one per position (active_c holds draft-point copies)."""
    by_pos: dict[int, object] = {}
    for lst in (state.active_r, state.active_p, state.active_c):
        for t in lst:
            if t.is_alive and t.position > position and t.position not in by_pos:
                by_pos[t.position] = t
    return [{"pos": p, "tok": by_pos[p].token_id, "step": by_pos[p].step} for p in sorted(by_pos)]


@torch.no_grad()
def generate_wavefront_logged(blocks, prompt_ids, max_new_tokens, policy, eos_token_id=None):
    """WFD dynamic scheduler with event logging; control flow identical to the original."""
    from src.decoding.scheduler import GenerationTrace, _finalize
    from src.decoding.state import ActiveToken

    trace = GenerationTrace()
    device = prompt_ids.device
    sampler = policy.make_sampler()
    drafts: list[dict] = []
    rejects: list[dict] = []
    tick = 0

    t0 = time.perf_counter()
    state, last_prompt_logits = blocks.prefill(
        prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids,
        T_total=policy.T_total,
        T_draft=policy.T_draft,
    )
    first_token = sampler(last_prompt_logits, state.prefix_len)
    state.committed.append(first_token)
    state.last_committed_position = state.prefix_len
    trace.tokens.append(first_token)
    if len(state.committed) >= max_new_tokens or (
        eos_token_id is not None and first_token == eos_token_id
    ):
        tokens, trace = _finalize(trace, state, t0, device, warmup_marker=None)
        return tokens, trace, state.prefix_len, drafts, rejects

    state.active_p.append(
        ActiveToken(token_id=first_token, position=state.prefix_len, role="draft")
    )
    trace.n_p_calls += 1
    call_p = True
    warmup_marker = None
    steady_size = policy.steady_state_active_size

    while len(state.committed) < max_new_tokens:
        rejected = False
        stop = False
        logits_per_pos: list = []
        if call_p:
            state, _ = blocks.P_dyn(state)
            call_p = False
        state, call_c = blocks.R_dyn(state)
        trace.n_r_calls += 1
        tick += 1
        n_verify = 0
        if call_c:
            state, n_verify, logits_per_pos = blocks.C_multi_dyn(state)
            trace.n_c_calls += 1
        draft_logits = logits_per_pos[n_verify:]

        if n_verify > 0:
            verifies = sorted(
                (t for t in state.active_c if t.role == "verify"), key=lambda t: t.position
            )
            for i, vtok in enumerate(verifies):
                vtok.is_alive = False
                next_pos = vtok.position + 1
                nxt = verifies[i + 1] if i + 1 < n_verify else state.find_active_prc(next_pos)
                if nxt is None:
                    gold = sampler(logits_per_pos[i], next_pos)
                    accepted = False
                else:
                    accepted, gold = sampler.verify(
                        nxt.token_id, nxt.draft_probs, logits_per_pos[i], next_pos
                    )

                if nxt is None:
                    committed_tok = gold
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    trace.tokens.append(gold)
                    state.active_p.append(
                        ActiveToken(token_id=gold, position=next_pos, role="draft")
                    )
                    trace.n_p_calls += 1
                    call_p = True
                elif accepted:
                    committed_tok = nxt.token_id
                    trace.n_drafts_proposed += 1
                    trace.n_drafts_accepted += 1
                    state.committed.append(nxt.token_id)
                    state.last_committed_position = next_pos
                    trace.tokens.append(nxt.token_id)
                else:
                    committed_tok = gold
                    trace.n_drafts_proposed += 1
                    trace.n_drafts_rejected += 1
                    trace.n_rollback_events += 1
                    rejected = True
                    rejects.append(
                        {
                            "tick": tick,
                            "pos": next_pos,
                            "draft": nxt.token_id,
                            "gold": gold,
                            "suffix": inflight(state, next_pos),
                        }
                    )
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    trace.tokens.append(gold)
                    state.rollback_prc(next_pos)
                    state.active_p.append(
                        ActiveToken(token_id=gold, position=next_pos, role="draft")
                    )
                    trace.n_p_calls += 1
                    call_p = True

                if (eos_token_id is not None and committed_tok == eos_token_id) or len(
                    state.committed
                ) >= max_new_tokens:
                    stop = True
                if stop or rejected or nxt is None:
                    break

        if stop:
            break

        if (not rejected) and draft_logits:
            draft_src = next((t for t in state.active_c if t.role == "draft"), None)
            if draft_src is not None:
                draft_token, draft_q = sampler.draft(draft_logits[0], draft_src.position + 1)
                top = torch.topk(draft_logits[0].float().reshape(-1), TOPK).indices.tolist()
                drafts.append(
                    {
                        "tick": tick,
                        "pos": draft_src.position + 1,
                        "src": draft_src.token_id,
                        "tok": draft_token,
                        "top": top,
                    }
                )
                state.active_p.append(
                    ActiveToken(
                        token_id=draft_token,
                        position=draft_src.position + 1,
                        role="draft",
                        draft_probs=draft_q,
                    )
                )
                trace.n_p_calls += 1
                call_p = True

        state.active_c = []
        if warmup_marker is None and len(state.active_r) >= steady_size:
            warmup_marker = time.perf_counter()

    tokens, trace = _finalize(trace, state, t0, device, warmup_marker)
    return tokens, trace, state.prefix_len, drafts, rejects


def counters(trace) -> dict:
    return {
        "n_r_calls": trace.n_r_calls,
        "n_p_calls": trace.n_p_calls,
        "n_c_calls": trace.n_c_calls,
        "proposed": trace.n_drafts_proposed,
        "accepted": trace.n_drafts_accepted,
        "rejected": trace.n_drafts_rejected,
        "rollbacks": trace.n_rollback_events,
    }


def init_wandb(args, n_prompts: int):
    token_path = Path(args.wandb_token_file)
    if not token_path.exists():
        print("[audit] W&B token file missing; tracking disabled", flush=True)
        return None
    os.environ["WANDB_API_KEY"] = token_path.read_text().strip()
    import wandb

    defaults = json.loads(Path(args.tracking_defaults).read_text())
    return wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        mode=args.wandb_mode or defaults["mode"],
        name=f"loop-spec-reuse-20261030-{args.model}",
        group="loop-spec-reuse-20261030",
        job_type="inference-audit",
        config=vars(args) | {"n_prompts": n_prompts, "training_updates": 0},
        dir=args.out_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["ouro-2.6b", "huginn-3.5b"])
    p.add_argument("--wfd-root", default="/wfd")
    p.add_argument("--questions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--T-total", type=int, required=True)
    p.add_argument("--T-draft", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--index-start", type=int, default=81)
    p.add_argument("--index-end", type=int, default=560)
    p.add_argument("--sanity-per-category", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-token-file", default="/run/secrets/wandb-token")
    p.add_argument("--tracking-defaults", default="/repo/configs/experiment-tracking-defaults.json")
    p.add_argument("--wandb-mode", default=None, help="Override the default mode, e.g. offline")
    args = p.parse_args()

    sys.path.insert(0, args.wfd_root)
    from experiments.exp2_spec_bench.run import build_blocks
    from src.decoding.policy import SsdPolicy
    from src.decoding.scheduler import generate_wavefront_dynamic

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    blocks, wrapper = build_blocks(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
        T_total=args.T_total,
        state_init="random",
        state_seed=args.seed,
    )
    eos = getattr(wrapper, "eos_token_id", None)
    policy = SsdPolicy(T_total=args.T_total, T_draft=args.T_draft, mode="greedy")
    rows = load_questions(Path(args.questions), args.index_start, args.index_end)

    def encode(row):
        ids = wrapper.encode(row["text"])
        return ids[:, : args.max_prompt_tokens]

    run = init_wandb(args, len(rows))

    sanity_path = out_dir / "sanity.json"
    if not sanity_path.exists():
        checks = []
        for row in sanity_subset(rows, args.sanity_per_category):
            ids = encode(row).squeeze(0)
            orig_tokens, orig_trace = generate_wavefront_dynamic(
                blocks, ids, args.max_new_tokens, policy, eos_token_id=eos
            )
            log_tokens, log_trace, _, _, _ = generate_wavefront_logged(
                blocks, ids, args.max_new_tokens, policy, eos_token_id=eos
            )
            ar_tokens, _ = blocks.ar_generate(
                ids, args.max_new_tokens, eos_token_id=eos, sampler=policy.make_sampler()
            )
            n = min(len(ar_tokens), len(log_tokens))
            checks.append(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "logged_equals_original_tokens": orig_tokens == log_tokens,
                    "logged_equals_original_counters": counters(orig_trace) == counters(log_trace),
                    "ar_match": sum(
                        a == b for a, b in zip(ar_tokens[:n], log_tokens[:n], strict=True)
                    ),
                    "ar_compared": n,
                    "ar_len": len(ar_tokens),
                    "wfd_len": len(log_tokens),
                }
            )
            print(f"[sanity] {checks[-1]}", flush=True)
        summary = {
            "all_tokens_equal": all(c["logged_equals_original_tokens"] for c in checks),
            "all_counters_equal": all(c["logged_equals_original_counters"] for c in checks),
            "ar_match_rate": sum(c["ar_match"] for c in checks)
            / max(1, sum(c["ar_compared"] for c in checks)),
            "checks": checks,
        }
        sanity_path.write_text(json.dumps(summary, indent=2))
        if run is not None:
            run.summary.update({f"sanity/{k}": v for k, v in summary.items() if k != "checks"})
        if not (summary["all_tokens_equal"] and summary["all_counters_equal"]):
            raise SystemExit("[sanity] logged scheduler differs from the original; stopping")

    events_path = out_dir / "events.jsonl"
    done = set()
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["question_id"])
    totals = {"tokens": 0, "proposed": 0, "accepted": 0, "rejected": 0, "n_r_calls": 0}
    with events_path.open("a") as fh:
        for i, row in enumerate(rows):
            if row["question_id"] in done:
                continue
            ids = encode(row)
            t0 = time.perf_counter()
            tokens, trace, prefix_len, drafts, rejects = generate_wavefront_logged(
                blocks, ids.squeeze(0), args.max_new_tokens, policy, eos_token_id=eos
            )
            record = {
                "question_id": row["question_id"],
                "category": row["category"],
                "prompt_ids": ids.squeeze(0).tolist(),
                "prefix_len": prefix_len,
                "tokens": tokens,
                "counters": counters(trace),
                "walltime_s": time.perf_counter() - t0,
                "drafts": drafts,
                "rejects": rejects,
            }
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            c = record["counters"]
            for key in ("proposed", "accepted", "rejected", "n_r_calls"):
                totals[key] += c[key]
            totals["tokens"] += len(tokens)
            if run is not None:
                run.log(
                    {
                        "prompt_index": i,
                        "acceptance_rate": c["accepted"] / max(1, c["proposed"]),
                        "rejections_per_100_tokens": 100 * c["rejected"] / max(1, len(tokens)),
                        "r_calls_per_token": c["n_r_calls"] / max(1, len(tokens)),
                        "tokens": len(tokens),
                        "walltime_s": record["walltime_s"],
                        "cum_acceptance_rate": totals["accepted"] / max(1, totals["proposed"]),
                    }
                )
            if (i + 1) % 20 == 0:
                print(f"[audit] {i + 1}/{len(rows)} totals={totals}", flush=True)
    (out_dir / "run-complete.json").write_text(json.dumps({"totals": totals, "n_rows": len(rows)}))
    if run is not None:
        run.summary.update({f"total/{k}": v for k, v in totals.items()})
        run.finish()


if __name__ == "__main__":
    main()
