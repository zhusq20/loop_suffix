"""Greedy outputs and per-depth next-token readouts of a looped LM (Ouro or Huginn).

For each Spec-Bench prompt: greedy decoding at full depth with the model's own
per-depth KV cache, then one teacher-forced pass over prompt + output that keeps
the top-k next-token predictions after each selected recurrence depth. Huginn
draws its initial latent from a fixed per-position table, so decoding and the
teacher-forced pass start from the same state. A depth-t readout here is the
model truncated to t recurrence steps with its own caches at that depth (Huginn's
coda attends to depth-t coda states). Uses only the Hugging Face model code.
--prompt-format chat wraps each question in the model's own chat template and also
stops at its end-of-turn token (protocol 7b).

Protocol: results/loop-suffix-spec-20261030/protocol.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
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
MODELS = {
    "ouro-2.6b": {
        "repo": "models--ByteDance--Ouro-2.6B",
        "revision": "1ed04250da1a9936042725d302e81c8fa2ab5abd",
        "depths": [1, 2, 3, 4],
        "end_turn": "<|im_end|>",
    },
    "huginn-3.5b": {
        "repo": "models--tomg-group-umd--huginn-0125",
        "revision": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "depths": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32],
        "end_turn": "<|end_turn|>",
    },
}


def load_questions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = (obj.get("turns") or [""])[0].strip()
        if not text:
            continue
        cat = "mt_bench" if obj["category"] in MT_BENCH else obj["category"]
        rows.append(
            {
                "question_id": obj["question_id"],
                "category": cat,
                "text": text,
                "turns": obj["turns"],
            }
        )
    return sorted(rows, key=lambda r: r["question_id"])


def topk(logits: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    probs = torch.softmax(logits.float(), dim=-1)
    p, i = torch.topk(probs, k, dim=-1)
    return i.cpu().numpy().astype(np.int32), p.cpu().numpy().astype(np.float16)


class OuroRunner:
    def __init__(self, model, depth_total: int):
        self.model = model
        self.T = depth_total

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden)

    def _new_cache(self):
        # The released UniversalTransformerCache assigns to key_cache, which is
        # a read-only property in Transformers 4.55. Ouro explicitly supports a
        # growable DynamicCache; its layer_idx = loop * layers + physical_layer
        # keeps all recurrence KV slots separate without modifying model code.
        from transformers import DynamicCache

        cache = DynamicCache()
        # 4.55 calls this append_new_layers; empty slots prevent Ouro from
        # replacing an undersized cache with its incompatible default class.
        cache.append_new_layers(self.T * self.model.config.num_hidden_layers - 1)
        return cache

    @torch.no_grad()
    def generate_with_readouts(self, prompt: torch.Tensor, max_new: int, stop: set[int]):
        """Capture top-1 at each loop on the actual AR cache, including prefill.

        Column i predicts output[i]. Each output-head call keeps the same [1,H]
        shape as generate(); batching depths would change BF16 GEMM numerics.
        The returned hidden states already include Ouro's per-loop final norm.
        """
        if max_new < 1:
            raise ValueError("max_new must be positive")
        tokens, reads = [], []
        cache = self._new_cache()
        step = prompt
        while len(tokens) < max_new:
            out, hs, _ = self.model.model(
                input_ids=step, past_key_values=cache, use_cache=True
            )
            cache = out.past_key_values
            if len(hs) != self.T:
                raise ValueError("unexpected number of loop hidden states")
            # Generate the full-depth reference before observing shallow heads.
            target = int(self._head(hs[-1][:, -1]).argmax(-1))
            reads.append([int(self._head(h[:, -1]).argmax(-1)) for h in hs[:-1]] + [target])
            tokens.append(target)
            if target in stop:
                break
            step = torch.tensor([[target]], device=prompt.device)
        return tokens, np.asarray(reads, dtype=np.int32).T

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, stop: set[int]) -> list[int]:
        out, hs, _ = self.model.model(
            input_ids=prompt, past_key_values=self._new_cache(), use_cache=True
        )
        tokens = [int(self._head(hs[-1][:, -1]).argmax(-1))]
        cache = out.past_key_values
        while len(tokens) < max_new and tokens[-1] not in stop:
            step = torch.tensor([[tokens[-1]]], device=prompt.device)
            out, hs, _ = self.model.model(input_ids=step, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            tokens.append(int(self._head(hs[-1][:, -1]).argmax(-1)))
        return tokens

    @torch.no_grad()
    def readouts(self, seq: torch.Tensor, start: int, depths: list[int], k: int):
        _, hs, _ = self.model.model(input_ids=seq, use_cache=False)
        return [topk(self._head(hs[t - 1][0, start:]), k) for t in depths]


class HuginnRunner:
    def __init__(self, model, depth_total: int, seed: int, max_len: int, device: str):
        self.model = model
        self.T = depth_total
        std = model.config.init_values["std"]
        table = torch.empty(max_len, model.config.n_embd)
        gen = torch.Generator().manual_seed(seed)
        torch.nn.init.trunc_normal_(table, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=gen)
        if model.emb_scale != 1:
            table = table * model.emb_scale
        self.init = table.to(device=device, dtype=next(model.parameters()).dtype)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, stop: set[int]) -> list[int]:
        P = prompt.shape[1]
        out = self.model(
            input_ids=prompt, input_states=self.init[:P][None], num_steps=self.T, use_cache=True
        )
        tokens = [int(out.logits[0, -1].argmax(-1))]
        cache = out.past_key_values
        while len(tokens) < max_new and tokens[-1] not in stop:
            pos = P + len(tokens) - 1
            out = self.model(
                input_ids=torch.tensor([[tokens[-1]]], device=prompt.device),
                input_states=self.init[pos : pos + 1][None],
                num_steps=self.T,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([pos], device=prompt.device),
            )
            cache = out.past_key_values
            tokens.append(int(out.logits[0, -1].argmax(-1)))
        return tokens

    @torch.no_grad()
    def readouts(self, seq: torch.Tensor, start: int, depths: list[int], k: int):
        m = self.model
        L = seq.shape[1]
        freqs = m.freqs_cis[:, :L]
        emb = m.transformer.wte(seq)
        if m.emb_scale != 1:
            emb = emb * m.emb_scale
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
        for block in m.transformer.prelude:
            block_idx += 1
            emb = block(emb, freqs, block_idx, None, None)
        x = self.init[:L][None].clone()
        found = {}
        for step in range(max(depths)):
            x, block_idx = m.core_block_forward(x, emb, freqs, None, None, block_idx, step)
            if step + 1 in depths:
                found[step + 1] = topk(m.predict_from_latents(x).logits[0, start:], k)
        return [found[t] for t in depths]


def init_wandb(args, n_rows: int):
    defaults = json.loads(Path(args.tracking_defaults).read_text())
    os.environ.setdefault("WANDB_MODE", "offline")
    import wandb

    return wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        name=f"loop-suffix-spec-20261030-{args.model}"
        f"{'-chat' if args.prompt_format == 'chat' else ''}"
        f"{'-turn2' if args.turn2_from else ''}{'-' + args.run_tag if args.run_tag else ''}"
        f"-shard{args.shard}",
        group="loop-suffix-spec-20261030",
        job_type="inference-readouts",
        config=vars(args) | {"n_rows": n_rows, "training_updates": 0},
        dir=args.out_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--hf-root", default="/hf/hub")
    p.add_argument("--questions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--depth-total", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit-per-category", type=int, default=0, help="pilot: first n per category")
    p.add_argument("--claim", action="store_true", help="take unclaimed questions from all rows")
    p.add_argument(
        "--turn2-from",
        default=None,
        help="MT-bench turn 2: prompt = turn-1 prompt + this run's turn-1 output + turn-2 text",
    )
    p.add_argument(
        "--prompt-format",
        choices=["raw", "chat"],
        default="raw",
        help="chat: the model's chat template, and its end-of-turn token also stops decoding",
    )
    p.add_argument("--run-tag", default="", help="extra W&B run-name suffix for a setting")
    p.add_argument("--tracking-defaults", default="/repo/configs/experiment-tracking-defaults.json")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = MODELS[args.model]
    path = Path(args.hf_root) / spec["repo"] / "snapshots" / spec["revision"]
    depths = [t for t in spec["depths"] if t <= args.depth_total]
    if args.depth_total not in depths:
        depths.append(args.depth_total)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    if args.model.startswith("ouro"):
        model.model.total_ut_steps = args.depth_total
        runner = OuroRunner(model, args.depth_total)
        eos = tok.eos_token_id if tok.eos_token_id is not None else model.config.eos_token_id
    else:
        max_len = args.max_prompt_tokens + args.max_new_tokens + 8
        runner = HuginnRunner(model, args.depth_total, args.seed, max_len, "cuda")
        eos = getattr(model.config, "eos_token_id", None)
    stop = {eos}
    if args.prompt_format == "chat":
        stop.add(tok.convert_tokens_to_ids(spec["end_turn"]))

    rows = load_questions(Path(args.questions))
    if args.turn2_from:
        rows = [r for r in rows if len(r["turns"]) >= 2 and r["turns"][1].strip()]

    def chat_ids(messages: list[dict]) -> list[int]:
        return list(tok.apply_chat_template(messages, add_generation_prompt=True))

    def first_turn(text: str) -> str:
        # cut the question, not the template, so the prompt still ends with the assistant header
        budget = args.max_prompt_tokens - len(chat_ids([{"role": "user", "content": ""}]))
        ids = tok(text, add_special_tokens=False).input_ids
        return text if len(ids) <= budget else tok.decode(ids[:budget])

    def prompt_ids(row) -> torch.Tensor | None:
        if not args.turn2_from:
            if args.prompt_format == "chat":
                ids = chat_ids([{"role": "user", "content": first_turn(row["text"])}])
                return torch.tensor([ids], device="cuda")
            ids = tok(row["text"], return_tensors="pt").input_ids[:, : args.max_prompt_tokens]
            return ids.to("cuda")
        first = Path(args.turn2_from) / "readouts" / f"{row['question_id']}.npz"
        if not first.exists():
            return None
        z = np.load(first)
        answer = [int(t) for t in z["output"]]
        if answer and answer[-1] in stop:
            answer = answer[:-1]
        if args.prompt_format == "chat":
            ids = chat_ids(
                [
                    {"role": "user", "content": first_turn(row["text"])},
                    {"role": "assistant", "content": tok.decode(answer)},
                    {"role": "user", "content": row["turns"][1].strip()},
                ]
            )
        else:
            follow = tok("\n\n" + row["turns"][1].strip(), add_special_tokens=False).input_ids
            ids = [int(t) for t in z["prompt"]] + answer + list(follow)
        return torch.tensor([ids[-args.max_prompt_tokens :]], device="cuda")

    if args.limit_per_category:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            if seen.get(r["category"], 0) < args.limit_per_category:
                kept.append(r)
                seen[r["category"]] = seen.get(r["category"], 0) + 1
        rows = kept
    if not args.claim:
        rows = rows[args.shard :: args.num_shards]
    out_dir = Path(args.out_dir)
    (out_dir / "readouts").mkdir(parents=True, exist_ok=True)
    run = init_wandb(args, len(rows))
    index_path = out_dir / f"index-shard{args.shard}.jsonl"
    done = set()
    if index_path.exists():
        done = {json.loads(x)["question_id"] for x in index_path.read_text().splitlines() if x}
    with index_path.open("a") as fh:
        for i, row in enumerate(rows):
            if row["question_id"] in done:
                continue
            ids = prompt_ids(row)
            if ids is None:
                print(f"[readouts] skip {row['question_id']}: no turn-1 output", flush=True)
                continue
            if args.claim:
                # workers on several GPUs share one queue: skip finished or claimed questions
                if (out_dir / "readouts" / f"{row['question_id']}.npz").exists():
                    continue
                try:
                    os.close(
                        os.open(
                            out_dir / "readouts" / f"{row['question_id']}.claim",
                            os.O_CREAT | os.O_EXCL,
                        )
                    )
                except FileExistsError:
                    continue
            t0 = time.perf_counter()
            output = runner.generate(ids, args.max_new_tokens, stop)
            t1 = time.perf_counter()
            seq = torch.cat([ids, torch.tensor([output], device="cuda")], dim=1)
            P = ids.shape[1]
            reads = runner.readouts(seq, P - 1, depths, args.topk)
            t2 = time.perf_counter()
            full = reads[depths.index(args.depth_total)][0][: len(output), 0]
            agree = float(np.mean(full == np.asarray(output)))
            np.savez_compressed(
                out_dir / "readouts" / f"{row['question_id']}.npz",
                prompt=ids[0].cpu().numpy().astype(np.int32),
                output=np.asarray(output, dtype=np.int32),
                depths=np.asarray(depths, dtype=np.int32),
                ids=np.stack([r[0] for r in reads]),
                probs=np.stack([r[1] for r in reads]),
            )
            rec = {
                "question_id": row["question_id"],
                "category": row["category"],
                "prompt_len": P,
                "output_len": len(output),
                "ended_eos": output[-1] in stop,
                "stop_token": output[-1] if output[-1] in stop else None,
                "full_depth_teacher_forced_agreement": agree,
                "generate_s": t1 - t0,
                "readout_s": t2 - t1,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            run.log({"row": i} | {k: v for k, v in rec.items() if k != "category"})
            print(f"[readouts] {i + 1}/{len(rows)} {rec}", flush=True)
    (out_dir / f"shard{args.shard}-complete.json").write_text(json.dumps({"n_rows": len(rows)}))
    run.finish()


if __name__ == "__main__":
    main()
