"""Wavefront decoding with shallow sibling branches (training-free, lossless).

Step 1 of the loop speculative-decoding plan. The main chain is WaveFront Decoding
(WFD, commit 4345209): a token reaching the draft depth proposes the next token, every
token is verified at full depth. In addition, when the shallow top-1 probability is
below `tau`, the shallow top-2 token starts a sibling lane at the same position. A lane
extends its own top-1 chain in the same batched recurrent calls. If the main draft is
rejected and the full-depth token equals the lane's first token, the lane replaces the
discarded main suffix, so the post-rejection restart is skipped. Every committed token
is still the full-depth argmax, so the output equals greedy AR in exact arithmetic.

Lane tokens read the main KV cache before their fork position and a pooled side buffer
from the fork on. Ticks without lane tokens run WFD's own code paths unchanged. Ticks
with lane tokens run every prelude / recurrence / coda layer as one batched call whose
attention reads each slot's main prefix once and applies a per-row lane mask, so extra
lane rows add no prefix copies and no extra weight passes.

Protocol: results/loop-spec-sibling-20261030/protocol.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_spec_reuse_run import load_questions  # noqa: E402


def spread_subset(rows: list[dict], per_category: int) -> list[dict]:
    """`per_category` prompts per category, evenly spaced by question_id."""
    by_cat: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: r["question_id"]):
        by_cat.setdefault(row["category"], []).append(row)
    out = []
    for cat_rows in by_cat.values():
        n = min(per_category, len(cat_rows))
        out += [cat_rows[(j * len(cat_rows)) // n] for j in range(n)]
    return sorted(out, key=lambda r: r["question_id"])


@dataclass
class Group:
    """Rows of one attention group (same recurrence step) and their key layout."""

    key: int
    main_upto: int  # last main position read by any row (-1: none)
    lanes: list[tuple[int, int]]  # (lane, number of side positions read)


class Layout:
    """Key segments and a boolean row mask for one batched lane-aware attention."""

    def __init__(self, lc: LaneKVCache, rows: list[tuple[int, int, int]], device):
        order: dict[int, list[int]] = {}
        for i, (_, _, key) in enumerate(rows):
            order.setdefault(key, []).append(i)
        self.groups: list[Group] = []
        allowed: list[list[tuple[int, int]]] = [[] for _ in rows]
        base = 0
        for key, idxs in order.items():
            main_upto = -1
            side_len: dict[int, int] = {}
            for i in idxs:
                lane, pos, _ = rows[i]
                if lane == 0:
                    main_upto = max(main_upto, pos)
                else:
                    f = lc.fork[lane]
                    main_upto = max(main_upto, f - 1)
                    side_len[lane] = max(side_len.get(lane, 0), pos - f + 1)
            lanes = sorted(side_len.items())
            self.groups.append(Group(key, main_upto, lanes))
            side_start = {}
            cursor = base + main_upto + 1
            for lane, n in lanes:
                side_start[lane] = cursor
                cursor += n
            for i in idxs:
                lane, pos, _ = rows[i]
                if lane == 0:
                    allowed[i].append((base, base + pos + 1))
                else:
                    f = lc.fork[lane]
                    allowed[i].append((base, base + f))
                    allowed[i].append((side_start[lane], side_start[lane] + pos - f + 1))
            base = cursor
        mask = torch.zeros(len(rows), base, dtype=torch.bool)
        for i, ranges in enumerate(allowed):
            for a, b in ranges:
                mask[i, a:b] = True
        self.n_keys = base
        self.mask = mask.to(device, non_blocking=True)[None, None]


class LaneKVCache:
    """The WFD main StaticKVCache plus a pooled side buffer for sibling lanes."""

    def __init__(self, main, slots: list[int], cap: int, pool: int):
        self.main = main
        self.cap = cap
        self.slot_index = {s: i for i, s in enumerate(slots)}
        self.slots = list(slots)
        self.pool = pool
        self.free = list(range(pool))
        self.idx: dict[int, int] = {}
        self.fork: dict[int, int] = {}
        self.length: dict[int, int] = {}
        self.SK: torch.Tensor | None = None
        self.SV: torch.Tensor | None = None

    def open(self, lane: int, fork_pos: int) -> bool:
        if not self.free:
            return False
        if self.SK is None:
            m = self.main
            shape = (self.pool, len(self.slots), self.cap, m.n_kv_head, m.head_dim)
            self.SK = torch.zeros(shape, dtype=m.dtype, device=m.device)
            self.SV = torch.zeros(shape, dtype=m.dtype, device=m.device)
        self.idx[lane] = self.free.pop()
        self.fork[lane] = fork_pos
        self.length[lane] = 0
        return True

    def close(self, lane: int) -> None:
        if lane in self.idx:
            self.free.append(self.idx.pop(lane))
        self.fork.pop(lane, None)
        self.length.pop(lane, None)

    def note(self, lane: int, pos: int) -> None:
        off = pos - self.fork[lane]
        if not 0 <= off < self.cap:
            raise IndexError(f"lane {lane}: offset {off} outside side buffer {self.cap}")
        self.length[lane] = max(self.length[lane], off + 1)

    def write_rows(self, slot_idx: torch.Tensor, lane_rows, k: torch.Tensor, v: torch.Tensor):
        """k, v: (n, H, D) for lane rows [(lane, pos)]; slot_idx: (n,) indices into slots."""
        li = torch.tensor([self.idx[ln] for ln, _ in lane_rows], device=k.device)
        off = torch.tensor([p - self.fork[ln] for ln, p in lane_rows], device=k.device)
        self.SK[li, slot_idx, off] = k
        self.SV[li, slot_idx, off] = v

    def side(self, lane: int, slot: int, n: int):
        li, si = self.idx[lane], self.slot_index[slot]
        return self.SK[li, si, :n].unsqueeze(0), self.SV[li, si, :n].unsqueeze(0)

    def attend(self, layout: Layout, q: torch.Tensor, slot_of) -> torch.Tensor:
        """q: (1, n, H, D) -> (1, n, H, D) attention over each row's lane-visible keys."""
        ks, vs = [], []
        for g in layout.groups:
            slot = slot_of(g.key)
            if g.main_upto >= 0:
                km, vm = self.main.gather(slot, g.main_upto)
                ks.append(km)
                vs.append(vm)
            for lane, n in g.lanes:
                kside, vside = self.side(lane, slot, n)
                ks.append(kside)
                vs.append(vside)
        k = ks[0] if len(ks) == 1 else torch.cat(ks, dim=1)
        v = vs[0] if len(vs) == 1 else torch.cat(vs, dim=1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=layout.mask
        )
        return out.transpose(1, 2)

    def promote(self, lane: int) -> None:
        """Replace main positions >= fork with the lane's side buffers."""
        f, li, n = self.fork[lane], self.idx[lane], self.length[lane]
        self.main.drop_positions_at_and_after(f)
        for si, slot in enumerate(self.slots):
            k = self.SK[li, si, :n].unsqueeze(0)
            v = self.SV[li, si, :n].unsqueeze(0)
            self.main.write_range(slot, f, k, v)
        self.close(lane)


def _lane_layer(lc, layout, rows, q, k, v, slot_of, layer_slot_idx):
    """Write each row's K/V (main rows to the main cache, lane rows to the pool) and attend."""
    main_rows = [i for i, (lane, _, _) in enumerate(rows) if lane == 0]
    lane_rows = [i for i, (lane, _, _) in enumerate(rows) if lane != 0]
    for i in main_rows:
        _, pos, key = rows[i]
        lc.main.write_range(slot_of(key), pos, k[:, i : i + 1], v[:, i : i + 1])
    if lane_rows:
        sel = torch.tensor(lane_rows, device=k.device)
        lc.write_rows(
            layer_slot_idx,
            [(rows[i][0], rows[i][1]) for i in lane_rows],
            k[0].index_select(0, sel),
            v[0].index_select(0, sel),
        )
    return lc.attend(layout, q, slot_of)


def make_lane_blocks(model_name: str, wrapper, T_total: int, seed: int):
    """Subclass the WFD blocks with batched lane-aware prelude / recurrence / coda."""
    from src.decoding.huginn_blocks import HuginnBlocks
    from src.decoding.ouro_blocks import OuroBlocks

    class OuroLaneBlocks(OuroBlocks):
        def cache_slots(self):
            return list(range(self.T_total * self.n_layers))

        def prelude_rows(self, toks, lc):
            ids = torch.tensor([[t.token_id for t in toks]], dtype=torch.long, device=self.device)
            x = self.inner.embed_tokens(ids)[0]
            for i, t in enumerate(toks):
                t.hidden, t.static_state = x[i], None

        def advance_lanes(self, toks, lc):
            if not any(t.lane for t in toks):
                self._advance_one_ut(toks, lc.main)  # WFD path, bit-identical
                return
            n = len(toks)
            rows = [(t.lane, t.position, t.step) for t in toks]
            x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
            cos, sin = self._rope(x, [t.position for t in toks])
            layout = Layout(lc, rows, self.device)
            lane_steps = [t.step for t in toks if t.lane]
            base = torch.tensor(
                [self._slot(s) * self.n_layers for s in lane_steps], device=self.device
            )
            for layer_idx, layer in enumerate(self.inner.layers):

                def slot_of(step, layer_idx=layer_idx):
                    return self._slot(step) * self.n_layers + layer_idx

                residual = x
                h = layer.input_layernorm(x)
                q, k, v = self._compute_qkv(layer.self_attn, h, cos, sin)
                out = _lane_layer(lc, layout, rows, q, k, v, slot_of, base + layer_idx)
                attn_out = layer.self_attn.o_proj(out.reshape(1, n, x.shape[-1]))
                x = self._layer_mlp(layer, residual + layer.input_layernorm_2(attn_out))
            x = self.inner.norm(x)
            for i, t in enumerate(toks):
                t.hidden = x[0, i]
                t.step += 1

        def coda_rows(self, toks, lc):
            return self._lm_head_logits(toks) if toks else []

    class HuginnLaneBlocks(HuginnBlocks):
        def cache_slots(self):
            core = self.n_prelude + self.T_total * self.n_core
            return list(range(core)) + [-1, -2]

        def _sandwich(self, block, x, freqs, lc, layout, rows, slot_of, slot_idx):
            h = block.norm_1(x)
            q, k, v = self._qkv(block.attn, h, freqs)
            out = _lane_layer(lc, layout, rows, q, k, v, slot_of, slot_idx)
            att = block.attn.proj(out.reshape(1, x.shape[1], x.shape[-1]))
            x = block.norm_2(att + x)
            return block.norm_4(block.mlp(block.norm_3(x)) + x)

        def _freqs(self, positions):
            return self.model.freqs_cis[:, torch.tensor(positions, device=self.device)]

        def _const_slot_idx(self, lc, slot, n_lane):
            return torch.full((n_lane,), lc.slot_index[slot], device=self.device)

        def prelude_rows(self, toks, lc):
            if not any(t.lane for t in toks):
                for t in toks:  # WFD path, bit-identical
                    t.hidden, t.static_state = self._prelude_forward(
                        lc.main, t.token_id, t.position
                    )
                return
            model = self.model
            positions = [t.position for t in toks]
            rows = [(t.lane, t.position, 0) for t in toks]
            layout = Layout(lc, rows, self.device)
            n_lane = sum(1 for t in toks if t.lane)
            ids = torch.tensor([[t.token_id for t in toks]], dtype=torch.long, device=self.device)
            x = model.transformer.wte(ids)
            if model.emb_scale != 1:
                x = x * model.emb_scale
            freqs = self._freqs(positions)
            for i, block in enumerate(model.transformer.prelude):
                x = self._sandwich(
                    block, x, freqs, lc, layout, rows, lambda _k, i=i: i,
                    self._const_slot_idx(lc, i, n_lane),
                )  # fmt: skip
            hidden0 = self._init_state_tokens(positions)
            for i, t in enumerate(toks):
                t.hidden, t.static_state = hidden0[i], x[0, i]

        def advance_lanes(self, toks, lc):
            if not any(t.lane for t in toks):
                self._advance_core(toks, lc.main)  # WFD path, bit-identical
                return
            model = self.model
            rows = [(t.lane, t.position, t.step) for t in toks]
            layout = Layout(lc, rows, self.device)
            x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
            e = torch.stack([t.static_state for t in toks], dim=0).unsqueeze(0)
            freqs = self._freqs([t.position for t in toks])
            x = model.transformer.adapter(torch.cat([x, e], dim=-1))
            lane_steps = [t.step for t in toks if t.lane]
            base = torch.tensor(
                [self.n_prelude + self._slot(s) * self.n_core for s in lane_steps],
                device=self.device,
            )
            for layer_idx, block in enumerate(model.transformer.core_block):

                def slot_of(step, layer_idx=layer_idx):
                    return self.n_prelude + self._slot(step) * self.n_core + layer_idx

                x = self._sandwich(block, x, freqs, lc, layout, rows, slot_of, base + layer_idx)
            for i, t in enumerate(toks):
                t.hidden = x[0, i]
                t.step += 1

        def coda_rows(self, toks, lc):
            if not toks:
                return []
            if not any(t.lane for t in toks):  # WFD path, bit-identical
                return self._coda_logits(SimpleNamespace(kv_cache=lc.main), toks)
            model = self.model
            rows = [(t.lane, t.position, 0) for t in toks]
            layout = Layout(lc, rows, self.device)
            n_lane = sum(1 for t in toks if t.lane)
            x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
            x = model.transformer.ln_f(x)
            freqs = self._freqs([t.position for t in toks])
            for i, block in enumerate(model.transformer.coda):
                sidx = -(i + 1)
                x = self._sandwich(
                    block, x, freqs, lc, layout, rows, lambda _k, s=sidx: s,
                    self._const_slot_idx(lc, sidx, n_lane),
                )  # fmt: skip
            logits = model.lm_head(model.transformer.ln_f(x)).float()
            return [logits[0, i] for i in range(len(toks))]

    cls = OuroLaneBlocks if model_name.startswith("ouro") else HuginnLaneBlocks
    blocks = cls(wrapper.model, T_total=T_total)
    if hasattr(blocks, "state_init"):
        blocks.state_init = "random"
        blocks.state_seed = seed
    return blocks


@dataclass(eq=False)
class Tok:
    token_id: int
    position: int
    lane: int
    step: int = 0
    hidden: torch.Tensor | None = None
    static_state: torch.Tensor | None = None
    fresh: bool = True
    should_exit: bool = False  # read by WFD's recurrence code; adaptive depth stays off
    exit_remaining: float = 1.0
    prev_hidden: torch.Tensor | None = None


@torch.no_grad()
def generate_sibling(blocks, prompt_ids, max_new_tokens, T_total, T_draft, tau, eos=None):
    """Greedy WFD plus shallow top-2 sibling lanes when top-1 probability < tau."""
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    device = prompt_ids.device
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(prompt_ids, T_total=T_total, T_draft=T_draft)
    width = T_total // T_draft
    lc = LaneKVCache(state.kv_cache, blocks.cache_slots(), cap=width + 2, pool=width + 2)
    stats = dict(ticks=0, rows=0, max_rows=0, lane_ticks=0, accepted=0, rejected=0)
    stats.update(promoted=0, siblings=0, sibling_rows=0, pool_full=0)
    first = int(last_logits.reshape(-1).argmax().item())
    committed = [first]
    main = [Tok(first, state.prefix_len, 0)]
    lanes: dict[int, list[Tok]] = {}
    next_lane = 1

    def drop_lanes(from_pos):
        for lane in [ln for ln in lanes if lc.fork[ln] >= from_pos]:
            lc.close(lane)
            lanes.pop(lane)

    if eos is not None and first == eos:
        max_new_tokens = 1
    while len(committed) < max_new_tokens:
        everyone = main + [t for ln in sorted(lanes) for t in lanes[ln]]
        fresh = [t for t in everyone if t.fresh]
        if fresh:
            blocks.prelude_rows(fresh, lc)
            for t in fresh:
                t.step, t.fresh = 0, False
        blocks.advance_lanes(everyone, lc)
        stats["ticks"] += 1
        stats["rows"] += len(everyone)
        stats["sibling_rows"] += len(everyone) - len(main)
        stats["lane_ticks"] += int(len(everyone) > len(main))
        stats["max_rows"] = max(stats["max_rows"], len(everyone))

        verify = [t for t in main if t.step == T_total]
        assert len(verify) <= 1
        c_rows = verify + [t for t in main if t.step == T_draft]
        c_rows += [t for ln in sorted(lanes) for t in lanes[ln] if t.step == T_draft]
        flat_logits = blocks.coda_rows(c_rows, lc)
        logits = {id(t): lg for t, lg in zip(c_rows, flat_logits, strict=True)}
        draft_srcs = [t for t in c_rows if t.step == T_draft]

        stop = False
        if verify:
            vt = verify[0]
            main.remove(vt)
            p = vt.position + 1
            gold = int(logits[id(vt)].reshape(-1).argmax().item())
            committed.append(gold)
            nxt = next((t for t in main if t.position == p), None)
            if nxt is None:
                main.append(Tok(gold, p, 0))
            elif nxt.token_id == gold:
                stats["accepted"] += 1
                for lane in [ln for ln in lanes if lc.fork[ln] == p]:
                    lc.close(lane)
                    lanes.pop(lane)
            else:
                stats["rejected"] += 1
                main = [t for t in main if t.position < p]
                hit = next(
                    (ln for ln in lanes if lc.fork[ln] == p and lanes[ln][0].token_id == gold),
                    None,
                )
                if hit is not None:
                    lc.promote(hit)
                    for t in lanes[hit]:
                        t.lane = 0
                    main += lanes.pop(hit)
                    stats["promoted"] += 1
                    drop_lanes(p)
                else:
                    lc.main.drop_positions_at_and_after(p)
                    drop_lanes(p)
                    main.append(Tok(gold, p, 0))
            if (eos is not None and gold == eos) or len(committed) >= max_new_tokens:
                stop = True
        if stop:
            break

        for src in draft_srcs:
            alive = src in main if src.lane == 0 else src in lanes.get(src.lane, [])
            if not alive:
                continue
            row = logits[id(src)].reshape(-1)
            tok1 = int(row.argmax().item())  # same tie-breaking as WFD's greedy draft
            child = Tok(tok1, src.position + 1, src.lane)
            if src.lane != 0:
                lc.note(src.lane, child.position)
                lanes[src.lane].append(child)
                continue
            main.append(child)
            if tau <= 0:
                continue
            row = row.float()
            if torch.softmax(row, dim=-1)[tok1].item() >= tau:
                continue
            if not lc.open(next_lane, child.position):
                stats["pool_full"] += 1
                continue
            rest = row.clone()
            rest[tok1] = float("-inf")
            lc.note(next_lane, child.position)
            lanes[next_lane] = [Tok(int(rest.argmax().item()), child.position, next_lane)]
            next_lane += 1
            stats["siblings"] += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    stats["walltime_s"] = time.perf_counter() - t0
    return committed[:max_new_tokens], stats


def timed_ar(blocks, ids, max_new_tokens, eos, sampler):
    if ids.is_cuda:
        torch.cuda.synchronize()
    tokens, walltime = blocks.ar_generate(ids, max_new_tokens, eos_token_id=eos, sampler=sampler)
    return tokens, walltime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["ouro-2.6b", "huginn-3.5b"])
    p.add_argument("--mode", required=True, choices=["check", "bench"])
    p.add_argument("--wfd-root", default="/wfd")
    p.add_argument("--questions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--T-total", type=int, required=True)
    p.add_argument("--T-draft", type=int, required=True)
    p.add_argument("--taus", default="0,0.5,0.8,2")
    p.add_argument("--per-category", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-token-file", default="/tmp/wandb-token")
    p.add_argument("--tracking-defaults", default="/repo/configs/experiment-tracking-defaults.json")
    p.add_argument("--wandb-mode", default=None)
    args = p.parse_args()

    sys.path.insert(0, args.wfd_root)
    from experiments.exp2_spec_bench.run import MODEL_REGISTRY
    from src.decoding.policy import SsdPolicy
    from src.decoding.scheduler import generate_wavefront_dynamic

    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    wrapper_cls, model_id, _ = MODEL_REGISTRY[args.model]
    wrapper = wrapper_cls.from_pretrained(model_id, dtype=dtype, device="cuda")
    blocks = make_lane_blocks(args.model, wrapper, args.T_total, args.seed)
    eos = getattr(wrapper, "eos_token_id", None)
    policy = SsdPolicy(T_total=args.T_total, T_draft=args.T_draft, mode="greedy")
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    rows = spread_subset(load_questions(Path(args.questions), 81, 560), args.per_category)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    token_path = Path(args.wandb_token_file)
    if token_path.exists():
        os.environ["WANDB_API_KEY"] = token_path.read_text().strip()
        import wandb

        defaults = json.loads(Path(args.tracking_defaults).read_text())
        run = wandb.init(
            entity=defaults["entity"],
            project=defaults["project"],
            mode=args.wandb_mode or defaults["mode"],
            name=f"loop-spec-sibling-20261030-{args.model}-{args.mode}-{args.dtype}",
            group="loop-spec-sibling-20261030",
            job_type="inference-benchmark",
            config=vars(args) | {"n_prompts": len(rows), "training_updates": 0},
            dir=str(out_dir),
        )

    out_path = out_dir / f"{args.mode}-{args.dtype}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["question_id"] for x in out_path.read_text().splitlines() if x}
    with out_path.open("a") as fh:
        for i, row in enumerate(rows):
            if row["question_id"] in done:
                continue
            ids = wrapper.encode(row["text"])[:, : args.max_prompt_tokens].squeeze(0)
            mnt = args.max_new_tokens
            # warm-up as in WFD's benchmark: short untimed runs of every decoder
            timed_ar(blocks, ids, 4, eos, policy.make_sampler())
            generate_wavefront_dynamic(blocks, ids, 4, policy, eos_token_id=eos)
            for tau in taus:
                generate_sibling(blocks, ids, 4, args.T_total, args.T_draft, tau, eos)

            ar_tokens, ar_time = timed_ar(blocks, ids, mnt, eos, policy.make_sampler())
            wfd_tokens, wfd_trace = generate_wavefront_dynamic(
                blocks, ids, mnt, policy, eos_token_id=eos
            )
            rec = {
                "question_id": row["question_id"],
                "category": row["category"],
                "ar": {"tokens": ar_tokens, "walltime_s": ar_time},
                "wfd": {
                    "tokens": wfd_tokens,
                    "walltime_s": wfd_trace.walltime_s,
                    "ticks": wfd_trace.n_r_calls,
                    "accepted": wfd_trace.n_drafts_accepted,
                    "rejected": wfd_trace.n_drafts_rejected,
                },
                "sibling": {},
            }
            for tau in taus:
                toks, st = generate_sibling(blocks, ids, mnt, args.T_total, args.T_draft, tau, eos)
                rec["sibling"][str(tau)] = {"tokens": toks, **st}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if args.mode == "check" and any(
                st["tokens"] != ar_tokens for st in rec["sibling"].values()
            ):
                raise SystemExit(f"[check] q{row['question_id']}: sibling output differs from AR")

            log = {"prompt_index": i}
            ar_tps = len(ar_tokens) / ar_time
            log["wfd_speedup"] = len(wfd_tokens) / wfd_trace.walltime_s / ar_tps
            for tau, st in rec["sibling"].items():
                log[f"tau{tau}/speedup"] = len(st["tokens"]) / st["walltime_s"] / ar_tps
                log[f"tau{tau}/ticks_per_token"] = st["ticks"] / max(1, len(st["tokens"]))
                log[f"tau{tau}/same_as_wfd"] = float(st["tokens"] == wfd_tokens)
                log[f"tau{tau}/same_as_ar"] = float(st["tokens"] == ar_tokens)
            print(f"[sibling] {i + 1}/{len(rows)} q{row['question_id']} {log}", flush=True)
            if run is not None:
                run.log(log)
    (out_dir / f"{args.mode}-{args.dtype}-complete.json").write_text(json.dumps({"n": len(rows)}))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
