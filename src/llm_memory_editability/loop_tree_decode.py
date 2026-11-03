"""Tree speculative decoding of a looped LM with suffix-tree drafts and shallow readouts.

The real decoder behind loop_suffix_spec.replay_request. One step grows a draft tree
under the pending token (the last committed token, not yet run through the loop), and
every block call advances all live nodes by one loop step:

1. Suffix-tree drafts from the committed context start at call 0.
2. A node reaching depth d gets a shallow readout. Its top-1 is added as a child if
   absent (a fill) and the suffix tree continues from the fill. With pruning on,
   children outside the shallow top-m are removed.
3. A node reaching depth T gets a full-depth readout. From the pending token on, the
   full-depth top-1 is committed; if it is a child, that child is verified next and its
   siblings are removed, otherwise it is the bonus token and the step ends.

Node creation, fills, pruning and acceptance follow replay_request, so in exact
arithmetic the block-call count equals the offline replay. The replay only models
fills on the greedy path; a real decoder does not know that path. `fill_scope="all"`
fills every live node (same calls as the replay, more rows); `fill_scope="path"` only
fills nodes on the predicted path (from the last accepted node, follow the child equal
to the shallow top-1, else the first live child).

Each node attends to the committed prefix in the main per-slot KV cache and to its
ancestors in a step-local tree buffer. The two parts are merged through their
log-sum-exp: on CUDA the prefix part is one FlashAttention varlen call per layer over
all slots (no prefix copies), the tree part a small masked attention. Committed nodes'
tree K/V are copied into the main cache at the end of the step.

WaveFront Decoding (WFD, arXiv 2609.23033) is the shallow chain without a draft cap
(`max_draft` above the output length). Two options exist only for it and are off by
default, so the other arms run exactly the engine of the first wall-clock batch:

- `shared_coda` (Huginn): WFD's draft and verify readouts share one coda cache, so a
  draft's coda reads the depth-d coda of in-flight ancestors and the full-depth coda
  of committed ones.
- `flush_every`: an uncapped step can grow to the whole output. Once `flush_every`
  finished chain nodes lie above the frontier they are copied into the main cache and
  the tree part starts after them (every surviving node descends from them and has a
  larger index), so the tree buffer stays as small as in a K-capped step.

Protocols: results/loop-suffix-spec-wallclock-20261031/protocol.md,
results/loop-wfd-same-engine-20261031/protocol.md
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from llm_memory_editability.loop_decode_steps import ModelCall, run_single
from llm_memory_editability.loop_suffix_spec import ArmConfig, SuffixCache


@dataclass(eq=False)
class TreeNode:
    idx: int
    token: int
    parent: int  # -1 for the pending token
    depth: int  # draft depth below the pending token
    start: int  # block call after which the node was created
    source: str  # root, suffix or fill
    step: int = 0  # loop steps done
    alive: bool = True
    children: dict[int, int] = field(default_factory=dict)  # token -> node index
    shallow_top: int | None = None
    full_top: int | None = None


class DraftTree:
    """Nodes of one decoding step and their ancestor-or-self matrix."""

    def __init__(self, max_nodes: int, max_draft: int):
        self.max_nodes = max_nodes
        self.max_draft = max_draft
        self.anc = np.zeros((max_nodes, max_nodes), dtype=bool)
        self.nodes: list[TreeNode] = []
        self.fresh: list[TreeNode] = []
        self.base_pos = 0
        self.lo = 0  # nodes below this index are in the main cache (or dead)
        self.cap_skips = 0

    def reset(self, base_pos: int) -> None:
        self.nodes, self.fresh, self.base_pos, self.lo = [], [], base_pos, 0

    def position(self, node: TreeNode) -> int:
        return self.base_pos + node.depth

    def add(self, token: int, parent: TreeNode | None, start: int, source: str):
        i = len(self.nodes)
        if i >= self.max_nodes:
            self.cap_skips += 1
            return None
        depth = 0 if parent is None else parent.depth + 1
        node = TreeNode(i, int(token), -1 if parent is None else parent.idx, depth, start, source)
        if parent is None:
            self.anc[i] = False
        else:
            self.anc[i] = self.anc[parent.idx]
            parent.children[node.token] = i
        self.anc[i, i] = True
        self.nodes.append(node)
        self.fresh.append(node)
        return node

    def attach(self, parent: TreeNode, draft, start: int) -> None:
        """Insert a suffix draft under `parent`, reusing existing children (as _attach)."""
        created: list[TreeNode | None] = []
        for i, (tok, par) in enumerate(zip(draft.token_ids, draft.parents, strict=True)):
            if par >= i:
                raise ValueError("suffix draft parents must precede children")
            base = parent if par < 0 else created[par]
            if base is None or base.depth >= self.max_draft:
                created.append(None)
                continue
            j = base.children.get(int(tok))
            created.append(self.add(tok, base, start, "suffix") if j is None else self.nodes[j])

    def kill(self, node: TreeNode) -> None:
        stack = [node]
        while stack:
            n = stack.pop()
            n.alive = False
            stack.extend(self.nodes[j] for j in n.children.values())

    def path_tokens(self, node: TreeNode) -> list[int]:
        """Tokens from depth 1 down to `node` (the pending token is already in the context)."""
        out = []
        while node.parent >= 0:
            out.append(node.token)
            node = self.nodes[node.parent]
        return out[::-1]

    def predicted_path(self, frontier: TreeNode) -> set[int]:
        on, node = set(), frontier
        while node is not None:
            on.add(node.idx)
            nxt = None
            j = node.children.get(node.shallow_top) if node.shallow_top is not None else None
            if j is not None and self.nodes[j].alive:
                nxt = self.nodes[j]
            else:
                nxt = next(
                    (self.nodes[j] for j in node.children.values() if self.nodes[j].alive), None
                )
            node = nxt
        return on

    def take_fresh(self) -> list[TreeNode]:
        fresh, self.fresh = self.fresh, []
        return fresh


@torch.no_grad()
def tree_decode(backbone, *args, **kwargs):
    """Decode one request using the same state machine as the batch scheduler."""
    return run_single(backbone, tree_decode_steps(backbone, *args, **kwargs))


def tree_decode_steps(
    backbone,
    prompt,
    arm: ArmConfig,
    max_new_tokens: int,
    stop: set[int],
    cache: SuffixCache | None = None,
    req_id=None,
    *,
    max_nodes: int = 256,
    fill_scope: str = "all",
    shared_coda: bool = False,
    flush_every: int | None = None,
) -> tuple[list[int], dict]:
    """Greedy decoding of one request; returns the output and per-request statistics.

    The output starts with the prefill token and ends at a stop token or at
    max_new_tokens, as in loop_suffix_spec_readouts.py. `calls` counts block calls
    after prefill, the quantity replay_request reports. `shared_coda` and
    `flush_every` are the WFD options described in the module docstring.
    """
    if fill_scope not in ("all", "path"):
        raise ValueError(fill_scope)
    if not len(prompt) or max_new_tokens < 1:
        raise ValueError("prompt must be nonempty and max_new_tokens must be positive")
    T, d, K = arm.depth_total, arm.draft_depth, arm.max_draft
    if arm.use_fill and d >= T:
        raise ValueError("draft depth must be below full depth")
    shallow = arm.use_fill or bool(arm.prune_top)
    use_suffix = arm.use_suffix and cache is not None and K > 0
    tree = DraftTree(max_nodes, K)
    stats = dict(tokens=0, steps=0, calls=0, rows=0, max_rows=0, nodes=0, readout_rows=0)
    stats.update(from_suffix=0, from_fill=0, speculate_calls=0, speculate_s=0.0)
    ctx_limit = getattr(cache, "max_tree_depth", None)

    def speculate(ctx: list[int], room: int):
        t0 = time.perf_counter()
        draft = cache.speculate(
            req_id,
            np.asarray(ctx[-ctx_limit:] if ctx_limit else ctx, dtype=np.int32),
            max_spec_tokens=room,
            max_spec_factor=arm.spec_factor,
            max_spec_offset=0.0,
            min_token_prob=arm.min_token_prob,
            use_tree_spec=arm.tree_spec,
        )
        stats["speculate_s"] += time.perf_counter() - t0
        stats["speculate_calls"] += 1
        return draft

    yield ModelCall("synchronize")
    t0 = time.perf_counter()
    out = [(yield ModelCall("prefill", (
        [int(t) for t in prompt], d if shallow and not shared_coda else None
    )))]
    t1 = time.perf_counter()
    ctx = [int(t) for t in prompt] + out
    if use_suffix:
        cache.start_request(req_id, np.asarray(prompt, dtype=np.int32))
        cache.add_active_response(req_id, np.asarray(out, dtype=np.int32))

    while len(out) < max_new_tokens and out[-1] not in stop:
        tree.reset(backbone.length)
        root = tree.add(out[-1], None, 0, "root")
        if use_suffix:
            tree.attach(root, speculate(ctx, K), 0)
        yield ModelCall("embed", (tree, tree.take_fresh()))
        frontier, chain, flushed, new, tick = root, [root], 0, [], 0
        while True:
            tick += 1
            nodes = tree.nodes[tree.lo :] if tree.lo else tree.nodes
            live = sorted((n for n in nodes if n.alive and n.step < T), key=lambda n: n.step)
            yield ModelCall("advance", (tree, live))
            for n in live:
                n.step += 1
            stats["rows"] += len(live)
            stats["max_rows"] = max(stats["max_rows"], len(live))
            sh = [n for n in live if n.step == d] if shallow else []
            fu = [n for n in live if n.step == T]
            sh_top, fu_top = (yield ModelCall("readout", (tree, sh, fu, arm.prune_top or 1, shared_coda)))
            stats["readout_rows"] += len(sh) + len(fu)
            for n, top in zip(sh, sh_top, strict=True):
                n.shallow_top = top[0]
            for n, top in zip(fu, fu_top, strict=True):
                n.full_top = top[0]

            done = False
            while frontier.full_top is not None:
                gold = frontier.full_top
                new.append(gold)
                j = frontier.children.get(gold)
                if j is not None:
                    stats["from_suffix" if tree.nodes[j].source == "suffix" else "from_fill"] += 1
                if j is None or gold in stop or len(out) + len(new) >= max_new_tokens:
                    done = True
                    break
                for k in frontier.children.values():
                    if k != j:
                        tree.kill(tree.nodes[k])
                frontier = tree.nodes[j]
                chain.append(frontier)
            if done:
                break
            if flush_every and len(chain) - 1 - flushed >= flush_every:
                backbone.commit(tree, chain[flushed:-1])  # finished, all others descend
                flushed = len(chain) - 1
                tree.lo = frontier.idx

            on_path = None
            if arm.use_fill and fill_scope == "path":
                on_path = tree.predicted_path(frontier)
            for n, top in zip(sh, sh_top, strict=True):
                if not n.alive:
                    continue
                fill = arm.use_fill and n.depth < K and top[0] not in n.children
                if fill and (on_path is None or n.idx in on_path):
                    f = tree.add(top[0], n, tick, "fill")
                    if f is not None and use_suffix and f.depth < K:
                        tree.attach(f, speculate(ctx + tree.path_tokens(f), K - f.depth), tick)
                if arm.prune_top:
                    keep = set(top[: arm.prune_top])
                    for tok in [t for t in n.children if t not in keep]:
                        tree.kill(tree.nodes[n.children.pop(tok)])
            fresh = tree.take_fresh()
            if fresh:
                yield ModelCall("embed", (tree, fresh))

        backbone.commit(tree, chain[flushed:])
        out += new
        ctx += new
        if use_suffix:
            cache.add_active_response(req_id, np.asarray(new, dtype=np.int32))
        stats["calls"] += tick
        stats["steps"] += 1
        stats["tokens"] += len(new)
        stats["nodes"] += len(tree.nodes)

    yield ModelCall("synchronize")
    t2 = time.perf_counter()
    if use_suffix:
        cache.stop_request(req_id)
    stats.update(prefill_s=t1 - t0, decode_s=t2 - t1, cap_skips=tree.cap_skips)
    return out, stats


# ---------------------------------------------------------------------------
# KV cache and attention
# ---------------------------------------------------------------------------


class LoopKV:
    """Main K/V of committed positions and tree K/V of the current step, per slot.

    K[slot, pos] holds position `pos`; every committed position is present in every
    slot, so all slots share one length. TK[slot, node] holds draft node `node`.
    """

    def __init__(self, n_slots, max_seq, max_nodes, heads, head_dim, dtype, device, storage=None):
        if storage is None:
            self.K = torch.zeros(n_slots, max_seq, heads, head_dim, dtype=dtype, device=device)
            self.V = torch.zeros_like(self.K)
            self.TK = torch.zeros(n_slots, max_nodes, heads, head_dim, dtype=dtype, device=device)
            self.TV = torch.zeros_like(self.TK)
        else:
            self.K, self.V, self.TK, self.TV = storage
        self.max_seq = max_seq
        self.length = 0
        self.fast = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
        self._stage: dict = {}  # (dtype, turn) -> (pinned buffer, event of its last copy)
        self._turn: dict = {}

    def upload(self, arr) -> torch.Tensor:
        """Host array -> device tensor. A copy from pageable memory can block the host
        until the GPU queue drains, so CUDA uploads go through two alternating pinned
        buffers per dtype; a buffer is reused only after its previous copy finished."""
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if self.K.device.type != "cuda":
            return t
        turn = self._turn[t.dtype] = 1 - self._turn.get(t.dtype, 1)
        buf, event = self._stage.get((t.dtype, turn), (None, None))
        if event is not None:
            event.synchronize()
        if buf is None or buf.numel() < t.numel():
            buf = torch.empty(max(t.numel(), 1 << 14), dtype=t.dtype, pin_memory=True)
        staged = buf[: t.numel()]
        staged.copy_(t.view(-1))
        out = staged.to(self.K.device, non_blocking=True).view(t.shape)
        event = torch.cuda.Event()
        event.record()
        self._stage[(t.dtype, turn)] = (buf, event)
        return out

    def dense(self, slot: int):
        """Causal attention over prompt rows starting at position 0 (prefill)."""

        def attend(q, k, v):
            n = q.shape[0]
            self.K[slot, :n] = k
            self.V[slot, :n] = v
            o = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                is_causal=True,
            )  # fmt: skip
            return o[0].transpose(0, 1)

        return attend

    def commit(self, nodes: list[int]) -> None:
        idx = self.upload(np.asarray(nodes, dtype=np.int64))
        a, b = self.length, self.length + len(nodes)
        if b > self.max_seq:
            raise IndexError(f"position {b - 1} beyond max_seq {self.max_seq}")
        self.K[:, a:b] = self.TK[:, idx]
        self.V[:, a:b] = self.TV[:, idx]
        self.length = b


@dataclass
class Rows:
    """Draft nodes of one block call, grouped by the slot their layer 0 uses.

    Rows are sorted by slot; group g holds rows [a, b). For the tree part, group g is
    padded to max_q rows: row r sits at flat index pad[r] of a (G * max_q) layout.
    """

    nodes: list[TreeNode]
    lo: int  # tree-part keys are nodes lo..N-1
    idx: torch.Tensor  # (R,) node indices
    pos: torch.Tensor  # (R,) absolute positions
    base: torch.Tensor  # (R,) slot of layer 0
    groups: list[tuple[int, int, int]]  # (slot of layer 0, first row, end row)
    gslot: torch.Tensor  # (G,) slot of layer 0
    pad: torch.Tensor  # (R,) flat padded index
    anc: torch.Tensor  # (G, max_q, N - lo) ancestor-or-self; pad rows see node lo only
    cu_q: torch.Tensor  # (G+1,) int32
    cu_k: torch.Tensor  # (G+1,) int32 key offsets of layer 0 in the flattened main cache
    seqused: torch.Tensor  # (G,) int32
    max_q: int


def make_rows(kv: LoopKV, tree: DraftTree, nodes: list[TreeNode], bases: list[int]) -> Rows:
    order = sorted(range(len(nodes)), key=lambda i: bases[i])
    nodes = [nodes[i] for i in order]
    bases = [bases[i] for i in order]
    groups, first = [], 0
    for i in range(1, len(nodes) + 1):
        if i == len(nodes) or bases[i] != bases[first]:
            groups.append((bases[first], first, i))
            first = i
    G = len(groups)
    sizes = [b - a for _, a, b in groups]
    max_q = max(sizes)
    idx = [n.idx for n in nodes]
    pad = np.concatenate([g * max_q + np.arange(n) for g, n in enumerate(sizes)])
    lo = tree.lo
    anc = np.zeros((G * max_q, len(tree.nodes) - lo), dtype=bool)
    anc[:, 0] = True
    anc[pad] = tree.anc[idx, lo : len(tree.nodes)]
    parts = {
        "idx": idx,
        "pos": [tree.position(n) for n in nodes],
        "base": bases,
        "gslot": [g[0] for g in groups],
        "pad": pad,
        "cu_q": np.concatenate([[0], np.cumsum(sizes)]),
        "cu_k": [g[0] * kv.max_seq for g in groups] + [groups[-1][0] * kv.max_seq + kv.length],
        "seqused": [kv.length] * G,
    }
    flat = kv.upload(np.concatenate(list(parts.values())).astype(np.int64))
    t, at = {}, 0
    for name, vals in parts.items():
        t[name] = flat[at : at + len(vals)]
        at += len(vals)
    return Rows(
        nodes, lo, t["idx"], t["pos"], t["base"], groups, t["gslot"], t["pad"],
        kv.upload(anc).view(G, max_q, -1),
        t["cu_q"].int(), t["cu_k"].int(), t["seqused"].int(), max_q,
    )  # fmt: skip


def _prefix_attention(q, kv: LoopKV, rows: Rows, layer: int, fast: bool):
    """Attention of each row over the committed prefix of its slot: (out, lse)."""
    R, H, D = q.shape
    if fast:
        out, lse = torch.ops.aten._flash_attention_forward(
            q,
            kv.K.view(-1, H, D),
            kv.V.view(-1, H, D),
            rows.cu_q,
            rows.cu_k + layer * kv.max_seq,
            rows.max_q,
            kv.length,
            0.0,
            False,
            False,
            seqused_k=rows.seqused,
        )[:2]
        if lse.dim() == 3:  # padded (G, H, max_q) layout of older FlashAttention builds
            lse = torch.cat([lse[g, :, : b - a] for g, (_, a, b) in enumerate(rows.groups)], 1)
        return out.float(), lse.reshape(H, -1)[:, :R].t().float()
    outs, lses = [], []
    scale = D**-0.5
    for base, a, b in rows.groups:
        k = kv.K[base + layer, : kv.length].float()
        v = kv.V[base + layer, : kv.length].float()
        s = torch.einsum("rhd,lhd->rhl", q[a:b].float(), k) * scale
        lse = torch.logsumexp(s, dim=-1)
        outs.append(torch.einsum("rhl,lhd->rhd", torch.exp(s - lse[..., None]), v))
        lses.append(lse)
    return torch.cat(outs), torch.cat(lses)


def tree_attention(q, k, v, kv: LoopKV, rows: Rows, layer: int, fast: bool | None = None):
    """q, k, v: (R, H, D) in row order. Writes k, v to the tree buffer, returns (R, H, D)."""
    fast = kv.fast if fast is None else fast
    R, H, D = q.shape
    kv.TK[rows.base + layer, rows.idx] = k
    kv.TV[rows.base + layer, rows.idx] = v
    G, M, n = rows.anc.shape
    lo = rows.lo
    kt = kv.TK[:, lo : lo + n][rows.gslot + layer].float()  # (G, N - lo, H, D)
    vt = kv.TV[:, lo : lo + n][rows.gslot + layer].float()
    qp = q.new_zeros(G * M, H, D)
    qp[rows.pad] = q
    s = torch.einsum("gihd,gnhd->ghin", qp.view(G, M, H, D).float(), kt) * D**-0.5
    s = s.masked_fill(~rows.anc[:, None], float("-inf"))
    lse = torch.logsumexp(s, dim=-1)  # (G, H, M)
    out = torch.einsum("ghin,gnhd->gihd", torch.exp(s - lse[..., None]), vt)
    out_t = out.reshape(G * M, H, D)[rows.pad]
    lse_t = lse.transpose(1, 2).reshape(G * M, H)[rows.pad]
    if kv.length == 0:
        return out_t.to(q.dtype)
    out_p, lse_p = _prefix_attention(q, kv, rows, layer, fast)
    m = torch.maximum(lse_p, lse_t)
    w_p = torch.exp(lse_p - m)[..., None]
    w_t = torch.exp(lse_t - m)[..., None]
    return ((out_p * w_p + out_t * w_t) / (w_p + w_t)).to(q.dtype)


def _top_probs(logits: torch.Tensor, m: int) -> list[tuple[list[int], list[float]]]:
    """Per row: (top-m ids, their softmax probabilities), argmax first as in _top."""
    if logits.shape[0] == 0:
        return []
    probs = torch.softmax(logits.float(), dim=-1)
    first = logits.argmax(dim=-1)
    p, i = torch.topk(probs, m, dim=-1)
    out = []
    for a, pa, ids, ps in zip(
        first.tolist(), probs.gather(1, first[:, None])[:, 0].tolist(), i.tolist(), p.tolist(),
        strict=True,
    ):  # fmt: skip
        rest = [(t, q) for t, q in zip(ids, ps, strict=True) if t != a][: m - 1]
        out.append(([a] + [t for t, _ in rest], [pa] + [q for _, q in rest]))
    return out


def _split(items: list, sizes: list[int]) -> list[list]:
    out, at = [], 0
    for n in sizes:
        out.append(items[at : at + n])
        at += n
    return out


def _top(logits: torch.Tensor, m: int) -> list[list[int]]:
    """Per row: argmax first (greedy tie-breaking), then the rest of the top-m."""
    if logits.shape[0] == 0:
        return []
    first = logits.argmax(dim=-1)
    if m <= 1:
        return [[t] for t in first.tolist()]
    rest = torch.topk(logits, m, dim=-1).indices
    return [
        [a] + [t for t in r if t != a][: m - 1]
        for a, r in zip(first.tolist(), rest.tolist(), strict=True)
    ]


# ---------------------------------------------------------------------------
# Model backbones (Hugging Face weights, re-implemented forward)
# ---------------------------------------------------------------------------


class _Backbone:
    kv: LoopKV

    @property
    def length(self) -> int:
        return self.kv.length

    def synchronize(self) -> None:
        if self.kv.K.device.type == "cuda":
            torch.cuda.synchronize(self.kv.K.device)

    def commit(self, tree: DraftTree, chain: list[TreeNode]) -> None:
        self.kv.commit([n.idx for n in chain])


class OuroLoop(_Backbone):
    """Ouro: embedding -> [48-layer stack + norm] x T -> lm_head (no coda attention).

    Slot of loop step s, layer l: s * n_layers + l (Ouro's own cache index).
    """

    def __init__(self, causal_lm, depth_total: int, max_seq: int, max_nodes: int, kv_factory=None):
        self.lm = causal_lm
        self.inner = causal_lm.model
        cfg = causal_lm.config
        self.n_layers = cfg.num_hidden_layers
        self.layers = list(self.inner.layers[: self.n_layers])
        self.T = depth_total
        p = next(causal_lm.parameters())
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.kv = (kv_factory or LoopKV)(
            depth_total * self.n_layers, max_seq, max_nodes, cfg.num_key_value_heads, head_dim,
            p.dtype, p.device,
        )  # fmt: skip
        if cfg.num_key_value_heads != cfg.num_attention_heads:
            raise NotImplementedError("tree attention assumes multi-head attention")
        self.hidden = torch.zeros(max_nodes, cfg.hidden_size, dtype=p.dtype, device=p.device)

    def _layer(self, layer, x, cos, sin, attend):
        R = x.shape[1]
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_proj(h).view(1, R, -1, a.head_dim)
        k = a.k_proj(h).view(1, R, -1, a.head_dim)
        v = a.v_proj(h).view(1, R, -1, a.head_dim)
        c, s = cos.unsqueeze(2), sin.unsqueeze(2)
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        o = attend(q[0], k[0], v[0]).reshape(1, R, -1)
        x = x + layer.input_layernorm_2(a.o_proj(o))
        return x + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))

    def prefill(self, prompt: list[int], shallow_depth: int | None) -> int:
        dev = self.kv.K.device
        ids = torch.tensor([prompt], device=dev)
        x = self.inner.embed_tokens(ids)
        cos, sin = self.inner.rotary_emb(x, torch.arange(len(prompt), device=dev)[None])
        for s in range(self.T):
            for li, layer in enumerate(self.layers):
                x = self._layer(layer, x, cos, sin, self.kv.dense(s * self.n_layers + li))
            x = self.inner.norm(x)
        self.kv.length = len(prompt)
        return int(self.lm.lm_head(x[:, -1]).argmax(-1))

    def embed(self, tree: DraftTree, nodes: list[TreeNode]) -> None:
        both = self.kv.upload(np.array([[n.idx for n in nodes], [n.token for n in nodes]]))
        self.hidden[both[0]] = self.inner.embed_tokens(both[1])

    def advance(self, tree: DraftTree, nodes: list[TreeNode]) -> None:
        rows = make_rows(self.kv, tree, nodes, [n.step * self.n_layers for n in nodes])
        x = self.hidden[rows.idx][None]
        cos, sin = self.inner.rotary_emb(x, rows.pos[None])
        for li, layer in enumerate(self.layers):
            x = self._layer(
                layer,
                x,
                cos,
                sin,
                lambda q, k, v, li=li: tree_attention(q, k, v, self.kv, rows, li),
            )
        self.hidden[rows.idx] = self.inner.norm(x)[0]

    def readout(self, tree, shallow, full, m, shared_coda=False):
        nodes = shallow + full
        if not nodes:
            return [], []
        idx = self.kv.upload(np.array([n.idx for n in nodes]))
        top = _top(self.lm.lm_head(self.hidden[idx]).float(), m)
        return top[: len(shallow)], top[len(shallow) :]

    def readout_top(self, tree, groups, full, m):
        """groups: [(depth, nodes)]. Per group: (ids, probs) of the top-m per node; and the
        full-depth argmax per node of `full`. m=0 returns grouped argmax IDs only.
        Ouro reads every depth through lm_head."""
        sizes = [len(g) for _, g in groups]
        nodes = [n for _, g in groups for n in g] + full
        if not nodes:
            return [[] for _ in groups], []
        idx = self.kv.upload(np.array([n.idx for n in nodes]))
        logits = self.lm.lm_head(self.hidden[idx]).float()
        k = sum(sizes)
        reads = logits[:k].argmax(-1).tolist() if m == 0 else _top_probs(logits[:k], m)
        return _split(reads, sizes), logits[k:].argmax(-1).tolist()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class HuginnLoop(_Backbone):
    """Huginn: prelude (2) -> [adapter + core (4)] x T -> coda (2) -> lm_head.

    Slots: prelude l -> l; core step s, layer l -> 2 + 4s + l; full-depth coda ->
    2 + 4T + l; shallow coda -> 2 + 4T + 2 + l. The shallow coda has its own cache
    so a depth-d readout attends to depth-d coda states, as the teacher-forced
    readouts of loop_suffix_spec_readouts.py do. The initial latent is the same fixed
    per-position table as that script.
    """

    def __init__(
        self, model, depth_total, max_seq, max_nodes, seed: int, init_len: int, extra_depths=(),
        kv_factory=None,
    ):
        self.model = model
        tr = model.transformer
        self.prelude, self.core, self.coda = list(tr.prelude), list(tr.core_block), list(tr.coda)
        self.T = depth_total
        self.n_pre, self.n_core = len(self.prelude), len(self.core)
        self.coda_full = self.n_pre + self.n_core * depth_total
        self.coda_shallow = self.coda_full + len(self.coda)
        cfg = model.config
        p = next(model.parameters())
        heads = cfg.num_attention_heads
        if cfg.num_key_value_heads != heads:
            raise NotImplementedError("tree attention assumes multi-head attention")
        # a further shallow coda cache per extra readout depth (probability-threshold arm)
        self.coda_extra = {
            r: self.coda_shallow + len(self.coda) * (j + 1) for j, r in enumerate(extra_depths)
        }
        n_slots = self.coda_shallow + len(self.coda) * (1 + len(self.coda_extra))
        self.kv = (kv_factory or LoopKV)(n_slots, max_seq, max_nodes, heads, cfg.n_embd // heads, p.dtype, p.device)
        if max_seq > model.freqs_cis.shape[1]:
            raise ValueError(f"max_seq {max_seq} beyond the RoPE table {model.freqs_cis.shape[1]}")
        self.rope = sys.modules[type(model).__module__].apply_rotary_emb_complex_like
        std = cfg.init_values["std"]
        # drawn serially on CPU: the first init_len rows equal the readout script's table,
        # the extra rows cover draft positions past the end of the longest output
        table = torch.empty(max(init_len, max_seq), cfg.n_embd)
        gen = torch.Generator().manual_seed(seed)
        torch.nn.init.trunc_normal_(table, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=gen)
        if model.emb_scale != 1:
            table = table * model.emb_scale
        self.init = table.to(device=p.device, dtype=p.dtype)
        self.hidden = torch.zeros(max_nodes, cfg.n_embd, dtype=p.dtype, device=p.device)
        self.static = torch.zeros_like(self.hidden)

    def _block(self, block, x, freqs, attend):
        R = x.shape[1]
        a = block.attn
        q, k, v = a.Wqkv(block.norm_1(x)).split(a.chunks, dim=2)
        q = q.view(1, R, a.n_head, a.head_dim)
        k = k.view(1, R, a.n_kv_heads, a.head_dim)
        v = v.view(1, R, a.n_kv_heads, a.head_dim)
        if self.model.config.qk_bias:
            q_bias, k_bias = a.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = self.rope(q, k, freqs_cis=freqs)
        o = attend(q[0], k[0], v[0]).reshape(1, R, -1)
        x = block.norm_2(a.proj(o) + x)
        return block.norm_4(block.mlp(block.norm_3(x)) + x)

    def _embed_tokens(self, ids):
        x = self.model.transformer.wte(ids)
        return x * self.model.emb_scale if self.model.emb_scale != 1 else x

    def prefill(self, prompt: list[int], shallow_depth: int | None) -> int:
        P = len(prompt)
        dev = self.kv.K.device
        freqs = self.model.freqs_cis[:, :P]
        x = self._embed_tokens(torch.tensor([prompt], device=dev))
        for li, block in enumerate(self.prelude):
            x = self._block(block, x, freqs, self.kv.dense(li))
        h = self.init[:P][None]
        for s in range(self.T):
            h = self.model.transformer.adapter(torch.cat([h, x], dim=-1))
            for li, block in enumerate(self.core):
                h = self._block(block, h, freqs, self.kv.dense(self.n_pre + self.n_core * s + li))
            if shallow_depth == s + 1:
                self._coda(h, freqs, lambda li: self.kv.dense(self.coda_shallow + li))
            if s + 1 in self.coda_extra:
                base = self.coda_extra[s + 1]
                self._coda(h, freqs, lambda li, base=base: self.kv.dense(base + li))
        y = self._coda(h, freqs, lambda li: self.kv.dense(self.coda_full + li))
        self.kv.length = P
        tr = self.model.transformer
        return int(self.model.lm_head(tr.ln_f(y[:, -1])).argmax(-1))

    def _coda(self, h, freqs, attend_for):
        x = self.model.transformer.ln_f(h)
        for li, block in enumerate(self.coda):
            x = self._block(block, x, freqs, attend_for(li))
        return x

    def embed(self, tree: DraftTree, nodes: list[TreeNode]) -> None:
        rows = make_rows(self.kv, tree, nodes, [0] * len(nodes))
        freqs = self.model.freqs_cis[:, rows.pos]
        x = self._embed_tokens(self.kv.upload(np.array([[n.token for n in rows.nodes]])))
        for li, block in enumerate(self.prelude):
            x = self._block(
                block, x, freqs, lambda q, k, v, li=li: tree_attention(q, k, v, self.kv, rows, li)
            )
        self.static[rows.idx] = x[0]
        self.hidden[rows.idx] = self.init[rows.pos]

    def advance(self, tree: DraftTree, nodes: list[TreeNode]) -> None:
        bases = [self.n_pre + self.n_core * n.step for n in nodes]
        rows = make_rows(self.kv, tree, nodes, bases)
        freqs = self.model.freqs_cis[:, rows.pos]
        x = torch.cat([self.hidden[rows.idx], self.static[rows.idx]], dim=-1)[None]
        x = self.model.transformer.adapter(x)
        for li, block in enumerate(self.core):
            x = self._block(
                block, x, freqs, lambda q, k, v, li=li: tree_attention(q, k, v, self.kv, rows, li)
            )
        self.hidden[rows.idx] = x[0]

    def readout(self, tree, shallow, full, m, shared_coda=False):
        """Coda + head. Shallow readouts use their own coda cache, or with shared_coda
        (WFD) the full-depth one: a node's coda K/V there is its depth-d coda until
        its full-depth readout overwrites it."""
        nodes = shallow + full
        if not nodes:
            return [], []
        kind = {n.idx: i < len(shallow) and not shared_coda for i, n in enumerate(nodes)}
        bases = [self.coda_shallow if kind[n.idx] else self.coda_full for n in nodes]
        rows = make_rows(self.kv, tree, nodes, bases)
        freqs = self.model.freqs_cis[:, rows.pos]
        y = self._coda(
            self.hidden[rows.idx][None],
            freqs,
            lambda li: lambda q, k, v: tree_attention(q, k, v, self.kv, rows, li),
        )
        logits = self.model.lm_head(self.model.transformer.ln_f(y[0])).float()
        by_idx = dict(zip((n.idx for n in rows.nodes), _top(logits, m), strict=True))
        return [by_idx[n.idx] for n in shallow], [by_idx[n.idx] for n in full]

    def readout_top(self, tree, groups, full, m):
        """As OuroLoop.readout_top. A depth-r readout attends to depth-r coda states: the
        shallow coda cache for the draft depth, an extra one per deeper readout depth."""
        sizes = [len(g) for _, g in groups]
        nodes = [n for _, g in groups for n in g] + full
        if not nodes:
            return [[] for _ in groups], []
        bases = [self.coda_extra.get(r, self.coda_shallow) for r, g in groups for _ in g]
        bases += [self.coda_full] * len(full)
        rows = make_rows(self.kv, tree, nodes, bases)
        freqs = self.model.freqs_cis[:, rows.pos]
        y = self._coda(
            self.hidden[rows.idx][None],
            freqs,
            lambda li: lambda q, k, v: tree_attention(q, k, v, self.kv, rows, li),
        )
        logits = self.model.lm_head(self.model.transformer.ln_f(y[0])).float()
        at = {n.idx: i for i, n in enumerate(rows.nodes)}
        order = self.kv.upload(np.array([at[n.idx] for n in nodes]))
        logits = logits[order]
        k = sum(sizes)
        reads = logits[:k].argmax(-1).tolist() if m == 0 else _top_probs(logits[:k], m)
        return _split(reads, sizes), logits[k:].argmax(-1).tolist()
