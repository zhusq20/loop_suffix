"""Released Raven Llama/TinyLlama and OLMo layers with tree-cache orchestration.

Only cache/readout orchestration is shared with Huginn. Construction, RoPE,
initialization, block normalization and coda math are Raven-specific. The pinned
native forward accepts input_states, used by the independent checkpoint oracle.
"""
from __future__ import annotations

import torch

from .loop_tree_decode import HuginnLoop, LoopKV, _rotate_half


INIT_POLICY = ('cpu-fp32-per-position-v1: generator seed=(seed+position) mod 2**63; '
               'randn then trunc_normal(-3std,3std), then embed_scale and dtype cast; '
               'same absolute-position state for all requests/methods; no global RNG consumption')


def initial_states(model, length, seed):
    """Native distribution and draw order, indexed independently of scheduling/size."""
    cfg = model.config
    table = torch.empty(length, cfg.n_embd, dtype=torch.float32)
    std = cfg.init_values['std']
    for position in range(length):
        gen = torch.Generator(device='cpu').manual_seed((seed + position) % (2**63))
        row = torch.randn(cfg.n_embd, generator=gen)
        if std > 0:
            torch.nn.init.trunc_normal_(row, std=std, a=-3 * std, b=3 * std, generator=gen)
            row *= model.emb_scale
        else:
            row.zero_()
        table[position] = row
    parameter = next(model.parameters())
    return table.to(device=parameter.device, dtype=parameter.dtype)


class RavenLoop(HuginnLoop):
    """Explicit Raven backend; inherits only the pre/core/post cache orchestration."""

    def __init__(self, model, depth_total, max_seq, max_nodes, seed=0, init_len=None,
                 extra_depths=(), kv_factory=None):
        self.model = model
        cfg = model.config
        if getattr(cfg, 'test_time_noise', 0) != 0 or getattr(cfg, 'qk_bias', False):
            raise ValueError('Released Raven backend requires zero test-time noise and no Q/K bias')
        self.position_limit = min(max_seq, cfg.max_position_embeddings)
        if max_seq > cfg.max_position_embeddings:
            raise ValueError('Raven cache capacity exceeds the declared model context')
        self.prelude = list(model.transformer.prelude)
        self.core = list(model.transformer.core_block)
        self.coda = list(model.transformer.coda)
        self.T, self.n_pre, self.n_core = depth_total, len(self.prelude), len(self.core)
        if not 0 < depth_total:
            raise ValueError('positive recurrence count required')
        if len(extra_depths) != len(set(extra_depths)) or any(not 0 < d < depth_total for d in extra_depths):
            raise ValueError('extra coda depths must be distinct intermediate depths')
        self.coda_full = self.n_pre + self.n_core * depth_total
        self.coda_shallow = self.coda_full + len(self.coda)
        self.coda_extra = {d: self.coda_shallow + len(self.coda) * (i + 1)
                           for i, d in enumerate(extra_depths)}
        n_slots = self.coda_shallow + len(self.coda) * (1 + len(self.coda_extra))
        p = next(model.parameters())
        attn = self.core[0].attn
        self.kv = (kv_factory or LoopKV)(n_slots, max_seq, max_nodes, attn.n_kv_heads,
                                       attn.head_dim, p.dtype, p.device)
        self.hidden = torch.zeros(max_nodes, cfg.n_embd, dtype=p.dtype, device=p.device)
        self.static = torch.zeros_like(self.hidden)
        self.init = initial_states(model, max_seq, seed)
        self.initialization_policy = INIT_POLICY
        self.olmo = hasattr(attn, 'q_norm')

    def _positions(self, positions):
        # Calls the checkpoint's Llama3/OLMo/TinyLlama rotary implementation,
        # preserving its scaling and float32 frequency buffers.
        return self.model.rotary_emb(self.hidden[:1][None], positions[None])

    def _block(self, block, x, freqs, attend):
        attn, count = block.attn, x.shape[1]
        h = x if self.olmo else block.norm_1(x)
        q, k, v = attn.Wqkv(h).split(attn.chunks, dim=-1)
        if self.olmo:
            # Released OLMo normalizes the complete Q/K projection, before heads.
            q, k = attn.q_norm(q), attn.k_norm(k)
        q = q.view(1, count, attn.n_head, attn.head_dim)
        k = k.view(1, count, attn.n_kv_heads, attn.head_dim)
        v = v.view(1, count, attn.n_kv_heads, attn.head_dim)
        cos, sin = (f.unsqueeze(2) for f in freqs)
        # OLMo returns FP32 sin/cos and casts the rotated vectors back to the
        # projection dtype; Llama returns sin/cos in the input dtype.
        q, k = ((q * cos + _rotate_half(q) * sin).to(q.dtype),
                (k * cos + _rotate_half(k) * sin).to(k.dtype))
        h = attn.proj(attend(q[0], k[0], v[0]).reshape(1, count, -1))
        if self.olmo:
            x = x + block.norm_1(h)
            return x + block.norm_2(block.mlp(x))
        x = x + h
        return x + block.mlp(block.norm_2(x))

    def _coda(self, h, freqs, attend_for):
        # Native Raven has no pre-coda ln_f; only the final head input is normalized.
        x = h
        for li, block in enumerate(self.coda):
            x = self._block(block, x, freqs, attend_for(li))
        return x
