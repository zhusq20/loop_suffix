"""LAD architecture from the paper with explicit locally chosen missing details."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LADConfig:
    target_hidden: int
    width: int = 512
    layers: int = 2
    heads: int = 8
    intermediate: int = 1536
    context: int = 64
    train_future: int = 7
    norm_eps: float = 1e-6
    init_std: float = .02

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class CausalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.width
        self.heads = config.heads
        self.norm1 = RMSNorm(w, config.norm_eps)
        self.norm2 = RMSNorm(w, config.norm_eps)
        self.qkv = nn.Linear(w, 3 * w, bias=False)
        self.attn_out = nn.Linear(w, w, bias=False)
        self.gate = nn.Linear(w, config.intermediate, bias=False)
        self.up = nn.Linear(w, config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, w, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads).unbind(2)
        attention = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                   v.transpose(1, 2), is_causal=True, dropout_p=0.)
        x = x + self.attn_out(attention.transpose(1, 2).reshape(batch, length, width))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class LookaheadDrafter(nn.Module):
    def __init__(self, config: LADConfig):
        super().__init__()
        if config.width % config.heads:
            raise ValueError('Width must be divisible by attention heads')
        self.config = config
        h, w = config.target_hidden, config.width
        # Normalization placement, bias=False and learned absolute positions are reproduction choices.
        self.input_norm = RMSNorm(h, config.norm_eps)
        self.context_proj = nn.Linear(h, w, bias=False)
        self.anchor_proj = nn.Linear(h, w, bias=False)
        self.query_state_proj = nn.Linear(h, w, bias=False)
        self.context_position = nn.Parameter(torch.empty(config.context, w))
        self.future_position = nn.Parameter(torch.empty(config.train_future, w))
        self.layers = nn.ModuleList([CausalLayer(config) for _ in range(config.layers)])
        self.output_norm = RMSNorm(w, config.norm_eps)
        self.output_proj = nn.Linear(w, h, bias=False)
        self.apply(self._init)
        nn.init.normal_(self.context_position, std=config.init_std)
        nn.init.normal_(self.future_position, std=config.init_std)

    def _init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.init_std)

    def forward(self, context, anchor_embedding, future=None):
        count = self.config.train_future if future is None else future
        if context.ndim != 3 or context.shape[1:] != (self.config.context, self.config.target_hidden):
            raise ValueError('LAD requires exactly64 actual shallow target states; do not fabricate padding')
        if anchor_embedding.shape != context[:, -1].shape or not 1 <= count <= self.config.train_future:
            raise ValueError('Invalid anchor embedding or future length')
        normalized = self.input_norm(context)
        past = self.context_proj(normalized) + self.context_position
        query = self.anchor_proj(anchor_embedding) + self.query_state_proj(normalized[:, -1])
        future_queries = query[:, None, :] + self.future_position[:count]
        hidden = torch.cat([past, future_queries], dim=1)
        for layer in self.layers:
            hidden = layer(hidden)
        return context[:, -1:, :] + self.output_proj(self.output_norm(hidden[:, -count:]))

    @torch.inference_mode()
    def propose(self, context, anchor_embedding, vocab, future):
        hidden = self(context, anchor_embedding, future)
        logits = F.linear(hidden, vocab.weight, vocab.bias).float()
        indices = logits.argmax(-1)
        tokens = vocab.ids[indices]
        confidence = logits.softmax(-1).gather(-1, indices[..., None])[..., 0]
        return tokens, confidence


def frequent_training_ids(counts, limit=8192):
    return [int(token) for token, _ in sorted(counts.items(), key=lambda x: (-x[1], int(x[0])))[:limit]]


class RequestVocabulary:
    """Frozen head rows for frequent training tokens, prompt, stops and committed tokens."""
    def __init__(self, head, frequent_ids, prompt_ids, stop_ids):
        self.head = head
        self.tokens = set(map(int, frequent_ids)) | set(map(int, prompt_ids)) | set(map(int, stop_ids))
        self._refresh()

    def _refresh(self):
        self.ids = torch.tensor(sorted(self.tokens), dtype=torch.long, device=self.head.weight.device)
        if self.ids.numel() == 0 or self.ids.min() < 0 or self.ids.max() >= self.head.weight.shape[0]:
            raise ValueError('Invalid request draft vocabulary')
        self.weight = self.head.weight.detach().index_select(0, self.ids)
        self.bias = None if getattr(self.head, 'bias', None) is None else self.head.bias.detach().index_select(0, self.ids)

    def commit(self, tokens):
        added = set(map(int, tokens)) - self.tokens
        if added:
            self.tokens.update(added)
            self._refresh()
