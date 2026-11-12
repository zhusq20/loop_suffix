"""Full-depth HF target and selected features; target checkpoint source is unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import importlib

import torch
from transformers import AutoModelForCausalLM, DynamicCache, Qwen3Config


@dataclass
class TargetOutput:
    logits: torch.Tensor
    features: torch.Tensor | None
    cache: object


class LoopTarget:
    def __init__(self, model, family, max_seq, seed=0):
        if family not in {'ouro', 'huginn'}:
            raise ValueError(f'Unknown loop family: {family}')
        self.model, self.family, self.max_seq = model.eval(), family, max_seq
        self.model.requires_grad_(False)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.capture = False
        self.counts, self.captured, self.hooks = {}, {}, []
        if family == 'ouro':
            self.depth = 4
            model.model.total_ut_steps = self.depth
            model.early_exit_step = model.early_exit_threshold = None
            layers = list(model.model.layers)
            if len(layers) not in (24, 48):
                raise ValueError('Only pinned Ouro architectures are supported')
            self.physical_ids = [int(round(1 + i * (len(layers) - 4) / 4)) for i in range(5)]
            self.expanded_ids = [3 * len(layers) + i for i in self.physical_ids]
            self.capture_at = {i: 4 for i in self.physical_ids}
            self.embedding, self.head = model.model.embed_tokens, model.lm_head
            self.hidden_size = model.config.hidden_size
            self.cache_factory = DynamicCache
            self.noise = None
        else:
            self.depth = 32
            assert len(model.transformer.prelude) == 2
            assert len(model.transformer.core_block) == 4 and len(model.transformer.coda) == 2
            layers = list(model.transformer.prelude) + list(model.transformer.core_block) + list(model.transformer.coda)
            self.physical_ids, self.expanded_ids = [1, 2, 4, 6, 7], [1, 126, 128, 130, 131]
            self.capture_at = {i: 32 if 2 <= i < 6 else 1 for i in self.physical_ids}
            self.embedding, self.head = model.transformer.wte, model.lm_head
            self.hidden_size = model.config.n_embd
            self.cache_factory = importlib.import_module(type(model).__module__).HuginnDynamicCache
            if max_seq > model.freqs_cis.shape[1]:
                raise ValueError('Noise/cache capacity exceeds Huginn context limit')
            gen = torch.Generator(device='cpu').manual_seed(seed)
            table = torch.empty((max_seq, self.hidden_size), dtype=torch.float32)
            std = model.config.init_values['std']
            torch.nn.init.trunc_normal_(table, mean=0., std=std, a=-3 * std, b=3 * std, generator=gen)
            self.noise = (table * model.emb_scale).to(device=self.device, dtype=self.dtype)
        for index in self.physical_ids:
            def hook(module, inputs, output, index=index):
                if self.capture:
                    self.counts[index] = self.counts.get(index, 0) + 1
                    if self.counts[index] == self.capture_at[index]:
                        self.captured[index] = output.detach()
            self.hooks.append(layers[index].register_forward_hook(hook))

    @classmethod
    def load(cls, path, family, max_seq, device='cuda', dtype=torch.bfloat16, seed=0):
        kwargs = dict(trust_remote_code=True, local_files_only=True, torch_dtype=dtype)
        if family == 'ouro':
            kwargs['attn_implementation'] = 'sdpa'
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device)
        # Mirror the existing benchmark's precision enforcement, preserving RoPE buffers.
        buffers = [(m, n, v) for m in model.modules() for n, v in m._buffers.items()]
        model.to(dtype=dtype)
        for module, name, value in buffers:
            module._buffers[name] = value
        assert {p.dtype for p in model.parameters() if p.is_floating_point()} == {dtype}
        return cls(model, family, max_seq, seed)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def new_cache(self):
        return self.cache_factory()

    def cache_lengths(self, cache):
        if self.family == 'huginn':
            return [len(v) for v in cache.key_cache.values()]
        return [int(layer.get_seq_length()) for layer in cache.layers]

    def crop(self, cache, length):
        if not 0 <= length <= cache.get_seq_length():
            raise ValueError('Crop must retain a nonnegative prefix of the existing cache')
        if self.family == 'huginn':
            # Inherited DynamicCache.crop() does not touch Huginn's dictionary storage.
            cache.clear_last_k_entries(cache.get_seq_length() - length)
        else:
            cache.crop(length)

    @torch.no_grad()
    def forward(self, input_ids, cache=None, *, features=True, use_cache=True,
                attention_mask=None, position_ids=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError('Expected nonempty [batch, sequence] token IDs')
        if use_cache and cache is None:
            cache = self.new_cache()
        start = cache.get_seq_length() if cache is not None else 0
        end = start + input_ids.shape[1]
        if end > self.max_seq:
            raise ValueError('Target context capacity exceeded; no silent truncation')
        self.counts.clear()
        self.captured.clear()
        self.capture = features
        positions = torch.arange(start, end, device=self.device)
        kwargs = dict(past_key_values=cache, use_cache=use_cache, cache_position=positions)
        if attention_mask is not None:
            if self.family != 'ouro':
                raise ValueError('Padded Huginn generation is unsupported; use identical-length batches')
            kwargs['attention_mask'] = attention_mask
        if position_ids is not None:
            if self.family != 'ouro' or position_ids.shape != input_ids.shape:
                raise ValueError('Per-example positions require Ouro and must match token shape')
            kwargs['position_ids'] = position_ids
        if self.family == 'huginn':
            kwargs.update(num_steps=32, input_states=self.noise[positions].unsqueeze(0).expand(input_ids.shape[0], -1, -1),
                          output_details=dict(return_logits=True, return_latents=False,
                                              return_head=False, return_stats=False))
        else:
            kwargs.update(use_weighted_exit=False, exit_at_step=None, exit_threshold=None)
        try:
            output = self.model(input_ids, **kwargs)
            if features:
                if self.counts != self.capture_at:
                    raise RuntimeError(f'Incomplete target execution: {self.counts}')
                selected = torch.cat([self.captured[i] for i in self.physical_ids], dim=-1)
            else:
                selected = None
        finally:
            self.capture = False
            self.captured.clear()
        if use_cache and output.past_key_values is not cache:
            raise RuntimeError('Target replaced the cache; rollback identity would be invalid')
        return TargetOutput(output.logits, selected, output.past_key_values)

    def draft_config(self, mask_token_id):
        if not 0 <= mask_token_id < self.model.config.vocab_size:
            raise ValueError('Mask token must use a frozen existing embedding row')
        cfg = self.model.config
        heads = cfg.num_attention_heads
        kwargs = dict(hidden_size=self.hidden_size, intermediate_size=cfg.intermediate_size,
            num_attention_heads=heads, num_key_value_heads=cfg.num_key_value_heads,
            head_dim=self.hidden_size // heads, num_hidden_layers=5,
            num_target_layers=192 if self.family == 'ouro' and len(self.model.model.layers) == 48 else
                              96 if self.family == 'ouro' else 132,
            vocab_size=cfg.vocab_size, max_position_embeddings=self.max_seq,
            rms_norm_eps=getattr(cfg, 'rms_norm_eps', getattr(cfg, 'norm_eps', 1e-6)),
            rope_theta=getattr(cfg, 'rope_theta', getattr(cfg, 'rope_base', 1000000.)),
            attention_bias=False, attention_dropout=0., layer_types=['full_attention'] * 5,
            block_size=16, dflash_config=dict(mask_token_id=mask_token_id, target_layer_ids=self.expanded_ids),
            tie_word_embeddings=False, use_sliding_window=False)
        out = Qwen3Config(**kwargs)
        out._attn_implementation = 'sdpa'
        return out

    @torch.no_grad()
    def shallow_readout_states(self, input_ids, depth):
        """Real LM-head input states at d1, for offline LoopAhead training only.

        Right-padded examples are causal; callers mask labels and retain only
        each real sequence length. The full-depth target setting is restored.
        """
        if not 0 < depth < self.depth or input_ids.ndim != 2 or input_ids.shape[1] > self.max_seq:
            raise ValueError('Invalid shallow feature depth or sequence shape')
        captured = []
        hook = self.head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
        positions = torch.arange(input_ids.shape[1], device=self.device)
        if self.family == 'ouro':
            original = self.model.model.total_ut_steps
            self.model.model.total_ut_steps = depth
            kwargs = dict(use_cache=False, use_weighted_exit=False, exit_at_step=None, exit_threshold=None)
        else:
            kwargs = dict(use_cache=False, num_steps=depth, cache_position=positions,
                input_states=self.noise[positions].unsqueeze(0).expand(input_ids.shape[0], -1, -1),
                output_details=dict(return_logits=True, return_latents=False, return_head=False, return_stats=False))
        try:
            self.model(input_ids, **kwargs)
        finally:
            hook.remove()
            if self.family == 'ouro':
                self.model.model.total_ut_steps = original
        if len(captured) != 1 or captured[0].shape != (*input_ids.shape, self.hidden_size):
            raise RuntimeError('Shallow readout did not expose every actual target state')
        return captured[0]
