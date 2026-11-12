"""Preallocated full-depth Huginn cache for teacher production, not benchmark timing.

The checkpoint's dictionary cache supports adaptive missing-loop states. Teacher
generation always executes all32 loops, so every slot has the same contiguous
prefix. Store owned K/V tensors instead of retaining per-position tensor views.
"""
import torch


class HuginnProductionCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('Positive capacity required')
        self.capacity = capacity
        self.key_cache, self.value_cache, self.lengths = {}, {}, {}
        self._seen_tokens = 0
        self.lookup_strategy = 'full'

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self._seen_tokens

    def update(self, key, value, step_idx_tensor, lookup_strategy=None):
        if lookup_strategy not in (None, 'full'):
            raise ValueError('Production cache supports complete fixed-depth states only')
        slot = int(step_idx_tensor)
        assert key.shape == value.shape and key.ndim == 4
        width = key.shape[-2]
        if slot == 0:
            self._seen_tokens += width
        end, start = self._seen_tokens, self._seen_tokens-width
        if not 0 <= start < end <= self.capacity:
            raise ValueError('Cache capacity exceeded or first prelude missing')
        if slot not in self.key_cache:
            shape = (*key.shape[:-2], self.capacity, key.shape[-1])
            self.key_cache[slot] = torch.empty(shape, dtype=key.dtype, device=key.device)
            self.value_cache[slot] = torch.empty_like(self.key_cache[slot])
            self.lengths[slot] = 0
        if self.lengths[slot] != start:
            raise RuntimeError('Missing/repeated recurrence slot would expose invalid KV')
        self.key_cache[slot][..., start:end, :].copy_(key)
        self.value_cache[slot][..., start:end, :].copy_(value)
        self.lengths[slot] = end
        return self.key_cache[slot][..., :end, :], self.value_cache[slot][..., :end, :]

    def clear_last_k_entries(self, k):
        if not 0 <= k <= self._seen_tokens or any(n != self._seen_tokens for n in self.lengths.values()):
            raise ValueError('Crop requires a completed full-depth call')
        self._seen_tokens -= k
        self.lengths = {slot:self._seen_tokens for slot in self.lengths}


