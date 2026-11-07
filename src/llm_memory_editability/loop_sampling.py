"""Target-only inverse-CDF sampling, independent of speculative execution order.

A request has one independent uniform per absolute output position. All candidate
prefixes at that position share the uniform; only the realized prefix is emitted.
Draft construction never consumes these uniforms. Precomputed off-path samples
therefore cannot shift the random stream used by the realized autoregressive path.
"""
import hashlib
import math
import torch


def request_seed(seed, model, dataset, source_id):
    key = f'{seed}/{model}/{dataset}/{source_id}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)


class TargetSampler:
    def __init__(self, temperature, seed, capacity, device):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('sampling temperature must be finite and positive')
        self.temperature = temperature
        rng = torch.Generator(device='cpu').manual_seed(seed)
        # CPU-generated uniforms make GPU assignment irrelevant to request RNG.
        self.uniforms = torch.rand(capacity, generator=rng, dtype=torch.float64).to(device)

    def __call__(self, logits, positions):
        if len(positions) != len(logits):
            raise ValueError('one absolute position required for each logit row')
        if not positions:
            return []
        if min(positions) < 0 or max(positions) >= len(self.uniforms):
            raise ValueError('sampling position outside request capacity')
        # The full target distribution, not a distribution restricted to tree children.
        probabilities = torch.softmax(logits.float() / self.temperature, dim=-1)
        cdf = probabilities.cumsum(dim=-1)
        # Renormalize tiny accumulation error, including the last bin.
        cdf = cdf / cdf[:, -1:]
        u = self.uniforms[torch.tensor(positions, device=logits.device)].to(cdf.dtype)
        u = u.clamp_max(1.0 - torch.finfo(cdf.dtype).eps)
        return torch.searchsorted(cdf.contiguous(), u[:, None].contiguous(), right=True).flatten().tolist()


def target_tokens(backbone, logits, positions):
    sampler = getattr(backbone, 'target_sampler', None)
    return logits.argmax(-1).tolist() if sampler is None else sampler(logits, positions)
