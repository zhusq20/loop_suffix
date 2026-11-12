"""One loop block call with rows from one request or from many (S3, multi-request part).

The same total number of rows R runs through one block call (Ouro: the whole layer
stack; Huginn: adapter + core blocks) in three layouts: all R rows from one request,
R / 8 requests with 8 rows each, and R requests with one row each. Every request has
its own committed prefix of P tokens in the K/V cache. Rows of one request read their
prefix together, rows of different requests each read their own, so long prefixes
should make the many-request layouts slow down at fewer rows. Attention is
FlashAttention varlen over [prefix, own rows] per request; the draft-tree mask is left
out, since it is small next to the prefix.

Protocol: results/loop-spec-prob-tree-20261031/protocol.md (S3)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loop_suffix_spec_readouts import MODELS  # noqa: E402
from loop_suffix_spec_simulate import MAIN  # noqa: E402

from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop  # noqa: E402

ROWS = (1, 4, 16, 64, 256, 1024)
MAX_PREFIX_TOKENS = 1 << 20  # K/V of all requests in one buffer: requests x prefix


class Batch:
    """K/V buffer and varlen metadata for `reqs` requests of `per` rows each."""

    def __init__(self, reqs: int, per: int, prefix: int, heads: int, dim: int, dev, dtype):
        span = prefix + per
        self.K = torch.randn(reqs * span, heads, dim, device=dev, dtype=dtype)
        self.V = torch.randn_like(self.K)
        first = torch.arange(reqs, device=dev)[:, None] * span + prefix
        self.slot = (first + torch.arange(per, device=dev)).reshape(-1)  # each row's own k / v
        self.cu_q = (torch.arange(reqs + 1, device=dev) * per).int()
        self.cu_k = (torch.arange(reqs + 1, device=dev) * span).int()
        self.seqused = torch.full((reqs,), span, device=dev, dtype=torch.int32)
        self.per, self.span = per, span
        # rows of one request sit at most a draft-tree depth past the prefix
        self.pos = (prefix + torch.arange(per, device=dev) % 64).repeat(reqs)

    def attend(self, q, k, v):
        self.K[self.slot] = k
        self.V[self.slot] = v
        out = torch.ops.aten._flash_attention_forward(
            q, self.K, self.V, self.cu_q, self.cu_k, self.per, self.span, 0.0, False, False,
            seqused_k=self.seqused,
        )[0]  # fmt: skip
        return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MAIN))
    p.add_argument("--hf-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="256,1024,3584")  # Huginn RoPE table: 4096
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    spec = MODELS[args.model]
    path = Path(args.hf_root) / spec["repo"] / "snapshots" / spec["revision"]
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    dev, dtype = torch.device("cuda"), torch.bfloat16
    if args.model.startswith("ouro"):
        cfg = model.config
        layers = list(model.model.layers[: cfg.num_hidden_layers])
        heads = cfg.num_key_value_heads
        dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        hidden = cfg.hidden_size

        def block(x, batch):
            cos, sin = model.model.rotary_emb(x, batch.pos[None])
            for layer in layers:
                x = OuroLoop._layer(None, layer, x, cos, sin, batch.attend)
            return model.model.norm(x)
    else:
        cfg = model.config
        heads = cfg.num_attention_heads
        dim = cfg.n_embd // heads
        hidden = cfg.n_embd
        tr = model.transformer
        rope = sys.modules[type(model).__module__].apply_rotary_emb_complex_like
        shim = SimpleNamespace(model=model, rope=rope)

        def block(x, batch):
            freqs = model.freqs_cis[:, batch.pos]
            x = tr.adapter(torch.cat([x, x], dim=-1))
            for b in tr.core_block:
                x = HuginnLoop._block(shim, b, x, freqs, batch.attend)
            return x

    records = []
    with torch.no_grad():
        for P in [int(x) for x in args.prefix.split(",")]:
            for R in ROWS:
                layouts = {"one request": (1, R)}
                if R >= 8:
                    layouts["8 rows per request"] = (R // 8, 8)
                if R > 1:
                    layouts["1 row per request"] = (R, 1)
                for name, (reqs, per) in layouts.items():
                    if reqs * (P + per) > MAX_PREFIX_TOKENS:
                        continue
                    batch = Batch(reqs, per, P, heads, dim, dev, dtype)
                    x = torch.randn(1, R, hidden, device=dev, dtype=dtype) * 0.1
                    for _ in range(args.warmup):
                        block(x, batch)
                    ms = []
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        block(x, batch)
                        torch.cuda.synchronize()
                        ms.append(1000 * (time.perf_counter() - t0))
                    rec = {"prefix": P, "rows": R, "layout": name, "requests": reqs, "per": per}
                    rec |= {"ms_median": float(np.median(ms)), "ms_mean": float(np.mean(ms))}
                    records.append(rec)
                    print(
                        f"[multireq] {args.model} P={P} R={R} {name}: {rec['ms_median']:.2f} ms",
                        flush=True,
                    )
                    del batch
                    torch.cuda.empty_cache()
    meta = {"model": args.model, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    Path(args.out).write_text(json.dumps({"meta": meta, "records": records}, indent=1))


if __name__ == "__main__":
    main()
