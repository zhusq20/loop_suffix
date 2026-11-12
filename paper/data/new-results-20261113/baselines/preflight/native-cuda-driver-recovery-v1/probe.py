"""GPU 2 startup/kernel probe for the process-local driver-library correction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
os.sched_setaffinity(0, {80, 81, 82, 83})
UUID = 'GPU-861ad2de-b899-21ca-707e-44c0a1ce1029'
os.environ['CUDA_VISIBLE_DEVICES'] = UUID

from gpu_startup import acquire_cuda_context

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert not args.out.exists()
lease, acquisition = acquire_cuda_context(2, UUID)
import torch
import transformers
import flash_attn
from flash_attn import flash_attn_func

x = torch.arange(256, device='cuda', dtype=torch.float32)
assert x.square().sum().item() == 5559680
a = torch.ones((64, 64), device='cuda', dtype=torch.bfloat16)
assert torch.equal(a @ a, torch.full_like(a, 64))
q = torch.full((1, 8, 2, 64), 0.125, device='cuda', dtype=torch.bfloat16)
v = torch.arange(q.numel(), device='cuda', dtype=torch.float32).reshape(q.shape).div(512).bfloat16()
y = flash_attn_func(q, q, v, causal=True)
reference = v.float().cumsum(1) / torch.arange(1, 9, device='cuda').view(1, 8, 1, 1)
assert torch.isfinite(y).all() and torch.allclose(y.float(), reference, atol=0.02, rtol=0.01)
torch.cuda.synchronize()
libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                    if any(name in line for name in ['libcuda.so', 'libnvidia-ptxjitcompiler.so', 'libnvidia-nvvm.so'])})
assert '/usr/lib64/libcuda.so.570.148.08' in libraries
assert not any('/cuda/compat/' in name for name in libraries)
result = dict(status='cuda_context_and_tensor_matmul_flash_attention_passed',
              executable=sys.executable, torch=torch.__version__, torch_cuda=torch.version.cuda,
              transformers=transformers.__version__, flash_attn=flash_attn.__version__,
              acquisition=acquisition, loaded_driver_libraries=libraries,
              flash_attention_max_abs_error=float((y.float() - reference).abs().max()),
              ld_cache_mtime=Path('/etc/ld.so.cache').stat().st_mtime,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              limitation='Startup and small GPU kernels only; not a full-model benchmark or distribution test.')
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
