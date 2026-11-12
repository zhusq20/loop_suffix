"""Start the authorised pilot after all four complete, hashed teacher shards exist."""
import json
import os
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parents[1]
teachers = base / 'data/teachers/dflash-pilot/ouro-2.6b'
while not all((teachers / f'shard{i}/complete.json').exists() for i in range(4)):
    if any((teachers / f'shard{i}/failure.json').exists() for i in range(4)):
        raise RuntimeError('Teacher generation failed; pilot cannot start')
    time.sleep(10)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='GPU-520fad11-804c-b419-7096-2a19c219449b,GPU-2dc90f87-ad67-93a7-d0c5-218a3543318a',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           TORCHINDUCTOR_CACHE_DIR='/tmp/loopfill-training-inductor', TRITON_CACHE_DIR='/tmp/loopfill-training-triton')
argv = ['/tmp/loopfill-specforge-venv/bin/torchrun', '--standalone', '--nproc_per_node=2',
        str(base / 'train_dflash.py'), '--model', 'ouro-2.6b', '--teachers', str(teachers),
        '--out', str(base / 'checkpoints/dflash-ouro-2.6b-pilot-v1'), '--cpu-groups', '4-7,52-55',
        '--micro-batch', '4', '--accumulation', '4', '--epochs', '1', '--save-every', '25']
with (base / 'logs/dflash-training-pilot-v1.log').open('x') as log:
    child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(base / 'logs/dflash-training-pilot-v1-launcher.json').write_text(json.dumps(
    dict(pid=child.pid, argv=argv, started_unix=time.time()), indent=2) + '\n')
print('Started pilot', child.pid, flush=True)
