"""Acquire an assigned exclusive GPU before slow model loading, outside timing.

This module only waits and acquires its own CUDA context. It never signals other
processes or changes GPU settings. The independent scoped guard owns cleanup.
"""
import os
import subprocess
import time

from gpu_allocation import allowed_gpus


def acquire_cuda_context(gpu, uuid, timeout_s=60):
    import torch

    assert gpu in allowed_gpus(), 'GPU released by the user'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    modes = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,compute_mode',
        '--format=csv,noheader'], text=True)
    modes = dict(tuple(part.strip() for part in line.split(',')) for line in modes.splitlines())
    assert modes[uuid] == 'Exclusive_Process', 'Exclusive GPU context required for timed evaluation'
    started = time.time()
    deadline = time.monotonic()+timeout_s
    attempts, busy_errors = 0, 0
    while True:
        assert gpu in allowed_gpus(), 'GPU allocation changed while waiting'
        lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader'], text=True)
        foreign = [int(pid.strip()) for device, pid in (line.split(',') for line in lines.splitlines())
            if device.strip() == uuid and int(pid.strip()) != os.getpid()]
        if not foreign:
            attempts += 1
            try:
                torch.cuda.set_device(0)
                lease = torch.empty(1, device='cuda')
                torch.cuda.synchronize()
                return lease, dict(gpu=gpu, gpu_uuid=uuid, pid=os.getpid(),
                    started_unix=started, acquired_unix=time.time(), attempts=attempts,
                    cuda_busy_errors=busy_errors, timing_scope='startup only; outside benchmark decode')
            except RuntimeError as exc:
                if 'busy or unavailable' not in str(exc):
                    raise
                busy_errors += 1
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Assigned GPU {gpu} remained unavailable before model loading')
        time.sleep(.2)
