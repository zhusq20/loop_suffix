"""CPU-only test of the real inherited lock and durable exit-receipt path."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

OUT = Path(__file__).resolve().parent
ROOT = Path('/ossfs/workspace/loop_suffix')
sys.path.insert(0, str(ROOT/'scripts'))
from loop_queue_recovery import spawn_supervised

with tempfile.TemporaryDirectory(prefix='loopfill-resume-lock-') as folder:
    folder = Path(folder)
    uuid = f'cpu-test-{os.getpid()}'
    path = Path(f'/tmp/loopfill-{uuid}.lock')
    target = folder/'worker.py'
    target.write_text("import fcntl,os,sys\n"
        "p='/tmp/loopfill-'+os.environ['CUDA_VISIBLE_DEVICES']+'.lock'\n"
        "with open(p,'a+') as f:\n"
        " fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        " print('inherited reservation acquired',flush=True)\n"
        "sys.exit(int(sys.argv[1]))\n")
    results = []
    with path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for expected in [0, 7]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, LOOPFILL_RETRY_LOCK_FD=str(lock.fileno()))
            argv = [sys.executable, '-u', str(ROOT/'results/gpu-balanced-queue-20261112-v1/lock_handoff.py'), str(target), str(expected)]
            launch, finish = folder/f'launch-{expected}.json', folder/f'exit-{expected}.json'
            with (folder/f'worker-{expected}.log').open('w') as log:
                child = spawn_supervised(argv, env=env, log=log, lock=lock,
                    launcher_path=launch, exit_path=finish, metadata={'name': f'cpu-test-{expected}'})
                deadline = time.monotonic()+15
                while (code := child.poll()) is None:
                    assert time.monotonic()<deadline
                    time.sleep(.02)
            receipt = json.loads(finish.read_text())
            assert code == receipt['exit_code'] == expected
            assert receipt['exit_evidence'] == 'supervisor_waitpid'
            assert receipt['argv'] == argv
            assert 'inherited reservation acquired' in (folder/f'worker-{expected}.log').read_text()
            with path.open('a+') as competing:
                try:
                    fcntl.flock(competing, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    raise AssertionError('Parent reservation was lost between pipeline steps')
            results.append({'expected_exit': expected, 'actual_exit': code, 'parent_reservation_preserved': True})
    with path.open('a+') as released:
        fcntl.flock(released, fcntl.LOCK_EX|fcntl.LOCK_NB)
    path.unlink()
report = {'status': 'passed', 'cpu_only': True, 'cases': results,
          'reservation_released_after_parent_close': True, 'no_gpu_context_created': True}
(OUT/'handoff-test.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
