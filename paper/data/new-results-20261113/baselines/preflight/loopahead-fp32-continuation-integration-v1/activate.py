"""Start the prepared CPU waiter; no completed GPU run is restarted."""
import fcntl
import os
from pathlib import Path
import sys
import time

BASE = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from loopahead_resume_attempt import ATTEMPT, verify_attempt
from packed_dflash_evidence import read, save, sha
sys.path.insert(0, str(BASE / 'logs/ouro14-packed-evaluation-v1/source'))
from loop_queue_recovery import spawn_supervised, matching_process

assert not (OUT / 'activation.json').exists()
assert read(OUT / 'prepare-exit.json')['exit_code'] == read(OUT / 'audit-exit.json')['exit_code'] == 0
audit = read(BASE / 'preflight/numerical-matrix-partial-v24/report.json')
assert audit['completed_formal'] == 29 and audit['completed_additional'] == 3
assert not audit['errors'] and len(audit['pending']) == 7
assert audit['pending']['formal/loopahead/ouro-2.6b/math500']['remaining_cases'] == 167
plan = verify_attempt()
assert read(Path(plan['measurement_queue']) / 'status.json')['ready']
assert not Path(plan['group']['steps'][0]['out']).exists()
assert not (ATTEMPT / 'STOP').exists()
before = sha(BASE / 'combined-results/results.csv')
with (ATTEMPT / 'waiter.lock').open('a+') as lease:
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    argv = ['taskset', '--cpu-list', '80-83', '/opt/conda/bin/python', '-u', str(BASE / 'queue_loopahead_math_continuation.py')]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(BASE))
    with (ATTEMPT / 'controller.log').open('x') as log:
        child = spawn_supervised(argv, env=env, log=log, lock=lease,
            launcher_path=ATTEMPT / 'controller-launcher.json', exit_path=ATTEMPT / 'controller-exit.json',
            metadata=dict(plan_sha256=sha(ATTEMPT / 'plan.json'), activation_source_sha256=sha(Path(__file__))), cwd=BASE)
deadline = time.monotonic() + 30
while not (ATTEMPT / 'status.json').exists():
    assert child.poll() is None and time.monotonic() < deadline
    time.sleep(.1)
assert child.poll() is None and matching_process(child.launch)
status = read(ATTEMPT / 'status.json')
assert status['stage'] == 'waiting_for_measurement_priority' and status['running'] is None
assert status['priority_pending']
assert sha(BASE / 'combined-results/results.csv') == before
save(OUT / 'activation.json', dict(status='queued_waiting_for_measurements', controller=child.launch,
    initial_status=status, plan_sha256=sha(ATTEMPT / 'plan.json'),
    unchanged_results_csv_sha256=before, reused_cases=262, remaining_cases=167,
    new_gpu_runs=0, signalled_processes=[], source_sha256=sha(Path(__file__)), finished_unix=time.time()))
print(dict(status='queued_waiting_for_measurements', pid=child.pid, remaining_cases=167, new_gpu_runs=0))
