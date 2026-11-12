"""Resume the unchanged six-group queue after this thread's later confirmation.

Preserve the prior stop evidence, never signal a process, and keep measurement
priority. The unfinished main LoopAhead MATH check is outside this queue.
"""
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time

BASE = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path[:0] = [str(ROOT / 'source'), str(BASE)]
from packed_dflash_evidence import read, save, sha
from loop_queue_recovery import matching_process, process_identity, spawn_supervised
from orchestrate_baseline_diagnostics_resume import verify_plan

assert not (OUT / 'activation.json').exists()
assert sha(ROOT / 'plan.json') == 'e3dc2b1f6a89502b1553fc801f2bcd13f21340460af9eb38ba6cd92b31ddbcc8'
assert sha(ROOT / 'spec.json') == '19b1497549548ca931eb4afcdc7db395b6404cfe279c327992bde11c7676343c'
plan = verify_plan(ROOT / 'plan.json')
assert len(plan['groups']) == 6 and read(ROOT / 'dispatch.json') == {}
assert all(not Path(g['out']).exists() for g in plan['groups'])
assert all(not (ROOT / name).exists() for name in ['complete.json', 'failure.json'])
prior = read(ROOT / 'incremental-launcher.json')
prior_exit = read(ROOT / 'incremental-exit.json')
assert prior_exit['exit_code'] == -15 and prior_exit['exit_evidence'] == 'supervisor_waitpid'
assert all(prior_exit[k] == v for k, v in prior.items())
assert not matching_process(prior) and not matching_process(prior['supervisor_identity'])
assert read(ROOT / 'status.json')['stage'] == 'cancelled_by_user'
known_stop = BASE / 'preflight/appendix-dflash-incremental-v1/subsequent-stop-observation/logs/appendix-dflash-diagnostics-v1'
for name in ['STOP', 'cancellation.json']:
    assert sha(ROOT / name) == sha(known_stop / name), 'A newer stop needs separate inspection.'
measurements = read(Path(plan['supplement_status_root']) / 'status.json')
assert measurements['ready'] and not measurements['failures']
before_csv = sha(BASE / 'combined-results/results.csv')
authority = dict(
    thread='01a1c3de-eb05-7b60-9b55-f210fe6e7d75',
    later_user_instruction='好的 你继续试验吧',
    context='After the stop was explained, the user supplied the identical original task document, confirmed its identity, and instructed this thread to continue experiments.',
    scope='Original required checks remain in scope; preserve all completed measurements and prioritize AR/profile work.',
    prior_cancellation_actor='Unverified; do not treat the shared file attribution as a verified user or chat identity.',
    observed_unix=time.time())
save(OUT / 'authorization-context.json', authority)
saved = {}
with (ROOT / 'controller.lock').open('a+') as guard, (ROOT / 'waiter.lock').open('a+') as lease:
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not matching_process(prior) and read(ROOT / 'dispatch.json') == {}
    for name in ['STOP', 'cancellation.json', 'status.json', 'waiter-status.json',
                 'incremental-launcher.json', 'incremental-exit.json']:
        original = ROOT / name
        dest = OUT / 'before' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        shutil.copy2(original, dest)
        assert sha(original) == sha(dest)
        saved[name] = sha(dest)
    assert sha(ROOT / 'STOP') == saved['STOP'] == sha(known_stop / 'STOP')
    assert sha(ROOT / 'cancellation.json') == saved['cancellation.json'] == sha(known_stop / 'cancellation.json')
    (ROOT / 'STOP').unlink()  # Exact archived flag; a later user instruction now authorizes continuation.
    fcntl.flock(guard, fcntl.LOCK_UN)
    argv = ['taskset', '--cpu-list', '80-83', '/opt/conda/bin/python', '-u',
            str(ROOT / 'source/orchestrate_baseline_diagnostics_resume.py'),
            '--plan', str(ROOT / 'plan.json')]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join([str(ROOT / 'source'), str(BASE)]))
    with (OUT / 'controller.log').open('x') as log:
        child = spawn_supervised(argv, env=env, log=log, lock=lease,
            launcher_path=OUT / 'controller-launcher.json', exit_path=OUT / 'controller-exit.json',
            metadata=dict(recovery=str(OUT), plan_sha256=sha(ROOT / 'plan.json'),
                authorization_context_sha256=sha(OUT / 'authorization-context.json')),
            cwd=BASE)

deadline = time.monotonic() + 30
while True:
    assert child.poll() is None, 'Preserve any failed attempt; do not restart blindly.'
    current = read(ROOT / 'status.json')
    if current.get('controller_pid') == child.pid:
        break
    assert time.monotonic() < deadline
    time.sleep(.1)
assert matching_process(child.launch)
assert current['expected_groups'] == 6 and current['priority_pending']
assert not current['running'] and not current['failures']
assert read(ROOT / 'dispatch.json') == {}
assert verify_plan(ROOT / 'plan.json') == plan
assert sha(BASE / 'combined-results/results.csv') == before_csv
save(ROOT / 'waiter-status.json', dict(stage='resumed_waiting_for_measurement_priority',
    controller_pid=child.pid, recovery=str(OUT), updated_unix=time.time()))
save(OUT / 'activation.json', dict(status='resumed_cpu_queue_waiting_for_measurements',
    controller=child.launch, live_identity=process_identity(child.pid), initial_status=current,
    plan_sha256=sha(ROOT / 'plan.json'), spec_sha256=sha(ROOT / 'spec.json'),
    prior_evidence_sha256=saved, authorization_context_sha256=sha(OUT / 'authorization-context.json'),
    unchanged_csv_sha256=before_csv, new_gpu_runs=0, signalled_processes=[],
    source_sha256=sha(Path(__file__)), completed_unix=time.time(),
    limitation='Six groups are queued, not complete. Main LoopAhead MATH remains separately interrupted.'))
print(dict(status='resumed_cpu_queue_waiting_for_measurements', pid=child.pid, new_gpu_runs=0))
