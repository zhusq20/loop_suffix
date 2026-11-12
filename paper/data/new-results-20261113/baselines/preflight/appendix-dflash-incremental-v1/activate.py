"""Replace only the idle CPU barrier with the unchanged incremental dispatcher."""
import fcntl
import os
from pathlib import Path
import signal
import sys
import time

BASE, OUT = (Path(x).resolve() for x in sys.argv[1:])
root = BASE / 'logs/appendix-dflash-diagnostics-v1'
sys.path[:0] = [str(root / 'source'), str(BASE)]
from packed_dflash_evidence import read, save, sha
from loop_queue_recovery import (matching_process, pidfd_open, pidfd_send_signal,
                                 process_identity, spawn_supervised)
from orchestrate_baseline_diagnostics_resume import verify_plan

assert not (OUT / 'activation.json').exists()
proof = read(OUT / 'report.json')
assert proof['status'] == 'candidate_prepared_not_activated'
assert read(OUT / 'prepare-exit.json')['exit_code'] == 0
assert read(OUT / 'dry-run-exit.json')['exit_code'] == 0
assert read(OUT / 'priority-dry-run.json')['worker_launches'] == 0
assert sha(OUT / 'plan.json') == proof['candidate_plan_sha256']
assert sha(root / 'spec.json') == proof['original_spec_sha256']
plan = verify_plan(OUT / 'plan.json')
for name, digest in plan['source_sha256'].items():
    assert sha(root / 'source' / name) == digest
assert sha(BASE / 'combined-results/results.csv') == proof['unchanged_combined_csv_sha256']
for name in ('plan.json', 'dispatch.json', 'complete.json', 'failure.json',
             'incremental-launcher.json', 'incremental-exit.json'):
    assert not (root / name).exists(), name
assert all(not Path(g['out']).exists() for g in plan['groups'])

old = read(root / 'launcher.json')
assert old['pid'] == 216747 and old['start_ticks'] == '49411305'
assert matching_process(old) and matching_process(old['supervisor_identity'])
status = read(root / 'waiter-status.json')
assert status['controller_pid'] == old['pid']
assert status['stage'] == 'waiting_for_existing_training_and_measurements'
assert status['gpu_workers'] == 0
assert not Path(f'/proc/{old["pid"]}/task/{old["pid"]}/children').read_text().strip()
assert not (Path(plan['supplement_status_root']) / 'complete.json').exists()
supervision = read(root / 'launcher.json.supervisor.json')
old_exit_path = Path(supervision['exit_path'])
assert not old_exit_path.exists()
save(OUT / 'waiter-before-launcher.json', old)
save(OUT / 'waiter-before-status.json', status)

with (OUT / 'handoff.lock').open('a+') as handoff, \
     (root / 'controller.lock').open('a+') as controller_guard:
    fcntl.flock(handoff, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(controller_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pidfd = pidfd_open(old['pid'])
    try:
        assert matching_process(old)
        pidfd_send_signal(pidfd, signal.SIGTERM)
        save(OUT / 'waiter-stop-request.json', dict(pid=old['pid'],
            start_ticks=old['start_ticks'], signal='SIGTERM', sent_unix=time.time(),
            reason='Scheduling handoff of idle CPU waiter only; preserve all GPU work and all six required diagnostic groups.'))
    finally:
        os.close(pidfd)
    deadline = time.monotonic() + 30
    while not old_exit_path.exists():
        assert time.monotonic() < deadline, 'Observe the same supervisor; never invent its exit.'
        time.sleep(.2)
    old_exit = read(old_exit_path)
    assert old_exit['exit_code'] == -signal.SIGTERM
    assert all(old_exit[k] == v for k, v in old.items())
    assert not matching_process(old)
    save(OUT / 'waiter-actual-exit.json', old_exit)

    waiter_lease = (root / 'waiter.lock').open('a+')
    deadline = time.monotonic() + 10
    while True:
        try:
            fcntl.flock(waiter_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            assert time.monotonic() < deadline, 'Old supervisor has not released its lease.'
            time.sleep(.1)
    with (root / 'plan.json').open('xb') as stream:
        stream.write((OUT / 'plan.json').read_bytes())
        stream.flush()
        os.fsync(stream.fileno())
    assert verify_plan(root / 'plan.json') == plan
    fcntl.flock(controller_guard, fcntl.LOCK_UN)
    argv = ['taskset', '--cpu-list', '80-83', '/opt/conda/bin/python', '-u',
            str(root / 'source/orchestrate_baseline_diagnostics_resume.py'),
            '--plan', str(root / 'plan.json')]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join([str(root / 'source'), str(BASE)]))
    with (root / 'incremental-controller.log').open('x') as log:
        child = spawn_supervised(argv, env=env, log=log, lock=waiter_lease,
            launcher_path=root / 'incremental-launcher.json',
            exit_path=root / 'incremental-exit.json',
            metadata=dict(scheduling_amendment=str(OUT),
                plan_sha256=sha(root / 'plan.json'), original_spec_sha256=sha(root / 'spec.json')),
            cwd=BASE)
    waiter_lease.close()  # The persistent supervisor and controller retain it.

deadline = time.monotonic() + 30
while True:
    assert child.poll() is None, 'Inspect the preserved dispatcher exit; do not restart blindly.'
    if (root / 'status.json').exists():
        current = read(root / 'status.json')
        if current['controller_pid'] == child.pid:
            break
    assert time.monotonic() < deadline
    time.sleep(.2)
assert current['expected_groups'] == 6 and not current['failures']
assert matching_process(child.launch)
assert sha(root / 'spec.json') == proof['original_spec_sha256']
assert sha(BASE / 'combined-results/results.csv') == proof['unchanged_combined_csv_sha256']
save(root / 'waiter-status.json', dict(stage='incremental_dispatch_active',
    controller_pid=child.pid, updated_unix=time.time(), expected_groups=6,
    plan_sha256=sha(root / 'plan.json'), amendment=str(OUT)))
save(OUT / 'activation.json', dict(status='incremental_dispatch_active',
    plan_sha256=sha(root / 'plan.json'), old_waiter_actual_exit=dict(
        path=str(old_exit_path), sha256=sha(old_exit_path), exit_code=old_exit['exit_code']),
    new_controller=child.launch, new_controller_live=process_identity(child.pid),
    initial_status=current, original_spec_sha256=sha(root / 'spec.json'),
    unchanged_combined_csv_sha256=sha(BASE / 'combined-results/results.csv'),
    changed_gpu_measurements=False, diagnostic_worker_code_unchanged=True,
    completed_unix=time.time()))
print(dict(status='incremental_dispatch_active', pid=child.pid,
           expected_groups=6, old_cpu_waiter_exit=old_exit['exit_code']))
