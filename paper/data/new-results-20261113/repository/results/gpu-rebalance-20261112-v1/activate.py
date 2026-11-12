"""Apply the reviewed placement plan without signaling GPU workers."""
import json
import os
from pathlib import Path
import subprocess
import time
import controller as migration

HERE = Path(__file__).resolve().parent
plan = migration.read(HERE / 'plan.json')
throughput, t1 = Path(plan['throughput']), Path(plan['t1'])
pause = t1 / 'pause-new-jobs'
hold = throughput / 'gpu-rebalance-hold.json'
assert not (HERE / 'activation.json').exists()
assert not pause.exists() and not hold.exists()
assert (throughput / 'pause-new-jobs').read_text() == plan['pause_content']
assert migration.read(throughput / 'status.json')['pending'] == [job['id'] for job in plan['jobs']]
for key in ['original_controller', 't1_controller', 'guard_controller']:
    actual = migration.identity(plan[key]['pid'])
    assert actual['start_ticks'] == plan[key]['start_ticks'] and actual['argv'] == plan[key]['argv']


def environment(pid):
    rows = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    return dict(row.decode().split('=', 1) for row in rows if row)


def launch(name, argv, env):
    with (HERE / f'{name}.log').open('x') as log:
        child = subprocess.Popen(argv, cwd=HERE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(pid=child.pid, argv=argv, started_unix=time.time())
    migration.save(HERE / f'{name}-launcher.json', receipt)
    return child


env_t1 = environment(plan['t1_controller']['pid'])
env_guard = environment(plan['guard_controller']['pid'])
env_throughput = environment(plan['original_controller']['pid'])
pause_text = json.dumps(dict(owner=str(HERE), reason='Brief coordinator handoff; preserve active workers')) + '\n'
with pause.open('x') as stream:
    stream.write(pause_text)
paused_at = time.time()
while migration.read(t1 / 'status.json')['updated_unix'] <= paused_at:
    assert time.time() - paused_at < 35, 'T1 controller did not acknowledge pause'
    time.sleep(1)
old_state = migration.read(t1 / 'status.json')
assert len(old_state['pending']) == 2
worker_identities = {key: migration.identity(entry['pid']) for key, entry in old_state['running'].items()}
migration.save(HERE / 't1-status-at-handoff.json', old_state)
migration.save(hold, dict(owner=str(HERE), created_unix=time.time(),
                         reason='Keep original throughput dispatch paused while migrated jobs use the same result paths'))
migration.stop_controller(plan['t1_controller'])
t1_child = launch('t1-controller', [migration.CONTROLLER_PYTHON, '-u',
                  str(HERE / 'orchestrate_t1_with_rebalance_hold.py'), '--out', str(t1), '--execute', '--resume'], env_t1)
deadline = time.time() + 40
while True:
    assert t1_child.poll() is None, 'Resumed T1 controller exited; inspect t1-controller.log'
    state = migration.read(t1 / 'status.json')
    if state['controller_pid'] == t1_child.pid:
        break
    assert time.time() < deadline, 'Resumed T1 controller did not publish status'
    time.sleep(1)
for key, expected in worker_identities.items():
    current = migration.identity(expected['pid'])
    assert current['start_ticks'] == expected['start_ticks'] and current['argv'] == expected['argv']
    assert state['running'][key]['pid'] == expected['pid']

# Preserve the guard's previous scope before changing the shared allocation.
migration.stop_controller(plan['guard_controller'])
guard = launch('guard', ['/tmp/loopfill-specforge-venv/bin/python', '-u',
                       str(HERE / 'guard_original_scope.py')], env_guard)
time.sleep(1)
assert guard.poll() is None, 'Scoped guard restart failed'
allocation = migration.read(HERE / 'allocation-after.json')
assert migration.read(migration.BASE / 'configs/gpu-allocation.json') == migration.read(HERE / 'allocation-before.json')
migration.save(migration.BASE / 'configs/gpu-allocation.json', allocation)
migration.save(throughput / 'gpu-placement-amendment.json', dict(
    plan=str(HERE / 'plan.json'), migrated_jobs=[job['id'] for job in plan['jobs']],
    migrated_gpus=migration.GPUS, status=str(HERE / 'status.json'), created_unix=time.time(),
    note='Original status lists migrated jobs as pending until receipt reconciliation; consult migration status for their live placement.'))
child = launch('controller', [migration.CONTROLLER_PYTHON, '-u', str(HERE / 'controller.py')], env_throughput)
time.sleep(1)
assert child.poll() is None, 'Migration controller failed to start'
assert pause.read_text() == pause_text
pause.unlink()
migration.save(HERE / 'activation.json', dict(activated_unix=time.time(), controller_pid=child.pid,
    t1_controller_pid=t1_child.pid, guard_pid=guard.pid,
    preserved_t1_workers=worker_identities, preserved_throughput=plan['original_running']))
print(json.dumps(migration.read(HERE / 'activation.json'), indent=2))
