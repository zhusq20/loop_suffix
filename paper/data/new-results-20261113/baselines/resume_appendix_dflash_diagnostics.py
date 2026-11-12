"""Wait for the existing appendix evaluator, then replay its six greedy cells.

This CPU controller never restarts training or timed evaluation. Its diagnostic
workers use the same GPU leases and unchanged replay implementation as the main
queue, and every launch/exit is recorded by that queue's durable supervisor.
"""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from packed_dflash_evidence import read, save, sha


def verify(spec_path):
    spec = read(spec_path)
    root = spec_path.parent
    for name, expected in spec['source_sha256'].items():
        assert sha(root/'source'/name) == expected, name
    for name, expected in spec['base_sha256'].items():
        assert sha(Path(spec['base'])/name) == expected, name
    assert sha(spec['evaluator_plan']) == spec['evaluator_plan_sha256']
    return spec


def prepare(root, spec):
    base, parent = Path(spec['base']), Path(spec['evaluator_plan']).parent
    finished = read(parent/'complete.json')
    assert finished['status'] == 'complete' and finished['completed_cells'] == 12
    assert not finished['failures']
    parent_plan = read(spec['evaluator_plan'])
    dispatch = read(parent/'dispatch.json')
    expected = {parent_plan['validation_job']['id']}
    expected |= {j['id']+'-'+stage for j in parent_plan['jobs'] for stage in ['timed', 'ar', 'profile']}
    assert set(dispatch) == expected and len(expected) == 37
    for name, record in dispatch.items():
        assert record['outcome'] == 'complete' and record['exit_code'] == 0, name
        launch, receipt = read(parent/f'logs/{name}-launcher.json'), read(parent/f'logs/{name}-exit.json')
        assert receipt['exit_code'] == 0 and all(receipt[k] == v for k, v in launch.items())
    sys.path.insert(0, str(base))
    from collect_combined_results import key, read_packed
    measurements, evidence = read_packed(parent/'supplement')
    assert len(measurements) == 12
    assert {(r['benchmark'], r['temperature']) for r in measurements} == {
        (j['dataset'], j['temperature']) for j in parent_plan['jobs']}
    jobs = {j['dataset']: j for j in parent_plan['jobs'] if j['temperature'] == 0}
    groups = []
    for row in measurements:
        if row['temperature'] != 0:
            continue
        assert row['model'] == 'ouro-1.4b' and row['method_exit_evidence'] == 'actual_exit0'
        job = jobs[row['benchmark']]
        folder = parent/'diagnostics'/job['id']
        assert not folder.exists(), f'Preserve and inspect existing diagnostic: {folder}'
        identity = job['id']+'-greedy'
        groups.append(dict(id=identity, kind='packed', job=job['id'], out=str(folder),
            source_evidence=evidence['/'.join(map(str, key(row)))], steps=[dict(
                id=identity, out=str(folder), marker=str(folder/'complete.json'), legacy_lock=False,
                argv=[parent_plan['python'], '-u', str(root/'source/diagnose_packed_dflash.py'),
                    '--plan', str(parent/'supplement/plan.json'), '--job', job['id'],
                    '--ar', row['ar_run'], '--out', str(folder), '--lock-fd', '{LOCK_FD}'])]))
    assert len(groups) == 6
    plan = dict(created_unix=time.time(), base=str(base), balanced_queue=parent_plan['balanced_queue'],
        supplement_plan=str(parent/'supplement/plan.json'), supplement_status_root=str(parent),
        gpu_policy=parent_plan['gpu_policy'], source_sha256=spec['source_sha256'],
        legacy_source_sha256=spec['base_sha256'], groups=groups, reused_groups={},
        limitation='Six Ouro1.4B DFlash T0 numerical checks after all twelve timed/AR/profile cells. '
                   'Diagnostic replay timing is excluded from benchmark measurements.',
        evaluator_complete_sha256=sha(parent/'complete.json'), spec_sha256=sha(root/'spec.json'))
    save(root/'plan.json', plan)
    save(root/'checks/preparation.json', dict(status='passed', groups=6, measured_cells=12,
        completed_evaluator_steps=37, plan_sha256=sha(root/'plan.json'),
        first_divergences=sum(len(g['source_evidence']['agreement']['first_divergences']) for g in groups),
        checked_unix=time.time()))


def main(spec_path):
    spec = verify(spec_path)
    root = spec_path.parent
    parent = Path(spec['evaluator_plan']).parent
    training = Path(read(spec['evaluator_plan'])['training_root'])
    with (root/'waiter.lock').open('a+') as waiter_lock:
        fcntl.flock(waiter_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not (parent/'complete.json').exists():
            verify(spec_path)
            for folder in [training, parent]:
                assert not (folder/'failure.json').exists(), f'Inspect existing failure before diagnostic continuation: {folder}'
            save(root/'waiter-status.json', dict(stage='waiting_for_existing_training_and_measurements',
                controller_pid=os.getpid(), updated_unix=time.time(), evaluator=str(parent),
                expected_groups=6, gpu_workers=0))
            time.sleep(15)
        verify(spec_path)
        if not (root/'plan.json').exists():
            prepare(root, spec)
        sys.path.insert(0, str(root/'source'))
        from orchestrate_baseline_diagnostics_resume import main as dispatch
        save(root/'waiter-status.json', dict(stage='dispatching_numerical_checks',
            controller_pid=os.getpid(), updated_unix=time.time(), plan_sha256=sha(root/'plan.json')))
        dispatch(root/'plan.json')
        save(root/'waiter-status.json', dict(stage=read(root/'complete.json')['status'],
            controller_pid=os.getpid(), updated_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.spec.resolve())
    except Exception:
        save(args.spec.parent/'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        raise
