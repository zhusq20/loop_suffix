"""Run the independent CPU audit once the existing portable pipeline finishes.

Only the audit is launched here. Training, GPU leases and timed measurements
belong to their existing controllers and are never restarted or signalled.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time
import traceback

from packed_dflash_evidence import read, save, sha
from orchestrate_formal import run_command


def verify(plan_path):
    plan = read(plan_path)
    base = Path(plan['base'])
    for name, digest in plan['source_sha256'].items():
        assert sha(base/name) == digest, name
        assert sha(plan_path.parent/'source'/name) == digest, name
    assert sha(plan['evaluation_plan']) == plan['evaluation_plan_sha256']
    assert sha(Path(plan['training_root'])/'plan.json') == plan['training_plan_sha256']
    return plan


def process_exists(pid):
    path = Path('/proc')/str(pid)/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, exists=True, state=fields[0], start_ticks=fields[19])
    except FileNotFoundError:
        return dict(pid=pid, exists=False)


def main(plan_path, execute):
    plan = verify(plan_path)
    root, training = plan_path.parent, Path(plan['training_root'])
    if not execute:
        print(dict(status='plan_verified', training_complete=(training/'complete.json').exists(),
                   cpu_affinity=plan['cpu_affinity'], cuda_visible_devices='', launches_gpu_work=False))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.sched_setaffinity(0, set(plan['cpu_affinity']))
    with (root/'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (root/'failure.json').exists(), 'Inspect prior failure before any continuation.'
        while not (training/'complete.json').exists():
            verify(plan_path)
            assert not (training/'failure.json').exists(), 'Existing training pipeline failed; inspect its evidence.'
            state = read(training/'status.json')
            save(root/'status.json', dict(stage='waiting_for_existing_training_and_parent_audit',
                controller_pid=os.getpid(), updated_unix=time.time(), training_stage=state['stage'],
                observed_training_process=process_exists(state['controller_pid']), gpu_workers=0))
            time.sleep(15)
        verify(plan_path)
        base, name, out = Path(plan['base']), plan['audit_name'], Path(plan['audit_out'])
        finish_path, launch_path = [base/f'logs/{name}-{kind}.json' for kind in ['finished','launcher']]
        if not finish_path.exists():
            assert not launch_path.exists() and not out.exists(), 'Preserve any prior audit process/artifacts; do not duplicate.'
            save(root/'status.json', dict(stage='running_cpu_audit', controller_pid=os.getpid(),
                updated_unix=time.time(), gpu_workers=0))
            run_command(name, [plan['python'], '-u', str(base/'audit_portable_dflash_training.py'),
                              '--out', str(out)], dict(os.environ))
        launch, finish = read(launch_path), read(finish_path)
        assert finish['exit_code'] == 0 and all(finish[k] == v for k,v in launch.items())
        report = read(out/'report.json')
        assert report['status'] == 'complete_training_audit_not_benchmark'
        assert report['model'] == 'ouro-1.4b' and report['epochs'] == 6
        assert not report['cuda_initialized'] and report['learning_rate_schedule']['all_logged_learning_rates_match']
        for path, digest in report['artifact_hashes'].items():
            assert sha(path) == digest, path
        assert sha(Path(report['checkpoint'])/'model.safetensors') == report['weights_sha256']
        result = dict(status='complete', report=str(out/'report.json'), report_sha256=sha(out/'report.json'),
                      actual_audit_exit=str(finish_path), actual_audit_exit_sha256=sha(finish_path),
                      plan_sha256=sha(plan_path), finished_unix=time.time(), gpu_workers=0)
        save(root/'complete.json', result)
        save(root/'status.json', dict(result, stage='complete', controller_pid=os.getpid()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        main(args.plan.resolve(), args.execute)
    except Exception:
        save(args.plan.parent/'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        raise
