"""Queue only the 167 unfinished FP32 cases behind the live measurements."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

from loopahead_resume_attempt import ATTEMPT, BASE, verify_attempt, validate_completed
from loopahead_partial_resume import load_partial
from packed_dflash_evidence import read, save, sha

# Frozen receipt writer and device-allocation helpers already used by the evaluator.
SUPPORT = BASE / 'logs/ouro14-packed-evaluation-v1/source'
sys.path.insert(0, str(SUPPORT))
from loop_queue_recovery import ObservedWorker, spawn_supervised
from native_cuda_env import native_cuda_environment
from orchestrate_packed_dflash_supplements import cards, occupied, cpus


def ready_to_launch(parent, evaluator, stopped):
    assert not parent.get('failures') and not evaluator['failures'], 'Inspect measurement failure first.'
    return not stopped and not parent['pending'] and not evaluator['ready']


def main():
    plan = verify_attempt()
    group, root = plan['group'], ATTEMPT
    step = group['steps'][0]
    launcher = root / f'logs/{step["id"]}-launcher.json'
    receipt = root / f'logs/{step["id"]}-exit.json'
    with (root / 'controller.lock').open('a+') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        worker = ObservedWorker(read(launcher), receipt) if launcher.exists() else None
        lease = None
        while True:
            if worker:
                code = worker.poll()
                if code is not None:
                    if lease: lease.close()
                    assert code == 0, f'Preserve failed/unknown continuation: {code}'
                    verify_attempt()
                    validated = validate_completed(plan)
                    save(root / 'complete.json', dict(status='complete', validation=validated,
                        actual_exit_sha256=sha(receipt), plan_sha256=sha(root / 'plan.json'),
                        original_interrupted_exit_code=-15, reused_cases=262, newly_executed_cases=167,
                        finished_unix=time.time()))
                    save(root / 'status.json', dict(stage='complete', controller_pid=os.getpid(), updated_unix=time.time()))
                    return
            parent = read(Path(plan['balanced_queue']) / 'status.json')
            evaluator = read(Path(plan['measurement_queue']) / 'status.json')
            stopped = (root / 'STOP').exists()
            allowed_to_start = ready_to_launch(parent, evaluator, stopped)
            if not worker and allowed_to_start:
                policy = read(plan['gpu_policy'])
                allowed = set(policy['allowed_gpus']) - set(policy.get('reserved_for_development', []))
                busy = occupied()
                for gpu, (uuid, mode) in sorted(cards().items()):
                    if gpu not in allowed or uuid in busy or mode != 'Exclusive_Process': continue
                    lease = open(f'/tmp/loopfill-{uuid}.lock', 'a+')
                    try: fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lease.close(); lease = None; continue
                    if uuid in occupied():
                        lease.close(); lease = None; continue
                    verify_attempt()
                    values = step['argv']
                    def arg(flag): return Path(values[values.index(flag)+1])
                    load_partial(arg('--resume-evidence'), argparse.Namespace(
                        ar=arg('--ar'), loopahead=arg('--loopahead'), observed_margin_report=arg('--observed-margin-report'),
                        paper_check=False, out=Path(step['out'])))
                    assert not (root / 'STOP').exists()
                    env, driver = native_cuda_environment()
                    save(root / 'native-driver.json', driver)
                    argv = [x.replace('{GPU}', str(gpu)).replace('{CPUS}', cpus(gpu)) for x in values]
                    argv.insert(2, str(BASE / 'logs/baseline-diagnostics-resume-v1/source/lock_handoff.py'))
                    argv = ['taskset', '--cpu-list', cpus(gpu), *argv]
                    env.update(CUDA_VISIBLE_DEVICES=uuid, LOOPFILL_RETRY_LOCK_FD=str(lease.fileno()),
                        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        TOKENIZERS_PARALLELISM='false', PYTHONPATH=str(BASE))
                    with (root / f'logs/{step["id"]}.log').open('x') as log:
                        worker = spawn_supervised(argv, env=env, log=log, lock=lease,
                            launcher_path=launcher, exit_path=receipt,
                            metadata=dict(group=group['id'], step=step['id'], gpu=gpu,
                                plan_sha256=sha(root / 'plan.json')), cwd=BASE)
                    break
            save(root / 'status.json', dict(stage='running' if worker else 'stopped' if stopped else 'waiting_for_measurement_priority',
                controller_pid=os.getpid(), updated_unix=time.time(),
                running=worker.launch if worker else None, reused_cases=262, remaining_cases=167,
                priority_pending=parent['pending'] or evaluator['ready']))
            time.sleep(10)


if __name__ == '__main__':
    try: main()
    except Exception:
        save(ATTEMPT / 'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        raise
