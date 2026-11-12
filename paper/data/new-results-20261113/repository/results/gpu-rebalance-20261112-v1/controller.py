"""Run the six unstarted throughput cells on GPUs 0/1/5/7, then reconcile.

The original controller retains its running child and records its real exit.
Its dispatch pause stays owned by the T1 controller until this queue drains.
No benchmark worker is stopped or adopted without its original exit receipt.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
GPUS = [0, 1, 5, 7]
CPUS = {0: '0-3', 1: '4-7', 5: '52-55', 7: '60-63'}
PYTHON = '/tmp/loopfill-venv/bin/python'
CONTROLLER_PYTHON = '/tmp/loopfill-specforge-venv/bin/python'


def read(path):
    return Path(path).read_text() if Path(path).suffix != '.json' else json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reservation_available(uuid):
    # A queue may keep this reservation between CUDA workers. CUDA occupancy
    # alone does not make a device available for another experiment suite.
    with Path(f'/tmp/loopfill-{uuid}.lock').open('a+') as reservation:
        try:
            fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(reservation, fcntl.LOCK_UN)
        return True


def archive_unstarted_lock_conflict(throughput, job_id, receipt):
    """Preserve a pre-CUDA lock collision and retry only an untouched cell."""
    destination = throughput / 'runs' / job_id
    log = throughput / 'logs' / f'{job_id}.log'
    if receipt['exit_code'] != 1 or not destination.is_dir() or any(destination.iterdir()):
        return False
    message = log.read_text()
    if 'BlockingIOError: [Errno 11] Resource temporarily unavailable' not in message or 'fcntl.flock' not in message:
        return False
    archive = HERE / 'prestart-lock-conflicts' / f"{job_id}-{receipt['pid']}"
    archive.mkdir(parents=True, exist_ok=False)
    destination.rename(archive / 'empty-run-directory')
    for suffix in ['.log', '-launcher.json', '-finished.json']:
        path = throughput / 'logs' / f'{job_id}{suffix}'
        if path.exists():
            path.rename(archive / path.name)
    for suffix in ['-launcher.json', '-finished.json']:
        path = HERE / f'{job_id}{suffix}'
        if path.exists():
            path.rename(archive / f'migration-{path.name}')
    save(archive / 'retry.json', dict(reason='GPU reservation raced before CUDA initialization; empty attempt archived.',
                                    archived_unix=time.time(), receipt=receipt))
    return True


def identity(pid):
    proc = Path(f'/proc/{pid}')
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=stat[19], state=stat[0],
                argv=(proc / 'cmdline').read_bytes().decode().split('\0')[:-1])


def stop_controller(expected):
    """Signal one checked coordinator PID, never its process group/children."""
    try:
        descriptor = os.pidfd_open(expected['pid'])
    except ProcessLookupError:
        return
    try:
        actual = identity(expected['pid'])
        assert actual['start_ticks'] == expected['start_ticks']
        if actual['state'] == 'Z':
            return
        assert actual['argv'] == expected['argv'], actual
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        assert select.select([descriptor], [], [], 10)[0], 'Coordinator did not exit'
    finally:
        os.close(descriptor)


def main():
    plan = read(HERE / 'plan.json')
    throughput = Path(plan['throughput'])
    t1 = Path(plan['t1'])
    pause = throughput / 'pause-new-jobs'
    hold = throughput / 'gpu-rebalance-hold.json'
    assert read(hold)['owner'] == str(HERE)
    assert pause.read_text() == plan['pause_content']
    source = HERE / 'source'
    for relative, digest in read(source / 'source-manifest.json').items():
        assert sha(source / relative) == digest, relative
    sys.path.insert(0, str(source))
    from native_cuda_env import native_cuda_environment
    env, evidence = native_cuda_environment()
    env.update(PYTHONPATH=str(source), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               TOKENIZERS_PARALLELISM='false')
    save(HERE / 'native-driver.json', evidence)
    uuids = {int(i): u.strip() for i, u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
    active, finished = {}, {}
    with (HERE / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (HERE / 'status.json').exists(), 'Inspect existing dispatch before restarting'
        while True:
            assert hold.exists() and read(hold)['owner'] == str(HERE)
            assert pause.exists() and pause.read_text() == plan['pause_content']
            for job_id, entry in list(active.items()):
                code = entry['process'].poll()
                if code is None:
                    continue
                entry['log'].close()
                receipt = dict(entry['receipt'], exit_code=code, finished_unix=time.time())
                save(throughput / 'logs' / f'{job_id}-finished.json', receipt)
                save(HERE / f'{job_id}-finished.json', receipt)
                if not archive_unstarted_lock_conflict(throughput, job_id, receipt):
                    finished[job_id] = receipt
                del active[job_id]
            pending = [job for job in plan['jobs'] if job['id'] not in active and job['id'] not in finished]
            allowed = read(BASE / 'configs/gpu-allocation.json')['allowed_gpus']
            busy = {line.strip() for line in subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()}
            reserved = {entry['receipt']['gpu'] for entry in active.values()}
            lock_reserved = [gpu for gpu in GPUS if not reservation_available(uuids[gpu])]
            free = [gpu for gpu in GPUS if gpu in allowed and gpu not in reserved
                    and gpu not in lock_reserved and uuids[gpu] not in busy]
            if not (HERE / 'pause-new-jobs').exists():
                for job in pending[:len(free)]:
                    job_id = job['id']
                    assert job['mode'] == 'formal'
                    destination = throughput / 'runs' / job_id
                    assert not destination.exists(), f'Existing attempt: {destination}'
                    assert not (throughput / 'logs' / f'{job_id}-launcher.json').exists()
                    gpu = free.pop(0)
                    argv = [PYTHON, '-u', str(source / 'evaluate_loopahead_throughput.py'),
                            '--model', job['model'], '--batch', str(job['batch']), '--mode', 'formal',
                            '--gpu', str(gpu), '--cpus', CPUS[gpu], '--out', str(destination)]
                    log = (throughput / 'logs' / f'{job_id}.log').open('x')
                    child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
                    receipt = dict(pid=child.pid, argv=argv, gpu=gpu, started_unix=time.time(),
                                   placement_amendment=str(HERE / 'plan.json'))
                    save(throughput / 'logs' / f'{job_id}-launcher.json', receipt)
                    save(HERE / f'{job_id}-launcher.json', receipt)
                    active[job_id] = dict(process=child, log=log, receipt=receipt)
                    print(json.dumps(dict(event='launch', job=job_id, **receipt)), flush=True)
            legacy = read(throughput / 'status.json')
            state = dict(updated_unix=time.time(), controller_pid=os.getpid(),
                         stage='running' if active else 'waiting_for_gpu_release',
                         allowed_gpus=GPUS, running={key: value['receipt'] for key, value in active.items()},
                         reserved_gpus=lock_reserved,
                         pending=[job['id'] for job in plan['jobs'] if job['id'] not in active and job['id'] not in finished],
                         completed=[key for key, value in finished.items() if value['exit_code'] == 0],
                         failed={key: value['exit_code'] for key, value in finished.items() if value['exit_code'] != 0},
                         original_running=legacy['running'], original_status=str(throughput / 'status.json'),
                         note='Migrated jobs are tracked here until final reconciliation; original dispatch remains paused.')
            save(HERE / 'status.json', state)
            if len(finished) == len(plan['jobs']) and not active:
                # The original parent must reap its own child before retirement.
                old_receipts = [throughput / 'logs' / f'{job_id}-finished.json' for job_id in plan['original_running']]
                if not legacy['running'] and all(path.exists() for path in old_receipts):
                    break
            time.sleep(10)
        stop_controller(plan['original_controller'])
        # With every throughput job possessing a genuine exit receipt, the
        # unchanged controller can safely resume, aggregate, and finish.
        protocol = read(throughput / 'protocol.json')
        assert all((throughput / 'logs' / f"{job['id']}-finished.json").exists()
                   for job in protocol['preflight'] + protocol['jobs'])
        save(HERE / 'legacy-controller-retired.json', dict(retired_unix=time.time(),
             controller=plan['original_controller'], reason='All original child exit receipts recorded'))
        with (HERE / 'reconcile.log').open('x') as log:
            code = subprocess.call(plan['original_controller']['argv'], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        assert (throughput / 'finished.json').exists(), 'Reconciliation did not publish finished.json'
        # T1 may still be using the pause as a priority gate. Remove it only
        # after that controller has relinquished its ownership.
        if read(t1 / 'status.json')['throughput_dispatch_resumed']:
            assert pause.read_text() == plan['pause_content']
            pause.unlink()
        hold.unlink()
        state.update(stage='complete' if code == 0 and not state['failed'] else 'needs_attention',
                     finished_unix=time.time(), reconcile_exit_code=code)
        save(HERE / 'status.json', state)
        save(HERE / 'finished.json', state)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(HERE / 'failure.json', dict(failed_unix=time.time(), error=traceback.format_exc()))
        raise
