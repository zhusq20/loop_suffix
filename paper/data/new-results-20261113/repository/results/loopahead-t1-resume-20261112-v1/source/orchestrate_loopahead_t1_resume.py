"""Finish the 12 existing LoopAhead T=1 cells, sharing the throughput GPU queue.

Only missing single-request cells are dispatched. A temporary throughput pause
lets them acquire free devices without interrupting any active measurement.
The pause is released once all missing cells have started and acquired CUDA.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0, str(BASE))
from collect_results import read_run, sha, validate_training
from gpu_allocation import dispatch_gpus
from native_cuda_env import native_cuda_environment
from orchestrate_formal import AttachedWorker, MODELS, TASKS, ready, save, selected
from result_evidence import validate_run_artifacts

PROTOCOL = BASE.parents[1] / 'results/loopfill-latency512-20261107/protocol.json'
GPUS = [2, 3, 4, 6]


def read(path):
    return json.loads(path.read_text())


def verify(job):
    folder = BASE / 'runs' / (job['id'] + '-v1')
    run = read_run(folder, read(PROTOCOL))
    meta = run['meta']
    assert meta['method'] == 'loopahead' and meta['temperature'] == 1
    assert meta['model'] == job['model'] and meta['dataset'] == job['dataset']
    evidence = validate_run_artifacts(run)
    validate_training(run)
    return dict(requests=len(run['rows']), evidence=evidence, artifacts={
        str(folder / name): sha(folder / name)
        for name in ['metadata.json', 'complete.json', 'requests.jsonl', 'source-manifest.json']})


def prepare(out, throughput):
    jobs = [dict(id=f'loopahead-{model}-{task}-t1', kind='loopahead', model=model,
                 dataset=task, temperature=1) for task in TASKS for model in MODELS]
    completed = {}
    pending = []
    for job in jobs:
        folder = BASE / 'runs' / (job['id'] + '-v1')
        if (folder / 'complete.json').exists():
            completed[job['id']] = verify(job)
        else:
            assert ready(job), f'Formal prerequisites missing: {job["id"]}'
            assert not folder.exists(), f'Inspect incomplete run before retry: {folder}'
            for suffix in ['.log', '-launcher.json', '-finished.json']:
                assert not (BASE / 'logs' / (folder.name + suffix)).exists()
            pending.append(job)
    assert len(jobs) == 12 and len(completed) == 7 and len(pending) == 5
    assert not (throughput / 'pause-new-jobs').exists(), 'Throughput already paused by another owner'
    assert (BASE / 'logs/formal-controller-v1/pause-new-jobs').exists()
    assert set(GPUS) <= set(read(BASE / 'configs/gpu-allocation.json')['allowed_gpus'])
    assert read(throughput / 'protocol.json')['allowed_gpus'] == GPUS
    source_files = [BASE / n for n in [
        'orchestrate_loopahead_t1_resume.py', 'orchestrate_formal.py', 'evaluate_loopahead.py',
        'native_cuda_env.py', 'gpu_startup.py', 'gpu_allocation.py', 'provenance.py',
        'collect_results.py', 'result_evidence.py', 'loopahead_diagnostic_inputs.py',
        'dflash_training_registry.py', 'configs/loopahead-ouro-2.6b-admission-v1.json',
        'configs/loopahead-lad.json', 'configs/loopahead-sampling-dev16.json']]
    source_files += sorted((BASE / 'loopahead').glob('*.py'))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in source_files:
        relative = str(path.relative_to(BASE))
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest[relative] = sha(path)
    save(out / 'plan.json', dict(created_unix=time.time(), all_jobs=jobs, jobs=pending,
        already_completed=completed, allowed_gpus=GPUS, throughput=str(throughput),
        throughput_status_at_prepare=read(throughput / 'status.json'),
        checkpoints={model: selected(model) for model in MODELS}, source_sha256=manifest,
        protocol_sha256=sha(PROTOCOL), temperature=1, batch_size=1, max_new_tokens=512,
        handoff='Existing throughput workers finish normally; missing T1 cells get free GPUs first; '
                'throughput dispatch resumes after every missing cell has acquired CUDA.'))


def check_sources(out, plan):
    for relative, digest in plan['source_sha256'].items():
        assert sha(BASE / relative) == sha(out / 'source' / relative) == digest, relative
    assert sha(PROTOCOL) == plan['protocol_sha256']
    for model, checkpoint in plan['checkpoints'].items():
        assert selected(model) == checkpoint


def execute(out, resume):
    plan = read(out / 'plan.json')
    check_sources(out, plan)
    for record in plan['already_completed'].values():
        for path, digest in record['artifacts'].items():
            assert sha(Path(path)) == digest, f'Existing result changed: {path}'
    throughput = Path(plan['throughput'])
    pause = throughput / 'pause-new-jobs'
    owner = json.dumps(dict(owner=str(out), reason='Dispatch remaining five LoopAhead T=1 cells'), sort_keys=True) + '\n'
    env, driver = native_cuda_environment()
    env.update(PYTHONPATH=str(BASE), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    save(out / f'native-driver-{int(time.time())}.json', driver)
    uuids = {int(i): u.strip() for i, u in (line.split(',') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
    with (out / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, completed, failed = {}, list(plan['already_completed']), {}
        lookup = {job['id']: job for job in plan['jobs']}
        released = False
        if resume:
            previous = read(out / 'status.json')
            completed, failed = previous['completed'], previous['failed']
            released = previous['throughput_dispatch_resumed']
            for identity, entry in previous['running'].items():
                active[identity] = dict(entry, process=AttachedWorker(identity, entry, lookup[identity], out))
        else:
            assert not (out / 'status.json').exists(), 'Use --resume for an existing controller'
        if not released:
            if pause.exists():
                assert pause.read_text() == owner, 'Another owner paused throughput'
            else:
                with pause.open('x') as stream:
                    stream.write(owner)
            save(out / 'throughput-pause.json', dict(path=str(pause), content=owner, created_unix=time.time()))
        save(out / 'controller-start.json', dict(pid=os.getpid(), argv=sys.argv, resumed=resume, started_unix=time.time()))
        while True:
            for identity, entry in list(active.items()):
                code = entry['process'].poll()
                if code is None:
                    continue
                receipt = dict(exit_code=code, finished_unix=time.time())
                try:
                    assert code == 0, f'Worker exited {code}'
                    receipt['validated'] = verify(lookup[identity])
                    completed.append(identity)
                except Exception as error:
                    failed[identity] = repr(error)
                    receipt['validation_error'] = repr(error)
                save(out / f'{identity}-finished.json', receipt)
                del active[identity]
            pending = [job for job in plan['jobs'] if job['id'] not in completed
                       and job['id'] not in failed and job['id'] not in active]
            busy = {line.strip() for line in subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines()}
            other = read(throughput / 'status.json')
            reserved = {entry['gpu'] for entry in active.values()}
            reserved.update(entry['gpu'] for entry in other['running'].values())
            free = [gpu for gpu in GPUS if gpu in dispatch_gpus() and gpu not in reserved and uuids[gpu] not in busy]
            owns_pause = pause.exists() and pause.read_text() == owner
            pause_seen = other['updated_unix'] > read(out / 'throughput-pause.json')['created_unix']
            if pending and owns_pause and pause_seen and not (out / 'pause-new-jobs').exists():
                for job in pending[:len(free)]:
                    assert job['kind'] == 'loopahead' and job['temperature'] == 1
                    assert ready(job)
                    check_sources(out, plan)
                    gpu = free.pop(0)
                    identity = job['id']
                    job_path = out / f'{identity}.json'
                    assert not job_path.exists(), f'Existing dispatch needs investigation: {identity}'
                    assert not (BASE / 'runs' / (identity + '-v1')).exists()
                    save(job_path, dict(job=job, gpus=[gpu], created_unix=time.time()))
                    argv = ['/opt/conda/bin/python', '-u', str(BASE / 'orchestrate_formal.py'), '--worker', str(job_path)]
                    with (out / f'{identity}.log').open('x') as log:
                        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    entry = dict(pid=child.pid, gpu=gpu, gpus=[gpu], started_unix=time.time(), argv=argv)
                    save(out / f'{identity}-launcher.json', entry)
                    active[identity] = dict(entry, process=child)
            pending_ids = [job['id'] for job in plan['jobs'] if job['id'] not in completed
                           and job['id'] not in failed and job['id'] not in active]
            # Use the previous occupancy sample: a newly launched worker cannot
            # release the pause until a later iteration observes its CUDA context.
            if not pending_ids and not released and all(uuids[e['gpu']] in busy for e in active.values()):
                if owns_pause:
                    pause.unlink()
                released = True
                save(out / 'throughput-resumed.json', dict(resumed_unix=time.time(), running=list(active),
                    note='All missing cells dispatched and current workers have acquired CUDA; throughput queue may reuse free devices.'))
            state = dict(updated_unix=time.time(), controller_pid=os.getpid(),
                stage='running' if active else ('waiting_for_gpu' if pending_ids else 'evaluations_finished'),
                target_cells=12, completed=completed, completed_count=len(completed), failed=failed,
                running={identity: {k: v for k, v in entry.items() if k != 'process'} for identity, entry in active.items()},
                pending=pending_ids, allowed_gpus=GPUS, throughput_dispatch_resumed=released,
                throughput_running=other['running'])
            if pending_ids and not owns_pause:
                state['stage'] = 'waiting_for_throughput_pause_ownership'
            save(out / 'status.json', state)
            if not pending_ids and not active:
                break
            time.sleep(15)
        with (out / 'collect-results.log').open('a') as log:
            code = subprocess.call(['/opt/conda/bin/python', str(BASE / 'collect_results.py'), '--partial'],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
        rows = [row for row in csv.DictReader((BASE / 'results.csv').open())
                if row['method'] == 'loopahead' and float(row['temperature']) == 1 and row['model'] in MODELS]
        state.update(stage='complete' if not failed and code == 0 and len(rows) == 12 else 'needs_attention',
                     finished_unix=time.time(), collect_exit_code=code, collected_t1_rows=len(rows))
        save(out / 'status.json', state)
        save(out / 'finished.json', state)
        assert state['stage'] == 'complete', state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--throughput', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert args.prepare != args.execute, 'Choose --prepare or --execute'
    assert not args.resume or args.execute
    if args.prepare:
        assert args.throughput is not None
        prepare(args.out.resolve(), args.throughput.resolve())
    else:
        execute(args.out.resolve(), args.resume)


if __name__ == '__main__':
    main()
