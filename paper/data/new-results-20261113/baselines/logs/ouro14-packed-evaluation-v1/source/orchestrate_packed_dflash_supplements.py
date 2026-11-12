"""Fill missing AR/profile evidence after the existing packed measurement queue."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from native_cuda_env import native_cuda_environment
from packed_dflash_evidence import (check_plan, check_profile, check_run, compare,
                                    read, save, sha, timed_evidence)


def cpus(gpu):
    first = gpu * 4 if gpu < 4 else 48 + (gpu - 4) * 4
    return f'{first}-{first + 3}'


def cards():
    value = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,compute_mode',
                                     '--format=csv,noheader'], text=True)
    return {int(p[0]): (p[1], p[2]) for line in value.splitlines()
            if (p := [x.strip() for x in line.split(',')])}


def occupied():
    return set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True).splitlines())


def aggregate(root, suite, jobs, state):
    result, agreement = [], {}
    for job in jobs:
        key = job['id']
        if not all(state.get(key + '-' + stage, {}).get('outcome') == 'complete' for stage in ['ar', 'profile']):
            continue
        ar_state, profile_state = [state[key + '-' + stage] for stage in ['ar', 'profile']]
        ar = check_run(root / ar_state['out'], suite, job, stage='ar')
        profile = check_run(root / profile_state['out'], suite, job, stage='profile')
        binding = read(root / f'bindings/{key}-ar.json')
        other_binding = read(root / f'bindings/{key}-profile.json')
        assert binding['timed'] == other_binding['timed']
        method = check_run(binding['timed']['path'], suite, job)
        assert method['hashes'] == binding['timed']['hashes']
        check_profile(profile['rows'], method['rows'])
        pair = compare(method['rows'], ar['rows'])
        agreement[key] = pair
        a, m, p = ar['summary'], method['summary'], profile['summary']
        limitation = binding['timed']['exit_evidence'].get('limitation', '')
        result.append(dict(method='dflash', model=job['model'], benchmark=job['dataset'],
            temperature=job['temperature'], eta=m['decode_token_s'] / a['decode_token_s'],
            eta_latency=a['decode_s'] / m['decode_s'], eta_throughput=m['decode_token_s'] / a['decode_token_s'],
            gamma=m['gamma'], method_tok_s=m['decode_token_s'], ar_tok_s=a['decode_token_s'],
            drafter_time_frac=p['drafter_time_frac'], match_rate=pair['match_rate'],
            aligned_token_match_rate=pair['aligned_token_match_rate'], requests=job['count'],
            engine='official_dflash_with_packed_loop_target', status='measured_diagnostics_pending',
            method_run=method['path'], ar_run=ar['path'], profile_run=profile['path'],
            timed_process_exit=binding['timed']['exit_evidence']['actual_exit_code'],
            notes='Fresh own-stack AR; timed DFlash reused; separate instrumented profile matches every token and verification. '
                  'Both eta definitions retained; final legacy table uses eta_latency. ' + limitation))
    save(root / 'agreements.json', agreement)
    save(root / 'measurement-results.json', result)
    if result:
        temporary = root / f'results.{os.getpid()}.tmp'
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        temporary.replace(root / 'results.csv')
    return len(result)


def main(plan_path):
    plan = check_plan(plan_path)
    root, suite = plan_path.parent, Path(plan['suite'])
    # The standalone recovery helper is copied and hashed with this controller.
    sys.path.insert(0, str(root / 'source'))
    from loop_queue_recovery import UNKNOWN_EXIT, ObservedWorker, spawn_supervised
    with (root / 'controller.lock').open('a+') as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment, driver = native_cuda_environment()
        environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
            WANDB_MODE='disabled')
        environment.pop('WANDB_API_KEY', None)
        save(root / 'native-driver.json', driver)
        state = read(root / 'dispatch.json') if (root / 'dispatch.json').exists() else {}
        jobs = plan['jobs']
        entries = {j['id'] + '-' + stage: (j, stage) for stage in ['ar', 'profile'] for j in jobs}
        active, evidence = {}, {}
        # Adopt acknowledged launches, including one made immediately before a crash.
        for key in entries:
            launch_path, exit_path = root / f'logs/{key}-launcher.json', root / f'logs/{key}-exit.json'
            if launch_path.exists() and state.get(key, {}).get('outcome') not in ('complete', 'failed', 'exit_unavailable'):
                launch = read(launch_path)
                state[key] = dict(launch, outcome='running', out=f'runs/{key}')
                active[key] = dict(process=ObservedWorker(launch, exit_path), lock=None, log=None)
        completed_cells = aggregate(root, suite, jobs, state)
        while True:
            changed = False
            for key, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    continue
                for handle in ('lock', 'log'):
                    if item[handle] is not None:
                        item[handle].close()
                record = state[key]
                record.update(outcome='failed' if code != UNKNOWN_EXIT else 'exit_unavailable',
                              exit_code=code if code != UNKNOWN_EXIT else None, finished_unix=time.time())
                if code == 0:
                    try:
                        job, stage = entries[key]
                        checked = check_run(root / record['out'], suite, job, stage=stage)
                        binding = read(root / f'bindings/{key}.json')
                        assert checked['metadata']['binding_sha256'] == sha(root / f'bindings/{key}.json')
                        assert checked['metadata']['plan_sha256'] == sha(plan_path)
                        assert checked['metadata']['supplement_source_sha256'] == plan['source_sha256']
                        if stage == 'profile':
                            method = check_run(binding['timed']['path'], suite, job)
                            assert method['hashes'] == binding['timed']['hashes']
                            check_profile(checked['rows'], method['rows'])
                        record['outcome'] = 'complete'
                        record['artifact_sha256'] = checked['hashes']
                    except Exception:
                        save(root / record['out'] / 'audit-failure.json', dict(error=traceback.format_exc()))
                del active[key]
                changed = True
                save(root / 'dispatch.json', state)
            if changed:
                completed_cells = aggregate(root, suite, jobs, state)
            parent = read(Path(plan['balanced_queue']) / 'status.json')
            dispatch = read(suite / 'dispatch.json')
            dispatch.update(read(Path(plan['balanced_queue']) / 'dispatch.json'))
            awaiting = []
            for job in jobs:
                if job['id'] not in evidence:
                    found = timed_evidence(suite, dispatch, job)
                    if found:
                        evidence[job['id']] = found
                    else:
                        awaiting.append(job['id'])
            pending = [key for key, (job, stage) in entries.items() if key not in state and job['id'] in evidence]
            # Existing timed measurements retain priority. Use only GPUs that
            # become free, respecting the shared reservation/ownership policy.
            if not parent['pending'] and not (root / 'STOP').exists():
                policy = read(plan['gpu_policy'])
                allowed = set(policy['allowed_gpus']) - set(policy.get('reserved_for_development', []))
                busy = occupied()
                for gpu, (uuid, mode) in sorted(cards().items()):
                    if gpu not in allowed or uuid in busy or mode != 'Exclusive_Process' or not pending:
                        continue
                    lock = open(f'/tmp/loopfill-{uuid}.lock', 'a+')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                        continue
                    if uuid in occupied():
                        lock.close()
                        continue
                    check_plan(plan_path)
                    key = pending.pop(0)
                    job, stage = entries[key]
                    binding_path = root / f'bindings/{key}.json'
                    assert not binding_path.exists(), f'Inspect unlaunched binding before resuming: {key}'
                    save(binding_path, dict(job=job, stage=stage, timed=evidence[job['id']],
                        plan_sha256=sha(plan_path), created_unix=time.time()))
                    output = root / f'runs/{key}'
                    argv = ['taskset', '--cpu-list', cpus(gpu), plan['python'], '-u',
                        str(root / 'source/supplement_packed_dflash.py'), '--plan', str(plan_path),
                        '--binding', str(binding_path), '--job', job['id'], '--stage', stage,
                        '--out', str(output), '--lock-fd', str(lock.fileno())]
                    log = (root / f'logs/{key}.log').open('x')
                    child_env = dict(environment, CUDA_VISIBLE_DEVICES=uuid,
                        PYTHONPATH=os.pathsep.join(map(str, [root / 'source', suite / 'source', suite / 'source/src'])))
                    child = spawn_supervised(argv, env=child_env, log=log, lock=lock,
                        launcher_path=root / f'logs/{key}-launcher.json', exit_path=root / f'logs/{key}-exit.json',
                        metadata=dict(job_id=job['id'], stage=stage, gpu=gpu, gpu_uuid=uuid,
                                      supplement_plan=str(plan_path)), cwd=root / 'source')
                    state[key] = dict(child.launch, outcome='running', out=f'runs/{key}')
                    active[key] = dict(process=child, lock=lock, log=log)
                    busy.add(uuid)
                    save(root / 'dispatch.json', state)
            save(root / 'dispatch.json', state)
            failures = {k: v['outcome'] for k, v in state.items() if v['outcome'] not in ('complete', 'running')}
            save(root / 'status.json', dict(stage='running' if active else 'waiting_for_existing_measurements',
                controller_pid=os.getpid(), updated_unix=time.time(), completed_cells=completed_cells,
                expected_cells=len(jobs), expected_supplement_runs=len(entries),
                running={k: state[k] for k in active}, awaiting_timed_measurements=awaiting,
                existing_queue_pending=parent['pending'], ready_supplements=pending, failures=failures))
            if len(state) == len(entries) and not active:
                save(root / 'complete.json', dict(status='complete' if not failures else 'needs_inspection',
                    completed_cells=completed_cells, expected_cells=len(jobs), failures=failures,
                    finished_unix=time.time(), limitation='Supplementary measurements only; greedy diagnostics and final source-aware merge remain separate.'))
                return
            if not active and awaiting and parent.get('stage') in ('complete', 'needs_attention'):
                terminal_missing = [key for key in awaiting if dispatch.get(key, {}).get('outcome') in ('failed', 'oom', 'interrupted')]
                if terminal_missing and not pending:
                    save(root / 'dependency-failure.json', dict(job_ids=terminal_missing, finished_unix=time.time()))
                    return
            time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.plan.resolve())
    except Exception:
        save(args.plan.parent / 'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
        raise
