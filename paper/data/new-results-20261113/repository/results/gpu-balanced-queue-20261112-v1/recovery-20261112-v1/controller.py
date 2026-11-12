"""Retry one LoopAhead cell, backfill all eight GPUs, and reconcile DFlash.

The original DFlash parent reaps its own existing children under a dispatch
pause. This controller exclusively owns the remaining jobs and records their
actual exits. It retires the old parent only after all its workers have exited.
Frozen inference code, inputs, and checkpoints are never modified.
"""
import csv
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
ROOT = HERE.parents[1]
SUITE = ROOT / 'results/dflash-table12-packed-20261112-v2'
T1 = ROOT / 'results/loopahead-t1-resume-20261112-v1'
SQL = ROOT / 'results/loopfill-agentic-sql-table1-20261112'
RETRY_ID = 'loopahead-ouro-2.6b-specbench-t1'
RETRY_NAME = RETRY_ID + '-lockretry-v1'
CPUS = {g: f'{g*4 if g<4 else 48+(g-4)*4}-{g*4+3 if g<4 else 51+(g-4)*4}' for g in range(8)}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    proc = Path(f'/proc/{pid}')
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=stat[19], state=stat[0],
                argv=(proc / 'cmdline').read_bytes().decode().split('\0')[:-1])


def stop_idle_controller(expected):
    descriptor = os.pidfd_open(expected['pid'])
    try:
        actual = identity(expected['pid'])
        assert actual['start_ticks'] == expected['start_ticks'] and actual['argv'] == expected['argv']
        assert not read(SUITE / 'status.json')['active']
        assert not any(v['outcome'] == 'running' for v in read(SUITE / 'dispatch.json').values())
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        assert select.select([descriptor], [], [], 10)[0]
    finally:
        os.close(descriptor)


def reserved_sql_gpus():
    jobs = read(SQL / 'jobs.json')
    bindings = read(SQL / 'dispatch.json')['bindings']
    return {bindings[f"{j['model']}/t{j['temperature']}"] for j in jobs
            if not (SQL / 'runs' / j['id'] / 'summary.json').exists()}


def inventory():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,compute_mode',
                                    '--format=csv,noheader'], text=True).splitlines()
    return {int(p[0]): (p[1], p[2]) for line in lines if (p := [x.strip() for x in line.split(',')])}


def busy():
    return set(x.strip() for x in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).splitlines())


def retry_state(stage, **extra):
    value = dict(stage=stage, updated_unix=time.time(), **extra)
    save(HERE / 'retry-status.json', value)
    state = read(T1 / 'status.json')
    state.update(stage='complete' if stage == 'complete' else 'retry_' + stage,
                 updated_unix=time.time(), retry=value, retry_controller_pid=os.getpid())
    state.setdefault('historical_failures', {}).update(state.get('failed', {}))
    state['failed'].pop(RETRY_ID, None)
    state['pending'] = [RETRY_ID] if stage == 'waiting_for_gpu' else []
    state['running'] = {RETRY_ID: extra} if stage == 'running' else {}
    if stage == 'failed':
        state['failed'][RETRY_ID] = extra.get('error', 'Retry failed; inspect receipt')
    if stage == 'complete':
        if RETRY_ID not in state['completed']:
            state['completed'].append(RETRY_ID)
        state.update(completed_count=len(state['completed']), collected_t1_rows=extra['collected_t1_rows'],
                     collect_exit_code=0, finished_unix=time.time())
    save(T1 / 'status.json', state)
    if stage == 'complete':
        save(T1 / 'finished.json', state)


def validate_retry():
    sys.path.insert(0, str(BASE))
    from collect_results import read_run, validate_training
    from result_evidence import validate_run_artifacts
    run = read_run(BASE / 'runs' / RETRY_NAME,
                   read(BASE.parents[1] / 'results/loopfill-latency512-20261107/protocol.json'))
    assert run['meta']['model'] == 'ouro-2.6b' and run['meta']['dataset'] == 'specbench'
    assert run['meta']['method'] == 'loopahead' and run['meta']['temperature'] == 1
    result = validate_run_artifacts(run)
    validate_training(run)
    save(HERE / 'retry-audit.json', dict(status='passed', requests=len(run['rows']),
         evidence=result, summary=run['summary'], startup_wrapper_sha256=sha(HERE / 'lock_handoff.py')))


def main():
    plan = read(HERE / 'plan.json')
    assert sha(HERE / 'lock_handoff.py') == plan['lock_handoff_sha256']
    assert sha(Path(__file__)) == plan['controller_sha256']
    assert sha(SUITE / 'protocol.json') == plan['dflash_protocol_sha256']
    for relative, expected in read(T1 / 'plan.json')['source_sha256'].items():
        assert sha(BASE / relative) == expected, relative
    source = SUITE / 'source'
    sys.path[:0] = [str(source / 'scripts'), str(source / 'runtime')]
    import loop_dflash_benchmark_suite as dflash
    from native_cuda_env import native_cuda_environment
    env, driver = native_cuda_environment()
    save(HERE / 'native-driver.json', driver)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('WANDB_API_KEY', None)
    protocol = read(SUITE / 'protocol.json')
    with (HERE / 'controller.lock').open('a+') as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (HERE / 'status.json').exists(), 'Inspect existing queue before restarting'
        assert (SUITE / 'STOP').read_text() == plan['stop_content']
        while read(SUITE / 'status.json').get('updated_unix', 0) <= plan['paused_unix']:
            time.sleep(1)
        original = read(SUITE / 'dispatch.json')
        jobs = [j for j in read(SUITE / 'jobs.json') if j['id'] not in original]
        assert all(j['kind'] != 'validate' for j in jobs)
        assert all(original[f"validate-{j['model']}"]['outcome'] == 'complete' for j in jobs)
        refs = {key: list(csv.DictReader((SUITE / 'references' / f'{key}-measurements.csv').open()))
                for key in ['table1-t0', 'table1-t1', 'table2']}
        def cost(job):
            rows = refs[f"table1-t{job['temperature']}" if job['kind'] == 'latency' else 'table2']
            row = next(r for r in rows if r['model'] == job['model'] and r['arm'] == 'ar' and
                       (r['dataset'] == job['dataset'] if job['kind'] == 'latency' else int(r['batch']) == job['batch']))
            return float(row['decode_s' if job['kind'] == 'latency' else 'wall_s']) / 3
        jobs.sort(key=cost, reverse=True)
        save(HERE / 'migration.json', dict(original_dispatch=original, pending_jobs=jobs,
             scheduling='Retry first; then longest historical AR time first / 3, next available GPU.',
             estimates={j['id']: cost(j) for j in jobs}, frozen_at_unix=time.time()))
        overlay, active, retired = {}, {}, False
        retry = None
        legacy_lock = None
        assigned = {g: 0. for g in range(8)}
        retry_state('waiting_for_gpu', allowed_gpus=list(range(8)))
        while True:
            assert (SUITE / 'STOP').read_text() == plan['stop_content']
            for key, entry in list(active.items()):
                code = entry['process'].poll()
                if code is None:
                    continue
                entry['log'].close()
                entry['lock'].close()
                receipt = dict(entry['receipt'], exit_code=code, finished_unix=time.time())
                save(HERE / f'{key}-exit.json', receipt)
                if key == RETRY_ID:
                    save(BASE / 'logs' / f'{RETRY_NAME}-finished.json', receipt)
                    retry = dict(receipt, outcome='failed')
                    try:
                        assert code == 0, f'Retry exited {code}'
                        validate_retry()
                        with (HERE / 'collect-results.log').open('x') as log:
                            collected = subprocess.call(['/opt/conda/bin/python', str(BASE / 'collect_results.py'), '--partial'],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
                        assert collected == 0, f'Collection exited {collected}'
                        rows = [r for r in csv.DictReader((BASE / 'results.csv').open())
                                if r['method'] == 'loopahead' and float(r['temperature']) == 1 and
                                r['model'] in ['huginn-3.5b', 'ouro-2.6b']]
                        assert len(rows) == 12
                        retry['outcome'] = 'complete'
                        retry_state('complete', run=str(BASE / 'runs' / RETRY_NAME), collected_t1_rows=len(rows))
                    except Exception as error:
                        retry_state('failed', error=repr(error), receipt=receipt)
                else:
                    state = overlay[key]
                    outcome = 'failed'
                    if code == 0:
                        try:
                            dflash.audit(SUITE, entry['job'], SUITE / state['out'])
                            outcome = 'complete'
                        except Exception as error:
                            save(SUITE / state['out'] / 'audit-failure.json', dict(error=repr(error)))
                    elif (SUITE / state['out'] / 'failure.json').exists():
                        outcome = read(SUITE / state['out'] / 'failure.json')['outcome']
                    state.update(outcome=outcome, exit_code=code, finished_unix=receipt['finished_unix'])
                    save(SUITE / 'logs' / f'{key}-attempt{state["attempt"]}-exit.json', state)
                del active[key]
            old = read(SUITE / 'dispatch.json')
            if (not retired and not any(v['outcome'] == 'running' for v in old.values())
                    and not read(SUITE / 'status.json')['active']):
                for key, state in original.items():
                    if state['outcome'] == 'running':
                        assert (SUITE / 'logs' / f'{key}-attempt{state["attempt"]}-exit.json').exists()
                stop_idle_controller(plan['original_controller'])
                legacy_lock = (SUITE / 'controller.lock').open('a+')
                fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                retired = True
                save(HERE / 'original-controller-retired.json', dict(controller=plan['original_controller'],
                     retired_unix=time.time(), reason='All original child exits recorded; no workers signaled'))
            allocation = read(protocol['gpu_policy'])
            allowed = set(allocation['allowed_gpus']) - set(allocation.get('reserved_for_development', []))
            cards, occupied = inventory(), busy()
            reserved = reserved_sql_gpus()
            free = [g for g in range(8) if g in allowed and g not in reserved and
                    cards[g][1] == 'Exclusive_Process' and cards[g][0] not in occupied]
            free.sort(key=lambda g: (assigned[g], g))
            if not (HERE / 'STOP').exists():
                for gpu in free:
                    pending = [j for j in jobs if j['id'] not in overlay]
                    is_retry = retry is None and RETRY_ID not in active
                    if not is_retry and not pending:
                        break
                    uuid = cards[gpu][0]
                    lock = dflash.try_gpu_lock(uuid)
                    if lock is None:
                        continue
                    if uuid in busy():
                        lock.close()
                        continue
                    child_env = dict(env, CUDA_VISIBLE_DEVICES=uuid)
                    if is_retry:
                        key = RETRY_ID
                        argv = list(plan['retry_original_argv'])
                        for flag, value in [('--out', str(BASE / 'runs' / RETRY_NAME)), ('--gpu', str(gpu)), ('--cpus', CPUS[gpu])]:
                            argv[argv.index(flag)+1] = value
                        argv.insert(2, str(HERE / 'lock_handoff.py'))
                        child_env.update(PYTHONPATH=str(BASE), LOOPFILL_RETRY_LOCK_FD=str(lock.fileno()))
                        log = (BASE / 'logs' / f'{RETRY_NAME}.log').open('x')
                        child = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                                 pass_fds=(lock.fileno(),), start_new_session=True)
                        receipt = dict(name=RETRY_NAME, pid=child.pid, argv=argv, gpu=gpu,
                                       started_unix=time.time(), placement_amendment=str(HERE / 'plan.json'))
                        save(BASE / 'logs' / f'{RETRY_NAME}-launcher.json', receipt)
                        retry_state('running', **receipt, run=str(BASE / 'runs' / RETRY_NAME))
                        job = None
                        assigned[gpu] += 3600
                    else:
                        job = pending[0]
                        key = job['id']
                        out = Path('runs') / key / 'attempt1'
                        assert not (SUITE / out).exists()
                        argv = ['taskset', '--cpu-list', CPUS[gpu], protocol['python'], '-u',
                                str(source / 'scripts/loop_dflash_benchmark_worker.py'), '--suite', str(SUITE),
                                '--job', key, '--out', str(SUITE / out), '--lock-fd', str(lock.fileno())]
                        child_env['PYTHONPATH'] = os.pathsep.join(map(str, [source, source / 'src', source / 'scripts']))
                        log = (SUITE / 'logs' / f'{key}-attempt1.log').open('x')
                        child = subprocess.Popen(argv, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(lock.fileno(),), start_new_session=True, cwd=source)
                        receipt = dict(pid=child.pid, argv=argv, gpu=gpu, started_unix=time.time(),
                                       placement_amendment=str(HERE / 'plan.json'))
                        overlay[key] = dict(receipt, outcome='running', attempt=1, out=str(out), gpu_uuid=uuid)
                        assigned[gpu] += cost(job)
                    save(HERE / f'{key}-launcher.json', receipt)
                    active[key] = dict(process=child, lock=lock, log=log, receipt=receipt, job=job)
            save(HERE / 'dispatch.json', overlay)
            merged = dict(old, **overlay)
            if retired:
                save(SUITE / 'dispatch.json', merged)
                rows = dflash.summarize(SUITE)
                counts = {s: sum(r['outcome'] == s for r in rows) for s in ['pending','running','complete','failed','oom','interrupted']}
                save(SUITE / 'status.json', dict(status='running', counts=counts,
                     active={k:v for k,v in merged.items() if v['outcome'] == 'running'},
                     allowed_gpus=list(range(8)), planned=protocol['planned_formal_cells'],
                     placement_amendment=str(HERE / 'plan.json'), updated_unix=time.time()))
            pending_ids = [j['id'] for j in jobs if j['id'] not in overlay]
            state = dict(stage='running', controller_pid=os.getpid(), updated_unix=time.time(),
                         allowed_gpus=list(range(8)), original_controller_retired=retired,
                         running={k:v['receipt'] for k,v in active.items()}, pending=pending_ids,
                         original_running={k:v for k,v in old.items() if v['outcome'] == 'running' and k not in overlay},
                         retry=read(HERE / 'retry-status.json'), reserved_for_sql=sorted(reserved),
                         completed=[k for k,v in overlay.items() if v['outcome'] == 'complete'],
                         failed={k:v['outcome'] for k,v in overlay.items() if v['outcome'] in ['failed','oom']})
            save(HERE / 'status.json', state)
            if retired and not pending_ids and not active and retry is not None:
                terminal = {k:v['outcome'] for k,v in merged.items()}
                assert len(terminal) == len(read(SUITE / 'jobs.json'))
                save(SUITE / 'status.json', dict(status='finished', outcomes=terminal,
                     allowed_gpus=list(range(8)), updated_unix=time.time(), placement_amendment=str(HERE / 'plan.json')))
                state.update(stage='complete' if retry['outcome'] == 'complete' and
                    all(x == 'complete' for x in terminal.values()) else 'needs_attention', finished_unix=time.time())
                save(HERE / 'status.json', state)
                save(HERE / 'finished.json', state)
                assert (SUITE / 'STOP').read_text() == plan['stop_content']
                (SUITE / 'STOP').unlink()
                legacy_lock.close()
                return
            time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(HERE / 'failure.json', dict(failed_unix=time.time(), error=traceback.format_exc()))
        raise
