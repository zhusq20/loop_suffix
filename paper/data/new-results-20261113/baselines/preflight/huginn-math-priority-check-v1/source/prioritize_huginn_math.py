"""Start both long Huginn MATH cells before shorter unstarted Huginn cells.

Use the existing live deferral policy. Running workers, benchmark arguments,
the original job plan, and the four-GPU allocation remain authoritative.
Temporary deferrals are removed after both priority workers actually start,
or on a handled error/interruption. This observer never launches GPU work.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from collect_results import read_run
from result_evidence import validate_run_artifacts, sha
from orchestrate_formal import save
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE/'logs/huginn-math-priority-v1'
POLICY = BASE/'configs/gpu-allocation.json'
CONTROL = BASE/'logs/formal-controller-v1'
KEY = 'temporary_huginn_math_priority'


def read(path):
    return json.loads(path.read_text())


def make_plan():
    jobs = read(CONTROL/'plan.json')['jobs']
    priority = [f'dflash-huginn-3.5b-math500-t{t}' for t in [0, 1]]
    end = next(i for i, job in enumerate(jobs) if job['id'] == priority[-1])
    postponed = [j['id'] for j in jobs[:end] if j['kind'] == 'dflash'
                 and j['model'] == 'huginn-3.5b' and j['id'] not in priority]
    assert len(postponed) == 6
    state = read(CONTROL/'status.json')
    assert set(priority+postponed) <= set(state['pending'])
    protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
    references = []
    for job in jobs:
        if job['kind'] != 'dflash' or job['model'] != 'huginn-3.5b':
            continue
        path = BASE/f'runs/archived-ar-huginn-3.5b-{job["dataset"]}-t{job["temperature"]}'
        run = read_run(path, protocol)
        validate_run_artifacts(run)
        references.append(dict(job=job['id'], path=str(path), requests=len(run['rows']),
                               decode_s=run['summary']['decode_s'], requests_sha256=sha(path/'requests.jsonl')))
    ranked = sorted(references, key=lambda row: row['decode_s'], reverse=True)
    assert {r['job'] for r in ranked[:2]} == set(priority)
    return dict(owner=str(OUT), priority_jobs=priority, postponed_jobs=postponed,
                historical_AR=ranked, basis='Historical AR durations are scheduling proxies, not new benchmark results or predicted finish times.')


def marker(plan):
    return {key: plan[key] for key in ['owner', 'priority_jobs', 'postponed_jobs']}


def install(value, plan):
    value = copy.deepcopy(value)
    assert KEY not in value
    assert not set(plan['priority_jobs']+plan['postponed_jobs']) & set(value.get('deferred_jobs', []))
    value['deferred_jobs'] = value.get('deferred_jobs', [])+plan['postponed_jobs']
    value[KEY] = marker(plan)
    value['updated_unix'] = time.time()
    return value


def restore(value, plan):
    value = copy.deepcopy(value)
    assert value.get(KEY) == marker(plan), 'Temporary priority policy ownership changed'
    value['deferred_jobs'] = [job for job in value['deferred_jobs'] if job not in plan['postponed_jobs']]
    del value[KEY]
    value['updated_unix'] = time.time()
    return value


def started(identity, state):
    assert identity not in state['failed'], f'Priority job failed: {identity}'
    job_path = CONTROL/f'{identity}.json'
    if identity in state['running']:
        entry = state['running'][identity]
        proc = Path(f'/proc/{entry["pid"]}')
        if not proc.exists():
            return None  # The controller may still be collecting its exit.
        stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        assert proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')[-3:] == [str(BASE/'orchestrate_formal.py'), '--worker', str(job_path)]
        boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime ')))
        assert abs(boot+int(stat[19])/os.sysconf('SC_CLK_TCK')-entry['started_unix']) < 3
        return dict(kind='actual_live_worker', pid=entry['pid'], start_ticks=stat[19], gpus=entry['gpus'], job_sha256=sha(job_path))
    if identity in state['completed']:
        receipt = read(job_path.with_name(job_path.stem+'-worker-exit.json'))
        assert receipt['exit_code'] == 0 and receipt['job_sha256'] == sha(job_path)
        return dict(kind='actual_completed_worker', receipt=receipt)
    return None


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Signal {signum}; release owned temporary deferrals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = make_plan()
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    os.sched_setaffinity(0, set(range(80, 84)))
    signal.signal(signal.SIGTERM, interrupt)
    with open('/tmp/loopfill-huginn-math-priority.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        save(OUT/'plan.json', plan)
        freeze_sources(OUT, [Path(__file__), BASE/'gpu_allocation.py', BASE/'orchestrate_formal.py', POLICY])
        # Install before the training coordinator reaches its policy write.
        training = BASE/'checkpoints/dflash-huginn-narrow-formal-v1'
        assert not (training/'complete.json').exists() and read(training/'progress.json')['step'] < 8786
        before = read(POLICY)
        assert before['allowed_gpus'] == [2, 3, 4, 6] and before['released_gpus'] == [0, 1, 5, 7]
        assert before['reserved_for_development'] == [3, 6]
        try:
            after = install(before, plan)
            save(POLICY, after)
            save(OUT/'installed.json', dict(pid=os.getpid(), installed_unix=time.time(), before=before, after=after))
            while True:
                assert read(POLICY).get(KEY) == marker(plan), 'Priority configuration changed while waiting'
                failure = BASE/'logs/huginn-narrow-formal-queue-v1/failure.json'
                assert not failure.exists(), 'Narrow training failed; restore the original queue order'
                state = read(CONTROL/'status.json')
                evidence = {identity: started(identity, state) for identity in plan['priority_jobs']}
                save(OUT/'status.json', dict(stage='waiting_for_both_actual_workers', updated_unix=time.time(), evidence=evidence))
                if all(evidence.values()):
                    break
                time.sleep(10)
        except BaseException:
            save(OUT/'failure.json', dict(error=traceback.format_exc(), finished_unix=time.time()))
            raise
        finally:
            current = read(POLICY)
            if current.get(KEY) == marker(plan):
                restored = restore(current, plan)
                save(POLICY, restored)
                save(OUT/'restored.json', dict(restored_unix=time.time(), before=current, after=restored))
        save(OUT/'complete.json', dict(evidence=evidence, restored_policy_sha256=sha(POLICY), finished_unix=time.time()))
        print(json.dumps(dict(status='priority_workers_started_and_other_jobs_released', evidence=evidence)), flush=True)


if __name__ == '__main__':
    main()
