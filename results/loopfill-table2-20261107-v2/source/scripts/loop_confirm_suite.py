"""Paired Huginn B1--16 timing repeats with strict reuse and output audits."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

from loop_batch_sweep import check_resources, classify, gpu_snapshot
from loop_capacity_suite import digest, save, verify_suite
from loop_online_suite import audit_run

BATCHES = [1, 2, 4, 8, 16]
ARMS = ['ar', 'wavefront', 'loopspec-dual', 'loopspec-capped',
        'budget-suffix', 'budget-raw', 'budget-online']
PLACEMENT = {1: 0, 2: 1, 4: 2, 8: 5, 16: 3}


def key(batch, arm, repeat):
    return f'huginn-3.5b-b{batch}-r{repeat}-{arm}'


def arm_order(repeat):
    if repeat == 1:
        return list(reversed(ARMS))
    return ARMS[repeat:] + ARMS[:repeat]


def command(protocol, lane, job, out, smoke=False):
    cmd = list(protocol['template_command'])
    cmd[4] = str(Path(protocol['source']) / 'scripts/loop_batch_bench.py')
    values = {'--cpu-list': lane['cpus'], '--questions': protocol['questions'],
              '--arms': job['arm'], '--batch-sizes': job['batch'],
              '--repeat-id': job['repeat'], '--order-seeds': 0,
              '--suffix-c-initial': .7 if job['arm'] == 'budget-online' else 1.,
              '--max-new-tokens': 16 if smoke else 1024,
              '--reference-file': out / ('validation/references' if smoke else 'references') / 'huginn-3.5b.jsonl',
              '--out': out / ('validation' if smoke else 'runs') / job['key']}
    for option, value in values.items():
        cmd[cmd.index(option) + 1] = str(value)
    if smoke:
        cmd += ['--limit', '8']
    return cmd


def mismatch_reason(meta, protocol, job, smoke=False):
    """Completion alone is insufficient: source, input, GPU and semantics must match."""
    if meta['source_sha256'] != protocol['execution_source_sha256']:
        return 'execution_source'
    if meta['questions_sha256'] != protocol['questions_sha256']:
        return 'input'
    # PyTorch omits nvidia-smi's GPU- prefix for the same physical UUID.
    if str(meta.get('gpu_uuid')).removeprefix('GPU-') != str(job['gpu_uuid']).removeprefix('GPU-'):
        return 'physical_gpu'
    args = meta['args']
    checks = dict(protocol['execution_args'], arms=job['arm'], batch_sizes=str(job['batch']),
                  repeat_id=job['repeat'], max_new_tokens=16 if smoke else 1024,
                  limit=8 if smoke else None)
    for name, value in checks.items():
        if args.get(name) != value:
            return f'argument:{name}'
    if meta['configs'].get(job['arm']) != protocol['configs'][job['arm']]:
        return 'arm_config'
    if meta.get('profiled') or meta.get('controller_profiled'):
        return 'profiled'
    return None


def audit(out, protocol, job, smoke=False):
    meta = json.loads((out / 'metadata.json').read_text())
    reason = mismatch_reason(meta, protocol, job, smoke)
    if reason:
        raise ValueError(f'Incompatible measurement: {reason}: {out}')
    ids = protocol['request_ids'][:8] if smoke else protocol['request_ids']
    summary, = audit_run(out, set(ids), protocol.get('require_serial_reference', True))
    if summary['cap_skips'] or summary['attention_policy'] != 'packed_prefix_and_tree':
        raise ValueError('Capacity truncation or wrong backend')
    if summary['repeat_id'] != job['repeat'] or summary['order_seed'] != 0:
        raise ValueError('Wrong repeat/order')
    rows = read_rows(out / f"{job['arm']}-b{job['batch']}-seed0.jsonl")
    if job['arm'].startswith('budget-') or job['arm'] == 'loopspec-capped':
        if any(r['stats']['peak_draft_nodes'] > 16 for r in rows):
            raise ValueError('N=16 violated')
    if job['arm'] == 'budget-online' and not smoke:
        if sum(r['stats']['controller_updates'] for r in rows) <= 0:
            raise ValueError('Online controller has no updates')
    return summary


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(previous, out):
    from loop_batch_bench import arm_config
    previous, out = previous.resolve(), out.resolve()
    old = verify_suite(previous)
    if out.exists():
        raise FileExistsError(out)
    source = out / 'source'
    shutil.copytree(old['source'], source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(Path(__file__).resolve(), source / 'scripts/loop_confirm_suite.py')
    tests = Path(__file__).resolve().parents[1] / 'tests/test_loop_confirm_suite.py'
    shutil.copy2(tests, source / 'tests' / tests.name)
    for folder in ('jobs', 'runs', 'logs', 'validation', 'inputs', 'references'):
        (out / folder).mkdir()
    questions = out / 'inputs/huginn-3.5b.jsonl'
    shutil.copy2(old['inputs']['huginn-3.5b'], questions)
    shutil.copy2(previous / 'references/huginn-3.5b.jsonl', out / 'references/huginn-3.5b.jsonl')
    shutil.copytree(previous / 'validation/references', out / 'validation/references')
    template = json.loads((previous / 'jobs/huginn-3.5b-b8-online-c07-eta01.json').read_text())
    meta = json.loads((Path(template['output']) / 'metadata.json').read_text())
    lanes = [dict(next(l for l in old['lanes'] if l['gpu'] == gpu), batches=[batch])
             for batch, gpu in PLACEMENT.items()]
    configs = {arm: asdict(arm_config(arm, 32, 4, 8, 4, draft_node_budget=16,
               model_top_k=2, child_suffix_caps=(4, 4), suffix_eta=.1,
               suffix_c_initial=.7 if arm == 'budget-online' else 1.)) for arm in ARMS}
    # Normalize tuples to the JSON representation used in measurement metadata.
    configs = json.loads(json.dumps(configs))
    execution_keys = ['model', 'model_path', 'scoring', 'schedule', 'prefill_chunk_size',
        'attention_backend', 'cuda_graph', 'max_cuda_graphs', 'allocation_guard',
        'order_seeds', 'max_seq', 'buffer_policy', 'max_nodes', 'prompt_format',
        'dtype', 'seed', 'warmup_tokens', 'threads', 'check', 'profile', 'trace_controller',
        'progress_every']
    protocol = dict(version='huginn-b1-b16-confirm-v1', created_unix=time.time(),
        parent=str(previous), source=str(source), questions=str(questions),
        questions_sha256=digest(questions), request_ids=sorted(r['question_id'] for r in read_rows(questions)),
        execution_source_sha256=meta['source_sha256'],
        execution_args={k: meta['args'][k] for k in execution_keys}, configs=configs,
        template_command=template['command'], lanes=lanes, batches=BATCHES, arms=ARMS,
        repeats=[0, 1, 2], order_seed=0, planned_cells=105,
        primary_arms=[a for a in ARMS if a != 'loopspec-capped'],
        auxiliary_arms=['loopspec-capped'],
        repeat_policy='Three independent timing runs, fixed input order/initialization; report sample SD and paired ratios. Repeat 1 reverses method order, repeat 2 rotates it. No claim that timing repeats remove data-selection bias.',
        reuse_policy='Same execution source identity, full configuration, input, physical GPU and repeat ID; audit full480 coverage. Adopt current online r0 after its GPU queue releases. Never overwrite prior artifacts.',
        capacity_policy='Actual full480 OOM recorded once for each B/method; subsequent timing repeats marked not_repeated_capacity_failure. No claimed speedup against an OOM method.',
        backend_policy='Existing frozen dense-KV backend; no paging or cross-recurrence sharing change.',
        baseline_scope='Same-engine implementations; loopspec-dual is the transferred original schedule, loopspec-capped N16 is a separately named tuned auxiliary control, not the official Raven checkpoint.',
        output_policy='Report serial reference agreement, matched optimized AR agreement per B/repeat, repeat determinism, output lengths and per-category coverage. These are not task quality scores.',
        validation_policy='Prior unchanged CUDA gates retained; each new B/arm gets a real-checkpoint 8-request/16-token smoke before full480. Smoke OOM still permits the full allocation attempt; other smoke errors block that configuration.',
        prior_cuda_validation=str(previous / 'validation'),
        prior_numeric_diagnostic=str(previous.parent / 'loop-output-diagnostic-20261104/report.json'),
        selection_scope='Q4 N16 c0=.7 eta=.1 selected on the same SpecBench480. This suite strengthens timing evidence, not held-out confirmation.',
        outstanding_scientific_checks=['Independent data confirmation', 'Task quality evaluation',
            'Full-population numerical divergence attribution; existing short diagnostics do not prove all BF16 differences benign'])
    for name, sha in meta['source_sha256'].items():
        assert digest(source / name) == sha, name
    assert len(protocol['request_ids']) == len(set(protocol['request_ids'])) == 480
    save(out / 'protocol.json', protocol)
    candidates = []
    for path in sorted(previous.parent.glob('*/jobs/*.json')):
        j = json.loads(path.read_text())
        if j.get('model') == 'huginn-3.5b' and j.get('outcome') == 'complete' and j.get('output'):
            candidates.append((path, j))
    reuse = []
    for lane in lanes:
        batch = lane['batches'][0]
        for repeat in protocol['repeats']:
            for arm in arm_order(repeat):
                job = dict(key=key(batch, arm, repeat), batch=batch, arm=arm, repeat=repeat,
                           gpu=lane['gpu'], gpu_uuid=lane['uuid'], outcome='pending')
                if repeat == 0 and arm == 'budget-online':
                    job.update(outcome='adopt_pending', reuse_from=str(previous / 'jobs' /
                        f'huginn-3.5b-b{batch}-online-c07-eta01.json'))
                for path, candidate in candidates:
                    if candidate.get('batch') != batch or candidate.get('arm') != arm:
                        continue
                    target = Path(candidate['output'])
                    measured_meta = json.loads((target / 'metadata.json').read_text())
                    if mismatch_reason(measured_meta, protocol, job):
                        continue
                    audit(target, protocol, job)
                    job.update(outcome='complete', output=str(target), reused=True, reuse_from=str(path))
                    reuse.append(dict(key=job['key'], source=str(path), output=str(target),
                                      metadata_sha256=digest(target / 'metadata.json')))
                    break
                save(out / 'jobs' / f"{job['key']}.json", job)
    save(out / 'reuse-provenance.json', dict(reused=reuse))
    save(out / 'source-hashes.json', {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
    save(out / 'input-hashes.json', {str(p): digest(p) for folder in ('inputs', 'references', 'validation/references') for p in (out / folder).glob('*.jsonl')})
    verify_suite(out)
    summarize(out)
    print(json.dumps(dict(out=str(out), cells=105, reused=len(reuse))))


def dump_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(out):
    jobs = [json.loads(p.read_text()) for p in sorted((out / 'jobs').glob('*.json'))]
    records, measurements = {}, []
    for j in jobs:
        row = {k: j[k] for k in ('key', 'batch', 'arm', 'repeat', 'gpu', 'outcome')}
        row.update(reused=j.get('reused', False), tok_s=None, wall_s=None, output_tokens=None,
            requests=None, same_as_serial_ar=None, same_as_batched_ar=None,
            same_as_repeat0=None, peak_allocated_gib=None, rows_per_token=None,
            controller_updates=None, output=j.get('output'), error=j.get('error'))
        if j['outcome'] == 'complete':
            target = Path(j['output'])
            prefix = f"{j['arm']}-b{j['batch']}-seed0"
            s = json.loads((target / f'{prefix}-summary.json').read_text())
            rs = {r['request_id']: r for r in read_rows(target / f'{prefix}.jsonl')}
            records[(j['batch'], j['arm'], j['repeat'])] = rs
            row.update(tok_s=s['output_tput'], wall_s=s['wall_s'], output_tokens=s['output_tokens'],
                requests=s['requests'], same_as_serial_ar=s.get('same_as_serial_ar'),
                peak_allocated_gib=s['peak_allocated_bytes'] / 2**30,
                rows_per_token=s['core_rows'] / s['output_tokens'],
                controller_updates=sum(r['stats'].get('controller_updates', 0) for r in rs.values()))
        measurements.append(row)
    categories = []
    for row in measurements:
        b, arm, repeat = row['batch'], row['arm'], row['repeat']
        rs = records.get((b, arm, repeat))
        if rs is None:
            continue
        ar = records.get((b, 'ar', repeat))
        first = records.get((b, arm, 0))
        if ar:
            row['same_as_batched_ar'] = sum(r['tokens'] == ar[rid]['tokens'] for rid, r in rs.items())
        if first:
            row['same_as_repeat0'] = sum(r['tokens'] == first[rid]['tokens'] for rid, r in rs.items())
        groups = defaultdict(list)
        for rid, r in rs.items():
            groups[r.get('category', 'unknown')].append((rid, r))
        for category, group in groups.items():
            categories.append(dict(batch=b, arm=arm, repeat=repeat, category=category,
                requests=len(group), output_tokens=sum(len(r['tokens']) for _, r in group),
                same_as_batched_ar=sum(r['tokens'] == ar[rid]['tokens'] for rid, r in group) if ar else None))
    aggregate = []
    by_job = {(r['batch'], r['arm'], r['repeat']): r for r in measurements if r['outcome'] == 'complete'}
    for batch in BATCHES:
        for arm in ARMS:
            group = [r for r in measurements if r['batch'] == batch and r['arm'] == arm and r['outcome'] == 'complete']
            speeds = [r['tok_s'] for r in group]
            ratios = [r['tok_s'] / by_job[(batch, 'ar', r['repeat'])]['tok_s'] for r in group if (batch, 'ar', r['repeat']) in by_job]
            raw_ratios = [r['tok_s'] / by_job[(batch, 'budget-raw', r['repeat'])]['tok_s'] for r in group if (batch, 'budget-raw', r['repeat']) in by_job]
            aggregate.append(dict(batch=batch, arm=arm, repeats_complete=len(group),
                tok_s_mean=statistics.mean(speeds) if speeds else None,
                tok_s_sd=statistics.stdev(speeds) if len(speeds) > 1 else None,
                tok_s_min=min(speeds) if speeds else None, tok_s_max=max(speeds) if speeds else None,
                wall_s_mean=statistics.mean(r['wall_s'] for r in group) if group else None,
                ar_pairs=len(ratios), ar_speedup_mean=statistics.mean(ratios) if ratios else None,
                raw_pairs=len(raw_ratios), raw_speedup_mean=statistics.mean(raw_ratios) if raw_ratios else None,
                outputs_repeat_identical=all(r['same_as_repeat0'] == 480 for r in group) if group and all(r['same_as_repeat0'] is not None for r in group) else None))
    dump_csv(out / 'measurements.csv', measurements)
    dump_csv(out / 'aggregate.csv', aggregate)
    dump_csv(out / 'category-output-audit.csv', categories)
    save(out / 'summary.json', dict(updated_unix=time.time(), counts=dict(Counter(j['outcome'] for j in jobs)),
        aggregate=aggregate, note='SD is timing repeat variation for one fixed order. Output agreement is not a task accuracy metric. OOM is a backend capacity result.'))


def run(out):
    out = out.resolve()
    protocol = verify_suite(out)
    guard = (out / 'supervisor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mutex, stop = threading.RLock(), threading.Event()
    status = dict(phase='running', supervisor_pid=os.getpid(), started_unix=time.time(), planned_cells=105, lanes={})

    def update(gpu=None, refresh=False, **fields):
        with mutex:
            (status if gpu is None else status['lanes'].setdefault(str(gpu), {})).update(fields)
            status['updated_unix'] = time.time()
            status['counts'] = dict(Counter(json.loads(p.read_text())['outcome'] for p in (out / 'jobs').glob('*.json')))
            save(out / 'status.json', status)
            if refresh:
                summarize(out)

    def heartbeat():
        while not stop.wait(30):
            try:
                snapshots = [gpu_snapshot(l['gpu']) for l in protocol['lanes']]
                with (out / 'gpu-telemetry.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(time=time.time(), gpus=snapshots)) + '\n')
                update()
            except Exception as exc:
                update(heartbeat_error=repr(exc))

    def launch(lane, job, smoke=False):
        verify_suite(out)
        cmd = command(protocol, lane, job, out, smoke)
        target = Path(cmd[cmd.index('--out') + 1])
        log = out / 'logs' / f"{'validation-' if smoke else ''}{job['key']}.log"
        if target.exists() or log.exists():
            raise RuntimeError(f'Incomplete artifact preserved; inspect before retry: {target}')
        check_resources(lane)
        source = Path(protocol['source'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONPATH=str(source / 'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)
        job.update(outcome='validating' if smoke else 'running', command=cmd, output=str(target),
                   log=str(log), started_unix=time.time())
        with log.open('x') as handle:
            child = subprocess.Popen(cmd, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            job['pid'] = child.pid
            save(out / 'jobs' / f"{job['key']}.json", job)
            update(lane['gpu'], phase=job['outcome'], current=job['key'], pid=child.pid)
            rc = child.wait()
        result = classify(rc, target)
        if result == 'complete':
            audit(target, protocol, job, smoke)
        elif result != 'oom':
            job['error'] = log.read_text()[-3000:]
        if smoke:
            save(out / 'validation' / f"{job['key']}-validation.json", dict(outcome=result, output=str(target), returncode=rc))
        job.update(outcome=result, returncode=rc, finished_unix=time.time())
        return result

    def worker(lane):
        gpu, batch = lane['gpu'], lane['batches'][0]
        with open(f'/tmp/loopfill-suite-gpu{gpu}.lock', 'a') as gpu_lock:
            update(gpu, phase='waiting_for_existing_gpu_queue')
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            for repeat in protocol['repeats']:
                for arm in arm_order(repeat):
                    path = out / 'jobs' / f'{key(batch, arm, repeat)}.json'
                    job = json.loads(path.read_text())
                    if job['outcome'] == 'complete':
                        audit(Path(job['output']), protocol, job)
                        continue
                    if job['outcome'] == 'adopt_pending':
                        original = json.loads(Path(job['reuse_from']).read_text())
                        if original['outcome'] != 'complete':
                            raise RuntimeError(f'Prior online job did not complete: {job["reuse_from"]}')
                        audit(Path(original['output']), protocol, job)
                        job.update(outcome='complete', output=original['output'], reused=True)
                        save(path, job)
                        update(refresh=True)
                        continue
                    if job['outcome'] != 'pending':
                        if job['outcome'] in ('running', 'validating'):
                            raise RuntimeError(f'Unreconciled prior process/artifact: {job["key"]}')
                        continue
                    if repeat:
                        first = json.loads((out / 'jobs' / f'{key(batch, arm, 0)}.json').read_text())
                        if first['outcome'] != 'complete':
                            job.update(outcome='not_repeated_capacity_failure' if first['outcome'] == 'oom' else 'blocked_by_first_run',
                                       first_run=first['key'])
                            save(path, job)
                            update(refresh=True)
                            continue
                    try:
                        if repeat == 0:
                            result = launch(lane, job, smoke=True)
                            job['validation_output'] = job['output']
                            job['validation_outcome'] = result
                            if result not in ('complete', 'oom'):
                                job['outcome'] = 'validation_failed'
                                save(path, job)
                                update(refresh=True)
                                continue
                        launch(lane, job)
                    except Exception as exc:
                        job.update(outcome='failed', error=repr(exc), finished_unix=time.time())
                    save(path, job)
                    update(refresh=True)
            update(gpu, phase='complete', current=None, pid=None)

    update(refresh=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    errors = []
    try:
        with ThreadPoolExecutor(max_workers=len(protocol['lanes'])) as pool:
            for task in as_completed([pool.submit(worker, lane) for lane in protocol['lanes']]):
                try:
                    task.result()
                except Exception as exc:
                    errors.append(repr(exc))
                    update(errors=errors)
        bad = sum(status['counts'].get(k, 0) for k in ('failed', 'validation_failed', 'blocked_by_first_run', 'pending', 'adopt_pending', 'running', 'validating'))
        update(phase='finished_with_errors' if errors or bad else 'complete', errors=errors, refresh=True)
    finally:
        stop.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--previous', type=Path, default=Path('results/loopfill-online-batch-huginn-20261104'))
    parser.add_argument('--out', type=Path, default=Path('results/loopfill-confirm-b1-b16-huginn-20261104'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.previous, args.out)
    elif args.action == 'run':
        run(args.out)
    else:
        summarize(args.out)
