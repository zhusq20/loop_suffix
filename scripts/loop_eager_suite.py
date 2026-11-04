"""Six-method Huginn full480 rerun, B1--16, without graphs or execution padding."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import loop_confirm_suite as confirmation
from loop_batch_sweep import check_resources, classify, gpu_snapshot
from loop_capacity_suite import digest, save, verify_suite

ARMS = ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-raw', 'budget-online']
BATCHES = [1, 2, 4, 8, 16]
REPO = Path(__file__).resolve().parents[1]


def summarize(out):
    # Reuse the existing output/serial-AR audits, restricted to these six methods.
    confirmation.ARMS = ARMS
    confirmation.summarize(out)
    path = out / 'summary.json'
    summary = json.loads(path.read_text())
    summary['note'] = ('One timing run per cell, fixed order, CUDA Graph and execution '
                       'padding disabled. Output agreement is not a task accuracy score.')
    save(path, summary)


def command(protocol, lane, job, out, smoke=False):
    return confirmation.command(protocol, lane, job, out, smoke)


def audit(out, protocol, job, smoke=False):
    summary = confirmation.audit(out, protocol, job, smoke)
    for name in ('cuda_graph', 'pad_decode_rows', 'cuda_graph_shapes',
                 'cuda_graph_replays', 'core_padding_rows', 'padding_sink_bytes'):
        if summary[name] != 0:
            raise ValueError(f'Graph/padding disabled contract violated: {name}')
    meta = json.loads((out / 'metadata.json').read_text())
    if meta['warmup_request_limit_factor'] != 2:
        raise ValueError('Warmup request count changed')
    return summary


def migrate_reference(previous, out, source, source_hashes, folder):
    """Copy unchanged serial outputs with an explicit source-identity migration audit."""
    origin = previous / folder / 'huginn-3.5b.jsonl'
    lines = origin.read_text().splitlines(keepends=True)
    header = json.loads(lines[0])
    old_hashes = header['reference_identity']['source_sha256']
    changed = {name for name, sha in old_hashes.items() if source_hashes.get(name) != sha}
    allowed = {'scripts/loop_batch_bench.py', 'src/llm_memory_editability/loop_optimized_decode.py'}
    if changed - allowed or set(source_hashes) != set(old_hashes):
        raise ValueError(f'Serial reference dependencies changed: {changed - allowed}')
    old_source = Path(json.loads((previous / 'protocol.json').read_text())['source'])
    old_bench = old_source / 'scripts/loop_batch_bench.py'
    if digest(old_bench) != old_hashes['scripts/loop_batch_bench.py']:
        # References can originate from an earlier frozen suite.
        matches = [p for p in previous.parent.glob('*/source/scripts/loop_batch_bench.py')
                   if digest(p) == old_hashes['scripts/loop_batch_bench.py']]
        if not matches:
            raise ValueError('Cannot locate serial reference source')
        old_bench = matches[0]
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    before, after = functions(old_bench), functions(source / 'scripts/loop_batch_bench.py')
    checked = ['arm_config', 'load_requests', 'enforce_precision', 'serial_reference']
    for name in checked:
        if before[name] != after[name]:
            raise ValueError(f'Serial reference function changed: {name}')
    header['reference_identity']['source_sha256'] = source_hashes
    target = out / folder / origin.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(header) + '\n' + ''.join(lines[1:]))
    return dict(origin=str(origin), origin_sha256=digest(origin), target=str(target),
                target_sha256=digest(target), changed_sources=sorted(changed),
                checked_functions=checked, original_benchmark=str(old_bench),
                tokens_unchanged=True, reason='All serial decoder modules and request/precision/reference functions unchanged; changes affect packed execution and CLI/warmup only.')


def prepare(previous, out):
    from loop_batch_bench import arm_config, parser
    previous, out = previous.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    old = json.loads((previous.parent / 'loopfill-confirm-b1-b16-huginn-20261104/protocol.json').read_text())
    assert old['primary_arms'] == ARMS
    source = out / 'source'
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(REPO / folder, source / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder in ('jobs', 'runs', 'logs', 'validation', 'inputs', 'references'):
        (out / folder).mkdir(exist_ok=True)
    questions = out / 'inputs/huginn-3.5b.jsonl'
    shutil.copy2(previous / 'inputs/huginn-3.5b.jsonl', questions)
    ids = [r['question_id'] for r in confirmation.read_rows(questions)]
    assert len(ids) == len(set(ids)) == 480
    hashes = {str(p.relative_to(source)): digest(p) for p in
              [source / 'scripts/loop_batch_bench.py'] + sorted((source / 'src/llm_memory_editability').glob('*.py'))}
    references = [migrate_reference(previous, out, source, hashes, f)
                  for f in ('references', 'validation/references')]
    save(out / 'reference-provenance.json', references)
    template = [a for a in old['template_command'] if a != '--cuda-graph']
    template += ['@' + str(source / 'configs/loopfill-eager.args')]
    configs = {arm: asdict(arm_config(arm, 32, 4, 8, 4, draft_node_budget=16,
               model_top_k=2, child_suffix_caps=(4, 4), suffix_eta=.1,
               suffix_c_initial=.7 if arm == 'budget-online' else 1.)) for arm in ARMS}
    configs = json.loads(json.dumps(configs))
    assert configs == {arm: old['configs'][arm] for arm in ARMS}
    lanes = []
    for gpu in range(8):
        snapshot = gpu_snapshot(gpu)
        lane = dict(gpu=gpu, uuid=snapshot['uuid'], cpus=f'{gpu*4}-{gpu*4+3}')
        check_resources(lane)
        lanes.append(lane)
    execution = dict(old['execution_args'], cuda_graph=False, pad_decode_rows=False,
                     warmup_request_limit_factor=2, verifier_shape_warmup='complete')
    protocol = dict(version='huginn-six-arms-eager-b1-b16-v1', created_unix=time.time(),
        parent=str(previous), source=str(source), questions=str(questions),
        questions_sha256=digest(questions), request_ids=ids,
        execution_source_sha256=hashes, execution_args=execution, configs=configs,
        template_command=template, lanes=lanes, batches=BATCHES, arms=ARMS,
        repeats=[0], planned_cells=30,
        scheduling='Shared queue on eight exclusive GPUs; B1 first, then B2/B4/B8/B16. No repeated timings.',
        execution_changes=['CUDA Graph disabled', 'Execution row padding disabled'],
        warmup_policy='Keep 32-token, 2*B request warmup; no graph capture or synthetic graph-shape priming.',
        failure_policy='Short checkpoint validation before each full480 attempt. Validation OOM still permits full480; other validation failures block that cell. Full480 OOM is recorded; other cells continue.',
        reference_policy='Reuse serial AR tokens only after unchanged serial dependency/function checks; migration provenance recorded separately.',
        reuse_policy='All 30 timing cells are new; no previous timing or OOM reused.',
        prior_extra_arm='loopspec-capped was auxiliary; excluded from the six primary methods.')
    save(out / 'protocol.json', protocol)
    for batch in BATCHES:
        for arm in ARMS:
            job = dict(key=confirmation.key(batch, arm, 0), model='huginn-3.5b', batch=batch,
                       arm=arm, repeat=0, gpu=None, gpu_uuid=None, outcome='pending')
            cmd = command(protocol, lanes[0], job, out)
            args = vars(parser().parse_args(cmd[5:]))
            assert all(args[k] == v for k, v in execution.items())
            save(out / 'jobs' / f"{job['key']}.json", job)
    save(out / 'source-hashes.json', {str(p.relative_to(source)): digest(p)
                                    for p in source.rglob('*') if p.is_file()})
    save(out / 'input-hashes.json', {str(p): digest(p) for folder in
         ('inputs', 'references', 'validation/references') for p in (out / folder).glob('*.jsonl')})
    verify_suite(out)
    summarize(out)
    print(json.dumps(dict(out=str(out), planned_cells=30, gpus=8)))


def run(out):
    out = out.resolve()
    protocol = verify_suite(out)
    guard = (out / 'supervisor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = queue.Queue()
    for batch in BATCHES:
        for arm in ARMS:
            job = json.loads((out / 'jobs' / f'{confirmation.key(batch, arm, 0)}.json').read_text())
            if job['outcome'] in ('running', 'validating'):
                raise RuntimeError(f'Unreconciled previous process: {job["key"]}')
            if job['outcome'] == 'pending':
                pending.put(job)
    mutex, stop = threading.RLock(), threading.Event()
    status = dict(phase='running', supervisor_pid=os.getpid(), started_unix=time.time(),
                  planned_cells=30, lanes={})

    def update(gpu=None, refresh=False, **fields):
        with mutex:
            (status if gpu is None else status['lanes'].setdefault(str(gpu), {})).update(fields)
            status['updated_unix'] = time.time()
            status['counts'] = dict(Counter(json.loads(p.read_text())['outcome']
                                           for p in (out / 'jobs').glob('*.json')))
            save(out / 'status.json', status)
            if refresh:
                summarize(out)

    def heartbeat():
        while not stop.wait(30):
            try:
                with (out / 'gpu-telemetry.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(time=time.time(), gpus=[gpu_snapshot(l['gpu'])
                                 for l in protocol['lanes']])) + '\n')
                update()
            except Exception as exc:
                update(heartbeat_error=repr(exc))

    def launch(lane, job, smoke=False):
        verify_suite(out)
        cmd = command(protocol, lane, job, out, smoke)
        target = Path(cmd[cmd.index('--out') + 1])
        log = out / 'logs' / f"{'validation-' if smoke else ''}{job['key']}.log"
        if target.exists() or log.exists():
            raise RuntimeError(f'Existing artifact preserved: {target}')
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
        job.update(outcome=result, returncode=rc, finished_unix=time.time())
        return result

    def worker(lane):
        with open(f'/tmp/loopfill-suite-gpu{lane["gpu"]}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    break
                job.update(gpu=lane['gpu'], gpu_uuid=lane['uuid'])
                try:
                    result = launch(lane, job, smoke=True)
                    job.update(validation_output=job['output'], validation_outcome=result)
                    if result in ('complete', 'oom'):
                        launch(lane, job)
                    else:
                        job['outcome'] = 'validation_failed'
                except Exception as exc:
                    job.update(outcome='failed', error=repr(exc), finished_unix=time.time())
                save(out / 'jobs' / f"{job['key']}.json", job)
                update(refresh=True)
                pending.task_done()
            update(lane['gpu'], phase='complete', current=None, pid=None)

    update(refresh=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=8) as workers:
            list(workers.map(worker, protocol['lanes']))
        bad = sum(n for state, n in status['counts'].items() if state not in ('complete', 'oom'))
        update(phase='finished_with_errors' if bad else 'complete', refresh=True)
    finally:
        stop.set()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'summarize'))
    p.add_argument('--previous', type=Path, default=Path('results/loopfill-missing-b1-b16-huginn-20261104'))
    p.add_argument('--out', type=Path, default=Path('results/loopfill-eager-b1-b16-huginn-20261105'))
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.previous, args.out)
    elif args.action == 'run':
        run(args.out)
    else:
        summarize(args.out)
