"""Pinned four-model input audit, staged validation and exclusive-GPU eager matrix.

TinyLlama's original full480 matrix is explicitly deferred on context failure.
Every other model has its own frozen eager-compatible suite. A throughput-only
successor can reuse verified preflight evidence and omit untimed serial references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

from llm_memory_editability.loop_model_registry import (
    MODELS, EXPANSION_MODELS, checkpoint_manifest, context_audit, resolve_stop_ids, tokenize_prompt,
    special_token_audit,
)
from llm_memory_editability.loop_raven_decode import INIT_POLICY
from loop_batch_bench import arm_config, parser as benchmark_parser
from loop_batch_sweep import check_resources, classify, gpu_snapshot
from loop_capacity_suite import digest, save, verify_suite
import loop_eager_suite as eager
from loop_accepted_length import MT_CATEGORIES, extract
from loop_online_suite import audit_run

REPO = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def validation_rows(rows):
    """Six task representatives plus shortest/longest prompt; no output inspection."""
    chosen = {}
    for row in rows:
        category = 'mt_bench' if row['category'] in MT_CATEGORIES else row['category']
        chosen.setdefault(category, row)
    items = {r['question_id']: r for r in chosen.values()}
    for r in (min(rows, key=lambda r: len(r['prompt_token_ids'])),
              max(rows, key=lambda r: len(r['prompt_token_ids']))):
        items[r['question_id']] = r
    for r in rows:
        if len(items) >= 8:
            break
        items[r['question_id']] = r
    return sorted(items.values(), key=lambda r: r['question_id'])


def freeze_source(out):
    source = out / 'source'
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(REPO / folder, source / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(REPO / 'pyproject.toml', source / 'pyproject.toml')
    hashes = {str(p.relative_to(source)): digest(p) for p in sorted(source.rglob('*')) if p.is_file()}
    execution = {name: value for name, value in hashes.items()
                 if name == 'scripts/loop_batch_bench.py' or
                 name.startswith('src/llm_memory_editability/') and name.endswith('.py')}
    return source, hashes, execution


def prepare(out, questions, python, validation_evidence, gpus):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    # Resolving the venv Python symlink would silently switch to the base
    # interpreter and its incompatible Transformers installation.
    out, questions, python = out.resolve(), questions.resolve(), python.absolute()
    if out.exists():
        raise FileExistsError(out)
    rows = sorted(read_rows(questions), key=lambda r: r['question_id'])
    if len(rows) != 480 or len({r['question_id'] for r in rows}) != 480:
        raise ValueError('The original 480 unique raw questions are required')
    if any(not r.get('turns') for r in rows):
        raise ValueError('Raw first-turn text required; do not transfer token IDs')
    out.mkdir(parents=True)
    source, hashes, execution_hashes = freeze_source(out)
    lanes = []
    for gpu in gpus:
        snap = gpu_snapshot(gpu)
        lane = dict(gpu=gpu, uuid=snap['uuid'], cpus=f'{gpu*4}-{gpu*4+3}')
        check_resources(lane)
        lanes.append(lane)
    suites, audits = {}, {}
    for mid in EXPANSION_MODELS:
        spec = MODELS[mid]
        path = Path(snapshot_download(spec.repo_id, revision=spec.revision, local_files_only=True))
        cfg = json.loads((path / 'config.json').read_text())
        if cfg['max_position_embeddings'] != spec.context_limit:
            raise ValueError(f'Registered context differs from checkpoint: {mid}')
        gp = path / 'generation_config.json'
        gen = json.loads(gp.read_text()) if gp.exists() else {}
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        stop, stop_audit = resolve_stop_ids(spec, tok, cfg.get('eos_token_id'), gen.get('eos_token_id'),
                                            cfg['vocab_size'], spec.prompt_format)
        inputs, contexts = [], []
        for row in rows:
            text = row['turns'][0]
            ids = tokenize_prompt(spec, tok, text)
            if any(type(i) is not int or not 0 <= i < cfg['vocab_size'] for i in ids):
                raise ValueError(f'{mid}: invalid token in {row["question_id"]}')
            inputs.append(dict(question_id=row['question_id'], category=row['category'],
                               prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                               prompt_token_ids=ids, model_id=mid))
            contexts.append(context_audit(spec, row['question_id'], ids, 1024,
                                          draft_margin=spec.depth_total+4))
        max_prompt = max(len(r['prompt_token_ids']) for r in inputs)
        required = max_prompt + 1024 + spec.depth_total + 4
        capacity = min(spec.context_limit, ((required + 127) // 128) * 128)
        contexts = [context_audit(spec, r['question_id'], r['prompt_token_ids'], 1024,
                                  max_seq=capacity, draft_margin=spec.depth_total+4) for r in inputs]
        bad = [r for r in contexts if not r['context_ok']]
        state = 'deferred_context_limit' if bad else 'pending_validation'
        suite = out / 'models' / mid
        for name in ('inputs', 'jobs', 'logs', 'runs', 'references', 'validation/references', 'evidence'):
            (suite / name).mkdir(parents=True, exist_ok=True)
        input_path = suite / 'inputs' / f'{mid}.jsonl'
        short_path = suite / 'inputs' / f'{mid}-validation.jsonl'
        write_rows(input_path, inputs)
        short = validation_rows(inputs)
        write_rows(short_path, short)
        audit = dict(checkpoint=checkpoint_manifest(spec, path), stop=stop_audit, contexts=contexts,
                     special_tokens=special_token_audit(tok, cfg, gen),
                     context_failures=bad, prompt_format=spec.prompt_format,
                     add_special_tokens=False if spec.prompt_format == 'raw' else 'released_chat_template',
                     raw_questions=str(questions), raw_questions_sha256=digest(questions),
                     max_prompt_tokens=max_prompt, max_seq=capacity, state=state)
        save(suite / 'input-audit.json', audit)
        audits[mid] = dict(state=state, over_limit=len(bad), max_prompt_tokens=max_prompt, max_seq=capacity)
        configs = {arm: asdict(arm_config(arm, spec.depth_total, *spec.loopfill_readout_depths, 4,
                   draft_node_budget=16, model_top_k=2, child_suffix_caps=(4, 4),
                   suffix_c_initial=.7 if arm == 'budget-online' else 1.,
                   loopspec_depths=spec.loopspec_depths, wavefront_depth=spec.wavefront_depth)) for arm in eager.ARMS}
        template = ['taskset', '--cpu-list', lanes[0]['cpus'], str(python), str(source/'scripts/loop_batch_bench.py')]
        values = {'--model':mid, '--model-path':path, '--questions':input_path,
                  '--arms':'ar', '--batch-sizes':1, '--repeat-id':0, '--order-seeds':0,
                  '--out':suite/'runs/placeholder', '--scoring':'raw', '--draft-node-budget':16,
                  '--model-top-k':2, '--suffix-max-draft':4, '--suffix-c-initial':1,
                  '--max-new-tokens':1024, '--max-seq':capacity, '--max-nodes':1152,
                  '--buffer-policy':'per-arm', '--schedule':'continuous', '--attention-backend':'packed',
                  '--prefill-chunk-size':128, '--warmup-tokens':32, '--warmup-request-limit-factor':2,
                  '--reference-file':suite/'references'/f'{mid}.jsonl', '--allocation-guard':'disabled',
                  '--prompt-format':spec.prompt_format, '--dtype':'bfloat16', '--seed':0,
                  '--threads':1, '--progress-every':16}
        for flag, value in values.items():
            template += [flag, str(value)]
        template += ['--child-suffix-caps','4','4','--check','--no-cuda-graph','--no-pad-decode-rows']
        execution = vars(benchmark_parser().parse_args(template[5:]))
        variable = {'out', 'questions', 'arms', 'batch_sizes', 'repeat_id', 'max_new_tokens',
                    'limit', 'reference_file', 'suffix_c_initial', 'reference_only'}
        execution = {k:v for k,v in execution.items() if k not in variable}
        evidence_files = []
        for dtype in ('float32', 'bfloat16'):
            origin = validation_evidence / f'{mid}-{"fp32" if dtype == "float32" else "bf16"}.json'
            if not origin.exists():
                raise ValueError(f'Missing native validation: {origin}')
            report = json.loads(origin.read_text())
            if report['outcome'] != 'complete' or report['model'] != mid or report['dtype'] != dtype:
                raise ValueError(f'Wrong native validation identity: {origin}')
            if report['checkpoint']['file_sha256'] != audit['checkpoint']['file_sha256']:
                raise ValueError(f'Checkpoint changed after validation: {mid}')
            if not report.get('validation_source_sha256') or any(
                hashes.get(name) != sha for name,sha in report['validation_source_sha256'].items()
            ):
                raise ValueError(f'Native validation source changed: {mid}/{dtype}')
            target = suite / 'evidence' / origin.name
            shutil.copy2(origin, target)
            evidence_files.append(target)
        physical = {arm: (1 if arm == 'ar' else 17 if arm.startswith('budget-') else
                         arm_config(arm, spec.depth_total, *spec.loopspec_depths, 4).physical_nodes
                         if arm == 'loopspec-dual' else 1152) for arm in eager.ARMS}
        kv_slots = spec.depth_total*spec.core_layers + spec.prelude_layers + spec.coda_layers*4
        protocol = dict(version='model-expansion-eager-v1', created_unix=time.time(), source=str(source),
                        questions=str(input_path), questions_sha256=digest(input_path),
                        validation_questions=str(short_path), validation_questions_sha256=digest(short_path),
                        request_ids=[r['question_id'] for r in inputs],
                        validation_ids=[r['question_id'] for r in short],
                        execution_source_sha256=execution_hashes, execution_args=execution,
                        configs=configs, template_command=template, lanes=lanes, arms=eager.ARMS,
                        batches=eager.BATCHES, repeats=[0], planned_cells=30, model=spec.to_dict(),
                        state=state, context_failure_policy='Defer original full480 matrix; no truncation/subset/budget reduction.',
                        initialization_policy=INIT_POLICY if spec.backend == 'raven' else 'fixed_four_loops',
                        coda_policy='Per-depth prefix/tree caches; WFD retains shared draft/verify coda.',
                        physical_nodes=physical,
                        kv_bytes_per_slot={arm: 2*kv_slots*(capacity+n)*spec.kv_heads*spec.head_dim*2
                                           for arm,n in physical.items()},
                        reference_policy='Fresh model-specific serial AR, never migrated from other checkpoints.',
                        baseline_scope='Same-engine AR/WFD/LoopSpec; Graph-off, no execution-row padding.',
                        timing_policy='One run; include prefill/decode/scheduling/suffix work; exclude loading/reference/warmup.',
                        warmup_policy='2B requests, at most32 generated tokens; discard suffix history.',
                        validation_policy='Native FP32/BF16 plus stratified eight-request six-method B2 preflight.')
        save(suite / 'protocol.json', protocol)
        save(suite / 'source-hashes.json', hashes)
        save(suite / 'input-hashes.json', {str(p): digest(p) for p in [input_path, short_path, suite/'input-audit.json', *evidence_files]})
        for batch in eager.BATCHES:
            for arm in eager.ARMS:
                key = eager.job_key(protocol, batch, arm, 0)
                save(suite/'jobs'/f'{key}.json', dict(key=key, model=mid, batch=batch, arm=arm,
                     repeat=0, gpu=None, gpu_uuid=None, outcome='deferred_context_limit' if bad else 'pending'))
        suites[mid] = str(suite)
        verify_suite(suite)
        eager.summarize(suite)
    save(out/'protocol.json', dict(version='model-expansion-parent-v1', suites=suites, lanes=lanes,
                                   planned_cells=120, runnable_cells=sum(a['state']!='deferred_context_limit' for a in audits.values())*30,
                                   input_audit=audits, source=str(source),
                                   historical_suites=[str(REPO/'results'/f'loopfill-eager-b1-b16-{m}-20261105') for m in ('ouro','huginn')],
                                   tinyllama_decision='User requested deferral of original full480 on 24 context failures.'))
    save(out/'status.json', dict(phase='prepared', counts={'pending':90,'deferred_context_limit':30}))
    print(json.dumps(audits), flush=True)


def command(protocol, lane, job, suite, phase):
    require_reference = protocol.get('require_serial_reference', True)
    if phase == 'references' and not require_reference:
        raise ValueError('Serial reference generation disabled by this protocol')
    smoke = phase == 'validate'
    cmd = eager.command(protocol, lane, job, suite, smoke)
    if smoke:
        cmd[cmd.index('--questions')+1] = protocol['validation_questions']
        cmd[cmd.index('--limit')+1] = str(len(protocol['validation_ids']))
        if '--check' not in cmd:
            cmd.append('--check')
    elif not require_reference:
        if '--check' in cmd:
            cmd.remove('--check')
        i = cmd.index('--reference-file')
        del cmd[i:i+2]
    if phase == 'references':
        cmd[cmd.index('--out')+1] = str(suite/'reference-build')
        cmd += ['--reference-only']
    return cmd


def audit_validation(target, protocol, job):
    meta = json.loads((target/'metadata.json').read_text())
    if meta['source_sha256'] != protocol['execution_source_sha256'] or meta['questions_sha256'] != protocol['validation_questions_sha256']:
        raise ValueError('Preflight source/input changed')
    if meta['configs'][job['arm']] != protocol['configs'][job['arm']]:
        raise ValueError('Preflight method changed')
    summary, = audit_run(target, set(protocol['validation_ids']), True)
    if summary['cap_skips'] or summary['attention_policy'] != 'packed_prefix_and_tree':
        raise ValueError('Preflight capacity/backend mismatch')
    if summary['core_padding_rows'] or summary['cuda_graph']:
        raise ValueError('Preflight graph/padding enabled')
    records = read_rows(target/f"{job['arm']}-b{job['batch']}-seed0.jsonl")
    for r in records:
        stats = r['stats']
        if stats['tokens'] != len(r['tokens']) - 1:
            raise ValueError('Decode token accounting mismatch')
        if job['arm'] == 'ar' and stats['calls'] != protocol['model']['depth_total']*stats['tokens']:
            raise ValueError('AR gamma differs from one')


def prepare_throughput(previous, out):
    """Preserve the old suite and reuse only fully audited, unchanged preflight.

    The benchmark and decoder hashes must match exactly. Full480 references and
    partially executed runs are never copied into this successor.
    """
    previous, out = previous.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    parent = json.loads((previous/'protocol.json').read_text())
    source, hashes, execution = freeze_source(out)
    suites = {}
    for mid, old_path in parent['suites'].items():
        old_suite = Path(old_path)
        protocol = verify_suite(old_suite)
        if execution != protocol['execution_source_sha256']:
            raise ValueError(f'Execution source changed since preflight: {mid}')
        marker = json.loads((old_suite/'validation-complete.json').read_text())
        expected = {eager.job_key(protocol, 2, arm, 0) for arm in eager.ARMS}
        if marker['source_sha256'] != execution or set(marker['jobs']) != expected:
            raise ValueError(f'Incomplete preflight evidence: {mid}')
        evidence = {}
        for key in sorted(expected):
            job_path = old_suite/'validate-jobs'/f'{key}.json'
            job = json.loads(job_path.read_text())
            if job['outcome'] != 'complete' or job['key'] != key:
                raise ValueError(f'Failed preflight: {key}')
            target = Path(job['output'])
            audit_validation(target, protocol, job)
            evidence[str(job_path)] = digest(job_path)
            for p in target.iterdir():
                if p.is_file():
                    evidence[str(p)] = digest(p)
        suite = out/'models'/mid
        for folder in ('inputs', 'evidence', 'validation', 'validate-jobs'):
            shutil.copytree(old_suite/folder, suite/folder)
        shutil.copy2(old_suite/'input-audit.json', suite/'input-audit.json')
        for folder in ('jobs', 'logs', 'runs', 'references'):
            (suite/folder).mkdir()
        protocol.update(source=str(source), parent=str(old_suite), created_unix=time.time(),
                        version='model-expansion-throughput-v1', require_serial_reference=False,
                        reference_policy='User requested skipping independent full480 serial AR generation. '
                                         'Formal serial agreement is unmeasured; timed AR remains a matrix arm.',
                        validation_policy='Reuse all six completed short preflights after exact execution '
                                          'hash checks and re-audit; native evidence retained unchanged.')
        for field in ('questions', 'validation_questions'):
            protocol[field] = str(suite/Path(protocol[field]).relative_to(old_suite))
        protocol['execution_args']['check'] = False
        template = protocol['template_command']
        template[4] = str(source/'scripts/loop_batch_bench.py')
        for flag, value in {'--questions':protocol['questions'],
                            '--reference-file':suite/'references'/f'{mid}.jsonl',
                            '--out':suite/'runs/placeholder'}.items():
            template[template.index(flag)+1] = str(value)
        if '--check' in template:
            template.remove('--check')
        save(suite/'protocol.json', protocol)
        save(suite/'source-hashes.json', hashes)
        save(suite/'preflight-provenance.json', dict(parent=str(old_suite),
             marker_sha256=digest(old_suite/'validation-complete.json'),
             evidence_sha256=evidence, execution_unchanged=True, reaudit_unix=time.time()))
        marker.update(reused_from=str(old_suite), reaudit_unix=time.time())
        save(suite/'validation-complete.json', marker)
        inputs = [suite/'input-audit.json', suite/'preflight-provenance.json',
                  suite/'validation-complete.json']
        inputs += [p for folder in ('inputs','evidence','validation','validate-jobs')
                   for p in (suite/folder).rglob('*') if p.is_file()]
        save(suite/'input-hashes.json', {str(p):digest(p) for p in inputs})
        for batch in protocol['batches']:
            for arm in protocol['arms']:
                key = eager.job_key(protocol, batch, arm, 0)
                save(suite/'jobs'/f'{key}.json', dict(key=key, model=mid, batch=batch, arm=arm,
                     repeat=0, gpu=None, gpu_uuid=None,
                     outcome='deferred_context_limit' if protocol['state']=='deferred_context_limit' else 'pending'))
        verify_suite(suite)
        eager.summarize(suite)
        suites[mid] = str(suite)
    parent.update(version='model-expansion-throughput-parent-v1', source=str(source), suites=suites,
                  previous=str(previous), require_serial_reference=False, created_unix=time.time(),
                  reference_decision='User requested immediate throughput matrix without independent serial references.')
    save(out/'protocol.json', parent)
    save(out/'status.json', dict(phase='prepared', counts={'pending':parent['runnable_cells'],
         'deferred_context_limit':parent['planned_cells']-parent['runnable_cells']}))
    print(json.dumps(dict(out=str(out), runnable_cells=parent['runnable_cells'],
                         reused_preflight_cells=len(suites)*6, require_serial_reference=False)), flush=True)


def run(out, phase):
    out = out.resolve()
    parent = json.loads((out/'protocol.json').read_text())
    guard = (out/'supervisor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for mid, path in parent['suites'].items():
        suite = Path(path)
        protocol = verify_suite(suite)
        audit = json.loads((suite/'input-audit.json').read_text())
        manifest = checkpoint_manifest(MODELS[mid], protocol['execution_args']['model_path'])
        if manifest != audit['checkpoint']:
            raise ValueError(f'Checkpoint changed after freeze: {mid}')
        if phase != 'validate' and protocol['state'] == 'deferred_context_limit':
            continue
        if phase == 'validate' and (suite/'validation-complete.json').exists():
            evidence = json.loads((suite/'validation-complete.json').read_text())
            if evidence['source_sha256'] != protocol['execution_source_sha256']:
                raise ValueError('Preflight source changed')
            continue
        if phase != 'validate':
            evidence = json.loads((suite/'validation-complete.json').read_text())
            if evidence['source_sha256'] != protocol['execution_source_sha256']:
                raise ValueError('Preflight source changed')
        if phase == 'run':
            if protocol.get('require_serial_reference', True) and not (suite/'reference-build/reference-complete.json').exists():
                raise ValueError(f'Build untimed references first: {mid}')
            for p in (suite/'jobs').glob('*.json'):
                j = json.loads(p.read_text())
                if j['outcome'] in ('running','validating'):
                    raise ValueError(f'Unreconciled previous process: {j["key"]}')
                if j['outcome'] == 'pending':
                    jobs.append((suite, protocol, j))
        else:
            if phase == 'references' and (suite/'reference-build/reference-complete.json').exists():
                continue
            for arm in eager.ARMS if phase == 'validate' else ['ar']:
                batch = 2 if phase == 'validate' else 1
                key = eager.job_key(protocol, batch, arm, 0)
                jobs.append((suite, protocol, dict(key=key, model=mid, arm=arm, batch=batch, repeat=0)))
    jobs.sort(key=lambda item: (item[2]['batch'], eager.ARMS.index(item[2]['arm']), item[2]['model']))
    pending = queue.Queue()
    for item in jobs:
        pending.put(item)
    mutex, stop = threading.RLock(), threading.Event()
    status = dict(phase=phase, supervisor_pid=os.getpid(), started_unix=time.time(), total=len(jobs), completed=0, errors=0, lanes={})
    outcomes = []
    def update(gpu=None, **fields):
        with mutex:
            (status if gpu is None else status['lanes'].setdefault(str(gpu), {})).update(fields)
            status['updated_unix'] = time.time()
            save(out/'status.json', status)
    def heartbeat():
        while not stop.wait(30):
            with (out/'gpu-telemetry.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(), gpus=[gpu_snapshot(lane['gpu']) for lane in parent['lanes']]))+'\n')
            update()
    def worker(lane):
        with open(f'/tmp/loopfill-suite-gpu{lane["gpu"]}.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                try:
                    suite, protocol, job = pending.get_nowait()
                except queue.Empty:
                    break
                job.update(gpu=lane['gpu'], gpu_uuid=lane['uuid'], started_unix=time.time())
                target_job = suite / ('jobs' if phase=='run' else f'{phase}-jobs') / f'{job["key"]}.json'
                try:
                    verify_suite(suite)
                    check_resources(lane)
                    cmd = command(protocol, lane, job, suite, phase)
                    target = Path(cmd[cmd.index('--out')+1])
                    log = suite/'logs'/f'{phase}-{job["key"]}.log'
                    if target.exists() or log.exists():
                        raise FileExistsError(f'Existing artifacts preserved: {target}')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                               PYTHONPATH=str(Path(protocol['source'])/'src'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                    env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', None)
                    job.update(outcome='running', command=cmd, output=str(target), log=str(log))
                    with log.open('x') as f:
                        child = subprocess.Popen(cmd, cwd=protocol['source'], env=env, stdin=subprocess.DEVNULL,
                                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                        job['pid'] = child.pid
                        save(target_job, job)
                        update(lane['gpu'], key=job['key'], pid=child.pid)
                        rc = child.wait()
                    outcome = classify(rc, target)
                    if outcome == 'complete':
                        if phase == 'run':
                            eager.audit(target, protocol, job)
                        elif phase == 'validate':
                            audit_validation(target, protocol, job)
                        else:
                            marker = json.loads((target/'reference-complete.json').read_text())
                            if marker['requests'] != 480:
                                raise ValueError('Incomplete serial reference')
                    elif outcome != 'oom':
                        job['error'] = log.read_text()[-4000:]
                    job.update(outcome='error' if outcome=='failed' else outcome, returncode=rc)
                except Exception as exc:
                    job.update(outcome='error', error=repr(exc))
                job['finished_unix'] = time.time()
                save(target_job, job)
                with mutex:
                    outcomes.append((suite, job))
                    status['completed'] += 1
                    status['errors'] += job['outcome'] not in ('complete','oom')
                update(lane['gpu'], key=None, pid=None)
                pending.task_done()
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        with ThreadPoolExecutor(max_workers=len(parent['lanes'])) as executor:
            list(executor.map(worker, parent['lanes']))
        if phase == 'validate':
            for path in parent['suites'].values():
                suite = Path(path)
                group = [j for s,j in outcomes if s == suite]
                if len(group)==6 and all(j['outcome']=='complete' for j in group):
                    protocol = verify_suite(suite)
                    save(suite/'validation-complete.json', dict(source_sha256=protocol['execution_source_sha256'],
                         jobs=[j['key'] for j in group], completed_unix=time.time()))
        if phase == 'run':
            for path in parent['suites'].values():
                suite = Path(path)
                eager.summarize(suite)
                extract(suite, suite/'accepted-length')
        update(phase=phase+'_finished', counts=dict(Counter(j['outcome'] for _,j in outcomes)))
    finally:
        stop.set()
    if any(j['outcome'] not in ('complete','oom') for _,j in outcomes):
        raise RuntimeError(f'{phase} has errors; inspect status/jobs/logs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','prepare-throughput','validate','references','run','summarize','pipeline'))
    p.add_argument('--previous', type=Path)
    p.add_argument('--out', type=Path, default=Path('results/loopfill-model-expansion-20261105'))
    p.add_argument('--questions', type=Path, default=Path('results/loop-spec-reuse-20261030/raw/data/spec_bench_question.jsonl'))
    p.add_argument('--python', type=Path, default=Path(sys.executable))
    p.add_argument('--validation-evidence', type=Path, default=Path('results/loopfill-model-expansion-audit-20261105'))
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.out, a.questions, a.python, a.validation_evidence, [int(i) for i in a.gpus.split(',')])
    elif a.action == 'prepare-throughput':
        if a.previous is None:
            p.error('prepare-throughput requires --previous')
        prepare_throughput(a.previous, a.out)
    elif a.action in ('summarize','pipeline'):
        from loop_accepted_length import summarize_models
        parent = json.loads((a.out/'protocol.json').read_text())
        suites = [Path(s) for s in parent['suites'].values()]
        suites += [Path(s) for s in parent['historical_suites']]
        if a.action == 'pipeline':
            run(a.out, 'validate')
            if parent.get('require_serial_reference', True):
                run(a.out, 'references')
            try:
                run(a.out, 'run')
            finally:
                # Failed/OOM cells still belong in the table as status-only cells.
                summarize_models(suites, a.out/'paper-summary')
        else:
            summarize_models(suites, a.out/'paper-summary')
    else:
        run(a.out, a.action)


if __name__ == '__main__':
    main()
