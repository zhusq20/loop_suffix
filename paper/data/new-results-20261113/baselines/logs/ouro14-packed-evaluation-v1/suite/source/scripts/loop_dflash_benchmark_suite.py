"""Freeze, queue, audit and summarize DFlash Table 1/2 remeasurements.

Preparation reads old tables but never modifies them. Every retry has a fresh
directory; throughput windows are never spliced. Shared UUID locks are inherited
by workers, and the current allocation file is checked before each launch.
"""
from __future__ import annotations

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

from loop_dflash_benchmark_worker import digest, read, save

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['gsm8k', 'math500', 'bbh', 'humanevalplus', 'mbppplus', 'specbench']


def prepare(args):
    from baselines.infer import tokenizer_and_stops
    from llm_memory_editability.loop_model_registry import MODELS, tokenize_prompt

    suite = args.out.resolve()
    if suite.exists():
        raise FileExistsError(suite)
    specs = {}
    for value in args.checkpoints:
        model, path = value.split('=', 1)
        if model in specs or model not in MODELS:
            raise ValueError('Duplicate or unknown checkpoint model')
        specs[model] = Path(path).resolve()
    old1, oldt1, old2 = [read(p/'protocol.json') for p in (args.table1, args.table1_t1, args.table2)]
    suite.mkdir(parents=True)
    for name in ['inputs', 'references', 'source', 'runs', 'logs']:
        (suite/name).mkdir()
    for name in ['src', 'baselines']:
        shutil.copytree(ROOT/name, suite/'source'/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    (suite/'source/scripts').mkdir()
    for name in ['loop_dflash_benchmark_worker.py', 'loop_dflash_benchmark_suite.py']:
        shutil.copy2(ROOT/'scripts'/name, suite/'source/scripts'/name)
    (suite/'source/runtime').mkdir()
    shutil.copy2(args.driver_wrapper, suite/'source/runtime/native_cuda_env.py')
    references = {}
    for label, directory in [('table1-t0', args.table1), ('table1-t1', args.table1_t1), ('table2', args.table2)]:
        for name in ['protocol.json', 'measurements.csv']:
            dest = suite/'references'/f'{label}-{name}'
            shutil.copy2(directory/name, dest)
            references[str(dest.relative_to(suite))] = digest(dest)
    models, jobs = {}, []
    for model, checkpoint in specs.items():
        if not (checkpoint/'complete.json').is_file() or not (checkpoint.parent/'complete.json').is_file():
            raise ValueError(f'Checkpoint training/export is incomplete: {checkpoint}')
        if read(checkpoint.parent/'complete.json').get('epochs') != 6:
            raise ValueError('Expected the completed six-epoch drafter')
        spec = MODELS[model]
        path = Path(old1['models'][model]['path'])
        tok, stops = tokenizer_and_stops(model, path, 'chat', None)
        models[model] = dict(path=str(path), family=spec.backend, revision=spec.revision,
            checkpoint=str(checkpoint), checkpoint_sha256={name:digest(checkpoint/name)
                for name in ['model.safetensors', 'config.json', 'complete.json']},
            training_complete_sha256=digest(checkpoint.parent/'complete.json'), stop_ids=sorted(stops))
        for dataset in DATASETS:
            original = args.table1/'inputs'/f'{dataset}.jsonl'
            if digest(original) != old1['inputs'][dataset]['sha256']:
                raise ValueError('Original Table 1 input changed')
            if digest(args.table1_t1/'inputs'/f'{dataset}.jsonl') != digest(original):
                raise ValueError('T0/T1 inputs differ; explicitly freeze separate sets')
            rows = [json.loads(line) for line in original.read_text().splitlines()]
            assert len(rows) == old1['inputs'][dataset]['count'] == oldt1['inputs'][dataset]['count']
            prepared = [dict(source_id=r['source_id'], prompt_sha256=r['prompt_sha256'],
                             prompt_tokens=tokenize_prompt(spec, tok, r['prompt'], 'chat')) for r in rows]
            required = max(len(r['prompt_tokens']) for r in prepared) + 512 + spec.depth_total + 4
            capacity = min(spec.context_limit, ((required + 255)//256)*256)
            if required > capacity:
                raise ValueError(f'Context exceeds capacity: {model}/{dataset}')
            dest = suite/'inputs'/f'{model}-{dataset}.jsonl'
            dest.write_text(''.join(json.dumps(r)+'\n' for r in prepared))
            for temp in [0, 1]:
                jobs.append(dict(id=f'table1-{model}-{dataset}-t{temp}', kind='latency', model=model,
                    dataset=dataset, temperature=temp, batch=1, max_seq=capacity, max_new_tokens=512,
                    count=len(rows), inputs=str(dest.relative_to(suite)), inputs_sha256=digest(dest),
                    original_inputs_sha256=digest(original)))
        original = args.table2/'inputs'/f'{model}.jsonl'
        if digest(original) != old2['models'][model]['inputs_sha256']:
            raise ValueError('Original Table 2 input changed')
        rows = [json.loads(line) for line in original.read_text().splitlines()]
        assert len(rows) == len({r['question_id'] for r in rows}) == 480
        prepared = []
        for r in rows:
            assert r['model_id'] == model
            assert r['prompt_token_ids'] == tokenize_prompt(spec, tok, r['prompt'], 'chat')
            prepared.append(dict(source_id=r['question_id'], prompt_tokens=r['prompt_token_ids']))
        dest = suite/'inputs'/f'{model}-table2.jsonl'
        dest.write_text(''.join(json.dumps(r)+'\n' for r in prepared))
        for batch in [1, 2, 4, 8, 16]:
            jobs.append(dict(id=f'table2-{model}-b{batch}', kind='throughput', model=model,
                dataset='specbench', temperature=0, batch=batch, max_seq=2688, max_new_tokens=1024,
                count=480, inputs=str(dest.relative_to(suite)), inputs_sha256=digest(dest),
                original_inputs_sha256=digest(original)))
        jobs.append(dict(id=f'validate-{model}', kind='validate', model=model, dataset='synthetic-prefixes',
            temperature=0, batch=2, max_seq=128, max_new_tokens=33, count=6,
            inputs=str(dest.relative_to(suite)), inputs_sha256=digest(dest)))
    jobs.sort(key=lambda j: (j['kind'] != 'validate', j['kind'] == 'latency',
                             j['batch'], j['temperature'], j['dataset'], j['model']))
    source = suite/'source'
    protocol = dict(created_unix=time.time(), name=suite.name, python=sys.executable,
        models=models, source_sha256={str(p.relative_to(source)):digest(p)
                                     for p in sorted(source.rglob('*')) if p.is_file()},
        reference_sha256=references, gpu_policy=str(args.gpu_policy.resolve()), allowed_gpus=args.gpus,
        cpu_bindings={str(g): f'{g*4 if g<4 else 48+(g-4)*4}-{g*4+3 if g<4 else 51+(g-4)*4}' for g in args.gpus},
        planned_formal_cells=sum(j['kind'] != 'validate' for j in jobs),
        formal_requests=sum(j['count'] for j in jobs if j['kind'] != 'validate'),
        table1=dict(temperatures=[0,1], max_new_tokens=512, batch=1, warmup_requests=2,
                    warmup_tokens=32, timing='synchronized decode only, excludes prefill/first token and persistence'),
        table2=dict(temperature=0, batches=[1,2,4,8,16], max_new_tokens=1024,
                    max_seq=2688, prefill_chunk_size=128, order_seed=0, warmup_requests='min(2B,480)',
                    warmup_tokens=32, timing='continuous finite-stream E2E including prefill, scheduling and fsync'),
        dtype='bfloat16', graph=False, compile=False, target_row_padding=False,
        drafter_batching='SDPA with explicit per-request prefix masks; group equal block widths',
        sampling='seed0 request hashes for Table1; positional target first token and p/q rejection at T1',
        repeats=1, checkpoint_training=False,
        tracking=dict(provider='wandb', mode=args.wandb_mode, entity='zsqzz',
                      project='loopfill-dflash-packed-table12', key_file=str(args.wandb_key_file)),
        reference_policy='Original same-host AR measurements remain historical denominators, with timestamps and hashes. No fresh AR or drift correction is claimed.',
        recovery='Require exit0, complete marker, ordered IDs and exact journal/summary hashes. Interrupted cells restart in new attempts; no timing-window splicing.',
        exclusions={m:'No completed DFlash checkpoint supplied' for m in old1['models'] if m not in models},
        limitations=['BF16 shapes can change output; throughput does not establish equal task quality.',
                     'Single seed and timing repeat; no confidence intervals.',
                     'Masked padding applies only inside the drafter; target packed rows remain unpadded.'])
    save(suite/'jobs.json', jobs)
    save(suite/'protocol.json', protocol)
    save(suite/'dispatch.json', {})
    save(suite/'status.json', dict(status='prepared', planned=protocol['planned_formal_cells'],
                                  requests=protocol['formal_requests']))
    print(json.dumps(dict(suite=str(suite), formal_cells=protocol['planned_formal_cells'],
                          formal_requests=protocol['formal_requests'])), flush=True)


def audit(suite, job, out):
    done, summary = read(out/'complete.json'), read(out/'summary.json')
    metadata = read(out/'metadata.json')
    assert metadata['job'] == job
    assert metadata['source_sha256'] == read(suite/'protocol.json')['source_sha256']
    assert digest(out/'requests.jsonl') == done['requests_sha256']
    assert digest(out/'summary.json') == done['summary_sha256']
    records = [json.loads(line) for line in (out/'requests.jsonl').read_text().splitlines()]
    inputs = [json.loads(line) for line in (suite/job['inputs']).read_text().splitlines()]
    assert len(records) == done['requests'] == summary['requests'] == job['count']
    key = 'source_id' if job['kind'] == 'latency' else 'request_id'
    ids = [r[key] for r in records]
    assert len(set(ids)) == len(ids)
    if job['kind'] == 'latency':
        assert ids == [r['source_id'] for r in inputs]
        assert [r['prompt_sha256'] for r in records] == [r['prompt_sha256'] for r in inputs]
        assert abs(summary['decode_s']-sum(r['stats']['decode_s'] for r in records)) < 1e-7
        assert summary['decode_tokens'] == sum(len(r['tokens'])-1 for r in records)
    elif job['kind'] == 'throughput':
        assert set(ids) == {r['source_id'] for r in inputs}
        assert summary['output_tokens'] == sum(len(r['tokens']) for r in records)
        assert abs(summary['output_tput']-summary['output_tokens']/summary['wall_s']) < 1e-9
        assert summary['concurrency'] == job['batch'] and summary['core_padding_rows'] == 0
        assert summary['input_tokens'] == sum(len(r['prompt_tokens']) for r in inputs)
        assert summary['prefill_chunk_tokens'] == summary['input_tokens']
        assert not summary['cuda_graph'] and not summary['pad_decode_rows']
    for r in records:
        assert 1 <= len(r['tokens']) <= job['max_new_tokens']
        assert r['stats']['decode_tokens'] == len(r['tokens'])-1
        assert sum(r['stats']['emitted_per_verification']) == len(r['tokens'])-1
    calls = sum(r['stats']['target_verifications'] for r in records)
    if calls:
        assert abs(summary['gamma']-sum(len(r['tokens'])-1 for r in records)/calls) < 1e-9
    return summary


def try_gpu_lock(uuid):
    lock = open(f'/tmp/loopfill-{uuid}.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return None
    return lock


def summarize(suite):
    jobs, dispatch = read(suite/'jobs.json'), read(suite/'dispatch.json')
    references = {key:list(csv.DictReader((suite/'references'/f'{key}-measurements.csv').open()))
                  for key in ['table1-t0', 'table1-t1', 'table2']}
    results = []
    for job in jobs:
        if job['kind'] == 'validate':
            continue
        state = dispatch.get(job['id'], {})
        row = {k:job[k] for k in ['id', 'kind', 'model', 'dataset', 'temperature', 'batch']}
        row['outcome'] = state.get('outcome', 'pending')
        if row['outcome'] == 'complete':
            out = suite/state['out']
            summary = audit(suite, job, out)
            row.update({k:v for k,v in summary.items() if isinstance(v, (int, float, str))})
            ref = next(r for r in references[f"table1-t{job['temperature']}" if job['kind'] == 'latency' else 'table2']
                if r['model'] == job['model'] and r['arm'] == 'ar' and
                (r.get('dataset') == job['dataset'] if job['kind'] == 'latency' else int(r['batch']) == job['batch']))
            if job['kind'] == 'latency':
                row.update(historical_ar_decode_s=float(ref['decode_s']),
                    latency_speedup_vs_historical_ar=float(ref['decode_s'])/summary['decode_s'],
                    throughput_ratio_vs_historical_ar=summary['decode_token_s']/float(ref['decode_token_s']),
                    decode_token_ratio_vs_historical_ar=summary['decode_tokens']/int(ref['decode_tokens']))
            else:
                row.update(historical_ar_output_tput=float(ref['output_tput']),
                    throughput_ratio_vs_historical_ar=summary['output_tput']/float(ref['output_tput']))
        results.append(row)
    fields = list(dict.fromkeys(k for row in results for k in row))
    with (suite/'measurements.csv.tmp').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (suite/'measurements.csv.tmp').replace(suite/'measurements.csv')
    save(suite/'measurements.json', results)
    lines = ['# DFlash packed 引擎重测', '',
        '表一：六任务 B1/max512，T=0/1；表二：SpecBench480/max1024，B=1/2/4/8/16。',
        'AR 分母来自原表历史测量，未作跨日期漂移修正。仅 complete 且审计通过的单元提供数值。', '',
        '| 单元 | 状态 | token/s | γ | 相对历史 AR |', '|---|---|---:|---:|---:|']
    for row in results:
        if row['outcome'] == 'complete':
            rate = row.get('decode_token_s', row.get('output_tput'))
            ratio = row.get('latency_speedup_vs_historical_ar', row.get('throughput_ratio_vs_historical_ar'))
            lines.append(f"| {row['id']} | complete | {rate:.3f} | {row['gamma']:.3f} | {ratio:.3f} |")
        else:
            lines.append(f"| {row['id']} | {row['outcome']} | — | — | — |")
    lines += ['', 'Ouro-1.4B 未提供完成训练的 DFlash 检查点，未启动训练或填入替代数值。',
              '原表和旧 DFlash 结果保持历史记录；单次 BF16 速度测量不证明输出相同或任务质量相同。']
    (suite/'report.md').write_text('\n'.join(lines)+'\n')
    return results


def run_queue(suite):
    protocol, jobs = read(suite/'protocol.json'), read(suite/'jobs.json')
    source = suite/'source'
    for rel, sha in protocol['source_sha256'].items():
        if digest(source/rel) != sha:
            raise ValueError(f'Frozen source changed: {rel}')
    for rel, sha in protocol['reference_sha256'].items():
        if digest(suite/rel) != sha:
            raise ValueError('Historical reference changed')
    sys.path.insert(0, str(source/'runtime'))
    from native_cuda_env import native_cuda_environment
    env, evidence = native_cuda_environment()
    save(suite/'native-driver.json', evidence)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [source, source/'src', source/'scripts'])),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    env.pop('WANDB_API_KEY', None)
    with (suite/'controller.lock').open('a+') as control:
        fcntl.flock(control, fcntl.LOCK_EX|fcntl.LOCK_NB)
        dispatch, active = read(suite/'dispatch.json'), {}
        for identity, state in dispatch.items():
            if state.get('outcome') == 'running':
                proc = Path(f"/proc/{state['pid']}/cmdline")
                if proc.exists() and identity.encode() in proc.read_bytes():
                    raise RuntimeError(f'Existing worker still alive: {identity}; wait before resuming')
                state['outcome'] = 'interrupted'
        save(suite/'dispatch.json', dispatch)
        while True:
            for identity, entry in list(active.items()):
                code = entry['process'].poll()
                if code is None:
                    continue
                entry['log'].close()
                entry['lock'].close()
                state = dispatch[identity]
                out = suite/state['out']
                outcome = 'failed'
                if code == 0:
                    try:
                        audit(suite, entry['job'], out)
                        outcome = 'complete'
                    except Exception as exc:
                        save(out/'audit-failure.json', dict(error=repr(exc)))
                elif (out/'failure.json').exists() and read(out/'failure.json')['outcome'] == 'oom':
                    outcome = 'oom'
                state.update(outcome=outcome, exit_code=code, finished_unix=time.time())
                save(suite/'logs'/f'{identity}-attempt{state["attempt"]}-exit.json', state)
                del active[identity]
            save(suite/'dispatch.json', dispatch)
            finished = {k for k,v in dispatch.items() if v.get('outcome') in ['complete','oom','failed']}
            if len(finished) == len(jobs):
                summarize(suite)
                save(suite/'status.json', dict(status='finished', outcomes={k:v['outcome'] for k,v in dispatch.items()},
                                              updated_unix=time.time()))
                return
            allocation = read(protocol['gpu_policy'])
            allowed = set(protocol['allowed_gpus']) & set(allocation['allowed_gpus'])
            allowed -= set(allocation.get('reserved_for_development', []))
            lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,compute_mode',
                                             '--format=csv,noheader'], text=True).splitlines()
            inventory = {int(p[0]):(p[1], p[2]) for line in lines if (p:=[x.strip() for x in line.split(',')])}
            busy = {x.strip() for x in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                                                '--format=csv,noheader'], text=True).splitlines()}
            pending = [j for j in jobs if j['id'] not in finished and j['id'] not in active]
            candidates = [j for j in pending if j['kind'] == 'validate' or
                          dispatch.get(f"validate-{j['model']}", {}).get('outcome') == 'complete']
            if not (suite/'STOP').exists():
                for gpu in sorted(allowed):
                    uuid, mode = inventory[gpu]
                    if uuid in busy or mode != 'Exclusive_Process' or not candidates:
                        continue
                    lock = try_gpu_lock(uuid)
                    if lock is None:
                        continue
                    occupied_now = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                                             '--format=csv,noheader'], text=True)
                    if uuid in {x.strip() for x in occupied_now.splitlines()}:
                        lock.close()
                        continue
                    job = candidates.pop(0)
                    identity = job['id']
                    attempt = dispatch.get(identity, {}).get('attempt', 0) + 1
                    out = Path('runs')/identity/f'attempt{attempt}'
                    cmd = ['taskset', '--cpu-list', protocol['cpu_bindings'][str(gpu)], protocol['python'], '-u',
                        str(source/'scripts/loop_dflash_benchmark_worker.py'), '--suite', str(suite),
                        '--job', identity, '--out', str(suite/out), '--lock-fd', str(lock.fileno())]
                    log = (suite/'logs'/f'{identity}-attempt{attempt}.log').open('x')
                    child_env = dict(env, CUDA_VISIBLE_DEVICES=uuid)
                    proc = subprocess.Popen(cmd, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                            pass_fds=(lock.fileno(),), start_new_session=True, cwd=source)
                    dispatch[identity] = dict(outcome='running', attempt=attempt, out=str(out), gpu=gpu,
                        gpu_uuid=uuid, pid=proc.pid, argv=cmd, started_unix=time.time())
                    active[identity] = dict(process=proc, log=log, lock=lock, job=job)
                    busy.add(uuid)
                    save(suite/'dispatch.json', dispatch)
            blocked = [j['id'] for j in pending if dispatch.get(f"validate-{j['model']}", {}).get('outcome') in ['failed','oom']]
            rows = summarize(suite)
            counts = {state:sum(r['outcome'] == state for r in rows)
                      for state in ['pending','running','complete','failed','oom','interrupted']}
            save(suite/'status.json', dict(status='running' if active else 'waiting_for_gpu',
                counts=counts, active={k:dispatch[k] for k in active}, blocked_by_validation=blocked,
                allowed_gpus=sorted(allowed), planned=protocol['planned_formal_cells'], updated_unix=time.time()))
            telemetry = subprocess.check_output(['nvidia-smi',
                '--query-gpu=index,uuid,clocks.current.sm,clocks.current.memory,power.limit,temperature.gpu,utilization.gpu,memory.used',
                '--format=csv,noheader'], text=True)
            with (suite/'telemetry.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(unix=time.time(), gpu_csv=telemetry))+'\n')
            if not active and ((blocked and not candidates) or (suite/'STOP').exists()):
                save(suite/'status.json', dict(status='validation_failed' if blocked else 'stopped',
                    counts=counts, blocked_by_validation=blocked, updated_unix=time.time()))
                return
            time.sleep(10)


def track(suite):
    """Independent network process; never imported or called by a timed worker."""
    import wandb
    protocol = read(suite/'protocol.json')
    tracking = protocol['tracking']
    if tracking['mode'] == 'disabled':
        return
    key = Path(tracking['key_file'])
    if tracking['mode'] == 'online' and 'WANDB_API_KEY' not in os.environ and key.is_file():
        os.environ['WANDB_API_KEY'] = key.read_text().strip()
    with (suite/'tracking.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        run = wandb.init(entity=tracking['entity'], project=tracking['project'],
            group=suite.name, name=suite.name, id=digest(suite/'protocol.json')[:16],
            resume='allow', mode=tracking['mode'], dir=str(suite), job_type='table12-remeasurement',
            config=protocol, settings=wandb.Settings(init_timeout=60))
        save(suite/'tracking.json', dict(url=run.url, id=run.id, mode=tracking['mode']))
        uploaded = set()
        offsets = {}
        while True:
            state = read(suite/'status.json')
            dispatch = read(suite/'dispatch.json')
            metrics = {'cells/'+k:v for k,v in state.get('counts', {}).items()}
            for identity, job in dispatch.items():
                out = suite/job['out']
                journal = out/'requests.jsonl'
                if journal.exists():
                    # Seek only to complete newline boundaries; a partially-written
                    # final line remains pending until the worker flushes it.
                    offset = offsets.get(str(journal), 0)
                    with journal.open('rb') as f:
                        f.seek(offset)
                        for line in f:
                            if not line.endswith(b'\n'):
                                break
                            row = json.loads(line)
                            offset += len(line)
                            stats = row.get('stats', {})
                            run.log({identity+'/'+k:v for k,v in stats.items()
                                     if k in ['decode_tokens','decode_s','gamma','target_verifications'] and v is not None})
                    offsets[str(journal)] = offset
                progress = out/'progress.json'
                if progress.exists():
                    metrics[identity+'/completed_requests'] = read(progress)['requests']
                receipt = suite/'logs'/f'{identity}-attempt{job["attempt"]}-exit.json'
                if receipt.exists() and str(out) not in uploaded:
                    artifact = wandb.Artifact(identity+'-attempt'+str(job['attempt']), type='benchmark-cell')
                    for name in ['requests.jsonl','metadata.json','summary.json','complete.json','failure.json','comparison.json','preflight.json']:
                        if (out/name).exists():
                            artifact.add_file(str(out/name), name=name)
                    artifact.add_file(str(receipt), name='process-exit.json')
                    run.log_artifact(artifact)
                    uploaded.add(str(out))
            if metrics:
                run.log(metrics)
            run.save(str(suite/'measurements.csv'), base_path=str(suite), policy='now')
            if state['status'] in ['finished','validation_failed','stopped']:
                run.summary['final_status'] = state
                run.finish()
                return
            time.sleep(15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','run','summarize','track'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoints', nargs='+')
    p.add_argument('--table1', type=Path, default=ROOT/'results/loopfill-latency512-20261107')
    p.add_argument('--table1-t1', type=Path, default=ROOT/'results/loopfill-latency512-t1-20261108')
    p.add_argument('--table2', type=Path, default=ROOT/'results/loopfill-table2-20261107-v2')
    p.add_argument('--gpu-policy', type=Path)
    p.add_argument('--gpus', type=int, nargs='+', default=[2,3,4,6])
    p.add_argument('--driver-wrapper', type=Path)
    p.add_argument('--wandb-mode', choices=['online','offline','disabled'], default='online')
    p.add_argument('--wandb-key-file', type=Path, default=Path('/tmp/loopfill-wandb-key'))
    args = p.parse_args()
    if args.action == 'prepare':
        if not args.checkpoints or not args.gpu_policy or not args.driver_wrapper:
            p.error('prepare needs --checkpoints, --gpu-policy and --driver-wrapper')
        prepare(args)
    elif args.action == 'run':
        run_queue(args.out.resolve())
    elif args.action == 'track':
        track(args.out.resolve())
    else:
        summarize(args.out.resolve())


if __name__ == '__main__':
    main()
