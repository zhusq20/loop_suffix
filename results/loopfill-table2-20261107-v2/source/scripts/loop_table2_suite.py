"""SpecBench Table 2: 75 eager cells, a shared eight-GPU queue, durable W&B tracking.

Run prepare once, then run. The frozen source is used for every benchmark process.
Interrupted cells start a new attempt; completed/OOM cells are retained on resume.
W&B runs live in supervisor processes, outside the measured GPU process.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
MODELS = ['huginn-3.5b', 'ouro-1.4b', 'ouro-2.6b']
ARMS = ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-online']
BATCHES = [1, 2, 4, 8, 16]
LABELS = dict(zip(ARMS, ['AR', 'WFD', 'LoopSpec', 'Suffix-only', 'LoopFill-online']))


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cells():
    rows = [dict(key=f'{m}-b{b}-{a}', model=m, batch=b, arm=a, outcome='pending')
            for m in MODELS for b in BATCHES for a in ARMS]
    # Longest expected cells first reduces the idle tail. Never choose by results.
    cost = {'ar': 4, 'wavefront': 2.5, 'loopspec-dual': 1.3,
            'budget-suffix': 2, 'budget-online': 1}
    model_cost = {'huginn-3.5b': 3, 'ouro-2.6b': 1, 'ouro-1.4b': .6}
    return sorted(rows, key=lambda r: -model_cost[r['model']] * cost[r['arm']] / r['batch']**.65)


def prepare(out, assets, preflight_from=None):
    from transformers import AutoTokenizer
    from llm_memory_editability.loop_model_registry import MODELS as specs, checkpoint_manifest, tokenize_prompt
    from loop_batch_sweep import check_resources, gpu_snapshot
    from loop_model_expansion_suite import validation_rows
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = out / 'source'
    for name in ['src', 'scripts', 'tests', 'configs']:
        shutil.copytree(REPO / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(REPO / 'pyproject.toml', source / 'pyproject.toml')
    raw_path = out / 'inputs/specbench480.jsonl'
    raw_path.parent.mkdir()
    shutil.copy2(assets / 'specbench480.jsonl', raw_path)
    rows = sorted([json.loads(x) for x in raw_path.read_text().splitlines()], key=lambda r: r['question_id'])
    assert len(rows) == len({r['question_id'] for r in rows}) == 480
    models = {}
    for mid in MODELS:
        spec = specs[mid]
        path = read(assets / f'{mid}.json')['path']
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        tokenized = [dict(question_id=r['question_id'], category=r['category'],
                          prompt=r['turns'][0], prompt_token_ids=tokenize_prompt(spec, tok, r['turns'][0]),
                          model_id=mid) for r in rows]
        max_prompt = max(len(r['prompt_token_ids']) for r in tokenized)
        assert max_prompt + 1024 <= 4096, (mid, max_prompt)
        capacity = ((max_prompt + 1024 + spec.depth_total + 4 + 127) // 128) * 128
        inputs = out / 'inputs' / f'{mid}.jsonl'
        short = out / 'inputs' / f'{mid}-smoke.jsonl'
        for target, data in [(inputs, tokenized), (short, validation_rows(tokenized))]:
            target.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data))
        models[mid] = dict(path=path, inputs=str(inputs), smoke_inputs=str(short),
                           inputs_sha256=digest(inputs), smoke_inputs_sha256=digest(short),
                           max_prompt_tokens=max_prompt, max_seq=capacity,
                           checkpoint=checkpoint_manifest(spec, path))
    lanes = []
    for gpu in range(8):
        start = gpu * 4 if gpu < 4 else 48 + (gpu - 4) * 4
        lane = dict(gpu=gpu, uuid=gpu_snapshot(gpu)['uuid'], cpus=f'{start}-{start+3}',
                    monitor_cpus=[start+2, start+3])
        check_resources(lane)
        lanes.append(lane)
    protocol = dict(version='specbench-table2-v1', created_unix=time.time(), source=str(source),
        python=sys.executable, assets=str(assets), models=models, arms=ARMS, batches=BATCHES,
        planned_cells=75, requests=480, lanes=lanes, raw_inputs_sha256=digest(raw_path),
        dataset_url='https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl',
        source_sha256={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()},
        project='loopfill-specbench-table2', entity='zsqzz', group=out.name,
        resource_policy='Shared cell queue on eight identical exclusive GPUs; refill immediately. Each cell binds one GPU UUID and four local NUMA CPU cores; no cross-method physical-GPU pairing.',
        graphics_clock_mhz=1980, memory_clock_mhz=3201, power_limit_w=700,
        cuda_graph=False, torch_compile=False, pad_decode_rows=False,
        dtype='bfloat16', max_new_tokens=1024, warmup_tokens=32, warmup_requests='min(2B,480)',
        schedule='continuous', prefill_chunk_size=128, attention_backend='packed',
        draft_node_budget=16, suffix_max_draft=4, model_top_k=2, child_suffix_caps=[4,4],
        suffix_c_initial=.7, suffix_eta=.1, suffix_c_min=.25, suffix_c_max=2.,
        seed=0, order_seed=0, timing_repeats=1,
        metric='generated output tokens / synchronized wall time including prefill, decode, CPU scheduling, suffix work and local result persistence; exclude model loading, warmup and network upload',
        extra_metric='(input tokens + output tokens) / same wall time',
        baseline_scope='Same-engine AR, WFD, LoopSpec, Suffix-only',
        failure_policy='Record actual OOM without reducing batch/context/budgets; continue every other cell.',
        recovery='Preserve finished and OOM cells. Restart interrupted cells as a new attempt; never splice throughput windows or suffix histories.',
        tracking='fsync per completed request; separate W&B worker logs every request and live files; durable local W&B spool',
        limits=['Single timing repeat', 'BF16 output lengths may differ; throughput does not establish quality or losslessness'])
    save(out / 'protocol.json', protocol)
    for job in cells():
        save(out / 'jobs' / f"{job['key']}.json", job)
    (out / 'logs').mkdir()
    (out / 'runs').mkdir()
    (out / 'preflight').mkdir()
    if preflight_from is not None:
        # Reuse only completed short streams with identical benchmark/decoder
        # hashes and actual method configs. Supervisor-only fixes do not rerun GPUs.
        from dataclasses import asdict
        from loop_batch_bench import arm_config
        for old_path in sorted((preflight_from/'preflight').glob('*.json')):
            job=read(old_path)
            if job.get('returncode') != 0:continue
            target=Path(job['output'])
            spec=specs[job['model']]
            expected=asdict(arm_config(job['arm'],spec.depth_total,*spec.loopfill_readout_depths,4,
                draft_node_budget=16,model_top_k=2,child_suffix_caps=(4,4),
                suffix_c_initial=.7 if job['arm']=='budget-online' else 1.,
                loopspec_depths=spec.loopspec_depths,wavefront_depth=spec.wavefront_depth))
            if read(target/'metadata.json')['configs'][job['arm']] != json.loads(json.dumps(expected)):
                continue
            try:summary=audit(target,job,protocol,True)
            except (AssertionError,FileNotFoundError,KeyError):continue
            job.update(outcome='complete',summary=summary,reused_from=str(old_path),
                reuse_reason='Identical benchmark/decoder hashes, input hashes and method config; supervisor-only fixes.')
            job.pop('error',None)
            save(out/'preflight'/old_path.name,job)
    save(out / 'runtime.json', dict(python=sys.version,
        pip_freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),
        arctic_revision=subprocess.check_output(['git','-C',str(assets/'ArcticInference'),'rev-parse','HEAD'],text=True).strip(),
        arctic_extension_sha256={p.name:digest(p) for p in (assets/'ArcticInference/arctic_inference/suffix_decoding').glob('*.so')}))
    summarize(out)
    print(json.dumps(dict(out=str(out), cells=75, max_prompts={m:x['max_prompt_tokens'] for m,x in models.items()})), flush=True)


def command(protocol, lane, job, target, smoke=False):
    model = protocol['models'][job['model']]
    cmd = ['taskset', '--cpu-list', lane['cpus'], protocol['python'],
           str(Path(protocol['source'])/'scripts/loop_batch_bench.py')]
    options = {'model':job['model'], 'model-path':model['path'],
        'questions':model['smoke_inputs' if smoke else 'inputs'], 'out':target,
        'arms':job['arm'], 'batch-sizes':2 if smoke else job['batch'], 'scoring':'raw',
        'draft-node-budget':16, 'model-top-k':2, 'suffix-max-draft':4,
        'suffix-c-initial':.7 if job['arm']=='budget-online' else 1., 'max-new-tokens':16 if smoke else 1024,
        'max-seq':model['max_seq'], 'max-nodes':1152, 'buffer-policy':'per-arm',
        'schedule':'continuous', 'attention-backend':'packed', 'prefill-chunk-size':128,
        'warmup-tokens':4 if smoke else 32, 'warmup-request-limit-factor':2,
        'allocation-guard':'disabled', 'prompt-format':'chat', 'dtype':'bfloat16',
        'order-seeds':0, 'seed':0, 'threads':1, 'progress-every':1}
    for k,v in options.items():
        cmd += ['--'+k, str(v)]
    return cmd + ['--child-suffix-caps','4','4','--no-cuda-graph','--no-pad-decode-rows','--durable-results']


def audit(target, job, protocol, smoke=False):
    batch = 2 if smoke else job['batch']
    key = f"{job['arm']}-b{batch}-seed0"
    summary = read(target/f'{key}-summary.json')
    metadata = read(target/'metadata.json')
    records = [json.loads(x) for x in (target/f'{key}.jsonl').read_text().splitlines()]
    durable = [json.loads(x) for x in (target/f'{key}-completed.jsonl').read_text().splitlines()]
    inputs = Path(protocol['models'][job['model']]['smoke_inputs' if smoke else 'inputs'])
    expected = {json.loads(x)['question_id'] for x in inputs.read_text().splitlines()}
    assert (target/'complete.json').exists()
    assert len(records) == len(durable) == summary['requests'] == len(expected)
    assert {r['request_id'] for r in records} == {r['request_id'] for r in durable} == expected
    assert {r['request_id']:r['tokens'] for r in records} == {r['request_id']:r['tokens'] for r in durable}
    assert summary['output_tokens'] == sum(r['n_tokens'] for r in records)
    assert abs(summary['output_tput'] - summary['output_tokens']/summary['wall_s']) < 1e-8
    assert summary['concurrency'] == batch and summary['schedule'] == 'continuous'
    for field in ['cuda_graph', 'pad_decode_rows','cuda_graph_shapes','cuda_graph_replays','core_padding_rows','padding_sink_bytes']:
        assert summary[field] == 0, field
    assert metadata['cuda_graph'] is False and metadata['compile'] is False
    assert metadata['args']['max_new_tokens'] == (16 if smoke else 1024)
    assert metadata['args']['max_seq'] == protocol['models'][job['model']]['max_seq']
    assert metadata['questions_sha256'] == digest(inputs)
    assert metadata['cpu_affinity'] == list(range(int(job['cpus'].split('-')[0]),int(job['cpus'].split('-')[1])+1))
    assert metadata['gpu_uuid'].removeprefix('GPU-') == job['gpu_uuid'].removeprefix('GPU-')
    for name,sha in metadata['source_sha256'].items():
        assert protocol['source_sha256'][name] == sha
    prompt_tokens = sum(len(json.loads(x)['prompt_token_ids']) for x in inputs.read_text().splitlines())
    summary.update(input_tokens=prompt_tokens, total_tput=(prompt_tokens+summary['output_tokens'])/summary['wall_s'])
    return summary


def summarize(out):
    jobs = [read(p) for p in sorted((out/'jobs').glob('*.json'))]
    rows = []
    for j in jobs:
        s = j.get('summary', {})
        rows.append({k:j.get(k) for k in ['key','model','batch','arm','outcome','gpu','gpu_uuid','wandb_url','attempt']} |
                    {k:s.get(k) for k in ['output_tput','total_tput','wall_s','requests','output_tokens','input_tokens','peak_allocated_bytes','result_persistence_s']})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    tmp = out/'measurements.csv.tmp'; tmp.write_text(stream.getvalue()); tmp.replace(out/'measurements.csv')
    save(out/'status.json', dict(updated_unix=time.time(), planned=75, counts=dict(Counter(j['outcome'] for j in jobs)),
        active=[{k:j.get(k) for k in ['key','gpu','wandb_url','progress','started_unix']} for j in jobs if j['outcome']=='running']))
    lookup = {(j['model'],j['arm'],j['batch']):j for j in jobs}
    lines = ['# Table 2 — SpecBench end-to-end output throughput (token/s)', '',
             '480 requests; up to 1024 new tokens; prefill + decode + scheduling; BF16; CUDA Graph off; one timing repeat.', '']
    for model in MODELS:
        lines += [f'## {model}', '', '| Method | B1 | B2 | B4 | B8 | B16 |', '|---|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            values = []
            for batch in BATCHES:
                j=lookup[(model,arm,batch)]
                values.append(f"{j['summary']['output_tput']:.2f}" if j['outcome']=='complete' else j['outcome'])
            lines.append('| '+LABELS[arm]+' | '+' | '.join(values)+' |')
        lines.append('')
    tmp=out/'table2.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(out/'table2.md')


def worker(out, job_path, lane_index, smoke=False):
    from loop_batch_sweep import check_resources, classify
    protocol=read(out/'protocol.json');lane=protocol['lanes'][lane_index]
    job=read(job_path)
    os.sched_setaffinity(0, lane['monitor_cpus'])
    lock=open(f"/tmp/loopfill-{lane['uuid']}.lock",'w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_resources(lane)
    attempt=job.get('attempt',0)+1
    target=out/('preflight' if smoke else 'runs')/job['key']/f'attempt{attempt}'
    target.mkdir(parents=True)
    job.update(attempt=attempt, output=str(target), gpu=lane['gpu'],gpu_uuid=lane['uuid'],
               cpus=lane['cpus'],outcome='starting',started_unix=time.time())
    save(job_path,job)
    import wandb
    run=wandb.init(entity=protocol['entity'],project=protocol['project'],group=protocol['group'],
        id=hashlib.sha256(f"{out.name}/{job['key']}/{attempt}/{smoke}".encode()).hexdigest()[:16],
        name=('preflight-' if smoke else '')+job['key']+f'-a{attempt}',
        job_type='preflight' if smoke else 'throughput',dir=str(target),
        config=dict(model=job['model'],method=job['arm'],batch=2 if smoke else job['batch'],
                    gpu_uuid=lane['uuid'],cpus=lane['cpus'],protocol_sha256=digest(out/'protocol.json'),
                    cuda_graph=False,max_new_tokens=16 if smoke else 1024,attempt=attempt),
        settings=wandb.Settings(disable_git=True,disable_code=True,console='off',init_timeout=90))
    job.update(wandb_url=run.url,outcome='running');save(job_path,job)
    run.save(str(out/'protocol.json'),base_path=str(out),policy='now')
    run.save(str(out/'runtime.json'),base_path=str(out),policy='now')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=lane['uuid'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
             PYTHONPATH=protocol['source']+'/src:'+protocol['assets']+'/ArcticInference')
    env.pop('WANDB_API_KEY',None)  # The measured process does not need credentials.
    cmd=command(protocol,lane,job,target,smoke)
    save(target/'command.json',cmd)
    key=f"{job['arm']}-b{2 if smoke else job['batch']}-seed0"
    journal=target/f'{key}-completed.jsonl'
    offset=0;count=0;tokens=0;registered=set();telemetry_at=0
    with (target/'process.log').open('w') as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=protocol['source'])
        job['pid']=process.pid;save(job_path,job)
        while True:
            rc=process.poll()
            if journal.exists():
                with journal.open() as handle:
                    handle.seek(offset)
                    while True:
                        line=handle.readline()
                        if not line or not line.endswith('\n'):break
                        row=json.loads(line);offset=handle.tell();count+=1;tokens+=row['n_tokens']
                        metrics={'progress/completed_requests':count,'progress/output_tokens':tokens,
                                 'progress/elapsed_s':row['completed_s'],
                                 'request/id':row['request_id'],'request/output_tokens':row['n_tokens'],
                                 'request/latency_s':row['latency_s'],'request/ttft_s':row['ttft_s']}
                        run.log(metrics)
                        job['progress']=dict(completed_requests=count,output_tokens=tokens,elapsed_s=row['completed_s'])
                save(job_path,job)
            for p in [journal,target/'metadata.json',target/'command.json',target/f'{key}-progress.json']:
                if p.exists() and str(p) not in registered:
                    run.save(str(p),base_path=str(target),policy='live');registered.add(str(p))
            if time.time()-telemetry_at>15:
                from loop_batch_sweep import gpu_snapshot
                snap=gpu_snapshot(lane['gpu']);telemetry_at=time.time()
                with (target/'gpu-telemetry.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(unix=time.time(),**snap))+'\n')
                run.log({'gpu/'+k:float(snap[k]) for k in ['clocks.sm','clocks.mem','power.limit','temperature.gpu','memory.used','utilization.gpu']})
            if rc is not None:break
            time.sleep(5)
    job.update(returncode=rc,outcome=classify(rc,target),finished_unix=time.time())
    if job['outcome']=='complete':
        try:
            job['summary']=audit(target,job,protocol,smoke)
        except Exception as exc:
            job.update(outcome='audit_failed',error=repr(exc))
    run.summary['outcome']=job['outcome']
    run.summary['completed_requests']=count
    for k,v in job.get('summary',{}).items():
        if isinstance(v,(int,float,str,bool)):
            run.summary[k]=v
    save(job_path,job)
    for p in target.iterdir():
        if p.is_file():run.save(str(p),base_path=str(target),policy='now')
    run.save(str(job_path),base_path=str(out),policy='now')
    run.finish(exit_code=0 if job['outcome']=='complete' else 1)
    job['tracking_finished']=True;save(job_path,job)
    print(json.dumps({k:job.get(k) for k in ['key','outcome','gpu','wandb_url']}),flush=True)


def run_suite(out):
    protocol=read(out/'protocol.json')
    lock=(out/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name,sha in protocol['source_sha256'].items():
        assert digest(Path(protocol['source'])/name)==sha, name
    for model in protocol['models'].values():
        assert digest(model['inputs'])==model['inputs_sha256']
        assert digest(model['smoke_inputs'])==model['smoke_inputs_sha256']
    save(out/'supervisor.json',dict(pid=os.getpid(),started_unix=time.time(),argv=sys.argv))
    # Every model/method gets a short stratified smoke run, never counted as Table 2.
    smoke_paths=[]
    for m in MODELS:
        for arm in ARMS:
            path=out/'preflight'/f'{m}-{arm}.json'
            if not path.exists():save(path,dict(key=f'{m}-{arm}',model=m,arm=arm,batch=2,outcome='pending'))
            if read(path)['outcome']!='complete':smoke_paths.append(path)
    status_lock=threading.Lock()
    def phase(paths,smoke):
        pending=queue.Queue()
        for p in paths:pending.put(p)
        def lane(index):
            while True:
                try:path=pending.get_nowait()
                except queue.Empty:return
                j=read(path)
                cmd=[protocol['python'],str(Path(protocol['source'])/'scripts/loop_table2_suite.py'),
                     'worker','--out',str(out),'--job',str(path),'--lane',str(index)]
                if smoke:cmd+=['--smoke']
                env=dict(os.environ,PYTHONPATH=protocol['source']+'/src:'+str(Path(protocol['source'])/'scripts')+':'+protocol['assets']+'/ArcticInference',
                         WANDB_SILENT='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                log_path=out/'logs'/f"{j['key']}-{'smoke' if smoke else 'full'}-worker.log"
                with log_path.open('a') as log:
                    p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    while p.poll() is None:
                        if not smoke:
                            with status_lock:summarize(out)
                        time.sleep(5)
                current=read(path)
                if p.returncode and current['outcome'] not in ['complete','oom','failed','audit_failed']:
                    current.update(outcome='worker_failed',worker_returncode=p.returncode,worker_log=str(log_path))
                    save(path,current)
                with status_lock:summarize(out)
                print(json.dumps(dict(key=j['key'],outcome=read(path)['outcome'],gpu=index,smoke=smoke)),flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lane,range(8)))
    phase(smoke_paths,True)
    failures=[p.name for p in (out/'preflight').glob('*.json') if read(p)['outcome']!='complete']
    if failures:
        save(out/'preflight-failed.json',failures)
        raise RuntimeError(f'Preflight failures: {failures}')
    jobs=[out/'jobs'/f"{j['key']}.json" for j in cells()]
    phase([p for p in jobs if read(p)['outcome'] not in ['complete','oom']],False)
    summarize(out)
    save(out/'suite-finished.json',read(out/'status.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','worker','summarize'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assets',type=Path,default=Path('/tmp/loopfill-assets'))
    p.add_argument('--job',type=Path)
    p.add_argument('--lane',type=int)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight-from',type=Path)
    args=p.parse_args();out=args.out.resolve()
    if args.action=='prepare':prepare(out,args.assets,args.preflight_from)
    elif args.action=='run':run_suite(out)
    elif args.action=='worker':worker(out,args.job,args.lane,args.smoke)
    else:summarize(out)


if __name__=='__main__':main()
