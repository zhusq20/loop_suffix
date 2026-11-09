"""Frozen, stratified stage-timing pilot; one physical GPU per model across methods."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import loop_table2_suite as base
from loop_accepted_length import MT_CATEGORIES, TASKS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['huginn-3.5b', 'ouro-2.6b']
ARMS = ['ar', 'loopspec-dual', 'budget-suffix', 'budget-online']
read, save, digest = base.read, base.save, base.digest


def select_ids(rows, per_task=4):
    groups = defaultdict(list)
    for row in rows:
        task = 'mt_bench' if row['category'] in MT_CATEGORIES else row['category']
        assert task in TASKS
        groups[task].append(row)
    assert set(groups) == TASKS
    chosen, smoke = [], []
    for task, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda r: hashlib.sha256(
            f"mechanism-stage-seed0/{r['question_id']}".encode()).hexdigest())
        assert len(ranked) >= per_task
        chosen += [r['question_id'] for r in ranked[:per_task]]
        smoke.append(ranked[0]['question_id'])
    return sorted(chosen), sorted(smoke)


def prepare(out, parent):
    if out.exists():
        raise FileExistsError(out)
    p = read(parent/'protocol.json')
    source = Path(p['source'])
    for rel, expected in p['source_sha256'].items():
        assert digest(source/rel) == expected, rel
    out.mkdir(parents=True)
    shutil.copytree(source, out/'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['loop_mechanism_suite.py', 'loop_stage_timing.py']:
        shutil.copy2(ROOT/'scripts'/name, out/'source/scripts'/name)
    # Supervisor-only GPU lock fixes; inference modules remain byte-identical.
    shutil.copy2(ROOT/'scripts/loop_table2_suite.py', out/'source/scripts/loop_table2_suite.py')
    for folder in ['inputs', 'jobs', 'runs', 'logs']:
        (out/folder).mkdir()
    raw = [json.loads(line) for line in (parent/'inputs/specbench480.jsonl').read_text().splitlines()]
    ids, smoke = select_ids(raw)
    models = {}
    for model in MODELS:
        info = dict(p['models'][model])
        original = Path(info['inputs'])
        assert digest(original) == info['inputs_sha256']
        tokenized = [json.loads(line) for line in original.read_text().splitlines()]
        for field, chosen in [('inputs', ids), ('smoke_inputs', smoke)]:
            path = out/'inputs'/f'{model}-{field}.jsonl'
            rows = sorted((r for r in tokenized if r['question_id'] in chosen), key=lambda r:r['question_id'])
            assert [r['question_id'] for r in rows] == chosen
            path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            info[field], info[field+'_sha256'] = str(path), digest(path)
        models[model] = info
    p.update(version='mechanism-stage-v1', created_unix=time.time(), source=str(out/'source'),
             python=sys.executable, parent_protocol=str(parent/'protocol.json'),
             models=models, arms=ARMS, batches=[16, 1], planned_cells=32,
             requests=24, request_ids=ids, smoke_ids=smoke,
             warmup_requests='min(2B,24); diagnostic smoke min(2B,6)',
             lanes=p['lanes'][:2], group=out.name, project='llm-memory-editability',
             resource_policy='One reserved GPU per model; all methods and both B sequential on that GPU.',
             selection='Six SpecBench tasks x4, SHA256(mechanism-stage-seed0/question_id); '
                       'chosen without looking at outputs; order-seed=0.',
             scope='16 configurations, each one normal timing and one separate CUDA-event diagnostic. '
                   '24 requests/stream, max1024, not new full480 performance measurements.',
             order='Per model: four diagnostic B2/max16 smoke checks, then B16, B1; '
                   'AR, LoopSpec, Suffix-only, LoopFill; normal then diagnostic for each configuration.',
             metric='Normal: output tokens / synchronized stream wall (prefill/decode/scheduling/persistence). '
                    'Diagnostic host-exclusive wall and CUDA stream spans are separate clocks; never sum them.',
             limits=['Single timing; short finite queue and cold history; B16 not established saturation.',
                     'CUDA event spans can contain host submission gaps; they are not kernel-active time.',
                     'Host scopes include waiting; combined readouts cannot isolate shallow LM-head time.',
                     'BF16 outputs and completion-order history can differ between normal and diagnostic.',
                     'No B4/B8 or additional source-transition trace is scheduled.'])
    p['source_sha256'] = {str(f.relative_to(out/'source')): digest(f)
                          for f in (out/'source').rglob('*') if f.is_file()}
    p['inference_reuse_audit'] = {rel: sha for rel, sha in read(parent/'protocol.json')['source_sha256'].items()
                                if rel.startswith('src/') or rel=='scripts/loop_batch_bench.py'}
    for rel, sha in p['inference_reuse_audit'].items():
        assert p['source_sha256'][rel] == sha
    save(out/'protocol.json', p)
    runtime = read(parent/'runtime.json')
    runtime.update(current_python=sys.version, parent_runtime=str(parent/'runtime.json'))
    save(out/'runtime.json', runtime)
    for lane, model in enumerate(MODELS):
        sequence = 0
        for phase, batches, modes in [('preflight', [2], ['diagnostic']),
                                      ('measurement', [16,1], ['normal','diagnostic'])]:
            for b in batches:
                for arm in ARMS:
                    for mode in modes:
                        key = f'{model}-b{b}-{arm}-{mode}'
                        save(out/'jobs'/f'{key}.json', dict(key=key, model=model, batch=b,
                             arm=arm, mode=mode, phase=phase, lane=lane, sequence=sequence,
                             outcome='pending'))
                        sequence += 1
    status(out)
    print(json.dumps(dict(out=str(out), requests_per_stream=24, configurations=16,
                          measurement_streams=32, preflight_streams=8, ids=ids)))


def status(out):
    jobs = [read(f) for f in sorted((out/'jobs').glob('*.json'))]
    save(out/'status.json', dict(updated_unix=time.time(),
         measurement_counts=dict(Counter(j['outcome'] for j in jobs if j['phase']=='measurement')),
         preflight_counts=dict(Counter(j['outcome'] for j in jobs if j['phase']=='preflight')),
         active=[{k:j.get(k) for k in ['key','pid','gpu','output','progress','started_unix','wandb_url']}
                 for j in jobs if j['outcome']=='running']))


def validate_job(target, job, p):
    summary = base.audit(target, job, p, smoke=job['phase']=='preflight')
    prefix = f"{job['arm']}-b{job['batch']}-seed0"
    rows = [json.loads(line) for line in (target/f'{prefix}.jsonl').read_text().splitlines()]
    metadata = read(target/'metadata.json')
    config = metadata['configs'][job['arm']]
    from dataclasses import asdict
    from loop_batch_bench import arm_config
    from llm_memory_editability.loop_model_registry import MODELS as specs
    spec = specs[job['model']]
    expected = asdict(arm_config(job['arm'], spec.depth_total, *spec.loopfill_readout_depths, 4,
        draft_node_budget=16, model_top_k=2, child_suffix_caps=(4,4),
        suffix_c_initial=.7 if job['arm']=='budget-online' else 1.,
        loopspec_depths=spec.loopspec_depths, wavefront_depth=spec.wavefront_depth))
    # Table 2 predates optional sampling fields. Its hashed inference snapshot
    # has the same deterministic defaults without serializing these fields.
    for field, default in [('selection_policy','top_n'),('draft_seed',0),('draft_model_id','')]:
        if field not in config and field in expected:
            assert expected[field] == default
            expected.pop(field)
    assert config == json.loads(json.dumps(expected)) == summary['config']
    assert metadata['history'] == 'cold_start_completion_only'
    assert not metadata['profiled'] and not metadata['controller_profiled']
    for row in rows:
        assert row['stats']['tokens'] == row['n_tokens']-1
        if job['arm']=='ar':
            assert row['stats']['calls'] == spec.depth_total*row['stats']['tokens']
    if job['mode']=='diagnostic':
        stages = read(target/f'{prefix}-stages.json')
        assert summary['stage_profiled'] and not summary['formal_throughput']
        assert stages['decode_tokens'] == sum(r['stats']['tokens'] for r in rows)
        assert abs(stages['host_accounting_error_s']) < 1e-6
        assert all(v>=0 for v in stages['host_exclusive_wall_s'].values())
        assert stages['cuda_event_pairs'].get('recurrent_forward',0)>0
    else:
        assert not metadata.get('stage_profiled',False)
    save(target/'audit.json', dict(passed=True, requests=len(rows),
         records_sha256=digest(target/f'{prefix}.jsonl'), metadata_sha256=digest(target/'metadata.json')))
    return summary


def lane_run(out, index):
    p = read(out/'protocol.json')
    lane = p['lanes'][index]
    reservation = base.try_gpu_lock(lane)
    while reservation is None:
        time.sleep(15)
        reservation = base.try_gpu_lock(lane)
    tracking = None
    try:
        key_file = Path('/tmp/loopfill-wandb-key')
        if 'WANDB_API_KEY' not in os.environ and key_file.exists():
            os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
        import wandb
        tracking = wandb.init(entity=p['entity'], project=p['project'], group=p['group'],
            name=MODELS[index]+'-stage-timing', job_type='mechanism-diagnostic', dir=str(out),
            config=dict(protocol_sha256=digest(out/'protocol.json'), model=MODELS[index], gpu=lane),
            settings=wandb.Settings(disable_git=True,disable_code=True,console='off',init_timeout=60))
        for name in ['protocol.json','runtime.json']:
            tracking.save(str(out/name), base_path=str(out), policy='now')
    except Exception as exc:
        save(out/f'tracking-error-lane{index}.json', dict(error=repr(exc)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane['uuid'], OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONUNBUFFERED='1', PYTHONPATH=p['source']+'/src:'+p['assets']+'/ArcticInference')
    env.pop('WANDB_API_KEY',None)
    jobs = sorted((read(f) for f in (out/'jobs').glob('*.json') if read(f)['lane']==index),
                  key=lambda j:j['sequence'])
    failed_preflight = False
    for job in jobs:
        if job['outcome']=='complete':
            continue
        if job['outcome']!='pending':
            raise RuntimeError(f"Inspect previous attempt before resume: {job['key']}")
        if failed_preflight and job['phase']=='measurement':
            break
        target = out/'runs'/job['key']
        target.mkdir(exist_ok=False)
        path = out/'jobs'/f"{job['key']}.json"
        job.update(output=str(target), gpu=lane['gpu'], gpu_uuid=lane['uuid'], cpus=lane['cpus'],
                   outcome='running', started_unix=time.time(), wandb_url=tracking.url if tracking else None)
        cmd = base.command(p,lane,job,target,smoke=job['phase']=='preflight')
        if job['mode']=='diagnostic':
            cmd[4] = str(Path(p['source'])/'scripts/loop_stage_timing.py')
        save(target/'command.json',cmd)
        save(path,job)
        try:
            with (target/'process.log').open('w') as log:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        cwd=p['source'], pass_fds=(reservation.fileno(),))
                job['pid']=proc.pid
                save(path,job)
                while proc.poll() is None:
                    progress = target/f"{job['arm']}-b{job['batch']}-seed0-progress.json"
                    if progress.exists():
                        job['progress']=read(progress)
                        save(path,job)
                    time.sleep(10)
                job['returncode']=proc.returncode
            if proc.returncode:
                raise RuntimeError(f'benchmark exited {proc.returncode}; see {target}/process.log')
            job['summary']=validate_job(target,job,p)
            job['outcome']='complete'
        except Exception as exc:
            job.update(outcome='failed', error=repr(exc))
            failed_preflight |= job['phase']=='preflight'
        job['finished_unix']=time.time()
        save(path,job)
        if tracking:
            try:
                values = dict(batch=job['batch'],method=job['arm'],mode=job['mode'],phase=job['phase'],
                              complete=job['outcome']=='complete')
                if job.get('summary'):
                    values.update({job['mode']+'/'+k:job['summary'][k]
                                   for k in ['wall_s','output_tokens','requests','output_tput']})
                tracking.log(values)
            except Exception as exc:
                save(out/f'tracking-log-error-lane{index}.json',dict(error=repr(exc)))
        print(job['key'], job['outcome'], flush=True)
    reservation.close()
    if tracking:
        tracking.finish()
    if failed_preflight:
        raise RuntimeError('Preflight failure; dependent measurement streams were not started')


def run(out):
    lock = (out/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(out/'protocol.json')
    for rel, sha in p['source_sha256'].items():
        assert digest(Path(p['source'])/rel)==sha, rel
    for info in p['models'].values():
        for field in ['inputs','smoke_inputs']:
            assert digest(info[field])==info[field+'_sha256']
    save(out/'supervisor.json',dict(pid=os.getpid(),started_unix=time.time()))
    children=[]
    for index in range(2):
        log=(out/'logs'/f'lane{index}.log').open('a')
        child=subprocess.Popen([p['python'],str(Path(p['source'])/'scripts/loop_mechanism_suite.py'),
            'lane','--out',str(out),'--lane',str(index)],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=p['source']+'/src:'+p['assets']+'/ArcticInference',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        log.close()
        children.append(child)
    while any(c.poll() is None for c in children):
        status(out)
        time.sleep(15)
    status(out)
    analyze(out)
    if any(c.returncode for c in children):
        raise RuntimeError(f'lane failures: {[c.returncode for c in children]}')


def analyze(out):
    p=read(out/'protocol.json')
    rows, stages, pairs=[] , [], []
    jobs=[read(f) for f in sorted((out/'jobs').glob('*.json'))]
    for j in jobs:
        if j['phase']!='measurement' or j['outcome']!='complete':
            continue
        target=Path(j['output'])
        s=validate_job(target,j,p)
        prefix=f"{j['arm']}-b{j['batch']}-seed0"
        requests=[json.loads(line) for line in (target/f'{prefix}.jsonl').read_text().splitlines()]
        depth=32 if j['model']=='huginn-3.5b' else 4
        tokens=sum(r['stats']['tokens'] for r in requests)
        calls=sum(r['stats']['calls'] for r in requests)
        item={k:j[k] for k in ['key','model','batch','arm','mode','gpu_uuid']}
        item.update(requests=len(requests),output_tokens=s['output_tokens'],decode_tokens=tokens,
                    wall_s=s['wall_s'],output_tput=s['output_tput'],gamma=depth*tokens/calls,
                    rows_per_token=sum(r['stats']['rows'] for r in requests)/tokens,
                    mean_decode_latency_s=sum(r['latency_s']-r['ttft_s'] for r in requests)/len(requests)
                    if j['batch']==1 else None)
        rows.append(item)
        if j['mode']=='diagnostic':
            timing=read(target/f'{prefix}-stages.json')
            for clock in ['host_exclusive_wall_s','cuda_stream_span_s']:
                for stage, seconds in timing[clock].items():
                    stages.append(dict(model=j['model'],batch=j['batch'],arm=j['arm'],clock=clock,
                                       stage=stage,seconds=seconds,
                                       ms_per_output_token=1000*seconds/s['output_tokens'],
                                       fraction_of_diagnostic_wall=seconds/timing['diagnostic_wall_s']))
            normal=out/'runs'/j['key'].replace('-diagnostic','-normal')
            normal_rows=[json.loads(line) for line in (normal/f'{prefix}.jsonl').read_text().splitlines()]
            assert [r['request_id'] for r in normal_rows]==[r['request_id'] for r in requests]
            ns=read(normal/f'{prefix}-summary.json')
            pairs.append(dict(model=j['model'],batch=j['batch'],arm=j['arm'],requests=len(requests),
                              same_output_requests=sum(a['tokens']==b['tokens'] for a,b in zip(normal_rows,requests)),
                              same_publication_order=ns.get('publications')==s.get('publications'),
                              diagnostic_over_normal_wall=s['wall_s']/ns['wall_s']))
    for filename, data in [('measurements.csv',rows),('stages.csv',stages),('pairs.csv',pairs)]:
        if data:
            with (out/filename).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    save(out/'audit.json',dict(complete_streams=len(rows),planned_streams=32,
         all_complete=len(rows)==32,pairs=pairs,normal_timing_streams=sum(r['mode']=='normal' for r in rows)))
    if stages:
        plot(out,rows,stages)
    lines=['# Stage timing pilot','',f'{len(rows)}/32 measurement streams complete.', '',
           '24 stratified SpecBench requests, max1024. Normal timings and diagnostic spans are separate. '
           'Host stacks partition diagnostic wall time and include waits; CUDA event spans are shown '
           'separately, may include submission gaps, and cannot be added to host times. '
           'Readout combines shallow/full-depth coda and LM head; recurrent includes attention KV work.','',
           '| Model | B | Method | Normal output token/s | Gamma | Rows/token |',
           '| --- | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        if r['mode']=='normal':
            lines.append(f"| {r['model']} | {r['batch']} | {r['arm']} | {r['output_tput']:.2f} | {r['gamma']:.3f} | {r['rows_per_token']:.2f} |")
    lines += ['', 'This short, single-run pilot does not replace full480 performance results. '
              'Output agreement and completion-history order are recorded in pairs.csv; equality '
              'does not establish causal stage-specific speedups. Profiler/event instrumentation '
              'overhead is retained, not subtracted from normal timings.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def plot(out,rows,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(15,13),constrained_layout=True)
    for i,(model,batch) in enumerate((m,b) for m in MODELS for b in [1,16]):
        normal={r['arm']:r for r in rows if r['model']==model and r['batch']==batch and r['mode']=='normal'}
        axes[i,0].bar(ARMS,[1000/normal[a]['output_tput'] if a in normal else 0 for a in ARMS])
        axes[i,0].set(title=f'{model}, B{batch}: normal',ylabel='E2E ms/output token')
        for col,clock in [(1,'host_exclusive_wall_s'),(2,'cuda_stream_span_s')]:
            subset=[r for r in stages if r['model']==model and r['batch']==batch and r['clock']==clock]
            names=sorted({r['stage'] for r in subset})
            bottom=[0.0]*len(ARMS)
            for k,stage in enumerate(names):
                values=[sum(r['ms_per_output_token'] for r in subset if r['arm']==a and r['stage']==stage) for a in ARMS]
                if col==1:
                    axes[i,col].bar(ARMS,values,bottom=bottom,label=stage)
                    bottom=[a+b for a,b in zip(bottom,values)]
                else:
                    width=.8/max(1,len(names))
                    axes[i,col].bar([x-.4+width*(k+.5) for x in range(len(ARMS))],values,width,label=stage)
                    axes[i,col].set_xticks(range(len(ARMS)),ARMS)
            axes[i,col].set(title='Diagnostic host wall partition' if col==1 else 'CUDA spans (separate clock)',
                            ylabel='ms/output token')
            axes[i,col].legend(fontsize=5,loc='upper right')
        for ax in axes[i]:
            ax.tick_params(axis='x',labelrotation=30)
    for ext in ['pdf','png']:
        fig.savefig(out/f'stage-timing.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','lane','status','analyze'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=Path('results/loopfill-table2-20261107-v2'))
    parser.add_argument('--lane',type=int)
    args=parser.parse_args()
    out=args.out.resolve()
    if args.action=='prepare':
        prepare(out,args.parent.resolve())
    elif args.action=='lane':
        lane_run(out,args.lane)
    else:
        globals()[args.action](out)
