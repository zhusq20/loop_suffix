"""Freeze and run the two-model Table 1 agentic supplement on four assigned GPUs."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from loop_agentic_worker import digest, save

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
PYTHON = '/tmp/loopfill-dflash-venv/bin/python'
ARMS = ['ar','wavefront','loopspec-dual','budget-suffix','loopahead','dflash','budget-online']


def freeze(suite):
    if (suite/'protocol.json').exists():
        raise FileExistsError('Protocol is already frozen')
    manifest = json.loads((suite/'inputs/manifest.json').read_text())
    source = suite/'source'
    source.mkdir(exist_ok=False)
    shutil.copytree(REPO/'src', source/'src', ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info'))
    shutil.copytree(REPO/'baselines', source/'baselines',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'scripts').mkdir()
    for name in ['loop_agentic_inputs.py','loop_agentic_sql.py','loop_agentic_worker.py','loop_agentic_suite.py',
                 'loop_batch_bench.py','loop_latency_worker.py']:
        shutil.copy2(REPO/'scripts'/name, source/'scripts'/name)
    shutil.copytree('/tmp/loopfill-assets/ArcticInference/arctic_inference',
                    source/'suffix/arctic_inference', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'runtime').mkdir()
    shutil.copy2(BASE/'native_cuda_env.py', source/'runtime/native_cuda_env.py')
    shutil.copy2(BASE/'configs/loopahead-ouro-2.6b-admission-v1.json', suite/'inputs/loopahead-admission.json')
    models={}
    for name in ['huginn-3.5b','ouro-2.6b']:
        lad_root=BASE/f'checkpoints/loopahead-{name}-formal-v1'
        lad=Path(json.loads((lad_root/'complete.json').read_text())['selected_checkpoint'])/'lad.pt'
        droot=BASE/'checkpoints'/('dflash-huginn-narrow-formal-v1' if name.startswith('huginn') else f'dflash-{name}-formal-v1')
        dflash=Path(json.loads((droot/'latest.json').read_text())['checkpoint'])
        assert (droot/'complete.json').exists() and (dflash/'complete.json').exists()
        models[name]=dict(model_path=manifest['model_paths'][name],input_sha256=manifest['model_inputs'][name]['sha256'],
            loopahead_checkpoint=str(lad),dflash_checkpoint=str(dflash),
            checkpoint_sha256=dict(loopahead=digest(lad),dflash=digest(dflash/'model.safetensors'),
                                   dflash_config=digest(dflash/'config.json')))
    protocol = dict(name=suite.name, created_unix=time.time(), models=models,
        dataset=manifest['dataset'],dataset_id=manifest.get('dataset_id','openhands100'),
        input_semantics=manifest['semantics'],
        max_seq=4096, max_new_tokens=512,
        input_sha256=manifest['input_sha256'], python=PYTHON, temperatures=[0,1],
        main_arms=ARMS, source_sha256={str(p.relative_to(source)):digest(p)
            for p in sorted(source.rglob('*')) if p.is_file()},
        input_files_sha256={p.name:digest(p) for p in sorted((suite/'inputs').glob('*'))},
        selection=manifest['selection'], cache='Empty global cache at cell start, completed outputs accumulated; '
            'all methods receive the same complete historical prompt. No ablations in this Table 1 batch.',
        allowed_gpus=[0,1,5,7],
        gpu_bindings={'huginn-3.5b/t0':0,'huginn-3.5b/t1':1,'ouro-2.6b/t0':5,'ouro-2.6b/t1':7},
        scheduling='User explicitly assigned the four idle GPUs 0/1/5/7 for this Table 1 supplement. '
                   'Do not queue on GPUs 2/3/4/6 or alter other experiment allocations. UUID locks and exclusive mode required.',
        sampling_seed=0, timing_repeats=1, graph=False, compile=False, pad_decode_rows=False,
        scope='Table 1 supplement: both Huginn-3.5B and Ouro-2.6B, all seven methods, temperatures 0/1. '
              'B1 decode latency and rate; not concurrent throughput or live-agent task success.',
        limitations=['Single sampling seed and timing repeat.',
            'Identical request selection across both models; <=3548 prompt tokens in each tokenizer; no truncation.',
            'DFlash configured training length is 3072; agentic context extrapolation is reported.',
            'BF16 execution shapes can change target logits; T1 samplers are not expected to give paired identical outputs.'])
    save(suite/'protocol.json', protocol)
    jobs = []
    for model in models:
        for temperature in [0,1]:
            for arm in ARMS:
                jobs.append(dict(id=f'{model}-{arm}-t{temperature}',model=model,arm=arm,
                    temperature=temperature,q=4,n=16,cache_mode='stream',stage='main'))
    save(suite/'jobs.json',jobs)
    save(suite/'status.json',dict(status='prepared',main_cells=28,completed=0,total=len(jobs)))
    print(json.dumps(dict(status='prepared',jobs=len(jobs),suite=str(suite))))


def allowed(protocol):
    return protocol['allowed_gpus']


def summarize(suite):
    jobs = json.loads((suite/'jobs.json').read_text())
    models = list(dict.fromkeys(job['model'] for job in jobs))
    table = []
    for job in jobs:
        directory = suite/'runs'/job['id']
        if (directory/'summary.json').exists():
            result = json.loads((directory/'summary.json').read_text())
            rows = [json.loads(line) for line in (directory/'requests.jsonl').open()]
            times = sorted(row['stats']['decode_s'] for row in rows)
            table.append(dict(**job, status=result['status'], count=len(rows),
                decode_s=result['decode_s'], decode_tokens=result['decode_tokens'],
                decode_token_s=result['decode_token_s'], gamma=result['gamma'],
                mean_decode_s=result['mean_decode_s'], p50_decode_s=times[(len(times)-1)//2],
                p95_decode_s=times[min(len(times)-1, int(.95*len(times)))],
                pooled_candidate_coverage=result.get('pooled_acceptance_rate'),
                prefill_s=result['prefill_s'], cap_stopped=result['cap_stopped'],
                reportable=result['reportable']))
    ar = {(r['model'],r['temperature']):r for r in table if r['arm']=='ar'}
    for row in table:
        base = ar.get((row['model'],row['temperature']))
        if base and row['reportable']:
            row.update(latency_speedup=base['decode_s']/row['decode_s'],
                       decode_rate_speedup=row['decode_token_s']/base['decode_token_s'],
                       output_token_ratio=row['decode_tokens']/base['decode_tokens'])
    fields = sorted({k for row in table for k in row})
    if fields:
        with (suite/'measurements.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    manifest=json.loads((suite/'inputs/manifest.json').read_text())
    lines=['# Agent-request decoding experiment', '',manifest['dataset'],'',
        f"Frozen requests: {manifest['count']}; tasks: {manifest['instances']}; phase counts: {manifest['phase_counts']}.",
        '', f"Table 1 supplement: B=1, {', '.join(models)}, BF16, max512, seed0, single timing repeat. Fixed historical prompts; "
        'new target generation. This is not a live-agent success-rate evaluation.', '',
        f"Completed formal cells: {len(table)}/{len(jobs)}. Missing cells are pending or failed; consult status.json.",
        '', '| Cell | Requests | Decode tok/s | Mean decode s | p95 decode s | γ | Latency speedup |',
        '|---|---:|---:|---:|---:|---:|---:|']
    phase_rows=[]
    retrieval={}
    for row in table:
        if row['reportable']:
            speed=f"{row['latency_speedup']:.3f}×" if 'latency_speedup' in row else 'pending AR'
            lines.append(f"| {row['id']} | {row['count']} | {row['decode_token_s']:.3f} | "
                         f"{row['mean_decode_s']:.3f} | {row['p95_decode_s']:.3f} | {row['gamma']:.3f} | {speed} |")
        records=[json.loads(line) for line in (suite/'runs'/row['id']/'requests.jsonl').open()]
        if row['stage']=='diagnostic':
            totals={kind:{} for kind in ('root','child')}
            for record in records:
                for kind,counts in record.get('retrieval',{}).items():
                    for k,v in counts.items():
                        totals[kind][k]=totals[kind].get(k,0)+v
            retrieval[row['id']]=totals
        elif row['reportable']:
            for phase in ('early','middle','late'):
                group=[r for r in records if r['phase']==phase]
                if not group:
                    continue
                seconds=sum(r['stats']['decode_s'] for r in group)
                tokens=sum(r['stats']['tokens'] for r in group)
                phase_rows.append(dict(cell=row['id'],phase=phase,count=len(group),
                    decode_s=seconds,decode_tokens=tokens,decode_token_s=tokens/seconds if seconds else None,
                    mean_prompt_tokens=sum(r['prompt_tokens'] for r in group)/len(group)))
    lines.extend(['', 'Decode throughput is generated decode tokens divided by synchronized decode time; '
        'it excludes prefill and is not concurrent serving throughput. Latency speedup is the ratio of total '
        'decode times; generated lengths may differ. Compare output_token_ratio in measurements.csv.', '',
        'DFlash uses the trained local block16 drafter and the packed target adapter. Its configured training '
        'length is 3072; requests above this length are extrapolation. All methods first run shortest/longest-context '
        'smoke checks. Diagnostic timings are excluded. No confidence interval or task-quality claim is made.'])
    lines.extend(['',manifest['semantics']])
    (suite/'report.md').write_text('\n'.join(lines)+'\n')
    # Separate supplement, preserving existing Table 1 columns and their raw data.
    index={(r['model'],r['temperature'],r['arm']):r for r in table if r['reportable']}
    labels={'ar':'AR','wavefront':'WFD','loopspec-dual':'LoopSpec','budget-suffix':'Suffix-only',
            'loopahead':'LoopAhead','dflash':'DFlash','budget-online':'LoopFill-online'}
    tex=['% Generated from this suite only. Cell = latency speedup / loop-normalized gamma.',
         '% Pending cells are --. See protocol.json and report.md for the workload and timing boundaries.']
    for model in models:
        tex += ['% '+model,r'\begin{tabular}{lrr}',r'\hline',
                r'Method & Agentic ($T=0$) & Agentic ($T=1$) \\',r'\hline']
        for arm in ARMS:
            cells=[]
            for temperature in (0,1):
                row=index.get((model,temperature,arm))
                cells.append(f"{row['latency_speedup']:.2f} / {row['gamma']:.2f}"
                             if row and 'latency_speedup' in row else '--')
            tex.append(' & '.join([labels[arm]]+cells)+r' \\')
        tex += [r'\hline',r'\end{tabular}','']
    (suite/'table1-agentic.tex').write_text('\n'.join(tex)+'\n')
    if phase_rows:
        with (suite/'phase-measurements.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(phase_rows[0]))
            writer.writeheader()
            writer.writerows(phase_rows)
    if retrieval:
        save(suite/'retrieval-diagnostics.json',dict(instrumented=True,formal_timing=False,cells=retrieval,
            definition='Actual queries; root path coverage scored against eventual generated continuation. '
                       'Query nonempty rate is different from accepted-node rate or saved computation.'))
    return table


def validate_scheduling(suite, scheduling):
    """Check an explicit scheduling amendment without changing frozen measurements."""
    if scheduling is None:
        return
    assert scheduling['placement'] == 'first_idle_per_cell'
    assert digest(suite/'protocol.json') == scheduling['protocol_sha256']
    assert digest(suite/'jobs.json') == scheduling['jobs_sha256']
    assert scheduling['allowed_gpus'] == list(range(8))
    for name, expected in scheduling['scheduler_sha256'].items():
        assert digest(Path(scheduling['scheduler_root'])/name) == expected, name


def run_queue(suite, validation_only=False, scheduling=None):
    validate_scheduling(suite, scheduling)
    protocol=json.loads((suite/'protocol.json').read_text())
    independent = scheduling is not None
    if independent:
        protocol['allowed_gpus'] = scheduling['allowed_gpus']
    jobs=json.loads((suite/'jobs.json').read_text())
    (suite/'logs').mkdir(exist_ok=True)
    with (suite/'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Legacy suites bind temperatures; amended suites place each cell independently.
        state_path=suite/'dispatch.json'
        dispatch=json.loads(state_path.read_text()) if state_path.exists() else dict(bindings=protocol['gpu_bindings'],attempts={})
        if independent:
            dispatch['placement'] = scheduling['placement']
            dispatch.setdefault('cell_gpus', {})
            dispatch.setdefault('cell_gpu_uuids', {})
        running={}
        failed=set(dispatch.get('failed', []))
        supervised = protocol.get('require_exit_receipts', False)
        if supervised:
            from loop_queue_recovery import ObservedWorker, spawn_supervised
            for key, saved in dispatch.get('workers', {}).items():
                running[key] = dict(process=ObservedWorker(saved['launch'], saved['exit_path']),
                    log=None, lock=None, gpu=saved['gpu'], smoke=saved['smoke'],
                    pid=saved['launch']['pid'])
        while True:
            if (suite/'STOP').exists():
                save(suite/'status.json',dict(status='stop_requested',running=list(running)))
                return
            for key, record in list(running.items()):
                code=record['process'].poll()
                if code is not None:
                    for resource in ('log', 'lock'):
                        if record[resource] is not None:
                            record[resource].close()
                    if supervised:
                        dispatch['workers'].pop(key)
                    else:
                        save(suite/'logs'/(key+'-exit.json'),dict(exit_code=code,finished_unix=time.time()))
                    del running[key]
                    target = suite/('validation' if record['smoke'] else 'runs')/key
                    if code != 0 or not (target/'summary.json').exists():
                        failed.add(key)
                    dispatch['failed'] = sorted(failed)
                    save(state_path, dispatch)
            complete={j['id'] for j in jobs if (suite/'runs'/j['id']/'summary.json').exists()}
            if len(complete)==len(jobs) and not running and not failed:
                summarize(suite)
                save(suite/'status.json',dict(status='complete',completed=len(complete),total=len(jobs),
                                            finished_unix=time.time()))
                return
            if failed and not running:
                summarize(suite)
                save(suite/'status.json',dict(status='failed',completed=len(complete),total=len(jobs),
                                            failed=sorted(failed),updated_unix=time.time()))
                return
            prior=[]
            inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,compute_mode',
                                               '--format=csv,noheader'],text=True)
            devices={int(parts[0]):(parts[1],parts[2]) for line in inventory.splitlines()
                     if (parts:=[s.strip() for s in line.split(',')])}
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                                              '--format=csv,noheader'],text=True).splitlines()
            occupied={u.strip() for u in occupied}
            busy={r['gpu'] for r in running.values()}
            pending=[j for j in jobs if j['id'] not in complete and j['id'] not in running and j['id'] not in failed]
            if any(j['stage']=='main' and j['id'] not in complete for j in jobs):
                pending=[j for j in pending if j['stage']=='main']
            missing_smokes=[j for j in jobs if not (suite/'validation'/j['id']/'summary.json').exists()]
            if validation_only and not missing_smokes and not running:
                save(suite/'status.json',dict(status='validation_complete',validated=len(jobs),
                    completed=0,total=len(jobs),updated_unix=time.time()))
                return
            if missing_smokes:
                needed={j['id'] for j in missing_smokes}
                pending=[j for j in pending if j['id'] in needed]
            # Every main method must pass the real shortest/longest-context smoke gate first.
            if not prior and not failed:
                for job in pending:
                    lane=f"{job['model']}/t{job['temperature']}"
                    bound=dispatch['bindings'][lane]
                    if independent:
                        previous_gpu = dispatch['cell_gpus'].get(job['id'])
                        candidates = [previous_gpu] if previous_gpu is not None else allowed(protocol)
                    else:
                        candidates=[bound]
                    for gpu in candidates:
                        if gpu not in allowed(protocol) or gpu in busy or gpu not in devices:
                            continue
                        uuid,mode=devices[gpu]
                        if uuid in occupied or mode != 'Exclusive_Process':
                            continue
                        lock=open(f'/tmp/loopfill-{uuid}.lock','a+')
                        try:
                            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            lock.close()
                            continue
                        current_occupied = subprocess.check_output(
                            ['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],
                            text=True).splitlines()
                        if uuid in {u.strip() for u in current_occupied}:
                            lock.close()
                            continue
                        smoke=suite/'validation'/job['id']
                        smoke_passed=(smoke/'summary.json').exists()
                        target=suite/'runs'/job['id'] if smoke_passed else smoke
                        target.mkdir(parents=True,exist_ok=True)
                        attempt=dispatch['attempts'].get(job['id'],0)+1
                        dispatch['attempts'][job['id']]=attempt
                        if independent:
                            dispatch['cell_gpus'][job['id']] = gpu
                            dispatch['cell_gpu_uuids'][job['id']] = uuid
                        save(state_path,dispatch)
                        prefix=f"{job['id']}-attempt{attempt}"
                        command=[protocol['python'],str(suite/'source/runtime/native_cuda_env.py'),
                                 '--record',str(suite/'logs'/(prefix+'-environment.json')),'--',
                                 protocol['python'],'-u',str(suite/'source/scripts/loop_agentic_worker.py'),
                                 '--suite',str(suite),'--out',str(target),'--model',job['model'],'--arm',job['arm'],
                                 '--temperature',str(job['temperature']),'--q',str(job['q']),
                                 '--n',str(job['n']),'--cache-mode',job['cache_mode']]
                        if not smoke_passed:
                            command.append('--smoke')
                        elif job.get('diagnostic'):
                            command.append('--diagnostic')
                        env=os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                   OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                        cpus={0:'0-3',1:'4-7',2:'8-11',3:'12-15',
                              4:'48-51',5:'52-55',6:'56-59',7:'60-63'}[gpu]
                        log=open(suite/'logs'/(prefix+'.log'),'w')
                        if supervised:
                            receipt = suite/'logs'/(prefix+'-exit.json')
                            process = spawn_supervised(['taskset','--cpu-list',cpus,*command],
                                env=env, log=log, lock=lock,
                                launcher_path=suite/'logs'/(prefix+'-launcher.json'), exit_path=receipt,
                                extra_exit=suite/'logs'/(job['id']+'-exit.json'),
                                metadata=dict(job_id=job['id'], smoke=not smoke_passed, gpu=gpu),
                                cwd=suite/'source')
                            dispatch.setdefault('workers', {})[job['id']] = dict(
                                launch=process.launch, exit_path=str(receipt), gpu=gpu, smoke=not smoke_passed)
                            save(state_path, dispatch)
                        else:
                            process=subprocess.Popen(['taskset','-c',cpus,*command],env=env,
                                                     stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                        running[job['id']]=dict(process=process,log=log,lock=lock,gpu=gpu,
                                              smoke=not smoke_passed,pid=process.pid)
                        busy.add(gpu)
                        break
            summarize(suite)
            save(suite/'status.json',dict(status='waiting_for_prior_suite' if prior else 'running' if running
                else 'failed' if failed else 'waiting_for_gpu',completed=len(complete),total=len(jobs),
                main_completed=sum(j['id'] in complete for j in jobs if j['stage']=='main'),
                prior_pids=prior,failed=sorted(failed),updated_unix=time.time(),
                running={k:{f:v for f,v in r.items() if f in ('gpu','smoke','pid')} for k,r in running.items()},
                bindings={} if independent else dispatch['bindings'],
                placement='first_idle_per_cell' if independent else 'same_device_per_temperature',
                allowed_gpus=allowed(protocol), cell_gpus=dispatch.get('cell_gpus', {})))
            if not running and failed and not pending:
                return
            time.sleep(3 if independent else 15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','validate','summarize'])
    parser.add_argument('--suite',type=Path,required=True)
    args=parser.parse_args()
    {'freeze':freeze,'run':run_queue,'validate':lambda p:run_queue(p,validation_only=True),
     'summarize':summarize}[args.action](args.suite.resolve())
