"""Freeze and queue B1 OpenHands experiments using the existing user GPU allocation."""
from __future__ import annotations

import argparse
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

from loop_agentic_worker import digest, save

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
PYTHON = '/tmp/loopfill-dflash-venv/bin/python'
ARMS = ['ar', 'budget-suffix', 'loopahead', 'dflash', 'budget-online']


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
    for name in ['loop_agentic_inputs.py','loop_agentic_worker.py','loop_agentic_suite.py',
                 'loop_batch_bench.py','loop_latency_worker.py']:
        shutil.copy2(REPO/'scripts'/name, source/'scripts'/name)
    shutil.copytree('/tmp/loopfill-assets/ArcticInference/arctic_inference',
                    source/'suffix/arctic_inference', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (source/'runtime').mkdir()
    shutil.copy2(BASE/'native_cuda_env.py', source/'runtime/native_cuda_env.py')
    shutil.copy2(BASE/'configs/loopahead-ouro-2.6b-admission-v1.json', suite/'inputs/loopahead-admission.json')
    lad_root = BASE/'checkpoints/loopahead-ouro-2.6b-formal-v1'
    lad = Path(json.loads((lad_root/'complete.json').read_text())['selected_checkpoint'])/'lad.pt'
    droot = BASE/'checkpoints/dflash-ouro-2.6b-formal-v1'
    dflash = Path(json.loads((droot/'latest.json').read_text())['checkpoint'])
    assert (droot/'complete.json').exists() and (dflash/'complete.json').exists()
    protocol = dict(name=suite.name, created_unix=time.time(), model='ouro-2.6b',
        model_path=manifest['model_path'], max_seq=16384, max_new_tokens=512,
        input_sha256=manifest['input_sha256'], python=PYTHON, temperatures=[0,1],
        main_arms=ARMS, source_sha256={str(p.relative_to(source)):digest(p)
            for p in sorted(source.rglob('*')) if p.is_file()},
        input_files_sha256={p.name:digest(p) for p in sorted((suite/'inputs').glob('*'))},
        loopahead_checkpoint=str(lad), dflash_checkpoint=str(dflash),
        checkpoint_sha256=dict(loopahead=digest(lad), dflash=digest(dflash/'model.safetensors'),
                               dflash_config=digest(dflash/'config.json')),
        selection=manifest['selection'], cache='main: empty global cache at cell start, completed outputs accumulated; '
            'ablation: reset global cache per call; all methods receive the same complete historical prompt',
        scheduling='Respect live GPU allocation, exclusive-process mode and UUID flock. '
                   'Wait for existing LoopAhead throughput suite to finish. No other processes stopped.',
        pending_prior_suite=str(REPO/'results/loopahead-table2-20261112-v2'),
        sampling_seed=0, timing_repeats=1, graph=False, compile=False, pad_decode_rows=False,
        scope='First-stage agent-request decoding experiment; not closed-loop task success. '
              'Batched serving and SQL execution workflow are subsequent stages.',
        limitations=['Single sampling seed and timing repeat.',
            'Selected prompts <=15840 tokens; context exclusions published; no truncation.',
            'DFlash configured training length is 3072; agentic context extrapolation is reported.',
            'BF16 execution shapes can change target logits; T1 samplers are not expected to give paired identical outputs.'])
    save(suite/'protocol.json', protocol)
    jobs = []
    for temperature in [0,1]:
        for arm in ARMS:
            jobs.append(dict(id=f'{arm}-t{temperature}-stream-q4-n16', arm=arm,
                             temperature=temperature, q=4,n=16,cache_mode='stream',stage='main'))
    # Fixed in advance; do not select budgets by test-set speed.
    for arm in ['budget-suffix','budget-online']:
        jobs.append(dict(id=f'{arm}-t1-request-q4-n16',arm=arm,temperature=1,
                         q=4,n=16,cache_mode='request',stage='cache_ablation'))
        for q in [16,64]:
            jobs.append(dict(id=f'{arm}-t1-stream-q{q}-n64',arm=arm,temperature=1,
                             q=q,n=64,cache_mode='stream',stage='budget_ablation'))
        for temperature in [0,1]:
            jobs.append(dict(id=f'{arm}-t{temperature}-diagnostic',arm=arm,temperature=temperature,
                             q=4,n=16,cache_mode='stream',stage='diagnostic',diagnostic=True))
    save(suite/'jobs.json',jobs)
    save(suite/'status.json',dict(status='prepared',main_cells=10,ablation_cells=6,
                                diagnostic_cells=4,completed=0,total=len(jobs)))
    print(json.dumps(dict(status='prepared',jobs=len(jobs),suite=str(suite))))


def allowed():
    policy = json.loads((BASE/'configs/gpu-allocation.json').read_text())
    return [g for g in policy['allowed_gpus'] if g not in policy.get('reserved_for_development', [])]


def prior_active(protocol):
    marker = protocol['pending_prior_suite']
    active = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            argv = (directory/'cmdline').read_bytes().split(b'\0')
            if any(b'orchestrate_loopahead_throughput.py' in arg for arg in argv) and any(marker.encode() in arg for arg in argv):
                active.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return active


def summarize(suite):
    jobs = json.loads((suite/'jobs.json').read_text())
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
    ar = {r['temperature']:r for r in table if r['arm']=='ar'}
    for row in table:
        base = ar.get(row['temperature'])
        if base and row['reportable']:
            row.update(latency_speedup=base['decode_s']/row['decode_s'],
                       decode_rate_speedup=row['decode_token_s']/base['decode_token_s'],
                       output_token_ratio=row['decode_tokens']/base['decode_tokens'])
    fields = sorted({k for row in table for k in row})
    if fields:
        with (suite/'measurements.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table)
    manifest=json.loads((suite/'inputs/manifest.json').read_text())
    lines=['# OpenHands agent-request decoding experiment', '',
        f"Frozen requests: {manifest['count']}; tasks: {manifest['instances']}; phase counts: {manifest['phase_counts']}.",
        '', 'B=1, Ouro-2.6B, BF16, max512, seed0, single timing repeat. Fixed historical prompts; '
        'new target generation. This is not a live-agent success-rate evaluation.', '',
        f"Completed cells: {len(table)}/{len(jobs)} (including diagnostics). Missing cells are pending or failed; consult status.json.",
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
                seconds=sum(r['stats']['decode_s'] for r in group)
                tokens=sum(r['stats']['tokens'] for r in group)
                phase_rows.append(dict(cell=row['id'],phase=phase,count=len(group),
                    decode_s=seconds,decode_tokens=tokens,decode_token_s=tokens/seconds if seconds else None,
                    mean_prompt_tokens=sum(r['prompt_tokens'] for r in group)/len(group)))
    lines.extend(['', 'Decode throughput is generated decode tokens divided by synchronized decode time; '
        'it excludes prefill and is not concurrent serving throughput. Latency speedup is the ratio of total '
        'decode times; generated lengths may differ. Compare output_token_ratio in measurements.csv.', '',
        'DFlash uses the trained local block16 drafter and the packed target adapter. Its configured training '
        'length is 3072; these longer contexts are extrapolation. All methods first run shortest/longest-context '
        'smoke checks. Diagnostic timings are excluded. No confidence interval or task-quality claim is made.'])
    (suite/'report.md').write_text('\n'.join(lines)+'\n')
    if phase_rows:
        with (suite/'phase-measurements.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(phase_rows[0]));writer.writeheader();writer.writerows(phase_rows)
    if retrieval:
        save(suite/'retrieval-diagnostics.json',dict(instrumented=True,formal_timing=False,cells=retrieval,
            definition='Actual queries; root path coverage scored against eventual generated continuation. '
                       'Query nonempty rate is different from accepted-node rate or saved computation.'))
    return table


def run_queue(suite):
    protocol=json.loads((suite/'protocol.json').read_text())
    jobs=json.loads((suite/'jobs.json').read_text())
    (suite/'logs').mkdir(exist_ok=True)
    with (suite/'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Keep each temperature on one device for direct same-device method comparisons.
        state_path=suite/'dispatch.json'
        dispatch=json.loads(state_path.read_text()) if state_path.exists() else dict(bindings={},attempts={})
        running={}
        failed=set()
        while True:
            if (suite/'STOP').exists():
                save(suite/'status.json',dict(status='stop_requested',running=list(running)))
                return
            for key, record in list(running.items()):
                code=record['process'].poll()
                if code is not None:
                    record['log'].close();record['lock'].close()
                    save(suite/'logs'/(key+'-exit.json'),dict(exit_code=code,finished_unix=time.time()))
                    del running[key]
                    if code:
                        failed.add(key)
            complete={j['id'] for j in jobs if (suite/'runs'/j['id']/'summary.json').exists()}
            if len(complete)==len(jobs):
                summarize(suite)
                save(suite/'status.json',dict(status='complete',completed=len(complete),total=len(jobs),
                                            finished_unix=time.time()))
                return
            if failed and not running:
                summarize(suite)
                save(suite/'status.json',dict(status='failed',completed=len(complete),total=len(jobs),
                                            failed=sorted(failed),updated_unix=time.time()))
                return
            prior=prior_active(protocol)
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
            # Every main method must pass the real shortest/longest-context smoke gate first.
            if not prior and not failed:
                for job in pending:
                    temperature=str(job['temperature'])
                    bound=dispatch['bindings'].get(temperature)
                    candidates=[bound] if bound is not None else allowed()
                    for gpu in candidates:
                        if gpu not in allowed() or gpu in busy:
                            continue
                        uuid,mode=devices[gpu]
                        if uuid in occupied or mode != 'Exclusive_Process':
                            continue
                        lock=open(f'/tmp/loopfill-{uuid}.lock','a+')
                        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        except BlockingIOError:
                            lock.close();continue
                        dispatch['bindings'][temperature]=gpu
                        smoke=suite/'validation'/f"{job['arm']}-t{job['temperature']}"
                        smoke_passed=(smoke/'summary.json').exists()
                        target=suite/'runs'/job['id'] if smoke_passed else smoke
                        target.mkdir(parents=True,exist_ok=True)
                        attempt=dispatch['attempts'].get(job['id'],0)+1
                        dispatch['attempts'][job['id']]=attempt
                        save(state_path,dispatch)
                        prefix=f"{job['id']}-attempt{attempt}"
                        command=[protocol['python'],str(suite/'source/runtime/native_cuda_env.py'),
                                 '--record',str(suite/'logs'/(prefix+'-environment.json')),'--',
                                 protocol['python'],'-u',str(suite/'source/scripts/loop_agentic_worker.py'),
                                 '--suite',str(suite),'--out',str(target),'--arm',job['arm'],
                                 '--temperature',str(job['temperature']),'--q',str(job['q']),
                                 '--n',str(job['n']),'--cache-mode',job['cache_mode']]
                        if not smoke_passed:command.append('--smoke')
                        elif job.get('diagnostic'):command.append('--diagnostic')
                        env=os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                                   OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                        cpus={2:'8-11',3:'12-15',4:'48-51',6:'56-59'}[gpu]
                        log=open(suite/'logs'/(prefix+'.log'),'w')
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
                bindings=dispatch['bindings']))
            if not running and failed and not pending:
                return
            time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','run','summarize'])
    parser.add_argument('--suite',type=Path,required=True)
    args=parser.parse_args()
    {'freeze':freeze,'run':run_queue,'summarize':summarize}[args.action](args.suite.resolve())
