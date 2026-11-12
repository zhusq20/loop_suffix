"""Run the main-paper matrix after the existing per-GPU production queues finish.

This controller never interrupts another process. Failed jobs retain all evidence
and block their dependants; independent jobs may continue. Appendix production is
deliberately a later phase. --execute is required to start GPU work.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE=Path(__file__).resolve().parent
CPUS=['0-3','4-7','8-11','12-15','48-51','52-55','56-59','60-63']
MODELS=['huginn-3.5b','ouro-2.6b']
TASKS=['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench']
RELEASE={0:'data/teachers/dflash-train/huginn-3.5b/shard6/complete.json',
    1:'data/teachers/dflash-train/huginn-3.5b/shard1/complete.json',
    2:'preflight/loopahead-trained-huginn-3.5b-gsm8k-ar-bfloat16-n20-v1/complete.json',
    3:'data/teachers/dflash-train/huginn-3.5b/shard3/complete.json',
    4:'data/teachers/dflash-train/huginn-3.5b/shard5/complete.json',
    5:'data/teachers/dflash-train/huginn-3.5b/shard2/complete.json',
    6:'data/teachers/dflash-train/huginn-3.5b/shard4/complete.json',
    7:'data/teachers/dflash-train/huginn-3.5b/shard7/complete.json'}


def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def run_command(name,argv,env=None):
    log=BASE/f'logs/{name}.log'
    with log.open('x') as f:
        child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env)
    event=dict(name=name,argv=argv,pid=child.pid,started_unix=time.time())
    save(BASE/f'logs/{name}-launcher.json',event)
    code=child.wait()
    save(BASE/f'logs/{name}-finished.json',dict(event,exit_code=code,finished_unix=time.time()))
    if code:raise RuntimeError(f'{name} exited {code}; see {log}')


def selected(model):
    p=BASE/f'checkpoints/loopahead-{model}-formal-v1/complete.json'
    return str(Path(json.loads(p.read_text())['selected_checkpoint'])/'lad.pt')


def worker(job,gpus,uuids):
    kind,model=job['kind'],job['model'];gpu=gpus[0]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids[g] for g in gpus),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if kind=='train':
        name=f'dflash-{model}-formal-v1';out=BASE/'checkpoints'/name
        micro=4
        if model=='huginn-3.5b':
            # Measure the official microbatch4 on the longest actual production
            # examples before reducing it. Keep headroom for DDP buckets and
            # allocator variation; accumulation always preserves global batch32.
            trials=[]
            for candidate in [4,2,1]:
                smoke_name=f'dflash-huginn-micro{candidate}-memory-smoke-v1'
                smoke=BASE/'checkpoints'/smoke_name
                smoke_argv=['/tmp/loopfill-specforge-venv/bin/python','-u',str(BASE/'train_dflash.py'),
                    '--model',model,'--teachers',str(BASE/f'data/teachers/dflash-train/{model}'),
                    '--out',str(smoke),'--cpu-groups',CPUS[gpu],'--epochs','1','--micro-batch',str(candidate),
                    '--accumulation','1','--checkpoint-layers','--smoke']
                try:
                    run_command(smoke_name,smoke_argv,env)
                except RuntimeError:
                    failure=smoke/'failure-rank0.json'
                    if not failure.exists():raise
                    error=json.loads(failure.read_text())['error']
                    if 'out of memory' not in error.lower() and 'OutOfMemoryError' not in error:raise
                    trials.append(dict(micro_batch=candidate,status='out_of_memory',failure=str(failure)))
                    continue
                metrics=json.loads((smoke/'progress.json').read_text())
                peak=metrics['peak_allocated_bytes']
                trials.append(dict(micro_batch=candidate,status='complete',peak_allocated_bytes=peak,
                    smoke=str(smoke),update_wall_s=metrics['update_wall_s']))
                if peak<=120*1024**3:
                    micro=candidate;break
            else:raise RuntimeError('No Huginn microbatch fit the measured memory budget')
            save(BASE/'training/huginn-dflash-microbatch-selection.json',dict(trials=trials,
                selected_micro_batch=micro,peak_budget_bytes=120*1024**3,world_size=len(gpus),
                global_batch=32,accumulation=32//(len(gpus)*micro),selected_unix=time.time(),
                note='One-update memory probes use the longest32 eligible examples from the completed50k teacher set; formal training starts fresh.'))
        argv=['/tmp/loopfill-specforge-venv/bin/torchrun','--standalone',f'--nproc_per_node={len(gpus)}',
            str(BASE/'train_dflash.py'),'--model',model,'--teachers',str(BASE/f'data/teachers/dflash-train/{model}'),
            '--out',str(out),'--cpu-groups',','.join(CPUS[g] for g in gpus),'--epochs','6',
            '--micro-batch',str(micro),'--accumulation',str(32//(len(gpus)*micro)),'--save-every','500']
        if model=='huginn-3.5b':argv.append('--checkpoint-layers')
        run_command(name,argv,env)
        export=BASE/'training'/name;export.mkdir(exist_ok=True)
        for p in out.glob('*.json*'):shutil.copy2(p,export/p.name)
        shutil.copytree(out/'source',export/'source',dirs_exist_ok=True)
    elif kind in ['huginn_check','trained_check']:
        root=BASE/'preflight';records={}
        for method in ['ar','loopahead']:
            p=root/f'loopahead-trained-{model}-gsm8k-{method}-bfloat16-n20-v1/requests.jsonl'
            records[method]=[json.loads(l) for l in p.open()]
        assert len(records['ar'])==len(records['loopahead'])==20
        exact=sum(a['tokens']==b['tokens'] for a,b in zip(records['ar'],records['loopahead']))
        report=dict(model=model,bfloat16_exact_sequences=exact,requests=20)
        if exact<20:
            fp={}
            for method in ['ar','loopahead']:
                name=f'loopahead-trained-{model}-gsm8k-{method}-float32-n20-v1'
                out=root/name
                argv=['/tmp/loopfill-venv/bin/python','-u',str(BASE/'evaluate_loopahead.py'),
                    '--model',model,'--method',method,'--dataset','gsm8k','--count','20',
                    '--pilot','--trace','--dtype','float32','--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(out)]
                if method=='loopahead':argv+=['--checkpoint',selected(model)]
                run_command(name,argv,env)
                fp[method]=[json.loads(l) for l in (out/'requests.jsonl').open()]
            report['float32_exact_sequences']=sum(a['tokens']==b['tokens'] for a,b in zip(fp['ar'],fp['loopahead']))
            assert report['float32_exact_sequences']==20,'Trained check requires investigation before formal evaluation'
            name=f'loopahead-trained-{model}-common-prefix-fp32-v1'
            diagnosis=root/name
            run_command(name,['/tmp/loopfill-venv/bin/python','-u',str(BASE/'diagnose_trained_loopahead.py'),
                '--ar',str(root/f'loopahead-trained-{model}-gsm8k-ar-bfloat16-n20-v1'),
                '--loopahead',str(root/f'loopahead-trained-{model}-gsm8k-loopahead-bfloat16-n20-v1'),
                '--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(diagnosis)],env)
            diagnostic=json.loads((diagnosis/'report.json').read_text())
            assert diagnostic['status']=='complete' and diagnostic['all_fp32_same_top1']
            assert diagnostic['cases']==20-exact
            report['common_prefix_diagnosis']=str(diagnosis/'report.json')
            report['common_prefix_diagnosis_sha256']=hashlib.sha256((diagnosis/'report.json').read_bytes()).hexdigest()
            report['limitation']='FP32 whole-sequence and observed BF16 common-prefix checks passed; finite-precision discrepancies remain explicitly reported, without claiming every formal benchmark sequence matches.'
        save(root/('loopahead-trained-huginn-gate.json' if model=='huginn-3.5b' else f'loopahead-trained-{model}-gate.json'),report)
    else:
        task,temp=job['dataset'],job['temperature']
        if kind=='loopahead':
            name=f'loopahead-{model}-{task}-t{temp}-v1';out=BASE/'runs'/name
            argv=['/tmp/loopfill-venv/bin/python','-u',str(BASE/'evaluate_loopahead.py'),
                '--model',model,'--method','loopahead','--dataset',task,'--temperature',str(temp),
                '--checkpoint',selected(model),'--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(out)]
            if temp and model.startswith('ouro'):
                argv+=['--admission',str(BASE/f'configs/loopahead-{model}-admission-v1.json')]
            run_command(name,argv,env)
        else:
            training=BASE/f'checkpoints/dflash-{model}-formal-v1'
            checkpoint=json.loads((training/'latest.json').read_text())['checkpoint']
            for method,profile in [('ar',False),('dflash',False),('dflash',True)]:
                name=f'dflash-stack-{model}-{task}-t{temp}-{method}'+('-profile' if profile else '')+'-v1'
                out=BASE/'runs'/name
                argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(BASE/'evaluate_dflash.py'),
                    '--model',model,'--method',method,'--dataset',task,'--temperature',str(temp),
                    '--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(out)]
                if method=='dflash':argv+=['--checkpoint',checkpoint]
                if profile:argv.append('--profile')
                run_command(name,argv,env)


def ready(job):
    kind,model=job['kind'],job['model']
    if kind=='train':
        return all((BASE/f'data/teachers/dflash-train/{model}/shard{s}/complete.json').exists() for s in range(8))
    if kind=='huginn_check':
        return all((BASE/f'preflight/loopahead-trained-huginn-3.5b-gsm8k-{m}-bfloat16-n20-v1/complete.json').exists() for m in ['ar','loopahead'])
    if kind=='dflash':return (BASE/f'checkpoints/dflash-{model}-formal-v1/complete.json').exists()
    paths=[BASE/f'checkpoints/loopahead-{model}-formal-v1/complete.json',BASE/'preflight/loopahead-B4-summary.json']
    if model=='huginn-3.5b':paths.append(BASE/'preflight/loopahead-trained-huginn-gate.json')
    if job['temperature'] and model.startswith('ouro'):paths.append(BASE/f'configs/loopahead-{model}-admission-v1.json')
    return all(p.exists() for p in paths)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',type=Path);args=p.parse_args()
    uuids={int(i):u.strip() for i,u in (l.split(',') for l in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    if args.worker:
        entry=json.loads(args.worker.read_text());worker(entry['job'],entry['gpus'],uuids);return
    jobs=[dict(id=f'train-{m}',kind='train',model=m) for m in ['ouro-2.6b','huginn-3.5b']]
    jobs.append(dict(id='check-huginn-lad',kind='huginn_check',model='huginn-3.5b'))
    jobs += [dict(id=f'{method}-{model}-{task}-t{temp}',kind=method,model=model,dataset=task,temperature=temp)
             for temp in [0,1] for task in TASKS for model in MODELS for method in ['loopahead','dflash']]
    plan=dict(jobs=jobs,release=RELEASE,training_global_batch=32,
        training_gpu_policy='4 free GPUs when available, otherwise2; never preempt',
        phase='main48 only; appendix production follows main results')
    if not args.execute:print(json.dumps(plan,indent=2));return
    folder=BASE/'logs/formal-controller-v1';folder.mkdir(exist_ok=False)
    save(folder/'plan.json',plan)
    with open('/tmp/loopfill-formal-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        running={};completed=[];failed=[]
        while len(completed)+len(failed)<len(jobs):
            for identity,entry in list(running.items()):
                code=entry['process'].poll()
                if code is None:continue
                (completed if code==0 else failed).append(identity)
                save(folder/f'{identity}-finished.json',dict(exit_code=code,finished_unix=time.time()))
                del running[identity]
            busy={u.strip() for u in subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
            reserved={g for entry in running.values() for g in entry['gpus']}
            free=[g for g in range(8) if g not in reserved and uuids[g] not in busy and (BASE/RELEASE[g]).exists()]
            if not (folder/'pause-new-jobs').exists():
                for job in jobs:
                    identity=job['id']
                    if identity in running or identity in completed or identity in failed or not ready(job):continue
                    n=(4 if len(free)>=4 else 2) if job['kind']=='train' else 1
                    if len(free)<n:
                        if job['kind']=='train':break  # retain the free lane for the next training launch
                        continue
                    gpus=free[:n];free=free[n:]
                    path=folder/f'{identity}.json';save(path,dict(job=job,gpus=gpus,created_unix=time.time()))
                    with (folder/f'{identity}.log').open('x') as out:
                        child=subprocess.Popen(['python',str(Path(__file__)),'--worker',str(path)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                    running[identity]=dict(process=child,gpus=gpus,started_unix=time.time())
            save(folder/'status.json',dict(updated_unix=time.time(),completed=completed,failed=failed,
                running={k:dict(pid=v['process'].pid,gpus=v['gpus'],started_unix=v['started_unix']) for k,v in running.items()},
                pending=[j['id'] for j in jobs if j['id'] not in completed+failed and j['id'] not in running]))
            time.sleep(15)
        save(folder/'complete.json',dict(completed=completed,failed=failed,finished_unix=time.time()))


if __name__=='__main__':main()
