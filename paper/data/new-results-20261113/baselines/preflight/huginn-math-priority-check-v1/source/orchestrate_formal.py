"""Run the main-paper matrix after the existing per-GPU production queues finish.

This controller never interrupts another process. Failed jobs retain all evidence
and block their dependants; independent jobs may continue. Appendix production is
deliberately a later phase. --execute is required to start GPU work.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from gpu_allocation import allowed_gpus, dispatch_gpus, policy as gpu_policy
import dflash_training_registry as dflash_registry

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
    if '--gpu' in argv:
        assert int(argv[argv.index('--gpu')+1]) in allowed_gpus(), 'GPU released by the user'
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


def verified_completed_dflash_stage(name, argv):
    """Reuse only a full, immutable leaf with an actual successful exit receipt."""
    out=Path(argv[argv.index('--out')+1])
    if not (out/'complete.json').exists():
        assert not out.exists(), f'Archive and investigate incomplete evaluation before retry: {out}'
        return None
    from collect_results import read_run
    from dflash_source_identity import validate_evaluation_sources
    protocol=json.loads((BASE.parents[1]/'results/loopfill-latency512-20261107/protocol.json').read_text())
    run=read_run(out,protocol,profile='--profile' in argv)
    meta=run['meta']
    for flag,key in [('--model','model'),('--method','method'),('--dataset','dataset')]:
        assert meta[key]==argv[argv.index(flag)+1]
    assert meta['temperature']==float(argv[argv.index('--temperature')+1])
    if meta['method']=='dflash':
        checkpoint=Path(argv[argv.index('--checkpoint')+1])
        assert Path(meta['checkpoint'])==checkpoint
        for relative,digest in meta['checkpoint_sha256'].items():
            assert hashlib.sha256((checkpoint/relative).read_bytes()).hexdigest()==digest
    validate_evaluation_sources(out)
    launch,finish=(json.loads((BASE/f'logs/{name}-{suffix}.json').read_text()) for suffix in ['launcher','finished'])
    assert all(finish[k]==v for k,v in launch.items()) and finish['exit_code']==0
    def without_placement(command):
        command=list(command)
        for flag in ['--gpu','--cpus']:
            index=command.index(flag);del command[index:index+2]
        return command
    assert without_placement(finish['argv'])==without_placement(argv)
    return dict(name=name,run=str(out),original_gpu_uuid=meta['gpu_uuid'],
        requests=len(run['rows']),requests_sha256=hashlib.sha256((out/'requests.jsonl').read_bytes()).hexdigest(),
        actual_exit_receipt=finish,reused_unix=time.time(),note='Existing measurement retained; no request rerun.')


class AttachedWorker:
    """Observe an existing worker after coordinator restart; never signal it."""
    def __init__(self, identity, entry, job, folder):
        self.pid=entry['pid'];self.job=job;self.folder=folder;self.identity=identity
        self.path=folder/f'{identity}.json';self.start_ticks=None
        self.completion_basis='worker_exit_receipt'
        proc=Path(f'/proc/{self.pid}')
        if proc.exists():
            stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
            if stat[0]!='Z':
                argv=proc.joinpath('cmdline').read_bytes().decode().split('\0')[:-1]
                assert argv[-3:]==[str(Path(__file__)), '--worker', str(self.path)], 'Refuse to adopt a different process'
                self.start_ticks=stat[19]
                boot=int(next(line.split()[1] for line in Path('/proc/stat').read_text().split('\n') if line.startswith('btime ')))
                started=boot+int(self.start_ticks)/os.sysconf('SC_CLK_TCK')
                assert abs(started-entry['started_unix'])<3., 'Worker PID start time differs from saved status'

    def poll(self):
        proc=Path(f'/proc/{self.pid}')
        if self.start_ticks is not None and proc.exists():
            try:
                stat=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
                if stat[0]!='Z' and stat[19]==self.start_ticks:return None
            except FileNotFoundError:pass
        receipt=self.folder/f'{self.identity}-worker-exit.json'
        if receipt.exists():
            row=json.loads(receipt.read_text())
            assert row['worker_pid']==self.pid
            assert row['job_sha256']==hashlib.sha256(self.path.read_bytes()).hexdigest()
            return row['exit_code']
        # The one already-running training worker predates worker-exit receipts.
        # Its actual torchrun exit and complete exported artifacts prove success.
        if self.job['kind']=='train':
            name=f'dflash-{self.job["model"]}-formal-v1'
            finish=BASE/f'logs/{name}-finished.json'
            source=BASE/'checkpoints'/name;export=BASE/'training'/name
            if finish.exists() and json.loads(finish.read_text())['exit_code']==0 and (export/'complete.json').exists():
                done=json.loads((source/'complete.json').read_text())
                meta=json.loads((source/'metadata.json').read_text())
                assert done==json.loads((export/'complete.json').read_text())
                assert done['step']==meta['total_updates'] and meta['epochs']==6 and not done['smoke']
                for rel,sha in json.loads((source/'source-manifest.json').read_text()).items():
                    assert hashlib.sha256((export/'source'/rel).read_bytes()).hexdigest()==sha
                self.completion_basis='legacy_torchrun_exit0_and_verified_complete_export'
                return 0
        if self.job['kind']=='loopahead':
            name=f'loopahead-{self.job["model"]}-{self.job["dataset"]}-t{self.job["temperature"]}-v1'
            finish=BASE/f'logs/{name}-finished.json';out=BASE/'runs'/name
            if finish.exists() and json.loads(finish.read_text())['exit_code']==0 and (out/'complete.json').exists():
                done=json.loads((out/'complete.json').read_text());meta=json.loads((out/'metadata.json').read_text())
                assert meta['reportable'] and meta['model']==self.job['model'] and meta['dataset']==self.job['dataset']
                assert meta['temperature']==self.job['temperature'] and done['requests']==meta['count']
                assert hashlib.sha256((out/'requests.jsonl').read_bytes()).hexdigest()==done['requests_sha256']
                self.completion_basis='legacy_evaluation_exit0_and_verified_complete_journal'
                return 0
        self.completion_basis='worker_disappeared_without_success_receipt'
        return 1


def worker(job,gpus,uuids):
    assert set(gpus) <= set(allowed_gpus()), 'Worker would occupy a GPU released by the user'
    kind,model=job['kind'],job['model'];gpu=gpus[0]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(uuids[g] for g in gpus),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if kind=='train':
        assert not dflash_registry.is_projected(model), 'Projected Huginn uses its separate gated training queue'
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
            training=dflash_registry.training_root(model)
            if dflash_registry.is_projected(model):dflash_registry.validate_projected_completion()
            checkpoint=json.loads((training/'latest.json').read_text())['checkpoint']
            for method,profile in [('ar',False),('dflash',False),('dflash',True)]:
                name=f'dflash-stack-{model}-{task}-t{temp}-{method}'+('-profile' if profile else '')+'-v1'
                out=BASE/'runs'/name
                argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(dflash_registry.evaluator(model)),
                    '--model',model,'--method',method,'--dataset',task,'--temperature',str(temp),
                    '--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(out)]
                if method=='dflash':argv+=['--checkpoint',checkpoint]
                if profile:argv.append('--profile')
                reused=verified_completed_dflash_stage(name,argv)
                if reused is not None:
                    with (BASE/'logs/formal-controller-v1/reused-dflash-stages.jsonl').open('a') as stream:
                        stream.write(json.dumps(reused)+'\n')
                    continue
                run_command(name,argv,env)


def ready(job):
    if job.get('id') in gpu_policy().get('deferred_jobs', []):
        return False
    kind,model=job['kind'],job['model']
    if kind=='train':
        if dflash_registry.is_projected(model):return False
        return all((BASE/f'data/teachers/dflash-train/{model}/shard{s}/complete.json').exists() for s in range(8))
    if kind=='huginn_check':
        return all((BASE/f'preflight/loopahead-trained-huginn-3.5b-gsm8k-{m}-bfloat16-n20-v1/complete.json').exists() for m in ['ar','loopahead'])
    if kind=='dflash':return dflash_registry.ready(model)
    paths=[BASE/f'checkpoints/loopahead-{model}-formal-v1/complete.json',BASE/'preflight/loopahead-B4-summary.json']
    if model=='huginn-3.5b':paths.append(BASE/'preflight/loopahead-trained-huginn-gate.json')
    if job['temperature'] and model.startswith('ouro'):paths.append(BASE/f'configs/loopahead-{model}-admission-v1.json')
    return all(p.exists() for p in paths)


def gpu_count(job, free_count, policy):
    if job['kind'] != 'train':
        return 1
    if job['model'] == 'huginn-3.5b' and policy.get('huginn_training_gpus'):
        return policy['huginn_training_gpus']
    return 4 if free_count >= 4 else 2


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true')
    p.add_argument('--worker',type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-failed',action='append',default=[])
    p.add_argument('--huginn-training-gpus',type=int,choices=[2,4],
        help='Wait for this many free GPUs before starting Huginn training; preserve global batch32')
    args=p.parse_args()
    if args.resume and not args.execute:p.error('--resume requires --execute')
    if args.retry_failed and not args.resume:p.error('--retry-failed requires --resume')
    uuids={int(i):u.strip() for i,u in (l.split(',') for l in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    if args.worker:
        entry=json.loads(args.worker.read_text());code=0
        receipt=dict(worker_pid=os.getpid(),started_unix=time.time(),
            job_sha256=hashlib.sha256(args.worker.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        try:worker(entry['job'],entry['gpus'],uuids)
        except BaseException:code=1;raise
        finally:save(args.worker.with_name(args.worker.stem+'-worker-exit.json'),dict(receipt,exit_code=code,finished_unix=time.time()))
        return
    jobs=[dict(id=f'train-{m}',kind='train',model=m) for m in ['ouro-2.6b','huginn-3.5b']]
    jobs.append(dict(id='check-huginn-lad',kind='huginn_check',model='huginn-3.5b'))
    jobs += [dict(id=f'{method}-{model}-{task}-t{temp}',kind=method,model=model,dataset=task,temperature=temp)
             for temp in [0,1] for task in TASKS for model in MODELS for method in ['loopahead','dflash']]
    plan=dict(jobs=jobs,release=RELEASE,training_global_batch=32,
        training_gpu_policy='4 free GPUs when available, otherwise2; never preempt',
        phase='main48 only; appendix production follows main results')
    if not args.execute:print(json.dumps(plan,indent=2));return
    folder=BASE/'logs/formal-controller-v1'
    if args.resume:
        assert json.loads((folder/'plan.json').read_text())==json.loads(json.dumps(plan))
    else:
        folder.mkdir(exist_ok=False);save(folder/'plan.json',plan)
    with open('/tmp/loopfill-formal-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        running={};completed=[];failed=[];runtime_policy={}
        if args.resume:
            state=json.loads((folder/'status.json').read_text())
            runtime_policy=dict(state.get('runtime_policy',{}))
            if args.huginn_training_gpus is not None:
                assert 'train-huginn-3.5b' not in state['running'] and 'train-huginn-3.5b' not in state['completed'], 'Cannot change the GPU count of an already-started training'
                runtime_policy['huginn_training_gpus']=args.huginn_training_gpus
            completed=list(state['completed']);failed=list(state['failed'])
            lookup={j['id']:j for j in jobs}
            assert set(args.retry_failed)<=set(failed)
            for identity in args.retry_failed:
                assert not (folder/f'{identity}.log').exists(), 'Archive failed attempt logs before retry'
                assert not (folder/f'{identity}-worker-exit.json').exists(), 'Archive stale worker receipt before retry'
                failed.remove(identity)
            for identity,entry in state['running'].items():
                process=AttachedWorker(identity,entry,lookup[identity],folder)
                running[identity]=dict(process=process,gpus=entry['gpus'],started_unix=entry['started_unix'])
            from provenance import freeze_sources
            snapshot=folder/f'resume-source-{int(time.time())}';snapshot.mkdir()
            freeze_sources(snapshot,[Path(__file__),BASE/'diagnose_trained_loopahead.py',
                BASE/'gpu_allocation.py',BASE/'configs/gpu-allocation.json',BASE/'dflash_training_registry.py',
                BASE/'dflash_source_identity.py',BASE/'gpu_startup.py',BASE/'evaluate_dflash.py',BASE/'collect_results.py'])
            with (folder/'resume-history.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time_unix=time.time(),pid=os.getpid(),retry_failed=args.retry_failed,
                    adopted_workers=state['running'],previous_failed=state['failed'],source_snapshot=str(snapshot),
                    runtime_policy=runtime_policy))+'\n')
        elif args.huginn_training_gpus is not None:
            runtime_policy['huginn_training_gpus']=args.huginn_training_gpus
        while len(completed)+len(failed)<len(jobs):
            external='train-huginn-3.5b'
            if external not in completed+failed and external not in running and dflash_registry.ready('huginn-3.5b'):
                evidence=dflash_registry.validate_projected_completion()
                save(folder/f'{external}-external-narrow-completion.json',dict(evidence,verified_unix=time.time()))
                completed.append(external)
            for identity,entry in list(running.items()):
                code=entry['process'].poll()
                if code is None:continue
                (completed if code==0 else failed).append(identity)
                save(folder/f'{identity}-finished.json',dict(exit_code=code,finished_unix=time.time(),
                    completion_basis=getattr(entry['process'],'completion_basis','direct_child_exit_code')))
                del running[identity]
            busy={u.strip() for u in subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
            reserved={g for entry in running.values() for g in entry['gpus']}
            free=[g for g in dispatch_gpus() if g not in reserved and uuids[g] not in busy and (BASE/RELEASE[g]).exists()]
            if not (folder/'pause-new-jobs').exists():
                for job in jobs:
                    identity=job['id']
                    if identity in running or identity in completed or identity in failed or not ready(job):continue
                    n=gpu_count(job,len(free),runtime_policy)
                    if len(free)<n:
                        if job['kind']=='train':break  # retain the free lane for the next training launch
                        continue
                    gpus=free[:n];free=free[n:]
                    path=folder/f'{identity}.json';save(path,dict(job=job,gpus=gpus,created_unix=time.time()))
                    with (folder/f'{identity}.log').open('x') as out:
                        child=subprocess.Popen(['python',str(Path(__file__)),'--worker',str(path)],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                    running[identity]=dict(process=child,gpus=gpus,started_unix=time.time())
            save(folder/'status.json',dict(updated_unix=time.time(),completed=completed,failed=failed,runtime_policy=runtime_policy,
                gpu_allocation=gpu_policy(),
                running={k:dict(pid=v['process'].pid,gpus=v['gpus'],started_unix=v['started_unix']) for k,v in running.items()},
                pending=[j['id'] for j in jobs if j['id'] not in completed+failed and j['id'] not in running]))
            time.sleep(15)
        save(folder/'complete.json',dict(completed=completed,failed=failed,finished_unix=time.time()))


if __name__=='__main__':main()
