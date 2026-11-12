"""Continue the existing Ouro1.4B small drafter through aligned evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from native_cuda_env import native_cuda_environment
from packed_dflash_evidence import (check_profile, check_run, read, save, sha, timed_evidence)
from orchestrate_packed_dflash_supplements import aggregate, cards, cpus, occupied


def verify(path):
    plan = read(path)
    for relative, expected in plan['frozen_sha256'].items():
        assert sha(path.parent / relative) == expected, relative
    training = Path(plan['training_root'])
    assert sha(training/'plan.json') == plan['training_plan_sha256']
    return plan


def bind_training(root, plan):
    training = Path(plan['training_root'])
    complete = read(training/'complete.json')
    report = read(training/'checks/training-audit.json')
    assert complete['status'] == 'complete' and report['status'] == 'passed'
    assert report['parameters'] == complete['parameters'] == 163597568
    for stage in ['teachers','smoke','train','audit']:
        launch, receipt = read(training/f'{stage}-launcher.json'), read(training/f'{stage}-exit.json')
        assert receipt['exit_code'] == 0
        assert all(receipt[k] == launch[k] for k in ['pid','argv','started_unix'])
    checkpoint = Path(report['checkpoint'])
    assert str(checkpoint) == complete['checkpoint']
    assert sha(checkpoint/'model.safetensors') == report['weights_sha256']
    assert sha(training/'training/metrics.jsonl') == report['metrics_sha256']
    assert sha(training/'training/metadata.json') == report['metadata_sha256']
    meta, done = read(training/'training/metadata.json'), read(training/'training/complete.json')
    assert meta['epochs'] == 6 and meta['draft_parameters'] == 163597568
    assert done['step'] == meta['total_updates'] == report['updates']
    assert not meta['bounded_experiment'] and not done['bounded_experiment']
    config = read(checkpoint/'config.json')
    expected_config = read(plan['draft_config'])
    for key,value in expected_config.items():
        if key not in ['architectures','transformers_version']:
            assert config[key] == value, key
    assert config['architectures'] == ['ProjectedDFlashDraftModel']
    assert config.get('dtype',config.get('torch_dtype')) == 'bfloat16'
    suite = root/'suite'
    protocol = dict(created_unix=time.time(),name=root.name,python=plan['python'],
        models={'ouro-1.4b':dict(path=plan['target_path'],family='ouro',revision=plan['revision'],
            checkpoint=str(checkpoint),checkpoint_sha256={n:sha(checkpoint/n) for n in ['model.safetensors','config.json','complete.json']},
            training_complete_sha256=sha(training/'training/complete.json'),stop_ids=plan['stop_ids'])},
        source_sha256={str(p.relative_to(suite/'source')):sha(p) for p in sorted((suite/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
        dtype='bfloat16',graph=False,compile=False,target_row_padding=False,
        table1=dict(temperatures=[0,1],max_new_tokens=512,batch=1,warmup_requests=2,warmup_tokens=32),
        training_audit=dict(path=str(training/'checks/training-audit.json'),sha256=sha(training/'checks/training-audit.json')),
        limitations=['Our projected Ouro1.4B adaptation: five layers, H1536, FF4096, 12 heads, block16.',
                     'Single timing repeat and seed; shared host; report observed BF16 divergences separately.'])
    save(suite/'protocol.json',protocol)
    save(suite/'jobs.json',[plan['validation_job'],*plan['jobs']])
    supplement=root/'supplement'
    supplement.mkdir(exist_ok=False)
    for name in ['source','logs','bindings','runs']:(supplement/name).mkdir()
    for p in (root/'source').glob('*.py'):shutil.copy2(p,supplement/'source'/p.name)
    save(supplement/'plan.json',dict(suite=str(suite),protocol_sha256=sha(suite/'protocol.json'),
        jobs_sha256=sha(suite/'jobs.json'),jobs=plan['jobs'],
        source_sha256={p.name:sha(p) for p in sorted((supplement/'source').glob('*.py'))},
        limitations=protocol['limitations']))
    save(root/'training-binding.json',dict(checkpoint=str(checkpoint),training_report=report,
        training_root_complete_sha256=sha(training/'complete.json'),protocol_sha256=sha(suite/'protocol.json'),
        supplementary_plan_sha256=sha(supplement/'plan.json'),bound_unix=time.time()))


def main(path):
    plan=verify(path);root=path.parent;suite=root/'suite';supplement=root/'supplement'
    sys.path.insert(0,str(root/'source'))
    from loop_queue_recovery import UNKNOWN_EXIT, ObservedWorker, spawn_supervised
    with (root/'controller.lock').open('a+') as control:
        fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (Path(plan['training_root'])/'complete.json').exists():
            if (Path(plan['training_root'])/'failure.json').exists():
                raise RuntimeError('Existing training failed; preserve evidence before any evaluation')
            save(root/'status.json',dict(stage='waiting_for_full_training',controller_pid=os.getpid(),updated_unix=time.time()))
            time.sleep(15)
        verify(path)
        if not (root/'training-binding.json').exists():bind_training(root,plan)
        binding=read(root/'training-binding.json')
        assert binding['protocol_sha256']==sha(suite/'protocol.json')
        assert binding['supplementary_plan_sha256']==sha(supplement/'plan.json')
        sys.path.insert(0,str(suite/'source/scripts'))
        from loop_dflash_benchmark_suite import audit
        env,driver=native_cuda_environment()
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
        env.pop('WANDB_API_KEY',None)
        save(root/'native-driver.json',driver)
        entries={plan['validation_job']['id']:(plan['validation_job'],'validate')}
        for stage in ['timed','ar','profile']:
            for job in plan['jobs']:
                if stage=='diagnose' and job['temperature']!=0:continue
                entries[job['id']+'-'+stage]=(job,stage)
        state=read(root/'dispatch.json') if (root/'dispatch.json').exists() else {}
        active={}
        def folder(job,stage):
            if stage in ['validate','timed']:return suite/'runs'/job['id']/'attempt1'
            if stage=='diagnose':return root/'diagnostics'/job['id']
            return supplement/'runs'/(job['id']+'-'+stage)
        for key in entries:
            launch=root/f'logs/{key}-launcher.json'
            if launch.exists() and state.get(key,{}).get('outcome') not in ['complete','failed','exit_unavailable']:
                record=read(launch);job,stage=entries[key];out=folder(job,stage)
                state[key]=dict(record,outcome='running',out=str(out.relative_to(supplement)) if stage in ['ar','profile'] else str(out))
                active[key]=dict(process=ObservedWorker(record,root/f'logs/{key}-exit.json'),lock=None,log=None)
        while True:
            for key,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                for name in ['lock','log']:
                    if item[name] is not None:item[name].close()
                job,stage=entries[key];out=folder(job,stage)
                state[key].update(outcome='failed' if code!=UNKNOWN_EXIT else 'exit_unavailable',
                    exit_code=None if code==UNKNOWN_EXIT else code,finished_unix=time.time())
                if code==0:
                    try:
                        if stage in ['validate','timed']:
                            summary=audit(suite,job,out)
                            if stage=='validate':assert summary['status']=='passed_full_weight_fp32'
                            else:check_run(out,suite,job)
                        elif stage in ['ar','profile']:
                            current=check_run(out,suite,job,stage=stage)
                            assert current['metadata']['binding_sha256']==sha(supplement/f'bindings/{job["id"]}-{stage}.json')
                            if stage=='profile':check_profile(current['rows'],check_run(folder(job,'timed'),suite,job)['rows'])
                        else:
                            done=read(out/'complete.json')
                            assert sha(out/'diagnosis.json')==done['diagnosis_sha256']
                            assert done['all_bf16_divergences_reproduced'] and done['all_fp32_argmax_agree']
                        state[key]['outcome']='complete'
                    except Exception:
                        save(out/'audit-failure.json',dict(error=traceback.format_exc()))
                if stage in ['validate','timed']:
                    state[key].update(out=str(out.relative_to(suite)),attempt=1)
                    # Copy only the genuine parent receipt, without changing its exit status.
                    save(suite/f'logs/{job["id"]}-attempt1-exit.json',read(root/f'logs/{key}-exit.json') if code!=UNKNOWN_EXIT else state[key])
                del active[key]
                save(root/'dispatch.json',state)
                aggregate(supplement,suite,plan['jobs'],state)
            pending=[]
            for key,(job,stage) in entries.items():
                if key in state:continue
                required=[]
                if stage=='timed':required=[plan['validation_job']['id']]
                elif stage in ['ar','profile']:required=[job['id']+'-timed']
                elif stage=='diagnose':required=[job['id']+'-timed',job['id']+'-ar',job['id']+'-profile']
                if all(state.get(r,{}).get('outcome')=='complete' for r in required):pending.append(key)
            # Give the original main-model latency and throughput work priority.
            main_status=read(Path(plan['balanced_queue'])/'status.json')
            if not main_status['pending'] and not (root/'STOP').exists():
                policy=read(plan['gpu_policy']);allowed=set(policy['allowed_gpus'])-set(policy.get('reserved_for_development',[]));busy=occupied()
                for gpu,(uuid,mode) in sorted(cards().items()):
                    if gpu not in allowed or uuid in busy or mode!='Exclusive_Process' or not pending:continue
                    lease=open(f'/tmp/loopfill-{uuid}.lock','a+')
                    try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:lease.close();continue
                    if uuid in occupied():lease.close();continue
                    verify(path)
                    key=pending.pop(0);job,stage=entries[key];out=folder(job,stage)
                    assert not out.exists(),f'Preserve partial run before retry: {out}'
                    if stage in ['validate','timed']:
                        argv=[plan['python'],'-u',str(suite/'source/scripts/loop_dflash_benchmark_worker.py'),'--suite',str(suite),'--job',job['id'],'--out',str(out),'--lock-fd',str(lease.fileno())]
                    elif stage in ['ar','profile']:
                        timed=state[job['id']+'-timed'];evidence=timed_evidence(suite,{job['id']:timed},job)
                        assert evidence is not None
                        proof=supplement/f'bindings/{job["id"]}-{stage}.json'
                        assert not proof.exists()
                        save(proof,dict(job=job,stage=stage,timed=evidence,plan_sha256=sha(supplement/'plan.json'),created_unix=time.time()))
                        argv=[plan['python'],'-u',str(supplement/'source/supplement_packed_dflash.py'),'--plan',str(supplement/'plan.json'),'--binding',str(proof),'--job',job['id'],'--stage',stage,'--out',str(out),'--lock-fd',str(lease.fileno())]
                    else:
                        argv=[plan['python'],'-u',str(root/'source/diagnose_packed_dflash.py'),'--plan',str(supplement/'plan.json'),'--job',job['id'],'--ar',str(folder(job,'ar')),'--out',str(out),'--lock-fd',str(lease.fileno())]
                    argv=['taskset','--cpu-list',cpus(gpu),*argv]
                    child_env=dict(env,CUDA_VISIBLE_DEVICES=uuid,PYTHONPATH=os.pathsep.join(map(str,[root/'source',suite/'source',suite/'source/src',suite/'source/scripts'])))
                    log=(root/f'logs/{key}.log').open('x')
                    child=spawn_supervised(argv,env=child_env,log=log,lock=lease,
                        launcher_path=root/f'logs/{key}-launcher.json',exit_path=root/f'logs/{key}-exit.json',
                        metadata=dict(stage=stage,job_id=job['id'],gpu=gpu,plan_sha256=sha(path)),cwd=suite/'source')
                    state[key]=dict(child.launch,outcome='running',out=str(out.relative_to(supplement)) if stage in ['ar','profile'] else str(out))
                    active[key]=dict(process=child,lock=lease,log=log)
                    busy.add(uuid);save(root/'dispatch.json',state)
            save(root/'dispatch.json',state)
            failures={k:v['outcome'] for k,v in state.items() if v['outcome'] not in ['running','complete']}
            save(root/'status.json',dict(stage='running' if active else 'waiting_for_gpu',controller_pid=os.getpid(),updated_unix=time.time(),
                expected_steps=len(entries),complete_steps=sum(v['outcome']=='complete' for v in state.values()),
                running={k:state[k] for k in active},ready=pending,failures=failures))
            if not active and (len(state)==len(entries) or (failures and not pending)):
                save(root/'complete.json',dict(status='complete' if not failures else 'needs_inspection',finished_unix=time.time(),
                    completed_cells=aggregate(supplement,suite,plan['jobs'],state),failures=failures,
                    limitation='DFlash appendix measurements; output-divergence diagnostics cancelled by user on 2026-11-12. Final paper merge and training-cost integration remain separate.'))
                return
            time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    try:main(args.plan.resolve())
    except Exception:
        save(args.plan.parent/'failure.json',dict(error=traceback.format_exc(),failed_unix=time.time()))
        raise
