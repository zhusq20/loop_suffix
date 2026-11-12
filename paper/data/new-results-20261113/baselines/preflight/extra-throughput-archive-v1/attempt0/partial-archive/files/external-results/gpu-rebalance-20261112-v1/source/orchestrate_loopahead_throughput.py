"""Validate and run the ten requested LoopAhead Table 2 throughput cells."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
TABLE2 = BASE.parents[1]/'results/loopfill-table2-20261107-v2'
PYTHON = '/tmp/loopfill-venv/bin/python'
MODELS = ['huginn-3.5b','ouro-2.6b']
CPUS = {2:'8-11',3:'12-15',4:'48-51',6:'56-59'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def save(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir()
    files=[BASE/n for n in ['evaluate_loopahead_throughput.py','orchestrate_loopahead_throughput.py',
                          'gpu_startup.py','gpu_allocation.py','configs/gpu-allocation.json','native_cuda_env.py']]
    files += sorted((BASE/'loopahead').glob('*.py'))
    manifest={}
    for p in files:
        rel=p.relative_to(BASE);target=source/rel;target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(p,target);manifest[str(rel)]=sha(target)
    save(source/'source-manifest.json',manifest)
    old=json.loads((TABLE2/'protocol.json').read_text())
    selected={}
    for model in MODELS:
        root=BASE/f'checkpoints/loopahead-{model}-formal-v1'
        path=Path(json.loads((root/'complete.json').read_text())['selected_checkpoint'])/'lad.pt'
        selected[model]=dict(path=str(path),sha256=sha(path),training_complete_sha256=sha(root/'complete.json'),
            input_sha256=old['models'][model]['inputs_sha256'],model_revision=old['models'][model]['checkpoint']['model']['revision'])
    jobs=[dict(id=f'{m}-b{b}',model=m,batch=b,mode='formal') for b in [1,2,4,8,16] for m in MODELS]
    preflight=[dict(id=f'{m}-fp32-b2',model=m,batch=2,mode='validate') for m in MODELS]
    preflight += [dict(id=f'{m}-bf16-b16',model=m,batch=16,mode='smoke') for m in MODELS]
    save(out/'protocol.json',dict(created_unix=time.time(),models=selected,batches=[1,2,4,8,16],
        planned_cells=10,requests_per_cell=480,max_new_tokens=1024,temperature=0,dtype='bfloat16',
        prefill_chunk_size=128,schedule='continuous',graph=False,compile=False,pad_decode_rows=False,
        max_seq=2688,seed=0,repeats=1,warmup_requests='min(2B,480)',warmup_max_new_tokens=32,
        table2_protocol=str(TABLE2/'protocol.json'),table2_protocol_sha256=sha(TABLE2/'protocol.json'),
        source_sha256=manifest,jobs=jobs,preflight=preflight,allowed_gpus=[2,3,4,6],
        timing='Output token count / finite-stream E2E wall time including prefill, scheduling and fsynced request journal.',
        dflash='Stopped at user request due to outdated code; this queue cannot launch DFlash.',
        historical_references='Original Table 2 AR and LoopFill; cross-date, no drift correction.',
        limitations=['No throughput data for Ouro-1.4B: a trained LoopAhead checkpoint is not available.',
                    'Single measurement per cell; shared-host conditions and BF16 output differences remain.']))
    for folder in ['logs','runs','preflight']:(out/folder).mkdir()
    return json.loads((out/'protocol.json').read_text())


def collect(out,protocol):
    historical=list(csv.DictReader((TABLE2/'measurements.csv').open()))
    results=[]
    for job in protocol['jobs']:
        root=out/'runs'/job['id']
        finish=out/'logs'/f"{job['id']}-finished.json"
        if not finish.exists():continue
        receipt=json.loads(finish.read_text())
        if (root/'failure.json').exists():
            failure=json.loads((root/'failure.json').read_text())
            results.append(dict(model=job['model'],batch=job['batch'],outcome=failure['status']))
            continue
        if receipt['exit_code'] != 0 or not (root/'complete.json').exists():continue
        done=json.loads((root/'complete.json').read_text())
        assert done['requests']==480 and sha(root/'requests.jsonl')==done['requests_sha256']
        summary=json.loads((root/'summary.json').read_text())
        row={k:summary[k] for k in ['model','batch','output_tput','total_tput','wall_s','output_tokens',
            'input_tokens','requests','gamma','peak_allocated_bytes','max_advance_requests','max_lad_batch']}
        row['outcome']='complete'
        for arm,label in [('ar','ar'),('budget-online','loopfill')]:
            ref=next(r for r in historical if r['model']==job['model'] and int(r['batch'])==job['batch'] and r['arm']==arm)
            row[f'historical_{label}_tput']=float(ref['output_tput'])
            row[f'ratio_to_historical_{label}']=summary['output_tput']/float(ref['output_tput'])
        results.append(row)
    if results:
        keys=list(dict.fromkeys(k for row in results for k in row))
        with (out/'measurements.csv.tmp').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(results)
        (out/'measurements.csv.tmp').replace(out/'measurements.csv')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();out=args.out.resolve()
    protocol=json.loads((out/'protocol.json').read_text()) if args.resume else prepare(out)
    if not args.execute:
        print(json.dumps(dict(out=str(out),planned_cells=10)));return
    from native_cuda_env import native_cuda_environment
    env,evidence=native_cuda_environment()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TOKENIZERS_PARALLELISM='false')
    save(out/'native-driver.json',evidence)
    env['PYTHONPATH']=str(out/'source')
    uuids={int(i):u.strip() for i,u in (l.split(',') for l in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    with (out/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active={};completed=[];failed=[]
        # Resume only completed jobs; interrupted finite-stream cells require a
        # separate attempt directory and are never spliced into one timing window.
        for job in protocol['preflight']+protocol['jobs']:
            finish=out/'logs'/f"{job['id']}-finished.json"
            if finish.exists():
                receipt=json.loads(finish.read_text())
                (completed if receipt['exit_code']==0 else failed).append(job['id'])
        while True:
            for identity,entry in list(active.items()):
                code=entry['process'].poll()
                if code is None:continue
                entry['log'].close()
                save(out/'logs'/f'{identity}-finished.json',dict(entry['receipt'],exit_code=code,finished_unix=time.time()))
                (completed if code==0 else failed).append(identity)
                del active[identity]
            preflight_ok=all(j['id'] in completed for j in protocol['preflight'])
            if preflight_ok:
                for j in protocol['preflight']:
                    root=out/'preflight'/j['id']
                    # FP32 parity must pass. Typed BF16 B16 OOM is recorded and
                    # re-observed in its formal allocation, without lowering B.
                    if j['mode']=='validate' or not (root/'failure.json').exists():
                        assert (root/'complete.json').exists()
                    else:assert json.loads((root/'failure.json').read_text())['status']=='oom'
            preflight_failed=any(j['id'] in failed for j in protocol['preflight'])
            candidates=protocol['jobs'] if preflight_ok else protocol['preflight']
            allowed=json.loads((BASE/'configs/gpu-allocation.json').read_text())['allowed_gpus']
            busy={x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                '--format=csv,noheader'],text=True).splitlines()}
            free=[g for g in protocol['allowed_gpus'] if g in allowed and g not in [e['gpu'] for e in active.values()] and uuids[g] not in busy]
            if not preflight_failed and not (out/'pause-new-jobs').exists():
                for job in candidates:
                    identity=job['id']
                    if identity in completed+failed or identity in active or not free:continue
                    gpu=free.pop(0)
                    destination=out/('runs' if job['mode']=='formal' else 'preflight')/identity
                    if destination.exists():raise RuntimeError(f'Inspect/archive interrupted attempt before restart: {destination}')
                    argv=[PYTHON,'-u',str(out/'source/evaluate_loopahead_throughput.py'),
                        '--model',job['model'],'--batch',str(job['batch']),'--mode',job['mode'],
                        '--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(destination)]
                    log=(out/'logs'/f'{identity}.log').open('x')
                    process=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    receipt=dict(pid=process.pid,argv=argv,gpu=gpu,started_unix=time.time())
                    save(out/'logs'/f'{identity}-launcher.json',receipt)
                    active[identity]=dict(process=process,log=log,gpu=gpu,receipt=receipt)
            results=collect(out,protocol)
            status=dict(updated_unix=time.time(),stage='formal' if preflight_ok else 'preflight',completed=completed,
                failed=failed,formal_completed=sum(r['outcome']=='complete' for r in results),
                formal_oom=sum(r['outcome']=='oom' for r in results),planned_cells=10,
                running={k:dict(pid=v['process'].pid,gpu=v['gpu'],started_unix=v['receipt']['started_unix']) for k,v in active.items()},
                pending=[j['id'] for j in protocol['jobs'] if j['id'] not in completed+failed and j['id'] not in active])
            save(out/'status.json',status)
            if not active and (preflight_failed or all(j['id'] in completed+failed for j in protocol['jobs'])):
                save(out/'finished.json',status)
                if failed:raise RuntimeError(f'Failed jobs: {failed}')
                return
            time.sleep(15)


if __name__=='__main__':main()
