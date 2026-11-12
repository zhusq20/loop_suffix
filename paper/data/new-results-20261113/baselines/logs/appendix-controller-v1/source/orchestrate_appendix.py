"""Ouro1.4 appendix phase, gated on all48 completed main-paper cells.

No GPU work starts until the main controller and result validation both succeed.
Every failed command stops dependent work and retains its original journal.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import time

from orchestrate_formal import BASE,CPUS,TASKS,run_command,save,selected,worker
from production_backend import teacher_backend_args
from gpu_allocation import allowed_gpus

MODEL='ouro-1.4b'
PY='/tmp/loopfill-dflash-venv/bin/python'
INFER='/tmp/loopfill-venv/bin/python'


def parallel(gpus,fn):
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(fn,g) for g in gpus]
        errors=[]
        for future in futures:
            try:future.result()
            except Exception as exc:errors.append(repr(exc))
    if errors:raise RuntimeError(errors)


def command(script,name,gpu,extra=(),out=None,python=PY):
    out=out or BASE/'preflight'/name
    run_command(name,[python,'-u',str(BASE/script),'--model',MODEL,'--gpu',str(gpu),
        '--cpus',CPUS[gpu],'--out',str(out),*map(str,extra)],
        dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))


def produce(gpu,stage,shard):
    name=f'teacher-{stage}-{MODEL}-shard{shard}-appendix-v1'
    out=BASE/f'data/teachers/{stage}/{MODEL}/shard{shard}'
    command('generate_teachers.py',name,gpu,
        ['--manifest',BASE/f'data/manifests/{stage}.jsonl','--shard',shard,'--shards',8,
         '--max-batch',32,'--kv-budget-gib',70,*teacher_backend_args(MODEL)],out)


def lad_data(gpu,shard):
    for stage in ['train','dev','confirmation']:produce(gpu,'loopahead-'+stage,shard)
    for stage in ['train','dev','confirmation']:
        name=f'features-loopahead-{stage}-{MODEL}-shard{shard}-appendix-v1'
        command('extract_loopahead_features.py',name,gpu,
            ['--teacher-shard',BASE/f'data/teachers/loopahead-{stage}/{MODEL}/shard{shard}'],
            BASE/f'features/loopahead-{stage}/{MODEL}/shard{shard}')


def train_lad(gpu,uuids):
    name=f'loopahead-{MODEL}-formal-v1';out=BASE/'checkpoints'/name
    command('train_loopahead.py',name,gpu,['--features',BASE/'features'],out)
    export=BASE/'training'/name;export.mkdir(exist_ok=True)
    for p in out.glob('*.json*'):shutil.copy2(p,export/p.name)
    shutil.copytree(out/'source',export/'source',dirs_exist_ok=True)
    for method in ['loopahead','ar']:
        name=f'loopahead-trained-{MODEL}-gsm8k-{method}-bfloat16-n20-v1'
        extra=['--method',method,'--dataset','gsm8k','--count',20,'--pilot','--trace']
        if method=='loopahead':extra+=['--checkpoint',selected(MODEL)]
        command('evaluate_loopahead.py',name,gpu,extra,python=INFER)
    worker(dict(kind='trained_check',model=MODEL),[gpu],uuids)
    paths={}
    for mode,flag in [('native','--native-only'),('profile','--profile-dev')]:
        name=f'loopahead-dev16-{MODEL}-{mode}-v1';paths[mode]=BASE/'preflight'/name
        command('evaluate_loopahead.py',name,gpu,
            ['--method','loopahead','--temperature',1,'--checkpoint',selected(MODEL),
             '--dev-calibration',flag],python=INFER)
    run_command(f'loopahead-admission-{MODEL}-v1',[PY,str(BASE/'calibrate_loopahead.py'),
        '--native-run',str(paths['native']),'--profile-run',str(paths['profile']),
        '--out',str(BASE/f'configs/loopahead-{MODEL}-admission-v1.json')])


def evaluate(kind,gpus,uuids):
    pending=queue.Queue()
    for temp in [0,1]:
        for task in TASKS:pending.put(dict(kind=kind,model=MODEL,dataset=task,temperature=temp))
    def lane(gpu):
        while True:
            try:job=pending.get_nowait()
            except queue.Empty:return
            worker(job,[gpu],uuids)
            pending.task_done()
    parallel(gpus,lane)
    assert pending.empty()


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    gpus=allowed_gpus()
    assert len(gpus)>=3, 'Appendix parallel training/evaluation needs at least three assigned GPUs'
    primary=gpus[0]
    shard_assignment={gpu:list(range(index,8,len(gpus))) for index,gpu in enumerate(gpus)}
    training_gpus=gpus[:max(2,len(gpus)//2)]
    evaluation_gpus=gpus[len(training_gpus):]
    if not args.execute:
        print(json.dumps(dict(model=MODEL,main_cells_required=48,appendix_cells=24,
            allowed_gpus=gpus,shard_assignment=shard_assignment,training_gpus=training_gpus,evaluation_gpus=evaluation_gpus,
            stages=['compatibility and graph probe','8-shard LAD data/features',
                'LAD training/check/dev calibration plus8-shard DFlash50k',
                'independent complete50k DFlash teacher audit',
                '6-epoch DFlash training and12 LAD cells share only the assigned GPUs',
                '12 DFlash cells with own AR and separate full-task draft profiling',
                'validate and export all72 cells',
                'replay all36 greedy cells for first-divergence probability margins',
                'render final training costs']),indent=2));return
    out=BASE/'logs/appendix-controller-v1';out.mkdir(exist_ok=False)
    from provenance import freeze_sources
    freeze_sources(out,[Path(__file__),BASE/'orchestrate_formal.py',BASE/'production_backend.py',
        BASE/'audit_dflash_teachers.py',BASE/'render_training_costs.py',
        BASE/'orchestrate_greedy_diagnostics.py',BASE/'diagnose_formal_loopahead.py',BASE/'diagnose_dflash.py',
        BASE/'gpu_allocation.py',BASE/'configs/gpu-allocation.json'])
    with open('/tmp/loopfill-appendix-controller.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main_dir=BASE/'logs/formal-controller-v1'
        save(out/'status.json',dict(stage='waiting_for_main48',started_unix=time.time()))
        while not (main_dir/'complete.json').exists():
            status=json.loads((main_dir/'status.json').read_text())
            if status['failed']:raise RuntimeError(f'Main phase failures require investigation: {status["failed"]}')
            time.sleep(15)
        assert not json.loads((main_dir/'complete.json').read_text())['failed']
        run_command('main48-results-export-v1',['python',str(BASE/'collect_results.py'),'--partial'])
        matrix=json.loads((BASE/'formal-matrix-status.json').read_text())
        assert matrix['completed']==48
        shutil.copy2(BASE/'results.csv',BASE/'main48-results.csv')
        shutil.copy2(BASE/'table2_rows.md',BASE/'main48-table2_rows.md')
        uuids={int(i):u.strip() for i,u in (l.split(',') for l in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
        while any(uuids[g] in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True) for g in gpus):time.sleep(15)
        save(out/'status.json',dict(stage='compatibility',updated_unix=time.time()))
        for script,name,extra,python in [
            ('probe_dflash_target.py','native-target-ouro-1.4b-appendix-v1',[],PY),
            ('smoke_dflash.py','dflash-ouro-1.4b-appendix-smoke-v1',[],PY),
            ('smoke_loopahead.py','loopahead-ouro-1.4b-appendix-fp32-smoke-v1',['--dtype','float32'],INFER)]:
            command(script,name,primary,extra,python=python)
        report=json.loads((BASE/'preflight/loopahead-ouro-1.4b-appendix-fp32-smoke-v1/report.json').read_text())
        assert report['greedy_ar_equal'] and report['repeat_exact']
        proof=BASE/'preflight/teacher-packed-graph-ouro-1.4b-v1'
        run_command('teacher-packed-graph-ouro-1.4b-v1',[PY,'-u',str(BASE/'probe_teacher_graph.py'),
            '--gpu',str(primary),'--cpus',CPUS[primary],'--models',MODEL,'--out',str(proof)])
        report=json.loads((proof/'report.json').read_text());assert report['status']=='complete'
        assert len(report['results'])==2
        for r in report['results']:
            assert r['model']==MODEL and r['graph_off_on_exact_sequences']==r['requests']==8
            if r['dtype']=='float32':assert r['native_fp32_exact_sequences']==8
        save(BASE/f'configs/teacher-graph-adoption-{MODEL}-v1.json',dict(enabled_models=[MODEL],
            proof=str(proof),proof_sha256=hashlib.sha256((proof/'report.json').read_bytes()).hexdigest(),
            adopted_unix=time.time(),scope='Appendix teacher production only; benchmark remains Graph OFF',
            limitations=report['limitations']))
        save(out/'status.json',dict(stage='LAD_teacher_and_features',updated_unix=time.time()))
        def lad_lane(gpu):
            for shard in shard_assignment[gpu]:lad_data(gpu,shard)
        parallel(gpus,lad_lane)
        save(out/'status.json',dict(stage='LAD_training_and_DFlash_teachers',updated_unix=time.time()))
        def lane(gpu):
            if gpu==primary:train_lad(gpu,uuids)
            for shard in shard_assignment[gpu]:produce(gpu,'dflash-train',shard)
        parallel(gpus,lane)
        save(out/'status.json',dict(stage='DFlash_teacher_audit',updated_unix=time.time()))
        audit=BASE/f'training/dflash-{MODEL}-data-v1/teacher-audit.json'
        run_command(f'audit-dflash-{MODEL}-full-v1',[PY,'-u',str(BASE/'audit_dflash_teachers.py'),
            '--model',MODEL,'--out',str(audit)])
        report=json.loads(audit.read_text())
        assert report['status']=='complete' and report['model']==MODEL
        assert report['completed_requests']==50000 and report['training_input_gate_passed']
        save(out/'status.json',dict(stage='DFlash_training_and_LAD_evaluation',updated_unix=time.time()))
        with ThreadPoolExecutor(max_workers=2) as pool:
            training=pool.submit(worker,dict(kind='train',model=MODEL),training_gpus,uuids)
            evaluation=pool.submit(evaluate,'loopahead',evaluation_gpus,uuids)
            training.result();evaluation.result()
        save(out/'status.json',dict(stage='DFlash_evaluation',updated_unix=time.time()))
        evaluate('dflash',gpus,uuids)
        run_command('all72-results-export-v1',['python',str(BASE/'collect_results.py')])
        save(out/'status.json',dict(stage='formal_greedy_divergence_diagnostics',updated_unix=time.time()))
        run_command('all36-greedy-diagnostics-v1',['python',str(BASE/'orchestrate_greedy_diagnostics.py'),'--execute'])
        run_command('all72-training-cost-report-v1',['python',str(BASE/'render_training_costs.py')])
        save(out/'complete.json',dict(expected_cells=72,finished_unix=time.time()))
        save(out/'status.json',dict(stage='completed_requires_final_requirement_audit',updated_unix=time.time()))


if __name__=='__main__':main()
