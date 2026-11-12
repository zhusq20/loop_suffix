"""Execute an explicit per-GPU teacher plan after existing jobs finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base))
from production_backend import teacher_backend_args
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
plan=json.loads(args.plan.read_text())
gpu,cpus=plan['gpu'],plan['cpus']
uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
events=base/f'logs/continue-dflash-teachers-gpu{gpu}.jsonl'
for relative in plan['dependencies']:
    while not (base/relative).exists():time.sleep(10)
for job in plan['jobs']:
    model,shard=job['model'],job['shard']
    while uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(10)
    name=f'teacher-dflash-train-{model}-shard{shard}-continuation-v1'
    argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'generate_teachers.py'),
        '--model',model,'--manifest',str(base/'data/manifests/dflash-train.jsonl'),
        '--shard',str(shard),'--shards','8','--max-batch','32','--kv-budget-gib','70',
        '--gpu',str(gpu),'--cpus',cpus,'--out',str(base/f'data/teachers/dflash-train/{model}/shard{shard}')]
    argv+=teacher_backend_args(model)
    with (base/f'logs/{name}.log').open('x') as f:
        child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    event=dict(name=name,pid=child.pid,argv=argv,started_unix=time.time())
    with events.open('a') as f:f.write(json.dumps(dict(event,event='start'))+'\n')
    code=child.wait()
    with events.open('a') as f:f.write(json.dumps(dict(event,event='finish',exit_code=code,finished_unix=time.time()))+'\n')
    if code:raise RuntimeError(f'{name} failed; subsequent planned jobs stopped')
