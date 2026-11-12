"""Finish assigned Huginn LAD shards, with validated teacher-only cache storage."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base))
from production_backend import teacher_backend_args
p=argparse.ArgumentParser()
p.add_argument('--gpu',type=int,required=True)
p.add_argument('--cpus',required=True)
p.add_argument('--shards',required=True)
p.add_argument('--first-training-already-running',action='store_true')
p.add_argument('--wait-for',type=Path,help='complete dependency before taking a shard')
args=p.parse_args()
shards=list(map(int,args.shards.split(',')))
uuid=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
events=base/f'logs/continue-huginn-loopahead-gpu{args.gpu}.jsonl'


def run(name,argv):
    while uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(10)
    with (base/f'logs/{name}.log').open('x') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    event=dict(name=name,pid=child.pid,argv=argv,started_unix=time.time())
    with events.open('a') as f:f.write(json.dumps(dict(event,event='start'))+'\n')
    code=child.wait()
    with events.open('a') as f:f.write(json.dumps(dict(event,event='finish',exit_code=code,finished_unix=time.time()))+'\n')
    if code:raise RuntimeError(f'{name} failed, dependent work stopped')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def finish_shard(index,shard):
    for stage in ['train','dev','confirmation']:
        folder=base/f'data/teachers/loopahead-{stage}/huginn-3.5b/shard{shard}'
        if (folder/'complete.json').exists():
            complete=json.loads((folder/'complete.json').read_text())
            meta=json.loads((folder/'metadata.json').read_text())['identity']
            assert meta['model']=='huginn-3.5b' and meta['shard']==shard and meta['shards']==8
            assert meta['manifest_sha256']==digest(base/f'data/manifests/loopahead-{stage}.jsonl')
            assert complete['teachers_sha256']==digest(folder/'teachers.jsonl')
            continue
        if index==0 and stage=='train' and args.first_training_already_running:
            while not (folder/'complete.json').exists():time.sleep(10)
        else:
            name=f'teacher-loopahead-{stage}-huginn-3.5b-shard{shard}-dense-v1'
            run(name,['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'generate_teachers.py'),
                '--model','huginn-3.5b','--manifest',str(base/f'data/manifests/loopahead-{stage}.jsonl'),
                '--shard',str(shard),'--shards','8','--max-batch','16','--kv-budget-gib','32',
                '--gpu',str(args.gpu),'--cpus',args.cpus,'--out',str(folder)]+teacher_backend_args('huginn-3.5b'))
    for stage in ['train','dev','confirmation']:
        folder=base/f'features/loopahead-{stage}/huginn-3.5b/shard{shard}'
        if (folder/'complete.json').exists():
            complete=json.loads((folder/'complete.json').read_text())
            meta=json.loads((folder/'metadata.json').read_text())
            teacher=json.loads((base/f'data/teachers/loopahead-{stage}/huginn-3.5b/shard{shard}/complete.json').read_text())
            assert meta['teacher_sha256']==teacher['teachers_sha256']
            assert complete['requests']==teacher['requests'] and complete['index_sha256']==digest(folder/'index.json')
            continue
        name=f'features-loopahead-{stage}-huginn-3.5b-shard{shard}-v1'
        run(name,['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'extract_loopahead_features.py'),
            '--model','huginn-3.5b','--teacher-shard',str(base/f'data/teachers/loopahead-{stage}/huginn-3.5b/shard{shard}'),
            '--gpu',str(args.gpu),'--cpus',args.cpus,'--out',str(base/f'features/loopahead-{stage}/huginn-3.5b/shard{shard}')])


if args.wait_for:
    while not args.wait_for.exists():time.sleep(10)
for index,shard in enumerate(shards):
    # Additional lanes may take shards4/7 before the long paper AR comparison
    # finishes. The original lane then validates and skips completed work.
    with (base/f'logs/huginn-loopahead-shard{shard}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        finish_shard(index,shard)
