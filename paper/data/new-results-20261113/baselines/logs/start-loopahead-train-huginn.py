"""Wait for all protected Huginn features, train, and run the first trained check."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

base=Path(__file__).resolve().parents[1]
dependencies=[base/f'features/loopahead-{stage}/huginn-3.5b/shard{shard}/complete.json'
              for stage in ['train','dev','confirmation'] for shard in range(8)]
while not all(p.exists() for p in dependencies):time.sleep(10)
uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
events=base/'logs/loopahead-train-huginn-events.jsonl'


def run(name,argv):
    while uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(10)
    with (base/f'logs/{name}.log').open('x') as f:
        child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    record=dict(name=name,pid=child.pid,argv=argv,started_unix=time.time())
    with events.open('a') as f:f.write(json.dumps(dict(record,event='start'))+'\n')
    code=child.wait()
    with events.open('a') as f:f.write(json.dumps(dict(record,event='finish',exit_code=code,finished_unix=time.time()))+'\n')
    if code:raise RuntimeError(f'{name} failed; dependent work stopped')


name='loopahead-huginn-3.5b-formal-v1';training=base/'checkpoints'/name
run(name,['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'train_loopahead.py'),
    '--model','huginn-3.5b','--features',str(base/'features'),'--gpu','2','--cpus','8-11','--out',str(training)])
export=base/'training'/name;export.mkdir(exist_ok=True)
for p in training.glob('*.json*'):shutil.copy2(p,export/p.name)
shutil.copytree(training/'source',export/'source',dirs_exist_ok=True)
selected=Path(json.loads((training/'complete.json').read_text())['selected_checkpoint'])/'lad.pt'
for method in ['loopahead','ar']:
    name=f'loopahead-trained-huginn-3.5b-gsm8k-{method}-bfloat16-n20-v1'
    argv=['/tmp/loopfill-venv/bin/python','-u',str(base/'evaluate_loopahead.py'),'--model','huginn-3.5b',
        '--method',method,'--dataset','gsm8k','--temperature','0','--count','20','--pilot','--trace',
        '--gpu','2','--cpus','8-11','--out',str(base/'preflight'/name)]
    if method=='loopahead':argv+=['--checkpoint',str(selected)]
    run(name,argv)
