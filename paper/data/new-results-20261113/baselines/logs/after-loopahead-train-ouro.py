"""Selected LAD -> fixed dev calibration -> B4 -> remaining Huginn data shards."""
import json
import os
from pathlib import Path
import subprocess
import time

base=Path(__file__).resolve().parents[1]
training=base/'checkpoints/loopahead-ouro-2.6b-formal-v1'
while not (training/'complete.json').exists():
    if (training/'failure.json').exists():raise RuntimeError('LAD training failed')
    time.sleep(10)
selected=Path(json.loads((training/'complete.json').read_text())['selected_checkpoint'])/'lad.pt'
uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
events=base/'logs/after-loopahead-train-ouro-events.jsonl'


def run(name,argv):
    while uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(5)
    with (base/f'logs/{name}.log').open('x') as f:
        child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    event=dict(name=name,pid=child.pid,argv=argv,started_unix=time.time())
    with events.open('a') as f:f.write(json.dumps(dict(event,event='start'))+'\n')
    code=child.wait()
    with events.open('a') as f:f.write(json.dumps(dict(event,event='finish',exit_code=code,finished_unix=time.time()))+'\n')
    if code:raise RuntimeError(f'{name} failed; dependent work stopped')


common=['/tmp/loopfill-venv/bin/python','-u',str(base/'evaluate_loopahead.py'),'--model','ouro-2.6b',
        '--gpu','2','--cpus','8-11']
paths={}
for mode,flag in [('native','--native-only'),('profile','--profile-dev')]:
    name=f'loopahead-dev16-ouro-2.6b-{mode}-v1';paths[mode]=base/'preflight'/name
    run(name,common+['--method','loopahead','--temperature','1','--checkpoint',str(selected),'--dev-calibration',
        flag,'--out',str(paths[mode])])
admission=base/'configs/loopahead-ouro-2.6b-admission-v1.json'
subprocess.run(['/tmp/loopfill-dflash-venv/bin/python',str(base/'calibrate_loopahead.py'),
    '--native-run',str(paths['native']),'--profile-run',str(paths['profile']),'--out',str(admission)],check=True)
for method in ['loopahead','ar']:
    name=f'loopahead-B4-ouro-2.6b-humaneval-{method}-v1'
    argv=common+['--method',method,'--temperature','0','--paper-humaneval','--out',str(base/'preflight'/name)]
    if method=='loopahead':argv+=['--checkpoint',str(selected)]
    run(name,argv)
subprocess.run(['python',str(base/'summarize_loopahead_b4.py')],check=True)
subprocess.run(['python',str(base/'logs/continue-huginn-loopahead.py'),'--gpu','2','--cpus','8-11','--shards','4,7'],check=True)
