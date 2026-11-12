"""Authorised sequential verification jobs after the original GPU4 calibration."""
import json
import os
from pathlib import Path
import subprocess
import time

base=Path(__file__).resolve().parents[1]
dependency=base/'preflight/ouro-2.6b-gsm8k-ar-n300/complete.json'
while not dependency.exists():
    try:os.kill(439962,0)
    except ProcessLookupError:raise RuntimeError('Calibration ended without a complete journal')
    time.sleep(10)
subprocess.run(['python',str(base/'summarize_calibration.py'),'--count','300'],check=True)
jobs=[]
for dtype in ['bfloat16','float32']:
    for method in ['loopahead','ar']:
        name=f'loopahead-B2-ouro-2.6b-gsm8k-{method}-{dtype}-n20-v1'
        argv=['/tmp/loopfill-venv/bin/python','-u',str(base/'evaluate_loopahead.py'),
              '--model','ouro-2.6b','--method',method,'--dataset','gsm8k','--temperature','0',
              '--dtype',dtype,'--count','20','--gpu','4','--cpus','48-51','--pilot','--trace','--out',str(base/'preflight'/name)]
        if method=='loopahead':argv+=['--random-lad']
        jobs.append((name,argv))
name='huginn-production-cache-v1'
jobs.append((name,['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'probe_production_cache.py'),
                  '--gpu','4','--cpus','48-51','--out',str(base/'preflight'/name)]))
for name,argv in jobs:
    with (base/f'logs/{name}.log').open('x') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(name=name,pid=child.pid,argv=argv,started_unix=time.time())
    with (base/'logs/after-calibration-gpu4-launcher.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    code=child.wait()
    record.update(exit_code=code,finished_unix=time.time())
    with (base/'logs/after-calibration-gpu4-completed.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if code:raise RuntimeError(f'{name} failed with exit code{code}; inspect before dependent work')
