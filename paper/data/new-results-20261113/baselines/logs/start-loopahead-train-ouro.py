"""Start the authorized LAD recipe only after all B2 checks and feature audits."""
import json
import os
from pathlib import Path
import subprocess
import time

base=Path(__file__).resolve().parents[1]
dependencies=[base/f'preflight/loopahead-B2-ouro-2.6b-gsm8k-{method}-{dtype}-n20-v1/complete.json'
              for method in ['ar','loopahead'] for dtype in ['bfloat16','float32']]
dependencies += [base/'preflight/loopahead-B2-common-prefix-fp32-v1/report.json',
                 base/'training/loopahead-ouro-2.6b-data-v1/feature-audit.json']
while not all(p.exists() for p in dependencies):time.sleep(10)
subprocess.run(['python',str(base/'summarize_loopahead_b2.py')],check=True)
uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
while uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(10)
name='loopahead-ouro-2.6b-formal-v1'
argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'train_loopahead.py'),
    '--model','ouro-2.6b','--features',str(base/'features'),'--gpu','2','--cpus','8-11',
    '--out',str(base/'checkpoints'/name)]
with (base/f'logs/{name}.log').open('x') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
record=dict(name=name,pid=child.pid,argv=argv,started_unix=time.time())
(base/f'logs/{name}-launcher.json').write_text(json.dumps(record,indent=2)+'\n')
code=child.wait();record.update(exit_code=code,finished_unix=time.time())
(base/f'logs/{name}-completed.json').write_text(json.dumps(record,indent=2)+'\n')
if code:raise RuntimeError('LAD training failed; inspect before downstream runs')
