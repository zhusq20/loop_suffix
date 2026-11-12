"""Insert a bounded experimental probe, then resume already authorized data work."""
import json
from pathlib import Path
import subprocess
import time

base=Path(__file__).resolve().parents[1]
while not (base/'data/teachers/loopahead-train/huginn-3.5b/shard0/complete.json').exists():time.sleep(10)
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
while uuid in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True):time.sleep(10)
name='teacher-packed-graph-v1'
argv=['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'probe_teacher_graph.py'),
      '--gpu','0','--cpus','0-3','--out',str(base/'preflight'/name)]
with (base/f'logs/{name}.log').open('x') as f:
    child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
event=dict(pid=child.pid,argv=argv,started_unix=time.time())
(base/f'logs/{name}-launcher.json').write_text(json.dumps(event,indent=2)+'\n')
code=child.wait();event.update(exit_code=code,finished_unix=time.time())
(base/f'logs/{name}-completed.json').write_text(json.dumps(event,indent=2)+'\n')
# A failed optional optimization does not stop the validated native producer.
# Graph adoption is a separate explicit action after inspecting its evidence.
subprocess.run(['python',str(base/'logs/continue-huginn-loopahead.py'),
    '--gpu','0','--cpus','0-3','--shards','0,2,5','--first-training-already-running'],check=True)
