"""Keep the completed BF16 diagnostic lane for its FP32 continuation."""
import json
from pathlib import Path
import subprocess
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
identity = 'loopahead-huginn-3.5b-gsm8k-v1'
name = 'formal-greedy-fp32-'+identity
parent_name = 'formal-greedy-margins-'+identity
folder = BASE/'preflight/formal-greedy-margins'/identity
pause = BASE/'logs/formal-controller-v1/pause-new-jobs'
with pause.open('x') as stream:
    stream.write('Reserve the currently running BF16 diagnostic lane for its FP32 continuation.\n')
child = None
try:
    deadline = time.time()+1800
    while not (BASE/f'logs/{parent_name}-finished.json').exists():
        if time.time()>deadline:
            raise RuntimeError('BF16 diagnostic did not terminate within30min')
        time.sleep(5)
    finished = json.loads((BASE/f'logs/{parent_name}-finished.json').read_text())
    assert finished['exit_code']==0
    report = json.loads((folder/'report.json').read_text())
    assert report['status']=='complete' and report['cases']==83
    assert all(r['both_original_prefixes_reproduced'] for r in report['results'])
    gpu = finished['gpu']
    cpus = ['0-3','4-7','8-11','12-15','48-51','52-55','56-59','60-63'][gpu]
    uuid = subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    argv = ['/tmp/loopfill-venv/bin/python','-u',str(BASE/'diagnose_trained_loopahead.py'),
        '--ar',str(BASE/'runs/archived-ar-huginn-3.5b-gsm8k-t0'),
        '--loopahead',str(BASE/'runs/loopahead-huginn-3.5b-gsm8k-t0-v1'),
        '--observed-margin-report',str(folder/'report.json'),
        '--gpu',str(gpu),'--cpus',cpus,'--out',str(folder/'fp32-common-prefix')]
    with (BASE/f'logs/{name}.log').open('x') as stream:
        child = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    event = dict(pid=child.pid,argv=argv,gpu=gpu,started_unix=time.time())
    (BASE/f'logs/{name}-launcher.json').write_text(json.dumps(event,indent=2)+'\n')
    print(json.dumps(event),flush=True)
    deadline = time.time()+300
    while child.poll() is None:
        busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        if any(line.split(',')[0].strip()==str(child.pid) and uuid in line for line in busy.splitlines()):
            break
        if time.time()>deadline:
            child.terminate()
            raise RuntimeError('FP32 child did not claim its GPU within5min')
        time.sleep(1)
finally:
    pause.unlink(missing_ok=True)
if child is not None:
    code = child.wait()
    (BASE/f'logs/{name}-finished.json').write_text(json.dumps(dict(event,exit_code=code,finished_unix=time.time()),indent=2)+'\n')
    assert code==0,code
