"""Resume the two checks after the archived, pre-generation GPU contention failure."""
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parents[1]
training = BASE/'checkpoints/loopahead-huginn-3.5b-formal-v1'
selected = Path(json.loads((training/'complete.json').read_text())['selected_checkpoint'])/'lad.pt'
uuid = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
events = BASE/'logs/loopahead-train-huginn-events.jsonl'
for method in ['loopahead', 'ar']:
    name = f'loopahead-trained-huginn-3.5b-gsm8k-{method}-bfloat16-n20-v1'
    argv = ['/tmp/loopfill-venv/bin/python', '-u', str(BASE/'evaluate_loopahead.py'), '--model', 'huginn-3.5b',
        '--method', method, '--dataset', 'gsm8k', '--temperature', '0', '--count', '20', '--pilot', '--trace',
        '--gpu', '2', '--cpus', '8-11', '--out', str(BASE/'preflight'/name)]
    if method == 'loopahead':
        argv += ['--checkpoint', str(selected)]
    while uuid in subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True):
        time.sleep(5)
    with (BASE/f'logs/{name}.log').open('x') as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'), start_new_session=True)
    event = dict(name=name, pid=process.pid, argv=argv, started_unix=time.time(), recovery='weather-contention-v2')
    with events.open('a') as stream:
        stream.write(json.dumps(dict(event, event='start'))+'\n')
    code = process.wait()
    with events.open('a') as stream:
        stream.write(json.dumps(dict(event, event='finish', exit_code=code, finished_unix=time.time()))+'\n')
    if code:
        raise RuntimeError(f'{name} failed with exit{code}')
