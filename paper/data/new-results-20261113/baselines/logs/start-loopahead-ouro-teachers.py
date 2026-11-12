"""Four shard chains on the cards released by the DFlash pilot teacher job."""
import json
import os
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parents[1]
pilot = base / 'data/teachers/dflash-pilot/ouro-2.6b'
while not all((pilot / f'shard{i}/complete.json').exists() for i in range(4)):
    if any((pilot / f'shard{i}/failure.json').exists() for i in range(4)):
        raise RuntimeError('Pilot teacher failed; inspect before reassigning its cards')
    time.sleep(10)
gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
uuids = {int(i.strip()): u.strip() for i, u in (line.split(',') for line in gpu_lines.splitlines())}
assignments = [(2, '8-11'), (3, '12-15'), (6, '56-59'), (7, '60-63')]
while True:
    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    if not any(uuids[gpu] in busy for gpu, _ in assignments):
        break
    time.sleep(5)

pending = []
for shard, (gpu, cpus) in enumerate(assignments):
    for stage in ['loopahead-train', 'loopahead-dev', 'loopahead-confirmation']:
        pending.append(dict(shard=shard, gpu=gpu, cpus=cpus, stage=stage))
running, completed = {}, []
while pending or running:
    for task in pending[:]:
        shard = task['shard']
        if shard in running:
            continue
        stage = task['stage']
        argv = ['/tmp/loopfill-dflash-venv/bin/python', '-u', str(base / 'generate_teachers.py'),
                '--model', 'ouro-2.6b', '--manifest', str(base / f'data/manifests/{stage}.jsonl'),
                '--shard', str(shard), '--shards', '4', '--max-batch', '32', '--gpu', str(task['gpu']),
                '--cpus', task['cpus'], '--out', str(base / f'data/teachers/{stage}/ouro-2.6b/shard{shard}')]
        with (base / f'logs/teacher-{stage}-ouro-2.6b-shard{shard}.log').open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(task, pid=child.pid, argv=argv, started_unix=time.time())
        with (base / 'logs/teacher-loopahead-ouro-launcher.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        running[shard] = (child, record)
        pending.remove(task)
    for shard, (child, record) in list(running.items()):
        code = child.poll()
        if code is not None:
            record.update(exit_code=code, finished_unix=time.time())
            completed.append(record)
            del running[shard]
            if code != 0:
                # Other authorised shards can finish, but no dependent stage on
                # this GPU is automatically restarted after an unexplained error.
                pending = [t for t in pending if t['shard'] != shard]
            (base / 'logs/teacher-loopahead-ouro-completed.json').write_text(json.dumps(completed, indent=2) + '\n')
    time.sleep(10)
