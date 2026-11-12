"""Continue each completed Ouro teacher lane with features, then new teacher work."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import threading
import time

base = Path(__file__).resolve().parents[1]
mutex = threading.Lock()


def record(event):
    with mutex, (base/'logs/loopahead-features-ouro-events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')


def run_job(name, argv):
    with (base/f'logs/{name}.log').open('x') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'))
    event = dict(name=name, argv=argv, pid=child.pid, started_unix=time.time())
    record(dict(event, event='start'))
    code = child.wait()
    record(dict(event, event='finish', exit_code=code, finished_unix=time.time()))
    if code:
        raise RuntimeError(f'{name} failed; dependent work stopped')


def lane(shard, gpu, cpus, next_stage=None, next_model=None, next_shard=None):
    dependency = base/f'data/teachers/loopahead-confirmation/ouro-2.6b/shard{shard}/complete.json'
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    while not dependency.exists() or uuid in subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'], text=True):
        time.sleep(10)
    for stage in ['train','dev','confirmation']:
        name = f'features-loopahead-{stage}-ouro-2.6b-shard{shard}-v1'
        run_job(name, ['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'extract_loopahead_features.py'),
            '--model','ouro-2.6b','--teacher-shard',str(base/f'data/teachers/loopahead-{stage}/ouro-2.6b/shard{shard}'),
            '--gpu',str(gpu),'--cpus',cpus,'--out',str(base/f'features/loopahead-{stage}/ouro-2.6b/shard{shard}')])
    if next_stage:
        name = f'teacher-{next_stage}-{next_model}-shard{next_shard}'
        run_job(name, ['/tmp/loopfill-dflash-venv/bin/python','-u',str(base/'generate_teachers.py'),
            '--model',next_model,'--manifest',str(base/f'data/manifests/{next_stage}.jsonl'),
            '--shard',str(next_shard),'--shards','8','--max-batch','16' if next_model.startswith('huginn') else '32',
            '--kv-budget-gib','32' if next_model.startswith('huginn') else '70','--gpu',str(gpu),'--cpus',cpus,
            '--out',str(base/f'data/teachers/{next_stage}/{next_model}/shard{next_shard}')])


with ThreadPoolExecutor(max_workers=4) as pool:
    jobs = [pool.submit(lane,0,2,'8-11'),
            pool.submit(lane,1,3,'12-15','dflash-train','ouro-2.6b',2),
            pool.submit(lane,2,6,'56-59','dflash-train','ouro-2.6b',3),
            pool.submit(lane,3,7,'60-63','loopahead-train','huginn-3.5b',1)]
    for job in jobs:
        job.result()
