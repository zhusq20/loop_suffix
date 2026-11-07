"""Durable, work-conserving eight-GPU queue for the five-method B=1 main table."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from loop_latency_worker import read, save, digest

REPO = Path(__file__).resolve().parents[1]
MODELS = ['huginn-3.5b', 'ouro-1.4b', 'ouro-2.6b']
ARMS = ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-online']
DATASETS = ['gsm8k', 'math500', 'bbh', 'humanevalplus', 'mbppplus', 'specbench']


def gpu_state():
    fields = ['index', 'uuid', 'memory.used', 'utilization.gpu', 'temperature.gpu', 'clocks.sm', 'clocks.mem']
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=' + ','.join(fields),
                                      '--format=csv,noheader,nounits'], text=True)
    rows = [dict(zip(fields, [v.strip() for v in line.split(',')])) for line in output.splitlines()]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.strip()}
    return {r['uuid']: dict(r, busy=r['uuid'] in busy) for r in rows}


def jobs():
    # Keep AR baselines early, but interleave models/datasets so every free GPU works.
    return [dict(key=f'{m}-{d}-{a}', model=m, dataset=d, arm=a, batch_size=1,
                 outcome='pending', failures=0, wandb_id=hashlib.sha256(f'{m}/{d}/{a}'.encode()).hexdigest()[:12])
            for a in ARMS for d in DATASETS for m in MODELS]


def prepare(out, assets, python, temperature=0.0, sampling_seed=0):
    out = Path(out).resolve()
    if temperature not in (0.0, 1.0):
        raise ValueError('supported experiment temperatures: 0 or 1')
    if (out / 'protocol.json').exists():
        raise FileExistsError('Protocol exists; use run to resume')
    manifest = read(out / 'inputs/manifest.json')
    source = out / 'source'
    source.mkdir(exist_ok=False)
    shutil.copytree(REPO / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'scripts').mkdir()
    for name in ['loop_latency_suite.py', 'loop_latency_worker.py', 'loop_latency_inputs.py', 'loop_batch_bench.py']:
        shutil.copy2(REPO / 'scripts' / name, source / 'scripts' / name)
    source_hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}
    assets = Path(assets).resolve()
    models = {m: read(assets / (m + '.json')) for m in MODELS}
    lanes = []
    for uuid, row in gpu_state().items():
        gpu = int(row['index'])
        start = gpu * 4 if gpu < 4 else 48 + (gpu - 4) * 4
        lanes.append(dict(gpu=gpu, uuid=uuid, cpus=f'{start}-{start+3}'))
    assert len(lanes) == 8
    # Shared with the other LoopFill queue: never run two cells on the same card.
    protocol = dict(name=out.name, created_unix=time.time(), source=str(source),
        source_sha256=source_hashes, python=str(Path(python).absolute()), assets=str(assets),
        models=models, lanes=lanes, inputs=manifest['datasets'], arms=ARMS, datasets=DATASETS,
        batches=[1], planned_cells=90, max_new_tokens=512, cuda_graph=False,
        temperature=temperature, top_p=1.0, top_k=0, sampling_seed=sampling_seed,
        sampler='target_inverse_cdf_absolute_position' if temperature else 'greedy',
        draft_sampling='deterministic shallow top-k and suffix retrieval',
        fixed_order_seed=20261107, repeats=1, warmup_requests=2, warmup_tokens=32,
        draft_node_budget=16, suffix_max_draft=4, model_top_k=2, child_suffix_caps=[4, 4],
        suffix_c_initial=.7, suffix_eta=.1,
        wandb_key_file='/tmp/loopfill-wandb-key',
        wandb_project='loopfill-latency512-t1' if temperature else 'loopfill-latency512',
        wandb_entity='zsqzz', resource_policy='one UUID and four disjoint NUMA-local CPU cores per cell; '
        'shared /tmp/loopfill-UUID.lock; poll real GPU occupancy before allocation; any free GPU takes next cell',
        recovery='append/fsync request journal; recover a prefix and replay only suffix publication; '
        'never regenerate completed requests; model and ordered input/source identity checked',
        timing='decode only after synchronized prefill; includes suffix work, excludes W&B and persistence',
        metric='speedup=sum(AR decode seconds)/sum(method decode seconds); '
        'also report pooled decode-token/s ratio and loop-normalized gamma',
        limits=['single timing repeat', '512-token capped generation', 'BF16 may differ from AR',
                'same-engine baselines, not official engine performance'])
    save(out / 'protocol.json', protocol)
    for folder in ['jobs', 'runs', 'validation', 'logs']:
        (out / folder).mkdir(exist_ok=True)
    for job in jobs():
        job['wandb_id'] = hashlib.sha256(f"{out.name}/{job['key']}".encode()).hexdigest()[:12]
        save(out / 'jobs' / (job['key'] + '.json'), job)
    save(out / 'status.json', dict(phase='prepared', planned_cells=90))
    print(json.dumps({'suite': str(out), 'cells': 90}), flush=True)


def aggregate(out):
    all_jobs = [read(p) for p in sorted((out / 'jobs').glob('*.json'))]
    summaries = {}
    for job in all_jobs:
        p = out / 'runs' / job['key'] / 'summary.json'
        if job['outcome'] == 'complete' and p.exists():
            summaries[job['key']] = read(p)
    rows = []
    for job in all_jobs:
        row = {k: job[k] for k in ['key', 'model', 'dataset', 'arm', 'outcome']}
        value = summaries.get(job['key'])
        if value:
            row.update({k: value[k] for k in ['completed_requests', 'decode_s', 'decode_tokens',
                       'decode_token_s', 'gamma', 'cap_stopped']})
            row.update({k: value[k] for k in ['mean_decode_s', 'draft_hits', 'mean_acceptance_rate', 'pooled_acceptance_rate'] if k in value})
            base = summaries.get(f"{job['model']}-{job['dataset']}-ar")
            if base:
                row['latency_speedup'] = base['decode_s'] / value['decode_s']
                row['decode_rate_speedup'] = value['decode_token_s'] / base['decode_token_s']
                row['output_token_ratio'] = value['decode_tokens'] / base['decode_tokens']
        rows.append(row)
    save(out / 'measurements.json', rows)
    columns = ['model', 'dataset', 'arm', 'outcome', 'completed_requests', 'decode_s', 'decode_tokens',
               'decode_token_s', 'gamma', 'cap_stopped', 'mean_decode_s', 'draft_hits', 'mean_acceptance_rate', 'pooled_acceptance_rate', 'latency_speedup', 'decode_rate_speedup', 'output_token_ratio']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, columns, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    temp = out / 'measurements.csv.tmp'
    temp.write_text(buf.getvalue())
    temp.replace(out / 'measurements.csv')
    counts = dict(Counter(j['outcome'] for j in all_jobs))
    completed_requests = 0
    for job in all_jobs:
        p = out / 'runs' / job['key'] / 'progress.json'
        if p.exists():
            completed_requests += read(p)['completed_requests']
    status = dict(updated_unix=time.time(), counts=counts, planned_cells=len(all_jobs),
                  completed_requests=completed_requests, planned_requests=2092 * 3 * 5)
    save(out / 'status.json', status)
    return status


def run(out):
    out = Path(out).resolve()
    supervisor_lock = (out / 'supervisor.lock').open('a')
    fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(out / 'protocol.json')
    source = Path(protocol['source'])
    env_base = dict(os.environ, PYTHONPATH=str(source / 'src') + ':' + str(Path(protocol['assets']) / 'ArcticInference'),
                    PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    # An interrupted controller must not duplicate a surviving worker.
    for path in (out / 'jobs').glob('*.json'):
        job = read(path)
        if job['outcome'] == 'running':
            try:
                os.kill(job['pid'], 0)
            except ProcessLookupError:
                job['outcome'] = 'pending'
                save(path, job)
            else:
                raise RuntimeError(f"Worker {job['pid']} is still alive; refusing duplicate controller")
    import wandb
    os.environ['WANDB_API_KEY'] = Path(protocol['wandb_key_file']).read_text().strip()
    tracking = wandb.init(entity=protocol['wandb_entity'], project=protocol['wandb_project'],
        group=protocol['name'], name='queue', id=hashlib.sha256(protocol['name'].encode()).hexdigest()[:12],
        resume='allow', dir=str(out), config={k: v for k, v in protocol.items() if k != 'wandb_key_file'},
        settings=wandb.Settings(init_timeout=90))
    save(out / 'launcher.json', dict(pid=os.getpid(), started_unix=time.time(), wandb_url=tracking.url,
                                    command=[sys.executable, __file__, 'run', '--out', str(out)]))
    active = {}
    last_report = 0
    validation = {}
    with (out / 'telemetry.jsonl').open('a', buffering=1) as telemetry:
        while True:
            for uuid, task in list(active.items()):
                code = task['process'].poll()
                if code is None:
                    continue
                task['log'].close()
                job = task['job']
                if task['smoke']:
                    passed = code == 0 and (out / 'validation' / job['key'] / 'complete.json').exists()
                    validation[(job['model'], job['arm'])] = passed
                    job['outcome'] = 'pending' if passed else 'validation_failed'
                else:
                    complete = code == 0 and (out / 'runs' / job['key'] / 'complete.json').exists()
                    job['outcome'] = 'complete' if complete else 'failed'
                    if not complete:
                        job['failures'] += 1
                job.update(exit_code=code, ended_unix=time.time())
                save(out / 'jobs' / (job['key'] + '.json'), job)
                fcntl.flock(task['lock'], fcntl.LOCK_UN)
                task['lock'].close()
                del active[uuid]
            pending = [read(out / 'jobs' / (j['key'] + '.json')) for j in jobs()]
            pending = [j for j in pending if j['outcome'] == 'pending']
            # Validate all model/method pairs early, before long formal cells fill the GPUs.
            pending.sort(key=lambda j: (j['model'], j['arm']) in validation)
            states = gpu_state()
            for lane in protocol['lanes']:
                uuid = lane['uuid']
                if uuid in active or not pending:
                    continue
                state = states[uuid]
                if state['busy'] or float(state['memory.used']) > 256 or float(state['utilization.gpu']) > 10:
                    continue
                lock = open(f'/tmp/loopfill-{uuid}.lock', 'a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock.close()
                    continue
                if gpu_state()[uuid]['busy']:
                    lock.close()
                    continue
                # One short real-checkpoint test per model/method, before that pair's formal cells.
                chosen = None
                for job in pending:
                    pair = (job['model'], job['arm'])
                    if any(t['smoke'] and (t['job']['model'], t['job']['arm']) == pair for t in active.values()):
                        continue
                    if pair in validation and not validation[pair]:
                        job['outcome'] = 'validation_failed'
                        save(out / 'jobs' / (job['key'] + '.json'), job)
                        continue
                    chosen = job
                    break
                if chosen is None:
                    lock.close()
                    continue
                pending.remove(chosen)
                pair = (chosen['model'], chosen['arm'])
                smoke = pair not in validation
                command = ['taskset', '--cpu-list', lane['cpus'], protocol['python'],
                    str(source / 'scripts/loop_latency_worker.py'), '--suite', str(out), '--job', chosen['key']]
                if smoke:
                    command += ['--smoke']
                logfile = out / 'logs' / (chosen['key'] + ('-smoke' if smoke else '') + '.log')
                log = logfile.open('a')
                env = dict(env_base, CUDA_VISIBLE_DEVICES=uuid)
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                chosen.update(outcome='running', phase='validation' if smoke else 'measured',
                              gpu_uuid=uuid, cpus=lane['cpus'], pid=process.pid, started_unix=time.time())
                save(out / 'jobs' / (chosen['key'] + '.json'), chosen)
                active[uuid] = dict(process=process, job=chosen, smoke=smoke, lock=lock, log=log)
            if time.time() - last_report >= 15:
                status = aggregate(out)
                tracking.log({**{'cells/' + k: v for k, v in status['counts'].items()},
                              'requests/completed': status['completed_requests'], 'gpus/active': len(active)})
                telemetry.write(json.dumps(dict(time=time.time(), gpus=states)) + '\n')
                telemetry.flush()
                os.fsync(telemetry.fileno())
                print(json.dumps(status), flush=True)
                last_report = time.time()
            if not active and not any(read(p)['outcome'] == 'pending' for p in (out / 'jobs').glob('*.json')):
                status = aggregate(out)
                tracking.summary.update(status)
                artifact = wandb.Artifact(protocol['name'], type='latency-suite')
                for file in ['protocol.json', 'status.json', 'measurements.csv', 'inputs/manifest.json']:
                    artifact.add_file(str(out / file))
                tracking.log_artifact(artifact)
                tracking.finish()
                return
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--assets', default='/tmp/loopfill-assets')
    parser.add_argument('--python', default='/tmp/loopfill-venv/bin/python')
    parser.add_argument('--temperature', type=float, default=0.)
    parser.add_argument('--sampling-seed', type=int, default=0)
    a = parser.parse_args()
    if a.action == 'prepare':
        prepare(a.out, a.assets, a.python, a.temperature, a.sampling_seed)
    elif a.action == 'run':
        run(a.out)
    else:
        print(json.dumps(aggregate(a.out)))
