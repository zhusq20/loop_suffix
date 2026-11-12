"""Resume the eight E00 cells; validate completed results and never overwrite runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('results/loopfill-8gpu-20261103')
ARMS = ['ar', 'wavefront', 'suffix', 'prob-low', 'prob-load']
REVISIONS = {
    'ouro-2.6b': ('ByteDance--Ouro-2.6B', '1ed04250da1a9936042725d302e81c8fa2ab5abd'),
    'huginn-3.5b': ('tomg-group-umd--huginn-0125', 'bb6621b65e90b6a4b9b29ef88dc83866d450470c'),
}


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jobs():
    result = []
    for gpu in range(8):
        model = 'ouro-2.6b' if gpu < 4 else 'huginn-3.5b'
        subset = 'check12' if gpu % 4 < 2 else 'validation32'
        budget = 64 if subset == 'check12' else 256
        batch = 1 + gpu % 2
        repo, rev = REVISIONS[model]
        out = ROOT / f'e00-{model}-{subset}-b{batch}'
        questions = f'results/loopfill-icml-20261103/external-calibration/{subset}.jsonl'
        calibration = f'results/loopfill-icml-20261103/calibration/calib-external64-{model}.json'
        envname = 'venv' if gpu < 4 else 'huginn-venv'
        cmd = [f'/root/.cache/loopfill/{envname}/bin/python', 'scripts/loop_batch_bench.py',
               '--model', model, '--model-path', f'/root/.cache/huggingface/hub/models--{repo}/snapshots/{rev}',
               '--questions', questions, '--calibration', calibration, '--arms', ','.join(ARMS),
               '--batch-sizes', str(batch), '--order-seeds', '0', '--dtype', 'float32',
               '--max-new-tokens', str(budget), '--max-seq', '4096', '--max-nodes', '1024',
               '--check', '--out', str(out)]
        # Append the extension: with_suffix would truncate the dotted model name.
        result.append(dict(gpu=gpu, model=model, subset=subset, batch=batch, budget=budget,
                           out=str(out), log=str(out) + '.log', questions=questions,
                           calibration=calibration, command=cmd))
    assert len({j['log'] for j in result}) == len(result)
    return result


def completed(job):
    out = Path(job['out'])
    if not out.exists():
        return False
    meta = json.loads((out / 'metadata.json').read_text())
    for field, path in [('questions', job['questions']), ('calibration', job['calibration'])]:
        if meta[field + '_sha256'] != digest(path):
            raise ValueError(f'{out}: {field} changed')
    expected = {'model': job['model'], 'batch_sizes': str(job['batch']), 'order_seeds': '0',
                'max_new_tokens': job['budget'], 'dtype': 'float32', 'check': True,
                'max_seq': 4096, 'max_nodes': 1024, 'arms': ','.join(ARMS)}
    for key, value in expected.items():
        if meta['args'][key] != value:
            raise ValueError(f'{out}: incompatible {key}')
    ids = {r['question_id'] for r in map(json.loads, Path(job['questions']).read_text().splitlines())}
    if list(out.glob('*failure.json')):
        raise ValueError(f'{out}: prior failure requires explicit diagnosis and a new attempt directory')
    for arm in ARMS:
        key = f'{arm}-b{job["batch"]}-seed0'
        summary = json.loads((out / (key + '-summary.json')).read_text())
        rows = list(map(json.loads, (out / (key + '.jsonl')).read_text().splitlines()))
        if len(rows) != len(ids) or {r['request_id'] for r in rows} != ids:
            raise ValueError(f'{out}: incomplete {arm} coverage')
        if summary['same_as_serial_ar'] != len(ids) or not all(r['same_as_serial_ar'] for r in rows):
            raise ValueError(f'{out}: FP32 mismatch in {arm}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--retry-incomplete', action='store_true')
    args = parser.parse_args()
    records = jobs()
    # Validate every path before starting any workers.
    for job in records:
        try:
            job['reused'] = completed(job)
        except FileNotFoundError:
            if not args.retry_incomplete:
                raise
            old_out = Path(job['out'])
            if not (old_out / 'metadata.json').exists() or list(old_out.glob('*-summary.json')):
                raise ValueError(f'{old_out}: only interrupted cells with no arm results may retry here')
            job['interrupted_attempt'] = job['out']
            job['out'] += f'-retry-{int(time.time())}'
            job['log'] = job['out'] + '.log'
            job['command'][-1] = job['out']
            job['reused'] = False
        if job['reused']:
            job['returncode'] = 0
        elif Path(job['log']).exists():
            raise FileExistsError(job['log'])
    if args.dry_run:
        print(json.dumps([{'gpu':j['gpu'], 'log':j['log'], 'reused':j['reused']} for j in records], indent=2))
        return
    statepath = ROOT / 'continuation-status.json'
    old = json.loads(statepath.read_text())
    write_json(ROOT / f'continuation-status-before-recovery-{int(time.time())}.json', old)
    prior_workers = ROOT / 'e00-workers.json'
    if prior_workers.exists():
        write_json(ROOT / f'e00-workers-before-recovery-{int(time.time())}.json', json.loads(prior_workers.read_text()))
    processes = {}
    try:
        write_json(statepath, {'phase':'five_arm_fp32_checks', 'updated_unix':time.time(),
                              'recovery':'unique log paths; verified complete cells reused'})
        for i, job in enumerate(records):
            if job['reused']:
                continue
            log = open(job['log'], 'x')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES':str(job['gpu']), 'PYTHONPATH':'src',
                   'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'}
            proc = subprocess.Popen(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
            processes[i] = (proc, log)
            job['pid'] = proc.pid
            job['started_unix'] = time.time()
            write_json(ROOT / 'e00-workers.json', records)
        while processes:
            write_json(ROOT / 'e00-heartbeat.json', {
                'updated_unix': time.time(), 'supervisor_pid': os.getpid(),
                'running_gpus': [records[i]['gpu'] for i in processes],
                'completed_cells': sum(r.get('returncode') == 0 for r in records),
                'failed_cells': sum(r.get('returncode', 0) != 0 for r in records),
            })
            for i, (proc, log) in list(processes.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                records[i]['returncode'] = rc
                records[i]['finished_unix'] = time.time()
                del processes[i]
                write_json(ROOT / 'e00-workers.json', records)
                print(f'GPU {records[i]["gpu"]} exit {rc}', flush=True)
            time.sleep(2)
        if any(j['returncode'] != 0 for j in records):
            raise RuntimeError('E00 worker failed; formal timing remains blocked; inspect per-task logs')
        for job in records:
            completed(job)
        write_json(statepath, {'phase':'g1_checks_finished_formal_timing_not_started',
                              'updated_unix':time.time(), 'remaining':['calibration diagnostics',
                              'official baseline adapters', 'capacity correction and load probes',
                              'formal experiment matrix']})
    except Exception as exc:
        write_json(statepath, {'phase':'failed','updated_unix':time.time(),'error':str(exc)})
        raise


if __name__ == '__main__':
    main()
