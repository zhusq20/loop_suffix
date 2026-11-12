"""One-GPU teacher generation and DFlash training after existing queues drain."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_bundle(plan):
    for relative, expected in plan['source_sha256'].items():
        assert digest(HERE / relative) == expected, f'Changed frozen source: {relative}'
    for path, expected in plan['input_sha256'].items():
        assert digest(path) == expected, f'Changed input: {path}'


def dependencies(plan, gpu):
    """Do not take gaps between already queued SQL/throughput workers."""
    migration = Path(plan['throughput_queue']) / 'status.json'
    if not migration.exists() or read(migration).get('stage') != 'complete':
        return False, 'waiting_for_six_migrated_LoopAhead_cells_and_reconciliation'
    sql = Path(plan['sql_suite'])
    bindings = read(sql / 'dispatch.json')['bindings']
    streams = [key for key, value in bindings.items() if value == gpu]
    if len(streams) != 1:
        return False, 'missing_unique_SQL_stream_binding'
    model, temp = streams[0].split('/')
    for arm in read(sql / 'protocol.json')['main_arms']:
        path = sql / 'runs' / f'{model}-{arm}-{temp}' / 'summary.json'
        if not path.exists():
            return False, f'waiting_for_SQL_{model}_{arm}_{temp}'
        summary = read(path)
        if summary.get('status') != 'complete' or summary.get('completed_requests') != 100:
            return False, f'waiting_for_complete_SQL_{model}_{arm}_{temp}'
    return True, 'existing_queues_complete_on_candidate_GPU'


def inventory():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,compute_mode',
                                     '--format=csv,noheader'], text=True).splitlines()
    return {int(parts[0]): (parts[1], parts[2]) for line in lines
            if (parts := [part.strip() for part in line.split(',')])}


def busy_uuids():
    return set(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                       '--format=csv,noheader'], text=True).splitlines())


def try_lock(uuid):
    handle = Path(f'/tmp/loopfill-{uuid}.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def status(stage, **details):
    save(HERE / 'status.json', dict(stage=stage, controller_pid=os.getpid(),
                                   updated_unix=time.time(), **details))


def execute_stage(name, argv, env, lock, gpu, progress_paths):
    log_path = HERE / 'logs' / f'{name}.log'
    with log_path.open('x') as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   pass_fds=(lock.fileno(),), start_new_session=True,
                                   cwd=HERE / 'source' / 'portable')
        receipt = dict(stage=name, pid=process.pid, argv=argv, gpu=gpu,
                       started_unix=time.time(), log=str(log_path))
        save(HERE / f'{name}-launcher.json', receipt)
        while process.poll() is None:
            progress = {}
            for path in progress_paths:
                if path.exists():
                    if path.suffix == '.jsonl':
                        # Only the recent metric is needed; avoid reading a growing journal.
                        with path.open('rb') as stream:
                            stream.seek(max(0, path.stat().st_size - 65536))
                            lines = stream.read().splitlines()
                        for line in reversed(lines):
                            try:
                                progress[str(path)] = json.loads(line)
                                break
                            except (ValueError, UnicodeDecodeError):
                                continue
                    else:
                        progress[str(path)] = read(path)
            status(name, gpu=gpu, worker=receipt, progress=progress)
            time.sleep(15)
        receipt.update(exit_code=process.returncode, finished_unix=time.time())
        save(HERE / f'{name}-exit.json', receipt)
        if process.returncode:
            raise RuntimeError(f'{name} exited {process.returncode}; see {log_path}')


def prepare_smoke(plan):
    shard = HERE / 'teachers' / 'shard0'
    complete, meta = read(shard / 'complete.json'), read(shard / 'metadata.json')['identity']
    assert complete['teachers_sha256'] == digest(shard / 'teachers.jsonl')
    assert meta['model'] == 'ouro-1.4b' and meta['full_depth'] == 4
    expected = read(plan['manifest'])['files']['dflash-train']
    assert meta['manifest_sha256'] == expected['sha256']
    rows = [json.loads(line) for line in (shard / 'teachers.jsonl').open()]
    ids = [row['source_id'] for row in rows]
    assert len(ids) == len(set(ids)) == complete['requests'] == expected['count'] == 50000
    assert set(ids) == set(expected['source_ids'])
    assert all(0 < len(r['output_tokens']) <= 512 and
               len(r['prompt_tokens']) + len(r['output_tokens']) <= 3072 for r in rows)
    eligible = [row for row in rows if len(row['output_tokens']) >= 32]
    selected = sorted(eligible, key=lambda r: (-len(r['prompt_tokens']) - len(r['output_tokens']), r['source_id']))[:32]
    assert len(selected) == 32
    path = HERE / 'checks' / 'longest32-teachers.jsonl'
    with path.open('x') as stream:
        for row in selected:
            stream.write(json.dumps(row) + '\n')
    save(HERE / 'checks' / 'teacher-audit.json', dict(status='passed', requests=len(rows),
         eligible=len(eligible), excluded_under32=len(rows) - len(eligible),
         teachers_sha256=complete['teachers_sha256'], longest32_sha256=digest(path),
         smoke_source_ids=[r['source_id'] for r in selected],
         expected_updates=6 * math.ceil(len(eligible) / 32)))
    return path


def train_command(plan, out, teachers):
    return [plan['train_python'], '-u', '-m', 'baselines', 'train-dflash',
            '--model', 'ouro-1.4b', '--target-path', plan['target_path'],
            '--teachers', str(teachers), '--out', str(out), '--draft-config', plan['draft_config'],
            '--device', 'cuda:0', '--dtype', 'bfloat16', '--max-seq', '3072',
            '--epochs', '6', '--micro-batch', '4', '--accumulation', '8',
            '--num-anchors', '512', '--attention', 'flex_attention', '--save-every', '500']


def audit(plan):
    import torch
    from safetensors import safe_open
    out = HERE / 'training'
    meta, done = read(out / 'metadata.json'), read(out / 'complete.json')
    teachers = read(HERE / 'checks' / 'teacher-audit.json')
    expected = teachers['expected_updates']
    assert meta['model'] == 'ouro-1.4b' and meta['draft_parameters'] == plan['draft_parameters']
    assert meta['config']['dflash_config']['target_layer_ids'] == [73, 78, 83, 88, 93]
    assert meta['epochs'] == 6 and meta['world_size'] * meta['micro_batch'] * meta['accumulation'] == 32
    assert not meta['bounded_experiment'] and not done['bounded_experiment']
    assert meta['total_updates'] == done['step'] == done['total_updates'] == expected
    assert read(HERE / 'train-exit.json')['exit_code'] == 0
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').open()]
    assert [row['step'] for row in rows] == list(range(1, expected + 1))
    assert all(math.isfinite(row[key]) for row in rows
               for key in ['loss', 'accuracy', 'grad_norm', 'lr', 'wall_s'])
    checkpoint = Path(read(out / 'latest.json')['checkpoint'])
    assert checkpoint == out / f'checkpoint-{expected:06d}'
    assert read(checkpoint / 'complete.json')['step'] == expected
    weights = checkpoint / 'model.safetensors'
    parameters = 0
    with safe_open(weights, framework='pt', device='cpu') as tensors:
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            assert torch.isfinite(value).all(), name
            parameters += value.numel()
    assert parameters == plan['draft_parameters']
    verify_bundle(plan)
    save(HERE / 'checks' / 'training-audit.json', dict(status='passed', parameters=parameters,
         checkpoint=str(checkpoint), updates=expected, weights_sha256=digest(weights),
         metrics_sha256=digest(out / 'metrics.jsonl'), metadata_sha256=digest(out / 'metadata.json'),
         training_scope='full50000_teachers_six_epochs; not a benchmark result'))


def run(plan):
    verify_bundle(plan)
    assert not (HERE / 'failure.json').exists(), 'Inspect the preserved failure before restarting'
    assert not (HERE / 'selection.json').exists(), 'An execution already owns this run; inspect its receipts'
    with (HERE / 'controller.lock').open('a+') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = None
        while selected is None:
            reasons = {}
            policy = read(plan['gpu_policy'])
            allowed = set(policy['allowed_gpus']) - set(policy.get('reserved_for_development', []))
            cards, busy = inventory(), busy_uuids()
            for gpu in plan['gpu_preference']:
                ready, reason = dependencies(plan, gpu)
                reasons[str(gpu)] = reason
                if not ready or gpu not in allowed:
                    continue
                uuid, mode = cards[gpu]
                if mode != 'Exclusive_Process' or uuid in busy:
                    reasons[str(gpu)] = 'GPU_busy_or_not_exclusive'
                    continue
                lock = try_lock(uuid)
                if lock is None:
                    reasons[str(gpu)] = 'GPU_reserved_by_another_queue'
                    continue
                if uuid in busy_uuids():
                    lock.close()
                    continue
                selected = gpu, uuid, lock
                break
            if selected is None:
                status('waiting_for_existing_queues', candidate_gpus=plan['gpu_preference'],
                       reasons=reasons, planned_stages=['teachers', 'smoke', 'train', 'audit'])
                time.sleep(15)
        gpu, uuid, lock = selected
        with lock:
            save(HERE / 'selection.json', dict(gpu=gpu, gpu_uuid=uuid, selected_unix=time.time(),
                 policy='one GPU; all scheduled throughput cells and this SQL stream completed'))
            sys.path.insert(0, plan['legacy_source'])
            from native_cuda_env import native_cuda_environment
            env, evidence = native_cuda_environment()
            save(HERE / 'native-driver.json', evidence)
            portable = HERE / 'source' / 'portable'
            env.update(CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                       LOOPFILL_QUEUE_LOCK_FD=str(lock.fileno()),
                       PYTHONPATH=f'{portable}:{portable / "src"}')
            cpus = plan['cpu_bindings'][str(gpu)]
            prefix = ['taskset', '--cpu-list', cpus]
            teachers = HERE / 'teachers' / 'shard0'
            argv = [plan['teacher_python'], '-u', str(Path(plan['legacy_source']) / 'generate_teachers.py'),
                    '--model', 'ouro-1.4b', '--manifest', plan['teacher_manifest'], '--shard', '0',
                    '--shards', '1', '--max-batch', '32', '--kv-budget-gib', '70',
                    '--gpu', str(gpu), '--cpus', cpus, '--out', str(teachers)]
            execute_stage('teachers', prefix + argv, env, lock, gpu, [teachers / 'progress.json'])
            smoke_input = prepare_smoke(plan)
            smoke = HERE / 'checks' / 'training-smoke'
            argv = train_command(plan, smoke, smoke_input) + ['--max-updates', '1']
            execute_stage('smoke', prefix + argv, env, lock, gpu, [smoke / 'metrics.jsonl'])
            assert read(smoke / 'complete.json')['step'] == 1
            assert read(smoke / 'metadata.json')['draft_parameters'] == plan['draft_parameters']
            argv = train_command(plan, HERE / 'training', HERE / 'teachers') + [
                '--manifest', plan['manifest'], '--stage', 'dflash-train']
            execute_stage('train', prefix + argv, env, lock, gpu, [HERE / 'training' / 'metrics.jsonl'])
            cpu_env = dict(env, CUDA_VISIBLE_DEVICES='')
            execute_stage('audit', prefix + [plan['train_python'], '-u', str(HERE / 'controller.py'), '--audit'],
                          cpu_env, lock, gpu, [])
            report = read(HERE / 'checks' / 'training-audit.json')
            save(HERE / 'complete.json', dict(report, status='complete', gpu=gpu, finished_unix=time.time()))
            status('complete', gpu=gpu, checkpoint=report['checkpoint'], updates=report['updates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    plan = read(HERE / 'plan.json')
    if args.audit:
        audit(plan)
        return
    try:
        run(plan)
    except Exception:
        error = traceback.format_exc()
        save(HERE / 'failure.json', dict(error=error, failed_unix=time.time()))
        status('needs_attention', error=error)
        raise


if __name__ == '__main__':
    main()
