"""Observe completed LAD training and verify three models on CPU only.

No training/evaluation process is started, stopped or changed by this observer.
Wait for the actual process exit and complete export, not just complete.json.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from audit_loopahead_training import BASE, read, sha, training_exit
from orchestrate_formal import run_command, save
from provenance import freeze_sources

MODELS = ['ouro-2.6b', 'huginn-3.5b', 'ouro-1.4b']
PYTHON = '/tmp/loopfill-venv/bin/python'


def training_ready(model):
    root = BASE/f'checkpoints/loopahead-{model}-formal-v1'
    if not (root/'complete.json').exists():
        return False
    if model != 'huginn-3.5b':
        suffix = 'completed' if model == 'ouro-2.6b' else 'finished'
        if not (BASE/f'logs/{root.name}-{suffix}.json').exists():
            return False
    else:
        records = [json.loads(line) for line in (BASE/'logs/loopahead-train-huginn-events.jsonl').open()]
        if not any(r['name'] == root.name and r['event'] == 'finish' for r in records):
            return False
    exit_record = training_exit(model, root)['receipt']
    export = BASE/'training'/root.name
    paths = [p.relative_to(root) for p in root.glob('*.json*')]
    paths += [Path('source')/name for name in read(root/'source-manifest.json')]
    for path in paths:
        if not (export/path).exists() or sha(export/path) != sha(root/path):
            assert time.time()-exit_record['finished_unix'] < 900, f'Incomplete LAD export: {export/path}'
            return False
    return True


def validate_audit(model):
    folder = BASE/f'training/loopahead-{model}-formal-audit-v2'
    path = folder/'report.json'
    report = read(path)
    root = BASE/f'checkpoints/loopahead-{model}-formal-v1'
    assert report['status'] == 'complete_LAD_training_audit_not_benchmark'
    assert report['model'] == model and Path(report['training_root']) == root
    assert report['epochs'] == 3 and not report['cuda_initialized']
    assert all(report[k] for k in ['approved_LAD_config_verified', 'all_logged_values_finite',
        'all_saved_parameters_finite', 'selected_by_frozen_dev_rule',
        'frequent8192_exactly_recomputed_from_training_counts'])
    for name, expected in report['artifact_hashes'].items():
        assert sha(root/name) == expected
    meta, best, complete = (read(root/name) for name in ['metadata.json', 'best.json', 'complete.json'])
    assert report['updates'] == meta['total_updates'] == complete['step']
    assert report['parameters'] == meta['parameters']
    assert report['selected_step'] == best['step']
    checkpoint = Path(best['checkpoint'])/'lad.pt'
    assert Path(report['selected_checkpoint']) == checkpoint
    assert sha(checkpoint) == report['checkpoint_sha256']
    assert training_exit(model, root)['receipt'] == report['actual_training_exit']['receipt']
    manifest = read(folder/'source-manifest.json')
    assert manifest['audit_loopahead_training.py'] == sha(BASE/'audit_loopahead_training.py')
    for relative, expected in manifest.items():
        assert sha(folder/'source'/relative) == expected
    name = f'audit-loopahead-{model}-formal-v2'
    launch, finish = (read(BASE/f'logs/{name}-{suffix}.json') for suffix in ['launcher', 'finished'])
    assert all(finish[k] == v for k, v in launch.items()) and finish['exit_code'] == 0
    assert finish['argv'] == [PYTHON, '-u', str(BASE/'audit_loopahead_training.py'),
        '--model', model, '--out', str(folder)]
    return dict(model=model, audit_report=str(path), audit_report_sha256=sha(path),
        selected_checkpoint=str(checkpoint), checkpoint_sha256=report['checkpoint_sha256'],
        actual_audit_exit=finish, training_gpu_hours=report['training_gpu_hours'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, default=BASE/'logs/loopahead-training-audits-v1')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.sched_setaffinity(0, {80,81,82,83})
    if not args.execute:
        print(json.dumps(dict(models=MODELS, ready={m: training_ready(m) for m in MODELS},
            existing_audits=[validate_audit(m) for m in MODELS
                if (BASE/f'training/loopahead-{m}-formal-audit-v2/report.json').exists()],
            cpu_affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices='')))
        return
    out = args.out
    out.mkdir(exist_ok=True)
    with open('/tmp/loopfill-loopahead-training-audit-controller.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out/'failure.json').exists(), 'Inspect preserved failure before resuming'
        freeze_sources(out, [Path(__file__), BASE/'audit_loopahead_training.py',
            BASE/'audit_dflash_training.py', BASE/'loopahead/lad.py', BASE/'provenance.py',
            BASE/'orchestrate_formal.py'])
        results = []
        try:
            for model in MODELS:
                while not training_ready(model):
                    save(out/'status.json', dict(stage='waiting_for_training_exit_and_export',
                        model=model, completed=results, updated_unix=time.time()))
                    time.sleep(15)
                folder = BASE/f'training/loopahead-{model}-formal-audit-v2'
                reused = (folder/'report.json').exists()
                if not reused:
                    assert not folder.exists(), f'Inspect incomplete audit: {folder}'
                    save(out/'status.json', dict(stage='cpu_training_audit', model=model,
                        completed=results, updated_unix=time.time()))
                    run_command(f'audit-loopahead-{model}-formal-v2', [PYTHON, '-u',
                        str(BASE/'audit_loopahead_training.py'), '--model', model,
                        '--out', str(folder)], dict(os.environ))
                result = dict(validate_audit(model), reused_existing_audit=reused)
                save(out/f'{model}-verified.json', dict(result, checked_unix=time.time()))
                results.append(result)
                print(json.dumps(result), flush=True)
            save(out/'complete.json', dict(status='all_three_formal_LAD_training_audits_complete',
                models=MODELS, results=results, finished_unix=time.time()))
            save(out/'status.json', dict(stage='complete', completed=results, updated_unix=time.time()))
        except Exception:
            save(out/'failure.json', dict(error=traceback.format_exc(), completed=results, failed_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
