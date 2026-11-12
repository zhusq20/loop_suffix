"""Run CPU-only final-training audits after real training exit and export.

This observer does not start, stop, or modify any training/evaluation process.
Existing audits are verified and reused. An incomplete/failed audit is retained
for investigation, never silently overwritten or retried.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from audit_dflash_training import sha
from orchestrate_formal import BASE, run_command, save
from provenance import freeze_sources
import dflash_training_registry as dflash_registry

MODELS = ['ouro-2.6b', 'huginn-3.5b', 'ouro-1.4b']
PYTHON = '/tmp/loopfill-specforge-venv/bin/python'


def read(path):
    return json.loads(path.read_text())


def training_receipts(model):
    if dflash_registry.is_projected(model) and not dflash_registry.ready(model):
        return None  # The narrow queue owns its CPU audit; do not race it.
    name = dflash_registry.training_name(model)
    root, export = BASE/'checkpoints'/name, BASE/'training'/name
    finish_path = BASE/'logs'/f'{name}-finished.json'
    if not finish_path.exists():
        return None
    finish = read(finish_path)
    launch_path = BASE/'logs'/f'{name}-launcher.json'
    launch = read(launch_path)
    assert all(finish[key] == value for key, value in launch.items())
    assert finish['exit_code'] == 0, f'Training failed: {name}'
    assert finish['name'] == name and finish['finished_unix'] >= finish['started_unix']
    argv = finish['argv']
    assert argv[:2] == ['/tmp/loopfill-specforge-venv/bin/torchrun', '--standalone']
    trainer = 'train_projected_dflash.py' if dflash_registry.is_projected(model) else 'train_dflash.py'
    assert argv[3] == str(BASE/trainer)
    for flag, value in [('--model', model), ('--out', str(root)), ('--epochs', '6'),
                        ('--teachers', str(BASE/f'data/teachers/dflash-train/{model}'))]:
        assert argv.count(flag) == 1 and argv[argv.index(flag)+1] == value
    metadata, complete = read(root/'metadata.json'), read(root/'complete.json')
    assert metadata['model'] == model and metadata['epochs'] == complete['epochs'] == 6
    assert not metadata['smoke'] and not complete['smoke']
    assert complete['step'] == metadata['total_updates']
    assert argv[2] == f'--nproc_per_node={metadata["world_size"]}'
    names = ['metadata.json', 'complete.json', 'latest.json', 'metrics.jsonl',
             'progress.json', 'source-manifest.json']
    names += [f'reservation-cost-rank{rank}.jsonl' for rank in range(metadata['world_size'])]
    names += ['source/'+name for name in read(root/'source-manifest.json')]
    exported = {}
    for relative in names:
        expected = sha(root/relative)
        if not (export/relative).exists() or sha(export/relative) != expected:
            # torchrun exits before the coordinator finishes copying the export.
            assert time.time()-finish['finished_unix'] < 900, f'Incomplete training export: {export/relative}'
            return None
        exported[relative] = expected
    reservations = []
    for rank in range(metadata['world_size']):
        rows = [json.loads(line) for line in (root/f'reservation-cost-rank{rank}.jsonl').open()]
        assert len(rows) == 1 and rows[0]['status'] == 'complete'
        entry = rows[0]
        assert finish['started_unix'] <= entry['started_unix'] < entry['finished_unix'] <= finish['finished_unix']
        assert abs(entry['reserved_gpu_seconds']-(entry['finished_unix']-entry['started_unix'])) < 1e-6
        reservations.append(entry)
    assert len({r['gpu_uuid'] for r in reservations}) == metadata['world_size']
    return dict(training_name=name, actual_torchrun_exit=finish,
        launcher_sha256=sha(launch_path), finished_receipt_sha256=sha(finish_path),
        export_files_verified_against_checkpoints=exported, reservations=reservations,
        training_gpu_hours=sum(r['reserved_gpu_seconds'] for r in reservations)/3600)


def validate_audit(model, folder):
    if dflash_registry.is_projected(model):
        assert folder.resolve() == dflash_registry.audit_root(model).resolve()
        result = dflash_registry.validate_projected_completion()
        return dict(model=model, **{key: result[key] for key in [
            'audit_report', 'audit_report_sha256', 'checkpoint', 'weights_sha256']})
    report_path = folder/'report.json'
    report = read(report_path)
    assert report['status'] == 'complete_training_audit_not_benchmark'
    assert report['model'] == model and report['epochs'] == 6 and report['global_batch'] == 32
    root = BASE/f'checkpoints/dflash-{model}-formal-v1'
    assert Path(report['training_root']) == root
    checkpoint = Path(read(root/'latest.json')['checkpoint'])
    assert Path(report['checkpoint']) == checkpoint
    for key, path in [('metadata_sha256', root/'metadata.json'), ('complete_sha256', root/'complete.json'),
                      ('metrics_sha256', root/'metrics.jsonl'), ('weights_sha256', checkpoint/'model.safetensors'),
                      ('checkpoint_config_sha256', checkpoint/'config.json')]:
        assert report[key] == sha(path), key
    metadata = read(root/'metadata.json')
    assert report['metrics']['updates'] == metadata['total_updates']
    assert report['metrics']['all_logged_metrics_finite']
    assert report['learning_rate_schedule']['all_logged_learning_rates_match']
    assert report['weights']['all_weights_finite'] and not report['weights']['cuda_initialized']
    assert report['weights']['parameters'] == metadata['draft_parameters']
    assert report['teacher_sources_hash_verified']
    assert report['teacher_manifest_sha256'] == sha(BASE/'data/manifests/dflash-train.jsonl')
    manifest = read(folder/'source-manifest.json')
    assert manifest['audit_dflash_training.py'] == sha(BASE/'audit_dflash_training.py')
    for relative, expected in manifest.items():
        assert sha(folder/'source'/relative) == expected
    return dict(model=model, audit_report=str(report_path), audit_report_sha256=sha(report_path),
        checkpoint=str(checkpoint), weights_sha256=report['weights_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, default=BASE/'logs/dflash-training-audits-v2')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(models=MODELS, cpu_affinity=[80,81,82,83], cuda_visible_devices='',
            readiness={model: training_receipts(model) is not None for model in MODELS},
            scope='CPU training audit only; all model inference and training queues continue independently.')))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    os.sched_setaffinity(0, {80,81,82,83})
    out = args.out
    out.mkdir(exist_ok=True)
    with open('/tmp/loopfill-training-audit-controller.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (out/'failure.json').exists(), 'Inspect the retained failure before resuming audits'
        freeze_sources(out, [Path(__file__), BASE/'audit_dflash_training.py', BASE/'provenance.py',
            BASE/'orchestrate_formal.py', BASE/'dflash_training_registry.py', BASE/'audit_projected_dflash_training.py'])
        results = []
        try:
            for model in MODELS:
                while True:
                    receipts = training_receipts(model)
                    if receipts is not None:
                        break
                    save(out/'status.json', dict(stage='waiting_for_training_exit_and_export',
                        model=model, completed=results, updated_unix=time.time()))
                    time.sleep(15)
                folder = dflash_registry.audit_root(model)
                audit_name = f'audit-dflash-{model}-formal-training-v1'
                reused = (folder/'report.json').exists()
                if not reused:
                    assert not dflash_registry.is_projected(model), 'Narrow queue must finish its own audit before this observer proceeds'
                    assert not folder.exists(), f'Inspect incomplete audit: {folder}'
                    save(out/'status.json', dict(stage='cpu_training_audit', model=model,
                        completed=results, updated_unix=time.time()))
                    run_command(audit_name, [PYTHON, '-u', str(BASE/'audit_dflash_training.py'),
                        '--model', model, '--out', str(folder)], dict(os.environ))
                    receipts['audit_process_exit'] = read(BASE/'logs'/f'{audit_name}-finished.json')
                    assert receipts['audit_process_exit']['exit_code'] == 0
                result = validate_audit(model, folder)
                result.update(reused_existing_audit=reused, training_gpu_hours=receipts['training_gpu_hours'])
                save(out/f'{model}-verified.json', dict(result, **receipts, checked_unix=time.time()))
                results.append(result)
                print(json.dumps(result), flush=True)
            save(out/'complete.json', dict(status='all_three_formal_dflash_training_audits_complete',
                models=MODELS, results=results, finished_unix=time.time()))
            save(out/'status.json', dict(stage='complete', completed=results, updated_unix=time.time()))
        except Exception:
            save(out/'failure.json', dict(error=traceback.format_exc(), completed=results, failed_unix=time.time()))
            raise


if __name__ == '__main__':
    main()
