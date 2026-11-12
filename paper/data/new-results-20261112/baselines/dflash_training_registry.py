"""Explicit final DFlash identities; interrupted wide Huginn stays archived.

Choosing the narrow recipe does not declare training or evaluation successful.
Huginn evaluation is gated by the separate full-training queue and CPU audit.
"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = ['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def is_projected(model):
    assert model in MODELS
    return model == 'huginn-3.5b'


def training_name(model):
    return 'dflash-huginn-narrow-formal-v1' if is_projected(model) else f'dflash-{model}-formal-v1'


def training_root(model):
    return BASE/'checkpoints'/training_name(model)


def audit_root(model):
    return BASE/'training'/('dflash-huginn-narrow-formal-audit-v1' if is_projected(model)
                            else f'dflash-{model}-formal-audit-v1')


def evaluator(model):
    return BASE/('evaluate_projected_dflash.py' if is_projected(model) else 'evaluate_dflash.py')


def ready(model):
    if not (training_root(model)/'complete.json').exists():
        return False
    if not is_projected(model):
        return True
    return (BASE/'logs/huginn-narrow-formal-queue-v1/complete.json').exists()


def validate_projected_completion():
    """Verify real formal training, export, weights and queue/audit binding."""
    model = 'huginn-3.5b'
    assert ready(model), 'Narrow formal training and audit are not complete'
    root, audit_dir = training_root(model), audit_root(model)
    export = BASE/'training'/root.name
    identity, done = read(root/'metadata.json'), read(root/'complete.json')
    assert identity['model'] == model and identity['training_scope'] == 'full50000'
    assert identity['selection_manifest'] is None and identity['selection_manifest_sha256'] is None
    assert identity['epochs'] == done['epochs'] == 6 and not identity['smoke'] and not done['smoke']
    assert identity['global_batch'] == identity['world_size']*identity['micro_batch']*identity['accumulation'] == 32
    assert done['step'] == identity['total_updates'] == 8886
    assert done['examples'] == identity['examples'] == 47387
    assert len(identity['excluded_under_32_tokens']) == 2613
    assert identity['draft_architecture'] == 'target-width-projections-v1'
    assert identity['draft_parameters'] == 332621056
    assert (identity['target_hidden_size'], identity['draft_hidden_size'], identity['draft_intermediate_size']) == (5280, 2048, 5632)
    latest = read(root/'latest.json')
    checkpoint = root/'checkpoint-008886'
    assert Path(latest['checkpoint']).resolve() == checkpoint.resolve() and latest['step'] == 8886
    assert read(checkpoint/'complete.json')['step'] == 8886
    report_path = audit_dir/'report.json'
    report = read(report_path)
    assert report['status'] == 'complete_training_audit_not_benchmark' and report['scope'] == 'formal'
    assert report['model'] == model and report['epochs'] == 6 and report['actual_exit_code'] == 0
    assert Path(report['checkpoint']).resolve() == checkpoint.resolve()
    assert report['metrics']['updates'] == 8886 and report['metrics']['all_logged_metrics_finite']
    assert report['learning_rate_schedule']['all_logged_learning_rates_match']
    assert report['weights']['parameters'] == 332621056 and report['weights']['all_weights_finite']
    assert report['full50k_teacher_hashes_and_id_coverage_verified'] and report['exact_selection_exclusions_verified']
    for key, path in [('metadata_sha256', root/'metadata.json'), ('complete_sha256', root/'complete.json'),
                      ('metrics_sha256', root/'metrics.jsonl'), ('weights_sha256', checkpoint/'model.safetensors'),
                      ('checkpoint_config_sha256', checkpoint/'config.json')]:
        assert report[key] == sha(path), key
    finish_path = BASE/f'logs/{root.name}-finished.json'
    finish, launch = read(finish_path), read(BASE/f'logs/{root.name}-launcher.json')
    assert all(finish[k] == v for k, v in launch.items()) and finish['exit_code'] == 0
    assert report['exit_receipt_sha256'] == sha(finish_path)
    argv = finish['argv']
    assert argv[:4] == ['/tmp/loopfill-specforge-venv/bin/torchrun', '--standalone',
                       '--nproc_per_node=2', str(BASE/'train_projected_dflash.py')]
    for flag, value in [('--model', model), ('--out', str(root)), ('--epochs', '6')]:
        assert argv.count(flag) == 1 and argv[argv.index(flag)+1] == value
    for path in root.glob('*.json*'):
        assert sha(path) == sha(export/path.name)
    for rel, expected in read(root/'source-manifest.json').items():
        assert sha(root/'source'/rel) == sha(export/'source'/rel) == expected
    for rel, expected in read(audit_dir/'source-manifest.json').items():
        assert sha(audit_dir/'source'/rel) == expected
    queue_path = BASE/'logs/huginn-narrow-formal-queue-v1/complete.json'
    queue = read(queue_path)
    assert queue['audit_sha256'] == sha(report_path)
    assert Path(queue['checkpoint']).resolve() == checkpoint.resolve()
    return dict(training_name=root.name, training_root=str(root), checkpoint=str(checkpoint),
        audit_report=str(report_path), audit_report_sha256=sha(report_path),
        queue_complete_sha256=sha(queue_path), weights_sha256=report['weights_sha256'],
        completion_basis='separate_projected_training_queue_exit0_export_and_full_CPU_audit',
        interrupted_wide_training='dflash-huginn-3.5b-formal-v1 remains incomplete and separately costed')
