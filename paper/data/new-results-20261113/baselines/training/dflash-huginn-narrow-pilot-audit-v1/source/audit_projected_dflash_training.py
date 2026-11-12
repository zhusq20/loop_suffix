"""CPU audit of the narrow Huginn pilot or its explicitly separate formal run."""
import argparse
import json
from pathlib import Path
import time

from audit_dflash_training import sha, inspect_metrics, inspect_lr_schedule, inspect_weights
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def audit(scope, out):
    started = time.time()
    name = f'dflash-huginn-narrow-{scope}-v1'
    root, export = BASE/'checkpoints'/name, BASE/'training'/name
    assert not out.exists()
    identity = json.loads((root/'metadata.json').read_text())
    done = json.loads((root/'complete.json').read_text())
    epochs, count, eligible, updates = (1, 2000, 1893, 60) if scope == 'pilot' else (6, 50000, 47387, 8886)
    assert identity['model'] == 'huginn-3.5b'
    assert identity['revision'] == 'bb6621b65e90b6a4b9b29ef88dc83866d450470c'
    assert identity['epochs'] == done['epochs'] == epochs
    assert identity['examples'] == done['examples'] == eligible
    assert done['step'] == identity['total_updates'] == updates
    assert not identity['smoke'] and not done['smoke']
    assert identity['training_scope'] == ('pilot2000' if scope == 'pilot' else 'formal50000')
    assert identity['global_batch'] == identity['world_size']*identity['micro_batch']*identity['accumulation'] == 32
    assert identity['draft_architecture'] == 'target-width-projections-v1'
    assert identity['target_hidden_size'] == 5280 and identity['draft_hidden_size'] == 2048
    assert identity['draft_intermediate_size'] == 5632 and identity['draft_parameters'] == 332621056
    assert identity['target_features'] == [1, 126, 128, 130, 131]
    assert identity['mask_token_id'] == 65513

    manifest_index = json.loads((BASE/'data/manifests/manifest.json').read_text())['files']
    selection_name = 'dflash-pilot' if scope == 'pilot' else 'dflash-train'
    selection = BASE/f'data/manifests/{selection_name}.jsonl'
    selection_sha = sha(selection)
    assert selection_sha == manifest_index[selection_name]['sha256']
    selected = {json.loads(line)['source_id'] for line in selection.open()}
    assert len(selected) == count
    if scope == 'pilot':
        assert Path(identity['selection_manifest']).resolve() == selection.resolve()
        assert identity['selection_manifest_sha256'] == selection_sha
    else:
        assert identity['selection_manifest'] is None and identity['selection_manifest_sha256'] is None
    full = BASE/'data/manifests/dflash-train.jsonl'
    assert sha(full) == manifest_index['dflash-train']['sha256']
    full_ids = {json.loads(line)['source_id'] for line in full.open()}
    seen, excluded = set(), set()
    sources = identity['sources']
    assert len(sources) == 8 and {s['metadata']['shard'] for s in sources} == set(range(8))
    for source in sources:
        path = Path(source['path'])
        assert source['metadata']['model'] == identity['model']
        assert source['metadata']['manifest_sha256'] == sha(full)
        assert sha(path) == source['sha256'] == json.loads(path.with_name('complete.json').read_text())['teachers_sha256']
        for line in path.open():
            row = json.loads(line)
            source_id = row['source_id']
            assert source_id not in seen
            seen.add(source_id)
            if source_id in selected and len(row['output_tokens']) < 32:
                excluded.add(source_id)
    assert seen == full_ids and len(full_ids) == 50000 and selected <= seen
    assert excluded == set(identity['excluded_under_32_tokens']) and len(excluded) == count-eligible

    bundle = json.loads((root/'source-manifest.json').read_text())
    assert bundle['train_projected_dflash.py'] == identity['source_sha256']
    assert bundle['configs/dflash-huginn-narrow-2048-v1.json'] == identity['draft_config_sha256']
    for rel, expected in bundle.items():
        assert sha(root/'source'/rel) == sha(export/'source'/rel) == expected
    exported = {}
    for path in root.glob('*.json*'):
        assert sha(path) == sha(export/path.name), path.name
        exported[path.name] = sha(path)
    checkpoint = root/f'checkpoint-{updates:06d}'
    latest = json.loads((root/'latest.json').read_text())
    assert Path(latest['checkpoint']).resolve() == checkpoint.resolve() and latest['step'] == updates
    assert json.loads((checkpoint/'complete.json').read_text())['step'] == updates
    config = json.loads((checkpoint/'config.json').read_text())
    assert config['hidden_size'] == 2048 and config['intermediate_size'] == 5632
    assert config['num_attention_heads'] == config['num_key_value_heads'] == 16 and config['head_dim'] == 128
    assert config['num_hidden_layers'] == 5 and config['block_size'] == 16
    assert config['dflash_config']['target_hidden_size'] == 5280
    assert config['dflash_config']['projection_adapter'] == identity['draft_architecture']
    assert config['dflash_config']['target_layer_ids'] == identity['target_features']
    assert config['dflash_config']['mask_token_id'] == 65513
    metrics_path, weights_path = root/'metrics.jsonl', checkpoint/'model.safetensors'
    metrics_sha, weights_sha = sha(metrics_path), sha(weights_path)
    metric_rows = [json.loads(line) for line in metrics_path.open()]
    metrics = inspect_metrics(metric_rows, identity)
    schedule = inspect_lr_schedule(metric_rows, identity)
    weights = inspect_weights(weights_path, 332621056)
    shapes = {r['name']: r['shape'] for r in weights['tensors']}
    assert shapes['fc.weight'] == [2048, 26400]
    assert shapes['input_projection.weight'] == [2048, 5280]
    assert shapes['output_projection.weight'] == [5280, 2048]
    assert sha(metrics_path) == metrics_sha and sha(weights_path) == weights_sha
    rng = {}
    for rank in range(identity['world_size']):
        path = checkpoint/f'rng-rank{rank}.pt'
        assert path.stat().st_size > 0
        rng[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    optimizer = checkpoint/'training-state.pt'
    assert optimizer.stat().st_size > 0
    receipt_path = BASE/f'logs/{name}-finished.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['exit_code'] == 0
    assert str(root) in receipt['argv'] and str(BASE/'train_projected_dflash.py') in receipt['argv']
    costs = []
    for rank in range(identity['world_size']):
        cost_path = root/f'reservation-cost-rank{rank}.jsonl'
        rows = [json.loads(line) for line in cost_path.open()]
        assert rows and rows[-1]['status'] == 'complete'
        assert all(row['finished_unix'] >= row['started_unix'] for row in rows)
        costs.extend(rows)
    report = dict(status='complete_training_audit_not_benchmark', scope=scope, reportable=False,
        training_root=str(root), checkpoint=str(checkpoint), model=identity['model'],
        metadata_sha256=sha(root/'metadata.json'), complete_sha256=sha(root/'complete.json'),
        metrics_sha256=metrics_sha, weights_sha256=weights_sha, checkpoint_config_sha256=sha(checkpoint/'config.json'),
        selection_manifest_sha256=selection_sha, examples=eligible, excluded_under_32_tokens=len(excluded),
        epochs=epochs, global_batch=32, world_size=identity['world_size'],
        full50k_teacher_hashes_and_id_coverage_verified=True, exact_selection_exclusions_verified=True,
        frozen_training_sources_verified=bundle, exported_file_hashes=exported,
        metrics=metrics, learning_rate_schedule=schedule, weights=weights,
        rng_files=rng, optimizer_state_bytes=optimizer.stat().st_size,
        exit_receipt_sha256=sha(receipt_path), actual_exit_code=0,
        closed_gpu_reservation_hours=sum(row['reserved_gpu_seconds'] for row in costs)/3600,
        limitations=[
            'Training artifact audit only; no held-out acceptance or speed claim.',
            'Pilot2000 is not formal50k training and cannot substitute for the final model.',
            'RNG files hashed; optimizer state presence/size checked, not a resume test.',
            'All weights and logged rank0 updates finite; this is not numerical decoding equivalence.'],
        started_unix=started, finished_unix=time.time())
    out.mkdir(parents=True)
    freeze_sources(out, [Path(__file__), BASE/'audit_dflash_training.py', BASE/'provenance.py',
        BASE/'third_party/SpecForge/specforge/lr_scheduler.py', BASE/'third_party/SpecForge/specforge/optimizer.py'])
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'scope', 'examples', 'epochs', 'weights_sha256', 'closed_gpu_reservation_hours']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['pilot', 'formal'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    audit(args.scope, args.out)
