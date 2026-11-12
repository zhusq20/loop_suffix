"""Independently inspect a completed formal DFlash training run on CPU."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time

from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_metrics(rows, identity):
    expected = identity['total_updates']
    per_epoch = math.ceil(identity['examples'] / identity['global_batch'])
    assert expected == per_epoch * identity['epochs']
    assert len(rows) == expected
    assert [row['step'] for row in rows] == list(range(1, expected + 1))
    epochs = Counter()
    for row in rows:
        assert row['total_updates'] == expected
        assert row['epoch'] == (row['step'] - 1) // per_epoch
        epochs[row['epoch']] += 1
        for key in ['loss', 'accuracy', 'lr', 'grad_norm', 'update_wall_s', 'finished_unix']:
            assert math.isfinite(row[key]), (row['step'], key)
        assert row['loss'] >= 0 and 0 <= row['accuracy'] <= 1
        assert row['lr'] >= 0
        assert row['grad_norm'] >= 0 and row['update_wall_s'] > 0
        assert row['peak_allocated_bytes'] > 0
    assert dict(epochs) == {epoch: per_epoch for epoch in range(identity['epochs'])}
    return dict(updates=expected, updates_per_epoch=per_epoch,
                epoch_update_counts=dict(epochs), all_logged_metrics_finite=True,
                first_loss=rows[0]['loss'], final_loss=rows[-1]['loss'],
                max_grad_norm=max(row['grad_norm'] for row in rows),
                rank0_peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows),
                sum_optimizer_update_wall_s=sum(row['update_wall_s'] for row in rows),
                timing_scope='Logged optimizer updates only; full GPU reservation cost is reported separately.')


def inspect_lr_schedule(rows, identity):
    import torch

    assert str(torch.__version__) == identity['torch'], 'Use the original training environment.'
    repo = BASE / 'third_party/SpecForge'
    pin = 'ea5a5f5902b3493b3b6045f249f5aa57cbd100a1'
    dependency_hashes = {}
    for name in ['specforge/lr_scheduler.py', 'specforge/optimizer.py']:
        original = subprocess.check_output(['git', '-C', str(repo), 'show', f'{pin}:{name}'])
        dependency_hashes[name] = hashlib.sha256(original).hexdigest()
        assert sha(repo / name) == dependency_hashes[name], name
    path = repo / 'specforge/lr_scheduler.py'
    spec = importlib.util.spec_from_file_location('audited_specforge_lr_scheduler', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameter = torch.nn.Parameter(torch.zeros(1, device='cpu'))
    optimizer = torch.optim.AdamW([parameter], lr=identity['lr'],
                                 weight_decay=identity['weight_decay'], betas=tuple(identity['adam_betas']))
    scheduler = module.CosineAnnealingWarmupLR(
        optimizer, total_steps=identity['total_updates'],
        warmup_steps=int(identity['warmup_ratio'] * identity['total_updates']))
    max_error = 0.
    for row in rows:
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        expected = optimizer.param_groups[0]['lr']
        max_error = max(max_error, abs(expected - row['lr']))
        assert math.isclose(expected, row['lr'], rel_tol=1e-11, abs_tol=1e-15), (row['step'], expected, row['lr'])
    assert not torch.cuda.is_initialized()
    return dict(all_logged_learning_rates_match=True, max_absolute_error=max_error,
                scheduler_source_sha256=sha(path), torch=str(torch.__version__),
                specforge_commit=pin, pinned_dependency_hashes=dependency_hashes,
                logged_max_lr=max(row['lr'] for row in rows), nominal_lr=identity['lr'],
                note='Replay the pinned official scheduler, including its warmup-to-cosine boundary behavior.')


def inspect_weights(path, expected_parameters):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    tensors = []
    with safe_open(str(path), framework='pt', device='cpu') as reader:
        for name in sorted(reader.keys()):
            tensor = reader.get_tensor(name)
            assert tensor.dtype == torch.bfloat16, (name, tensor.dtype)
            assert tensor.numel() > 0 and bool(torch.isfinite(tensor).all()), name
            tensors.append(dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                                parameters=tensor.numel(), max_abs=float(tensor.abs().max())))
            del tensor
    total = sum(row['parameters'] for row in tensors)
    assert total == expected_parameters, (total, expected_parameters)
    assert not torch.cuda.is_initialized()
    return dict(parameters=total, tensor_count=len(tensors), all_weights_finite=True,
                cuda_initialized=False, tensors=tensors)


def audit(model, out):
    started = time.time()
    root = BASE / 'checkpoints' / f'dflash-{model}-formal-v1'
    assert (root / 'complete.json').exists(), 'Formal training is not complete.'
    assert not out.exists(), 'Keep existing audit reports; use a fresh output directory.'
    identity = json.loads((root / 'metadata.json').read_text())
    complete = json.loads((root / 'complete.json').read_text())
    assert identity['model'] == model and identity['epochs'] == 6 and not identity['smoke']
    assert identity['global_batch'] == 32
    assert identity['world_size'] * identity['micro_batch'] * identity['accumulation'] == 32
    assert complete['epochs'] == 6 and not complete['smoke']
    assert complete['examples'] == identity['examples']
    assert complete['step'] == identity['total_updates']
    manifest_path = BASE / 'data/manifests/dflash-train.jsonl'
    manifest_index = BASE / 'data/manifests/manifest.json'
    manifest = json.loads(manifest_index.read_text())['files']['dflash-train']
    assert manifest['count'] == 50000 and sha(manifest_path) == manifest['sha256']
    assert identity['examples'] + len(identity['excluded_under_32_tokens']) == 50000
    sources = identity['sources']
    assert len(sources) == 8
    assert {source['metadata']['shard'] for source in sources} == set(range(8))
    for source in sources:
        assert source['metadata']['shards'] == 8
        assert source['metadata']['model'] == model
        assert source['metadata']['manifest_sha256'] == manifest['sha256']
        assert sha(source['path']) == source['sha256']

    bundle = json.loads((root / 'source-manifest.json').read_text())
    assert bundle['train_dflash.py'] == identity['source_sha256']
    for name, expected in bundle.items():
        assert sha(root / 'source' / name) == expected, name
    latest = json.loads((root / 'latest.json').read_text())
    checkpoint = root / f'checkpoint-{complete["step"]:06d}'
    assert Path(latest['checkpoint']).resolve() == checkpoint.resolve()
    assert latest['step'] == complete['step']
    assert json.loads((checkpoint / 'complete.json').read_text())['step'] == complete['step']
    config = json.loads((checkpoint / 'config.json').read_text())
    assert config['num_hidden_layers'] == 5 and config['block_size'] == 16
    assert config['dflash_config']['mask_token_id'] == identity['mask_token_id']
    assert config['dflash_config']['target_layer_ids'] == identity['target_features']
    assert config.get('dtype', config.get('torch_dtype')) == 'bfloat16'
    metrics_path = root / 'metrics.jsonl'
    metrics_sha = sha(metrics_path)
    with metrics_path.open() as stream:
        metric_rows = [json.loads(line) for line in stream]
    metrics = inspect_metrics(metric_rows, identity)
    schedule = inspect_lr_schedule(metric_rows, identity)
    weights_path = checkpoint / 'model.safetensors'
    weights_sha = sha(weights_path)
    weights = inspect_weights(weights_path, identity['draft_parameters'])
    assert sha(weights_path) == weights_sha and sha(metrics_path) == metrics_sha
    rng_files = {}
    for rank in range(identity['world_size']):
        path = checkpoint / f'rng-rank{rank}.pt'
        assert path.stat().st_size > 0
        rng_files[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    optimizer = checkpoint / 'training-state.pt'
    assert optimizer.stat().st_size > 0
    report = dict(
        status='complete_training_audit_not_benchmark', model=model,
        training_root=str(root.resolve()), checkpoint=str(checkpoint.resolve()),
        metadata_sha256=sha(root / 'metadata.json'), complete_sha256=sha(root / 'complete.json'),
        metrics_sha256=metrics_sha, weights_sha256=weights_sha,
        checkpoint_config_sha256=sha(checkpoint / 'config.json'),
        teacher_manifest_sha256=manifest['sha256'], teacher_sources_hash_verified=True,
        examples=identity['examples'], excluded_under_32_tokens=len(identity['excluded_under_32_tokens']),
        epochs=6, global_batch=32, world_size=identity['world_size'],
        frozen_training_sources_verified=bundle, metrics=metrics, learning_rate_schedule=schedule, weights=weights,
        rng_files=rng_files, optimizer_state_bytes=optimizer.stat().st_size,
        limitations=[
            'Final weights and all rank0 logged updates inspected; this is not benchmark accuracy or speed.',
            'Teacher-source hashes are checked; semantic data coverage is established by the separate teacher audit.',
            'Optimizer state checked for presence/size and RNG files for hashes; no optimizer deserialization or resume test.',
            'Job exit receipts and total GPU reservation costs are checked separately by orchestration and cost reporting.'
        ], started_unix=started, finished_unix=time.time())
    out.mkdir(parents=True)
    freeze_sources(out, [Path(__file__), BASE / 'provenance.py',
                         BASE / 'third_party/SpecForge/specforge/lr_scheduler.py',
                         BASE / 'third_party/SpecForge/specforge/optimizer.py'])
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['status', 'model', 'checkpoint', 'epochs', 'global_batch']}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['ouro-2.6b', 'huginn-3.5b', 'ouro-1.4b'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.model, args.out)


if __name__ == '__main__':
    main()
