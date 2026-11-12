"""CPU audit of complete LAD optimization, held-out selection and saved weights.

This checks the published logging cadence, not nonexistent per-update logs.
Feature index bindings are verified here; tensor contents retain their separate
production/FeatureStore audits. Nothing is generated or evaluated on GPU.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from audit_dflash_training import sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def training_exit(model, root):
    name = root.name
    if model == 'huginn-3.5b':
        path = BASE/'logs/loopahead-train-huginn-events.jsonl'
        records = [json.loads(line) for line in path.open()]
        matches = [r for r in records if r['name'] == name and r['event'] == 'finish']
        assert len(matches) == 1
        result = matches[0]
        starts = [r for r in records if r['name'] == name and r['event'] == 'start']
        assert len(starts) == 1 and starts[0]['pid'] == result['pid'] and starts[0]['argv'] == result['argv']
    else:
        suffix = 'completed' if model == 'ouro-2.6b' else 'finished'
        path = BASE/f'logs/{name}-{suffix}.json'
        result = read(path)
        launch = read(BASE/f'logs/{name}-launcher.json')
        assert all(result[k] == v for k, v in launch.items())
    assert result['exit_code'] == 0
    argv = result['argv']
    assert str(BASE/'train_loopahead.py') in argv
    for flag, value in [('--model', model), ('--out', str(root))]:
        assert argv.count(flag) == 1 and argv[argv.index(flag)+1] == value
    return dict(path=str(path), sha256=sha(path), receipt=result)


def inspect_dev(record, pools):
    assert record['requests'] == sum(pools.values()) == 782
    assert set(record['domains']) == set(pools)
    for pool, domain in record['domains'].items():
        assert domain['windows'] == pools[pool]*8
        assert 0 <= domain['prefix_sum'] <= 4*domain['windows']
        assert domain['labels'] > 0 and domain['ce_sum'] >= 0
        assert math.isclose(domain['accepted_prefix'], domain['prefix_sum']/domain['windows'], abs_tol=1e-12)
        assert math.isclose(domain['ce'], domain['ce_sum']/domain['labels'], abs_tol=1e-12)
    assert math.isclose(record['accepted_prefix'], sum(d['accepted_prefix'] for d in record['domains'].values())/5, abs_tol=1e-12)
    assert math.isclose(record['ce'], sum(d['ce'] for d in record['domains'].values())/5, abs_tol=1e-12)


def audit(model, out):
    import torch
    from loopahead.lad import LADConfig, LookaheadDrafter, frequent_training_ids

    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    root = BASE/f'checkpoints/loopahead-{model}-formal-v1'
    export = BASE/'training'/root.name
    assert (root/'complete.json').exists(), 'Training incomplete'
    assert not out.exists(), 'Preserve previous audit evidence'
    started = time.time()
    identity, done, best = (read(root/name) for name in ['metadata.json', 'complete.json', 'best.json'])
    confirmation = read(root/'confirmation.json')
    assert identity['model'] == model and identity['epochs'] == 3 and identity['seed'] == 42
    assert identity['effective_batch'] == identity['microbatch']*identity['accumulation'] == 64
    assert identity['anchors_per_request_per_epoch'] == 32
    assert identity['eval_interval'] == 500 and identity['eval_anchors_per_request'] == 8
    assert identity['eval_future'] == 3 and identity['eval_counts_anchor']
    assert identity['betas'] == [.9, .95] and identity['weight_decay'] == .01 and identity['max_grad_norm'] == 1.
    assert identity['peak_lr'] == 2e-4 and identity['min_lr'] == 2e-5 and identity['warmup_fraction'] == .05
    bundle = read(root/'source-manifest.json')
    for rel, digest in bundle.items():
        assert sha(root/'source'/rel) == sha(export/'source'/rel) == digest
    for path in root.glob('*.json*'):
        assert sha(path) == sha(export/path.name), path.name

    manifest = read(BASE/'data/manifests/manifest.json')['files']
    counts, provenance, exclusions = Counter(), {}, {}
    for stage in ['train', 'dev', 'confirmation']:
        name = f'loopahead-{stage}'
        assert sha(BASE/f'data/manifests/{name}.jsonl') == manifest[name]['sha256']
        ids, excluded = [], []
        sources = identity['feature_sources'][stage]
        assert sources
        for source in sources:
            index_path = Path(source['path'])
            assert sha(index_path) == source['sha256'] == read(index_path.with_name('complete.json'))['index_sha256']
            index = read(index_path)
            assert index['identity']['model'] == model
            assert index['identity']['teacher_identity']['manifest_sha256'] == manifest[name]['sha256']
            assert index['frozen_components_sha256'] == identity['components_sha256']
            for chunk in index['chunks']:
                assert len(chunk['source_ids']) == len(chunk['eligible_anchors']) == chunk['requests']
                ids += chunk['source_ids']
                excluded += [sid for sid, eligible in zip(chunk['source_ids'], chunk['eligible_anchors']) if eligible == 0]
                assert all(eligible >= 0 for eligible in chunk['eligible_anchors'])
            if stage == 'train':
                counts.update({int(k): v for k, v in index['output_token_counts'].items()})
        assert len(ids) == len(set(ids)) == manifest[name]['count']
        assert set(ids) == set(manifest[name]['source_ids'])
        provenance[stage] = dict(requests=len(ids), eligible=len(ids)-len(excluded), index_sources=sources)
        exclusions[stage] = excluded
    assert set(exclusions['train']) == set(identity['excluded_no_real_64_state_context'])
    updates_epoch = math.ceil(provenance['train']['eligible']*32/64)
    total = updates_epoch*3
    assert total == identity['total_updates'] == done['step']
    metric_rows = [json.loads(line) for line in (root/'metrics.jsonl').open()]
    assert [r['step'] for r in metric_rows] == [1, *range(25, total, 25), total]
    warmup = max(1, int(.05*total))
    lr_error = 0.
    for row in metric_rows:
        assert row['epoch'] == (row['step']-1)//updates_epoch and row['total_updates'] == total
        assert all(math.isfinite(row[k]) for k in ['loss', 'lr', 'grad_norm', 'update_wall_s'])
        assert row['loss'] >= 0 and row['grad_norm'] >= 0 and row['update_wall_s'] > 0
        step = row['step']-1
        expected = 2e-4*(step+1)/warmup if step < warmup else 2e-5+.5*(2e-4-2e-5)*(1+math.cos(math.pi*(step-warmup)/max(1, total-warmup-1)))
        lr_error = max(lr_error, abs(expected-row['lr']))
        assert math.isclose(expected, row['lr'], rel_tol=1e-12, abs_tol=1e-15)
    dev = [json.loads(line) for line in (root/'dev-metrics.jsonl').open()]
    assert [r['step'] for r in dev] == [*range(500, total, 500), total]
    for row in dev:
        inspect_dev(row, manifest['loopahead-dev']['pools'])
        assert row['ineligible_requests'] == len(exclusions['dev'])
    inspect_dev(confirmation, manifest['loopahead-confirmation']['pools'])
    assert confirmation['ineligible_requests'] == len(exclusions['confirmation'])
    selected = max(dev, key=lambda r: (r['accepted_prefix'], -r['ce']))
    assert selected['step'] == best['step'] == confirmation['selected_step']
    assert {k: v for k, v in best.items() if k != 'checkpoint'} == selected
    assert done['selected_checkpoint'] == best['checkpoint']
    assert done['dev_accepted_prefix'] == selected['accepted_prefix']
    assert done['confirmation_accepted_prefix'] == confirmation['accepted_prefix']

    frequent = frequent_training_ids(counts)
    assert len(frequent) == len(set(frequent)) == 8192
    assert read(root/'frequent-token-ids.json') == frequent
    # Match all saved LAD tensor keys/shapes against the actual architecture.
    with torch.device('meta'):
        structural = LookaheadDrafter(LADConfig(**identity['config']))
    shapes = {k: tuple(t.shape) for k, t in structural.state_dict().items()}
    assert sum(p.numel() for p in structural.parameters()) == identity['parameters']
    checkpoints = []
    for row in dev:
        folder = root/f'checkpoint-{row["step"]:06d}'
        marker = read(folder/'complete.json')
        assert marker['step'] == row['step'] and marker['dev'] == {k: v for k, v in row.items() if k != 'step'}
        path = folder/'lad.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['step'] == row['step'] and payload['config'] == identity['config']
        assert payload['frequent_ids'] == frequent
        assert {k: tuple(t.shape) for k, t in payload['model'].items()} == shapes
        assert all(t.dtype == torch.bfloat16 and bool(torch.isfinite(t).all()) for t in payload['model'].values())
        assert sum(t.numel() for t in payload['model'].values()) == identity['parameters']
        checkpoints.append(dict(step=row['step'], path=str(path), sha256=sha(path), all_weights_finite=True))
        del payload
    checkpoint = Path(best['checkpoint'])/'lad.pt'
    selected_hash = next(c['sha256'] for c in checkpoints if c['step'] == best['step'])
    assert checkpoint == root/f'checkpoint-{best["step"]:06d}'/'lad.pt' and sha(checkpoint) == selected_hash
    exit_evidence = training_exit(model, root)
    reservations = [json.loads(line) for line in (root/'reservation-cost.jsonl').open()]
    assert len(reservations) == 1 and reservations[0]['status'] == 'complete'
    reservation = reservations[0]
    assert exit_evidence['receipt']['started_unix'] <= reservation['started_unix'] < reservation['finished_unix'] <= exit_evidence['receipt']['finished_unix']
    assert abs(reservation['reserved_gpu_seconds']-(reservation['finished_unix']-reservation['started_unix'])) < 1e-6
    assert not torch.cuda.is_initialized()
    report = dict(status='complete_LAD_training_audit_not_benchmark', model=model,
        training_root=str(root), selected_checkpoint=str(checkpoint), selected_step=best['step'],
        checkpoint_sha256=selected_hash, parameters=identity['parameters'],
        epochs=3, updates=total, logged_optimization_records=len(metric_rows),
        all_logged_values_finite=True, logged_learning_rate_max_error=lr_error,
        saved_checkpoints=checkpoints, all_saved_parameters_finite=True,
        selected_by_frozen_dev_rule=True, confirmation_files=1,
        dev_accepted_prefix=selected['accepted_prefix'], confirmation_accepted_prefix=confirmation['accepted_prefix'],
        frequent8192_exactly_recomputed_from_training_counts=True,
        feature_index_coverage=provenance, frozen_source_hashes=bundle,
        artifact_hashes={name: sha(root/name) for name in ['metadata.json', 'complete.json', 'best.json', 'confirmation.json', 'metrics.jsonl', 'dev-metrics.jsonl', 'frequent-token-ids.json']},
        actual_training_exit=exit_evidence, training_gpu_hours=reservation['reserved_gpu_seconds']/3600,
        cuda_initialized=False, started_unix=started, finished_unix=time.time(),
        limitations=[
            'Checks every recorded optimization row at the actual1/25/final cadence; not a full per-update log.',
            'Recomputes checkpoint selection from saved domain metrics; does not rerun GPU dev/confirmation evaluation.',
            'One confirmation artifact and trainer control flow support the once-after-selection procedure; not an external invocation count.',
            'Checks feature index hashes/IDs/counts; large feature tensors and frozen head contents retain their separate data audits.',
            'Finite trained weights and selection checks do not establish decoding correctness or benchmark speed.'])
    out.mkdir(parents=True)
    freeze_sources(out, [Path(__file__), BASE/'audit_dflash_training.py', BASE/'loopahead/lad.py', BASE/'train_loopahead.py', BASE/'provenance.py'])
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ['status', 'model', 'updates', 'selected_step', 'parameters', 'checkpoint_sha256', 'training_gpu_hours']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    audit(args.model, args.out)
