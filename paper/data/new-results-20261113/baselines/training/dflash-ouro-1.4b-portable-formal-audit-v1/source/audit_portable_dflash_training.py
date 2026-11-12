"""Independently audit the completed portable Ouro1.4B DFlash run on CPU.

The portable log has no epoch, per-update timestamp or peak-memory field. Do
not synthesize those observations to fit the older trainer's audit schema.
"""
import argparse
import inspect
import math
from pathlib import Path
import time

from audit_dflash_training import inspect_lr_schedule, inspect_weights
from external_dflash_costs import collect, EVALUATION_PLAN
from packed_dflash_evidence import read, rows, save, sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def inspect_portable_metrics(records, metadata):
    batch = metadata['world_size'] * metadata['micro_batch'] * metadata['accumulation']
    assert metadata['epochs'] == 6 and batch == 32
    assert metadata['examples'] > 0 and not metadata['bounded_experiment']
    per_epoch = math.ceil(metadata['examples'] / batch)
    expected = 6 * per_epoch
    assert metadata['total_updates'] == expected and len(records) == expected
    assert [r['step'] for r in records] == list(range(1, expected + 1))
    for r in records:
        assert r['total_updates'] == expected
        for name in ['loss', 'accuracy', 'grad_norm', 'lr', 'wall_s']:
            assert math.isfinite(r[name]), (r['step'], name)
        assert r['loss'] >= 0 and 0 <= r['accuracy'] <= 1
        assert r['grad_norm'] >= 0 and r['lr'] >= 0 and r['wall_s'] > 0
    return dict(updates=expected, updates_per_epoch_from_recipe=per_epoch,
                all_logged_updates_present_and_finite=True,
                first_loss=records[0]['loss'], final_loss=records[-1]['loss'],
                max_grad_norm=max(r['grad_norm'] for r in records),
                sum_logged_update_wall_s=sum(r['wall_s'] for r in records),
                epoch_observation='Epoch boundaries inferred from the frozen recipe and contiguous steps; not logged.',
                timing_scope='Logged update durations exclude checkpoint writes; reservation cost is separate.',
                peak_allocated_bytes=None)


def inspect_recipe(meta, plan):
    recipe = plan['train_recipe']
    expected = dict(model='ouro-1.4b', epochs=6, world_size=1, micro_batch=4,
                    accumulation=8, seed=42, num_anchors=512, loss_decay_gamma=7.,
                    lr=6e-4, warmup_ratio=.04, max_grad_norm=1., max_seq=3072,
                    attention='flex_attention', dtype='torch.bfloat16',
                    gradient_checkpointing=False, bounded_experiment=False,
                    draft_parameters=163597568, target_path=plan['target_path'])
    for name, value in expected.items():
        assert meta[name] == value, (name, meta[name], value)
    for field, name in [('epochs','epochs'), ('world_size','world_size'),
                        ('micro_batch','micro_batch'), ('accumulation','accumulation'),
                        ('seed','seed'), ('num_anchors','anchors'), ('lr','learning_rate'),
                        ('warmup_ratio','warmup_ratio'), ('max_seq','max_seq'),
                        ('loss_decay_gamma','loss_decay_gamma'), ('attention','attention')]:
        assert meta[field] == recipe[name], name
    assert recipe['global_batch'] == 32 and recipe['save_every'] == 500
    return expected


def audit(out):
    import torch

    started = time.time()
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    evaluation = read(EVALUATION_PLAN)
    root = Path(evaluation['training_root'])
    assert (root/'complete.json').exists(), 'Full training and its parent audit have not completed.'
    assert not out.exists(), 'Keep prior audit evidence; choose a fresh output directory.'
    # This checks the original frozen plan, all source/input hashes, full 50k
    # coverage, excluded IDs, actual four-stage exits and the parent audit.
    accounting = collect()
    assert accounting['status'] == 'complete' and Path(accounting['root']) == root
    plan = read(root/'plan.json')
    training = root/'training'
    meta, marker = read(training/'metadata.json'), read(training/'complete.json')
    recipe = inspect_recipe(meta, plan)
    checkpoint = Path(read(training/'latest.json')['checkpoint'])
    assert checkpoint == training/f'checkpoint-{meta["total_updates"]:06d}'
    assert read(training/'latest.json')['step'] == marker['step'] == meta['total_updates']
    assert not marker['bounded_experiment'] and marker['total_updates'] == meta['total_updates']
    config = read(plan['draft_config'])
    exported = read(checkpoint/'config.json')
    for name, value in config.items():
        if name == 'transformers_version':
            continue  # Export records the actual installed Transformers version.
        assert meta['config'][name] == value and exported[name] == value, name
    assert exported.get('dtype', exported.get('torch_dtype')) == 'bfloat16'
    assert meta['environment']['torch'] == str(torch.__version__)
    package = root/'source/portable/baselines'
    assert Path(meta['environment']['package']) == package
    observed_sources = meta['environment']['source_sha256']
    expected_sources = {str(Path(p).relative_to('source/portable/baselines')): h
                        for p, h in plan['source_sha256'].items()
                        if p.startswith('source/portable/baselines/') and p.endswith('.py')}
    assert observed_sources == expected_sources
    for name, digest in observed_sources.items():
        assert sha(package/name) == digest, name
    for name, digest in meta['environment']['executor_sha256'].items():
        assert sha(root/'source/portable/src/llm_memory_editability'/f'{name}.py') == digest, name
    assert len(meta['sources']) == 1
    source = meta['sources'][0]
    assert Path(source['path']) == root/'teachers/shard0/teachers.jsonl'
    assert sha(source['path']) == source['sha256']
    assert source['metadata'] == read(root/'teachers/shard0/metadata.json')['identity']
    argv = read(root/'train-launcher.json')['argv']
    assert not any(flag in argv for flag in ['--resume', '--max-updates', '--stop-after', '--checkpoint-layers'])
    for name, value in {'--epochs':'6', '--micro-batch':'4', '--accumulation':'8',
                        '--num-anchors':'512', '--attention':'flex_attention',
                        '--save-every':'500', '--dtype':'bfloat16', '--max-seq':'3072',
                        '--draft-config':plan['draft_config'], '--stage':'dflash-train',
                        '--manifest':plan['manifest'], '--teachers':str(root/'teachers')}.items():
        assert argv.count(name) == 1 and argv[argv.index(name)+1] == value, name
    metrics_path = training/'metrics.jsonl'
    before = sha(metrics_path)
    records = rows(metrics_path)
    metrics = inspect_portable_metrics(records, meta)
    # The reviewed, pinned portable optimizer uses AdamW's default betas and
    # weight_decay=0, FP32 masters, clip=1; step() logs after scheduler.step().
    assert sha(package/'vendor/optimizer.py') == 'fa28c4935037e729c938763bdca8a1d154df125c90a7a0fa12a2e97b75cb16e2'
    assert sha(package/'vendor/lr_scheduler.py') == sha(BASE/'third_party/SpecForge/specforge/lr_scheduler.py')
    betas = inspect.signature(torch.optim.AdamW).parameters['betas'].default
    assert betas == (.9, .999)
    schedule = inspect_lr_schedule(records, dict(torch=meta['environment']['torch'],
        lr=meta['lr'], weight_decay=0., adam_betas=betas,
        total_updates=meta['total_updates'], warmup_ratio=meta['warmup_ratio']))
    weights_path = checkpoint/'model.safetensors'
    weights_sha = sha(weights_path)
    weights = inspect_weights(weights_path, 163597568)
    assert weights_sha == accounting['details']['weights_sha256']
    assert sha(metrics_path) == before and sha(weights_path) == weights_sha
    rng = checkpoint/'rng-rank0.pt'
    optimizer = checkpoint/'training-state.pt'
    assert rng.stat().st_size > 0 and optimizer.stat().st_size > 0
    artifacts = [root/'plan.json', root/'complete.json', root/'checks/training-audit.json',
                 training/'metadata.json', training/'metrics.jsonl', training/'complete.json',
                 checkpoint/'config.json', checkpoint/'complete.json', training/'latest.json']
    artifacts += [root/f'{stage}-{kind}.json' for stage in ['teachers','smoke','train','audit']
                  for kind in ['launcher','exit']]
    report = dict(status='complete_training_audit_not_benchmark', model='ouro-1.4b',
        training_root=str(training), checkpoint=str(checkpoint), recipe=recipe,
        epochs=6, global_batch=32, world_size=1, examples=meta['examples'],
        excluded_under_32_tokens=len(meta['excluded_under_32_tokens']),
        metadata_sha256=sha(training/'metadata.json'), complete_sha256=sha(training/'complete.json'),
        metrics_sha256=before, weights_sha256=weights_sha,
        checkpoint_config_sha256=sha(checkpoint/'config.json'),
        artifact_hashes={str(p):sha(p) for p in artifacts},
        metrics=metrics, learning_rate_schedule=schedule, weights=weights,
        frozen_training_sources_verified=observed_sources,
        optimizer_defaults=dict(weight_decay=0., adam_betas=betas,
            portable_source_sha256=sha(package/'vendor/optimizer.py')),
        rng_files={rng.name:dict(bytes=rng.stat().st_size, sha256=sha(rng))},
        optimizer_state_bytes=optimizer.stat().st_size,
        limitations=['Training evidence only; not benchmark accuracy, equivalence or speed.',
                     'No epoch timestamps or peak GPU-memory measurement were logged.',
                     'Optimizer presence/size and RNG hash only; no deserialization or resume test.',
                     'Final weight inspection does not inspect every intermediate checkpoint.'],
        started_unix=started, finished_unix=time.time(), cuda_initialized=torch.cuda.is_initialized())
    assert not report['cuda_initialized']
    out.mkdir(parents=True)
    freeze_sources(out, [Path(__file__), BASE/'audit_dflash_training.py',
        BASE/'external_dflash_costs.py', BASE/'packed_dflash_evidence.py', BASE/'provenance.py',
        BASE/'third_party/SpecForge/specforge/lr_scheduler.py', BASE/'third_party/SpecForge/specforge/optimizer.py'])
    save(out/'report.json', report)
    print(dict(status=report['status'], updates=meta['total_updates'], parameters=163597568))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    audit(parser.parse_args().out)
