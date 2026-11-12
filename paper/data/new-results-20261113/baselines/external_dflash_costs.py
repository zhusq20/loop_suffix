"""Account closed intervals of the existing portable Ouro1.4B training.

The parent retains one GPU lease across teachers, smoke, training and CPU audit.
Its child exit receipts provide stage intervals; selection and stage boundaries
provide disjoint held-GPU gaps. Nested teacher reservations are evidence only.
"""
from collections import Counter
import math
from pathlib import Path

from packed_dflash_evidence import read, sha

BASE = Path(__file__).resolve().parent
EVALUATION_PLAN = BASE/'logs/ouro14-packed-evaluation-v1/plan.json'
STAGES = ('teachers', 'smoke', 'train', 'audit')
CATEGORIES = dict(teachers='generation', smoke='training_smoke', train='training', audit='held_gpu_cpu_audit')


def intervals(selection, launches, receipts, finished=None):
    """Partition observed closed intervals; never extrapolate an active stage."""
    result = []
    previous = selection['selected_unix']
    assert math.isfinite(previous)
    for index, stage in enumerate(STAGES):
        if stage not in launches:
            assert not any(s in launches or s in receipts for s in STAGES[index:])
            break
        launch = launches[stage]
        start = launch['started_unix']
        assert math.isfinite(start) and start >= previous
        assert launch['stage'] == stage and launch['gpu'] == selection['gpu']
        assert type(launch['pid']) is int and launch['pid'] > 0
        result.append(dict(stage='before_'+stage, category='held_gpu_between_stages',
            started_unix=previous, finished_unix=start, status='complete'))
        if stage not in receipts:
            assert finished is None and not any(s in launches or s in receipts for s in STAGES[index+1:])
            break
        receipt = receipts[stage]
        assert all(receipt[k] == v for k, v in launch.items()), stage
        end = receipt['finished_unix']
        assert math.isfinite(end) and end >= start
        assert type(receipt['exit_code']) is int
        result.append(dict(stage=stage, category=CATEGORIES[stage], started_unix=start,
            finished_unix=end, status='complete' if receipt['exit_code'] == 0 else 'failed'))
        previous = end
        if receipt['exit_code']:
            assert finished is None and not any(s in launches or s in receipts for s in STAGES[index+1:])
            break
    if finished is not None:
        assert set(receipts) == set(STAGES) and all(r['exit_code'] == 0 for r in receipts.values())
        assert math.isfinite(finished) and finished >= previous
        result.append(dict(stage='after_audit', category='held_gpu_between_stages',
            started_unix=previous, finished_unix=finished, status='complete'))
    for r in result:
        r['reserved_gpu_seconds'] = r['finished_unix'] - r['started_unix']
    assert all(a['finished_unix'] <= b['started_unix'] for a, b in zip(result, result[1:]))
    if finished is not None:
        assert abs(sum(r['reserved_gpu_seconds'] for r in result) - (finished-selection['selected_unix'])) < 1e-6
    return result


def collect():
    evaluation = read(EVALUATION_PLAN)
    root = Path(evaluation['training_root'])
    plan = read(root/'plan.json')
    assert sha(root/'plan.json') == evaluation['training_plan_sha256']
    assert plan['model'] == 'ouro-1.4b' and plan['draft_parameters'] == 163597568
    for relative, expected in plan['source_sha256'].items():
        assert sha(root/relative) == expected, relative
    for name in ['teacher_manifest', 'manifest', 'draft_config']:
        path = Path(plan[name])
        expected = plan['input_sha256'].get(str(path), plan['source_sha256'].get(str(path.relative_to(root))))
        assert expected and sha(path) == expected, name
    manifest = read(BASE/'data/manifests/manifest.json')['files']['dflash-train']
    assert read(plan['manifest'])['files']['dflash-train'] == manifest
    assert sha(plan['teacher_manifest']) == manifest['sha256'] and manifest['count'] == 50000
    selected = read(root/'selection.json')
    done = read(root/'complete.json') if (root/'complete.json').exists() else None
    launches, receipts = {}, {}
    for stage in STAGES:
        launch_path, exit_path = root/f'{stage}-launcher.json', root/f'{stage}-exit.json'
        if launch_path.exists():
            launches[stage] = read(launch_path)
            argv = launches[stage]['argv']
            assert launches[stage]['gpu'] == selected['gpu']
            if stage == 'audit':
                assert argv[-2:] == [str(root/'controller.py'), '--audit']
            else:
                output = root/('teachers/shard0' if stage == 'teachers' else 'checks/training-smoke' if stage == 'smoke' else 'training')
                assert argv.count('--out') == 1 and Path(argv[argv.index('--out')+1]) == output
                assert argv[argv.index('--model')+1] == 'ouro-1.4b'
        if exit_path.exists():
            assert stage in launches
            receipts[stage] = read(exit_path)
    partition = intervals(selected, launches, receipts, done['finished_unix'] if done else None)
    records = []
    for interval in partition:
        stage = interval['stage']
        if stage in STAGES:
            path = root/f'{stage}-exit.json'
            evidence = [path, root/f'{stage}-launcher.json', root/'selection.json']
        elif stage.startswith('before_'):
            target = stage[len('before_'):]
            path = root/f'{target}-launcher.json'
            index = STAGES.index(target)
            evidence = [path, root/f'{STAGES[index-1]}-exit.json' if index else root/'selection.json']
        else:
            path = root/'complete.json'
            evidence = [path, root/'audit-exit.json']
        records.append(dict(interval, path=str(path), line=1, gpu_uuid=selected['gpu_uuid'],
            scope='portable_ouro14_dflash', interval_id=stage,
            evidence={str(p): sha(p) for p in evidence}))

    completed_teachers = 0
    excluded, source_counts = [], Counter()
    if receipts.get('teachers', {}).get('exit_code') == 0:
        teacher = root/'teachers/shard0'
        marker, metadata = read(teacher/'complete.json'), read(teacher/'metadata.json')['identity']
        assert metadata['model'] == 'ouro-1.4b' and metadata['full_depth'] == 4
        assert metadata['manifest_sha256'] == manifest['sha256']
        assert marker['teachers_sha256'] == sha(teacher/'teachers.jsonl')
        source_pools = {r['source_id']: r['pool'] for r in json_rows(Path(plan['teacher_manifest']))}
        seen = set()
        for row in json_rows(teacher/'teachers.jsonl'):
            identity = row['source_id']
            assert identity not in seen and identity in source_pools
            seen.add(identity)
            assert 0 < len(row['output_tokens']) <= 512
            assert len(row['prompt_tokens']) + len(row['output_tokens']) <= 3072
            if len(row['output_tokens']) < 32:
                excluded.append(identity)
            else:
                source_counts[source_pools[identity]] += 1
        assert len(seen) == marker['requests'] == manifest['count'] == 50000
        assert seen == set(manifest['source_ids'])
        completed_teachers = len(seen)
        # Do not append this inner reservation: the parent teacher interval
        # already includes it, including process startup and exit observation.
        inner = list(json_rows(teacher/'reservation-cost.jsonl'))
        assert len(inner) == 1 and inner[0]['status'] == 'complete'
        item = inner[0]
        assert item['gpu_uuid'] == selected['gpu_uuid']
        assert launches['teachers']['started_unix'] <= item['started_unix'] <= item['finished_unix'] <= receipts['teachers']['finished_unix']

    details = {}
    if done:
        assert done['status'] == 'complete' and completed_teachers == 50000
        assert set(receipts) == set(STAGES) and all(r['exit_code'] == 0 for r in receipts.values())
        audit = read(root/'checks/training-audit.json')
        meta, marker = read(root/'training/metadata.json'), read(root/'training/complete.json')
        checkpoint = Path(audit['checkpoint'])
        assert audit['status'] == 'passed' and audit['parameters'] == meta['draft_parameters'] == 163597568
        assert all(done[k] == v for k, v in audit.items() if k != 'status')
        assert meta['model'] == 'ouro-1.4b' and meta['epochs'] == 6 and meta['world_size'] == 1
        assert not meta['bounded_experiment'] and not marker['bounded_experiment']
        assert meta['world_size'] * meta['micro_batch'] * meta['accumulation'] == 32
        assert set(meta['excluded_under_32_tokens']) == set(excluded)
        assert len(meta['excluded_under_32_tokens']) == len(excluded)
        assert meta['examples'] == sum(source_counts.values()) == 50000-len(excluded)
        steps = 6 * math.ceil(meta['examples']/32)
        assert steps == marker['step'] == meta['total_updates'] == audit['updates']
        assert checkpoint == root/'training'/f'checkpoint-{steps:06d}'
        assert read(checkpoint/'complete.json')['step'] == steps
        for filename, field in [('metadata.json', 'metadata_sha256'), ('metrics.jsonl', 'metrics_sha256')]:
            assert sha(root/'training'/filename) == audit[field]
        assert sha(checkpoint/'model.safetensors') == audit['weights_sha256']
        # The actual successful CPU audit checked every weight for finiteness
        # and every update. Bind that audit to the exact archived source/inputs.
        for path, expected in plan['input_sha256'].items():
            assert sha(path) == expected, path
        details = dict(steps=steps, epochs=6, eligible=meta['examples'],
            eligible_source_counts=dict(sorted(source_counts.items())), gpus=1,
            selection=f'{checkpoint.name};163597568 parameters', rank0_peak_allocated_bytes=None,
            peak_memory_limitation='Portable trainer did not record peak allocated GPU memory.',
            metadata_sha256=sha(root/'training/metadata.json'), complete_sha256=sha(root/'training/complete.json'),
            audit_sha256=sha(root/'checks/training-audit.json'), weights_sha256=audit['weights_sha256'])
    return dict(root=str(root), status='complete' if done else 'pending',
        counts={'dflash-train': dict(completed=completed_teachers, expected=50000)}, details=details,
        reservations=records, training_plan_sha256=sha(root/'plan.json'),
        coverage='Closed parent child-stage intervals plus observed gaps while the same GPU lease is held. '
                 'Teacher nested reservations are not added again. CPU audit and one-update smoke are separate extras. '
                 'Running-stage durations are not estimated. Final bound is the completion-marker timestamp; '
                 'the unrecorded final marker-to-lock-release interval is excluded.',
        total_recorded_gpu_hours=sum(r['reserved_gpu_seconds'] for r in records)/3600)


def json_rows(path):
    import json
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)
