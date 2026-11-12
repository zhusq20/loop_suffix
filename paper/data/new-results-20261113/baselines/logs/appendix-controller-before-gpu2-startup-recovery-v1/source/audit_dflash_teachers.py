"""Independently validate a complete50k teacher set and its training input gate."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b'], required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {80, 81, 82, 83})
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    root = BASE/'data/teachers/dflash-train'/args.model
    markers = [root/f'shard{s}/complete.json' for s in range(8)]
    while args.wait and not all(p.exists() for p in markers):
        time.sleep(30)
    assert all(p.exists() for p in markers), 'All eight complete teacher shards required'
    started = time.time()
    manifest_path = BASE/'data/manifests/dflash-train.jsonl'
    manifest_sha = digest(manifest_path)
    manifest_meta = read(BASE/'data/manifests/manifest.json')['files']['dflash-train']
    assert manifest_sha == manifest_meta['sha256'] and manifest_meta['count'] == 50000
    rows = [json.loads(line) for line in manifest_path.open()]
    manifest = {r['source_id']: r for r in rows}
    assert len(manifest) == len(rows) == 50000
    protocol = read(BASE.parents[1]/'results/loopfill-latency512-20261107/protocol.json')
    cfg = read(Path(protocol['models'][args.model]['path'])/'config.json')
    depth = 32 if args.model == 'huginn-3.5b' else 4
    slots = {'huginn-3.5b': 132, 'ouro-2.6b': 192, 'ouro-1.4b': 96}[args.model]
    first = read(root/'shard0/metadata.json')['identity']
    max_batch = first['max_batch']
    kv_gib = first.get('kv_budget_gib', 70)
    bytes_per_position = 2*slots*cfg['num_key_value_heads']*cfg['head_dim']*2
    groups = defaultdict(list)
    for row in rows:
        groups[row['prompt_lengths'][args.model]].append(row['source_id'])
    batches, batch = [], []
    for length, ids in sorted(groups.items()):
        limit = min(max_batch, max(1, kv_gib*1024**3//(bytes_per_position*(length+512))))
        if args.model == 'huginn-3.5b':
            batches.extend(ids[i:i+limit] for i in range(0, len(ids), limit))
        else:
            for key in ids:
                if len(batch) >= limit:
                    batches.append(batch)
                    batch = []
                batch.append(key)
    if batch:
        batches.append(batch)
    assignment_sha = hashlib.sha256(json.dumps(batches).encode()).hexdigest()
    all_ids, completed, costs, graph_checks = set(), [], [], []
    for shard in range(8):
        folder = root/f'shard{shard}'
        meta = read(folder/'metadata.json')
        ident = meta['identity']
        assert ident['model'] == args.model and ident['shard'] == shard and ident['shards'] == 8
        assert ident['full_depth'] == depth and ident['max_new_tokens'] == 512
        assert ident['max_batch'] == max_batch and ident.get('kv_budget_gib', 70) == kv_gib
        assert ident['manifest_sha256'] == manifest_sha and ident['batch_plan_sha256'] == assignment_sha
        assert ident['model_revision'] == first['model_revision']
        assignment = [[i, ids] for i, ids in enumerate(batches) if i % 8 == shard]
        expected = [key for _, ids in assignment for key in ids]
        assert len(expected) == len(set(expected)) == meta['planned_requests']
        assert not all_ids.intersection(expected)
        all_ids.update(expected)
        complete = read(folder/'complete.json')
        assert digest(folder/'teachers.jsonl') == complete['teachers_sha256']
        teachers = [json.loads(line) for line in (folder/'teachers.jsonl').open()]
        ids = [r['source_id'] for r in teachers]
        assert len(ids) == len(set(ids)) == len(expected) == complete['requests']
        assert set(ids) == set(expected)
        for row in teachers:
            source = manifest[row['source_id']]
            assert row['prompt_sha256'] == hashlib.sha256(source['prompt'].encode()).hexdigest()
            assert row['question_sha256'] == source['question_sha256']
            assert len(row['prompt_tokens']) == source['prompt_lengths'][args.model]
            assert 1 <= len(row['output_tokens']) <= 512
            assert all(type(t) is int and 0 <= t < cfg['vocab_size'] for t in row['prompt_tokens']+row['output_tokens'])
        assert sum(len(r['output_tokens']) for r in teachers) == complete['output_tokens']
        if (folder/'batch-plans.json').exists():
            plan = read(folder/'batch-plans.json')
            assert plan['assignment'] == assignment
            execution = [key for _, keys in plan['execution'] for key in keys]
            assert len(execution) == len(set(execution)) == len(expected) and set(execution) == set(expected)
            assert hashlib.sha256(json.dumps(plan['execution']).encode()).hexdigest() == ident['execution_batch_plan_sha256']
            graphs = [json.loads(line) for line in (folder/'graph-batches.jsonl').open()]
            assert len(graphs) == len(plan['execution']) == meta['planned_batches']
            for graph, (_, keys) in zip(graphs, plan['execution']):
                summary = graph['summary']
                assert graph['source_ids'] == keys and summary['requests'] == len(keys)
                assert summary['cuda_graph'] and summary['cuda_graph_replays'] > 0
                assert summary['core_rows'] == depth*(summary['output_tokens']-summary['requests'])
            assert sum(g['summary']['output_tokens'] for g in graphs) == complete['output_tokens']
            graph_checks.append(dict(shard=shard, batches=len(graphs), full_depth_all_batches=True))
        reservations = [json.loads(line) for line in (folder/'reservation-cost.jsonl').open()]
        assert reservations and reservations[-1]['status'] == 'complete'
        assert all(r['gpu_uuid'] == meta['gpu_uuid'] for r in reservations)
        costs.extend(reservations)
        completed.append(dict(shard=shard, **complete))
        print(json.dumps(dict(shard=shard, requests=len(teachers), status='verified')), flush=True)
    assert all_ids == set(manifest)
    from train_dflash import read_teachers
    training, sources, excluded = read_teachers(SimpleNamespace(teachers=root, model=args.model, smoke=False))
    assert len(training)+len(excluded) == 50000 and len(sources) == 8
    report = dict(status='complete', model=args.model, manifest_sha256=manifest_sha,
        stage='full_teacher_audit_not_benchmark', completed_requests=50000, completed_shards=completed,
        global_assignment_sha256=assignment_sha, all_eight_shard_assignments_disjoint_and_exhaustive=True,
        graph_checks=graph_checks, training_input_gate_passed=True, eligible_training_examples=len(training),
        excluded_under_32_tokens=len(excluded), eligible_source_counts=dict(Counter(r['pool'] for r in training)),
        total_output_tokens=sum(r['output_tokens'] for r in completed),
        actual_generation_gpu_hours=sum(r['reserved_gpu_seconds'] for r in costs)/3600,
        training_reader_sha256=digest(BASE/'train_dflash.py'), audit_source_sha256=digest(Path(__file__)),
        wall_s=time.time()-started, finished_unix=time.time())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['completed_shards','graph_checks']}), flush=True)


if __name__ == '__main__':
    main()
