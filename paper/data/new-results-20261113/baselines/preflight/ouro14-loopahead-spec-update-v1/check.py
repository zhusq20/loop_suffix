"""Bind the appendix specification update to completed training and probe artifacts."""
from pathlib import Path
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, rows, save, sha


def finished(name):
    launch, done = [BASE/f'logs/{name}-{suffix}.json' for suffix in ['launcher', 'finished']]
    a, z = read(launch), read(done)
    assert z['exit_code'] == 0 and all(z[k] == v for k, v in a.items())
    return dict(path=str(done), sha256=sha(done), exit_code=0)


audit_path = BASE/'training/loopahead-ouro-1.4b-formal-audit-v1/report.json'
audit = read(audit_path)
training = Path(audit['training_root'])
assert audit['status'] == 'complete_LAD_training_audit_not_benchmark'
for name, expected in audit['artifact_hashes'].items():
    assert sha(training/name) == expected, name
for item in audit['feature_index_coverage'].values():
    for index in item['index_sources']:
        assert sha(index['path']) == index['sha256']
checkpoint = Path(audit['selected_checkpoint'])
assert sha(checkpoint) == audit['checkpoint_sha256']
assert audit['epochs'] == 3 and audit['updates'] == audit['selected_step'] == 13533
assert audit['parameters'] == 11051008 and audit['logged_optimization_records'] == 543
assert audit['all_saved_parameters_finite'] and len(audit['saved_checkpoints']) == 28
assert audit['selected_by_frozen_dev_rule'] and audit['confirmation_files'] == 1
assert audit['feature_index_coverage']['train']['requests'] == 9025
assert audit['feature_index_coverage']['train']['eligible'] == 9022
assert len(read(training/'metadata.json')['excluded_no_real_64_state_context']) == 3
train_exit = finished('loopahead-ouro-1.4b-formal-v1')
assert train_exit['sha256'] == audit['actual_training_exit']['sha256']
audit_exit = finished('audit-loopahead-ouro14-resume-v1')
probes, exits = {}, {}
for dtype in ['bfloat16', 'float32']:
    for method in ['ar', 'loopahead']:
        name = f'loopahead-trained-ouro-1.4b-gsm8k-{method}-{dtype}-n20-v1'
        folder = BASE/'preflight'/name
        data, meta, done = rows(folder/'requests.jsonl'), read(folder/'metadata.json'), read(folder/'complete.json')
        assert len(data) == done['requests'] == 20 and sha(folder/'requests.jsonl') == done['requests_sha256']
        assert meta['model'] == 'ouro-1.4b' and meta['dtype'] == dtype and meta['temperature'] == 0
        if method == 'loopahead':
            assert meta['checkpoint_sha256'] == audit['checkpoint_sha256']
        probes[dtype,method] = data
        exits[name] = finished(name)
matches, divergences = {}, []
for dtype in ['bfloat16', 'float32']:
    a, z = probes[dtype,'ar'], probes[dtype,'loopahead']
    assert [r['source_id'] for r in a] == [r['source_id'] for r in z]
    matches[dtype] = sum(x['tokens'] == y['tokens'] for x, y in zip(a, z, strict=True))
    if dtype == 'bfloat16':
        for x, y in zip(a, z, strict=True):
            if x['tokens'] == y['tokens']:
                continue
            position = next(i for i,(u,v) in enumerate(zip(x['tokens'], y['tokens'])) if u != v)
            divergences.append((x['source_id'], position))
assert matches == {'bfloat16':4, 'float32':20}
gate = read(BASE/'preflight/loopahead-trained-ouro-1.4b-gate.json')
diag_path = Path(gate['common_prefix_diagnosis'])
diag = read(diag_path)
assert sha(diag_path) == gate['common_prefix_diagnosis_sha256']
assert diag['checkpoint_sha256'] == audit['checkpoint_sha256']
assert diag['status'] == 'complete' and diag['cases'] == 16 and diag['all_fp32_same_top1']
assert all(r['same_top1'] for r in diag['results'])
assert sorted((r['source_id'],r['position']) for r in diag['results']) == sorted(divergences)
for source in diag['source_journals'].values():
    assert sha(Path(source['path'])/'requests.jsonl') == source['sha256']
exits[diag_path.parent.name] = finished(diag_path.parent.name)
admission_path = BASE/'configs/loopahead-ouro-1.4b-admission-v1.json'
admission = read(admission_path)
assert admission['split'] == 'development_only' and admission['checkpoint_sha256'] == audit['checkpoint_sha256']
dev_ids = {r['source_id'] for r in rows(BASE/'data/manifests/loopahead-dev.jsonl')}
assert len(admission['source_ids']) == len(set(admission['source_ids'])) == 16
assert set(admission['source_ids']) <= dev_ids
for folder, expected in admission['source_runs'].items():
    folder = Path(folder)
    assert sha(folder/'requests.jsonl') == expected
    meta = read(folder/'metadata.json')
    assert meta['dev_calibration'] and meta['checkpoint_sha256'] == audit['checkpoint_sha256']
    assert {r['source_id'] for r in rows(folder/'requests.jsonl')} == set(admission['source_ids'])
    exits[folder.name] = finished(folder.name)
exits['admission'] = finished('loopahead-admission-ouro-1.4b-v1')
save(Path(__file__).parent/'report.json', dict(status='passed', checked_unix=time.time(),
    training_audit=dict(path=str(audit_path),sha256=sha(audit_path)), selected_checkpoint=str(checkpoint),
    checkpoint_sha256=audit['checkpoint_sha256'], parameters=11051008, epochs=3, updates=13533,
    train_requests=9025, eligible_train_requests=9022, dev_requests=782, confirmation_requests=782,
    training_exit=train_exit, audit_exit=audit_exit, probe_exits=exits,
    gsm20_whole_sequence_matches=matches, fp32_common_prefix_cases=16,
    admission=dict(path=str(admission_path),sha256=sha(admission_path),development_requests=16),
    limitation='Completed training and small probes only; full formal numerical checks remain separate.',
    check_source_sha256=sha(Path(__file__))))
print(dict(status='passed', train_eligible=9022, updates=13533, matches=matches, common_prefix_cases=16))
