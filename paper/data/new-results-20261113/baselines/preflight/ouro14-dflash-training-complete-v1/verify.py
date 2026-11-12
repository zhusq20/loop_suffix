"""Bind completed training, its CPU audit, measured costs and interface check."""
from pathlib import Path
import hashlib
import json
import math
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = Path('/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3')
evidence = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    evidence[str(path)] = sha(path)
    return json.loads(path.read_text())


audit_dir = BASE / 'training/dflash-ouro-1.4b-portable-formal-audit-v1'
audit = read(audit_dir / 'report.json')
assert audit['status'] == 'complete_training_audit_not_benchmark'
assert audit['epochs'] == 6 and audit['examples'] == 48550
assert audit['excluded_under_32_tokens'] == 1450 and not audit['cuda_initialized']
assert audit['metrics']['updates'] == 9108 and audit['metrics']['all_logged_updates_present_and_finite']
assert audit['weights']['parameters'] == 163597568 and audit['weights']['all_weights_finite']
assert audit['learning_rate_schedule']['all_logged_learning_rates_match']
assert audit['learning_rate_schedule']['max_absolute_error'] == 0
for path, digest in audit['artifact_hashes'].items():
    assert sha(path) == digest
    evidence[path] = digest
for name, digest in read(audit_dir / 'source-manifest.json').items():
    assert sha(audit_dir / 'source' / name) == sha(BASE / name) == digest
    evidence[str(audit_dir / 'source' / name)] = digest
launch = read(BASE / 'logs/audit-portable-dflash-ouro14-final-v1-launcher.json')
exit_record = read(BASE / 'logs/audit-portable-dflash-ouro14-final-v1-finished.json')
assert exit_record['exit_code'] == 0 and all(exit_record[k] == v for k, v in launch.items())
assert launch['started_unix'] <= audit['started_unix'] <= audit['finished_unix'] <= exit_record['finished_unix']

hours = {}
for stage in ['teachers', 'smoke', 'train', 'audit']:
    launch, receipt = read(ROOT / (stage + '-launcher.json')), read(ROOT / (stage + '-exit.json'))
    assert receipt['exit_code'] == 0
    assert all(receipt[k] == launch[k] for k in ['pid', 'argv', 'started_unix'])
    hours[stage] = (receipt['finished_unix'] - receipt['started_unix']) / 3600
done = read(ROOT / 'complete.json')
assert done['status'] == 'complete' and done['updates'] == 9108
assert done['weights_sha256'] == audit['weights_sha256']

archive = BASE / 'external-evidence/portable-dflash-ouro14-v3'
latest = read(archive / 'latest.json')
assert sha(latest['manifest']) == latest['sha256']
manifest = read(latest['manifest'])
assert manifest['pipeline_status'] == 'complete'
assert set(manifest['closed_stages']) == {'teachers', 'smoke', 'train'}
for original, record in manifest['files'].items():
    assert sha(original) == sha(record['archived']) == record['sha256']
    assert Path(record['archived']).stat().st_size == record['bytes']
assert sum(record['bytes'] for record in manifest['files'].values()) == manifest['bytes']

cost = read(BASE / 'training_cost.json')
ledger = read(BASE / 'training_cost-ledger.json')
assert not cost['partial'] and len(cost['rows']) == 6
assert all(row['status'] == 'complete' for row in cost['rows'])
assert cost['ledger_sha256'] == sha(BASE / 'training_cost-ledger.json')
assert math.isclose(sum(row['reserved_gpu_seconds'] for row in ledger['reservations']) / 3600,
                    cost['total_recorded_gpu_hours'], abs_tol=1e-9)
row = next(row for row in cost['rows'] if row['method'] == 'dflash' and row['model'] == 'ouro-1.4b')
assert row['details']['weights_sha256'] == audit['weights_sha256']
assert row['training_gpu_hours']['complete'] == hours['train']
assert row['generation_gpu_hours']['complete'] == hours['teachers']
portable = [row for row in ledger['reservations'] if row.get('scope') == 'portable_ouro14_dflash']
assert math.isclose(sum(row['reserved_gpu_seconds'] for row in portable) / 3600,
                    manifest['closed_pipeline_gpu_hours'], abs_tol=1e-12)

queue = BASE / 'logs/ouro14-packed-evaluation-v1'
protocol = read(queue / 'suite/protocol.json')
assert protocol['models']['ouro-1.4b']['checkpoint_sha256']['model.safetensors'] == audit['weights_sha256']
validation = queue / 'suite/runs/validate-ouro-1.4b/attempt1'
comparison = read(validation / 'comparison.json')
assert len(comparison['expected']) == len(comparison['actual']) == 6
assert [row['tokens'] for row in comparison['actual']] == comparison['expected']
summary = read(validation / 'summary.json')
assert summary['status'] == 'passed_full_weight_fp32' and summary['reportable'] is False
assert read(queue / 'logs/validate-ouro-1.4b-exit.json')['exit_code'] == 0
csv_digest = sha(BASE / 'combined-results/results.csv')
assert csv_digest == 'ed407418e235c10bf37b7b4b729e1486f5ac7c3026a9ebbec3c7c598e6621ff4'
for name in ['training_cost.md', 'DESIGN.md', 'EVAL_CONFIG.md', 'NOTES.md']:
    evidence[str(BASE / name)] = sha(BASE / name)
report = dict(status='completed_training_cost_and_interface_evidence_bound',
    updates=9108, epochs=6, parameters=163597568, source_sha256=sha(__file__),
    weights_sha256=audit['weights_sha256'], original_stage_gpu_hours=hours,
    pipeline_gpu_hours=manifest['closed_pipeline_gpu_hours'],
    total_training_related_gpu_hours=cost['total_recorded_gpu_hours'],
    cost_report_completed_training_rows=6, archived_files=len(manifest['files']),
    archived_bytes=manifest['bytes'], interface_sequences=6,
    unchanged_formal_results_sha256=csv_digest, evidence_sha256=evidence,
    checked_unix=time.time(), new_gpu_runs=0,
    limitations=['Training and six-prefix interface check do not establish formal performance or numerical equivalence.',
                 'Original measurement matrix remains 60/72; the remaining measurements and diagnostics are running.',
                 'Cost scope excludes benchmarks and separate numerical diagnostics; missing peak memory is not estimated.'])
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['evidence_sha256', 'limitations']}))
