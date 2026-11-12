"""Summarize the complete main-model matrix without declaring final acceptance."""
import csv
import itertools
import math
from pathlib import Path
import statistics
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from packed_dflash_evidence import read, save, sha

OUT = Path(__file__).resolve().parent
assert not (OUT/'report.json').exists()
source = BASE/'combined-results/results.csv'
snapshot = source.read_bytes()
(OUT/'results-snapshot.csv').write_bytes(snapshot)
with source.open() as stream:
    all_rows = list(csv.DictReader(stream))
models = ['huginn-3.5b','ouro-2.6b']
tasks = dict(gsm8k=300,math500=500,bbh=270,humanevalplus=164,mbppplus=378,specbench=480)
rows = [r for r in all_rows if r['model'] in models]
key = lambda r: (r['method'],r['model'],r['benchmark'],int(r['temperature']))
expected = set(itertools.product(['dflash','loopahead'],models,tasks,[0,1]))
assert len(rows) == len(expected) == 48 and {key(r) for r in rows} == expected
assert all(r['status'] == 'complete_with_declared_limitations' for r in rows)
collection = read(BASE/'preflight/combined-results-cell56-v1/report.json')
assert collection['status'] == 'passed' and collection['all_previous_rows_unchanged']
assert collection['completed'] == len(all_rows) == 56
assert collection['previous_rows'] == 55 and collection['results_sha256'] == sha(source)
assert len(collection['added']) == 1
assert key(collection['added'][0]) == ('dflash','ouro-2.6b','specbench',1)
for receipt in collection['collector_exits'].values():
    assert sha(receipt['path']) == receipt['sha256'] and read(receipt['path'])['exit_code'] == 0
evidence = read(BASE/'combined-results/evidence.json')
assert sha(BASE/'combined-results/evidence.json') == collection['evidence_sha256']
unavailable = set()
for row in rows:
    identity = '/'.join(map(str,key(row)))
    proof = evidence['cells'][identity]
    assert proof['agreement']['requests'] == tasks[row['benchmark']]
    assert tasks[row['benchmark']] <= 500
    for item in proof['sources'].values():
        for filename,digest in item['sha256'].items():
            assert sha(Path(item['path'])/filename) == digest
    for name in ['eta_latency','eta_throughput','gamma','method_tok_s','ar_tok_s','method_decode_s','ar_decode_s']:
        assert math.isfinite(float(row[name])) and float(row[name]) > 0
    assert math.isclose(float(row['eta_latency']),float(row['ar_decode_s'])/float(row['method_decode_s']),rel_tol=1e-12)
    assert math.isclose(float(row['eta_throughput']),float(row['method_tok_s'])/float(row['ar_tok_s']),rel_tol=1e-12)
    if row['method'] == 'dflash':
        assert 0 <= float(row['drafter_time_frac']) <= 1
    if row['method_exit_evidence'] == 'unavailable_artifacts_verified':
        unavailable.add(key(row))
        assert proof['original_timed_exit']['actual_exit_code'] is None
    else:
        assert row['method_exit_evidence'] == 'actual_exit0'
assert unavailable == {('dflash','huginn-3.5b','math500',0),
                       ('dflash','huginn-3.5b','math500',1),
                       ('dflash','ouro-2.6b','specbench',1)}
means = []
for model,temp,method in itertools.product(models,[0,1],['dflash','loopahead']):
    group = [r for r in rows if (r['model'],int(r['temperature']),r['method']) == (model,temp,method)]
    assert len(group) == 6
    backends = sorted({r['target_backend'] for r in group})
    means.append(dict(model=model,temperature=temp,method=method,
        mean_eta_latency=statistics.mean(float(r['eta_latency']) for r in group),
        mean_eta_throughput=statistics.mean(float(r['eta_throughput']) for r in group),
        mean_gamma=statistics.mean(float(r['gamma']) for r in group),
        target_backends=backends,mixed_target_backends=len(backends)>1))
assert source.read_bytes() == snapshot
report = dict(status='main_model_measurements_complete_with_declared_limitations',
    main_cells=48,all_collected_cells=56,expected_all_cells=72,requests_per_cell=tasks,
    measured_method_requests=sum(tasks[r['benchmark']] for r in rows),
    actual_timed_exit0_cells=45,original_timed_exit_unavailable=sorted(unavailable),means=means,
    results_snapshot_sha256=sha(OUT/'results-snapshot.csv'),
    collection_report_sha256=sha(BASE/'preflight/combined-results-cell56-v1/report.json'),
    evidence_sha256=sha(BASE/'combined-results/evidence.json'),check_source_sha256=sha(Path(__file__)),
    checked_unix=time.time(),new_gpu_measurements=0,
    limitations=['Only main-model measurements are complete; full 72-cell delivery, sixth training and remaining numerical checks are pending.',
                 'Three original packed timed exit receipts are unavailable; complete artifact checks and actual AR/profile exits do not reconstruct them.',
                 'DFlash HF and packed target backends retain separate measured AR denominators; mixed-backend means are explicitly labelled.',
                 'LoopAhead uses the historical same-host AR measurements; cross-date timing drift is uncorrected.',
                 'BF16 whole-output differences and T1 same-seed statistics are retained; this summary does not establish numerical or distribution equivalence.'])
save(OUT/'report.json',report)
print(dict(status=report['status'],main_cells=48,all_cells=56,unknown_original_exits=3))
