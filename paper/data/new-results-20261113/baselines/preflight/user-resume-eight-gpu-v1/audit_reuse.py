"""Audit existing measurements before removing duplicate pending work."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
SUITE = Path('/ossfs/workspace/loop_suffix/results/dflash-table12-packed-20261112-v2')
sys.path.insert(0, str(BASE))
from collect_results import read_run, compare
from result_evidence import validate_run_artifacts


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return tuple(row[k] for k in ['method', 'model', 'benchmark', 'temperature'])


assert not (OUT / 'reuse.json').exists()
protocol = read(ROOT / 'results/loopfill-latency512-20261107/protocol.json')
rows = list(csv.DictReader((BASE / 'results.csv').open()))
current = {key(r): r for r in rows}
old = {key(r): r for r in csv.DictReader(io.StringIO(subprocess.check_output(
    ['git', 'show', 'HEAD:experiments/baselines/results.csv'], cwd=ROOT, text=True)))}
assert len(old) == 31 and len(current) >= 35
assert all(current[k] == v for k, v in old.items())
agreements = read(BASE / 'formal-agreement.json')
new_results = []
for k, row in current.items():
    if k in old:
        continue
    assert row['method'] == 'loopahead' and row['temperature'] == '1'
    ar, method = [read_run(Path(row[name]), protocol) for name in ['ar_run', 'method_run']]
    evidence = {name: validate_run_artifacts(run) for name, run in [('ar', ar), ('loopahead', method)]}
    assert compare(method, ar) == agreements['/'.join([row['engine'], row['model'], row['benchmark'], '1', 'loopahead'])]
    assert sha(Path(method['meta']['checkpoint'])) == method['meta']['checkpoint_sha256']
    assert len(method['rows']) <= 500
    assert float(row['eta_latency']) == ar['summary']['decode_s'] / method['summary']['decode_s']
    assert float(row['eta_throughput']) == method['summary']['tok_s'] / ar['summary']['tok_s']
    new_results.append({'key': k, 'row': row, 'execution_evidence': evidence})

dispatch = read(SUITE / 'dispatch.json')
jobs = read(SUITE / 'jobs.json')
legacy = {f"table1-{r['model']}-{r['benchmark']}-t{r['temperature']}": r for r in rows if r['method'] == 'dflash'}
reuse = {}
for job in jobs:
    identity = job['id']
    if identity not in legacy or identity in dispatch:
        continue
    row = legacy[identity]
    method_path, ar_path = Path(row['method_run']), Path(row['ar_run'])
    profile_path = method_path.with_name(method_path.name.replace('-dflash-v1', '-dflash-profile-v1'))
    runs = {'ar': read_run(ar_path, protocol), 'dflash': read_run(method_path, protocol),
            'profile': read_run(profile_path, protocol, profile=True)}
    evidence = {name: validate_run_artifacts(run) for name, run in runs.items()}
    assert job['count'] == len(runs['dflash']['rows']) <= 500
    assert job['max_new_tokens'] == runs['dflash']['meta']['max_new_tokens'] == 512
    assert all(t['tokens'] == p['tokens'] and t['stats']['target_verifications'] == p['stats']['target_verifications']
               for t, p in zip(runs['dflash']['rows'], runs['profile']['rows'], strict=True))
    assert compare(runs['dflash'], runs['ar']) == agreements['/'.join([
        row['engine'], row['model'], row['benchmark'], row['temperature'], 'dflash'])]
    if row['model'] == 'huginn-3.5b':
        assert runs['dflash']['meta']['draft_parameters'] == 332621056
    reuse[identity] = {'row': row, 'execution_evidence': evidence,
        'runs': {label: {'path': str(run['path']), 'requests_sha256': sha(run['path']/'requests.jsonl'),
                         'metadata_sha256': sha(run['path']/'metadata.json'),
                         'complete_sha256': sha(run['path']/'complete.json')} for label, run in runs.items()},
        'provenance_note': 'Reuse the existing HF-target measurement and its original AR denominator. This is not a completed packed-backend measurement.'}
assert len(reuse) == 9
result = {'status': 'verified_existing_measurements_for_reuse', 'checked_unix': time.time(),
          'authorization': 'User explicitly resumed both baselines, small DFlash, <=500 questions per dataset, all eight available GPUs, no repeat of existing results.',
          'preserved_previous_rows': len(old), 'current_rows': len(current), 'new_results': new_results,
          'reuse': reuse, 'already_running_overlap': [k for k in legacy if dispatch.get(k, {}).get('outcome') == 'running'],
          'formal_outputs_sha256': {name: sha(BASE/name) for name in ['results.csv', 'formal-agreement.json', 'formal-matrix-status.json', 'table2_rows.md']},
          'source_sha256': sha(Path(__file__)), 'limitation': 'No new GPU measurement and no numerical equivalence claim.'}
(OUT/'reuse.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'preserved_rows': len(old), 'new_rows': len(new_results),
                  'avoid_pending_repeats': list(reuse), 'already_running_overlap': result['already_running_overlap']}))
