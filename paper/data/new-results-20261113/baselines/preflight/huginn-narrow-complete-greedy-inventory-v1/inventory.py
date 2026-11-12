"""Inventory every first divergence in the three completed narrow Huginn cells.

This is a CPU input/readiness audit, not a BF16/FP32 replay or a completion marker.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from collect_results import compare, read_run
from result_evidence import validate_run_artifacts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


assert not (OUT / 'report.json').exists()
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
formal_names = ['results.csv', 'formal-agreement.json', 'formal-matrix-status.json', 'table2_rows.md']
formal_hashes = {name: sha(BASE / name) for name in formal_names}
for name in formal_names:
    assert (BASE / name).read_bytes() == subprocess.check_output(
        ['git', 'show', f'{head}:experiments/baselines/{name}'], cwd=ROOT)
rows = list(csv.DictReader((BASE / 'results.csv').open()))
assert len(rows) == read(BASE / 'formal-matrix-status.json')['completed'] == 31
selected = [r for r in rows if (r['method'], r['model'], r['temperature']) == ('dflash', 'huginn-3.5b', '0')]
expected_counts = {'gsm8k': (300, 76), 'humanevalplus': (164, 26), 'mbppplus': (378, 70)}
assert {r['benchmark'] for r in selected} == set(expected_counts)
agreements = read(BASE / 'formal-agreement.json')
protocol = read(ROOT / 'results/loopfill-latency512-20261107/protocol.json')
cells = []
all_cases = []
for row in selected:
    task = row['benchmark']
    ar, method = [read_run(Path(row[key]), protocol) for key in ['ar_run', 'method_run']]
    evidence = {label: validate_run_artifacts(run) for label, run in [('ar', ar), ('dflash', method)]}
    assert method['meta']['draft_parameters'] == 332621056
    assert method['meta']['draft_architecture'] == 'target-width-projections-v1'
    agreement = compare(method, ar)
    key = '/'.join([row['engine'], row['model'], task, '0', row['method']])
    assert agreement == agreements[key]
    cases = []
    for a, d in zip(ar['rows'], method['rows'], strict=True):
        assert a['source_id'] == d['source_id'] and a['prompt_sha256'] == d['prompt_sha256']
        if a['tokens'] == d['tokens']:
            continue
        position = next((i for i, (u, v) in enumerate(zip(a['tokens'], d['tokens'])) if u != v), None)
        assert position is not None, 'Length-only divergence needs an EOS-specific diagnosis'
        assert 0 < position < 512
        assert a['tokens'][:position] == d['tokens'][:position]
        cases.append({'benchmark': task, 'source_id': a['source_id'], 'position': position,
                      'prompt_sha256': a['prompt_sha256'], 'ar_token': a['tokens'][position],
                      'dflash_token': d['tokens'][position], 'ar_length': len(a['tokens']),
                      'method_length': len(d['tokens']), 'request_seed': d['stats']['seed'],
                      'common_output_sha256': hashlib.sha256(json.dumps(a['tokens'][:position],
                          separators=(',', ':')).encode()).hexdigest()})
    assert (len(ar['rows']), len(cases)) == expected_counts[task]
    assert [(c['source_id'], c['position']) for c in cases] == [
        (c['source_id'], c['position']) for c in agreement['first_divergences']]
    folder = BASE / 'preflight/formal-greedy-margins' / f'dflash-huginn-3.5b-{task}-v1'
    assert not folder.exists(), 'Existing numerical diagnostics need separate validation'
    cells.append({'benchmark': task, 'requests': len(ar['rows']), 'divergences': len(cases),
                  'position_min': min(c['position'] for c in cases),
                  'position_max': max(c['position'] for c in cases),
                  'numerical_replay': 'pending', 'expected_diagnostic_folder': str(folder),
                  'source_runs': {label: {'path': str(run['path']), 'metadata_sha256': sha(run['path']/'metadata.json'),
                      'journal_sha256': sha(run['path']/'requests.jsonl'), 'complete_sha256': sha(run['path']/'complete.json')}
                      for label, run in [('ar', ar), ('dflash', method)]}, 'execution_evidence': evidence})
    all_cases.extend(cases)

# Verify that the current runner retains the original full 512-token block budget
# and only stops instrumented replay after covering the first differing token.
runner = BASE / 'diagnose_dflash.py'
tree = ast.parse(runner.read_text())
calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'generate']
assert len(calls) == 1
assert isinstance(calls[0].args[2], ast.Constant) and calls[0].args[2].value == 512
stop = next(k.value for k in calls[0].keywords if k.arg == 'diagnostic_stop_after')
assert ast.dump(stop) == ast.dump(ast.parse("case['position'] + 1", mode='eval').body)
assert len(all_cases) == 172 and sum(c['requests'] for c in cells) == 842
case_path = OUT / 'cases.jsonl'
case_path.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in all_cases))
assert {name: sha(BASE / name) for name in formal_names} == formal_hashes
report = {'status': 'cpu_inventory_complete_numerical_replay_pending', 'checked_unix': time.time(),
          'head': head, 'completed_formal_cells': 31, 'requests_audited': 842, 'pending_first_divergences': 172,
          'cells': cells, 'case_manifest_sha256': sha(case_path), 'formal_output_sha256': formal_hashes,
          'diagnostic_runner_sha256': sha(runner), 'runner_preserves_full_512_budget': True,
          'source_sha256': sha(Path(__file__)),
          'limitation': 'No GPU execution, probability margins, numerical equivalence, or new benchmark completion claimed.'}
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['cells', 'formal_output_sha256']}))
