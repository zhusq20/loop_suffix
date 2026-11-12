"""Admit the complete Ouro MATH-500/T1 DFlash cell and preserve previous results."""
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from collect_results import read_run, compare, sha
from result_evidence import validate_run_artifacts


def read(path):
    return json.loads(path.read_text())


def keyed(content):
    return {tuple(row[key] for key in ['method', 'model', 'benchmark', 'temperature']): row
            for row in csv.DictReader(io.StringIO(content))}


out = Path(__file__).parent
assert not (out/'report.json').exists()
names = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
before = {}
for name in names:
    raw = (BASE/name).read_bytes()
    assert raw == subprocess.check_output(['git', 'show', f'{head}:experiments/baselines/{name}'], cwd=ROOT)
    before[name] = raw.decode()
old_rows = keyed(before['results.csv'])
old_agreement = json.loads(before['formal-agreement.json'])
assert len(old_rows) == len(old_agreement) == 28
protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
stages = ['ar', 'dflash', 'dflash-profile']
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity']
runs, receipts, evidence = {}, {}, {}
for stage in stages:
    path = BASE/f'runs/dflash-stack-ouro-2.6b-math500-t1-{stage}-v1'
    run = read_run(path, protocol, profile=stage == 'dflash-profile')
    evidence[stage] = validate_run_artifacts(run)
    assert run['meta']['model'] == 'ouro-2.6b' and run['meta']['dataset'] == 'math500'
    assert run['meta']['temperature'] == 1 and len(run['rows']) == 500
    finish = read(BASE/f'logs/{path.name}-finished.json')
    argv = finish['argv']
    assert argv[argv.index('--gpu')+1] == '4' and argv[argv.index('--cpus')+1] == '48-51'
    acquisition = read(path/'cuda-acquisition.json')
    assert acquisition['pid'] == finish['pid'] and acquisition['gpu_uuid'] == run['meta']['gpu_uuid']
    assert finish['started_unix'] <= acquisition['started_unix'] <= acquisition['acquired_unix'] <= run['rows'][0]['finished_unix']
    runs[stage], receipts[stage] = run, finish
for stage in stages[1:]:
    assert all(runs[stage]['meta'][key] == runs['ar']['meta'][key] for key in common)
for a, b in zip(stages, stages[1:]):
    assert receipts[a]['finished_unix'] <= receipts[b]['started_unix']
assert [receipts[stage]['pid'] for stage in stages] == [420686, 134675, 217392]
timed, profile, ar = runs['dflash'], runs['dflash-profile'], runs['ar']
assert timed['meta']['checkpoint_sha256'] == profile['meta']['checkpoint_sha256']
for t, p in zip(timed['rows'], profile['rows']):
    assert t['tokens'] == p['tokens']
    assert t['stats']['target_verifications'] == p['stats']['target_verifications']
    assert math.isfinite(p['stats']['draft_gpu_s']) and p['stats']['draft_gpu_s'] >= 0
agreement = compare(timed, ar)
draft_s = sum(r['stats']['draft_gpu_s'] for r in profile['rows'])
job = BASE/'logs/formal-controller-v1/dflash-ouro-2.6b-math500-t1.json'
worker = read(job.with_name(job.stem+'-worker-exit.json'))
outer = read(job.with_name(job.stem+'-finished.json'))
assert worker['exit_code'] == outer['exit_code'] == 0 and worker['worker_pid'] == 420683
assert outer['completion_basis'] == 'worker_exit_receipt'
assert worker['job_sha256'] == sha(job) and worker['source_sha256'] == sha(BASE/'orchestrate_formal.py')
assert worker['started_unix'] <= receipts['ar']['started_unix']
assert receipts['dflash-profile']['finished_unix'] <= worker['finished_unix'] <= outer['finished_unix']
audit_path = BASE/'training/dflash-ouro-2.6b-formal-audit-v1/report.json'
audit = read(audit_path)
assert audit['status'] == 'complete_training_audit_not_benchmark'
assert sha(audit_path) == 'c19089e7e77c7fa61701414b17a253666137397bf85e29c7aa64644b6b494475'

completed = subprocess.run([sys.executable, str(BASE/'collect_results.py'), '--partial'],
                           cwd=ROOT, text=True, capture_output=True)
(out/'collector.stdout').write_text(completed.stdout)
(out/'collector.stderr').write_text(completed.stderr)
assert completed.returncode == 0, completed.stderr
rows = keyed((BASE/'results.csv').read_text())
agreements = read(BASE/'formal-agreement.json')
key = ('dflash', 'ouro-2.6b', 'math500', '1')
agreement_key = 'official/ouro-2.6b/math500/1/dflash'
assert set(rows)-set(old_rows) == {key} and len(rows) == 29
assert set(agreements)-set(old_agreement) == {agreement_key} and len(agreements) == 29
assert all(rows[k] == v for k, v in old_rows.items())
assert all(agreements[k] == v for k, v in old_agreement.items())
assert agreements[agreement_key] == agreement
assert read(BASE/'formal-matrix-status.json')['completed'] == 29
result = rows[key]
assert float(result['eta_latency']) == ar['summary']['decode_s']/timed['summary']['decode_s']
assert float(result['eta_throughput']) == timed['summary']['tok_s']/ar['summary']['tok_s']
assert float(result['drafter_time_frac']) == draft_s/profile['summary']['decode_s']
report = dict(status='complete_formal_sampling_cell_with_declared_limitations', checked_unix=time.time(),
              completed_cells=29, expected_cells=72, previous_commit=head,
              previous_28_rows_and_agreements_unchanged=True, requests_per_stage={s: len(r['rows']) for s, r in runs.items()},
              common_config_keys=common, summaries={s: r['summary'] for s, r in runs.items()},
              journal_sha256={s: sha(r['path']/'requests.jsonl') for s, r in runs.items()},
              execution_evidence=evidence, actual_exit_receipts=receipts, worker_exit=worker, outer_exit=outer,
              training_audit_sha256=sha(audit_path), checkpoint_sha256=timed['meta']['checkpoint_sha256'],
              profile_all_tokens_and_verification_counts_match_timed_run=True, draft_gpu_s=draft_s,
              profile_wall_relative_difference=profile['summary']['decode_s']/timed['summary']['decode_s']-1,
              result=result, agreement={k: v for k, v in agreement.items() if k != 'first_divergences'},
              output_sha256={name: sha(BASE/name) for name in names}, collector_exit_code=completed.returncode,
              collection_source_sha256={name: sha(BASE/name) for name in ['collect_results.py', 'result_evidence.py']},
              verification_source_sha256=sha(Path(__file__)),
              limitations=['T1 same-seed token agreement is descriptive, not a sampling-distribution proof.',
                           'Shared-host timings remain uncorrected; profile wall differences include run variation.',
                           'The original AR and worker were retained across the native-driver environment correction.',
                           '29/72 cells complete; final training, remaining benchmarks and diagnostics are pending.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: report[key] for key in ['completed_cells', 'result', 'agreement', 'draft_gpu_s', 'profile_wall_relative_difference']}))
