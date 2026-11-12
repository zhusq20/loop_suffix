"""Admit the first complete final narrow Huginn cell, preserving prior results."""
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
from dflash_training_registry import validate_projected_completion


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
assert len(old_rows) == len(old_agreement) == 27
protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
stages = ['ar', 'dflash', 'dflash-profile']
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity']
runs, receipts, evidence = {}, {}, {}
for stage in stages:
    path = BASE/f'runs/dflash-stack-huginn-3.5b-gsm8k-t0-{stage}-v1'
    run = read_run(path, protocol, profile=stage == 'dflash-profile')
    evidence[stage] = validate_run_artifacts(run)
    assert run['meta']['model'] == 'huginn-3.5b' and run['meta']['dataset'] == 'gsm8k'
    assert run['meta']['temperature'] == 0 and len(run['rows']) == 300
    finish = read(BASE/f'logs/{path.name}-finished.json')
    argv = finish['argv']
    assert argv[argv.index('--gpu')+1] == '2' and argv[argv.index('--cpus')+1] == '8-11'
    assert run['meta']['gpu_uuid'] == 'GPU-861ad2de-b899-21ca-707e-44c0a1ce1029'
    runs[stage], receipts[stage] = run, finish
for stage in stages[1:]:
    assert all(runs[stage]['meta'][key] == runs['ar']['meta'][key] for key in common)
for a, b in zip(stages, stages[1:]):
    assert receipts[a]['finished_unix'] <= receipts[b]['started_unix']
assert [receipts[stage]['pid'] for stage in stages] == [182221, 219749, 235583]
timed, profile, ar = runs['dflash'], runs['dflash-profile'], runs['ar']
binding = validate_projected_completion()
for run in [timed, profile]:
    assert run['meta']['training_completion_evidence'] == binding
    assert run['meta']['checkpoint'] == binding['checkpoint']
    assert run['meta']['draft_parameters'] == 332621056
    assert run['meta']['draft_architecture'] == 'target-width-projections-v1'
    assert run['meta']['checkpoint_sha256'] == {'model.safetensors': binding['weights_sha256'],
                                              'config.json': sha(Path(binding['checkpoint'])/'config.json')}
for t, p in zip(timed['rows'], profile['rows']):
    assert t['tokens'] == p['tokens']
    assert t['stats']['target_verifications'] == p['stats']['target_verifications']
    assert math.isfinite(p['stats']['draft_gpu_s']) and p['stats']['draft_gpu_s'] >= 0
agreement = compare(timed, ar)
assert agreement['exact_sequences'] == 224 and len(agreement['first_divergences']) == 76
draft_s = sum(r['stats']['draft_gpu_s'] for r in profile['rows'])
job = BASE/'logs/formal-controller-v1/dflash-huginn-3.5b-gsm8k-t0.json'
worker = read(job.with_name(job.stem+'-worker-exit.json'))
outer = read(job.with_name(job.stem+'-finished.json'))
assert worker['exit_code'] == outer['exit_code'] == 0 and worker['worker_pid'] == 182214
assert worker['job_sha256'] == sha(job) and worker['source_sha256'] == sha(BASE/'orchestrate_formal.py')
assert worker['started_unix'] <= receipts['ar']['started_unix']
assert receipts['dflash-profile']['finished_unix'] <= worker['finished_unix'] <= outer['finished_unix']

completed = subprocess.run([sys.executable, str(BASE/'collect_results.py'), '--partial'],
                           cwd=ROOT, text=True, capture_output=True)
(out/'collector.stdout').write_text(completed.stdout)
(out/'collector.stderr').write_text(completed.stderr)
assert completed.returncode == 0, completed.stderr
rows = keyed((BASE/'results.csv').read_text())
agreements = read(BASE/'formal-agreement.json')
key = ('dflash', 'huginn-3.5b', 'gsm8k', '0')
agreement_key = 'official/huginn-3.5b/gsm8k/0/dflash'
assert set(rows)-set(old_rows) == {key} and len(rows) == 28
assert set(agreements)-set(old_agreement) == {agreement_key} and len(agreements) == 28
assert all(rows[k] == v for k, v in old_rows.items())
assert all(agreements[k] == v for k, v in old_agreement.items())
assert agreements[agreement_key] == agreement
assert read(BASE/'formal-matrix-status.json')['completed'] == 28
result = rows[key]
assert float(result['eta_latency']) == ar['summary']['decode_s']/timed['summary']['decode_s']
assert float(result['eta_throughput']) == timed['summary']['tok_s']/ar['summary']['tok_s']
assert float(result['drafter_time_frac']) == draft_s/profile['summary']['decode_s']
report = dict(status='complete_formal_narrow_cell_greedy_diagnostics_pending', checked_unix=time.time(),
              completed_cells=28, expected_cells=72, previous_commit=head,
              previous_27_rows_and_agreements_unchanged=True, requests_per_stage={s: len(r['rows']) for s, r in runs.items()},
              common_config_keys=common, summaries={s: r['summary'] for s, r in runs.items()},
              journal_sha256={s: sha(r['path']/'requests.jsonl') for s, r in runs.items()},
              execution_evidence=evidence, actual_exit_receipts=receipts, worker_exit=worker, outer_exit=outer,
              training_binding=binding, checkpoint_sha256=timed['meta']['checkpoint_sha256'],
              profile_all_tokens_and_verification_counts_match_timed_run=True, draft_gpu_s=draft_s,
              profile_wall_relative_difference=profile['summary']['decode_s']/timed['summary']['decode_s']-1,
              result=result, agreement={k: v for k, v in agreement.items() if k != 'first_divergences'},
              first_divergences=len(agreement['first_divergences']),
              output_sha256={name: sha(BASE/name) for name in names}, collector_exit_code=completed.returncode,
              collection_source_sha256={name: sha(BASE/name) for name in ['collect_results.py', 'result_evidence.py']},
              verification_source_sha256=sha(Path(__file__)),
              limitations=['All 76 greedy first divergences still require actual-prefix BF16 and FP32 diagnostics.',
                           'Shared-host timings remain uncorrected; profile wall differences include run variation.',
                           'Own 332.6M Huginn width adaptation, not an author checkpoint or equal-training wide/narrow comparison.',
                           '28/72 cells complete; appendix training, remaining benchmarks and diagnostics are pending.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: report[key] for key in ['completed_cells', 'result', 'agreement', 'draft_gpu_s', 'profile_wall_relative_difference']}))
