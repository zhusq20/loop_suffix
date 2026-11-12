"""Verify the completed SpecBench cell, then collect it without changing old rows."""
import csv
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from collect_results import read_run, compare, sha
from dflash_source_identity import validate_evaluation_sources

def read(path):
    return json.loads(path.read_text())

def keyed(text):
    return {tuple(r[k] for k in ['method', 'model', 'benchmark', 'temperature']): r
            for r in csv.DictReader(io.StringIO(text))}

out = Path(__file__).parent
assert not (out/'report.json').exists()
names = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
before = {}
for name in names:
    raw = (BASE/name).read_bytes()
    committed = subprocess.check_output(['git', 'show', f'{head}:experiments/baselines/{name}'], cwd=ROOT)
    assert raw == committed, f'Uncollected changes in {name}'
    before[name] = raw.decode()
old_rows = keyed(before['results.csv'])
old_agreement = json.loads(before['formal-agreement.json'])
assert len(old_rows) == len(old_agreement) == 22

protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
stages = ['ar', 'dflash', 'dflash-profile']
runs, receipts = {}, {}
common = ['model', 'model_revision', 'dataset', 'temperature', 'top_p', 'top_k', 'seed',
          'count', 'full_count', 'max_new_tokens', 'max_seq', 'input_sha256', 'stop_audit',
          'torch', 'transformers', 'dtype', 'cuda', 'graph', 'compile', 'gpu_uuid', 'cpu_affinity']
for stage in stages:
    name = f'dflash-stack-ouro-2.6b-specbench-t0-{stage}-v1'
    path = BASE/'runs'/name
    run = read_run(path, protocol, profile=stage == 'dflash-profile')
    assert run['meta']['model'] == 'ouro-2.6b' and run['meta']['dataset'] == 'specbench'
    assert run['meta']['temperature'] == 0 and len(run['rows']) == 480
    validate_evaluation_sources(path)
    launch = read(BASE/f'logs/{name}-launcher.json')
    finish = read(BASE/f'logs/{name}-finished.json')
    assert finish['exit_code'] == 0 and all(finish[k] == v for k, v in launch.items())
    argv = finish['argv']
    assert argv[argv.index('--out')+1] == str(path)
    assert argv[argv.index('--method')+1] == ('ar' if stage == 'ar' else 'dflash')
    assert argv[argv.index('--dataset')+1] == 'specbench'
    assert argv[argv.index('--temperature')+1] == '0'
    assert argv[argv.index('--gpu')+1] == '2' and argv[argv.index('--cpus')+1] == '8-11'
    assert ('--profile' in argv) == (stage == 'dflash-profile')
    assert finish['started_unix'] <= run['rows'][0]['finished_unix'] <= run['rows'][-1]['finished_unix']
    assert run['rows'][-1]['finished_unix'] <= read(path/'complete.json')['finished_unix'] <= finish['finished_unix']
    assert all(math.isfinite(row['stats']['decode_s']) and row['stats']['decode_s'] >= 0 for row in run['rows'])
    assert all(row['stats']['decode_tokens'] == len(row['tokens']) - 1 for row in run['rows'])
    if stage != 'ar':
        assert not any(read(path/'checkpoint-loading.json').values())
    runs[stage], receipts[stage] = run, finish
for stage in stages[1:]:
    assert all(runs[stage]['meta'][key] == runs['ar']['meta'][key] for key in common)
for a, b in zip(stages, stages[1:]):
    assert receipts[a]['finished_unix'] <= receipts[b]['started_unix']

timed, profile, ar = runs['dflash'], runs['dflash-profile'], runs['ar']
assert timed['meta']['checkpoint_sha256'] == profile['meta']['checkpoint_sha256']
for t, p in zip(timed['rows'], profile['rows']):
    assert t['tokens'] == p['tokens']
    assert t['stats']['target_verifications'] == p['stats']['target_verifications']
    assert math.isfinite(p['stats']['draft_gpu_s']) and p['stats']['draft_gpu_s'] >= 0
agreement = compare(timed, ar)
draft_s = sum(r['stats']['draft_gpu_s'] for r in profile['rows'])
job = BASE/'logs/formal-controller-v1/dflash-ouro-2.6b-specbench-t0.json'
worker = read(job.with_name(job.stem+'-worker-exit.json'))
outer = read(job.with_name(job.stem+'-finished.json'))
assert worker['exit_code'] == outer['exit_code'] == 0
assert worker['worker_pid'] == 287000
assert worker['job_sha256'] == sha(job) and worker['source_sha256'] == sha(BASE/'orchestrate_formal.py')
assert worker['started_unix'] <= receipts['ar']['started_unix']
assert receipts['dflash-profile']['finished_unix'] <= worker['finished_unix'] <= outer['finished_unix']

# The collector also validates the complete training, teacher manifests, and
# actual checkpoint hashes. Its completed-only export cannot promote this run
# without all three full-task journals.
completed = subprocess.run([sys.executable, str(BASE/'collect_results.py'), '--partial'],
                           cwd=ROOT, text=True, capture_output=True)
(out/'collector.stdout').write_text(completed.stdout)
(out/'collector.stderr').write_text(completed.stderr)
assert completed.returncode == 0, completed.stderr
rows = keyed((BASE/'results.csv').read_text())
audits = read(BASE/'formal-agreement.json')
key = ('dflash', 'ouro-2.6b', 'specbench', '0')
audit_key = 'official/ouro-2.6b/specbench/0/dflash'
assert set(rows) - set(old_rows) == {key} and len(rows) == 23
assert set(audits) - set(old_agreement) == {audit_key} and len(audits) == 23
assert all(rows[k] == v for k, v in old_rows.items())
assert all(audits[k] == v for k, v in old_agreement.items())
assert audits[audit_key] == agreement
assert read(BASE/'formal-matrix-status.json')['completed'] == 23
result = rows[key]
assert float(result['eta_latency']) == ar['summary']['decode_s'] / timed['summary']['decode_s']
assert float(result['eta_throughput']) == timed['summary']['tok_s'] / ar['summary']['tok_s']
assert float(result['drafter_time_frac']) == draft_s / profile['summary']['decode_s']
six = [r for k, r in rows.items() if k[0] == 'dflash' and k[1] == 'ouro-2.6b' and k[3] == '0']
assert len(six) == 6
six_agreements = [v for k, v in audits.items() if k.startswith('official/ouro-2.6b/') and k.endswith('/0/dflash')]
means = {field: statistics.mean(float(r[field]) for r in six) for field in ['eta_latency', 'eta_throughput', 'gamma']}
means.update(tasks=6, requests=sum(a['requests'] for a in six_agreements),
             exact_sequences=sum(a['exact_sequences'] for a in six_agreements),
             convention='Arithmetic mean of the six task metrics; not a pooled ratio.')
report = dict(status='complete_formal_cell_with_greedy_diagnostics_pending', checked_unix=time.time(),
              completed_cells=23, expected_cells=72, previous_commit=head,
              previous_22_rows_and_agreements_unchanged=True, common_config_keys=common,
              requests_per_stage={k: len(r['rows']) for k, r in runs.items()},
              summaries={k: r['summary'] for k, r in runs.items()},
              journal_sha256={k: sha(r['path']/'requests.jsonl') for k, r in runs.items()},
              source_manifest_sha256={k: sha(r['path']/'source-manifest.json') for k, r in runs.items()},
              actual_exit_receipts=receipts, outer_exit=outer, worker_exit=worker,
              profile_all_tokens_and_verification_counts_match_timed_run=True,
              draft_gpu_s=draft_s,
              profile_wall_relative_difference=profile['summary']['decode_s']/timed['summary']['decode_s']-1,
              result=result, agreement={k: v for k, v in agreement.items() if k != 'first_divergences'},
              first_divergences_pending=len(agreement['first_divergences']),
              ouro_dflash_t0_six_task_means=means,
              limitations=['All first-divergence BF16/FP32 prefix diagnostics for this cell remain pending.',
                           'Shared host timings are retained without correction or CPU isolation claims.',
                           'Profile wall-time difference includes run variation, not just instrumentation overhead.',
                           'This cell completed before the native driver loader recovery.'],
              output_sha256={name: sha(BASE/name) for name in names},
              collector_exit_code=completed.returncode, collector_stdout=completed.stdout,
              verification_source_sha256=sha(Path(__file__)))
(out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['completed_cells', 'result', 'agreement', 'draft_gpu_s',
                  'profile_wall_relative_difference', 'first_divergences_pending', 'ouro_dflash_t0_six_task_means']}))
