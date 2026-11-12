"""Admit the completed Ouro BBH/T1 cell with source and actual-exit evidence."""
import csv
import io
import json
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
assert len(old_rows) == len(old_agreement) == 24
protocol = read(ROOT/'results/loopfill-latency512-20261107/protocol.json')
paths = dict(loopahead=BASE/'runs/loopahead-ouro-2.6b-bbh-t1-v1',
             ar=BASE/'runs/archived-ar-ouro-2.6b-bbh-t1')
runs = {key: read_run(path, protocol) for key, path in paths.items()}
evidence = {key: validate_run_artifacts(run) for key, run in runs.items()}
assert all(len(run['rows']) == 270 for run in runs.values())
method, ar = runs['loopahead'], runs['ar']
meta = method['meta']
assert meta['model'] == 'ouro-2.6b' and meta['dataset'] == 'bbh' and meta['temperature'] == 1
assert meta['config'] == dict(total=4, first=1, second=2, future=3, retain=32,
                             execute=16, guard=2, stride=1, family='ouro', temperature=1., top_p=1.)
assert meta['physical_nodes'] == 32
admission = BASE/'configs/loopahead-ouro-2.6b-admission-v1.json'
assert sha(admission) == meta['admission_sha256']
audit_path = BASE/'training/loopahead-ouro-2.6b-formal-audit-v2/report.json'
audit = read(audit_path)
assert audit['status'] == 'complete_LAD_training_audit_not_benchmark'
assert audit['model'] == meta['model'] and audit['selected_step'] == 13000
assert audit['selected_checkpoint'] == meta['checkpoint']
assert audit['checkpoint_sha256'] == meta['checkpoint_sha256'] == sha(Path(meta['checkpoint']))
receipt = read(BASE/f'logs/{paths["loopahead"].name}-finished.json')
assert receipt['pid'] == 516319
argv = receipt['argv']
assert argv[argv.index('--admission')+1] == str(admission)
assert argv[argv.index('--gpu')+1] == '2' and argv[argv.index('--cpus')+1] == '8-11'
acquisition = read(paths['loopahead']/'cuda-acquisition.json')
assert acquisition['pid'] == receipt['pid'] and acquisition['gpu'] == 2
assert acquisition['gpu_uuid'] == meta['gpu_uuid']
assert receipt['started_unix'] <= acquisition['started_unix'] <= acquisition['acquired_unix'] <= method['rows'][0]['finished_unix']
job = BASE/'logs/formal-controller-v1/loopahead-ouro-2.6b-bbh-t1.json'
worker = read(job.with_name(job.stem+'-worker-exit.json'))
outer = read(job.with_name(job.stem+'-finished.json'))
assert worker['exit_code'] == outer['exit_code'] == 0 and worker['worker_pid'] == 516314
assert worker['job_sha256'] == sha(job) and worker['source_sha256'] == sha(BASE/'orchestrate_formal.py')
assert worker['started_unix'] <= receipt['started_unix']
assert receipt['finished_unix'] <= worker['finished_unix'] <= outer['finished_unix']
agreement = compare(method, ar)
completed = subprocess.run([sys.executable, str(BASE/'collect_results.py'), '--partial'],
                           cwd=ROOT, text=True, capture_output=True)
(out/'collector.stdout').write_text(completed.stdout)
(out/'collector.stderr').write_text(completed.stderr)
assert completed.returncode == 0, completed.stderr
rows = keyed((BASE/'results.csv').read_text())
agreements = read(BASE/'formal-agreement.json')
key = ('loopahead', 'ouro-2.6b', 'bbh', '1')
agreement_key = 'ours/ouro-2.6b/bbh/1/loopahead'
assert set(rows) - set(old_rows) == {key} and len(rows) == 25
assert set(agreements) - set(old_agreement) == {agreement_key} and len(agreements) == 25
assert all(rows[k] == v for k, v in old_rows.items())
assert all(agreements[k] == v for k, v in old_agreement.items())
assert agreements[agreement_key] == agreement
assert read(BASE/'formal-matrix-status.json')['completed'] == 25
result = rows[key]
assert float(result['eta_latency']) == ar['summary']['decode_s']/method['summary']['decode_s']
assert float(result['eta_throughput']) == method['summary']['tok_s']/ar['summary']['tok_s']
report = dict(status='complete_formal_sampling_cell_with_declared_limitations', checked_unix=time.time(),
              completed_cells=25, expected_cells=72, previous_commit=head,
              previous_24_rows_and_agreements_unchanged=True, requests=270,
              summaries={k: r['summary'] for k, r in runs.items()},
              journal_sha256={k: sha(path/'requests.jsonl') for k, path in paths.items()},
              metadata_sha256={k: sha(path/'metadata.json') for k, path in paths.items()},
              execution_evidence=evidence, actual_exit_receipt=receipt,
              worker_exit=worker, outer_exit=outer, config=meta['config'],
              training_audit_sha256=sha(audit_path), checkpoint_sha256=meta['checkpoint_sha256'],
              admission_sha256=sha(admission), cuda_acquisition=acquisition,
              native_driver_live_evidence_sha256=sha(BASE/'preflight/formal-cell24-loopahead-huginn-bbh-t1-v1/next-run-live.json'),
              result=result, agreement={k: v for k, v in agreement.items() if k != 'first_divergences'},
              output_sha256={name: sha(BASE/name) for name in names},
              collector_exit_code=completed.returncode, collector_stdout=completed.stdout,
              collection_source_sha256={name: sha(BASE/name) for name in ['collect_results.py', 'result_evidence.py']},
              verification_source_sha256=sha(Path(__file__)),
              limitations=['Own paper-based implementation; no author training code or weights used.',
                           'AR uses the approved archived same-host measurement; cross-date and shared-host timing differences remain uncorrected.',
                           'T1 same-seed token agreement is descriptive, not a sampling-distribution proof.',
                           'Original CUDA-startup failure remains archived.',
                           '25/72 cells complete; final training, remaining benchmarks and diagnostics are pending.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: report[key] for key in ['completed_cells', 'result', 'agreement']}))
