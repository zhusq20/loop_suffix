"""Verify closed timed runs only; do not publish incomplete speedup cells."""
import csv
import hashlib
import io
from pathlib import Path
import sys
import time

BASE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import check_frozen, check_run, read, save, sha
from provenance import freeze_sources

assert not (OUT / 'report.json').exists(), 'Preserve earlier evidence.'
root = BASE / 'logs/ouro14-packed-evaluation-v1'
suite = root / 'suite'
plan_path = root / 'plan-no-output-diagnostics.json'
plan = read(plan_path)
for name, digest in plan['frozen_sha256'].items():
    assert sha(root / name) == digest, name
for name, digest in plan['original_inputs_sha256'].items():
    assert sha(name) == digest, name
assert sha(root / 'plan.json') == plan['scope_amendment']['original_plan_sha256']
assert sha(Path(plan['training_root']) / 'plan.json') == plan['training_plan_sha256']

counts = dict(gsm8k=300, math500=500, bbh=270, humanevalplus=164,
              mbppplus=378, specbench=480)
jobs = plan['jobs']
assert len(jobs) == 12
assert {(j['dataset'], j['temperature']) for j in jobs} == {
    (task, temperature) for task in counts for temperature in (0, 1)}
assert all(j['model'] == 'ouro-1.4b' and j['count'] == counts[j['dataset']]
           for j in jobs)
protocol, _ = check_frozen(suite, jobs[0], weights=True)
assert protocol['table1'] == dict(temperatures=[0, 1], max_new_tokens=512,
    batch=1, warmup_requests=2, warmup_tokens=32)
assert protocol['target_row_padding'] is False
model = protocol['models']['ouro-1.4b']
assert model['checkpoint_sha256']['model.safetensors'] == (
    'd25b56c5c1f43e35d1c4ced2f32eb9bb102cb3f34e5f0b03a378a8cf2bd48742')
binding = read(root / 'training-binding.json')
assert binding['protocol_sha256'] == sha(suite / 'protocol.json')

csv_path = BASE / 'combined-results/results.csv'
before = csv_path.read_bytes()
dispatch = read(root / 'dispatch.json')
results, artifacts = {}, {}
for job in jobs:
    key = job['id'] + '-timed'
    state = dispatch[key]
    assert state['outcome'] == 'complete' and state['exit_code'] == 0, key
    folder = suite / state['out']
    checked = check_run(folder, suite, job)
    assert read(folder / 'preflight.json') == dict(status='passed', requests=2,
                                                   max_new_tokens=16)
    for row in checked['rows']:
        seed_key = f'0/{job["model"]}/{job["dataset"]}/{row["source_id"]}'.encode()
        expected_seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:8],
                                       'little') % (2**63 - 1)
        assert row['stats']['seed'] == expected_seed, (key, row['source_id'])
    launch_path = root / 'logs' / f'{key}-launcher.json'
    receipt_path = root / 'logs' / f'{key}-exit.json'
    suite_receipt = suite / 'logs' / f'{job["id"]}-attempt1-exit.json'
    launch, receipt = read(launch_path), read(receipt_path)
    assert type(receipt['exit_code']) is int and receipt['exit_code'] == 0
    assert receipt['exit_evidence'] == 'supervisor_waitpid'
    assert receipt['plan_sha256'] == sha(plan_path)
    for field in ('pid', 'argv', 'started_unix', 'start_ticks'):
        assert receipt[field] == launch[field] == state[field], (key, field)
    assert receipt['started_unix'] < read(folder / 'complete.json')['finished_unix'] <= receipt['finished_unix']
    assert read(suite_receipt) == receipt
    assert Path(receipt['argv'][receipt['argv'].index('--out') + 1]) == folder
    assert receipt['argv'][receipt['argv'].index('--job') + 1] == job['id']
    for name, digest in checked['hashes'].items():
        artifacts[str(folder / name)] = digest
    artifacts[str(folder / 'preflight.json')] = sha(folder / 'preflight.json')
    for path in (launch_path, receipt_path, suite_receipt):
        artifacts[str(path)] = sha(path)
    results[job['id']] = dict(summary=checked['summary'],
        actual_exit_code=0, run=str(folder), receipt=str(receipt_path),
        requests_sha256=checked['hashes']['requests.jsonl'])

assert csv_path.read_bytes() == before, 'This check must not alter the collected results.'
source = freeze_sources(OUT, [Path(__file__), BASE / 'packed_dflash_evidence.py',
                              BASE / 'provenance.py'])
save(OUT / 'report.json', dict(
    status='complete_timed_runs_not_complete_benchmark_cells',
    expected_timed_runs=12, verified_timed_runs=len(results),
    requests=sum(v['summary']['requests'] for v in results.values()),
    results=results, artifact_sha256=artifacts,
    plan_sha256=sha(plan_path), original_plan_sha256=sha(root / 'plan.json'),
    protocol_sha256=sha(suite / 'protocol.json'),
    jobs_sha256=sha(suite / 'jobs.json'),
    checkpoint_sha256=model['checkpoint_sha256'],
    preflight_per_timed_run=dict(requests=2, max_new_tokens=16, included_in_timing=False),
    warmup_per_timed_run=dict(requests=2, max_new_tokens=32, included_in_timing=False),
    request_seed_policy='Frozen SHA256 derivation from base seed 0, model, dataset and source_id; all recorded request seeds checked.',
    collected_cells_at_check=len(list(csv.DictReader(io.StringIO(before.decode())))),
    unchanged_combined_csv_sha256=sha(csv_path), source_sha256=source,
    new_gpu_runs=0, checked_unix=time.time(),
    limitations=[
        'Timed method measurements only. Each speedup still requires its own completed AR and profile evidence.',
        'Separate frozen numerical checks and final 72-cell collection remain required.',
        'Original BF16 outputs and all measurement values are preserved. No timing reruns or final-table publication.',
    ]))
print(dict(verified_timed_runs=len(results), new_gpu_runs=0,
           report=str(OUT / 'report.json')))
