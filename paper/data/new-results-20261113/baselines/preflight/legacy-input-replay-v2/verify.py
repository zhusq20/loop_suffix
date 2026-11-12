"""Verify the separate archive reader without changing the frozen live reader."""
import hashlib
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
OLD = OUT.with_name('legacy-input-replay-v1')
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'report.json').exists()
launch, receipt = read(OUT/'launcher.json'), read(OUT/'exit.json')
assert receipt['exit_code'] == 0
assert all(receipt[k] == v for k, v in launch.items())
assert sha(BASE/'collect_combined_results_archived.py') == launch['source_sha256']
assert sha(OUT/'source/collect_combined_results_archived.py') == launch['source_sha256']
before = read(OUT/'before.json')
assert all(sha(BASE/name) == digest for name, digest in before.items())
spec = read(BASE/'logs/appendix-dflash-diagnostics-v1/spec.json')
assert sha(BASE/'collect_combined_results.py') == spec['base_sha256']['collect_combined_results.py']
for name in ['results.csv', 'table2_rows.md']:
    assert (OUT/'replayed'/name).read_bytes() == (OLD/'replayed'/name).read_bytes()
prior, current = read(OLD/'replayed/evidence.json'), read(OUT/'replayed/evidence.json')
assert prior['cells'] == current['cells']
assert prior['legacy_source'] == current['legacy_source']
assert current['collector_sha256'] == launch['source_sha256']
assert sha(OLD/'before-results.csv') == current['legacy_source']['sha256']
assert sha(OLD/'before-formal-agreement.json') == current['legacy_source']['agreement_sha256']
status = read(OUT/'replayed/status.json')
assert status['completed'] == 58 and status['expected'] == 72
assert status['results_sha256'] == sha(OUT/'replayed/results.csv')
assert launch['started_unix'] <= status['updated_unix'] <= receipt['finished_unix']
report = dict(status='passed', actual_exit_code=0, combined_cells=58,
    legacy_cells=46, new_gpu_runs=0, all_existing_rows_byte_identical=True,
    cell_evidence_unchanged=True, live_inputs_and_tables_unchanged=True,
    pinned_collector_restored=True, pinned_spec_unchanged=True,
    archive_reader=str(BASE/'collect_combined_results_archived.py'),
    archive_reader_sha256=launch['source_sha256'],
    actual_exit_sha256=sha(OUT/'exit.json'), verification_sha256=sha(Path(__file__)),
    output_sha256={n:sha(OUT/'replayed'/n) for n in ['results.csv','table2_rows.md','evidence.json','status.json']},
    checked_unix=time.time(),
    limitation='Replays the preserved 58-cell snapshot only; final publication still requires complete legacy input archives.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:report[k] for k in ['status','combined_cells','pinned_collector_restored','new_gpu_runs']})
