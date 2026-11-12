"""Check collection from explicit archived inputs against the original rows."""
from pathlib import Path
import csv
import hashlib
import json
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'report.json').exists()
before = read(OUT/'before.json')['files']
for name,record in before.items():
    assert sha(OUT/record['snapshot']) == record['sha256']
    if name != 'collect_combined_results.py':
        assert sha(BASE/name) == record['sha256']
launch,done = read(OUT/'launcher.json'),read(OUT/'exit.json')
assert done['exit_code'] == 0 and all(done[k] == v for k,v in launch.items())
assert sha(BASE/'collect_combined_results.py') == launch['source_sha256']
new = OUT/'replayed'
original_csv = OUT/'before-combined-results-results.csv'
with original_csv.open() as stream:
    original = list(csv.DictReader(stream))
with (new/'results.csv').open() as stream:
    replayed = list(csv.DictReader(stream))
assert len(original) == len(replayed) == 58 and original == replayed
assert sha(original_csv) == sha(new/'results.csv')
assert (BASE/'combined-results/table2_rows.md').read_bytes() == (new/'table2_rows.md').read_bytes()
old_evidence,new_evidence = read(BASE/'combined-results/evidence.json'),read(new/'evidence.json')
collection = read(BASE/'preflight/combined-results-cell58-v1/report.json')
assert sha(BASE/'combined-results/evidence.json') == collection['evidence_sha256']
assert old_evidence['cells'] == new_evidence['cells']
source = new_evidence['legacy_source']
assert source['rows'] == 46
assert Path(source['path']) == OUT/'before-results.csv'
assert Path(source['agreement_path']) == OUT/'before-formal-agreement.json'
assert source['sha256'] == sha(OUT/'before-results.csv')
assert source['agreement_sha256'] == sha(OUT/'before-formal-agreement.json')
assert new_evidence['collector_sha256'] == launch['source_sha256']
status = read(new/'status.json')
assert status['completed'] == 58 and status['expected'] == 72
assert status['results_sha256'] == sha(new/'results.csv')
assert launch['started_unix'] <= status['updated_unix'] <= done['finished_unix']
source_dir = OUT/'source'
source_dir.mkdir()
for name in ['collect_combined_results.py','collect_results.py','packed_dflash_evidence.py']:
    (source_dir/name).write_bytes((BASE/name).read_bytes())
report = dict(status='explicit_legacy_inputs_reproduce_existing_collection',actual_exit_code=0,
    combined_cells=58,legacy_cells=46,packed_cells=12,all_existing_rows_byte_identical=True,
    all_cell_evidence_unchanged=True,source_results_unchanged=True,new_gpu_runs=0,
    before_source_sha256=before['collect_combined_results.py']['sha256'],
    after_source_sha256=launch['source_sha256'],
    archived_input_sha256={name:sha(OUT/name) for name in ['before-results.csv','before-formal-agreement.json']},
    actual_exit_sha256=sha(OUT/'exit.json'),verification_sha256=sha(Path(__file__)),
    output_sha256={name:sha(new/name) for name in ['results.csv','table2_rows.md','evidence.json','status.json']},
    checked_unix=time.time(),
    limitation='This is a 58-cell reproducibility check. The final 72-cell publication must preserve its own complete legacy inputs first.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:report[k] for k in ['status','combined_cells','all_existing_rows_byte_identical','new_gpu_runs']})
