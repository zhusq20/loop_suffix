"""CPU input and mocked scheduling checks; never start a CUDA process."""
import csv
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import diagnose_huginn_dflash_dev as development
import orchestrate_greedy_diagnostics as controller
import provenance

real_read, real_digest = development.read, development.digest
hashes = {}

def cached_digest(path):
    path = Path(path)
    if path not in hashes:
        hashes[path] = real_digest(path)
    return hashes[path]

prepared = {}
with patch.object(development, 'digest', cached_digest):
    for label in ['narrow', 'wide']:
        prepared[label] = development.prepare(label)

checks = []

def reject(name, operation):
    try:
        operation()
    except AssertionError:
        checks.append(name)
    else:
        raise AssertionError(f'Unexpected acceptance: {name}')

def altered_metadata(change):
    def read(path):
        value = real_read(path)
        if path == development.SOURCE/'metadata.json':
            change(value)
        return value
    return read

for name, change in [
    ('formal_budget_rejected', lambda m: m['plan'].update(max_new_tokens=512)),
    ('wrong_model_rejected', lambda m: m['plan'].update(model='ouro-2.6b')),
    ('formal_scope_rejected', lambda m: m['plan'].update(reportable=True)),
    ('wrong_target_depth_rejected', lambda m: m.update(target_depth=4)),
]:
    with patch.object(development, 'digest', cached_digest), patch.object(development, 'read', altered_metadata(change)):
        reject(name, lambda: development.prepare('narrow'))

archived_decoder = development.SOURCE/'source/dflash_impl/decode.py'
with patch.object(development, 'digest', lambda p: 'changed' if Path(p) == archived_decoder else cached_digest(p)):
    reject('changed_archived_decoder_rejected', lambda: development.prepare('narrow'))

real_rows = development.rows
with patch.object(development, 'digest', cached_digest), patch.object(development, 'rows',
        lambda p: list(reversed(real_rows(p))) if p == development.SOURCE/'inputs.jsonl' else real_rows(p)):
    reject('reordered_inputs_rejected', lambda: development.prepare('narrow'))

def wrong_case(path):
    value = real_read(path)
    if path == development.SOURCE/'report.json':
        value['comparisons']['narrow']['first_differences'][0]['position'] += 1
    return value

with patch.object(development, 'digest', cached_digest), patch.object(development, 'read', wrong_case):
    reject('wrong_first_divergence_rejected', lambda: development.prepare('narrow'))

finish = BASE/'logs/huginn-dflash-dev-compare-v1-finished.json'
def failed_exit(path):
    value = real_read(path)
    if path == finish:
        value['exit_code'] = 1
    return value

with patch.object(development, 'digest', cached_digest), patch.object(development, 'read', failed_exit):
    reject('failed_source_process_rejected', lambda: development.prepare('narrow'))

# The real current matrix is incomplete. Its gate must reject execution before
# any output directory, child command, or device acquisition is reached.
current_status = real_read(BASE/'formal-matrix-status.json')
assert current_status['completed'] < 72
with patch.object(sys, 'argv', ['orchestrate_greedy_diagnostics.py', '--execute']), \
     patch.object(controller, 'run_command', side_effect=AssertionError('Unexpected child launch')) as launch:
    reject('incomplete_matrix_gate', controller.main)
    launch.assert_not_called()

paper_row, paper_agreement, paper_identity = controller.paper_check_plan()
assert paper_identity['cases'] == 67 and paper_agreement['requests'] == 164
original_results = {n: real_digest(BASE/n) for n in ['results.csv', 'formal-agreement.json', 'formal-matrix-status.json']}
commands = []
with tempfile.TemporaryDirectory(prefix='loopfill-diagnostic-schedule-check-') as tmp:
    temp = Path(tmp)
    (temp/'preflight').mkdir()
    (temp/'formal-matrix-status.json').write_text(json.dumps(dict(completed=72, expected=72, missing=[])))
    fake_rows = [dict(method=method, model=model, benchmark=task, temperature='0',
                     engine='ours' if method == 'loopahead' else 'official',
                     ar_run=str(temp/f'inputs/ar-{model}-{task}'), method_run=str(temp/f'inputs/{method}-{model}-{task}'))
                 for method, model, task in itertools.product(['dflash', 'loopahead'],
                     ['huginn-3.5b', 'ouro-2.6b', 'ouro-1.4b'], controller.TASKS)]
    with (temp/'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fake_rows[0]))
        writer.writeheader()
        writer.writerows(fake_rows)
    (temp/'formal-agreement.json').write_text(json.dumps({
        '/'.join([r['engine'], r['model'], r['benchmark'], '0', r['method']]): {} for r in fake_rows}))

    def fake_launch(name, argv):
        commands.append(dict(name=name, argv=argv))
        out = Path(argv[argv.index('--out')+1])
        out.mkdir(parents=True, exist_ok=False)
        # A single synthetic case also exercises the FP32 dispatch branch.
        (out/'report.json').write_text(json.dumps(dict(status='complete', cases=1)))
        (out/'complete.json').write_text('{}')

    def fake_validate(row, folder, agreement):
        return dict(method=row['method'], model=row['model'], benchmark=row['benchmark'], cases=1)

    def fake_dev_validate(label, folder):
        return dict(method='dflash', model='huginn-3.5b', benchmark='heldout-huginn-dflash-dev20',
                    label=label, development_only=True, cases=8 if label == 'narrow' else 7)

    with patch.object(controller, 'BASE', temp), \
         patch.object(controller, 'paper_check_plan', return_value=(paper_row, paper_agreement, paper_identity)), \
         patch.object(controller, 'development_check_plan', return_value=[prepared[n][3] for n in ['narrow', 'wide']]), \
         patch.object(controller, 'run_command', fake_launch), \
         patch.object(controller, 'validate', fake_validate), \
         patch.object(development, 'validate', fake_dev_validate), \
         patch.object(provenance, 'freeze_sources', lambda *args: None), \
         patch.object(sys, 'argv', ['orchestrate_greedy_diagnostics.py', '--execute']):
        controller.main()
    synthetic = real_read(temp/'preflight/formal-greedy-margins/complete.json')
    assert synthetic['cells'] == len(synthetic['results']) == 36
    assert synthetic['additional_paper_check']['benchmark'] == 'paper-humaneval164'
    assert len(synthetic['additional_development_checks']) == 2
    assert sum(x['cases'] for x in synthetic['additional_development_checks']) == 15
    assert len(commands) == 58  #18 DFlash,36 LA BF16/FP32,2 B4,2 development
    assert {int(c['argv'][c['argv'].index('--gpu')+1]) for c in commands} <= {2, 3, 4, 6}
    paper_commands = [c for c in commands if '--paper-check' in c['argv']]
    assert len(paper_commands) == 2 and all('B4' in c['name'] for c in paper_commands)
    dev_commands = [c for c in commands if '--label' in c['argv']]
    assert {c['argv'][c['argv'].index('--label')+1] for c in dev_commands} == {'narrow', 'wide'}
    checks.append('mock_dispatch_preserves36_formal_B4_and_adds2_development_jobs')

assert all(real_digest(BASE/n) == expected for n, expected in original_results.items())
assert 'torch' not in sys.modules
out = Path(__file__).parent
assert not (out/'report.json').exists()
provenance.freeze_sources(out, [BASE/'diagnose_huginn_dflash_dev.py',
                              BASE/'orchestrate_greedy_diagnostics.py', Path(__file__)])
report = dict(status='cpu_inputs_and_mock_dispatch_passed_gpu_replay_pending', checked_unix=time.time(),
              development={label: dict(identity=value[3], first_divergences=[
                  {k: case[k] for k in ['source_id', 'position', 'ar_token', 'dflash_token']} for case in value[2]])
                  for label, value in prepared.items()},
              checks=checks, original_B4_cases_verified=67, mock_child_commands=len(commands),
              current_formal_cells=current_status['completed'], production_results_unchanged=True,
              torch_imported=False, actual_gpu_jobs_started=0,
              limitations=['No development GPU replay has run. Both narrow8 and wide7 cases remain pending.',
                           'Scheduling test uses synthetic records in a temporary directory and a mocked child launcher.',
                           'Real frozen data, both checkpoint files and source hashes were checked once per unique path; negative checks reuse that hash cache.',
                           'max128 development diagnostics are separate from36 formal T0 cells and B4.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(status=report['status'], checks=len(checks), cases=15, mocked_commands=len(commands))))
