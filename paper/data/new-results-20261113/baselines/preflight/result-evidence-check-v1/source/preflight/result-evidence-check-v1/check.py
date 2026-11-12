"""CPU checks for artifact admission; temporary output only, no GPU work."""
import copy
import csv
import io
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
import collect_results as collector
import result_evidence as evidence
from provenance import freeze_sources

protocol = evidence.read(BASE.parents[1]/'results/loopfill-latency512-20261107/protocol.json')
checked = {}
for marker in sorted((BASE/'runs').glob('*/complete.json')):
    meta = evidence.read(marker.parent/'metadata.json')
    run = collector.read_run(marker.parent, protocol, meta.get('profiling', False))
    checked[marker.parent.name] = evidence.validate_run_artifacts(run)

leaf = BASE/'runs/loopahead-huginn-3.5b-gsm8k-t1-v1'
run = collector.read_run(leaf, protocol)
finish = BASE/f'logs/{leaf.name}-finished.json'
real_read = evidence.read
tests = []

def must_reject(name, action, error=AssertionError):
    try:
        action()
    except error:
        tests.append(name)
    else:
        raise AssertionError(f'Accepted invalid evidence: {name}')

def bad_receipt(field, value):
    def read(path):
        result = real_read(path)
        if path == finish:
            result[field] = value
        return result
    return read

for field, value in [('exit_code', 1), ('pid', -1), ('finished_unix', 0)]:
    with patch.object(evidence, 'read', bad_receipt(field, value)):
        must_reject('reject_receipt_'+field, lambda: evidence.validate_run_artifacts(run))

exists = Path.exists
with patch.object(Path, 'exists', lambda path: False if path == finish else exists(path)):
    must_reject('await_missing_exit_receipt', lambda: evidence.validate_run_artifacts(run), evidence.ExitReceiptPending)

for name, mutate in [
    ('wrong_revision', lambda r: r['meta'].update(model_revision='wrong')),
    ('wrong_checkpoint', lambda r: r['meta'].update(checkpoint='/tmp/wrong-checkpoint/lad.pt')),
    ('profile_flag_mismatch', lambda r: r['meta'].update(profiling=True)),
    ('wrong_temperature_source', lambda r: r['meta'].update(frozen_source_sha256=protocol['source_sha256'])),
    ('negative_request_latency', lambda r: r['rows'][0]['stats'].update(decode_s=-1)),
    ('wrong_request_token_count', lambda r: r['rows'][0]['stats'].update(tokens=-1)),
]:
    altered = copy.deepcopy(run)
    mutate(altered)
    must_reject(name, lambda: evidence.validate_run_artifacts(altered))

ar = collector.read_run(BASE/'runs/archived-ar-huginn-3.5b-bbh-t1', protocol)
real_sha = evidence.sha
original_request = Path(ar['meta']['measurement_provenance']['source'])/'requests.jsonl'
with patch.object(evidence, 'sha', lambda path: 'changed' if path == original_request else real_sha(path)):
    must_reject('changed_archived_AR_bytes', lambda: evidence.validate_run_artifacts(ar))

original_validate = evidence.validate_run_artifacts
original_outputs = {name: (BASE/name).read_bytes() for name in
                    ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']}

def keyed(text):
    return {tuple(r[k] for k in ['method', 'model', 'benchmark', 'temperature']): r
            for r in csv.DictReader(io.StringIO(text))}

with tempfile.TemporaryDirectory(prefix='loopfill-result-evidence-') as tmp:
    temp = Path(tmp)
    (temp/'data').symlink_to(BASE/'data', target_is_directory=True)
    with patch.object(collector, 'BASE', temp), patch.object(sys, 'argv',
            ['collect_results.py', '--runs', str(BASE/'runs'), '--partial']):
        collector.main()
    actual = keyed((temp/'results.csv').read_text())
    old = keyed(original_outputs['results.csv'].decode())
    assert all(actual[k] == v for k, v in old.items())
    old_agreement = json.loads(original_outputs['formal-agreement.json'])
    new_agreement = evidence.read(temp/'formal-agreement.json')
    assert all(new_agreement[k] == v for k, v in old_agreement.items())
    tests.append('existing_rows_and_agreements_unchanged')

    def pending_one(item):
        if item['path'] == leaf:
            raise evidence.ExitReceiptPending('simulated launcher wait')
        return original_validate(item)

    with patch.object(collector, 'BASE', temp), patch.object(sys, 'argv',
            ['collect_results.py', '--runs', str(BASE/'runs'), '--partial']), \
         patch.object(evidence, 'validate_run_artifacts', pending_one):
        collector.main()
    skipped = keyed((temp/'results.csv').read_text())
    omitted = ('loopahead', 'huginn-3.5b', 'gsm8k', '1')
    assert omitted not in skipped and all(skipped[k] == v for k, v in actual.items() if k != omitted)
    tests.append('partial_collection_does_not_promote_missing_exit')

    with patch.object(collector, 'BASE', temp), patch.object(sys, 'argv',
            ['collect_results.py', '--runs', str(BASE/'runs')]), \
         patch.object(evidence, 'validate_run_artifacts', pending_one):
        must_reject('strict_collection_requires_exit_receipt', collector.main, evidence.ExitReceiptPending)

assert all((BASE/name).read_bytes() == raw for name, raw in original_outputs.items())
assert 'torch' not in sys.modules, 'Artifact checks must not import Torch'
out = Path(__file__).parent
assert not (out/'report.json').exists()
freeze_sources(out, [BASE/'collect_results.py', BASE/'result_evidence.py', Path(__file__)])
report = dict(status='cpu_artifact_admission_checks_passed', checked_unix=time.time(),
              completed_runs_checked=checked, tests=tests, production_outputs_unchanged=True,
              existing_formal_rows_verified=len(old), torch_imported=False,
              development_correction='Initial read-only check assumed T0 frozen sources for T1. Corrected to the existing temperature-specific source suite; no experiment or result was changed.',
              limitations=['Metadata, byte identity and actual exit evidence only; not a numerical or sampling-distribution proof.',
                           'Historical AR remains an archived measurement, with uncorrected cross-date timing differences.'])
(out/'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(status=report['status'], runs=len(checked), tests=len(tests),
                     existing_rows=len(old), outputs_unchanged=True)))
