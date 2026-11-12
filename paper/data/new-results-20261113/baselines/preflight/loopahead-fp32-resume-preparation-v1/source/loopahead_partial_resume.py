"""Strictly bind one interrupted Ouro MATH diagnostic to its preserved prefix.

This module is CPU-only. It does not turn an interrupted attempt into exit 0.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPORT_SHA = '5853743677d22273d43d495e4c1ffab2eabff4af9c1ccd97a0cb658f4aef5bc7'
ORIGINAL_SOURCE_SHA = '8e4c18ebc54ef0c0b1475ead9573ec80ca4da24c8af18c670231d28526addc7f'
NAME = 'loopahead-ouro-2.6b-math500-v1'
EXPECTED_CASES, REUSED_CASES = 429, 262


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_partial(path, args):
    """Accept only the audited, immutable 262-record interruption capsule."""
    path = Path(path).resolve()
    assert sha(path) == REPORT_SHA, 'Unknown or changed partial-evidence capsule'
    root, proof = path.parent, read(path)
    assert proof['status'] == 'interrupted_partial_stdout_preserved_not_complete_diagnostic'
    assert (proof['expected_cases'], proof['recovered_cases'], proof['remaining_cases']) == (429, 262, 167)
    assert proof['actual_original_exit_code'] == -15
    assert proof['counts_toward_completed_numerical_groups'] is False
    assert not args.paper_check
    assert sha(BASE / 'diagnose_trained_loopahead.py') == ORIGINAL_SOURCE_SHA
    recovery = read(root / 'exit.json')
    assert recovery['exit_code'] == 0 and recovery['exit_evidence'] == 'subprocess_wait'
    assert recovery['source_sha256'] == proof['source_sha256'] == sha(root / 'recover.py')
    assert recovery['started_unix'] <= proof['checked_unix'] <= recovery['finished_unix']
    for original, item in proof['artifact_bindings'].items():
        assert sha(original) == item['sha256'], original
        if 'snapshot' in item:
            assert sha(root / item['snapshot']) == item['sha256']
    logs = BASE / 'logs/baseline-diagnostics-resume-v1/logs'
    launch_path = logs / f'{NAME}-fp32-launcher.json'
    exit_path = logs / f'{NAME}-fp32-exit.json'
    launch, stopped = read(launch_path), read(exit_path)
    assert stopped['exit_code'] == -15 and stopped['exit_evidence'] == 'supervisor_waitpid'
    assert all(stopped[k] == value for k, value in launch.items())
    original_argv = stopped['argv']
    for flag, supplied in [('--ar', args.ar), ('--loopahead', args.loopahead),
                           ('--observed-margin-report', args.observed_margin_report)]:
        assert original_argv.count(flag) == 1
        assert Path(original_argv[original_argv.index(flag)+1]).resolve() == Path(supplied).resolve()
    old_out = Path(original_argv[original_argv.index('--out')+1]).resolve()
    assert not (old_out / 'report.json').exists(), 'A completed original report must be inspected and reused.'
    assert not Path(args.out).exists(), 'Use a new attempt directory.'
    assert Path(args.out).resolve() != old_out and old_out not in Path(args.out).resolve().parents
    assert sha(old_out / 'source/diagnose_trained_loopahead.py') == ORIGINAL_SOURCE_SHA
    for name, digest in read(old_out / 'source-manifest.json').items():
        assert sha(old_out / 'source' / name) == digest
        assert sha(BASE / name) == digest, 'Resume requires the original implementation: '+name
    recovered_path = root / 'recovered-records.partial.jsonl'
    assert sha(recovered_path) == proof['recovered_records_sha256']
    records = [json.loads(line) for line in recovered_path.read_text().splitlines()]
    log_path = logs / f'{NAME}-fp32.log'
    emitted = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == len(emitted) == REUSED_CASES
    for record, raw in zip(records, emitted, strict=True):
        assert {k: v for k, v in record.items() if k not in ['common_prefix', 'observed_bf16']} == raw
        assert record['same_top1'] is True
        assert record['serial']['top_ids'][0] == record['loopahead']['top_ids'][0]
        assert math.isfinite(record['max_abs_logit_difference']) and record['max_abs_logit_difference'] >= 0
        # The archive retains the whole BF16 record. Restore the original run schema.
        record['observed_bf16'] = {label: record['observed_bf16'][label] for label in ['ar', 'loopahead']}
    return dict(records=records, evidence=dict(
        partial_report=str(path), partial_report_sha256=sha(path),
        recovered_records=str(recovered_path), recovered_records_sha256=sha(recovered_path),
        original_log=str(log_path), original_log_sha256=sha(log_path),
        original_interrupted_exit=str(exit_path), original_interrupted_exit_sha256=sha(exit_path),
        original_interrupted_exit_code=-15, original_source_sha256=ORIGINAL_SOURCE_SHA,
        reused_cases=REUSED_CASES, expected_cases=EXPECTED_CASES,
        limitation='The original attempt exited -15. Only its emitted, source-bound records are reused; this attempt must execute every remaining case and actually exit 0.'))


def reconcile(cases, partial):
    """Require an exact contiguous prefix, including BF16 traces and forced tokens."""
    assert len(cases) == EXPECTED_CASES
    records = partial['records']
    assert len(records) == REUSED_CASES and 0 < len(records) < len(cases)
    assert len({(c['source_id'], c['position']) for c in cases}) == len(cases)
    for case, record in zip(cases[:len(records)], records, strict=True):
        for key in ['source_id', 'position', 'common_prefix', 'observed_bf16', 'ar_token', 'loopahead_token']:
            assert record[key] == case[key], key
    return copy.deepcopy(records), cases[len(records):]
