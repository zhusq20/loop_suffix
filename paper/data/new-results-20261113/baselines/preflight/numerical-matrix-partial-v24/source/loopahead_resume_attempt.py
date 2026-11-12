"""One explicit continuation of the interrupted Ouro-2.6B MATH FP32 check."""
from pathlib import Path

from packed_dflash_evidence import read, rows, sha
from loopahead_partial_resume import NAME, REPORT_SHA, REUSED_CASES, EXPECTED_CASES

BASE = Path(__file__).resolve().parent
ATTEMPT = BASE / 'logs/loopahead-math-fp32-continuation-v1'
ORIGINAL = BASE / 'logs/baseline-diagnostics-resume-v1'
ORIGINAL_PLAN_SHA = 'c321d04eac2871bf2945dab3e463fb8308853cb5e37b3f4574d86664d40f4b39'
PREPARATION = BASE / 'preflight/loopahead-fp32-resume-preparation-v1'


def verify_attempt(path=ATTEMPT / 'plan.json'):
    plan = read(path)
    assert Path(path).resolve() == ATTEMPT / 'plan.json'
    assert plan['original_plan'] == str(ORIGINAL / 'plan.json')
    assert sha(plan['original_plan']) == plan['original_plan_sha256'] == ORIGINAL_PLAN_SHA
    old = next(g for g in read(plan['original_plan'])['groups'] if g['id'] == NAME)
    group = plan['group']
    assert group['id'] == NAME and group['kind'] == 'legacy'
    for key in ['row', 'agreement']:
        assert group[key] == old[key]
    assert plan['original_group'] == old
    assert group['out'] == str(BASE / 'preflight/formal-greedy-margins/loopahead-ouro-2.6b-math500-resume-v1')
    assert len(group['steps']) == 1
    step = group['steps'][0]
    assert step['id'] == NAME + '-fp32-continuation'
    assert step['out'] == str(Path(group['out']) / 'fp32-common-prefix')
    assert step['marker'] == str(Path(step['out']) / 'report.json')
    expected = list(old['steps'][1]['argv'])
    expected[2] = str(BASE / 'diagnose_trained_loopahead_resume.py')
    expected[expected.index('--out')+1] = step['out']
    expected += ['--resume-evidence', str(BASE / 'preflight/interrupted-loopahead-math-fp32-v1/report.json')]
    assert step['argv'] == expected
    assert sha(expected[-1]) == REPORT_SHA
    for original, digest in plan['artifact_sha256'].items():
        assert sha(original) == digest, original
    for name, digest in plan['source_sha256'].items():
        assert sha(BASE / name) == digest, name
        assert sha(ATTEMPT / 'source' / name) == digest, name
    assert sha(Path(group['out']) / 'report.json') == sha(Path(old['out']) / 'report.json')
    prepared = read(PREPARATION / 'inputs-report.json')
    assert (prepared['reused_cases'], prepared['pending_cases']) == (262, 167)
    for name, digest in prepared['source_sha256'].items():
        assert sha(BASE / name) == digest
    for label in ['tests', 'inputs']:
        launch, receipt = read(PREPARATION / f'{label}-launcher.json'), read(PREPARATION / f'{label}-exit.json')
        assert all(receipt[k] == v for k, v in launch.items()) and receipt['exit_code'] == 0
    proof_path = BASE / 'preflight/interrupted-loopahead-math-fp32-v1/report.json'
    proof = read(proof_path)
    assert sha(proof_path) == REPORT_SHA and proof['actual_original_exit_code'] == -15
    for original, item in proof['artifact_bindings'].items():
        assert sha(original) == item['sha256'], original
        if 'snapshot' in item:
            assert sha(proof_path.parent / item['snapshot']) == item['sha256']
    evidence = prepared['partial_evidence']
    for field in ['partial_report', 'recovered_records', 'original_log', 'original_interrupted_exit']:
        assert sha(evidence[field]) == evidence[field+'_sha256']
    assert read(evidence['original_interrupted_exit'])['exit_code'] == -15
    return plan


def verify_join(report, old_records, emitted, evidence):
    """The final report must contain the old prefix and exactly 167 new records."""
    assert report['status'] == 'complete' and report['all_fp32_same_top1'] is True
    assert report['cases'] == len(report['results']) == EXPECTED_CASES
    assert report['executed_case_count'] == len(emitted) == EXPECTED_CASES - REUSED_CASES
    assert report['resume_provenance'] == evidence
    normalized = []
    for r in old_records:
        normalized.append(dict(r, observed_bf16={key: r['observed_bf16'][key] for key in ['ar', 'loopahead']}))
    assert len(normalized) == REUSED_CASES and report['results'][:REUSED_CASES] == normalized
    for record, raw in zip(report['results'][REUSED_CASES:], emitted, strict=True):
        assert {k: v for k, v in record.items() if k not in ['common_prefix', 'observed_bf16']} == raw
    assert all(r['same_top1'] is True for r in report['results'])


def validate_completed(plan):
    from orchestrate_greedy_diagnostics import validate
    group = plan['group']
    folder = Path(group['out'])
    step = group['steps'][0]
    log = ATTEMPT / f'logs/{step["id"]}.log'
    # Model loader messages are not JSON case records; preserve the raw log too.
    import json
    emitted = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"source_id":')]
    evidence = read(PREPARATION / 'inputs-report.json')['partial_evidence']
    verify_join(read(folder / 'fp32-common-prefix/report.json'), rows(evidence['recovered_records']), emitted, evidence)
    return validate(group['row'], folder, group['agreement'])


def audit_resumed(queue, old_plan, old_group, canonical, check_receipt, inspect_cases):
    """Audit both actual successful stages and preserve the interrupted attempt."""
    assert queue == ORIGINAL and old_group['id'] == NAME
    plan = verify_attempt()
    assert plan['original_group'] == old_group
    group, old_step = plan['group'], old_group['steps'][0]
    step = group['steps'][0]
    old_paths = [ORIGINAL / f'logs/{old_step["id"]}-{suffix}.json' for suffix in ['launcher', 'exit']]
    new_paths = [ATTEMPT / f'logs/{step["id"]}-{suffix}.json' for suffix in ['launcher', 'exit']]
    check_receipt(read(old_paths[0]), read(old_paths[1]), old_step, old_group, ORIGINAL_PLAN_SHA, False)
    if new_paths[1].exists():
        check_receipt(read(new_paths[0]), read(new_paths[1]), step, group, sha(ATTEMPT / 'plan.json'), False)
    missing = [str(p) for p in new_paths if not p.exists()]
    if missing:
        return dict(status='pending_actual_exit', missing=missing,
            continuation_plan=str(ATTEMPT / 'plan.json'), continuation_plan_sha256=sha(ATTEMPT / 'plan.json'),
            preserved_interrupted_exit=-15, reused_cases=262, remaining_cases=167)
    artifacts, exits = dict(plan['artifact_sha256']), []
    artifacts[str(ATTEMPT / 'plan.json')] = sha(ATTEMPT / 'plan.json')
    for paths, current in [(old_paths, old_step), (new_paths, step)]:
        launch, receipt = map(read, paths)
        marker = Path(current['marker'])
        assert receipt['started_unix'] <= read(marker)['finished_unix'] <= receipt['finished_unix']
        for p in [*paths, marker]: artifacts[str(p)] = sha(p)
        exits.append(dict(path=str(paths[1]), sha256=sha(paths[1]), exit_code=0))
        manifest = Path(current['out']) / 'source-manifest.json'
        artifacts[str(manifest)] = sha(manifest)
        for name, digest in read(manifest).items():
            p = manifest.parent / 'source' / name
            assert sha(p) == digest
            artifacts[str(p)] = digest
            if current is step:
                assert plan['source_sha256'][name] == digest
    log = ATTEMPT / f'logs/{step["id"]}.log'
    artifacts[str(log)] = sha(log)
    validated = validate_completed(plan)
    folder = Path(group['out'])
    meta, fp32 = read(folder / 'report.json'), read(folder / 'fp32-common-prefix/report.json')
    artifacts[str(folder / 'report.json')] = sha(folder / 'report.json')
    originals = {}
    for label, record in meta['source_journals'].items():
        p = Path(record['path']) / 'requests.jsonl'
        assert sha(p) == record['sha256']
        artifacts[str(p)] = sha(p)
        originals['ar' if label == 'ar' else 'method'] = rows(p)
    summary = inspect_cases('loopahead', meta['results'], fp32['results'], originals)
    assert summary['first_divergences'] == validated['cases'] == 429
    scope = 'formal/loopahead/ouro-2.6b/math500'
    row = canonical[scope]
    for name in ['eta_latency', 'eta_throughput', 'gamma', 'match_rate', 'method_decode_s', 'ar_decode_s', 'ar_run', 'method_run']:
        assert row[name] == group['row'][name]
    assert float(row['match_rate']) == summary['whole_sequence_match_rate_bf16']
    measurement = read(Path(row['method_run']) / 'metadata.json')
    assert sha(measurement['checkpoint']) == meta['checkpoint_sha256']
    return dict(status='validated_actual_diagnostic', scope=scope, group=group['id'], **summary,
        queue_plan=str(ATTEMPT / 'plan.json'), queue_plan_sha256=sha(ATTEMPT / 'plan.json'),
        actual_exits=exits, artifact_sha256=artifacts, existing_validation=validated,
        continuation_provenance=fp32['resume_provenance'], newly_executed_cases=167,
        original_interrupted_exit_code=-15,
        limitation='The original FP32 attempt remains interrupted; 262 emitted records are reused and 167 cases belong to a separately supervised continuation.')
