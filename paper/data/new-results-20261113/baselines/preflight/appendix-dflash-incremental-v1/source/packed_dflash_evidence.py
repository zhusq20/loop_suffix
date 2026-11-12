"""Evidence checks for supplemental AR/profile runs of the frozen packed suite."""
import hashlib
import json
import math
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def check_plan(path):
    path = Path(path)
    plan = read(path)
    for name, expected in plan['source_sha256'].items():
        assert sha(path.parent / 'source' / name) == expected, name
    suite = Path(plan['suite'])
    assert sha(suite / 'protocol.json') == plan['protocol_sha256']
    assert sha(suite / 'jobs.json') == plan['jobs_sha256']
    return plan


def check_frozen(suite, job, *, weights=False):
    suite = Path(suite)
    protocol = read(suite / 'protocol.json')
    assert job in read(suite / 'jobs.json')
    assert job['kind'] == 'latency' and job['batch'] == 1
    assert job['max_new_tokens'] == 512 and job['temperature'] in (0, 1)
    assert not protocol['graph'] and not protocol['compile']
    assert protocol['dtype'] == 'bfloat16'
    for name, expected in protocol['source_sha256'].items():
        assert sha(suite / 'source' / name) == expected, name
    assert sha(suite / job['inputs']) == job['inputs_sha256']
    inputs = rows(suite / job['inputs'])
    assert len(inputs) == job['count'] <= 500
    assert len({r['source_id'] for r in inputs}) == len(inputs)
    model = protocol['models'][job['model']]
    if weights:
        for name, expected in model['checkpoint_sha256'].items():
            assert sha(Path(model['checkpoint']) / name) == expected, name
    return protocol, inputs


def totals(records, job, *, profile=False, ar=False):
    assert len(records) == job['count']
    for r in records:
        s = r['stats']
        assert 1 <= len(r['tokens']) <= job['max_new_tokens']
        assert s['decode_tokens'] == len(r['tokens']) - 1 == s['output_tokens'] - 1
        assert sum(s['emitted_per_verification']) == s['decode_tokens']
        assert len(s['emitted_per_verification']) == s['target_verifications']
        assert s['loop_steps'] == s['target_verifications'] * (32 if job['model'] == 'huginn-3.5b' else 4)
        assert s['temperature'] == job['temperature'] and s['top_p'] == 1. and s['top_k'] == 0
        assert not s['graph'] and not s['diagnostic_trace'] and s['diagnostic_stop_after'] is None
        assert s['instrumented_draft_profile'] == profile
        assert s['stop_reason'] == r['stop_reason']
        assert math.isfinite(s['decode_s']) and s['decode_s'] > 0
        if ar:
            assert s['target_verifications'] == s['decode_tokens']
        if profile:
            assert math.isfinite(s['draft_gpu_s']) and 0 <= s['draft_gpu_s'] <= s['decode_s']
        else:
            assert s['draft_gpu_s'] is None
    n = sum(r['stats']['decode_tokens'] for r in records)
    seconds = sum(r['stats']['decode_s'] for r in records)
    calls = sum(r['stats']['target_verifications'] for r in records)
    assert n > 0 and calls > 0
    result = dict(requests=len(records), decode_tokens=n, decode_s=seconds,
                  decode_token_s=n / seconds, gamma=n / calls,
                  output_tokens=sum(len(r['tokens']) for r in records), target_verifications=calls,
                  cap_stopped=sum(r['stop_reason'] == 'length' for r in records))
    if profile:
        result['draft_gpu_s'] = sum(r['stats']['draft_gpu_s'] for r in records)
        result['drafter_time_frac'] = result['draft_gpu_s'] / seconds
    return result


def check_run(folder, suite, job, *, stage=None):
    folder, suite = Path(folder), Path(suite)
    protocol, inputs = check_frozen(suite, job)
    meta, done = read(folder / 'metadata.json'), read(folder / 'complete.json')
    assert meta['job'] == job
    assert meta['source_sha256'] == protocol['source_sha256']
    model = protocol['models'][job['model']]
    assert meta['checkpoint'] == model['checkpoint']
    assert meta['checkpoint_sha256'] == model['checkpoint_sha256']
    assert not meta['graph'] and not meta['compile']
    assert meta['target_dtype'] == 'torch.bfloat16' and len(meta['cpu_affinity']) == 4
    assert sha(folder / 'requests.jsonl') == done['requests_sha256']
    assert sha(folder / 'summary.json') == done['summary_sha256']
    records = rows(folder / 'requests.jsonl')
    assert done['requests'] == len(records) == len(inputs)
    assert [r['source_id'] for r in records] == [r['source_id'] for r in inputs]
    assert [r['prompt_sha256'] for r in records] == [r['prompt_sha256'] for r in inputs]
    if stage is not None:
        assert meta['stage'] == stage
        assert meta['reportable'] == (stage == 'ar')
        assert meta['profiling'] == (stage == 'profile')
        assert meta['target_backend'] == 'PackedDFlashTarget'
    else:
        assert meta['reportable']
    summary = totals(records, job, profile=stage == 'profile', ar=stage == 'ar')
    saved = read(folder / 'summary.json')
    for key, value in summary.items():
        assert saved[key] == value, key
    return dict(path=str(folder), metadata=meta, rows=records, summary=summary,
                hashes={name: sha(folder / name) for name in
                        ['metadata.json', 'complete.json', 'requests.jsonl', 'summary.json']})


def timed_evidence(suite, dispatch, job):
    state = dispatch.get(job['id'], {})
    if state.get('outcome') not in ('complete', 'complete_artifact_exit_unavailable'):
        return None
    run = check_run(Path(suite) / state['out'], suite, job)
    if state['outcome'] == 'complete':
        receipt_path = Path(suite) / 'logs' / f"{job['id']}-attempt{state['attempt']}-exit.json"
        receipt = read(receipt_path)
        assert receipt['exit_code'] == 0
        for key in ('pid', 'argv', 'started_unix'):
            assert receipt[key] == state[key], key
        assert receipt['finished_unix'] >= read(Path(run['path']) / 'complete.json')['finished_unix']
        exit_evidence = dict(path=str(receipt_path), sha256=sha(receipt_path), actual_exit_code=0)
    else:
        assert state.get('exit_code') is None
        assert state['preserved_artifact_audit'] == 'passed'
        exit_evidence = dict(actual_exit_code=None,
            limitation='Original parent exit receipt unavailable; artifacts verified independently. No synthetic exit status and no timed rerun.')
    return dict(path=run['path'], hashes=run['hashes'], exit_evidence=exit_evidence)


def compare(method, ar):
    exact = matched = denominator = 0
    divergences = []
    for m, a in zip(method, ar, strict=True):
        assert m['source_id'] == a['source_id'] and m['prompt_sha256'] == a['prompt_sha256']
        x, y = m['tokens'], a['tokens']
        exact += x == y
        matched += sum(u == v for u, v in zip(x, y))
        denominator += max(len(x), len(y))
        if x != y:
            position = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            divergences.append(dict(source_id=m['source_id'], position=position,
                                    method_length=len(x), ar_length=len(y)))
    return dict(match_rate=exact / len(method), aligned_token_match_rate=matched / denominator,
                exact_sequences=exact, requests=len(method), first_divergences=divergences)


def check_profile(profile, method):
    for p, m in zip(profile, method, strict=True):
        assert p['source_id'] == m['source_id'] and p['tokens'] == m['tokens'], p['source_id']
        for key in ('seed', 'target_verifications', 'accepted_draft_tokens', 'emitted_per_verification', 'stop_reason'):
            assert p['stats'][key] == m['stats'][key], (p['source_id'], key)

