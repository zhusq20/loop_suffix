"""Independently recompute the frozen CPU sample counts and simultaneous bounds."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


plan, report = read(OUT / 'plan.json'), read(OUT / 'report.json')
receipt = read(BASE / 'logs' / (OUT.name + '-exit.json'))
assert receipt['exit_code'] == 0 and report['status'] == 'passed'
assert receipt['started_unix'] <= report['started_unix'] <= report['finished_unix'] <= receipt['finished_unix']
assert receipt['source_sha256'] == plan['source_sha256']['check_loopahead_retention_sampling.py']
assert plan['source_sha256'] == report['source_sha256'] == read(OUT / 'source-manifest.json')
for name, digest in plan['source_sha256'].items():
    assert sha(OUT / 'source' / name) == sha(BASE / name) == digest
engine = BASE.parents[1] / 'results/loopfill-latency512-20261107/source/src/llm_memory_editability'
assert plan['engine_sha256'] == report['engine_sha256']
for name, digest in plan['engine_sha256'].items():
    assert sha(engine / name) == digest
assert report['cuda_initialized'] is False and report['new_gpu_runs'] == 0
assert plan['samples_per_configuration'] == report['samples_per_configuration'] == 32000
assert plan['max_new_tokens'] == 12 and plan['familywise_alpha'] == 1e-6
assert plan['measured_path_coverage'] == report['measured_path_coverage']
assert sha(BASE / 'combined-results/results.csv') == plan['measured_path_coverage']['results_sha256']

expected = {}
frontier = [((), 1.)]
while frontier:
    sequence, probability = frontier.pop()
    if len(sequence) == 12 or (sequence and sequence[-1] == plan['eos']):
        expected[sequence] = probability
        continue
    context = tuple(plan['prompt']) + sequence
    probabilities = plan['target'][(sum(context) + len(context)) % len(plan['target'])]
    assert math.isclose(sum(probabilities), 1., abs_tol=1e-15)
    for token, mass in enumerate(probabilities):
        frontier.append((sequence + (token,), probability * mass))
assert len(expected) == 8191 and math.isclose(sum(expected.values()), 1., abs_tol=1e-13)


def aggregate(histogram):
    result = Counter()
    for sequence, weight in histogram.items():
        result['full/' + ','.join(map(str, sequence))] += weight
        for length in range(1, min(4, len(sequence)) + 1):
            result['prefix/' + ','.join(map(str, sequence[:length]))] += weight
        for position in range(12):
            token = sequence[position] if position < len(sequence) else -1
            result[f'position/{position}/{token}'] += weight
        result[f'length/{len(sequence)}'] += weight
    return result


probabilities = aggregate(expected)
assert len(probabilities) == plan['events_per_configuration'] == 8295
assert set(report['configurations']) == set(plan['configurations'])
logterm = math.log(2 * 8295 * len(plan['configurations']) / 1e-6)
summary = {}
for name, result in report['configurations'].items():
    assert result == read(OUT / (name + '.json'))
    counts = {tuple(row['sequence']): row['count'] for row in result['sequence_counts']}
    assert len(counts) == len(result['sequence_counts']) and sum(counts.values()) == 32000
    assert set(counts) <= set(expected) and all(type(n) is int and n > 0 for n in counts.values())
    observed = aggregate(counts)
    rows = {row['event']: row for row in result['details']}
    assert len(rows) == len(result['details']) == result['events'] == 8295
    assert rows.keys() == probabilities.keys()
    for event, probability in probabilities.items():
        row = rows[event]
        bound = math.sqrt(2 * probability * (1 - probability) * logterm / 32000) + 2 * logterm / (3 * 32000)
        error = abs(observed[event] / 32000 - probability)
        assert row['count'] == observed[event]
        assert math.isclose(row['expected_probability'], probability, abs_tol=1e-13)
        assert math.isclose(row['bound'], bound, abs_tol=1e-13)
        assert math.isclose(row['absolute_error'], error, abs_tol=1e-13)
        assert row['passed'] and error <= bound
    counters = result['execution_counters']
    assert all(type(n) is int and n >= 0 for n in counters.values())
    assert counters['retention_evictions'] > 0 and counters['cascade_commits'] > 0
    assert result['passed'] and result['coverage_passed'] and not result['violations']
    maximum = max(abs(observed[key] / 32000 - probability) for key, probability in probabilities.items())
    assert math.isclose(result['max_absolute_error'], maximum, abs_tol=1e-13)
    summary[name] = dict(max_event_absolute_error=maximum,
        retention_evictions=counters['retention_evictions'], cascade_commits=counters['cascade_commits'])

bad = aggregate({(plan['eos'],): 32000})
assert any(abs(bad[key] / 32000 - probability) >
           math.sqrt(2 * probability * (1 - probability) * logterm / 32000) + 2 * logterm / (3 * 32000)
           for key, probability in probabilities.items())

measured, totals = {}, Counter()
for cell, record in plan['measured_path_coverage']['cells'].items():
    path = Path(record['requests_path'])
    assert sha(path) == record['requests_sha256']
    counts = Counter()
    for line in path.read_text().splitlines():
        stats = json.loads(line)['stats']
        counts['requests'] += 1
        for field in ['retention_evictions', 'cascade_commits']:
            counts[field] += stats[field]
            counts[field + '_requests'] += stats[field] > 0
    assert dict(counts) == record['counters']
    manifest = path.parent / 'source-manifest.json'
    source = read(manifest)
    for name in ['loopahead/decode.py', 'loopahead/sampling.py', 'loopahead/admission.py']:
        assert source[name] == report['source_sha256'][name]
        assert sha(path.parent / 'source' / name) == source[name]
    measured[cell] = dict(requests_sha256=sha(path), source_manifest_sha256=sha(manifest))
    totals.update(counts)
assert len(measured) == 18 and totals['requests'] == 6276

result = dict(status='independently_recomputed', configurations=summary,
    synthetic_sequences=96000, exact_terminal_sequences=8191, events_per_configuration=8295,
    negative_control_rejected=True, measured_LoopAhead_T1_cells=18,
    measured_path_counts=dict(totals), measured_artifacts=measured,
    source_sha256=report['source_sha256'], report_sha256=sha(OUT / 'report.json'),
    plan_sha256=sha(OUT / 'plan.json'), actual_exit_sha256=sha(BASE / 'logs' / (OUT.name + '-exit.json')),
    verifier_sha256=sha(__file__), verified_unix=time.time(), new_gpu_runs=0,
    limitations=['Finite synthetic sampling evidence, not a universal distribution proof.',
                 'Real target numerical differences and correctness of model tensor operations remain separately scoped.'])
(OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status=result['status'], synthetic_sequences=96000, configurations=summary)))
