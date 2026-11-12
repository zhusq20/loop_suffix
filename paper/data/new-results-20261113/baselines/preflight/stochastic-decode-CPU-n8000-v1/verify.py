"""Check the saved integration result and bind it to measured T1 source hashes."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'verification.json').exists()
report,plan = read(OUT/'report.json'),read(OUT/'plan.json')
prefix = BASE/'logs/stochastic-decode-CPU-n8000-v1'
launch = read(Path(str(prefix)+'-launcher.json'))
done = read(Path(str(prefix)+'-exit.json'))
assert done['exit_code'] == 0 and all(done[k] == v for k,v in launch.items())
assert launch['started_unix'] <= report['started_unix'] <= report['finished_unix'] <= done['finished_unix']
assert report['status'] == 'passed_stochastic_CPU_integration'
assert report['rejected_uncorrected_synthetic_counts'] and report['cuda_initialized'] is False
assert report['samples_per_configuration'] == plan['samples_per_configuration'] == 8000
assert plan['source_sha256'] == report['source_sha256']
assert launch['source_sha256'] == sha(OUT/'source/check_stochastic_decode.py') == sha(BASE/'check_stochastic_decode.py')
expected = {}
for length in range(1,6):
    for tokens in itertools.product(range(3),repeat=length):
        if 2 in tokens[:-1] or (length < 5 and tokens[-1] != 2):
            continue
        prefix_tokens = list(plan['prompt'])
        probability = 1.
        for token in tokens:
            probability *= plan['target'][(sum(prefix_tokens)+len(prefix_tokens))%3][token]
            prefix_tokens.append(token)
        expected[tokens] = probability
assert len(expected) == 63 and math.isclose(sum(expected.values()),1.,abs_tol=1e-14)
assert set(report['configurations']) == set(plan['configurations'])
for name,result in report['configurations'].items():
    details = result['outcomes_detail']
    assert {tuple(x['sequence']):x['expected_probability'] for x in details} == expected
    assert sum(x['count'] for x in details) == result['samples'] == 8000
    for item in details:
        assert type(item['count']) is int and 0 <= item['count'] <= 8000
        assert item['empirical_probability'] == item['count']/8000
        assert item['absolute_error'] == abs(item['empirical_probability']-item['expected_probability'])
        p = item['expected_probability'];logterm = math.log(2*63*4/1e-6)
        bound = math.sqrt(2*p*(1-p)*logterm/8000)+2*logterm/(3*8000)
        assert item['bound'] == bound and item['absolute_error'] <= bound
    counters = result['execution_counters']
    if name.startswith('loopahead'):
        assert all(counters[k] > 0 for k in ['first_proposals','second_proposals','accepted_first','accepted_second','fallback_tokens'])
        if name == 'loopahead-ouro-tight':
            assert counters['native_budget_rejections'] > 0 and counters['lad_calls'] == 0
        else:
            assert counters['lad_calls'] > 0
coverage_path = BASE/'preflight/sampling-source-coverage-partial-v1/report.json'
coverage = read(coverage_path)
assert coverage['completed_cells'] == 28
for cell in coverage['cells'].values():
    for name,digest in cell['tested_source_sha256'].items():
        assert report['source_sha256'][name] == digest == sha(BASE/name)
assert sha(BASE/'combined-results/results.csv') == coverage['unchanged_results_csv_sha256']
verification = dict(status='saved_stochastic_integration_and_source_binding_verified',
    configurations=4,sequences=32000,enumerated_outcomes_each=63,actual_exit_code=0,
    measured_T1_cells_with_identical_sampling_sources=28,source_coverage_sha256=sha(coverage_path),
    actual_exit_sha256=sha(Path(str(prefix)+'-exit.json')),report_sha256=sha(OUT/'report.json'),
    verification_source_sha256=sha(Path(__file__)),checked_unix=time.time(),new_gpu_runs=0,
    limitations=['Synthetic CPU evidence only; no real-model benchmark reruns.',
                 'These stochastic scenarios triggered no cascade commits or retention evictions; those are covered only by separate scheduler oracles.',
                 'Tiny-vocabulary finite sampling is not a universal real-model distribution proof.'])
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print({k:verification[k] for k in ['status','sequences','measured_T1_cells_with_identical_sampling_sources','new_gpu_runs']})
