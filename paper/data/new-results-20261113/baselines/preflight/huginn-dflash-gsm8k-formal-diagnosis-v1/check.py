"""Bind all 76 completed formal first-divergence checks to the original run."""
import csv
import math
from pathlib import Path
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, rows, save, sha
from orchestrate_greedy_diagnostics import validate

queue = BASE/'logs/baseline-diagnostics-resume-v1'
plan = read(queue/'plan.json')
identity = 'dflash-huginn-3.5b-gsm8k-v1'
group = next(g for g in plan['groups'] if g['id'] == identity)
folder = Path(group['out'])
marker = read(folder/'complete.json')
assert marker['cases'] == 76 and marker['all_bf16_divergences_reproduced']
assert marker['all_fp32_argmax_agree']
for name, digest in plan['source_sha256'].items():
    assert sha(queue/'source'/name) == digest, name
for name, digest in plan['legacy_source_sha256'].items():
    assert sha(BASE/name) == digest, name
source_bundle = read(folder/'source-manifest.json')
for name, digest in source_bundle.items():
    assert sha(folder/'source'/name) == digest, name
step = group['steps'][0]
assert len(group['steps']) == 1
launch_path = queue/f'logs/{step["id"]}-launcher.json'
exit_path = queue/f'logs/{step["id"]}-exit.json'
launch, done = read(launch_path), read(exit_path)
assert done['exit_code'] == 0 and all(done[k] == v for k,v in launch.items())
assert done['plan_sha256'] == sha(queue/'plan.json')
assert Path(done['argv'][done['argv'].index('--out')+1]) == folder
assert done['started_unix'] <= marker['finished_unix'] <= done['finished_unix']
summary = validate(group['row'], folder, group['agreement'])
cases, provenance = read(folder/'diagnosis.json'), read(folder/'metadata.json')
data = {k:rows(Path(v['path'])/'requests.jsonl') for k,v in provenance['source_journals'].items()}
assert len(data['ar']) == len(data['dflash']) == 300
source_rows = {label:{r['source_id']:r for r in records} for label,records in data.items()}
assert len(source_rows['ar']) == len(source_rows['dflash']) == 300
assert list(source_rows['ar']) == list(source_rows['dflash'])
whole_matches = sum(a['tokens'] == d['tokens'] for a,d in zip(data['ar'],data['dflash'],strict=True))
assert whole_matches == 224 and len(cases) == 300-whole_matches
assert len({c['source_id'] for c in cases}) == 76
ties = 0
for case in cases:
    sid, position = case['source_id'], case['position']
    ar, method = [source_rows[label][sid]['tokens'] for label in ['ar','dflash']]
    assert ar[:position] == method[:position] == case['common_output']
    assert ar[position] == case['ar_token'] != case['dflash_token'] == method[position]
    assert not case['bf16']['argmax_equal'] and case['fp32']['argmax_equal']
    for precision in ['bf16','fp32']:
        observed = case[precision]
        assert math.isfinite(observed['max_abs_logit_difference']) and observed['max_abs_logit_difference'] >= 0
        for label in ['ar','dflash']:
            side = observed[label]
            assert len(side['top_ids']) == len(side['top_logits']) == len(side['top_probs']) == 3
            assert all(math.isfinite(x) for x in side['top_logits']+side['top_probs'])
            assert all(0 <= x <= 1 for x in side['top_probs'])
            assert side['top2_probability_margin'] >= 0 and math.isclose(
                side['top2_probability_margin'], side['top_probs'][0]-side['top_probs'][1], abs_tol=1e-7, rel_tol=1e-6)
            assert side['top1_margin'] >= 0 and math.isclose(
                side['top1_margin'], side['top_logits'][0]-side['top_logits'][1], abs_tol=1e-7, rel_tol=1e-6)
    ties += any(case['bf16'][label]['top1_margin'] == 0 for label in ['ar','dflash'])
checkpoint = Path(read(Path(group['row']['method_run'])/'metadata.json')['checkpoint'])
for name, digest in provenance['checkpoint_sha256'].items():
    assert sha(checkpoint/name) == digest, name
with (BASE/'combined-results/results.csv').open() as stream:
    row = next(r for r in csv.DictReader(stream) if (r['method'],r['model'],r['benchmark'],r['temperature']) ==
               ('dflash','huginn-3.5b','gsm8k','0'))
assert row['match_rate'] == group['row']['match_rate'] and float(row['match_rate']) == 224/300
for name in ['eta_latency','eta_throughput','gamma','method_decode_s','ar_decode_s']:
    assert row[name] == group['row'][name], name
report = dict(status='all_observed_formal_first_divergences_checked', method='dflash', model='huginn-3.5b',
    benchmark='gsm8k', temperature=0, formal_requests=300, whole_sequence_matches_bf16=224,
    whole_sequence_match_rate_bf16=224/300, first_divergences=76,
    bf16_divergences_reproduced=76, at_least_one_bf16_top2_tie=ties,
    no_bf16_top2_tie=76-ties, fp32_argmax_equal=76,
    max_fp32_logit_difference=max(c['fp32']['max_abs_logit_difference'] for c in cases),
    bf16_probability_margin_range={label:[min(c['bf16'][label]['top2_probability_margin'] for c in cases),
        max(c['bf16'][label]['top2_probability_margin'] for c in cases)] for label in ['ar','dflash']},
    actual_exit=dict(path=str(exit_path),sha256=sha(exit_path),exit_code=0),
    artifacts={str(p):sha(p) for p in [queue/'plan.json',folder/'metadata.json',folder/'complete.json',
        folder/'diagnosis.json',launch_path,exit_path]}, source_journals=provenance['source_journals'],
    checkpoint_sha256=provenance['checkpoint_sha256'], existing_validation=summary,
    original_timed_row_unchanged=row, check_source_sha256=sha(Path(__file__)), checked_unix=time.time(),
    limitation='All observed first divergences of this BF16 cell only. FP32 common-prefix agreement does not '
               'turn its original 224/300 whole-sequence agreement into exact BF16 equivalence or certify other cells.')
save(Path(__file__).parent/'report.json', report)
print({k:report[k] for k in ['status','first_divergences','at_least_one_bf16_top2_tie',
                            'no_bf16_top2_tie','fp32_argmax_equal','max_fp32_logit_difference']})
