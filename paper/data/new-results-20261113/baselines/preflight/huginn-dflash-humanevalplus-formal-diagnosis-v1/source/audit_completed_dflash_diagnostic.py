"""Audit a completed legacy DFlash first-divergence group without GPU reruns."""
import argparse
import csv
import math
from pathlib import Path
import time

from packed_dflash_evidence import read, rows, save, sha
from orchestrate_greedy_diagnostics import validate
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def audit(identity, out):
    assert not out.exists(), 'Keep existing checks; choose a fresh output directory.'
    queue = BASE/'logs/baseline-diagnostics-resume-v1'
    plan = read(queue/'plan.json')
    group = next(g for g in plan['groups'] if g['id'] == identity)
    assert group['kind'] == 'legacy' and group['row']['method'] == 'dflash'
    folder = Path(group['out'])
    marker = read(folder/'complete.json')
    assert marker['all_bf16_divergences_reproduced'] and marker['all_fp32_argmax_agree']
    for name,digest in plan['source_sha256'].items():
        assert sha(queue/'source'/name) == digest, name
    for name,digest in plan['legacy_source_sha256'].items():
        assert sha(BASE/name) == digest, name
    for name,digest in read(folder/'source-manifest.json').items():
        assert sha(folder/'source'/name) == digest, name
    assert len(group['steps']) == 1
    step = group['steps'][0]
    launch_path, exit_path = [queue/f'logs/{step["id"]}-{kind}.json' for kind in ['launcher','exit']]
    launch, done = read(launch_path), read(exit_path)
    assert done['exit_code'] == 0 and all(done[k] == v for k,v in launch.items())
    assert done['plan_sha256'] == sha(queue/'plan.json')
    assert Path(done['argv'][done['argv'].index('--out')+1]) == folder
    assert done['started_unix'] <= marker['finished_unix'] <= done['finished_unix']
    summary = validate(group['row'],folder,group['agreement'])
    cases, provenance = read(folder/'diagnosis.json'), read(folder/'metadata.json')
    data = {k:rows(Path(v['path'])/'requests.jsonl') for k,v in provenance['source_journals'].items()}
    count = len(data['ar'])
    assert count == len(data['dflash']) == group['agreement']['requests']
    source = {label:{r['source_id']:r for r in records} for label,records in data.items()}
    assert len(source['ar']) == len(source['dflash']) == count and list(source['ar']) == list(source['dflash'])
    whole = sum(a['tokens'] == d['tokens'] for a,d in zip(data['ar'],data['dflash'],strict=True))
    assert len(cases) == count-whole == marker['cases'] == len(group['agreement']['first_divergences'])
    assert len({c['source_id'] for c in cases}) == len(cases)
    ties = 0
    for case in cases:
        sid, position = case['source_id'],case['position']
        ar, method = [source[label][sid]['tokens'] for label in ['ar','dflash']]
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
                assert side['top2_probability_margin'] >= 0 and math.isclose(side['top2_probability_margin'],
                    side['top_probs'][0]-side['top_probs'][1], abs_tol=1e-7, rel_tol=1e-6)
                assert side['top1_margin'] >= 0 and math.isclose(side['top1_margin'],
                    side['top_logits'][0]-side['top_logits'][1], abs_tol=1e-7, rel_tol=1e-6)
        ties += any(case['bf16'][label]['top1_margin'] == 0 for label in ['ar','dflash'])
    checkpoint = Path(read(Path(group['row']['method_run'])/'metadata.json')['checkpoint'])
    for name,digest in provenance['checkpoint_sha256'].items():
        assert sha(checkpoint/name) == digest, name
    with (BASE/'combined-results/results.csv').open() as stream:
        row = next(r for r in csv.DictReader(stream) if all(r[k] == group['row'][k]
                   for k in ['method','model','benchmark','temperature']))
    assert float(row['match_rate']) == whole/count
    for name in ['match_rate','eta_latency','eta_throughput','gamma','method_decode_s','ar_decode_s']:
        assert row[name] == group['row'][name], name
    report = dict(status='all_observed_formal_first_divergences_checked',method='dflash',model=row['model'],
        benchmark=row['benchmark'],temperature=0,formal_requests=count,whole_sequence_matches_bf16=whole,
        whole_sequence_match_rate_bf16=whole/count,first_divergences=len(cases),
        bf16_divergences_reproduced=len(cases),at_least_one_bf16_top2_tie=ties,
        no_bf16_top2_tie=len(cases)-ties,fp32_argmax_equal=len(cases),
        max_fp32_logit_difference=max((c['fp32']['max_abs_logit_difference'] for c in cases),default=None),
        bf16_probability_margin_range={label:[
            min((c['bf16'][label]['top2_probability_margin'] for c in cases),default=None),
            max((c['bf16'][label]['top2_probability_margin'] for c in cases),default=None)] for label in ['ar','dflash']},
        actual_exit=dict(path=str(exit_path),sha256=sha(exit_path),exit_code=0),
        artifacts={str(p):sha(p) for p in [queue/'plan.json',folder/'metadata.json',folder/'complete.json',
                   folder/'diagnosis.json',launch_path,exit_path]},source_journals=provenance['source_journals'],
        checkpoint_sha256=provenance['checkpoint_sha256'],existing_validation=summary,
        original_timed_row_unchanged=row,check_source_sha256=sha(Path(__file__)),checked_unix=time.time(),
        limitation='Every observed first divergence of this cell only. FP32 common-prefix agreement does not '
                   'replace the original BF16 whole-sequence agreement or certify other cells.')
    out.mkdir(parents=True)
    freeze_sources(out,[Path(__file__),BASE/'provenance.py',BASE/'packed_dflash_evidence.py',
                        BASE/'orchestrate_greedy_diagnostics.py'])
    save(out/'report.json',report)
    print({k:report[k] for k in ['status','model','benchmark','first_divergences',
                                'at_least_one_bf16_top2_tie','no_bf16_top2_tie',
                                'fp32_argmax_equal','max_fp32_logit_difference']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',required=True)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    audit(args.group,args.out.resolve())
