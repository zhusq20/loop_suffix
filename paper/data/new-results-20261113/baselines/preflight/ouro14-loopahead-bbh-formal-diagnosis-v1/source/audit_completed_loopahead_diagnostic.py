"""Check a completed formal LoopAhead diagnostic against its original journals.

This CPU check verifies both actual GPU exits, source and weight bindings, every
observed first divergence and reported margins. It does not regenerate outputs.
"""
import argparse
import csv
import math
from pathlib import Path
import time

from orchestrate_greedy_diagnostics import validate
from packed_dflash_evidence import read, rows, save, sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def check_margin(side, bf16=False):
    assert len(side['top_ids']) == len(side['top_logits']) == len(side['top_probs']) == 3
    assert all(math.isfinite(x) for x in side['top_logits'] + side['top_probs'])
    assert all(0 <= x <= 1 for x in side['top_probs'])
    assert side['top2_probability_margin'] >= 0
    assert math.isclose(side['top2_probability_margin'],
                        side['top_probs'][0]-side['top_probs'][1], abs_tol=1e-7, rel_tol=1e-6)
    if bf16:
        assert side['top1_logit_margin'] >= 0
        assert math.isclose(side['top1_logit_margin'],
                            side['top_logits'][0]-side['top_logits'][1], abs_tol=1e-7, rel_tol=1e-6)


def audit(queue, identity, out):
    assert not out.exists(), 'Preserve earlier evidence; choose a fresh output directory.'
    plan_path = queue/'plan.json'
    plan = read(plan_path)
    group = next(g for g in plan['groups'] if g['id'] == identity)
    row = group['row']
    assert group['kind'] == 'legacy' and row['method'] == 'loopahead'
    assert row['temperature'] == '0' and not row.get('paper_check')
    folder = Path(group['out'])
    for name,digest in plan['source_sha256'].items():
        assert sha(queue/'source'/name) == digest, name
    for name,digest in plan['legacy_source_sha256'].items():
        assert sha(BASE/name) == digest, name
    validated = validate(row, folder, group['agreement'])
    observed = read(folder/'report.json')
    count = observed['cases']
    fp32 = read(folder/'fp32-common-prefix/report.json') if count else None
    assert len(group['steps']) == (2 if count else 1)
    exits, artifacts = {}, [plan_path]
    for step in group['steps']:
        target, marker = Path(step['out']), Path(step['marker'])
        launch_path, exit_path = [queue/f'logs/{step["id"]}-{kind}.json' for kind in ['launcher','exit']]
        launch, done, report = read(launch_path), read(exit_path), read(marker)
        assert done['exit_code'] == 0 and all(done[k] == v for k,v in launch.items())
        assert done['plan_sha256'] == sha(plan_path)
        assert done['group'] == identity and done['step'] == step['id']
        assert Path(done['argv'][done['argv'].index('--out')+1]) == target
        assert done['started_unix'] <= report['started_unix'] <= report['finished_unix'] <= done['finished_unix']
        source_manifest = target/'source-manifest.json'
        for name,digest in read(source_manifest).items():
            assert sha(target/'source'/name) == digest, name
        artifacts += [marker, source_manifest, launch_path, exit_path]
        exits[step['id']] = dict(path=str(exit_path), sha256=sha(exit_path), exit_code=0)
    source = {label:rows(Path(observed['source_journals'][label]['path'])/'requests.jsonl')
              for label in ['ar','loopahead']}
    ids = [r['source_id'] for r in source['ar']]
    assert len(ids) == len(set(ids)) == group['agreement']['requests']
    assert ids == [r['source_id'] for r in source['loopahead']]
    indexed = {label:{r['source_id']:r for r in records} for label,records in source.items()}
    whole = sum(a['tokens'] == d['tokens'] for a,d in zip(source['ar'],source['loopahead'],strict=True))
    assert len(ids)-whole == count == len(observed['results'])
    assert len({r['source_id'] for r in observed['results']}) == count
    ties = 0
    for case, precise in zip(observed['results'], fp32['results'] if fp32 else [], strict=True):
        sid, position = case['source_id'], case['position']
        assert (sid,position) == (precise['source_id'],precise['position'])
        ar, method = [indexed[label][sid]['tokens'] for label in ['ar','loopahead']]
        assert ar[:position] == method[:position] == precise['common_prefix']
        assert ar[position] == precise['ar_token'] == case['ar']['argmax']
        assert method[position] == precise['loopahead_token'] == case['loopahead']['argmax']
        assert ar[position] != method[position] and case['both_original_prefixes_reproduced']
        assert precise['observed_bf16'] == {label:case[label] for label in ['ar','loopahead']}
        assert precise['same_top1'] and math.isfinite(precise['max_abs_logit_difference'])
        assert precise['max_abs_logit_difference'] >= 0
        for label in ['ar','loopahead']:
            check_margin(case[label], bf16=True)
        for label in ['serial','loopahead']:
            check_margin(precise[label])
        tied = any(case[label]['top1_logit_margin'] == 0 for label in ['ar','loopahead'])
        assert tied == case['at_least_one_top2_tie']
        ties += tied
    meta_path = Path(row['method_run'])/'metadata.json'
    meta = read(meta_path)
    assert sha(meta['checkpoint']) == meta['checkpoint_sha256'] == observed['checkpoint_sha256']
    with (BASE/'combined-results/results.csv').open() as stream:
        current = next(r for r in csv.DictReader(stream) if all(r[k] == row[k]
                       for k in ['method','model','benchmark','temperature']))
    assert float(current['match_rate']) == whole/len(ids)
    for name in ['match_rate','eta_latency','eta_throughput','gamma','method_decode_s','ar_decode_s']:
        assert current[name] == row[name], name
    report = dict(status='all_observed_formal_first_divergences_checked', method='loopahead',
        model=row['model'],benchmark=row['benchmark'],temperature=0,formal_requests=len(ids),
        whole_sequence_matches_bf16=whole,whole_sequence_match_rate_bf16=whole/len(ids),
        first_divergences=count,bf16_divergences_reproduced=count,
        at_least_one_bf16_top2_tie=ties,no_bf16_top2_tie=count-ties,fp32_argmax_equal=count,
        max_fp32_logit_difference=max((r['max_abs_logit_difference'] for r in fp32['results']),default=None) if fp32 else None,
        actual_exits=exits,artifact_sha256={str(p):sha(p) for p in artifacts+[meta_path]},
        checkpoint=meta['checkpoint'],checkpoint_sha256=meta['checkpoint_sha256'],
        source_journals=observed['source_journals'],existing_validation=validated,
        original_timed_row_unchanged=current,checked_unix=time.time(),new_gpu_measurements=0,
        limitations=['All observed first divergences of this formal cell only; full BF16 output agreement is unchanged.',
                     'Forced FP32 common-prefix replay may change the proposed tree; it is not complete FP32 benchmark generation.',
                     'Logged top-three values and margins are checked arithmetically; full logits are not regenerated by this CPU audit.'])
    out.mkdir(parents=True)
    freeze_sources(out,[Path(__file__),BASE/'orchestrate_greedy_diagnostics.py',
                        BASE/'packed_dflash_evidence.py',BASE/'provenance.py'])
    save(out/'report.json',report)
    print({k:report[k] for k in ['status','model','benchmark','first_divergences',
                               'at_least_one_bf16_top2_tie','no_bf16_top2_tie',
                               'fp32_argmax_equal','max_fp32_logit_difference']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--group',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    audit(args.queue.resolve(),args.group,args.out.resolve())
