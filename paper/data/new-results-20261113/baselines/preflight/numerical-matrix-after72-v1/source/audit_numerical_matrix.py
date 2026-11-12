"""Audit the full numerical-check scope from immutable reports and actual exits.

36 formal greedy cells, one paper-setting check and two development checks are
separate requirements. Pending FP32 work never counts as a completed group.
This entry point performs no inference and never starts or restarts a worker.
"""
import argparse
import csv
import itertools
import math
from pathlib import Path
import time
import traceback

from orchestrate_baseline_diagnostics_resume import verify_plan, validate_group
from packed_dflash_evidence import check_frozen, read, rows, save, sha
from provenance import freeze_sources
from loopahead_resume_attempt import ATTEMPT, ORIGINAL, NAME, audit_resumed

BASE = Path(__file__).resolve().parent
MODELS = ['huginn-3.5b','ouro-2.6b','ouro-1.4b']
TASKS = ['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench']
EXPECTED = {'formal/'+ '/'.join(x) for x in itertools.product(['dflash','loopahead'],MODELS,TASKS)}
EXPECTED |= {'paper/loopahead/ouro-2.6b/humaneval164','development/narrow','development/wide'}
QUEUES = ['baseline-diagnostics-resume-v1','appendix-loopahead-diagnostics-v1','appendix-dflash-diagnostics-v1']


def scope_key(plan, group):
    if group['kind'] == 'development':
        return 'development/'+group['label']
    if group['kind'] == 'packed':
        supplement = read(plan['supplement_plan'])
        job = next(j for j in supplement['jobs'] if j['id'] == group['job'])
        assert job['temperature'] == 0
        return '/'.join(['formal','dflash',job['model'],job['dataset']])
    row = group['row']
    assert group['kind'] == 'legacy' and row['temperature'] == '0'
    if row.get('paper_check'):
        assert row['method'] == 'loopahead' and row['model'] == 'ouro-2.6b'
        assert row['benchmark'] == 'paper-humaneval164'
        return 'paper/loopahead/ouro-2.6b/humaneval164'
    return '/'.join(['formal',row['method'],row['model'],row['benchmark']])


def receipt_paths(queue, plan, group, step):
    if group['id'] in plan['reused_groups']:
        assert group['id'] == 'loopahead-huginn-3.5b-gsm8k-v1'
        prefix = 'formal-greedy-fp32-' if step['id'].endswith('-fp32') else 'formal-greedy-margins-'
        stem = prefix+group['id']
        return BASE/'logs'/f'{stem}-launcher.json', BASE/'logs'/f'{stem}-finished.json'
    return queue/f'logs/{step["id"]}-launcher.json', queue/f'logs/{step["id"]}-exit.json'


def check_receipt(launch, receipt, step, group, plan_hash, reused):
    assert type(receipt['exit_code']) is int and receipt['exit_code'] == 0
    assert all(receipt[k] == v for k,v in launch.items())
    assert math.isfinite(receipt['started_unix']) and math.isfinite(receipt['finished_unix'])
    assert receipt['started_unix'] <= receipt['finished_unix']
    argv = receipt['argv']
    assert argv.count('--out') == 1 and argv[argv.index('--out')+1] == step['out']
    for flag in ['--ar','--dflash','--loopahead','--observed-margin-report','--label','--job','--resume-evidence']:
        if flag in step['argv']:
            assert argv.count(flag) == 1
            assert argv[argv.index(flag)+1] == step['argv'][step['argv'].index(flag)+1]
    assert ('--paper-check' in argv) == ('--paper-check' in step['argv'])
    if not reused:
        assert receipt['group'] == group['id'] and receipt['step'] == step['id']
        assert receipt['plan_sha256'] == plan_hash


def check_margin(side):
    assert len(side['top_ids']) == len(side['top_logits']) == len(side['top_probs']) == 3
    assert len(set(side['top_ids'])) == 3 and all(type(x) is int and x >= 0 for x in side['top_ids'])
    assert all(math.isfinite(v) for v in side['top_logits']+side['top_probs'])
    assert side['top_logits'] == sorted(side['top_logits'],reverse=True)
    assert side['top_probs'] == sorted(side['top_probs'],reverse=True)
    assert all(0 <= v <= 1 for v in side['top_probs'])
    assert 0 <= side['top2_probability_margin']
    assert math.isclose(side['top2_probability_margin'],side['top_probs'][0]-side['top_probs'][1],abs_tol=1e-7,rel_tol=1e-6)
    for field in ['top1_margin','top1_logit_margin']:
        if field in side:
            assert side[field] >= 0
            assert math.isclose(side[field],side['top_logits'][0]-side['top_logits'][1],abs_tol=1e-7,rel_tol=1e-6)


def inspect_cases(method, cases, precise, originals):
    """Check actual first-divergence positions, prefixes and all recorded margins."""
    ar, target = originals['ar'], originals['method']
    ids = [r['source_id'] for r in ar]
    assert len(ids) == len(set(ids)) and ids == [r['source_id'] for r in target]
    expected, whole = [], 0
    for a,d in zip(ar,target,strict=True):
        if a['tokens'] == d['tokens']:
            whole += 1
            continue
        position = next((i for i,(x,y) in enumerate(zip(a['tokens'],d['tokens'])) if x != y),None)
        assert position is not None, 'Length-only differences need separate investigation.'
        expected.append((a['source_id'],position))
    assert [(c['source_id'],c['position']) for c in cases] == expected
    by_id = {r['source_id']:(r,t) for r,t in zip(ar,target,strict=True)}
    assert len(cases) == len(ids)-whole
    if method == 'loopahead':
        assert [(c['source_id'],c['position']) for c in precise] == expected
    ties, differences = 0, []
    for i,case in enumerate(cases):
        a,d = by_id[case['source_id']]
        pos = case['position']
        if method == 'loopahead':
            f = precise[i]
            assert case['both_original_prefixes_reproduced'] and f['same_top1']
            assert a['tokens'][:pos] == d['tokens'][:pos] == f['common_prefix']
            assert a['tokens'][pos] == case['ar']['argmax'] == f['ar_token']
            assert d['tokens'][pos] == case['loopahead']['argmax'] == f['loopahead_token']
            assert f['observed_bf16'] == {s:case[s] for s in ['ar','loopahead']}
            bf16 = [case['ar'],case['loopahead']]
            fp32 = [f['serial'],f['loopahead']]
            difference = f['max_abs_logit_difference']
        else:
            assert method == 'dflash'
            assert a['tokens'][:pos] == d['tokens'][:pos] == case['common_output']
            assert a['tokens'][pos] == case['ar_token'] != case['dflash_token'] == d['tokens'][pos]
            assert not case['bf16']['argmax_equal'] and case['fp32']['argmax_equal']
            bf16 = [case['bf16']['ar'],case['bf16']['dflash']]
            fp32 = [case['fp32']['ar'],case['fp32']['dflash']]
            assert math.isfinite(case['bf16']['max_abs_logit_difference'])
            assert case['bf16']['max_abs_logit_difference'] >= 0
            difference = case['fp32']['max_abs_logit_difference']
        for side in bf16+fp32:
            check_margin(side)
        if all(s['top_logits'][0] > s['top_logits'][1] for s in fp32):
            assert fp32[0]['top_ids'][0] == fp32[1]['top_ids'][0]
        tied = any(s['top_logits'][0] == s['top_logits'][1] for s in bf16)
        if method == 'loopahead':
            assert tied == case['at_least_one_top2_tie']
        ties += tied
        assert math.isfinite(difference) and difference >= 0
        differences.append(difference)
    return dict(requests=len(ids),whole_sequence_matches_bf16=whole,
        whole_sequence_match_rate_bf16=whole/len(ids),first_divergences=len(cases),
        bf16_divergences_reproduced=len(cases),at_least_one_bf16_top2_tie=ties,
        no_bf16_top2_tie=len(cases)-ties,fp32_argmax_equal=len(cases),
        max_fp32_logit_difference=max(differences,default=None))


def audit_group(queue, plan, group, canonical):
    paths = [receipt_paths(queue,plan,group,s) for s in group['steps']]
    missing = [str(p) for pair in paths for p in pair if not p.exists()]
    # A failed earlier stage must not disappear behind a missing later receipt.
    for step,(launch_path,exit_path) in zip(group['steps'],paths,strict=True):
        if exit_path.exists():
            assert launch_path.exists(), 'An exit receipt is missing its launcher.'
            check_receipt(read(launch_path),read(exit_path),step,group,
                          sha(queue/'plan.json'),group['id'] in plan['reused_groups'])
    if missing:
        return dict(status='pending_actual_exit',missing=missing)
    artifacts, exits = {}, []
    for step,(launch_path,exit_path) in zip(group['steps'],paths,strict=True):
        launch, receipt = read(launch_path), read(exit_path)
        check_receipt(launch,receipt,step,group,sha(queue/'plan.json'),group['id'] in plan['reused_groups'])
        marker = Path(step['marker'])
        assert marker.exists(), 'Actual successful exit is missing its required report.'
        done = read(marker)
        assert receipt['started_unix'] <= done['finished_unix'] <= receipt['finished_unix']
        for p in [launch_path,exit_path,marker]:
            artifacts[str(p)] = sha(p)
        exits.append(dict(path=str(exit_path),sha256=sha(exit_path),exit_code=0))
        source_manifest = Path(step['out'])/'source-manifest.json'
        if source_manifest.exists():
            artifacts[str(source_manifest)] = sha(source_manifest)
            for name,digest in read(source_manifest).items():
                assert sha(source_manifest.parent/'source'/name) == digest,name
    validated = validate_group(plan,group)
    folder = Path(group['out'])
    method = validated['method']
    development = group['kind'] == 'development'
    if method == 'loopahead':
        meta = read(folder/'report.json')
        cases = meta['results']
        precise = read(folder/'fp32-common-prefix/report.json')['results'] if cases else []
    elif development:
        meta = read(folder/'report.json')
        cases, precise = meta['results'], []
    else:
        meta = read(folder/'metadata.json')
        cases, precise = read(folder/'diagnosis.json'), []
        artifacts[str(folder/'metadata.json')] = sha(folder/'metadata.json')
        artifacts[str(folder/'diagnosis.json')] = sha(folder/'diagnosis.json')
    originals = {}
    for label,record in meta['source_journals'].items():
        p = Path(record['path']) if development else Path(record['path'])/'requests.jsonl'
        assert sha(p) == record['sha256']
        artifacts[str(p)] = record['sha256']
        originals['ar' if label == 'ar' else 'method'] = rows(p)
    summary = inspect_cases(method,cases,precise,originals)
    assert summary['first_divergences'] == validated['cases']
    scope = scope_key(plan,group)
    if scope.startswith('formal/'):
        row = canonical[scope]
        assert float(row['match_rate']) == summary['whole_sequence_match_rate_bf16']
        for label,field in [('ar','ar_run'),(method,'method_run')]:
            assert Path(meta['source_journals'][label]['path']) == Path(row[field])
        if group['kind'] == 'legacy':
            for name in ['eta_latency','eta_throughput','gamma','match_rate','method_decode_s','ar_decode_s']:
                assert row[name] == group['row'][name]
    if group['kind'] == 'packed':
        supplement = read(plan['supplement_plan'])
        job = next(j for j in supplement['jobs'] if j['id'] == group['job'])
        check_frozen(supplement['suite'],job,weights=True)
        assert meta['diagnostic_source_sha256'] == sha(queue/'source/diagnose_packed_dflash.py')
    elif not development:
        measurement = read(Path(group['row']['method_run'])/'metadata.json')
        checkpoint = Path(measurement['checkpoint'])
        if method == 'loopahead':
            assert sha(checkpoint) == meta['checkpoint_sha256']
        else:
            for name,digest in meta['checkpoint_sha256'].items():
                assert sha(checkpoint/name) == digest
    if development:
        artifacts[str(folder/'report.json')] = sha(folder/'report.json')
    return dict(status='validated_actual_diagnostic',scope=scope,group=group['id'],
        **summary,queue_plan=str(queue/'plan.json'),queue_plan_sha256=sha(queue/'plan.json'),
        actual_exits=exits,artifact_sha256=artifacts,existing_validation=validated)


def completion_gate(results, pending, errors):
    assert not pending and not errors, 'Numerical scope still has pending or invalid groups.'
    assert set(results) == EXPECTED and len(results) == 39
    assert all(r['status'] == 'validated_actual_diagnostic' for r in results.values())


def audit(out, partial):
    assert not out.exists(), 'Keep earlier snapshots; choose a fresh output directory.'
    csv_path = BASE/'combined-results/results.csv'
    before = sha(csv_path)
    with csv_path.open() as stream:
        canonical = {'/'.join(['formal',r['method'],r['model'],r['benchmark']]):r
                     for r in csv.DictReader(stream) if r['temperature'] == '0'}
    groups, results, pending, errors, plans = {}, {}, {}, {}, {}
    for name in QUEUES:
        queue = BASE/'logs'/name
        if not (queue/'plan.json').exists():
            continue
        plan = verify_plan(queue/'plan.json')
        plans[str(queue/'plan.json')] = sha(queue/'plan.json')
        for group in plan['groups']:
            scope = scope_key(plan,group)
            assert scope in EXPECTED and scope not in groups, ('Unexpected or duplicate scope',scope)
            groups[scope] = group['id']
            try:
                if queue == ORIGINAL and group['id'] == NAME and (ATTEMPT/'plan.json').exists():
                    result = audit_resumed(queue,plan,group,canonical,check_receipt,inspect_cases)
                    plans[str(ATTEMPT/'plan.json')] = sha(ATTEMPT/'plan.json')
                else:
                    result = audit_group(queue,plan,group,canonical)
                if result['status'] == 'pending_actual_exit':
                    pending[scope] = result
                else:
                    results[scope] = result
            except Exception:
                errors[scope] = dict(group=group['id'],error=traceback.format_exc())
    for scope in EXPECTED-set(groups):
        pending[scope] = dict(status='waiting_for_frozen_plan')
    assert set(results)|set(pending)|set(errors) == EXPECTED
    assert not (set(results)&set(pending) or set(results)&set(errors) or set(pending)&set(errors))
    complete = not pending and not errors
    if complete:
        completion_gate(results,pending,errors)
    assert sha(csv_path) == before
    out.mkdir(parents=True)
    source_manifest = freeze_sources(out,[Path(__file__),BASE/'provenance.py',BASE/'packed_dflash_evidence.py',
        BASE/'orchestrate_baseline_diagnostics_resume.py',BASE/'orchestrate_greedy_diagnostics.py',
        BASE/'diagnose_huginn_dflash_dev.py',BASE/'loopahead_resume_attempt.py',BASE/'loopahead_partial_resume.py'])
    report = dict(status='complete_numerical_scope' if complete else 'needs_inspection' if errors else 'partial_numerical_scope',
        complete=complete,partial_requested=partial,expected_formal=36,expected_additional=3,
        completed_formal=sum(k.startswith('formal/') for k in results),
        completed_additional=sum(not k.startswith('formal/') for k in results),
        results=results,pending=pending,errors=errors,plan_sha256=plans,
        source_sha256=source_manifest,unchanged_results_csv_sha256=before,
        checked_unix=time.time(),new_gpu_runs=0,
        limitations=['Only completed actual first-divergence diagnostics are counted; pending BF16/FP32 stages are not successes.',
                     'Original BF16 whole-output match rates remain unchanged; FP32 common-prefix checks are not full FP32 benchmark outputs.',
                     'This is greedy numerical evidence, not proof of T1 sampling distributions or task accuracy.',
                     'Paper-setting and unequal-budget development checks are separate from the 36 formal cells.'])
    save(out/'report.json',report)
    print({k:report[k] for k in ['status','completed_formal','completed_additional','new_gpu_runs']})
    assert not errors, 'Inspect preserved per-group errors before using this report.'
    if not partial:
        completion_gate(results,pending,errors)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args = parser.parse_args()
    audit(args.out.resolve(),args.partial)
