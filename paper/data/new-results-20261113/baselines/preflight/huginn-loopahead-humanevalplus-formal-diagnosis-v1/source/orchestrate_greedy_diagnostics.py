"""Diagnose36 formal greedy cells, B4, and the frozen Huginn width comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import queue
import time

from orchestrate_formal import BASE, CPUS, MODELS, TASKS, run_command, save
from gpu_allocation import allowed_gpus


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_check_plan():
    from diagnose_huginn_dflash_dev import prepare
    return [prepare(label)[3] for label in ['narrow', 'wide']]


def paper_check_plan():
    from diagnose_formal_loopahead import prepare
    from loopahead_diagnostic_inputs import input_provenance
    ar = BASE/'preflight/loopahead-B4-ouro-2.6b-humaneval-ar-v1'
    target = BASE/'preflight/loopahead-B4-ouro-2.6b-humaneval-loopahead-v1'
    meta, cases = prepare(ar, target, paper_check=True)
    summary_path = BASE/'preflight/loopahead-B4-summary.json'
    agreement = json.loads(summary_path.read_text())
    assert agreement['requests'] == 164 and agreement['exact_sequences'] == 97
    assert [(r['source_id'], r['position']) for r in cases] == [
        (r['source_id'], r['position']) for r in agreement['first_divergences']]
    assert len(cases) == 67
    row = dict(method='loopahead', model=meta['model'], benchmark=meta['dataset'],
        temperature='0', paper_check=True, ar_run=str(ar), method_run=str(target))
    provenance = dict(row=row, cases=len(cases), summary_sha256=digest(summary_path),
        input_provenance=input_provenance(meta, json.loads((ar/'metadata.json').read_text()), True),
        source_runs={label: {name: digest(folder/name) for name in
            ['metadata.json', 'source-manifest.json', 'complete.json', 'requests.jsonl']}
            for label, folder in [('ar', ar), ('loopahead', target)]})
    return row, agreement, provenance


def validate(row, folder, agreement):
    method = row['method']
    ar, target = Path(row['ar_run']), Path(row['method_run'])
    if method == 'loopahead':
        report = json.loads((folder/'report.json').read_text())
        assert report['status'] == 'complete'
        results = report['results']
        assert all(case['both_original_prefixes_reproduced'] for case in results)
        assert all('top2_probability_margin' in case[label] for case in results for label in ['ar','loopahead'])
        provenance = report
    else:
        report = json.loads((folder/'complete.json').read_text())
        assert report['all_bf16_divergences_reproduced']
        assert report['all_fp32_argmax_agree'], 'Formal DFlash FP32 disagreements require investigation'
        results = json.loads((folder/'diagnosis.json').read_text())
        assert all('top2_probability_margin' in case['bf16'][label] for case in results for label in ['ar','dflash'])
        provenance = json.loads((folder/'metadata.json').read_text())
    assert provenance['model'] == row['model'] and provenance['dataset'] == row['benchmark']
    meta = json.loads((target/'metadata.json').read_text())
    assert provenance['checkpoint_sha256'] == meta['checkpoint_sha256']
    if row.get('paper_check'):
        from loopahead_diagnostic_inputs import input_provenance
        assert provenance['input_provenance'] == input_provenance(
            meta, json.loads((ar/'metadata.json').read_text()), True)
    for label,path in [('ar',ar),(method,target)]:
        source = provenance['source_journals'][label]
        assert Path(source['path']).resolve() == path.resolve() and source['sha256'] == digest(path/'requests.jsonl')
    expected = [(case['source_id'],case['position']) for case in agreement['first_divergences']]
    actual = [(case['source_id'],case['position']) for case in results]
    assert actual == expected and report['cases'] == len(expected)
    fp32_sha = None
    if method == 'loopahead' and expected:
        fp32_path = folder/'fp32-common-prefix/report.json'
        fp32 = json.loads(fp32_path.read_text())
        assert fp32['status'] == 'complete' and fp32['all_fp32_same_top1']
        assert fp32['cases'] == len(expected) and fp32['checkpoint_sha256'] == meta['checkpoint_sha256']
        assert fp32['observed_margin_report']['sha256'] == digest(folder/'report.json')
        assert fp32['source_journals'] == provenance['source_journals']
        if row.get('paper_check'):
            assert fp32['input_provenance'] == provenance['input_provenance']
        assert [(r['source_id'],r['position']) for r in fp32['results']] == expected
        fp32_sha = digest(fp32_path)
    return dict(method=method, model=row['model'], benchmark=row['benchmark'], cases=len(expected),
        folder=str(folder), result_sha256=digest(folder/('report.json' if method=='loopahead' else 'diagnosis.json')),
        loopahead_fp32_report_sha256=fp32_sha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(cells=36, models=MODELS+['ouro-1.4b'], benchmarks=TASKS,
            additional_paper_check=paper_check_plan()[2],
            additional_development_checks=development_check_plan(),
            requirement='Every greedy first divergence: exact source-prefix replay and BF16 top2 probability margins',
            loopahead='Actual AR and trained LAD BF16 replay, followed by FP32 common-prefix checks for all observed first divergences',
            dflash='Actual block trace plus BF16 and FP32 common-prefix replay',
            scope='Diagnostics only; original benchmark journals and timings remain unchanged'),indent=2))
        return
    status = json.loads((BASE/'formal-matrix-status.json').read_text())
    assert status['completed'] == status['expected'] == 72 and not status['missing']
    rows = [r for r in csv.DictReader((BASE/'results.csv').open()) if r['temperature']=='0']
    assert len(rows) == 36
    agreements = json.loads((BASE/'formal-agreement.json').read_text())
    paper_row, paper_agreement, paper_provenance = paper_check_plan()
    development_provenance = development_check_plan()
    out = BASE/'preflight/formal-greedy-margins'
    out.mkdir(exist_ok=True)
    plan = out/'plan.json'
    current = dict(cells=36, results_csv_sha256=digest(BASE/'results.csv'),
        agreement_sha256=digest(BASE/'formal-agreement.json'), rows=rows,
        additional_paper_check=paper_provenance, additional_development_checks=development_provenance)
    if plan.exists():
        assert json.loads(plan.read_text()) == current
    else:
        save(plan,current)
    from provenance import freeze_sources
    snapshot = out/f'controller-source-{int(time.time())}'
    snapshot.mkdir()
    freeze_sources(snapshot,[Path(__file__),BASE/'diagnose_formal_loopahead.py',BASE/'diagnose_dflash.py',
        BASE/'diagnose_trained_loopahead.py',BASE/'loopahead_diagnostic_inputs.py',
        BASE/'diagnose_huginn_dflash_dev.py',
        BASE/'gpu_allocation.py',BASE/'configs/gpu-allocation.json'])
    pending = queue.Queue()
    for row in rows:
        pending.put(row)
    pending.put(paper_row)
    for item in development_provenance:
        pending.put(dict(development_only=True, label=item['label']))

    def lane(gpu):
        completed = []
        while True:
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return completed
            if row.get('development_only'):
                from diagnose_huginn_dflash_dev import validate as validate_development
                label = row['label']
                folder = BASE/f'preflight/huginn-dflash-dev-greedy-{label}-v1'
                if not (folder/'complete.json').exists():
                    assert not folder.exists(), f'Investigate incomplete development diagnostic: {folder}'
                    run_command(f'formal-greedy-dev-huginn-dflash-{label}-v1',
                        ['/tmp/loopfill-dflash-venv/bin/python', '-u', str(BASE/'diagnose_huginn_dflash_dev.py'),
                         '--label', label, '--gpu', str(gpu), '--cpus', CPUS[gpu], '--out', str(folder)])
                completed.append(validate_development(label, folder))
                save(out/f'lane{gpu}-progress.json', dict(completed=completed, updated_unix=time.time()))
                pending.task_done()
                continue
            method,model,task = row['method'],row['model'],row['benchmark']
            identity = f'{method}-{model}-{task}-v1'
            paper_check = row.get('paper_check', False)
            if paper_check:
                identity = 'loopahead-B4-ouro-2.6b-humaneval-v1'
                folder = BASE/'preflight/loopahead-B4-greedy-margins-v1'
                agreement = paper_agreement
            else:
                folder = out/identity
                key = '/'.join([row['engine'],model,task,'0',method])
                agreement = agreements[key]
            scope_args = ['--paper-check'] if paper_check else []
            marker = folder/('report.json' if method=='loopahead' else 'complete.json')
            if not marker.exists():
                assert not folder.exists(), f'Investigate/archive incomplete diagnostic before retry: {folder}'
                script = 'diagnose_formal_loopahead.py' if method=='loopahead' else 'diagnose_dflash.py'
                python = '/tmp/loopfill-venv/bin/python' if method=='loopahead' else '/tmp/loopfill-dflash-venv/bin/python'
                argv = [python,'-u',str(BASE/script),'--ar',row['ar_run'],'--'+method,row['method_run'],
                    '--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(folder)] + scope_args
                if method == 'dflash':
                    argv += ['--shard','0','--shards','1']
                run_command('formal-greedy-margins-'+identity,argv)
            if method == 'loopahead':
                bf16 = json.loads(marker.read_text())
                assert bf16['status'] == 'complete'
                fp32 = folder/'fp32-common-prefix'
                if bf16['cases'] and not (fp32/'report.json').exists():
                    assert not fp32.exists(), f'Investigate incomplete FP32 diagnostic: {fp32}'
                    run_command('formal-greedy-fp32-'+identity,['/tmp/loopfill-venv/bin/python','-u',
                        str(BASE/'diagnose_trained_loopahead.py'),'--ar',row['ar_run'],'--loopahead',row['method_run'],
                        '--observed-margin-report',str(marker),'--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(fp32)] + scope_args)
            completed.append(validate(row,folder,agreement))
            save(out/f'lane{gpu}-progress.json',dict(completed=completed,updated_unix=time.time()))
            pending.task_done()

    results, errors = [], []
    gpus = allowed_gpus()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane,gpu) for gpu in gpus]
        for future in futures:
            try:
                results += future.result()
            except Exception as exc:
                errors.append(repr(exc))
    if errors:
        save(out/'failure.json',dict(errors=errors,finished_unix=time.time()))
        raise RuntimeError(errors)
    assert pending.empty() and len(results) == 39
    formal_results = [r for r in results if r['benchmark'] != 'paper-humaneval164' and not r.get('development_only')]
    paper_results = [r for r in results if r['benchmark'] == 'paper-humaneval164']
    development_results = [r for r in results if r.get('development_only')]
    assert len(formal_results) == 36 and len(paper_results) == 1 and len(development_results) == 2
    assert {r['label'] for r in development_results} == {'narrow', 'wide'}
    save(out/'complete.json',dict(cells=36, cases=sum(r['cases'] for r in formal_results),
        results=sorted(formal_results,key=lambda r:(r['method'],r['model'],r['benchmark'])),
        additional_paper_check=paper_results[0],
        additional_development_checks=sorted(development_results, key=lambda r: r['label']),
        finished_unix=time.time(), limitation='Sampling(T1) same-seed differences are not classified as greedy numerical discrepancies.'))


if __name__ == '__main__':
    main()
