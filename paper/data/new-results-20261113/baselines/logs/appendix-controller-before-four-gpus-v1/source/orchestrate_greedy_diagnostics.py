"""Reproduce and record every first divergence in the36 formal greedy cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import queue
import time

from orchestrate_formal import BASE, CPUS, MODELS, TASKS, run_command, save


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for label,path in [('ar',ar),(method,target)]:
        source = provenance['source_journals'][label]
        assert Path(source['path']).resolve() == path.resolve() and source['sha256'] == digest(path/'requests.jsonl')
    expected = [(case['source_id'],case['position']) for case in agreement['first_divergences']]
    actual = [(case['source_id'],case['position']) for case in results]
    assert actual == expected and report['cases'] == len(expected)
    return dict(method=method, model=row['model'], benchmark=row['benchmark'], cases=len(expected),
        folder=str(folder), result_sha256=digest(folder/('report.json' if method=='loopahead' else 'diagnosis.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(cells=36, models=MODELS+['ouro-1.4b'], benchmarks=TASKS,
            requirement='Every greedy first divergence: exact source-prefix replay and BF16 top2 probability margins',
            loopahead='Actual AR and trained LAD replay; FP32 evidence remains the separately reported preflight checks',
            dflash='Actual block trace plus BF16 and FP32 common-prefix replay',
            scope='Diagnostics only; original benchmark journals and timings remain unchanged'),indent=2))
        return
    status = json.loads((BASE/'formal-matrix-status.json').read_text())
    assert status['completed'] == status['expected'] == 72 and not status['missing']
    rows = [r for r in csv.DictReader((BASE/'results.csv').open()) if r['temperature']=='0']
    assert len(rows) == 36
    agreements = json.loads((BASE/'formal-agreement.json').read_text())
    out = BASE/'preflight/formal-greedy-margins'
    out.mkdir(exist_ok=True)
    plan = out/'plan.json'
    current = dict(cells=36, results_csv_sha256=digest(BASE/'results.csv'),
        agreement_sha256=digest(BASE/'formal-agreement.json'), rows=rows)
    if plan.exists():
        assert json.loads(plan.read_text()) == current
    else:
        save(plan,current)
    from provenance import freeze_sources
    snapshot = out/f'controller-source-{int(time.time())}'
    snapshot.mkdir()
    freeze_sources(snapshot,[Path(__file__),BASE/'diagnose_formal_loopahead.py',BASE/'diagnose_dflash.py'])
    pending = queue.Queue()
    for row in rows:
        pending.put(row)

    def lane(gpu):
        completed = []
        while True:
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return completed
            method,model,task = row['method'],row['model'],row['benchmark']
            identity = f'{method}-{model}-{task}-v1'
            folder = out/identity
            key = '/'.join([row['engine'],model,task,'0',method])
            marker = folder/('report.json' if method=='loopahead' else 'complete.json')
            if not marker.exists():
                assert not folder.exists(), f'Investigate/archive incomplete diagnostic before retry: {folder}'
                script = 'diagnose_formal_loopahead.py' if method=='loopahead' else 'diagnose_dflash.py'
                python = '/tmp/loopfill-venv/bin/python' if method=='loopahead' else '/tmp/loopfill-dflash-venv/bin/python'
                argv = [python,'-u',str(BASE/script),'--ar',row['ar_run'],'--'+method,row['method_run'],
                    '--gpu',str(gpu),'--cpus',CPUS[gpu],'--out',str(folder)]
                if method == 'dflash':
                    argv += ['--shard','0','--shards','1']
                run_command('formal-greedy-margins-'+identity,argv)
            completed.append(validate(row,folder,agreements[key]))
            save(out/f'lane{gpu}-progress.json',dict(completed=completed,updated_unix=time.time()))
            pending.task_done()

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lane,gpu) for gpu in range(8)]
        for future in futures:
            try:
                results += future.result()
            except Exception as exc:
                errors.append(repr(exc))
    if errors:
        save(out/'failure.json',dict(errors=errors,finished_unix=time.time()))
        raise RuntimeError(errors)
    assert pending.empty() and len(results) == 36
    save(out/'complete.json',dict(cells=36, cases=sum(r['cases'] for r in results),
        results=sorted(results,key=lambda r:(r['method'],r['model'],r['benchmark'])),
        finished_unix=time.time(), limitation='Sampling(T1) same-seed differences are not classified as greedy numerical discrepancies.'))


if __name__ == '__main__':
    main()
