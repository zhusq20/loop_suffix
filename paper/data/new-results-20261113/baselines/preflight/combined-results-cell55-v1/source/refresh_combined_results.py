"""Refresh partial results with real collector receipts and preserve old rows.

This CPU wrapper invokes the existing frozen collectors. It never changes their
measurement rules or launches inference. A fresh proof directory is required.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import subprocess
import time

from packed_dflash_evidence import read, save, sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent


def key(row):
    return tuple(row[k] for k in ['method','model','benchmark','temperature'])


def refresh(out, legacy):
    assert not out.exists(), 'Preserve earlier collection evidence; choose a fresh output directory.'
    out.mkdir(parents=True)
    os.sched_setaffinity(0, {80,81,82,83})
    csv_path = BASE/'combined-results/results.csv'
    before = csv_path.read_bytes()
    (out/'before-results.csv').write_bytes(before)
    prior_rows = list(csv.DictReader(io.StringIO(before.decode())))
    prior = {key(r):r for r in prior_rows}
    assert len(prior) == len(prior_rows)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    scripts = [('legacy','collect_results.py',['--partial'])] if legacy else []
    scripts.append(('combined','collect_combined_results.py',[]))
    exits = {}
    for name, script, arguments in scripts:
        argv = ['/opt/conda/bin/python','-u',str(BASE/script),*arguments]
        with (out/f'{name}.log').open('x') as log:
            child = subprocess.Popen(argv, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
        launch = dict(argv=argv,pid=child.pid,started_unix=time.time(),source_sha256=sha(BASE/script))
        save(out/f'{name}-launcher.json',launch)
        code = child.wait()
        save(out/f'{name}-exit.json',dict(launch,exit_code=code,finished_unix=time.time()))
        assert code == 0, f'Inspect retained {name} failure; do not overwrite its evidence.'
        exits[name] = dict(path=str(out/f'{name}-exit.json'),sha256=sha(out/f'{name}-exit.json'),exit_code=code)
    with csv_path.open() as stream:
        current_rows = list(csv.DictReader(stream))
    current = {key(r):r for r in current_rows}
    assert len(current) == len(current_rows) and len(current) >= len(prior)
    assert all(current.get(k) == v for k,v in prior.items()), 'An existing row changed; inspect before publication.'
    added = [r for k,r in current.items() if k not in prior]
    evidence = read(BASE/'combined-results/evidence.json')
    new_proof = { '/'.join(key(r)):evidence['cells']['/'.join(key(r))] for r in added }
    for item in new_proof.values():
        for source in item['sources'].values():
            for name,digest in source['sha256'].items():
                assert sha(Path(source['path'])/name) == digest
        for receipt in item.get('supplement_exits',{}).values():
            assert sha(receipt['path']) == receipt['sha256'] and read(receipt['path'])['exit_code'] == 0
        timed = item.get('original_timed_exit')
        if timed is not None:
            if timed['actual_exit_code'] == 0:
                assert sha(timed['path']) == timed['sha256'] and read(timed['path'])['exit_code'] == 0
            else:
                assert timed['actual_exit_code'] is None and timed['limitation']
    status = read(BASE/'combined-results/status.json')
    assert status['completed'] == len(current) and status['results_sha256'] == sha(csv_path)
    sources = freeze_sources(out,[Path(__file__),BASE/'provenance.py',BASE/'packed_dflash_evidence.py',
                                  *[BASE/script for _,script,_ in scripts]])
    report = dict(status='passed',completed=len(current),expected=72,previous_rows=len(prior),
        all_previous_rows_unchanged=True,added=added,new_cell_evidence=new_proof,collector_exits=exits,
        prior_csv_sha256=sha(out/'before-results.csv'),results_sha256=sha(csv_path),
        evidence_sha256=sha(BASE/'combined-results/evidence.json'),source_sha256=sources,
        new_gpu_measurements=0,checked_unix=time.time(),
        limitation='Measurement collection only. Numerical checks and unresolved original exit statuses retain their declared scope.')
    save(out/'report.json',report)
    print(dict(completed=len(current),unchanged=len(prior),added=[key(r) for r in added]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--refresh-legacy',action='store_true')
    args = parser.parse_args()
    refresh(args.out.resolve(),args.refresh_legacy)
