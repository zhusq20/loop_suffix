"""Bind measured T1 cells to the actually tested sampling implementations.

This reuses CPU test evidence and never repeats benchmark inference. It does
not turn small-vocabulary checks into a claim of real-model distribution equality.
"""
import argparse
import csv
import itertools
from pathlib import Path
import time

from packed_dflash_evidence import read, save, sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
EXPECTED = set(itertools.product(['dflash','loopahead'],['huginn-3.5b','ouro-2.6b','ouro-1.4b'],
                               ['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench']))
MODULES = dict(dflash=['dflash_impl/decode.py'],
               loopahead=['loopahead/sampling.py','loopahead/decode.py','loopahead/admission.py'])


def validate_tests(folder):
    tests, launch, done = [read(folder/name) for name in ['tests.json','launcher.json','exit.json']]
    manifest = read(folder/'source-manifest.json')
    assert type(done['exit_code']) is int and done['exit_code'] == 0
    assert all(done[k] == v for k,v in launch.items())
    assert launch['started_unix'] <= tests['finished_unix'] <= done['finished_unix']
    assert tests['status'] == 'passed_existing_sampling_and_scheduler_tests'
    assert tests['tests'] == 17 and tests['failures'] == tests['errors'] == 0
    assert tests['cuda_initialized'] is False and tests['source_sha256'] == manifest
    assert sha(folder/'run_tests.py') == launch['runner_sha256']
    assert sha(folder/'source-manifest.json') == launch['source_manifest_sha256']
    assert launch['argv'][2:] == [str(folder/'run_tests.py'),str(BASE),str(folder)]
    for name,digest in manifest.items():
        assert sha(folder/'source'/name) == digest
        assert sha(BASE/name) == digest
    return manifest


def measured_sources(row, tested):
    run = Path(row['method_run'])
    meta = read(run/'metadata.json')
    if row['target_backend'] == 'packed_full_loop':
        assert row['method'] == 'dflash'
        job = meta['job']
        assert job['temperature'] == 1 and job['max_new_tokens'] == 512 and job['batch'] == 1
        assert (job['model'],job['dataset']) == (row['model'],row['benchmark'])
        manifest = meta['source_sha256']
        filename = 'baselines/dflash/decode.py'
        suite = next(p for p in run.parents if (p/'source'/filename).is_file())
        mapping = {'dflash_impl/decode.py':suite/'source'/filename}
        assert manifest[filename] == tested['dflash_impl/decode.py']
    else:
        assert meta['temperature'] == 1 and meta['top_p'] == 1 and meta['top_k'] == 0
        assert meta['max_new_tokens'] == 512 and meta['model'] == row['model'] and meta['dataset'] == row['benchmark']
        manifest = read(run/'source-manifest.json')
        mapping = {name:run/'source'/name for name in MODULES[row['method']]}
        for name in mapping:
            assert manifest[name] == tested[name]
    assert not meta['graph'] and not meta['compile']
    for name,path in mapping.items():
        assert sha(path) == tested[name]
    return dict(method_run=str(run),metadata_sha256=sha(run/'metadata.json'),
                target_backend=row['target_backend'],actual_source_paths={k:str(v) for k,v in mapping.items()},
                tested_source_sha256={name:tested[name] for name in mapping},
                runtime_torch=meta['torch'],test_scope='source identity plus recorded CPU algorithm checks')


def audit(tests, out, partial):
    assert not out.exists(), 'Preserve older snapshots; choose a fresh output directory.'
    tested = validate_tests(tests)
    csv_path = BASE/'combined-results/results.csv'
    snapshot_hash = sha(csv_path)
    with csv_path.open() as stream:
        items = [r for r in csv.DictReader(stream) if r['temperature'] == '1']
    keys = [(r['method'],r['model'],r['benchmark']) for r in items]
    assert len(keys) == len(set(keys)) and set(keys) <= EXPECTED
    evidence = read(BASE/'combined-results/evidence.json')
    status = read(BASE/'combined-results/status.json')
    assert status['results_sha256'] == snapshot_hash
    actual, artifacts = {}, {}
    for row,identity in zip(items,keys,strict=True):
        assert row['status'] == 'complete_with_declared_limitations'
        proof = evidence['cells']['/'.join([*identity,'1'])]
        for record in proof['sources'].values():
            for name,digest in record['sha256'].items():
                path = str(Path(record['path'])/name)
                if path not in artifacts:
                    artifacts[path] = sha(path)
                assert artifacts[path] == digest
        actual['/'.join(identity)] = measured_sources(row,tested)
    missing = sorted(EXPECTED-set(keys))
    planned_path = BASE/'logs/ouro14-packed-evaluation-v1/suite/source/baselines/dflash/decode.py'
    assert sha(planned_path) == tested['dflash_impl/decode.py']
    assert sha(csv_path) == snapshot_hash
    out.mkdir(parents=True)
    sources = freeze_sources(out,[Path(__file__),BASE/'packed_dflash_evidence.py',BASE/'provenance.py'])
    report = dict(status='complete_T1_source_coverage_not_distribution_proof' if not missing else 'partial_T1_source_coverage',
        expected_cells=36,completed_cells=len(actual),missing=missing,cells=actual,
        tests_directory=str(tests),test_artifact_sha256={name:sha(tests/name) for name in
            ['tests.json','launcher.json','exit.json','source-manifest.json','run_tests.py']},
        verified_measurement_artifact_sha256=artifacts,unchanged_results_csv_sha256=snapshot_hash,
        report_source_sha256=sources,new_gpu_runs=0,checked_unix=time.time(),
        planned_ouro14_DFlash=dict(path=str(planned_path),sha256=sha(planned_path),
                                  status='frozen_code_matches_not_completed_measurement'),
        limitations=['CPU tests cover small-vocabulary conditional enumeration/Monte Carlo and scheduler/EOS/cap/admission oracles.',
                     'Code identity binds those tests to measured implementations; it does not statistically prove distribution equality for every real-model prompt.',
                     'The scheduler sampling oracle is near-deterministic; general nondegenerate rejection tests exercise the separate sampling routines.',
                     'BF16 full-depth logits can vary with execution shape; common-prefix numerical evidence and output differences are reported separately.',
                     'T1 same-seed token agreement is descriptive. No benchmark outputs were regenerated here.',
                     'Future Ouro1.4B DFlash source identity is a preparation check and does not count toward completed cells.'])
    save(out/'report.json',report)
    print(dict(status=report['status'],completed_cells=len(actual),missing=len(missing),new_gpu_runs=0))
    if not partial:
        assert not missing, 'All 36 actual T1 measurements must exist for complete source coverage.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args = parser.parse_args()
    audit(args.tests.resolve(),args.out.resolve(),args.partial)
