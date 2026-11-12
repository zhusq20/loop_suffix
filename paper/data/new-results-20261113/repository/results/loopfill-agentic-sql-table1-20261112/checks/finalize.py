"""Audit the complete 28-cell agentic supplement and export its Table 1 column."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def finalize(suite, repository):
    protocol, jobs = read(suite/'protocol.json'), read(suite/'jobs.json')
    if read(suite/'status.json')['status'] != 'complete' or len(jobs) != 28:
        raise ValueError('All 28 formal cells must complete before paper export')
    for name, expected in protocol['input_files_sha256'].items():
        assert digest(suite/'inputs'/name) == expected, name
    for name, expected in protocol['source_sha256'].items():
        assert digest(suite/'source'/name) == expected, name
    expected_rows = {model: [json.loads(line) for line in (suite/'inputs'/(model+'.jsonl')).open()]
                     for model in protocol['models']}
    assert all(len(rows) == 100 for rows in expected_rows.values())
    audits, devices = [], {}
    for job in jobs:
        directory = suite/'runs'/job['id']
        metadata, summary = read(directory/'metadata.json'), read(directory/'summary.json')
        assert summary['status'] == 'complete' and summary['reportable']
        assert not metadata['smoke'] and not metadata['diagnostic']
        assert metadata['max_new_tokens'] == 512 and metadata['batch'] == 1
        assert metadata['source_sha256'] == protocol['source_sha256']
        assert metadata['input_sha256'] == protocol['models'][job['model']]['input_sha256']
        assert metadata['checkpoint_sha256'] == protocol['models'][job['model']]['checkpoint_sha256']
        lane = (job['model'], job['temperature'])
        assert devices.setdefault(lane, metadata['gpu_uuid']) == metadata['gpu_uuid']
        journal = directory/'requests.jsonl'
        assert digest(journal) == summary['requests_sha256']
        rows = [json.loads(line) for line in journal.open()]
        assert len(rows) == 100
        for row, expected in zip(rows, expected_rows[job['model']]):
            for key in ('source_id','question_id','phase','instance_id','prompt_tokens','prompt_sha256'):
                assert row[key] == expected[key], (job['id'], key)
            assert 1 <= len(row['tokens']) <= 512
            assert row['stats']['tokens'] == len(row['tokens']) - 1
            assert row['stats']['decode_s'] > 0
            assert row['stats']['prefill_s'] >= 0
        tokens = sum(row['stats']['tokens'] for row in rows)
        seconds = sum(row['stats']['decode_s'] for row in rows)
        calls = sum(row['stats']['calls'] for row in rows)
        depth = 32 if job['model'].startswith('huginn') else 4
        recomputed = dict(decode_tokens=tokens, decode_s=seconds, decode_token_s=tokens/seconds,
            gamma=depth*tokens/calls, mean_decode_s=seconds/100,
            prefill_s=sum(row['stats']['prefill_s'] for row in rows),
            output_tokens=sum(len(row['tokens']) for row in rows),
            cap_stopped=sum(row['stop_reason']=='length' for row in rows))
        for key, value in recomputed.items():
            assert math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12), (job['id'], key)
        if job['arm'] == 'ar':
            assert recomputed['gamma'] == 1.
        audits.append(dict(cell=job['id'], requests=100, requests_sha256=digest(journal), **recomputed))
    assert len(set(devices.values())) == 4
    # Regenerate only from the frozen implementation and its complete journals.
    subprocess.run([sys.executable, str(suite/'source/scripts/loop_agentic_suite.py'),
                    'summarize','--suite',str(suite)], check=True)
    measured = list(csv.DictReader((suite/'measurements.csv').open()))
    index = {row['id']:row for row in measured}
    assert len(index) == 28
    for audit in audits:
        row = index[audit['cell']]
        for key in ('decode_tokens','decode_s','decode_token_s','gamma','mean_decode_s'):
            assert math.isclose(float(row[key]), audit[key], rel_tol=1e-12, abs_tol=1e-12)
    checks = suite/'checks'
    checks.mkdir(exist_ok=True)
    result = dict(passed=True, cells=28, requests=2800, verified_unix=time.time(),
        same_device_per_model_temperature=True, complete_input_and_source_hashes=True,
        cells_audited=audits, note='Integrity and metric audit, not a task-quality evaluation.')
    (checks/'final-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    destinations = {'table1-agentic.tex':repository/'paper/tables/table1-agentic-sql.tex',
                    'measurements.csv':repository/'paper/data/agentic-sql-measurements.csv'}
    for filename, target in destinations.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(suite/filename, target)
    manifest = read(suite/'inputs/manifest.json')
    provenance = dict(suite=str(suite), dataset=manifest['dataset'], semantics=manifest['semantics'],
        input_sha256=manifest['input_sha256'], phase_counts=manifest['phase_counts'],
        models=list(protocol['models']), methods=protocol['main_arms'], temperatures=[0,1],
        max_new_tokens=512, batch=1, sampling_seed=0, timing_repeats=1,
        audit_sha256=digest(checks/'final-audit.json'),
        outputs={str(target.relative_to(repository)):digest(target) for target in destinations.values()},
        timing='Decode-only; prefill, warmup, loading and journal I/O excluded. Includes suffix publication.',
        limitations='Fixed historical SQL requests, not closed-loop task performance. '
                    'B1 token/s is not concurrent serving throughput. Output lengths can differ.')
    (repository/'paper/data/agentic-sql-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(status='audited_and_exported', cells=28, requests=2800,
                          outputs=[str(p) for p in destinations.values()])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    finalize(args.suite.resolve(), args.repository.resolve())
