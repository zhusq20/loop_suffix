"""Archive completed numerical records needed for paper tables and figures.

Preserves all statistics, per-request timings, wait entries and histograms. Generated
token arrays are replaced by their SHA256 and length, retaining output-match groups.
Never reads an unfinished request stream or changes a running experiment's files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from loop_table2_suite import read, save, digest


def archive(suite, destination, kind, keys=None):
    destination.mkdir(parents=True, exist_ok=True)
    protocol = read(suite/'protocol.json')
    jobs = [read(p) for p in sorted((suite/'jobs').glob('*.json'))]
    if kind == 'timing':
        jobs = [j for j in jobs if j['phase']=='measurement']
    completed = [j for j in jobs if j['outcome']=='complete']
    if keys is not None:
        completed = [j for j in completed if j['key'] in keys]
        assert {j['key'] for j in completed} == set(keys)
    target = destination/'paper-data.jsonl.gz'
    temporary = target.with_suffix('.gz.tmp')
    provenance, count = [], 0
    with temporary.open('wb') as binary:
        with gzip.GzipFile(filename='', mode='wb', fileobj=binary, mtime=0) as stream:
            for job in completed:
                if kind=='headroom':
                    run = suite/'runs'/job['key']
                    journal = run/'requests.jsonl'
                    done = read(run/'complete.json')
                    assert digest(journal)==done['requests_sha256']
                    batch = 1
                else:
                    run = Path(job['output'])
                    assert (run/'complete.json').exists()
                    batch = job['batch']
                    journal = run/f"{job['arm']}-b{batch}-seed0.jsonl"
                before = digest(journal)
                ids, tokens, calls, rows, output_tokens = set(), 0, 0, 0, 0
                for line in journal.read_text().splitlines():
                    source = json.loads(line)
                    rid = source.get('request_id',source.get('source_id'))
                    assert rid not in ids
                    ids.add(rid)
                    assert source['stats']['tokens']==len(source['tokens'])-1
                    record = {k:v for k,v in source.items() if k!='tokens'}
                    record.update(job_key=job['key'],model=job['model'],batch=batch,
                        variant=job.get('variant'),arm=job.get('arm'),mode=job.get('mode'),
                        output_token_count=len(source['tokens']),
                        output_sha256=hashlib.sha256(json.dumps(source['tokens'],separators=(',',':')).encode()).hexdigest())
                    stream.write((json.dumps(record,sort_keys=True,separators=(',',':'))+'\n').encode())
                    count += 1
                    tokens += source['stats']['tokens']
                    calls += source['stats']['calls']
                    rows += source['stats']['rows']
                    output_tokens += len(source['tokens'])
                assert digest(journal)==before
                if kind in ['ablation','tree_sampling']:
                    assert len(ids)==480
                elif kind=='headroom':
                    assert len(ids)==done['requests']==120
                elif kind=='timing':
                    assert len(ids)==protocol['requests']==24
                if 'summary' in job:
                    assert output_tokens==job['summary']['output_tokens']
                provenance.append(dict(job_key=job['key'],model=job['model'],batch=batch,
                    arm=job.get('arm'),variant=job.get('variant'),mode=job.get('mode'),
                    requests=len(ids),decode_tokens=tokens,loop_steps=calls,rows=rows,
                    output_tokens=output_tokens,source=str(journal),source_sha256=before,
                    summary=job.get('summary')))
    temporary.replace(target)
    manifest = dict(version=1,kind=kind,complete_jobs=len(completed),planned_jobs=len(jobs),
        all_complete=len(completed)==len(jobs),requests=count,file=target.name,
        sha256=digest(target),bytes=target.stat().st_size,
        protocol_sha256=digest(suite/'protocol.json'),
        exporter_sha256=digest(Path(__file__)),
        schema='One JSON per request: all original stats/trace/waits/wait_entries/timings retained. '
               'Top-level tokens replaced with output_token_count and SHA256 of compact JSON integer array. '
               'Use output_sha256 for equal-output groups; no generated text is needed for these analyses.',
        sources=provenance)
    save(destination/'paper-data-manifest.json',manifest)
    print(json.dumps(dict(file=str(target),requests=count,jobs=len(completed),bytes=target.stat().st_size)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--kind',choices=['ablation','headroom','tree_sampling','timing'],required=True)
    args=parser.parse_args()
    archive(args.suite.resolve(),(args.out or args.suite).resolve(),args.kind)
