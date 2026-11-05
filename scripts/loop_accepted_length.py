"""Extract loop-normalized accepted length from completed benchmark artifacts.

No inference or timing is performed. Aggregate gamma is T * sum(tokens) /
sum(calls), using per-request decode counters and excluding the prefill token.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time


MT_CATEGORIES = {'coding', 'writing', 'extraction', 'stem', 'reasoning',
                 'roleplay', 'math', 'humanities'}
TASKS = {'translation', 'summarization', 'math_reasoning', 'qa', 'rag', 'mt_bench'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def extract(suite, out):
    suite, out = suite.resolve(), out.resolve()
    protocol_path = suite / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    expected_ids = set(protocol['request_ids'])
    overall, categories, per_request, provenance = [], [], [], []
    jobs = [json.loads(p.read_text()) for p in (suite / 'jobs').glob('*.json')]
    for job in sorted(jobs, key=lambda j: (j['batch'], j['arm'], j['repeat'])):
        base = dict(key=job['key'], model=job.get('model', protocol.get('execution_args', {}).get('model')),
                    batch=job['batch'], arm=job['arm'], repeat=job['repeat'])
        result = dict(**base, outcome=job['outcome'], depth=None, requests=0,
                      decode_tokens=None, loop_steps=None, prefill_only_requests=None, gamma=None)
        overall.append(result)
        if job['outcome'] != 'complete':
            continue
        target = Path(job['output'])
        meta_path = target / 'metadata.json'
        meta = json.loads(meta_path.read_text())
        depth = meta['configs'][job['arm']]['depth_total']
        prefix = f"{job['arm']}-b{job['batch']}-seed0"
        rows_path = target / (prefix + '.jsonl')
        summary_path = target / (prefix + '-summary.json')
        rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
        summary = json.loads(summary_path.read_text())
        assert (target / 'complete.json').exists(), target
        assert len(rows) == len(expected_ids) and {r['request_id'] for r in rows} == expected_ids
        assert sum(len(r['tokens']) for r in rows) == summary['output_tokens']
        assert meta['questions_sha256'] == protocol['questions_sha256']
        assert meta['source_sha256'] == protocol['execution_source_sha256']
        groups = defaultdict(list)
        for row in rows:
            tokens, calls = row['stats']['tokens'], row['stats']['calls']
            assert type(tokens) is int and type(calls) is int and tokens >= 0 and calls >= 0
            assert tokens == len(row['tokens']) - 1 == row['n_tokens'] - 1
            assert (tokens == 0) == (calls == 0), (job['key'], row['request_id'])
            if job['arm'] == 'ar':
                assert calls == depth * tokens, (job['key'], row['request_id'])
            category = 'mt_bench' if row['category'] in MT_CATEGORIES else row['category']
            assert category in TASKS, category
            item = dict(**base, request_id=row['request_id'], category=category,
                        source_category=row['category'], depth=depth, decode_tokens=tokens,
                        loop_steps=calls, prefill_only=(calls == 0),
                        gamma=depth * tokens / calls if calls else None)
            per_request.append(item)
            groups[category].append(item)
        def aggregate(items):
            tokens = sum(r['decode_tokens'] for r in items)
            calls = sum(r['loop_steps'] for r in items)
            return dict(depth=depth, requests=len(items), decode_tokens=tokens,
                        loop_steps=calls, prefill_only_requests=sum(r['prefill_only'] for r in items),
                        gamma=depth * tokens / calls if calls else None)
        items = [r for group in groups.values() for r in group]
        result.update(aggregate(items))
        for category, group in sorted(groups.items()):
            categories.append(dict(**base, category=category, **aggregate(group)))
        provenance.append(dict(key=job['key'], files={str(p): digest(p) for p in
                          (meta_path, rows_path, summary_path)}))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'overall.csv', overall)
    write_csv(out / 'by-category.csv', categories)
    write_csv(out / 'per-request.csv', per_request)
    report = dict(created_unix=time.time(), suite=str(suite), protocol_sha256=digest(protocol_path),
                  formula='gamma = depth_total * sum(stats.tokens) / sum(stats.calls)',
                  zero_policy='Per-request gamma is undefined for tokens=calls=0; retain and count these requests.',
                  aggregation='Ratio of sums, not the arithmetic mean of request gamma values.',
                  category_mapping={'mt_bench': sorted(MT_CATEGORIES)},
                  counts=dict(Counter(j['outcome'] for j in jobs)),
                  validated_requests=len(per_request), provenance=provenance)
    tmp = out / 'audit.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(out / 'audit.json')
    print(json.dumps(dict(out=str(out), counts=report['counts'], validated_requests=len(per_request))))
    for row in overall:
        print(row['batch'], row['arm'], row['outcome'], row['gamma'])


def summarize_models(suites, out):
    """Rebuild separate throughput/gamma tables; keep each suite's own provenance."""
    out.mkdir(parents=True, exist_ok=True)
    measurements, accepted, protocols = [], [], []
    for suite in suites:
        suite = suite.resolve()
        protocol = json.loads((suite/'protocol.json').read_text())
        model = protocol['execution_args']['model']
        extract(suite, suite/'accepted-length')
        with (suite/'accepted-length/overall.csv').open() as f:
            gamma = {r['key']: r for r in csv.DictReader(f)}
        for path in sorted((suite/'jobs').glob('*.json')):
            job = json.loads(path.read_text())
            row = dict(model=model, arm=job['arm'], batch=job['batch'], outcome=job['outcome'],
                       throughput=None, gamma=None, output_tokens=None, wall_s=None,
                       peak_allocated_bytes=None, same_as_serial_ar=None, suite=str(suite))
            if job['outcome']=='complete':
                prefix = f"{job['arm']}-b{job['batch']}-seed0"
                summary = json.loads((Path(job['output'])/f'{prefix}-summary.json').read_text())
                row.update(throughput=summary['output_tokens']/summary['wall_s'],
                           gamma=float(gamma[job['key']]['gamma']) if gamma[job['key']]['gamma'] else None,
                           output_tokens=summary['output_tokens'], wall_s=summary['wall_s'],
                           peak_allocated_bytes=summary.get('peak_allocated_bytes'),
                           same_as_serial_ar=summary.get('same_as_serial_ar'))
            measurements.append(row)
        category_path = suite/'accepted-length/by-category.csv'
        if category_path.exists():
            with category_path.open() as f:
                accepted.extend(dict(r, model=model) for r in csv.DictReader(f))
        protocols.append(dict(model=model, suite=str(suite), protocol_sha256=digest(suite/'protocol.json'),
                              input_sha256=protocol['questions_sha256'],
                              execution_source_sha256=protocol['execution_source_sha256'],
                              prompt_format=protocol['execution_args']['prompt_format']))
    keys = [(r['model'], r['arm'], r['batch']) for r in measurements]
    if len(keys) != len(set(keys)):
        raise ValueError('Multiple suites supply the same model/method/batch cell')
    write_csv(out/'measurements.csv', measurements)
    write_csv(out/'by-category.csv', accepted)
    names = {'ar':'AR', 'wavefront':'WFD (same engine)', 'loopspec-dual':'LoopSpec (same engine)',
             'budget-suffix':'Suffix-only', 'budget-raw':'LoopFill-raw', 'budget-online':'LoopFill-online'}
    for metric in ('throughput','gamma'):
        table = []
        for model in dict.fromkeys(r['model'] for r in measurements):
            for arm, label in names.items():
                row = dict(model=model, method=label)
                for batch in (1,2,4,8,16):
                    item = next(r for r in measurements if (r['model'],r['arm'],r['batch'])==(model,arm,batch))
                    value = item[metric]
                    row[f'B{batch}'] = f'{value:.3f}' if value is not None else item['outcome']
                table.append(row)
        write_csv(out/f'{metric}.csv', table)
        columns = list(table[0])
        lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        lines += ['| '+' | '.join(str(r[k]) for k in columns)+' |' for r in table]
        (out/f'{metric}.md').write_text('\n'.join(lines)+'\n')
    save = dict(protocols=protocols, cells=len(measurements),
                counts=dict(Counter(r['outcome'] for r in measurements)),
                formulas={'throughput':'sum(output_tokens) / wall_s',
                          'gamma':'T * sum(stats.tokens) / sum(stats.calls)'},
                limitations=['Raven uses raw first-turn text without automatic BOS; Ouro/Huginn use released chat templates.',
                             'Historical two-model results keep their original frozen backend/protocol.',
                             'TinyLlama full480 deferred; OOM/error/deferred cells contain no metrics.',
                             'Gamma measures recurrent steps, not FLOPs or cross-model wall efficiency.',
                             'BF16 output disagreement is retained; no quality/equal-output claim.'])
    (out/'audit.json').write_text(json.dumps(save, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path,
                        default=Path('results/loopfill-eager-b1-b16-huginn-20261105'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    extract(args.suite, args.out or args.suite / 'accepted-length')
