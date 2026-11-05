"""Compare diagnostic work counts, output agreement, and common-prefix readiness."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def totals(rows):
    c = Counter()
    for row in rows:
        for key in ('tokens', 'calls', 'rows', 'wasted_rows', 'speculate_calls', 'readout_rows'):
            c[key] += row['stats'][key]
    return dict(c, calls_per_token=c['calls']/c['tokens'],
                rows_per_token=c['rows']/c['tokens'], gamma=4*c['tokens']/c['calls'],
                discarded_rows_pct=100*c['wasted_rows']/c['rows'])


def analyze(out, additional=()):
    protocol = json.loads((out/'protocol.json').read_text())
    paths = {'raw': Path(protocol['baseline']), **{j['arm']:out/'runs'/j['arm'] for j in protocol['jobs']}}
    for name, path in additional:
        if name in paths:
            raise ValueError('duplicate analysis arm: '+name)
        paths[name] = Path(path)
    rows, entries, traces = {}, {}, {}
    for name, path in paths.items():
        assert (path/'complete.json').exists(), path
        rows[name] = {r['request_id']:r for r in read(path/'budget-raw-b4-seed0.jsonl')}
        entries[name] = {(r['request_id'],r['position']):r for r in read(path/'budget-raw-b4-root-entries.jsonl') if not r['terminal']}
        traces[name] = json.loads((path/'budget-raw-b4-lifecycle-summary.json').read_text())['totals']
        assert set(rows[name]) == set(protocol['request_ids'])
        assert sum(r['stats']['calls'] for r in rows[name].values()) == traces[name]['calls']
    table = []
    for name, rr in rows.items():
        t = dict(arm=name, **totals(rr.values()))
        t['same_output_requests_vs_raw'] = sum(r['tokens']==rows['raw'][rid]['tokens'] for rid,r in rr.items())
        t['ready_on_entry_pct'] = 100*traces[name].get('root_step_4',0)/traces[name]['root_entries']
        t['zero_selected_pct'] = 100*traces[name].get('selected_zero_entries',0)/traces[name]['selected_entries']
        for key in ('calls_per_token', 'rows_per_token'):
            t[key+'_change_pct'] = 100*(t[key]/table[0][key]-1) if table else 0.
        table.append(t)
    opportunities = json.loads((Path(protocol['parent'])/'runs/budget-raw/source-opportunities.json').read_text())
    pairs = {}
    for name in rows:
        if name == 'raw':
            continue
        same = [rid for rid in rows[name] if rows[name][rid]['tokens']==rows['raw'][rid]['tokens']]
        prefixes = {}
        for rid, row in rows[name].items():
            length = 0
            for a,b in zip(row['tokens'],rows['raw'][rid]['tokens']):
                if a != b:
                    break
                length += 1
            prefixes[rid] = length
        counts = Counter()
        for key, entry in entries['raw'].items():
            rid, pos = key
            if pos >= prefixes[rid] or key not in entries[name]:
                continue
            delta = entries[name][key]['step'] - entry['step']
            counts['positions'] += 1
            counts['more_ready' if delta>0 else 'less_ready' if delta<0 else 'same_ready'] += 1
            counts['remaining_wait_change'] -= delta
        earlier = []
        for e in opportunities['examples']:
            rid, pos = e['request_id'],e['position'];key=(rid,pos)
            if pos >= prefixes[rid] or key not in entries[name]:
                continue
            earlier.append(dict(request_id=rid, position=pos,
                raw_step=entries['raw'][key]['step'],variant_step=entries[name][key]['step']))
        pairs[name] = dict(same_output_request_ids=sorted(same),
            same_output_totals={arm:totals(rows[arm][rid] for rid in same) for arm in ('raw',name)} if same else {},
            common_prefix_readiness=dict(counts), original_opportunity_readiness=earlier,
            original_opportunities_comparable=len(earlier),
            original_opportunities_more_ready=sum(e['variant_step']>e['raw_step'] for e in earlier),
            original_opportunities_less_ready=sum(e['variant_step']<e['raw_step'] for e in earlier))
    common = set.intersection(*[{rid for rid,r in rr.items() if r['tokens']==rows['raw'][rid]['tokens']} for rr in rows.values()])
    report = dict(formal_throughput=False, input_runs={k:str(v) for k,v in paths.items()},
        table=table, pairs=pairs, common_identical_output_request_ids=sorted(common),
        common_identical_output_totals={name:totals(rr[rid] for rid in common) for name,rr in rows.items()} if common else {},
        source_probe_counts=opportunities['counts'],
        caveats=protocol['limitations']+['Identical output subsets and shared prefixes still do not hold completion-history state fixed. Gamma is normalized model work, not measured throughput.'],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    with (out/'analysis.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    parser.add_argument('--additional',nargs=2,action='append',default=[],metavar=('NAME','RUN'))
    args=parser.parse_args();analyze(args.out,args.additional)
