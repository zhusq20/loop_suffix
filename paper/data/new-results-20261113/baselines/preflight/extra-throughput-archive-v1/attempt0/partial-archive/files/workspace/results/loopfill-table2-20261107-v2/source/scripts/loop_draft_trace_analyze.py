"""Summarize lifecycle counts; never interpret instrumented wall time as throughput."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(c):
    roots, selected, selections = c['root_entries'], c['selected_entries'], c['selections']
    return dict(root_entries=roots,
        **{f'root_step_{d}_pct':100*c.get(f'root_step_{d}',0)/roots for d in range(5)},
        **{f'wait_from_step_{d}':(4-d)*c.get(f'root_step_{d}',0) for d in range(4)},
        selected_suffix_pct=100*c['selected_suffix_entries']/selected,
        selected_zero_pct=100*c['selected_zero_entries']/selected,
        **{f'selection_changed_c{v}_pct':100*c.get(f'changed_c{v}',0)/selections for v in ('0.25','1.0','2.0')},
        **{f'entries_replaced_c{v}_pct':100*c.get(f'replaced_entries_c{v}',0)/selected for v in ('0.25','1.0','2.0')},
        discarded_compute_pct=100*sum(c.get(f'release_{reason}_rows',0) for reason in ('budget','rejected','end'))/c['rows'],
        correct_path_budget_waste_pct_of_rows=100*c['correct_path_budget_waste']/c['rows'],
        correct_path_rebuilt_pct=100*c['correct_path_rebuilt_nodes']/c['correct_path_allocated_nodes'],
        delayed_correct_admission_pct=100*c['delayed_correct_admissions']/c['correct_path_proposed_nodes'])


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text())
    table, details, examples = [], {}, []
    for job in protocol['jobs']:
        arm=job['arm'];run=out/'runs'/arm;prefix=f'{arm}-b4'
        assert (run/'complete.json').exists(), run
        trace=json.loads((run/f'{prefix}-lifecycle-summary.json').read_text())
        rows=read(run/f'{prefix}-seed0.jsonl')
        assert {r['request_id'] for r in rows}==set(protocol['request_ids'])
        tokens=sum(r['stats']['tokens'] for r in rows)
        c=trace['totals']
        assert c['calls']==sum(r['stats']['calls'] for r in rows)
        assert c['rows']==sum(r['stats']['rows'] for r in rows)
        if arm in ('budget-raw','budget-fixed-length'):
            assert c['changed_c1.0']==0
        summary=dict(arm=arm,decode_tokens=tokens,gamma=4*tokens/c['calls'],rows_per_token=c['rows']/tokens,
                     calls=c['calls'],**metrics(c))
        table.append(summary)
        categories={}
        for r in trace['requests']:
            cat=r['category'] if r['category'] in ('translation','summarization','qa','rag','math_reasoning') else 'mt_bench'
            categories.setdefault(cat,Counter()).update(r['counters'])
        details[arm]=dict(totals=c,categories={k:metrics(v) for k,v in categories.items()})
        entries={(r['request_id'],r['position']):r for r in read(run/f'{prefix}-root-entries.jsonl')}
        nodes=read(run/f'{prefix}-correct-node-lifecycle.jsonl')
        candidates=[]
        for n in nodes:
            entry=entries.get((n['request_id'],n['position']))
            if entry and not entry['terminal']:
                lost=sum(a['spent'] for a in n['allocations'] if a['reason']=='budget')
                candidates.append(dict(arm=arm,**n,entry=entry,lost_correct_rows=lost))
        examples.extend(sorted(candidates,key=lambda n:n['lost_correct_rows'],reverse=True)[:5])
    report=dict(formal_throughput=False,protocol=str(out/'protocol.json'),table=table,details=details,
                limitations=protocol['limitations'],examples=examples)
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    with (out/'analysis.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(table,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    analyze(p.parse_args().out)
