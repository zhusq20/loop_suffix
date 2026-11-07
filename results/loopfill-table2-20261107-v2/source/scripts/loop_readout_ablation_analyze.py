"""Analyze paired readout ablations without mixing trace time into throughput."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def work(rows):
    c=Counter()
    for row in rows:
        for k in ('tokens','calls','rows','readout_rows','speculate_calls','wasted_rows'):
            c[k]+=row['stats'][k]
    return dict(c,calls_per_token=c['calls']/c['tokens'],rows_per_token=c['rows']/c['tokens'],
                readout_rows_per_token=c['readout_rows']/c['tokens'],
                discarded_rows_pct=100*c['wasted_rows']/c['rows'])


def analyze(out):
    protocol=json.loads((out/'protocol.json').read_text())
    table=[];runs={};inputs={}
    for j in protocol['jobs']:
        path=out/'runs'/j['arm']
        assert (path/'complete.json').exists(),path
        rows={r['request_id']:r for r in read(path/'budget-raw-b4-seed0.jsonl')}
        assert set(rows)==set(protocol['request_ids'])
        summary=json.loads((path/'budget-raw-b4-seed0-summary.json').read_text())
        cfg=summary['config'];v=protocol['variants'][j['variant']]
        assert cfg['top_m']==v['top_k'] and cfg['readout_depths']==v['depths']
        assert cfg['draft_node_budget']==16 and cfg['scoring']=='raw'
        assert cfg['use_suffix'] and cfg['child_suffix_caps']==[4]*v['top_k']
        assert cfg['positive_only'] is False
        assert not summary['profiled'] and not summary['controller_profiled']
        entry=dict(variant=j['variant'],kind=j['kind'],repeat=j['repeat'],**work(rows.values()))
        if j['kind']=='timing':
            assert not summary.get('diagnostic',False)
            entry.update(output_tokens=summary['output_tokens'],wall_s=summary['wall_s'],tokens_per_s=summary['output_tput'])
        else:
            assert summary['diagnostic']
            trace=json.loads((path/'budget-raw-b4-lifecycle-summary.json').read_text())['totals']
            assert trace['calls']==entry['calls'] and trace['rows']==entry['rows']
            entry['root_steps']={str(k):trace.get(f'root_step_{k}',0) for k in range(5)}
        table.append(entry);runs[j['arm']]=rows
        for file in path.glob('*.json*'):
            inputs[str(file.relative_to(out))]=hashlib.sha256(file.read_bytes()).hexdigest()
    timing=[]
    for v in protocol['variants']:
        measured=[r for r in table if r['variant']==v and r['kind']=='timing']
        rates=[r['tokens_per_s'] for r in measured]
        timing.append(dict(variant=v,repeats=len(rates),tokens_per_s_mean=mean(rates),
            tokens_per_s_stdev=stdev(rates) if len(rates)>1 else None,
            tokens_per_s_min=min(rates),tokens_per_s_max=max(rates),
            wall_s_mean=mean(r['wall_s'] for r in measured)))
    pairs=[]
    for repeat in range(3):
        a,b=(next(t for t in table if t['kind']=='timing' and t['repeat']==repeat and t['variant']==v)
             for v in ('f1-top4','f1f2-top2'))
        ra,rb=(runs[f'timing-r{repeat}-{v}'] for v in ('f1-top4','f1f2-top2'))
        same=[rid for rid in ra if ra[rid]['tokens']==rb[rid]['tokens']]
        pairs.append(dict(repeat=repeat,top4_vs_dual_tput_change_pct=100*(a['tokens_per_s']/b['tokens_per_s']-1),
            top4_vs_dual_wall_change_pct=100*(a['wall_s']/b['wall_s']-1),
            identical_output_request_ids=sorted(same),
            identical_output_work={v:work(rr[rid] for rid in same) for v,rr in [('f1-top4',ra),('f1f2-top2',rb)]} if same else {}))
    stable={}
    for v in protocol['variants']:
        base=runs[f'timing-r0-{v}']
        stable[v]={label:sum(rr[rid]['tokens']==base[rid]['tokens'] for rid in base)
                   for label,rr in runs.items() if label.endswith(v)}
    common=set(protocol['request_ids'])
    reference=next(iter(runs.values()))
    for rows in runs.values():
        common&={rid for rid in rows if rows[rid]['tokens']==reference[rid]['tokens']}
    trace_compare={v:work(runs[f'trace-r0-{v}'][rid] for rid in common)
                   for v in protocol['variants']} if common else {}
    trace_entries={v:{(e['request_id'],e['position']):e for e in read(
        out/'runs'/f'trace-r0-{v}'/'budget-raw-b4-root-entries.jsonl') if not e['terminal']}
        for v in protocol['variants']}
    cold_causes={}
    for v in protocol['variants']:
        nodes={(n['request_id'],n['position']):n for n in read(
            out/'runs'/f'trace-r0-{v}'/'budget-raw-b4-correct-node-lifecycle.jsonl')}
        counts=Counter()
        for key,entry in trace_entries[v].items():
            if entry['step']!=0:
                continue
            node=nodes[key];proposal=node['first_proposal']
            if proposal is None or proposal>=entry['tick']:
                counts['no_earlier_proposal']+=1
            elif node['allocations'][0]['tick']<entry['tick']:
                counts['earlier_allocated_then_lost']+=1
            else:
                counts['proposed_but_not_earlier_allocated']+=1
        cold_causes[v]=dict(counts)
    transitions=Counter(); transition_examples=[]
    a,b=(runs[f'trace-r0-{v}'] for v in ('f1-top4','f1f2-top2'))
    for key,dual in trace_entries['f1f2-top2'].items():
        rid,pos=key
        if key not in trace_entries['f1-top4'] or a[rid]['tokens'][:pos+1]!=b[rid]['tokens'][:pos+1]:
            continue
        first=trace_entries['f1-top4'][key]
        transitions[f"{dual['step']}->{first['step']}"]+=1
        if dual['step']==2 and first['step']==0:
            transition_examples.append(dict(request_id=rid,position=pos,dual_origin=dual['origin'],
                dual_step=2,top4_step=0))
    report=dict(protocol=str(out/'protocol.json'),timing_summary=timing,paired_timing=pairs,
        output_agreement_with_own_repeat0=stable,common_identical_output_request_ids=sorted(common),
        common_identical_output_trace_work=trace_compare,runs=table,
        common_prefix_root_steps_dual_to_top4=dict(transitions),
        step_zero_node_diagnosis=cold_causes,
        lost_two_round_headstart_examples=transition_examples,
        caveats=protocol['limitations']+['Identical-output subsets are post hoc diagnostics, not independent or history-controlled causal estimates.'],
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(report,indent=2))
    (out/'analysis-input-sha256.json').write_text(json.dumps(inputs,indent=2))
    with (out/'timing.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(timing[0]));writer.writeheader();writer.writerows(timing)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    analyze(p.parse_args().out)
