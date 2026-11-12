"""Observe both suffix sources at actual query times without changing the winner."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from loop_draft_trace import DraftTrace


def snapshot(draft):
    return dict(tokens=[int(x) for x in draft.token_ids], parents=[int(x) for x in draft.parents],
                probs=[float(x) for x in draft.probs], match_len=int(draft.match_len), score=float(draft.score))


def paths(base, draft):
    result=[]
    for token, parent in zip(draft['tokens'], draft['parents'], strict=True):
        assert -1 <= parent < len(result)
        result.append((base if parent < 0 else result[parent]) + (token,))
    return result


class SourceProbe:
    def __init__(self, cache, traces):
        self.cache, self.traces = cache, traces
        self.queries=[]
        self.latest={}

    def speculate(self, rid, context, **options):
        local=self.cache.local.speculate(rid, context, **options)
        history=self.cache.history.speculate(rid, context, **options)
        winner='local' if local.score >= history.score else 'history'
        query=dict(request_id=rid, tick=self.traces[rid].counts['calls'],
                   parent=tuple(self.cache.responses[rid]), kind='root', winner=winner,
                   local=snapshot(local), history=snapshot(history),
                   completed_history=len(self.cache.publications))
        self.queries.append(query)
        self.latest[rid]=query
        return local if winner=='local' else history

    def observe(self, rid, event, **data):
        self.traces[rid].observe(event, **data)
        if event=='child_suffix_query' and data['cap']>0:
            q=self.latest[rid]
            assert q['kind']=='root'
            q.update(kind='child', parent=data['path'], rank=data['rank'])


def analyze(queries, traces, rows):
    by_id={r['request_id']:r for r in rows}
    counts=Counter()
    early={rid:{} for rid in by_id}
    for q in queries:
        counts['queries']+=1
        counts['winner_'+q['winner']]+=1
        winner=set(paths(tuple(q['parent']),q[q['winner']]))
        loser=set(paths(tuple(q['parent']),q['history' if q['winner']=='local' else 'local']))
        extra=loser-winner
        counts['discarded_unique_candidate_occurrences']+=len(extra)
        counts['queries_with_extra_candidates']+=bool(extra)
        tokens=by_id[q['request_id']]['tokens']
        if tuple(tokens[:len(q['parent'])])==tuple(q['parent']):
            counts['queries_on_final_prefix']+=1
        for path in extra:
            if tuple(tokens[:len(path)])!=path:
                continue
            counts['discarded_correct_candidate_occurrences']+=1
            key=len(path)-1
            old=early[q['request_id']].get(key)
            if old is None or q['tick']<old['tick']:
                early[q['request_id']][key]=dict(tick=q['tick'],kind=q['kind'],
                    winner=q['winner'],position=key,token=path[-1],
                    query_position=len(q['parent'])-1,completed_history=q['completed_history'])
    examples=[]
    details=[]
    for rid,trace in traces.items():
        trace.finish(by_id[rid]['tokens'],by_id[rid]['stats'])
        local=Counter()
        assert sum(q['request_id']==rid for q in queries)==by_id[rid]['stats']['speculate_calls']
        for e in trace.entries:
            if e['terminal']:
                continue
            local['root_entries']+=1
            local['one_round_wait_nodes']+=e['step']==trace.depth-1
            if e['step']==trace.depth:
                continue
            local['waiting_nodes']+=1
            path=e['path'];pos=len(path)-1;n=trace.nodes[path]
            first_alloc=n['allocations'][0]['tick']
            candidate=early[rid].get(pos)
            if candidate is None:
                continue
            local['waiting_nodes_with_discarded_correct_candidate']+=1
            if candidate['tick']>=first_alloc:
                continue
            local['waiting_nodes_with_earlier_discarded_candidate']+=1
            if n['first_proposal'] is not None and candidate['tick']>=n['first_proposal']:
                local['already_proposed_earlier_by_existing_decoder']+=1
                continue
            local['waiting_nodes_with_new_earlier_candidate']+=1
            local['one_round_wait_with_new_earlier_candidate']+=e['step']==trace.depth-1
            examples.append(dict(request_id=rid,**candidate,actual_first_proposal=n['first_proposal'],
                actual_first_allocation=first_alloc,actual_root_tick=e['tick'],actual_root_step=e['step'],
                lead_ticks=first_alloc-candidate['tick']))
        counts.update(local)
        details.append(dict(request_id=rid,counts=dict(local)))
    return dict(counts=dict(counts),requests=details,examples=examples,
                interpretation='Potential earlier proposals on the original trajectory only. No budget admission, survival, changed outputs, or realized speedup is assumed.')


def main():
    import loop_batch_bench as bench
    import llm_memory_editability.loop_batch_decode as batch_module
    args=bench.parser().parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'DIAGNOSTIC.json').write_text(json.dumps(dict(purpose=__doc__,formal_throughput=False),indent=2))
    original=bench.batch_decode

    def measured(*a,**kw):
        engine=kw.get('executor')
        if engine is None or engine.graphs.warming:
            return original(*a,**kw)
        traces={};cache=a[4];probe=SourceProbe(cache,traces)
        old_speculate=cache.speculate;old_steps=batch_module.budget_tree_decode_steps
        def steps(*sa,**sk):
            rid=sk['req_id'];traces[rid]=DraftTrace(sa[2].depth_total)
            return old_steps(*sa,**dict(sk,observer=lambda event,**data:probe.observe(rid,event,**data)))
        cache.speculate=probe.speculate;batch_module.budget_tree_decode_steps=steps
        try:
            rows,summary=original(*a,**kw)
        finally:
            cache.speculate=old_speculate;batch_module.budget_tree_decode_steps=old_steps
        report=analyze(probe.queries,traces,rows)
        (out/'source-opportunities.json').write_text(json.dumps(report,indent=2))
        with (out/'source-queries.jsonl').open('w') as f:
            for q in probe.queries:f.write(json.dumps(q)+'\n')
        summary.update(diagnostic=True,formal_throughput=False)
        return rows,summary
    bench.batch_decode=measured
    bench.main()


if __name__=='__main__':main()
