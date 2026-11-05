"""Instrument proposal/admission/recycling/readiness; diagnostic times are not benchmarks."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

from llm_memory_editability.loop_budget_decode import suffix_score


class DraftTrace:
    def __init__(self, depth):
        self.depth = depth
        self.nodes = {}
        self.counts = Counter()
        self.entries = []

    def node(self, path):
        if path not in self.nodes:
            self.nodes[path] = dict(first_proposal=None, first_source=None, allocations=[], active=None)
        return self.nodes[path]

    def observe(self, event, **d):
        c = self.counts
        if event == 'proposal':
            n = self.node(d['path'])
            if n['first_proposal'] is None:
                n.update(first_proposal=d['tick'], first_source=d['source'])
            c['proposals'] += 1
        elif event == 'allocate':
            n = self.node(d['path'])
            assert n['active'] is None
            a = dict(tick=d['tick'], first_advance=None, spent=0, release_tick=None, reason=None)
            n['allocations'].append(a)
            n['active'] = a
            c['allocations'] += 1
        elif event == 'advance':
            c['calls'] += 1
            for path, step in d['steps'].items():
                a = self.node(path)['active']
                assert a is not None and step == a['spent'] + 1
                if a['first_advance'] is None:
                    a['first_advance'] = d['tick']
                a['spent'] = step
                c['rows'] += 1
        elif event == 'release':
            n = self.node(d['path'])
            a = n['active']
            assert a is not None and a['spent'] == d['spent']
            a.update(release_tick=d['tick'], reason=d['reason'])
            n['active'] = None
            c['release_' + d['reason'] + '_rows'] += d['spent']
        elif event == 'rescore':
            c['model_rescores'] += 1
            c['suffix_overwritten'] += d['previous_source'] == 'suffix'
            c['absent_zero_rescores'] += d['absent'] and d['score'] == 0
            c['absent_positive_rescores'] += d['absent'] and d['score'] > 0
        elif event == 'verify':
            path = d['parent'] + (d['gold'],)
            self.entries.append(dict(path=path, tick=d['tick'], step=d['next_step'],
                covered=d['covered'], terminal=d['terminal'], origin=d['next_origin'],
                score_source=d['next_score_source']))
        elif event == 'selection_candidates':
            c['selections'] += 1
            root, candidates = d['root'], d['candidates']
            selected = set(d['selected']) - {root}
            c['selected_entries'] += len(selected)
            c['selected_suffix_entries'] += sum(candidates[p][0] == 'suffix' for p in selected)
            c['selected_zero_entries'] += sum(candidates[p][3] == 0 for p in selected)
            c['candidate_entries'] += len(candidates)-1
            c['candidate_suffix_entries'] += sum(v[0] == 'suffix' for p, v in candidates.items() if p != root)
            c['contested_selections'] += len(candidates)-1 > d['budget']
            paths = sorted((p for p in candidates if p != root), key=lambda p: (len(p), p))
            for coefficient in (.25, 1., 2.):
                scores = {root: 1.}
                for p in paths:
                    source, q, length, _ = candidates[p]
                    edge = suffix_score(q, coefficient, length if d['length_conditioned'] else None) if source == 'suffix' else q
                    scores[p] = scores[p[:-1]] * edge
                ranked = sorted(paths, key=lambda p: (-scores[p], len(p), p))
                alternative = set(p for p in ranked if not d['positive_only'] or scores[p] > 0)
                alternative = set([p for p in ranked if p in alternative][:d['budget']])
                tag = str(coefficient)
                c['changed_c' + tag] += alternative != selected
                c['replaced_entries_c' + tag] += len(selected - alternative)

    def finish(self, tokens, stats):
        c = self.counts
        assert c['calls'] == stats['calls'] and c['rows'] == stats['rows']
        assert c['allocations'] == stats['nodes']
        assert c['selections'] == stats['selection_events']
        assert len(self.entries) == stats['tokens']
        assert sum(e['covered'] for e in self.entries) == stats['covered_tokens']
        for reason in ('budget', 'rejected', 'end'):
            assert c['release_' + reason + '_rows'] == stats['wasted_' + reason + '_rows']
        continuing = [e for e in self.entries if not e['terminal']]
        c['root_entries'] = len(continuing)
        c['initial_root_wait'] = self.depth if self.entries else 0
        for e in continuing:
            c[f"root_step_{e['step']}"] += 1
            c['remaining_root_wait'] += self.depth - e['step']
            c['covered_root_entries'] += e['covered']
        assert c['initial_root_wait'] + c['remaining_root_wait'] == stats['calls']
        gold_nodes = []
        for path, n in self.nodes.items():
            if tuple(tokens[:len(path)]) != path:
                continue
            allocs = n['allocations']
            lost = sum(a['spent'] for a in allocs if a['reason'] == 'budget')
            c['correct_path_budget_waste'] += lost
            c['correct_path_rebuilt_nodes'] += len(allocs) > 1
            c['correct_path_allocated_nodes'] += bool(allocs)
            if allocs and n['first_proposal'] is not None:
                c['correct_path_proposed_nodes'] += 1
                c['proposal_to_allocation_ticks'] += allocs[0]['tick'] - n['first_proposal']
                c['delayed_correct_admissions'] += allocs[0]['tick'] > n['first_proposal']
            gold_nodes.append(dict(position=len(path)-1, token=path[-1],
                first_proposal=n['first_proposal'], first_source=n['first_source'], allocations=allocs))
        return dict(c), gold_nodes, [dict(e, position=len(e['path'])-1, path=None) for e in self.entries]


def main():
    import loop_batch_bench as bench
    import llm_memory_editability.loop_batch_decode as batch_module
    args = bench.parser().parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'DIAGNOSTIC.json').write_text(json.dumps(dict(
        purpose=__doc__, instrumented=True, formal_throughput=False,
        argv=sys.argv[1:], limitation='Short stratified cold-start run; not formal throughput or same-prefix causal comparison.'), indent=2))
    inputs = {r['question_id']:r for r in map(json.loads, Path(args.questions).read_text().splitlines())}
    original_decode = bench.batch_decode

    def measured(*a, **kw):
        engine = kw.get('executor')
        if engine is None or engine.graphs.warming:
            return original_decode(*a, **kw)
        traces = {}
        original_steps = batch_module.budget_tree_decode_steps
        def steps(*sa, **sk):
            trace = DraftTrace(sa[2].depth_total)
            traces[sk['req_id']] = trace
            return original_steps(*sa, **dict(sk, observer=trace.observe))
        batch_module.budget_tree_decode_steps = steps
        try:
            rows, summary = original_decode(*a, **kw)
        finally:
            batch_module.budget_tree_decode_steps = original_steps
        totals, details = Counter(), []
        label = f'{a[2].name}-b{len(a[1])}'
        with (out/f'{label}-correct-node-lifecycle.jsonl').open('w') as nodes, (out/f'{label}-root-entries.jsonl').open('w') as entries:
            for r in rows:
                rid = r['request_id']
                counts, gold_nodes, root_entries = traces[rid].finish(r['tokens'], r['stats'])
                totals.update(counts)
                details.append(dict(request_id=rid, category=inputs[rid]['category'], counters=counts))
                for n in gold_nodes:
                    nodes.write(json.dumps(dict(request_id=rid, **n))+'\n')
                for e in root_entries:
                    entries.write(json.dumps(dict(request_id=rid, **e))+'\n')
        (out/f'{label}-lifecycle-summary.json').write_text(json.dumps(dict(
            diagnostic=True, formal_throughput=False, totals=dict(totals), requests=details,
            reconstruction='calls = T for first root + sum(T - step_on_root_entry) excluding terminal token',
            sensitivity='Same candidate pool rescored at different c; not counterfactual decoding speed.'), indent=2))
        summary.update(diagnostic=True, formal_throughput=False)
        return rows, summary

    bench.batch_decode = measured
    bench.main()


if __name__ == '__main__':
    main()
