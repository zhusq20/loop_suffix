"""Observe actual suffix retrieval hits without changing the production decoder.

Forwards benchmark CLI arguments; only the measured request prefix is shortened.
Child-query events identify query type. Draft quality is scored after generation
against the recorded output, without exposing future tokens to the cache.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import loop_batch_bench as bench
import llm_memory_editability.loop_batch_decode as batch_module
from loop_suffix_cache_diagnose import matching_depth
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-requests', type=int, default=32)
    diagnostic, remaining = parser.parse_known_args()
    if diagnostic.diagnostic_requests < 1:
        parser.error('--diagnostic-requests must be positive')
    sys.argv[1:] = remaining
    out = Path(remaining[remaining.index('--out') + 1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'DIAGNOSTIC.json').write_text(json.dumps(dict(
        purpose=__doc__, instrumented=True, formal_throughput=False,
        measured_request_limit=diagnostic.diagnostic_requests, argv=remaining,
    ), indent=2))
    decode = bench.batch_decode

    def measured(*args, **kwargs):
        engine = kwargs.get('executor')
        if engine is None or engine.graphs.warming:
            return decode(*args, **kwargs)
        args = (args[0][:diagnostic.diagnostic_requests], *args[1:])
        cache = args[4]
        if cache is None:
            raise ValueError('Suffix hit diagnostics require a suffix cache')
        speculate = cache.speculate
        steps = batch_module.budget_tree_decode_steps
        queries, latest = [], {}

        def probe(req_id, context, **options):
            draft = speculate(req_id, context, **options)
            q = dict(request_id=req_id, kind='root',
                     confirmed_length=len(cache.responses[req_id]),
                     token_ids=list(draft.token_ids), parents=list(draft.parents),
                     probs=list(draft.probs), match_len=int(draft.match_len))
            queries.append(q)
            latest[req_id] = q
            return draft

        def observed_steps(*step_args, **step_kwargs):
            req_id = step_kwargs['req_id']
            previous = step_kwargs.get('observer')

            def observe(event, **data):
                if event == 'child_suffix_query' and data['cap'] > 0:
                    q = latest[req_id]
                    assert q['kind'] == 'root'
                    assert len(q['token_ids']) == data['returned_nodes']
                    q.update(kind='child', parent_path=list(data['path']))
                if previous is not None:
                    previous(event, **data)

            return steps(*step_args, **dict(step_kwargs, observer=observe))

        cache.speculate = probe
        batch_module.budget_tree_decode_steps = observed_steps
        try:
            records, summary = decode(*args, **kwargs)
        finally:
            cache.speculate = speculate
            batch_module.budget_tree_decode_steps = steps
        by_id = {r['request_id']: r for r in records}
        totals = {k: Counter() for k in ['all', 'root', 'child']}
        histograms = {k: Counter() for k in totals}
        root_quality = Counter()
        for q in queries:
            count = len(q['token_ids'])
            for kind in ['all', q['kind']]:
                totals[kind].update(queries=1, nonempty=int(count > 0), nodes=count,
                                    nodes_prob_below_point1=sum(p < .1 for p in q['probs']))
                histograms[kind][count] += 1
            if q['kind'] == 'root':
                future = by_id[q['request_id']]['tokens'][q['confirmed_length']:]
                depth = matching_depth(SimpleNamespace(**q), future)
                root_quality.update(queries=1, first_correct=int(depth > 0), matching_tokens=depth)
        assert totals['all']['queries'] == sum(r['stats']['speculate_calls'] for r in records)
        assert totals['child']['queries'] == sum(r['stats']['child_suffix_queries'] for r in records)
        for kind, counts in totals.items():
            n = counts['queries']
            counts['nonempty_rate'] = counts['nonempty'] / n if n else None
            counts['nodes_per_query'] = counts['nodes'] / n if n else None
        label = f'{args[2].name}-b{len(args[1])}'
        (out / f'{label}-suffix-hits.json').write_text(json.dumps(dict(
            diagnostic=True, formal_throughput=False, requests=len(records),
            config=vars(args[2]), totals=totals, returned_node_histograms=histograms,
            root_quality=root_quality,
            root_quality_definition='Matching draft paths against eventual output; not actual retained-node acceptance in fusion',
        ), indent=2))
        (out / f'{label}-suffix-queries.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in queries))
        summary.update(diagnostic=True, suffix_hit_instrumented=True)
        return records, summary

    bench.batch_decode = measured
    bench.main()


if __name__ == '__main__':
    main()
