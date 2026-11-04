"""Replay fixed accepted outputs to compare completion-only and streaming history.

No model generation, tuning or future-token insertion. The future output is used
only to score a draft after retrieval. Round-robin token admission is synthetic;
these acceptance statistics are not end-to-end decoding speed measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache


def matching_depth(draft, future):
    """Longest draft path matching the fixed continuation (excludes bonus)."""
    paths = []
    best = 0
    for token, parent in zip(draft.token_ids, draft.parents, strict=True):
        path = ([] if parent < 0 else paths[parent]) + [token]
        paths.append(path)
        if path == future[:len(path)]:
            best = max(best, len(path))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--q', type=int, default=4)
    a = p.parse_args()
    if a.batch < 1 or a.q < 1:
        p.error('batch and q must be positive')
    prompts = {}
    for line in a.questions.read_text().splitlines():
        r = json.loads(line)
        prompts[r.get('question_id', r.get('request_id'))] = r['prompt_token_ids']
    records = [json.loads(line) for line in a.records.read_text().splitlines()]
    caches = {'completion_only': CompletionSuffixCache.arctic(),
              'streaming_official': SuffixDecodingCache(64, 100000)}
    totals = {name: Counter() for name in caches}
    transitions = Counter()
    active, next_request = {}, 0

    def timed(name, method, *args, **kwargs):
        start = time.perf_counter()
        result = getattr(caches[name], method)(*args, **kwargs)
        totals[name][method + '_s'] += time.perf_counter() - start
        return result

    while active or next_request < len(records):
        for slot in range(a.batch):
            if slot not in active and next_request < len(records):
                r = records[next_request]
                next_request += 1
                rid = r['request_id']
                active[slot] = [r, 0]
                for name in caches:
                    timed(name, 'start_request', rid, np.asarray(prompts[rid], dtype=np.int32))
        for slot, (r, pos) in list(active.items()):
            rid, tokens = r['request_id'], r['tokens']
            accepted = np.asarray(tokens[pos:pos+1], dtype=np.int32)
            pos += 1
            for name in caches:
                timed(name, 'add_active_response', rid, accepted)
            if pos == len(tokens):
                for name in caches:
                    timed(name, 'stop_request', rid)
                del active[slot]
                continue
            active[slot][1] = pos
            context = np.asarray((prompts[rid] + tokens[:pos])[-64:], dtype=np.int32)
            depths = []
            for name in caches:
                draft = timed(name, 'speculate', rid, context, max_spec_tokens=a.q,
                              max_spec_factor=1., max_spec_offset=0.,
                              min_token_prob=.001, use_tree_spec=True)
                depth = matching_depth(draft, tokens[pos:pos+a.q])
                depths.append(depth)
                totals[name].update(queries=1, nodes=len(draft.token_ids), matching_tokens=depth,
                                    nonempty=int(bool(draft.token_ids)), first_correct=int(depth > 0))
            transitions['streaming_better' if depths[1] > depths[0] else
                        'streaming_worse' if depths[1] < depths[0] else 'equal'] += 1
    result = dict(purpose=__doc__, records=str(a.records), requests=len(records),
                  batch=a.batch, q=a.q, totals=totals, paired=transitions)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
