"""Diagnostic raw-only union: keep winner edges, add unique loser paths.

The advertised cap is the combined cap; each existing source gets half.
Model node budget is unchanged. This intentionally does not define a merged
match length or a new confidence estimator.
"""
from types import SimpleNamespace

from llm_memory_editability.loop_prob_tree import _edge_probs


def merge_drafts(local, history):
    winner, loser = (local, history) if local.score >= history.score else (history, local)
    tokens, parents, probs, indices = [], [], [], {}
    for draft in (winner, loser):
        source_paths = []
        for token, parent, edge in zip(draft.token_ids, draft.parents, _edge_probs(draft), strict=True):
            prefix = () if parent < 0 else source_paths[parent]
            path = prefix + (token,)
            source_paths.append(path)
            if path in indices:
                continue
            merged_parent = -1 if not prefix else indices[prefix]
            indices[path] = len(tokens)
            tokens.append(token)
            parents.append(merged_parent)
            probs.append(edge * (1. if merged_parent < 0 else probs[merged_parent]))
    return SimpleNamespace(token_ids=tokens, parents=parents, probs=probs,
                           match_len=None, score=winner.score)


def union_speculate(self, req_id, context, **options):
    cap = options['max_spec_tokens']
    if cap < 2 or cap % 2:
        raise ValueError('union probe requires a positive even combined cap')
    options = dict(options, max_spec_tokens=cap // 2)
    local = self.local.speculate(req_id, context, **options)
    history = self.history.speculate(req_id, context, **options)
    merged = merge_drafts(local, history)
    if len(merged.token_ids) > cap:
        raise ValueError('source exceeded its half of the combined cap')
    return merged


def main():
    import loop_batch_bench as bench
    import loop_draft_trace
    from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
    args = bench.parser().parse_args()
    if args.arms != 'budget-raw':
        raise ValueError('union probe only supports budget-raw; sources have distinct match lengths')
    original = CompletionSuffixCache.speculate
    CompletionSuffixCache.speculate = union_speculate
    try:
        loop_draft_trace.main()
    finally:
        CompletionSuffixCache.speculate = original


if __name__ == '__main__':
    main()
