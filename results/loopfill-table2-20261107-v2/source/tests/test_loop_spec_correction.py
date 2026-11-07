"""Correction variants must preserve greedy output, ancestry and slot isolation."""
from dataclasses import replace

import pytest
import torch

from llm_memory_editability.loop_batch_decode import batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_budget_decode import Candidate, select_candidates, budget_tree_decode
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor
from scripts.loop_batch_bench import arm_config
from test_loop_budget_decode import BudgetFakeLoop, BoundedSuffix
from test_loop_tree_decode import greedy
from test_loop_optimized_decode import setup, requests


VARIANTS = [
    dict(model_top_k=2), dict(model_top_k=1),
    dict(model_top_k=2, child_suffix_policy="best1"),
    dict(model_top_k=2, child_suffix_policy="none"),
    dict(model_top_k=2, positive_only=True),
    dict(model_top_k=2, draft_node_budget=2),
    dict(arm="budget-shallow"), dict(),
    dict(model_top_k=2, child_suffix_caps=(4, 4)),
    dict(model_top_k=2, child_suffix_caps=(4, 2)),
    dict(model_top_k=2, child_suffix_caps=(4, 0)),
]


def configuration(variant):
    options = dict(draft_node_budget=8, **{k: v for k, v in variant.items() if k != 'draft_node_budget'})
    options['draft_node_budget'] = variant.get('draft_node_budget', 8)
    arm = options.pop('arm', 'budget-raw')
    return arm_config(arm, 4, 1, 2, 64, **options)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_positive_admission_drops_zero_subtree_but_preserves_root():
    root = (1,)
    pool = {root: Candidate(root, 1, 'model', 'root')}
    for path, score in [((1, 2), 0), ((1, 2, 3), 1), ((1, 4), .1), ((1, 4, 5), .1)]:
        pool[path] = Candidate(path, score, 'model', 'fill')
    assert set(select_candidates(pool, root, 64, 1, positive_only=True)) == {root, (1, 4), (1, 4, 5)}
    assert len(select_candidates(pool, root, 64, 1)) == 5
    assert list(select_candidates(pool, root, 0, 1, positive_only=True)) == [root]


@pytest.mark.parametrize('variant', VARIANTS)
def test_variants_preserve_greedy_and_budget_and_query_contract(variant):
    cfg = configuration(variant)
    events = []
    prompt = [1, 2, 3, 1, 2]
    seq, stats = budget_tree_decode(BudgetFakeLoop(4, 1), prompt, cfg, 40, set(),
                                    BoundedSuffix() if cfg.use_suffix else None,
                                    observer=lambda event, **data: events.append((event, data)))
    assert seq == greedy(prompt, 40, set())
    assert stats['wasted_rows'] == sum(stats[f'wasted_{r}_rows'] for r in ['budget', 'rejected', 'end'])
    assert stats['peak_draft_nodes'] <= cfg.draft_node_budget
    for kind, event in events:
        if kind != 'selection':
            continue
        paths = set(event['paths'])
        assert all(p == event['root'] or p[:-1] in paths for p in paths)
        if cfg.positive_only:
            assert all(v[-1] > 0 for p, v in event['scores'].items() if p != event['root'])
    if not cfg.use_suffix:
        assert stats['speculate_calls'] == stats['child_suffix_queries'] == 0
    if cfg.child_suffix_policy == 'none':
        assert stats['child_suffix_queries'] == 0
    if cfg.child_suffix_policy == 'best1':
        assert stats['child_suffix_queries'] <= len([e for e, _ in events if e == 'selection'])


@pytest.mark.parametrize('family', ['ouro', 'huginn'])
@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_variant_packed_recycling_graphs_and_progress(family, variant, device):
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    make = setup(family, device, dtype)
    cfg = configuration(variant)
    def cache():
        return CompletionSuffixCache.arctic() if cfg.use_suffix else None
    refs, _ = batch_decode(requests(), [make(), make()], cfg, cache=cache(), schedule='continuous')
    pool = SharedKVPool(2)
    slots = [make(pool), make(pool)]
    engine = PackedLoopExecutor(slots, cuda_graph=device == 'cuda', max_graphs=32)
    batch_decode(requests(), slots, cfg, cache=cache(), schedule='continuous',
                 executor=engine, prefill_chunk_size=2)
    engine.graphs.freeze()
    progress = []
    rows, summary = batch_decode(requests(), slots, cfg, cache=cache(), schedule='continuous',
                                executor=engine, prefill_chunk_size=2,
                                progress_callback=lambda r, t: progress.append((len(r), t)))
    assert [r['tokens'] for r in rows] == [r['tokens'] for r in refs]
    assert [n for n, _ in progress] == list(range(1, len(requests()) + 1))
    assert all(a[1] <= b[1] for a, b in zip(progress, progress[1:]))
    assert all(r['stats']['cap_skips'] == 0 for r in rows)
    if device == 'cuda':
        assert summary['cuda_graph_replays'] > 0
