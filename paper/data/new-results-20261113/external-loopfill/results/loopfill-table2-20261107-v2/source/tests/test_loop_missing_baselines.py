"""Reuse is configuration-based, not a requirement to match GPU or new counters."""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from loop_missing_baselines import canonical_config, compatible


def test_ar_unused_draft_cap_does_not_force_rerun():
    cfg = dict(name='ar', use_fill=False, use_suffix=False, max_draft=64)
    assert canonical_config(cfg) == canonical_config(dict(cfg, max_draft=4))


def test_old_global_suffix_cap_matches_explicit_rank_caps():
    old = dict(name='budget-raw', child_suffix_policy='all', suffix_room=4, top_m=2)
    new = dict(old, c_initial=1., child_suffix_caps=[4, 4])
    assert canonical_config(old) == canonical_config(new)
    assert canonical_config(old) != canonical_config(dict(new, child_suffix_caps=[4, 0]))
    assert 'c_initial' not in old


def test_reuse_ignores_gpu_but_rejects_different_parameters_or_workload():
    cfg = dict(name='budget-online', c_initial=.7, eta=.1, draft_node_budget=16,
               child_suffix_policy='all', suffix_room=4, top_m=2)
    args = dict(model='huginn-3.5b', schedule='continuous', attention_backend='packed',
        prefill_chunk_size=128, cuda_graph=True, max_cuda_graphs=24,
        max_new_tokens=1024, max_seq=2816, buffer_policy='per-arm', dtype='bfloat16',
        warmup_tokens=32, seed=0, order_seeds='0', repeat_id=0, limit=None,
        check=True, profile=False, trace_controller=False)
    meta = dict(args=args, questions_sha256='data', configs={'budget-online':cfg}, gpu_uuid='any-gpu')
    assert compatible(meta, cfg, 'data')
    changed = copy.deepcopy(meta)
    changed['gpu_uuid'] = 'another-gpu'
    assert compatible(changed, cfg, 'data')
    assert not compatible(meta, dict(cfg, c_initial=1.), 'data')
    assert not compatible(meta, dict(cfg, draft_node_budget=64), 'data')
    assert not compatible(meta, cfg, 'other-data')
    changed['args']['schedule'] = 'fixed'
    assert not compatible(changed, cfg, 'data')
