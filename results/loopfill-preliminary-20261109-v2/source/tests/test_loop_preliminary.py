from types import SimpleNamespace

import numpy as np
import pytest

from llm_memory_editability.loop_preliminary import (
    ProbeConfig, attach, bootstrap_summary, correct_length, probe_position,
    replay_request, request_totals,
)


def draft(tokens, parents=None):
    return SimpleNamespace(token_ids=tokens, parents=(list(range(-1, len(tokens) - 1))
                           if parents is None else parents), match_len=2)


class AuditedCache:
    def __init__(self, query):
        self.query = query
        self.calls = []
        self.history = []
        self.active = []

    def start_request(self, req_id, prompt):
        self.active = list(prompt)

    def add_active_response(self, req_id, tokens):
        self.active.extend(tokens)

    def stop_request(self, req_id):
        self.history.append(self.active)

    def speculate(self, req_id, context, **kwargs):
        self.calls.append((list(context), list(self.active), kwargs))
        return self.query(list(context))


def test_shallow_bridge_and_requery_only_once():
    cache = AuditedCache(lambda context: draft([2, 3, 4]) if context[-1] == 1 else draft([]))
    r = probe_position(cache, 0, [10], 1, [1, 2, 3, 4, 5])
    assert r['lengths'] == [0, 1, 4]
    assert r['nodes'] == [0, 1, 4]
    assert len(cache.calls) == 2
    assert cache.calls[1][0] == [10, 1]


def test_no_future_visibility_and_prefill_excluded():
    cache = AuditedCache(lambda context: draft([]))
    output = [1, 2, 3]
    predictions = np.array([output] * 4)
    rows = list(replay_request(cache, 7, [10], output, predictions))
    assert [r['output_index'] for r in rows] == [1, 2]
    assert [c[1] for c in cache.calls] == [[10, 1], [10, 1], [10, 1, 2], [10, 1, 2]]
    assert cache.history == [[10, 1, 2, 3]]
    assert all(r['lengths'] == [0, 1, 1] for r in rows)


def test_wrong_shallow_cannot_change_common_suffix_candidates():
    cache = AuditedCache(lambda ctx: draft([8, 7, 6, 5]) if ctx[-1] == 9 else draft([1, 2]))
    r = probe_position(cache, 0, [10], 9, [1, 2, 3])
    assert r['lengths'] == [2, 2, 2]
    assert r['nodes'] == [2, 3, 7]


def test_path_dedup_preserves_branches_and_existing_nodes():
    paths = {(1,), (1, 2)}
    attach(paths, draft([1, 2, 9, 2], [-1, 0, -1, 2]), (), 4)
    assert paths == {(1,), (1, 2), (9,), (9, 2)}
    attach(paths, draft([2, 3]), (1,), 4)
    assert len(paths) == 4
    assert correct_length(paths, [1, 2, 3]) == 2
    assert correct_length(paths, [1]) == 1  # no bonus / beyond-EOS scoring


def test_requery_respects_remaining_room_without_eviction():
    cache = AuditedCache(lambda ctx: draft([2]) if ctx[-1] == 1 else draft([9, 8, 7, 6]))
    r = probe_position(cache, 0, [10], 1, [1, 2], ProbeConfig(node_budget=6))
    assert r['nodes'] == [4, 5, 6]
    assert cache.calls[-1][2]['max_spec_tokens'] == 1
    assert r['lengths'] == [0, 1, 2]


def test_existing_shallow_path_is_requeried_and_deduplicated():
    cache = AuditedCache(lambda ctx: draft([2, 3]) if ctx[-1] == 1 else draft([1, 2]))
    r = probe_position(cache, 0, [10], 1, [1, 2, 3])
    assert r['nodes'] == [2, 2, 3]
    assert r['lengths'] == [2, 2, 3]


def test_invalid_alignment_rejected_and_zero_decode_kept():
    cache = AuditedCache(lambda ctx: draft([]))
    with pytest.raises(ValueError, match='full-depth'):
        list(replay_request(cache, 0, [10], [1], [[1], [1], [1], [2]]))
    rows = list(replay_request(cache, 0, [10], [1], [[1]] * 4))
    assert rows == []
    assert request_totals(rows)['positions'] == 0
    with pytest.raises(ValueError, match='parents'):
        attach(set(), draft([1], [0]), (), 16)


def test_agreement_coverage_gains_and_paired_bootstrap():
    cache = AuditedCache(lambda ctx: draft([]))
    predictions = np.array([[1, 9, 3], [1, 2, 9], [1, 2, 3], [1, 2, 3]])
    rows = list(replay_request(cache, 0, [10], [1, 2, 3], predictions))
    totals = request_totals(rows)
    assert totals['agreement'] == [1, 1, 2]
    assert totals['coverage'] == [1, 2, 2]
    assert totals['gains'] == [1, 1, 0]
    result = bootstrap_summary([totals] * 3, repeats=50)
    assert result['metrics']['agreement']['mean'] == [0.5, 0.5, 1]
    assert result['metrics']['length_deltas']['ci95'][1] == [0, 0]
    assert result['metrics']['suffix_zero_fraction']['mean'] == [1]


def test_bootstrap_pools_positions_instead_of_averaging_request_rates():
    cache = AuditedCache(lambda ctx: draft([]))
    a = request_totals(list(replay_request(cache, 0, [10], [1, 2], [[1, 2]] * 4)))
    b = request_totals(list(replay_request(cache, 1, [10], [1, 2, 3, 4],
                          [[9, 9, 9, 9]] * 3 + [[1, 2, 3, 4]])))
    assert bootstrap_summary([a, b], repeats=50)['metrics']['agreement']['mean'] == [.25] * 3


def test_online_readouts_match_separate_ar_and_stop_at_eos():
    import importlib.util
    from pathlib import Path
    import torch
    spec = importlib.util.spec_from_file_location('readouts', Path(__file__).resolve().parents[1]
                                                  / 'scripts/loop_suffix_spec_readouts.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeInner:
        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            pos = past_key_values if isinstance(past_key_values, int) else 0
            hs = []
            for d in range(4):
                h = torch.zeros(1, input_ids.shape[1], 10)
                h[0, -1, pos + (1 if d == 3 else 2)] = 1
                hs.append(h)
            return SimpleNamespace(past_key_values=pos + 1), hs, None
    runner = mod.OuroRunner(SimpleNamespace(model=FakeInner(), lm_head=lambda x: x,
                                           config=SimpleNamespace(num_hidden_layers=2)), 4)
    cache = runner._new_cache()
    assert len(cache.layers) == 8
    for i in range(8):
        key = torch.full((1, 1, 2, 1), float(i))
        cache.update(key, key + 1, i)
    assert all(cache.get_seq_length(i) == 2 for i in range(8))
    assert cache.layers[0].keys[0, 0, 0, 0] == 0
    assert cache.layers[7].keys[0, 0, 0, 0] == 7
    output, reads = runner.generate_with_readouts(torch.tensor([[8, 9]]), 8, {3})
    assert output == runner.generate(torch.tensor([[8, 9]]), 8, {3}) == [1, 2, 3]
    assert reads.tolist() == [[2, 3, 4]] * 3 + [[1, 2, 3]]
