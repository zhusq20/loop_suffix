from types import SimpleNamespace

import numpy as np

from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request, summarize

PROMPT = [100, 101, 102]
OUT = list(range(1, 10))  # output[0] comes from prefill; 8 counted tokens


class ChainCache:
    """Drafts a fixed chain chosen by a callback on the context."""

    def __init__(self, chain_for):
        self.chain_for = chain_for
        self.added: list[int] = []

    def start_request(self, req_id, prompt_token_ids):
        self.added = []

    def add_active_response(self, req_id, token_ids):
        self.added.extend(int(t) for t in token_ids)

    def stop_request(self, req_id):
        pass

    def speculate(self, req_id, context, max_spec_tokens=None, **kwargs):
        chain = self.chain_for([int(t) for t in context])[:max_spec_tokens]
        return SimpleNamespace(token_ids=chain, parents=list(range(-1, len(chain) - 1)))


def perfect_readouts(depth: int) -> dict[int, np.ndarray]:
    return {depth: np.array([[t, 0] for t in OUT])}


def true_continuation(context: list[int]) -> list[int]:
    n = len(context) - len(PROMPT)
    return OUT[n:]


def arm(**kw) -> ArmConfig:
    base = dict(
        name="x", depth_total=8, draft_depth=2, max_draft=3, use_suffix=False, use_fill=False
    )
    return ArmConfig(**(base | kw))


def test_ar_costs_full_depth_per_token():
    s = replay_request(0, PROMPT, OUT, perfect_readouts(2), arm(max_draft=0), None)
    assert s["tokens"] == 8 and s["steps"] == 8 and s["calls"] == 64


def test_shallow_chain_waits_d_calls_per_draft():
    s = replay_request(0, PROMPT, OUT, perfect_readouts(2), arm(use_fill=True), None)
    # each step: fills start at 2, 4, 6; the last one is verified by call 6 + 8
    assert s["steps"] == 2 and s["tokens"] == 8 and s["calls"] == 2 * 14
    assert s["from_fill"] == 6 and s["from_suffix"] == 0


def test_correct_suffix_draft_needs_only_full_depth():
    cache = ChainCache(true_continuation)
    s = replay_request(0, PROMPT, OUT, perfect_readouts(2), arm(use_suffix=True), cache)
    assert s["steps"] == 2 and s["calls"] == 2 * 8 and s["from_suffix"] == 6
    assert cache.added == OUT


def test_fill_replaces_wrong_suffix_token():
    def wrong_third(context):
        cont = true_continuation(context)
        return cont[:2] + [999] if len(cont) >= 3 else cont

    cache = ChainCache(wrong_third)
    s = replay_request(
        0, PROMPT, OUT, perfect_readouts(2), arm(use_suffix=True, use_fill=True), cache
    )
    # first step: suffix gives two correct tokens at call 0, the second one's shallow
    # readout fills the third at call 2, so the step ends at 2 + 8 with 4 tokens
    first = replay_request(
        0,
        PROMPT,
        OUT[:5],
        perfect_readouts(2),
        arm(use_suffix=True, use_fill=True),
        ChainCache(wrong_third),
    )
    assert first["steps"] == 1 and first["calls"] == 10 and first["tokens"] == 4
    assert first["from_suffix"] == 2 and first["from_fill"] == 1
    assert s["tokens"] == 8


def test_pruning_removes_wrong_branch_rows():
    def wrong_first(context):
        return [999, 998, 997]

    kw = dict(use_suffix=True, use_fill=True)
    loose = replay_request(0, PROMPT, OUT, perfect_readouts(2), arm(**kw), ChainCache(wrong_first))
    tight = replay_request(
        0, PROMPT, OUT, perfect_readouts(2), arm(prune_top=1, **kw), ChainCache(wrong_first)
    )
    assert loose["calls"] == tight["calls"] and loose["tokens"] == tight["tokens"]
    assert tight["rows"] < loose["rows"]


def test_summary_speedup_matches_definition():
    s = replay_request(0, PROMPT, OUT, perfect_readouts(2), arm(use_fill=True), None)
    out = summarize([s], 8)
    assert out["speedup_vs_ar"] == 8 * 8 / 28
