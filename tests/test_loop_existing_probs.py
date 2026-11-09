"""Existing-child scores must use real probabilities without expanding top-k."""
from dataclasses import replace

import pytest
import torch

from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode
from llm_memory_editability.loop_budget_decode import budget_tree_decode
from llm_memory_editability.loop_optimized_decode import PackedLoopExecutor, SharedKVPool
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_tree_decode import _top_probs, tree_decode
from scripts.loop_batch_bench import arm_config, parser
from test_loop_budget_decode import BoundedSuffix, BudgetFakeLoop, config
from test_loop_optimized_decode import setup
from test_loop_tree_decode import greedy, greedy_next, shallow_next


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_extra_scores_share_full_softmax_and_preserve_topk_ties(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    logits = torch.tensor([[4., 4., 2., -1., 0.], [-1., 0., 3., 2., 1.]], device=device)
    old = _top_probs(logits, 2)
    extra = [[4, 0, 4, 2], [0, 3]]
    got = _top_probs(logits, 2, extra)
    probs = logits.softmax(-1).cpu()
    for row, (ids, ps) in enumerate(got):
        assert ids[:2] == old[row][0] and ps[:2] == old[row][1]
        assert set(ids) == set(old[row][0]) | set(extra[row])
        assert len(ids) == len(set(ids))
        assert ps == pytest.approx(probs[row, ids].tolist())
    assert _top_probs(logits[:0], 2, []) == []
    with pytest.raises(ValueError, match="one list per"):
        _top_probs(logits, 2, [[]])


class ExactFakeLoop(BudgetFakeLoop):
    def readout_top(self, tree, groups, full, m, extra_token_ids=None, return_log_probs=False):
        weights = torch.tensor([.55, .2, .1, .06, .04, .03, .01, .004, .003, .002, .001])
        logits, sizes = [], []
        for depth, nodes in groups:
            sizes.append(len(nodes))
            for node in nodes:
                first = shallow_next(self.context(tree, node), depth)
                logits.append(torch.roll(weights, first).log())
        scores = _top_probs(torch.stack(logits), m, extra_token_ids, return_log_probs) if logits else []
        reads, at = [], 0
        for size in sizes:
            reads.append(scores[at:at + size])
            at += size
        return reads, [greedy_next(self.context(tree, n)) for n in full]


@pytest.mark.parametrize("positive_only", [False, True])
def test_outside_topk_is_rescored_without_new_proposals_and_preserves_greedy(positive_only):
    prompt = [1, 2, 3, 4, 5, 6, 1, 2]
    cfg = config(8, scoring="raw", top_m=2, suffix_room=4, positive_only=positive_only)
    observations = {}
    for exact in (False, True):
        events = []
        out, stats = budget_tree_decode(
            ExactFakeLoop(4, 1), prompt, replace(cfg, existing_child_probs=exact), 60, set(),
            BoundedSuffix(), 0, observer=lambda event, **data: events.append((event, data)))
        assert out == greedy(prompt, 60, set())
        assert stats["peak_draft_nodes"] <= 8 and stats["max_rows"] <= 9
        absent = [d for e, d in events if e == "rescore" and d["absent"]]
        assert absent
        assert all((d["score"] > 0) == exact for d in absent)
        assert stats["absent_positive_rescores"] == (len(absent) if exact else 0)
        # Every neural proposal is still among that parent's two largest values.
        for event, data in events:
            if event == "proposal" and data["source"] in ("fill", "sibling", "deep"):
                assert data["score"] == pytest.approx(.55) or data["score"] == pytest.approx(.2)
        if positive_only:
            assert stats["selected_zero_nodes_sum"] == 0
        observations[exact] = stats
    assert observations[True]["existing_probability_values"] > 0


@pytest.mark.parametrize("family", ["ouro", "huginn"])
@pytest.mark.parametrize("positive_only", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("selection", ["top_n", "sample"])
@torch.no_grad()
def test_single_fused_and_packed_exact_scores_preserve_ar_and_recycled_slots(family, positive_only, device, selection):
    if selection == "sample" and positive_only:
        pytest.skip("sampling and positive-only are separate selection policies")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    make = setup(family, device, torch.bfloat16 if device == "cuda" else torch.float32, max_nodes=5)
    cfg = config(4, scoring="raw", top_m=2, suffix_room=4,
                 existing_child_probs=True, positive_only=positive_only,
                 selection_policy=selection, draft_model_id=family)
    reqs = [BatchRequest(i, p, n) for i, (p, n) in enumerate([
        ([1, 2, 3, 1, 2], 20), ([6, 7, 8], 13), ([4, 2, 1, 4], 17), ([7, 2], 1)])]
    refs, serial = [], []
    for r in reqs:
        ref, _ = tree_decode(make(), r.prompt, ArmConfig("ar", 4, 1, 0, False, False),
                             r.max_new_tokens, r.stop)
        refs.append(ref)
        serial.append(budget_tree_decode(make(), r.prompt, cfg, r.max_new_tokens,
                                        r.stop, BoundedSuffix(), r.request_id))
    fused, _ = batch_decode(reqs, [make(), make()], cfg, cache=BoundedSuffix(), schedule="continuous")
    pool = SharedKVPool(2)
    slots = [make(pool), make(pool)]
    packed, _ = batch_decode(reqs, slots, cfg, cache=BoundedSuffix(), schedule="continuous",
                              executor=PackedLoopExecutor(slots), prefill_chunk_size=2)
    assert [r[0] for r in serial] == refs
    assert [r["tokens"] for r in fused] == refs == [r["tokens"] for r in packed]
    for rows in (fused, packed):
        assert sum(r["stats"]["absent_positive_rescores"] for r in rows) > 0
        assert any(r["stats"]["nodes"] > 5 for r in rows)
        assert all(r["stats"]["peak_draft_nodes"] <= 4 for r in rows)


def test_existing_probabilities_cli_is_explicit_and_budget_only():
    args = parser().parse_args(["--model", "ouro-2.6b", "--model-path", "unused",
        "--questions", "unused", "--out", "unused", "--existing-child-probs"])
    assert args.existing_child_probs
    assert not arm_config("budget-raw", 4, 1, 2, 4).existing_child_probs
    assert arm_config("budget-raw", 4, 1, 2, 4, existing_child_probs=True).existing_child_probs
    for arm in ("ar", "loopspec-dual", "budget-suffix"):
        with pytest.raises(ValueError, match="require budget arms"):
            arm_config(arm, 4, 1, 2, 4, existing_child_probs=True)
