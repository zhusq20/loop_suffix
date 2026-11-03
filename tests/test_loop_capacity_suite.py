"""Capacity checks must execute the real fused kernels at full batch and tree bounds."""
import importlib.util
from pathlib import Path

import pytest
import torch

from llm_memory_editability.loop_budget_decode import BudgetConfig
from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop
from test_loop_batch_decode import TinyHuginn, TinyOuro

spec = importlib.util.spec_from_file_location(
    "capacity_suite", Path(__file__).resolve().parents[1] / "scripts/loop_capacity_suite.py"
)
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


@pytest.mark.parametrize("family", ["ouro", "huginn"])
@pytest.mark.parametrize("arm", ["ar", "wavefront", "loopspec-dual",
                                 "budget-suffix", "budget-raw", "budget-online"])
def test_saturated_capacity_shapes_execute_without_exceeding_buffers(family, arm):
    torch.set_num_threads(1)
    model = (TinyOuro() if family == "ouro" else TinyHuginn()).eval()
    if arm.startswith("budget-"):
        cfg = BudgetConfig(arm, 4, 1, (1, 2), use_model=arm != "budget-suffix")
    elif arm == "loopspec-dual":
        cfg = LoopSpecConfig(arm, 4, 1, 2)
    else:
        cfg = ArmConfig(arm, 4, 1, 1 << 30, False, arm == "wavefront")
    nodes = capacity.physical_nodes(arm, cfg,
        {"draft_node_budget": 64, "physical_nodes": {"wavefront": 1152}})
    slots = [OuroLoop(model, 4, 128, nodes) if family == "ouro" else
             HuginnLoop(model, 4, 128, nodes, 0, 128, extra_depths=(2,)) for _ in range(2)]
    with torch.inference_mode():
        for b in slots:
            b.prefill([1, 2, 3], 1)
        report = capacity.stress_tree_shapes(slots, cfg, 50, nodes)
    assert all(b.kv.length == 50 for b in slots)
    assert report["retained_tree_nodes"] <= nodes
    assert all(torch.isfinite(b.hidden).all() for b in slots)
    if arm == "wavefront":
        assert report["live_rows_per_request"] == 4
        assert report["retained_tree_nodes"] == 20
    if arm.startswith("budget-"):
        assert report["live_rows_per_request"] == 65


def test_guard_boundary_matches_existing_main_allocation_policy():
    # 100 bytes per slot, 1000 free after first; only 850 of those may be used.
    batch = capacity.guarded_batch_limit(100, 1000)
    assert batch == 9
    assert (batch - 1) * 100 <= 1000 * 0.85
    assert batch * 100 > 1000 * 0.85


def test_oversized_late_context_is_rejected_before_device_execution():
    model = TinyOuro().eval()
    b = OuroLoop(model, 4, 128, 65)
    cfg = BudgetConfig("budget-raw", 4, 1, (1, 2))
    with pytest.raises(ValueError, match="positions exceed"):
        capacity.stress_tree_shapes([b], cfg, 100, 65)
