"""Dual-proposal schedule parity with the pinned authors' pure-Python scheduler."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode, FusedLoopExecutor
from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig, loopspec_decode
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop, tree_decode
from test_loop_batch_decode import TinyHuginn, TinyOuro, FakeExecutor
from test_loop_tree_decode import FakeLoop, greedy, greedy_next, shallow_next


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class DualFake(FakeLoop):
    def context(self, tree, node):
        path = tree.path_tokens(node)
        while node.parent >= 0:
            node = tree.nodes[node.parent]
        return self.seq[:tree.base_pos] + [node.token] + path

    def readout_top(self, tree, groups, full, m):
        assert m == 0
        return ([[shallow_next(self.context(tree, n), d // self.d) for n in group]
                 for d, group in groups],
                [greedy_next(self.context(tree, n)) for n in full])


@pytest.mark.parametrize("T,d1,d2", [(4, 1, 2), (32, 4, 8)])
@pytest.mark.parametrize("length,stop", [(1, set()), (120, set()), (120, {5})])
def test_dual_correctness_recycling_and_batch(T, d1, d2, length, stop):
    cfg = LoopSpecConfig("loopspec-dual", T, d1, d2)
    prompts = [[1, 2], [3, 1, 5], [8, 7]]
    serial = [loopspec_decode(DualFake(T, d1), p, cfg, length, stop, max_nodes=128)
              for p in prompts]
    rows, summary = batch_decode(
        [BatchRequest(i, p, length, stop) for i, p in enumerate(prompts)],
        [DualFake(T, d1), DualFake(T, d1)], cfg, max_nodes=128, executor=FakeExecutor())
    for p, row, (tokens, stats) in zip(prompts, rows, serial, strict=True):
        assert tokens == row["tokens"] == greedy(p, length, stop)
        assert row["stats"]["calls"] == stats["calls"]
        assert row["stats"]["rows"] == stats["rows"]
    assert summary["core_rows"] == sum(s["rows"] for _, s in serial)
    if length > 1 and not stop:
        assert sum(s["accepted_second"] for _, s in serial) > 0
        assert sum(s["second_agreements"] for _, s in serial) > 0


@pytest.mark.parametrize("family", ["ouro", "huginn"])
@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), ("cuda", torch.bfloat16)])
def test_real_backbones_dual_batch_and_serial_ar(family, device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(41)
    model = (TinyOuro() if family == "ouro" else TinyHuginn()).to(device, dtype).eval()
    serial_logits, batch_logits, ar_logits = {}, {}, {}
    def capture(jobs, target, invoke):
        chunks = []
        hook = model.lm_head.register_forward_hook(lambda m, args, output: chunks.append(output.detach().float().cpu()))
        result = invoke()
        hook.remove()
        if not chunks:
            return result
        logits = chunks[-1]
        offset = 0
        for b, tree, groups, full in jobs:
            nodes = [n for _, g in groups for n in g] + full
            bases = ([b.coda_extra.get(d, b.coda_shallow) for d, g in groups for _ in g]
                     + [b.coda_full] * len(full)) if family == "huginn" else [0] * len(nodes)
            ordered = sorted(zip(nodes, bases), key=lambda item: item[1])
            for index, (n, _) in enumerate(ordered):
                if n in full:
                    path = tree.path_tokens(n)
                    root = n
                    while root.parent >= 0:
                        root = tree.nodes[root.parent]
                    context = tuple(b.committed) + (root.token,) + tuple(path)
                    target[context] = logits[offset + index]
            offset += len(nodes)
        return result
    def backbone():
        b = (OuroLoop(model, 4, 128, 128) if family == "ouro" else
             HuginnLoop(model, 4, 128, 128, 9, 128, extra_depths=(2,)))
        prefill, commit, readout, ar_readout = b.prefill, b.commit, b.readout_top, b.readout
        def recorded_prefill(prompt, depth):
            b.committed = list(prompt)
            return prefill(prompt, depth)
        def recorded_commit(tree, nodes):
            commit(tree, nodes)
            b.committed.extend(n.token for n in nodes)
        b.prefill, b.commit = recorded_prefill, recorded_commit
        b.readout_top = lambda tree, groups, full, m: capture(
            [(b, tree, groups, full)], serial_logits, lambda: readout(tree, groups, full, m))
        b.readout = lambda tree, shallow, full, m, shared=False: capture(
            [(b, tree, [(1, shallow)], full)], ar_logits,
            lambda: ar_readout(tree, shallow, full, m, shared))
        return b
    class RecordedExecutor(FusedLoopExecutor):
        def execute(self, name, jobs):
            if name == "readout_top":
                return capture([(b, *call.args[:3]) for b, call in jobs], batch_logits,
                               lambda: super(RecordedExecutor, self).execute(name, jobs))
            return super().execute(name, jobs)
    reqs = [BatchRequest(i, p, 48, set())
            for i, p in enumerate([[1, 2, 3], [4, 7], [6, 8, 9, 2]])]
    cfg = LoopSpecConfig("loopspec-dual", 4, 1, 2)
    refs = [tree_decode(backbone(), r.prompt, ArmConfig("ar", 4, 1, 0, False, False),
                        r.max_new_tokens, r.stop, max_nodes=128)[0] for r in reqs]
    serial = [loopspec_decode(backbone(), r.prompt, cfg, r.max_new_tokens, r.stop,
                              max_nodes=128) for r in reqs]
    slots = [backbone(), backbone()]
    rows, _ = batch_decode(reqs, slots, cfg, max_nodes=128, executor=RecordedExecutor(slots))
    if dtype == torch.float32:
        assert [r["tokens"] for r in rows] == [t for t, _ in serial] == refs
        assert [r["stats"]["rows"] for r in rows] == [s["rows"] for _, s in serial]
    else:
        # Check single/batch implementation parity in BF16. AR uses different
        # matrix shapes: report its common-prefix error, without inventing an
        # accuracy bound for accumulated recurrent rounding.
        assert [r["tokens"] for r in rows] == [t for t, _ in serial]
        common = serial_logits.keys() & batch_logits.keys() & ar_logits.keys()
        assert len(common) >= 10
        max_error, flips = 0., 0
        for context in common:
            torch.testing.assert_close(serial_logits[context], batch_logits[context],
                                       rtol=2 * torch.finfo(dtype).eps,
                                       atol=2 * torch.finfo(dtype).eps)
            x = ar_logits[context]
            for y in (serial_logits[context], batch_logits[context]):
                error = (x - y).abs().max().item()
                max_error = max(max_error, error)
                assert torch.isfinite(x).all() and torch.isfinite(y).all()
                if x.argmax() != y.argmax():
                    flips += 1
                    top = x.topk(2).values
                    assert top[0] - top[1] <= 2 * error
        print(f"BF16 {family}: common_prefixes={len(common)}, argmax_flips={flips}, max_abs_logit_error={max_error}")


@pytest.mark.parametrize("T,d1,d2", [(4, 1, 2), (32, 4, 8)])
def test_schedule_matches_authors_pipeline(T, d1, d2):
    repo = Path(os.environ.get("LOOPSPEC_ROOT", "/root/.cache/loopfill/loopspec"))
    path = repo / "sglang_recurrent/loopspec/pipeline.py"
    if not path.exists():
        pytest.skip("set LOOPSPEC_ROOT to the recorded official revision")
    spec = importlib.util.spec_from_file_location("official_loopspec_pipeline", path)
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    prompt = [1, 2, 7]
    trace = []
    class Executor:
        schedule = official.PipelineSchedule(T, d1, d2 // d1)
        paths = {}
        def forward(self, lanes):
            results = []
            for lane in lanes:
                if lane.branch not in self.paths:
                    self.paths[lane.branch] = self.paths.get(lane.parent_branch, ()) + (lane.token,)
            trace.append(sorted((self.paths[n.branch], n.stage) for n in lanes))
            for lane in lanes:
                depth = lane.stage + d1
                ctx = prompt + list(self.paths[lane.branch])
                token = greedy_next(ctx) if depth == T else shallow_next(ctx, depth // d1)
                proposed = depth == d1 or (depth == d2 and token != lane.first_token)
                accepted = None
                if depth == T:
                    accepted = (d1 - 1 if token == lane.first_token else
                                d2 - 1 if token == lane.second_token else None)
                results.append(official.PipelineResult(
                    state=0, prediction=token, proposed=proposed, accepted_stage=accepted,
                    proposal_action="argmax" if depth == d2 and proposed else None))
            return results
        def should_stop(self, tokens):
            return len(tokens) >= 100
        def retain_branches(self, *args):
            pass
    reference = official.continue_pipelined(greedy_next(prompt), 100, Executor())
    ours = []
    tokens, stats = loopspec_decode(
        DualFake(T, d1), prompt, LoopSpecConfig("loopspec-dual", T, d1, d2),
        100, set(), max_nodes=128,
        observer=lambda event, **data: ours.append(data["lanes"]))
    assert tokens == reference
    assert ours == trace
    assert stats["calls"] == len(trace) * d1
    assert stats["rows"] == sum(len(lanes) for lanes in trace) * d1


def test_reference_reuse_rejects_incompatible_or_corrupt_data(tmp_path):
    spec = importlib.util.spec_from_file_location("batch_cli", Path(__file__).parents[1] /
                                                 "scripts/loop_batch_bench.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    path = tmp_path / "ref.jsonl"
    reqs = [BatchRequest(1, [3], 2), BatchRequest(2, [7], 2)]
    calls = []
    def generate(r):
        calls.append(r.request_id)
        return [r.prompt[0], 9]
    cli.serial_reference(path, {"version": 1}, reqs[:1], generate)
    out = cli.serial_reference(path, {"version": 1}, reqs, generate)
    assert calls == [1, 2] and out == {1: [3, 9], 2: [7, 9]}
    with pytest.raises(ValueError, match="identity"):
        cli.serial_reference(path, {"version": 2}, reqs, generate)
    path.write_text(path.read_text() + '{"request_id":1,"tokens":[0]}\n')
    with pytest.raises(ValueError, match="duplicate"):
        cli.serial_reference(path, {"version": 1}, reqs, generate)


def test_explicit_precision_preserves_complex_rope():
    spec = importlib.util.spec_from_file_location("batch_cli_precision", Path(__file__).parents[1] /
                                                 "scripts/loop_batch_bench.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    model = torch.nn.Linear(2, 3)
    rope = torch.tensor([1 + 2j, 3 - 4j])
    model.register_buffer("rope", rope.clone())
    frequencies = torch.tensor([0.123456, 0.234567], dtype=torch.float32)
    model.register_buffer("freqs_cis", frequencies.clone())
    cli.enforce_precision(model, "bfloat16")
    assert model.weight.dtype == torch.bfloat16
    assert torch.equal(model.rope, rope) and model.rope.is_complex()
    assert model.freqs_cis.dtype == torch.float32 and torch.equal(model.freqs_cis, frequencies)
    cli.enforce_precision(model, "float32")
    assert model.weight.dtype == torch.float32
    assert torch.equal(model.rope, rope)
