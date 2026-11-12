"""Multi-request parity, isolation, and actual projection fusion on tiny models."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode
from llm_memory_editability.loop_batch_suffix import CompletionSuffixCache
from llm_memory_editability.loop_prob_decode import prob_tree_decode
from llm_memory_editability.loop_prob_tree import ProbConfig
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_tree_decode import HuginnLoop, OuroLoop, tree_decode
from test_loop_prob_decode import ProbFakeLoop, ProbFakeSuffix, calibration
from test_loop_tree_decode import greedy


# Minimal random layers implement the same interfaces as the released models.
# No downloads or checkpoint fixtures are needed to test the production backbones.
def module(**attrs):
    m = nn.Module()
    for key, value in attrs.items():
        setattr(m, key, value)
    return m


class Rotary(nn.Module):
    def forward(self, x, positions):
        phase = positions[..., None].float() * torch.tensor([0.1, 0.2, 0.3, 0.4], device=x.device)
        phase = torch.cat([phase, phase], dim=-1)
        return phase.cos().to(x.dtype), phase.sin().to(x.dtype)


def apply_rotary_emb_complex_like(q, k, freqs_cis):
    # Huginn's callback signature, using position-dependent real rotations here.
    phase = freqs_cis[..., None, None].float() * 0.1

    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    return (q * phase.cos() + rotate(q) * phase.sin()).to(q.dtype), (
        k * phase.cos() + rotate(k) * phase.sin()
    ).to(k.dtype)


class TinyHuginn(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=2,
            n_embd=16,
            init_values={"std": 0.1},
            qk_bias=False,
        )
        self.emb_scale = 1

        def block():
            return module(
                attn=module(
                    Wqkv=nn.Linear(16, 48),
                    proj=nn.Linear(16, 16),
                    n_head=2,
                    n_kv_heads=2,
                    head_dim=8,
                    chunks=(16, 16, 16),
                ),
                norm_1=nn.LayerNorm(16),
                norm_2=nn.LayerNorm(16),
                norm_3=nn.LayerNorm(16),
                norm_4=nn.LayerNorm(16),
                mlp=nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 16)),
            )

        self.transformer = module(
            wte=nn.Embedding(19, 16),
            prelude=nn.ModuleList([block()]),
            core_block=nn.ModuleList([block(), block()]),
            coda=nn.ModuleList([block()]),
            adapter=nn.Linear(32, 16),
            ln_f=nn.LayerNorm(16),
        )
        self.lm_head = nn.Linear(16, 19, bias=False)
        self.register_buffer("freqs_cis", torch.arange(128)[None])


class TinyOuro(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            vocab_size=19,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=16,
            head_dim=8,
        )

        def layer():
            return module(
                self_attn=module(
                    q_proj=nn.Linear(16, 16),
                    k_proj=nn.Linear(16, 16),
                    v_proj=nn.Linear(16, 16),
                    o_proj=nn.Linear(16, 16),
                    head_dim=8,
                ),
                input_layernorm=nn.LayerNorm(16),
                input_layernorm_2=nn.LayerNorm(16),
                post_attention_layernorm=nn.LayerNorm(16),
                post_attention_layernorm_2=nn.LayerNorm(16),
                mlp=nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 16)),
            )

        self.model = module(
            embed_tokens=nn.Embedding(19, 16),
            layers=nn.ModuleList([layer(), layer()]),
            rotary_emb=Rotary(),
            norm=nn.LayerNorm(16),
        )
        self.lm_head = nn.Linear(16, 19, bias=False)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config_for(arm, suffix=False):
    if arm.startswith("prob"):
        return ProbConfig(
            arm,
            4,
            1,
            (1, 2),
            tau=0.01 if arm == "prob-low" else 0.1,
            use_suffix=suffix,
            joint=True,
            prune=True,
            suffix_calibration="insitu",
        )
    return ArmConfig(arm, 4, 1, 1 << 30, suffix, arm == "wavefront")


@pytest.mark.parametrize("family", ["ouro", "huginn"])
@pytest.mark.parametrize("arm", ["ar", "wavefront", "prob-low", "prob-load"])
def test_fused_batch_equals_serial_and_reuses_slots(family, arm):
    torch.manual_seed(7)
    model = (TinyOuro() if family == "ouro" else TinyHuginn()).eval()

    def backbone():
        return (
            OuroLoop(model, 4, 128, 128)
            if family == "ouro"
            else HuginnLoop(model, 4, 128, 128, 0, 128, extra_depths=(2,))
        )

    requests = [
        BatchRequest(i, prompt, budget)
        for i, (prompt, budget) in enumerate(
            [([1, 2, 3], 4), ([3, 2, 1, 5, 6], 15), ([7, 8], 5), ([3, 7, 4, 8], 9), ([5], 2)]
        )
    ]
    cfg, cal = config_for(arm), calibration((1, 2))
    refs = []
    for r in requests:
        b = backbone()
        if isinstance(cfg, ProbConfig):
            refs.append(
                prob_tree_decode(
                    b, r.prompt, cfg, cal, r.max_new_tokens, r.stop, max_nodes=128, flush_every=3
                )
            )
        else:
            refs.append(
                tree_decode(
                    b,
                    r.prompt,
                    cfg,
                    r.max_new_tokens,
                    r.stop,
                    max_nodes=128,
                    flush_every=3,
                    shared_coda=arm == "wavefront",
                )
            )
    # End one request on its first predicted token, while a longer request remains.
    requests[2].stop = {refs[2][0][0]}
    seen_rows = []
    layer = (
        model.model.layers[0].self_attn.q_proj
        if family == "ouro"
        else (model.transformer.core_block[0].attn.Wqkv)
    )
    hook = layer.register_forward_pre_hook(lambda m, args: seen_rows.append(args[0].shape[1]))
    records, summary = batch_decode(
        requests, [backbone(), backbone()], cfg, cal, max_nodes=128, flush_every=3
    )
    hook.remove()
    for r, got, (ref, stats) in zip(requests, records, refs):
        expected = ref[:1] if r.request_id == 2 else ref
        assert got["tokens"] == expected
        if r.request_id != 2:
            assert got["stats"]["calls"] == stats["calls"]
            assert got["stats"]["rows"] == stats["rows"]
        assert 0 <= got["ttft_s"] <= got["latency_s"]
    assert records[2]["admitted_s"] >= max(r["completed_s"] for r in records[:2])
    assert records[4]["admitted_s"] >= max(r["completed_s"] for r in records[2:4])
    assert summary["output_tokens"] == sum(len(r["tokens"]) for r in records)
    assert summary["output_tput"] == summary["output_tokens"] / summary["wall_s"]
    assert summary["core_calls"] < sum(r["stats"]["calls"] for r in records)
    assert summary["core_rows"] == sum(r["stats"]["rows"] for r in records)
    assert max(seen_rows) >= 2


class FakeExecutor:
    def __init__(self):
        self.core_calls = self.core_rows = self.max_core_rows = 0

    def execute(self, name, jobs):
        if name == "advance":
            self.core_calls += 1
            n = sum(len(c.args[1]) for _, c in jobs)
            self.core_rows += n
            self.max_core_rows = max(n, self.max_core_rows)
        return [getattr(b, name)(*c.args) for b, c in jobs]


@pytest.mark.parametrize("max_nodes", [3, 128])
def test_batch_probability_suffix_rejection_and_node_cap(max_nodes):
    reqs = [
        BatchRequest(i, p, n, stop)
        for i, (p, n, stop) in enumerate(
            [([1, 2, 3, 4, 1, 2], 40, set()), ([7, 8], 33, {5}), ([5, 4, 3], 12, set())]
        )
    ]
    cfg = config_for("prob-low", suffix=True)
    out, summary = batch_decode(
        reqs,
        [ProbFakeLoop(4, 1), ProbFakeLoop(4, 1)],
        cfg,
        calibration((1, 2)),
        ProbFakeSuffix(),
        max_nodes=max_nodes,
        executor=FakeExecutor(),
        flush_every=3,
    )
    for r, got in zip(reqs, out):
        assert got["tokens"] == greedy(r.prompt, r.max_new_tokens, r.stop)
    if max_nodes == 3:
        assert sum(r["stats"]["cap_skips"] for r in out) > 0
    assert summary["requests"] == 3


class RecordingCache:
    """Public Arctic interface stand-in; exposes every globally visible token."""

    def __init__(self, max_cached_requests):
        self.max_cached_requests = max_cached_requests
        self.max_tree_depth = 64
        self.local, self.global_ = {}, {}

    @property
    def cached_requests(self):
        return self.global_.keys()

    def start_request(self, rid, tokens):
        self.local[rid] = list(tokens)
        if self.max_cached_requests:
            self.global_[rid] = []

    def add_active_response(self, rid, tokens):
        self.local[rid].extend(tokens)
        if self.max_cached_requests:
            self.global_[rid].extend(tokens)

    def speculate(self, rid, context, **kw):
        tokens = self.local[rid] + [t for ts in self.global_.values() for t in ts]
        return SimpleNamespace(token_ids=tokens, score=len(tokens))

    def stop_request(self, rid):
        del self.local[rid]

    def evict_cached_response(self, rid):
        del self.global_[rid]


def test_suffix_only_publishes_completed_outputs_and_aborts_cleanly():
    cache = CompletionSuffixCache(RecordingCache(0), RecordingCache(100))
    cache.start_request("a", np.array([1, 2], dtype=np.int32))
    cache.start_request("b", np.array([3], dtype=np.int32))
    cache.add_active_response("a", np.array([91, 92, 93], dtype=np.int32))
    assert 91 in cache.speculate("a", []).token_ids
    assert 91 not in cache.speculate("b", []).token_ids
    cache.stop_request("a")
    assert 91 in cache.speculate("b", []).token_ids
    assert 1 not in cache.speculate("b", []).token_ids  # no leaked prompt
    cache.add_active_response("b", [99])
    cache.abort_request("b")
    assert "b" not in cache.history.cached_requests
    assert cache.responses == {}


def test_invalid_batch_fails_before_model_execution():
    cfg = config_for("ar")
    r = BatchRequest(1, [1, 2], 1)
    for requests in ([r, r], [replace(r, prompt=[])], [replace(r, max_new_tokens=0)]):
        with pytest.raises(ValueError):
            batch_decode(requests, [ProbFakeLoop(4, 1)], cfg, executor=FakeExecutor())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("family", ["ouro", "huginn"])
def test_cuda_bf16_fused_forward_matches_independent_attention(family):
    """Exercise the real FlashAttention prefix path, including ragged depths/prefixes."""
    from llm_memory_editability.loop_batch_decode import FusedLoopExecutor
    from llm_memory_editability.loop_decode_steps import ModelCall
    from llm_memory_editability.loop_tree_decode import DraftTree

    torch.manual_seed(29)
    model = (TinyOuro() if family == "ouro" else TinyHuginn()).to("cuda", torch.bfloat16).eval()

    def backbone():
        return (
            OuroLoop(model, 4, 128, 16)
            if family == "ouro"
            else HuginnLoop(model, 4, 128, 16, 0, 128, extra_depths=(2,))
        )

    serial, fused, trees = [], [], []
    with torch.no_grad():
        for prompt in ([1, 2, 3], [6, 7, 8, 9, 2]):
            a, b = backbone(), backbone()
            a.prefill(prompt, 1)
            b.prefill(prompt, 1)
            tree = DraftTree(16, 16)
            tree.reset(a.length)
            root = tree.add(4, None, 0, "root")
            tree.add(5, root, 0, "fill")
            tree.add(7, root, 0, "sibling")
            a.embed(tree, tree.nodes)
            b.embed(tree, tree.nodes)
            serial.append(a)
            fused.append(b)
            trees.append(tree)
        engine = FusedLoopExecutor(fused)
        for _ in range(3):
            for a, t in zip(serial, trees):
                a.advance(t, t.nodes)
            engine.execute(
                "advance", [(b, ModelCall("advance", (t, t.nodes))) for b, t in zip(fused, trees)]
            )
            for a, b, t in zip(serial, fused, trees):
                torch.testing.assert_close(a.hidden[:3], b.hidden[:3], atol=0.08, rtol=0.03)
                for n in t.nodes:
                    n.step += 1


@pytest.mark.parametrize("family", ["ouro", "huginn"])
def test_batch_one_matches_original_decoder(family):
    torch.manual_seed(11)
    model = (TinyOuro() if family == "ouro" else TinyHuginn()).eval()
    b = (
        OuroLoop(model, 4, 128, 128)
        if family == "ouro"
        else HuginnLoop(model, 4, 128, 128, 0, 128, extra_depths=(2,))
    )
    cfg, cal = config_for("prob-low", suffix=True), calibration((1, 2))
    req = BatchRequest(7, [1, 3, 4, 1, 3], 20)
    expected, stats = prob_tree_decode(
        b,
        req.prompt,
        cfg,
        cal,
        req.max_new_tokens,
        set(),
        ProbFakeSuffix(),
        req.request_id,
        max_nodes=128,
    )
    got, _ = batch_decode([req], [b], cfg, cal, ProbFakeSuffix(), max_nodes=128)
    assert got[0]["tokens"] == expected
    for key in ("calls", "rows", "readout_rows", "from_suffix", "from_fill", "cap_skips"):
        assert got[0]["stats"][key] == stats[key]


@pytest.mark.parametrize("schedule", ["fixed", "continuous"])
def test_cli_writes_four_arm_paired_results(tmp_path, monkeypatch, schedule):
    import json
    import sys
    import transformers
    from scripts import loop_batch_bench as bench

    requests = tmp_path / "input.jsonl"
    requests.write_text(
        "\n".join(
            json.dumps(dict(question_id=i, prompt_token_ids=p, max_new_tokens=4, category="test"))
            for i, p in enumerate([[1, 2, 3], [3, 7]])
        )
    )
    cal_path = tmp_path / "cal.json"
    cal_path.write_text(json.dumps({"dev": calibration((1, 2)).to_json()}))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: TinyOuro().eval()
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **kw: TinyOuro().config)
    tokenizer = SimpleNamespace(eos_token_id=None, get_vocab=lambda: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)

    def new_cache():
        cache = ProbFakeSuffix()
        cache.publications = []
        return cache

    monkeypatch.setattr(bench.CompletionSuffixCache, "arctic", new_cache)
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "loop_batch_bench.py",
            "--model",
            "ouro-2.6b",
            "--model-path",
            str(tmp_path),
            "--questions",
            str(requests),
            "--out",
            str(out),
            "--batch-sizes",
            "1,2",
            "--schedule",
            schedule,
            "--order-seeds",
            "5",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-seq",
            "128",
            "--max-nodes",
            "64",
            "--warmup-tokens",
            "2",
            "--check",
            "--calibration",
            str(cal_path),
        ],
    )
    bench.main()
    summaries = [json.loads(p.read_text()) for p in out.glob("*-summary.json")]
    assert len(summaries) == 8
    for summary in summaries:
        assert summary["requests"] == summary["same_as_serial_ar"] == 2
        assert summary["output_tokens"] == 8
        assert summary["request_order"] == summaries[0]["request_order"]
        assert summary["schedule"] == schedule
    assert json.loads((out / "metadata.json").read_text())["cuda_graph"] is False
    with pytest.raises(FileExistsError):
        bench.main()
