"""Raven operators vs an independent dense causal oracle; no checkpoint download."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from llm_memory_editability.loop_raven_decode import RavenLoop, initial_states
from llm_memory_editability.loop_tree_decode import DraftTree, grouped_scores, grouped_values, tree_decode
from llm_memory_editability.loop_suffix_spec import ArmConfig
from llm_memory_editability.loop_batch_decode import BatchRequest, batch_decode
from llm_memory_editability.loop_optimized_decode import SharedKVPool, PackedLoopExecutor, prefill_chunk
from test_loop_batch_decode import module, Rotary


class RavenNorm(nn.Module):
    # Exact released RMSNorm: accumulate in FP32, cast, then apply learned weight.
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        fp = x.float()
        return (fp * torch.rsqrt(fp.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype) * self.weight


class OlmoRotary(Rotary):
    def forward(self, x, positions):
        return super().forward(x.float(), positions)


class TinyRaven(nn.Module):
    def __init__(self, ratio=4, olmo=False):
        super().__init__()
        heads, dim = 8, 8
        width, kv_heads = heads * dim, heads // ratio
        self.config = SimpleNamespace(n_embd=width, max_position_embeddings=128,
                                      test_time_noise=0, qk_bias=False, init_values={'std': .02})
        self.emb_scale = 1.3
        self.rotary_emb = OlmoRotary() if olmo else Rotary()
        self.olmo = olmo

        def norm(n=width):
            return RavenNorm(n)

        def block():
            attn = module(Wqkv=nn.Linear(width, width + 2 * kv_heads * dim, bias=False),
                          proj=nn.Linear(width, width, bias=False), n_head=heads,
                          n_kv_heads=kv_heads, head_dim=dim,
                          chunks=(width, kv_heads * dim, kv_heads * dim))
            if olmo:
                attn.q_norm, attn.k_norm = norm(), norm(kv_heads * dim)
            # Real gated MLP, independently exercised by dense reference below.
            mlp = Gated(width)
            return module(attn=attn, norm_1=norm(), norm_2=norm(), mlp=mlp)
        self.transformer = module(wte=nn.Embedding(23, width),
                                  prelude=nn.ModuleList([block()]),
                                  core_block=nn.ModuleList([block(), block()]),
                                  coda=nn.ModuleList([block()]),
                                  adapter=nn.Linear(width * 2, width, bias=False), ln_f=norm())
        self.lm_head = nn.Linear(width, 23, bias=False)


class Gated(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc, self.proj = nn.Linear(width, 80, bias=False), nn.Linear(40, width, bias=False)

    def forward(self, x):
        gate, value = self.fc(x).chunk(2, -1)
        return self.proj(F.silu(gate) * value)


def dense_reference(model, tokens, init, depth):
    """Uses explicit repeated KV + full causal matrix, no tree/backend helpers."""
    tr = model.transformer
    x = tr.wte(torch.tensor([tokens])) * model.emb_scale
    cos, sin = model.rotary_emb(x, torch.arange(len(tokens))[None])

    def block(layer, hidden):
        a = layer.attn
        inp = hidden if model.olmo else layer.norm_1(hidden)
        q, k, v = a.Wqkv(inp).split(a.chunks, -1)
        if model.olmo:
            q, k = a.q_norm(q), a.k_norm(k)
        q = q.view(1, -1, a.n_head, a.head_dim).transpose(1, 2)
        k = k.view(1, -1, a.n_kv_heads, a.head_dim).transpose(1, 2)
        v = v.view(1, -1, a.n_kv_heads, a.head_dim).transpose(1, 2)
        def rotate(t):
            left, right = t.chunk(2, -1)
            return (t * cos[:, None] + torch.cat([-right, left], -1) * sin[:, None]).to(t.dtype)
        q, k = rotate(q), rotate(k)
        ratio = a.n_head // a.n_kv_heads
        k, v = k.repeat_interleave(ratio, 1), v.repeat_interleave(ratio, 1)
        scores = (q @ k.transpose(-1, -2)) * a.head_dim**-.5
        mask = torch.ones(len(tokens), len(tokens), dtype=torch.bool).tril()
        weights = scores.masked_fill(~mask, float('-inf')).softmax(-1)
        attn = a.proj((weights @ v).transpose(1, 2).reshape_as(hidden))
        if model.olmo:
            hidden = hidden + layer.norm_1(attn)
            return hidden + layer.norm_2(layer.mlp(hidden))
        hidden = hidden + attn
        return hidden + layer.mlp(layer.norm_2(hidden))

    for layer in tr.prelude:
        x = block(layer, x)
    h = init[:len(tokens)][None].clone()
    for _ in range(depth):
        h = tr.adapter(torch.cat([h, x], -1))
        for layer in tr.core_block:
            h = block(layer, h)
    latent = h.clone()
    for layer in tr.coda:
        h = block(layer, h)
    return latent, model.lm_head(tr.ln_f(h))


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('ratio', [1, 4, 8])
def test_grouped_kernels_match_repeated_dense(ratio):
    torch.manual_seed(2)
    q, k, v = torch.randn(2, 3, 8, 6), torch.randn(2, 7, 8 // ratio, 6), torch.randn(2, 7, 8 // ratio, 6)
    expected = torch.einsum('gqhd,gnhd->ghqn', q, k.repeat_interleave(ratio, 2))
    torch.testing.assert_close(grouped_scores(q, k), expected)
    probs = expected.softmax(-1)
    torch.testing.assert_close(grouped_values(probs, v),
                               torch.einsum('ghqn,gnhd->gqhd', probs, v.repeat_interleave(ratio, 2)))


@pytest.mark.parametrize('ratio,olmo', [(1, False), (4, False), (8, False), (1, True)])
@torch.no_grad()
def test_serial_states_readout_caches_and_chunked_prefill(ratio, olmo):
    torch.manual_seed(73)
    model = TinyRaven(ratio, olmo).eval()
    b = RavenLoop(model, 4, 96, 20, extra_depths=(2,))
    rng = torch.random.get_rng_state()
    torch.testing.assert_close(initial_states(model, 96, 0)[:12], initial_states(model, 12, 0), rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    prompt = [1, 4, 2, 9]
    _, expected = dense_reference(model, prompt, b.init, 4)
    assert b.prefill(prompt, 1) == int(expected[0, -1].argmax())
    for chunk in (1, 3, 4):
        c = RavenLoop(model, 4, 96, 20, extra_depths=(2,))
        start = 0
        while start < len(prompt):
            start, token = prefill_chunk(c, prompt, 1, start, chunk)
        assert token == int(expected[0, -1].argmax())
        # Chunked GEMM/SDPA reduction order can accumulate ~4.2e-6 in the
        # recurrent OLMo KV state; use the same FP32 bound as latent/logit
        # comparisons below. Independent FP64 logit checks cover both paths.
        torch.testing.assert_close(c.kv.K[:, :len(prompt)], b.kv.K[:, :len(prompt)], atol=5e-6, rtol=5e-5)
    tree = DraftTree(20, 6)
    tree.reset(len(prompt))
    root = tree.add(5, None, 0, 'root')
    tree.add(7, root, 0, 'first')
    right = tree.add(8, root, 0, 'second')
    b.embed(tree, tree.take_fresh())
    for step in range(1, 5):
        b.advance(tree, tree.nodes)
        for n in tree.nodes:
            n.step += 1
            path = prompt + [5] + ([] if n is root else [n.token])
            latent, logits = dense_reference(model, path, b.init, step)
            torch.testing.assert_close(b.hidden[n.idx], latent[0, -1], atol=5e-6, rtol=5e-5)
        if step in (1, 2, 4):
            before = b.hidden.clone()
            captured = []
            hook = model.lm_head.register_forward_hook(lambda _, args, out: captured.append(out.clone()))
            reads, full = b.readout_top(tree, [(step, tree.nodes)] if step < 4 else [], tree.nodes if step == 4 else [], 2)
            hook.remove()
            for i, n in enumerate(tree.nodes):
                _, ref = dense_reference(model, prompt + [5] + ([] if n is root else [n.token]), b.init, step)
                torch.testing.assert_close(captured[0][i], ref[0, -1], atol=5e-6, rtol=5e-5)
            assert torch.equal(before, b.hidden)
    b.commit(tree, [root, right])
    assert b.length == len(prompt) + 2
    torch.testing.assert_close(b.kv.K[:, len(prompt):len(prompt) + 2], b.kv.TK[:, [root.idx, right.idx]])


@pytest.mark.parametrize('olmo', [False, True])
@pytest.mark.parametrize('packed', [False, True])
@torch.no_grad()
def test_fused_packed_continuous_isolation(olmo, packed):
    torch.manual_seed(53)
    model = TinyRaven(1 if olmo else 8, olmo).eval()
    requests = [BatchRequest(i, p, n) for i, (p, n) in enumerate([([1, 2], 5), ([2, 6, 3], 8), ([8], 4), ([7, 5, 4, 2], 6)])]
    cfg = ArmConfig('ar', 4, 1, 0, False, False)
    refs = [tree_decode(RavenLoop(model, 4, 96, 1), r.prompt, cfg, r.max_new_tokens, set(), max_nodes=1)[0] for r in requests]
    pool = SharedKVPool(2) if packed else None
    slots = [RavenLoop(model, 4, 96, 1, kv_factory=pool) for _ in range(2)]
    executor = PackedLoopExecutor(slots) if packed else None
    records, _ = batch_decode(requests, slots, cfg, None, None, max_nodes=1,
                              schedule='continuous', executor=executor, prefill_chunk_size=2)
    for r in records:
        assert r['tokens'] == refs[r['request_id']]
        assert r['stats']['calls'] == 4 * r['stats']['tokens']

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('ratio,olmo', [(1,False),(4,False),(8,False),(1,True)])
@torch.no_grad()
def test_cuda_gqa_prefix_tree_and_packed(ratio, olmo):
    from llm_memory_editability.loop_batch_decode import FusedLoopExecutor
    from llm_memory_editability.loop_decode_steps import ModelCall
    from llm_memory_editability.loop_tree_decode import make_rows, tree_attention
    torch.manual_seed(21)
    model = TinyRaven(ratio, olmo).to('cuda', torch.bfloat16).eval()
    pool = SharedKVPool(2)
    slots = [RavenLoop(model, 4, 96, 12, extra_depths=(2,), kv_factory=pool) for _ in range(2)]
    jobs = []
    for i, b in enumerate(slots):
        b.prefill([1, 2, 3][:i+2], 1)
        t = DraftTree(12, 6)
        t.reset(b.length)
        root = t.add(i + 3, None, 0, 'root')
        t.add(7, root, 0, 'left')
        t.add(9, root, 0, 'right')
        b.embed(t, t.take_fresh())
        jobs.append((b, ModelCall('advance', (t, t.nodes))))
        rows = make_rows(b.kv, t, t.nodes, [b.n_pre] * 3)
        a = b.core[0].attn
        q = torch.randn(3, a.n_head, a.head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(3, a.n_kv_heads, a.head_dim, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        fast = tree_attention(q, k, v, b.kv, rows, 0, fast=True)
        dense = tree_attention(q, k, v, b.kv, rows, 0, fast=False)
        torch.testing.assert_close(fast, dense, atol=.016, rtol=.016)
    before = [b.hidden.clone() for b in slots]
    FusedLoopExecutor(slots).execute('advance', jobs)
    reference = [b.hidden.clone() for b in slots]
    for b, h in zip(slots, before):
        b.hidden.copy_(h)
    PackedLoopExecutor(slots).execute('advance', jobs)
    for b, h in zip(slots, reference):
        torch.testing.assert_close(b.hidden, h, atol=.02, rtol=.02)


@pytest.mark.parametrize('arm', ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-raw', 'budget-online'])
@torch.no_grad()
def test_six_methods_raven_t32_two_eight(arm):
    from llm_memory_editability.loop_budget_decode import BudgetConfig
    from llm_memory_editability.loop_loopspec_decode import LoopSpecConfig
    from test_loop_prob_decode import ProbFakeSuffix
    torch.manual_seed(55)
    model = TinyRaven(8).eval()
    if arm == 'loopspec-dual':
        cfg = LoopSpecConfig(arm, 32, 2, 8)
        nodes = cfg.physical_nodes
    elif arm.startswith('budget-'):
        cfg = BudgetConfig(arm, 32, 2, (2, 8), draft_node_budget=16, top_m=2,
                           suffix_room=4, child_suffix_caps=(4, 4), use_model=arm != 'budget-suffix',
                           scoring='online_suffix_i' if arm == 'budget-online' else 'raw',
                           c_initial=.7 if arm == 'budget-online' else 1)
        nodes = 17
    else:
        cfg = ArmConfig(arm, 32, 2, 1 << 30, False, arm == 'wavefront')
        nodes = 64
    reqs = [BatchRequest(0, [1, 2, 3], 6), BatchRequest(1, [8, 5, 9, 2], 7), BatchRequest(2, [7], 4)]
    ar = ArmConfig('ar', 32, 2, 0, False, False)
    refs = [tree_decode(RavenLoop(model, 32, 96, 1), r.prompt, ar, r.max_new_tokens, set(), max_nodes=1)[0] for r in reqs]
    reqs[2].stop = {refs[2][0]}
    pool = SharedKVPool(2)
    slots = [RavenLoop(model, 32, 96, nodes, extra_depths=(8,), kv_factory=pool) for _ in range(2)]
    cache = ProbFakeSuffix() if cfg.use_suffix else None
    records, _ = batch_decode(reqs, slots, cfg, None, cache, max_nodes=nodes, schedule='continuous',
                              executor=PackedLoopExecutor(slots), prefill_chunk_size=2)
    for r in records:
        expected = refs[r['request_id']]
        assert r['tokens'] == (expected[:1] if r['request_id'] == 2 else expected)
        assert r['stats']['calls'] > 0 or r['stats']['tokens'] == 0
        if arm == 'ar':
            assert r['stats']['calls'] == 32 * r['stats']['tokens']

@torch.no_grad()
def test_wavefront_tail_never_computes_past_context():
    torch.manual_seed(91)
    model = TinyRaven(8).eval()
    b = RavenLoop(model, 4, 12, 64)
    cfg = ArmConfig('wavefront', 4, 1, 1 << 30, False, True)
    seen = []
    original = b.embed
    def embed(tree, nodes):
        seen.extend(tree.position(n) for n in nodes)
        original(tree, nodes)
    b.embed = embed
    output, _ = tree_decode(b, [1, 2, 4, 5, 6, 7], cfg, 6, set(), max_nodes=64, shared_coda=True)
    assert len(output) == 6 and max(seen) < 12
    with pytest.raises(ValueError, match='context'):
        tree_decode(b, [1] * 10, cfg, 3, set(), max_nodes=64)


@torch.no_grad()
def test_olmo_fp32_chunk_roundoff_against_independent_fp64_logits():
    import copy
    import types
    torch.manual_seed(73)
    model = TinyRaven(1, True).eval()
    reference = copy.deepcopy(model).double()
    # The oracle accumulates normalization in FP64, unlike the released FP32
    # normalization. Its dense attention already uses separate explicit math.
    def precise_norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight
    for norm in reference.modules():
        if isinstance(norm, RavenNorm):
            norm.forward = types.MethodType(precise_norm, norm)
    prompt = [1, 4, 2, 9]
    init = initial_states(model, 96, 0)
    _, expected = dense_reference(reference, prompt, init.double(), 4)
    for chunk in (1, 3, 4):
        b = RavenLoop(model, 4, 96, 20, extra_depths=(2,))
        captured = []
        hook = model.lm_head.register_forward_hook(lambda _, args, out: captured.append(out.clone()))
        if chunk == 4:
            b.prefill(prompt, 1)
        else:
            start = 0
            while start < len(prompt):
                start, _ = prefill_chunk(b, prompt, 1, start, chunk)
        hook.remove()
        torch.testing.assert_close(captured[-1].double(), expected[:, -1], atol=3e-6, rtol=3e-6)
