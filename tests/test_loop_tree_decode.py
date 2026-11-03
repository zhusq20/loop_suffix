import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability.loop_suffix_spec import ArmConfig, replay_request
from llm_memory_editability.loop_tree_decode import (
    DraftTree,
    LoopKV,
    make_rows,
    tree_attention,
    tree_decode,
)

VOCAB = 11


def greedy_next(ctx):
    return (3 * ctx[-1] + ctx[-2] + (len(ctx) % 5 == 0)) % VOCAB


def shallow_next(ctx, depth):
    g = greedy_next(ctx)
    return g if (7 * sum(ctx[-3:]) + len(ctx) + depth) % 4 else (g + 1) % VOCAB


def top_list(first, m):
    return [(first + i) % VOCAB for i in range(m)]


class FakeLoop:
    """Readouts are functions of the token context; checks the scheduler's bookkeeping."""

    def __init__(self, T, d):
        self.T, self.d = T, d
        self.seq: list[int] = []
        self.advanced = 0

    @property
    def length(self):
        return len(self.seq)

    def synchronize(self):
        pass

    def prefill(self, prompt, shallow_depth):
        self.seq = list(prompt)
        return greedy_next(self.seq)

    def context(self, tree, node):
        # the step's own chain may already be (partly) committed into self.seq
        return self.seq[: tree.base_pos] + [tree.nodes[0].token] + tree.path_tokens(node)

    def embed(self, tree, nodes):
        assert all(n.step == 0 for n in nodes)

    def advance(self, tree, nodes):
        assert all(n.alive and n.step < self.T for n in nodes)
        self.advanced += 1

    def readout(self, tree, shallow, full, m, shared_coda=False):
        sh = [top_list(shallow_next(self.context(tree, n), self.d), m) for n in shallow]
        fu = [top_list(greedy_next(self.context(tree, n)), m) for n in full]
        return sh, fu

    def commit(self, tree, chain):
        assert chain and all(n.step == self.T and n.alive for n in chain)
        start = len(self.seq)
        assert [tree.position(n) for n in chain] == list(range(start, start + len(chain)))
        assert all(n.idx >= tree.lo for n in chain)
        self.seq += [n.token for n in chain]


class FakeSuffix:
    """Chain from the latest earlier occurrence of the last token, plus one sibling."""

    def __init__(self):
        self.text: dict = {}

    def start_request(self, req_id, prompt_token_ids):
        self.text[req_id] = [int(t) for t in prompt_token_ids]

    def add_active_response(self, req_id, token_ids):
        self.text[req_id] += [int(t) for t in token_ids]

    def stop_request(self, req_id):
        del self.text[req_id]

    def speculate(self, req_id, context, max_spec_tokens=None, **kwargs):
        text, last = self.text[req_id], int(context[-1])
        hits = [i for i in range(len(text) - 1) if text[i] == last]
        if not hits or not max_spec_tokens:
            return SimpleNamespace(token_ids=[], parents=[])
        chain = text[hits[-1] + 1 : hits[-1] + 1 + max_spec_tokens]
        toks, parents = list(chain), list(range(-1, len(chain) - 1))
        if len(hits) > 1 and text[hits[-2] + 1] != chain[0]:
            toks.append(text[hits[-2] + 1])
            parents.append(-1)
        return SimpleNamespace(token_ids=toks, parents=parents)


def greedy(prompt, max_new, stop):
    out = [greedy_next(list(prompt))]
    while len(out) < max_new and out[-1] not in stop:
        out.append(greedy_next(list(prompt) + out))
    return out


def fake_readouts(prompt, out, T, d, k=4):
    rows = [list(prompt) + out[:i] for i in range(len(out))]
    return {
        d: np.array([top_list(shallow_next(c, d), k) for c in rows]),
        T: np.array([top_list(greedy_next(c), k) for c in rows]),
    }


def arms(T, d, K=8):
    return [
        ArmConfig("ar", T, d, 0, use_suffix=False, use_fill=False),
        ArmConfig("shallow", T, d, K, use_suffix=False, use_fill=True),
        ArmConfig("suffix", T, d, K, use_suffix=True, use_fill=False),
        ArmConfig("both", T, d, K, use_suffix=True, use_fill=True),
        ArmConfig("both prune2", T, d, K, use_suffix=True, use_fill=True, prune_top=2),
        ArmConfig("both K3", T, d, 3, use_suffix=True, use_fill=True),
        ArmConfig("wavefront", T, d, 1 << 30, use_suffix=False, use_fill=True),
    ]


@pytest.mark.parametrize("T,d", [(4, 1), (6, 2), (8, 4)])
@pytest.mark.parametrize("stop", [set(), {5}])
def test_calls_match_replay_and_output_is_greedy(T, d, stop):
    prompt = [1, 2, 3, 4, 5, 6, 1, 2]
    max_new = 60
    out = greedy(prompt, max_new, stop)
    reads = fake_readouts(prompt, out, T, d)
    for arm in arms(T, d):
        ref = replay_request(0, prompt, out, reads, arm, FakeSuffix())
        for scope in ("all", "path"):
            got, st = tree_decode(
                FakeLoop(T, d), prompt, arm, max_new, stop, FakeSuffix(), 0, fill_scope=scope
            )
            assert got == out, (arm.name, scope)
            assert st["tokens"] == ref["tokens"] == len(out) - 1
            if scope == "all":
                assert st["calls"] == ref["calls"], (arm.name, st["calls"], ref["calls"])
                assert st["from_suffix"] == ref["from_suffix"]
                assert st["from_fill"] == ref["from_fill"]
    assert arms(T, d)[0].max_draft == 0
    _, st = tree_decode(FakeLoop(T, d), prompt, arms(T, d)[0], max_new, stop)
    assert st["calls"] == T * (len(out) - 1) and st["max_rows"] == 1


def test_node_cap_skips_but_stays_lossless():
    prompt, T, d = [1, 2, 3, 4, 5, 6, 1, 2], 6, 2
    out = greedy(prompt, 40, set())
    arm = ArmConfig("both", T, d, 8, use_suffix=True, use_fill=True)
    got, st = tree_decode(FakeLoop(T, d), prompt, arm, 40, set(), FakeSuffix(), 0, max_nodes=3)
    assert got == out and st["cap_skips"] > 0 and st["max_rows"] <= 3


def build_tree(parents):
    tree = DraftTree(16, 16)
    tree.reset(base_pos=7)
    nodes = []
    for i, p in enumerate(parents):
        nodes.append(tree.add(100 + i, None if p < 0 else nodes[p], 0, "suffix"))
    return tree, nodes


def test_tree_attention_matches_dense_reference():
    torch.manual_seed(0)
    H, D, L, S = 3, 8, 7, 6
    kv = LoopKV(S, 32, 16, H, D, torch.float32, torch.device("cpu"))
    kv.K.normal_()
    kv.V.normal_()
    kv.TK.normal_()
    kv.TV.normal_()
    kv.length = L
    tree, nodes = build_tree([-1, 0, 0, 1, 3, 2])
    rows_nodes, bases = [nodes[5], nodes[0], nodes[3], nodes[4]], [2, 0, 2, 4]
    rows = make_rows(kv, tree, rows_nodes, bases)
    assert [n.idx for n in rows.nodes] == [0, 5, 3, 4] and rows.max_q == 2
    q, k, v = torch.randn(3, 4, H, D).unbind(0)
    layer = 1
    out = tree_attention(q, k, v, kv, rows, layer)
    for r, node in enumerate(rows.nodes):
        slot = int(rows.base[r]) + layer
        assert torch.equal(kv.TK[slot, node.idx], k[r])
        anc = np.flatnonzero(tree.anc[node.idx, : len(tree.nodes)])
        keys = torch.cat([kv.K[slot, :L], kv.TK[slot, anc]])
        vals = torch.cat([kv.V[slot, :L], kv.TV[slot, anc]])
        w = torch.softmax(torch.einsum("hd,lhd->hl", q[r], keys) * D**-0.5, dim=-1)
        torch.testing.assert_close(out[r], torch.einsum("hl,lhd->hd", w, vals))


def test_tree_attention_after_flushing_the_chain_head():
    """Flushed chain nodes are read from the main cache, never twice."""
    torch.manual_seed(1)
    H, D, L, S = 2, 8, 5, 4
    kv = LoopKV(S, 32, 16, H, D, torch.float32, torch.device("cpu"))
    for t in (kv.K, kv.V, kv.TK, kv.TV):
        t.normal_()
    kv.length = L
    tree, nodes = build_tree([-1, 0, 1, 2, 2, 3])  # chain 0-1-2, children 3/4 of 2, 5 under 3
    tree.base_pos = L
    kv.commit([0, 1])
    tree.lo = 2
    rows_nodes = [nodes[3], nodes[4], nodes[2], nodes[5]]
    rows = make_rows(kv, tree, rows_nodes, [0, 0, 2, 0])
    assert rows.lo == 2 and rows.anc.shape[-1] == 4
    q, k, v = torch.randn(3, 4, H, D).unbind(0)
    out = tree_attention(q, k, v, kv, rows, 1)
    for r, node in enumerate(rows.nodes):
        slot = int(rows.base[r]) + 1
        anc = [i for i in np.flatnonzero(tree.anc[node.idx, : len(tree.nodes)]) if i > 1]
        keys = torch.cat([kv.K[slot, : L + 2], kv.TK[slot, anc]])
        vals = torch.cat([kv.V[slot, : L + 2], kv.TV[slot, anc]])
        w = torch.softmax(torch.einsum("hd,lhd->hl", q[r], keys) * D**-0.5, dim=-1)
        torch.testing.assert_close(out[r], torch.einsum("hl,lhd->hd", w, vals))


def test_uncapped_chain_has_one_step_per_rejection():
    """Without a draft cap (WFD) every output token after the first was drafted, so the
    tokens not taken from a draft are exactly the wrong drafts; with K=4 there are more
    (cap-ended steps) and more calls."""
    prompt, T, d = [1, 2, 3, 4, 5, 6, 1, 2], 8, 2
    out = greedy(prompt, 80, set())
    wrong = sum(shallow_next(prompt + out[:j], d) != out[j] for j in range(1, len(out)))
    wf = ArmConfig("wavefront", T, d, 1 << 30, use_suffix=False, use_fill=True)
    capped = ArmConfig("shallow", T, d, 4, use_suffix=False, use_fill=True)
    got, st = tree_decode(FakeLoop(T, d), prompt, wf, 80, set(), max_nodes=128)
    _, st4 = tree_decode(FakeLoop(T, d), prompt, capped, 80, set())
    assert got == out and st["cap_skips"] == 0
    assert st["tokens"] - st["from_fill"] == wrong
    assert st4["tokens"] - st4["from_fill"] > wrong and st4["calls"] > st["calls"]
    for every in (1, 3, 16):  # flushing finished chain nodes changes nothing but storage
        got_f, st_f = tree_decode(
            FakeLoop(T, d), prompt, wf, 80, set(), max_nodes=128, flush_every=every
        )
        assert got_f == out and st_f["calls"] == st["calls"]


def test_commit_copies_chain_to_main_cache():
    kv = LoopKV(2, 16, 8, 1, 2, torch.float32, torch.device("cpu"))
    kv.TK.normal_()
    kv.length = 3
    kv.commit([0, 2])
    assert kv.length == 5
    assert torch.equal(kv.K[:, 3], kv.TK[:, 0]) and torch.equal(kv.K[:, 4], kv.TK[:, 2])


def test_init_table_rows_do_not_depend_on_length():
    def table(n):
        t = torch.empty(n, 6)
        gen = torch.Generator().manual_seed(0)
        return torch.nn.init.trunc_normal_(t, std=0.1, a=-0.3, b=0.3, generator=gen)

    assert torch.equal(table(9), table(40)[:9])


# ---------------------------------------------------------------------------
# Tiny random Ouro / Huginn from the checkpoints' own modeling code (CPU, fp32)
# ---------------------------------------------------------------------------

HF_ROOT = Path(os.environ.get("LOOP_HF_ROOT", "/hf/hub"))
OURO = HF_ROOT / "models--ByteDance--Ouro-2.6B/snapshots/1ed04250da1a9936042725d302e81c8fa2ab5abd"
HUGINN = (
    HF_ROOT
    / "models--tomg-group-umd--huginn-0125/snapshots/bb6621b65e90b6a4b9b29ef88dc83866d450470c"
)


def tiny_ouro():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(OURO, trust_remote_code=True)
    cfg.update(
        dict(
            hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, head_dim=16, layer_types=["full_attention"] * 2,
            max_window_layers=2, vocab_size=97, total_ut_steps=4,
        )
    )  # fmt: skip
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()


def tiny_huginn(T):
    from transformers import AutoConfig, AutoModelForCausalLM

    base = AutoConfig.from_pretrained(HUGINN, trust_remote_code=True)
    cfg = type(base)(
        n_embd=64, n_heads=4, n_layers=8, block_size=256, vocab_size=97, intermediate_size=96,
        mean_recurrence=T,
    )  # fmt: skip
    cfg.auto_map, cfg._name_or_path = base.auto_map, base._name_or_path
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).float().eval()


def check_against_hf(backbone_for, runner, T, d, depths):
    """Every arm reproduces the HF greedy output and the replay's call count."""
    prompt = [5, 9, 13, 5, 9, 2, 40, 41, 5, 9, 13, 7]
    max_new = 40
    ids = torch.tensor([prompt])
    out = runner.generate(ids, max_new, set())
    seq = torch.cat([ids, torch.tensor([out])], dim=1)
    reads = runner.readouts(seq, len(prompt) - 1, depths, 4)
    readouts = {t: r[0] for t, r in zip(depths, reads, strict=True)}
    for arm in arms(T, d):
        ref = replay_request(0, prompt, out, readouts, arm, FakeSuffix())
        got, st = tree_decode(backbone_for(), prompt, arm, max_new, set(), FakeSuffix(), 0)
        assert got == out, arm.name
        assert st["calls"] == ref["calls"], (arm.name, st["calls"], ref["calls"])
    # flushing and WFD's shared coda cache (Huginn) never change the output
    wf = arms(T, d)[-1]
    ref = replay_request(0, prompt, out, readouts, wf, None)
    got, st = tree_decode(backbone_for(), prompt, wf, max_new, set(), flush_every=3)
    assert got == out and st["calls"] == ref["calls"]
    got, _ = tree_decode(backbone_for(), prompt, wf, max_new, set(), shared_coda=True)
    assert got == out
    return prompt, out


WFD_ROOT = Path(os.environ.get("LOOP_WFD_ROOT", "/wfd"))


def wfd_reference(model, kind, T, d, prompt, max_new, init=None):
    """Tokens and R calls of WFD's own scheduler (reference code, CPU)."""
    import sys

    sys.path.insert(0, str(WFD_ROOT))
    from src.decoding.policy import SsdPolicy
    from src.decoding.scheduler import generate_wavefront_dynamic

    if kind == "ouro":
        from src.decoding.ouro_blocks import OuroBlocks as Blocks
    else:
        from src.decoding.huginn_blocks import HuginnBlocks as Blocks
    blocks = Blocks(model, T_total=T, max_seq_len=128)
    if init is not None:
        blocks._init_state_tokens = lambda pos: init[torch.as_tensor(pos)]
        blocks._init_state_prefill = lambda b, n: init[:n].unsqueeze(0).expand(b, -1, -1)
    toks, trace = generate_wavefront_dynamic(
        blocks, torch.tensor(prompt), max_new, SsdPolicy(T_total=T, T_draft=d, mode="greedy")
    )
    return toks, trace.n_r_calls


@pytest.mark.skipif(not OURO.exists(), reason="Ouro snapshot not mounted")
def test_ouro_backbone_matches_hf_greedy_and_replay():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from loop_suffix_spec_readouts import OuroRunner

    from llm_memory_editability.loop_tree_decode import OuroLoop

    model = tiny_ouro()
    with torch.no_grad():
        check_against_hf(
            lambda: OuroLoop(model, 4, 96, 64), OuroRunner(model, 4), 4, 1, [1, 2, 3, 4]
        )


@pytest.mark.skipif(not (OURO.exists() and WFD_ROOT.exists()), reason="Ouro or WFD not mounted")
@pytest.mark.parametrize("d", [1, 2])
def test_ouro_wavefront_equals_wfd_scheduler(d):
    from llm_memory_editability.loop_tree_decode import OuroLoop

    model = tiny_ouro()
    prompt, max_new, T = [5, 9, 13, 5, 9, 2, 40, 41, 5, 9, 13, 7], 40, 4
    wf = ArmConfig("wavefront", T, d, 1 << 30, use_suffix=False, use_fill=True)
    with torch.no_grad():
        got, st = tree_decode(OuroLoop(model, T, 96, 64), prompt, wf, max_new, set(), flush_every=4)
        ref, r_calls = wfd_reference(model, "ouro", T, d, prompt, max_new)
    assert got == ref and st["calls"] == r_calls, (st["calls"], r_calls)


@pytest.mark.skipif(not HUGINN.exists(), reason="Huginn snapshot not mounted")
def test_huginn_backbone_matches_hf_greedy_and_replay():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from loop_suffix_spec_readouts import HuginnRunner

    from llm_memory_editability.loop_tree_decode import HuginnLoop

    T, d = 6, 2
    model = tiny_huginn(T)
    init_len = 64
    runner = HuginnRunner(model, T, 0, init_len, "cpu")
    with torch.no_grad():
        check_against_hf(
            lambda: HuginnLoop(model, T, 96, 64, seed=0, init_len=init_len),
            runner, T, d, [d, T],
        )  # fmt: skip


@pytest.mark.skipif(not (HUGINN.exists() and WFD_ROOT.exists()), reason="Huginn or WFD absent")
@pytest.mark.parametrize("shared", [True, False])
def test_huginn_wavefront_equals_wfd_scheduler_only_with_shared_coda(shared):
    from llm_memory_editability.loop_tree_decode import HuginnLoop

    T, d = 6, 2
    model = tiny_huginn(T)
    prompt, max_new = [5, 9, 13, 5, 9, 2, 40, 41, 5, 9, 13, 7], 40
    wf = ArmConfig("wavefront", T, d, 1 << 30, use_suffix=False, use_fill=True)
    with torch.no_grad():
        loop = HuginnLoop(model, T, 96, 64, seed=0, init_len=64)
        got, st = tree_decode(loop, prompt, wf, max_new, set(), shared_coda=shared, flush_every=4)
        ref, r_calls = wfd_reference(model, "huginn", T, d, prompt, max_new, init=loop.init)
    assert got == ref
    if shared:
        assert st["calls"] == r_calls, (st["calls"], r_calls)
    else:  # informative: separate shallow coda changes the drafts
        print("separate coda calls", st["calls"], "wfd", r_calls)
