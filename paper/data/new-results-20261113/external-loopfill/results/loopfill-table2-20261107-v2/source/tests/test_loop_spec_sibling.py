import torch

from scripts.loop_spec_sibling import LaneKVCache, Layout, spread_subset


class FakeMain:
    """Minimal stand-in for WFD's StaticKVCache: buffer index == position."""

    def __init__(self, length=16, heads=2, dim=4):
        self.n_kv_head, self.head_dim = heads, dim
        self.dtype, self.device = torch.float32, torch.device("cpu")
        self.K, self.V = {}, {}
        self.shape = (1, length, heads, dim)
        self.dropped = None

    def _slot(self, s):
        if s not in self.K:
            self.K[s] = torch.zeros(self.shape)
            self.V[s] = torch.zeros(self.shape)

    def write_range(self, s, pos, k, v):
        self._slot(s)
        self.K[s][:, pos : pos + k.shape[1]] = k
        self.V[s][:, pos : pos + k.shape[1]] = v

    def gather(self, s, upto):
        self._slot(s)
        return self.K[s][:, : upto + 1], self.V[s][:, : upto + 1]

    def drop_positions_at_and_after(self, pos):
        self.dropped = pos


def _setup():
    torch.manual_seed(0)
    main = FakeMain()
    lc = LaneKVCache(main, slots=[10, 11], cap=4, pool=3)
    for s in (10, 11):
        main.write_range(s, 0, torch.randn(1, 6, 2, 4), torch.randn(1, 6, 2, 4))
    assert lc.open(1, 3) and lc.open(2, 4)
    side = [(1, 3), (1, 4), (2, 4)]
    for lane, pos in side:
        lc.note(lane, pos)
    for s in (10, 11):
        idx = torch.full((3,), lc.slot_index[s])
        lc.write_rows(idx, side, torch.randn(3, 2, 4), torch.randn(3, 2, 4))
    return main, lc


def _reference(main, lc, lane, pos, slot, q_row):
    km, vm = main.gather(slot, (pos if lane == 0 else lc.fork[lane] - 1))
    ks, vs = [km[0]], [vm[0]]
    if lane:
        n = pos - lc.fork[lane] + 1
        kside, vside = lc.side(lane, slot, n)
        ks.append(kside[0])
        vs.append(vside[0])
    k, v = torch.cat(ks), torch.cat(vs)  # (L, H, D)
    scores = torch.einsum("hd,lhd->hl", q_row, k) / q_row.shape[-1] ** 0.5
    return torch.einsum("hl,lhd->hd", scores.softmax(-1), v)


def test_attend_matches_per_row_reference():
    main, lc = _setup()
    rows = [(0, 5, 7), (1, 4, 7), (2, 4, 7), (0, 4, 8), (1, 3, 8)]  # (lane, pos, step key)
    slot_of = {7: 10, 8: 11}.get
    layout = Layout(lc, rows, torch.device("cpu"))
    q = torch.randn(1, len(rows), 2, 4)
    out = lc.attend(layout, q, slot_of)
    for i, (lane, pos, key) in enumerate(rows):
        ref = _reference(main, lc, lane, pos, slot_of(key), q[0, i])
        torch.testing.assert_close(out[0, i], ref)


def test_promote_copies_side_buffers_into_main():
    main, lc = _setup()
    expected = lc.side(1, 10, 2)[0][0].clone()
    lc.promote(1)
    assert main.dropped == 3
    torch.testing.assert_close(main.K[10][0, 3:5], expected)
    assert 1 not in lc.fork and len(lc.free) == 2


def test_pool_exhaustion_refuses_new_lane():
    _, lc = _setup()
    assert lc.open(3, 5)
    assert not lc.open(4, 5)


def test_spread_subset_covers_category_evenly():
    rows = [{"question_id": q, "category": "a"} for q in range(10)]
    rows += [{"question_id": 100 + q, "category": "b"} for q in range(3)]
    picked = spread_subset(rows, 2)
    assert [r["question_id"] for r in picked] == [0, 5, 100, 101]
