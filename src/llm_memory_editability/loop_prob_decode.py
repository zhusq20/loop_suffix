"""Real decoder for the probability-threshold draft tree.

The rules of loop_prob_tree (its replay) on the engine of loop_tree_decode: every block
call advances all live nodes one loop step; a node reaching a readout depth offers its
top-8 tokens, each placed iff p(parent) * e > tau, with e from the calibration tables;
suffix-tree drafts are offered under every placed node; deeper readouts can score
jointly with the depth-d readout and prune; accepted tokens enter the suffix tree at
once and the suffix tree drafts again from the newly accepted node. Every committed
token is the full-depth argmax, so the output equals greedy decoding.

Differences from the replay are only where the replay had no data: off-path nodes get
their real readouts here (the replay used the greedy path's), so in exact arithmetic
the block-call count equals the replay's and the row count is the true one.

Protocol: results/loop-spec-prob-tree-20261031/protocol.md (S4)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from llm_memory_editability.loop_prob_tree import ABSENT, Calibration, ProbConfig, _edge_probs
from llm_memory_editability.loop_tree_decode import DraftTree


@dataclass
class _Info:
    p: float
    e: float
    src: str
    suffix: bool
    locked: bool = False
    reads: dict = field(default_factory=dict)  # depth -> (ids, probs)


@torch.no_grad()
def prob_tree_decode(
    backbone,
    prompt,
    cfg: ProbConfig,
    calib: Calibration,
    max_new_tokens: int,
    stop: set[int],
    cache=None,
    req_id=None,
    *,
    max_nodes: int | None = None,
    flush_every: int = 16,
) -> tuple[list[int], dict]:
    """Greedy decoding of one request; returns the output and per-request statistics."""
    if cfg.legacy or cfg.row_cap is not None or cfg.max_draft is not None:
        raise NotImplementedError("legacy, row caps and K caps run in loop_tree_decode")
    T, d = cfg.depth_total, cfg.draft_depth
    depths = sorted(cfg.readout_depths)
    use_suffix = cfg.use_suffix and cache is not None
    eager = cfg.eager_commit and use_suffix
    kv = getattr(backbone, "kv", None)
    if max_nodes is None:  # the backbone's tree buffers bound the node count
        max_nodes = kv.TK.shape[1] if kv is not None else 8192
    tree = DraftTree(max_nodes, 1 << 30)
    max_pos = getattr(kv, "max_seq", None)
    ctx_limit = getattr(cache, "max_tree_depth", None)
    info: dict[int, _Info] = {}
    stats = dict(tokens=0, steps=0, calls=0, rows=0, max_rows=0, nodes=0, readout_rows=0)
    stats.update(speculate_calls=0, speculate_s=0.0, prune_kills=0, cap_skips=0)
    for src in ("suffix", "fill", "sibling", "deep"):
        stats[f"from_{src}"] = 0

    def speculate(ctx: list[int]):
        t0 = time.perf_counter()
        draft = cache.speculate(
            req_id,
            np.asarray(ctx[-ctx_limit:] if ctx_limit else ctx, dtype=np.int32),
            max_spec_tokens=cfg.suffix_room,
            max_spec_factor=cfg.spec_factor,
            max_spec_offset=0.0,
            min_token_prob=cfg.suffix_min_prob,
            use_tree_spec=True,
        )
        stats["speculate_s"] += time.perf_counter() - t0
        stats["speculate_calls"] += 1
        return draft

    def node(i):
        return tree.nodes[i]

    def child(n, tok):
        j = n.children.get(tok)
        return None if j is None or not node(j).alive else node(j)

    def add(tok, parent, tick, src, e, suffix):
        if max_pos is not None and tree.position(parent) + 1 >= max_pos:
            return None
        c = tree.add(tok, parent, tick, src)
        if c is None:
            stats["cap_skips"] += 1
            return None
        info[c.idx] = _Info(info[parent.idx].p * e, e, src, suffix)
        return c

    def kill(n):
        par = node(n.parent)
        if par.children.get(n.token) == n.idx:
            del par.children[n.token]
        tree.kill(n)

    def reprob(n):
        stack = [n]
        while stack:
            m = stack.pop()
            info[m.idx].p = info[m.parent].p * info[m.idx].e
            stack.extend(node(j) for j in m.children.values() if node(j).alive)

    def rescore(c, e, deep):
        ci = info[c.idx]
        if ci.locked or (deep and not cfg.deep_raise and e > ci.e):
            return
        ci.e = e
        reprob(c)
        if cfg.prune and ci.p <= cfg.tau:
            kill(c)
            stats["prune_kills"] += 1

    def place_suffix(n, tick):
        draft = speculate(ctx + tree.path_tokens(n))
        edge = _edge_probs(draft)
        tau = cfg.tau if cfg.tau_suffix is None else cfg.tau_suffix
        created = []
        for k, (tok, par) in enumerate(zip(draft.token_ids, draft.parents, strict=True)):
            base = n if par < 0 else created[par]
            if base is None or not base.alive:
                created.append(None)
                continue
            tok = int(tok)
            c = child(base, tok)
            if c is not None:
                info[c.idx].suffix = True
                created.append(c)
                continue
            rival = any(node(j).alive for j in base.children.values())
            lead = base.depth - frontier[0].depth
            if cfg.suffix_calibration == "insitu":
                e = calib.suffix_prob_insitu(edge[k], draft.match_len, rival, lead)
            else:
                e = calib.suffix_prob(edge[k], draft.match_len)
            ok = info[base.idx].p * e > tau
            created.append(add(tok, base, tick, "suffix", e, True) if ok else None)

    def readout(n, r, tick):
        ids, probs = info[n.idx].reads[r]
        deep = r > d
        joint = deep and cfg.joint
        rank_d = {t: j for j, t in enumerate(info[n.idx].reads[d][0])} if joint else {}
        tau = cfg.tau if (not deep or cfg.tau_deep is None) else cfg.tau_deep

        def edge_prob(tok, k, q, flag):
            if joint:
                return calib.joint_prob(r, rank_d.get(tok, ABSENT), k, q, flag)
            return calib.shallow_prob(r, k, q, flag)

        seen = set()
        for k, (tok, q) in enumerate(zip(ids, probs, strict=True)):
            seen.add(tok)
            c = child(n, tok)
            if c is not None:
                if cfg.rescore:
                    rescore(c, edge_prob(tok, k, q, info[c.idx].suffix), deep)
            elif k < cfg.top_m and (cfg.deep_expand or not deep):
                e = edge_prob(tok, k, q, False)
                if info[n.idx].p * e > tau:
                    src = "deep" if deep else ("fill" if k == 0 else "sibling")
                    c = add(tok, n, tick, src, e, False)
                    if c is not None and use_suffix:
                        place_suffix(c, tick)
        if not cfg.rescore:
            return
        for j in list(n.children.values()):
            c = node(j)
            if not c.alive or c.token in seen:
                continue
            if joint:
                e = calib.joint_prob(r, rank_d.get(c.token, ABSENT), ABSENT, 0.0, info[j].suffix)
            elif info[j].suffix:
                e = calib.shallow_prob(r, ABSENT, 0.0, True)
            else:
                continue
            rescore(c, e, deep)

    backbone.synchronize()
    t0 = time.perf_counter()
    out = [backbone.prefill([int(t) for t in prompt], d)]
    t1 = time.perf_counter()
    ctx = [int(t) for t in prompt] + out
    if use_suffix:
        cache.start_request(req_id, np.asarray(prompt, dtype=np.int32))
        cache.add_active_response(req_id, np.asarray(out, dtype=np.int32))

    while len(out) < max_new_tokens and out[-1] not in stop:
        tree.reset(backbone.length)
        info.clear()
        root = tree.add(out[-1], None, 0, "root")
        info[root.idx] = _Info(1.0, 1.0, "root", False, locked=True)
        frontier = [root]  # last accepted node; a list so the closures see updates
        if use_suffix:
            place_suffix(root, 0)
        backbone.embed(tree, tree.take_fresh())
        chain, flushed, new, tick = [root], 0, [], 0
        while True:
            tick += 1
            nodes = tree.nodes[tree.lo :] if tree.lo else tree.nodes
            live = sorted((n for n in nodes if n.alive and n.step < T), key=lambda n: n.step)
            backbone.advance(tree, live)
            for n in live:
                n.step += 1
            stats["rows"] += len(live)
            stats["max_rows"] = max(stats["max_rows"], len(live))
            groups = [(r, [n for n in live if n.step == r]) for r in depths]
            fu = [n for n in live if n.step == T]
            reads, fu_top = backbone.readout_top(tree, groups, fu, 8)
            stats["readout_rows"] += sum(len(g) for _, g in groups) + len(fu)
            for (r, g), got in zip(groups, reads, strict=True):
                for n, top in zip(g, got, strict=True):
                    info[n.idx].reads[r] = top
            for n, top in zip(fu, fu_top, strict=True):
                n.full_top = top

            done = False
            while frontier[0].full_top is not None:
                gold = frontier[0].full_top
                new.append(gold)
                if eager:
                    cache.add_active_response(req_id, np.asarray([gold], dtype=np.int32))
                nxt = child(frontier[0], gold)
                if nxt is not None:
                    stats[f"from_{info[nxt.idx].src}"] += 1
                if nxt is None or gold in stop or len(out) + len(new) >= max_new_tokens:
                    done = True
                    break
                for j in list(frontier[0].children.values()):
                    if j != nxt.idx and node(j).alive:
                        kill(node(j))
                frontier[0] = nxt
                chain.append(nxt)
                info[nxt.idx].locked = True
                info[nxt.idx].e = 1.0
                reprob(nxt)
                if eager:
                    place_suffix(nxt, tick)
            if done:
                break
            if flush_every and len(chain) - 1 - flushed >= flush_every:
                backbone.commit(tree, chain[flushed:-1])  # finished, all others descend
                flushed = len(chain) - 1
                tree.lo = frontier[0].idx

            # ancestors first, so their rescoring or pruning precedes their descendants'
            # expansion within one block call (the replay's event order)
            due = sorted(
                ((n, r) for r, g in groups for n in g), key=lambda x: (x[0].depth, x[0].idx)
            )
            for n, r in due:
                if n.alive:
                    readout(n, r, tick)
            fresh = tree.take_fresh()
            if fresh:
                backbone.embed(tree, fresh)

        backbone.commit(tree, chain[flushed:])
        out += new
        ctx += new
        if use_suffix and not eager:
            cache.add_active_response(req_id, np.asarray(new, dtype=np.int32))
        stats["calls"] += tick
        stats["steps"] += 1
        stats["tokens"] += len(new)
        stats["nodes"] += len(tree.nodes)

    backbone.synchronize()
    t2 = time.perf_counter()
    if use_suffix:
        cache.stop_request(req_id)
    stats.update(prefill_s=t1 - t0, decode_s=t2 - t1)
    return out, stats
