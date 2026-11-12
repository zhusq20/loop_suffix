"""Paper-based retained-tree scheduler on the frozen loop execution interface."""
from __future__ import annotations

from dataclasses import dataclass
import time

from llm_memory_editability.loop_budget_decode import RecyclingTree
from llm_memory_editability.loop_decode_steps import ModelCall

from .sampling import Proposal, probabilities, draw, second_proposal, verify


@dataclass(frozen=True)
class LoopAheadConfig:
    total: int = 4
    first: int = 1
    second: int = 2
    future: int = 3
    retain: int = 32
    execute: int = 16
    guard: int = 2
    stride: int = 1
    family: str = 'ouro'
    temperature: float = 0.
    top_p: float = 1.

    def __post_init__(self):
        assert 0 < self.first < self.second < self.total
        assert 1 <= self.execute <= self.retain and 0 <= self.future <= 7
        assert self.guard >= 0 and self.stride >= 1
        assert all(d % self.stride == 0 for d in (self.first, self.second, self.total))


def conflicted(tree, node, sampling):
    while node.parent >= 0:
        parent = tree.nodes[node.parent]
        feedback = getattr(parent, 'native_tokens', [])
        if feedback and node.token not in (feedback if sampling else feedback[-1:]):
            return True
        node = parent
    return False


def native_path(tree, node):
    while node.parent >= 0:
        if not (node.origins & {'first', 'second', 'root'}):
            return False
        node = tree.nodes[node.parent]
    return True


def execution_nodes(tree, nodes, cfg, quality=None):
    """Every ancestor's next-loop KV must already exist or be written this call."""
    def key(n):
        q = 0. if quality is None else float(quality(n))
        if cfg.family == 'huginn':
            return (not native_path(tree, n), n.depth, -q, n.created)
        return (n.depth, -q, n.created)

    selected, identities = [], set()
    for node in sorted(nodes, key=key):
        if node.step >= cfg.total:
            continue
        if node.depth > cfg.guard and conflicted(tree, node, cfg.temperature > 0):
            continue
        parent_id, ready = node.parent, True
        while parent_id >= 0:
            parent = tree.nodes[parent_id]
            if parent.step < node.step or (parent.step == node.step and parent.idx not in identities):
                ready = False
                break
            parent_id = parent.parent
        if ready:
            selected.append(node)
            identities.add(node.idx)
        if len(selected) == cfg.execute:
            break
    return selected


def decode_steps(backbone, prompt, cfg, max_new_tokens, stop, *, generator, admission=None,
                 quality=None, observer=None):
    if not prompt or max_new_tokens < 1:
        raise ValueError('Expected nonempty prompt and positive output budget')
    tree = RecyclingTree(cfg.retain)
    pool, serial = {}, 0
    stats = dict(tokens=0, steps=0, calls=0, rows=0, max_rows=0, readout_rows=0,
        nodes=0, peak_live_nodes=0, lad_calls=0, lad_tokens=0, first_proposals=0,
        second_proposals=0, accepted_first=0, accepted_second=0, fallback_tokens=0,
        cascade_commits=0, native_budget_rejections=0, retention_evictions=0,
        suspended_node_cycles=0, implementation='paper_reproduction_same_engine',
        retention_budget=cfg.retain, execution_budget=cfg.execute,
        sampling_calibrated=cfg.temperature == 0 or cfg.family != 'ouro' or admission is not None)
    yield ModelCall('synchronize')
    started = time.perf_counter()
    out = [int((yield ModelCall('prefill', (list(prompt), cfg.first))))]
    prefilled = time.perf_counter()
    tree.base_pos = backbone.length
    root_path = tuple(out)

    def release(node):
        tree.release(node)
        del pool[node.path]

    def add(path, parent, source):
        nonlocal serial
        if len(path) >= max_new_tokens:
            return None
        if path in pool:
            pool[path].origins.add(source)
            return pool[path]
        if len(pool) >= cfg.retain:
            if source == 'lad':
                return None
            # Explicit local tie-break: native proposals may evict the farthest
            # LAD-only or conflicting leaf, never the root or a live ancestor.
            candidates = [n for n in pool.values() if n.path != root_path and n is not parent and not n.children
                          and (n.origins == {'lad'} or conflicted(tree, n, cfg.temperature > 0))]
            if not candidates:
                stats['native_budget_rejections'] += 1
                return None
            victim = max(candidates, key=lambda n: (n.depth, n.origins == {'lad'}, n.created))
            release(victim)
            stats['retention_evictions'] += 1
        node = tree.add(path[-1], parent, stats['calls'], source)
        if node is None:
            raise RuntimeError('Physical tree capacity diverged from retained-node count')
        node.path, node.created, node.origins = path, serial, {source}
        node.native_tokens, node.proposals = [], []
        node.shallow_state, node.full_logits = None, None
        serial += 1
        pool[path] = node
        stats['nodes'] += 1
        stats['peak_live_nodes'] = max(stats['peak_live_nodes'], len(pool))
        return node

    if len(out) < max_new_tokens and out[-1] not in stop:
        add(root_path, None, 'root')
    pending_anchors = []
    while len(out) < max_new_tokens and out[-1] not in stop:
        fresh = tree.take_fresh()
        if fresh:
            yield ModelCall('embed', (tree, fresh))
        selected = execution_nodes(tree, list(pool.values()), cfg, quality)
        root = pool[root_path]
        if root.step < cfg.total and root not in selected:
            raise RuntimeError('Root starvation or unsatisfied prefix cache dependency')
        if not selected:
            raise RuntimeError('No executable node before root completion')
        stats['suspended_node_cycles'] += sum(n.depth > cfg.guard and n.step < cfg.total
            and conflicted(tree, n, cfg.temperature > 0) for n in pool.values())
        if observer:
            observer('advance', tree=tree, selected=selected, pool=pool)
        for _ in range(cfg.stride):
            yield ModelCall('advance', (tree, selected))
            for node in selected:
                node.step += 1
            stats['calls'] += 1
            stats['rows'] += len(selected)
        stats['max_rows'] = max(stats['max_rows'], len(selected))
        groups = [(d, [n for n in selected if n.step == d]) for d in (cfg.first, cfg.second)]
        full = [n for n in selected if n.step == cfg.total]
        reads, finals = yield ModelCall('lad_readout', (tree, groups, full))
        stats['readout_rows'] += len(reads) + len(finals)
        for node in full:
            node.full_logits = finals[node.idx]
        for depth, group in groups:
            for node in group:
                if not node.alive:
                    continue
                _, logits = reads[node.idx]
                proposal = None
                if cfg.temperature > 0:
                    q = probabilities(logits, cfg.temperature, cfg.top_p)
                    if depth == cfg.first:
                        proposal = Proposal(draw(q, generator), q, depth)
                    else:
                        proposal = second_proposal(node.proposals[0], q, depth, generator)
                    if proposal is None:
                        continue
                    node.proposals.append(proposal)
                    token = proposal.token
                else:
                    token = int(logits.argmax(-1))
                    if depth == cfg.second and token == node.native_tokens[0]:
                        continue
                node.native_tokens.append(token)
                source = 'first' if depth == cfg.first else 'second'
                stats[source + '_proposals'] += 1
                child = add(node.path + (token,), node, source)
                if observer:
                    observer('native', path=node.path+(token,), root_path=root_path,
                             distance=node.depth+1, source=source, depth=depth)
                if depth == cfg.first:
                    pending_anchors.append(node)

        for node in full:
            if not node.alive:
                continue
            if cfg.temperature > 0:
                node.verified_token, node.verified_kind = verify(
                    probabilities(node.full_logits, cfg.temperature, cfg.top_p), node.proposals, generator)
            else:
                node.verified_token = int(node.full_logits.argmax(-1))
                node.verified_kind = next((i for i,v in enumerate(node.native_tokens) if v == node.verified_token), None)

        commits = 0
        while len(out) < max_new_tokens and out[-1] not in stop and pool[root_path].step == cfg.total:
            root = pool[root_path]
            if root.full_logits is None:
                raise RuntimeError('Full-depth node has no verified distribution')
            token, accepted = root.verified_token, root.verified_kind
            counter = 'accepted_first' if accepted == 0 else 'accepted_second' if accepted == 1 else 'fallback_tokens'
            stats[counter] += 1
            stats['tokens'] += 1
            stats['steps'] += 1
            out.append(token)
            if observer:
                observer('commit', position=len(out)-1, token=token, logits=root.full_logits,
                         path=root.path, accepted_proposal=accepted)
            next_path = root_path + (token,)
            selected_root = pool.get(next_path)
            yield ModelCall('lad_commit', (tree, root, token))
            # Descendants are released before parents to keep child mappings clean.
            discarded = [n for p, n in pool.items() if p[:len(next_path)] != next_path]
            for node in sorted(discarded, key=lambda n: -n.depth):
                release(node)
            tree.base_pos += 1
            for node in pool.values():
                node.depth -= 1
            if selected_root is not None:
                selected_root.parent = -1
            root_path = next_path
            commits += 1
            if len(out) < max_new_tokens and token not in stop and selected_root is None:
                add(root_path, None, 'root')
        stats['cascade_commits'] += max(0, commits - 1)
        # Rerooting can free retention slots. Reconsider previously recorded
        # native children without resampling their proposal distributions.
        if len(out) < max_new_tokens and out[-1] not in stop:
            for parent in sorted(list(pool.values()), key=lambda n: (n.depth, n.created)):
                if parent.alive:
                    for i, token in enumerate(parent.native_tokens):
                        add(parent.path + (token,), parent, 'first' if i == 0 else 'second')
        # At most one LAD call per scheduling stride. An anchor is reconsidered
        # only within the stride in which its parent reached d1.
        if len(out) < max_new_tokens and out[-1] not in stop:
            eligible = []
            for parent in pending_anchors:
                if parent.alive:
                    anchor = pool.get(parent.path + (parent.native_tokens[0],))
                    if anchor is not None and not anchor.children:
                        eligible.append((parent, anchor))
            pending_anchors.clear()
            if eligible and len(pool) < cfg.retain:
                parent, anchor = min(eligible, key=lambda pair: (pair[1].depth, pair[1].created))
                cap = min(cfg.future, cfg.retain - len(pool), max_new_tokens - len(anchor.path) - 1)
                if cfg.family == 'huginn':
                    cap = min(cap, max(0, cfg.execute - len(selected)))
                if admission is not None:
                    cap = min(cap, admission.call_length(anchor.depth, cap))
                if cap > 0:
                    tokens, confidence = yield ModelCall('lad_propose', (tree, parent, anchor, cap))
                    if tokens:
                        stats['lad_calls'] += 1
                    keep = len(tokens) if admission is None else admission.keep_length(anchor.depth, confidence)
                    anchor_distance = anchor.depth
                    for offset, (token, conf) in enumerate(zip(tokens[:keep], confidence[:keep]), 1):
                        child = add(anchor.path + (int(token),), anchor, 'lad')
                        if child is None:
                            break
                        child.lad_confidence = float(conf)
                        child.lad_offset = offset
                        child.lad_root_distance = anchor_distance
                        if observer:
                            observer('lad', path=child.path, root_path=root_path, distance=anchor_distance,
                                     offset=offset, confidence=float(conf))
                        anchor = child
                        stats['lad_tokens'] += 1

    yield ModelCall('synchronize')
    stats.update(prefill_s=prefilled - started, decode_s=time.perf_counter() - prefilled)
    return out, stats
