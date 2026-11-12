"""Request-local, event-addressed sampling of an ancestor-closed candidate tree."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


class DraftTreeSampler:
    """Reconstruct the stream from identity and next event; never touch global RNGs."""

    def __init__(self, seed, model, request_id, event=0):
        if request_id is None or not isinstance(request_id, (str, int)):
            raise ValueError("tree sampling requires a stable string/integer request ID")
        if not model or type(event) is not int or event < 0:
            raise ValueError("tree sampling requires model identity and nonnegative event")
        self.identity = json.dumps([seed, model, request_id], ensure_ascii=False).encode()
        self.event = event

    def generator(self):
        key = hashlib.sha256(self.identity + b"/" + str(self.event).encode()).digest()
        self.event += 1
        return np.random.Generator(np.random.PCG64(int.from_bytes(key[:16], "little")))


def sample_candidates(pool, root, budget, coefficient, rng, *, length_conditioned=False):
    """Exactly budget iid draws, with whole-path admission and no refill.

    Imported lazily to keep Candidate and the legacy selector in their original
    module. Sorting uses logical token paths only. Raw probabilities are retained
    for legacy ranking; stable log values recover underflowed model probabilities.
    """
    from .loop_budget_decode import _score, select_candidates, suffix_score

    logs = {root: 0.0}
    paths = sorted((p for p in pool if p != root), key=lambda p: (len(p), p))
    pool[root].p = pool[root].e = 1.0
    for path in paths:
        if path[:len(root)] != root or path[:-1] not in logs:
            raise ValueError("candidate tree must be rooted and ancestor closed")
        c = pool[path]
        q = _score(c.raw_score)
        if c.score_source == "suffix":
            length = c.suffix_match_len if length_conditioned else None
            if length_conditioned and (type(length) is not int or length < 0):
                raise ValueError("length scoring requires nonnegative match length")
            c.e = suffix_score(q, coefficient, length)
            edge = math.log(c.e) if c.e else -math.inf
        elif c.score_source == "model":
            c.e = q
            edge = c.raw_log_score
            if edge is None:
                edge = math.log(q) if q else -math.inf
            if math.isnan(edge) or edge > 0:
                raise ValueError("invalid model log probability")
            if q > 0 and not math.isfinite(edge):
                raise ValueError("positive probability has nonfinite log probability")
            # Match the scoring arm's softmax values exactly where representable;
            # only underflowed edges need the stable value recovered from logits.
            if q > 0:
                edge = math.log(q)
        else:
            raise ValueError("unknown candidate score source")
        logs[path] = logs[path[:-1]] + edge
        c.p = pool[path[:-1]].p * c.e
    info = dict(draws=0, unique_draws=0, duplicates=0, ancestor_additions=0,
                budget_skips=0, all_zero_fallbacks=0, candidates=len(paths),
                closure_nodes=0, retained=0, empty_events=int(not paths or budget == 0))
    if not paths or budget == 0:
        return {root: pool[root]}, info
    maximum = max(logs[p] for p in paths)
    if maximum == -math.inf:
        selected = select_candidates(pool, root, budget, coefficient,
                                     length_conditioned=length_conditioned)
        info.update(all_zero_fallbacks=1, retained=len(selected)-1)
        return selected, info
    weights = np.asarray([math.exp(logs[p] - maximum) for p in paths], dtype=np.float64)
    weights /= weights.sum()
    draws = [paths[i] for i in rng.choice(len(paths), size=budget, replace=True, p=weights)]
    selected, closure = {root}, {root}
    for path in draws:
        missing = []
        cursor = path
        while cursor not in selected:
            missing.append(cursor)
            cursor = cursor[:-1]
        cursor = path
        while cursor not in closure:
            closure.add(cursor)
            cursor = cursor[:-1]
        if len(selected) - 1 + len(missing) > budget:
            info["budget_skips"] += 1
            continue
        info["ancestor_additions"] += max(0, len(missing) - 1)
        selected.update(missing)
    info.update(draws=budget, unique_draws=len(set(draws)), duplicates=budget-len(set(draws)),
                closure_nodes=len(closure)-1, retained=len(selected)-1)
    return {p: pool[p] for p in sorted(selected, key=lambda p: (len(p), p))}, info
