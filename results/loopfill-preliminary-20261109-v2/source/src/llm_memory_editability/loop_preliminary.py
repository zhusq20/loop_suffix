"""Paired, causal candidate-coverage probes for the preliminary figure.

No model execution or scheduling-speed claims belong in these statistics.
All three arms share one read-only suffix snapshot at each reference position.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ProbeConfig:
    node_budget: int = 16
    query_budget: int = 4
    shallow_depth: int = 1
    spec_factor: float = 1.0
    min_token_prob: float = 0.1
    max_tree_depth: int = 64
    max_cached_requests: int = 100000

    def __post_init__(self):
        if self.query_budget < 1 or self.node_budget < self.query_budget + 1:
            raise ValueError("budget must preserve the root query plus one shallow candidate")
        if self.shallow_depth != 1:
            raise ValueError("the frozen preliminary probe uses depth 1")


def attach(paths: set[tuple[int, ...]], draft, base: tuple[int, ...], budget: int):
    """Deduplicate by complete token path; never evict existing candidates."""
    created = []
    for i, (token, parent) in enumerate(zip(draft.token_ids, draft.parents, strict=True)):
        if not -1 <= parent < i:
            raise ValueError("suffix parents must precede their children")
        prefix = base if parent == -1 else created[parent]
        path = None if prefix is None else prefix + (int(token),)
        if path is not None and (path in paths or len(paths) < budget):
            paths.add(path)
            created.append(path)
        else:
            created.append(None)


def correct_length(paths, remaining):
    path = ()
    length = 0
    for token in remaining:
        path += (int(token),)
        if path not in paths:
            break
        length += 1
    return length


def probe_position(cache, req_id, context, shallow_token, remaining, config=ProbeConfig()):
    def query(ctx, room):
        return cache.speculate(
            req_id, np.asarray(ctx, dtype=np.int32),
            max_spec_tokens=min(config.query_budget, room),
            max_spec_factor=config.spec_factor, max_spec_offset=0.0,
            min_token_prob=config.min_token_prob, use_tree_spec=True,
        )

    root = query(context, config.query_budget)
    if len(root.token_ids) > config.query_budget:
        raise ValueError("suffix query exceeded its node cap")
    suffix = set()
    attach(suffix, root, (), config.node_budget)
    shallow = suffix | {(int(shallow_token),)}
    requery = set(shallow)
    room = config.node_budget - len(requery)
    extra = None
    if room:
        extra = query(list(context) + [int(shallow_token)], room)
        if len(extra.token_ids) > min(config.query_budget, room):
            raise ValueError("suffix requery exceeded its node cap")
        attach(requery, extra, (int(shallow_token),), config.node_budget)
    assert suffix <= shallow <= requery
    assert len(requery) <= config.node_budget
    groups = [suffix, shallow, requery]
    lengths = [correct_length(paths, remaining) for paths in groups]
    assert lengths == sorted(lengths)
    return dict(lengths=lengths, nodes=[len(paths) for paths in groups],
                root_match_len=int(getattr(root, 'match_len', 0)),
                requery_match_len=int(getattr(extra, 'match_len', 0)),
                requery_nodes_added=len(requery) - len(shallow))


def replay_request(cache, req_id, prompt, output, predictions, config=ProbeConfig()):
    """Yield observations before publishing the next confirmed reference token."""
    output = [int(t) for t in output]
    predictions = np.asarray(predictions)
    if predictions.shape != (4, len(output)) or not output:
        raise ValueError("expected four aligned top-1 readouts and a nonempty reference")
    if not np.array_equal(predictions[-1], output):
        raise ValueError("full-depth readouts do not match the AR reference")
    cache.start_request(req_id, np.asarray(prompt, dtype=np.int32))
    cache.add_active_response(req_id, np.asarray(output[:1], dtype=np.int32))
    context = [int(t) for t in prompt] + output[:1]
    for i in range(1, len(output)):
        # Future tokens are used only for scoring, never supplied to the cache.
        result = probe_position(cache, req_id, context, predictions[0, i], output[i:], config)
        hits = (predictions[:3, i] == output[i]).astype(int)
        cumulative = np.maximum.accumulate(hits)
        yield dict(question_id=req_id, output_index=i, target=output[i],
                   predictions=predictions[:, i].astype(int).tolist(),
                   agreement=hits.tolist(), coverage=cumulative.tolist(),
                   gains=np.diff(np.r_[0, cumulative]).tolist(), **result)
        cache.add_active_response(req_id, np.asarray([output[i]], dtype=np.int32))
        context.append(output[i])
    cache.stop_request(req_id)


def request_totals(records):
    n = len(records)
    result = dict(positions=n)
    for key in ('agreement', 'coverage', 'gains', 'lengths', 'nodes'):
        result[key] = np.asarray([r[key] for r in records], dtype=np.int64).reshape(n, 3).sum(0).tolist()
    zero = [r for r in records if r['lengths'][0] == 0]
    result['suffix_zero_positions'] = len(zero)
    result['suffix_zero_lengths'] = np.asarray(
        [r['lengths'] for r in zero], dtype=np.int64).reshape(len(zero), 3).sum(0).tolist()
    return result


def bootstrap_summary(requests, repeats=2000, seed=0):
    """Pooled token ratios with paired request-cluster percentile intervals."""
    if not requests or repeats < 1:
        raise ValueError("nonempty requests and positive bootstrap repeats required")
    n = np.asarray([r['positions'] for r in requests])
    if n.sum() == 0:
        raise ValueError("no eligible decode positions")
    zero_n = np.asarray([r['suffix_zero_positions'] for r in requests])
    sums = {k: np.asarray([r[k] for r in requests], dtype=float) for k in
            ('agreement', 'coverage', 'gains', 'lengths', 'nodes', 'suffix_zero_lengths')}
    sums['length_deltas'] = np.column_stack((sums['lengths'][:, 1] - sums['lengths'][:, 0],
                                           sums['lengths'][:, 2] - sums['lengths'][:, 1],
                                           sums['lengths'][:, 2] - sums['lengths'][:, 0]))
    sums['suffix_zero_delta'] = (sums['suffix_zero_lengths'][:, 2] -
                                 sums['suffix_zero_lengths'][:, 1])[:, None]
    sums['suffix_zero_fraction'] = zero_n[:, None].astype(float)
    # Shared cluster weights preserve all method/depth pairings.
    weights = np.random.default_rng(seed).multinomial(
        len(requests), np.full(len(requests), 1 / len(requests)), size=repeats)
    result = dict(requests=len(requests), positions=int(n.sum()),
                  suffix_zero_positions=int(zero_n.sum()), bootstrap_repeats=repeats,
                  bootstrap_seed=seed, config=asdict(ProbeConfig()), metrics={})
    for key, values in sums.items():
        denominator = zero_n if key in ('suffix_zero_lengths', 'suffix_zero_delta') else n
        total = denominator.sum()
        boot_n = weights @ denominator
        valid = boot_n > 0
        if total == 0:
            result['metrics'][key] = dict(mean=None, ci95=None, valid_bootstrap=0)
            continue
        sampled = (weights[valid] @ values) / boot_n[valid, None]
        result['metrics'][key] = dict(mean=(values.sum(0) / total).tolist(),
            ci95=np.quantile(sampled, [0.025, 0.975], axis=0).T.tolist(),
            valid_bootstrap=int(valid.sum()))
    return result
