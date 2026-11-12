"""CPU distribution check that must exercise retention eviction and cascades.

Uses the unchanged real scheduler with a prefix-conditioned three-token oracle.
The longer cap and non-EOS LAD paths address branches absent from the earlier
cap5 integration check. Synthetic frequencies are not benchmark measurements.
"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

import check_stochastic_decode as tiny
from loopahead.admission import AdmissionPolicy
from loopahead.decode import LoopAheadConfig
from packed_dflash_evidence import save, sha
from provenance import freeze_sources

BASE = Path(__file__).resolve().parent
CAP = 12
ALPHA = 1e-6
COUNTERS = ['lad_calls', 'lad_tokens', 'first_proposals', 'second_proposals',
            'accepted_first', 'accepted_second', 'fallback_tokens',
            'cascade_commits', 'native_budget_rejections', 'retention_evictions']


class BranchingOracle(tiny.StochasticTreeOracle):
    def execute(self, call):
        if call.name == 'lad_propose':
            _, _, anchor, cap = call.args
            # Deterministic non-EOS paths can remain live across several
            # commits, unlike the earlier deliberately all-EOS fixture.
            start = (sum(anchor.path) + len(anchor.path)) % 2
            return [(start + i) % 2 for i in range(cap)], [.95] * cap
        return super().execute(call)


def events(sequence):
    yield 'full/' + ','.join(map(str, sequence))
    for length in range(1, min(4, len(sequence)) + 1):
        yield 'prefix/' + ','.join(map(str, sequence[:length]))
    for position in range(CAP):
        token = sequence[position] if position < len(sequence) else -1
        yield f'position/{position}/{token}'
    yield f'length/{len(sequence)}'


def event_distribution(sequences):
    result = Counter()
    for sequence, weight in sequences.items():
        for key in events(sequence):
            result[key] += weight
    return result


def distribution_check(counts, expected_sequences, samples, families):
    assert sum(counts.values()) == samples and set(counts) <= set(expected_sequences)
    expected = event_distribution(expected_sequences)
    observed = event_distribution(counts)
    assert set(observed) <= set(expected)
    logterm = math.log(2 * len(expected) * families / ALPHA)
    rows = []
    for key, probability in sorted(expected.items()):
        bound = (math.sqrt(2 * probability * (1 - probability) * logterm / samples)
                 + 2 * logterm / (3 * samples))
        error = abs(observed[key] / samples - probability)
        rows.append(dict(event=key, expected_probability=probability,
                         count=observed[key], absolute_error=error,
                         bound=bound, passed=error <= bound))
    return dict(passed=all(row['passed'] for row in rows), events=len(rows),
                max_absolute_error=max(row['absolute_error'] for row in rows),
                violations=[row for row in rows if not row['passed']], details=rows)


def measured_path_coverage():
    csv_path = BASE / 'combined-results/results.csv'
    before = sha(csv_path)
    rows = list(csv.DictReader(csv_path.open()))
    result = {}
    for row in rows:
        if row['method'] != 'loopahead' or row['temperature'] != '1':
            continue
        path = Path(row['method_run']) / 'requests.jsonl'
        digest, counters = sha(path), Counter()
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                counters['requests'] += 1
                for key in ['retention_evictions', 'cascade_commits']:
                    counters[key] += record['stats'][key]
                    counters[key + '_requests'] += record['stats'][key] > 0
        assert sha(path) == digest
        result[row['model'] + '/' + row['benchmark']] = dict(
            requests_path=str(path), requests_sha256=digest, counters=dict(counters))
    assert len(result) == 18 and sha(csv_path) == before
    return dict(results_sha256=before, cells=result)


def run(out, samples):
    assert samples >= 1 and not out.exists()
    torch.set_num_threads(1)
    tiny.CAP = CAP  # Fixture state only; no source or production config changes.
    out.mkdir(parents=True)
    sources = freeze_sources(out, [Path(__file__), BASE / 'check_stochastic_decode.py',
        BASE / 'loopahead/decode.py', BASE / 'loopahead/sampling.py',
        BASE / 'loopahead/admission.py', BASE / 'dflash_impl/decode.py',
        BASE / 'packed_dflash_evidence.py', BASE / 'provenance.py'])
    engine = {name: sha(tiny.ENGINE / 'llm_memory_editability' / name) for name in
              ['loop_budget_decode.py', 'loop_prob_tree.py', 'loop_decode_steps.py']}
    policy = AdmissionPolicy(dict(split='development_only',
        costs=dict(t_save_s=1., t_head_s=.55, t_node_s=.1),
        quality=dict(offset={}, distance_offset={}, distance_offset_confidence={}, native_distance={})))
    configurations = [
        ('ouro-default', LoopAheadConfig(temperature=1.), policy),
        ('ouro-retain8', LoopAheadConfig(temperature=1., retain=8, execute=4), policy),
        ('huginn-default', LoopAheadConfig(total=32, first=4, second=8, future=2,
            guard=1, stride=2, family='huginn', temperature=1.), None)]
    measured = measured_path_coverage()
    expected = tiny.enumerate_target()
    expected_events = event_distribution(expected)
    save(out / 'plan.json', dict(samples_per_configuration=samples, max_new_tokens=CAP,
        prompt=tiny.PROMPT, eos=tiny.EOS, target=tiny.TARGET, q1=tiny.Q1, q2=tiny.Q2,
        configurations={name: asdict(cfg) for name, cfg, _ in configurations},
        admission_fixture=policy.record, lad='Alternating non-EOS 0/1 from anchor parity',
        seed_base=932800, seed_configuration_stride=1000000,
        familywise_alpha=ALPHA, events_per_configuration=len(expected_events),
        test='Simultaneous Bernstein bounds for complete sequences, prefixes up to 4, positions and lengths',
        require_each_configuration=['retention_evictions > 0', 'cascade_commits > 0'],
        measured_path_coverage=measured, source_sha256=sources, engine_sha256=engine,
        scope='CPU synthetic integration; no training, GPU inference or benchmark reruns'))
    negative = distribution_check(Counter({(tiny.EOS,): samples}), expected,
                                  samples, len(configurations))
    assert not negative['passed'], 'The fixed bound must reject deliberately all-EOS output.'
    started, results = time.time(), {}
    for index, (name, cfg, admission) in enumerate(configurations):
        counts, counters = Counter(), Counter()
        for seed in range(samples):
            generator = torch.Generator().manual_seed(932800 + index * 1000000 + seed)
            tokens, stats = BranchingOracle(cfg, generator, admission).run()
            assert 1 <= len(tokens) <= CAP and (tiny.EOS not in tokens or tokens[-1] == tiny.EOS)
            counts[tuple(tokens)] += 1
            for field in COUNTERS:
                counters[field] += stats[field]
            if (seed + 1) % 1000 == 0:
                save(out / 'progress.json', dict(configuration=name, completed=seed + 1,
                     samples=samples, execution_counters=dict(counters), updated_unix=time.time()))
        result = distribution_check(counts, expected, samples, len(configurations))
        result.update(execution_counters=dict(counters),
            coverage_passed=counters['retention_evictions'] > 0 and counters['cascade_commits'] > 0,
            sequence_counts=[dict(sequence=list(seq), count=count) for seq, count in sorted(counts.items())])
        results[name] = result
        save(out / (name + '.json'), result)
        print(name, dict(distribution_passed=result['passed'], coverage_passed=result['coverage_passed'],
                         counters=dict(counters), violations=len(result['violations'])), flush=True)
    assert not torch.cuda.is_initialized()
    assert all(sha(BASE / name) == digest for name, digest in sources.items())
    assert all(sha(tiny.ENGINE / 'llm_memory_editability' / name) == digest for name, digest in engine.items())
    assert sha(BASE / 'combined-results/results.csv') == measured['results_sha256']
    passed = all(row['passed'] and row['coverage_passed'] for row in results.values())
    save(out / 'report.json', dict(status='passed' if passed else 'failed',
        samples_per_configuration=samples, configurations=results,
        source_sha256=sources, engine_sha256=engine, measured_path_coverage=measured,
        negative_control_rejected=True, cuda_initialized=False, new_gpu_runs=0,
        started_unix=started, finished_unix=time.time(), torch=str(torch.__version__),
        limitations=['Three-token synthetic target and cap12, not real-model quality or speed.',
            'Finite Monte Carlo with a predeclared bound; not a universal distribution proof.',
            'Oracle replaces target tensor operations; real-model numerical evidence remains separate.']))
    if not passed:
        raise AssertionError('Distribution or required scheduler-path coverage failed; preserve outputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32000)
    args = parser.parse_args()
    run(args.out.resolve(), args.samples)
