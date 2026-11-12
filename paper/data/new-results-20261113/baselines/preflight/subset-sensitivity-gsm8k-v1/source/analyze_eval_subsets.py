"""Paired finite-pool subset sensitivity; never changes the formal eval protocol."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time

from collect_results import BASE, ROOT, read_run, sha


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def measures(method, ar, indices):
    def totals(run):
        official = run['stack'] == 'official'
        depth = 32 if run['meta']['model'] == 'huginn-3.5b' else 4
        rows = [run['rows'][i]['stats'] for i in indices]
        tokens = sum(r['decode_tokens' if official else 'tokens'] for r in rows)
        seconds = sum(r['decode_s'] for r in rows)
        loops = sum(r['target_verifications'] * depth if official else r['calls'] for r in rows)
        assert tokens > 0 and seconds > 0 and loops > 0
        return tokens, seconds, depth * tokens / loops

    mn, ms, gamma = totals(method)
    an, ars, _ = totals(ar)
    return dict(eta_latency=ars / ms, eta_throughput=(mn / ms) / (an / ars), gamma=gamma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default='gsm8k')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 200])
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.repeats >= 100 and len(set(args.sizes)) == len(args.sizes)
    assert not args.out.exists(), 'Use a fresh output directory; keep prior analyses.'
    csv_path = BASE / 'results.csv'
    csv_sha = sha(csv_path)
    cells = [r for r in csv.DictReader(csv_path.open())
             if r['benchmark'] == args.benchmark and float(r['temperature']) == 0]
    assert cells, 'No completed greedy cells for this benchmark.'
    protocol_path = ROOT / 'results/loopfill-latency512-20261107/protocol.json'
    protocol = json.loads(protocol_path.read_text())
    count = protocol['inputs'][args.benchmark]['count']
    assert all(0 < n < count for n in args.sizes), 'Subsets must be smaller than the frozen pool.'
    draws = {}
    draw_hashes = {}
    for n in args.sizes:
        rng = random.Random(f'{args.seed}:{args.benchmark}:{n}')
        draws[n] = [sorted(rng.sample(range(count), n)) for _ in range(args.repeats)]
        draw_hashes[n] = hashlib.sha256(json.dumps(draws[n], separators=(',', ':')).encode()).hexdigest()

    report = dict(
        status='complete_offline_sensitivity_only', created_unix=time.time(),
        benchmark=args.benchmark, frozen_pool_count=count, subset_sizes=args.sizes,
        repeats=args.repeats, seed=args.seed, python_version=platform.python_version(),
        sampling='uniform without replacement within each draw; repeated draws are independent; '
                 'same indices for every method/model and its paired AR',
        draw_indices_sha256=draw_hashes, results_csv_sha256=csv_sha,
        protocol_sha256=sha(protocol_path), analyzer_sha256=sha(Path(__file__)),
        collector_sha256=sha(BASE / 'collect_results.py'),
        limitations=[
            'Uses already measured per-question times and loop/token counts; no new GPU timing.',
            'Percentiles describe subset composition within this frozen finite pool, not a confidence '
            'interval for the full source dataset or repeated timing noise.',
            'Does not assess unseen questions, change the formal question set, or count as a new benchmark cell.',
            'Uniform sampling does not preserve task/category strata; no smaller formal protocol is adopted.',
            'Archived AR timing drift and BF16 output differences remain as declared for the source runs.'
        ], cells=[])
    expected_ids = None
    for cell in cells:
        method = read_run(Path(cell['method_run']), protocol)
        ar = read_run(Path(cell['ar_run']), protocol)
        ids = [r['source_id'] for r in method['rows']]
        assert ids == [r['source_id'] for r in ar['rows']]
        assert expected_ids is None or expected_ids == ids
        expected_ids = ids
        full = measures(method, ar, range(count))
        for name, value in full.items():
            assert math.isclose(value, float(cell[name]), rel_tol=1e-12, abs_tol=1e-12), (name, value)
        result = dict(method=cell['method'], model=cell['model'], temperature=0, full=full,
                      method_run=cell['method_run'], ar_run=cell['ar_run'],
                      method_journal_sha256=sha(Path(cell['method_run']) / 'requests.jsonl'),
                      ar_journal_sha256=sha(Path(cell['ar_run']) / 'requests.jsonl'), subsets=[])
        for n, samples in draws.items():
            values = [measures(method, ar, indices) for indices in samples]
            metrics = {}
            for name, reference in full.items():
                raw = [v[name] for v in values]
                relative = [100 * (value / reference - 1) for value in raw]
                absolute = [abs(value) for value in relative]
                metrics[name] = dict(p025=percentile(raw, .025), median=percentile(raw, .5),
                                     p975=percentile(raw, .975),
                                     relative_error_pct_p025=percentile(relative, .025),
                                     relative_error_pct_p975=percentile(relative, .975),
                                     absolute_relative_error_pct_p95=percentile(absolute, .95),
                                     absolute_relative_error_pct_max=max(absolute))
            result['subsets'].append(dict(count=n, metrics=metrics))
        report['cells'].append(result)
    assert sha(csv_path) == csv_sha, 'Results changed during analysis; rerun with a fresh snapshot.'

    args.out.mkdir(parents=True)
    source = args.out / 'source'
    source.mkdir()
    for path in [Path(__file__), BASE / 'collect_results.py']:
        shutil.copyfile(path, source / path.name)
    (args.out / 'results-snapshot.csv').write_bytes(csv_path.read_bytes())
    (args.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = [f'# {args.benchmark} 子集敏感性（离线分析）', '',
             f'固定池{count}题；每种子集大小重复{args.repeats}次无放回抽样，seed={args.seed}。'
             '每次方法与AR使用相同题目，各模型共用抽样索引。', '',
             '表内为相对完整固定池结果的绝对偏差95分位（百分比），不是置信区间。', '',
             '| 方法 | 模型 | 子集题数 | η_latency偏差P95 | η_throughput偏差P95 | γ偏差P95 |',
             '|---|---|---:|---:|---:|---:|']
    for cell in report['cells']:
        for subset in cell['subsets']:
            errors = [subset['metrics'][name]['absolute_relative_error_pct_p95']
                      for name in ['eta_latency', 'eta_throughput', 'gamma']]
            lines.append(f'| {cell["method"]} | {cell["model"]} | {subset["count"]} | '
                         + ' | '.join(f'{x:.2f}%' for x in errors) + ' |')
    lines += ['', '仅重算已测逐题记录，未增加GPU测评，正式矩阵题集保持不变。'
              '该分析不能推断GSM8K完整1319题与随机500题的差异，也不覆盖重新运行的计时噪声。'
              '历史AR跨日期漂移及BF16输出差异仍保留。完整数值与来源哈希见report.json。']
    (args.out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'report': str(args.out / 'report.md'), 'cells': len(report['cells'])}))


if __name__ == '__main__':
    main()
