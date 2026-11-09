"""Read-only output audit and standalone figures for the B16 tree-sampling suite."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random

import loop_tree_sampling_suite as suite


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def output_audit(out):
    result = {}
    for model in suite.MODELS:
        streams = {}
        for variant in suite.VARIANTS:
            job = suite.read(out / 'jobs' / f'{model}-b16-{variant}.json')
            assert job['outcome'] == 'complete'
            path = Path(job['output']) / 'budget-raw-b16-seed0.jsonl'
            streams[variant] = {r['request_id']: r for r in suite.records(path)}
        common = [rid for rid, r in streams['A'].items()
                  if all(streams[v][rid]['tokens'] == r['tokens'] for v in suite.VARIANTS)]
        depth = 32 if model.startswith('huginn') else 4
        comparisons = []
        for x, y in [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D')]:
            differences = []
            for rid, r in streams[x].items():
                other = streams[y][rid]
                position = first_difference(r['tokens'], other['tokens'])
                if position is not None:
                    differences.append(dict(request_id=rid, category=r['category'],
                        first_difference=position, old_length=r['n_tokens'], new_length=other['n_tokens'],
                        old_stop=r['stats']['stop_reason'], new_stop=other['stats']['stop_reason']))
            comparisons.append(dict(comparison=y+'-'+x, differing_requests=len(differences),
                difference_categories=dict(Counter(d['category'] for d in differences)),
                differences=differences))
        result[model] = dict(common_output_requests=len(common), common_ids=common,
            common_work={v: suite.work([stream[rid] for rid in common], depth)[0]
                         for v, stream in streams.items()}, comparisons=comparisons)
    suite.save(out / 'output-audit.json', dict(models=result,
        limitation='BF16 tree-shape and completion-history differences are retained. Token comparison is not a task-quality evaluation.',
        protocol_sha256=suite.digest(out / 'protocol.json'),
        analyzer_sha256=suite.digest(Path(__file__))))
    return result


def plot(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = ['#536878', '#3B82A0', '#D28B39', '#B35161']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, model in enumerate(suite.MODELS):
        table = [next(r for r in data['table'] if r['model'] == model and r['variant'] == v)
                 for v in suite.VARIANTS]
        for col, (metric, title) in enumerate([('tokens_per_s', 'B16 output throughput (token/s)'),
                                             ('rows_per_token', 'Executed rows / decode token')]):
            vals = [r[metric] for r in table]
            bars = axes[row, col].bar(list(suite.VARIANTS), vals, color=colors, width=.65)
            axes[row, col].bar_label(bars, fmt='%.1f', padding=3)
            axes[row, col].set_ylim(0, max(vals)*1.18)
            axes[row, col].set_title(model + '\n' + title)
            axes[row, col].spines[['top', 'right']].set_visible(False)
        for v, color in zip(suite.VARIANTS, colors, strict=True):
            hist = data['distributions'][f'{model}-b16-{v}']['histograms']['retained_drafts']
            denom = sum(hist.values())
            axes[row, 2].plot(np.arange(17), [hist.get(str(i), 0)/denom for i in range(17)],
                              label=v, color=color, marker='.', markersize=3)
        axes[row, 2].set(title=model+'\nFinal retained draft count', xlabel='Draft nodes', ylabel='Fraction of boundaries', xticks=[0, 4, 8, 12, 16])
        axes[row, 2].legend(frameon=False, ncol=4)
        axes[row, 2].spines[['top', 'right']].set_visible(False)
    fig.suptitle('A: legacy + Top-N   B: exact + Top-N   C: legacy + sampling   D: exact + sampling\nSingle seed 0, one timed stream per arm; no uncertainty bars')
    fig.savefig(out/'tree-sampling.pdf')
    fig.savefig(out/'tree-sampling.png', dpi=180)
    plt.close(fig)


def main(out):
    protocol = suite.read(out/'protocol.json')
    for name, expected in protocol['source_sha256'].items():
        assert suite.digest(Path(protocol['source'])/name) == expected, name
    for model, info in protocol['models'].items():
        for kind in ['inputs', 'smoke_inputs']:
            assert suite.digest(info[kind]) == info[kind+'_sha256'], (model, kind)
        order = sorted(row['question_id'] for row in suite.records(info['inputs']))
        random.Random(0).shuffle(order)
        for variant in suite.VARIANTS:
            job = suite.read(out/'jobs'/f'{model}-b16-{variant}.json')
            summary = suite.read(Path(job['output'])/'budget-raw-b16-seed0-summary.json')
            assert summary['request_order'] == order, (model, variant, 'request order')
    assert suite.digest(out/'design-plan.md') == protocol['design_plan_sha256']
    suite.save(out/'provenance-audit.json', dict(passed=True,
        frozen_source_files=len(protocol['source_sha256']), input_files=4,
        protocol_sha256=suite.digest(out/'protocol.json'),
        reporter_sha256=suite.digest(Path(__file__))))
    suite.analyze(out)
    data = suite.read(out/'analysis.json')
    assert len(data['table']) == 8
    common = output_audit(out)
    plot(out, data)
    sample_effects = [r for r in data['paired'] if r['comparison'] in ['C-A', 'D-B']]
    if all(r['throughput_change_pct'] < 0 for r in sample_effects):
        report = (out/'report.md').read_text()
        lead = ('**结论：本轮两模型、两种评分下，树采样均未获得净加速。** '
                '虽然计算行数减少，但γ下降、loop steps/token增加，且选择/采样引入额外开销。'
                '本轮证据不支持用该采样规则替换确定性Top-N；按计划未触发B1延迟补测。\n\n')
        at = report.index('| 模型 | 臂 |')
        (out/'report.md').write_text(report[:at] + lead + report[at:])
    lines = ['', '## 代价分解与输出审计', '',
             '| 模型 | 臂 | loop/token | readout rows/token | 选择CPU秒 | 其中采样秒 | suffix查询数 | 预算丢弃rows | 错路径丢弃rows | 已算节点重建 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['table']:
        lines.append(f"| {r['model']} | {r['variant']} | {r['loop_steps_per_token']:.3f} | {r['readout_rows_per_token']:.3f} | {r['selection_s']:.2f} | {r['sampling_s']:.2f} | {r['speculate_calls']} | {r['wasted_budget_rows']} | {r['wasted_rejected_rows']} | {r['computed_nodes_rebuilt']} |")
    lines += ['', '选择时间包含采样；这些CPU函数计时仅用于开销归因，不替代同步端到端吞吐。推进行数是node-loop计数，不是FLOPs。', '',
              '| 模型 | 采样臂 | 平均不同抽中节点 | 重复率 | 平均闭包大小 | 平均最终草稿 | 平均保留深度 | 超预算跳过 | 全零回退 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in data['table']:
        if r['variant'] in ['C', 'D']:
            lines.append(f"| {r['model']} | {r['variant']} | {r['mean_sample_unique']:.2f} | {100*r['duplicate_rate']:.1f}% | {r['mean_sample_closure']:.2f} | {r['mean_sample_retained']:.2f} | {r['mean_retained_depth']:.2f} | {r['sample_budget_skips']} | {r['sample_all_zero_fallbacks']} |")
    lines += ['', '闭包大小指全部抽样结果的祖先并集，尚未应用预算限制；最终草稿数已应用预算跳过。事件平均包含空候选事件，重复率以实际抽取次数为分母；空候选计数见CSV。平均保留深度按最终边界中的节点出现次数加权。', '',
              '| 模型 | 四臂共同输出请求 | A rows/token | B rows/token | C rows/token | D rows/token |',
              '|---|---:|---:|---:|---:|---:|']
    for model, result in common.items():
        values = [result['common_work'][v]['rows_per_token'] for v in suite.VARIANTS]
        lines.append(f"| {model} | {result['common_output_requests']}/480 | " + ' | '.join(f'{v:.2f}' if v is not None else 'N/A' for v in values) + ' |')
    lines += ['', '共同输出子集属于事后条件诊断；当前请求输出相同不保证各臂此前的完成历史相同。逐请求首次分歧位置见output-audit.json。', '',
              '| 模型 | 臂 | 平均未完成节点 | 平均执行行数 | 零分执行行数 | 停止token结束 | 达到长度上限 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in data['table']:
        stops = data['distributions'][f"{r['model']}-b16-{r['variant']}"]['stop_reasons']
        lines.append(f"| {r['model']} | {r['variant']} | {r['mean_unfinished_nodes']:.2f} | {r['mean_execution_rows']:.2f} | {r['zero_score_rows']} | {stops.get('stop_token', 0)} | {stops.get('max_new_tokens', 0)} |")
    lines += ['', '未完成节点在每请求最终选择边界记录，包含根；执行行数按每请求推进tick记录。零分行数使用当前浮点路径分数，完整直方图见analysis.json。', '',
              '![吞吐、计算行数与最终草稿分布](tree-sampling.png)', '']
    with (out/'report.md').open('a') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
