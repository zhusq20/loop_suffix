"""Audit and summarize accepted-token origins from frozen Table 2 logs; no inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

from loop_accepted_length import MT_CATEGORIES, TASKS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('suffix', 'fill', 'sibling', 'deep', 'uncovered')
LABELS = ('Suffix', 'First readout: rank-1', 'First readout: rank-2',
          'Second readout', 'No draft match')
COLORS = ('#4477AA', '#EE9944', '#EECB77', '#66AA88', '#BBBBBB')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def counts(row):
    """Partition every emitted token, including terminal tokens and prefill-only cases."""
    s = row['stats']
    values = [s[k] for k in ('tokens', 'covered_tokens', 'verified_parents')]
    values += [s['from_' + k] for k in SOURCES[:-1]]
    if any(type(n) is not int or n < 0 for n in values):
        raise ValueError('Source and token counters must be nonnegative integers')
    emitted = len(row['tokens'])
    if emitted != row['n_tokens'] or emitted != s['tokens'] + 1:
        raise ValueError('Output must contain exactly one prefill token')
    result = Counter({k: s['from_' + k] for k in SOURCES[:-1]})
    covered = sum(result.values())
    if covered != s['covered_tokens'] or covered > s['tokens']:
        raise ValueError('Source counts must partition covered tokens')
    if s['verified_parents'] != s['tokens']:
        raise ValueError('Every decode token must have a verified parent')
    result.update(uncovered=s['tokens'] - covered, requests=1, decode_tokens=s['tokens'],
                  covered_tokens=covered, prefill=1, output_tokens=emitted,
                  prefill_only_requests=int(s['tokens'] == 0))
    return result


def metrics(c):
    result = dict(c)
    result['model'] = c['fill'] + c['sibling'] + c['deep']
    for key in (*SOURCES, 'model'):
        result[key + '_decode_pct'] = 100 * result[key] / c['decode_tokens'] if c['decode_tokens'] else None
        result[key + '_output_pct'] = 100 * result[key] / c['output_tokens'] if c['output_tokens'] else None
    result['prefill_output_pct'] = 100 * c['prefill'] / c['output_tokens'] if c['output_tokens'] else None
    for key in (*SOURCES[:-1], 'model'):
        result[key + '_covered_pct'] = 100 * result[key] / c['covered_tokens'] if c['covered_tokens'] else None
    return result


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def table(rows, detailed=False):
    keys = SOURCES if detailed else ('suffix', 'model', 'uncovered')
    columns = ['后缀', '首次读出第1候选', '首次读出第2候选', '第二次读出', '未命中候选'] if detailed else ['后缀', '浅层模型合计', '未命中候选']
    lines = ['| 模型 | B | Decode tokens | ' + ' | '.join(columns) + ' |',
             '|---|---:|---:|' + '---:|' * len(keys)]
    for r in rows:
        cells = [r['model_name'], str(r['batch']), f"{r['decode_tokens']:,}"]
        cells += [f"{r[k + '_decode_pct']:.2f}%" for k in keys]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def plot(rows, by_task, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    for ax, batch in zip(axes, (1, 16)):
        selected = [r for r in rows if r['batch'] == batch]
        left = [0.] * len(selected)
        for key, label, color in zip(SOURCES, LABELS, COLORS):
            widths = [r[key + '_decode_pct'] for r in selected]
            ax.barh(range(len(selected)), widths, left=left, label=label, color=color)
            for i, width in enumerate(widths):
                if width >= 7:
                    ax.text(left[i] + width / 2, i, f'{width:.1f}', ha='center', va='center', fontsize=8)
            left = [x + w for x, w in zip(left, widths)]
        ax.set_yticks(range(len(selected)), [r['model_name'] for r in selected])
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_title(f'B = {batch}')
        ax.set_xlabel('Share of emitted decode tokens')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .17, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'token-sources.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    order = ['mt_bench', 'translation', 'summarization', 'qa', 'math_reasoning', 'rag']
    names = ['MT-Bench', 'Translation', 'Summary', 'QA', 'Math', 'RAG']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, model in zip(axes, sorted({r['model_name'] for r in rows})):
        index = {r['category']: r for r in by_task if r['model_name'] == model and r['batch'] == 16}
        left = [0.] * len(order)
        for key, label, color in zip(SOURCES, LABELS, COLORS):
            widths = [index[c][key + '_decode_pct'] for c in order]
            ax.barh(range(len(order)), widths, left=left, label=label, color=color)
            left = [x + w for x, w in zip(left, widths)]
        ax.set_yticks(range(len(order)), names)
        ax.set_xlim(0, 100)
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.set_title(model + ' / B16')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'token-sources-by-task.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def analyze(suite, out):
    suite, out = suite.resolve(), out.resolve()
    protocol = read(suite / 'protocol.json')
    jobs = [(f, read(f)) for f in (suite / 'jobs').glob('*-budget-online.json')]
    assert len(jobs) == len(protocol['models']) * len(protocol['batches']) == 15
    overall, by_task, per_request, provenance = [], [], [], []
    checked_sources = {}
    for job_path, job in sorted(jobs, key=lambda pair: (pair[1]['model'], pair[1]['batch'])):
        assert job['outcome'] == 'complete'
        # Select the completed attempt from its job, never glob W&B copies or failed attempts.
        target = suite / 'runs' / job['key'] / f"attempt{job['attempt']}"
        assert (target / 'complete.json').is_file()
        prefix = f"budget-online-b{job['batch']}-seed0"
        journal = target / (prefix + '.jsonl')
        summary = read(target / (prefix + '-summary.json'))
        meta = read(target / 'metadata.json')
        inputs_path = suite / 'inputs' / (job['model'] + '.jsonl')
        assert digest(inputs_path) == meta['questions_sha256']
        inputs = {r['question_id']: r for r in map(json.loads, inputs_path.open())}
        assert len(inputs) == protocol['requests'] == 480
        cfg = meta['configs']['budget-online']
        assert cfg['top_m'] == 2 and len(cfg['readout_depths']) == 2
        assert cfg['readout_depths'][0] == cfg['draft_depth']
        assert cfg['draft_node_budget'] == 16 and cfg['scoring'] == 'online_suffix_i'
        for rel, sha in meta['source_sha256'].items():
            if rel not in checked_sources:
                checked_sources[rel] = digest(suite / 'source' / rel)
            assert checked_sources[rel] == sha
        totals, groups, seen = Counter(), defaultdict(Counter), set()
        base = dict(key=job['key'], model_name=job['model'], batch=job['batch'])
        with journal.open() as stream:
            for line in stream:
                row = json.loads(line)
                rid = row['request_id']
                assert rid not in seen and rid in inputs
                seen.add(rid)
                assert row['category'] == inputs[rid]['category']
                category = 'mt_bench' if row['category'] in MT_CATEGORIES else row['category']
                assert category in TASKS
                c = counts(row)
                totals.update(c)
                groups[category].update(c)
                per_request.append(dict(**base, request_id=rid, category=category, **metrics(c)))
        assert seen == set(inputs)
        assert totals['output_tokens'] == summary['output_tokens'] == job['summary']['output_tokens']
        assert all(group['requests'] == 80 for group in groups.values()) and len(groups) == 6
        overall.append(dict(**base, **metrics(totals)))
        for category, c in sorted(groups.items()):
            by_task.append(dict(**base, category=category, **metrics(c)))
        paths = [job_path, journal, target / 'metadata.json', target / 'complete.json',
                 target / (prefix + '-summary.json'), inputs_path]
        provenance.append(dict(key=job['key'], files={str(p.relative_to(ROOT)): digest(p) for p in paths}))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('overall', overall), ('by-task', by_task), ('per-request', per_request)]:
        write_csv(out / (name + '.csv'), rows)
    audit = dict(passed=True, cells=len(overall), requests=len(per_request),
                 decode_tokens=sum(r['decode_tokens'] for r in overall),
                 output_tokens=sum(r['output_tokens'] for r in overall),
                 formula='sum(source counts) / sum(emitted decode tokens); prefill excluded',
                 attribution='origin of the surviving candidate incarnation at verification',
                 source_fields=list(SOURCES), duplicate_proposals='not counted twice',
                 terminal_tokens='included exactly as stored in generated outputs',
                 overlap='not recoverable from these aggregate logs',
                 new_inference=False, protocol_sha256=digest(suite / 'protocol.json'),
                 analyzer_sha256=digest(Path(__file__)), frozen_sources=checked_sources,
                 inputs=provenance)
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    plot(overall, by_task, out)
    lines = [
        '# LoopFill 最终输出 token 的来源比例', '',
        '2026-11-09。复用表2 LoopFill-online：三模型 × B=1/2/4/8/16，共15组、7,200次请求。'
        '每组完整SpecBench 480题、greedy、最多1024新token、N16、双读出top-2、BF16、Graph-off；未启动新推理。', '',
        '## 统计定义', '',
        '主分母为所有最终输出的decode token，即Σstats.tokens；排除每请求prefill直接产生的首token，'
        '包含原始输出中保留的停止token。按token数加权汇总，不平均各请求百分比。', '',
        '| 类别 | 原字段 | 含义 |', '|---|---|---|',
        '| 后缀 | from_suffix | 验证命中由suffix首先引入的现存候选 |',
        '| 首次读出第1候选 | from_fill | 首次浅层读出的第一候选 |',
        '| 首次读出第2候选 | from_sibling | 首次浅层读出的第二候选；本协议top-k=2 |',
        '| 第二次读出 | from_deep | 第二次浅层读出新引入的候选，含其top-1和top-2 |',
        '| 未命中候选 | tokens − covered_tokens | 全深度输出时不存在匹配草稿候选 |', '',
        'Huginn两次读出为loop 4/8，Ouro为loop 1/2。deep仍是浅层提案，不是全深度输出。'
        '浅层模型合计=fill+sibling+deep。所有类别的实际输出均由全深度预测决定。', '',
        '同一现存候选被多个来源重复提出时，保留其origin，不因后来的重新打分改变归属。'
        '候选被淘汰后重新创建，按最终存活候选的创建来源计数；不是该token路径在整个请求中的最早来源。'
        '该规则受实际提案次序影响，不能解释成来源独占贡献或因果加速比例。'
        '现有日志无法另外恢复“两源都提出”、suffix根查询/孩子续查、或接受时已经完成计算的来源交叉比例。', '',
        'CSV同时提供 *_decode_pct、*_covered_pct 和 *_output_pct：分别以全部decode输出、'
        '命中草稿的输出、含prefill的全部输出为分母。含prefill口径另有prefill_output_pct，六类合计100%。'
        '零分母记空值。每请求核验四种来源计数之和等于covered_tokens，五类之和等于tokens，'
        '再加prefill等于实际输出长度。', '',
        '## 主结果：B1与B16', '',
        table([r for r in overall if r['batch'] in (1, 16)]), '',
        '## 五类来源明细', '', table(overall, detailed=True), '',
        '## 文件与复现', '',
        '- [总体CSV](overall.csv)、[分任务CSV](by-task.csv)、[逐请求CSV](per-request.csv)、[哈希与审计](audit.json)。',
        '- [B1/B16图PDF](token-sources.pdf)、[分任务图PDF](token-sources-by-task.pdf)。',
        '- 运行：`python scripts/loop_token_sources.py`。仅从job指定的完成attempt读取，排除W&B副本。', '',
        '![B1/B16来源构成](token-sources.png)', '',
        '![B16六任务来源构成](token-sources-by-task.png)', '',
    ]
    (out / 'report.md').write_text('\n'.join(lines))
    print(table([r for r in overall if r['batch'] == 16]))
    print(json.dumps({k: audit[k] for k in ('passed', 'cells', 'requests', 'decode_tokens', 'output_tokens')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT / 'results/loopfill-table2-20261107-v2')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/loopfill-token-sources-20261109')
    parser.add_argument('--paper-assets', type=Path,
                        help='Also copy lightweight summaries and figures into a versioned paper directory')
    args = parser.parse_args()
    analyze(args.suite, args.out)
    if args.paper_assets:
        args.paper_assets.mkdir(parents=True, exist_ok=True)
        for name in ('overall.csv', 'by-task.csv', 'audit.json', 'token-sources.pdf',
                     'token-sources.png', 'token-sources-by-task.pdf', 'token-sources-by-task.png'):
            shutil.copy2(args.out / name, args.paper_assets / name)
