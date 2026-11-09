"""Audit and plot existing load/budget and root-readiness evidence without inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from loop_four_ablation_suite import audit as ablation_audit, metrics
from loop_table2_suite import read, save, digest

MODELS = ['huginn-3.5b', 'ouro-2.6b']


def csv_write(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def readiness(rows, depth):
    totals, hist = Counter(), Counter()
    for r in rows:
        stats, trace, waits, entries = r['stats'], r['trace'], r['waits'], r['wait_entries']
        assert stats['tokens'] == len(r['tokens']) - 1
        assert stats['calls'] == trace.get('calls', 0)
        assert stats['rows'] == trace.get('rows', 0)
        assert not waits.get('reach_violations', 0)
        assert len(entries) == waits.get('entries', 0) == trace.get('root_entries', 0)
        assert len({e[0] for e in entries}) == len(entries)
        initial = depth if stats['tokens'] else 0
        assert initial == trace.get('initial_root_wait', 0)
        assert stats['calls'] == initial + sum(depth - e[1] for e in entries)
        assert sum(e[2] for e in entries) == waits.get('proposal_wait', 0)
        assert sum(e[3] for e in entries) == waits.get('ranking_wait', 0)
        assert sum(e[1] == depth for e in entries) == waits.get('full_ready_entries', 0)
        for _, step, proposal, ranking in entries:
            assert min(step, proposal, ranking) >= 0 and step + proposal + ranking == depth
            hist[step] += 1
        totals.update(requests=1, decode_tokens=stats['tokens'], loop_steps=stats['calls'],
                      rows=stats['rows'], entries=len(entries), initial_wait=initial,
                      proposal_wait=waits.get('proposal_wait', 0),
                      ranking_wait=waits.get('ranking_wait', 0),
                      correct_budget_waste=trace.get('correct_path_budget_waste', 0),
                      rebuilt_nodes=trace.get('correct_path_rebuilt_nodes', 0),
                      allocated_correct_nodes=trace.get('correct_path_allocated_nodes', 0))
    n = totals['entries']
    return dict(**totals, full_ready_fraction=hist[depth]/n if n else None,
                mean_progress_fraction=sum(k*v for k, v in hist.items())/depth/n if n else None,
                mean_remaining_steps=sum((depth-k)*v for k, v in hist.items())/n if n else None,
                gamma=depth*totals['decode_tokens']/totals['loop_steps']
                if totals['loop_steps'] else None), hist


def analyze(ablation, headroom, out):
    out.mkdir(parents=True, exist_ok=True)
    protocol = read(ablation/'protocol.json')
    expected = {r['key']: r for r in csv.DictReader((ablation/'measurements.csv').open())}
    records, provenance = [], []
    for path in sorted((ablation/'jobs').glob('*.json')):
        job = read(path)
        assert job['outcome'] == 'complete'
        target = Path(job['output'])
        summary = ablation_audit(target, job, protocol)
        journal = target/f"{job['arm']}-b{job['batch']}-seed0.jsonl"
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        depth = protocol['mother_configs'][job['model']]['depth_total']
        derived = metrics(rows, depth, job['batch'])
        for k, value in derived.items():
            if value is not None:
                assert abs(value-float(expected[job['key']][k])) < 1e-8, (job['key'], k)
        records.append(dict(key=job['key'], model=job['model'], batch=job['batch'],
                            nodes=job['nodes'], variant=job['variant'], **derived,
                            output_tput=summary['output_tput'], wall_s=summary['wall_s']))
        provenance.append(dict(key=job['key'], requests=len(rows), path=str(journal),
                               sha256=digest(journal)))
    assert len(records) == 36
    budgets = [r for r in records if r['variant'] in ('n8', 'r2-k2-q4', 'n32', 'n64')]
    assert len(budgets) == 16
    csv_write(out/'ablation-efficiency.csv', records)
    csv_write(out/'load-budget.csv', budgets)

    hp = read(headroom/'protocol.json')
    for rel, sha in hp['source_sha256'].items():
        assert digest(Path(hp['source'])/rel) == sha
    summaries, cdfs = [], []
    for model in MODELS:
        key = f'{model}-n16-online-c0.7'
        target = headroom/'runs'/key
        journal = target/'requests.jsonl'
        done = read(target/'complete.json')
        assert digest(journal) == done['requests_sha256']
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        ids = []
        for dataset, info in hp['inputs'].items():
            ipath = headroom/'inputs'/f'{dataset}.jsonl'
            assert digest(ipath) == info['sha256']
            ids += [json.loads(line)['source_id'] for line in ipath.read_text().splitlines()]
        assert [r['source_id'] for r in rows] == ids and len(rows) == done['requests']
        grouped = defaultdict(list)
        for r in rows:
            grouped[r['dataset']].append(r)
        grouped['all_six_datasets'] = rows
        for dataset, group in grouped.items():
            depth = hp['models'][model]['depth_total']
            summary, hist = readiness(group, depth)
            summaries.append(dict(model=model, dataset=dataset, depth=depth, **summary))
            cumulative = 0
            for step in range(depth+1):
                cumulative += hist[step]
                cdfs.append(dict(model=model, dataset=dataset, step=step, progress=step/depth,
                                 count=hist[step], cdf=cumulative/summary['entries']))
        provenance.append(dict(key=key, requests=len(rows), path=str(journal), sha256=digest(journal)))
    csv_write(out/'readiness.csv', summaries)
    csv_write(out/'readiness-cdf.csv', cdfs)
    save(out/'audit.json', dict(passed=True, ablation_configurations=36,
         ablation_requests=sum(r['requests'] for r in records), readiness_requests=240,
         source_protocols={str(p): digest(p) for p in [ablation/'protocol.json', headroom/'protocol.json']},
         provenance=provenance,
         missing_fields=['Headroom journals omit per-handover origin and lifecycle events; '
                         'source-transition progress cannot be reconstructed.'],
         scope='Readiness: existing B1/N16/512-token diagnostics, 20 requests each from '
               'GSM8K/MATH500/BBH/HumanEval+/MBPP+/SpecBench. Not the full480 protocol.'))
    plot(out, budgets, summaries, cdfs, records)
    lines = ['# Mechanism evidence (existing runs)', '',
             '36 ablation configurations / 17,280 requests and 240 existing readiness requests audited.', '',
             '## Load and node budget', '',
             '| Model | B | N | Decode s/request | Decode ms/token | Output token/s | Gamma | Rows/token |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(budgets, key=lambda r: (r['model'], r['batch'], r['nodes'])):
        cells = [r['model'], str(r['batch']), str(r['nodes'])]
        cells += ['—' if r[k] is None else f'{r[k]:.3f}' for k in
                  ['mean_decode_latency_s', 'decode_ms_per_token', 'output_tput', 'gamma', 'rows_per_token']]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', 'Endpoints already establish a load-dependent budget tradeoff. No B4/B8 inference '
              'is added. N64 vs N32 at B1 is a small single-run difference; no significance claim.', '',
              '## Root readiness', '',
              'Existing online-c0.7, B1/N16, max512, six datasets ×20 requests per model. '
              'These are six different datasets, not six balanced SpecBench tasks. '
              'Terminal tokens are excluded from handover denominators; initial-root wait is separate.', '',
              '| Model | Handovers | Mean r/T | Immediately ready | Remaining steps/handover | Rebuilt correct nodes |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in summaries:
        if r['dataset'] == 'all_six_datasets':
            lines.append(f"| {r['model']} | {r['entries']} | {r['mean_progress_fraction']:.3f} | "
                         f"{100*r['full_ready_fraction']:.2f}% | {r['mean_remaining_steps']:.3f} | {r['rebuilt_nodes']} |")
    lines += ['', 'Every request satisfies calls = initial wait + sum(T-r) and T-r = proposal wait + ranking wait. '
              'Source transitions cannot be reconstructed from these aggregate journals. '
              'No new trace is required for the progress/wait/rebuild claims above. '
              'Different outputs and histories limit causal interpretation. No diagnostic wall time enters throughput.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'audit': str(out/'audit.json'), 'report': str(out/'report.md'),
                      'new_gpu_runs': 0, 'readiness': [r for r in summaries if r['dataset']=='all_six_datasets']}))


def plot(out, budgets, summaries, cdfs, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), constrained_layout=True)
    for i, model in enumerate(MODELS):
        for b, color in [(1, '#31688e'), (16, '#e1812c')]:
            rows = sorted((r for r in budgets if r['model']==model and r['batch']==b), key=lambda r:r['nodes'])
            xs = [r['nodes'] for r in rows]
            if b == 1:
                for j, field in enumerate(['mean_decode_latency_s', 'decode_ms_per_token']):
                    axes[i,j].plot(xs, [r[field] for r in rows], 'o-', color=color)
            else:
                axes[i,2].plot(xs, [r['output_tput'] for r in rows], 'o-', color=color)
            axes[i,3].plot([r['rows_per_token'] for r in rows], [r['gamma'] for r in rows], 'o-', label=f'B={b}')
            for r in rows:
                axes[i,3].annotate(f"N{r['nodes']}", (r['rows_per_token'],r['gamma']), fontsize=7)
        for j, title in enumerate(['B1 decode seconds/request', 'B1 decode ms/token', 'B16 output token/s', 'Compute vs serial savings']):
            axes[i,j].set_title(f'{model}\n{title}')
            axes[i,j].grid(alpha=.2)
            axes[i,j].set_xlabel('Node budget N' if j<3 else 'Rows/decode token')
        axes[i,3].set_ylabel('Gamma')
        axes[i,3].legend()
    for ext in ['pdf', 'png']:
        fig.savefig(out/f'load-budget.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for model in MODELS:
        points = [r for r in cdfs if r['model']==model and r['dataset']=='all_six_datasets']
        axes[0].step([r['progress'] for r in points], [r['cdf'] for r in points], where='post', label=model)
    axes[0].set(xlabel='Progress at root handover r/T', ylabel='CDF', ylim=(0,1.02))
    axes[0].legend()
    rows = [r for r in summaries if r['dataset']=='all_six_datasets']
    bottom = [0.0]*len(rows)
    for field, label in [('initial_wait','Initial root'), ('proposal_wait','Proposal wait'), ('ranking_wait','Ranking wait')]:
        values = [r[field]/r['loop_steps'] for r in rows]
        axes[1].bar(MODELS, values, bottom=bottom, label=label)
        bottom = [a+b for a,b in zip(bottom, values)]
    axes[1].set(ylabel='Fraction of per-request loop steps', ylim=(0,1.05))
    axes[1].legend()
    axes[2].bar(MODELS, [100*r['full_ready_fraction'] for r in rows])
    axes[2].set(ylabel='Immediately ready handovers (%)')
    for ax in axes:
        ax.grid(alpha=.2)
    for ext in ['pdf','png']:
        fig.savefig(out/f'readiness.{ext}', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        rows = [r for r in records if r['model']==model and r['batch']==16]
        for r in rows:
            ax.scatter(r['rows_per_token'], r['loop_steps_per_token'])
            ax.annotate(r['variant'], (r['rows_per_token'], r['loop_steps_per_token']), fontsize=6)
        ax.set(title=model+' / B16', xlabel='Rows/decode token', ylabel='Loop steps/decode token')
        ax.grid(alpha=.2)
    for ext in ['pdf','png']:
        fig.savefig(out/f'ablation-efficiency.{ext}', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ablation', type=Path, default=Path('results/loopfill-four-ablation-20261109'))
    p.add_argument('--headroom', type=Path, default=Path('results/loopfill-score-headroom-t0-20261109'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    analyze(a.ablation.resolve(), a.headroom.resolve(), a.out.resolve())
