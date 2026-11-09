"""Single-seed B16 2x2 tree-sampling experiment, frozen inputs/source and cold streams."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import loop_table2_suite as base
from loop_batch_sweep import check_resources, gpu_snapshot

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['huginn-3.5b', 'ouro-2.6b']
VARIANTS = {
    'A': dict(exact=False, selection='top_n', label='旧评分 + Top-N'),
    'B': dict(exact=True, selection='top_n', label='补概率 + Top-N'),
    'C': dict(exact=False, selection='sample', label='旧评分 + 采样'),
    'D': dict(exact=True, selection='sample', label='补概率 + 采样'),
}
read, save, digest = base.read, base.save, base.digest


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def command(p, job, target, smoke=False):
    lane = p['lanes'][job['model']]
    cmd = base.command(p, lane, job, target, smoke)
    cmd += ['--tree-selection', VARIANTS[job['variant']]['selection'], '--draft-seed', '0']
    if VARIANTS[job['variant']]['exact']:
        cmd += ['--existing-child-probs']
    return cmd


def prepare(out):
    parent = ROOT / 'results/loopfill-table2-20261107-v2'
    p = read(parent / 'protocol.json')
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    for folder in ['src', 'scripts', 'tests', 'configs']:
        shutil.copytree(ROOT / folder, source / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    for folder in ['inputs', 'jobs', 'preflight', 'runs', 'logs']:
        (out / folder).mkdir()
    for model in MODELS:
        info = p['models'][model]
        for kind in ['inputs', 'smoke_inputs']:
            original = Path(info[kind])
            assert digest(original) == info[kind + '_sha256']
            target = out / 'inputs' / original.name
            shutil.copy2(original, target)
            info[kind] = str(target)
        assert len(records(info['inputs'])) == 480
        assert len({r['question_id'] for r in records(info['inputs'])}) == 480
    lanes = {m: p['lanes'][g] for m, g in zip(MODELS, [0, 4], strict=True)}
    for lane in lanes.values():
        check_resources(lane)
    p.update(version='tree-sampling-b16-seed0-v1', source=str(source), python=sys.executable,
             parent_protocol=str(parent / 'protocol.json'), created_unix=time.time(),
             models={m: p['models'][m] for m in MODELS}, lanes=lanes,
             variants=VARIANTS, batches=[16], planned_cells=8, arms=['budget-raw'],
             suffix_c_initial=1., timing_repeats=1, draft_seed=0,
             generation='greedy', history='cold_start_completion_only',
             user_scope='Single seed 0, one timing per arm. User explicitly declined multi-seed experiments.',
             tracking='Local durable per-request journals, frozen source/input hashes and supervisor logs; no network upload.',
             resource_policy='Two exclusive GPUs, one per model; four arms sequentially share that GPU and four local CPU cores.',
             paired_gpu_policy='All four arms for each model run sequentially on the same GPU and CPU affinity.',
             recovery='Completed streams retained; interrupted streams restart cold in a new attempt. No midstream splicing. Request RNG is reconstructed from seed/model/request/event.',
             limits=['Single timing and seed: neither timing variance nor seed robustness is established.',
                     'SpecBench informed prior parameter design; this is same-distribution mechanism/performance validation.',
                     'Preselection remains deterministic and limits the candidate pool before continuation queries.',
                     'BF16 outputs and completion histories may differ; output agreement is not a task-quality score.',
                     'Reserved KV capacity stays N+1; fewer rows does not imply lower memory allocation.',
                     'No full observer traces in timing; aggregate counters and histograms are included in wall time.'])
    p['source_sha256'] = {str(f.relative_to(source)): digest(f) for f in source.rglob('*') if f.is_file()}
    p['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    shutil.copy2(ROOT / 'docs/loopfill-tree-sampling-plan.md', out / 'design-plan.md')
    p['design_plan_sha256'] = digest(out / 'design-plan.md')
    (out / 'working-tree.diff').write_text(subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True))
    save(out / 'protocol.json', p)
    runtime = read(parent / 'runtime.json')
    runtime.update(python=sys.version, pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines())
    ext = Path(p['assets']) / 'ArcticInference/arctic_inference/suffix_decoding'
    for name, sha in runtime['arctic_extension_sha256'].items():
        assert digest(ext / name) == sha
    save(out / 'runtime.json', runtime)
    for model in MODELS:
        for variant in VARIANTS:
            job = dict(key=f'{model}-b16-{variant}', model=model, batch=16,
                       arm='budget-raw', variant=variant, outcome='pending')
            save(out / 'jobs' / (job['key'] + '.json'), job)
    status(out)
    print(json.dumps(dict(out=str(out), experiments=8, batch=16, draft_seed=0)), flush=True)


def status(out):
    jobs = [read(f) for f in sorted((out / 'jobs').glob('*.json'))]
    pre = [read(f) for f in sorted((out / 'preflight').glob('*.json'))]
    save(out / 'status.json', dict(updated_unix=time.time(), planned=8,
        counts=dict(Counter(j['outcome'] for j in jobs)),
        preflight_counts=dict(Counter(j['outcome'] for j in pre)),
        active=[{k: j.get(k) for k in ['key', 'outcome', 'gpu', 'pid', 'output']}
                for j in jobs + pre if j['outcome'] == 'running']))


def audit(target, job, p, smoke=False):
    summary = base.audit(target, job, p, smoke)
    cfg = summary['config']
    variant = VARIANTS[job['variant']]
    assert cfg['selection_policy'] == variant['selection']
    assert cfg['existing_child_probs'] == variant['exact']
    assert cfg['draft_seed'] == 0 and cfg['draft_model_id'] == job['model']
    assert cfg['scoring'] == 'raw' and cfg['c_initial'] == 1.
    assert cfg['draft_node_budget'] == 16 and cfg['top_m'] == 2
    assert cfg['suffix_room'] == 4 and cfg['child_suffix_caps'] == [4, 4]
    assert cfg['readout_depths'] == ([4, 8] if job['model'].startswith('huginn') else [1, 2])
    assert not cfg['positive_only'] and not summary['profiled'] and not summary['controller_profiled']
    meta = read(target / 'metadata.json')
    assert meta['history'] == 'cold_start_completion_only'
    batch = 2 if smoke else 16
    for row in records(target / f'budget-raw-b{batch}-seed0.jsonl'):
        s = row['stats']
        assert s['tokens'] == len(row['tokens']) - 1
        assert s['max_rows'] <= 17 and s['peak_draft_nodes'] <= 16
        assert s['controller_updates'] == 0 and s['c_final'] == 1.
        hist = s['tree_histograms']['execution_rows']
        assert sum(int(k) * v for k, v in hist.items()) == s['rows']
        assert sum(hist.values()) == s['calls']
        assert s['wasted_rows'] == sum(s['wasted_' + reason + '_rows'] for reason in ['budget', 'rejected', 'end'])
        if variant['selection'] == 'sample':
            counts = s['sampling_counts']
            assert counts.get('draws', 0) == 16 * (s['sampling_events'] - counts.get('empty_events', 0) - counts.get('all_zero_fallbacks', 0))
            assert counts.get('unique_draws', 0) + counts.get('duplicates', 0) == counts.get('draws', 0)
        else:
            assert s['sampling_events'] == 0
    return summary


def run(out):
    p = read(out / 'protocol.json')
    source = Path(p['source'])
    for rel, sha in p['source_sha256'].items():
        assert digest(source / rel) == sha, rel
    for spec in p['models'].values():
        for kind in ['inputs', 'smoke_inputs']:
            assert digest(spec[kind]) == spec[kind + '_sha256']
    mutex = threading.Lock()

    def execute(job, smoke):
        folder = 'preflight' if smoke else 'jobs'
        saved = out / folder / (job['key'] + '.json')
        if saved.exists():
            old = read(saved)
            if old['outcome'] == 'complete':
                audit(Path(old['output']), old, p, smoke)
                return old
            if old.get('pid') and old['outcome'] == 'running':
                try:
                    os.kill(old['pid'], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError(f"child still alive: {old['pid']}")
            job = old
        lane = p['lanes'][job['model']]
        attempt = job.get('attempt', 0) + 1
        label = ('smoke-' if smoke else '') + job['key'] + f'-attempt{attempt}'
        target = out / 'runs' / label
        job.update(attempt=attempt, gpu=lane['gpu'], gpu_uuid=lane['uuid'], cpus=lane['cpus'],
                   command=command(p, job, target, smoke), output=str(target), started_unix=time.time())
        try:
            job['resource_before'] = check_resources(lane)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=lane['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                PYTHONPATH=str(source / 'src') + ':' + str(Path(p['assets']) / 'ArcticInference'),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with (out / 'logs' / (label + '.log')).open('x') as log:
                child = subprocess.Popen(job['command'], cwd=source, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(pid=child.pid, outcome='running')
                save(saved, job)
                with mutex:
                    status(out)
                rc = child.wait()
            job['returncode'] = rc
            if rc != 0:
                raise RuntimeError(f'benchmark exited {rc}; see {label}.log')
            job.update(summary=audit(target, job, p, smoke), outcome='complete')
        except Exception as exc:
            job.update(outcome='failed', error=repr(exc))
        job.update(finished_unix=time.time(), resource_after=gpu_snapshot(lane['gpu']))
        save(saved, job)
        with mutex:
            status(out)
        print(json.dumps({k: job.get(k) for k in ['key', 'outcome', 'error', 'output']}), flush=True)
        return job

    def model_run(model):
        lane = p['lanes'][model]
        with open(f"/tmp/loopfill-suite-gpu{lane['gpu']}.lock", 'a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            jobs = [read(out / 'jobs' / f'{model}-b16-{v}.json') for v in VARIANTS]
            smoke_results = [execute(copy.deepcopy(j), True) for j in jobs]
            if any(j['outcome'] != 'complete' for j in smoke_results):
                return
            for job in jobs:
                execute(job, False)
    with open(out / 'supervisor.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ThreadPoolExecutor(max_workers=2) as workers:
            list(workers.map(model_run, MODELS))
    analyze(out)


def work(rows, depth):
    counts, sampling, histograms, stops = Counter(), Counter(), defaultdict(Counter), Counter()
    keys = ['tokens', 'calls', 'rows', 'readout_rows', 'speculate_calls', 'child_suffix_queries',
            'selection_s', 'sampling_s', 'speculate_s', 'wasted_rows', 'wasted_budget_rows',
            'wasted_rejected_rows', 'computed_nodes_rebuilt', 'nodes_released', 'zero_score_rows',
            'selected_zero_nodes_sum', 'sampling_events']
    for row in rows:
        s = row['stats']
        counts.update({k: s[k] for k in keys})
        sampling.update(s['sampling_counts'])
        for k, hist in s['tree_histograms'].items():
            histograms[k].update(hist)
        stops[s['stop_reason']] += 1
    tok, calls = counts['tokens'], counts['calls']
    flat = dict(requests=len(rows), output_tokens=sum(r['n_tokens'] for r in rows), **counts)
    flat.update(gamma=depth*tok/calls if calls else None,
                rows_per_token=counts['rows']/tok if tok else None,
                loop_steps_per_token=calls/tok if tok else None,
                readout_rows_per_token=counts['readout_rows']/tok if tok else None,
                mean_output_length=flat['output_tokens']/len(rows) if rows else None,
                duplicate_rate=sampling['duplicates']/sampling['draws'] if sampling['draws'] else None,
                **{'sample_' + k: v for k, v in sampling.items()})
    for k, hist in histograms.items():
        flat['mean_' + k] = sum(int(x)*n for x, n in hist.items())/sum(hist.values()) if sum(hist.values()) else None
    return flat, dict(histograms), dict(stops)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(out):
    p = read(out / 'protocol.json')
    table, distributions, by_run, pairs, hashes, task_rows = [], {}, {}, [], {}, []
    for model in MODELS:
        for variant in VARIANTS:
            key = f'{model}-b16-{variant}'
            job = read(out / 'jobs' / (key + '.json'))
            if job['outcome'] != 'complete':
                continue
            target = Path(job['output'])
            summary = audit(target, job, p)
            rows = records(target / 'budget-raw-b16-seed0.jsonl')
            depth = summary['config']['depth_total']
            metrics, hist, stops = work(rows, depth)
            table.append(dict(model=model, variant=variant, batch=16, seed=0,
                gpu=job['gpu'], wall_s=summary['wall_s'], tokens_per_s=summary['output_tput'], **metrics))
            distributions[key] = dict(histograms=hist, stop_reasons=stops)
            by_run[(model, variant)] = {r['request_id']: r for r in rows}
            groups = defaultdict(list)
            for row in rows:
                groups[row['category'] if row['category'] in ['translation', 'summarization', 'math_reasoning', 'qa', 'rag'] else 'mt_bench'].append(row)
            for task, group in groups.items():
                metrics, _, _ = work(group, depth)
                task_rows.append(dict(model=model, variant=variant, task=task, **metrics))
            for file in target.glob('*.json*'):
                hashes[str(file.relative_to(out))] = digest(file)
        for v0, v1 in [('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D')]:
            if (model, v0) not in by_run or (model, v1) not in by_run:
                continue
            r0, r1 = by_run[model, v0], by_run[model, v1]
            same = [rid for rid in r0 if r0[rid]['tokens'] == r1[rid]['tokens']]
            a, b = [next(r for r in table if r['model'] == model and r['variant'] == v) for v in [v0, v1]]
            depth = 32 if model.startswith('huginn') else 4
            pairs.append(dict(model=model, comparison=v1+'-'+v0,
                throughput_change_pct=100*(b['tokens_per_s']/a['tokens_per_s']-1),
                rows_change_pct=100*(b['rows_per_token']/a['rows_per_token']-1),
                identical_outputs=len(same), requests=len(r0), identical_ids=same,
                matched_work={v: work([rs[rid] for rid in same], depth)[0] for v, rs in [(v0, r0), (v1, r1)]}))
    write_csv(out / 'measurements.csv', table)
    write_csv(out / 'task-metrics.csv', task_rows)
    save(out / 'analysis.json', dict(table=table, paired=pairs, distributions=distributions, limitations=p['limits']))
    save(out / 'audit.json', dict(passed=len(table)==8, completed=len(table), expected=8,
         requests=sum(r['requests'] for r in table), input_hashes=hashes, analyzer_sha256=digest(Path(__file__))))
    lines = ['# LoopFill B16 树采样实验', '',
        '单种子0、单次计时；2模型 × 4臂 × 480题，最多1024新token。c=1，N16，双读出Top-2，根Q4、孩子Q(4,4)，BF16 greedy，Graph-off，无行填充。', '',
        '| 模型 | 臂 | token/s | γ | rows/token | 平均保留草稿 | 输出token |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['model']} | {r['variant']} | {r['tokens_per_s']:.2f} | {r['gamma']:.3f} | {r['rows_per_token']:.2f} | {r['mean_retained_drafts']:.2f} | {r['output_tokens']} |")
    lines += ['', '| 模型 | 对比 | 吞吐变化 | rows/token变化 | 同输出请求 |', '|---|---|---:|---:|---:|']
    for r in pairs:
        lines.append(f"| {r['model']} | {r['comparison']} | {r['throughput_change_pct']:+.2f}% | {r['rows_change_pct']:+.2f}% | {r['identical_outputs']}/{r['requests']} |")
    lines += ['', 'A=旧评分+Top-N，B=补概率+Top-N，C=旧评分+采样，D=补概率+采样。', '',
        '前置确定性Top-N保留；每个后续边界仅最终选择采样一次。完整计数、动态规模分布、停止原因见analysis.json；逐任务数据见task-metrics.csv。', '',
        '限制：只有单次计时，不能推断跨运行或跨种子的稳定性；同输出子集仅作条件诊断。SpecBench参与过参数设计，不是独立held-out评测。BF16输出差异和任务质量需区分，当前不主张无损或质量改善。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    status(out)
    print(json.dumps(dict(completed=len(table), expected=8, report=str(out/'report.md'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.action](args.out.resolve())
