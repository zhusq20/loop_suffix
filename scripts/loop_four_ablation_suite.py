"""Four B16 ablations plus B1 node budgets, using the frozen Table 2 engine.

The existing Table 2 worker supplies GPU locks, durable results and W&B tracking.
Only the supervisor and analysis are new; inference code is copied byte-for-byte.
Prepare from an existing ablation directory to audit and reuse completed cells;
the child-Q=2 extension adds four B16 streams to the original 36-cell matrix.
"""
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
import queue
import shutil
import subprocess
import sys
import threading
import time

import loop_table2_suite as base
from loop_accepted_length import MT_CATEGORIES, TASKS

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['huginn-3.5b', 'ouro-2.6b']
read, save, digest = base.read, base.save, base.digest
BASE_COMMAND, BASE_AUDIT = base.command, base.audit


def variants():
    result = []
    for readouts in [1, 2]:
        for k in [1, 2, 4]:
            result.append(dict(variant=f'r{readouts}-k{k}-q4', readouts=readouts,
                               top_k=k, child_q=4, nodes=16, c=.7, arm='budget-online'))
    for q in [2, 8]:
        for r in [1, 2]:
            result.append(dict(variant=f'r{r}-k2-q{q}', readouts=r, top_k=2,
                               child_q=q, nodes=16, c=.7, arm='budget-online'))
    mother = result[4]
    assert mother['variant'] == 'r2-k2-q4'
    result += [dict(mother, variant=f'n{n}', nodes=n) for n in [8, 32, 64]]
    result += [dict(mother, variant=f'fixed-c{c}', c=c, arm='budget-fixed')
               for c in [.5, .7, 1.]]
    return result


def cells(variant_specs=None):
    rows = []
    for batch in [16, 1]:
        for v in variants() if variant_specs is None else variant_specs:
            if batch == 1 and v['variant'] not in ['r2-k2-q4', 'n8', 'n32', 'n64']:
                continue
            for model in MODELS:
                rows.append(dict(v, model=model, batch=batch, outcome='pending',
                                 key=f"{model}-b{batch}-{v['variant']}"))
    return rows


def expected_config(protocol, job):
    cfg = copy.deepcopy(protocol['mother_configs'][job['model']])
    cfg.update(name=job['arm'], readout_depths=cfg['readout_depths'][:job['readouts']],
               top_m=job['top_k'], child_suffix_caps=[job['child_q']]*job['top_k'],
               draft_node_budget=job['nodes'], c_initial=job['c'],
               scoring='online_suffix_i' if job['arm']=='budget-online' else 'fixed_suffix')
    return cfg


def command(protocol, lane, job, target, smoke=False):
    cmd = BASE_COMMAND(protocol, lane, job, target, smoke)
    for flag, value in [('draft-node-budget', job['nodes']), ('model-top-k', job['top_k']),
                        ('suffix-c-initial', job['c'])]:
        cmd[cmd.index('--'+flag)+1] = str(value)
    start = cmd.index('--child-suffix-caps')+1
    end = start
    while end < len(cmd) and not cmd[end].startswith('--'):
        end += 1
    cmd[start:end] = [str(job['child_q'])]*job['top_k']
    return cmd + ['--model-readout-depths'] + [str(x) for x in expected_config(protocol, job)['readout_depths']]


def audit(target, job, protocol, smoke=False):
    summary = BASE_AUDIT(target, job, protocol, smoke)
    meta = read(target/'metadata.json')
    assert meta['configs'][job['arm']] == expected_config(protocol, job)
    assert summary['config'] == expected_config(protocol, job)
    assert meta['history'] == 'cold_start_completion_only'
    assert not meta['profiled'] and not meta['controller_profiled']
    for field in ['torch', 'transformers']:
        assert meta[field] == protocol['engine_versions'][field]
    prefix = f"{job['arm']}-b{2 if smoke else job['batch']}-seed0"
    records = [json.loads(x) for x in (target/f'{prefix}.jsonl').read_text().splitlines()]
    for row in records:
        s = row['stats']
        assert s['tokens'] == row['n_tokens']-1
        assert (s['tokens']==0) == (s['calls']==0)
        assert s['peak_draft_nodes'] <= job['nodes']
        assert s['c_initial'] == job['c']
        if job['arm'] == 'budget-fixed':
            assert s['c_final'] == job['c']
    return summary


def metrics(records, depth, batch):
    sums = Counter()
    for r in records:
        for k in ['tokens','calls','rows','readout_rows','wasted_rows','speculate_calls',
                  'child_suffix_queries','selected_draft_nodes_sum','selection_events']:
            sums[k] += r['stats'][k]
    tok, calls = sums['tokens'], sums['calls']
    latency = sum(r['latency_s']-r['ttft_s'] for r in records)
    return dict(requests=len(records), decode_tokens=tok, loop_steps=calls,
                prefill_only_requests=sum(r['stats']['calls']==0 for r in records),
                gamma=depth*tok/calls if calls else None,
                loop_steps_per_token=calls/tok if tok else None,
                rows_per_token=sums['rows']/tok if tok else None,
                readout_rows_per_token=sums['readout_rows']/tok if tok else None,
                wasted_rows_per_token=sums['wasted_rows']/tok if tok else None,
                suffix_queries=sums['speculate_calls'], child_suffix_queries=sums['child_suffix_queries'],
                mean_selected_nodes=sums['selected_draft_nodes_sum']/sums['selection_events'] if sums['selection_events'] else None,
                peak_draft_nodes=max(r['stats']['peak_draft_nodes'] for r in records),
                mean_decode_latency_s=latency/len(records) if batch==1 else None,
                decode_ms_per_token=1000*latency/tok if batch==1 and tok else None)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def summarize(out):
    protocol = read(out/'protocol.json')
    jobs = [read(p) for p in sorted((out/'jobs').glob('*.json'))]
    overall, tasks = [], []
    for j in jobs:
        item = {k:j.get(k) for k in ['key','model','batch','variant','nodes','readouts','top_k',
                                     'child_q','c','arm','outcome','reused_from','wandb_url']}
        if j['outcome']=='complete':
            target = Path(j['output'])
            rows = [json.loads(x) for x in (target/f"{j['arm']}-b{j['batch']}-seed0.jsonl").read_text().splitlines()]
            depth = protocol['mother_configs'][j['model']]['depth_total']
            item.update(metrics(rows, depth, j['batch']), output_tput=j['summary']['output_tput'],
                        wall_s=j['summary']['wall_s'], output_tokens=j['summary']['output_tokens'])
            grouped = defaultdict(list)
            for r in rows:
                cat = 'mt_bench' if r['category'] in MT_CATEGORIES else r['category']
                assert cat in TASKS
                grouped[cat].append(r)
            for cat, group in sorted(grouped.items()):
                tasks.append(dict(key=j['key'], model=j['model'], batch=j['batch'],
                                  variant=j['variant'], task=cat, **metrics(group, depth, j['batch'])))
        overall.append(item)
    write_csv(out/'measurements.csv', overall)
    write_csv(out/'task-metrics.csv', tasks)
    preflight = [read(p) for p in (out/'preflight').glob('*.json')]
    save(out/'status.json', dict(updated_unix=time.time(), planned=len(jobs),
        counts=dict(Counter(j['outcome'] for j in jobs)),
        preflight_counts=dict(Counter(j['outcome'] for j in preflight)),
        active=[{k:j.get(k) for k in ['key','gpu','progress','wandb_url']} for j in jobs+preflight
                if j['outcome'] in ['starting','running']]))


def prepare(out, parent):
    if out.exists():
        raise FileExistsError(out)
    p = read(parent/'protocol.json')
    parent_is_ablation = 'mother_configs' in p
    planned = cells()
    old_source = Path(p['source'])
    for name, sha in p['source_sha256'].items():
        assert digest(old_source/name)==sha, name
    out.mkdir(parents=True)
    shutil.copytree(old_source, out/'source', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__, out/'source/scripts'/Path(__file__).name)
    # The newer supervisor has GPU-lock handoff fixes; inference stays frozen.
    shutil.copy2(ROOT/'scripts/loop_table2_suite.py', out/'source/scripts/loop_table2_suite.py')
    shutil.copytree(parent/'inputs', out/'inputs')
    for folder in ['jobs','runs','logs','preflight']:
        (out/folder).mkdir()
    p.update(version='four-ablation-v2-child-q2', created_unix=time.time(), source=str(out/'source'),
             python=sys.executable, parent_protocol=str(parent/'protocol.json'),
             group=out.name, project='llm-memory-editability', models={m:p['models'][m] for m in MODELS},
             planned_cells=len(planned), batches=[16,1], mother_configs={},
             generation='greedy', main_figure_model='ouro-2.6b',
             primary_metric='B16 end-to-end output throughput; B1 node-budget decode latency',
             secondary_metrics='gamma=T*sum(stats.tokens)/sum(stats.calls); rows/token and task breakdown',
             b1_decode_latency='latency_s - ttft_s; includes scheduler and suffix work after first token; token-weighted ms/token also reported',
             scope=f'B16: {len(variants())} variants/model, child Q=2/4/8; B1: N8/16/32/64 only; no repeat timing',
             diagnostic_trace='Separate follow-up collection; never mixed into measured throughput',
             history='cold_start_completion_only; reset before each stream; warmup discarded',
             limitations=['SpecBench has informed parameter design; not held-out confirmation.',
                         'B16 is a fixed high-load point, not established saturation.',
                         'Single timing repeat; BF16 output differences and quality remain unresolved.',
                         'Per-task gamma is available; mixed-stream wall time is not apportioned into task throughput.'])
    for m in MODELS:
        model = p['models'][m]
        for kind in ['inputs','smoke_inputs']:
            model[kind] = str(out/'inputs'/Path(model[kind]).name)
            assert digest(model[kind]) == model[kind+'_sha256']
        mother_key = f'{m}-b16-r2-k2-q4' if parent_is_ablation else f'{m}-b16-budget-online'
        old = read(parent/'jobs'/f'{mother_key}.json')
        meta = read(Path(old['output'])/'metadata.json')
        p['mother_configs'][m] = meta['configs']['budget-online']
        p['engine_versions'] = {k:meta[k] for k in ['torch','transformers']}
    p['variants'] = variants()
    p['source_sha256'] = {str(x.relative_to(out/'source')):digest(x)
                          for x in (out/'source').rglob('*') if x.is_file()}
    runtime = read(parent/'runtime.json')
    ext = Path(p['assets'])/'ArcticInference/arctic_inference/suffix_decoding'
    for name, sha in runtime['arctic_extension_sha256'].items():
        assert digest(ext/name)==sha
    current_freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    runtime.update(parent_runtime=str(parent/'runtime.json'), current_pip_freeze=current_freeze)
    save(out/'runtime.json',runtime)
    save(out/'protocol.json',p)
    reuse = []
    for j in planned:
        if parent_is_ablation:
            path = parent/'jobs'/f"{j['key']}.json"
        elif j['variant']=='r2-k2-q4':
            path = parent/'jobs'/f"{j['model']}-b{j['batch']}-budget-online.json"
        else:
            path = None
        if path is not None and path.exists():
            old = read(path)
            assert old['outcome']=='complete'
            j.update({k:old[k] for k in ['output','gpu','gpu_uuid','cpus','wandb_url']})
            j['summary'] = audit(Path(j['output']),j,p)
            j.update(outcome='complete',reused_from=str(path))
            reuse.append(dict(key=j['key'], from_job=str(path),
                              metadata_sha256=digest(Path(j['output'])/'metadata.json'),
                              records_sha256=digest(Path(j['output'])/f"{j['arm']}-b{j['batch']}-seed0.jsonl")))
        save(out/'jobs'/f"{j['key']}.json",j)
    save(out/'reuse-audit.json',reuse)
    summarize(out)
    print(json.dumps(dict(planned=len(planned),reused=len(reuse),new=len(planned)-len(reuse))),flush=True)


def run(out):
    protocol = read(out/'protocol.json')
    # A historical frozen protocol must retain its original experiment matrix.
    planned = cells(protocol['variants'])
    lock = (out/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name, sha in protocol['source_sha256'].items():
        assert digest(Path(protocol['source'])/name)==sha, name
    for m in protocol['models'].values():
        for k in ['inputs','smoke_inputs']:
            assert digest(m[k])==m[k+'_sha256']
    # Do not silently restart a live or interrupted process. Resume is explicit.
    for path in list((out/'jobs').glob('*.json'))+list((out/'preflight').glob('*.json')):
        assert read(path)['outcome'] not in ['starting','running'], f'Inspect interrupted job before resuming: {path}'
    save(out/'supervisor.json',dict(pid=os.getpid(),started_unix=time.time(),script_sha256=digest(__file__)))
    status_lock = threading.Lock()

    def phase(paths, smoke):
        pending = queue.Queue()
        for path in paths:
            pending.put(path)
        def lane(index):
            while not pending.empty():
                reservation = base.try_gpu_lock(protocol['lanes'][index])
                if reservation is None:
                    time.sleep(5)
                    continue
                try:
                    try:
                        path = pending.get_nowait()
                    except queue.Empty:
                        return
                    job = read(path)
                    cmd = [sys.executable, str(Path(__file__).resolve()),'worker','--out',str(out),
                           '--job',str(path),'--lane',str(index),'--gpu-lock-fd',str(reservation.fileno())]
                    if smoke:
                        cmd.append('--smoke')
                    env = dict(os.environ, PYTHONPATH=protocol['source']+'/src:'+protocol['source']+'/scripts:'+protocol['assets']+'/ArcticInference',
                               WANDB_SILENT='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    log_path = out/'logs'/f"{job['key']}-{'smoke' if smoke else 'full'}.log"
                    with log_path.open('a') as log:
                        process = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                                                   pass_fds=(reservation.fileno(),),start_new_session=True)
                finally:
                    reservation.close()
                while process.poll() is None:
                    with status_lock:
                        summarize(out)
                    time.sleep(5)
                current = read(path)
                if process.returncode and current['outcome'] not in ['complete','oom','failed','audit_failed']:
                    current.update(outcome='worker_failed',worker_returncode=process.returncode,worker_log=str(log_path))
                    save(path,current)
                with status_lock:
                    summarize(out)
        with ThreadPoolExecutor(max_workers=len(protocol['lanes'])) as pool:
            list(pool.map(lane,range(len(protocol['lanes']))))

    smokes = []
    for j in planned:
        if j['batch'] != 16 or read(out/'jobs'/f"{j['key']}.json")['outcome']=='complete':
            continue
        path = out/'preflight'/f"{j['key']}.json"
        if not path.exists():
            save(path,j)
        if read(path)['outcome']!='complete':
            smokes.append(path)
    phase(smokes,True)
    failures = [p.name for p in (out/'preflight').glob('*.json') if read(p)['outcome']!='complete']
    if failures:
        save(out/'preflight-failed.json',failures)
        raise RuntimeError(f'Preflight failed: {failures}')
    phase([out/'jobs'/f"{j['key']}.json" for j in planned
           if read(out/'jobs'/f"{j['key']}.json")['outcome'] not in ['complete','oom']],False)
    summarize(out)
    save(out/'suite-finished.json',read(out/'status.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker','summarize'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'results/loopfill-table2-20261107-v2')
    parser.add_argument('--job',type=Path)
    parser.add_argument('--lane',type=int)
    parser.add_argument('--gpu-lock-fd',type=int)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action=='prepare':
        prepare(out,args.parent.resolve())
    elif args.action=='run':
        run(out)
    elif args.action=='worker':
        base.command, base.audit = command, audit
        base.worker(out,args.job,args.lane,args.smoke,args.gpu_lock_fd)
    else:
        summarize(out)


if __name__=='__main__':
    main()
