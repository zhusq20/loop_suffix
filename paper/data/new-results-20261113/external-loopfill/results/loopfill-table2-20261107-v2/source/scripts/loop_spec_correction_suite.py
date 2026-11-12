"""Prioritize correction experiments while preserving/adopting frozen baseline jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from loop_capacity_suite import digest, save, verify_suite
from loop_batch_sweep import check_resources, classify, gpu_snapshot, summarize
from loop_online_suite import audit_run

BASELINES = ('ar', 'wavefront', 'loopspec-dual', 'budget-suffix')
VARIANTS = [
    dict(name='raw-top2', arm='budget-raw', top_k=2, nodes=64, child='all', positive=False),
    dict(name='raw-top1', arm='budget-raw', top_k=1, nodes=64, child='all', positive=False),
    dict(name='raw-top2-child1', arm='budget-raw', top_k=2, nodes=64, child='best1', positive=False),
    dict(name='raw-top2-root-only', arm='budget-raw', top_k=2, nodes=64, child='none', positive=False),
    dict(name='raw-top2-positive', arm='budget-raw', top_k=2, nodes=64, child='all', positive=True),
    dict(name='raw-top2-n16', arm='budget-raw', top_k=2, nodes=16, child='all', positive=False),
    dict(name='shallow-top8', arm='budget-shallow', top_k=8, nodes=64, child='all', positive=False),
    dict(name='raw-top8-control', arm='budget-raw', top_k=8, nodes=64, child='all', positive=False),
]


def q_variants():
    """Three Q levels locate the coarse tradeoff before any finer sweep."""
    def variant(name, arm='budget-raw', k=2, q=64):
        return dict(name=name, arm=arm, top_k=k, nodes=64, child='all', positive=False, suffix_q=q)
    return [
        variant('raw-top2'),  # Adopt the already-running identical Q=64 cell.
        variant('raw-top2-q4', q=4), variant('raw-top2-q16', q=16),
        variant('shallow-top2', arm='budget-shallow'),
        variant('raw-top1-q4', k=1, q=4), variant('raw-top1-q16', k=1, q=16),
        variant('raw-top1', k=1), variant('shallow-top1', arm='budget-shallow', k=1),
        variant('suffix-q4', arm='budget-suffix', q=4),
        variant('suffix-q16', arm='budget-suffix', q=16),
        variant('suffix-q64', arm='budget-suffix'),
    ]


def n_variants():
    variants=[]
    for n in (8,16,32,64):
        for label,arm in [('raw-top2-q4','budget-raw'),('shallow-top2','budget-shallow'),
                          ('suffix-q4','budget-suffix'),('loopspec-capped','loopspec-capped')]:
            variants.append(dict(name=f'{label}-n{n}',arm=arm,top_k=2,nodes=n,child='all',
                                 positive=False,suffix_q=4))
    return variants


def live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def replace_option(command, option, value):
    command[command.index(option) + 1] = str(value)


def baseline_command(templates, lane, job, parent):
    """Select model/environment by job, and CPU/GPU placement by physical lane."""
    candidates=[j for j in templates if j['model']==job['model'] and j.get('command')]
    template=next((j for j in candidates if j['gpu']==lane['gpu']), candidates[0])
    cmd=list(template['command'])
    for option,value in [('--cpu-list',lane['cpus']),('--arms',job['arm']),
                         ('--batch-sizes',job['batch']),('--out',parent/'runs'/job['key'])]:
        replace_option(cmd,option,value)
    if cmd[cmd.index('--model')+1]!=job['model']:
        raise ValueError('template model differs from job model')
    return cmd


def prepare(parent, out, focus_huginn_b8=False, resume_correction=None, budget_alignment=False):
    parent, out = parent.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError(out)
    old = verify_suite(parent)
    if live(json.loads((parent / 'correction-handoff.json').read_text())['supervisor_pid']):
        raise RuntimeError('old supervisor still running')
    repo = Path(__file__).resolve().parents[1]
    source = out / 'source'
    source.mkdir(parents=True)
    for name in ['src', 'scripts', 'tests', 'configs']:
        shutil.copytree(repo / name, source / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['README.md', 'pyproject.toml']:
        shutil.copy2(repo / name, source / name)
    (source / 'docs').mkdir()
    shutil.copy2(repo / 'docs/loopfill-spec-correction.md', source / 'docs/loopfill-spec-correction.md')
    for name in ['jobs', 'runs', 'logs', 'inputs', 'references', 'validation']:
        (out / name).mkdir()
    inputs = {}
    for model in old['model_revisions']:
        target = out / 'inputs' / f'{model}.jsonl'
        shutil.copy2(old['inputs'][f'specbench480/{model}'], target)
        inputs[model] = str(target)
    variants = n_variants() if budget_alignment else q_variants() if focus_huginn_b8 else VARIANTS
    planned = 16 if budget_alignment else 11 if focus_huginn_b8 else 32
    focused=focus_huginn_b8 or budget_alignment
    protocol = dict(version='matched-node-budget-huginn-b8-v1' if budget_alignment else 'spec-correction-huginn-b8-q-v2' if focus_huginn_b8 else 'spec-correction-v1', parent=str(parent), source=str(source),
                    created_unix=time.time(), variants=variants, inputs=inputs,
                    lanes=[l for l in old['lanes'] if not budget_alignment or l['gpu'] in (3,6,7)], model_revisions=old['model_revisions'], planned_cells=planned,
                    batches=[8] if focused else [4, 8],
                    correction_models=['huginn-3.5b'] if focused else list(old['model_revisions']),
                    manage_baselines=not budget_alignment,wait_for_gpu=budget_alignment,
                    resume_correction=str(resume_correction.resolve()) if resume_correction else None,
                    priority='adopt running baseline/correction, corrections, remaining baseline',
                    max_new_tokens=1024, seed=0, repeat=0, development_experiment=True,
                    validation='8 first-ID requests, 16 generated tokens, each model/B/variant before full480',
                    retained_baselines=list(BASELINES), telemetry_seconds=15)
    save(out / 'protocol.json', protocol)
    hashes = {str(f.relative_to(source)): digest(f) for f in source.rglob('*') if f.is_file()}
    save(out / 'source-hashes.json', hashes)
    # Serial reference uses tree_decode with ArmConfig(ar). Only non-reference
    # budget logic, batch progress and CLI additions changed; preserve original rows.
    old_source = Path(old['source'])
    allowed = {'scripts/loop_batch_bench.py', 'src/llm_memory_editability/loop_batch_decode.py',
               'src/llm_memory_editability/loop_budget_decode.py',
               'src/llm_memory_editability/loop_loopspec_decode.py'}
    expected_sources = [source / 'scripts/loop_batch_bench.py'] + sorted((source / 'src/llm_memory_editability').glob('*.py'))
    reference_hashes = {str(f.relative_to(source)): digest(f) for f in expected_sources}
    if budget_alignment:
        if resume_correction is None:
            raise ValueError('matched N requires the existing Q control suite')
        control_protocol = verify_suite(resume_correction.resolve())
        control_source = Path(control_protocol['source'])
        control_hashes = {name: digest(control_source/name) for name in reference_hashes}
        reuse_changes = [name for name in reference_hashes if reference_hashes[name] != control_hashes[name]]
        reviewed = {'scripts/loop_batch_bench.py', 'src/llm_memory_editability/loop_loopspec_decode.py'}
        if set(reuse_changes) - reviewed:
            raise ValueError(f'budget control execution changed: {reuse_changes}')
        if digest(inputs['huginn-3.5b']) != digest(control_protocol['inputs']['huginn-3.5b']):
            raise ValueError('Q control input differs')
        save(out/'reuse-provenance.json', dict(control_suite=str(resume_correction.resolve()),
             control_source_sha256=control_hashes, changed_source_files=reuse_changes,
             rationale='Reviewed CLI changes only add the capped LoopSpec arm. Budget decoder, batch executor, suffix cache, model backends and remaining source are byte-identical.'))
    changed = [name for name in reference_hashes if not (old_source/name).exists()
               or digest(old_source/name) != reference_hashes[name]]
    if set(changed) - allowed:
        raise RuntimeError(f'unreviewed reference-source changes: {set(changed)-allowed}')
    provenance = dict(changed_source_files=changed,
                      rationale='Serial tree_decode, backbone, model/precision loading and AR config unchanged; batch callback, budget variants and capped LoopSpec do not execute in serial AR. Headers migrated, token rows byte-preserved.',
                      references={})
    for model in old['model_revisions']:
        original = parent / 'references' / f'{model}.jsonl'
        lines = original.read_text().splitlines(keepends=True)
        header = json.loads(lines[0])
        original_header = json.loads(lines[0])
        header['reference_identity']['source_sha256'] = reference_hashes
        target = out / 'references' / original.name
        target.write_text(json.dumps(header) + '\n' + ''.join(lines[1:]))
        assert target.read_text().splitlines()[1:] == original.read_text().splitlines()[1:]
        provenance['references'][model] = dict(source=str(original), source_sha256=digest(original),
                                               destination_sha256=digest(target), original_header=original_header)
    save(out / 'reference-provenance.json', provenance)
    save(out / 'input-hashes.json', {str(f): digest(f) for f in list((out/'inputs').glob('*.jsonl')) + list((out/'references').glob('*.jsonl'))})
    for lane in old['lanes']:
        batch = lane['batches'][0]
        if budget_alignment:
            if lane['gpu'] not in (3,6,7):continue
            batch=8
        elif batch not in protocol['batches'] or lane['model'] not in protocol['correction_models']:
            continue
        for variant in variants:
            if budget_alignment and {8:3,16:6,32:7,64:3}[variant['nodes']]!=lane['gpu']:continue
            model='huginn-3.5b' if budget_alignment else lane['model']
            key = f"{model}-b{batch}-{variant['name']}"
            job = dict(key=key, model=model, batch=batch, arm=variant['arm'], variant=variant,
                       gpu=lane['gpu'], gpu_uuid=lane['uuid'], outcome='pending')
            if budget_alignment and variant['nodes']==64 and variant['arm']!='loopspec-capped':
                if resume_correction is None:raise ValueError('N64 controls require the Q suite')
                control={'budget-raw':'raw-top2-q4','budget-shallow':'shallow-top2','budget-suffix':'suffix-q4'}[variant['arm']]
                previous=resume_correction.resolve()/'jobs'/f'huginn-3.5b-b8-{control}.json'
                if not previous.exists():raise FileNotFoundError(previous)
                job.update(outcome='reuse_pending',reuse_from=str(previous),
                           reuse_rationale='Budget decoder, batch executor and optimized backend unchanged vs Q suite; new CLI only adds LoopSpec-capped branch. Reuse same GPU3 N64 measured controls.')
            elif resume_correction and variant['name']=='raw-top2':
                previous = resume_correction.resolve()/'jobs'/f'{key}.json'
                adopted=json.loads(previous.read_text())
                if adopted['outcome'] not in ('running','complete'):
                    raise ValueError('expected a live or completed raw-top2 Q64 job to adopt')
                meta=json.loads((Path(adopted['output'])/'metadata.json').read_text())
                if meta['source_sha256'] != reference_hashes:
                    raise ValueError('adopted measurement decoder/bench source differs')
                cfg=meta['configs'][variant['arm']]
                for k,expected in dict(top_m=2,suffix_room=64,draft_node_budget=64,
                                       child_suffix_policy='all',positive_only=False,use_suffix=True).items():
                    if cfg[k]!=expected:raise ValueError(f'adopted variant mismatch: {k}')
                job.update(adopted)
                job.update(variant=variant,adopted_from=str(previous))
            save(out / 'jobs' / f'{key}.json', job)
    print(json.dumps(dict(out=str(out), planned_cells=planned, reference_source_changes=changed)))


def run(out):
    out = out.resolve()
    protocol = verify_suite(out)
    parent = Path(protocol['parent'])
    old = verify_suite(parent)
    source = Path(protocol['source'])
    lock = (out / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manage_baselines=protocol.get('manage_baselines',True)
    if manage_baselines:
        parent_lock = (parent / 'supervisor.lock').open('a')
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    mutex = threading.RLock()
    stop = threading.Event()
    status = dict(phase='running', supervisor_pid=os.getpid(), started_unix=time.time(), planned_cells=protocol['planned_cells'], lanes={})

    def update(gpu=None, **fields):
        with mutex:
            (status if gpu is None else status['lanes'].setdefault(str(gpu), {})).update(fields)
            status['updated_unix'] = time.time()
            jobs = [json.loads(f.read_text()) for f in sorted((out/'jobs').glob('*.json'))]
            status['counts'] = dict(Counter(j['outcome'] for j in jobs))
            save(out/'status.json', status)
            measurements = []
            for j in jobs:
                row = dict(key=j['key'], model=j['model'], batch=j['batch'], variant=j['variant']['name'],
                           model_top_k=j['variant']['top_k'],suffix_q=j['variant'].get('suffix_q',64),
                           gpu=j['gpu'], outcome=j['outcome'], tok_s=None, wall_s=None, requests=None,
                           output_tokens=None, same_as_serial_ar=None, rows_per_token=None,
                           wasted_row_fraction=None, readout_rows_per_token=None, graph_hit_fraction=None,
                           completed_requests=None, error=j.get('error'))
                row['child_suffix_caps']=j['variant'].get('child_suffix_caps')
                row['c_initial']=j['variant'].get('c_initial',1.0)
                row['eta']=j['variant'].get('eta',.1)
                rank_metrics=[f'child_suffix_rank{rank}_{metric}' for rank in (1,2)
                              for metric in ('queries','returned_nodes','skipped')]
                for metric in [*rank_metrics,'suffix_query_s']:row[metric]=None
                for metric in ['scoring','controller_updates','requests_with_updates','c_final_mean',
                               'c_observed_min','c_observed_max','controller_s','peak_feedback_records']:
                    row[metric]=None
                if j['outcome']=='complete':
                    target=Path(j['output']); prefix=f"{j['arm']}-b{j['batch']}-seed0"
                    summary=json.loads((target/(prefix+'-summary.json')).read_text())
                    records=[json.loads(l) for l in (target/(prefix+'.jsonl')).read_text().splitlines()]
                    row.update(tok_s=summary['output_tput'], wall_s=summary['wall_s'], requests=summary['requests'],
                               output_tokens=summary['output_tokens'], same_as_serial_ar=summary['same_as_serial_ar'],
                               rows_per_token=summary['core_rows']/summary['output_tokens'],
                               wasted_row_fraction=sum(r['stats']['wasted_rows'] for r in records)/max(1,summary['core_rows']),
                               readout_rows_per_token=sum(r['stats']['readout_rows'] for r in records)/summary['output_tokens'],
                               graph_hit_fraction=summary['cuda_graph_replays']/max(1,summary['cuda_graph_replays']+summary['cuda_graph_eager_misses']),
                               completed_requests=summary['requests'])
                    for metric in rank_metrics:
                        row[metric]=sum(r['stats'].get(metric,0) for r in records)
                    row['suffix_query_s']=sum(r['stats'].get('speculate_s',0) for r in records)
                    row.update(scoring=summary['config'].get('scoring'),
                        controller_updates=sum(r['stats'].get('controller_updates',0) for r in records),
                        requests_with_updates=sum(r['stats'].get('controller_updates',0)>0 for r in records),
                        c_final_mean=sum(r['stats'].get('c_final',1) for r in records)/len(records),
                        c_observed_min=min(r['stats'].get('c_observed_min',1) for r in records),
                        c_observed_max=max(r['stats'].get('c_observed_max',1) for r in records),
                        controller_s=sum(r['stats'].get('controller_s',0) for r in records),
                        peak_feedback_records=max(r['stats'].get('peak_feedback_records',0) for r in records))
                elif j.get('output'):
                    pf=Path(j['output'])/f"{j['arm']}-b{j['batch']}-seed0-progress.json"
                    if pf.exists(): row['completed_requests']=json.loads(pf.read_text())['completed_requests']
                measurements.append(row)
            save(out/'summary.json', dict(updated_unix=time.time(),counts=status['counts'],measurements=measurements))
            temp=out/'measurements.csv.tmp'
            with temp.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(measurements[0]));writer.writeheader();writer.writerows(measurements)
            temp.replace(out/'measurements.csv')
            if manage_baselines:
                summarize(parent)
                save(parent/'status.json',dict(phase='baselines_managed_by_correction',supervisor_pid=os.getpid(),
                     manager=str(out/'status.json'),updated_unix=time.time(),lanes=status['lanes']))

    def heartbeat():
        while not stop.wait(15):
            try:
                snapshots=[gpu_snapshot(l['gpu']) for l in protocol['lanes']]
                with (out/'gpu-telemetry.jsonl').open('a') as f:
                    f.write(json.dumps(dict(time=time.time(),gpus=snapshots))+'\n')
                update()
            except Exception as e:
                update(heartbeat_error=repr(e))

    def expected(model, smoke=False):
        rows=[json.loads(l) for l in Path(protocol['inputs'][model]).read_text().splitlines()]
        ids=sorted(r['question_id'] for r in rows)
        return set(ids[:8] if smoke else ids)

    def finish(job, target, rc, smoke=False):
        outcome=classify(rc, target) if rc is not None else ('complete' if (target/'complete.json').exists() else 'failed')
        if outcome=='complete':
            summaries=audit_run(target,expected(job['model'],smoke),True)
            for s in summaries:
                if s.get('cap_skips') or s.get('attention_policy')!='packed_prefix_and_tree' or not s.get('cuda_graph'):
                    raise ValueError('capacity skips or backend mismatch')
            if 'variant' in job:
                v=job['variant'];cfg=summaries[0]['config']
                checks=dict(draft_node_budget=v['nodes'],use_suffix=False) if job['arm']=='loopspec-capped' else dict(top_m=v['top_k'],draft_node_budget=v['nodes'],
                                  child_suffix_policy=v['child'],positive_only=v['positive'],
                                  suffix_room=v.get('suffix_q',64),
                                  use_suffix=v['arm']!='budget-shallow')
                # Q has no meaning in shallow-only; reused control retains its old placeholder.
                if job['arm']=='budget-shallow':checks.pop('suffix_room')
                if 'child_suffix_caps' in v:
                    checks['child_suffix_caps']=v['child_suffix_caps']
                if job['arm'].startswith('budget-'):
                    checks['scoring']={'budget-online':'online_suffix_i','budget-fixed':'fixed_suffix'}.get(job['arm'],'raw')
                    for k in ('eta','c_min','c_max','c_initial'):
                        if k in v:checks[k]=v[k]
                for k,want in checks.items():
                    if cfg[k]!=want: raise ValueError(f'config mismatch {k}')
                updates=0
                for f in target.glob('*-seed0.jsonl'):
                    for l in f.read_text().splitlines():
                        stats=json.loads(l)['stats']
                        updates+=stats.get('controller_updates',0)
                        if job['arm']=='budget-fixed' and (stats.get('controller_updates',0) or
                            stats.get('feedback_created',0) or abs(stats['c_final']-v['c_initial'])>1e-12):
                            raise ValueError('fixed coefficient changed or collected feedback')
                        if stats['peak_draft_nodes']>v['nodes'] or stats['max_rows']>v['nodes']+1:
                            raise ValueError('node budget exceeded')
                if job['arm']=='budget-online' and not smoke and updates==0:
                    raise ValueError('online run completed without any controller feedback')
        return outcome

    def baseline_cmd(lane, job):
        templates=[json.loads(f.read_text()) for f in (parent/'jobs').glob('*.json')]
        return baseline_command(templates,lane,job,parent)

    def correction_cmd(lane, job, smoke):
        cmd=baseline_cmd(lane,job)
        cmd[cmd.index(str(Path(old['source'])/'scripts/loop_batch_bench.py'))]=str(source/'scripts/loop_batch_bench.py')
        target=out/('validation' if smoke else 'runs')/job['key']
        refs=out/('validation/references' if smoke else 'references')/f"{job['model']}.jsonl"
        v=job['variant']
        for option,value in [('--questions',protocol['inputs'][job['model']]),('--out',target),
                             ('--reference-file',refs),('--draft-node-budget',v['nodes']),
                             ('--max-new-tokens',16 if smoke else 1024)]:
            replace_option(cmd,option,value)
        cmd+=['--model-top-k',str(v['top_k']),'--child-suffix-policy',v['child'],'--progress-every','16']
        if '--suffix-max-draft' in cmd:
            replace_option(cmd,'--suffix-max-draft',v.get('suffix_q',64))
        else:
            cmd+=['--suffix-max-draft',str(v.get('suffix_q',64))]
        if v['positive']:cmd+=['--positive-only']
        if 'child_suffix_caps' in v:
            cmd+=['--child-suffix-caps',*[str(q) for q in v['child_suffix_caps']]]
        for k,option in [('eta','--suffix-eta'),('c_min','--suffix-c-min'),('c_max','--suffix-c-max')]:
            if k in v:replace_option(cmd,option,v[k])
        if 'c_initial' in v:cmd+=['--suffix-c-initial',str(v['c_initial'])]
        if smoke:cmd+=['--limit','8']
        return cmd,target

    def launch(lane, job, smoke=False, baseline=False):
        root=parent if baseline else out
        frozen=Path(old['source']) if baseline else source
        verify_suite(root)
        if baseline:
            cmd=baseline_cmd(lane,job);target=parent/'runs'/job['key']
        else:
            cmd,target=correction_cmd(lane,job,smoke)
        log=root/'logs'/(('validation-' if smoke else '')+job['key']+'.log')
        if target.exists() or log.exists(): raise RuntimeError(f'preserving existing target: {target}')
        check_resources(lane)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=lane['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 PYTHONPATH=str(frozen/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        env.pop('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',None)
        job.update(command=cmd,output=str(target),log=str(log),started_unix=time.time(),
                   outcome='validating' if smoke else 'running')
        with log.open('x') as f:
            child=subprocess.Popen(cmd,cwd=frozen,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            job['pid']=child.pid
            save(root/'jobs'/f"{job['key']}.json",job)
            update(lane['gpu'],phase=job['outcome'],current_cell=job['key'],pid=child.pid,baseline=baseline)
            rc=child.wait()
        result=finish(job,target,rc,smoke)
        if result!='complete':job['error']=log.read_text()[-4000:]
        if smoke:
            save(out/'validation'/f"{job['key']}-validation.json",dict(outcome=result,returncode=rc,output=str(target)))
        job.update(outcome=result,returncode=rc,finished_unix=time.time())
        return result

    def worker(lane):
        gpu=lane['gpu']
        with open(f'/tmp/loopfill-suite-gpu{gpu}.lock','a') as guard:
            if protocol.get('wait_for_gpu'):
                update(gpu,phase='waiting_for_prior_gpu_queue',current_cell=None,pid=None)
                fcntl.flock(guard,fcntl.LOCK_EX)
            else:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if protocol['version'].startswith('matched-node-budget'):
                check_resources(lane)
                update(gpu,phase='cuda_correctness_gate',current_cell=None,pid=None)
                environment=protocol['model_revisions']['huginn-3.5b'][2]
                cmd=['taskset','--cpu-list',lane['cpus'],f'/root/.cache/loopfill/{environment}/bin/python',
                     '-m','pytest',*protocol.get('cuda_correctness_tests', ['tests/test_loop_budget_alignment.py']),'-k','cuda','-q']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=lane['uuid'],
                         PYTHONPATH=str(source/'src')+os.pathsep+str(source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                log=out/'validation'/f'gpu{gpu}-correctness-tests.log'
                with log.open('x') as handle:
                    result=subprocess.run(cmd,cwd=source,env=env,stdout=handle,stderr=subprocess.STDOUT)
                save(out/'validation'/f'gpu{gpu}-correctness-tests.json',dict(returncode=result.returncode,command=cmd))
                if result.returncode:raise RuntimeError(f'GPU{gpu} capped LoopSpec tests failed; see {log}')
            # Adopt live baselines without restarting or changing their command/source.
            for path in ((parent/'jobs').glob('*.json') if manage_baselines else []):
                j=json.loads(path.read_text())
                if j['gpu']!=gpu or j['outcome']!='running':continue
                if j['arm'] not in BASELINES:raise RuntimeError('uncancelled old method job')
                update(gpu,phase='adopted_baseline',current_cell=j['key'],pid=j['pid'],baseline=True)
                while live(j['pid']):time.sleep(2)
                try:j['outcome']=finish(j,Path(j['output']),None)
                except Exception as e:j.update(outcome='failed',error=repr(e))
                j.update(adopted_by=os.getpid(),finished_unix=time.time());save(path,j);update()
            # Explicit job assignment permits independent configurations on idle GPUs.
            order={v['name']:i for i,v in enumerate(protocol['variants'])}
            assigned=[p for p in (out/'jobs').glob('*.json')
                      if json.loads(p.read_text())['gpu']==gpu]
            assigned.sort(key=lambda p:order[json.loads(p.read_text())['variant']['name']])
            for path in assigned:
                job=json.loads(path.read_text())
                if job['outcome']=='reuse_pending':
                    control=json.loads(Path(job['reuse_from']).read_text())
                    try:
                        if control['outcome']!='complete':raise RuntimeError('Q-suite control did not complete')
                        provenance=json.loads((out/'reuse-provenance.json').read_text())
                        meta=json.loads((Path(control['output'])/'metadata.json').read_text())
                        if meta['source_sha256']!=provenance['control_source_sha256']:
                            raise ValueError('Q control measured source differs from audited source')
                        if control['gpu_uuid']!=job['gpu_uuid']:
                            raise ValueError('Q control GPU differs')
                        job['output']=control['output']
                        job['outcome']=finish(job,Path(job['output']),None)
                        job.update(reused=True,finished_unix=time.time())
                    except Exception as e:job.update(outcome='failed',error=repr(e))
                    save(path,job);update()
                    continue
                if job['outcome']=='running' and job.get('adopted_from'):
                    update(gpu,phase='adopted_correction',current_cell=job['key'],pid=job['pid'],baseline=False)
                    while live(job['pid']):time.sleep(2)
                    try:job['outcome']=finish(job,Path(job['output']),None)
                    except Exception as e:job.update(outcome='failed',error=repr(e))
                    job.update(finished_unix=time.time(),adopted_by=os.getpid())
                    save(path,job)
                    previous=Path(job['adopted_from'])
                    old_job=json.loads(previous.read_text())
                    old_job.update(outcome=job['outcome'],finished_unix=job['finished_unix'],managed_by=str(out))
                    if job.get('error'):old_job['error']=job['error']
                    save(previous,old_job)
                    update()
                if job['outcome']!='pending':continue
                try:
                    result=launch(lane,job,smoke=True)
                    if result=='complete':
                        job['validation_output']=job['output'];job.pop('error',None)
                        launch(lane,job)
                    else:job['outcome']='validation_failed'
                except Exception as e:job.update(outcome='failed',error=repr(e),finished_unix=time.time())
                save(path,job);update()
            for batch in (lane['batches'] if manage_baselines else []):
                for arm in BASELINES:
                    path=parent/'jobs'/f"{lane['model']}-b{batch}-{arm}.json"
                    job=json.loads(path.read_text())
                    if job['outcome']!='pending':continue
                    try:launch(lane,job,baseline=True)
                    except Exception as e:job.update(outcome='failed',error=repr(e),finished_unix=time.time())
                    save(path,job);update()
            update(gpu,phase='complete',current_cell=None,pid=None)

    update()
    threading.Thread(target=heartbeat,daemon=True).start()
    errors=[]
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(worker,lane) for lane in protocol['lanes']]):
                try:future.result()
                except Exception as e:errors.append(repr(e));update(errors=errors)
        update(phase='finished_with_errors' if errors or status['counts'].get('failed') or status['counts'].get('validation_failed') else 'complete',errors=errors)
    finally:stop.set()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--parent',type=Path,default=Path('results/loopfill-optimized480-20261104'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--focus-huginn-b8',action='store_true')
    parser.add_argument('--resume-correction',type=Path)
    parser.add_argument('--budget-alignment',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.parent,args.out,args.focus_huginn_b8,args.resume_correction,args.budget_alignment)
    else:run(args.out)
