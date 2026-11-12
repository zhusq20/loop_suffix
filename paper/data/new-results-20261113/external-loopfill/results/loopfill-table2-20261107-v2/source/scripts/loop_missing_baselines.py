"""Reuse completed configurations across GPUs and fill only missing B1--16 cells."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import time

from loop_capacity_suite import digest, save, verify_suite
from loop_online_suite import audit_run

BATCHES = [1, 2, 4, 8, 16]
ARMS = ['ar', 'wavefront', 'loopspec-dual', 'loopspec-capped',
        'budget-suffix', 'budget-raw', 'budget-online']


def canonical_config(config):
    cfg = copy.deepcopy(config)
    if cfg['name'] == 'ar' and not cfg['use_fill'] and not cfg['use_suffix']:
        # AR never proposes any draft; this CLI placeholder does not execute.
        cfg['max_draft'] = 0
    if cfg['name'].startswith('budget-'):
        cfg.setdefault('c_initial', 1.)
        cfg.setdefault('child_suffix_caps', None)
        # Unrestricted per-rank queries use the same global Q for both ranks.
        if cfg['child_suffix_caps'] is None and cfg.get('child_suffix_policy') == 'all':
            cfg['child_suffix_caps'] = [cfg['suffix_room']] * cfg['top_m']
    if cfg['name'] == 'loopspec-dual':
        cfg.setdefault('draft_node_budget', None)
    return cfg


def compatible(meta, target, questions_sha):
    args = meta['args']
    execution = dict(model='huginn-3.5b', schedule='continuous', attention_backend='packed',
        prefill_chunk_size=128, cuda_graph=True, max_cuda_graphs=24,
        max_new_tokens=1024, max_seq=2816, buffer_policy='per-arm', dtype='bfloat16',
        warmup_tokens=32, seed=0, order_seeds='0', repeat_id=0, limit=None,
        check=True, profile=False, trace_controller=False)
    return (meta['questions_sha256'] == questions_sha
        and all(args.get(k) == v for k, v in execution.items())
        and canonical_config(meta['configs'][target['name']]) == canonical_config(target))


def prepare(previous, out):
    previous, out = previous.resolve(), out.resolve()
    old = verify_suite(previous)
    confirmation = json.loads((previous.parent / 'loopfill-confirm-b1-b16-huginn-20261104/protocol.json').read_text())
    if out.exists():
        raise FileExistsError(out)
    source = out / 'source'
    shutil.copytree(old['source'], source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Some original AR/WFD records predate waste counters. Preserve missingness.
    runner = source / 'scripts/loop_spec_correction_suite.py'
    original = runner.read_text()
    before = "wasted_row_fraction=sum(r['stats']['wasted_rows'] for r in records)/max(1,summary['core_rows']),"
    after = "wasted_row_fraction=(sum(r['stats']['wasted_rows'] for r in records)/max(1,summary['core_rows']) if all('wasted_rows' in r['stats'] for r in records) else None),"
    assert original.count(before) == 1
    updated = original.replace(before, after)
    validation = '                    result=launch(lane,job,smoke=True)'
    resume_validation = "                    smoke_target=out/'validation'/job['key']\n                    result=finish(job,smoke_target,0,smoke=True) if (smoke_target/'complete.json').exists() else launch(lane,job,smoke=True)"
    assert updated.count(validation) == 1
    runner.write_text(updated.replace(validation, resume_validation))
    shutil.copy2(Path(__file__).resolve(), source / 'scripts/loop_missing_baselines.py')
    test = Path(__file__).resolve().parents[1] / 'tests/test_loop_missing_baselines.py'
    shutil.copy2(test, source / 'tests' / test.name)
    for folder in ('jobs', 'runs', 'logs', 'inputs', 'references', 'validation'):
        (out / folder).mkdir()
    inputs = {}
    for model, path in old['inputs'].items():
        target = out / 'inputs' / f'{model}.jsonl'
        shutil.copy2(path, target)
        inputs[model] = str(target)
    for path in (previous / 'references').glob('*.jsonl'):
        shutil.copy2(path, out / 'references' / path.name)
    shutil.copytree(previous / 'validation/references', out / 'validation/references')
    expected = {json.loads(line)['question_id'] for line in Path(inputs['huginn-3.5b']).read_text().splitlines()}
    qsha = digest(inputs['huginn-3.5b'])
    candidates = {}
    # A source result can be referenced by several older suites: deduplicate by output path.
    for path in sorted(previous.parent.glob('*/jobs/*.json')):
        job = json.loads(path.read_text())
        if job.get('model') != 'huginn-3.5b' or job.get('batch') not in BATCHES or job.get('arm') not in ARMS:
            continue
        if job.get('outcome') not in ('complete', 'oom'):
            continue
        target = Path(job.get('output') or path.parent.parent / 'runs' / job['key'])
        if not (target / 'metadata.json').exists():
            continue
        meta = json.loads((target / 'metadata.json').read_text())
        if compatible(meta, confirmation['configs'][job['arm']], qsha):
            candidates.setdefault(str(target), (path, job, meta))
    variants = []
    for arm in ARMS:
        budget = arm.startswith('budget-') or arm == 'loopspec-capped'
        variants.append(dict(name=arm, arm=arm, top_k=2 if budget else None,
            nodes=16 if budget else None, child='all', positive=False,
            suffix_q=4, child_suffix_caps=[4, 4], c_initial=.7 if arm == 'budget-online' else 1.,
            eta=.1, c_min=.25, c_max=2.))
        if not arm.startswith('budget-'):
            variants[-1].pop('child_suffix_caps')
    jobs, reused, missing = [], [], []
    for batch in BATCHES:
        for variant in variants:
            arm = variant['arm']
            job = dict(key=f'huginn-3.5b-b{batch}-{arm}', model='huginn-3.5b', batch=batch,
                arm=arm, variant=variant, outcome='pending')
            matches = [(path, j, m) for path, j, m in candidates.values() if j['batch'] == batch and j['arm'] == arm]
            # Prefer the current inference source when available, never the fastest measurement.
            matches.sort(key=lambda entry: (
                entry[1]['outcome'] != 'complete',
                entry[2]['source_sha256'] != confirmation['execution_source_sha256'], str(entry[0])))
            if matches:
                path, original, meta = matches[0]
                target = Path(original.get('output') or path.parent.parent / 'runs' / original['key'])
                if original['outcome'] == 'complete':
                    summary, = audit_run(target, expected, True)
                    if summary['cap_skips'] or summary['attention_policy'] != 'packed_prefix_and_tree':
                        raise ValueError(f'Invalid existing measurement: {target}')
                else:
                    failure = json.loads((target / 'failure.json').read_text())
                    assert failure['exception_type'] == 'OutOfMemoryError' and failure['outcome'] == 'oom'
                measured_source = Path(meta['args']['out']).parent.parent / 'source'
                # Validate the actual measured revision, without requiring it to equal today's revision.
                for name, sha in meta['source_sha256'].items():
                    assert digest(measured_source / name) == sha, (target, name)
                for name in ['src/llm_memory_editability/loop_tree_decode.py',
                             'src/llm_memory_editability/loop_optimized_decode.py']:
                    assert digest(measured_source / name) == digest(source / name), name
                changed = [name for name, sha in meta['source_sha256'].items() if digest(source / name) != sha]
                job.update(outcome=original['outcome'], gpu=original['gpu'], gpu_uuid=original['gpu_uuid'],
                    output=str(target), reused=True, reuse_from=str(path))
                reused.append(dict(key=job['key'], source_job=str(path), output=str(target),
                    outcome=job['outcome'], original_gpu=job['gpu'], changed_source_files=changed,
                    original_config=meta['configs'][arm], canonical_config=canonical_config(meta['configs'][arm]),
                    progress_every=meta['args'].get('progress_every', 0),
                    source_sha256=meta['source_sha256'],
                    artifact_sha256={str(f): digest(f) for f in target.iterdir() if f.is_file()}))
            else:
                missing.append(job)
            jobs.append(job)
    assert len(missing) <= len(old['lanes'])
    lanes = []
    for index, job in enumerate(missing):
        lane = copy.deepcopy(next(l for l in old['lanes'] if l['gpu'] == index))
        lane.update(model='huginn-3.5b', batches=[job['batch']])
        lanes.append(lane)
        job.update(gpu=lane['gpu'], gpu_uuid=lane['uuid'])
    protocol = copy.deepcopy(old)
    protocol.update(version='matched-node-budget-missing-baselines-huginn-v1', source=str(source),
        created_unix=time.time(), inputs=inputs, variants=variants, lanes=lanes,
        planned_cells=len(jobs), new_cells=len(missing), reused_cells=len(reused), batches=BATCHES,
        manage_baselines=False, wait_for_gpu=True, control_suite=str(previous),
        priority='User correction: reuse completed matching configurations regardless of physical GPU; no timing repeats',
        baseline_policy='Reuse AR/WFD/native LoopSpec, matched Q4 N16 raw/suffix and N16 capped LoopSpec when available; retain source/GPU/instrumentation differences in provenance.',
        reuse_policy='Input, model, precision, optimized backend, output budget and effective arm parameters must match. GPU and whole-source identity may differ. Never choose the fastest prior run.',
        source_review='Tree/model backend and packed backend are byte-identical. Historical changes add budget variants/controller initialization/rank query caps, progress callbacks and extra counters. Default child caps None means global Q4 at every rank, equivalent to [4,4]. AR max_draft is inactive with both proposals disabled. Historical timing instrumentation differs; reused comparisons are not same-source or same-GPU paired repeats.',
        uncertainty='One measured stream per cell; no newly requested repeats or standard-error/significance claim. Data was used for parameter selection; no held-out confirmation.',
        failure_policy='Reuse previously recorded actual OOM; do not retry unchanged failures. Every missing cell gets one full480 attempt, after short validation. No capacity pruning of other cells.')
    save(out / 'protocol.json', protocol)
    for job in jobs:
        save(out / 'jobs' / f"{job['key']}.json", job)
    save(out / 'reuse-provenance.json', dict(reused=reused, review=protocol['source_review']))
    save(out / 'source-hashes.json', {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
    save(out / 'input-hashes.json', {str(p): digest(p) for folder in ('inputs', 'references', 'validation/references') for p in (out / folder).glob('*.jsonl')})
    verify_suite(out)
    print(json.dumps(dict(out=str(out), counts=dict(Counter(j['outcome'] for j in jobs)),
                         missing=[j['key'] for j in missing]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('results/loopfill-online-batch-huginn-20261104'))
    parser.add_argument('--out', type=Path, default=Path('results/loopfill-missing-b1-b16-huginn-20261104'))
    args = parser.parse_args()
    prepare(args.previous, args.out)
