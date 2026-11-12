"""Freeze the selected online setting and fill the nine-point Huginn batch curve."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from loop_batch_sweep import BATCHES, check_resources
from loop_capacity_suite import digest, save, verify_suite
from loop_online_suite import audit_run


def prepare(previous: Path, out: Path):
    previous, out = previous.resolve(), out.resolve()
    old = verify_suite(previous)
    if out.exists():
        raise FileExistsError(out)
    model = 'huginn-3.5b'
    chosen = json.loads((previous / 'jobs' / f'{model}-b8-online-c07-eta01.json').read_text())
    if chosen['outcome'] != 'complete':
        raise ValueError('Selected B8 measurement is incomplete')
    measured = Path(chosen['output'])
    meta = json.loads((measured / 'metadata.json').read_text())
    expected = {json.loads(line)['question_id'] for line in Path(old['inputs'][model]).read_text().splitlines()}
    assert len(expected) == 480
    audit_run(measured, expected, True)
    cfg = meta['configs']['budget-online']
    for key, value in dict(top_m=2, draft_node_budget=16, suffix_room=4,
                           child_suffix_caps=[4, 4], c_initial=.7, eta=.1,
                           scoring='online_suffix_i', trace_controller=False).items():
        if cfg[key] != value:
            raise ValueError(f'Selected measurement differs: {key}')
    for name, sha in meta['source_sha256'].items():
        if digest(Path(old['source']) / name) != sha:
            raise ValueError(f'Measured source differs: {name}')
    for key, value in dict(max_new_tokens=1024, order_seeds='0', repeat_id=0,
                           schedule='continuous', attention_backend='packed',
                           prefill_chunk_size=128, cuda_graph=True, warmup_tokens=32,
                           dtype='bfloat16', limit=None).items():
        if meta['args'][key] != value:
            raise ValueError(f'Selected execution differs: {key}')
    parent = verify_suite(Path(old['parent']))
    lanes = []
    missing = [b for b in BATCHES if b != 8]
    for gpu, batch in enumerate(missing):
        lane = copy.deepcopy(next(l for l in parent['lanes'] if l['gpu'] == gpu))
        lane.update(model=model, batches=[batch])
        check_resources(lane)
        lanes.append(lane)

    source = out / 'source'
    shutil.copytree(old['source'], source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder in ('jobs', 'runs', 'logs', 'inputs', 'references', 'validation'):
        (out / folder).mkdir()
    inputs = {}
    for name, path in old['inputs'].items():
        target = out / 'inputs' / f'{name}.jsonl'
        shutil.copy2(path, target)
        inputs[name] = str(target)
    for path in (previous / 'references').glob('*.jsonl'):
        shutil.copy2(path, out / 'references' / path.name)
    shutil.copytree(previous / 'validation/references', out / 'validation/references')

    # Only orchestration changes: an allocation OOM in the short check must not
    # prune the requested full480 attempt. Ordinary correctness failures still block.
    runner = source / 'scripts/loop_spec_correction_suite.py'
    text = runner.read_text()
    before = "                    if result=='complete':\n                        job['validation_output']=job['output'];job.pop('error',None)"
    after = "                    if result in ('complete', 'oom'):\n                        job['validation_outcome']=result\n                        job['validation_output']=job['output'];job.pop('error',None)"
    assert text.count(before) == 1
    runner.write_text(text.replace(before, after))
    # All code covered by the benchmark's source identity stays byte-identical.
    for name, sha in meta['source_sha256'].items():
        assert digest(source / name) == sha

    variant = chosen['variant']
    protocol = copy.deepcopy(old)
    for key in ('pairs', 'paired_controls', 'zero_policy', 'rank_policy', 'followup_policy'):
        protocol.pop(key, None)
    protocol.update(version='matched-node-budget-online-batch-huginn-v1',
        source=str(source), created_unix=time.time(), inputs=inputs, lanes=lanes,
        variants=[variant], planned_cells=9, new_cells=8, reused_cells=1,
        batches=BATCHES, control_suite=str(previous), manage_baselines=False,
        priority='Each missing batch on one exclusive GPU; reuse matching complete B8',
        controller=dict(c_initial=.7, eta=.1, c_min=.25, c_max=2., scope='request', trace=False),
        selection_context='User selected online c0=.7 eta=.1. Freeze Q4 N16 top2 ranks(4,4) across all batches: N16 leads N8/32/64 at B8 Q4; Q4 N16 is stronger than N32 at B16, and has a measured warm-start online B8 result. Q8 N16 raw has higher token/s but longer wall time at B8. This is an evidence-backed fixed choice, not an established global optimum.',
        baseline_policy='Existing baseline artifacts retained; this invocation fills only the selected online curve.',
        failure_policy='Attempt each full480 batch even if its short check hits actual CUDA OOM; never reduce batch, N, prompts, or output budget. Non-OOM correctness failures block that cell.',
        development_experiment=True,
        data_scope='Full480 formal workload after selection on the same data; not a held-out confirmation.')
    save(out / 'protocol.json', protocol)
    for lane in lanes:
        batch = lane['batches'][0]
        key = f'{model}-b{batch}-{variant["name"]}'
        save(out / 'jobs' / f'{key}.json', dict(key=key, model=model, batch=batch,
             arm='budget-online', variant=variant, gpu=lane['gpu'], gpu_uuid=lane['uuid'], outcome='pending'))
    chosen.update(reused=True, reused_from=str(previous / 'jobs' / f'{chosen["key"]}.json'))
    save(out / 'jobs' / f'{chosen["key"]}.json', chosen)
    save(out / 'reuse-provenance.json', dict(source_suite=str(previous),
         reused_job=chosen['key'], original_output=str(measured),
         execution_source_sha256=meta['source_sha256'],
         measurement_hashes={str(p): digest(p) for p in measured.iterdir() if p.is_file()},
         input_identical=digest(inputs[model]) == meta['questions_sha256'],
         reference_sha256=digest(out / 'references' / f'{model}.jsonl'),
         source_changes=['scripts/loop_spec_correction_suite.py: preserve full480 attempts following smoke allocation OOM']))
    save(out / 'source-hashes.json', {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()})
    save(out / 'input-hashes.json', {str(p): digest(p) for folder in ('inputs', 'references', 'validation/references') for p in (out / folder).glob('*.jsonl')})
    save(out / 'prelaunch-audit.json', dict(planned=9, pending=8, reused=[8],
         source_verified=True, reference_bytes_identical=True, config=cfg,
         prior_validation=str(previous / 'prelaunch-audit.json')))
    verify_suite(out)
    print(json.dumps(dict(out=str(out), pending=missing, reused=[8])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('results/loopfill-online-tuning-huginn-b8-20261104'))
    parser.add_argument('--out', type=Path, default=Path('results/loopfill-online-batch-huginn-20261104'))
    args = parser.parse_args()
    prepare(args.previous, args.out)
