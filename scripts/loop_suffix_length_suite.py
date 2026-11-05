"""Freeze and launch two match-length arms against the existing Ouro eager protocol."""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import loop_eager_suite as eager
from loop_batch_bench import arm_config, parser
from loop_batch_sweep import check_resources
from loop_capacity_suite import digest, save, verify_suite

ARMS = ['budget-fixed-length', 'budget-online-length']
REPO = Path(__file__).resolve().parents[1]


def migrate_reference(previous, out, source_hashes, folder):
    old = json.loads((previous / 'protocol.json').read_text())
    origin = previous / folder / 'ouro-2.6b.jsonl'
    lines = origin.read_text().splitlines(keepends=True)
    header = json.loads(lines[0])
    old_hashes = header['reference_identity']['source_sha256']
    assert old_hashes == old['execution_source_sha256']
    assert set(old_hashes) == set(source_hashes)
    changed = {k for k in old_hashes if old_hashes[k] != source_hashes[k]}
    assert changed <= {'scripts/loop_batch_bench.py', 'src/llm_memory_editability/loop_budget_decode.py'}
    def functions(path):
        return {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    before = functions(Path(old['source']) / 'scripts/loop_batch_bench.py')
    after = functions(out / 'source/scripts/loop_batch_bench.py')
    assert before.keys() == after.keys()
    checked = sorted(set(before) - {'parser', 'arm_config'})
    assert all(before[k] == after[k] for k in checked), 'Serial reference generation changed'
    assert asdict(arm_config('ar', 4, 1, 2, 4)) == old['configs']['ar']
    header['reference_identity']['source_sha256'] = source_hashes
    target = out / folder / origin.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(header) + '\n' + ''.join(lines[1:]))
    assert target.read_text().splitlines()[1:] == origin.read_text().splitlines()[1:]
    return dict(origin=str(origin), origin_sha256=digest(origin), target=str(target),
                target_sha256=digest(target), changed_sources=sorted(changed),
                unchanged_benchmark_functions=checked, ar_config_unchanged=True,
                tokens_unchanged=True, reason='Only budget scoring and its CLI wiring changed; '
                'full benchmark main including serial generator, AR config and all AR backend modules unchanged.')


def prepare(previous, out, gpus):
    previous, out = previous.resolve(), out.resolve()
    old = verify_suite(previous)
    assert old['execution_args']['model'] == 'ouro-2.6b'
    assert old['batches'] == [1, 2, 4, 8, 16] and old['repeats'] == [0]
    if out.exists():
        raise FileExistsError(out)
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError('GPU lanes must be nonempty and unique')
    lanes = [next(dict(lane) for lane in old['lanes'] if lane['gpu'] == gpu) for gpu in gpus]
    for lane in lanes:
        check_resources(lane)
    source = out / 'source'
    for folder in ('src', 'scripts', 'configs', 'tests'):
        shutil.copytree(REPO / folder, source / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for folder in ('jobs', 'runs', 'logs', 'validation', 'inputs', 'references'):
        (out / folder).mkdir(exist_ok=True)
    questions = out / 'inputs/ouro-2.6b.jsonl'
    shutil.copy2(old['questions'], questions)
    assert digest(questions) == old['questions_sha256']
    hashes = {str(p.relative_to(source)): digest(p) for p in
              [source / 'scripts/loop_batch_bench.py'] + sorted((source / 'src/llm_memory_editability').glob('*.py'))}
    refs = [migrate_reference(previous, out, hashes, f)
            for f in ('references', 'validation/references')]
    save(out / 'reference-provenance.json', refs)
    configs = {arm: asdict(arm_config(arm, 4, 1, 2, 4, draft_node_budget=16,
               model_top_k=2, child_suffix_caps=(4, 4), suffix_eta=.1,
               suffix_c_initial=1.)) for arm in ARMS}
    template = list(old['template_command'])
    template[4] = str(source / 'scripts/loop_batch_bench.py')
    template = ['@' + str(source / 'configs/loopfill-eager.args') if a.startswith('@') else a for a in template]
    protocol = dict(old, version='ouro-two-length-arms-eager-b1-b16-v1', created_unix=time.time(),
        parent=str(previous), source=str(source), questions=str(questions),
        execution_source_sha256=hashes, configs=configs, template_command=template,
        lanes=lanes, arms=ARMS, planned_cells=10,
        scheduling='Shared queue on explicitly selected idle GPUs; B1 first, then B2/B4/B8/B16.',
        execution_changes=['Continuous length weighting; fixed c=1 or online c0=1; no other decoding/engine setting changed.'],
        reference_policy='Migrate serial reference identity only after unchanged main/serial helpers, AR configuration and backend audit.',
        reuse_policy='Ten new measurements; existing raw/online measurements retained in parent suite, no rerun.',
        model_transfer=None, prior_extra_arm=None,
        length_weighting=dict(formula='p=c*q*(1+L)/(1+c*q*(1+L))', initial=1., eta=.1,
            bounds=[.25, 2.], offline_fitting=False, per_request_reset=True,
            fixed_control='budget-fixed-length', learned='budget-online-length'),
        comparison_limitations='Same workload and engine settings, but new scoring source revision and single timings; '
                               'do not attribute small timing differences solely to the algorithm.')
    save(out / 'protocol.json', protocol)
    for batch in protocol['batches']:
        for arm in ARMS:
            job = dict(key=eager.job_key(protocol, batch, arm, 0), model='ouro-2.6b', batch=batch,
                       arm=arm, repeat=0, gpu=None, gpu_uuid=None, outcome='pending')
            cmd = eager.command(protocol, lanes[0], job, out)
            args = vars(parser().parse_args(cmd[5:]))
            assert all(args[k] == v for k, v in protocol['execution_args'].items())
            assert args['suffix_c_initial'] == 1 and args['arms'] == arm
            save(out / 'jobs' / f"{job['key']}.json", job)
    save(out / 'source-hashes.json', {str(p.relative_to(source)): digest(p)
                                    for p in source.rglob('*') if p.is_file()})
    save(out / 'input-hashes.json', {str(p): digest(p) for folder in
         ('inputs', 'references', 'validation/references') for p in (out / folder).glob('*.jsonl')})
    verify_suite(out)
    eager.summarize(out)
    print(json.dumps(dict(out=str(out), planned_cells=10, gpus=gpus)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'run', 'summarize'))
    p.add_argument('--previous', type=Path, default=Path('results/loopfill-eager-b1-b16-ouro-20261105'))
    p.add_argument('--out', type=Path, default=Path('results/loopfill-length-b1-b16-ouro-20261105'))
    p.add_argument('--gpus', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7])
    a = p.parse_args()
    if a.action == 'prepare':
        prepare(a.previous, a.out, a.gpus)
    elif a.action == 'run':
        eager.run(a.out)
    else:
        eager.summarize(a.out)
