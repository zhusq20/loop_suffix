"""Extend the frozen SQL replay to Ouro-1.4B after its full DFlash training."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from loop_agentic_worker import digest, digest_tokens, save


def read(path):
    return json.loads(Path(path).read_text())


def tokenize_requests(rows, tokenizer):
    """Keep the original selection, messages and order; never filter or truncate."""
    output = []
    for row in rows:
        ids = tokenizer.apply_chat_template(row['messages'], add_generation_prompt=True)
        if not ids or len(ids) > 3548:
            raise ValueError(f"Unsupported full context: {row['source_id']}")
        output.append(dict(**{k: v for k, v in row.items() if k != 'tokenizations'},
            prompt_token_ids=ids, prompt_tokens=len(ids), prompt_sha256=digest_tokens(ids),
            model='ouro-1.4b'))
    return output


def prepare(suite, parent, evaluation_plan, repository):
    from transformers import AutoTokenizer
    from loop_agentic_suite import ARMS, BASE

    parent_protocol = read(parent/'protocol.json')
    manifest = read(parent/'inputs/manifest.json')
    assert read(parent/'checks/final-audit.json')['passed']
    for folder, key in [('source', 'source_sha256'), ('inputs', 'input_files_sha256')]:
        for name, expected in parent_protocol[key].items():
            assert digest(parent/folder/name) == expected, name
    evaluation = read(evaluation_plan)
    training = Path(evaluation['training_root'])
    assert digest(training/'plan.json') == evaluation['training_plan_sha256']
    master = parent/'inputs/spider-repair100.jsonl'
    assert digest(master) == manifest['input_sha256']
    rows = [json.loads(line) for line in master.open()]
    assert len(rows) == len({r['source_id'] for r in rows}) == 100
    tokenizer = AutoTokenizer.from_pretrained(evaluation['target_path'],
        trust_remote_code=True, local_files_only=True)
    tokenized = tokenize_requests(rows, tokenizer)
    lad_root = BASE/'checkpoints/loopahead-ouro-1.4b-formal-v1'
    lad = Path(read(lad_root/'complete.json')['selected_checkpoint'])/'lad.pt'
    admission = BASE/'configs/loopahead-ouro-1.4b-admission-v1.json'
    assert read(admission)['model'] == 'ouro-1.4b'
    assert read(admission)['checkpoint_sha256'] == digest(lad)

    suite.mkdir(parents=True, exist_ok=False)
    for name in ['inputs', 'checks', 'logs', 'provenance']:
        (suite/name).mkdir()
    shutil.copy2(master, suite/'inputs'/master.name)
    shutil.copy2(admission, suite/'inputs/loopahead-admission.json')
    shutil.copy2(evaluation['draft_config'], suite/'inputs/dflash-expected-config.json')
    for name in ['protocol.json', 'inputs/manifest.json', 'checks/final-audit.json']:
        shutil.copy2(parent/name, suite/'provenance'/Path(name).name)
    input_path = suite/'inputs/ouro-1.4b.jsonl'
    input_path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in tokenized))
    lengths = [row['prompt_tokens'] for row in tokenized]
    manifest.update(model_paths={'ouro-1.4b': evaluation['target_path']},
        model_inputs={'ouro-1.4b': dict(path=input_path.name, sha256=digest(input_path),
            min_prompt_tokens=min(lengths), max_prompt_tokens=max(lengths),
            mean_prompt_tokens=sum(lengths)/len(lengths))},
        extension_parent=str(parent), selection=manifest['selection']+
            ' Ouro-1.4B extension reuses these exact 100 historical requests without reselection.')
    save(suite/'inputs/manifest.json', manifest)
    shutil.copytree(parent/'source', suite/'source',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['loop_agentic_worker.py', 'loop_agentic_suite.py',
                 'loop_agentic_finalize.py', 'loop_agentic_ouro14.py', 'loop_queue_recovery.py']:
        shutil.copy2(repository/'scripts'/name, suite/'source/scripts'/name)
    jobs = [dict(id=f'ouro-1.4b-{arm}-t{temperature}', model='ouro-1.4b', arm=arm,
                 temperature=temperature, q=4, n=16, cache_mode='stream', stage='main')
            for temperature in [0, 1] for arm in ARMS]
    save(suite/'jobs.json', jobs)
    plan = dict(created_unix=time.time(), parent_suite=str(parent), repository=str(repository),
        training_root=str(training), training_plan_sha256=digest(training/'plan.json'),
        evaluation_plan=str(evaluation_plan), evaluation_plan_sha256=digest(evaluation_plan),
        target_path=evaluation['target_path'], target_revision=evaluation['revision'],
        loopahead_checkpoint=str(lad), loopahead_sha256=digest(lad),
        jobs_sha256=digest(suite/'jobs.json'), expected_cells=14, expected_requests=1400,
        gpu_bindings={'ouro-1.4b/t0': 5, 'ouro-1.4b/t1': 7}, allowed_gpus=[5, 7],
        output_divergence_diagnostics='cancelled_by_user',
        source_sha256={str(p.relative_to(suite/'source')): digest(p)
            for p in sorted((suite/'source').rglob('*')) if p.is_file()},
        input_files_sha256={p.name: digest(p) for p in sorted((suite/'inputs').iterdir())},
        provenance_sha256={p.name: digest(p) for p in sorted((suite/'provenance').iterdir())})
    save(suite/'plan.json', plan)
    save(suite/'checks/input-audit.json', dict(passed=True, requests=100,
        instances=len({r['instance_id'] for r in rows}), phase_counts=dict(Counter(r['phase'] for r in rows)),
        master_sha256=digest(master), unchanged_selection_messages_and_order=True,
        min_prompt_tokens=min(lengths), max_prompt_tokens=max(lengths),
        max_total_tokens=max(lengths)+512, truncated_or_excluded=0))
    save(suite/'status.json', dict(status='prepared_waiting_for_full_training',
        completed=0, total=14, updated_unix=time.time()))
    print(json.dumps(dict(status='prepared', suite=str(suite), cells=14, requests=1400)))


def verify(suite):
    plan = read(suite/'plan.json')
    for folder, key in [('source', 'source_sha256'), ('inputs', 'input_files_sha256'),
                        ('provenance', 'provenance_sha256')]:
        for name, expected in plan[key].items():
            assert digest(suite/folder/name) == expected, name
    assert digest(suite/'jobs.json') == plan['jobs_sha256']
    assert digest(Path(plan['training_root'])/'plan.json') == plan['training_plan_sha256']
    assert digest(plan['loopahead_checkpoint']) == plan['loopahead_sha256']
    return plan


def checked_checkpoint(plan, expected_config):
    training = Path(plan['training_root'])
    complete = read(training/'complete.json')
    report = read(training/'checks/training-audit.json')
    assert complete['status'] == 'complete' and report['status'] == 'passed'
    assert complete['parameters'] == report['parameters'] == 163597568
    for stage in ['teachers', 'smoke', 'train', 'audit']:
        launch, receipt = read(training/f'{stage}-launcher.json'), read(training/f'{stage}-exit.json')
        assert receipt['exit_code'] == 0
        assert all(receipt[k] == launch[k] for k in ['pid', 'argv', 'started_unix'])
    checkpoint = Path(report['checkpoint'])
    assert str(checkpoint) == complete['checkpoint']
    assert digest(checkpoint/'model.safetensors') == report['weights_sha256']
    assert digest(training/'training/metrics.jsonl') == report['metrics_sha256']
    assert digest(training/'training/metadata.json') == report['metadata_sha256']
    meta, done = read(training/'training/metadata.json'), read(training/'training/complete.json')
    assert meta['epochs'] == 6 and meta['draft_parameters'] == 163597568
    assert done['step'] == meta['total_updates'] == report['updates']
    assert not meta['bounded_experiment'] and not done['bounded_experiment']
    config = read(checkpoint/'config.json')
    assert config['architectures'] == ['ProjectedDFlashDraftModel']
    for key, value in expected_config.items():
        if key not in ['architectures', 'transformers_version']:
            assert config[key] == value, key
    return checkpoint, report


def run(suite, scheduling=None):
    from loop_agentic_suite import run_queue
    from loop_agentic_finalize import finalize
    with (suite/'activation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify(suite)
        training = Path(plan['training_root'])
        while not (training/'complete.json').exists():
            if (suite/'STOP').exists():
                save(suite/'status.json', dict(status='cancelled', completed=0, total=14))
                return
            if (training/'failure.json').exists():
                raise RuntimeError('DFlash training failed; SQL evaluation cannot start')
            save(suite/'status.json', dict(status='waiting_for_full_training',
                controller_pid=os.getpid(), updated_unix=time.time(), completed=0, total=14,
                dependency=str(training), running={}, bindings=plan['gpu_bindings']))
            time.sleep(30)
        plan = verify(suite)
        checkpoint, report = checked_checkpoint(plan, read(suite/'inputs/dflash-expected-config.json'))
        model = dict(model_path=plan['target_path'], target_revision=plan['target_revision'],
            input_sha256=plan['input_files_sha256']['ouro-1.4b.jsonl'],
            loopahead_checkpoint=plan['loopahead_checkpoint'], dflash_checkpoint=str(checkpoint),
            loopahead_admission='loopahead-admission.json',
            checkpoint_sha256=dict(loopahead=plan['loopahead_sha256'],
                dflash=digest(checkpoint/'model.safetensors'), dflash_config=digest(checkpoint/'config.json')))
        if not (suite/'protocol.json').exists():
            protocol = read(suite/'provenance/protocol.json')
            manifest = read(suite/'inputs/manifest.json')
            protocol.update(name=suite.name, created_unix=time.time(), models={'ouro-1.4b': model},
                source_sha256=plan['source_sha256'], input_files_sha256=plan['input_files_sha256'],
                selection=manifest['selection'], allowed_gpus=plan['allowed_gpus'],
                gpu_bindings=plan['gpu_bindings'], require_exit_receipts=True,
                scope='Ouro-1.4B extension: seven methods, temperatures 0/1, 100 fixed SQL requests per cell.',
                scheduling='Wait for the complete audited DFlash checkpoint; use free GPU 5/7 with UUID locks. '
                    'Each temperature stays on one GPU. Existing jobs are allowed to finish.',
                paper_export_suffix='-ouro-1.4b', output_divergence_diagnostics='cancelled_by_user',
                preparation_plan_sha256=digest(suite/'plan.json'),
                training_audit=dict(path=str(training/'checks/training-audit.json'),
                    sha256=digest(training/'checks/training-audit.json')))
            save(suite/'protocol.json', protocol)
            save(suite/'checks/training-binding.json', dict(checkpoint=str(checkpoint), report=report))
        else:
            assert read(suite/'protocol.json')['models'] == {'ouro-1.4b': model}
        run_queue(suite, scheduling=scheduling)
        if read(suite/'status.json')['status'] == 'complete':
            finalize(suite, Path(plan['repository']), scheduling=scheduling)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'verify'])
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--evaluation-plan', type=Path)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scheduling', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.suite.resolve(), args.parent.resolve(), args.evaluation_plan.resolve(), args.repository.resolve())
    elif args.action == 'verify':
        verify(args.suite.resolve())
        print('Frozen extension inputs and sources verified')
    else:
        try:
            run(args.suite.resolve(), read(args.scheduling) if args.scheduling else None)
        except Exception:
            save(args.suite/'failure.json', dict(error=traceback.format_exc(), failed_unix=time.time()))
            save(args.suite/'status.json', dict(status='needs_attention', updated_unix=time.time()))
            raise
