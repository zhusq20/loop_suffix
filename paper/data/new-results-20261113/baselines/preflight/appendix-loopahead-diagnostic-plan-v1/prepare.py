"""Freeze six completed appendix greedy runs for separate numerical replay."""
import csv
from pathlib import Path
import shutil
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, save, sha, compare, rows
from diagnose_formal_loopahead import prepare
from orchestrate_baseline_diagnostics_resume import verify_plan

root = BASE / 'logs/appendix-loopahead-diagnostics-v1'
original = BASE / 'logs/baseline-diagnostics-resume-v1'
previous = verify_plan(original / 'plan.json')
records = [r for r in csv.DictReader((BASE / 'results.csv').open())
           if (r['method'], r['model'], r['temperature']) == ('loopahead', 'ouro-1.4b', '0')]
assert len(records) == 6
assert {r['benchmark'] for r in records} == {'gsm8k', 'math500', 'bbh', 'humanevalplus', 'mbppplus', 'specbench'}
agreements = read(BASE / 'formal-agreement.json')
groups, evidence = [], {}
for row in records:
    identity = f"loopahead-ouro-1.4b-{row['benchmark']}-v1"
    assert identity not in {g['id'] for g in previous['groups']}
    ar, method = Path(row['ar_run']), Path(row['method_run'])
    agreement = compare(rows(method / 'requests.jsonl'), rows(ar / 'requests.jsonl'))
    assert agreement == agreements['/'.join([row['engine'], row['model'], row['benchmark'], '0', 'loopahead'])]
    meta, cases = prepare(ar, method)
    assert [(r['source_id'], r['position']) for r in cases] == [
        (r['source_id'], r['position']) for r in agreement['first_divergences']]
    receipt = BASE / f'logs/{method.name}-finished.json'
    assert read(receipt)['exit_code'] == 0
    folder = BASE / 'preflight/formal-greedy-margins' / identity
    assert not folder.exists(), f'Preserve any existing diagnostic before preparing: {folder}'
    python = '/tmp/loopfill-venv/bin/python'
    marker = folder / 'report.json'
    common = ['--ar', str(ar), '--loopahead', str(method), '--gpu', '{GPU}', '--cpus', '{CPUS}']
    steps = [dict(id=identity+'-observed', out=str(folder), marker=str(marker), legacy_lock=True,
        argv=[python, '-u', str(BASE/'diagnose_formal_loopahead.py'), *common, '--out', str(folder)])]
    if cases:
        fp = folder / 'fp32-common-prefix'
        steps.append(dict(id=identity+'-fp32', requires=[steps[0]['id']], out=str(fp),
            marker=str(fp/'report.json'), legacy_lock=True,
            argv=[python, '-u', str(BASE/'diagnose_trained_loopahead.py'), *common,
                  '--observed-margin-report', str(marker), '--out', str(fp)]))
    groups.append(dict(id=identity, kind='legacy', row=row, agreement=agreement, out=str(folder), steps=steps))
    evidence[identity] = dict(cases=len(cases), checkpoint_sha256=meta['checkpoint_sha256'],
        exit_receipt=dict(path=str(receipt), sha256=sha(receipt), exit_code=0),
        sources={label: {name: sha(path/name) for name in ['metadata.json', 'complete.json', 'requests.jsonl']}
                 for label, path in [('ar', ar), ('loopahead', method)]})

root.mkdir(exist_ok=False)
for name in ['source', 'logs', 'checks']:
    (root/name).mkdir()
for name, expected in previous['source_sha256'].items():
    source = original/'source'/name
    assert sha(source) == expected
    shutil.copy2(source, root/'source'/name)
# The executable is unchanged except for the completion report's scope text.
controller = root/'source/orchestrate_baseline_diagnostics_resume.py'
text = controller.read_text()
old = "limitation='This queue covers already available main-model measurements, packed additions, B4 and development checks. Appendix model checks are added after its training/evaluation.'"
assert text.count(old) == 1
controller.write_text(text.replace(old, "limitation=plan['limitation']"))
plan = dict(created_unix=time.time(), base=str(BASE), balanced_queue=previous['balanced_queue'],
    supplement_plan=previous['supplement_plan'], gpu_policy=previous['gpu_policy'],
    source_sha256={p.name: sha(p) for p in sorted((root/'source').glob('*.py'))},
    legacy_source_sha256=previous['legacy_source_sha256'], groups=groups, reused_groups={},
    limitation='Six Ouro1.4B LoopAhead T0 cells only. Replays are excluded from benchmark timing. '
               'DFlash appendix diagnostics follow its separate training/evaluation; sampling T1 is not a greedy-equivalence test.',
    preparation=dict(path=str(Path(__file__)), sha256=sha(Path(__file__))),
    input_evidence=evidence)
save(root/'plan.json', plan)
verify_plan(root/'plan.json')
save(root/'checks/preparation.json', dict(status='passed', groups=6,
    steps=sum(len(g['steps']) for g in groups), cases=sum(e['cases'] for e in evidence.values()),
    plan_sha256=sha(root/'plan.json'), frozen_sources=plan['source_sha256'], checked_unix=time.time()))
print(read(root/'checks/preparation.json'))
