"""Freeze a new attempt; preserve the original plan, reports and actual -15."""
import copy
import json
from pathlib import Path
import shutil
import sys
import time

BASE = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, save, sha
from provenance import freeze_sources
from loopahead_resume_attempt import ATTEMPT, NAME, ORIGINAL, PREPARATION, verify_attempt

assert read(OUT / 'tests-exit.json')['exit_code'] == 0
for name, digest in read(OUT / 'tests-exit.json')['source_sha256'].items():
    assert sha(BASE / name) == digest
old_plan = read(ORIGINAL / 'plan.json')
old = next(g for g in old_plan['groups'] if g['id'] == NAME)
group = copy.deepcopy(old)
group['out'] = str(BASE / 'preflight/formal-greedy-margins/loopahead-ouro-2.6b-math500-resume-v1')
folder = Path(group['out'])
assert not folder.exists() and not ATTEMPT.exists()
step = copy.deepcopy(old['steps'][1])
step['id'] = NAME + '-fp32-continuation'
step.pop('requires')
step['out'] = str(folder / 'fp32-common-prefix')
step['marker'] = str(Path(step['out']) / 'report.json')
step['argv'][2] = str(BASE / 'diagnose_trained_loopahead_resume.py')
step['argv'][step['argv'].index('--out')+1] = step['out']
step['argv'] += ['--resume-evidence', str(BASE / 'preflight/interrupted-loopahead-math-fp32-v1/report.json')]
group['steps'] = [step]
ATTEMPT.mkdir()
(ATTEMPT / 'logs').mkdir()
folder.mkdir()
shutil.copy2(Path(old['out']) / 'report.json', folder / 'report.json')
support = BASE / 'logs/ouro14-packed-evaluation-v1/source'
sources = [BASE / name for name in ['loopahead_resume_attempt.py', 'queue_loopahead_math_continuation.py',
    'diagnose_trained_loopahead_resume.py', 'loopahead_partial_resume.py', 'diagnose_trained_loopahead.py',
    'loopahead_diagnostic_inputs.py', 'provenance.py', 'gpu_startup.py', 'gpu_allocation.py',
    'packed_dflash_evidence.py', 'orchestrate_greedy_diagnostics.py', 'test_loopahead_resume_attempt.py']]
sources += sorted((BASE / 'loopahead').glob('*.py'))
sources += [support / name for name in ['loop_queue_recovery.py', 'native_cuda_env.py', 'orchestrate_packed_dflash_supplements.py']]
sources += [BASE / 'logs/baseline-diagnostics-resume-v1/source/lock_handoff.py']
frozen = freeze_sources(ATTEMPT, sources)
capsule = BASE / 'preflight/interrupted-loopahead-math-fp32-v1'
artifacts = [ORIGINAL / 'plan.json', Path(old['out']) / 'report.json', folder / 'report.json']
artifacts += [ORIGINAL / f'logs/{s["id"]}-{kind}.json' for s in old['steps'] for kind in ['launcher', 'exit']]
artifacts += [ORIGINAL / f'logs/{old["steps"][1]["id"]}.log']
artifacts += [ORIGINAL / name for name in ['STOP', 'cancellation.json'] if (ORIGINAL / name).exists()]
artifacts += [capsule / name for name in ['report.json', 'recover.py', 'exit.json', 'launcher.json', 'recovered-records.partial.jsonl', 'binding-check.json']]
artifacts += [PREPARATION / name for name in ['inputs-report.json', 'verification.json', 'tests-launcher.json',
    'tests-exit.json', 'tests.log', 'inputs-launcher.json', 'inputs-exit.json', 'inputs.log', 'source-manifest.json']]
artifacts += [OUT / name for name in ['prepare.py', 'tests-launcher.json', 'tests-exit.json', 'tests.log']]
save(ATTEMPT / 'plan.json', dict(base=str(BASE), group=group, original_group=old,
    original_plan=str(ORIGINAL / 'plan.json'), original_plan_sha256=sha(ORIGINAL / 'plan.json'),
    balanced_queue=old_plan['balanced_queue'], measurement_queue=str(BASE / 'logs/ouro14-packed-evaluation-v1'),
    gpu_policy=old_plan['gpu_policy'], source_sha256=frozen,
    artifact_sha256={str(p): sha(p) for p in artifacts}, created_unix=time.time(),
    authority='Current thread: user reconfirmed the original task document and then said 好的 你继续试验吧.',
    limitation='Continuation only: preserve original exit -15, reuse 262 cases, execute only 167 remaining cases. No new GPU work is launched by preparation.'))
plan = verify_attempt()
save(OUT / 'prepared.json', dict(status='prepared_not_launched', plan=str(ATTEMPT / 'plan.json'),
    plan_sha256=sha(ATTEMPT / 'plan.json'), source_sha256=sha(Path(__file__)),
    new_gpu_runs=0, reused_cases=262, pending_cases=167, finished_unix=time.time()))
print(dict(status='prepared_not_launched', reused_cases=262, pending_cases=167))
