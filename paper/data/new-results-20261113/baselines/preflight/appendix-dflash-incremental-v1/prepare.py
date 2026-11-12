"""Prepare an incremental queue using the unchanged existing dispatcher.

This prepares a candidate only. It never replaces a live plan or starts a GPU.
"""
import copy
from pathlib import Path
import shutil
import sys
import time

BASE = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, save, sha
from orchestrate_baseline_diagnostics_resume import verify_plan
from audit_numerical_matrix import EXPECTED, scope_key

root = BASE / 'logs/appendix-dflash-diagnostics-v1'
spec_path = root / 'spec.json'
spec = read(spec_path)
assert not (OUT / 'report.json').exists()
assert not (root / 'plan.json').exists() and not (root / 'dispatch.json').exists()
assert not (root / 'complete.json').exists() and not (root / 'failure.json').exists()
for name, digest in spec['source_sha256'].items():
    source = root / 'source' / name
    assert sha(source) == digest
    dest = OUT / 'source' / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    shutil.copy2(source, dest)
for name, digest in spec['base_sha256'].items():
    assert sha(BASE / name) == digest

evaluator_plan = Path(spec['evaluator_plan'])
assert sha(evaluator_plan) == spec['evaluator_plan_sha256']
parent = evaluator_plan.parent
parent_plan = read(evaluator_plan)
timed_proof = BASE / 'preflight/ouro14-dflash-timed-complete-v1'
timed, timed_exit = read(timed_proof / 'report.json'), read(timed_proof / 'exit.json')
assert timed['status'] == 'complete_timed_runs_not_complete_benchmark_cells'
assert timed['verified_timed_runs'] == 12 and timed['requests'] == 4184
assert timed_exit['exit_code'] == 0
assert timed_exit['source_sha256'] == sha(timed_proof / 'verify.py')
assert timed_exit['started_unix'] <= timed['checked_unix'] <= timed_exit['finished_unix']
for name, digest in timed['artifact_sha256'].items():
    assert sha(name) == digest
assert timed['plan_sha256'] == sha(evaluator_plan)

legacy = dict(spec['base_sha256'])
for name, digest in parent_plan['frozen_sha256'].items():
    path = parent / name
    assert sha(path) == digest
    legacy[str(path.relative_to(BASE))] = digest
for path in [evaluator_plan, parent / 'suite/protocol.json', parent / 'suite/jobs.json',
             parent / 'training-binding.json', spec_path,
             timed_proof / 'report.json', timed_proof / 'exit.json']:
    legacy[str(path.relative_to(BASE))] = sha(path)

groups = []
for job in parent_plan['jobs']:
    if job['temperature'] != 0:
        continue
    folder = parent / 'diagnostics' / job['id']
    assert not folder.exists(), 'Preserve any already started diagnostic.'
    identity = job['id'] + '-greedy'
    groups.append(dict(id=identity, kind='packed', job=job['id'], out=str(folder),
        dependency='Frozen evaluator marks own AR complete only after actual exit 0 and input/output validation.',
        steps=[dict(id=identity, out=str(folder), marker=str(folder / 'complete.json'),
            legacy_lock=False, argv=[parent_plan['python'], '-u',
                str(root / 'source/diagnose_packed_dflash.py'),
                '--plan', str(parent / 'supplement/plan.json'), '--job', job['id'],
                '--ar', str(parent / 'supplement/runs' / (job['id'] + '-ar')),
                '--out', str(folder), '--lock-fd', '{LOCK_FD}'])]))
assert len(groups) == 6
plan = dict(created_unix=time.time(), base=str(BASE),
    balanced_queue=parent_plan['balanced_queue'],
    supplement_plan=str(parent / 'supplement/plan.json'),
    supplement_status_root=str(parent), gpu_policy=parent_plan['gpu_policy'],
    source_sha256=copy.deepcopy(spec['source_sha256']),
    legacy_source_sha256=legacy, groups=groups, reused_groups={},
    spec_sha256=sha(spec_path), evaluator_plan_sha256=sha(evaluator_plan),
    preparation_sha256=sha(Path(__file__)),
    amendment='Remove only the all-37-stage launch barrier. Keep the unchanged per-cell AR gate, measurement priority, GPU leases, diagnostic code and six-group scope.',
    limitation='Six Ouro1.4B DFlash T0 numerical checks may overlap remaining measurements. '
               'Own completed timed and AR inputs are required; profile and all 72 formal cells remain final requirements. '
               'Diagnostic replay timing is excluded from benchmark measurements.')
save(OUT / 'plan.json', plan)
assert verify_plan(OUT / 'plan.json') == plan
scope = {scope_key(plan, group) for group in groups}
assert scope == {'formal/dflash/ouro-1.4b/' + task for task in
                 ['gsm8k', 'math500', 'bbh', 'humanevalplus', 'mbppplus', 'specbench']}
all_scopes = []
for name in ['baseline-diagnostics-resume-v1', 'appendix-loopahead-diagnostics-v1']:
    prior = verify_plan(BASE / 'logs' / name / 'plan.json')
    all_scopes.extend(scope_key(prior, group) for group in prior['groups'])
all_scopes.extend(scope)
assert len(all_scopes) == len(set(all_scopes)) == 39 and set(all_scopes) == EXPECTED

dispatch = read(parent / 'dispatch.json')
ready = [group['job'] for group in groups
         if dispatch.get(group['job'] + '-ar', {}).get('outcome') == 'complete']
save(OUT / 'report.json', dict(status='candidate_prepared_not_activated',
    candidate_plan_sha256=sha(OUT / 'plan.json'), original_spec_sha256=sha(spec_path),
    unchanged_diagnostic_sources=spec['source_sha256'],
    expected_groups=6, exact_total_scope=39, initial_completed_ar_dependencies=ready,
    measurement_priority_unchanged=True, gpu_lease_code_unchanged=True,
    diagnostic_worker_code_unchanged=True, changed_live_sources=False, new_gpu_runs=0,
    unchanged_combined_csv_sha256=sha(BASE / 'combined-results/results.csv'),
    preparation_sha256=sha(Path(__file__)), checked_unix=time.time()))
print(dict(candidate=str(OUT / 'plan.json'), groups=6, total_scope=39,
           unchanged_dispatcher=True, changed_live_sources=False))
