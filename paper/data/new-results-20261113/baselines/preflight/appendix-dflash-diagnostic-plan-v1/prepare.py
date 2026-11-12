"""Freeze a CPU-only continuation; no training or measurements are launched."""
import ast
from pathlib import Path
import shutil
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, save, sha

root = BASE/'logs/appendix-dflash-diagnostics-v1'
original = BASE/'logs/baseline-diagnostics-resume-v1'
root.mkdir(exist_ok=False)
for name in ['source', 'logs', 'checks']:
    (root/name).mkdir()
original_plan = read(original/'plan.json')
for name, expected in original_plan['source_sha256'].items():
    assert sha(original/'source'/name) == expected
    shutil.copy2(original/'source'/name, root/'source'/name)
shutil.copy2(BASE/'resume_appendix_dflash_diagnostics.py', root/'source/resume_appendix_dflash_diagnostics.py')
controller = root/'source/orchestrate_baseline_diagnostics_resume.py'
text = controller.read_text()
changes = {
    "supplements = read(Path(plan['supplement_plan']).parent/'status.json')":
        "supplements = read(Path(plan['supplement_status_root'])/'status.json')",
    "supplement_state = read(Path(plan['supplement_plan']).parent/'dispatch.json')":
        "supplement_state = read(Path(plan['supplement_status_root'])/'dispatch.json')",
    "priority_pending = parent['pending'] or supplements['ready_supplements']":
        "priority_pending = parent['pending'] or supplements['ready']",
    "limitation='This queue covers already available main-model measurements, packed additions, B4 and development checks. Appendix model checks are added after its training/evaluation.'":
        "limitation=plan['limitation']",
}
for old, new in changes.items():
    assert text.count(old) == 1, old
    text = text.replace(old, new)
controller.write_text(text)
for path in (root/'source').glob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
parent = BASE/'logs/ouro14-packed-evaluation-v1'
evaluator_plan = parent/'plan-no-output-diagnostics.json'
parent_plan = read(evaluator_plan)
assert len(parent_plan['jobs']) == 12
assert sum(j['temperature'] == 0 for j in parent_plan['jobs']) == 6
assert not (parent/'complete.json').exists()
for job in parent_plan['jobs']:
    assert not (parent/'diagnostics'/job['id']).exists()
base_names = set(original_plan['legacy_source_sha256'])
base_names.update(['collect_combined_results.py', 'packed_dflash_evidence.py'])
spec = dict(created_unix=time.time(), base=str(BASE), evaluator_plan=str(evaluator_plan),
    evaluator_plan_sha256=sha(evaluator_plan),
    source_sha256={p.name: sha(p) for p in sorted((root/'source').glob('*.py'))},
    base_sha256={name: sha(BASE/name) for name in sorted(base_names)},
    authorization='Continue the active complete baseline goal under the direct thread instructions. '
                  'The unattributed cancellation file is not treated as a user waiver.',
    scope='Only six numerical diagnostic groups after the existing 37 evaluation stages finish; no repeated training or measurement.',
    scheduler_changes=changes, preparation_sha256=sha(Path(__file__)))
save(root/'spec.json', spec)
from resume_appendix_dflash_diagnostics import verify
verify(root/'spec.json')
save(root/'checks/preparation.json', dict(status='passed', expected_groups=6,
    waits_for_completed_measurements=12, expected_parent_stages=37,
    gpu_work_started=False, spec_sha256=sha(root/'spec.json'),
    scheduler_adaptations_only=list(changes), checked_unix=time.time()))
print(read(root/'checks/preparation.json'))
