"""CPU checks of real job readiness and isolated priority-observer cleanup."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import prioritize_huginn_math as priority
import orchestrate_formal as controller
from provenance import freeze_sources
from result_evidence import sha

out = Path(__file__).parent
assert not (out/'report.json').exists()
names = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json', 'configs/gpu-allocation.json']
before = {name: sha(BASE/name) for name in names}
plan = priority.make_plan()
value = priority.read(priority.POLICY)
installed = priority.install(value, plan)
state = priority.read(priority.CONTROL/'status.json')
jobs = [j for j in priority.read(priority.CONTROL/'plan.json')['jobs'] if j['id'] in state['pending']]
checks = []


def runnable(policy, trained):
    with patch.object(controller, 'gpu_policy', return_value=policy), patch.object(controller.dflash_registry, 'ready', side_effect=lambda model: model != 'huginn-3.5b' or trained):
        return [j['id'] for j in jobs if controller.ready(j)]


assert runnable(value, False) == runnable(installed, False)
checks.append('Unfinished Huginn training leaves other runnable jobs unchanged')
assert runnable(installed, True)[:2] == plan['priority_jobs']
assert runnable(value, True)[:2] != plan['priority_jobs']
checks.append('Existing controller readiness dispatches both MATH cells first after training')
assert {k:v for k,v in installed.items() if k not in ['deferred_jobs', priority.KEY, 'updated_unix']} == {k:v for k,v in value.items() if k not in ['deferred_jobs', 'updated_unix']}
checks.append('Install preserves GPU allocation, training reservations and unrelated policy fields')
changed = copy.deepcopy(installed)
changed['reserved_for_development'] = []
changed['deferred_jobs'].append('unrelated-temporary-hold')
restored = priority.restore(changed, plan)
assert restored['reserved_for_development'] == []
assert restored['deferred_jobs'] == value['deferred_jobs']+['unrelated-temporary-hold']
assert runnable(restored, True) == runnable(value, True)
checks.append('Cleanup preserves coordinator reservation release and unrelated deferrals')
for bad, operation in [(installed, priority.install), (value, priority.restore)]:
    try:
        operation(bad, plan)
    except AssertionError:
        pass
    else:
        raise AssertionError('Policy ownership conflict accepted')
checks.append('Duplicate installation and changed ownership are rejected')
assert all(priority.started(identity, state) is None for identity in plan['priority_jobs'])
fake = copy.deepcopy(state)
fake['running'][plan['priority_jobs'][0]] = dict(pid=999999999, started_unix=0, gpus=[3])
assert priority.started(plan['priority_jobs'][0], fake) is None
fake['running'][plan['priority_jobs'][0]]['pid'] = os.getpid()
try:
    priority.started(plan['priority_jobs'][0], fake)
except AssertionError:
    pass
else:
    raise AssertionError('An unrelated live PID was accepted as a priority worker')
checks.append('Pending/state-only and unrelated live processes cannot satisfy startup evidence')

for mode in ['success', 'error', 'interrupt', 'still_waiting']:
    with tempfile.TemporaryDirectory(prefix='loopfill-priority-test-') as temporary:
        root = Path(temporary)
        policy_path = root/'allocation.json'
        policy_path.write_text(json.dumps(value))
        result = root/'observer'

        def observe(identity, status):
            if mode == 'error':
                raise RuntimeError('Simulated worker failure')
            if mode == 'interrupt':
                raise KeyboardInterrupt('Simulated interruption')
            if mode == 'still_waiting':
                return None
            current = priority.read(policy_path)
            current['reserved_for_development'] = []
            if 'unrelated-temporary-hold' not in current['deferred_jobs']:
                current['deferred_jobs'].append('unrelated-temporary-hold')
            priority.save(policy_path, current)
            return dict(kind='mock_worker_for_CPU_control_flow_only', identity=identity)

        with patch.object(priority, 'OUT', result), patch.object(priority, 'POLICY', policy_path), patch.object(priority, 'make_plan', return_value=plan), patch.object(priority, 'freeze_sources'), patch.object(priority, 'started', side_effect=observe), patch.object(priority.signal, 'signal'), patch.object(priority.time, 'sleep', side_effect=RuntimeError('End the isolated wait')), patch.object(sys, 'argv', ['priority', '--execute']):
            try:
                priority.main()
            except (RuntimeError, KeyboardInterrupt):
                assert mode != 'success'
            else:
                assert mode == 'success'
        final = priority.read(policy_path)
        assert priority.KEY not in final
        assert not set(plan['postponed_jobs']) & set(final['deferred_jobs'])
        assert set(value['deferred_jobs']) <= set(final['deferred_jobs'])
        assert (result/'complete.json').exists() == (mode == 'success')
        assert (result/'failure.json').exists() == (mode != 'success')
        if mode == 'success':
            assert final['reserved_for_development'] == [] and 'unrelated-temporary-hold' in final['deferred_jobs']
        checks.append(f'Actual observer main restores isolated policy on {mode}')

assert before == {name: sha(BASE/name) for name in names}
assert 'torch' not in sys.modules
freeze_sources(out, [Path(__file__), BASE/'prioritize_huginn_math.py', BASE/'orchestrate_formal.py', BASE/'gpu_allocation.py'])
report = dict(status='CPU_priority_and_cleanup_checks_passed_actual_dispatch_pending', checks=checks,
              plan=plan, production_policy_and_results_unchanged=before, gpu_jobs_launched=0, torch_imported=False)
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(status=report['status'], checks=len(checks), priority_jobs=plan['priority_jobs'], postponed_jobs=plan['postponed_jobs'])))
