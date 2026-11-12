"""Bind actual final training, closed costs and the two live MATH workers."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from dflash_training_registry import validate_projected_completion, sha
from dflash_source_identity import validate_evaluation_sources


def read(path):
    return json.loads(path.read_text())


out = Path(__file__).parent
assert not (out/'report.json').exists()
binding = validate_projected_completion()
audit_path = Path(binding['audit_report'])
audit = read(audit_path)
assert audit['weights']['tensor_count'] == 60 and not audit['weights']['cuda_initialized']
root = Path(binding['training_root'])
train_exit = read(BASE/'logs/dflash-huginn-narrow-formal-v1-finished.json')
audit_exit = read(BASE/'logs/audit-dflash-huginn-narrow-formal-v1-finished.json')
assert train_exit['exit_code'] == audit_exit['exit_code'] == 0
assert train_exit['finished_unix'] <= audit_exit['started_unix'] <= audit['started_unix']
assert audit['finished_unix'] <= audit_exit['finished_unix']
queue = read(BASE/'logs/huginn-narrow-formal-queue-v1/complete.json')
assert audit_exit['finished_unix'] <= queue['finished_unix']
costs = [json.loads((root/f'reservation-cost-rank{rank}.jsonl').read_text()) for rank in [0, 1]]
assert {r['gpu_uuid'] for r in costs} == {'GPU-bb9dac43-6fce-fc43-0f10-8f64a80742c3', 'GPU-e54049b5-ca93-d066-efc5-6942d59bf762'}
assert all(r['status'] == 'complete' for r in costs)
assert sum(r['reserved_gpu_seconds'] for r in costs)/3600 == audit['closed_gpu_reservation_hours']
cost_path = BASE/'training/current-cost-summary.json'
current_cost = read(cost_path)
previous_cost = json.loads(subprocess.check_output(['git', 'show', 'HEAD:experiments/baselines/training/current-cost-summary.json'], cwd=ROOT))
assert sum(r['status'] == 'complete' for r in current_cost['rows']) == 4
for old, new in zip(previous_cost['rows'], current_cost['rows']):
    if (old['method'], old['model']) != ('dflash', 'huginn-3.5b'):
        assert old == new
assert math.isclose(current_cost['total_recorded_gpu_hours']-previous_cost['total_recorded_gpu_hours'], audit['closed_gpu_reservation_hours'], abs_tol=1e-10)

priority = BASE/'logs/huginn-math-priority-v1'
done, restored = read(priority/'complete.json'), read(priority/'restored.json')
observer_exit = read(BASE/'logs/huginn-math-priority-v1-finished.json')
assert observer_exit['exit_code'] == 0 and observer_exit['pid'] == 150154
assert done['finished_unix'] <= observer_exit['finished_unix']
policy = read(BASE/'configs/gpu-allocation.json')
assert policy == restored['after'] and sha(BASE/'configs/gpu-allocation.json') == done['restored_policy_sha256']
assert policy['allowed_gpus'] == [2, 3, 4, 6] and policy['released_gpus'] == [0, 1, 5, 7]
assert policy['reserved_for_development'] == [] and policy['deferred_jobs'] == ['train-huginn-3.5b']
assert 'temporary_huginn_math_priority' not in policy
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
gpu_pids = {(uuid.strip(), int(pid)) for uuid, pid in (line.split(',') for line in apps.splitlines())}
state = read(BASE/'logs/formal-controller-v1/status.json')
assert not state['failed']
inputs = [json.loads(line) for line in (ROOT/'results/loopfill-latency512-20261107/inputs/math500.jsonl').open()]
live = {}
for temperature, gpu, worker_pid, child_pid, cpus in [(0, 3, 174369, 174380, {12, 13, 14, 15}), (1, 6, 174370, 174381, {56, 57, 58, 59})]:
    identity = f'dflash-huginn-3.5b-math500-t{temperature}'
    job = BASE/f'logs/formal-controller-v1/{identity}.json'
    expected = done['evidence'][identity]
    assert expected['pid'] == state['running'][identity]['pid'] == worker_pid
    assert expected['job_sha256'] == sha(job) and state['running'][identity]['gpus'] == [gpu]
    parent = Path('/proc')/str(worker_pid)
    parent_stat = parent.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    assert parent_stat[0] in ['R', 'S', 'D'] and parent_stat[19] == expected['start_ticks']
    assert parent.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')[-3:] == [str(BASE/'orchestrate_formal.py'), '--worker', str(job)]
    path = BASE/f'runs/dflash-stack-huginn-3.5b-math500-t{temperature}-ar-v1'
    meta = read(path/'metadata.json')
    validate_evaluation_sources(path)
    launch = read(BASE/f'logs/{path.name}-launcher.json')
    proc = Path('/proc')/str(child_pid)
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    assert stat[0] in ['R', 'S', 'D'] and int(stat[1]) == worker_pid
    assert launch['pid'] == child_pid and proc.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0') == launch['argv']
    assert os.sched_getaffinity(child_pid) == cpus
    assert (meta['gpu_uuid'], child_pid) in gpu_pids
    assert '/usr/lib64/libcuda.so.570.148.08' in proc.joinpath('maps').read_text()
    assert meta['model'] == 'huginn-3.5b' and meta['method'] == 'ar' and meta['temperature'] == temperature
    assert meta['count'] == 500 and meta['max_new_tokens'] == 512 and meta['graph'] is False
    assert meta['torch'] == '2.6.0+cu126' and meta['transformers'] == '4.57.1'
    lines = (path/'requests.jsonl').read_bytes().splitlines(keepends=True)
    prefix = [json.loads(line) for line in lines[:3] if line.endswith(b'\n')]
    assert prefix, 'Wait for an actual complete request, not just process startup'
    for row, expected_input in zip(prefix, inputs):
        assert row['source_id'] == expected_input['source_id'] and row['prompt_sha256'] == expected_input['prompt_sha256']
        assert len(row['tokens']) <= 512
    prefix_path = out/f'math-t{temperature}-observed-prefix.jsonl'
    prefix_path.write_text(''.join(json.dumps(row)+'\n' for row in prefix))
    live[identity] = dict(worker_pid=worker_pid, child_pid=child_pid, child_start_ticks=stat[19], gpu=gpu,
                          argv=launch['argv'], metadata_sha256=sha(path/'metadata.json'),
                          observed_requests=len(prefix), prefix_sha256=sha(prefix_path), stage='same_stack_AR_only')
names = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
for name in names:
    assert (BASE/name).read_bytes() == subprocess.check_output(['git', 'show', f'HEAD:experiments/baselines/{name}'], cwd=ROOT)
assert read(BASE/'formal-matrix-status.json')['completed'] == 26
report = dict(status='complete_narrow_training_audit_and_actual_formal_AR_dispatch', checked_unix=time.time(),
              binding=binding, actual_training_exit=train_exit, actual_audit_exit=audit_exit,
              training_reservations=costs, training_gpu_hours=audit['closed_gpu_reservation_hours'],
              closed_project_gpu_hours=current_cost['total_recorded_gpu_hours'], other_cost_rows_unchanged=True,
              cost_summary_sha256=sha(cost_path), final_metrics=audit['metrics'], live_AR=live,
              priority_observer_exit=observer_exit, priority_complete_sha256=sha(priority/'complete.json'),
              policy_sha256=sha(BASE/'configs/gpu-allocation.json'), temporary_deferrals_and_reservations_cleared=True,
              formal_results_unchanged=True, verification_source_sha256=sha(Path(__file__)),
              limitations=['Training completion is not a speed or decoding-correctness result.',
                           'Two MATH cells are at same-stack AR; final DFlash timed/profile stages remain pending.',
                           'Optimizer/RNG artifacts preserved; no actual resumed optimization claimed.',
                           'Old interrupted wide training and failed costs remain separate.',
                           '26/72 benchmark cells and 4/6 formal trainings complete; remaining work continues.'])
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ['status', 'training_gpu_hours', 'closed_project_gpu_hours', 'live_AR']}))
