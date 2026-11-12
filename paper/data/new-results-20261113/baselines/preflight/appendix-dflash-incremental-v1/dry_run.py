"""Exercise one real scheduler cycle while forbidding GPU discovery or launch."""
from pathlib import Path
import sys
import time
from unittest.mock import patch

base, root = (Path(x).resolve() for x in sys.argv[1:])
sys.path[:0] = [str(root / 'source'), str(base)]
import orchestrate_baseline_diagnostics_resume as scheduler
import loop_queue_recovery
from packed_dflash_evidence import read, save, sha

class EndCycle(Exception):
    pass

def forbidden(*args, **kwargs):
    raise AssertionError('Measurement-priority dry run attempted GPU discovery or launch')

plan = read(root / 'plan.json')
parent = Path(plan['supplement_status_root'])
assert read(parent / 'status.json')['ready'], 'This check requires real queued measurement work.'
assert not (base / 'logs/appendix-dflash-diagnostics-v1/plan.json').exists()
outputs = [Path(g['out']) for g in plan['groups']]
assert all(not p.exists() for p in outputs)
with patch.object(scheduler, 'native_cuda_environment', return_value=(
        {}, {'test_only': 'CPU dry run; no CUDA environment or measurement'})), \
     patch.object(scheduler, 'cards', side_effect=forbidden), \
     patch.object(scheduler, 'occupied', side_effect=forbidden), \
     patch.object(loop_queue_recovery, 'spawn_supervised', side_effect=forbidden), \
     patch.object(scheduler.time, 'sleep', side_effect=EndCycle):
    try:
        scheduler.main(root / 'plan.json')
    except EndCycle:
        pass
    else:
        raise AssertionError('Expected exactly one live-data scheduling cycle')
status = read(root / 'status.json')
assert status['expected_groups'] == 6 and not status['running'] and not status['failures']
assert status['priority_pending'] and not read(root / 'dispatch.json')
assert all(not p.exists() for p in outputs)
save(root / 'priority-dry-run.json', dict(
    status='passed_cpu_only_measurement_priority_check',
    candidate_plan_sha256=sha(root / 'plan.json'),
    scheduler_sha256=sha(root / 'source/orchestrate_baseline_diagnostics_resume.py'),
    new_gpu_runs=0, gpu_discovery_calls=0, worker_launches=0,
    expected_groups=6, actual_priority_pending=status['priority_pending'],
    changed_canonical_queue=False, checked_unix=time.time()))
print(dict(status='passed', groups=6, worker_launches=0, changed_canonical_queue=False))
