"""Verify completed optimization, held-out selection, and saved Huginn LAD tensors."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.sched_setaffinity(0, {80, 81, 82, 83})
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import torch

torch.set_num_threads(1)
BASE = Path(__file__).resolve().parents[2]
ROOT = BASE/'checkpoints/loopahead-huginn-3.5b-formal-v1'


def read(name):
    return json.loads((ROOT/name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


metadata, complete, best = read('metadata.json'), read('complete.json'), read('best.json')
confirmation = read('confirmation.json')
assert metadata['model'] == 'huginn-3.5b' and metadata['epochs'] == 3
assert metadata['effective_batch'] == 64 and metadata['total_updates'] == complete['step'] == 13539
assert metadata['parameters'] == 17673376 and metadata['components_sha256'] == 'd96da1e706c8e4635150ae9cf8b6b35932d961f18f84a9d23a6fd20556cd76c3'
for rel, sha in read('source-manifest.json').items():
    assert digest(ROOT/'source'/rel) == sha
audit = json.loads((BASE/'training/loopahead-huginn-3.5b-data-v1/feature-audit.json').read_text())
for stage in ['train', 'dev', 'confirmation']:
    assert metadata['feature_sources'][stage] == audit[stage]['source']
metrics = [json.loads(line) for line in (ROOT/'metrics.jsonl').open()]
assert [r['step'] for r in metrics] == [1, *range(25, 13539, 25), 13539]
assert all(math.isfinite(r[k]) for r in metrics for k in ['loss', 'grad_norm', 'lr', 'update_wall_s'])
dev = [json.loads(line) for line in (ROOT/'dev-metrics.jsonl').open()]
assert [r['step'] for r in dev] == [*range(500, 13539, 500), 13539]
selected = max(dev, key=lambda r: (r['accepted_prefix'], -r['ce']))
assert selected['step'] == best['step'] == confirmation['selected_step'] == 13500
assert complete['selected_checkpoint'] == best['checkpoint']
assert complete['dev_accepted_prefix'] == best['accepted_prefix'] == selected['accepted_prefix']
assert complete['confirmation_accepted_prefix'] == confirmation['accepted_prefix']
assert confirmation['requests'] == 782 and confirmation['ineligible_requests'] == 0
path = Path(best['checkpoint'])/'lad.pt'
payload = torch.load(path, map_location='cpu', weights_only=True)
assert payload['step'] == 13500 and payload['config'] == metadata['config']
assert len(payload['frequent_ids']) == len(set(payload['frequent_ids'])) == 8192
assert sum(t.numel() for t in payload['model'].values()) == metadata['parameters']
assert all(t.dtype == torch.bfloat16 and torch.isfinite(t).all().item() for t in payload['model'].values())
cost = [json.loads(line) for line in (ROOT/'reservation-cost.jsonl').open()]
assert len(cost) == 1 and cost[0]['status'] == 'complete'
report = dict(status='complete', model=metadata['model'], epochs=3, updates=13539,
    selected_step=13500, checkpoint=str(path), checkpoint_sha256=digest(path), parameters=metadata['parameters'],
    all_saved_parameters_finite=True, all_logged_optimization_values_finite=True,
    selected_by_frozen_dev_rule=True, confirmation_evaluations=1,
    dev_accepted_prefix=best['accepted_prefix'], confirmation_accepted_prefix=confirmation['accepted_prefix'],
    training_gpu_hours=cost[0]['reserved_gpu_seconds']/3600,
    peak_allocated_bytes=read('progress.json')['peak_allocated_bytes'],
    audit_source_sha256=digest(Path(__file__)), finished_unix=time.time(),
    limitation='Accepted-prefix validation scores are checkpoint-selection diagnostics, not benchmark gamma or speedup; trained-generation checks are separate.')
Path(__file__).with_name('training-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
