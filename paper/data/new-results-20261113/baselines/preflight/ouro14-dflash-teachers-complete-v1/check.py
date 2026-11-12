"""Check completed teacher evidence and the recipe of the running formal job."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from audit_portable_dflash_training import inspect_recipe
from external_dflash_costs import collect, json_rows
from packed_dflash_evidence import read, save, sha
from provenance import freeze_sources

OUT = Path(__file__).resolve().parent
assert not (OUT/'report.json').exists()
accounting = collect()
root = Path(accounting['root'])
plan = read(root/'plan.json')
meta = read(root/'training/metadata.json')
recipe = inspect_recipe(meta, plan)
shard = root/'teachers/shard0'
marker = read(shard/'complete.json')
identity = read(shard/'metadata.json')['identity']
inputs = {r['source_id']: r for r in json_rows(Path(plan['teacher_manifest']))}
assert len(inputs) == 50000
config = read(plan['draft_config'])
seen, excluded = set(), []
source_counts, eligible_counts = Counter(), Counter()
batches = defaultdict(list)
output_tokens = 0
for row in json_rows(shard/'teachers.jsonl'):
    key = row['source_id']
    assert key in inputs and key not in seen
    seen.add(key)
    source = inputs[key]
    assert row['pool'] == source['pool']
    assert row['question_sha256'] == source['question_sha256']
    assert row['prompt_sha256'] == hashlib.sha256(source['prompt'].encode()).hexdigest()
    prompt, output = row['prompt_tokens'], row['output_tokens']
    assert len(prompt) == source['prompt_lengths']['ouro-1.4b']
    assert 0 < len(output) <= 512 and len(prompt) + len(output) <= 3072
    assert all(type(t) is int and 0 <= t < config['vocab_size'] for t in prompt + output)
    source_counts[row['pool']] += 1
    output_tokens += len(output)
    batches[row['batch_id']].append((len(prompt), len(output)))
    if len(output) < 32:
        excluded.append(key)
    else:
        eligible_counts[row['pool']] += 1
assert seen == set(inputs)
assert len(seen) == marker['requests'] == 50000
assert output_tokens == marker['output_tokens'] == 21342831
assert sha(shard/'teachers.jsonl') == marker['teachers_sha256']
assert set(excluded) == set(meta['excluded_under_32_tokens'])
assert len(excluded) == len(meta['excluded_under_32_tokens']) == 1450
assert sum(eligible_counts.values()) == meta['examples'] == 48550
assert meta['total_updates'] == 6 * math.ceil(meta['examples']/32) == 9108
batch_logs = list(json_rows(shard/'batches.jsonl'))
assert [r['batch_id'] for r in batch_logs] == list(range(len(batches)))
for batch in batch_logs:
    observed = batches[batch['batch_id']]
    assert batch['requests'] == len(observed)
    assert sorted(batch['prompt_tokens']) == sorted(p for p, _ in observed)
    assert batch['output_tokens'] == sum(n for _, n in observed)
    assert math.isfinite(batch['inference_wall_s']) and batch['inference_wall_s'] > 0
    assert batch['peak_allocated_bytes'] > 0
parent_check = read(root/'checks/teacher-audit.json')
assert parent_check['eligible'] == meta['examples']
assert parent_check['expected_updates'] == meta['total_updates']
assert parent_check['teachers_sha256'] == marker['teachers_sha256']
assert parent_check['longest32_sha256'] == sha(root/'checks/longest32-teachers.jsonl')
assert meta['sources'] == [dict(path=str(shard/'teachers.jsonl'),
    sha256=marker['teachers_sha256'], metadata=identity)]
package = root/'source/portable/baselines'
expected_sources = {str(Path(p).relative_to('source/portable/baselines')): h
    for p, h in plan['source_sha256'].items()
    if p.startswith('source/portable/baselines/') and p.endswith('.py')}
assert meta['environment']['source_sha256'] == expected_sources
for name, digest in expected_sources.items():
    assert sha(package/name) == digest
for stage in ['teachers', 'smoke']:
    launch, receipt = [read(root/f'{stage}-{kind}.json') for kind in ['launcher', 'exit']]
    assert all(receipt[k] == v for k,v in launch.items()) and receipt['exit_code'] == 0
smoke = read(root/'checks/training-smoke/complete.json')
assert smoke['step'] == smoke['total_updates'] == 1 and smoke['bounded_experiment']
argv = read(root/'train-launcher.json')['argv']
assert not any(flag in argv for flag in ['--resume','--max-updates','--stop-after','--checkpoint-layers'])
for flag, value in {'--epochs':'6','--micro-batch':'4','--accumulation':'8',
                    '--num-anchors':'512','--attention':'flex_attention','--save-every':'500',
                    '--dtype':'bfloat16','--max-seq':'3072',
                    '--teachers':str(root/'teachers'),'--out':str(root/'training')}.items():
    assert argv.count(flag) == 1 and argv[argv.index(flag)+1] == value
artifacts = [root/'plan.json',root/'teachers-launcher.json',root/'teachers-exit.json',
    root/'smoke-launcher.json',root/'smoke-exit.json',root/'train-launcher.json',
    shard/'complete.json',shard/'metadata.json',shard/'batches.jsonl',
    shard/'teachers.jsonl',root/'checks/teacher-audit.json',
    root/'checks/training-smoke/complete.json',root/'training/metadata.json']
sources = freeze_sources(OUT,[Path(__file__),BASE/'audit_portable_dflash_training.py',
    BASE/'audit_dflash_training.py',BASE/'external_dflash_costs.py',
    BASE/'packed_dflash_evidence.py',BASE/'provenance.py'])
report = dict(status='completed_teachers_and_running_recipe_verified',
    model='ouro-1.4b',requests=50000,output_tokens=output_tokens,
    source_counts=dict(sorted(source_counts.items())),eligible=meta['examples'],
    eligible_source_counts=dict(sorted(eligible_counts.items())),excluded=len(excluded),
    batches=len(batch_logs),recipe=recipe,expected_formal_updates=9108,
    teachers_actual_exit_code=0,smoke_actual_exit_code=0,
    teacher_gpu_hours=sum(r['reserved_gpu_seconds'] for r in accounting['reservations']
                         if r['category']=='generation')/3600,
    closed_pipeline_gpu_hours=accounting['total_recorded_gpu_hours'],
    artifact_sha256={str(p):sha(p) for p in artifacts},source_sha256=sources,
    checked_unix=time.time(),new_gpu_measurements=0,
    limitations=['Formal training is running; its completion, final weights, metrics and cost are not yet audited.',
                 'Teacher tokens are checked against recorded IDs, prompts, hashes, ranges and batches; no model regeneration.',
                 'No benchmark speed, output equivalence or model quality conclusion follows from this check.'])
save(OUT/'report.json',report)
print({k:report[k] for k in ['status','requests','eligible','excluded','batches','teacher_gpu_hours']})
