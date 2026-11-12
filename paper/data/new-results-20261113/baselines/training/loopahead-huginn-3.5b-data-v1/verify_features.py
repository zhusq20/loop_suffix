"""Audit all frozen Huginn LAD features with the actual training data reader."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
os.sched_setaffinity(0, {80, 81, 82, 83})
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import torch
from loopahead.training_data import FeatureStore, digest

torch.set_num_threads(1)
started = time.time()
report = {}
components = set()
for stage in ('train', 'dev', 'confirmation'):
    name = 'loopahead-' + stage
    store = FeatureStore(BASE/'features'/name/'huginn-3.5b', name, 'huginn-3.5b')
    assert len(store.provenance) == 8
    teachers = {}
    sources = []
    for shard in range(8):
        folder = BASE/'data/teachers'/name/'huginn-3.5b'/f'shard{shard}'
        complete = json.loads((folder/'complete.json').read_text())
        sha = digest(folder/'teachers.jsonl')
        assert sha == complete['teachers_sha256']
        rows = [json.loads(line) for line in (folder/'teachers.jsonl').open()]
        assert len(rows) == complete['requests']
        index_path = BASE/'features'/name/'huginn-3.5b'/f'shard{shard}'/'index.json'
        index = json.loads(index_path.read_text())
        assert index['identity']['teacher_sha256'] == sha
        assert index['identity']['depth'] == 4
        assert index['identity']['teacher_identity']['full_depth'] == 32
        for row in rows:
            assert row['source_id'] not in teachers
            teachers[row['source_id']] = row
        sources.append(dict(shard=shard, requests=len(rows), teachers_sha256=sha))
    counts = Counter()
    state_tokens = 0
    for row in store.rows:
        teacher = teachers[row['source_id']]
        assert row['prompt_tokens'] == teacher['prompt_tokens']
        assert row['output_tokens'] == teacher['output_tokens']
        assert tuple(row['states'].shape) == (len(row['tokens']), 5280)
        assert torch.isfinite(row['states']).all().item()
        state_tokens += len(row['tokens'])
        counts.update(row['output_tokens'])
    assert counts == store.counts and set(teachers) == {r['source_id'] for r in store.rows}
    components.add(store.component_hash)
    report[stage] = dict(count=len(store.rows), eligible=len(store.eligible),
        ineligible_ids=store.ineligible, state_tokens=state_tokens,
        component_sha256=store.component_hash, source=store.provenance,
        teacher_sources=sources, all_states_finite=True, all_teacher_token_ids_equal=True)
    print(json.dumps(dict(stage=stage, count=len(store.rows), eligible=len(store.eligible))), flush=True)
    del store, teachers
assert len(components) == 1
report['audit'] = dict(status='complete', wall_s=time.time()-started, finished_unix=time.time(),
    script_sha256=digest(Path(__file__)), reader_sha256=digest(BASE/'loopahead/training_data.py'),
    checks='Actual FeatureStore validates all24 indices/chunks/components/manifests and IDs; additionally full teacher token equality, full-depth provenance, state shapes, finite values, and output-token counts.')
Path(__file__).with_name('feature-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report['audit']), flush=True)
