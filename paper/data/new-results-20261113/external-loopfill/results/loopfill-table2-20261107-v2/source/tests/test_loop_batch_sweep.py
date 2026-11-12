"""A failed cell must not prune the experiment matrix or manufacture throughput."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('batch_sweep', SCRIPTS / 'loop_batch_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_all_eight_lanes_cover_each_cell_once_and_pair_methods():
    arms = ['ar', 'wavefront', 'loopspec-dual', 'budget-suffix', 'budget-raw', 'budget-online']
    jobs = [dict(cell, gpu=lane['gpu']) for lane in sweep.LANES
            for cell in sweep.cells(lane['model'], arms, lane['batches'])]
    assert len(jobs) == len({j['key'] for j in jobs}) == 108
    assert {j['gpu'] for j in jobs} == set(range(8))
    for model in {j['model'] for j in jobs}:
        assert {j['batch'] for j in jobs if j['model'] == model} == set(sweep.BATCHES)
        for b in sweep.BATCHES:
            assert len({j['gpu'] for j in jobs if j['model'] == model and j['batch'] == b}) == 1


def test_oom_and_unrelated_failure_do_not_cancel_later_cells(tmp_path):
    visited = []
    def execute(cell):
        visited.append((cell['batch'], cell['arm']))
        out = tmp_path / cell['key']
        out.mkdir()
        # Real child exits simulate independent process failure and clean restart.
        oom = cell['batch'] in (2, 256) and cell['arm'] == 'ar'
        child = subprocess.run([sys.executable, '-c', 'import sys; sys.exit(1)' if oom else 'pass'])
        if oom:
            (out / 'failure.json').write_text(json.dumps(dict(outcome='oom', exception_type='OutOfMemoryError')))
        if cell['batch'] == 4 and cell['arm'] == 'ar':
            raise RuntimeError('non-memory error')
        return dict(cell, outcome=sweep.classify(child.returncode, out))
    results = sweep.run_cells(sweep.cells('tiny', ['ar', 'wavefront']), execute)
    assert len(results) == len(visited) == 18
    assert results[-1]['outcome'] == 'complete'  # B=256 other arm still ran
    assert sum(r['outcome'] == 'oom' for r in results) == 2
    assert sum(r['outcome'] == 'failed' for r in results) == 1


def test_untyped_memory_guard_is_not_reported_as_oom(tmp_path):
    (tmp_path / 'failure.json').write_text(json.dumps(dict(outcome='oom', exception_type='MemoryError')))
    assert sweep.classify(1, tmp_path) == 'failed'


def test_oom_summary_contains_no_throughput(tmp_path):
    sweep.save(tmp_path / 'jobs' / 'test.json', dict(key='test', model='tiny', batch=256, arm='ar', outcome='oom'))
    sweep.summarize(tmp_path)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['counts'] == {'oom': 1}
    assert summary['measurements'][0]['tok_s'] is None


def test_shards_merge_exactly_480_and_reject_duplicates(tmp_path):
    rows = [dict(question_id=i) for i in range(480)]
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs' / 'specbench480-tiny.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    header = dict(reference_identity=dict(source_sha256={'decoder': 'same'}))
    (tmp_path / 'references').mkdir()
    for shard in range(4):
        path = tmp_path / 'references' / f'tiny-shard{shard}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in [header] + [dict(request_id=i, tokens=[i+1]) for i in range(shard,480,4)]))
    sweep.merge_references(tmp_path, 'tiny')
    merged = [json.loads(s) for s in (tmp_path / 'references' / 'tiny.jsonl').read_text().splitlines()]
    assert [r['request_id'] for r in merged[1:]] == list(range(480))
    with (tmp_path / 'references' / 'tiny-shard1.jsonl').open('a') as f:
        f.write(json.dumps(dict(request_id=0, tokens=[1]))+'\n')
    with pytest.raises(ValueError, match='duplicate'):
        sweep.merge_references(tmp_path, 'tiny')
