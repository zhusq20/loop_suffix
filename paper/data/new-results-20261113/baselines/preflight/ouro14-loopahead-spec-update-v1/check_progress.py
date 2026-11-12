"""Check the two newly collected cells against the immutable prior 48 rows."""
import csv
import io
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, save, sha

relative = 'experiments/baselines/combined-results/results.csv'
previous = subprocess.check_output(['git','show','b1eaefe:'+relative], cwd=BASE, text=True)
with (BASE/'combined-results/results.csv').open() as stream:
    current = list(csv.DictReader(stream))
key = lambda r: tuple(r[k] for k in ['method','model','benchmark','temperature'])
old = {key(r):r for r in csv.DictReader(io.StringIO(previous))}
now = {key(r):r for r in current}
assert len(old) == 48 and len(now) == len(current) == 50
assert all(now[k] == v for k,v in old.items())
added = [v for k,v in now.items() if k not in old]
assert {key(r) for r in added} == {('dflash','ouro-2.6b','humanevalplus','1'),
                                    ('loopahead','ouro-1.4b','gsm8k','1')}
evidence = read(BASE/'combined-results/evidence.json')
assert evidence['collector_sha256'] == sha(BASE/'collect_combined_results.py')
assert evidence['legacy_source']['rows'] == 43
assert sha(evidence['legacy_source']['path']) == evidence['legacy_source']['sha256']
new_evidence = {}
for r in added:
    identity = '/'.join(key(r))
    proof = evidence['cells'][identity]
    for source in proof['sources'].values():
        for name, digest in source['sha256'].items():
            assert sha(Path(source['path'])/name) == digest, (identity,name)
    assert float(r['eta_latency']) == float(r['ar_decode_s']) / float(r['method_decode_s'])
    assert float(r['eta_throughput']) == float(r['method_tok_s']) / float(r['ar_tok_s'])
    assert r['method_exit_evidence'] == 'actual_exit0'
    if r['method'] == 'dflash':
        receipts = list(proof['supplement_exits'].values()) + [proof['original_timed_exit']]
        for receipt in receipts:
            assert sha(receipt['path']) == receipt['sha256']
            assert read(receipt['path'])['exit_code'] == 0
    else:
        name = Path(r['method_run']).name
        launch, done = [read(BASE/f'logs/{name}-{suffix}.json') for suffix in ['launcher','finished']]
        assert done['exit_code'] == 0 and all(done[k] == v for k,v in launch.items())
    new_evidence[identity] = proof
status = read(BASE/'combined-results/status.json')
assert status['completed'] == 50 and status['expected'] == 72
assert status['results_sha256'] == sha(BASE/'combined-results/results.csv')
save(Path(__file__).parent/'result-progress.json', dict(status='passed', completed=50,
    unchanged_prior_rows=48, prior_git_revision='b1eaefe', added=added,
    new_cell_evidence=new_evidence, new_gpu_measurements=0,
    check_source_sha256=sha(Path(__file__)), results_sha256=status['results_sha256'],
    evidence_sha256=sha(BASE/'combined-results/evidence.json'), checked_unix=time.time(),
    limitation='T1 same-seed agreement is descriptive; full formal numerical checks remain pending.'))
print(dict(status='passed', completed=50, unchanged=48, added=[key(r) for r in added]))
