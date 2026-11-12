"""Collect completed measurements and retain an actual CPU-process exit receipt."""
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path('/ossfs/workspace/sz/loopfill/experiments/baselines')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from packed_dflash_evidence import read, save, sha

assert read(BASE/'logs/packed-dflash-supplement-v1/status.json')['completed_cells'] >= 8
csv_path = BASE/'combined-results/results.csv'
before = csv_path.read_bytes()
(OUT/'before-results.csv').write_bytes(before)
key = lambda r: tuple(r[k] for k in ['method','model','benchmark','temperature'])
prior = {key(r):r for r in csv.DictReader(io.StringIO(before.decode()))}
assert len(prior) == 50
argv = ['/opt/conda/bin/python','-u',str(BASE/'collect_combined_results.py')]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with (OUT/'collector.log').open('x') as log:
    child = subprocess.Popen(argv, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT)
launch = dict(argv=argv, pid=child.pid, started_unix=time.time(), collector_sha256=sha(BASE/'collect_combined_results.py'))
save(OUT/'launcher.json', launch)
code = child.wait()
save(OUT/'exit.json', dict(launch, exit_code=code, finished_unix=time.time()))
assert code == 0, 'Preserve the collector log and inspect its failure.'
with csv_path.open() as stream:
    data = list(csv.DictReader(stream))
current = {key(r):r for r in data}
assert len(current) == len(data) == 51
assert all(current[k] == v for k,v in prior.items())
added = [r for k,r in current.items() if k not in prior]
assert len(added) == 1 and key(added[0]) == ('dflash','huginn-3.5b','mbppplus','1')
proof = read(BASE/'combined-results/evidence.json')['cells']['/'.join(key(added[0]))]
for source in proof['sources'].values():
    for name, digest in source['sha256'].items():
        assert sha(Path(source['path'])/name) == digest
for receipt in [*proof['supplement_exits'].values(), proof['original_timed_exit']]:
    assert sha(receipt['path']) == receipt['sha256'] and read(receipt['path'])['exit_code'] == 0
status = read(BASE/'combined-results/status.json')
assert status['completed'] == 51 and status['results_sha256'] == sha(csv_path)
save(OUT/'report.json', dict(status='passed', completed=51, expected=72,
    previous50_unchanged=True, collector_actual_exit_code=code, new_gpu_measurements=0,
    added=added, new_cell_evidence=proof, source_sha256=sha(Path(__file__)),
    prior_csv_sha256=sha(OUT/'before-results.csv'), results_sha256=sha(csv_path),
    collector_exit_sha256=sha(OUT/'exit.json'), checked_unix=time.time(),
    limitation='T1 same-seed agreement is descriptive; no sampling distribution conclusion from agreement alone.'))
print(dict(completed=51, unchanged=50, added=added))
