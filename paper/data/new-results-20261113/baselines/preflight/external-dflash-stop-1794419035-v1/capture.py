"""Preserve interrupted runs without changing queues, receipts, or results."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE.parents[1]
assert not (OUT / 'report.json').exists()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


files = {}


def preserve(path):
    raw = path.read_bytes()
    relative = path.relative_to(BASE)
    destination = OUT / 'snapshot' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    destination.write_bytes(raw)
    assert destination.read_bytes() == raw
    files[str(relative)] = {'sha256': sha(raw), 'bytes': len(raw)}
    return raw


head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
outputs = ['results.csv', 'table2_rows.md', 'formal-agreement.json', 'formal-matrix-status.json']
for name in outputs:
    raw = preserve(BASE / name)
    assert raw == subprocess.check_output(['git', 'show', f'{head}:experiments/baselines/{name}'], cwd=ROOT)
assert len(list(csv.DictReader(io.StringIO((BASE / 'results.csv').read_text())))) == 31
assert read(BASE / 'formal-matrix-status.json')['completed'] == 31
for path in sorted((BASE / 'logs/user-stop-dflash-1794419035').iterdir()):
    if path.is_file():
        preserve(path)
preserve(BASE / 'configs/gpu-allocation.json')
preserve(BASE / 'logs/formal-controller-v1/pause-new-jobs')
policy = read(OUT / 'snapshot/configs/gpu-allocation.json')
assert policy['allowed_gpus'] == [2, 3, 4, 6]
assert policy['released_gpus'] == [0, 1, 5, 7]
before = read(OUT / 'snapshot/logs/user-stop-dflash-1794419035/before.json')
signals = read(OUT / 'snapshot/logs/user-stop-dflash-1794419035/signals.json')
original_processes = {p['pid']: p for p in before['processes']}
stages = [
    ('math500', 0, 'ar', 495, 174380, 174369),
    ('math500', 1, 'ar', 491, 174381, 174370),
    ('bbh', 0, 'dflash-profile', 36, 401558, 252990),
    ('specbench', 0, 'ar', 40, 401821, 401812),
]
report = []
for dataset, temperature, stage, count, pid, worker_pid in stages:
    name = f'dflash-stack-huginn-3.5b-{dataset}-t{temperature}-{stage}-v1'
    run = BASE / 'runs' / name
    assert not (run / 'complete.json').exists()
    for path in sorted(run.iterdir()):
        if path.is_file():
            preserve(path)
    for suffix in ['-launcher.json', '-finished.json', '.log']:
        preserve(BASE / 'logs' / (name + suffix))
    job = BASE / 'logs/formal-controller-v1' / f'dflash-huginn-3.5b-{dataset}-t{temperature}.json'
    preserve(job)
    worker_path = job.with_name(job.stem + '-worker-exit.json')
    preserve(worker_path)
    meta = read(run / 'metadata.json')
    finish = read(BASE / 'logs' / (name + '-finished.json'))
    worker = read(worker_path)
    assert finish['pid'] == pid and finish['exit_code'] == worker['exit_code'] == 1
    assert worker['worker_pid'] == worker_pid
    assert pid in signals['evaluators_sigint']
    assert finish['argv'] == original_processes[pid]['argv']
    assert signals['time_unix'] <= finish['finished_unix'] <= worker['finished_unix']
    assert 'KeyboardInterrupt' in (BASE / 'logs' / (name + '.log')).read_text()
    for old_pid in [pid, worker_pid]:
        stat = Path(f'/proc/{old_pid}/stat')
        if stat.exists():
            fields = stat.read_text().rsplit(') ', 1)[1].split()
            assert fields[19] != str(original_processes[old_pid]['start_ticks']) or fields[0] == 'Z'
    rows = [json.loads(line) for line in (run / 'requests.jsonl').read_bytes().splitlines()]
    source = ROOT / f'results/loopfill-latency512-20261107/inputs/{dataset}.jsonl'
    expected = [json.loads(line) for line in source.read_bytes().splitlines()]
    assert sha(source.read_bytes()) == meta['input_sha256']
    assert len(rows) == count < meta['count'] == len(expected)
    for row, original in zip(rows, expected):
        assert row['source_id'] == original['source_id']
        assert row['prompt_sha256'] == sha(original['prompt'].encode())
        assert 1 <= len(row['tokens']) <= 512
        assert row['stats']['decode_tokens'] == len(row['tokens']) - 1
        assert math.isfinite(row['stats']['decode_s']) and row['stats']['decode_s'] > 0
        assert finish['started_unix'] <= row['finished_unix'] <= finish['finished_unix']
    progress = read(run / 'progress.json')
    assert progress['requests'] == count
    assert progress['decode_tokens'] == sum(r['stats']['decode_tokens'] for r in rows)
    assert math.isclose(progress['decode_s'], sum(r['stats']['decode_s'] for r in rows), rel_tol=1e-12)
    report.append({'run': name, 'completed_prefix': count, 'expected': len(expected),
                   'actual_exit_code': 1, 'summary': progress, 'admitted_as_complete': False})

result = {
    'status': 'interruption_preserved_scope_clarification_pending',
    'checked_unix': time.time(), 'head': head, 'completed_formal_cells': 31,
    'formal_outputs_unchanged': True, 'interrupted_stages': report,
    'scope_note': 'An external shared-file record claims a user-requested scope change. This is not treated as an instruction in this thread; clarification is pending.',
    'actions': 'Copied evidence only. No process signals, queue edits, result admission, or new GPU jobs.',
    'snapshot_files': files, 'capture_source_sha256': sha(Path(__file__).read_bytes()),
}
(OUT / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ['snapshot_files', 'interrupted_stages']}))
