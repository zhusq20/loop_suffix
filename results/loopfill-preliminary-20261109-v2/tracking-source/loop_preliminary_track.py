"""Track a frozen preliminary run without putting network I/O in model workers."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import time

from loop_preliminary_suite import read, save, sha


def track(args):
    import wandb
    out = Path(args.out).resolve()
    lock = (out / 'tracking.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    defaults = read(args.defaults)
    if not defaults['enabled']:
        raise ValueError('tracking is disabled in defaults')
    os.environ['WANDB_API_KEY'] = Path(args.key_file).read_text().strip()
    os.environ['WANDB_SILENT'] = 'true'
    run = wandb.init(entity=defaults['entity'], project=defaults['project'],
        name=out.name, group='loopfill-preliminary', job_type='coverage-probe',
        id=hashlib.sha256(out.name.encode()).hexdigest()[:12], resume='allow',
        dir=str(out), config=read(out / 'protocol.json'), mode=defaults['mode'],
        settings=wandb.Settings(init_timeout=60))
    save(out / 'tracking.json', dict(pid=os.getpid(), url=run.url, state='running',
        tracker_sha256=sha(__file__), started_unix=time.time()))
    uploaded_summaries = set()
    while True:
        status = read(out / 'status.json')
        metrics = {'progress/completed_requests': status.get('completed_requests', 0),
                   'progress/planned_requests': status.get('planned_requests', 480),
                   'progress/formal_phase': int(status.get('phase') == 'full')}
        for phase in ('pilot', 'full'):
            complete = out / phase / 'complete.json'
            if not complete.exists() or phase in uploaded_summaries:
                continue
            summary = read(out / phase / 'summary.json')
            for key, value in summary['metrics'].items():
                if value['mean'] is not None:
                    for i, mean in enumerate(value['mean']):
                        metrics[f'{phase}/{key}_{i + 1}'] = mean
            metrics[f'{phase}/positions'] = summary['positions']
            metrics[f'{phase}/figure'] = wandb.Image(str(out / phase / 'preliminary.png'))
            uploaded_summaries.add(phase)
        run.log(metrics)
        run.summary['state'] = status['state']
        if status['state'] in ('complete', 'failed'):
            break
        time.sleep(defaults.get('poll_seconds', 15))
    artifact = wandb.Artifact(out.name, type='preliminary-coverage-experiment',
                             metadata={'protocol_sha256': sha(out / 'protocol.json'),
                                       'state': status['state']})
    manifest = {}
    selected = [out / 'protocol.json', out / 'questions.jsonl', out / 'status.json']
    for directory in ('source', 'suffix', 'pilot', 'full', 'logs'):
        selected.extend(p for p in (out / directory).rglob('*') if p.is_file()
                        and p.suffix in ('.json', '.jsonl', '.csv', '.npz', '.pdf', '.png',
                                         '.py', '.md', '.so', '.txt', '.log'))
    for path in selected:
        name = str(path.relative_to(out))
        artifact.add_file(str(path), name=name)
        manifest[name] = sha(path)
    save(out / 'artifact-manifest.json', manifest)
    artifact.add_file(str(out / 'artifact-manifest.json'), name='artifact-manifest.json')
    logged = run.log_artifact(artifact)
    logged.wait(timeout=600)
    save(out / 'tracking.json', dict(pid=os.getpid(), url=run.url, state='uploaded',
        artifact=logged.qualified_name, files=len(manifest), finished_unix=time.time(),
        tracker_sha256=sha(__file__)))
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--key-file', required=True)
    parser.add_argument('--defaults', default='configs/experiment-tracking-defaults.json')
    arguments = parser.parse_args()
    try:
        track(arguments)
    except Exception as exc:
        save(Path(arguments.out) / 'tracking-error.json',
             dict(error_type=type(exc).__name__, message=str(exc), time=time.time()))
        raise
