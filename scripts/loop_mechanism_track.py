"""Track an already running mechanism suite without touching its timed processes."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import time

from loop_table2_suite import read, save, digest, process_identity


def track(out, key_file):
    # Existing experiment credential convention. Never emit the credential.
    if 'WANDB_API_KEY' not in os.environ and key_file.exists():
        os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
    import wandb
    p = read(out/'protocol.json')
    run = wandb.init(entity=p['entity'], project=p['project'], group=p['group'],
        name=out.parent.name+'-timing', job_type='mechanism-tracking', dir=str(out),
        config=dict(protocol_sha256=digest(out/'protocol.json'), requests_per_stream=24,
                    measurement_streams=32, tracking_script_sha256=digest(Path(__file__))),
        settings=wandb.Settings(disable_git=True,disable_code=True,console='off',init_timeout=90))
    save(out/'tracking.json',dict(pid=os.getpid(),url=run.url,started_unix=time.time(),
         script=str(Path(__file__).resolve()),script_sha256=digest(Path(__file__)),
         note='Sidecar uses the existing experiment key file; original unauthenticated lane errors retained.'))
    for name in ['protocol.json','runtime.json']:
        run.save(str(out/name),base_path=str(out),policy='now')
    logged=set()
    supervisor=read(out/'supervisor.json')['pid']
    identity=process_identity(supervisor)
    while True:
        jobs=[read(f) for f in sorted((out/'jobs').glob('*.json'))]
        for job in jobs:
            if job['key'] in logged or job['outcome'] not in ['complete','failed']:
                continue
            values={k:job[k] for k in ['model','batch','arm','mode','phase']}
            values.update(job_key=job['key'],complete=job['outcome']=='complete')
            if job.get('summary'):
                values.update({job['mode']+'/'+k:job['summary'][k]
                               for k in ['wall_s','output_tokens','requests','output_tput']})
            run.log(values)
            logged.add(job['key'])
        counts=Counter(j['outcome'] for j in jobs if j['phase']=='measurement')
        run.log({'progress/'+k:v for k,v in counts.items()})
        if process_identity(supervisor)!=identity:
            break
        time.sleep(15)
    artifact=wandb.Artifact(out.parent.name+'-timing',type='mechanism-results')
    for name in ['protocol.json','runtime.json','status.json','measurements.csv','stages.csv',
                 'pairs.csv','audit.json','report.md','stage-timing.pdf']:
        if (out/name).exists():
            artifact.add_file(str(out/name),name=name)
    for path in sorted((out/'runs').glob('*/*-stages.json')):
        artifact.add_file(str(path),name=str(path.relative_to(out)))
    for name in ['loop_stage_timing.py','loop_mechanism_suite.py']:
        artifact.add_file(str(Path(p['source'])/'scripts'/name),name='scripts/'+name)
    run.log_artifact(artifact).wait()
    run.summary['measurement_complete']=counts['complete']
    run.summary['measurement_failed']=counts['failed']
    run.finish()
    state=read(out/'tracking.json')
    state.update(finished_unix=time.time(),measurement_counts=dict(counts),artifact_uploaded=True)
    save(out/'tracking.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--key-file',type=Path,default=Path('/tmp/loopfill-wandb-key'))
    args=parser.parse_args()
    track(args.out.resolve(),args.key_file)
