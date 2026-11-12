"""Freeze Ouro sampling admission from the prespecified16 development requests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from loopahead.admission import quality_counts


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(path):
    meta=json.loads((path/'metadata.json').read_text())
    complete=json.loads((path/'complete.json').read_text())
    assert meta['dev_calibration'] and not meta['reportable']
    assert complete['requests']==meta['count']==16
    assert digest(path/'requests.jsonl')==complete['requests_sha256']
    return meta,[json.loads(line) for line in (path/'requests.jsonl').open()]


def incremental_fit(events,name):
    rows=[e for e in events if e['name']==name and e['rows']>0]
    n=np.array([e['rows'] for e in rows],dtype=float)
    position=np.array([e['prefix_length']+e['node_depth_mean'] for e in rows],dtype=float)
    duration=np.array([e['wall_s'] for e in rows],dtype=float)
    assert len(rows)>100 and len(set(n))>=3
    center=float(np.median(position))
    # Include context length and its interaction with row count. Evaluate the
    # incremental row cost at the observed median context, without test inputs.
    x=np.stack([np.ones_like(n),n,(position-center)/1000,n*(position-center)/1000],axis=1)
    coef,_,rank,_=np.linalg.lstsq(x,duration,rcond=None)
    residual=duration-x@coef
    raw=float(coef[1])
    return dict(name=name,observations=len(rows),rows=sorted(set(map(int,n))),
        median_position=center,coefficients=coef.tolist(),rank=int(rank),
        residual_rms_s=float(np.sqrt(np.mean(residual**2))),
        incremental_raw_s=raw,incremental_nonnegative_s=max(0.,raw),
        formula='duration = b0 + b1*rows + b2*(position-median)/1000 + b3*rows*(position-median)/1000')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--native-run',type=Path,required=True)
    p.add_argument('--profile-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    native,native_rows=read_run(args.native_run)
    profile,profile_rows=read_run(args.profile_run)
    assert native['native_only'] and not native['profile_dev']
    assert profile['profile_dev'] and not profile['native_only']
    assert native['model'].startswith('ouro')
    for key in ['model','model_revision','checkpoint_sha256','temperature','input_sha256','dtype','max_new_tokens']:
        assert native[key]==profile[key],key
    assert profile['temperature']==1. and profile['dtype']=='bfloat16'
    assert [r['source_id'] for r in native_rows]==[r['source_id'] for r in profile_rows]
    events=[e for r in profile_rows for e in r['profile']]
    head=[e['wall_s'] for e in events if e['name']=='lad_propose' and e['produced_tokens']]
    assert len(head)>10
    advance,readout=(incremental_fit(events,name) for name in ['advance','lad_readout'])
    costs=dict(t_save_s=sum(r['stats']['decode_s'] for r in native_rows)/sum(r['stats']['calls'] for r in native_rows),
        t_head_s=float(np.median(head)),
        t_node_s=4*advance['incremental_nonnegative_s']+3*readout['incremental_nonnegative_s'])
    record=dict(split='development_only',model=native['model'],checkpoint_sha256=native['checkpoint_sha256'],
        source_ids=[r['source_id'] for r in native_rows],costs=costs,quality=quality_counts(profile_rows),
        smoothing_strength=4.,confidence_bins=[0.,.25,.5,.75,1.],
        source_runs={str(p.resolve()):digest(p/'requests.jsonl') for p in [args.native_run,args.profile_run]},
        cost_measurement=dict(advance=advance,readout=readout,head_calls=len(head),head_median_s=float(np.median(head))),
        local_defaults=['t_save is the native-only unprofiled pipeline wall time per physical loop cycle',
            't_head is median synchronized actual LAD call including context, projection, vocabulary head and CPU token transfer',
            't_node is4 advance incremental row costs plus3 shallow/final readout row costs; negative estimated row costs clipped to0',
            'Costs are measured on16 fixed dev prompts only; author calibration code is unavailable',
            'Beta(1,1) pooled probabilities; strength4 hierarchical shrinkage for distance and confidence cells'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:
        json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(dict(model=native['model'],costs=costs,head_calls=len(head))))


if __name__=='__main__':main()
