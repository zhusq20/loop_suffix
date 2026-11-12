"""Aggregate measured production/training reservations without double-counting exports.

Running jobs have no final reservation record and are listed as incomplete. The
ledger is not a claim to cover CPU work or all earlier diagnostic GPU work.
"""
from collections import defaultdict
import json
from pathlib import Path
import time

BASE=Path(__file__).resolve().parent


def main():
    records=[];pending=[]
    for category,root in [('generation',BASE/'data/teachers'),('features',BASE/'features'),('training',BASE/'checkpoints')]:
        for path in sorted(root.glob('**/reservation-cost*.json*')):
            if 'source' in path.relative_to(root).parts:continue
            values=[json.loads(l) for l in path.open()] if path.suffix=='.jsonl' else [json.loads(path.read_text())]
            for line,value in enumerate(values,1):
                seconds=value['reserved_gpu_seconds']
                assert seconds>=0 and value['finished_unix']>=value['started_unix']
                assert abs(seconds-(value['finished_unix']-value['started_unix']))<1.
                records.append(dict(category=category,path=str(path.relative_to(BASE)),line=line,**value))
        for path in sorted(root.glob('**/metadata.json')):
            if 'source' in path.relative_to(root).parts or 'checkpoint-' in str(path.parent.name):continue
            if not (path.parent/'complete.json').exists():
                pending.append(dict(category=category,path=str(path.parent.relative_to(BASE)),
                    failed=bool(list(path.parent.glob('failure*.json'))),
                    note='No successful completion marker; final cost may not yet exist'))
    groups=defaultdict(lambda:dict(gpu_seconds=0.,reservations=0))
    for r in records:
        k=r['category']+'/'+r['status'];groups[k]['gpu_seconds']+=r['reserved_gpu_seconds'];groups[k]['reservations']+=1
    for v in groups.values():v['gpu_hours']=v['gpu_seconds']/3600
    report=dict(generated_unix=time.time(),groups=dict(groups),reservations=records,incomplete=pending,
        coverage='Finished reservation records in data/teachers,features,checkpoints only. Excludes duplicate training exports, archived AR, CPU work and separate environment/inference diagnostics. Running reservations are not estimated.',
        total_recorded_gpu_hours=sum(r['reserved_gpu_seconds'] for r in records)/3600)
    out=BASE/'training/cost-ledger.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['reservations','incomplete']}))


if __name__=='__main__':main()
