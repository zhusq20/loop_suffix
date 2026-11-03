"""Summarize the transfer pilot, keeping diagnostic and timing runs separate."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('results/loop-calibration-transfer-20261103')


def metrics(a):
    n=a[0]
    if not n:return [None]*3
    brier=a[3]/n
    # bins: count, sum probability, sum hit
    bins=a[4:].reshape(10,3)
    ece=np.abs(bins[:,1]-bins[:,2]).sum()/n
    bias=(a[1]-a[2])/n
    return [float(brier),float(ece),float(bias)]


def main():
    result={'scope':'exploratory dev12 transfer, 256-token cap; not formal OOD generalization or code quality evidence','reliability':[],'performance':[]}
    for model in ['ouro-2.6b','huginn-3.5b']:
        for domain in ['id','ood']:
            d=ROOT/f'diagnostic-{model}-{domain}'
            if not (d/'complete.json').exists():continue
            ids=[json.loads(s)['request_id'] for s in (d/'requests.jsonl').read_text().splitlines()]
            groups={};fallback={};zero={};eventcounts={};strata=defaultdict(lambda: defaultdict(lambda: np.zeros(34)))
            for line in (d/'events.jsonl').open():
                e=json.loads(line);source=e['source'];rid=e['request_id'];p=e['probability'];hit=int(e['hit'])
                table=groups.setdefault(source,{i:np.zeros(34) for i in ids});a=table[rid]
                a[:4]+=np.array([1,p,hit,(p-hit)**2]);b=min(int(p*10),9)
                a[4+b*3:7+b*3]+=np.array([1,p,hit])
                tags=[]
                if source=='shallow' and e['features'][1]==0:tags.append('shallow_top1')
                if source=='joint' and e['features'][2]==0:tags.append('joint_deep_top1')
                if p>=.1:tags.append(source+'_p_ge_0.1')
                for tag in tags:
                    z=strata[tag][rid];z[:4]+=np.array([1,p,hit,(p-hit)**2]);z[4+b*3:7+b*3]+=np.array([1,p,hit])
                trace=e['lookup'];fallback[source]=fallback.get(source,0)+int(any(x['level']!=0 for x in trace))
                zero[source]=zero.get(source,0)+int(bool(trace) and trace[-1]['level'] is None)
                eventcounts[source]=eventcounts.get(source,0)+1
            for source,table in groups.items():
                a=np.stack(list(table.values()));point=metrics(a.sum(0));rng=np.random.default_rng(20261103)
                boot=np.array([metrics(a[rng.integers(len(a),size=len(a))].sum(0)) for _ in range(1000)],dtype=float)
                ci=np.nanquantile(boot,[.025,.975],axis=0).T.tolist()
                bins=a.sum(0)[4:].reshape(10,3)
                result['reliability'].append(dict(model=model,domain=domain,source=source,requests=len(ids),events=eventcounts[source],
                    brier=point[0],ece=point[1],predicted_minus_observed=point[2],request_bootstrap_95ci=dict(zip(['brier','ece','bias'],ci)),
                    fallback_event_fraction=fallback[source]/eventcounts[source],unsupported_default_fraction=zero[source]/eventcounts[source],
                    bins=[dict(n=int(n),predicted=float(p/n) if n else None,observed=float(y/n) if n else None) for n,p,y in bins]))
            result.setdefault('stratified_reliability',[])
            for tag,table in strata.items():
                total=sum(table.values());m=metrics(total)
                result['stratified_reliability'].append(dict(model=model,domain=domain,stratum=tag,events=int(total[0]),brier=m[0],ece=m[1],predicted_minus_observed=m[2]))
            perf=ROOT/f'performance-{model}-{domain}'
            summaries=[json.loads(p.read_text()) for p in perf.glob('*-summary.json')]
            ar=next((s for s in summaries if s['arm']=='ar'),None)
            arfile=perf/'ar-b1-seed0.jsonl'
            aroutputs={r['request_id']:r['tokens'] for r in map(json.loads,arfile.read_text().splitlines())} if arfile.exists() else {}
            for s in summaries:
                methodfile=perf/f"{s['arm']}-b1-seed0.jsonl"
                methodrows=list(map(json.loads,methodfile.read_text().splitlines())) if methodfile.exists() else []
                equal=sum(aroutputs.get(r['request_id'])==r['tokens'] for r in methodrows)
                result['performance'].append(dict(model=model,domain=domain,arm=s['arm'],requests=s['requests'],output_tokens=s['output_tokens'],wall_s=s['wall_s'],tput=s['output_tput'],speedup_vs_ar=s['output_tput']/ar['output_tput'] if ar else None,core_rows=s['core_rows'],cap_skips=s['cap_skips'],same_as_ar=equal))
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'reliability_groups':len(result['reliability']),'performance_cells':len(result['performance'])}))


if __name__=='__main__':main()
