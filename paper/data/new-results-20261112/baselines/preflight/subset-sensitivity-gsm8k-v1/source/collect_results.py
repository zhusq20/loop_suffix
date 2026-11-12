"""Build paper rows only from complete, hash-verified formal benchmark journals."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
MODELS=['huginn-3.5b','ouro-2.6b','ouro-1.4b']
TASKS=['gsm8k','math500','bbh','humanevalplus','mbppplus','specbench']
METHODS=['dflash','loopahead']
VALIDATED_TRAININGS=set()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(8*1024**2):h.update(block)
    return h.hexdigest()


def read_run(path,protocol,profile=False):
    meta=json.loads((path/'metadata.json').read_text())
    complete=json.loads((path/'complete.json').read_text())
    if profile:
        assert meta.get('profiling') and meta['method']=='dflash' and not meta['reportable']
    else:
        assert meta['reportable']
    assert not meta.get('pilot',False) and meta.get('stage','formal')=='formal'
    assert not meta.get('trace',False) and not meta.get('random_lad',False)
    assert not meta.get('paper_check',False) and not meta.get('dev_calibration',False)
    assert meta['dataset'] in TASKS and meta['model'] in MODELS
    assert meta['count']==meta['full_count']==protocol['inputs'][meta['dataset']]['count']
    assert meta['input_sha256']==protocol['inputs'][meta['dataset']]['sha256']
    assert meta['max_new_tokens']==512 and meta['temperature'] in [0.,1.]
    assert meta['top_p']==1. and meta['top_k']==0 and meta['seed']==0
    assert meta.get('dtype',meta.get('precision'))=='bfloat16'
    assert not meta['graph'] and not meta['compile'] and len(meta['cpu_affinity'])==4
    assert complete['requests']==meta['count'] and sha(path/'requests.jsonl')==complete['requests_sha256']
    rows=[json.loads(l) for l in (path/'requests.jsonl').open()]
    expected=[json.loads(l) for l in (ROOT/f'results/loopfill-latency512-20261107/inputs/{meta["dataset"]}.jsonl').open()]
    assert [r['source_id'] for r in rows]==[r['source_id'] for r in expected]
    for row,source in zip(rows,expected):
        assert row['prompt_sha256']==hashlib.sha256(source['prompt'].encode()).hexdigest()
        assert 1<=len(row['tokens'])<=512
    stack='official' if 'denominator' in meta else 'ours'
    token_key='decode_tokens' if stack=='official' else 'tokens'
    depth=32 if meta['model']=='huginn-3.5b' else 4
    n=sum(r['stats'][token_key] for r in rows)
    assert n==sum(len(r['tokens'])-1 for r in rows)
    seconds=sum(r['stats']['decode_s'] for r in rows)
    loops=sum(r['stats']['target_verifications']*depth if stack=='official' else r['stats']['calls'] for r in rows)
    assert n>0 and loops>0 and math.isfinite(seconds) and seconds>0
    summary=dict(decode_tokens=n,decode_s=seconds,loops=loops,tok_s=n/seconds,gamma=depth*n/loops)
    if meta['method']=='ar':assert summary['gamma']==1.
    return dict(path=path,meta=meta,rows=rows,stack=stack,summary=summary)


def validate_training(run):
    meta=run['meta'];method=meta['method']
    checkpoint=Path(meta['checkpoint'])
    if method=='loopahead':checkpoint=checkpoint.parent
    train=checkpoint.parent
    complete=json.loads((train/'complete.json').read_text())
    identity=json.loads((train/'metadata.json').read_text())
    validation_key=(method,meta['model'],str(checkpoint.resolve()),json.dumps(meta['checkpoint_sha256'],sort_keys=True),sha(train/'metadata.json'))
    if validation_key in VALIDATED_TRAININGS:return
    manifests=json.loads((BASE/'data/manifests/manifest.json').read_text())['files']
    assert identity['model']==meta['model']
    if method=='dflash':
        assert identity['epochs']==6 and not identity['smoke'] and identity['global_batch']==32
        expected=manifests['dflash-train']
        assert expected['count']==50000
        assert sha(BASE/'data/manifests/dflash-train.jsonl')==expected['sha256']
        assert identity['examples']+len(identity['excluded_under_32_tokens'])==expected['count']
        assert all(s['metadata']['manifest_sha256']==expected['sha256'] for s in identity['sources'])
        assert len(identity['sources'])==identity['sources'][0]['metadata']['shards']
        assert {s['metadata']['shard'] for s in identity['sources']}==set(range(identity['sources'][0]['metadata']['shards']))
        assert all(sha(Path(s['path']))==s['sha256'] for s in identity['sources'])
        assert complete['step']==identity['total_updates']==json.loads((checkpoint/'complete.json').read_text())['step']
        for name,expected in meta['checkpoint_sha256'].items():assert sha(checkpoint/name)==expected
    else:
        assert identity['epochs']==3 and Path(complete['selected_checkpoint']).resolve()==checkpoint.resolve()
        assert (train/'confirmation.json').exists()
        assert sha(checkpoint/'lad.pt')==meta['checkpoint_sha256']
        for stage in ['train','dev','confirmation']:
            expected=manifests['loopahead-'+stage]
            manifest_path=BASE/f'data/manifests/loopahead-{stage}.jsonl'
            assert sha(manifest_path)==expected['sha256']
            allowed={json.loads(l)['source_id'] for l in manifest_path.open()}
            ids=[]
            for source in identity['feature_sources'][stage]:
                path=Path(source['path']);assert sha(path)==source['sha256']
                index=json.loads(path.read_text())
                assert index['identity']['teacher_identity']['manifest_sha256']==expected['sha256']
                assert index['frozen_components_sha256']==identity['components_sha256']
                ids += [sid for chunk in index['chunks'] for sid in chunk['source_ids']]
            assert len(ids)==len(set(ids))==expected['count'] and set(ids)==allowed
    VALIDATED_TRAININGS.add(validation_key)


def compare(method,ar):
    for key in ['model','model_revision','dataset','temperature','top_p','top_k','seed','count','max_new_tokens','max_seq','input_sha256','stop_audit','torch','transformers']:
        assert method['meta'][key]==ar['meta'][key],key
    assert method['stack']==ar['stack']
    exact=0;matched=0;denom=0;first=[]
    for m,a in zip(method['rows'],ar['rows']):
        assert m['source_id']==a['source_id']
        x,y=m['tokens'],a['tokens']
        exact+=int(x==y);matched+=sum(i==j for i,j in zip(x,y));denom+=max(len(x),len(y))
        if x!=y:
            pos=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)))
            first.append(dict(source_id=m['source_id'],position=pos,method_length=len(x),ar_length=len(y)))
    return dict(match_rate=exact/len(method['rows']),aligned_token_match_rate=matched/denom,
                exact_sequences=exact,requests=len(method['rows']),first_divergences=first)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,default=BASE/'runs')
    p.add_argument('--partial',action='store_true',help='export completed cells only; never imply full matrix completion')
    args=p.parse_args()
    protocol=json.loads((ROOT/'results/loopfill-latency512-20261107/protocol.json').read_text())
    runs={};profiles={}
    for path in sorted(args.runs.glob('*/complete.json')):
        folder=path.parent;meta=json.loads((folder/'metadata.json').read_text())
        profile=meta.get('profiling',False)
        r=read_run(folder,protocol,profile)
        key=(r['stack'],meta['model'],meta['dataset'],int(meta['temperature']),meta['method'])
        mapping=profiles if profile else runs
        assert key not in mapping,f'Duplicate completed run for{key}'
        mapping[key]=r
    result=[];audits={};missing=[]
    for model,temp,method,task in itertools.product(MODELS,[0,1],METHODS,TASKS):
        stack='official' if method=='dflash' else 'ours';key=(stack,model,task,temp,method)
        if key not in runs or (*key[:4],'ar') not in runs:
            missing.append(key);continue
        r,a=runs[key],runs[(*key[:4],'ar')];validate_training(r)
        agreement=compare(r,a);s,t=r['summary'],a['summary'];profile_fraction='NA'
        if method=='dflash':
            if key not in profiles:
                missing.append((*key,'draft_profile'));continue
            pro=profiles[key];compare(pro,a)
            assert pro['meta']['checkpoint_sha256']==r['meta']['checkpoint_sha256']
            assert [x['tokens'] for x in pro['rows']]==[x['tokens'] for x in r['rows']]
            profile_fraction=sum(x['stats']['draft_gpu_s'] for x in pro['rows'])/pro['summary']['decode_s']
        notes=['η in CSV is pooled throughput ratio; table uses latency ratio',
               'match_rate is exact whole-sequence fraction; aligned token fraction separately reported']
        if temp:notes.append('Same-seed token agreement is descriptive, not a sampling distribution test')
        if method=='dflash':notes.append('Official DFlash model/loss with full-loop target and timing/rejection adapter; separate full-task draft profiling run')
        else:notes.append('Own paper-based implementation; no official author code/weights; B4 gamma reference unavailable')
        provenance=a['meta'].get('measurement_provenance',{})
        if provenance.get('kind')=='archived_same_host_AR':
            notes.append('AR denominator is the original same-host frozen-table measurement, not a new run; cross-date timing drift is uncorrected')
        if model=='huginn-3.5b' and method=='loopahead':notes.append('Huginn migration, not a model evaluated in the LoopAhead paper')
        item=dict(method=method,model=model,benchmark=task,temperature=temp,eta=s['tok_s']/t['tok_s'],gamma=s['gamma'],
            method_tok_s=s['tok_s'],ar_tok_s=t['tok_s'],drafter_time_frac=profile_fraction,match_rate=agreement['match_rate'],
            engine=stack,status='complete_with_declared_limitations',notes='; '.join(notes),
            eta_latency=t['decode_s']/s['decode_s'],eta_throughput=s['tok_s']/t['tok_s'],
            aligned_token_match_rate=agreement['aligned_token_match_rate'],method_decode_tokens=s['decode_tokens'],
            ar_decode_tokens=t['decode_tokens'],method_decode_s=s['decode_s'],ar_decode_s=t['decode_s'],
            ar_provenance=provenance.get('kind','new_same_stack_AR'),
            ar_first_completed_unix=provenance.get('first_request_completed_unix',a['rows'][0].get('finished_unix')),
            ar_last_completed_unix=provenance.get('last_request_completed_unix',a['rows'][-1].get('finished_unix')),
            method_run=str(r['path'].resolve()),ar_run=str(a['path'].resolve()))
        result.append(item);audits['/'.join(map(str,key))]=agreement
    if not args.partial:assert not missing and len(result)==72,f'Matrix incomplete:{missing}'
    if not result:raise RuntimeError('No complete formal cells; refusing to publish an empty results table')
    with (BASE/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    bykey={(r['model'],r['temperature'],r['method'],r['benchmark']):r for r in result}
    text='| Model | Temp. | Method | GSM8K η γ | MATH-500 η γ | BBH η γ | HumanEval+ η γ | MBPP+ η γ | SpecBench η γ | Mean η γ |\n'
    text+='|'+'---|'*10+'\n'
    for model,temp,method in itertools.product(MODELS,[0,1],METHODS):
        cells=[bykey.get((model,temp,method,task)) for task in TASKS]
        if not any(cells):continue
        values=[f'{c["eta_latency"]:.2f} {c["gamma"]:.2f}' if c else 'pending' for c in cells]
        mean=f'{statistics.mean(c["eta_latency"] for c in cells):.2f} {statistics.mean(c["gamma"] for c in cells):.2f}' if all(cells) else 'pending'
        label='DFlash*' if method=='dflash' else 'LoopAhead†'+(' (Huginn适配)' if model=='huginn-3.5b' else '')
        text+='| '+' | '.join([model,str(temp),label,*values,mean])+' |\n'
    text+='\nη按原LoopFill表的总decode时间比；六任务Mean为算术平均。CSV另报吞吐比。\n\n'
    text+='* DFlash使用官方草稿模型/损失和本地循环目标适配，η相对其同栈AR；训练数据版本/规模差异见DESIGN。草稿耗时来自独立完整题集profiling。\n\n'
    text+='† LoopAhead为按论文自行复现，无官方训练代码或LAD权重；Huginn为迁移。B4对照及BF16数值限制见NOTES。\n'
    text+='LoopAhead沿用原表同一主机、同一冻结配置的AR测量；原始时间戳保留于CSV与导入记录，不算作本次新测。跨日期时延漂移未作修正，独立GSM8K全300题环境复测见preflight/calibration-n300.md。\n'
    if missing:text+=f'\n部分结果：已完成{len(result)}/72格，不是最终完整表。\n'
    (BASE/'table2_rows.md').write_text(text)
    (BASE/'formal-agreement.json').write_text(json.dumps(audits,indent=2)+'\n')
    (BASE/'formal-matrix-status.json').write_text(json.dumps(dict(completed=len(result),expected=72,missing=missing),indent=2)+'\n')
    print(json.dumps(dict(completed=len(result),expected=72,missing=len(missing))))


if __name__=='__main__':main()
