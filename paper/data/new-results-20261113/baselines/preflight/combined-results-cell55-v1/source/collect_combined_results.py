"""Reconcile legacy and packed measurements, retaining each actual AR denominator.

Writes a separate partial table while the frozen legacy collector remains active.
Existing completed legacy cells take precedence; extra backend measurements are
retained as variants, never silently substituted or counted twice.
"""
import argparse
import csv
import itertools
from pathlib import Path
import statistics
import time

from packed_dflash_evidence import (check_frozen, check_plan, check_profile, check_run,
                                    compare as compare_packed, read, rows, save, sha)
from collect_results import MODELS, TASKS, METHODS, read_run, compare, validate_training
from result_evidence import validate_run_artifacts

BASE = Path(__file__).resolve().parent


def key(row):
    return row['method'], row['model'], row['benchmark'], int(row['temperature'])


def artifact_record(path):
    path = Path(path)
    names = ['metadata.json','complete.json','requests.jsonl']
    names += [name for name in ['summary.json','source-manifest.json'] if (path/name).exists()]
    return dict(path=str(path),sha256={name:sha(path/name) for name in names})


def read_legacy():
    csv_path = BASE/'results.csv'
    before = sha(csv_path)
    original = list(csv.DictReader(csv_path.open()))
    assert len({key(r) for r in original}) == len(original)
    protocol = read(BASE.parents[1]/'results/loopfill-latency512-20261107/protocol.json')
    agreements = read(BASE/'formal-agreement.json')
    result, evidence = {}, {}
    for row in original:
        method, ar = [read_run(Path(row[name]),protocol) for name in ['method_run','ar_run']]
        validate_run_artifacts(method)
        validate_run_artifacts(ar)
        validate_training(method)
        agreement = compare(method,ar)
        assert agreement == agreements['/'.join([row['engine'],row['model'],row['benchmark'],row['temperature'],row['method']])]
        m, a = method['summary'], ar['summary']
        for name,value in dict(eta=m['tok_s']/a['tok_s'],eta_latency=a['decode_s']/m['decode_s'],
                              eta_throughput=m['tok_s']/a['tok_s'],gamma=m['gamma'],
                              method_tok_s=m['tok_s'],ar_tok_s=a['tok_s'],
                              match_rate=agreement['match_rate']).items():
            assert float(row[name]) == value, (key(row),name)
        item = dict(row,target_backend='legacy_HF_full_loop' if row['method']=='dflash' else 'packed_LoopAhead',
                    method_exit_evidence='actual_exit0',profile_run='',numerical_diagnostics='not_all_completed')
        sources = dict(method=artifact_record(method['path']),ar=artifact_record(ar['path']))
        if row['method']=='dflash':
            profile_path=method['path'].with_name(method['path'].name.replace('-dflash-v1','-dflash-profile-v1'))
            profile=read_run(profile_path,protocol,profile=True)
            validate_run_artifacts(profile)
            check_profile(profile['rows'],method['rows'])
            assert float(row['drafter_time_frac']) == sum(r['stats']['draft_gpu_s'] for r in profile['rows'])/profile['summary']['decode_s']
            item['profile_run']=str(profile_path)
            sources['profile']=artifact_record(profile_path)
        result[key(item)]=item
        evidence['/'.join(map(str,key(item)))]=dict(agreement=agreement,sources=sources,origin='verified_legacy_collector')
    assert sha(csv_path)==before, 'Legacy collector changed its table during inspection; retry from its new complete snapshot'
    return result,evidence,dict(path=str(csv_path),sha256=before,rows=len(original))


def check_supplement_exit(root,job,stage,run):
    name=job['id']+'-'+stage
    candidates=[root/'logs'/f'{name}-exit.json',root.parent/'logs'/f'{name}-exit.json']
    receipt_path=next((p for p in candidates if p.exists()),None)
    assert receipt_path is not None, name
    launcher=read(receipt_path.with_name(receipt_path.name.replace('-exit.json','-launcher.json')))
    receipt=read(receipt_path)
    assert receipt['exit_code']==0
    assert all(receipt[k]==launcher[k] for k in ['pid','argv','started_unix'])
    argv=receipt['argv']
    assert argv[argv.index('--job')+1]==job['id'] and argv[argv.index('--stage')+1]==stage
    assert Path(argv[argv.index('--out')+1]).resolve()==Path(run['path']).resolve()
    done=read(Path(run['path'])/'complete.json')
    assert all(receipt['started_unix']<=r['finished_unix']<=done['finished_unix']<=receipt['finished_unix'] for r in run['rows'])
    return dict(path=str(receipt_path),sha256=sha(receipt_path),exit_code=0)


def check_packed_training(suite,job):
    protocol=read(suite/'protocol.json');model=protocol['models'][job['model']]
    checkpoint=Path(model['checkpoint'])
    assert sha(checkpoint.parent/'complete.json')==model['training_complete_sha256']
    metadata,complete=read(checkpoint.parent/'metadata.json'),read(checkpoint.parent/'complete.json')
    assert metadata['model']==job['model'] and metadata['epochs']==6
    assert complete['step']==metadata['total_updates']==read(checkpoint/'complete.json')['step']
    if job['model']=='ouro-1.4b':
        path=Path(protocol['training_audit']['path'])
        assert sha(path)==protocol['training_audit']['sha256']
    else:
        name='dflash-huginn-narrow-formal-audit-v1' if job['model']=='huginn-3.5b' else 'dflash-ouro-2.6b-formal-audit-v1'
        path=BASE/'training'/name/'report.json'
    audit=read(path)
    assert audit['status'] in ['passed','complete_training_audit_not_benchmark']
    assert Path(audit['checkpoint']).resolve()==checkpoint.resolve()
    assert audit['weights_sha256']==model['checkpoint_sha256']['model.safetensors']
    for name,label in [('metadata.json','metadata_sha256'),('complete.json','complete_sha256'),('metrics.jsonl','metrics_sha256')]:
        if label in audit:assert sha(checkpoint.parent/name)==audit[label]
    return dict(path=str(path),sha256=sha(path),checkpoint=str(checkpoint))


def read_packed(root):
    if not (root/'measurement-results.json').exists():return [],{}
    plan=check_plan(root/'plan.json');suite=Path(plan['suite'])
    ready=read(root/'measurement-results.json');jobs={j['id']:j for j in plan['jobs']}
    result,evidence=[],{}
    checked_models=set()
    for exported in ready:
        identity=f"table1-{exported['model']}-{exported['benchmark']}-t{int(exported['temperature'])}"
        job=jobs[identity]
        if job['model'] not in checked_models:
            check_frozen(suite,job,weights=True);checked_models.add(job['model'])
        training=check_packed_training(suite,job)
        ar=check_run(Path(exported['ar_run']),suite,job,stage='ar')
        profile=check_run(Path(exported['profile_run']),suite,job,stage='profile')
        method=check_run(Path(exported['method_run']),suite,job)
        exits={stage:check_supplement_exit(root,job,stage,r) for stage,r in [('ar',ar),('profile',profile)]}
        bindings=[read(root/f'bindings/{identity}-{stage}.json') for stage in ['ar','profile']]
        assert bindings[0]['timed']==bindings[1]['timed']
        binding=bindings[0]['timed'];assert binding['hashes']==method['hashes'] and binding['path']==method['path']
        for stage,run in [('ar',ar),('profile',profile)]:
            meta=run['metadata']
            assert meta['plan_sha256']==sha(root/'plan.json')
            assert meta['binding_sha256']==sha(root/f'bindings/{identity}-{stage}.json')
            assert meta['supplement_source_sha256']==plan['source_sha256']
            assert meta['torch']==method['metadata']['torch'] and meta['cuda']==method['metadata']['cuda']
            assert [r['stats']['seed'] for r in run['rows']]==[r['stats']['seed'] for r in method['rows']]
        original_exit=binding['exit_evidence']
        if original_exit['actual_exit_code']==0:
            receipt_path=Path(original_exit['path'])
            assert sha(receipt_path)==original_exit['sha256'] and read(receipt_path)['exit_code']==0
            receipt=read(receipt_path);argv=receipt['argv']
            assert Path(argv[argv.index('--out')+1]).resolve()==Path(method['path']).resolve()
            assert argv[argv.index('--job')+1]==job['id']
        else:
            assert original_exit['actual_exit_code'] is None and original_exit['limitation']
        check_profile(profile['rows'],method['rows'])
        agreement=compare_packed(method['rows'],ar['rows'])
        m,a,p=method['summary'],ar['summary'],profile['summary']
        values=dict(eta=m['decode_token_s']/a['decode_token_s'],eta_latency=a['decode_s']/m['decode_s'],
            eta_throughput=m['decode_token_s']/a['decode_token_s'],gamma=m['gamma'],method_tok_s=m['decode_token_s'],
            ar_tok_s=a['decode_token_s'],drafter_time_frac=p['drafter_time_frac'],match_rate=agreement['match_rate'],
            aligned_token_match_rate=agreement['aligned_token_match_rate'])
        assert all(float(exported[k])==v for k,v in values.items())
        limitation=original_exit.get('limitation','')
        item=dict(method='dflash',model=job['model'],benchmark=job['dataset'],temperature=job['temperature'],
            **values,engine='official_dflash_with_packed_loop_target',status='complete_with_declared_limitations',
            notes='Official frozen DFlash drafter/decoder with our packed full-loop target adapter; fresh own-stack AR; '
                  'separate full-task profile matches every token and verification; numerical diagnostics not all completed. '+limitation,
            method_decode_tokens=m['decode_tokens'],ar_decode_tokens=a['decode_tokens'],
            method_decode_s=m['decode_s'],ar_decode_s=a['decode_s'],ar_provenance='new_same_packed_stack_AR',
            ar_first_completed_unix=ar['rows'][0]['finished_unix'],ar_last_completed_unix=ar['rows'][-1]['finished_unix'],
            method_run=method['path'],ar_run=ar['path'],profile_run=profile['path'],target_backend='packed_full_loop',
            method_exit_evidence='actual_exit0' if original_exit['actual_exit_code']==0 else 'unavailable_artifacts_verified',
            numerical_diagnostics='not_all_completed')
        result.append(item)
        evidence['/'.join(map(str,key(item)))]=dict(origin='verified_packed_supplement',agreement=agreement,
            sources={label:artifact_record(r['path']) for label,r in [('method',method),('ar',ar),('profile',profile)]},
            supplement_exits=exits,original_timed_exit=original_exit,training_audit=training)
    return result,evidence


def table_text(items):
    lookup={key(r):r for r in items}
    text='| Model | T | Method | GSM8K η γ | MATH-500 η γ | BBH η γ | HumanEval+ η γ | MBPP+ η γ | SpecBench η γ | Mean η γ |\n'
    text+='|'+'---|'*10+'\n'
    for model,temp,method in itertools.product(MODELS,[0,1],METHODS):
        cells=[lookup.get((method,model,task,temp)) for task in TASKS]
        if not any(cells):continue
        values=[]
        for row in cells:
            if row is None:values.append('pending');continue
            mark=(' H' if row['target_backend']=='legacy_HF_full_loop' else ' P') if method=='dflash' else ''
            values.append(f"{float(row['eta_latency']):.2f} {float(row['gamma']):.2f}{mark}")
        mean=f"{statistics.mean(float(c['eta_latency']) for c in cells):.2f} {statistics.mean(float(c['gamma']) for c in cells):.2f}" if all(cells) else 'pending'
        if all(cells) and len({c['target_backend'] for c in cells})>1:mean+=' (混合实现)'
        text+='| '+' | '.join([model,str(temp),'DFlash*' if method=='dflash' else 'LoopAhead†',*values,mean])+' |\n'
    text+='\nη为原表的总decode时间比；CSV另报吞吐比。Mean仅在六任务齐全时取算术平均。\n\n'
    text+='* H：保留的HF循环目标实现；P：packed循环目标实现。两者都使用冻结的DFlash草稿器/解码逻辑及各自实测的同栈AR，逐格保留实现标识。混合实现的Mean不能表述为同一目标后端的统一测量。Huginn为332.6M窄版适配，Ouro1.4B为163.6M窄版适配。\n\n'
    text+='† LoopAhead为按论文自行实现；Huginn是迁移。AR沿用原同主机冻结测量，时间漂移未修正。跨方法比较应同时查看绝对速度、输出长度与AR来源。\n\n'
    text+='所有正式格子均沿用原题序/题数（300/500/270/164/378/480）、max512、BF16、Graph OFF。T1同seed一致率仅为描述指标；数值分歧诊断尚未全部完成，不宣称所有输出等价。退出凭证缺失的旧packed测速在CSV逐项标出，未补造退出状态，也未自动重跑。\n'
    text+=f'\n当前完整测量 {len(items)}/72 格；尚非最终完整论文表。\n' if len(items)<72 else '\n72格测量齐全；最终交付仍需训练成本与未完成验证范围的审查。\n'
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=BASE/'combined-results')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    canonical,evidence,legacy_source=read_legacy();variants=[]
    for root in [BASE/'logs/packed-dflash-supplement-v1',BASE/'logs/ouro14-packed-evaluation-v1/supplement']:
        additions,proof=read_packed(root)
        for row in additions:
            k=key(row)
            if k in canonical:variants.append(row)
            else:
                canonical[k]=row;evidence['/'.join(map(str,k))]=proof['/'.join(map(str,k))]
    items=sorted(canonical.values(),key=lambda r:(MODELS.index(r['model']),int(r['temperature']),METHODS.index(r['method']),TASKS.index(r['benchmark'])))
    fields=list(items[0])
    assert all(set(r)==set(fields) for r in items), 'Inconsistent result schema'
    temporary=args.out/'results.csv.tmp'
    with temporary.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(items)
    temporary.replace(args.out/'results.csv')
    (args.out/'table2_rows.md').write_text(table_text(items))
    save(args.out/'evidence.json',dict(legacy_source=legacy_source,collector_sha256=sha(Path(__file__)),cells=evidence))
    save(args.out/'extra-variants.json',variants)
    expected=set(itertools.product(METHODS,MODELS,TASKS,[0,1]))
    save(args.out/'status.json',dict(completed=len(items),expected=72,missing=sorted(expected-set(canonical)),
        additional_variant_count=len(variants),updated_unix=time.time(),
        results_sha256=sha(args.out/'results.csv'),table_sha256=sha(args.out/'table2_rows.md'),
        source_policy='Existing complete legacy measurements take precedence; each added packed cell has its own fresh AR and full profile.',
        numerical_diagnostics='not_all_completed'))
    print({'completed':len(items),'expected':72,'legacy':legacy_source['rows'],'packed_additions':len(items)-legacy_source['rows']})


if __name__=='__main__':main()
