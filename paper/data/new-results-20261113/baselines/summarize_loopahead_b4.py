"""Report the paper-setting comparison without inventing an unavailable gamma."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
records={}
for method in ['ar','loopahead']:
    folder=BASE/f'preflight/loopahead-B4-ouro-2.6b-humaneval-{method}-v1'
    complete=json.loads((folder/'complete.json').read_text())
    meta=json.loads((folder/'metadata.json').read_text())
    assert meta['paper_check'] and not meta['reportable'] and complete['requests']==164
    assert meta['max_new_tokens']==1024 and meta['temperature']==0 and meta['prompt_format']=='completion'
    assert hashlib.sha256((folder/'requests.jsonl').read_bytes()).hexdigest()==complete['requests_sha256']
    records[method]=dict(meta=meta,summary=json.loads((folder/'summary.json').read_text()),
        rows=[json.loads(l) for l in (folder/'requests.jsonl').open()])
a,d=records['ar'],records['loopahead']
assert a['meta']['input_sha256']==d['meta']['input_sha256']
matches=0;first=[]
for ar,la in zip(a['rows'],d['rows']):
    assert ar['source_id']==la['source_id'] and ar['prompt_sha256']==la['prompt_sha256']
    if ar['tokens']==la['tokens']:matches+=1
    else:
        pos=next((i for i,(x,y) in enumerate(zip(ar['tokens'],la['tokens'])) if x!=y),min(len(ar['tokens']),len(la['tokens'])))
        first.append(dict(source_id=ar['source_id'],position=pos,ar_length=len(ar['tokens']),method_length=len(la['tokens'])))
eta=d['summary']['tok_s']/a['summary']['tok_s']
timing_incidents=[]
incident_path=BASE/'logs/weather-contention-recovery-v1.json'
if incident_path.exists():
    incident=json.loads(incident_path.read_text())
    worker=next(p for p in incident['processes'] if any('resume_forecast_shard.py' in s for s in p['argv']))
    start=worker['process_started_unix']
    end=incident['contention_interval_end_upper_bound_unix']
    affected={}
    for method,record in records.items():
        previous=0.
        ids=[]
        for row in record['rows']:
            finished=row['finished_unix']
            if previous < end and finished > start:
                ids.append(row['source_id'])
            previous=finished
        if ids:affected[method]=ids
    timing_incidents.append(dict(source=str(incident_path),sha256=hashlib.sha256(incident_path.read_bytes()).hexdigest(),
        worker_interval_unix=[start,end],potentially_affected_requests=affected,
        interval_policy='Conservative request windows from previous request publication to current publication; no timing values removed or replaced',
        limitation='Weather worker used GPU0 while B4 used GPU2, but its192-CPU affinity overlapped B4 CPUs8-11. Timing impact cannot be isolated from these logs.'))
report=dict(model='ouro-2.6b',benchmark='HumanEval164_original_completion',temperature=0,max_new_tokens=1024,
    ar=a['summary'],method=d['summary'],eta_throughput=eta,eta_latency=a['summary']['decode_s']/d['summary']['decode_s'],
    gamma=d['summary']['gamma'],paper_eta_throughput=4.72,eta_relative_difference=eta/4.72-1,
    paper_gamma=None,gamma_15_percent_check='not_verifiable_no_per_cell_gamma_published',
    exact_sequences=matches,requests=164,first_divergences=first,
    prefill_eos_requests={k:sum(len(r['tokens'])==1 and r['stop_reason']=='eos' for r in v['rows']) for k,v in records.items()},
    output_cap_requests={k:sum(len(r['tokens'])==1024 for r in v['rows']) for k,v in records.items()},
    interpretation='paper-based implementation with declared adaptations; not an exact numerical replication of author code',
    remaining_differences=['Author training code/LAD weights/evaluation launcher not public',
        'Training BBH portions replaced by protected non-test reasoning data; all local training/architecture defaults recorded',
        'Current8-GPU cluster identifies as L20X; author A40 backend and graph behavior differ',
        'Author-specific additional completion delimiters and exact target checkpoint revisions unavailable',
        'Native and speculative BF16 shapes can differ in logits; observed B2 divergences were checked in FP32'],
    formal_matrix_gate='comparison_and_limitations_recorded; gamma threshold is unavailable, not passed')
report['timing_incidents']=timing_incidents
report['uncontended_timing_comparison']=not any(i['potentially_affected_requests'] for i in timing_incidents)
if not report['uncontended_timing_comparison']:
    report['remaining_differences'].append('B4 AR overlapped a weather worker with shared CPU affinity; ratios are retained as observed diagnostics and cannot establish an uncontended paper-speedup replication. Formal matrix results are separate.')
(BASE/'preflight/loopahead-B4-summary.json').write_text(json.dumps(report,indent=2)+'\n')
text=f'''# LoopAhead B4论文设置对照

Ouro-2.6B、原始HumanEval164题、completion、T0、max1024；新跑同栈AR和已训练LAD，decode-only。正式LoopFill矩阵仍是chat/max512，与本对照分开保存。

| 指标 | 本次实际值 | LoopAhead论文 |
|---|---:|---:|
| η吞吐比 | {eta:.6f} | 4.72 |
| η时间比 | {report['eta_latency']:.6f} | 未单列 |
| γ | {report['gamma']:.6f} | 未公布此单元 |
| AR tok/s | {a['summary']['tok_s']:.6f} | 未公布此单元 |
| 方法 tok/s | {d['summary']['tok_s']:.6f} | 未公布此单元 |
| BF16整序列一致 | {matches}/164 | 原文称一致 |

η相对原文差异为{report['eta_relative_difference']:+.2%}。γ缺少可比参考，不能判定“差异小于15%”，更不能将其他任务混合的图中倒数当作本题集参考。

计时存在并发干扰记录：{not report['uncontended_timing_comparison']}。若为True，上表η仅保留原始观测，不用于宣称无干扰的论文速度复现；具体时段、可能受影响题目及原始事件哈希见同名JSON中的timing_incidents。未删除慢题或替换计时。

首token即EOS：AR {report['prefill_eos_requests']['ar']}/164，LoopAhead {report['prefill_eos_requests']['loopahead']}/164；达到1024上限：AR {report['output_cap_requests']['ar']}/164，LoopAhead {report['output_cap_requests']['loopahead']}/164。前两题另用原生HF完整4轮CPU前向复查，原始completion prompt下EOS均为top1；此诊断不能代替GPU数值等价证明，但支持早停来自目标的prompt响应。未为提升结果而关闭EOS或改写正式prompt。

以下差异限制精确数值复现：

'''+''.join('- '+v+'\n' for v in report['remaining_differences'])+'''
该结果作为本地论文方法复现的检查记录，不能标为作者官方实现或精确复现。差异与限制已记录后继续已授权的统一LoopFill矩阵；不根据测试表现重新训练或选择检查点。
'''
(BASE/'preflight/loopahead-B4-summary.md').write_text(text)
print(json.dumps({k:v for k,v in report.items() if k not in ['first_divergences','ar','method']}))
