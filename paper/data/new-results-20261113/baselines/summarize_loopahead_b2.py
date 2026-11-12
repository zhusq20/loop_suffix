"""Audit the completed20-request random-LAD correctness checks before training."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
OUT=BASE/'preflight'
report=dict(model='ouro-2.6b',stage='B2_random_LAD',cases={})
for dtype in ['bfloat16','float32']:
    runs={}
    for method in ['ar','loopahead']:
        path=OUT/f'loopahead-B2-ouro-2.6b-gsm8k-{method}-{dtype}-n20-v1'
        meta=json.loads((path/'metadata.json').read_text())
        complete=json.loads((path/'complete.json').read_text())
        assert complete['requests']==meta['count']==20 and not meta['reportable']
        assert hashlib.sha256((path/'requests.jsonl').read_bytes()).hexdigest()==complete['requests_sha256']
        rows=[json.loads(l) for l in (path/'requests.jsonl').open()]
        assert len(rows)==20
        runs[method]=rows
    exact=0;divergences=[]
    for ar,la in zip(runs['ar'],runs['loopahead']):
        assert ar['source_id']==la['source_id'] and ar['prompt_sha256']==la['prompt_sha256']
        if ar['tokens']==la['tokens']:
            exact+=1
        else:
            pos=next((i for i,(a,b) in enumerate(zip(ar['tokens'],la['tokens'])) if a!=b),min(len(ar['tokens']),len(la['tokens'])))
            divergences.append(dict(source_id=ar['source_id'],position=pos))
    report['cases'][dtype]=dict(exact_sequences=exact,requests=20,first_divergences=divergences)
common=json.loads((OUT/'loopahead-B2-common-prefix-fp32-v1/report.json').read_text())
assert common['status']=='complete' and all(r['same_top1'] for r in common['results'])
assert {(r['source_id'],r['position']) for r in common['results']}=={(r['source_id'],r['position']) for r in report['cases']['bfloat16']['first_divergences']}
report['common_prefix_fp32']=dict(cases=len(common['results']),same_top1=sum(r['same_top1'] for r in common['results']),
    maximum_logit_difference=max(r['max_abs_logit_difference'] for r in common['results']))
report['training_gate']=report['cases']['float32']['exact_sequences']==20
report['limitations']=['BF16 complete sequences do differ; do not describe this as100% BF16 token equality',
    'FP32 full20 and same-prefix comparisons support a numerical batch-shape explanation for these observed BF16 differences',
    'random LAD correctness checks are not trained method quality or formal timing results']
(OUT/'loopahead-B2-summary.json').write_text(json.dumps(report,indent=2)+'\n')
text=f'''# LoopAhead B2实际检查

Ouro-2.6B，冻结GSM8K前20题，greedy，max512，随机LAD。

- BF16整序列一致：{report['cases']['bfloat16']['exact_sequences']}/20。
- FP32整序列一致：{report['cases']['float32']['exact_sequences']}/20。
- BF16首次分歧的FP32共同前缀复查：{report['common_prefix_fp32']['same_top1']}/{report['common_prefix_fp32']['cases']} top1一致，最大logit差{report['common_prefix_fp32']['maximum_logit_difference']:.8g}。

共同前缀探针保留512输出预算，临时强制此前已经观察到的共同token，并在比较位置立即停止；该诊断不计入任何benchmark。串行对照保留单个活跃节点，每个位置跑满4轮。FP32树形状可与原BF16不同。证据支持已观察分歧来自不同batch形状下的数值差异，不能宣称BF16逐token100%无损。

首次分歧、top3 logits和概率差分别保存在loopahead-B2-bfloat16-first-divergences.json及loopahead-B2-common-prefix-fp32-v1/report.json。训练门槛{'通过' if report['training_gate'] else '未通过'}；训练后的质量与论文设置对照仍须另做。
'''
(OUT/'loopahead-B2-summary.md').write_text(text)
print(json.dumps(report))
assert report['training_gate'],'FP32 correctness gate failed; inspect before training'
