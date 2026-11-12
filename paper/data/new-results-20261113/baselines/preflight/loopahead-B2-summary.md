# LoopAhead B2实际检查

Ouro-2.6B，冻结GSM8K前20题，greedy，max512，随机LAD。

- BF16整序列一致：16/20。
- FP32整序列一致：20/20。
- BF16首次分歧的FP32共同前缀复查：4/4 top1一致，最大logit差4.7683716e-05。

共同前缀探针保留512输出预算，临时强制此前已经观察到的共同token，并在比较位置立即停止；该诊断不计入任何benchmark。串行对照保留单个活跃节点，每个位置跑满4轮。FP32树形状可与原BF16不同。证据支持已观察分歧来自不同batch形状下的数值差异，不能宣称BF16逐token100%无损。

首次分歧、top3 logits和概率差分别保存在loopahead-B2-bfloat16-first-divergences.json及loopahead-B2-common-prefix-fp32-v1/report.json。训练门槛通过；训练后的质量与论文设置对照仍须另做。
