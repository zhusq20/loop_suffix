# LoopAhead B4论文设置对照

Ouro-2.6B、原始HumanEval164题、completion、T0、max1024；新跑同栈AR和已训练LAD，decode-only。正式LoopFill矩阵仍是chat/max512，与本对照分开保存。

| 指标 | 本次实际值 | LoopAhead论文 |
|---|---:|---:|
| η吞吐比 | 5.203022 | 4.72 |
| η时间比 | 5.059748 | 未单列 |
| γ | 6.206698 | 未公布此单元 |
| AR tok/s | 6.302910 | 未公布此单元 |
| 方法 tok/s | 32.794181 | 未公布此单元 |
| BF16整序列一致 | 97/164 | 原文称一致 |

η相对原文差异为+10.23%。γ缺少可比参考，不能判定“差异小于15%”，更不能将其他任务混合的图中倒数当作本题集参考。

计时存在并发干扰记录：True。若为True，上表η仅保留原始观测，不用于宣称无干扰的论文速度复现；具体时段、可能受影响题目及原始事件哈希见同名JSON中的timing_incidents。未删除慢题或替换计时。

首token即EOS：AR 23/164，LoopAhead 23/164；达到1024上限：AR 95/164，LoopAhead 99/164。前两题另用原生HF完整4轮CPU前向复查，原始completion prompt下EOS均为top1；此诊断不能代替GPU数值等价证明，但支持早停来自目标的prompt响应。未为提升结果而关闭EOS或改写正式prompt。

以下差异限制精确数值复现：

- Author training code/LAD weights/evaluation launcher not public
- Training BBH portions replaced by protected non-test reasoning data; all local training/architecture defaults recorded
- Current8-GPU cluster identifies as L20X; author A40 backend and graph behavior differ
- Author-specific additional completion delimiters and exact target checkpoint revisions unavailable
- Native and speculative BF16 shapes can differ in logits; observed B2 divergences were checked in FP32
- B4 AR overlapped a weather worker with shared CPU affinity; ratios are retained as observed diagnostics and cannot establish an uncontended paper-speedup replication. Formal matrix results are separate.

该结果作为本地论文方法复现的检查记录，不能标为作者官方实现或精确复现。差异与限制已记录后继续已授权的统一LoopFill矩阵；不根据测试表现重新训练或选择检查点。
