# OpenHands agent 请求解码实验（2026-11-12）

按用户确认的计划，首先执行 Ouro-2.6B 的 B=1、temperature=0/1 五方法比较。
结果目录为 [loopfill-agentic-20261112](../results/loopfill-agentic-20261112/)，
实时进展以 [status.json](../results/loopfill-agentic-20261112/status.json) 为准。
[report.md](../results/loopfill-agentic-20261112/report.md) 与 measurements.csv 随完成单元更新。
排队和 CPU 验证完成不代表 GPU 实验已完成。

## 输入与适用范围

来源为 SWE-bench Verified 的 OpenHands + Devstral Small 2505 公开轨迹，
[官方提交元数据](https://github.com/SWE-bench/experiments/blob/main/evaluation/verified/20250520_openhands_devstral_small/metadata.yaml)
指向公开 S3 桶。已下载500条轨迹，共92,564,662字节；逐文件URL、ETag和SHA256保存在
[输入清单](../results/loopfill-agentic-20261112/inputs/manifest.json)。

每次调用只使用它之前的完整消息，通过 Ouro 发布的原生 chat template 编码。
目标模型重新生成下一条回复；当前参考回复和未来消息不会作为当前请求输入。
完整历史中已有的助手输出、工具观察保留。不同调用使用预先冻结的历史，不把新生成的
工具调用实际执行后接入下一轮，因此这是固定 agent 请求的解码性能实验，不能报告
SWE-bench 解决率或完整 agent 的端到端速度。

初始希望取早、中、晚各约三分之一。以原始轨迹的助手调用序号划分三个阶段，
在最大15,840个prompt token的预算内，分别有3,295、148、18次合格调用。
晚期全部纳入，缺额在其他阶段均分；最终100次调用为早期42、中期40、晚期18，
来自57个任务、12个仓库。各阶段内按仓库轮转、固定SHA256排名抽样，不按成功率、
输出重复率或实测速度筛选。完整上下文不截断；全部超长排除项保留。
prompt长度为6,367–15,839 token，均值12,485.19。

输入文件SHA256：`78453b1481b188368d0040618181128a844ec08d3a7fb68591f749fe96ea6360`。
该长度筛选和阶段配额意味着样本不代表全部500条轨迹的自然请求分布；阶段结果单列。

## 冻结比较

主实验为 AR、Suffix-only、LoopAhead、DFlash、LoopFill-online，两个温度，共10单元、
1,000次请求。五方法均使用新的同环境计时、相同完整目标权重及 packed 目标执行器，
不复用旧AR分母。温度0为greedy；温度1为完整softmax，无top-p/top-k截断；seed=0。
BF16、最多512新token、EOS/chat-end停止、Graph/compile/行填充关闭、B=1。
输出上限包含prefill产生的首token；decode指标不包含该首token。

LoopFill / Suffix-only 主配置为N16、根Q4、孩子Q(4,4)、浅层Top2；online初值0.7。
LoopAhead复用已完成训练选出的LAD及既有开发集采样准入表，不在agent测试集调参。
DFlash复用已完成训练的block16 drafter，通过新packed adapter执行目标验证。
其训练配置的max_position_embeddings为3,072；本组所有prompt都超出这一长度，属于
上下文外推比较，必须与结果一起说明。没有重新训练或改变位置配置。
这是本次用户授权的新实验，不恢复其他已停止的DFlash旧表队列。

每个温度的各方法绑定同一GPU。运行每个方法前，先对所选输入的最短和最长上下文
做两条、最多24 token的真实GPU smoke验证；失败保留日志并停止继续派发。
正式请求前另做两条、最多32 token预热；suffix预热缓存独立。

主实验从空全局suffix cache开始，仅在请求完成后发布本方法实际生成的输出，
随后请求逐步积累历史。恢复中断时，仅从已完成日志重建同一历史前缀。
参考答案、其他方法输出和预热输出不进入正式缓存。

主实验完成后，自动执行预先固定的补充单元：

- T1：两种suffix方法分别在每次请求重置缓存，隔离跨请求历史收益，共2单元。
- T1：两种suffix方法分别测N64/Q16和N64/Q64，共4单元；全部报告，不事后只选最快点。
- T0/T1：两种suffix方法各对前12次请求做实际查询诊断，共4单元、48次请求。
  诊断计时不进入性能表；区分根与孩子查询，记录非空检索、匹配长度和根候选路径覆盖。

合计20个单元：10个主比较、6个消融、4个诊断；正式和消融1,600次请求，诊断48次。
smoke及预热单列，不计入上述数量。

## 指标与复现

主指标为平均/p50/p95 decode latency、总decode token/s、γ（按完整目标循环深度归一化）、
相对新AR的延迟比和token/s比。另存输出长度比、触顶数量、prefill时间、峰值显存、
阶段分组统计及可适用方法的候选覆盖率。单请求decode token/s不是并发服务吞吐。
检索非空率、目标候选覆盖率与计算复用不可混为一谈；instrumented诊断速度不进入正式表。
仅一个随机种子和一次计时，不能据此给出跨种子稳定性、速度置信区间或任务质量结论。

调度遵循现有GPU白名单（当前2/3/4/6）、独占模式和UUID锁；等待已有
`loopahead-table2-20261112-v2` 控制器退出后再派发，不停止其他作业，不使用已释放GPU。
驱动库修复仅作用于子进程环境，沿用已验证的native_cuda_env包装器。
源码、ArcticInference扩展、输入、准入表和drafter检查点均记录哈希；冻结源码不在运行中修改。

准备入口：[loop_agentic_inputs.py](../scripts/loop_agentic_inputs.py)；
单元执行：[loop_agentic_worker.py](../scripts/loop_agentic_worker.py)；
冻结、排队与汇总：[loop_agentic_suite.py](../scripts/loop_agentic_suite.py)。

```bash
python scripts/loop_agentic_suite.py summarize --suite results/loopfill-agentic-20261112
```

重复启动前检查controller.lock与已记录的控制器PID；单实例锁阻止重复派发。
本批不把后续B4/8/16并发吞吐或Spider执行—修复工作流写成已完成实验。
