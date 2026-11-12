# Table 1 agentic 补充实验（2026-11-12）

**Ouro-1.4B 补充（2026-11-12）**：新增独立批次
[loopfill-agentic-sql-ouro14-20261112](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-ouro14-20261112/plan.json)，
复用原批次完全相同的100条请求、顺序和历史消息，以Ouro-1.4B自身tokenizer重新编码；
不重新生成teacher、不重新选题、不截断。全部七方法×T=0/1共14组、1,400次请求，
B1/max512、N16/Q4、seed0、单次计时及计时边界沿用下文。
LoopAhead使用该模型已完成的selected LAD及独立准入表；DFlash使用163,597,568参数的
projected drafter，等待5万条teacher生成、六轮正式训练和训练验收完成后绑定检查点哈希。
随后全部14项最短/最长上下文接口验证通过，才开始正式测速。
**2026-11-13调度修订**：用户明确要求所有剩余实验按单个单元并行派发，取消温度组同卡限制。
已完成的T0七组保留；剩余T1七组各自从GPU 0–7领取空闲且UUID锁可用的设备，
有多张空闲卡就同时运行多个方法，正在运行的任务继续完成。原冻结推理源码和输入不变。
实际设备逐单元记录；AR仍为同模型、同温度对照，跨卡差异不再通过同卡绑定控制。
调度及验收实现独立冻结在
[并行调度修订](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-ouro14-20261112/amendments/parallel-cells-20261113/scheduling.json)，
原计划和协议中的GPU5/7绑定作为历史记录保留，由该修订覆盖。
见[实时状态](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-ouro14-20261112/status.json)和
[输入核验](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-ouro14-20261112/checks/input-audit.json)。
完整结果经逐请求审计及真实退出码核验后自动导出独立的
`paper/tables/table1-agentic-sql-ouro-1.4b.tex`与对应CSV，原两模型结果保留。
按用户要求，输出差异诊断已取消；接口验证及产物完整性检查继续。
以下两模型、28组和原GPU绑定描述的是原批次协议。

本轮只补充 Table 1：Huginn-3.5B、Ouro-2.6B 两组模型，AR、WFD、LoopSpec、
Suffix-only、LoopAhead、DFlash、LoopFill-online 全部七个方法，temperature=0/1。
共28个正式单元，每个单元100次相同历史请求、最多512新token、B=1。
GPU仅使用用户本轮明确指定的空闲卡0、1、5、7；不在2、3、4、6排队。

正式数据目录是 [loopfill-agentic-sql-table1-20261112](../results/loopfill-agentic-sql-table1-20261112/)。
轨迹生成状态见 [trace-status.json](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-table1-20261112/trace-status.json)。
100次输入已冻结，正式批次已启动；实时测速状态见
[status.json](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-table1-20261112/status.json)，
[report.md](../paper/data/new-results-20261113/repository/results/loopfill-agentic-sql-table1-20261112/report.md)随完成单元更新。
GPU接口验证不等于正式测速完成。

## 工作负载

原方案使用公开OpenHands轨迹。Huginn最多4096上下文；两模型共同筛选完整请求后，
100次调用中99次来自原轨迹早期、1次来自中期，无法充分覆盖多轮重复。
该输入仅用于接口验证，未产生正式表格数据。两模型×七方法×两温度的28项验证全部通过，
记录位于 [validation](../results/loopfill-agentic-table1-20261112/validation/)。
更早的单Ouro、16K方案已标记SUPERSEDED，不能用于本轮Table 1。

本轮默认推进公开 [Spider开发集](https://yale-lily.github.io/spider) 上的SQL执行—修复工作流，
与SuffixDecoding论文未公开的AgenticSQL数据集明确区分。官方分发ZIP约206MB；
下载来源和SHA256保存在trace-protocol.json，开发集、SQLite文件另有逐文件哈希。

在1034道开发集题目中，以数据库轮转和固定SHA256排序预选100题，覆盖20个数据库。
初始完整schema与问题须同时满足两模型不超过1900个prompt token；本次没有题目因此排除。
不按正确率、输出重复率、suffix命中率或测速结果挑题。标准答案不进入任务输入。

固定Ouro-2.6B、greedy、512 token上限的AR teacher生成查询，并实际执行只读SQLite。
成功执行即结束；格式错误或真实执行错误最多触发两次修复。错误信息来自实际工具，
不伪造错误，不把标准SQL当作反馈。执行成功仅代表SQL可运行，不代表答案正确。
SQLite查询限时2秒，反馈保留实际错误；SQL工具及trace构建时间不计入模型解码指标。

每个调用保存它生成之前的完整消息，用两组模型各自发布的chat template编码。
所有方法使用相同历史文本、相同每模型token输入；不执行被测方法新生成的工具调用。
因此这是固定agent请求回放上的自由生成，不能报告闭环agent速度或Spider准确率。
本次teacher的当前回复、当前执行结果、未来回复从当前回放输入中删除。
历史中已经发生的SQL和错误反馈保留；不预填充suffix cache。

输入冻结时计划从初次请求、第一次修复、第二次修复各取34/33/33次调用。
某阶段数量不足时，缺额按预定规则分配到仍有容量的阶段，不复制或人为制造修复。
最终100次调用为74次初始、13次首次修复、13次二次修复，来自78道题。
原始100题共生成126次调用，87题成功执行、13题达到三次调用上限；无上下文排除。
Huginn输入242–1226 token，Ouro输入238–1289 token；含512输出预算均未超过3072。
详情以inputs/manifest.json为准。
完整历史必须在两个tokenizer下均不超过3548 token，不截断历史。
这套阶段配额并非自然流量分布，必须连同结果一起报告。

## 比较协议

两温度均固定seed=0；T0 greedy，T1完整softmax，无top-p/top-k截断。
BF16、Graph/compile/行填充关闭；使用相同packed完整目标执行器与新测AR分母。
Huginn固定32次循环、Ouro固定4次循环；Huginn位置噪声表固定seed=0。
所有方法沿用已冻结配置与训练检查点，不在agentic测试数据上训练或调参。

LoopFill与Suffix-only使用N16、根Q4、孩子Q(4,4)、浅层Top2；online初值0.7。
LoopAhead复用既有selected LAD及Ouro T1准入表。DFlash复用block16 drafter，
Huginn使用已完成的narrow projected drafter，Ouro使用既有正式检查点。
DFlash训练配置长度3072；超过此长度的请求属于上下文外推，逐请求保留长度供审计。

每个正式单元开始时全局suffix cache为空，只发布本方法已经完成的实际输出。
预热缓存独立；中断恢复仅从本单元已完成日志重建历史，其他方法和teacher输出不注入缓存。
每个方法先验证所选输入中的最短与最长上下文（各最多24 token），全部通过才派发正式单元。
正式计时前另做两条最多32 token预热。任何验证失败停止继续派发并保留日志。
本轮不自动追加参数扫描、缓存消融或并发吞吐实验，以完成Table 1为先。

## 指标与产物

Table 1单元按既有口径导出 latency speedup / loop-normalized γ。
同时保存平均/p50/p95 decode latency、总decode token/s、相对AR的token/s比、
输出长度比、prefill时间、触顶数量、峰值显存和适用方法的候选覆盖率。
时钟在同步prefill之后开始，包含draft、验证、采样和suffix发布；排除模型加载、
预热、磁盘日志与prefill。prefill产生的首token不计入decode token/s。
输出长度可能不同；延迟比必须结合token/s与输出长度比解读。

这里只有B=1 decode token/s，不等于Table 2并发服务吞吐。
仅一个采样种子和一次计时，不据此报告跨种子稳定性或速度置信区间。
候选覆盖率也不是suffix检索非空率，不能直接称为树hit rate。

正式单元完成后自动更新report.md、measurements.csv、phase-measurements.csv和
独立的table1-agentic.tex。缺失单元显示--。28项全部完成后，后台finalization watcher
逐项核对冻结输入/源码哈希、题序、prompt哈希、GPU一致性和2800条日志，复算时间、
token/s与γ；审计通过才导出paper/tables/table1-agentic-sql.tex、
paper/data/agentic-sql-measurements.csv与provenance.json，不覆盖原Table 1数据列。
审计失败保留checks/finalization-status.json，不写成实验完成。
正式指标输出目录会保存输入、源码、依赖扩展、准入表和drafter哈希及逐请求日志。

GPU绑定为Huginn/T0→0、Huginn/T1→1、Ouro/T0→5、Ouro/T1→7，
确保同模型同温度的方法在同一设备比较。调度同时检查占用和Exclusive_Process模式，
使用UUID文件锁；不修改全局分配文件，不终止其他实验。
驱动库修复通过既有native_cuda_env.py只作用于子进程环境。

入口：[SQL轨迹构建](../scripts/loop_agentic_sql.py)、
[单元执行](../scripts/loop_agentic_worker.py)、[冻结/调度/汇总](../scripts/loop_agentic_suite.py)、
[最终核验与论文导出](../scripts/loop_agentic_finalize.py)。

```bash
python scripts/loop_agentic_suite.py summarize --suite results/loopfill-agentic-sql-table1-20261112
```
