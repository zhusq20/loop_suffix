# Table 1：temperature=1 完整延迟实验（2026-11-08）

本轮是独立的随机采样实验，不覆盖temperature=0结果。状态与汇总入口：

- [状态](../results/loopfill-latency512-t1-20261108/status.json)
- [完整指标CSV](../results/loopfill-latency512-t1-20261108/measurements.csv)
- [协议](../results/loopfill-latency512-t1-20261108/protocol.json)
- [W&B](https://wandb.ai/zsqzz/loopfill-latency512-t1)

## 冻结范围

三个模型：Huginn-3.5B、Ouro-1.4B、Ouro-2.6B。五种方法：AR、同引擎WFD、同引擎LoopSpec、Suffix-only、LoopFill-online。六个数据集：GSM8K 300、MATH-500 500、BBH 270、HumanEval+ 164、MBPP+ 378、SpecBench 480。90单元，31,380次正式请求。完全复用原表1的冻结输入及题序，已核对六数据集SHA256。

B=1、BF16、最多512新token；temperature=1、top_p=1、不做top-k截断；EOS/chat-end停止。Graph、compile、行填充关闭。N16/Q4/model top2/child Q(4,4)，online c0=.7、eta=.1、范围[.25,2]。每单元计时一次，固定sampling seed=0；这不是多随机种子重复实验。

## 采样语义

目标输出由全深度logits的完整softmax分布按inverse-CDF采样产生，包括prefill产生的首token。浅层top-k及后缀草稿保持确定性；命中已存在候选则复用计算，否则正常继续。不将目标概率限制到树内，不重复抽样直到命中，不使用树排序分数作为拒绝采样概率。本轮WFD/LoopSpec是共享此目标采样器的同引擎调度实现，不是官方采样实现性能。

请求种子由SHA256(base seed/model/dataset/source ID)派生，不含方法。每个绝对token位置固定一个独立伪随机uniform；不同候选前缀共享该位置的uniform。无关分支、节点回收、重排和warmup不消耗正式输出的随机流。各方法共享请求级随机数，warmup另用独立命名空间。随机数表初始化位于请求设置阶段，实际目标softmax/CDF及抽样在相应prefill或decode计时内。

在相同目标条件分布和理想数值条件下，实际路径的每次选择服从目标分布。实现使用FP32概率/CDF，BF16前向；不同计算形状仍可能产生数值差异，所以固定随机数不代表真实大模型各方法必须逐token完全相同，也不等同于已完成任务质量评估。

## 指标

- `mean_decode_s`：各题decode墙钟时间的算术平均；prefill后同步开始，包含后缀建树/检索/更新/发布和采样。
- `latency_speedup`：同模型/数据集AR总decode时间除以方法总decode时间。
- `gamma`：T × decode token总数 / 实际loop steps总数，沿用原表1的循环归一化接受长度；AR=1。
- `mean_acceptance_rate`：每题“全深度采样token命中已有草稿候选数 / decode token数”的算术平均；零decode token请求不进入此平均。
- `pooled_acceptance_rate`：全部命中次数 / 全部decode token数。首token不进入分子或分母。AR无草稿，命中率为0。

此命中率是候选覆盖率，不是“接受草稿节点数/全部创建节点数”，也不是经典proposal/rejection算法的接受概率。预算方法包括已存在的逻辑候选，不保证候选已经完成全部计算；所以该比率不能直接换算为计算节省，需结合gamma及实测延迟。另报告token/s、输出长度比和达到512上限的数量。

## 保存、验证与运行

逐题JSONL append/flush/fsync，冻结源码和输入哈希；各单元独立后缀历史，只包含当前单元已完成输出。八张GPU共享互斥队列，空闲补位。每模型/方法先做两题、16-token真实权重短验证；预热最多32token，不进入正式计时或历史。

CPU回归118 passed、36 skipped；另外采样CPU/GPU及队列检查29 passed。新增检查覆盖分布频率、温度影响、off-path/行重排随机流隔离，以及两种小模型架构五方法相对AR的输出一致性。日志位于本批`checks/`。这些不替代真实权重短验证或完整质量实验。

本轮在本地测量完成后先写`complete.json`，将W&B上传状态单独记入`upload-status.json`；上传失败保留pending，避免把已完成推理错误标成未完成。W&B初始化或逐题上传期间的其他异常仍保留错误并停止该单元，恢复时按冻结逐题前缀继续。

```bash
/tmp/loopfill-venv/bin/python \
  results/loopfill-latency512-t1-20261108/source/scripts/loop_latency_suite.py run \
  --out results/loopfill-latency512-t1-20261108
```

恢复前必须核查现有控制器和工作进程，不重复启动。冻结副本不可在运行期间修改。只汇总可将`run`换成`summarize`。原temperature=0全量结果继续保存在`results/loopfill-latency512-20261107/`。

启动核验：15/15真实模型/方法短验证完成（每组2题×最多16token），八卡已进入正式单元。Huginn和Ouro-1.4B部分短输出与同seed AR不同，Ouro-2.6B本次短验证全部一致；逐组结果见`checks/real-checkpoint-smoke.json`。这些差异尚未逐项归因，不作为分布保持的实证证明。
