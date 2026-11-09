# LoopFill：耗时、提前计算与负载—预算分析

2026-11-09。执行范围来自[论文实验规划 §5.1–5.3](loopfill-paper-experiments.md)。已有结果核验与出图已完成；阶段耗时8/8条真实模型预检通过，GPU 0/1分别执行Huginn/Ouro测量队列。完成数以[实时状态](../results/loopfill-mechanisms-20261109/timing/status.json)为准，跟踪见[W&B](https://wandb.ai/zsqzz/llm-memory-editability/runs/6lcz65kl)。

## 已有证据：核验完成

- 四组消融36组、17,280条请求：重新读取原始逐请求日志，核验输入、推理源码、配置、请求完整性与计数，并重新计算指标，与原CSV逐项核对。
- 提前计算：复用Huginn-3.5B与Ouro-2.6B的online-c0.7诊断，各120条请求，共240条。逐请求验证 `calls = initial_wait + Σ(T−r)`、`T−r = proposal_wait + ranking_wait`，终止token排除在根交接分母外。
- [审计](../results/loopfill-mechanisms-20261109/existing/audit.json)、[完整报告](../results/loopfill-mechanisms-20261109/existing/report.md)、[预算CSV](../results/loopfill-mechanisms-20261109/existing/load-budget.csv)、[提前计算CSV](../results/loopfill-mechanisms-20261109/existing/readiness.csv)。

| 模型 | 非终止根交接数 | 平均进度r/T | 立即就绪比例 | 每次交接剩余loop steps | 正确路径重建节点数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Huginn-3.5B | 19,614 | 0.8683 | 41.18% | 4.2139 | 979 |
| Ouro-2.6B | 32,258 | 0.8326 | 38.02% | 0.6696 | 146 |

提前计算样本来自原score-headroom协议：B1/N16、greedy、最多512 token，GSM8K、MATH500、BBH、HumanEval+、MBPP+、SpecBench各20题。**这六个数据集不是六类SpecBench任务，也不是完整480题实验。** 按数据集拆分的统计保留在CSV；不与1024-token吞吐结果拼接。现有日志未保存逐交接origin和完整生命周期事件，因此不声称恢复了来源切换的进度交叉统计。当前进度、等待与重建论点已有足够字段，不新增trace推理。

预算端点已显示负载取舍：Huginn在B1增加N可以降低延迟，而B16的实测吞吐在N16后下降；Ouro在B1的N32相对N16降低decode ms/token，B16增至N32/64则损失吞吐。N32/64的B1均值差及Ouro/B16的N8/16差较小，单次计时不判断显著性。不把逐点最优包络当成自适应算法。本轮无需补B4/B8。

图：[负载与预算 PDF](../results/loopfill-mechanisms-20261109/existing/load-budget.pdf)、[进度CDF与等待分解 PDF](../results/loopfill-mechanisms-20261109/existing/readiness.pdf)、[36组消融计算效率 PDF](../results/loopfill-mechanisms-20261109/existing/ablation-efficiency.pdf)。效率图使用B16的28组核心配置，CSV同时保留B1补充配置。

## 新补采：阶段耗时

固定协议见[protocol.json](../results/loopfill-mechanisms-20261109/timing/protocol.json)。

- 模型：Huginn-3.5B、Ouro-2.6B；B=1/16。
- 方法：同引擎AR、LoopSpec、Suffix-only、LoopFill-online，共16个配置。
- 每配置24题，SpecBench六任务各4题。按 `SHA256(mechanism-stage-seed0/question_id)` 选择，两个模型使用同一组题号，未使用输出或运行耗时选题；随后按order-seed=0排列。
- 每配置一次正常计时、一次独立阶段诊断，共32条测量流、768次请求。另有两模型×四方法的8条短预检流，B2、六任务各1题、最多16 token，预检不进入正式小样本汇总。
- 主补采最多1024新token；BF16、greedy、continuous/packed、Graph-off、无执行行填充，N16/Q4/top2、孩子Q(4,4)、online初值0.7/η0.1。推理源码与表2逐字节一致，外部包装器只在诊断流安装计时钩子。
- 每流冷启动完成请求历史；预热为`min(2B,24)`请求、最多32 token，预热历史丢弃。同一模型所有方法和两个B固定在同一物理GPU，独占锁与CPU亲和性沿用现有约定。
- 顺序预先固定：预检后B16、B1；每个B内AR、LoopSpec、Suffix-only、LoopFill；每配置先正常再诊断。记录实际执行顺序、源码/输入哈希、逐请求输出和W&B跟踪。

正常计时采用含prefill、decode、调度、suffix工作与本地结果落盘的流墙钟。该24题有限队列只用于本次机制补采，完整480题主性能结果仍按原协议保留。

阶段诊断分成两种时钟，**二者不能相加**：

1. 主机互斥墙钟区间：模型提交与等待、prefill、suffix查询/更新、树选择/分配/剪枝、KV提交、元数据上传及等待、其余调度。嵌套区间扣除子区间后统计；堆叠图只堆叠这些互斥区间。它是含等待的主机墙钟归属，不是CPU利用率。
2. CUDA events记录packed forward、prefill、KV提交的流时间跨度。记录后统一同步读取，不逐阶段新增同步；时间跨度可能包含主机发射间隙，不能写成纯kernel活跃时间。设备时间单独画图，不与主机区间堆叠。

浅层与全深度读出使用融合调用，统一报告`combined_readout`，包含coda/LM head与全深度验证；不按行数分摊出虚假的浅层耗时。recurrent包含attention内的KV读写，外部KV提交单列。诊断的实际输出、γ、rows/token和相对正常流墙钟变化均保留；检查输出一致数和完成历史发布顺序，不能假定两次轨迹完全一致。

## 运行与检查

```bash
# 不运行GPU：复核已有证据并出图
PYTHONPATH=src /tmp/loopfill-venv/bin/python scripts/loop_mechanism_analyze.py \
  --out results/loopfill-mechanisms-20261109/existing

# 查看已启动队列；不要重复prepare或另启动同一队列
cat results/loopfill-mechanisms-20261109/timing/status.json

# 重新汇总已完成的小样本流
PYTHONPATH=src /tmp/loopfill-venv/bin/python scripts/loop_mechanism_suite.py analyze \
  --out results/loopfill-mechanisms-20261109/timing
```

相关测试26项通过，新增脚本和测试通过ruff。入口：[队列](../scripts/loop_mechanism_suite.py)、[阶段计时](../scripts/loop_stage_timing.py)、[已有证据分析](../scripts/loop_mechanism_analyze.py)、[计数与口径测试](../tests/test_loop_mechanisms.py)。图表、CSV、分布和压缩逐请求数值已按用户要求入库，见[论文数据索引](../paper/data/README.md)；耗时部分使用[已完成流快照](../results/loopfill-mechanisms-20261109/timing-snapshot/snapshot.json)，完整运行日志与重复源码快照保留本地。

跟踪在独立[sidecar](../scripts/loop_mechanism_track.py)中运行，读取既有实验凭据约定，不进入测量进程。原lane因未设置凭据环境变量而产生的初始化错误保留；sidecar已接通并从首批预检开始补记。冻结推理源码和运行中的队列未因此重启或修改。
