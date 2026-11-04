# 纯后缀基线性能修复（2026-11-05）

**当前对照设置（按用户最新要求）**：关闭 CUDA Graph 和新增的执行行填充。调用工作区的 `scripts/loop_batch_bench.py` 时，在原有参数末尾追加 `@configs/loopfill-eager.args`，显式覆盖两个开关，并保留原 Graph-on 实验的 2×batch 请求预热量。其余参数保持原样；packed attention 自身的布局不变。六方法、B=1/2/4/8/16的完整480题重跑已启动，见[独立实验状态](../results/loopfill-eager-b1-b16-huginn-20261105/status.json)。下文均为此前 Graph-on 实验记录。

已修复执行后端造成的明显性能损失。Q=4、N=16、后缀候选、完成后共享缓存、验证与接受规则、逐轮调度均保持原样。新结果仍是 Huginn 上本仓库的纯后缀实现，不称为官方 SuffixDecoding 的性能复现。

## 已完成测量

单位为输出 token/s，包含本轮请求的 prefill、解码、CPU 调度和后缀操作。模型加载、预热、图捕获在计时外。BF16，seed=0，continuous batching，128-token chunked prefill，packed attention，最多24张 CUDA Graph。每配置单次计时，没有显著性结论。

| B | 覆盖 | AR（执行填充后端） | 纯后缀：完整验证形状预热 | 吞吐比 | 纯后缀：仅请求预热 |
|---|---|---:|---:|---:|---:|
| 1 | 60题，每题最多128新token | 18.29 | 21.77 | 1.19× | 15.42 |
| 8 | 完整480题，每题最多1024新token | 77.54 | 105.37 | 1.36× | 61.72 |
| 16 | 完整480题，每题最多1024新token | 103.86 | 123.85 | 1.19× | **131.94** |

B1 的60题是完整题号列表每隔8题抽一题，用于工程验证，不是独立留出评测，不能冒充480题结果。B2/B4尚未补测修复后的全量曲线。

完整形状预热消除了 B1/B8 对预热请求是否恰好出现所需形状的依赖。B16在24张图预算下，优先保留全部验证图挤占了部分 embedding/readout 图，完整预热比仅请求预热慢。**后续融合对照应保留 B16 已测更强的131.94 token/s纯后缀点（相对本轮AR为1.27×）**，不能只挑较慢的123.85作为基线。这是已测工程配置的比较，不代表全局最优。

旧版全量纯后缀在B8/B16分别为56.17/82.23 token/s。修复后B8完整预热为105.37，B16仅请求预热为131.94；但执行填充改变了BF16数值形状，输出不同，不能把这些比值称为等输出无损加速。

AR复用本轮 `source-v2` 的执行填充后端测量，没有重复跑已经完成的控制。其数值计算路径与 `source-v3` 相同，完整验证形状预热是后续加入的预热策略。完整预热结果来自冻结的 `source-v3`。记录保留来源GPU、源码及输入哈希，不宣称同卡同源码的严格配对重复实验。

## 修复内容

1. 按执行行数分桶，并稳定小型草稿树的 attention 宽度，使相近大小的树能复用 CUDA Graph。填充行不成为草稿节点，不参与接受判定。
2. 给填充行分配独立的 KV 写入缓冲区，不写真实请求的 KV，attention 也不读该缓冲区。测试中用 NaN 填充这个缓冲区检查隔离。
3. 对 AR 和纯后缀，根据事先声明的 B 与单次草稿上限枚举验证形状。只做合成状态预热，不读测试输出、不向后缀缓存写入合成内容、不在计时期间捕获新图。Q=4时，B1/B8/B16分别需要4/16/20张验证图。
4. 分开记录逻辑计算行数、实际执行行数和填充行数，避免把填充算力隐藏在草稿预算中。

主实现见 [loop_optimized_decode.py](../src/llm_memory_editability/loop_optimized_decode.py)。解码状态机、候选选择和请求调度文件与任务开始时的快照逐字节相同。

最初试验过将连续循环合并执行。该方案在B8的AR上退化，已从最终工作区移除；试验源码与结果保留在 `source-v1`、`source-v2`，不将其作为本次推荐修复。

## 正确性与证据边界

- 最终优化及相关解码回归：152项通过，6项因测试默认外部模型/WFD路径未挂载跳过。
- 另用本地真实 Huginn 权重做 FP32 检查：8题、最多16新token；AR与纯后缀均8/8逐token匹配串行AR。包含最终固定形状预热路径。
- 完整形状预热相对仅请求预热：B1的60/60、B8/B16的480/480题输出完全相同，calls、rows、后缀查询次数也逐题相同。这部分计时差异不来自输出变化。
- 执行填充相对旧版全量纯后缀：B8为373/480、B16为369/480题输出完全相同。相对本轮填充AR，纯后缀为50/60、369/480、367/480题相同。BF16输出分歧仍存在，任务质量尚未评测，不认定这些差异无害。
- 完整预热的纯后缀/AR输出总量分别为：B1 3929/3926，B8 52304/51554，B16 51897/51590。报告吞吐与耗时，并保留逐题输出。
- Graph命中率是调用数占比，不是GPU时间占比。完整预热的B1/B8/B16总调用命中率分别为98.3%/94.4%/75.9%；B16较低主要反映有限图预算下的非验证操作回退。

## 使用与产物

在原有 packed benchmark 命令中加入：

```bash
--buffer-policy per-arm --cuda-graph --max-cuda-graphs 24 \
--pad-decode-rows --verifier-shape-warmup complete
```

B16的更快已测方案对应 `--verifier-shape-warmup observed`。AR和待比较方法应使用明确记录的执行后端；后续融合结果需按修复后基线重新比较。

默认不加 `--pad-decode-rows` 时保持原有形状策略，便于历史结果复现。完整形状预热目前面向不超过65个物理树槽的AR/纯后缀；需要的验证图数超过显式图预算时会报错，不暗中增加预算。

- [协议及源码快照说明](../results/loopfill-suffix-fix-20261105/protocol.json)
- [机器可读比较与逐题一致性审计](../results/loopfill-suffix-fix-20261105/comparison.json)
- [最终回归记录](../results/loopfill-suffix-fix-20261105/final-validation.log)
- [真实模型FP32检查](../results/loopfill-suffix-fix-20261105/fp32-final/complete.json)
- [B8完整预热结果](../results/loopfill-suffix-fix-20261105/primed-full-b8/budget-suffix-b8-seed0-summary.json)
- [B16仅请求预热结果](../results/loopfill-suffix-fix-20261105/full-b16-budget-suffix/budget-suffix-b16-seed0-summary.json)
- [B16完整预热结果](../results/loopfill-suffix-fix-20261105/primed-full-b16/budget-suffix-b16-seed0-summary.json)

工作区最终版在冻结的 `source-v3` 之后补了稳定的同频图淘汰顺序、大型共享树缓冲区的预热适用范围判断，以及 `observed/complete` 显式切换；没有再改模型算子、草稿策略或数值精度。完整计时保留原冻结源码身份。
