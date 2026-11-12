# SpecBench 连续批量端到端吞吐

已完成20组：两个正文模型 × B=1/2/4/8/16 × 两种方法，每组480题。两种方法的题集和prompt token逐项相同，全部测量有真实exit0。**实际请求提交顺序不同：DFlash与原表一样采用shuffle(seed=0)，LoopAhead沿用输入文件顺序。**因此只有10组DFlash满足原表顺序；10组LoopAhead作为异序补充记录，不视为严格对齐的加速结果。

BF16，温度0，max_new_tokens=1024，max_seq=2688，128-token分块prefill，continuous batching，Graph/compile/目标行padding关闭，每个worker绑定4个CPU核、单线程；每格一次测量。吞吐为全部输出token（含prefill产生的首token）除以有限请求流总壁钟，包含prefill、调度及fsync日志写入，排除加载与预热。γ按decode token计算，排除各请求的prefill首token。

表中AR和LoopFill均为原表同机历史测量；DFlash相对AR的列是端到端吞吐比，未做跨日期漂移校正。这20组是额外吞吐测评，与72组max512纯decode延迟矩阵分别报告。

| 模型 | B | 历史AR tok/s | 历史LoopFill tok/s | DFlash tok/s | DFlash / AR | DFlash γ |
|---|---:|---:|---:|---:|---:|---:|
| huginn-3.5b | 1 | 5.89 | 26.17 | 18.36 | 3.12× | 3.11 |
| huginn-3.5b | 2 | 11.11 | 40.73 | 33.18 | 2.99× | 3.17 |
| huginn-3.5b | 4 | 19.00 | 59.50 | 57.47 | 3.03× | 3.15 |
| huginn-3.5b | 8 | 32.99 | 88.18 | 94.08 | 2.85× | 3.12 |
| huginn-3.5b | 16 | 48.34 | 117.52 | 129.18 | 2.67× | 3.12 |
| ouro-2.6b | 1 | 5.53 | 27.12 | 16.17 | 2.93× | 2.97 |
| ouro-2.6b | 2 | 10.65 | 51.17 | 28.85 | 2.71× | 2.96 |
| ouro-2.6b | 4 | 20.75 | 86.22 | 55.83 | 2.69× | 3.01 |
| ouro-2.6b | 8 | 38.72 | 140.94 | 96.72 | 2.50× | 3.00 |
| ouro-2.6b | 16 | 67.48 | 189.56 | 153.48 | 2.27× | 2.97 |

LoopAhead异序补充测量（未作为对齐的AR加速比）：

| 模型 | B | LoopAhead† tok/s | LoopAhead γ |
|---|---:|---:|---:|
| huginn-3.5b | 1 | 23.95 | 5.33 |
| huginn-3.5b | 2 | 35.82 | 5.45 |
| huginn-3.5b | 4 | 52.21 | 5.29 |
| huginn-3.5b | 8 | 76.12 | 5.39 |
| huginn-3.5b | 16 | 100.24 | 5.34 |
| ouro-2.6b | 1 | 22.76 | 4.48 |
| ouro-2.6b | 2 | 43.05 | 4.51 |
| ouro-2.6b | 4 | 74.52 | 4.44 |
| ouro-2.6b | 8 | 124.17 | 4.45 |
| ouro-2.6b | 16 | 186.65 | 4.47 |

† LoopAhead为论文自行实现；Huginn为迁移适配。DFlash使用已完成正式训练的草稿器及官方解码/训练组件，通过我们的packed完整循环目标适配器执行；Huginn为332.6M投影窄版。两种方法均使用冻结检查点，逐格来源见throughput-evidence.json。

限制：请求顺序改变可能影响连续批处理的搭配与耗时，不能把它当成只改了记录格式。CSV保留历史比值的原始算术结果，并显式标记order_matches_historical_reference=false；这些比值不支持严格对齐的LoopAhead加速结论。遵照已有结果不重复跑的要求，当前保留全部原测量，未重跑这10组额外吞吐。另有共享主机、单次测量、无置信区间限制；BF16可能产生不同输出，吞吐和γ不证明逐token数值等价或任务质量相同。Ouro1.4B未包含在这份已完成的额外吞吐矩阵中；其常规延迟矩阵另行完成。本报告不把历史AR吞吐比写成新测同栈AR加速比。

复现汇总：`python experiments/baselines/collect_throughput_results.py`。此命令仅重验与汇总已存测量，不启动GPU任务。
