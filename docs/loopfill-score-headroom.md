# LoopFill：分数协调的上限诊断（宽范围固定 c + oracle 重排）

日期：2026-11-09。真实模型实验 **36/36组、4,320/4,320次请求完成，审计通过**。结果目录：`results/loopfill-score-headroom-t0-20261109/`。

## 实测结果（2026-11-09，已完成）

**结论：Huginn 有明显的排序改善空间，单个来源权重只能兑现其中一小部分；两个 Ouro 在 N16 下的 oracle γ 改善较小。当前 online 标量控制器未显示稳定优势。**

B=1、N=16、greedy、BF16、Graph-off；三模型 × 六数据集各20题 × 12变体，共36组、4,320次请求，最多512个新token。36组均一次完成，审计通过。γ 为 `T × 解码token数 / loop steps`，不包含prefill产生的第一个token。

| 模型 | γ(c=1) | 最佳固定c | γ(最佳固定c) | 最佳固定c相对c=1 | online相对c=1 | oracle-first相对c=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huginn-3.5b | 7.298 | 0.25 | 7.497 | +2.73% | +0.04% | +19.44% |
| ouro-1.4b | 5.798 | 0.7 | 5.833 | +0.60% | -0.18% | +1.90% |
| ouro-2.6b | 5.917 | 0.7 | 5.937 | +0.34% | -0.87% | +1.68% |

上表使用每组全部120题。最佳c是在本诊断子集上的事后选优，不是独立验证集结论。固定c全范围并非平坦：最佳/最差γ差为8.8%/13.2%/13.9%，三个模型的最差点均为c=0.001。较准确的结论是：**从c=1继续优化单一常数的收益较小，极小权重会退化。**

BF16下不同树形会改变输出，因此另在每模型12个变体输出都完全相同的请求交集上比较：

| 模型 | 共同输出题数 | γ(c=1) | γ(oracle-first) | oracle-first增益 | 最佳固定c增益 | oracle-only rows/token降幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| huginn-3.5b | 57/120 | 7.946 | 10.346 | +30.21% | +5.70% | 46.83% (60.65→32.25) |
| ouro-1.4b | 31/120 | 8.395 | 8.814 | +4.99% | +0.68% | 48.87% (7.86→4.02) |
| ouro-2.6b | 37/120 | 8.606 | 8.864 | +3.01% | +0.00% | 47.89% (7.71→4.02) |

只要求oracle-first与c=1两者输出相同时，Huginn/Ouro-1.4B/Ouro-2.6B分别有84/56/70题，γ增益为35.13%/4.77%/3.59%。这与上表共同输出交集的样本定义不同，不能混用分母。所有同输出结果均为条件子集分析。

### 等待分解与含义

| 模型 | c=1排序等待/总loop steps | c=1提案等待/总loop steps | 正分候选超过N的选择边界 | 同候选池改为c=0.25后选集改变 |
| --- | ---: | ---: | ---: | ---: |
| huginn-3.5b | 40.61% | 55.05% | 61.96% | 26.31% |
| ouro-1.4b | 7.78% | 90.32% | 55.24% | 25.01% |
| ouro-2.6b | 6.10% | 91.73% | 57.53% | 25.65% |

提案等待表示正确节点首次被提出得太晚，即使之后每个tick都计算，也无法在接棒时完成全部深度；排序等待表示它已被提出，但实际计算进度低于这个可达进度。两者占比之和还需加上每题初始根的等待，才等于总loop steps。它们不是错误token比例。

- **Huginn：**oracle-first在全体和同输出口径均有明显收益，值得研究更细的节点排序、准入和淘汰策略。本轮常数扫描和online控制器没有兑现大部分oracle收益。N16下oracle仍会截断超预算的正确路径，因此其残余排序等待不能全部归因于打分错误。
- **Ouro：**c接近1已经接近本次常数扫描最优；oracle-first对γ的改善约几个百分点，优先研究更早的正确路径提案有依据。不过父节点晚准入会让孩子晚提案，proposal_wait不是纯提案因素的因果归因。
- **计算行数：**oracle-only在共同输出子集上大幅减少rows/token，Ouro也有较大幅度。这说明减少无效分支计算仍值得关注；能否转化为墙钟吞吐收益需要无观察器的性能实验。
- **控制器：**online相对相同初值的固定c=0.7，三模型全体γ分别低0.78%/0.78%/1.21%。本轮没有显示online优势，但单次子集诊断不能作为统计显著性或所有负载下的结论。

这些oracle使用参考答案路径信息，且排序会改变后续候选池；它们是乐观诊断，**不是所有可能评分策略的严格全局上界**。结果仅覆盖本次N16/B1/greedy子集，不是实测吞吐加速，也没有评估任务质量。

产物：[全部变体及分数据集结果](../results/loopfill-score-headroom-t0-20261109/report.md)、[分析JSON](../results/loopfill-score-headroom-t0-20261109/analysis.json)、[指标CSV](../results/loopfill-score-headroom-t0-20261109/table.csv)、[两两同输出配对CSV](../results/loopfill-score-headroom-t0-20261109/paired.csv)、[审计](../results/loopfill-score-headroom-t0-20261109/audit.json)、[曲线PDF](../results/loopfill-score-headroom-t0-20261109/score-headroom.pdf)。

审计逐题验证请求顺序/唯一ID、输入和源码SHA256、结果日志SHA256、节点预算、token/loop/row计数、等待分解恒等式、零reach violations及oracle输出一致标记。可用 `PYTHONPATH=src /tmp/loopfill-venv/bin/python scripts/loop_score_headroom_audit.py --out results/loopfill-score-headroom-t0-20261109` 重新审计并出图。

## 要回答的问题

消融显示 online 控制器与固定 c 差别不大。在继续改进两种来源的分数协调之前，先测量：在当前 N 和提案规则下，固定分数权重有多敏感，使用正确路径信息的排序能实现多少 γ 改善。

1. **宽范围固定 c**：c ∈ {0.001, 0.05, 0.25, 0.5, 0.7, 1, 2, 4, 1000}。两端分别近似"模型候选优先"和"suffix 候选优先"（`min(1,cq)` 在 c 很大时饱和为 1）。如果 γ 在整个范围内几乎不变，说明该范围内固定标量的敏感性较低；这不能排除按上下文变化的权重带来收益。
2. **oracle-first**：候选生成、分数和 N 都与 fixed-c1 完全相同，只在选择时把位于参考输出路径上的候选排在最前，剩余名额按 c=1 的分数顺序填。参考输出取同模型、同 N 的 fixed-c1 运行结果。它提供使用正确路径信息的乐观排序诊断，不新增提案规则。排序会改变后续候选生成的时机和候选池，因此不能视为所有可能评分策略的严格全局上界。
3. **oracle-only**：只保留参考路径上的候选，用来测量理想路径筛选可减少多少 rows/token；偏离参考后回退普通排序。它也不是所有候选生成轨迹下的严格全局下界。
4. **online-c0.7**：沿用论文母配置的在线控制器，作为对照。

实际输出一旦偏离参考（BF16 树形差异或采样），之后就没有参考路径上的候选，选择会退回 c=1 的顺序。这类请求会单独标出，报告同时给出 γ_common，即只统计所有变体输出完全一致的请求。

## 等待分解

每次正确节点接棒（成为新根）时，按节点拆成两部分：

```text
proposal_wait = T − min(T, 距首次提案的 tick 数)      # 提案太晚或从未提案
ranking_wait  = min(T, 距首次提案的 tick 数) − 接棒时的进度  # 早已提案，但没分到位置或被预算淘汰
calls = T + Σ(proposal_wait + ranking_wait)
```

ranking_wait 是逐节点的上界：父节点准入晚，也会让子节点的提案变晚，这部分被计入子节点的 proposal_wait。oracle-first 测量的是实际能实现的收益。另外还记录：

- 正分候选数大于 N 的选择边界比例（positive-contested）。只有在这些边界上，对两种来源分数的任何单调重加权才可能改变选择结果。
- 用同一候选池换成 c=0.25/2 重新排序时，被选集合发生变化的比例。这一项来自 `DraftTrace`。

## 运行方式

与表 1 的 B=1 协议一致：N16、Q4、top2、孩子 Q(4,4)、BF16、Graph-off，每个数据集冷启动 suffix 历史，只发布已完成的输出，greedy 或 temperature=1 的目标采样。默认三个模型、六个数据集各分层抽 20 题（按类别轮询，组内按 sha256(source_id) 排序），最多 512 个新 token。每个 job 是一个"模型 × N × 变体"组合，默认 3×1×12=36 个。oracle job 会等对应的 fixed-c1 job 完成后再启动。

```bash
# 资产目录约定与 latency suite 相同：<model>.json（含 path）和 ArcticInference/
PYTHONPATH=src python scripts/loop_score_headroom_suite.py prepare \
  --out results/loopfill-score-headroom-t0 \
  --assets /tmp/loopfill-assets --python /tmp/loopfill-venv/bin/python
python results/loopfill-score-headroom-t0/source/scripts/loop_score_headroom_suite.py run \
  --out results/loopfill-score-headroom-t0 --gpus 0,1,2,3,4,5,6,7
PYTHONPATH=src python scripts/loop_score_headroom_suite.py analyze \
  --out results/loopfill-score-headroom-t0
```

常用选项：`--temperature 1`；`--budgets 4,8,16`，因为 N 越小预算越紧，打分越可能起作用；`--per-dataset 40`；`--models ouro-2.6b`。`run` 可以中断后重启：worker 按请求写入 journal 续跑，重放已完成输出以重建 suffix 历史；失败的 job 自动重试一次，加 `--retry-failed` 会清零失败计数。单个 job 可以先用 `loop_score_headroom.py --suite … --job … --smoke` 做 2 题×16 token 的检查。

产物：`analysis.json`、`table.csv`、`report.md`（包括 headline、全部变体和各数据集 γ），以及逐请求的 `runs/<job>/requests.jsonl`，其中含 stats、trace 计数、等待分解和逐接棒记录。

## 读法（运行前约定）

| 观察 | 含义 |
| --- | --- |
| oracle-first 相对 c=1 提升很小，且 fixed-c 全范围 γ 差很小 | 本诊断未检出明显的排序收益。可优先研究提案，但不能据此证明所有评分策略都没有收益。 |
| oracle-first 提升明显，但 fixed-c 范围很窄 | 排序有空间，但单个标量做不到，需要依赖上下文的校准（冲突/一致、深度、匹配长度）。 |
| fixed-c 范围明显，online ≈ 最佳固定 c | 本轮未显示online优势，固定常数是值得保留的基线。 |
| online 明显低于最佳固定 c | 提示检查控制器反馈、收敛和应用样本之间的差异，尚不能仅凭该观察确定原因。 |

## 限制

- 带观察器的诊断：只看 loop-step 计数，不作为吞吐。
- 只用分层子集，suffix 历史比全量运行短。
- 单次运行，没有题序或重复。
- BF16 下各变体的输出可能不同。
- oracle 只重排解码器自己提出的候选，不衡量新增提案规则的收益；排序本身会影响后续提案时机。
- 各变体维护自己的 suffix 历史；当前请求输出相同不保证此前历史相同。同输出子集属于条件分析，不能无条件外推到全部请求。
- proposal_wait 是逐节点的记账分类；父节点排序延迟引起的孩子晚提案也会计入其中，不能将该占比直接解释为纯提案的因果贡献。

实现：[worker](../scripts/loop_score_headroom.py)、[队列与分析](../scripts/loop_score_headroom_suite.py)、[测试](../tests/test_loop_score_headroom.py)。oracle 通过在诊断进程内临时替换 `loop_budget_decode.select_candidates` 实现，没有修改生产解码器。

## 本轮执行记录（2026-11-09）

- 36组 × 120题，共4,320次请求；9个固定c、online-c0.7、oracle-first、oracle-only；最多512个新token。
- 运行前修复整数题号抽样，并保留原始题号、以数据集前缀保证配对标识唯一；修复oracle-only偏离参考后的回退。
- 诊断及关联测试共134项通过；修改文件通过ruff检查。
- 源码和输入已冻结并记录SHA256；逐请求日志、进度和队列状态持久保存。
- Huginn oracle-first首次短测因与GPU测试争用独占卡而未能加载模型，日志保留；正式队列使用逐GPU锁和占用检查。
- 三模型 × fixed-c1/oracle-first/oracle-only 共9个真实模型短测均已完成；Huginn oracle-first使用逐GPU锁补跑通过，先前失败日志保留。
- 最终36/36组、4,320/4,320次请求完成，正式任务失败/重试数为0；完整审计通过，曲线已导出PDF/PNG。
