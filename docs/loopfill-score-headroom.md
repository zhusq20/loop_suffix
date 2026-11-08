# LoopFill：分数协调的上限诊断（宽范围固定 c + oracle 重排）

日期：2026-11-08。脚本已写好并通过 CPU 测试，**尚未在真实模型上运行**，下文没有任何实测结论。

## 要回答的问题

消融显示 online 控制器与固定 c 差别不大。在继续改进两种来源的分数协调之前，先测量：在当前 N 下，**任何**更好的打分最多还能带来多少 γ。

1. **宽范围固定 c**：c ∈ {0.001, 0.05, 0.25, 0.5, 0.7, 1, 2, 4, 1000}。两端分别近似"模型候选优先"和"suffix 候选优先"（`min(1,cq)` 在 c 很大时饱和为 1）。如果 γ 在整个范围内几乎不变，说明单个标量无论怎么学都改变不了结果。
2. **oracle-first**：候选生成、分数和 N 都与 fixed-c1 完全相同，只在选择时把位于参考输出路径上的候选排在最前，剩余名额按 c=1 的分数顺序填。参考输出取同模型、同 N 的 fixed-c1 运行结果。它给出"只改排序"能达到的 γ 上限，不新增任何提案。
3. **oracle-only**：只保留参考路径上的候选，用来给出相同提案下 rows/token 的下限。
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
| oracle-first 相对 c=1 提升很小，且 fixed-c 全范围 γ 差很小 | 当前 N 下分数协调没有空间。online 可以去掉，或改写为"对来源权重不敏感"；精力转向提案（看 proposal_wait）。 |
| oracle-first 提升明显，但 fixed-c 范围很窄 | 排序有空间，但单个标量做不到，需要依赖上下文的校准（冲突/一致、深度、匹配长度）。 |
| fixed-c 范围明显，online ≈ 最佳固定 c | 一个常数就够，online 不必要。 |
| online 明显低于最佳固定 c | 控制器收敛到了错误的位置，与反馈样本和应用样本不一致的问题一致。 |

## 限制

- 带观察器的诊断：只看 loop-step 计数，不作为吞吐。
- 只用分层子集，suffix 历史比全量运行短。
- 单次运行，没有题序或重复。
- BF16 下各变体的输出可能不同。
- oracle 只重排解码器自己提出的候选，不衡量提案改进带来的收益。

实现：[worker](../scripts/loop_score_headroom.py)、[队列与分析](../scripts/loop_score_headroom_suite.py)、[测试](../tests/test_loop_score_headroom.py)。oracle 通过在诊断进程内临时替换 `loop_budget_decode.select_candidates` 实现，没有修改生产解码器。
