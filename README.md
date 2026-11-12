# LoopFill (loop_suffix)

**论文新数据归档（2026-11-12）**：[集中索引](paper/data/new-results-20261112/README.md)提供 DFlash/LoopAhead 新基线、SQL补充、完整机制耗时分析、五个已完成草稿器的训练与成本，以及进行中任务的状态快照。包含184个已结束请求流、48,744条逐请求记录与SHA256清单，可用标准Python离线核验；各后端、AR分母和缺失退出码状态保留。

**八卡任务调度与失败单元重跑（2026-11-12）**：剩余 DFlash 重测单元进入 GPU 0–7 共享队列，
优先补跑 LoopAhead/Ouro-2.6B/SpecBench/T=1 的启动锁冲突单元。
已有 GPU 工作进程继续完成；待运行长任务优先填入空闲卡，同卡比较组保留绑定。
见[共享队列实时状态](results/gpu-balanced-queue-20261112-v1/status.json)、
[重跑状态](results/gpu-balanced-queue-20261112-v1/retry-status.json)与
[调度修订](docs/dflash-table12-remeasurement.md)。

**Agentic Table 1 补充（2026-11-12）**：Huginn-3.5B、Ouro-2.6B × 全部七方法 × temperature=0/1，
共28项、2,800次SQL执行—修复请求回放；仅使用本轮指定的GPU 0/1/5/7。
见[实验协议](docs/loopfill-agentic.md)、[实时状态](results/loopfill-agentic-sql-table1-20261112/status.json)和
[结果报告](results/loopfill-agentic-sql-table1-20261112/report.md)。全部完成且逐请求审计通过后自动导出论文表格。

**DFlash 表一/表二重测（2026-11-12）**：新增共享 KV 的多请求目标验证和 batched drafter，
按原冻结输入准备 Huginn-3.5B、Ouro-2.6B 的34个单元，表二覆盖 B=1/2/4/8/16。
真实权重验证通过后才进入对应正式单元；进度、协议和恢复入口见
[重测说明](docs/dflash-table12-remeasurement.md)。

**DFlash / LoopAhead 独立入口（2026-11-12）**：实现位于仓库根目录的 [`baselines/`](baselines/)，根目录运行 `python run_baseline.py --help`。支持训练、特征提取、推理与回归测试，运行时无需 SZ checkout；权重和数据通过显式路径传入。Huginn DFlash 默认332.6M drafter。[安装与命令](docs/standalone-baselines.md)。

**DFlash 共用目标执行器（2026-11-12）**：新增 B=1 的 packed target 适配器，复用已训练 drafter，将目标特征提取、块验证和 KV 回退接入 LoopFill 后端。独立验证与同 GPU 小样本比较见[实现和运行说明](docs/dflash-packed-backend.md)；诊断结果不混入已有正式表。

**论文数据归档**：[图表数据索引](paper/data/README.md)列出主表、四组消融、来源构成、分数协调、树采样和机制图所需的CSV/JSON及压缩逐请求统计。支持仅用已提交数据重画；耗时补采保存明确标记的已完成部分快照。

**三项机制分析（2026-11-09，已开始）**：36组消融/17,280条请求及240条提前计算诊断已重新核验，负载—预算曲线、进度CDF与等待分解图已生成。阶段耗时小样本队列已启动：两模型、B1/B16、四方法，每流固定六任务分层24题，正常计时与诊断分开；本轮不补B4/B8。见[执行范围、已有结果与计时口径](docs/loopfill-mechanisms.md)、[补采实时状态](results/loopfill-mechanisms-20261109/timing/status.json)。

**最终接受token来源统计（2026-11-09，已完成）**：复用表2三模型×五档B的15组、7,200次请求，无新推理。B16下后缀来源占Huginn/Ouro-1.4B/Ouro-2.6B最终decode输出的38.19%/41.41%/42.23%，浅层模型合计56.56%/56.35%/56.67%；其余为未命中候选的全深度输出，prefill首token单列。逐请求计数与哈希审计通过。见[口径、结果与分任务图](docs/loopfill-token-sources.md)。长上下文、历史积累、高熵场景分析已按用户要求取消。

**分数协调上限诊断（2026-11-09）已完成**：三模型、N16/B1/greedy、36组/4,320次请求，审计通过。oracle-first相对c=1的全体γ改善：Huginn **+19.4%**、Ouro-1.4B **+1.9%**、Ouro-2.6B **+1.7%**；全部变体共同输出子集为+30.2%/+5.0%/+3.0%。最佳固定c的全体改善仅+2.73%/+0.60%/+0.34%，online未显示稳定优势。此为排序诊断，不是吞吐或严格全局上界。见[结论与限制](docs/loopfill-score-headroom.md)、[完整结果](results/loopfill-score-headroom-t0-20261109/report.md)、[曲线](results/loopfill-score-headroom-t0-20261109/score-headroom.pdf)、[审计](results/loopfill-score-headroom-t0-20261109/audit.json)。

**候选树概率采样实验（2026-11-09，已完成）**：按用户要求仅种子0、单次计时，两模型四臂，8/8个B16完整480题流、3,840次请求完成且审计通过。树采样将平均草稿数降至约8个、rows/token减少21%—36%，但相对同评分Top-N的吞吐下降 **11.4%—14.6%**，本轮不支持替换Top-N，未触发B1补测。见[设计与实测结论](docs/loopfill-tree-sampling-plan.md)、[完整报告](results/loopfill-tree-sampling-b16-seed0-20261109/report.md)、[指标CSV](results/loopfill-tree-sampling-b16-seed0-20261109/measurements.csv)、[图PDF](results/loopfill-tree-sampling-b16-seed0-20261109/tree-sampling.pdf)。

**四组消融与Q2补充已完成（2026-11-09）**：Huginn-3.5B与Ouro-2.6B共40/40个单元完成、19,200条逐题记录通过审计：B16四组消融以吞吐为主，B1补充N=8/16/32/64的decode latency。child suffix budget增加Q=2的仅首次/两次读出，新增4组B16、1,920次请求，复用原36组；固定Top-2、N16、根部Q4、greedy、480题、最多1024 token和表2冻结推理版本。见[完成状态](results/loopfill-four-ablation-q2-20261109/status.json)、[合并指标CSV](results/loopfill-four-ablation-q2-20261109/measurements.csv)、[逐请求数值归档](results/loopfill-four-ablation-q2-20261109/paper-data.jsonl.gz)、[新协议](results/loopfill-four-ablation-q2-20261109/protocol.json)。后续控制器诊断/调参/配对评测计划仍取消。见[执行记录](docs/loopfill-paper-experiments.md)及[原36组协议](results/loopfill-four-ablation-20261109/protocol.json)。

**Preliminary 命中率实验（2026-11-09）已完成**：Ouro-2.6B、480题、80,933个有效位置。loop 1/2/3 单层一致率为88.83%/95.63%/98.20%，累计覆盖率为88.83%/96.72%/98.95%。Suffix → +Shallow → +Requery 平均连续正确草稿长度为1.343 → 1.732 → 2.276；续查配对增量+0.544 token（95% CI [0.502, 0.587]），不代表实测速度收益。见[计划与结果](docs/loopfill-preliminary-plan.md)、[汇总](results/loopfill-preliminary-20261109-v2/full/summary.json)、[图（单层与额外命中率）](paper/figures/loopfill-preliminary-v2/preliminary-hit-rates.pdf)。

**temperature=1 完整表1实验已完成（2026-11-09归档）**：三个模型×六个数据集×五种方法，90/90组、31,380/31,380次请求；B=1、最多512 token。LoopFill-online在六数据集上的延迟加速比几何均值：Huginn-3.5B 2.37×、Ouro-1.4B 3.22×、Ouro-2.6B 3.49×。原greedy结果保留。见[协议与验证](docs/loopfill-latency512-t1.md)、[结果表](results/loopfill-latency512-t1-20261108/table1.md)、[候选命中率](results/loopfill-latency512-t1-20261108/candidate-hit-rates.md)、[汇总CSV](results/loopfill-latency512-t1-20261108/measurements.csv)。

**表 1、表 2 最终结果（2026-11-08）**：表1全部90组、31,380条请求完成；表2为74组成功、1组实测OOM（Huginn/WFD/B16）。表1曾因W&B上传超时标记失败的单元已补传，云端artifact下载哈希与本地一致，原失败记录保留。实验均为单次计时，BF16输出差异与任务质量需单独评估。

- [表1：512-token、B=1 decode延迟加速比与γ](results/loopfill-latency512-20261107/table1.md) · [完整指标CSV](results/loopfill-latency512-20261107/measurements.csv)
- [表2：1024-token、B=1/2/4/8/16端到端吞吐与加速比](results/loopfill-table2-20261107-v2/table2-final.md) · [完整指标CSV](results/loopfill-table2-20261107-v2/measurements.csv)
- LaTeX：[表1](paper/tables/table1-latency512.tex)、[表2](paper/tables/table2-throughput.tex)。使用 `python scripts/loop_final_tables.py` 从汇总数据重新生成。
- 协议与归档：[表1](docs/loopfill-latency512.md)、[表2](docs/loopfill-table2.md)、[归档范围](docs/loopfill-table12-archive.md)。

**论文实验规划**：[当前精简计划](docs/loopfill-paper-experiments.md)统一主指标（throughput 与按 loop steps 归一化的平均接受长度 γ）、一行四面板消融雷达图（节点预算、浅层读出、suffix接续预算、权重控制）及机制分析；每模型/B共14个唯一消融配置，与旧实验计划冲突时以此文为准。

LoopFill 是面向循环（looped / recurrent-depth）语言模型的免训练投机解码。它把两种草稿来源放进同一棵草稿树：

- **后缀树匹配**（同 SuffixDecoding）：从提示、已确认输出和历史请求中检索续写；
- **早期循环读出**（同 LoopSpec / WFD）：位置在跑完全部 T 轮之前，用第 d 轮的读出先提出下一个词。

两种来源可以互相接续。v4 按**每请求未接受草稿节点预算 N**控制树规模，按路径分数排序并支持满额替换。已实现 `budget-raw`（c=1）、`budget-online`（逐请求在线 suffix 权重）及 `budget-suffix`，无需离线校准集或校准表；旧阈值与四表实现保留用于历史复现。分数控制提前计算，输出由已验证前缀上的全深度预测决定。模型为 Ouro-2.6B（T=4，读出1/2）与 Huginn-3.5B（T=32，读出4/8）。新版全尺寸性能尚未评测，既有 Ouro BF16 输出分歧仍待定位。实现、统一 N=64 的拟议480题对比和命令见 [在线控制器方案](docs/loopfill-online-suffix-control.md#11-已实现接口与下一轮比较方案)。

本仓库从 [zhusq20/llm-memory-editability](https://github.com/zhusq20/llm-memory-editability) 中抽出 LoopFill 相关的文档、代码和小体积实验产物（2026-11-02 快照）。为保持报告里的链接和 import 不变，目录结构与 Python 包名 `llm_memory_editability` 沿用原仓库。

**2026-11-05 当前对照重跑**：关闭 CUDA Graph 和执行行填充，保留 2×batch 请求预热量和其他实验参数。Huginn 六组主方法（AR、WFD、LoopSpec、纯后缀、LoopFill-raw、LoopFill-online），B=1/2/4/8/16，共30组完整480题，在8张GPU上通过共享队列并行执行。见[协议](results/loopfill-eager-b1-b16-huginn-20261105/protocol.json)、[实时状态](results/loopfill-eager-b1-b16-huginn-20261105/status.json)、[结果表](results/loopfill-eager-b1-b16-huginn-20261105/measurements.csv)。配置为 [`@configs/loopfill-eager.args`](configs/loopfill-eager.args)，本轮不复用历史计时或OOM结果。

**2026-11-05 纯后缀基线修复（此前 Graph-on 实验）**：执行行填充与固定验证形状预热已实现。Huginn完整480题，B8纯后缀105.37 token/s，对本轮AR为1.36×；B16仅请求预热的更快已测点为131.94 token/s，对AR为1.27×。B1另做60题短验证，尚未补齐修复后的全量曲线。BF16输出分歧与任务质量仍需单列。见[修复、配置及验证报告](docs/loopfill-suffix-baseline-fix.md)。

**2026-11-04 更新**：主实验使用 continuous batching + 128-token chunked prefill + 跨请求 packed attention + decode CUDA Graph。两模型、六臂在完整480条数据上测试 B=1/2/4/8/16/32/64/128/256，共108组，8张GPU并行；B表示最大在场请求数。实际 OOM 单独记录，其他组继续。GPU统一独占、固定频率和CPU亲和性，同模型同B六臂绑定同一卡。见[优化后端协议](results/loopfill-optimized480-20261104/protocol.json)、[状态](results/loopfill-optimized480-20261104/status.json)、[结果表](results/loopfill-optimized480-20261104/measurements.csv)、[验证记录](results/loopfill-optimized480-20261104/validation.json)。图仅覆盖预热中常见的decode形状，未覆盖形状走eager并计数；预热和捕获在计时外。旧固定batch及未优化的continuous结果分别保留在 `results/loopfill-batch480-20261104/`、`results/loopfill-continuous480-20261104/`，不混入新结果。

## 历史四表结果（详见 [docs/loopfill-results.md](docs/loopfill-results.md)）

单请求引擎 E2E，sb1024 chat，bf16，每模型 60 题：

| 模型 | C(0.01) 相对 AR | C(0.01) 相对同引擎 WFD |
| --- | ---: | ---: |
| Ouro-2.6B | 5.23× [4.49, 6.18] | 1.95× [1.76, 2.20] |
| Huginn-3.5B | 6.94× [5.26, 8.79] | 2.34× [2.16, 2.56] |

**当前进度（2026-11-03）**：两模型、B=1/4、raw/simple/four 的八卡消融已完成320流（128调参+192评测），见 [消融协议与产物](results/loop-score-ablation-20261103/protocol.json)。ID24/CanItEdit dev12 为开发规模证据；正式 Spec-Bench 全量、任务质量和满载曲线仍待完成。

**当前计划 v4（2026-11-04）**：[投稿总计划](docs/loopfill-icml-experiments.md)保留完整实验范围；下一轮按 [在线控制器方案第11节](docs/loopfill-online-suffix-control.md#11-已实现接口与下一轮比较方案)比较 AR/WFD/同引擎 LoopSpec 双提案/预算纯后缀/预算 raw/在线版本。事前统一 N=64，不再要求外部开发集选 N；参数不根据480题结果选优。N 不计已确认前缀与根，计入算完但未接受的草稿。预算入口已接入 [batch 脚本](scripts/loop_batch_bench.py)，旧机器协议与已有结果不改名。Ouro 输出分歧仍需针对性诊断。

## 目录

| 路径 | 内容 |
| --- | --- |
| `docs/design/` | 设计讨论页（HTML，下载后用浏览器打开）：`loop-suffix-tree-spec-20261030.html` 最初的方案原理图；`loop-spec-schedule-comparison-20261101.html` 调度对比、按猜对概率放候选的方法、v3 C 一步的完整例子与消融图 |
| `paper/` | 方法与预备知识（`method-loopfill.tex`、`preliminaries-loopfill.tex`）、设计稿 `suffix-loop-design.tex/.bib`、图源（excalidraw / build 脚本 / 导出图） |
| `docs/` | 实验规划（`loopfill-plan.md`）与结果摘要（`loopfill-results.md`） |
| `src/llm_memory_editability/` | `loop_suffix_spec.py` 后缀树+循环调度的离线回放；`loop_prob_tree.py` 概率草稿树（校准、打分、剪枝，v3 C）；`loop_tree_decode.py` Ouro/Huginn 的真实树解码引擎；`loop_prob_decode.py` 概率树解码器 |
| `scripts/` | 读出采集（`loop_suffix_spec_readouts.py`）、CPU 回放与消融（`loop_suffix_spec_simulate.py`、`loop_prob_tree_replay.py`、`loop_prob_tree_ablation.py`）、调度比较、墙钟测速（`loop_suffix_spec_wallclock.py`、`loop_spec_sibling.py`）、多请求微基准与分析出图 |
| `tests/` | 单元测试与对 WFD 参考实现的逐词一致性测试 |
| `results/` | 各批次 `protocol.md` / `report.md`、校准表、汇总 JSON、图表与启动脚本；`raw/` 子目录是从服务器持久目录复制的逐题记录（逐题计时 JSONL、校准表、轨迹、微基准、运行日志） |

各批次：

- `loop-suffix-spec-20261030`：四种解码器的离线调用数、MT-Bench 二轮、loop 调度对后缀树的贡献（§10）、v3 C 消融（§11）
- `loop-spec-prob-tree-20261031`：概率草稿树设计 v1–v3、校准、S4 单请求墙钟、多请求微基准
- `loop-suffix-spec-wallclock-20261031`、`loop-wfd-same-engine-20261031`：首批墙钟与同引擎 WFD 对照
- `loop-spec-reuse-20261030`、`loop-spec-sibling-20261030`：草稿复用与兄弟分支的上限测量

## 进度（截至 2026-11-03）

| 日期 | 内容 | 位置 |
| --- | --- | --- |
| 10-30 | 方案原理：后缀树与浅层读出共用一棵草稿树 | `docs/design/loop-suffix-tree-spec-20261030.html` |
| 10-30 | 前期探索（基于 WFD 调度）：拒绝后复用统计、浅层第 2 候选分支。只有协议和逐题记录，没有写正式报告；主要统计数字引在 sibling 协议开头 | `results/loop-spec-reuse-20261030`、`results/loop-spec-sibling-20261030` |
| 10-30–31 | 离线回放（块调用数）：四种解码器、raw / chat-256 / sb1024、MT-Bench 二轮 | `results/loop-suffix-spec-20261030` |
| 10-31 | 首批墙钟与同引擎 WFD 对照：both-path 相对 WFD 快 1.34–1.40× | `results/loop-suffix-spec-wallclock-20261031`、`results/loop-wfd-same-engine-20261031` |
| 10-31 | 按猜对概率放候选：设计 v1→v3、校准、回放、多请求微基准；loop 给后缀树带来什么（报告 §10） | `results/loop-spec-prob-tree-20261031`、`results/loop-suffix-spec-20261030/report.md` §10 |
| 11-01 | v3 C 单请求墙钟（S4）：相对 both-path 快 1.26–1.73×；v3 C 四栏消融（报告 §11）；单请求 E2E 汇总（见上表） | `results/loop-spec-prob-tree-20261031/report.md` §9、`docs/loopfill-results.md` |
| 11-02 | 论文方法节、预备知识节与图 | `paper/` |
| 11-03 | 最小固定 batch 入口、小模型正确性测试、均衡预检输入与正式吞吐执行计划；后续完成320流全尺寸评分消融，正式全量矩阵待执行 | `docs/loopfill-batch.md`、`results/loop-batch-20261103` |

**尚未完成**（当前投稿计划与顺序见 [docs/loopfill-icml-experiments.md](docs/loopfill-icml-experiments.md)）：

- 本轮：节点预算与在线控制器代码已实现，待全尺寸验证及容量审计；先按统一 N=64 的六臂方案做两个模型 B=1/4、单题序单次计时的24流全量比较，再扩展负载曲线。Ouro 分歧诊断单列，区分共同 B 对照与各臂最高吞吐。
- 投稿必做：官方 LoopSpec/WFD 直接对照；MATH-500 数学质量；CanItEdit 代码编辑速度与功能正确性。
- 投稿必做：GPU 核心消融（含关闭新孩子立即后缀查询）、节点预算/历史阈值策略与评分消融、历史/二轮、容量/上下文和性能分解；统计与复现归档。
- 不在本轮范围：连续到达/SLO 服务、新 drafter 训练、完整 SQL / SWE-bench agent 系统。
- 数据缺口：Huginn 摘要题 253、288、295 缺 sb1024 读出（显存不足）；预留题 249、273、281 缺 S4 计时。

`results/*/RUNNING.txt` 是运行当时的交接记录，状态以 `report.md` 和上表为准。

## 运行

```bash
pip install -e ".[dev]"            # 纯 CPU 测试只需 numpy + pytest（部分测试需 torch）
PYTHONPATH=src python -m pytest tests
```

需要模型推理的部分另需：

- `torch`、`transformers`，以及 Ouro-2.6B / Huginn-3.5B 权重；
- [ArcticInference](https://github.com/snowflakedb/ArcticInference) 的后缀树（固定版本 `aca5d9a`）；
- WFD 参考实现（[wavefront-decoding](https://github.com/summerbro-hhj/wavefront-decoding)），默认挂载于 `/wfd`，可用 `LOOP_WFD_ROOT` 或 `--wfd-root` 指定。

**未纳入本仓库**（都在原服务器 `lm-memory-persistent/workspace/llm-memory-editability/<批次>/` 下）：各深度的模型读出 `runs/`（约 150MB，CPU 回放的输入，需用 `scripts/loop_suffix_spec_readouts.py` 在 GPU 上重新生成）、模型权重 `hf/`、容器 Python 环境 `env/`、v1/v2 全量扫参 `sweep/`（59MB；报告已含汇总）、主机监控流水 `monitor*.jsonl`、W&B 离线缓存（已同步到 W&B 项目 `zsqzz/llm-memory-editability`）。`results/` 中的绝对路径只记录原运行环境。
