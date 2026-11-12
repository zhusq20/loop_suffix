# 2026-11-12 论文新数据快照

采集时间：北京时间 2026-11-12 10:25–10:27。归档包含 2,269 个证据文件、184 个已结束请求流、48,744 条逐请求记录，约 43.7 MiB。请求流包括方法、AR 对照、独立 profiling 和预检，不能将总数解释为独立题目数。

## 写论文时从这里读取

| 用途 | 数据入口 | 完成范围与口径 |
| --- | --- | --- |
| 表1：LoopAhead 与原生 DFlash | [results.csv](baselines/results.csv)、[结果说明](baselines/table2_rows.md)、[输出一致性](baselines/formal-agreement.json) | 36 行：LoopAhead 两模型 × 六任务 × temperature=0/1 共24行；原生/HF-target DFlash 12行。每行保留原 AR 分母、后端、延迟与吞吐比；`eta_latency` 用于延迟表，`eta_throughput` 用于吞吐比 |
| 表1/表2：packed DFlash | [measurements.csv](repository/results/dflash-table12-packed-20261112-v2/measurements.csv)、[报告](repository/results/dflash-table12-packed-20261112-v2/report.md)、[状态](repository/results/dflash-table12-packed-20261112-v2/status.json) | 34 单元：19 complete（含表2全部10组）、9复用原生后端、6结果完整但退出码未知。原汇总只给 complete 单元正式数值；其余状态及逐请求证据保留 |
| 表2：LoopAhead 吞吐 | [measurements.csv](repository/results/loopahead-table2-20261112-v2/measurements.csv)、[完整审计](baselines/preflight/user-resume-eight-gpu-v1/loopahead-throughput-audit.json) | 两模型 × B=1/2/4/8/16，10/10组、4,800请求；相对历史 AR 的比值有明确标记 |
| SQL agent 请求回放补充 | [measurements.csv](repository/results/loopfill-agentic-sql-table1-20261112/measurements.csv)、[阶段数据](repository/results/loopfill-agentic-sql-table1-20261112/phase-measurements.csv)、[报告](repository/results/loopfill-agentic-sql-table1-20261112/report.md)、[审计](repository/results/loopfill-agentic-sql-table1-20261112/checks/final-audit.json) | 28/28组、2,800请求；固定历史 SQL 请求的解码性能，不是闭环 agent 成功率。原 [LaTeX 表](../../tables/table1-agentic-sql.tex) 保留 |
| 机制耗时分解 | [measurements.csv](repository/results/loopfill-mechanisms-20261109/timing/measurements.csv)、[stages.csv](repository/results/loopfill-mechanisms-20261109/timing/stages.csv)、[pairs.csv](repository/results/loopfill-mechanisms-20261109/timing/pairs.csv)、[图PDF](repository/results/loopfill-mechanisms-20261109/timing/stage-timing.pdf)、[报告](repository/results/loopfill-mechanisms-20261109/timing/report.md) | 32/32请求流、768记录；16正常计时 + 16诊断流，每流24题。取代旧 timing-snapshot 的未完成快照；不替代完整480题性能表 |
| 草稿器训练、曲线与成本 | [成本表](training-costs/current-cost-summary.md)、[成本JSON](training-costs/current-cost-summary.json)、[逐项账本](training-costs/current-cost-summary-ledger.json)、[训练记录目录](baselines/training/) | 5个正式训练已完成，包括 Ouro-1.4B LoopAhead；保留逐步 loss、开发集指标、选择的检查点哈希及训练审计。Ouro-1.4B DFlash 尚未训练完成 |
| 输出差异与子集诊断 | [已完成差异诊断](baselines/preflight/formal-greedy-margins/loopahead-huginn-3.5b-gsm8k-v1/report.json)、[子集敏感性](baselines/preflight/subset-sensitivity-gsm8k-v1/report.md)、[诊断队列状态](baselines/logs/baseline-diagnostics-resume-v1/status.json) | 仅已有证据；27组诊断队列尚未全部完成 |
| DFlash 配套补采 | [队列快照](baselines/logs/packed-dflash-supplement-v1/status.json)、[已完成阶段](baselines/logs/packed-dflash-supplement-v1/runs/) | 快照收入3个已完成 AR 阶段、944请求；AR和profile均完成才构成整组。不能把部分阶段视为完整配对结果 |
| Ouro-1.4B 后续实验 | [DFlash进度](repository/results/dflash-ouro-1.4b-training-20261112-v3/status.json)、[teacher进度](repository/results/dflash-ouro-1.4b-training-20261112-v3/teachers/shard0/progress.json)、[LoopAhead预检](baselines/preflight/) | 只保存采集时状态及两组已完成20题预检；未完成正式评测不填入论文主表 |

## 数据完整性与使用限制

[manifest.json](manifest.json)逐文件记录原始路径、原始SHA256、归档SHA256、大小、所属分析和读取时间；[run-audits.json](run-audits.json)记录请求数量、输出量和完成证据的核验结果。[validation.json](validation.json)记录125行主要表格指标、冻结输入身份及链接核验结果。原始绝对路径仅供追溯，读取归档无需访问原机器。相关第三方源代码的许可保留在 [licenses](licenses/)。

`*.jsonl.gz` 是原始 JSONL 的无损 gzip，保留全部 token IDs、逐请求统计及计时。解压后 SHA256 与原文件一致。源代码、输入、协议及审计文件保持原字节；文件内的时间、运行状态和本地路径都是历史证据。不同文件有各自的采集时间，运行中的队列不宣称原子快照。

packed DFlash 的6项 `complete_artifact_exit_unavailable` 保留原缺失退出码状态；不能把完整产物核验解释为退出码0。9项 `reused_existing_measurement` 来自原生/HF target，不能标作 packed 后端新测。原生 DFlash 的新测同栈 AR 和 packed DFlash 的历史 AR 分母分别保留，不能互换。部分原报告中的“未启动训练”等措辞描述该协议范围，以本页的后续任务快照判断当前状态。

成本报告只累计已结束的 GPU 占用记录；DFlash Ouro-1.4B 的 `0/50000` 是已完成完整分片的计费口径，teacher 正在生成的逐请求进度另见上表。五个训练完成不代表全部模型的推理评测完成。

模型权重、优化器状态、训练特征缓存和未结束请求流不在此论文数据包中。既有主表、消融、token来源、树采样和分数协调数据仍从[总索引](../README.md)读取。

## 离线核验

在仓库根目录执行，仅需 Python 标准库：

```bash
python scripts/loop_paper_snapshot.py verify --out paper/data/new-results-20261112
python paper/data/new-results-20261112/validate_metrics.py
```

读取逐请求记录，例如：

```python
import gzip, json
path = 'paper/data/new-results-20261112/repository/results/loopfill-agentic-sql-table1-20261112/runs/huginn-3.5b-budget-online-t0/requests.jsonl.gz'
with gzip.open(path, 'rt') as stream:
    rows = [json.loads(line) for line in stream]
print(len(rows), sum(row['stats']['decode_s'] for row in rows))
```
