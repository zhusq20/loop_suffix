# 2026-11-13 最终新实验归档

本轮需要运行的正式实验已完成。本目录保留最新结果、逐请求记录、冻结源码、配置、训练曲线、真实退出凭据以及历史尝试；此前的 [11月12日快照](../new-results-20261112/README.md)和[更早论文数据](../README.md)继续保留。只归档现有产物，不重新运行模型或已取消的差异诊断。

| 内容 | 入口 | 范围 |
| --- | --- | --- |
| 新基线延迟与接受长度 | [results.csv](baselines/combined-results/results.csv)、[表格](baselines/combined-results/table2_rows.md)、[完成状态](baselines/combined-results/status.json) | DFlash/LoopAhead × 三模型 × 六任务 × T=0/1，共72/72组；保留每行的方法、后端、AR分母及单次计时限制 |
| 额外并发吞吐 | [throughput-results.csv](baselines/combined-results/throughput-results.csv)、[说明](baselines/combined-results/throughput-summary.md)、[审计](baselines/combined-results/throughput-evidence.json) | 两方法 × 两模型 × B=1/2/4/8/16，共20组、9,600请求；其中10组LoopAhead的入场顺序不匹配历史参考，不能解释为完全对齐的加速比 |
| Huginn/Ouro-2.6B SQL请求回放 | [汇总](repository/results/loopfill-agentic-sql-table1-20261112/measurements.csv)、[报告](repository/results/loopfill-agentic-sql-table1-20261112/report.md) | 28/28组、2,800请求，审计通过 |
| Ouro-1.4B SQL请求回放 | [汇总](repository/results/loopfill-agentic-sql-ouro14-20261112/measurements.csv)、[报告](repository/results/loopfill-agentic-sql-ouro14-20261112/report.md)、[最终审计](repository/results/loopfill-agentic-sql-ouro14-20261112/checks/final-audit.json) | 14/14组、1,400请求，审计通过；T=1使用空闲GPU按单元调度，跨卡比较明确披露 |
| 六个草稿器训练与成本 | [成本表](baselines/training_cost.md)、[成本JSON](baselines/training_cost.json)、[账本](baselines/training_cost-ledger.json)、[训练记录](baselines/training/)、[Ouro-1.4B DFlash](repository/results/dflash-ouro-1.4b-training-20261112-v3/) | 六个正式训练全部完成；累计已结束生成、特征提取、训练及额外占卡143.383645 GPU小时，计费口径见成本表 |
| Ouro-1.4B DFlash完整补测 | [完成凭据](baselines/logs/ouro14-packed-evaluation-v1/complete.json)、[配套AR与耗时分析](baselines/logs/ouro14-packed-evaluation-v1/supplement/) | 12个正式测速单元及其AR/profile均结束，控制器37/37步骤完成、无失败 |
| 历史新基线与预检 | [原始汇总](baselines/results.csv)、[请求流](baselines/runs/)、[检查记录](baselines/preflight/)、[调度与退出记录](baselines/logs/) | 保留原状态、原后端、原时间及失败/部分结果，不将历史试跑提升为正式结果 |
| 差异诊断取消 | [最终取消凭据](repository/results/output-diagnostics-cancelled-20261113-045608/cancellation.json)、[策略](baselines/output-divergence-diagnostics-policy.json) | 七组未完成诊断已取消；先前已有完整或部分证据原样保存，不再运行 |

Hugging Face 上的正式草稿器仓库、固定revision和权重SHA256见[发布清单](../published-drafter-weights.json)。目标模型仍从原始发布方下载；Git中不包含模型权重、优化器状态、训练特征缓存或50,000条teacher训练输入。

## 读取与离线核验

文件映射与哈希见 [manifest.json](manifest.json)，请求流核验见 [run-audits.json](run-audits.json)，最终覆盖与指标复算见 [validation.json](validation.json)。所有JSONL以及大于1 MiB的JSON/log采用标准gzip无损压缩，解压后的字节SHA256与原文件一致。原始绝对路径只用于来源追溯；离线核验根据manifest映射定位归档，不依赖原机器。

```bash
python scripts/loop_final_snapshot.py verify --out paper/data/new-results-20261113
python scripts/loop_paper_snapshot.py verify --out paper/data/new-results-20261112
```

例如读取本轮MATH500的AR记录：

```python
import gzip, json
path = 'paper/data/new-results-20261113/baselines/logs/ouro14-packed-evaluation-v1/supplement/runs/table1-ouro-1.4b-math500-t0-ar/requests.jsonl.gz'
with gzip.open(path, 'rt') as stream:
    rows = [json.loads(line) for line in stream]
print(len(rows), sum(r['stats']['decode_s'] for r in rows))
```

## 状态与口径

历史六项 `complete_artifact_exit_unavailable` 继续表示退出码未知，归档不补造成功退出。当前72行汇总允许带已披露限制的既有测量，并分别保留原生/HF或packed执行器以及历史/新测AR分母。CSV中的 `eta_latency` 为延迟比，`eta_throughput` 为吞吐比；不可混用。

部分控制器最后一条 `status.json` 写为 `waiting_for_gpu`，但同时记录37/37完成且没有运行或待派发任务；相邻 `complete.json` 和最终72/72汇总才是本轮结束凭据。某些历史汇总仍标记 `numerical_diagnostics=not_all_completed`，这是已取消项目的历史状态，不代表还有必须运行的正式实验。

SQL为固定历史请求回放上的自由生成解码计时，不是闭环agent成功率或Spider准确率。旧历史快照中的“进行中”描述只对应当时采集时间。本目录也保留历史部分产物，但只有 `run-audits.json` 所列请求流经过本次完整请求数及聚合检查。
