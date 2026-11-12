# 论文表格与图的数据索引

**2026-11-13 最终结果**：[最终新实验归档](new-results-20261113/README.md)收齐72/72组DFlash/LoopAhead延迟基线、20组吞吐补测、42组SQL回放及六个已完成草稿器的训练与成本。保留全部历史快照、原始限制和差异诊断取消记录；权重单独发布至Hugging Face，见[发布清单](published-drafter-weights.json)。

本索引中的CSV、JSON、图和压缩逐请求数值记录均纳入Git。`manifest.json`记录文件大小、SHA256与所属分析；原始输入、模型输出全文和运行日志的本地路径仅用于来源追溯，重新绘制下列图不需要这些本地路径。

**2026-11-12 新数据已归档**：[新数据索引](new-results-20261112/README.md)集中保存 DFlash/LoopAhead 新基线、Agentic SQL完整证据、32/32组机制耗时分析、五个已完成草稿器的训练曲线与成本，以及未完成实验的状态快照。共184个已结束请求流、48,744条逐请求记录；后端、历史/新测AR分母及缺失退出码状态分别标注。其[文件清单](new-results-20261112/manifest.json)可用 `python scripts/loop_paper_snapshot.py verify --out paper/data/new-results-20261112` 离线核验。

| 表或图 | 已提交的数据 | 范围与状态 |
| --- | --- | --- |
| 表1：greedy延迟与γ | [measurements.csv](../../results/loopfill-latency512-20261107/measurements.csv)、[JSON](../../results/loopfill-latency512-20261107/measurements.json) | 90配置、31,380请求，B1/max512；原逐请求日志此前已入库 |
| 表1：temperature=1 | [measurements.csv](../../results/loopfill-latency512-t1-20261108/measurements.csv)、[JSON](../../results/loopfill-latency512-t1-20261108/measurements.json) | 90配置、31,380请求；与greedy分开 |
| 表2：吞吐随B变化 | [measurements.csv](../../results/loopfill-table2-20261107-v2/measurements.csv)、[主表](../../results/loopfill-table2-20261107-v2/table2-final.md) | 74成功、1实测OOM；B1/2/4/8/16、max1024；原逐请求日志此前已入库 |
| Preliminary覆盖、命中率、接续收益 | [figure-data.json](../figures/loopfill-preliminary-v2/figure-data.json)、[完整summary](../../results/loopfill-preliminary-20261109-v2/full/summary.json)、[metrics.csv](../../results/loopfill-preliminary-20261109-v2/full/metrics.csv) | 完整480题，原分析产物此前已入库 |
| 四组消融主表、四面板六任务雷达图（含child Q2） | [measurements.csv](../../results/loopfill-four-ablation-q2-20261109/measurements.csv)、[task-metrics.csv](../../results/loopfill-four-ablation-q2-20261109/task-metrics.csv)、[逐请求数值](../../results/loopfill-four-ablation-q2-20261109/paper-data.jsonl.gz) | 40配置、19,200请求全部完成；每模型B16的16组核心配置及B1节点预算。复用原36组，仅新增4组Q2、1,920请求 |
| 最终接受token来源堆叠图 | [overall.csv](../../results/loopfill-token-sources-20261109/overall.csv)、[by-task.csv](../../results/loopfill-token-sources-20261109/by-task.csv)、[per-request.csv](../../results/loopfill-token-sources-20261109/per-request.csv) | 15组、7,200请求；总体、分任务、逐请求分母均保留 |
| 固定c与oracle分数协调诊断 | [table.csv](../../results/loopfill-score-headroom-t0-20261109/table.csv)、[paired.csv](../../results/loopfill-score-headroom-t0-20261109/paired.csv)、[analysis.json](../../results/loopfill-score-headroom-t0-20261109/analysis.json)、[逐请求数值](../../results/loopfill-score-headroom-t0-20261109/paper-data.jsonl.gz) | 36组、4,320请求；含等待分解、逐位置进度、共同输出匹配数据 |
| 树采样吞吐、计算量及草稿数分布 | [measurements.csv](../../results/loopfill-tree-sampling-b16-seed0-20261109/measurements.csv)、[task-metrics.csv](../../results/loopfill-tree-sampling-b16-seed0-20261109/task-metrics.csv)、[analysis.json](../../results/loopfill-tree-sampling-b16-seed0-20261109/analysis.json)、[逐请求数值](../../results/loopfill-tree-sampling-b16-seed0-20261109/paper-data.jsonl.gz) | 8组、3,840请求；保存完整直方图、计数与输出差异审计 |
| 负载—节点预算与计算效率 | [load-budget.csv](../../results/loopfill-mechanisms-20261109/existing/load-budget.csv)、[ablation-efficiency.csv](../../results/loopfill-mechanisms-20261109/existing/ablation-efficiency.csv) | B1/B16、N8/16/32/64；效率CSV保留36配置 |
| 提前计算CDF、立即就绪、等待分解 | [readiness.csv](../../results/loopfill-mechanisms-20261109/existing/readiness.csv)、[readiness-cdf.csv](../../results/loopfill-mechanisms-20261109/existing/readiness-cdf.csv) | 复用两模型240条B1/max512诊断；底层逐请求wait_entries在分数诊断数值包中 |
| 耗时分解（已完成） | [measurements.csv](new-results-20261112/repository/results/loopfill-mechanisms-20261109/timing/measurements.csv)、[stages.csv](new-results-20261112/repository/results/loopfill-mechanisms-20261109/timing/stages.csv)、[pairs.csv](new-results-20261112/repository/results/loopfill-mechanisms-20261109/timing/pairs.csv)、[图PDF](new-results-20261112/repository/results/loopfill-mechanisms-20261109/timing/stage-timing.pdf) | 32/32流完成，每流24题，共768记录；正常与诊断分开。旧 `timing-snapshot` 保留为历史部分结果 |

## 重画与读取

```bash
# 无需GPU或原始大日志：从已提交CSV/JSON重画机制、分数诊断、树采样与来源图
PYTHONPATH=src python scripts/loop_paper_replot.py --out /tmp/loopfill-paper-figures

# 表1/表2按已提交汇总数据生成
python scripts/loop_final_tables.py

# 查看逐请求数值包（标准gzip压缩JSONL）
python - <<'PY'
import gzip, json
path = 'results/loopfill-score-headroom-t0-20261109/paper-data.jsonl.gz'
with gzip.open(path, 'rt') as f:
    row = json.loads(next(f))
print(row['job_key'], row['stats']['tokens'], row['stats']['calls'])
PY
```

`paper-data.jsonl.gz`保留原始`stats`、请求计时、`trace`、`waits`、`wait_entries`及直方图。输出token数组用长度和SHA256替代，能恢复共同输出与配对子集；各包的`paper-data-manifest.json`记录来源哈希、每组计数和汇总。逐请求数值足以重新聚合本文图表，不能替代任务质量评测所需的输出文本。

实验计时口径仍按各自协议：512/1024 token、正常/诊断、B1延迟/B16吞吐不混用。11月12日和13日归档中的JSONL采用无损压缩并保留完整token数组；上述旧包的token哈希转换规则保持不变。新基线的配套补采已完成，最终结果从11月13日归档读取；旧部分快照保持原貌。
