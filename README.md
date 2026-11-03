# LoopFill (loop_suffix)

LoopFill 是面向循环（looped / recurrent-depth）语言模型的免训练投机解码。它把两种草稿来源放进同一棵草稿树：

- **后缀树匹配**（同 SuffixDecoding）：从提示、已确认输出和历史请求中检索续写；
- **早期循环读出**（同 LoopSpec / WFD）：位置在跑完全部 T 轮之前，用第 d 轮的读出先提出下一个词。

两种来源可以互相接续，并按校准得到的“与全深度贪心一致”概率给路径打分，路径分数低于阈值 τ 的节点被剪掉。只有已验证前缀上的全深度预测决定实际输出（FP32 下与贪心逐词一致；BF16 下因数值误差部分题输出不同，见结果摘要）。模型为 Ouro-2.6B（T=4, d=1）与 Huginn-3.5B（T=32, d=4），基准为 Spec-Bench。

本仓库从 [zhusq20/llm-memory-editability](https://github.com/zhusq20/llm-memory-editability) 中抽出 LoopFill 相关的文档、代码和小体积实验产物（2026-11-02 快照）。为保持报告里的链接和 import 不变，目录结构与 Python 包名 `llm_memory_editability` 沿用原仓库。

## 主要结果（详见 [docs/loopfill-results.md](docs/loopfill-results.md)）

单请求引擎 E2E，sb1024 chat，bf16，每模型 60 题：

| 模型 | C(0.01) 相对 AR | C(0.01) 相对同引擎 WFD |
| --- | ---: | ---: |
| Ouro-2.6B | 5.23× [4.49, 6.18] | 1.95× [1.76, 2.20] |
| Huginn-3.5B | 6.94× [5.26, 8.79] | 2.34× [2.16, 2.56] |

多请求（连续批处理）吞吐尚未实测，计划见 [docs/loopfill-plan.md](docs/loopfill-plan.md)。

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

## 进度（截至 2026-11-02）

| 日期 | 内容 | 位置 |
| --- | --- | --- |
| 10-30 | 方案原理：后缀树与浅层读出共用一棵草稿树 | `docs/design/loop-suffix-tree-spec-20261030.html` |
| 10-30 | 前期探索（基于 WFD 调度）：拒绝后复用统计、浅层第 2 候选分支。只有协议和逐题记录，没有写正式报告；主要统计数字引在 sibling 协议开头 | `results/loop-spec-reuse-20261030`、`results/loop-spec-sibling-20261030` |
| 10-30–31 | 离线回放（块调用数）：四种解码器、raw / chat-256 / sb1024、MT-Bench 二轮 | `results/loop-suffix-spec-20261030` |
| 10-31 | 首批墙钟与同引擎 WFD 对照：both-path 相对 WFD 快 1.34–1.40× | `results/loop-suffix-spec-wallclock-20261031`、`results/loop-wfd-same-engine-20261031` |
| 10-31 | 按猜对概率放候选：设计 v1→v3、校准、回放、多请求微基准；loop 给后缀树带来什么（报告 §10） | `results/loop-spec-prob-tree-20261031`、`results/loop-suffix-spec-20261030/report.md` §10 |
| 11-01 | v3 C 单请求墙钟（S4）：相对 both-path 快 1.26–1.73×；v3 C 四栏消融（报告 §11）；单请求 E2E 汇总（见上表） | `results/loop-spec-prob-tree-20261031/report.md` §9、`docs/loopfill-results.md` |
| 11-02 | 论文方法节、预备知识节与图 | `paper/` |

**尚未完成**（计划与顺序见 [docs/loopfill-plan.md](docs/loopfill-plan.md)）：

- 多请求 E2E：固定批与连续批处理、N=4/16 跑 Spec-Bench 全量请求流。目前只有按行数的估计和一次块调用的微基准。
- 在相同负载下补纯 SuffixDecoding 和 LoopSpec (1,2) 基线。
- GPU 消融；“关闭新孩子立即查询后缀索引”这项优先消融。
- 外部工作负载：CanItEdit 可行性检查；SQL / SWE-bench 只是备选。
- 数据缺口：Huginn 摘要题 253、288、295 缺 sb1024 读出（显存不足）；预留题 249、273、281 缺 S4 计时。

`results/*/RUNNING.txt` 是运行当时的交接记录，状态以 `report.md` 和上表为准。

## 运行

```bash
pip install -e ".[dev]"            # 纯 CPU 测试只需 numpy + pytest（部分测试需 torch）
PYTHONPATH=src pytest tests
```

需要模型推理的部分另需：

- `torch`、`transformers`，以及 Ouro-2.6B / Huginn-3.5B 权重；
- [ArcticInference](https://github.com/snowflakedb/ArcticInference) 的后缀树（固定版本 `aca5d9a`）；
- WFD 参考实现（[wavefront-decoding](https://github.com/summerbro-hhj/wavefront-decoding)），默认挂载于 `/wfd`，可用 `LOOP_WFD_ROOT` 或 `--wfd-root` 指定。

**未纳入本仓库**（都在原服务器 `lm-memory-persistent/workspace/llm-memory-editability/<批次>/` 下）：各深度的模型读出 `runs/`（约 150MB，CPU 回放的输入，需用 `scripts/loop_suffix_spec_readouts.py` 在 GPU 上重新生成）、模型权重 `hf/`、容器 Python 环境 `env/`、v1/v2 全量扫参 `sweep/`（59MB；报告已含汇总）、主机监控流水 `monitor*.jsonl`、W&B 离线缓存（已同步到 W&B 项目 `zsqzz/llm-memory-editability`）。`results/` 中的绝对路径只记录原运行环境。
