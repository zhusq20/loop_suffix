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
| `paper/` | 方法与预备知识（`method-loopfill.tex`、`preliminaries-loopfill.tex`）、设计稿 `suffix-loop-design.tex/.bib`、图源（excalidraw / build 脚本 / 导出图） |
| `docs/` | 实验规划（`loopfill-plan.md`）与结果摘要（`loopfill-results.md`） |
| `src/llm_memory_editability/` | `loop_suffix_spec.py` 后缀树+循环调度的离线回放；`loop_prob_tree.py` 概率草稿树（校准、打分、剪枝，v3 C）；`loop_tree_decode.py` Ouro/Huginn 的真实树解码引擎；`loop_prob_decode.py` 概率树解码器 |
| `scripts/` | 读出采集（`loop_suffix_spec_readouts.py`）、CPU 回放与消融（`loop_suffix_spec_simulate.py`、`loop_prob_tree_replay.py`、`loop_prob_tree_ablation.py`）、调度比较、墙钟测速（`loop_suffix_spec_wallclock.py`、`loop_spec_sibling.py`）、多请求微基准与分析出图 |
| `tests/` | 单元测试与对 WFD 参考实现的逐词一致性测试 |
| `results/` | 各批次 `protocol.md` / `report.md`、校准表、汇总 JSON、图表与启动脚本 |

各批次：

- `loop-suffix-spec-20261030`：四种解码器的离线调用数、MT-Bench 二轮、loop 调度对后缀树的贡献（§10）、v3 C 消融（§11）
- `loop-spec-prob-tree-20261031`：概率草稿树设计 v1–v3、校准、S4 单请求墙钟、多请求微基准
- `loop-suffix-spec-wallclock-20261031`、`loop-wfd-same-engine-20261031`：首批墙钟与同引擎 WFD 对照
- `loop-spec-reuse-20261030`、`loop-spec-sibling-20261030`：草稿复用与兄弟分支的上限测量

## 运行

```bash
pip install -e ".[dev]"            # 纯 CPU 测试只需 numpy + pytest（部分测试需 torch）
PYTHONPATH=src pytest tests
```

需要模型推理的部分另需：

- `torch`、`transformers`，以及 Ouro-2.6B / Huginn-3.5B 权重；
- [ArcticInference](https://github.com/snowflakedb/ArcticInference) 的后缀树（固定版本 `aca5d9a`）；
- WFD 参考实现（[wavefront-decoding](https://github.com/summerbro-hhj/wavefront-decoding)），默认挂载于 `/wfd`，可用 `LOOP_WFD_ROOT` 或 `--wfd-root` 指定。

原始读出、逐题轨迹和模型权重体积较大，未纳入本仓库；`results/` 中 JSON 与脚本里的绝对路径仅记录原运行环境。
