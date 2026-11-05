# LoopFill 模型扩展实施记录

日期：2026-11-05。对应 [实施计划](loopfill-model-expansion-implementation.md)。P0–P4 已完成。24个真实检查点短流预检全部通过；按用户最新要求，已停止独立完整AR参考生成，八卡直接运行90个正式吞吐单元。正式结果尚未完成，以下短样本不作为论文性能数据。

## 输入与检查点（P0）

四个新增检查点使用计划中的完整 revision，权重保存在 Hugging Face 缓存。480 个原始题目重新分词，保存原始首轮文本、文本哈希、模型 ID 和 token IDs。Raven 使用 raw、`add_special_tokens=False`；Ouro-1.4B 使用发布的 chat template。

| 模型 | 最长 prompt | 输出预算 | 上下文上限 | 超限题数 | 停止 IDs |
| --- | ---: | ---: | ---: | ---: | --- |
| Ouro-1.4B | 1588 | 1024 | 65536 | 0 | 0、2 |
| Raven-Llama-3.2 | 1421 | 1024 | 131072 | 0 | 128001 |
| Raven-OLMo-2-0425 | 1422 | 1024 | 4096 | 0 | 100257 |
| Raven-TinyLlama-3T | 1733 | 1024 | 2048 | 24 | 2 |

Raven 的模型/generation config 都继承了 EOS=65505；Llama/OLMo 中虽在词表范围内，也不对应 tokenizer 的 EOS；TinyLlama 中更是越界。新协议以 tokenizer 的实际 EOS 为准并保留所有冲突来源。

用户已明确选择：**逐题列出 TinyLlama 超限问题，暂不运行其正式矩阵**。保留全部480题输入，不降低1024输出预算、不静默删题或添加上下文扩展。因此本轮90个可运行正式单元，另30个 TinyLlama 单元标记 `deferred_context_limit`。TinyLlama 后端和短样本仍验证。

## 后端与数值验证（P1–P3）

- Ouro-1.4B 复用24层 Ouro 后端，固定4循环，关闭原生提前退出选项。
- Raven 独立实现构造、层归一化、RoPE、初始化和 coda 算子；共享前置/core/后置缓存编排。Llama/TinyLlama 使用前置 RMSNorm，OLMo 使用残差分支上的 RMSNorm，并对拆头前的完整 Q/K 投影归一化。Raven coda 前不执行 Huginn 的额外 norm。
- 使用检查点自身的 rotary 实现，保留 Llama3 scaling；OLMo 在 FP32 RoPE 后转回 Q/K 原 dtype。
- 初始化采用固定绝对位置表：CPU FP32，每位置独立 generator，seed=(seed+position) mod 2**63；先 randn 再截断正态，再乘 embed_scale 和转换dtype。该约定保留原生分布与每位置随机数消耗顺序，且不依赖请求调度、分块或缓存容量。原生对照显式传入相同 `input_states`，不声称仅设相同全局 seed 就一致。
- KV 按真实 KV 头存储。树 attention 使用分组计算而不复制KV；CUDA前缀采用原有 varlen FlashAttention 的 GQA 路径。全段/分块 prefill 使用支持 GQA 的 SDPA。
- 浅层读出使用独立深度 coda 缓存，保持循环 latent 不变；WFD 延续同引擎已有的共享 draft/verify coda 规则。新增模型的尾部草稿受实际上下文和缓存位置边界限制。

真实检查点 FP32 对照覆盖两个提示、三个树分支和分块 prefill；显式 logit 容差 `atol=2e-4, rtol=2e-4`，贪心首选全部一致。测得最大 logit 误差：

| 模型 | 最大绝对误差 |
| --- | ---: |
| ouro-1.4b | 1.7166138e-05 |
| raven-llama-3.2 | 3.1471252e-05 |
| raven-olmo-2-0425 | 3.528595e-05 |
| raven-tinyllama-3t | 2.3841858e-05 |

BF16 参数与 KV 均显式转换并审计。BF16 数值差异保留，未通过放宽 FP32 阈值掩盖错误。完整证据位于 `results/loopfill-model-expansion-audit-20261105/`，包含锁定模型文件哈希、框架版本、验证源码哈希和每样本误差。

## 评测入口（P4–P6）

新增 `scripts/loop_model_expansion_suite.py`：

```bash
export PYTHONPATH=src
PY=/root/.cache/loopfill/venv/bin/python
OUT=results/loopfill-model-expansion-new  # 使用未存在的目录建立新批次
$PY scripts/loop_model_expansion_suite.py prepare --out "$OUT"
export PYTHONPATH="$OUT/source/src"
$PY "$OUT/source/scripts/loop_model_expansion_suite.py" pipeline --out "$OUT"
```

默认 `pipeline` 依次执行 validate、references、run、summarize。正式运行使用准备阶段冻结的 `source/` 副本；不要修改冻结文件。每模型六方法，B=1/2/4/8/16，N16、Q4、top2、孩子Q(4,4)，online c₀=.7、η=.1、范围[.25,2]；seed=0、repeat=0、BF16、continuous/packed、128-token 分块 prefill、Graph-off、无执行行填充，2B请求各最多32token预热。

`validate` 采用按六类任务与长短输入选出的8题，每题16token、B2。默认协议在完整参考完成后运行正式矩阵。本次按用户指示采用 `prepare-throughput --previous <已验证套件> --out <新目录>` 创建继任协议，设置 `require_serial_reference=false`；逐项核对执行源码、输入与检查点身份，重新审计并复用24个成功预检，随后跳过 references 阶段。正式命令不带 `--check` 或 `--reference-file`，独立串行一致率留空；六方法中的计时 AR 基线保留，仍可汇总同 batch AR 输出对比。每单元独占GPU，沿用八卡共享队列、固定资源检查、GPU UUID/CPU affinity记录和遥测。实际 OOM 与其他错误分别记录。

汇总脚本重新读取逐请求 `stats.tokens` 和 `stats.calls`，验证 AR 的 γ=1，输出独立 `throughput.csv/md` 与 `gamma.csv/md`，保留零输出推进请求、OOM、失败和暂缓单元。六模型汇总保留旧两模型的原冻结协议，披露 Raven raw prompt 与旧 chat prompt 差异。


## 本次冻结产物与状态

当前正式套件：[protocol.json](../results/loopfill-model-expansion-throughput-20261105/protocol.json)、[status.json](../results/loopfill-model-expansion-throughput-20261105/status.json)。最初准备批次在模型元数据 JSON tuple/list 一致性检查处停止，未执行任何模型生成或计时；修复后另建 v2。v2 完成24个预检后，在独立参考生成阶段按用户要求停止；部分参考保留于原目录（Ouro 286题、Llama 42题、OLMo 37题），未作为完整参考复用。停止记录见 [reference-generation-stopped.json](../results/loopfill-model-expansion-20261105-v2/reference-generation-stopped.json)。继任套件沿用完全相同的基准和解码源码哈希，只调整启动、审计及汇总策略。

各模型物理缓存长度按输入重新计算：Ouro-1.4B=2688，Raven-Llama/OLMo=2560，TinyLlama=2048。AR=1树槽，预算臂=17，LoopSpec为Ouro15/Raven499，WFD=1152。正式Graph关闭，不填充执行行。上述均为容量配置，不代表扩展模型的上下文能力。

已核验六模型汇总入口能够重建现有两个模型的59个成功单元和1个OOM；新增模型的未完成单元只显示状态。当前表格：[吞吐](../results/loopfill-model-expansion-throughput-20261105/paper-summary/throughput.md)、[γ](../results/loopfill-model-expansion-throughput-20261105/paper-summary/gamma.md)。正式运行结束后由 pipeline 自动更新。

TinyLlama 超限清单（所需上下文=prompt长度+1024）：

| 题目ID | prompt tokens | 所需上下文 |
| --- | ---: | ---: |
| 248 | 1236 | 2260 |
| 253 | 1559 | 2583 |
| 256 | 1028 | 2052 |
| 258 | 1222 | 2246 |
| 264 | 1072 | 2096 |
| 266 | 1036 | 2060 |
| 267 | 1054 | 2078 |
| 269 | 1310 | 2334 |
| 273 | 1485 | 2509 |
| 274 | 1164 | 2188 |
| 278 | 1130 | 2154 |
| 279 | 1441 | 2465 |
| 280 | 1150 | 2174 |
| 282 | 1146 | 2170 |
| 285 | 1277 | 2301 |
| 288 | 1701 | 2725 |
| 295 | 1681 | 2705 |
| 297 | 1505 | 2529 |
| 298 | 1325 | 2349 |
| 306 | 1650 | 2674 |
| 311 | 1276 | 2300 |
| 316 | 1563 | 2587 |
| 317 | 1733 | 2757 |
| 318 | 1490 | 2514 |


## P4 完成与长任务启动

四模型六方法共24个预检单元全部 complete，无OOM/error；AR均满足calls=T×tokens、γ=1，预算臂峰值不超过16。BF16逐请求与串行AR的输出分歧已保留在每单元JSONL，预检一致请求数为4–7/8，不声明等输出。[预检汇总](../results/loopfill-model-expansion-20261105-v2/validation-summary.json)。

后台流程由新冻结源码启动，启动信息见 [launcher.json](../results/loopfill-model-expansion-throughput-20261105/launcher.json)，运行日志见 [pipeline.log](../results/loopfill-model-expansion-throughput-20261105/pipeline.log)。当前阶段为 run，监督进程 PID=361434；启动核验为8个正式单元运行、82个排队、0个错误，八张GPU各有一个独占进程。流程自动消费90单元共享队列并更新六模型表格。TinyLlama正式30单元一直保留deferred_context_limit。

吞吐切换回归：协议命令、无独立参考审计、缺失一致率汇总及旧协议兼容共29项测试通过；完整480题覆盖检查仍强制执行。新增脚本与测试通过Ruff，`git diff --check`通过；旧 `loop_confirm_suite.py` 的3项既有lint问题未改动。

CPU合并测试发现OLMo分块KV舍入差异约4.2e-6；独立FP64 dense对照测得整段/1token分块/3token分块logit最大误差分别为9.99e-7、6.14e-7、7.98e-7。KV单测统一采用与latent单测相同的atol=5e-6、rtol=5e-5；独立FP64对照额外以atol/rtol=3e-6验证。真实检查点FP32阈值2e-4未变，推理源码没有因该单测调整而修改。


最终回归：CPU全仓451 passed、102 skipped（主要为CUDA与外部检查点条件）；独立FP64新增测试通过。此前包含CUDA路径的受影响回归170 passed，新增OLMo BF16 GQA与协议检查11 passed。代码通过Ruff和`git diff --check`。这些验证日志保存在 `results/loopfill-model-expansion-audit-20261105/`。

最终新增模型专项（GPU可用）：`34 passed in 9.51s`。
