# LoopFill 模型扩展实现方案：Ouro-1.4B 与 Raven 三模型

日期：2026-11-05。状态：**实施中；新模型后端已接入，P0–P3 验证与 P4–P6 状态见 [实施记录](loopfill-model-expansion-progress.md)**。

本文落实新增 Ouro-1.4B、Raven-Llama-3.2、Raven-OLMo-2-0425、Raven-TinyLlama-3T 的实现路径，依据 LoopAhead、LoopSpec 的模型选择与公开检查点，以及当前仓库后端。模型事实与检查点版本已经核查；下文标为“建议”的配置属于实施决策，须写入新批次的冻结协议，不代表已有测量结果。

实验指标与总体范围沿用 [论文实验规划](loopfill-paper-experiments.md)。原设计交付为本文档；现按用户要求实施。新增运行使用独立冻结协议，历史运行结果和协议保持原定义。

## 1. 目标与范围

保留已评测的 Ouro-2.6B、Huginn-3.5B，新增四模型，形成六模型评测范围。新增后覆盖 LoopAhead 与 LoopSpec 共同采用的全部五个非 Thinking 模型，并保留 Huginn 的额外架构覆盖。

| 模型 | LoopAhead | LoopSpec | 本项目状态 |
| --- | --- | --- | --- |
| Ouro-1.4B | 使用 | 使用 | 本次拟新增 |
| Ouro-2.6B | 使用 | 使用 | 已有六方法五档 B 结果 |
| Raven-Llama-3.2 | 使用 | 使用 | 本次拟新增 |
| Raven-OLMo-2-0425 | 使用 | 使用 | 本次拟新增 |
| Raven-TinyLlama-3T | 使用 | 使用 | 本次拟新增 |
| Ouro-1.4B-Thinking | 使用 | 使用 | 后续可选，不纳入本次矩阵 |
| Ouro-2.6B-Thinking | 未列入模型表 | 使用 | 后续可选，不纳入本次矩阵 |
| Huginn-3.5B | 未列入模型表 | 未列入模型表 | 已有六方法五档 B 结果 |

来源：[LoopAhead 附录 A.1、表4](https://arxiv.org/html/2610.21594v1#A1.SS1)、[LoopSpec 附录 A](https://arxiv.org/html/2609.17184v1)、[LoopSpec 官方模型列表](https://github.com/kaist-flexml-lab/loopspec#supported-models)。

本次扩展的是目标模型，主对照仍是 AR、同引擎 WFD、同引擎 LoopSpec、Suffix-only、LoopFill-raw、LoopFill-online。不因此增加 LoopAhead 的训练型 LAD drafter，也不自动恢复已撤销的全矩阵重复计时、任务质量评分或官方独立引擎测速。

## 2. 检查点与版本锁定

以下 revision 与 LoopAhead 表4的版本前缀一致，并已通过 Hugging Face 元数据核对完整 SHA。下载模型、tokenizer 和远程模型代码时使用同一完整 revision；正式实验不使用浮动的 `main`。

| 内部模型 ID（建议） | Hugging Face 检查点 | 完整 revision | 权重参数量约 |
| --- | --- | --- | ---: |
| `ouro-1.4b` | [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | 1.435B |
| `raven-llama-3.2` | [smcleish/Recurrent-Llama-3.2-train-recurrence-32](https://huggingface.co/smcleish/Recurrent-Llama-3.2-train-recurrence-32) | `a5f6f126e9d9302445346844f1d7d29d111a06eb` | 1.385B |
| `raven-olmo-2-0425` | [smcleish/Recurrent-OLMo-2-0425-train-recurrence-32](https://huggingface.co/smcleish/Recurrent-OLMo-2-0425-train-recurrence-32) | `e1208e9af61d9e2fe1be95857e3490efb524294f` | 1.359B |
| `raven-tinyllama-3t` | [smcleish/Recurrent-TinyLlama-3T-train-recurrence-32](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-32) | `2579b6a0e9837dcea0f9fcaa85788f74d4a1f2b4` | 0.844B |

参数量来自各检查点的 Hugging Face `safetensors.total` 元数据，不是循环展开后的等效规模。Llama-3.2 中的 3.2 是版本号。用户所称“Raven Omo 2”对应上表的 Raven-OLMo-2-0425。

冻结产物至少包含：模型 ID、repo ID、revision、config/tokenizer/模型代码哈希、权重文件清单、实际加载的模型类、框架版本、参数与 KV 的实际 dtype。权重放入模型缓存，不提交 Git；轻量配置和代码按仓库现有方式管理。

## 3. 模型结构和读出设置

### 3.1 已核查的架构差异

| 模型 | T | 前置层 / 共享核心层 / 后置层 | Q / KV 头数 | checkpoint 权重精度 | 配置中的最大位置数 |
| --- | ---: | --- | --- | --- | ---: |
| Ouro-1.4B | 4 | embedding / 24 / norm + head | 16 / 16 | BF16 | 65,536 |
| Raven-Llama-3.2 | 32 | 4 / 6 / 4 | 32 / 8 | FP32 | 131,072 |
| Raven-OLMo-2-0425 | 32 | 4 / 6 / 4 | 16 / 16 | FP32 | 4,096 |
| Raven-TinyLlama-3T | 32 | 4 / 8 / 4 | 32 / 4 | FP32 | 2,048 |

以上来自锁定检查点的 `config.json`、`raven_modeling_minimal.py` 与权重元数据。配置中的最大位置数不是已经验证的可用上下文范围；物理缓存容量也不等于模型上下文能力。

Ouro-1.4B 与现有 Ouro-2.6B 共用 Ouro 架构；当前 `OuroLoop` 已从配置读取共享层数。Raven 与 Huginn 都有前置层、循环核心和后置读出层，但层内归一化、位置编码、GQA 与初始化细节不同，不能通过模型名称替换直接视为兼容。

### 3.2 文献配置与本项目建议

| 模型 | LoopSpec 第一 / 第二读出 | LoopAhead 第一 / 第二读出 | 本次 LoopFill 建议默认 |
| --- | --- | --- | --- |
| Ouro-1.4B | 1 / 2 | 1 / 2 | 1 / 2 |
| Raven-Llama-3.2 | 2 / 8 | 4 / 8 | 2 / 8 |
| Raven-OLMo-2-0425 | 2 / 8 | 4 / 8 | 2 / 8 |
| Raven-TinyLlama-3T | 2 / 8 | 2 / 8 | 2 / 8 |

LoopSpec 设置参考[官方 README](https://github.com/kaist-flexml-lab/loopspec#compare-decoding-modes)与[实验配置](https://github.com/kaist-flexml-lab/loopspec/blob/main/experiments/110-speed-math.yaml)；LoopAhead 设置参考[附录 A、表7](https://arxiv.org/html/2610.21594v1)。

实施建议：新增 Raven 的 LoopFill 先固定 2/8，与同模型 LoopSpec 的读出位置对齐；WFD 首次读出固定为 2。Raven-Llama/OLMo 的 4/8 仅作为后续明确命名的可选消融，不在480题结果上选优后替换主表。现有 Huginn 的 4/8 结果保持原定义。

将 `loopspec_depths`、`wavefront_depth`、`loopfill_readout_depths` 分开登记。当前 `--model-readout-depths` 只适用于预算臂，不应隐式修改 LoopSpec。LoopSpec 保持 `0 < d1 < d2 < T` 且 `T % d1 == d2 % d1 == 0`。

## 4. 当前代码边界与拟修改文件

| 文件 | 当前限制 | 拟修改内容 |
| --- | --- | --- |
| [loop_batch_bench.py](../scripts/loop_batch_bench.py) | `SPECS` 仅两模型；非 Ouro 分支直接构造 Huginn；EOS、深度及容量入口绑定旧模型 | 使用模型注册表；分开方法读出配置；分派 Raven；审计 tokenizer、停止规则和长度 |
| [loop_tree_decode.py](../src/llm_memory_editability/loop_tree_decode.py) | Ouro/Huginn backbone；注意力和 Huginn 构造器存在 Q/KV 头数相等假设 | 保留旧模型路径；为公共 KV/attention 增加 GQA 支持；接入 Raven 所需接口 |
| `src/llm_memory_editability/loop_raven_decode.py`（拟新增） | 尚不存在 | 实现 Raven backbone，封装 Llama/TinyLlama 与 OLMo 的层差异 |
| `src/llm_memory_editability/loop_model_registry.py`（拟新增） | 尚不存在 | 模型 ID、检查点、结构、深度、tokenizer/停止策略与 backend 类型的统一登记 |
| [loop_batch_decode.py](../src/llm_memory_editability/loop_batch_decode.py) | `FusedLoopExecutor` 限定两个 backbone 类型，并内嵌 Huginn 运算 | 增加 Raven 的融合执行路径，确保请求隔离 |
| [loop_optimized_decode.py](../src/llm_memory_editability/loop_optimized_decode.py) | packed attention、chunked prefill、读出和槽布局有 Ouro/Huginn 分支 | 支持 Raven/GQA、读出深度缓存、分块 prefill；保持 Graph-off 主协议 |
| [loop_eager_suite.py](../scripts/loop_eager_suite.py) | 准备流程含固定模型路径、旧输入及参考迁移 | 抽取可复用准备逻辑；新增模型独立建立输入、AR参考和协议 |
| `scripts/loop_model_expansion_suite.py`（拟新增） | 尚不存在 | 冻结并调度四模型六方法五档 B；复用现有独占 GPU 队列与结果格式 |
| [loop_accepted_length.py](../scripts/loop_accepted_length.py) | 已按元数据读取 T，但会检查 suite 输入与来源哈希 | 复用提取逻辑，核对新 suite schema；增加六模型表格汇总 |
| `tests/test_loop_raven_decode.py`、`tests/test_loop_model_registry.py`（拟新增） | 尚不存在 | 新架构计算、配置、tokenizer/长度审计和后端一致性验证 |

拟新增路径是实施目标，不应作为当前可运行接口引用。优先采用小范围 backbone 分派或明确的能力接口；不要让 Raven 落入所有“非 Ouro 即 Huginn”的旧分支。检索序列推理、融合执行、packed 执行和容量估算中的相关分支，逐项检查。

## 5. Backbone 与执行器实现

### 5.1 Ouro-1.4B

复用 `OuroLoop`，登记 `T=4`、读出1/2和锁定检查点。显式固定循环数并禁用自适应提前退出。核对24层均执行、每循环后的 norm 位置、tokenizer及停止规则；不根据2.6B结果推断1.4B已经通过验证。

### 5.2 Raven 独立实现

提供与现有解码状态机兼容的 `prefill`、`embed`、`advance`、`readout`、`readout_top`、`commit`、`length`、`synchronize` 接口，并支持预算臂的可选已有孩子概率提取参数。每次 `advance` 推进一次完整 recurrent core，`TreeNode.step` 记录实际已完成循环数。

需要逐项对齐锁定模型代码：

1. **前置层与输入注入**：embedding、prelude、adapter 的运算顺序及静态输入保存方式。
2. **层计算**：Llama/TinyLlama 使用对应的残差和归一化顺序；OLMo 保留 Q/K normalization 及其归一化维度。不要复用 Huginn 的四个 norm 假设。
3. **RoPE**：调用或忠实实现各检查点的位置编码；Llama-3.2 保留 `llama3` scaling，TinyLlama 与 OLMo 使用各自参数，不继承 Huginn 的 complex RoPE 表假设。
4. **循环初始化**：核查公开模型的 latent 初始化和随机数消耗。AR与六方法使用一致、与调度顺序无关的初始化约定，并写入协议；与原生参考比较时对齐相同初始状态。不能仅以“都设了 seed”认为状态相同。
5. **读出**：中间深度和最终深度均执行正确的 coda 与 norm/head；读出使用临时状态，不覆盖继续循环所需的 latent。
6. **精度**：Raven 权重以FP32发布，正式BF16测试需显式转换并验证实际参数/KV dtype，复用 `enforce_precision` 的审计方式。

### 5.3 GQA 与 KV 隔离

Raven-Llama 的 Q/KV 头数为32/8，TinyLlama为32/4。KV 按真实 KV 头数存储；attention 将对应查询头映射到同一 KV 组，不把 Q 头数误当作 KV 存储维度。

同时覆盖全段 prefill、chunked prefill、单请求树 attention、融合 batch、packed batch。若使用临时 `repeat_kv` 验证正确性，其时间和临时显存必须计入，不将该原型性能冒充已优化实现。支持 GQA 的 kernel 路径应与独立 dense 参考比较。

缓存必须区分请求、循环深度、层和树分支。接受节点提交后才能回收；被拒绝分支不能污染前缀。coda 缓存还要区分浅层读出深度与全深度验证：明确每种方法采用独立深度缓存还是其既有共享规则，核对参考实现并登记，不能无意混用。

### 5.4 Packed 执行和分块 prefill

Raven 必须支持当前主协议的 continuous batching、packed attention 与128-token chunked prefill。先验证串行实现，再验证 fused、最后验证 packed；正式性能不能以 per-request attention 结果填入 packed 主表。

prefill/decode 交错时，前置层、各循环核心和各读出深度的有效前缀长度均需正确。不同请求混合在一批时，使用各自位置和树掩码。主实验关闭 CUDA Graph 和执行行填充，新增实现不以 Graph 支持作为首阶段前置工作。

## 6. 输入、停止规则与上下文审计

使用同一组480个原始题目 ID；每模型使用自己的 tokenizer 重新生成输入，不复制 Ouro/Huginn 的 `prompt_token_ids`。冻结原始文本、最终 token IDs、输入哈希、特殊 token 设置和每题输出预算。

**Prompt 建议**：Ouro-1.4B 沿用其发布的 chat template，与当前 Ouro 实验保持设置一致；三款 Raven 的 tokenizer 配置未提供 chat template，建议使用明确登记的原始首轮文本，`add_special_tokens=False`，与两篇论文的 base-model 路线一致。不要添加自造 chat template，也不要在 raw 路径依赖 tokenizer 的默认 BOS 行为。Raven 的 prompt 格式差异必须在跨模型汇总中披露；同一模型六方法使用完全相同的输入。

**停止规则是必修项**：当前 `load_requests` 对非 Ouro 模型优先采用 `model.config.eos_token_id`。核查发现 Raven-TinyLlama 的配置包含 `eos_token_id=65505`，但其词表大小为32000，该值不能直接作为合法 EOS。实施时比较 tokenizer、generation config 与模型 config，解析实际有效的 EOS/停止集合；检查 `0 <= token_id < vocab_size`，对冲突记录来源及选择理由。不得将 Huginn 的 `<|end_turn|>` 作为 Raven 默认停止符。

**长度审计**：逐题计算 prompt 长度、最大输出长度及草稿访问的最高位置，分别报告模型支持范围和物理缓存需求。TinyLlama 配置标注2048，不能因为申请了2816缓存槽就认定支持2816上下文。

- 有效上下文内才允许发起对应模型计算；限制尾部草稿不得越过允许的位置。
- 列出超过上下文的题目及长度，不静默截断输入、减少1024输出预算或删除长题。
- 若全480题无法满足该检查点的支持范围，先形成明确的上下文扩展方案或单独的短上下文协议；该模型不得标记为已完成原始480题协议。
- CPU内存、显存或模型限制分别记录；不能将非OOM错误记为OOM。

此节的 prompt 与上下文选择需要在实施阶段随审计结果冻结，不对新模型假设现有2816容量和停止 token 均可复用。

## 7. 正式评测矩阵与容量

每个新增模型按以下六方法、五档 B 建立30个独立单元；四模型计划120个单元，全部成功时为57,600次正式请求生成。加上现有两模型，计划矩阵共180个单元；现有 Huginn WFD B16 的OOM记录保留，不能计为成功。

| 展示名 | CLI arm | 关键配置 |
| --- | --- | --- |
| AR | `ar` | 固定 T 完整验证，无草稿 |
| WFD（同引擎） | `wavefront` | Ouro首次读出1；Raven建议首次读出2；保留自身调度 |
| LoopSpec（同引擎） | `loopspec-dual` | Ouro1/2、Raven2/8；选择性第二提案；不套用N16 |
| Suffix-only | `budget-suffix` | N16、Q4，仅后缀提案 |
| LoopFill-raw | `budget-raw` | N16、Q4、top2、孩子Q(4,4)、固定c=1 |
| LoopFill-online | `budget-online` | 与raw相同提案设置；c₀=0.7、η=0.1、范围[0.25,2] |

统一设置：B=1/2/4/8/16、greedy、BF16、完整480题、最多1024新token、seed=0、repeat-id=0、continuous/packed、128-token chunked prefill、Graph-off、不填充执行行、2B个请求各最多32token预热。预热历史丢弃；每方法历史冷启动，只发布完成请求的输出。新增零分过滤、已有孩子真实概率、来源合并和长度评分均不叠加到这批主配置。

沿用八卡共享队列、每单元独占一卡及既有机器设置，记录 GPU UUID、CPU affinity 与运行遥测。保留单次计时口径，预热和加载不计入正式墙钟，prefill/decode/CPU调度/后缀操作计入。

物理容量必须按新模型和方法重新计算，不能照抄旧套件：

| 方法/配置 | 当前实现的物理树槽数 | 说明 |
| --- | ---: | --- |
| AR | 1 | 已确认根 |
| 三个预算臂，N16 | 17 | 根不计入N；采用可回收槽 |
| LoopSpec，T4、1/2 | 15 | `2 * live_bound + 1`，live_bound=7 |
| 现有Huginn LoopSpec，T32、4/8 | 109 | live_bound=54，仅作旧配置参照 |
| 新Raven LoopSpec，T32、2/8 | **499** | live_bound=249，应由 `LoopSpecConfig.physical_nodes` 计算 |
| WFD | 沿用当前1152起点并核对上界 | 其槽管理不同于预算臂，不能替换成N+1 |

499是本仓库完整调度的分配上界，不是每一步实际活跃节点数，也不是显存必然不足的结论。KV大小还依赖前置/核心/coda层数、T、KV头数和上下文。分配失败记录实际OOM；不自动降低B、N、循环数或输出预算，不把 `loopspec-capped` 偷换成主基线。

## 8. 指标、汇总与复现产物

主表分开呈现两张表：**吞吐 token/s** 和 **循环归一化平均接受长度 γ**，每个模型六行、五档B五列，避免将两项指标合在同一单元格造成歧义。

```text
throughput = 整流 output_tokens / 正式 wall_s
gamma      = T * sum(request.stats.tokens) / sum(request.stats.calls)
```

`stats.tokens` 排除 prefill 首token，`stats.calls` 统计实际逐请求 recurrent core 推进轮数，包括没有输出的轮次。即使调度一次推进d1轮，也必须累计d1个loop steps；不得用全GPU合并后的 `core_calls` 或调度步数替代。AR应满足 `calls = T * tokens`、γ=1。

报告整体和六类任务γ，使用总量之比而非逐请求γ的算术平均。零token/零calls请求保留并计数；OOM及失败不填写吞吐或γ。混合流墙钟不分摊成类别吞吐。不同模型的T、每轮层数和coda成本不同，γ只衡量循环步节省，不代表FLOPs或跨模型墙钟效率。

保存以下产物，命名尽量复用现有 eager suite：

- `protocol.json`、模型配置/输入/源码哈希、运行环境及检查点来源。
- `inputs/<model>.jsonl`、tokenizer/停止规则审计、逐题上下文审计。
- 模型专属的计时外AR参考及新后端验证记录；不复用其他模型参考。
- `jobs/`、`status.json`、`measurements.csv`、`aggregate.csv`、逐请求输出和统计。
- `accepted-length/overall.csv`、`by-category.csv`、`per-request.csv`、`audit.json`。
- 吞吐表、γ表，以及输出量、整批墙钟、显存、输出分歧和OOM说明。

所有基线注明“同引擎实现”；LoopSpec原论文采用的SGLang/CUDA Graph结果不直接填入当前eager表。模型扩展本身不证明等输出或质量无损；BF16分歧按现有规则保留并报告。

## 9. 验证和推进顺序

以下是新架构接入所需的工程验证，不扩展为既有模型的全量数值差异研究或任务质量评测。通过已有阶段后，只有新改动、失败或未解决问题才重复相应验证。

| 阶段 | 工作 | 完成标准 |
| --- | --- | --- |
| P0：登记与输入审计 | 锁定四检查点；生成模型注册信息；审计tokenizer、EOS、上下文及容量 | 输入覆盖和停止策略明确；没有无效ID或静默截断；未决上下文项单列 |
| P1：Ouro-1.4B | 复用Ouro后端；验证加载、循环数和六方法短流 | 独立参考与共享引擎对照完成；六方法能运行并正确记录统计 |
| P2：Raven串行后端 | 实现前置层、core、coda、初始化、RoPE、GQA和OLMo差异 | 小模型FP32与独立dense参考的状态/logits/greedy对齐；真实检查点短样本验证 |
| P3：Raven fused/packed | 跨请求执行、树attention、分块prefill、缓存提交回收 | 不同batch/分块方式、请求补位和分支拒绝均保持正确缓存语义 |
| P4：六方法预检 | 固定分层短样本，逐模型检查六方法、统计和资源 | γ口径正确；无结构性实现错误；BF16差异有记录；容量异常分类准确 |
| P5：正式矩阵 | 用冻结源码和输入运行新增120个计划单元 | 每单元有complete/oom/error状态；成功单元覆盖全部480题 |
| P6：论文汇总 | 重提吞吐、γ并输出六模型主表和复现说明 | 数字可由逐请求记录重建；历史结果协议清晰，未混用后端或参数 |

必要测试重点：

- MHA与GQA：Q/KV头比1、4、8；祖先可见、兄弟不可见、请求间不可见。
- Raven真实层算子的小尺寸版本：Llama/TinyLlama与OLMo分别对照，不能只用假logits调度器证明架构正确。
- 同一确定初始状态下，串行/fused/packed、全段/分块prefill的对应状态和输出；读出不改写循环latent。
- 多深度coda缓存、提交与回收、连续接受、EOS、输出边界、请求结束后槽复用。
- 六种方法的AR参考、N预算与LoopSpec2/8调度；`calls`按实际循环计数，AR γ=1。
- tokenizer的BOS/EOS冲突、没有chat template、越界停止ID、上下文超限以及新模型参考不能误复用。
- 对修改到的公共路径运行现有Ouro/Huginn相关回归，避免改变历史默认行为。

FP32计算对照使用显式数值容差，并报告最大误差；greedy分歧检查首个位置与logit间隔，不能通过放宽阈值掩盖错误。真实BF16短流与全量参考差异分别记录。当前阶段不把“GPU上能跑完”作为适配正确的充分条件。

## 10. 实施阶段仍需冻结的事项

| 事项 | 本文建议 / 处理原则 |
| --- | --- |
| Raven LoopFill读出深度 | 主配置先2/8；4/8作为可选消融，独立命名 |
| Raven prompt格式 | 原始首轮文本、无自动BOS；按模型冻结，披露与旧chat协议差异 |
| Raven随机初始化 | 对齐公开模型语义，确定请求/位置与seed规则，使六方法不因执行顺序消耗不同随机状态 |
| coda历史语义 | 核对官方参考和本地方法约定后登记，禁止浅层状态污染最终验证 |
| TinyLlama上下文 | 完成全量分词和位置审计后决定；未完成不能承诺原协议480题全覆盖 |
| EOS与generation config冲突 | 以核实后的tokenizer语义和合法ID生成明确停止集合，记录选择依据 |
| 首轮正式配置 | 将本文建议写入新的冻结协议；不根据正式480题计时选优 |

这些事项属于后续实现的检查与决策清单，不要求为本文档另行发起批准流程。若审计导致评测范围或生成任务发生实质变化，应明确记录协议变化，而不是静默沿用旧实验名。

## 11. 参考来源

1. [LoopAhead：论文与附录模型表、执行参数](https://arxiv.org/html/2610.21594v1)。本次模型选择、revision前缀和LoopAhead读出设置的来源。
2. [LoopSpec：论文与附录](https://arxiv.org/html/2609.17184v1)。模型范围、Ouro/Raven循环深度及调度背景。
3. [LoopSpec README](https://github.com/kaist-flexml-lab/loopspec/blob/main/README.md)、[110-speed-math.yaml](https://github.com/kaist-flexml-lab/loopspec/blob/main/experiments/110-speed-math.yaml)、[download.py](https://github.com/kaist-flexml-lab/loopspec/blob/main/download.py)。实际支持检查点及读出配置。仓库既有同引擎调度参考固定为 `41aa710ed7c9ce908cffa03597be5d1df19d7b56`；若实施时改用其他版本，应记录差异。
4. [Ouro-1.4B 锁定版本](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)。
5. [Raven-Llama 锁定版本](https://huggingface.co/smcleish/Recurrent-Llama-3.2-train-recurrence-32/tree/a5f6f126e9d9302445346844f1d7d29d111a06eb)。
6. [Raven-OLMo 锁定版本](https://huggingface.co/smcleish/Recurrent-OLMo-2-0425-train-recurrence-32/tree/e1208e9af61d9e2fe1be95857e3490efb524294f)。
7. [Raven-TinyLlama 锁定版本](https://huggingface.co/smcleish/Recurrent-TinyLlama-3T-train-recurrence-32/tree/2579b6a0e9837dcea0f9fcaa85788f74d4a1f2b4)。
8. 本仓库 [论文实验规划](loopfill-paper-experiments.md)、[Huginn eager汇总](loopfill-eager-paper-summary.md)、[eager参数文件](../configs/loopfill-eager.args)、[LoopSpec状态机](../src/llm_memory_editability/loop_loopspec_decode.py)。
