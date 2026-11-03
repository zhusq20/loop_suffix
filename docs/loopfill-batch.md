# LoopFill batch 吞吐实验

投稿范围与必做数据以 [ICML 实验总计划](loopfill-icml-experiments.md) 为准；本页说明现有实现，不表示全部投稿实验接口已完成。

**2026-11-04：已新增预算版 `budget-raw`、`budget-online`、`budget-suffix`。使用 `--draft-node-budget` 配置 N；无需校准表。完整命令和本轮480题对比方案见 [在线控制器文档第11节](loopfill-online-suffix-control.md#11-已实现接口与下一轮比较方案)。下文旧示例保留用于 v3 阈值复现；`--max-nodes` 不等于 N。**

2026-11-03 v3 历史实现：主方法改用 raw，四表标记 legacy。八卡评分消融已完成320流，正式全量矩阵仍待执行；见 [执行协议](../results/loop-batch-20261103/protocol.md)。已有外部开发输入可用。raw 的 [树决策说明](loopfill-raw-tree.md) 与 [阈值登记](../results/loopfill-icml-20261103/raw-score-policy.json) 是旧阈值版复现的配置依据。

## 实现范围

- `loop_prob_decode.py` / `loop_tree_decode.py` 把模型操作暴露为可暂停步骤；原单请求入口仍执行同一套步骤。
- `loop_budget_decode.py` 提供固定 N、候选排序、缓存槽回收和可选逐请求在线 suffix 积分控制；batch 调度器直接调用同一状态机。
- `loop_batch_decode.py` 在每次循环推进前汇集所有活跃请求，拼接节点做共享投影、MLP、embedding/prelude 和读出。各请求拥有独立 KV、树、位置和停止条件。不是逐条调用完整 `generate()`。
- `--attention-backend per-request` 保留原 eager 实现；`packed` 使用所有请求共享的连续 KV 存储，各请求拥有互不重叠的视图。每层一次 varlen FlashAttention 处理全部请求的 committed prefix，一批带祖先 mask 的计算处理树节点，再按 log-sum-exp 权重合并两部分。不会跨请求或跨分支读取内容。不是 paged KV。
- `--prefill-chunk-size 128`：每次推进一个最多128词的 prompt 块，保留每层、每个循环深度的 KV；不同 prefill 请求轮转，块间插入活跃请求的共享 decode advance。Prefill 仍为 eager，不将多个请求的 prefill 块拼成一次前向，也不与 decode 行混在同一 attention 调用中。默认0保留整条prompt的旧路径。
- `--cuda-graph`：在 packed 后端捕获 decode 的 recurrent/prelude/coda/readout 操作。预热先统计形状，再捕获最多 `--max-cuda-graphs 24` 个常见形状；固定元数据缓冲区在重放前更新。计时期间不新增图，未覆盖形状走 eager 并计数。CPU 调度、树操作和输出提取仍在图外；这是同引擎实现，不是直接运行官方 LoopSpec 的 SGLang 后端。
- `--schedule continuous`：B为最大在场请求数，每次调度迭代补满空闲槽位；保持原输入顺序，不按输出长度重排。有限480条队列从开始即可用，尾部逐步排空。可与 chunked prefill、packed attention 和 CUDA Graph 同时使用，不模拟外部请求到达率。
- `--schedule fixed`：保留旧固定批次模式用于复现；完成的槽位等整批结束后补充。CLI兼容默认是fixed，本轮八卡主实验显式指定continuous，两个口径分开记录。
- 支持 `ar`、`wavefront`、`prob-low`（C，τ=0.01）、`prob-load`（C，τ=0.1），以及现有 `suffix` 实现。`wavefront` 为本引擎 WFD 调度，Huginn 沿用 shared-coda；`suffix` 为本引擎纯后缀草稿对照，并非官方框架性能复现。
- raw 显式指定 `--scoring raw --tau <冻结值>`，不加载校准表；simple/four 对照使用冻结外部64表。均不预装校准/开发输出。这里采用在线、完成后发布历史的协议，和旧 S4 提前恢复历史的单请求协议不同。

本轮优化后端对全部六臂统一启用：`--schedule continuous --prefill-chunk-size 128 --attention-backend packed --cuda-graph --max-cuda-graphs 24 --warmup-tokens 32`。预热取题序中前 `min(2B,480)` 条，每条最多32个新词，使用同样的分块及补位调度；统计形状后捕获常见图。预热历史与正式历史分离。正式108组的资源、输入、失败策略及代码哈希由[冻结协议](../results/loopfill-optimized480-20261104/protocol.json)登记，不重新探测或筛选batch。

## 输入和依赖

使用原 Spec-Bench JSONL 的第一轮 `turns[0]`，或提供包含 `question_id`、`prompt_token_ids` 的 JSONL。每条可指定 `max_new_tokens`，否则使用命令行预算。遇 EOS/回合结束符自然停止，不强制固定输出长度。原样保留 category，输入过长或重复 ID 会报错，不静默删除或截断题目。

需要本地模型权重、兼容模型代码的 torch/transformers 环境；后缀相关方法需要已安装 ArcticInference。与原实验保持同一模型 revision 和 ArcticInference `aca5d9a` 版本。只有 AR/WFD 时不导入 ArcticInference。

旧阈值版的 Ouro B=4 命令示例如下。路径需使用实际模型位置与兼容环境；正式运行须先完成协议中的容量/覆盖审计。这里使用已冻结 τ=.01，并只保留一个概率树入口；`prob-low` 的实际评分由 `--scoring raw` 指定。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/loop_batch_bench.py \
  --model ouro-2.6b --model-path /models/Ouro-2.6B \
  --questions results/loop-spec-reuse-20261030/raw/data/spec_bench_question.jsonl \
  --arms ar,wavefront,suffix,prob-low --scoring raw --tau 0.01 \
  --batch-sizes 4 --order-seeds 0,1,2 \
  --max-new-tokens 1024 --max-nodes 1024 --dtype bfloat16 \
  --out results/batch-ouro-raw-t001-b4-r0
```

Ouro B=1 改用 τ=.03；Huginn B=1/4 均用 τ=.03、相应模型名和路径。不同 τ 的 B 分开启动；计时重复另换输出目录，CLI 没有 `--repeat`。旧默认评分 `four` 和两个 `prob-*` 默认阈值仅为历史兼容保留，新计划不能省略 `--scoring`/`--tau`。raw 命令不传 `--calibration`；legacy 复现显式传 `--scoring four` 与外部表路径。

已有 reference/短检查结果可复用。按用户选择不重启长 validation32 FP32 检查；修改代码后只核验相关边界。Ouro BF16 分歧仍需定位。`max_seq=4096`，Huginn 还受 RoPE 表限制，超窗请求需在正式运行前登记共同可行集合。

每个请求槽使用预分配的 dense KV，包括循环深度槽和草稿节点。CLI 默认 8192 个节点沿用旧概率树预算；当前实验协议显式传 1024，B=1 开发题另做 8192 对照。脚本估算额外槽位显存并报错，不自动降低 batch 或树上限。所有方法使用同一个登记预算，报告 `cap_skips`；1024 不是原来无触顶工作点的等价实现。运行期间还需要 attention 临时空间，显存预检不是一定能完成的保证。

## 计时与历史

权重、槽位缓存分配、分词和 warmup 在计时外。每个方法、每个题序的计时从首批入场前开始，到所有请求完成并同步 GPU 为止，覆盖 prefill、树构建/查询/更新、合批、验证、解码和跨批调度。

`output_tput = output_tokens / wall_s`，统计实际生成 token，包含第一词和返回 token ID 序列中的终止 token；不含 prompt。不要平均逐请求 token/s，也不要累加并发请求的 `decode_s` 当作整批时间。逐请求 `prefill_s/decode_s` 是协作调度下的时间跨度，可能包含等待，不是独占 GPU 耗时。

`CompletionSuffixCache` 通过 ArcticInference 的公开 API 分离两个缓存：本地缓存立即接收已验证输出；共享历史只在请求完成时接收该请求完整输出。其他活跃请求看不到未完成请求的输出，历史不包含其他请求的 prompt。每个方法/题序从空缓存启动，warmup 使用独立缓存。日志记录发布顺序；调度顺序及 BF16 数值差异可能改变后续历史。

## 输出与复核

每个输出目录对应一次实验，已有 `metadata.json` 时拒绝覆盖。

- `metadata.json`：配置、git revision/dirty 状态、torch/CUDA/GPU、输入与校准文件 SHA256、后端设置。
- `<arm>-b<B>-seed<S>.jsonl`：请求 ID、prompt 长度、输出 token、实际长度、调用/行数、cap skips、入场/完成时间、服务延迟和 TTFT。
- 同名 `-summary.json`：整流 token/s、request/s、总耗时、p50/p95、合批 block 调用数/行数、峰值显存、题序和历史发布顺序；另记录 `schedule`、`refilled_requests`、时间加权 `average_active_requests` 和 `slot_utilization`。
- 优化后端另记录 `prefill_chunk_size/prefill_chunks/prefill_chunk_tokens`、`attention_policy/attention_batches`、`cuda_graph_shapes/cuda_graph_replays/cuda_graph_eager_misses/cuda_graph_capture_s`。图覆盖率可按 replays/(replays+eager_misses) 统计，只代表有计算的模型调用覆盖率，不代表 GPU 时间占比。
- `--check`：计时外逐条运行 AR，对比每条输出和首个分歧位置。FP32 出现分歧则保存日志后失败，BF16 记录一致率供分析；不会筛掉分歧请求再报主结果。
- 可选 `--profile` 导出 PyTorch trace；对应计时标记为 `profiled=true`，仅诊断使用，不作正式吞吐数字。

以相同题序配对比较不同方法，按整个流汇总波动。同一后端比较是本文实验的范围；这些数字不代表优化后的 SGLang/vLLM 服务吞吐。连续批处理已接入全部六臂；仍不模拟在线到达率，不报告外部排队时间或线上SLO。逐请求延迟从槽位入场算起。

## 测试

```bash
PYTHONPATH=src python -m pytest tests
```

新增测试使用小型随机模型检查两个实际 backbone 的 batch/serial 输出与调用数、B=1 对齐、长短请求、EOS、槽位复用、节点触顶、后缀历史隔离、四臂命令行产物，以及 CUDA/BF16 FlashAttention 路径。已有全尺寸开发消融与短检查只支撑其实际覆盖范围；预算版正式质量和全量性能按 v4 总计划执行。
