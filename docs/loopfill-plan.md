# LoopFill 推理解码实验规划

**当前权威计划：[LoopFill ICML 投稿实验定稿](loopfill-icml-experiments.md)**。已确定 E00–E09 的必做矩阵、数据划分、官方基线、质量/泛化验证、消融、统计与交付标准；现有 [机器可读矩阵](../results/loopfill-icml-20261103/experiment-matrix.json) 仍对应 v3 阈值版，预算版机器协议待实现后登记。本文保留文献依据和历史候选，早先“以后再做”的项目若在新总计划中标为必做，以新总计划为准。

摘自原仓库 `docs/loop-roadmap.md` §6（2026-11-02 快照），链接已改为本仓库路径。

**2026-11-04 v4 已确定：每请求未验证草稿节点预算 N 为主方案，raw 分数用于排序。** 规则见 [节点预算定稿](loopfill-raw-tree.md)，预算版尚待实现。下文保留历史方案与文献依据；旧阈值、1024 槽容量及 320 流评分结果不代表预算版配置或证据。正式安排以 v4 总计划为准。

以下保留历史设计讨论和文献依据，包括旧四表方案、旧阈值与连续批候选；不代表当前执行顺序。全部480题仍用于正式evaluation，历史设计参与需披露。

### 主文消融

| 论文维度 | 使用的比较 | 能回答的问题 |
| --- | --- | --- |
| 路径阈值 τ | v3 C，τ 为 0.3、0.1、0.03、0.01 | 同时看每输出词串行块调用与处理行数，展示延迟和计算量的取舍 |
| 草稿树如何生长 | 新候选取 top-1 或 top-8，另单独限制每步路径深度 K=16。更深读出能给已有分支改分，也能加孩子 | 哪些规则使正确词及早进树？K 限制路径长度，top-1 限制一次模型读出的新候选 |
| 草稿树如何更新 | A→B 引入两次读出的联合打分，B→C 单独打开剪枝。另比较后缀候选的两种校准表，以及接受即写入并从新根续查和步末写入 | 哪些规则减少错枝上的计算？两表均事前拟合，解码器内表取实际提出的候选，并区分竞争分支和草稿深度 |
| 何时读出 | Ouro 的 (d₁,d₂) 为 (1,2)、(2,3)、(3,无)，Huginn 为 (2,4)、(4,8)、(8,16)、(16,24)。每对使用各自在开发题上拟合的表 | 比较整套读出时机。冻结点为 Ouro (1,2)、Huginn (4,8) |

中间两行可放在同一幅「草稿树的生长与更新」图中。E 保留第一次读出并剪枝，D 第二次只降分和剪枝，A 第二次能加候选并重打分，C 用联合分数扩枝和剪枝。D→C 同时允许旧边升分和新增孩子，A→C 同时改变联合打分和剪枝。各策略可直接对比，单独归因剪枝用 B→C。旧解码器只作整体参照。四栏的 CPU 回放均已完成，全量为 Ouro 480 题、Huginn 477 题，分别含 420/417 道开发题，同一组 60 道题为原确认集。C(0.01) 在确认集上相对 AR 的串行步数加速为 6.02×/10.29×，可直接与单提案理想上限 T/d₁=4/8 比较。主图并列每词调用和行数，后者对错枝使用近似读出。[消融§11](../results/loop-suffix-spec-20261030/report.md#11-冻结设计v3-c的消融11-01追加) · [v3 配置](../scripts/loop_prob_tree_replay.py)

**优先新增的一项消融。** 在 C 上只关闭「模型读出新增孩子时，立即以这个尚未核对的孩子查询后缀索引」。保留从已验证根查询、接受即写入、模型补位、重打分和剪枝。这让两种来源仍可用，直接测浅层提案提前引出后缀串的增量。沿用两个模型与 τ=0.01 的已有逐题轨迹及 C 对照，只回放新变体。在原确认题上配对报告每词调用、后缀查询数、被接受的后缀词和估计行数。这是已分析过的基准上的机制对照。若另要量化第二次读出**新增孩子本身**的作用，可只关 `deep_expand`，允许旧边升降分，复用 C 和 D 作对照。[既有来源记账§10](../results/loop-suffix-spec-20261030/report.md#10-loop调度给后缀树带来什么10-31追加) · [续查与重打分实现](../src/llm_memory_editability/loop_prob_tree.py)

### 批处理测法的文献依据

| 工作 | 怎样组批、何时结束 | 具体测量 |
| --- | --- | --- |
| WFD，最接近 LoopFill 的架构 | Spec-Bench 主表为单用户、真实贪心生成，遇 EOS 或达到 512 个新词结束。附录 C 图 5 另扫用户批大小，固定批内每条序列独立推进，完成后空槽，一批全部完成才结束 | 主表用提交 token 总数除以含 prefill 和 decode 的墙钟。图 5 用随机 token 提示、1024 个输入 token、512 个输出 token，接受事件按实测总体概率独立抽样。模型前向和 KV 回退真实执行。[论文](https://arxiv.org/html/2609.23033v1) · [实验入口](https://raw.githubusercontent.com/summerbro-hhj/wavefront-decoding/434520932897fcc1e26898891c7e2feaa6ab099f/experiments/exp4_emu/run.py) · [批调度器](https://raw.githubusercontent.com/summerbro-hhj/wavefront-decoding/434520932897fcc1e26898891c7e2feaa6ab099f/src/decoding/scheduler_batch.py) |
| Su 等，The Synergy of Speculative Decoding and Batching | 在 Hugging Face Transformers 原型中把 1000 条真实提示分成固定批，批大小 1、2、4、8、16、32，每条生成 128 个新词 | 计整套提示完成的时间，含分词。动态流量是另一项实验，用相同的到达时间序列给各方法发请求，服务器合并队列中待处理的请求（最多 16 条），从客户端发送计到返回。[论文 §5](https://arxiv.org/html/2310.18813) |
| BASS | 改造 DeepSpeed，在同一提示上批量生成多条序列。每条可接受不同数目的草稿，比较补齐到最大长度的 attention 与按序列拆分的计算 | 报批内最先完成、最后完成和平均序列的每词延迟。各序列的延迟保留自己的耗时，另报整批吞吐。[论文 §3–4](https://aclanthology.org/2024.findings-acl.489.pdf) |
| MagicDec | 主结果用基于 GPT-Fast、FlashInfer 的自建后端，固定批大小和长上下文。论文 §5.1 的 PG-19 实验描述为 20 批，每条贪心生成 96 个 token | 公开 StreamingLLM 自投机脚本在 prefill 后同步并开始计时，按实际生成 token 总数除以耗时。该脚本最多读 10 批，最长一行达到其生成阈值或出现批级 EOS 条件时整批结束。复用结果时须区分这份脚本和论文的预算。[论文](https://openreview.net/pdf?id=CS2JWaziYr) · [测速脚本](https://raw.githubusercontent.com/Infini-AI-Lab/MagicDec/main/tests/StreamingLLM/selfspec_benchmark.py) |
| SpecInfer | 在 FlexFlow 上自建按迭代调度的服务。每次解码后返回已经完成的请求，允许新请求加入仍在运行的批 | 图 7 扫并发请求数 1、2、4、8、16，每题最多 128 个新词，比较每输出词延迟。这是采用自建运行时做连续批处理的实例。[论文 §5–6](https://arxiv.org/pdf/2305.09781) |

SuffixDecoding 的 Spec-Bench 对照使用修改过的 Transformers 脚本，单 GPU、批大小 1。其附录 A.1 的 SWE-Bench 配置使用 vLLM、TP=4、最多 16 个任务 worker；任务并发不能直接当作固定请求 batch。LoopAhead 的本地 PyTorch 评测每张卡同时一个请求，计时从 prefill 后开始。LoopSpec 的原评测使用 SGLang，其附录所报的有效批大小指活跃草稿分支。[SuffixDecoding 附录 A.1](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf) · [LoopAhead 附录 A.5](https://arxiv.org/html/2610.21594v1) · [LoopSpec §6 与附录 D](https://arxiv.org/html/2609.17184v1)

### 旧版端到端候选（已被上述固定 batch 协议替代）

以下为此前连续批处理方案，留作后续研究记录，当前不执行。当前主图使用固定批，并在同一机器、相同历史协议下重跑 B=1；旧 S4 只能单列。E2E 时间覆盖请求进入引擎、提示预填充、草稿与核对，直到返回最后一个 token。并发 N 表示同时处理的独立请求数。每个请求可以有多个草稿节点。[已有单请求计时与数据覆盖](loopfill-results.md#loopfill-engine-e2e)

1. **先跑通固定批，再加入连续补位。** 两种调度共享同一条计算路径。每次循环把不同请求的活跃位置合入同一次块计算，各题独立维护 KV、草稿树、接受长度、回退和 EOS。固定批同时入场，短题结束后空槽，最后一题结束再开下一批。连续批在槽空时补入下一题。两者都用真实提示、真实模型核对和自然停止。先以并发 1 对接原实现，再以并发 2 跑至少三道长短不同的开发题，核对补入、缓存隔离及 FP32 逐词输出。已有多请求微基准只执行一次块调用。[现有单请求解码器](../src/llm_memory_editability/loop_prob_decode.py) · [块调用微基准](../scripts/loop_prob_tree_multireq.py) · [按迭代调度的设计](https://www.usenix.org/conference/osdi22/presentation/yu)

2. **两个模型均在 N=4、16 下跑同一份全量请求流。** 保留六类各 80 题和原确认题号，使用原模型、完整提示、贪心、EOS/回合结束符和 1024 新词上限，从原题直接生成。Huginn 原来缺离线读出的三题也入队，记录实际完成数和任何显存失败。各臂使用同一张卡、同一显存预算、题序和预热，分别从相同的历史快照起步。快照排除本批答案，每题接受的 token 立即进入本题索引，题目完成后才向其他请求开放其输出。现用 ArcticInference 会把每个已确认 token 同时写入本地树和全局树。旧测速每次只有一个活跃请求。新批处理需要将本题更新与跨请求发布拆开，并以两条重叠请求检查可见时刻。固定校准表和 τ。N=16 时吞吐仍在上升且显存允许，再测 N=32。若论文计入完整多轮对话，80 个第二轮分别等待本方法自己的首轮回答，单列为 480 个会话共 560 个生成轮次。[题库与第二轮设置](../results/loop-suffix-spec-20261030/protocol.md) · [已测后缀历史的来源](loopfill-results.md#loopfill-suffix-history)

   **方法。** 两模型先完成 AR、同引擎 WFD、LoopFill C(0.01)、C(0.1)。在相同负载补纯 SuffixDecoding，Ouro-2.6B 再补双提案 LoopSpec (1,2)。只支持单请求的实现单列 N=1，跨引擎报告绝对时间及各自 AR。同引擎复现先核对官方实现的输出与调度。纯 SuffixDecoding 只用后缀草稿并做全深度核对，旧墙钟与最终 C 分属不同机器。Ouro 的双提案对同一个位置在第 1 轮先猜，第 2 轮有分歧时开备选分支，第 4 轮确定输出。已有 S4 单请求五臂按原环境引用，Huginn 实测的 60 题中有 57 道预留题。若补齐原单请求预留集，仅缺 249、273、281 的计时。[旧纯后缀与 WFD 墙钟](../results/loop-wfd-same-engine-20261031/report.md) · [SuffixDecoding](https://suffix-decoding.github.io/) · [LoopSpec 原文](https://arxiv.org/html/2609.17184v1) · [官方配置](https://github.com/kaist-flexml-lab/loopspec)

   **区分算法和补位的收益。** 连续批仍是主曲线。固定批只留一个配对负载，例如 N=16（受显存限制时预先选共同可行值），在相同题序上比较 AR、C(0.01)、C(0.1)。同一调度内比较 LoopFill 与 AR 的实际输出 token/s，同一方法跨调度则比较从固定批改成连续批的整体收益。请求完成顺序也会影响后续请求可见的后缀历史，两种调度均记录后缀命中。固定精度、显存预算、兼容的 attention 后端和编译／CUDA Graph 策略，记录实际 KV 布局、padding 和草稿行数。有专用优化时注明启用范围。

   **计时。** 模型预热后，从第一批请求提交到最后一题返回计整批用时，涵盖 prefill、调度、核对和解码，同时分列 decode 耗时。输出吞吐取实际返回的 token 总数除以整批用时，按题另记从提交到结束的延迟 p50/p95、首词时间、首词后平均每词时间和输出长度。固定 N、完成即补测的是给定并发的吞吐。若评价给定到达率下的排队延迟，才引入各方法共用的到达时间表，从实际到达开始计时。记录峰值显存、实际块调用与处理行数。至少三个事前固定的全量题序，各方法按相同题序配对，按整段流报告波动。预留 60 题的逐题延迟和一致率可从全量日志提取。若单独测这 60 题的整批吞吐，需独立跑同一份 60 题流。[既有微基准与负载估计](../results/loop-spec-prob-tree-20261031/report.md)
3. **GPU 消融只验证影响墙钟的两项机制。** 在同一批处理实现上固定 τ=0.01，取并发 1 和 16（显存受限时在预检中确定各臂均可运行的高负载点），配对比较 C、B（关剪枝）、E（关第二次读出）。C/B 测少算的行能否换来吞吐，C/E 测第二次读出的整体收益能否覆盖它的成本。复用主图中的 C 记录，并记录 LM head／Huginn coda 读出行数或耗时。新增实现先在小样本用 FP32 逐词对 AR，检查并发请求的隔离。BF16 以全量计时为主，另报逐题完全一致率和同文本子集。在首个分叉的共同前缀对照全深度 logits 和前两名分差，以区分数值边界与实现错误。单请求的重复计时在题内合并。
4. <a id="loopfill-code-edit"></a>**外部代码工作负载优先试单轮编辑。** 主结果仍是 Spec-Bench 全量多请求 E2E，已有 MT-Bench 二轮回放可用于说明上下文重复。若要再检验“已有代码能从提示中复制，改动处仍要由模型补词”，优先试 CanItEdit。它有 105 道 Python 编辑题，给修改前的程序和编辑指令，以隐藏测试判定输出。原论文中修改前代码平均约 43 行。这个设置减少了仓库搜索和工具调用的负担，两模型在该任务的实际通过率仍待测。[已有二轮对照](../results/loop-suffix-spec-20261030/report.md#5-mt-bench第二轮与第一轮配对同80题) · [CanItEdit 表 1](https://arxiv.org/html/2312.12450) · [题库与测试](https://github.com/nuprl/CanItEdit)

   **建议的可行性检查。** 先按官方三类编辑各预选 4 题，共 12 题。两模型用描述明确的指令和普通贪心 AR 各作答一次，只给修改前的代码和指令，按各自 tokenizer 核查上下文并给输出留足空间。检查能否生成可运行的修改、自然结束并通过隐藏测试，同时记录原样照抄、截断和重复输出。如果两模型能做出有效编辑，再在预先固定的其余题上比较 AR、纯 SuffixDecoding、WFD 和 LoopFill。保留全部正式题的速度和测试通过率，将 12 题的开发结果单列。

   <a id="loopfill-agenticsql"></a>**SQL 工作负载备选。** SuffixDecoding 的 AgenticSQL 是 Snowflake 内部多智能体流程。作者公开了阶段说明和解码实现，目前未找到可重放的原始请求。若获得按时间排列的阶段提示，可做请求级测速。要测原系统整题用时，还需其检索、数据库、纠错和合并流程。原论文附录另用公开 Spider 和 DAIL-SQL 提示测试分布变化，采用模拟验证器。[Snowflake 说明](https://www.snowflake.com/en/blog/engineering/suffixdecoding-arctic-inference-vllm/) · [论文附录 A.2](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf#page=19)

   如果选 SQL 作外部用例，先在 Spider 1.0 训练划分中预选的一小组题上用 Ouro 与 Huginn 各自的普通贪心 AR 检查提示长度、输出长度、可执行率和执行结果正确率。若要测多阶段复用，可在 Spider 上接公开的 MAC-SQL，用同一目标模型做表列选择、查询生成和基于执行反馈的修正。各臂共用目标模型、模板、重试上限和历史可见规则，参考 SQL 只用于评分。保留所有尝试，分列模型生成、数据库执行和整题耗时。结果以 Spider 或 MAC-SQL 的名称报告。这是与单轮代码编辑择需选择的候选，尚无本模型实测。[Spider 1.0](https://yale-lily.github.io/spider) · [MAC-SQL](https://github.com/wbbeyourself/MAC-SQL) · [执行评测](https://github.com/taoyds/test-suite-sql-eval)

   <a id="loopfill-swe-bench"></a>**SWE-bench Verified 的定位。** 完整 OpenHands 修复留作远期选项。我们尚无 Ouro-2.6B 或 Huginn-3.5B 在该设置的解决率。SuffixDecoding 的现场实验用专门微调的 32B 模型，报告 37.2% 解决率。其请求测速中部分上下文超过 8192 token，Huginn 当前窗口为 4096。Ouro 作者报告 4K 训练上下文、可扩展至 64K。能容纳提示以及能独立调用工具并修好 issue，需要分别验证。[原论文 §4.2–4.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf#page=7) · [Huginn 配置](https://huggingface.co/tomg-group-umd/huginn-0125/blob/main/config.json) · [Ouro 模型说明](https://huggingface.co/ByteDance/Ouro-2.6B)

   如需工具使用场景的补充证据，可从登记的 OpenHands 轨迹还原固定请求，让各模型真实生成。报告可容纳的 issue／请求覆盖率、请求延迟、输出一致性及后缀命中。当前只核对到轨迹索引，提示完整性和可容纳比例均待核实。若要论证“更快修好 issue”，先用 AR 在独立开发 issue 上验证有效修复，再决定是否做现场评测，同时报告工具耗时、超时、全部尝试的官方解决率和任务耗时。[Verified 题库](https://www.swebench.com/SWE-bench/guides/datasets/) · [OpenHands 轨迹登记](https://github.com/SWE-bench/experiments/blob/main/evaluation/verified/20241029_OpenHands-CodeAct-2.1-sonnet-20241022/metadata.yaml)

   若比较扩展到有训练的草稿器，再选取匹配检查点和可用权重的 LoopAhead。[LoopAhead 的模型和测量设置](https://arxiv.org/html/2610.21594v1)
