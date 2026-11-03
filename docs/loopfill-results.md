# LoopFill 结果摘要

摘自原仓库 `docs/results.md` 中 LoopFill / Loop 投机解码条目（2026-11-02 快照）。完整数字见 `results/` 下各批次报告。

<a id="loop-v3c-ablation-20261101"></a>
**冻结设计 v3 C 的四栏消融均已完成 CPU 回放。** sb1024 chat 全量 480/477 题（Ouro/Huginn），包括用于选择配置的题。在 τ=0.01，只放第 1 名、加 K=16 上限、改为步末写入并省去接受后的立即续查、关闭更深读出，按块调用计的加速分别降 16%、25%、15%、13%（Ouro）和 27%、28%、11%、37%（Huginn）。两种模型在冻结的 d=1/4 时全量调用最少，60 道确认题的各改动同向。A→C 包含联合打分与剪枝两个变化。固定联合打分，只比较 B→C，每输出词的行数为 Ouro 14.6→9.7、Huginn 228.2→141.2，调用加速分别为 7.06→7.04 倍和 9.59→9.50 倍。行数为回放近似值，墙钟和读出成本另计。[报告§11](../results/loop-suffix-spec-20261030/report.md#11-冻结设计v3-c的消融11-01追加) · [Ouro 逐配置汇总](../results/loop-spec-prob-tree-20261031/sim/summary-v3-sb-chat-ouro-2.6b.json) · [Huginn 逐配置汇总](../results/loop-spec-prob-tree-20261031/sim/summary-v3-sb-chat-huginn-3.5b.json)

<a id="loopfill-engine-e2e"></a>
**Spec-Bench 数据覆盖。** 原始题库有 480 个题号，六类各 80 题，其中 80 道 MT-Bench 各有第二轮。以第一轮为单位，raw-256 和 chat-256 两个模型各保存 480 题的贪心输出与各深度读出。主设置 sb1024 chat 中 Ouro 保存 480 题，Huginn 保存 477 题，摘要题 253、288、295 因显存不足缺这批读出。CPU 回放预留同一组 60 个题号（每类 10 个），其余 420/417 题用于校准和选配置。[原始题库](https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl) · [设置与缺失记录](../results/loop-suffix-spec-wallclock-20261031/protocol.md) · [固定划分](../results/loop-spec-prob-tree-20261031/protocol.md)

<a id="loopfill-suffix-history"></a>
**Spec-Bench 的后缀索引。** 每个模型、每套回放配置从空缓存起，按题号升序处理。每题新建一棵本地树，先放入该模型的提示 token，再随解码追加全深度贪心已确定的输出 token。全局树按请求分开存输出 token，跨题和类别累积，历史提示随各自本地树释放。两棵树的索引深度上限为 64，历史请求上限为 100000，本批未触及容量上限。解码时拿当前前缀的末尾分别查询两棵树，选草稿得分较高的一份，平分取本地树。v3 C 还会在浅层刚提出的新孩子上查询。未经核对的草稿可作查询上下文，只有已确定的输出才写入后缀索引。候选是否进入本题草稿树另由校准表和路径阈值决定。[回放顺序](../scripts/loop_prob_tree_replay.py) · [解码器](../src/llm_memory_editability/loop_prob_decode.py) · [本批固定版本的缓存实现](https://github.com/snowflakedb/ArcticInference/blob/aca5d9a8a62474035c15d114d40a01abc8c94b51/arctic_inference/suffix_decoding/cache.py)

**S4 单请求测速复原相同历史。** 例如测第 89 题前，脚本先装入该模型第 81–88 题的已存贪心输出，包含未抽进测速集的题。预热和每臂每次试跑结束后撤回该次输出，各臂面对同一份既往历史。此前题目的历史装载在计时外，本题建树、查询和逐 token 更新在生成计时内。校准表用开发题拟合，后缀历史按实际题序增长，首题从空全局树开始，后面的题可借用前题输出。[S4 测速程序](../scripts/loop_suffix_spec_wallclock.py) · [运行协议](../results/loop-suffix-spec-wallclock-20261031/protocol.md)

**LoopFill 单请求引擎 E2E，事后汇总已有 S4 逐题计时。** 两个模型各实测 60 题，每类 10 题，每题每臂两次。Ouro 的计时题与预留题完全一致。Huginn 的测速程序在 77 道可用摘要题中按位置等距抽样，实测名单纳入了开发题 248、272、280，预留题 249、273、281 有读出但缺 S4 计时。下表保留两个模型各自原有的全部测量。[Huginn 实测题号](../results/loop-spec-prob-tree-20261031/wallclock/huginn-3.5b-bench-bfloat16-meta.json) · [测速抽样](../scripts/loop_spec_sibling.py) · [回放划分](../scripts/loop_prob_tree_replay.py)

对每题取两次 `wall_s` 的均值，再以基线时间之和除以 C 时间之和。计时从输入 token 进入生成函数至输出完成，含 prefill、解码和当前请求的树操作。模型已预热，分词、既往请求的后缀历史准备、服务排队和网络均在该计时范围之外。表中区间为按六类分层、配对重采样题目 4000 次的 95% 区间（种子 20261101）。

| 模型 | AR 平均秒/题 | 同引擎 WFD 平均秒/题 | C(0.01) 平均秒/题 | C(0.01) 相对 AR | C(0.01) 相对 WFD | C(0.1) 平均秒/题 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ouro-2.6B | 46.32 | 17.31 | 8.86 | 5.23× [4.49, 6.18] | 1.95× [1.76, 2.20] | 9.89 |
| Huginn-3.5B | 24.14 | 8.14 | 3.48 | 6.94× [5.26, 8.79] | 2.34× [2.16, 2.56] | 4.06 |

单取 Huginn 已实测的 **57 道预留题**，C(0.01) 相对 AR 为 6.98× [5.25, 8.76]，相对同引擎 WFD 为 2.35× [2.16, 2.57]。这 57 题含 7 道摘要题，其余五类各 10 题。

BF16 下各臂有不同的输出长度。按实际输出 token 总数除以总 `wall_s`，C(0.01) 相对 AR 为 5.59×/5.72×。第一次计时中，C(0.01) 与 AR 全文逐词相同的题为 36/60、42/60。仅取五臂第一次输出均相同的 28/37 题并只用该次计时，C(0.01) 相对同引擎 WFD 为 2.11×/2.12×。全量、同文本子集和原预设的每词 decode 指标是三个不同口径。逐题数据位置见 [S4 报告§9](../results/loop-spec-prob-tree-20261031/report.md#9-s4解码器与墙钟)，新增推理运行 0。

<a id="loop-suffix-schedule-20261031"></a>
**2026-10-31 loop给后缀树带来什么：缺口由浅层在d次而不是T次调用内补上，同等算力下每词调用少2.5–3.7倍，新增训练0。** 只用CPU回放已存的贪心输出与各深度读出。每个提交的词按“比前一个词晚几次块调用开始计算”记账：后缀树给出的词晚0次，浅层补上的晚d次，草稿树里没有、等全深度输出的晚T次。合用臂中后缀树给出的词与只用后缀树几乎相同（少0.4–2.2%），把每次补位改按T计价后调用数回到只用后缀树的水平（多0.5–2.4%），所以合用相对普通模型SuffixDecoding的收益全部来自补缺口。同一后缀树、同一按猜对概率放候选的规则下，普通调度（一步核对一次）与loop调度（prob-tree v3 C）相比：每词行数不多于普通调度最好点时，loop每词调用少2.5–3.7倍（sb1024 chat与raw-256，两模型）。τ=0.3时loop每词行数只比逐词多16–25%，调用数为逐词的0.11–0.20。普通调度的错草稿全部跑满T层才被删，loop读出时剪枝后Ouro 63%的白跑行在深度d内被删。行数为回放近似值，墙钟与多请求实测未做。[报告§10](../results/loop-suffix-spec-20261030/report.md#10-loop调度给后缀树带来什么10-31追加)、[协议§10](../results/loop-suffix-spec-20261030/protocol.md#10-追加loop调度给后缀树带来什么用户2026-10-31要求只用cpu回放)。

<a id="loop-spec-prob-tree-20261101"></a>
**2026-11-01 v3 C 单请求解码比 `both-path` 快 1.26–1.73 倍。** 本机 `sn4622129202`，五个臂同批重测，sb1024 chat，bf16。每个模型实测 60 题，Huginn 其中 57 题属于预留集，见[题号核对](#loopfill-engine-e2e)。相对 `both-path` 的解码时间之比及 95% CI 为 τ=0.01 时 Ouro 1.57 [1.45, 1.76]、Huginn 1.70 [1.36, 2.06]，τ=0.1 时 Ouro 1.26 [1.16, 1.34]、Huginn 1.73 [1.38, 2.13]。收益来自每词块调用数减少，Huginn 每次调用随行数变贵 18%，所以两个 τ 墙钟持平。τ=0.1 时每词行数为 `both-path` 的 0.44（Ouro）和 0.37（Huginn），τ=0.01 时 Huginn 行数反而多 12%。多请求吞吐只有按行数的估计，新增训练 0。[报告](../results/loop-spec-prob-tree-20261031/report.md) · [协议](../results/loop-spec-prob-tree-20261031/protocol.md)

<a id="loop-spec-wallclock-20261031"></a>
**2026-10-31 Loop投机解码墙钟测速完成：同一引擎下，后缀树+浅层读出比WFD快1.34–1.40倍，新增训练0。** Spec-Bench sb1024 chat 60题×2次，bf16单请求，GPU 6与CPU 24-31独占；两批主机监控均为60/60题无干扰。decode速度对AR：Ouro-2.6B（T=4、d=1）后缀树+浅层3.57–3.59倍，WFD调度在本引擎2.67倍；Huginn-3.5B（T=32、d=4）3.52–3.57倍对2.56倍。加速来自每词调用数下降（both-path对WFD：Ouro 1.02对1.38，Huginn 6.67对9.25），每次调用耗时相同（49.7对49.6 ms，6.35对6.39 ms）。本引擎中的WFD与原代码在fp32下逐词相同、调用数相同。WFD自带引擎每次调用更快（34.5对49.6 ms，5.3对6.4 ms），所以跨引擎比值为0.93–0.94倍（Ouro）和1.07–1.08倍（Huginn）。K=16草稿上限使只用浅层的速度为WFD的0.97／0.93倍。Huginn翻译为0.91–0.97倍，是唯一低于WFD的类别。WFD与其他臂分两个时段测；位置相同，fp32下AR每次调用耗时也相同（48.09 ms），但仍不如同题交替可靠。[同引擎报告](../results/loop-wfd-same-engine-20261031/report.md)、[首批测速与结果](../results/loop-suffix-spec-wallclock-20261031/protocol.md#wallclock-results)、[离线调用数结果](../results/loop-suffix-spec-20261030/report.md)；W&B组`loop-suffix-spec-wallclock-20261031`、`loop-wfd-same-engine-20261031-wfdonly`已同步并核验（[记录](../results/loop-wfd-same-engine-20261031/wandb-sync.json)）。

