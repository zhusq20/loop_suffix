# LoopAhead 复现规格（已批准配方与实现记录）

2026-11-10用户确认本规格默认做法，随后授权本任务后续自主实现、训练、检查点与测评。第7节列出的已批准默认值作为当前实现约定；若实际验证要求调整，记录依据与差异，不宣称作者实现如此。

依据：[arXiv:2610.21594v1](https://arxiv.org/html/2610.21594v1)，正文 3、4 节与附录 A–E。PDF/HTML 和下载哈希保存在 `../papers/`。2026-11-10 检查论文链接及公开检索，未找到作者发布的训练/推理仓库或 drafter 权重；这是检索结论，不是对不存在代码的证明。

## 1. 范围

论文模型为 Ouro-1.4B、Ouro-2.6B、Ouro-1.4B-Thinking 和三种 Raven（Llama/OLMo/TinyLlama），**没有 Huginn-3.5B**。Ouro 可做方法复现；Huginn 是迁移实现，须标注该差异，不能声称复现其原文数字。

需要两个独立协议：LoopFill 对齐协议用于交付表格；LoopAhead 原文协议用于 B4 对照。原文 GSM8K=1319、MATH500=500、HumanEval=164、MBPP=500、BBH=540，输出上限依次512/2048/1024/1024/1024；基本模型使用 completion、不自动加 BOS。LoopFill 的 300/500/164/378/270/480、统一 max512、chat 模板和 T1 top-p=1 与它不同。

## 2. LAD 结构与输入/输出

- 输入为同一候选分支上、完成 d1 轮且经过对应 readout 后的最近 w=64 个真实 target 隐状态（LM head 前），以及 native anchor 的冻结 token embedding。
- 位置 t 的状态来自输入 token x[t-1]，预测位置 t。native anchor 是该状态经原生 head 的预测 token，greedy 为 argmax，sampling 来自温度/top-p 后的 native 分布。训练 anchor 用 teacher y[t]，推理 anchor 不要求已经验证。
- context 映射到宽度 512；K 个 future query 由 anchor embedding 的投影、最近 context 状态和位置 embedding 构成。两层 causal Transformer，8 heads，FFN=1536，RMSNorm，SiLU-gated FFN。
- query 可读 context 与前面的 query 表征；所有 query 在 token 选择前构造。Ouro base LAD 一次并行产生所有 future hidden，不把上一预测 token 递归喂入下一 query。
- 输出投影回 target hidden size，并加到最新浅层状态：h_tilde[t,j] = h[t,d1] + D_j(context, embedding(anchor))。经冻结 target LM head 得 logits。
- 训练 K_train=7；Ouro 推理 K=3 个额外 token，不含 anchor；Raven-Llama/OLMo K=2，TinyLlama K=3。
- target/embedding/head 均冻结。论文报告 Ouro LAD 约11.1M参数；Raven-Llama base=11,051,008，R4残差头8,472,576，总19,523,584。Huginn 没有报告参数量，应实现后实数，不移用 Raven 数字。
- draft vocabulary：teacher 训练输出频率 top8192（同频按 token ID），加 prompt、EOS、已提交 token。LAD 两种温度都取 argmax；native 和 final readout 始终全词表。

## 3. 数据与优化

本次沿用原文的 teacher 生成方式：各目标模型分别以 full-depth greedy 重生成回答，目标权重冻结，不用数据集的原答案直接充当 teacher。下表为本次已冻结的实际数据配方，条数及样本ID以 `../data/manifests/manifest.json` 为准；三种目标模型共用问题清单，各自生成回答和浅层特征。

| 来源 | train | dev | confirmation |
|---|---:|---:|---:|
| GSM8K train | 2000 | 128 | 128 |
| Python CodeAlpaca | 2000 | 128 | 128 |
| MATH 原 train | 2000 | 128 | 128 |
| Nemotron V1 STEM 推理数据（替换原文 BBH 份额） | 2025 | 270 | 270 |
| UltraChat train_sft 首个 user turn | 1000 | 128 | 128 |
| 合计 | 9025 | 782 | 782 |

原文隔离方法：NFKC、大小写和空白归一后的 exact；保留标点/数学算子，token 5-gram Jaccard≥0.9 且共享≥8 个；BBH 忽略同子任务≥20%例子共有的5-gram。我们还必须覆盖完整六任务测试问题及 demonstrations，尤其 SpecBench/MT-Bench，全流程保存剔除原因和哈希。

**与原文的配方差异：原文使用 BBH 的自定义划分，未发布划分 ID，无法核实它与 LoopFill 测试题互斥。按用户已批准方案，本次保护全部官方 BBH 题目，将其 train/dev/confirmation 份额分别替换为2025/270/270条 Nemotron V1 STEM 推理问题。完整六任务测试源、示例及开发/确认隔离检查见 `../data/manifest-audit.json`；词面去重不能保证排除所有语义改写。此替换须在复现说明中披露。**

base：3 epochs，每个请求每 epoch 采32个 anchor；microbatch 8，accumulate 8，effective batch64；AdamW betas=(.9,.95)，wd=.01，grad clip1；seed42；前5% updates linear warmup至2e-4，cosine降至2e-5。目标为所有有效 future label 的 CE 均值，越界 label mask，不训练 target。

按已批准方案，“batch8”计 anchor window，则9025×32×3/64约13538 updates（本地采用ceil，尾batch从本epoch已打乱列表开头补齐），不是把9025个请求直接除64得到几百步。每500 optimizer updates与最后一次评dev，每请求固定8 anchors；以五领域等权平均 accepted-prefix length 选 checkpoint，CE打破平局。选定后 confirmation 只评一次。

Raven-Llama/OLMo 另有 R2 base冻结、在实际R4特征上训练的64维 token-conditioned residual head：V RMSNorm(hidden)+B embedding(previous token)，SiLU后U映射词表。500 train/80 dev requests、32 GSM8K train每题最多16在线窗口；25%online+75%offline；600 updates，micro4/effective32，peak5e-4；teacher-forcing/self-rollout loss均值，位置权重1/.75/.5，rollout第一错误后mask，每100updates选择。**Huginn是否使用这阶段无原文依据，须作为迁移决策。**

## 4. 推理与调度

Ouro：d1=1、d2=2、T=4；B_ret=32、B_exe=16、d_guard=2、stride=1。Raven-Llama/OLMo：4/8/32，K2，32/16，guard1，stride2；TinyLlama：2/8/32，K3，48/32，guard1，stride2。

每个节点存 token、parent、absolute position、完成深度、hidden、每层/轮KV、native proposal记录。相同(parent,token)合并，保留 native/LAD来源标记。

1. 从 root/prefix 出发，挑至多B_exe个未完成且依赖KV可用的节点。每个节点只看prefix、祖先、自身；所需祖先对应深度KV必须已存在，或在同一步先算好。
2. greedy 用父节点最深反馈判断冲突；sampling 候选不同于父节点所有已记录native token才冲突。任意祖先边冲突则整条路径冲突。
3. 距root不超过guard的节点免于冲突暂停；更远冲突路径保留KV/hidden/depth但暂停。反馈或root变化后重新检查；未获执行slot不等于删除。
4. Ouro按距root排序，sampling平局用开发集估计的path质量；Raven优先native路径，再距离。root不能饥饿。候选排序不得改变目标分布。
5. 每个step最多一次LAD调用，从刚到d1的节点的native子节点中选最近且尚无children的anchor，并要求有可用真实context window。输出链挂在anchor后；LAD节点未来也可到d1并触发递归draft。
6. Ouro greedy：有free retention slot则call，保留min(K,free)前缀。保留budget与执行budget不同，不能把ret32直接当每步执行32。
7. full-depth时greedy argmax／sampling拒绝校正决定唯一输出；匹配child变root保留该subtree及prefix KV，竞争子树释放。如果没有child则从depth0新建；新root若已满深可同step继续提交。EOS/长度上限逐token检查。

Raven特有：admission同时受free retention、spare execution、剩余输出预算（保留1位置）限制；动态门槛8→16或16→20/20→24，每次扩大间隔至少4已提交token。TinyLlama可剪除被深反馈否定且subtree无native路径的LAD枝。Huginn是否继承这些规则必须记录，不能默认与Raven相同。

Ouro sampling admission在16个dev prompts（8 GSM8K+8 CodeAlpaca）上校准：用offset/root距离估计p_j，draft后加4等宽confidence bins；最大化 U_call(l)=t_save Σp_j − t_head − l t_node；U_keep(k)=t_save Σp_hat_j − k t_node。非正选0，平局选短。原文A40的7.524/.602/.407ms不能直接当本机实测，需要本机dev重标定并冻结。

## 5. 温度1无损机制

遵循附录D，不用LAD argmax当全深度输出。d1记录u1~q1；到d2若q2_raw(u1)<q1(u1)，采第二native proposal自normalize(max(q2_raw−q1,0))并存真实分布。

满深得到p；按native记录顺序，以min(1,p(u)/q(u))接受，拒绝后p←normalize(max(p−q,0))，全部拒绝则从剩余p采样。LAD只提供能复用的候选节点，没有自己的额外接受概率；决策token可匹配任一来源的child。

概率处理FP32；温度和top-p先于残差。对零残差、重复proposal、proposal缺失、树裁剪、分支重排、bonus、EOS与cap做验收。构造小词表条件分布用大量样本检验频率，且用精确概率枚举检查残差分布。不以“同seed逐token相同”作为两个不同随机算法等分布的必要条件。

原文T1是top-p=.9、seed{0,42,271828}，只报告Ouro-1.4B/Raven-Llama。LoopFill表格改用top-p=1、seed0；本次B4另存的原文设置对照为Ouro-2.6B/HumanEval greedy，见下文已完成记录。原文T1三seed结果未复现，不能将LoopFill T1格子当作该组原文结果，两个协议不混表。

## 6. 可对照的原文数字

下表为原文表1的greedy throughput speedup，**不是本项目实验结果**。

| 模型 | GSM8K | MATH500 | HumanEval | MBPP | BBH |
|---|---:|---:|---:|---:|---:|
| Ouro-1.4B | 4.27 | 3.73 | 4.07 | 4.01 | 4.30 |
| Ouro-2.6B | 5.21 | 4.37 | 4.72 | 4.76 | 5.12 |

Ouro-1.4B sampling对应3.03/3.06/3.42/3.00/3.09（各自三seed均值）。最大7.82来自Raven-TinyLlama/HumanEval，不能用于Huginn对照。原文没有逐模型/逐任务γ表；正文4.3报告Ouro-1.4B五任务汇总 normalized loops/output=0.162，对应γ≈6.173（由已舍入数字推导，非原文精确实测γ）；LoopSpec为0.285。严格15% γ阈值需作者提供原始loops/tokens或图3底层数据，不能对不同题集直接套阈值。

## 7. 已批准的实现默认值与原文缺失细节

| 项目 | 建议默认值/依据 |
|---|---|
| Huginn缺原文配方 | d1=4/d2=8/T32、K2、ret32/exe16/guard1；先用直接R4 base LAD，是否需R2→R4 residual仅由独立dev决定；标迁移实验 |
| BBH数据冲突、split IDs缺失 | 保护整个BBH测试源；另选非测试reasoning数据替换2025/270/270份额；记录配方差异 |
| LAD投影bias、context位置编码、norm放置、dropout、输出投影初始化 | bias=False、pre-RMSNorm、dropout0、明确learned position编码；用论文约11.1M仅作量级核验，不能声称逐参数复现 |
| 短于64状态的context | 不凭空制造hidden；可用left mask填充但需确认是否论文其实延后call到64；默认不足64不draft，更接近“64 actual states”文字 |
| B_ret是否包含root | 默认包含root；每步验证计数不超budget；请作者核对 |
| native children满budget时优先级/平局 | root与native依赖优先，稳定按distance/creation ID；不随遍历顺序随机变化 |
| 中间readout语义 | Ouro最后norm/head；Huginn独立浅coda KV并保留对应depth真实状态，不用末层latent直接替代 |
| teacher生成长度、prompt模板未完全给出 | 训练completion cap512，沿域task模板且排除test demos泄漏；与当前benchmark模板差异记录；采集小批后再确定正式成本 |
| batch单位、采anchor重采/尾batch | 按anchor windows计microbatch，每epoch重新采32；不足长度mask；精确步数由manifest记录 |
| sampling校准数据与空bin | 固定dev16，平滑并回退到offset/root pooled估计，禁止从test拟合 |
| B4精确γ参考缺失 | 报自身γ及原文可比η；缺失值NA，不能标“已通过15%” |

后续实现只放独立baseline目录，复用冻结目标引擎接口，不改LoopFill/LoopSpec。首先Ouro-2.6B/GSM8K前20题greedy，随机或小训练LAD必须保持目标语义；保存第一次分歧位置及同prefix top2 margin，进一步FP32/同shape回放后才判断BF16误差。

## 已实现的具体约定

训练completion模板已冻结在manifest：GSM为Q/A、MATH为Problem/Solution、其余为Instruction/Response；输出cap512。这些并非作者公开的prompt文件。LAD dev遵循附录原文：计native anchor和最多3个预测的正确连续前缀；native anchor错误则长度0，CE仅作teacher-anchor tie-break。32个训练anchor/8个固定dev anchor在足够时无放回，否则有放回采样；必须有至少1个future label。没有64个真实状态的训练请求排除并报告；dev该类请求只做native-only评价，保留领域权重。

LAD参数已实算：Ouro11,051,008，Huginn17,673,376；结构默认见lad.py/configs。Ouro2.6B与Huginn均已完成3epoch/13539更新，按固定开发集分别选择13000/13500步，确认集仅评估一次。Ouro的sampling dev16校准已完成，Huginn采用声明过的固定迁移策略；Ouro1.4B的数据、训练与校准仍待附录阶段。

上述两个模型另经统一CPU验收，报告在`../training/loopahead-<model>-formal-audit-v2/report.json`：每个模型的543条实际优化日志、28份开发集记录与全部28份BF16检查点均核验；批准的LAD结构、训练独占词频8192、三个split的特征索引/ID、导出、实际训练退出与选中权重哈希一致。LR重算最大差4.07e-20。验收只检查已经记录的优化步，不补造未记录步；从原始领域指标重算选模，没有重新执行GPU开发/确认集推理。单份确认记录及训练控制流程支持选模后一次评估，不等于外部独立计数。Ouro1.4B的相同验收由独立CPU观察器在附录训练完成后执行。

Ouro随机LAD的GSM20检查为BF16整序列16/20、FP32为20/20，4个BF16共同前缀的FP32 top1一致。Huginn已训练LAD的GSM20 BF16为15/20，五个首次分歧的实际top2概率差已保存；FP32完整20题均与AR整序列一致，五个BF16共同前缀的FP32 top1也全部一致，gate证据见 `../preflight/loopahead-trained-huginn-gate.json`。这些诊断不保证全部正式BF16测评整序列一致；正式分歧另外逐项登记与检查。原始HumanEval164的B4对照已完成，BF1697/164一致且部分AR计时受CPU任务干扰；自身γ和原文可比η分开记录，原文缺少逐单元γ，不能标为通过15%阈值。详见preflight相应检查记录，不把开发集分数或这些独立诊断当作正式max512矩阵结果。

## 开发集成本测量实现补充

采样 admission 使用预先冻结的16条开发题，测试题不参与。先在关闭LAD的同一调度器上测量每物理loop周期的decode时间，作为`t_save`；另以同步计时记录实际LAD调用，取中位数为`t_head`。针对实际advance/readout调用，以行数、上下文位置及交互项做线性拟合，在开发题中位上下文处估计每新增候选行的耗时；`t_node=4×advance增量+3×readout增量`，负拟合增量截为零并保留原值。成本、原始调用记录及拟合残差全部保存。作者未公开校准实现，以上是本地明确默认值；不直接移植A40的毫秒数，也不把同步profiling日志用于正式η。

执行环境补充：为处理Exclusive_Process GPU在任务切换时被外部队列短暂占用，评测及诊断在加载模型前先限时取得自身CUDA上下文，仅等待用户仍授权的卡；该启动逻辑不进入decode计时、不改变LAD/调度/验证/采样。旧正式及B4日志的源码身份通过固定入口哈希兼容检查保留，其他算法源码仍逐字一致。
