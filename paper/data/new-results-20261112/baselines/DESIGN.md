# DFlash / LoopAhead baseline 设计与偏差记录

状态：正文模型的正式实验进行中。Ouro2.6B/Huginn各50k DFlash教师均已完成并通过全量审计；Ouro2.6B DFlash的6epoch正式训练及独立CPU验收已完成，Huginn DFlash原宽版已按用户最新要求停止，保留4000步完整检查点；2048维缩窄版已完成2k试跑、独立开发集对照及从头6epoch/8886步正式训练，最终CPU验收通过，已接续同栈AR与正式评测队列。两个正文模型的LoopAhead LAD正式训练与统一CPU验收均已完成，部分正式测评已完成，实际单元状态以results.csv与formal-matrix-status.json为准；Ouro1.4B仍待附录阶段。Ouro的2k DFlash试跑、20题T0检查及其6处首次分歧FP32诊断均已完成；正式测评的全部greedy分歧诊断仍未完成，不能把试跑证据推广至所有正式单元。新增代码只在本目录，原LoopFill和其他baseline不变。

## DFlash 官方版本与适配边界

官方来源：[DFlash](https://github.com/z-lab/dflash)，[SpecForge](https://github.com/sgl-project/SpecForge)，[论文v2](https://arxiv.org/html/2602.06036v2)。仓库当前主分支已引入DFlash2与Transformers5，本任务复现2602.06036的DFlash，不能默默换成DFlash2。

已下载官方完整DFlash历史。审阅的论文时期版本为 `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`（2026-05-10，论文v2提交前最后commit），Transformers依赖4.57.1；发现时main为`07ebd93db9f472af339b644bb70221ad8428328a`。SpecForge主分支发现时为`b15ad386d7425acdfeead50275a8d0278214a1f9`，已获取至论文日期的历史，训练固定为`ea5a5f5902b3493b3b6045f249f5aa57cbd100a1`（2026-05-28），官方OnlineDFlashModel、DFlashDraftModel导入已验证。原引擎环境为torch2.6.0+cu126/transformers4.55.0，第三方实验必须使用独立venv，不升级共享环境。

官方README支持的Transformers目标列表不包含Ouro或Huginn。不能将支持trust_remote_code等同于已经支持循环KV和feature capture。优先用官方Transformers DFlash model；SGLang/vLLM没有已验证的这两个目标架构注册，不宣称能直接运行。

### 目标模型完整前向和缓存

- Ouro checkpoint的`OuroModel.forward`明确遍历`range(self.total_ut_steps)`与每层，故固定4轮的整块前向是可行的；其attention以轮次偏移维护独立KV。关闭early_exit_step/early_exit_threshold/use_weighted_exit。应断言每个token访问4×48（2.6B）或4×24（1.4B）个cache slot；回滚每个slot，并验证下一token与未推测前向一致。
- 官方DFlash推理用DynamicCache和`.crop(start)`。Ouro自带UniversalTransformerCache有自定义key/value列表，不能假定DynamicCache会始终被原地保留，或其替换对象已实现crop。适配器必须返回并持有target实际使用的cache，所有轮次一致截断，避免crop了一个未使用的旧对象。
- Ouro输出的`hidden_states`受HF记录装饰器与多轮重复调用影响，必须用实际执行计数/forward hooks采指定层输出，不能假定元组长度等于单轮层数。hooks只采所需层、同一call最多5份，不保留全192层的巨大激活。
- Huginn使用`transformer.wte`，不是官方DFlash硬编码的`target.model.embed_tokens`。需独立接口适配`get_input_embeddings`和LM head；`num_steps=32`必须显式传递，不能走默认随机轮数。
- Huginn forward支持input_states，可按绝对position传入与LoopFill一致的CPU seed0截断正态表；这样不必改target源码。coda、prelude、每轮core的cache及负索引须统一处理；原生Raven缓存结构与普通DynamicCache不同，先验证后选。
- 每次block verify所有位置完成全深；每个候选只能看到真实prefix和本块自身的左侧候选；被拒后完整清理其KV。相同AR加载/精度/固定深度/初始状态，用block_size=1或同target一步前向建立官方栈AR分母。

### 五个条件特征（已用hook核验）

默认按任务文档只取Ouro最后一轮：2.6B单轮0-based层`[1,12,23,34,45]`（展开执行索引`[145,156,167,178,189]`）；1.4B为`[1,6,11,16,21]`（`[73,78,83,88,93]`）。这与官方在非循环网络第二层到倒数第三层均匀选5层相对应，但不同于LoopAhead论文DFlash adaptation沿整个展开网络取样的方案，结果必须标记。

Huginn选择prelude末层、最后一轮core两个代表层、两个coda层，共5份；展开执行0-based索引`[1,126,128,130,131]`。模型拓扑2 prelude +32×4 core+2 coda。此为任务要求的解释，不是DFlash论文已有Huginn配方。feature均来自目标前向，不从未验证future泄漏给draft；只提交被接受前缀的features。

### 草稿器与训练

官方五层block diffusion，block16=1 clean anchor+15 masks，单步去噪；embedding/LM head与target复用冻结，5份target feature拼接→线性投影→RMSNorm→每层KV注入。块内双向注意力，各anchor块之间隔离，context features只能来自anchor之前。官方CE位置权重exp(-(k-1)/7)。6 epochs、AdamW、lr6e-4、clip1、warmup4%、cosine、max_seq3072、每sequence512随机anchors；batch等未给参数遵循所pin SpecForge recipe并完整保存。

Huginn target为hidden=5280、intermediate=17920、55heads×96。最初直接继承target宽度的五层草稿器实数为2,116,288,320参数；用户随后要求缩窄，当前正式训练采用5层/H2048/FF5632/16头及target宽度投影，共332,621,056参数，明确标为本地Huginn适配。原宽版中断记录和窄版验证过程见下文。mask token选已有合适token，不新增未训练embedding行。

本轮八卡续跑保留已经开始的Ouro1.4B小版训练：5层/H1536/FF4096/12头/block16、target H2048，三处宽度投影后为163,597,568个可训练参数。CPU meta-device实例化核验参数量，教师继续沿用已开始的同一50k生成任务；这项Ouro缩窄配置也必须标为本地适配，不能称作作者公开检查点或已完成训练。冻结训练来源位于`/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3`。独立后续队列`logs/ouro14-packed-evaluation-v1`等待完整6epoch、实际退出与训练审计，再运行FP32串行/批量兼容性检查、12组max512正式测速、各自新测AR与独立profile，以及6组greedy首次分歧检查。输入仍为原300/500/270/164/378/480题，不新增教师生成，也不以缩窄参数量推算速度。

续跑中的packed目标实现复用原完整轮数及缓存语义，DFlash草稿器和拒绝采样源代码单独冻结。已完成的HF-target格子保留原结果，新packed格子独立测自己的AR；最终逐行注明目标实现，不能将两种实现标成同一后端。详见EVAL_CONFIG.md的续跑补测协议。当前仍需等待实测与数值诊断，不能把排队或CPU预检视作性能、精度验收。

数据按用户规划50k混合数学/代码/对话（原DFlash论文约800k，规模缩小是明确差异），来源与替换详见下文；回答由各target重生成，数学/代码/对话配额在正式生成前冻结。已先完成Ouro2.6B的2k流程校验，之后完成两个正文模型各自的50k教师生成与全量审计。实际生成和训练成本见training_cost.md；尚未结束的作业不以试跑外推值代替实测。

### sampling与计时

论文时期官方DFlash实现的block草稿token为argmax，再逐位置从target分布采样作匹配；这是deterministic proposal的特殊情形，不应误称为对softmax draft q的标准拒绝采样。任务文档明确要求温度1标准拒绝采样：已将官方新版本的采q/接受min(1,p/q)/残差纠正逻辑移植为单独wrapper，保留其源码出处及差异。对扩散一次并行给出的分布，q_j必须是实际采样生成候选时的分布，不能在事后换q。

官方旧实现会在第一次draft调用后重置decode_start，并将prefill首token计入num_output_tokens；还可能在上限外多验证一块后再切输出。为对齐LoopFill，独立benchmark从prefill后开始、包括首次draft；以实际提交decode token计算γ和tok/s，并裁剪最后block到剩余输出预算。AR和DFlash计时边界相同。

在summary另记draft GPU时间（CUDA events，在测量后统一同步）、完整wall时间、verify次数、实际提交数和token IDs。单独instrumented profiling run用于草稿时间占比；逐题核对输出和验证次数与主测速相同，再报告两次wall时间差。该差值包含运行波动，不能称为纯events开销；不在每个阶段强制同步改变主η测量。

## LoopAhead

实现规范见`loopahead/SPEC.md`，默认值已获用户确认。需新增独立LAD、scheduler和method runner；复用PackedLoopExecutor与KV存储，不能为了注册新method修改旧baseline逻辑。Huginn明确是迁移，BBH训练数据与用户隔离要求有冲突，必须按SPEC处理。

## 验收

1. CPU小词表拒绝采样/残差概率守恒，树暂停恢复不会改变最终target决策。
2. 目标全forward与block+crop后forward在固定prefix上比对logits、KV长度、feature形状和完整loop调用计数。
3. Ouro2.6/GSM8K前20题T0全token对比；第一分歧在共同prefix重跑，记录top1/top2 logits/probs及margin；FP32/固定shape检查后才归因。
4. Huginn相同绝对位置噪声及32轮断言。重复完全相同方法也先验证可重复性。
5. T1小模型分布检验与真实模型流程检查；同seed输出差异和统计等分布分开记录。
6. 去重报告未通过、正式配方未确认、B4复现差距未解释时不生成成功状态的论文格子。
7. 每个正式DFlash训练完成后，用训练时的SpecForge环境在CPU验收：Ouro运行`audit_dflash_training.py --model <model> --out <fresh-directory>`，Huginn窄版运行`audit_projected_dflash_training.py --scope formal --out <fresh-directory>`。检查完整6epoch的逐步日志、固定版本官方学习率调度重放、最终BF16权重的全量有限值/参数量、模型配置、冻结源码和教师文件哈希。该检查拒绝未完成训练；优化器文件只核对存在与大小、各rank RNG文件核对哈希，不声称已经做过优化器恢复测试。CPU验收不计入GPU训练或benchmark时间。
   自动执行由`orchestrate_training_audits.py --execute`负责：等待实际torchrun退出0、完整6epoch及导出文件逐字一致，再调用CPU验收；已有报告与当前最终权重/日志哈希重新绑定后复用。队列固定CPU80–83、CUDA不可见，独立于训练和推理控制器；任一验收失败保留错误并停止该验收队列。最终须检查当前窄版适配后的`logs/dflash-training-audits-v2/complete.json`及其中三个模型的真实验收报告，启动成功不能作为训练验收已通过的证据。
8. 首次分歧诊断覆盖36个正式T0单元，以及单独的B4原始HumanEval164对照。后者97/164完整序列相同，67处首次分歧必须全部记录实际BF16两侧top2概率差并完成FP32共同前缀核对，不能用早停探针代替。`orchestrate_greedy_diagnostics.py`在72格完成后安排这些检查；B4保留completion/max1024/EOS0/AR物理节点1/LoopAhead物理节点32，结果放在`preflight/loopahead-B4-greedy-margins-v1`，不增加正式矩阵格数或改变原始测速。另安排Huginn开发集宽/窄两组共15处分歧，保持其独立20题/max128协议，见下文。B4旧版评测入口只接受已审阅的固定哈希；其他engine/decode/LAD源码须与归档逐字一致。CPU计划验收见`preflight/loopahead-B4-diagnostic-plan-audit-v1/report.json`，不能据此宣称GPU诊断已完成。
9. LoopAhead三个模型统一运行`audit_loopahead_training.py`：核对批准结构与target宽度、3epoch完整更新数、实际日志频率下的有限值/LR、全部保存权重的结构与有限值、训练词频8192、三个数据split的特征索引覆盖、固定开发集选模、确认集记录、实际退出0及导出。Ouro2.6B/Huginn的v2验收已通过，各13539更新、543条优化日志、28份检查点；不把稀疏日志称为逐步全量日志，也不以CPU重算选模代替GPU重跑。`orchestrate_loopahead_training_audits.py`仅使用CPU80–83、CUDA不可见，复用并绑定这两份验收，等待Ouro1.4B真正退出及导出后检查第三份；最终完成凭证为`logs/loopahead-training-audits-v1/complete.json`，当前第三模型尚未完成。该验收不证明benchmark无损或速度。

## 已落实的适配与数据来源

原生完整前向/cache探针、随机官方drafter的短文本T0/T1流程、固定shape回滚重算均通过。拒绝采样/预算/EOS共6项单测通过。Ouro不同长度左padding批量生成在实际remote模型类的24层×4轮小宽度FP32实例上，对未padding逐样本前向、features及3个缓存步均通过；Huginn只按相同prompt长度组batch。上述检查不替代训练后20题逐token验收。

DFlash mask使用现有冻结行：Ouro token0，Huginn token65513（fim_pad），不增添随机embedding。五层草稿器实际参数量Ouro277,898,496；Huginn最初按target宽度直接适配为2,116,288,320，现正式配置改为带投影的332,621,056参数窄版。两者均为DFlash草稿器，不能误报成LoopAhead的LAD。

Nemotron V2受访问限制，现有授权无法下载。改用官方公开V1固定revision，V1代码题被删成占位符的部分经TACO train原始索引恢复，22,638个来源metadata逐条对应核验；聊天题缺失，使用UltraChat train_sft替代。DFlash配额50,000：Nemotron数学20k、恢复的Nemotron/TACO代码10k、CodeAlpaca5k、UltraChat15k。2k试跑为其中分层子集。此版本、规模和对话来源与DFlash原配方不同，须在最终表注说明。

LoopAhead训练9,025条，dev/confirmation各782条；原文BBH训练份额按已批准方案由非测试Nemotron STEM替换。三套tokenizer确认prompt≤2560，输出上限512，不截断；LA使用固定completion模板，DF沿用released chat模板。原始来源与选择细节见data/sources.jsonl、data/manifests/manifest.json。

隔离保护10,392个规范化问题（包括全部BBH、GSM8K测试、MATH500、EvalPlus、SpecBench各轮及demo、原始MBPP），用规范化相等、字面包含和5-gram Jaccard≥0.9过滤。独立audit核对清单hash、配额、样本唯一性、dev/confirmation跨方法隔离及61份冻结benchmark文件hash，全部通过（data/manifest-audit.json）。词面去重不能证明不存在语义改写，保留此限制。

## A2试跑与当前数值诊断

Ouro2.6B：2,000条教师回答全部生成，按官方≥32 loss-token过滤后1,923条，1epoch/61更新。生成占卡1.0260 GPU小时、训练0.1082 GPU小时。训练使用原样SpecForge OnlineDFlashModel/DFlashDraftModel/BF16Optimizer，DDP替代官方launcher的FSDP/SGLang，2卡×micro4×acc4=global32；FP32 master AdamW(.9,.999)、wd0，512 anchors/sequence，6e-4、warmup4%、clip1、cosine。保存master/optimizer状态，原target均冻结。正式配方仍为6epochs。

20题T0完整序列一致14/20。6个第一分歧均在相同prompt/已接受前缀及实际块形状上重现；BF16 top2概率差与logits详见preflight/dflash-A2-pilot.json。FP32下6个位置top1全部一致、logit最大差≤6.87e-5，支持BF16形状相关数值差异；不是20题完整FP32逐token证明。试跑γ1.0282、η_latency0.9289、η_throughput0.9139，仅为1epoch流程试跑，不进入正式表。已训练checkpoint的20题T1也完成；CPU拒绝采样/EOS/cap检查通过。

T1 AR保留冻结的request/absolute-position uniforms；DFlash采用真实softmax-q及标准拒绝校正，request seed相同、prefill首token使用冻结target sampler。拒绝校正有额外随机数，不能要求它与target-only AR逐token相同。当前实现用u<min(1,p/q)，零残差的被拒情况显式报错，不静默换回未校正分布。

## LoopAhead实现状态

LAD和retained-tree调度已实现，包括32保留/16执行、祖先KV就绪检查、冲突暂停、stride、原生双proposal、满深cascade、每step一次LAD。原文Algorithm2的顺序为advance/readout→native children→cascade reroot→LAD；已按此顺序实现。Huginn原生路径优先、guard1、stride2，直接R4 base迁移，无未经验证的残差头。原文未给的budget平局采用稳定creation ID；native可驱逐最远的LAD-only或冲突叶子，此为本地明确默认。

CPU oracle已覆盖误导草稿、冲突、祖先KV、retention/execution不同预算、EOS/cap与cascade。真实随机LAD短文本：Ouro BF16可重复但与AR有分歧，完整FP32 64token运行与AR一致；Huginn BF16短文本与AR一致且可重复。两者均实际调用了LAD且T1流程完成；未替代训练后的GSM8K20题验收。

训练feature提取与LAD优化已在Ouro完整9025/782/782数据上完成：3epoch、13539更新、开发集选择13000步。按论文附录dev指标，accepted-prefix包括native anchor及最多3个future token；native shallow argmax不匹配teacher则长度0。CE tie-break使用teacher-anchor监督。不足64真实状态的请求不造padding；训练无可用anchor则排除并记录，dev只计算native-only长度0/1并保留在领域均值中。本次Ouro所有请求都有可用anchor。确认集仅在选定后评一次。

Huginn的9025/782/782回答与全部24个特征分片也已完成并独立审计；正式LAD训练同为3epoch、13539更新，开发集选择13500步，确认集仅评估选定权重一次。开发/确认的平均接受前缀为1.993791956/1.974241898，按原文指标计native anchor和最多3个future，不是benchmark γ；Huginn正式推理采用已批准的K2迁移设置。数据证据见training/loopahead-huginn-3.5b-data-v1，训练与权重选择记录见training/loopahead-huginn-3.5b-formal-v1。

实际资源调度与任务文档的模型顺序有差异：两种方法均在Ouro2.6B数据先就绪后先启动其正式训练，同时继续Huginn较慢的32轮教师生成。Ouro LAD先于Huginn LAD完成；Ouro DFlash也先启动2卡训练，Huginn完整数据通过审计且4卡可用后再启动4卡训练。正文两模型仍优先，Ouro1.4B后置；此安排利用空闲设备，不改变各自数据、训练配方或测试隔离。

## Huginn生产缓存显存

首次Huginn生成使用32 batch/逻辑KV上限70GiB时，原生字典KV的tensor views与stack临时存储触发OOM。失败输出和0.0448 GPU小时成本保留在data/teachers/failed/huginn-loopahead-train-kv70-shard0。原生缓存作业改为batch≤16、逻辑KV上限32GiB、expandable_segments，重新冻结batch计划后运行，未复用失败计划的一条回答。

后续生产专用预分配缓存通过真实模型16样本×128解码步验证：全部logits逐位相同、2048/2048 top1相同、132个缓存槽及长度一致。该合成探针峰值分配从23.51GB降为13.95GB、单次探针墙钟18.93秒降为9.71秒；不是benchmark加速结果。新教师任务显式指定proof并校验实现哈希才启用，已有运行不被替换。Huginn DFlash50k新计划使用batch≤32/KV≤70GiB；LoopAhead剩余shard继续原batch≤16/KV≤32GiB计划以保持样本分配。正式DFlash测速仍使用原生缓存。

Huginn浅层feature与冻结推理接口在相同shape下BF16/FP32完全一致，改动未来token不影响此前feature。右padding改变批形后BF16最大feature差0.33203125、FP32为3.9339e-5；训练保留实际批量计算的浅层状态，明确记录这种数值形状差异。
# 教师生产Graph路径（2026-11-10补充）

`teacher_graph.py`复用冻结仓库的PackedLoopExecutor，全深度AR、独立请求KV、同一Huginn固定噪声；仅训练回答生成使用。正式benchmark保持Graph OFF。原有分片继续其原生/预分配缓存路径，之后启动的Huginn/Ouro2.6分片由`configs/teacher-graph-adoption-v1.json`显式启用Graph。清单、各shard的题目归属和512输出上限不变，每片metadata记录后端与proof哈希；训练数据因此可能混合两个已记录的生产后端。首个Graph分片保留原批次，之后新启动的Graph分片将各自已分配的题目按长度重新合批，支持不同长度的独立请求；全局assignment哈希仍固定，实际execution分组及其独立哈希另存于batch-plans.json，不能将它们描述为相同执行批形状。

在8条非测试训练题、五领域、不同8–64输出上限（覆盖活动batch缩小）的实模检查中，两模型的BF16/FP32 Graph开关输出均8/8一致；FP32也均8/8匹配原生HF串行目标。重复Graph回放一致。证据在`preflight/teacher-packed-graph-v1`，生成器核验其完成状态及目标适配/Graph源码哈希后才允许使用。Huginn BF16短批次10.466秒→4.766秒，Ouro10.987秒→3.943秒；此为生产兼容性实测，不是正式η或全数据速度估计。

限制：64-token探针不能证明所有输入上的逐位等价，BF16原生与packed算术不承诺逐位一致；全深度目标语义和数值形状差异需区分。分片的原始回答、源码及配置均保留，不能将混合后端生成描述为同一BF16执行形状。

Huginn正式DFlash训练的microbatch已先实测再确定：在完整50k教师集上选择最长32条有效样本做一次更新的显存检查，从官方microbatch4开始，预设仅显存不足或峰值超过120GiB时依次尝试2和1，其他错误须调查。实际microbatch4一次通过，峰值58,022,046,720字节，未降低microbatch；正式配置为world4×micro4×acc2=global32、gradient checkpointing、6epoch。120GiB阈值为本机约140GiB可用显存预留DDP及分配器空间。正式权重从头初始化，探针权重不用于正式结果。实测选择见training/huginn-dflash-microbatch-selection.json，探针元数据及成本见training/dflash-huginn-memory-micro4-v1；正式训练仍运行中，其最终成本与完整验收另行记录。


## 用户调整：四卡运行与Huginn窄版适配

用户要求暂时只用4张GPU，并重新考虑Huginn的2,116,288,320参数草稿器。已释放本任务在GPU0/1/5/7上的进程，保留GPU2/3/4/6；`configs/gpu-allocation.json`是主调度、附录和诊断的共同白名单。附录仍保留原8个数据分片ID，四卡分批处理，不改变数据划分。窄版正式训练及CPU验收已完成，GPU3/6的临时训练预留已解除，四张卡现均用于正式测评。旧全卡预留守护已停止，当前守护仅在用户授权的四卡上处理已明确识别的Weather占用，不操作交还的卡或未知进程。

Huginn宽版训练已执行至4245步，最新完整检查点为4000步（权重、优化器/master和4个rank RNG均保留，未实测恢复后的更新）；4卡共消耗22.947958704 GPU小时。此次为用户资源与架构调整导致的主动中断，不是数值训练失败，也不算完成6epoch。原始进程退出和成本日志保留其真实记录，解释及释放核验见`logs/four-gpu-reallocation-v1/`。旧训练自动CPU验收观察器已停止，现由识别窄版身份的v2观察器接续，不能将队列启动视为Huginn验收完成。

已采用的窄版为5层、内部H=2048、FFN=5632、16个Q/KV头、head_dim=128、block16。`dflash_impl/projected_draft.py`保留官方DFlash/SpecForge层和loss，增加冻结target embedding后的5280→2048投影、将五份特征的26400→2048融合层，以及2048→5280输出投影，继续使用冻结的target LM head。实数参数332,621,056，比宽版减少84.282810%；1024/3072/8头候选实数106,018,048，目前只核对参数，不作为优先训练方案。LoopAhead论文图4给出Ouro DFlash277.9M、Raven-Llama DFlash325.1M；窄版处于相同量级，但原文没有Huginn配置，须标为我们的Huginn宽度适配。

CPU检查`preflight/huginn-narrow-dflash-cpu-v1/report.json`通过：官方训练loss及梯度可通过三个投影和transformer块传播，外部embedding/head保持冻结，训练/推理层FP32输出完全相同，checkpoint跨训练/推理加载逐值一致，KV crop后重算最大差2.98e-8。该检查使用小型随机模型；完整候选参数由meta模型逐项实数，不是训练或加速结果。后续真实Huginn GPU前后向/内存探针、小规模训练与独立开发集接受长度/墙钟速度比较已完成，过程与限制见下文。已有50k教师回答可复用；新形状不能直接载入宽版草稿权重，须单独训练并记录新增成本。缩小草稿不会消除32轮target特征重算和词表head成本，不预估等比例训练或推理提速。


窄版真实GPU兼容性探针现已通过（`preflight/huginn-narrow-dflash-gpu-v2`）：冻结完整Huginn32轮与官方loss/optimizer进行一次更新，三个投影均有有效梯度，部署类权重兼容，短合成提示的greedy AR/重复一致及sampling流程完成。micro1短例峰值15.236GB不是正式显存结论。独立窄版trainer保留宽版代码与4000步检查点原样；该阶段队列先检查最长32源例的micro4显存，再以固定pilot2000中1893个有效回答训练1epoch/60步，global32；后续开发比较及完整训练记录见下文。不把小规模试训计入72个正式单元。

窄版2k试训已完成并通过独立CPU验收（training/dflash-huginn-narrow-pilot-audit-v1）：1893有效例/60更新、332,621,056个有限BF16参数、全部60步官方LR重放无差异，最终权重SHA256 f7cd716a686b7d887d89cbe5f3261cc58b9414aedeb9cfdfd7a8aa0cbdc0222c。它与旧宽版4000步在正式推理软件torch2.6.0+cu126/transformers4.57.1中均无缺失或错误加载。开发集比较固定20条分层问题（数学8、Nemotron代码4、CodeAlpaca2、对话6），按seed20261111与source_id的SHA256选取，仍属于原隔离dev1000；与两个方法的训练清单及pilot的ID/规范化问题哈希无重叠。每题循环AR/窄/宽顺序，用同一GPU、max128、Graph OFF；另跑窄/宽profile并逐token对齐未插桩输出。计划哈希79d093f0d4e1a0e417b30e5841d5436d4854746c38f50f2e118aacc64a58f01c，路径preflight/huginn-dflash-dev-plan-v1。该短开发集不是max512正式benchmark，且两检查点训练量不同，不能据此将接受率差异归因于容量。

独立orchestrate_projected_huginn.py队列先等待开发集实际退出0、所有20题完整且profile输出一致、pilot验收及权重哈希绑定通过，再等待GPU3/6现有任务自然结束。随后先做world2×micro4×acc4=global32的DDP显存/单步检查，通过后从头训练正式50k教师/47387有效例、6epoch/8886更新。独立目录checkpoints/dflash-huginn-narrow-formal-v1；旧宽版和pilot均保留，权重不续接。完整训练用于判断窄版最终效果，不把少量pilot的速度当作相同训练预算的架构胜负。两卡临时调度预留写入gpu-allocation.json，GPU2/4继续正式评测；用户的总4卡白名单不变。正式训练结束后导出、CPU验收并释放临时调度预留，评测入口/collector/最终成本还须绑定实际选定检查点，不能仅凭队列启动记为完成。

正式路径现由dflash_training_registry.py统一绑定：Huginn指向独立narrow-formal-v1及narrow-formal-audit-v1，Ouro两模型仍保持原路径。evaluate_projected_dflash.py是独立入口，复用相同target/decode、原题集与采样/计时函数；Ouro入口仅有下文明确记录的CUDA启动修复，测评算法与计时函数保持不变。投影版仅在完整8886步/6epoch、实际torchrun退出0、导出一致、CPU权重与日志审计、队列完成标记互相绑定后可进入正式评测。主控制器将这份证据单独记为external-narrow-completion，保留旧宽版失败worker记录，不重写为成功。collector、数值诊断、最终成本及CPU验收观察器均识别窄版；旧宽版22.947959 GPU小时单列额外中断成本。

CPU路由核验preflight/projected-formal-routing-check-v1通过：未完成窄版和真实2k pilot不能通过正式门槛，去除调度defer也不会重启宽版训练；新旧runner的run_one采样/计时AST一致，Ouro已完成run的原评测/target/decode源码逐字未变；原Ouro完整训练退出/导出/独立审计仍可校验，CUDA未初始化。主控制器已重载并接续原worker，当前CPU验收观察器目录为logs/dflash-training-audits-v2，在独立窄版队列完成验收后复用，不与训练队列争写报告；现已复用最终窄版报告并等待附录Ouro1.4B。原v1目录作为历史证据保留。

开发集20条/max128对照现已完成并独立核验：窄版pilot60步η_latency0.834691、η_throughput0.851594、γ1.065227；宽版4000步2.946173/3.005833/3.671318，AR10.649274 token/s。该60步窄版pilot尚未有加速，且训练预算不等，不能将此差异全部归因于容量。宽版draft events占profile decode约3.194683%，只说明计入这些events的草稿GPU工作占比较小；其余时间包含target计算和CPU调度等，未逐项分解，不能据此确定主要瓶颈，也不能从84%减参推出同比总提速。窄/宽各20条profile输出完全复现未插桩运行；BF16与AR分别12/20和13/20一致，15个首次分歧已存档但未独立FP32诊断。完整报告见preflight/huginn-dflash-dev-compare-v1/summary.md。该诊断不进入正式表，正式模型按完整配方训练后仍按max512评测。

这15处分歧现由独立`diagnose_huginn_dflash_dev.py`纳入矩阵后的收尾队列：窄版8例、宽版7例，原20条输入、seed0、max128、两个原检查点和共同前缀均固定；不能复用正式max512预算，两个版本均有首次分歧在位置127的样例。实际BF16草稿trace须逐token复现原前缀，AR路径也逐token核对，随后记录BF16两侧top2概率差及FP32共同前缀结果。保留原双drafter常驻及预热次序，诊断时间不计入原速度。结果目录为`preflight/huginn-dflash-dev-greedy-{narrow,wide}-v1`，严格验证真实退出0与来源哈希。CPU输入/检查点验证及模拟调度通过（`preflight/huginn-dev-diagnostic-plan-v1`）；尚未执行GPU重放，不提前将分歧归因为数值误差。收尾总共36个正式单元、1个B4对照和2组开发检查，正式矩阵仍为72格。

正式启动时的记录：双卡显存/DDP检查通过后，独立窄版torchrun PID65439及rank65472/65473在GPU3/6实际启动，47387例/6epoch/8886更新、world2×micro4×acc4=global32。前12步有限指标、实际进程命令/ticks/CPU绑定和CUDA上下文通过核验，证据preflight/huginn-narrow-formal-launch-v1。该启动证据只证明当时开始训练；随后实际完成的训练、最终权重验收和正式测速证据见下文，不以启动记录代替完成证明。


正式DFlash入口增加启动争用处理：在用户仍授权的独占GPU上先取得CUDA上下文，再做慢速模型加载；只对busy/unavailable等待，未知错误立即失败，等待上限60秒，启动时间不进入decode计时。后续系统动态加载器开始优先解析不匹配的compat驱动，另增加进程局部原生驱动库环境及必要时的入口re-exec；不修改系统驱动/缓存。两次入口修复均只改main，所有非main AST不变，dflash_source_identity.py仅接受已审阅的两份旧入口到当前版本的固定哈希，归档源码和target/decode仍精确匹配。旧宽版Huginn与窄版的训练/推理选择不变。争用和驱动失配分别造成两次、六次无请求启动失败，原始证据分别保留于logs/gpu2-eval-startup-recovery-v1与logs/native-cuda-driver-recovery-v1；探针范围及真实接续证据见相应preflight目录和EVAL_CONFIG.md。

正式调度器重试DFlash任务时，仅复用拥有完整固定题集、匹配采样/检查点、正确源码及实际退出0凭证的已完成阶段，记录原GPU与日志哈希；不将不完整目录当作可续测数据。首次恢复复用MBPP+的378题AR和378题测速，补跑profile；SpecBench从尚未产生输出的AR阶段重跑。启动失败原始证据归档，不覆盖。复用测量后仍需完整profile与测速输出逐题相同才能进入结果表。

LoopAhead评测与三个分歧诊断入口复用同一GPU启动获取函数，保持所有非main函数AST一致。loopahead_diagnostic_inputs.py只放行已审阅的启动版本哈希；正式12个T0单元和独立B4的分歧清单在修改前后逐项一致，输入与模型运行代码仍精确核对。CPU兼容性报告在preflight/loopahead-cuda-startup-fix-v1，不替代实际BF16/FP32回放。

附录的教师生成、特征提取、LAD训练和四个兼容性/Graph检查单卡入口也使用同一GPU启动保护，并将启动模块保存到各自源码快照。七个入口的非main AST与修改前一致；数据选择、生成、训练、检查函数不变。已有训练/数据快照保留，CPU检查不代表附录已经执行。两个DFlash trainer均保持原版本，原DDP入口仍在模型加载前获取其本地设备。证据preflight/appendix-cuda-startup-fix-v1。

窄版正式训练现已完成：GPU3/6、47387有效例、6epoch各1481步共8886更新、global32，从头训练并实际退出0。最终checkpoint-008886有60个有限BF16张量/332,621,056参数，全部8886条优化日志有限，固定官方LR重放误差0；最终loss1.052720、rank0峰值28,501,811,200字节，训练占卡24.489464004 GPU小时。完整50k教师哈希/ID覆盖与2613项排除、冻结源码/导出、两rank成本、实际训练与CPU审计退出均通过；权重SHA256 fde0e64955405f5f38de8566d09459db89a8547d367ab321edf38586324b3d48，审计报告SHA256 02bc2ffd720010a53686a14123797f43a7c12032eb713c8d307647ef451daaa6。验收目录training/dflash-huginn-narrow-formal-audit-v1，独立完成/接续核验preflight/huginn-narrow-formal-completion-v1。训练完整性不代表正式速度、数值解码等价或优化器恢复已实测。

两张训练卡已自动接续Huginn MATH-500/T0、T1的同栈AR，之后执行窄版测速及全任务profile。基于历史AR耗时的临时排队优先级已完成，实际worker174369/174370启动后，观察器退出0并解除六项临时延期及GPU3/6的训练预留；只保留旧宽版训练延期。四卡白名单仍2/3/4/6，释放0/1/5/7。完整72格评测与BF16/FP32诊断仍在进行，当前单元数见formal-matrix-status.json。

最终checkpoint-008886另已在正式推理环境Torch2.6.0+cu126/Transformers4.57.1做CPU加载：与正式入口相同的inference_draft_class/from_pretrained配置，无缺失、多余、错形状或加载错误，60个BF16张量与保存权重逐项相同，参数量332,621,056，CUDA未初始化；实际进程退出0。证据preflight/huginn-narrow-final-inference-cpu-v1。这项检查只验证最终产物的跨训练/推理环境序列化兼容性，不替代GPU前向、完整测速或数值解码检查。

最终窄版首个完整正式单元GSM8K/T0已完成：同栈AR、未插桩测速、独立profile均为冻结的300题，三个实际进程和worker退出0。η_latency=2.345148、η_throughput=2.383521、γ=2.852721；测速22.802557 token/s、同栈AR9.566752 token/s。profile的全部输出和target验证次数逐题等于测速，草稿CUDA events共38.975477秒，占profile decode1362.750783秒的2.860059%。profile壁钟比测速低0.540271%，包含共享主机运行波动。224/300条整序列与AR相同，76处首次分歧仍待实际前缀BF16/FP32诊断；单元验收不等于数值等价已证明。证据preflight/formal-cell28-huginn-dflash-gsm8k-t0-v1。这不是同等训练预算的宽窄版比较，也不能把此单元加速外推至全部任务。
