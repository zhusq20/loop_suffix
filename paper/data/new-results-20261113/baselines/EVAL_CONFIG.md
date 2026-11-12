# LoopFill 表 2 评测配置与证据

状态：已从冻结代码、输入、逐题记录确认；2026-11-10用户同意本文旧表评测口径与双η记录。新环境20题和完整300题四组校验均已完成，全部与各自历史输出及循环数逐题一致。新增baseline的统一阶段状态见 `combined-results/results.csv` 和 `combined-results/status.json`；旧格式收集器已完成，根目录 `results.csv` 与 `formal-matrix-status.json` 暂时保留旧格式产物，完整结果收尾后才统一发布。全部36个LoopAhead格子已齐全，六个草稿器均已完成训练；全部72个正式格子的测速、AR与配套DFlash profile现已齐全，收集核验见`preflight/combined-results-cell72-v1/report.json`；六组附录DFlash检查与正文LoopAhead MATH接续队列于2026-11-13 04:56再次被共享控制动作停止，真实−15退出及部分产物已保留。取消文件引用“不需要做差异诊断实验”，来源尚未获本线程用户确认；测量继续，检查暂不反复重启，等待已发出的范围澄清。LoopAhead原262项FP32记录仍保留、167项接续未执行；DFlash GSM仅有53/244个草稿trace，尚无共同前缀对照完成记录。本文件记录冻结评测协议，按所附PDF表号命名。校验结果见 `preflight/calibration-n300.md`，可用 `summarize_calibration.py --count 300` 重算。

## 1. 必须纠正的对应关系

| PDF | 仓库历史名称与路径 | 测量内容 |
|---|---|---|
| 表 2，温度 0；附录表 4，温度 0 | `results/loopfill-latency512-20261107/`，历史称 table1 | 六任务，B1，最多 512 新 token，decode-only |
| 表 2，温度 1；附录表 4，温度 1 | `results/loopfill-latency512-t1-20261108/` | 相同输入和题序，T=1，top-p=1，seed=0 |
| 表 3 | `results/loopfill-table2-20261107-v2/table2-final.md` | SpecBench 480，最多 1024 新 token，B1/2/4/8/16，端到端吞吐 |

因此不能用任务文档提供的 `table2-final.md` 的 1024-token 协议补六任务表。两个协议的实际元数据均为 **CUDA Graph OFF、torch.compile OFF、行填充 OFF**；附录写 Graph 的句子与已测结果不符。只记录此差异，不修改原论文或旧实现。

关键证据：两套 latency 的 `protocol.json`、`source/scripts/loop_latency_worker.py`、`runs/*/metadata.json`、`requests.jsonl`；throughput 的 protocol/source/metadata。`collect_evidence.py` 已核对 151 个源码/输入哈希，全部通过，详见 `preflight/frozen-evidence.json`。

## 2. 数据、模板和抽样

冻结输入位于 `results/loopfill-latency512-20261107/inputs/<dataset>.jsonl`。温度 1 使用相同内容。每条有 source_id、原始 prompt、prompt_sha256、gold；推理只使用 prompt，不将 gold 传给模型。

| 文件 | 数量 | 原始模板 |
|---|---:|---|
| gsm8k.jsonl | 300 | GSM8K test；lm-evaluation-harness 固定前三个 CoT 示例，结尾 `Q: ...\nA:` |
| math500.jsonl | 500 | MATH-500 全量；固定四个 Minerva 示例，结尾 `Problem:\n...\n\nSolution:` |
| bbh.jsonl | 270 | 27 子任务各 10 题；官方 3-shot CoT，结尾 `A: Let's think step by step.` |
| humanevalplus.jsonl | 164 | EvalPlus 发布的函数补全 prompt |
| mbppplus.jsonl | 378 | EvalPlus 发布的 prompt；不是原始 MBPP 500 题 |
| specbench.jsonl | 480 | Spec-Bench 全部 480 题的第一轮 turns[0] |

所有模型均用其发布 tokenizer 的 `apply_chat_template([{'role':'user','content':prompt}], add_generation_prompt=True)` 包装，不另加 system prompt。不能直接复用 DFlash 自带 benchmark 的题集和模板。

数据准备入口：`scripts/loop_latency_inputs.py`。sample/order 种子均为 20261107；GSM8K 按 SHA256(seed:sample:dataset:source_id) 排序取最低 300；BBH 每子任务独立选 10，排除 demonstrations；执行次序用独立的 `order` 哈希。其他任务全量再排序。应直接复用已冻结输入，不重新抽样。

数据 revision 和源文件哈希见同目录 `inputs/manifest.json`。包括 GSM8K `3101c7d5072418e28b9008a6636bde82a006892c`、BBH `9ee07bd481feebf959a6b59d61ea57bdcf30964d`、harness `d6de81643928d653435c431bae19945d41d32520`；MATH-500 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`；HumanEval+ `d32357cf319e50e9c8d8dab5ea876c72b0fd321b`；MBPP+ `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`。

## 3. 模型与运行参数

| 模型 | HF checkpoint / revision | T | LoopSpec d1,d2 |
|---|---|---:|---|
| Huginn-3.5B | tomg-group-umd/huginn-0125 / bb6621b65e90b6a4b9b29ef88dc83866d450470c | 32 | 4,8 |
| Ouro-2.6B | ByteDance/Ouro-2.6B / 1ed04250da1a9936042725d302e81c8fa2ab5abd | 4 | 1,2 |
| Ouro-1.4B | ByteDance/Ouro-1.4B / 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | 4 | 1,2 |

权重已存在 `/tmp/loopfill-assets/hub/` 对应 snapshot；模型加载 trust_remote_code=True，local_files_only=True，BF16。Ouro 固定 total_ut_steps=4，关闭 adaptive early exit。B=1，每个 worker 四个 NUMA 本地 CPU 核，PyTorch/OMP 单线程；两条 prompt 各最多 32 token 预热，预热独立，不进入计时。

新 token 上限 512（含 prefill 产出的首 token）；EOS/chat-end 提前停止，无任务特定字符串停止。`resolve_stop_ids` 合并有效 tokenizer/model/generation EOS 和 Huginn `<|end_turn|>`、Ouro `<|im_end|>`；实际 ID 与排除的越界 ID 记录在每单元 metadata.stop_audit。

cache max_seq 按该模型/任务所有 prompt 的最大长度加 512+T+4，向上取 256 的倍数，再受模型 context_limit 限制。不截断题目，不减少输出上限。

温度 0：greedy，torch.manual_seed(0)。温度 1：temperature=1、top_p=1、top_k=0、sampling_seed=0；请求随机种子由模型/任务/source ID 派生，不含方法。LoopFill 现有采样器使用按绝对位置固定的 uniform 与完整 target softmax/CDF。新 DFlash/LoopAhead 的拒绝采样与 AR 不必同 seed 逐 token 一致；必须另外验证分布保持。

**Huginn 初始状态：固定的按绝对位置索引的截断正态表，不是零初始化。** `HuginnLoop.__init__` 使用 CPU Generator(seed=0)，生成 `[max_seq,5280]` 的表，std 来自 config，截断 ±3std，再乘 emb_scale，转换 BF16。prefill/decode/不同树形复用同一绝对位置状态。仅调用全局 manual_seed 不能使原生 HF 随机初始化在不同 block 形状间等价。新增适配器必须复用相同规则。

六任务走 B1 `drive()` + PackedLoopExecutor；continuous batching 与 128-token chunked prefill 是端到端吞吐套件的配置，不能把吞吐套件描述原样套到六任务计时边界。

## 4. η、γ 和 Mean

实际 PDF 表 2 的 η = sum(AR decode_s) / sum(method decode_s)。任务文档定义的是吞吐比，二者在生成 token 数不同时不等价。建议同时保留：

- `eta_latency`：以上时间比，用于与旧表数值对齐。
- `eta_throughput`（任务文档 results.csv 的 eta）：(method decode_tokens / method decode_s) / (AR decode_tokens / AR decode_s)。
- `gamma`：T × sum(decode_tokens) / sum(loop_steps)；AR 必须为 1。
- DFlash 每次 target block verification 消耗 T loop steps；γ=实际 decode 输出数/验证次数。末尾 EOS/长度截断只计实际输出。prefill 的首 token 不算 decode 输出；草稿时间进入 η，不进入 target loop steps。
- PDF 的 Mean 明确是六任务**算术平均**。仓库旧表的 GM 曾为几何平均，两者不要混用。

CSV另外保留两种输出一致率，定义与`collect_results.compare`、`packed_dflash_evidence.compare`一致：

- `match_rate`：完整输出token列表与AR完全相同的题数 / 总题数。
- `aligned_token_match_rate`：逐题按相同输出位置比较，所有相同token数之和 / 每题两份输出较长长度之和。较长输出多出的token计入分母；首次分歧之后的位置也继续比较。这两项均使用保存的完整输出，包括prefill产生的首token，计数范围与decode速度不同。
- `drafter_time_frac`：独立profile中草稿CUDA event耗时之和 / 该profile的decode壁钟时间之和。profile必须逐题复现原测速的输出和验证次数，其总时间不替换正式测速时间；LoopAhead该字段保留NA。

T0的一致率与全部首次分歧的数值证据共同报告。T1的一致率仅描述同seed运行的输出，不据此判断分布是否保持。以上指标均不是benchmark答题正确率或代码测试通过率。

计时从 prefill 后 CUDA synchronize 完成开始，到 decode、后缀发布与最终同步完成结束；包括草稿/采样/调度/目标验证，排除加载、预热、prefill、序列化、fsync、网络上传。DFlash 原始代码有排除第一次 draft 调用和计数首 token 的差异，需要独立评测封装修正边界，不能照抄其 time_per_output_token。

## 5. 已恢复的历史结果（不是本轮重跑）

从各 300 条逐题日志重算，匹配 PDF 的舍入数字：

| 模型 | AR decode tok/s | LoopSpec decode tok/s | η_latency | η_throughput | γ | 与 AR 完整序列相同 |
|---|---:|---:|---:|---:|---:|---:|
| Huginn-3.5B | 7.023526 | 24.676602 | 3.465637 | 3.513421 | 4.879450 | 211/300 |
| Ouro-2.6B | 6.393514 | 21.040567 | 3.233484 | 3.290924 | 3.592205 | 205/300 |

旧实验已有 BF16 输出差异，不能把这两个完整序列一致率当作新 baseline 的质量验收标准，也不能未经定位就归因为数值误差。原始详情 `preflight/historical-gsm8k.json`。表 3 的 5.89/5.53 tok/s 是另一个包含 prefill、输出上限 1024 的 SpecBench 协议，不能拿来对比本节 GSM8K 的绝对速度。

## 6. 入口和命令

原始单元入口为冻结 `source/scripts/loop_latency_worker.py --suite <suite> --job <model>-<dataset>-<arm>`；正式调度入口为同目录 `loop_latency_suite.py run --out <suite>`。这些历史 suite 已完成，不应对它们直接启动恢复命令来写新结果。

本轮校验入口独立放在 `experiments/baselines/preflight_worker.py`，只导入冻结引擎、arm_config、drive/summarize，保存到新目录，无 W&B 上传。例如：

```bash
/tmp/loopfill-venv/bin/python experiments/baselines/collect_evidence.py
/tmp/loopfill-venv/bin/python -u experiments/baselines/preflight_worker.py \
  --repo /ossfs/workspace/sz/loopfill --model ouro-2.6b --method loopspec-dual \
  --gpu 5 --cpus 52-55 --count 20 --max-new-tokens 512 \
  --out experiments/baselines/preflight/ouro-2.6b-gsm8k-loopspec-dual-n20
```

四个已启动单元的精确 argv/PID 在 `preflight/launcher.json`（本轮启动时为只读复用原 checkout 的冻结源码，SHA256 已核对）。每次重跑必须使用新的 out，避免覆盖。完整 300 题预期耗时依历史分别约 Huginn AR 74 分钟、LoopSpec 21 分钟、Ouro AR 124 分钟、LoopSpec 38 分钟；加加载预热和环境波动，总墙钟按 2–3 小时预留、占 4 张卡；这是计划估计，不是新实测。

## 7. 已确认决策

正文新增行采用与现有表一致的 η_latency，同时提供任务文档定义的 η_throughput；六任务输入、chat template、max512、T1 top-p=1、Graph OFF 均按冻结实测。若要改成吞吐比或开启 Graph，需要将旧表也按统一定义重算/重跑，不能只改新方法。

按任务文档检查点 0，此配置已获用户确认；不等同于已获论文作者额外确认。表 2/3 编号、计时和指标冲突已有确定证据，不再标作推测。后续授权见NOTES.md。

### 新baseline的温度1随机数

同栈AR使用旧表冻结的按request/absolute-position目标采样器。DFlash保持同样的request seed与prefill首token采样；后续使用实际softmax-q拒绝校正所需的独立随机抽取，不能以同seed的逐token相等代替等分布验证。LoopAhead的ordered-native拒绝校正也按其实际proposal分布执行，并单独冻结/记录request seed。两种新方法均top-p=1、top-k=0；原文LoopAhead top-p=.9/三seed只用于独立原文协议对照。
# AR分母来源补充（2026-11-10）

LoopAhead正式表沿用原表36个AR完整测量。导入前验证了全部GPU UUID属于当前8卡主机、冻结源码校验和、模型revision、BF16、Graph/compile/padding OFF、batch1、4核、题序/prompt/计数和逐题日志校验和。`import_archived_ar.py`逐字节复制原始请求日志，保留原测量时间戳，`preflight/archived-ar-import.json`记录来源；这不是本次新测量。

本次独立GSM8K全300题AR/LoopSpec校验全部复现同方法原输出与循环数。新旧AR吞吐变化另见`preflight/archived-ar-calibration-drift.json`。正式LoopAhead分母保持原表数值，不按校验任务推算其他任务的漂移修正；跨日期测量是结果限制。DFlash按每格实际使用的HF或packed目标栈另测自身AR；正式汇总不使用上述导入AR作为DFlash分母，实际后端逐格保留。

因此两种方法各自的η不能直接作为跨方法绝对速度排名。读表时必须同时查看CSV的method_tok_s、method_decode_s、method_decode_tokens、ar_tok_s及ar_provenance；T1输出长度还可能不同。此说明保留已经批准的各自分母与全部原始数值，不把差异归因于某个软件版本，也不依据局部校验作跨任务时延修正。


历史四卡调整：此前本任务仅使用GPU2/3/4/6，GPU0/1/5/7已交还。各评测worker固定原4核/单线程/Graph OFF设置。释放后其他项目进程已启动，观察到其CPU亲和范围覆盖本任务评测核；因此不能声称整机CPU隔离。保留原始时序和数据，在最终结果中说明共享主机条件，不用参数量推算或校正实际速度。释放时刻、当时活跃任务/CPU亲和以及原Huginn宽版训练中断证据见`logs/four-gpu-reallocation-v1/release-verification.json`。

最新续跑授权：用户明确要求恢复两种baseline，使用可用的全部8张卡，沿用小版DFlash和每任务最多500题，并复用已有结果。共享白名单`configs/gpu-allocation.json`现为0–7；Ouro1.4B LAD流水线已完成全部12格，其GPU3预留已由控制器自动释放，后续检查按共享队列接续。Huginn仍使用332,621,056参数、8886步的已完成检查点；六任务题数仍为300/500/270/164/378/480。旧调度器的暂停标记保留以避免与已接管的队列重复启动，不代表继续停止DFlash。续跑证据在`preflight/user-resume-eight-gpu-v1`。

去重按已有实际测量处理：9个未启动的重复DFlash latency项复用原HF-target结果及其原AR分母，状态为`reused_existing_measurement`；不能将这些数据标为新packed实现的实测。外部原packed表保留自己的源代码、计时和当时使用的历史AR分母；本任务正式汇总改用下段所述的新测同栈AR，历史外部比值仅归档。汇总必须保留实现来源，不能悄悄覆盖旧测速值。已完成全部请求但因旧父进程退出而缺少实际退出码的packed结果另标`complete_artifact_exit_unavailable`，先保留并核验完整记录，不自动重跑，也不伪造exit0。缺少退出凭证的限制须在采用该测量时披露。

续跑补测协议：`logs/packed-dflash-supplement-v1/plan.json`只为尚缺正式结果的12个正文DFlash格子安排同栈AR和独立profile，共24个补测阶段。已有12个完整HF-target格子全部保留；3个在去重前已启动的重复packed测量作为额外实现结果留档。补测直接导入packed测速所冻结的原始推理源码、输入和检查点，AR调用同一`generate`与`PackedDFlashTarget`但不加载草稿器。题目上限500、max512、2次32-token预热、BF16、Graph/compile OFF不变；profile记录CUDA草稿事件，并逐题核验tokens、seed、接受数与验证次数和未插桩测速完全一致。profile耗时不得代替正式测速。最终packed加速比使用新测的同栈AR，原外部表使用历史AR的比值仅保留作历史产物，不能充当此分母。原测速缺少实际退出凭证时仍显式保留该限制，不能因补测成功而补造退出码。

数值检查可独立推进，无需等待全部72格。`logs/baseline-diagnostics-resume-v1/plan.json`冻结27组检查：24个正文greedy格子、1个原文B4对照及2个开发集宽窄版检查；已完成检查直接复用。共享目录曾出现取消记录和STOP标记，但来源及操作者未核实，不能归因为用户指令或其他会话；本线程仍明确授权完成原任务。原冻结队列曾恢复。Ouro1.4B的6组LoopAhead检查由`logs/appendix-loopahead-diagnostics-v1`执行并已完成；6组DFlash检查后来改为各格测速及自身AR完成后可接续，仍让全部测量优先，不必等待整个37阶段队列；此次调度调整及随后发生的停止见下文。Ouro1.4B的6epoch/9108步训练及独立CPU验收现已完成。37阶段的Ouro1.4B DFlash测评控制器保持现有修订版本，包含接口验证、12组测速、12组同栈AR和12组profile，诊断另行排队；不重启已完成训练或测速。诊断保持原始生成预算，在首次分歧后停止，不缩短草稿块预算来制造一致。原取消记录与更正前文档均保留备查，当前验收仍覆盖36个正式greedy格子、B4及2个开发检查。

Huginn DFlash窄版开发诊断单列于preflight/huginn-dflash-dev-compare-v1：隔离dev1000中预先固定分层20题、max128、T0、Graph OFF，比较旧宽版4000步与窄版pilot60步，训练预算不同。此诊断不进入72格、不替代本文件的冻结题集/max512正式口径；正式窄版完成50k/6epoch训练后仍须按原完整题集、同栈AR、未插桩计时加独立全任务profile评测。

正式Huginn窄版使用独立evaluate_projected_dflash.py入口；模型仍执行完整32轮，target embedding/head冻结，草稿器5层/H2048/FF5632/16头/block16、332,621,056参数。正式门槛为原50k教师/47387有效例、6epoch/8886更新及独立训练验收；2k pilot及旧宽版4000步不能进入正式表。DFlash表注将该配置标为我们对Huginn的窄版适配。

2026-11-11驱动加载环境补充：运行中的训练/MATH进程使用系统原生570.148.08驱动库；随后观察到`/etc/ld.so.cache`修改时间1794378966.2280037，新进程默认解析到CUDA compat的570.86.15，而内核模块仍570.148.08，导致CUDA初始化Error 803。缓存修改来源未知。修复仅为本任务子进程设置私有原生驱动库目录的`LD_LIBRARY_PATH`；未修改系统驱动、缓存、CUDA/Torch包或GPU设置。新主调度与附录调度已用以下入口启动，子进程继承修正环境：

```bash
python experiments/baselines/native_cuda_env.py \
  --record <新的环境记录.json路径> -- <原启动命令及参数>
```

记录路径必须不存在；实际启动argv、原生库SHA256及两环境GPU检查见`logs/native-cuda-driver-recovery-v1`与`preflight/native-cuda-driver-recovery-v1`。此前已启动的MATH外层worker仍继承旧环境，因此DFlash入口另在导入Torch/创建输出目录前检查并必要时以相同argv重新exec，记录保存至`logs/<run-name>-native-driver.json`；decode、采样和两次预热逻辑保持不变。已有环境修正时不重复exec。所有驱动处理均在正式decode计时之外，不校正测量值；GPU探针只验证启动、小矩阵和FlashAttention内核，不能替代完整benchmark或数值分歧诊断。

正式收集额外核验`result_evidence.py`：本轮测量须有与原launcher的PID/argv/启动时间绑定的实际退出0凭证，逐题时间/计数、模型revision、归档源码、检查点参数加载和启动参数一致。LoopAhead按温度分别核对T0/T1冻结引擎源码；DFlash只接受已审阅的入口启动差异。历史AR逐文件验证原目录、导入副本及冻结协议，保留其历史测量身份，不补造本轮退出或计时。`--partial`会跳过已写complete但尚无退出凭证的单元，严格收集则拒绝；非零退出或证据错配始终报错。该CPU检查不替代数值分歧/采样分布验收；加入此旧格式检查时的23行原值已保留。这段规则适用于旧格式收集器；续跑packed原测速的缺失退出凭证按上文单列，并不伪装成满足本段exit0要求。当前全量计数以combined-results/status.json为准。

统一汇总入口为`collect_combined_results.py`，阶段输出在`combined-results/results.csv`、`combined-results/table2_rows.md`及逐格`evidence.json`。原`collect_results.py`已完成LoopAhead流水线的旧格式收集，其结果仍作为统一收集器的输入；阶段性统一表独立保存，待收尾再发布到根目录正式交付文件。合并时逐项重验旧结果、新packed同栈AR/profile、实际退出凭证或其缺失限制、训练检查点和输入来源；同一method/model/task/temperature键优先保留已经完成的旧测量。DFlash单元标H（原HF目标）或P（packed目标）；同一行含两种实现时，均值显式标“混合实现”，不能表述为统一后端测量。

当前数值验收范围以本线程的明确授权为准。共享目录存在“用户取消诊断”的记录，但没有可核实的会话或操作者信息，本线程未收到明确取消或豁免要求；因此原必要检查仍保留；其冻结队列曾恢复并置于训练/测速之后，随后共享目录再次出现停止操作；旧启动记录不能证明当前仍运行，最新接续状态见文末。既有完成检查直接复用，未知来源记录和原始STOP文本保存在`preflight/restore-diagnostics-authorized-goal-v1`，不把它们作为数值验证已通过的证据。

最终表发布前须保留旧格式`results.csv`与匹配的`formal-agreement.json`作为收集输入；不能让混合后端的最终CSV被旧格式reader当作原输入。独立入口`collect_combined_results_archived.py`支持同时传入`--legacy-results <保存的旧CSV>`和`--legacy-agreement <保存的agreement>`，并用`--out <新目录>`独立重算。运行中队列继续使用原字节的`collect_combined_results.py`，其源码哈希已冻结，不能为添加选项而修改。两份输入均检查读取前后哈希，输出证据记录实际路径与哈希。已用保存的46行旧格式输入重算当时58格，CSV逐字节、论文表及每格来源证据均与原汇总相同，真实exit0且未重跑GPU，见`preflight/legacy-input-replay-v1/report.json`。这是阶段性重算验证；最终发布时仍需保存届时齐全的旧格式输入并记录其命令，不能拿此58格快照冒充完整结果。

曾将这组选项直接加入冻结的收集器，触发附录DFlash CPU等待器的源码保护并使其停止；训练、测速没有重启。已保留异常与旧进程记录，恢复收集器的原哈希，并将归档重算功能移到上述独立入口；未放宽计划或源码校验。等待器当时由持久监督器重新启动，等待原训练及37阶段测评后执行六组检查；此后已进行下述增量调度切换。旧等待器没有真实退出码凭证，保留为unknown，不从异常文本推造退出码；恢复证据在`logs/appendix-dflash-diagnostics-v1/recovery-pinned-collector-v1`。

独立归档入口已使用同一保存输入重新运行，实际exit0；58格CSV、论文表逐字节相同，逐格来源证据不变，当前59格表与冻结收集器/spec均未改动。实际验证见`preflight/legacy-input-replay-v2/report.json`。

最终发布所需的旧格式输入现已保存在`preflight/final-legacy-inputs-v1`：48行包含全部36个LoopAhead格子和保留的12个HF DFlash格子，配套agreement及原收集器源码哈希均已核对。该目录的`report.json`保存独立重算命令模板；packed DFlash格子仍由统一收集器另行读取。当时统一表为60/72，余下12个Ouro1.4B DFlash格子尚待训练和测评，不能把输入归档当作全任务完成。最终覆盖根目录CSV前继续保留该快照。

Ouro1.4B小版DFlash的12组正式主测速现已全部完成，共4,184条请求，独立CPU核验与原12个测速进程均实际退出0；证据在`preflight/ouro14-dflash-timed-complete-v1`。核验逐题输入、原始计数/计时、模型与源码、最终权重、退出凭证，以及由base seed 0/model/dataset/source_id确定的SHA256请求种子。冻结入口每组先执行首题和最长prompt各一次max16功能探针，再对前两题各做max32预热；这四次调用均不计入正式decode时间。主测速完成不等于12个加速比格子齐全：各自AR、独立profile及另排的6组数值检查仍须完成，当前统一完整矩阵仍为60/72。


2026-11-13增量调度与共享目录停止记录：`preflight/appendix-dflash-incremental-v1`保留候选计划、CPU优先级验证及切换的实际exit0。原8份冻结源码、6组检查范围、单格AR完成门槛、GPU租约和测量优先规则均未改变，仅解除“全部37阶段完成才启动等待器”的总门槛；空闲CPU等待器216747计划内停止，实际exit为−15，替代控制器519800于北京时间01:37:19启动，当时没有诊断GPU子任务。01:40:42，该控制器及正文最后一组Ouro2.6B LoopAhead MATH FP32检查又被停止，两份实际监督器凭证均为−15；共享取消文件声称来自“loop_suffix conversation”，但操作者、会话身份与指令归属未核实，不能归因为本线程用户，也未读取其他聊天。原始记录及现场状态保存在同目录`subsequent-stop-observation`。本轮观察未重启有争用的诊断队列，8个AR测量继续；已验收正式数值组29/36及补充3组不变，未完成的7组仍不能算通过或豁免。原始60格CSV、冻结spec和源码哈希均未改变，全部72格及其配套profile仍是测量交付范围。


用户在核对重新贴出的同一任务文档后，再次明确要求“好的 你继续试验吧”。据此，六组Ouro1.4B DFlash检查于`logs/appendix-dflash-diagnostics-v1/resume-after-reconfirmation-v1`恢复排队，启动核验实际exit0。冻结plan/spec及全部源码不变；原控制器已停止、dispatch为空、六组输出均不存在。只在确认STOP与已说明的历史记录逐字相同后，保存其副本并清除该旧标记；原取消记录与真实−15退出均保留，不推断其操作者。新控制器只用CPU等待测量优先队列，启动时未运行GPU检查，现有八个测量进程均继续。此接续不覆盖或重跑任何完成的测速；正文最后一组LoopAhead MATH的262项部分记录已另行归档，尚余167项，须采用独立接续记录处理，不能直接覆盖旧退出凭证。


正文LoopAhead MATH中止检查的独立接续入口已通过CPU准备核验，见`preflight/loopahead-fp32-resume-preparation-v1`。它严格绑定原中止日志及262项连续前缀，按原429项清单只运行剩余167项；原推理入口与实现保持冻结，旧−15凭证继续保留。8项边界/AST测试及6类错误输入绑定拒绝检查通过，不代表GPU接续已完成；独立attempt和最终验收来源映射仍待接入。

- 2026-11-13：最后一组LoopAhead MATH独立接续队列已启用（`logs/loopahead-math-fp32-continuation-v1`），当前仅CPU等待，控制器64214，未启动新的GPU检查。保留旧实际−15退出和262条源绑定记录；只计划执行剩余167项，原429项BF16报告逐字复制至新attempt并明确保留来源。28项CPU测试全部通过，准备、统一部分验收及启动均实际exit0，证据`preflight/loopahead-fp32-continuation-integration-v1`。`preflight/numerical-matrix-partial-v24/report.json`仍仅计29/36正式组加3项补充，7组待完成、无验收错误；60格结果CSV未改。新验收映射仅适用于此确定中断，不允许用部分日志或成功的CPU启动代替最终GPU实际exit0。

2026-11-13收尾保存第68格结果时，本仓库Git自动整理短暂具有全CPU亲和范围，可能与仍在运行的两个MATH AR及三个profile任务重叠。已将自身整理进程限制到非测量核80–83，保留原始时间和相关进程证据（`preflight/git-housekeeping-affinity-v1/report.json`）；不据此作时延修正，也不将共享主机描述为完全隔离。后续本任务的提交不自动触发GC。

2026-11-13测量收尾：全部72格已核验并保存在`combined-results/results.csv`，SHA256 `5948308581a4b80fef8666921103486bf0136448842e7ea086bb08b0aab5c438`；最后一次收集与外层刷新均实际exit0，原71行逐字段不变。六项训练及实际成本的统一对应记录为`preflight/six-training-completion-v1`（成功凭证为`exit-attempt2.json`）。最终T1源码核验`preflight/sampling-source-coverage-final-v1`实际exit0，36/36格全部绑定既有17项CPU检查、32k与96k合成解码记录；没有重做benchmark或随机样本，不将有限检查称为普遍等分布证明。

与72格CSV绑定的数值范围审查`preflight/numerical-matrix-after72-v1`实际exit1，未通过完整验收：29个正式组加3项补充仍有完整有效证据，另外5个附录DFlash组缺实际完成记录，GSM检查因实际−15退出未通过，LoopAhead MATH接续因原STOP/cancellation被后一次共享控制改写而未通过冻结来源校验。后两项原因已逐项核对于`preflight/numerical-after72-stop-review-v1`：旧STOP与取消文件的原哈希副本仍在，其他27份接续计划产物未变。没有修改停止文件、原计划或验收门槛。当前72格是已完成的性能测量；7组未完成检查仍保留，正式最终表发布与全部验证完成不能据此宣告。
