# Preliminary 实验计划：浅层预测与后缀续查

状态：2026-11-09 全部完成。前12题验证通过后，正式480题从头采集，共80,933个有效位置；回放、2000次配对 bootstrap、出图及 W&B artifact 上传均已完成。目标是在 ICML **一个单栏宽度内并排两张子图**，为现有调度 teaser 提供实证动机，不在此图报告端到端加速。

## 共同设置

- 主图用 Ouro-2.6B，最终深度 T=4；固定现有 SpecBench 480 题及题序，每题最多 512 个新 token，greedy、原 chat template 与 EOS 规则。记录输入、权重版本及配置哈希。此实验独立于 temperature=1 性能实验。
- 先生成 full-depth AR 参考轨迹；各对照使用完全相同的参考前缀、目标 token 和后缀索引快照。只评估生成区间中 prefill 首 token 之后的位置；按实际解码语义读取中间状态及输出头，检查最终读出与参考 token 一致。
- 后缀索引仅包含 prompt、当前已确认输出和题序中已完成请求的参考输出；不得提前写入当前或未来答案。每个位置评测完各对照后才推进快照。
- 主指标汇总全部合格位置；配对按请求 bootstrap 2,000 次给出 95% CI，随机种子 0。零命中与无匹配位置均保留。保存任务类别以供附录分组，不依据结果挑选任务或深度。

## (a) Shallow agreement：浅层读出是否互补？

在同一前缀上，记第 d 次 loop 的 top-1 为预测 p_d，最终 greedy token 为 y。按2026-11-09用户最终澄清，左图在同一坐标轴上同时画 **单层命中率和额外命中率** 两组数据：

- 蓝色实线、圆点：第1、2、3层各自的单层命中率 A_d=P(p_d=y)。
- 橙色虚线、方点：第2层 G_2=P(p_2=y 且 p_1≠y)，第3层 G_3=P(p_3=y 且 p_1≠y 且 p_2≠y)。额外命中率只在第2、3层画点。
- 两组数据的分母均为全部合格解码位置，不是前层未命中的位置数。横轴为 Loop depth 1、2、3，共用百分比纵轴，不使用双纵轴。

累计覆盖 C_d=P(存在 j≤d，使 p_j=y) 仍保存在表中，但不画在左图。G_d=C_d−C_(d−1)，C_0=0。额外命中率衡量后续读出对已有候选的补充，不代表可以提前确定正确 token。原冻结实验和原图保留，新图由同一批逐位置数据核对后生成。

## (b) Suffix synergy：浅层提案能否重新接上续写？

固定浅层读出 d=1、top-1；图中三个柱使用相同总候选节点预算 N=16（不计根），单次 suffix 查询最多 Q=4 个节点，复用同一后缀检索与排序规则。这里只构造候选树并与参考轨迹匹配，不运行完整流水线。

| 柱 | 候选构造 |
|---|---|
| Suffix | 在已确认前缀 x 上查询 suffix |
| + Shallow | 保留根部 suffix 树，加入浅层 token p_1；不从该 token 续查 |
| + Requery | 与上一组相同，再用 x+p_1 查询 suffix 并连接返回的续写 |

候选按 token 路径去重；先保留共同的根部 suffix 候选和浅层候选，续查只使用剩余预算，不替换共同候选。仅从指定浅层 token 续查一次，不递归扩张；记录实际节点数，不能将相同预算上限描述为相同计算量。

纵轴为平均连续正确草稿长度 L：从根开始，候选树中与参考后续 token 完全一致的最长路径长度；首 token 未覆盖则为 0，不计验证器额外产生的 bonus token。输出剩余长度不足时只计真实参考区间，不越过 EOS。报告三组均值与配对差值，尤其是 ΔL=L(+Requery)−L(+Shallow)。该差值衡量续查的额外候选覆盖，不直接解释为速度收益。

另在结果表保存 Suffix 的 L=0 子集占比、该子集三组的平均 L 和 ΔL。单栏空间有限，子集结果放图注或正文一句话，不再加 inset。若 ΔL 很小，如实报告，并将机制论述重点转向调度重叠与计算复用。

## 版式、交付与执行顺序

- 使用单栏 figure，两张子图各约 0.48\columnwidth；左图同轴画单层命中率与额外命中率两组折线，右图三个柱，最终尺寸下文字约 8 pt。右图短标签用 Suffix / +Shallow / +Requery，定义写在共同图注中。输出矢量 PDF 与绘图脚本，不画重复程度相关性主图。
- 先用固定题序前 12 题检查 token 对齐、索引可见性、零命中计数、节点去重和续查开关；修复后以冻结配置重新运行完整 480 题，试跑不作为独立结果合并。
- 复用 `scripts/loop_suffix_spec_readouts.py` 的采集框架；先核对其中读出与实际解码的 cache/输出头语义。原逐 token 读出未纳入仓库，需要恢复或重新采集。`scripts/loop_accepted_length.py` 的 T×tokens/steps 不是此处 L，不能代用。
- 保存逐位置读出与三组 L、实际节点数、逐请求汇总、bootstrap 表、配置哈希和两图 PDF。主图完成后再考虑 Huginn 或重复程度分组的附录验证。

拟用图意：**早期 loop 提供互补候选；浅层 token 后的后缀续查可否带来额外的连续正确草稿，由右图配对增量检验。**

## 2026-11-09 执行记录

- 当前批次：[冻结协议](../results/loopfill-preliminary-20261109-v2/protocol.json)、[实时状态](../results/loopfill-preliminary-20261109-v2/status.json)、[队列日志](../results/loopfill-preliminary-20261109-v2/queue.log)。入口为 `scripts/loop_preliminary_suite.py`；统计实现在 `src/llm_memory_editability/loop_preliminary.py`。
- 前12题已全部通过独立 AR 逐词核验，4,085个有效位置的回放、2000次 bootstrap 和图均已完成：[试跑汇总](../results/loopfill-preliminary-20261109-v2/pilot/summary.json)、[试跑图](../results/loopfill-preliminary-20261109-v2/pilot/preliminary.pdf)。这些题都属于固定题序开头的 MT-Bench，不代表全480题结果。正式采集在隔离的 `full/` 中完成。[W&B 跟踪](https://wandb.ai/zsqzz/llm-memory-editability/runs/feb3521f193f)。
- 参考和浅层 top-1 在同一真实逐 token AR 前向中采集；使用模型原生每轮归一化后的状态，输出头每次保持 `[1,H]`，不把多层读出合并成另一种 BF16 GEMM 形状。各题可分卡生成；suffix 回放严格使用按 question_id 排序的题序，共享历史不会按 GPU 完成顺序更新。
- 前 12 题都再运行一次独立 AR 生成入口，逐 token 验证与带读出的生成一致。通过后才开始 `full/`，不复用或拼入 `pilot/` 输出。每题的四层读出、prompt、完整参考输出保存在 `readouts/<qid>.npz`，旁边 JSON 保存 SHA256 与验证信息。
- prompt 使用发布的 chat template，不截断；生成最多 512 token，tokenizer EOS 与 `<|im_end|>` 停止。固定 suffix `max_spec_factor=1`、`min_token_prob=.1`、tree depth=64、历史容量100000，采用已有 `CompletionSuffixCache` 的局部/历史分数择优规则。同一位置先查询根，再查询 `x+p1`，所有对照评分后才发布下一参考 token。
- 相同预算上限 N16 不等于相同节点数：根查询至多4个节点，浅层至多新增1个，一次续查至多新增4个，所以本实验实际至多9个节点。`nodes` 和 `requery_nodes_added` 均逐位置记录。
- 输出包括 `positions.jsonl`、`requests.json`、`summary.json`、`metrics.csv`、`by-category.json` 和单栏两子图 `preliminary.pdf/png`。置信区间按请求整体重采样2000次、seed=0，以位置数为分母重新计算 pooled 指标与配对差；无 suffix 命中子集单独保存。
- 初次试跑 `loopfill-preliminary-20261109/` 因发布版 `UniversalTransformerCache` 与 Transformers 4.55 的只读 `key_cache` 属性不兼容而失败，未采集有效请求。日志和冻结源码保留。v2 显式初始化 HF `DynamicCache` 的 T×48 个独立 KV 槽，使用发布模型已支持的传入缓存接口；未改权重或模型源码。小尺寸原生 Ouro 模型的缓存/整段参考与两种生成入口一致性检查通过。
- 16项读出/回放测试通过；此前相关 suffix/batch 回归共45项通过。W&B 依据现有 tracking defaults 由独立进程同步，网络状态不改变实验成功条件；跟踪状态见 `tracking.json`，失败可重新启动跟踪器。

恢复命令（必须使用冻结源码；运行中的控制器/worker 会阻止重复启动）：

```bash
PYTHONPATH=results/loopfill-preliminary-20261109-v2/source/src:results/loopfill-preliminary-20261109-v2/suffix \
  /tmp/loopfill-venv/bin/python \
  results/loopfill-preliminary-20261109-v2/source/scripts/loop_preliminary_suite.py run \
  --out results/loopfill-preliminary-20261109-v2
```

每题完成后原子保存并校验配置/数据哈希；恢复跳过已完成题目，从空索引按完整题序重新回放。控制器与GPU共享锁沿用既有实验约定，不抢占其他实验。`status.json` 是当前运行状态；本记录中的启动状态不是完成声明。

## 正式结果（480题）

全部80,933个有效位置保留，包括零命中；prefill首token不计入。以下区间均为按请求配对重采样2000次的95% percentile CI，seed=0。

| Loop深度 | 单层一致率 A_d | 累计候选覆盖 C_d | 新增覆盖 G_d |
| --- | --- | --- | --- |
| 1 | 88.83% [87.70, 89.97] | 88.83% [87.70, 89.97] | 88.83 pp |
| 2 | 95.63% [95.12, 96.13] | 96.72% [96.30, 97.12] | 7.89 pp |
| 3 | 98.20% [97.99, 98.42] | 98.95% [98.82, 99.09] | 2.23 pp |

按逐位置预测直接核对：第2层补中第一层遗漏的 **6,388 / 80,933 = 7.89%**，95% CI [7.10%, 8.65%]；第3层补中前两层都遗漏的 **1,807 / 80,933 = 2.23%**，95% CI [1.95%, 2.54%]。第1层命中71,891个位置，前三层均未命中847个位置，四项合计80,933。[修改后的图PDF](../paper/figures/loopfill-preliminary-v2/preliminary-hit-rates.pdf)、[PNG](../paper/figures/loopfill-preliminary-v2/preliminary-hit-rates.png)、[绘图脚本](../paper/figures/loopfill-preliminary-v2/build_figure.py)及旁边的 `figure-data.json` 保存精确数据与原汇总SHA256。无需重新采集模型输出。

| 候选构造 | 平均连续正确草稿长度 L | 平均实际节点数 |
| --- | --- | --- |
| Suffix | 1.343 [1.187, 1.509] | 2.607 |
| +Shallow | 1.732 [1.600, 1.873] | 3.092 |
| +Requery | 2.276 [2.105, 2.459] | 4.640 |

续查相对只加浅层提案的配对增量为 **+0.544 token [0.502, 0.587]**。相比 Suffix，总增量为+0.933 token [0.912, 0.954]。Suffix 的 L=0 子集占48.97%，该子集三组平均 L 为0 / 0.796 / 1.202，续查额外贡献+0.407 token [0.374, 0.445]。

结果支持浅层候选互补，以及在浅层提案后续查能增加连续正确草稿覆盖；实际节点数也增加，不能直接解释为端到端加速。

产物：[正式汇总](../results/loopfill-preliminary-20261109-v2/full/summary.json)、[指标CSV](../results/loopfill-preliminary-20261109-v2/full/metrics.csv)、[分任务统计](../results/loopfill-preliminary-20261109-v2/full/by-category.json)、[图PDF](../results/loopfill-preliminary-20261109-v2/full/preliminary.pdf)、[逐位置记录](../results/loopfill-preliminary-20261109-v2/full/positions.jsonl)。W&B artifact 为 `zsqzz/llm-memory-editability/loopfill-preliminary-20261109-v2:v0`，1206个文件已上传；未另行进行云端下载哈希核验。

收尾复核：冻结源码、依赖版本、题库及480份读出哈希通过；四层读出对齐、最终层参考一致性、80,933条记录的共同候选长度单调性及节点预算通过；从逐位置记录重算 A/C/G/L/节点数与汇总一致。8张GPU已释放。
