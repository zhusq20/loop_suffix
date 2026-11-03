# Loop投机解码墙钟测速：后缀树+浅层读出的真实解码器（2026-10-31，已完成，结果见§8）

用户2026-10-31要求实现代码、先不运行，并要求测速前考虑其他程序的干扰。离线批次见[协议](../loop-suffix-spec-20261030/protocol.md)与[报告](../loop-suffix-spec-20261030/report.md)。

## 1. 问题

离线回放给出的是每词块调用数的倍数，例如raw-256主比较中后缀树+浅层为Ouro 4.71倍、Huginn 6.82倍。这个倍数假设一次块调用的耗时不随树宽变化。本批在GPU上实际运行同一过程，测量墙钟时间：

- 后缀树+浅层相对只用浅层、相对逐词，在墙钟上还剩多少加速；
- 调用数倍数兑现成墙钟倍数时损失在哪里：每次调用的耗时、行数、后缀树查找、prefill。

已有墙钟参照：sibling批次的WFD在60题raw-256上为Ouro 2.25倍、Huginn 2.65倍，对应的调用数倍数为3.0和4.6倍；在Huginn上，分支实现每次调用的耗时随行数翻倍，最终比WFD还慢。

## 2. 实现

- **解码器**：[loop_tree_decode.py](../../src/llm_memory_editability/loop_tree_decode.py)。建点、补位、剪枝和接受规则与`replay_request`相同，每次块调用推进所有存活节点一步。精确算术下，块调用数应与离线回放相同。
- **注意力**：每个节点读取两部分并按log-sum-exp合并：
  - 已提交前缀：读本深度槽的主KV缓存，每层一次FlashAttention varlen调用覆盖所有深度槽，用`seqused_k`只读前缀，不拷贝前缀。
  - 树内部分：读本步树缓冲中的祖先，按深度槽分组补齐后做一次批量掩码注意力。
  - sibling之所以慢，是因为每层每组都要把整段前缀拼接一次，并在全部拼接后的键上计算带掩码的注意力。本实现两件事都不做。
- **Huginn**：浅层读出的coda用单独的缓存，与离线teacher forcing一致，即深度d的读出看深度d的coda状态。初始潜状态使用与读出脚本相同的逐位置表。
- **补位范围**：离线回放只在贪心路径上补位；真实解码器事先不知道哪条路径正确。因此设两个臂：
  - `both`：所有存活节点到深度d都补位，调用数与回放相同，但行数更多；
  - `both-path`：只在预测路径上补位。预测路径从最后接受的节点出发，优先走与浅层第1候选相同的子节点，否则走第一个存活子节点。行数更少，调用数可能略多。
- **WFD参照**：WFD提交`4345209`，与新解码器共用同一份权重。Huginn的初始状态换成同一张表（算法不变），使各臂输出可以互相对照。WFD只能在一个停止词处停：chat格式用回合结束符，raw格式用EOS。
- **测速脚本**：[loop_suffix_spec_wallclock.py](../../scripts/loop_suffix_spec_wallclock.py)。主机监控：[loop_suffix_spec_env_monitor.py](../../scripts/loop_suffix_spec_env_monitor.py)。

**已做的校验（CPU，未用GPU）**：[单元测试](../../tests/test_loop_tree_decode.py)共22项全部通过。

- 用规则构造的假模型：AR、只用浅层、只用后缀树、后缀树+浅层、剪枝、K=3六种设置，输出都等于贪心，块调用数与回放逐题相同；节点数达到上限时仍然无损。
- 用两个检查点自带建模代码搭出的微型随机Ouro和Huginn（fp32）：六种设置的输出都等于HF贪心，块调用数等于用HF teacher forcing读出所做回放的结果。
- 树注意力与稠密参照一致；提交拷贝正确；初始状态表的前缀行与表长无关。
- 还没有在GPU上验证的部分：FlashAttention路径和pinned上传。脚本启动时先在GPU上比较flash路径与数学路径，误差超过2e-2即退出；`--mode check`用fp32逐题核对各臂与AR是否一致。

## 3. 设置

| 设置 | 格式与停止 | prompt | 新词上限 | 离线平均输出（Ouro / Huginn） | 正常停止 |
| --- | --- | --- | ---: | --- | --- |
| chat-256（协议§7b） | 模型自带chat模板；EOS或回合结束符 | 问题截到总长512，模板保留 | 256 | 141 / 91词 | 339/480、409/480 |
| sb1024（协议§7c） | 同上 | 不截断（最长1588 / 1564） | 1024 | 212 / 111词 | 453/480、466/477 |

- **题目**：每类按question_id均匀取10题，6类共60题，与sibling测速批次相同。prompt的token序列直接取自离线运行的npz，保证与离线完全相同。
- **全局后缀树**：装入离线运行中所有更早题目的贪心输出。每次运行结束后撤回本次的回答，使所有臂面对相同的树。
- **Huginn sb1024缺题**：缺253、288、295三题（摘要类，OOM后未补跑），全局树里也少了这三条输出。
- **参数**：Ouro T=4、d=1；Huginn T=32、d=4；K=16，α=1，不剪枝，即离线主比较的设置。

## 4. 计时与指标

**每题流程：**

1. 在CPU上用离线读出计算各臂的回放调用数，不计时。
2. 记录一次环境快照。
3. 每个臂做8词的不计时预热。
4. 各臂按轮换顺序正式运行，第i题把臂列表左移i位。

**计时方式：**

- 每次运行前后各做一次CUDA同步。
- 运行前回收Python垃圾；运行期间关闭GC。
- 同时记录墙钟时间和进程CPU时间。
- `--repeats`控制每题重复次数。

**指标：**

- 主指标是decode tok/s，不含prefill和第一个词，以及相对本实现AR和相对WFD AR的倍数。WFD臂的decode时间等于总时间减去单独测得的WFD prefill时间。另报端到端tok/s。
- 每词调用数，与回放对照；每次调用的毫秒数；每次调用的行数；后缀树查找占decode时间的比例。
- 各臂输出与离线输出、与本实现AR是否一致。bf16下的近似平局会使少数题分叉，单独报告。
- 结果分全部题目与无干扰标记题目两份汇总，并按类别拆分。

## 5. 环境一致性

| 干扰来源 | 影响 | 控制与记录 |
| --- | --- | --- |
| 同一张GPU上的其他进程 | 分时占用SM和显存带宽，影响最大 | 只用本项目常用的空闲卡2/6/7（按记忆，0/1/3/4/8通常属于其他用户）。启动前若有计算进程或3秒内利用率不为0，脚本拒绝运行。每题前后用NVML记录进程列表，监控每秒记录一次；出现外来进程的题目标记后删除记录重跑（记录可按题续跑）。 |
| 主机CPU争用 | 解码受kernel启动开销限制：Ouro的WFD每次调用约35 ms，大部分是CPU时间。其他用户（当前uid 1051有5个进程各占约95%）及本项目自己的CPU任务（离线回放用16进程、W&B同步等）都会拖慢 | 用`--cpuset-cpus`把容器绑到GPU所在NUMA节点的物理核，并让这些核的SMT兄弟核空着；torch和OMP只用1个线程；测速期间不跑本项目其他任务；监控记录绑定核及其兄弟核的忙碌度与当前频率。其他用户的进程仍可能被调度到这些核上，无法完全阻止，只能标记。 |
| CPU频率 | EPYC的boost频率随同一插槽的负载变化 | 同一题内各臂交替运行，缓慢漂移对各臂影响相同；监控记录频率。 |
| NUMA | 跨节点访问内存更慢 | GPU 2→节点0（CPU 0-23、48-71）；GPU 6/7→节点1（CPU 24-47、72-95）。`--cpuset-mems`与之对应。 |
| GPU时钟、温度、功耗 | 机箱里其他卡在300 W功耗上限下满载；本测试受启动开销限制，功耗较低 | 每题前后记录SM时钟、温度、功耗和throttle原因，出现功耗或温度降频即标记。锁频需要root权限，不做。 |
| 两个模型同时测 | 同一NUMA节点上会争用核、LLC和内存控制器 | 推荐在同一张卡上先测Ouro再测Huginn。sb1024单次约2.5小时。若并行，两个容器分放两个NUMA节点。 |
| 磁盘、NFS、W&B | I/O和后台日志线程占用CPU | 权重和离线数据在计时前读入；每题结束才写JSONL；W&B在全部计时结束后离线写入，计时期间进程内没有任何采样线程。 |
| 显存分配、GC | 首次分配和GC暂停 | 本实现的KV缓存一次性预分配；WFD每次调用自行分配，这属于它本身的成本，预热后由缓存分配器复用；GC在每次运行前回收、运行中关闭。 |

无法消除的部分：其他用户的CPU负载和机箱温度。本批的核心比较都是同一进程、同一题内交替运行的比值，环境漂移对各臂基本同向；绝对tok/s只对本机这一时段有效。

## 6. 运行命令（已于04:57–13:02 UTC执行）

```bash
REPO=/mnt/disk2_from_server2/siqizhu4/llm-memory-editability
P=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability
OUT=$P/loop-suffix-spec-wallclock-20261031; mkdir -p $OUT && chmod 777 $OUT
# 主机监控：绑在测速不用的核上，记录GPU 6所在节点的绑定核及其SMT兄弟核
taskset -c 47 python3 $REPO/scripts/loop_suffix_spec_env_monitor.py --cpus 24-31,72-79 --out $OUT/monitor.jsonl &
run() {  # name model setting extra...；前台运行，上一个容器退出后才启动下一个
  docker --context lm-memory run --name loop-suffix-spec-wallclock-20261031-$1 \
    --gpus device=6 --cpuset-cpus 24-31 --cpuset-mems 1 --network none --user 1015:1016 \
    -e PYTHONPATH=/repo/src -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/tmp/hf -e HF_MODULES_CACHE=/tmp/hf_modules -e WANDB_MODE=offline -e OMP_NUM_THREADS=1 \
    -v $REPO:/repo:ro -v $P/loop-spec-reuse-20261030/hf:/hf:ro \
    -v /mnt/disk2_from_server2/siqizhu4/loop-spec-src/wavefront-decoding:/wfd:ro \
    -v $P/loop-suffix-spec-20261030/arctic-suffix/pkg:/arctic:ro \
    -v $P/loop-suffix-spec-20261030/runs:/runs:ro -v $OUT:/out \
    --entrypoint python lm-memory-sdft:sn8200-20261028-ready /repo/scripts/loop_suffix_spec_wallclock.py \
    --model $2 --runs-dir /runs/$3/$2 --out-dir /out/$3/$2 --prompt-format chat "${@:4}"
}
# 1) fp32核对：每类1题、64词，各臂必须与AR逐词一致
run check-ouro ouro-2.6b sb-chat --mode check --max-new-tokens 64 --max-prompt-tokens 3000
# 2) 正式测速（sb1024）
run ouro ouro-2.6b sb-chat --max-new-tokens 1024 --max-prompt-tokens 3000
run huginn huginn-3.5b sb-chat --max-new-tokens 1024 --max-prompt-tokens 3000
# 3) 停掉监控后汇总，按时间戳接入监控记录
run sum-ouro ouro-2.6b sb-chat --max-new-tokens 1024 --max-prompt-tokens 3000 --summarize --monitor /out/monitor.jsonl
run sum-huginn huginn-3.5b sb-chat --max-new-tokens 1024 --max-prompt-tokens 3000 --summarize --monitor /out/monitor.jsonl
```

chat-256用`--runs-dir /runs/full-chat/<model> --max-new-tokens 256 --max-prompt-tokens 512`。

## 7. 判读

- 主比较在本实现内部进行：`both`和`both-path`对比`shallow`与`ar`。WFD是外部参照，它的引擎开销不同，不与本实现的倍数直接相减。
- 如果`both`每次调用的耗时随行数上升，抵消了调用数上的优势，就说明树宽成本占主导，看`both-path`是否把行数压下来。
- 倍数只对应本机、eager PyTorch、不用CUDA graph、单请求的条件，不外推到服务框架。

<a id="wallclock-results"></a>
## 8. 结果（2026-10-31，04:57–13:02 UTC）

全部步骤正常退出，fp32核对中各臂与AR逐词一致。干扰标记按用户判断（[后续批次协议§6](../loop-wfd-same-engine-20261031/protocol.md)），两个模型都是60/60题无干扰。原汇总把容器内外进程号不同误判为外来进程，标记了全部题目；原文件保留，修正后的汇总为`sb-chat/<model>/bench-bfloat16-summary-userflags.json`。

| 臂 | Ouro 对AR | Ouro 对shallow | Huginn 对AR | Huginn 对shallow |
| --- | ---: | ---: | ---: | ---: |
| shallow | 2.60 | 1.00 | 2.39 | 1.00 |
| suffix | 1.55 | 0.60 | 2.03 | 0.85 |
| both | 3.57 | 1.37 | 3.52 | 1.47 |
| both-path | 3.59 | 1.38 | 3.57 | 1.50 |
| 参照：WFD自带引擎 | 3.82 | 1.47 | 3.31 | 1.38 |

- both-path每次调用的行数约为both的一半到三分之一（Ouro 13.5对28.0，Huginn 21.3对76.8）。Huginn上每次调用因此从6.82 ms降到6.35 ms；Ouro上两者相同（49.9对49.7 ms）。
- 跨引擎对WFD：both／both-path在Ouro为0.93／0.94倍，在Huginn为1.07／1.08倍。这个比值包含引擎差异；同一引擎下为1.34–1.40倍，见[后续批次报告](../loop-wfd-same-engine-20261031/report.md)。
- W&B：组`loop-suffix-spec-wallclock-20261031`，2026-10-31已同步并核验，见[wandb-sync.json](wandb-sync.json)。
