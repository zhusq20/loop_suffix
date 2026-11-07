# Table 1：B=1、512-token decode latency 主实验

2026-11-07 创建。用户确认三个模型、六个数据集，Ours 只保留 online。

| 项目 | 冻结设置 |
| --- | --- |
| 模型 | Huginn-3.5B、Ouro-1.4B、Ouro-2.6B；revision 沿用模型注册表 |
| 方法 | AR、同引擎 WFD、同引擎 LoopSpec、Suffix-only、LoopFill-online |
| 数据 | GSM8K 300、MATH-500 500、BBH 270（27任务各10）、HumanEval+ 164、MBPP+ 378、Spec-Bench 480首轮 |
| 生成 | B=1、greedy、BF16、最多512新token，EOS/chat-end提前停止 |
| 后端 | packed attention；CUDA Graph、torch.compile、执行行填充均关闭 |
| 融合配置 | N16、Q4、top2、孩子Q(4,4)、online c0=.7、eta=.1、范围[.25,2] |
| 预热 | 每单元前2条输入各最多32词，使用独立缓存，不进入正式历史 |
| 规模 | 90单元，31,380次正式请求生成；每单元单次计时 |

输入由 `scripts/loop_latency_inputs.py` 冻结。GSM8K 按 SHA256(seed:sample:dataset:source_id) 排序取300，BBH 各任务独立取10，seed=20261107。执行顺序使用独立的 order 哈希。所有模型与方法使用同一题目和题序。

GSM8K 使用 harness 的前三个固定 CoT 示例；MATH 使用四个固定 Minerva 示例；BBH 使用官方各任务的三个 CoT 示例；代码使用 EvalPlus 发布的 prompt。全部经各模型发布的 chat template 包装。当前只用 EOS/chat-end 停止，不额外使用 few-shot 任务的文本分隔符。因此这是本项目独立冻结协议，不是 LoopSpec/LoopAhead 原设置的逐项复现。输入清单记录原始来源、revision、哈希及题号。

## 计时和指标

在 prefill 执行完并同步 GPU 后开始墙钟计时，至 decode 完成、suffix 完成输出发布及最终 GPU 同步为止。包括 suffix 建树、检索、更新、模型执行和 Python 调度。加载、预热、JSONL写入、fsync、W&B网络上传均在计时外。预填充产生的首token不进入decode token计数。

- latency speedup = 同一数据集 AR decode时间之和 / 方法 decode时间之和。
- 同时报告 pooled decode token/s 的比值和输出token数量比，不将生成长度差异解释为同输出加速。
- gamma = 完整循环深度 × decode token总数 / 实际loop steps总数。AR必须为1。
- 记录每题长度、停止原因、达到512上限的请求数、完整token输出和各解码器计数。上限停止不等价于已证明答案不完整。

每个方法/数据集从空共享历史开始，只发布已完成的选中请求输出。预热、标准答案、未入选请求输出均不进入后缀历史。BBH打乱后的任务顺序对所有方法一致。

## 队列、保存和恢复

本批目录为 `results/loopfill-latency512-20261107/`，入口：

- `protocol.json`：冻结配置、源码、输入哈希和GPU/CPU分配。
- `status.json`、`jobs/*.json`：实时状态；`queue.log` 与 `logs/*.log`：运行日志。
- `runs/<单元>/requests.jsonl`：每完成一题立即 append、flush、fsync。
- `measurements.csv/json`：定期汇总；AR完成后自动生成对应speedup。
- `validation/`：每模型/方法先运行两题、16-token短验证，成功后才允许对应正式单元。
- W&B项目：<https://wandb.ai/zsqzz/loopfill-latency512>。

统一使用八张GPU，每单元绑定一个GPU UUID、四个NUMA本地CPU核。共享 `/tmp/loopfill-<UUID>.lock` 与既有表二队列互斥，并检查真实GPU进程、显存和利用率。任何空闲卡均可领取下一单元，不做跨方法GPU配对。每5秒检查空闲资源，每15秒刷新队列汇总及遥测。

2026-11-07核查：本批实际运行的冻结调度器已经在领取任务前非阻塞取得共享锁；锁忙时跳过该卡，任务保持pending，并在取得锁后再次核查真实GPU进程。锁由调度器持有直到该任务退出，再释放供两个队列使用。表二锁修复期间本批调度器与工作进程持续运行。新增回归直接执行延迟队列循环，用表二锁模拟竞争，验证被占用卡不消耗任务、其他空闲卡继续工作、释放后自动补位及任务仅执行一次。受测调度器与本批冻结副本逐字节一致，冻结源码18个文件的哈希全部通过。

2026-11-07 02:01（北京时间）本批队列已启动，PID见 `queue-process.json` 和 `launcher.json`。当时八卡正在运行另一套表二实验，因此本批90单元排队，尚无正式结果。已有表二是独立1024-token吞吐协议，本批没有停止或改写它，结果不可混合。实时状态以 `status.json` 和进程为准。

每题W&B指标在计时外提交，逐题JSONL配置实时文件同步；结束后上传完整单元artifact。持久目录同时保留W&B本地缓存。网络中断不会删除本地结果；不要只依赖云端汇总。

恢复使用冻结源码，先确认原控制器和GPU进程状态，不能重复启动存活控制器：

```bash
/tmp/loopfill-venv/bin/python \
  results/loopfill-latency512-20261107/source/scripts/loop_latency_suite.py run \
  --out results/loopfill-latency512-20261107
```

中断单元保留逐题前缀，只重建已完成请求的suffix历史，继续剩余请求。完成单元不会重跑。损坏的最后一条未完成写入可截去；前缀ID、输入及源码哈希不匹配则报错。失败单元保留错误并继续其他单元，不静默缩短输出或删题。

当前解释器、模型下载缓存与 suffix 扩展在 `/tmp/loopfill-venv`、`/tmp/loopfill-assets`，是可重建运行依赖，不是持久数据。机器重建后需按记录的revision恢复模型、环境和私有W&B认证文件；不能修改冻结协议来接受不同输入/源码。认证文件为本机600权限文件，不提交密钥。

## 验证

全部18个模型/数据集组合的上下文检查通过，未删题、未截断prompt。CPU回归72 passed、12 skipped；新增测试覆盖prefill计时边界、suffix发布计时、崩溃日志恢复、确定性抽样、五方法矩阵和汇总指标。GPU回归曾因其他队列启用独占模式而无法获得设备，未作为代码失败或验证通过证据；真实检查点短验证已排在正式单元前。初次数据准备遇GitHub匿名API限流，失败目录保留；随后使用核实的固定commit拉取，六数据集准备成功。

## 最终完成状态（2026-11-08）

90/90单元完成，31,380/31,380请求。Ouro-2.6B / SpecBench / LoopFill-online原已完成480请求，但在`run.log_artifact`阶段遇W&B服务超时，未写完成标记。收尾时核对冻结源码与输入哈希、逐题题序和prompt哈希，并从原始记录复算所有汇总指标；恢复原run并补传artifact，下载验证三个文件SHA256一致后才写完成标记及更新队列。没有重新生成请求或替换计时。原始退出码、失败次数与原job/status快照保留在`recovery/upload-finalization/`。

最终[表格](../results/loopfill-latency512-20261107/table1.md)报告各方法/数据集的延迟加速比与γ，GM为六数据集加速比的等权几何均值。[上传核验](../results/loopfill-latency512-20261107/recovery/upload-finalization/wandb-verification.json)记录云端artifact与哈希。
