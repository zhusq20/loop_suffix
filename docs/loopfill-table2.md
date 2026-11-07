# SpecBench Table 2（2026-11-07）

本轮正式目录：[loopfill-table2-20261107-v2](../results/loopfill-table2-20261107-v2/)。
W&B：[loopfill-specbench-table2](https://wandb.ai/zsqzz/loopfill-specbench-table2)，正式 group 为 `loopfill-table2-20261107-v2`，`job_type=throughput`。
`setup` 和 `preflight` 不属于正式结果；无 `v2` 的早期目录只保留包装层调试与预检记录。

- 模型：Huginn-3.5B、Ouro-1.4B、Ouro-2.6B，使用注册表固定 revision。
- 方法：同引擎 AR、WFD、LoopSpec、Suffix-only、LoopFill-online。
- batch：1、2、4、8、16，共 75 组；完整 SpecBench 480 题的第一轮输入，每题最多生成 1024 token。
- 主指标 `output_tput`：生成 token 总数 / 从第一次请求进入到最终 CUDA 同步的墙钟时间。包括 prefill、decode、CPU 调度、后缀操作和本地逐请求持久化；不含模型加载、预热和 W&B 网络上传。
- 辅助指标 `total_tput`：(输入 token + 生成 token) / 同一墙钟时间。另保留耗时、生成量、内存峰值和持久化耗时。
- BF16，continuous/packed，prefill 每块 128 token；CUDA Graph、torch.compile 和执行行填充均关闭。
- 预热使用题序前 `min(2B,480)` 个请求，每题最多 32 个新 token，正式后缀历史冷启动。
- N=16、Q=4、model top-2、孩子 Q=(4,4)；online c₀=0.7、η=0.1、c∈[0.25,2]。N 不约束 WFD/LoopSpec 的独立调度规则。
- 输入最长为 Huginn 1577、Ouro 1588 token。三模型均事前固定 `max_seq=2688`，覆盖完整 prompt、1024 个生成 token 及草稿余量；没有截断题目或缩短输出上限。

八张 L20X 使用统一共享任务队列，长任务优先，空闲卡自动补位。同一模型/batch 的不同方法可以使用不同 GPU。每个任务独占一张固定 UUID GPU，绑定四个同 NUMA CPU 核，Torch/OMP/MKL 均单线程。GPU 为 Exclusive Process；SM 锁定 1980 MHz，应用显存频率 3201 MHz，功耗上限 700 W。每 15 秒记录实际频率、利用率、显存与温度。

2026-11-07 03:25（北京时间）修复共享 GPU 锁冲突：原调度器在锁忙时直接记为 `worker_failed` 并继续领取任务，造成67组尚未计算就启动失败。修复后先取得 `/tmp/loopfill-<UUID>.lock`、核查GPU资源，再从共享队列领取任务；锁忙或GPU被占用时等待并重试，不消耗任务。调度器将同一个锁文件描述符交给工作进程，避免交接时被另一队列抢占。任务结束后释放GPU，其他等待任务继续补位。

本次只替换调度器，接管了3个存活工作进程；5组完成结果保留，67组恢复 `pending`。冻结源码、输入、协议和计时方式未改动。修复调度器与修复前状态保存在 [`recovery/gpu-lock-v1/`](../results/loopfill-table2-20261107-v2/recovery/gpu-lock-v1/)，每个重新启动的单元另记 `scheduler.json` 及调度器哈希。原失败日志和任务中的 `launch_failures` 保留。表二、512-token延迟与batch sweep相关回归共21项通过，覆盖真实文件锁竞争、跨进程锁交接、存活进程接管及两队列共享锁后的空闲补位。

结果保存在 [measurements.csv](../results/loopfill-table2-20261107-v2/measurements.csv)、[table2.md](../results/loopfill-table2-20261107-v2/table2.md) 和 [status.json](../results/loopfill-table2-20261107-v2/status.json)。每完成一个请求，benchmark 追加 `*-completed.jsonl` 并 `fsync`；独立进程每约 5 秒将请求指标与文件同步 W&B。`runs/<cell>/attemptN/wandb/` 保留本地 W&B spool。密钥通过进程环境注入，不进入配置、源码、命令行参数或测量子进程。

源码、输入、依赖版本和资源配置已冻结在 [protocol.json](../results/loopfill-table2-20261107-v2/protocol.json)、`source/` 和 `runtime.json`。运行期间不要编辑冻结副本。OOM 单独登记，其他组合继续；不降低 batch 或预算重跑成成功结果。中断的单元使用新的 attempt 完整重跑，不拼接不同计时窗口或后缀历史。成功/OOM 的单元在恢复时保留。

恢复同一套任务时，将 `WANDB_API_KEY` 放入环境（不要放在命令行参数中），然后执行：

```bash
export PYTHONPATH=results/loopfill-table2-20261107-v2/source/src:results/loopfill-table2-20261107-v2/source/scripts:/tmp/loopfill-assets/ArcticInference
/tmp/loopfill-venv/bin/python \
  results/loopfill-table2-20261107-v2/recovery/gpu-lock-v1/loop_table2_suite.py run \
  --out results/loopfill-table2-20261107-v2
```

本批使用上面的修复调度器作为恢复入口，`PYTHONPATH` 仍使用原冻结源码。该入口检查原始源码和输入哈希，保留成功/OOM结果，接管仍存活的表二工作进程，并拒绝与存活的调度器重复启动。不要再使用 `source/scripts/loop_table2_suite.py run` 的旧入口。

仅汇总已有记录可将 `run` 换成 `summarize`。模型权重与运行环境位于本机 `/tmp`，实验结果和冻结源码位于持久化工作区；更换机器后须按固定 revision 重建环境与路径，不能直接混用新硬件计时。

本轮为固定 seed=0 的单次计时。BF16 下不同方法可能生成不同输出，吞吐结果不等同于任务质量或无损性验证。
