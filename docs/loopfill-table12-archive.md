# 表 1、表 2 实验归档（2026-11-08 更新）

2026-11-08最终归档：表1全部90组、31,380条请求完成；表2全部75组结束，其中74组成功、Huginn/WFD/B16为实测OOM。已纳入完整逐题记录、最终汇总、Markdown/LaTeX表格及上传恢复证据。表1失败单元补传到原W&B run并下载核验SHA256；没有重新生成或修改计时。下文2026-11-07的数量与测试结果为首次快照历史。`running` 和进程 PID 表示原机器采集时的状态，不表示克隆仓库后存在对应进程。

| 表格 | 正式目录 | 制表入口 | 协议 |
| --- | --- | --- | --- |
| 表 1：B=1、最多512-token decode latency | [loopfill-latency512-20261107](../results/loopfill-latency512-20261107/) | [measurements.csv](../results/loopfill-latency512-20261107/measurements.csv)、[measurements.json](../results/loopfill-latency512-20261107/measurements.json) | [运行说明](loopfill-latency512.md)、[protocol.json](../results/loopfill-latency512-20261107/protocol.json) |
| 表 2：SpecBench480、B=1/2/4/8/16、最多1024-token吞吐 | [loopfill-table2-20261107-v2](../results/loopfill-table2-20261107-v2/) | [table2.md](../results/loopfill-table2-20261107-v2/table2.md)、[measurements.csv](../results/loopfill-table2-20261107-v2/measurements.csv) | [运行说明](loopfill-table2.md)、[protocol.json](../results/loopfill-table2-20261107-v2/protocol.json) |

归档读取时间为2026-11-07 08:31–08:32（北京时间），包含948个实验文件，约55 MiB。所采集状态中，表 1 完成8/90组（另有8组running、74组pending），表 2 完成14/75组（61组pending）。完成情况分别见各目录的 `status.json` 与 `jobs/*.json`；运行中日志和周期汇总按文件读取，各自更新时间可能略有差异。只将 `complete` 单元纳入正式数值，保留尚未完成和失败状态。两张表的输出长度上限和计时范围不同，不混用结果。

## 文件范围

- `scripts/loop_latency_inputs.py` 冻结六个数据集的输入；`loop_latency_worker.py` 执行单个延迟单元；`loop_latency_suite.py` 管理队列、恢复和表 1 汇总。
- `scripts/loop_table2_suite.py` 管理表 2 的准备、预检、队列、逐请求持久化和汇总；`loop_batch_bench.py` 提供实际吞吐测量及持久化接口。公共解码代码位于 `src/llm_memory_editability/`。
- 两个正式目录均包含协议、冻结输入、源码快照、任务状态、已产生的逐题输出、汇总、验证和日志；表 1 另含输入来源与固定题序清单，表 2 另含环境记录及 `recovery/gpu-lock-v1/` 修复入口。
- 提交前核对了协议记录的全部120个源码哈希及两组冻结输入哈希，并检查了归档 JSON/JSONL 的可解析性。

模型权重、Python环境、编译产物、W&B本地缓存、临时文件、进程锁和认证密钥不入库。表 2 早期准备失败与包装层调试目录不属于正式结果；正式目录内的预检、失败尝试和恢复记录保留。新实验输出仍默认由 `.gitignore` 忽略，本次显式纳入的文件后续可以正常提交更新。

## 从现有记录重新汇总

在独立克隆或离线副本的仓库根目录运行以下命令；汇总不加载模型、不启动实验，也不需要 W&B认证。它会更新副本中的汇总和状态文件，不要在正在运行的原实验目录并发执行。

```bash
python scripts/loop_latency_suite.py summarize \
  --out results/loopfill-latency512-20261107

python scripts/loop_table2_suite.py summarize \
  --out results/loopfill-table2-20261107-v2
```

表 1 使用各已完成单元的 `summary.json`，其逐题依据为 `runs/<单元>/requests.jsonl`；相对AR指标需相同模型、数据集的AR单元完成。表 2 使用 `jobs/*.json` 中经审计的单元汇总，其依据为 `runs/<单元>/attemptN/` 的最终 JSONL、逐题持久化 JSONL、summary及metadata。部分请求日志用于恢复和核查，不补写为已完成单元。

## 复现与验证

重新执行实验所需的模型名称及固定revision、输入哈希、源码哈希、GPU设置和本机路径在协议中。依赖版本和ArcticInference版本另见表 2 的 [runtime.json](../results/loopfill-table2-20261107-v2/runtime.json)。协议里的绝对路径、GPU UUID与CPU亲和性来自原机器；更换机器须重建环境和新实验目录，不能将新硬件计时追加到本次结果。

原机器恢复入口见各表运行说明。表 2 当前批次必须使用已归档的 `recovery/gpu-lock-v1/loop_table2_suite.py`；其原冻结源码保持不变。认证通过运行环境提供，密钥不随仓库分发。

提交前在暂存文件的独立副本运行了两条离线汇总命令，重新生成的两份 `measurements.csv` 均与归档逐字节一致。

对延迟、表 2、batch sweep、batch decode和continuous batching回归执行CPU验证：**68 passed、14 skipped**。14项为CUDA测试；本次验证显式屏蔽GPU，不占用正在执行实验的设备。命令如下：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src:scripts:/tmp/loopfill-assets/ArcticInference \
  /tmp/loopfill-venv/bin/python -m pytest \
  tests/test_loop_latency_suite.py tests/test_loop_table2_suite.py \
  tests/test_loop_batch_sweep.py tests/test_loop_batch_decode.py \
  tests/test_loop_continuous_batch.py -q
```

## 最终归档验证

表1核对全部90份逐题记录的完成哈希、冻结输入题序及prompt哈希，并复算全部汇总指标（31,380请求）。表2核对75组CSV状态，成功组的吞吐、墙钟时间、输出量和请求数与job汇总一致。W&B队列已核验为finished、90 complete、0 failed。最终表格由 `python scripts/loop_final_tables.py` 生成；完整核验记录见 `results/loopfill-latency512-20261107/recovery/upload-finalization/final-validation.json`。云端artifact下载副本不重复入库，保留哈希核验记录。
