# DFlash 新引擎表一、表二重测（2026-11-12）

本轮使用仓库根目录 `baselines/` 中的 DFlash 实现及已完成训练的检查点，
新增真正的多请求目标验证与 drafter 批处理。正式目录为
[`results/dflash-table12-packed-20261112-v2/`](../results/dflash-table12-packed-20261112-v2/)。
无 `v2` 的目录仅用于准备检查，没有 GPU 测量。

## 范围与协议

| 项目 | 本轮配置 |
|---|---|
| 模型 | Huginn-3.5B（332,621,056 参数窄 drafter）、Ouro-2.6B（原完成训练的 drafter） |
| 表一 | 六任务、B=1、T=0/1、最多512新 token，共24单元、8,368请求 |
| 表二 | SpecBench480、T=0、B=1/2/4/8/16、最多1024新 token，共10单元、4,800请求 |
| 输入 | 从原表冻结文件验证哈希后复制；题目、chat template、题序和停止 token 沿用原协议 |
| 表一计时 | 同步后的 decode；排除 prefill/首 token、预热、文件写入和网络上传 |
| 表二计时 | 首次接纳请求至最终 CUDA 同步；包括128-token分块 prefill、调度及逐请求 fsync |
| 执行 | BF16，完整目标循环深度；Graph/compile/目标执行行填充关闭；单次计时 |
| 表二预热 | order-seed=0 打乱后前 min(2B,480) 请求，每题最多32 token |
| 参考分母 | 原表同主机 AR 历史记录；CSV明确命名 historical，不宣称新测 AR 或已校正跨日期漂移 |

Ouro-1.4B 没有可用的完成训练的 DFlash 检查点，因此本轮不含其单元，也不启动新训练。
旧表及旧 DFlash 原生后端结果保持历史记录。本轮新数值写入独立 CSV 和报告。

## 批处理实现

[`baselines/dflash/batch.py`](../baselines/dflash/batch.py) 提供 `BatchedDFlashTarget`
和 `batch_decode`。一个共享 KV 池为各请求分配互不重叠的视图；验证块作为各自的
因果链，共用一次 packed 执行。五组训练特征按请求和验证位置拆分，拒绝后分别
缩短可见 KV 长度。请求结束后槽位可以立即接纳下一请求。

drafter 将同宽验证块合成 batch。不同长度的上下文使用显式 SDPA mask；真实
target 特征才进入各请求的独立 DynamicCache，临时 noise KV 不保留。drafter
上下文有 padding，目标执行行没有 padding。记录 `max_verify_batch`、
`max_draft_batch`、`max_active_requests` 和 `refilled_requests`，可核查实际并发。

原单请求 `PackedDFlashTarget.forward` 与 `run_baseline.py infer` 的 B=1 接口保留。
表一使用单请求入口；表二由新批处理入口执行。

## 验证、调度与结果

启动前49项CPU回归通过，Ruff与diff检查通过，51个冻结源码文件和全部输入哈希
已核对。证据见正式目录的 `cpu-validation-final.log` 与
`implementation-validation.json`。启动时 GPU 验证和正式测量仍在等待设备，
不把CPU测试或队列启动视为GPU实验完成。

CPU回归覆盖两种目标结构的特征/logits、部分拒绝后的重算、不同上下文长度的
drafter KV、T=0/1 与串行参考对齐、EOS、输出预算、槽位复用及 B8/B16 缓存隔离。
每个模型先在真实目标权重和真实 drafter 上做 FP32 B2 串行/批处理对照，通过后
才能运行该模型的正式单元。每个表二单元在相同 batch 和上下文容量下预热。

GPU仅使用当前分配策略允许的2/3/4/6号卡。调度器等待已有任务释放设备，通过
`/tmp/loopfill-<UUID>.lock` 与其他队列互斥，锁文件描述符传给工作进程，控制器退出
不会提前释放仍在运行的 GPU 工作进程的锁。每单元绑定四个 NUMA 本地 CPU 核，
实际频率、显存和温度写入 `telemetry.jsonl`。

`status.json`、`dispatch.json` 和逐单元 `progress.json` 是运行状态依据。
只有退出码0、完整题号覆盖、逐题日志及汇总哈希通过审计的单元才标为 `complete`。
OOM保留真实失败，不降低 B、上下文或输出上限。中断单元使用新的 attempt 重跑，
不拼接吞吐计时窗口。`STOP` 文件只停止领取新任务，正在运行的任务继续完成。

W&B在独立进程中同步逐题指标和单元 artifact，不进入计时工作进程。
本机 spool 和原始 JSONL 即使网络上传失败也保留。项目为
`zsqzz/loopfill-dflash-packed-table12`；实际 run URL 见 `tracking.json`。

复核已有测量：

```bash
/tmp/loopfill-dflash-venv/bin/python \
  results/dflash-table12-packed-20261112-v2/source/scripts/loop_dflash_benchmark_suite.py \
  summarize --out results/dflash-table12-packed-20261112-v2
```

恢复队列时，将 `summarize` 改为 `run`，先检查 `launcher.json` 与现有进程。
控制器锁禁止重复启动；仍活着的旧工作进程必须先自然退出。恢复独立网络同步可用
`track` 子命令。新机器需要重新准备新目录，不把不同硬件时间追加到本轮。

本机网络同步进程使用已安装 W&B 的 `/tmp/loopfill-venv/bin/python`；推理工作进程
使用 `/tmp/loopfill-dflash-venv/bin/python`。二者解释器分别记在启动记录中。
首次 tracker 因推理环境未安装 W&B 而退出，日志保留为 `track.log`；恢复后的
跟踪记录为 `tracking-launcher-v2.json` 与 `track-v2.log`。该问题不影响 GPU 计时。

独立准备入口为 `scripts/loop_dflash_benchmark_suite.py prepare --help`。
通过 `--checkpoints MODEL=/path/to/checkpoint`、`--gpu-policy`、`--driver-wrapper`
显式指定本地资源。准备过程冻结源码、输入、参考表和检查点哈希。

BF16形状变化可能改变输出；同 seed 的随机解码不构成所有方法逐 token 相同的承诺。
本轮测速度和接受长度，不把速度结果解释为任务质量验证。
