# DFlash 新引擎表一、表二重测（2026-11-12）

本轮使用仓库根目录 `baselines/` 中的 DFlash 实现及已完成训练的检查点，
新增真正的多请求目标验证与 drafter 批处理。正式目录为
[`results/dflash-table12-packed-20261112-v2/`](../results/dflash-table12-packed-20261112-v2/)。
无 `v2` 的目录仅用于准备检查，没有 GPU 测量。

**2026-11-12 八卡调度更新**：按用户要求，尚未启动的单元已交给
[`gpu-balanced-queue-20261112-v1`](../results/gpu-balanced-queue-20261112-v1/status.json)，
允许使用 GPU 0–7，优先补跑 LoopAhead/Ouro-2.6B/SpecBench/T=1 的锁冲突单元，
随后按历史 AR 耗时估计优先派发长任务，空闲卡自动领取。
Agentic 尚未完成的同卡比较组保留绑定，已有训练和测速进程正常完成。
原 DFlash 控制器暂停领取新单元，继续回收自己启动的工作进程并记录真实退出码；
这些进程全部退出后，新队列合并记录并接管原目录的状态和汇总。
过渡期间以链接中的共享队列状态及原 `dispatch.json` 合并查看，避免遗漏已迁移单元。
冻结推理源码、输入和参数保持原协议；调度修订见
[`plan.json`](../results/gpu-balanced-queue-20261112-v1/plan.json)。

**2026-11-12 调度器恢复**：原控制器停止后写出的状态不含 `active`，共享队列因此
触发 `KeyError`。已修复停止状态兼容性，并增加显式 `--resume`、进程身份校验、
启动凭据与队列状态核对。恢复时保留原进程及其 GPU 锁，未中断正在计时的任务。
新任务由独立监护进程等待真实退出并原子写入凭据，控制器退出后仍可恢复。

旧共享控制器崩溃后，部分工作进程已由系统接管，无法补领真实退出码。
即使 `complete.json` 和产物哈希通过，这类 attempt 仍保留为 `interrupted`，
使用新 attempt 完整重跑；不推断退出码为零、不拼接计时、不改动原结果。
已结束的 LoopAhead SpecBench T=1 的 `lockretry-v1` 同样保留，正式补跑使用
`lockretry-v2`。因此恢复后的正式完成数可能少于按完成文件统计的产物数。
恢复前快照、源码哈希和验证记录见
[`recovery-amendment.json`](../results/gpu-balanced-queue-20261112-v1/recovery-amendment.json)。

共享队列恢复入口（有活动控制器时由互斥锁拒绝重复启动）：

```bash
/opt/conda/bin/python -u results/gpu-balanced-queue-20261112-v1/controller.py --resume
```

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

初始调度使用2/3/4/6号卡；后续八卡分配遵循上面的调度修订。调度器等待已有任务释放设备，通过
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
