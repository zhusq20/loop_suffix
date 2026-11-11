# DFlash 的 LoopFill 目标后端

**独立运行入口已提供**：当前仓库的训练、推理和测试命令见
[独立运行 DFlash 和 LoopAhead](standalone-baselines.md)。下文 probe 命令保留为
此前接入 SZ 冻结实验的历史诊断流程；独立入口不需要它。

`baselines/dflash/packed_target.py` 将现有 DFlash 的 target 接口接到
`PackedLoopExecutor`、`SharedKVPool` 和 `OuroLoop` / `HuginnLoop`。Drafter 权重、
block16、拒绝采样规则和目标完整循环深度保持原配置；不需要重新训练。

验证块作为一条因果链执行。中间特征从与训练对应的 block 输出提取：Ouro 使用
最后一轮的五个指定层，Huginn 使用 prelude、最后一轮 core 和 coda 的五个指定层。
所有验证位置先提交到预分配 KV；拒绝后缩短可见长度，后续提交覆盖被拒绝位置。
缓存带所有者检查，开始新请求后旧缓存失效。当前仅支持 B=1、无 padding、
Graph/compile 关闭；不提供多请求并发或无缓存训练接口。

## 验证与试跑

基础回归：

```bash
PYTHONPATH=src python -m pytest -q tests/test_loop_dflash_target.py
```

实际 DFlash 依赖、原始模型类和检查点位于现有基线实验目录。探针默认加载该目录
中原表1冻结的执行器代码，新增适配器单独加载，因此不会修改冻结源码。
`--engine-source` 可显式选择其他源码目录，每次运行记录实际来源及哈希。

```bash
/tmp/loopfill-dflash-venv/bin/python scripts/loop_dflash_packed_probe.py \
  --tiny --out <新的CPU结果目录>

/tmp/loopfill-dflash-venv/bin/python \
  /ossfs/workspace/sz/loopfill/experiments/baselines/native_cuda_env.py \
  --record <新的驱动环境记录.json> -- \
  /tmp/loopfill-dflash-venv/bin/python scripts/loop_dflash_packed_probe.py \
  --gpu 2 --cpus 8-11 --out <新的GPU结果目录>
```

GPU 探针遵守现有 GPU 白名单和共享锁，等待指定 GPU 空闲；不会终止其他作业。
默认每模型使用冻结 GSM8K、HumanEval+ 各前两题，每题最多128个新 token。
先用完整目标权重验证 FP32 logits、五组特征及回退重算，再在同一 GPU、同一
BF16 环境下轮换执行 native AR、packed AR、native DFlash、packed DFlash。
两种 DFlash 使用同一最终检查点。预热不计时，报告 decode token/s、时间、γ、
输出长度和整序列一致数。

这些是独立诊断，不是完整正式评测。BF16 形状变化可能改变输出；不能把短样本
速度当作六任务结论，也不能直接覆盖旧表的数值。原始实验和训练产物保持独立。
本次记录位于 `results/dflash-packed-backend-20261112/`，以各子目录 `report.json`
及实际进程退出记录为完成依据。

## 2026-11-12 验证结果

四项缓存回归及32组原始模型类的小尺寸解码对照通过。完整目标权重的 CPU/GPU
FP32 logits、五组中间特征和部分拒绝后的重算检查通过，GPU 上两侧 top1 一致。
真实 GPU 进程退出0，32条测量记录及汇总、源码快照哈希通过核对，临时 GPU2
调度预留已释放。以下仅为每模型四题、max128、T0 的同 GPU 原始测量：

| 模型 | HF AR token/s | Packed AR token/s | HF DFlash token/s | Packed DFlash token/s | Packed DFlash 相对 Packed AR 的延迟加速 |
|---|---:|---:|---:|---:|---:|
| Huginn-3.5B，332.6M drafter | 11.04 | 7.02 | 30.63 | 25.33 | 3.61× |
| Ouro-2.6B，277.9M drafter | 11.67 | 6.25 | 44.11 | 24.58 | 3.93× |

Packed DFlash 与 Packed AR 的整序列一致数分别为4/4、3/4；HF DFlash 与
Packed DFlash 分别为3/4、3/4。FP32短输入检查不代替这些 BF16 分歧的逐位置诊断。
本样本支持接口可用，不支持“换成 packed 后端必然更快”的判断。Huginn 四方法
均产生448个 decode token；Ouro 的 HF AR 为443个，其余方法为441个。
原始数据见 `comparison.csv`、`gpu-v1/report.json` 和各模型逐题 journal，
最终核对记录为 `validation.json`。
