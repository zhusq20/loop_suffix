# 独立运行 DFlash 和 LoopAhead

实现位于 `baselines/`。安装当前 `loop_suffix` 仓库即可使用，
运行时不需要 SZ 仓库、它的 `results/` 源码快照、第三方 checkout 或调度脚本。
模型权重、模型自带的 Hugging Face Python 实现、tokenizer、训练数据和已有 drafter
检查点仍是输入文件，通过参数传入。可以将这些产物放到任意目录。

根目录结构：

```text
/ossfs/workspace/loop_suffix/
├── run_baseline.py
├── baselines/
│   ├── train_dflash.py
│   ├── train_loopahead.py
│   ├── extract_features.py
│   ├── infer.py
│   ├── dflash/          # 包括核心适配器 packed_target.py
│   ├── loopahead/
│   ├── configs/
│   └── vendor/
└── tests/test_baselines/
```

新增基线实现已从 `src/llm_memory_editability` 移出；复用的 LoopFill 共享执行器
仍在本仓库原有的 `src` 包中。

| 入口 | 用途 | 目标执行方式 |
|---|---|---|
| `train-dflash` | DFlash 训练、保存、断点恢复；支持单进程和 torchrun DDP | 冻结目标模型的无缓存 HF 前向，提取五组训练特征 |
| `extract-features` | 为 LoopAhead 提取真实浅层状态和冻结 embedding/head | 冻结目标模型的浅层 HF 前向 |
| `train-loopahead` | LAD 训练、dev 选点、confirmation 检查、断点恢复 | 读取浅层状态，无需加载完整目标权重 |
| `infer` | AR / DFlash / LoopAhead 的逐题生成和 decode 计时 | 默认本仓库 `PackedLoopExecutor` + `SharedKVPool` |

`baselines/dflash/packed_target.py` 是**推理适配器**。训练入口单独提供；不能把这个 B=1、
`inference_mode` 的适配器直接作为训练前向使用。DFlash 的训练与推理 drafter
采用同一检查点张量格式，分别使用本地收录的 SpecForge / DFlash 模型实现。
Huginn 默认配置为 **332,621,056** 个可训练参数，五层、hidden2048、FFN5632；
不包括冻结目标模型。Ouro 默认沿用五层、目标宽度的 drafter。

上游代码的固定版本、来源文件哈希及许可证在
`baselines/vendor/ORIGINS.json`、`LICENSE.dflash`、`LICENSE.SpecForge`；
不需要安装完整 SpecForge、SGLang 或 DeepSpeed。

## 安装与环境

```bash
cd /path/to/loop_suffix
python -m pip install '.[baselines,dev]'
loop-baseline --help
```

在仓库根目录可直接运行 `python run_baseline.py --help`，将后面的 `loop-baseline`
替换成 `python run_baseline.py` 即可。也支持 `PYTHONPATH=src python -m baselines ...`。
包内固定 Transformers **4.57.1**，torch 要求 `>=2.6,<3`。本次 CPU 训练/恢复/
推理测试使用 Python **3.10.16**、torch **2.6.0+cu126**、SDPA。
先前全尺寸 DFlash 训练使用 torch **2.9.1+cu128**、FlexAttention；
先前推理使用 torch **2.6.0+cu126**。这是两个已使用过的环境，不能把不同环境
计时混成同栈比较。新入口每次记录实际版本和本地源码哈希。

自行选择设备，例如 `CUDA_VISIBLE_DEVICES=2` 后传 `--device cuda`；代码没有
机器专属 GPU 白名单或驱动目录。当前共享服务器的 GPU 预留与 native-driver
环境修正应由启动环境提供。CPU 小检查用 `--device cpu --dtype float32`，
DFlash 训练再加 `--attention sdpa`。默认训练 FlexAttention 面向 GPU。

## DFlash 训练

提供 JSONL，每行至少包含下面三个字段。token ID 必须属于同一个目标 tokenizer；
prompt 不参与训练 loss，DFlash 会排除 continuation 不足32 token 的样本。

```json
{"source_id":"train/0001","prompt_tokens":[1,2,3],"output_tokens":[4,5,6]}
```

以上仅展示字段，实际 DFlash continuation 须至少32 token。也可直接传入现有
`shard*/teachers.jsonl` 的根目录；存在旧 metadata 时会验证完成记录和内容哈希。
`--manifest /data/manifest.json --stage dflash-train` 可额外要求完整覆盖指定 split。

```bash
loop-baseline train-dflash \
  --model huginn-3.5b --target-path /models/huginn \
  --teachers /data/dflash-train.jsonl --out /runs/dflash-332m \
  --device cuda --dtype bfloat16 --attention flex_attention \
  --micro-batch 4 --accumulation 1 --epochs 6 --checkpoint-layers
```

默认直接读取包内 `configs/dflash-huginn-332m.json`，不需要手动配置窄 drafter。
可用 `--draft-config` 显式覆盖；入口验证目标宽度、特征层、词表、block16 和五层结构。
正式配方沿用512 anchors、loss decay7、AdamW FP32 master、lr6e-4、warmup4%、clip1。
可用 `--max-updates 1` 做有限步检查，该状态会记录为 bounded experiment。

```bash
torchrun --standalone --nproc_per_node=2 \
  -m baselines.train_dflash \
  --model huginn-3.5b --target-path /models/huginn \
  --teachers /data/dflash-train.jsonl --out /runs/dflash-ddp \
  --micro-batch 4 --accumulation 1 --checkpoint-layers
```

输出 `checkpoint-XXXXXX/config.json`、`model.safetensors`、FP32 master、optimizer
和各 rank RNG 状态。`latest.json` 指向最近检查点。用原命令追加
`--resume /runs/dflash-332m/checkpoint-000250` 可恢复本入口保存的训练状态。
`--stop-after N` 保存后中断，保持原学习率计划，便于恢复检查。
历史 drafter 权重可用于推理；历史训练状态不宣称可无缝恢复，因源码/运行身份不同会拒绝。

## LoopAhead 特征与训练

三个 split 使用互不重叠的 `source_id`，同样采用 prompt/output token JSONL。
可额外提供 `pool` 表示数据域；不提供时归入 `default`。

```bash
loop-baseline extract-features \
  --model huginn-3.5b --target-path /models/huginn \
  --teachers /data/loopahead-train.jsonl \
  --features /data/lad-features --stage loopahead-train
```

对 dev、confirmation 文件分别再执行，`--stage` 改成 `loopahead-dev` 和
`loopahead-confirmation`。产生 `FEATURES/manifest.json` 及
`FEATURES/loopahead-STAGE/MODEL/shard000/`，保存逐 token 的 BF16 浅层状态、
冻结 embedding/head、token 频次和校验哈希。Huginn 提取 d1=4，Ouro 提取 d1=1。
不会补造不足64个真实状态的上下文。

```bash
loop-baseline train-loopahead \
  --model huginn-3.5b --target-path /models/huginn \
  --features /data/lad-features --out /runs/loopahead \
  --device cuda --dtype bfloat16
```

已有完整特征可以直接传入根目录，并用 `--manifest` 指定配套 manifest。
如显式提供 `--stop-ids`，训练无需读取目标 tokenizer/config。
默认 width512、两层、context64、future7、三轮、每请求32个 anchors、有效 batch64，
沿用原 LAD 学习率和优化器设置。每500步及结束时做 dev 选点，选定后只做一次
confirmation。`complete.json` 的 `selected_checkpoint` 指向可推理的 `lad.pt`。
数据域数量由实际数据决定；哈希、split 覆盖、重复 ID、跨 split 泄漏都会检查。

## 推理和逐题测试

输入 JSONL 支持 `source_id` + `prompt_tokens`，或者 `source_id` + `prompt` 文本。
文本默认 raw；需要 chat template 时显式加 `--prompt-format chat`。
传 token IDs 且显式给出 `--stop-ids` 时，不需要 tokenizer。

```bash
loop-baseline infer \
  --model huginn-3.5b --target-path /models/huginn \
  --method dflash --checkpoint /runs/dflash-332m/checkpoint-000250 \
  --inputs /data/test.jsonl --out /runs/test-dflash \
  --max-new-tokens 512 --temperature 0 --backend packed

loop-baseline infer \
  --model huginn-3.5b --target-path /models/huginn \
  --method loopahead --checkpoint /runs/loopahead/checkpoint-000500/lad.pt \
  --inputs /data/test.jsonl --out /runs/test-loopahead \
  --max-new-tokens 512 --temperature 0

loop-baseline infer \
  --model huginn-3.5b --target-path /models/huginn \
  --method ar --inputs /data/test.jsonl --out /runs/test-ar \
  --max-new-tokens 512 --temperature 0 --backend packed
```

支持 T=0/1。DFlash/AR 可用 `--backend native` 对照原目标 HF 前向；LoopAhead
使用 packed 后端。LoopAhead 校准可通过 `--admission` 传入，验证 model 和
LAD 权重哈希；未传时使用基础调度，不应标记为已校准结果。

每次输出 metadata、`outputs.jsonl` 和完成汇总，保留实际 token、采样/解码统计及
源码哈希。计时排除 prefill/首 token，无单独预热，因此是逐题诊断数据。
这套入口当前为 B=1、Graph 关闭；不包含原六任务评分、全量正式表聚合或多卡队列。
需要正式速度比较时应另设同 GPU、预热、轮换顺序和任务评分协议。

## 回归验证

```bash
python -m pytest -q tests/test_baselines tests/test_loop_dflash_target.py
```

这些测试不读取目标权重、不访问网络、不需要 SZ checkout 或 GPU：

- 两种 DFlash 结构的实际 optimizer 更新、HF 格式保存/重载、断点恢复逐参数一致。
- 默认 Huginn 参数量精确为332,621,056，冻结目标没有梯度或权重变化。
- 两模型的 LoopAhead 特征提取、训练、held-out 选点和推理入口。
- Packed greedy 对齐 AR，T0/T1 输出预算、EOS、拒绝采样分布、KV crop 和缓存隔离。
- 训练特征与标签对齐、数据文件损坏检测、LAD 因果性和调度依赖。

2026-11-12：将 wheel 安装到仓库外目录，并通过 Python audit hook 禁止读取
SZ 目录和当前仓库 `src/` 后，上述29项测试全部通过（包括两种训练的中断恢复）。
wheel 内29个基线文件与本地源码/配置/许可证逐字节一致，CLI 帮助、Ruff 和
diff 检查通过。记录及可移植 wheel 位于
`results/standalone-baselines-20261112/`。本次没有重新启动全尺寸 GPU 训练，
也没有重跑正式性能表；DDP/FlexAttention 的完整 GPU 训练仍需在目标环境验证。

同日按目录要求迁至根目录 `baselines/`，新增根目录启动器 `run_baseline.py`。
迁移后33项相关回归通过；重新打包并禁止访问源码 checkout 后，29项独立测试通过，
安装包中的30个基线文件与根目录实现一致，且不含旧命名空间下的基线代码。
迁移验证及新 wheel 保存在上述结果目录的 `root-layout/` 子目录。

旧 `scripts/loop_dflash_packed_probe.py` 是此前连接 SZ 冻结实验的**历史诊断入口**；
它没有被用于上述独立运行流程。旧报告及训练结果保持原义。
