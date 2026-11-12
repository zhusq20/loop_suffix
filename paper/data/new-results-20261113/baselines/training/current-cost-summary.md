# 草稿器实测成本

阶段性报告：尚未完成全部正式训练。

GPU小时为各GPU实际占用时间之和，包括该作业内的加载、预热、保存及验证；不是纯kernel时间。只累计已结束的reservation记录或有真实起止记录的占卡时段，运行中作业不作估算。生成、特征提取、训练分别报告，导出副本不重复计费。

| 方法 | 模型 | 完整分片数据条数/目标 | 生成GPU小时 | 特征GPU小时 | 训练GPU小时 | 更新步数 | 状态 |
|---|---|---|---:|---:|---:|---:|---|
| dflash | huginn-3.5b | train 50000/50000 | 23.727772 | —（在线计算，计入训练） | 24.489464 | 8886 | complete |
| dflash | ouro-2.6b | train 50000/50000 | 19.719691 | —（在线计算，计入训练） | 12.010793 | 9030 | complete |
| dflash | ouro-1.4b | train 50000/50000 | 12.060123 | —（在线计算，计入训练） | NA | NA | pending |
| loopahead | huginn-3.5b | train 9025/9025; dev 782/782; confirmation 782/782 | 11.704820 | 0.401170 | 0.199502 | 13539 | complete |
| loopahead | ouro-2.6b | train 9025/9025; dev 782/782; confirmation 782/782 | 4.853005 | 0.147511 | 0.148247 | 13539 | complete |
| loopahead | ouro-1.4b | train 9025/9025; dev 782/782; confirmation 782/782 | 2.456944 | 0.083524 | 0.146427 | 13533 | complete |

未完成行中的数字仅为已完成分片的实测小计，不代表该模型的总成本。

Ouro1.4B DFlash使用父进程记录的生成/训练起止时间，包含进程加载、保存及退出观察等待；教师内部reservation已包含在该区间，不重复相加。最长32条的一步试跑、CPU审计期间仍保留的GPU及阶段间空档单列于额外费用。全流程范围以获得GPU后selection记录到完成标记时间为界；未记录的最后写标记至释放锁间隔不计入。

数据清单与去重：

- `dflash-train`：50000条；codealpaca 5000、nemotron_code 10000、nemotron_math 20000、ultrachat 15000；SHA256 `01375c1058ef2611418ef697a4f1008091360cb4ebc6156ab8a846cbb095ddb4`。
- `loopahead-train`：9025条；gsm8k 2000、math 2000、python 2000、reasoning 2025、ultrachat 1000；SHA256 `7bfb0f750708ad9b09d0f700f913eb849eba20984d3c6f645352f0933de4ae34`。
- `loopahead-dev`：782条；gsm8k 128、math 128、python 128、reasoning 270、ultrachat 128；SHA256 `74e437e92b451867ce705b774d0b97cc201bf56f805ef13d1e7ad6b954db6682`。
- `loopahead-confirmation`：782条；gsm8k 128、math 128、python 128、reasoning 270、ultrachat 128；SHA256 `34885fc01386ab9c56cdaf1a554e0bd89d2e6d3805e0a6bb90e8b0dbbaf37798`。

保护10392个去重测试问题；精确测试/跨方法heldout重叠为0，核验61个源文件哈希。近重复规则见data/manifest-audit.json及清单保护策略；不据此宣称排除了所有语义改写。原始HumanEval补充隔离检查也通过。

Nemotron V2受限而使用公开V1；缺失代码按TACO训练元数据恢复，对话份额由UltraChat补足。LoopAhead的BBH训练份额替换为非测试reasoning数据。详细来源、版本与适配理由见DESIGN.md和loopahead/SPEC.md。

已完成训练的记录：

- dflash / huginn-3.5b：有效训练样本47387，6 epochs，2张GPU，8886次更新；选择`final narrow Huginn checkpoint;332621056 parameters`；rank0峰值allocated显存28501811200字节。
  有效样本来源：codealpaca 3470、nemotron_code 9884、nemotron_math 19900、ultrachat 14133。
- dflash / ouro-2.6b：有效训练样本48137，6 epochs，2张GPU，9030次更新；选择`final checkpoint`；rank0峰值allocated显存41377969152字节。
  有效样本来源：codealpaca 3857、nemotron_code 10000、nemotron_math 19670、ultrachat 14610。
- loopahead / huginn-3.5b：有效训练样本9025，3 epochs，1张GPU，13539次更新；选择`checkpoint-013500`；rank0峰值allocated显存1875459584字节。
  有效样本来源：gsm8k 2000、math 2000、python 2000、reasoning 2025、ultrachat 1000。
- loopahead / ouro-2.6b：有效训练样本9025，3 epochs，1张GPU，13539次更新；选择`checkpoint-013000`；rank0峰值allocated显存735053312字节。
  有效样本来源：gsm8k 2000、math 2000、python 2000、reasoning 2025、ultrachat 1000。
- loopahead / ouro-1.4b：有效训练样本9022，3 epochs，1张GPU，13533次更新；选择`checkpoint-013533`；rank0峰值allocated显存735053824字节。
  有效样本来源：gsm8k 2000、math 2000、python 1999、reasoning 2025、ultrachat 998。

失败与额外试跑（不混入上表成功作业）：

Huginn旧宽版中断训练保留原始failed退出/成本记录，原因是用户资源与架构调整；其费用单列，不转记为窄版成功训练。

| 作业 | 状态 | GPU小时 |
|---|---|---:|
| `/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3/smoke-exit.json#smoke` | complete | 0.012540 |
| `/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3/smoke-launcher.json#before_smoke` | complete | 0.001213 |
| `/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3/teachers-launcher.json#before_teachers` | complete | 0.000087 |
| `/ossfs/workspace/loop_suffix/results/dflash-ouro-1.4b-training-20261112-v3/train-launcher.json#before_train` | complete | 0.000021 |
| `checkpoints/dflash-huginn-3.5b-formal-v1` | failed | 22.947959 |
| `checkpoints/dflash-huginn-micro4-memory-smoke-v1` | complete | 0.061968 |
| `checkpoints/dflash-huginn-narrow-ddp2-smoke-v1` | complete | 0.031748 |
| `checkpoints/dflash-huginn-narrow-micro4-smoke-v1` | complete | 0.010887 |
| `checkpoints/dflash-huginn-narrow-pilot-v1` | complete | 0.167468 |
| `checkpoints/dflash-huginn-training-smoke-v1` | complete | 0.032955 |
| `checkpoints/dflash-ouro-2.6b-pilot-v1` | complete | 0.108154 |
| `checkpoints/dflash-ouro-2.6b-smoke-v1` | complete | 0.006974 |
| `data/teachers/dflash-pilot/ouro-2.6b/shard0` | complete | 0.258746 |
| `data/teachers/dflash-pilot/ouro-2.6b/shard1` | complete | 0.261753 |
| `data/teachers/dflash-pilot/ouro-2.6b/shard2` | complete | 0.261288 |
| `data/teachers/dflash-pilot/ouro-2.6b/shard3` | complete | 0.244222 |
| `data/teachers/failed/huginn-dflash-train-shard6-gpu-contention-v1` | failed | 0.009553 |
| `data/teachers/failed/huginn-loopahead-train-kv70-shard0` | failed | 0.044838 |

全部已结束的生成/特征/训练及额外占卡记录合计：136.611365 GPU小时。不计CPU算力费用、benchmark推理及独立环境/数值探针；Ouro1.4B训练控制器持续保留GPU期间的试跑和CPU审计占卡时间按上述口径单列。

GPU型号按本机nvidia-smi实录保留；用户将集群称为8×H200，系统显示如下：

```text
0, NVIDIA L20X, GPU-994c5418-ecc5-78ff-ad5e-0cee6dcd1111, 143771 MiB
1, NVIDIA L20X, GPU-520fad11-804c-b419-7096-2a19c219449b, 143771 MiB
2, NVIDIA L20X, GPU-861ad2de-b899-21ca-707e-44c0a1ce1029, 143771 MiB
3, NVIDIA L20X, GPU-bb9dac43-6fce-fc43-0f10-8f64a80742c3, 143771 MiB
4, NVIDIA L20X, GPU-a688c63c-924d-a0b0-49f3-d8fbe7df3dda, 143771 MiB
5, NVIDIA L20X, GPU-2dc90f87-ad67-93a7-d0c5-218a3543318a, 143771 MiB
6, NVIDIA L20X, GPU-e54049b5-ca93-d066-efc5-6942d59bf762, 143771 MiB
7, NVIDIA L20X, GPU-afc353e9-924d-40c1-75bb-c126dc83e41e, 143771 MiB
```

原始账本快照：`current-cost-summary-ledger.json`，SHA256 `5fc429bdb304173afaf57d39438f3329675ca1240a85c572a01163161b1d0ff8`。
可复现命令：`python experiments/baselines/render_training_costs.py`。
