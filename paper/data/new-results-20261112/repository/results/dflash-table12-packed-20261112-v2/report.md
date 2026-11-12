# DFlash packed 引擎重测

表一：六任务 B1/max512，T=0/1；表二：SpecBench480/max1024，B=1/2/4/8/16。
AR 分母来自原表历史测量，未作跨日期漂移修正。仅 complete 且审计通过的单元提供数值。

| 单元 | 状态 | token/s | γ | 相对历史 AR |
|---|---|---:|---:|---:|
| table2-huginn-3.5b-b1 | complete | 18.363 | 3.108 | 3.119 |
| table2-ouro-2.6b-b1 | complete | 16.171 | 2.972 | 2.925 |
| table2-huginn-3.5b-b2 | complete | 33.178 | 3.172 | 2.987 |
| table2-ouro-2.6b-b2 | complete | 28.846 | 2.961 | 2.708 |
| table2-huginn-3.5b-b4 | complete | 57.473 | 3.149 | 3.025 |
| table2-ouro-2.6b-b4 | complete | 55.831 | 3.008 | 2.691 |
| table2-huginn-3.5b-b8 | complete | 94.081 | 3.119 | 2.851 |
| table2-ouro-2.6b-b8 | complete | 96.719 | 2.995 | 2.498 |
| table2-huginn-3.5b-b16 | complete | 129.178 | 3.118 | 2.672 |
| table2-ouro-2.6b-b16 | complete | 153.479 | 2.968 | 2.274 |
| table1-huginn-3.5b-bbh-t0 | complete | 26.453 | 4.254 | 3.799 |
| table1-ouro-2.6b-bbh-t0 | reused_existing_measurement | — | — | — |
| table1-huginn-3.5b-gsm8k-t0 | reused_existing_measurement | — | — | — |
| table1-ouro-2.6b-gsm8k-t0 | reused_existing_measurement | — | — | — |
| table1-huginn-3.5b-humanevalplus-t0 | reused_existing_measurement | — | — | — |
| table1-ouro-2.6b-humanevalplus-t0 | reused_existing_measurement | — | — | — |
| table1-huginn-3.5b-math500-t0 | complete_artifact_exit_unavailable | — | — | — |
| table1-ouro-2.6b-math500-t0 | complete_artifact_exit_unavailable | — | — | — |
| table1-huginn-3.5b-mbppplus-t0 | reused_existing_measurement | — | — | — |
| table1-ouro-2.6b-mbppplus-t0 | reused_existing_measurement | — | — | — |
| table1-huginn-3.5b-specbench-t0 | complete | 17.601 | 2.899 | 2.501 |
| table1-ouro-2.6b-specbench-t0 | complete_artifact_exit_unavailable | — | — | — |
| table1-huginn-3.5b-bbh-t1 | complete | 17.868 | 2.868 | 2.473 |
| table1-ouro-2.6b-bbh-t1 | reused_existing_measurement | — | — | — |
| table1-huginn-3.5b-gsm8k-t1 | complete | 12.223 | 1.991 | 1.791 |
| table1-ouro-2.6b-gsm8k-t1 | reused_existing_measurement | — | — | — |
| table1-huginn-3.5b-humanevalplus-t1 | complete | 17.619 | 2.844 | 2.729 |
| table1-ouro-2.6b-humanevalplus-t1 | complete | 16.566 | 2.998 | 2.448 |
| table1-huginn-3.5b-math500-t1 | complete_artifact_exit_unavailable | — | — | — |
| table1-ouro-2.6b-math500-t1 | complete_artifact_exit_unavailable | — | — | — |
| table1-huginn-3.5b-mbppplus-t1 | complete | 17.294 | 2.815 | 2.415 |
| table1-ouro-2.6b-mbppplus-t1 | complete | 16.316 | 2.968 | 2.417 |
| table1-huginn-3.5b-specbench-t1 | complete | 11.803 | 1.888 | 1.847 |
| table1-ouro-2.6b-specbench-t1 | complete_artifact_exit_unavailable | — | — | — |

Ouro-1.4B 未提供完成训练的 DFlash 检查点，未启动训练或填入替代数值。
原表和旧 DFlash 结果保持历史记录；单次 BF16 速度测量不证明输出相同或任务质量相同。
