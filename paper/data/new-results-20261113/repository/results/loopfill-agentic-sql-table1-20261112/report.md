# Agent-request decoding experiment

Spider development SQL execution/repair request replay

Frozen requests: 100; tasks: 78; phase counts: {'early': 74, 'middle': 13, 'late': 13}.

Table 1 supplement: B=1, Huginn-3.5B and Ouro-2.6B, BF16, max512, seed0, single timing repeat. Fixed historical prompts; new target generation. This is not a live-agent success-rate evaluation.

Completed formal cells: 28/28. Missing cells are pending or failed; consult status.json.

| Cell | Requests | Decode tok/s | Mean decode s | p95 decode s | γ | Latency speedup |
|---|---:|---:|---:|---:|---:|---:|
| huginn-3.5b-ar-t0 | 100 | 6.774 | 13.077 | 75.205 | 1.000 | 1.000× |
| huginn-3.5b-wavefront-t0 | 100 | 21.203 | 4.341 | 16.659 | 4.166 | 3.013× |
| huginn-3.5b-loopspec-dual-t0 | 100 | 27.550 | 3.193 | 14.684 | 5.567 | 4.096× |
| huginn-3.5b-budget-suffix-t0 | 100 | 14.115 | 6.433 | 23.510 | 2.224 | 2.033× |
| huginn-3.5b-loopahead-t0 | 100 | 28.773 | 3.097 | 12.227 | 6.266 | 4.223× |
| huginn-3.5b-dflash-t0 | 100 | 20.999 | 3.992 | 15.258 | 3.455 | 3.276× |
| huginn-3.5b-budget-online-t0 | 100 | 36.281 | 2.423 | 9.898 | 7.607 | 5.397× |
| huginn-3.5b-ar-t1 | 100 | 6.732 | 17.664 | 58.244 | 1.000 | 1.000× |
| huginn-3.5b-wavefront-t1 | 100 | 12.169 | 10.111 | 37.197 | 2.344 | 1.747× |
| huginn-3.5b-loopspec-dual-t1 | 100 | 13.290 | 8.975 | 27.375 | 2.677 | 1.968× |
| huginn-3.5b-budget-suffix-t1 | 100 | 9.539 | 14.008 | 47.806 | 1.485 | 1.261× |
| huginn-3.5b-loopahead-t1 | 100 | 17.910 | 8.360 | 28.203 | 3.971 | 2.113× |
| huginn-3.5b-dflash-t1 | 100 | 13.230 | 8.321 | 30.589 | 2.154 | 2.123× |
| huginn-3.5b-budget-online-t1 | 100 | 15.745 | 8.100 | 33.685 | 3.296 | 2.181× |
| ouro-2.6b-ar-t0 | 100 | 6.296 | 5.851 | 15.376 | 1.000 | 1.000× |
| ouro-2.6b-wavefront-t0 | 100 | 18.019 | 2.571 | 5.695 | 3.080 | 2.276× |
| ouro-2.6b-loopspec-dual-t0 | 100 | 18.622 | 1.962 | 4.536 | 3.276 | 2.982× |
| ouro-2.6b-budget-suffix-t0 | 100 | 12.103 | 3.468 | 9.258 | 2.113 | 1.687× |
| ouro-2.6b-loopahead-t0 | 100 | 20.456 | 2.033 | 4.282 | 3.887 | 2.878× |
| ouro-2.6b-dflash-t0 | 100 | 14.991 | 2.487 | 6.101 | 2.690 | 2.353× |
| ouro-2.6b-budget-online-t0 | 100 | 33.342 | 1.261 | 3.456 | 6.227 | 4.642× |
| ouro-2.6b-ar-t1 | 100 | 6.190 | 14.894 | 78.334 | 1.000 | 1.000× |
| ouro-2.6b-wavefront-t1 | 100 | 13.222 | 7.168 | 31.526 | 2.285 | 2.078× |
| ouro-2.6b-loopspec-dual-t1 | 100 | 13.862 | 7.178 | 32.960 | 2.443 | 2.075× |
| ouro-2.6b-budget-suffix-t1 | 100 | 8.914 | 12.294 | 50.593 | 1.565 | 1.211× |
| ouro-2.6b-loopahead-t1 | 100 | 17.658 | 5.000 | 26.944 | 3.430 | 2.979× |
| ouro-2.6b-dflash-t1 | 100 | 11.529 | 9.163 | 46.595 | 2.099 | 1.625× |
| ouro-2.6b-budget-online-t1 | 100 | 19.248 | 4.551 | 20.415 | 3.619 | 3.273× |

Decode throughput is generated decode tokens divided by synchronized decode time; it excludes prefill and is not concurrent serving throughput. Latency speedup is the ratio of total decode times; generated lengths may differ. Compare output_token_ratio in measurements.csv.

DFlash uses the trained local block16 drafter and the packed target adapter. Its configured training length is 3072; requests above this length are extrapolation. All methods first run shortest/longest-context smoke checks. Diagnostic timings are excluded. No confidence interval or task-quality claim is made.

Fixed causal SQL request replay. Ouro-2.6B greedy teacher, actual read-only SQLite execution, at most two error-triggered repairs. No gold SQL feedback. Successful execution ends a trace. Current teacher response, tool result and future calls removed from replay inputs. No live-agent or official Spider accuracy claim. Distinct from the unpublished SuffixDecoding AgenticSQL dataset.
