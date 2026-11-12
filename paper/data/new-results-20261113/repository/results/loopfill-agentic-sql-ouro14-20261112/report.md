# Agent-request decoding experiment

Spider development SQL execution/repair request replay

Frozen requests: 100; tasks: 78; phase counts: {'early': 74, 'middle': 13, 'late': 13}.

Table 1 supplement: B=1, ouro-1.4b, BF16, max512, seed0, single timing repeat. Fixed historical prompts; new target generation. This is not a live-agent success-rate evaluation.

Completed formal cells: 14/14. Missing cells are pending or failed; consult status.json.

| Cell | Requests | Decode tok/s | Mean decode s | p95 decode s | γ | Latency speedup |
|---|---:|---:|---:|---:|---:|---:|
| ouro-1.4b-ar-t0 | 100 | 12.088 | 2.998 | 7.655 | 1.000 | 1.000× |
| ouro-1.4b-wavefront-t0 | 100 | 29.892 | 1.221 | 2.867 | 2.682 | 2.456× |
| ouro-1.4b-loopspec-dual-t0 | 100 | 33.812 | 1.080 | 2.598 | 3.070 | 2.777× |
| ouro-1.4b-budget-suffix-t0 | 100 | 22.566 | 1.619 | 4.921 | 2.029 | 1.851× |
| ouro-1.4b-loopahead-t0 | 100 | 35.007 | 1.053 | 2.527 | 3.525 | 2.847× |
| ouro-1.4b-dflash-t0 | 100 | 30.325 | 1.179 | 3.042 | 2.878 | 2.543× |
| ouro-1.4b-budget-online-t0 | 100 | 55.023 | 0.659 | 1.772 | 5.389 | 4.548× |
| ouro-1.4b-ar-t1 | 100 | 12.069 | 7.912 | 42.059 | 1.000 | 1.000× |
| ouro-1.4b-wavefront-t1 | 100 | 24.650 | 2.980 | 11.854 | 2.199 | 2.655× |
| ouro-1.4b-loopspec-dual-t1 | 100 | 25.254 | 3.309 | 13.567 | 2.312 | 2.391× |
| ouro-1.4b-budget-suffix-t1 | 100 | 19.717 | 4.418 | 16.418 | 1.769 | 1.791× |
| ouro-1.4b-loopahead-t1 | 100 | 27.889 | 2.687 | 12.294 | 2.904 | 2.945× |
| ouro-1.4b-dflash-t1 | 100 | 22.274 | 4.230 | 20.821 | 2.152 | 1.870× |
| ouro-1.4b-budget-online-t1 | 100 | 36.643 | 2.699 | 10.426 | 3.626 | 2.931× |

Decode throughput is generated decode tokens divided by synchronized decode time; it excludes prefill and is not concurrent serving throughput. Latency speedup is the ratio of total decode times; generated lengths may differ. Compare output_token_ratio in measurements.csv.

DFlash uses the trained local block16 drafter and the packed target adapter. Its configured training length is 3072; requests above this length are extrapolation. All methods first run shortest/longest-context smoke checks. Diagnostic timings are excluded. No confidence interval or task-quality claim is made.

Fixed causal SQL request replay. Ouro-2.6B greedy teacher, actual read-only SQLite execution, at most two error-triggered repairs. No gold SQL feedback. Successful execution ends a trace. Current teacher response, tool result and future calls removed from replay inputs. No live-agent or official Spider accuracy claim. Distinct from the unpublished SuffixDecoding AgenticSQL dataset.
