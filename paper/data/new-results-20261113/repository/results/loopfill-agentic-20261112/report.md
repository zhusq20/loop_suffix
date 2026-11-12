# OpenHands agent-request decoding experiment

Frozen requests: 100; tasks: 57; phase counts: {'early': 42, 'middle': 40, 'late': 18}.

B=1, Ouro-2.6B, BF16, max512, seed0, single timing repeat. Fixed historical prompts; new target generation. This is not a live-agent success-rate evaluation.

Completed cells: 0/20 (including diagnostics). Missing cells are pending or failed; consult status.json.

| Cell | Requests | Decode tok/s | Mean decode s | p95 decode s | γ | Latency speedup |
|---|---:|---:|---:|---:|---:|---:|

Decode throughput is generated decode tokens divided by synchronized decode time; it excludes prefill and is not concurrent serving throughput. Latency speedup is the ratio of total decode times; generated lengths may differ. Compare output_token_ratio in measurements.csv.

DFlash uses the trained local block16 drafter and the packed target adapter. Its configured training length is 3072; these longer contexts are extrapolation. All methods first run shortest/longest-context smoke checks. Diagnostic timings are excluded. No confidence interval or task-quality claim is made.
