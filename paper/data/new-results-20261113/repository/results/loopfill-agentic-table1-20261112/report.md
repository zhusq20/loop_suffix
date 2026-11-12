# Agent-request decoding experiment

SWE-bench Verified / OpenHands CodeAct 2.1 Sonnet 20241022

Frozen requests: 100; tasks: 84; phase counts: {'early': 99, 'middle': 1}.

Table 1 supplement: B=1, Huginn-3.5B and Ouro-2.6B, BF16, max512, seed0, single timing repeat. Fixed historical prompts; new target generation. This is not a live-agent success-rate evaluation.

Completed formal cells: 0/28. Missing cells are pending or failed; consult status.json.

| Cell | Requests | Decode tok/s | Mean decode s | p95 decode s | γ | Latency speedup |
|---|---:|---:|---:|---:|---:|---:|

Decode throughput is generated decode tokens divided by synchronized decode time; it excludes prefill and is not concurrent serving throughput. Latency speedup is the ratio of total decode times; generated lengths may differ. Compare output_token_ratio in measurements.csv.

DFlash uses the trained local block16 drafter and the packed target adapter. Its configured training length is 3072; requests above this length are extrapolation. All methods first run shortest/longest-context smoke checks. Diagnostic timings are excluded. No confidence interval or task-quality claim is made.

Fixed historical request prefixes, target free generation. API tool calls serialized as explicit JSON blocks; tool observations preserved as user messages; API cache hints removed. Held-out next responses never enter inputs or suffix cache. This is decoding on agent requests, not a live agent or task-success evaluation.
