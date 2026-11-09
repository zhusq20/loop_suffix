# Mechanism evidence (existing runs)

36 ablation configurations / 17,280 requests and 240 existing readiness requests audited.

## Load and node budget

| Model | B | N | Decode s/request | Decode ms/token | Output token/s | Gamma | Rows/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| huginn-3.5b | 1 | 8 | 5.221 | 46.682 | 20.561 | 4.780 | 53.473 |
| huginn-3.5b | 1 | 16 | 4.116 | 36.236 | 26.169 | 6.404 | 74.879 |
| huginn-3.5b | 1 | 32 | 3.517 | 30.569 | 30.678 | 7.937 | 120.087 |
| huginn-3.5b | 1 | 64 | 3.488 | 30.323 | 30.896 | 8.675 | 216.924 |
| huginn-3.5b | 16 | 8 | — | — | 107.970 | 4.766 | 53.561 |
| huginn-3.5b | 16 | 16 | — | — | 117.518 | 6.400 | 74.830 |
| huginn-3.5b | 16 | 32 | — | — | 100.670 | 7.823 | 121.412 |
| huginn-3.5b | 16 | 64 | — | — | 66.457 | 8.569 | 219.611 |
| ouro-2.6b | 1 | 8 | 8.604 | 41.578 | 23.310 | 5.021 | 6.992 |
| ouro-2.6b | 1 | 16 | 7.363 | 35.511 | 27.117 | 5.909 | 11.137 |
| ouro-2.6b | 1 | 32 | 7.227 | 34.637 | 27.800 | 6.267 | 20.051 |
| ouro-2.6b | 1 | 64 | 7.203 | 35.828 | 26.880 | 6.246 | 38.263 |
| ouro-2.6b | 16 | 8 | — | — | 191.663 | 5.020 | 6.985 |
| ouro-2.6b | 16 | 16 | — | — | 189.556 | 5.876 | 11.185 |
| ouro-2.6b | 16 | 32 | — | — | 174.237 | 6.238 | 20.081 |
| ouro-2.6b | 16 | 64 | — | — | 119.686 | 6.192 | 38.513 |

Endpoints already establish a load-dependent budget tradeoff. No B4/B8 inference is added. N64 vs N32 at B1 is a small single-run difference; no significance claim.

## Root readiness

Existing online-c0.7, B1/N16, max512, six datasets ×20 requests per model. These are six different datasets, not six balanced SpecBench tasks. Terminal tokens are excluded from handover denominators; initial-root wait is separate.

| Model | Handovers | Mean r/T | Immediately ready | Remaining steps/handover | Rebuilt correct nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| huginn-3.5b | 19614 | 0.868 | 41.18% | 4.214 | 979 |
| ouro-2.6b | 32258 | 0.833 | 38.02% | 0.670 | 146 |

Every request satisfies calls = initial wait + sum(T-r) and T-r = proposal wait + ranking wait. Source transitions cannot be reconstructed from these aggregate journals. No new trace is required for the progress/wait/rebuild claims above. Different outputs and histories limit causal interpretation. No diagnostic wall time enters throughput.
