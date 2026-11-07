# Table 2 — SpecBench end-to-end output throughput

74 complete cells and one observed OOM; 480 requests per complete cell, up to 1024 new tokens. Entries are output token/s (speedup vs same-model, same-batch AR). Timing includes prefill, decode, scheduling and local persistence; one repeat.

## huginn-3.5b

| Method | B1 | B2 | B4 | B8 | B16 |
| --- | --- | --- | --- | --- | --- |
| AR | 5.89 (1.00×) | 11.11 (1.00×) | 19.00 (1.00×) | 32.99 (1.00×) | 48.34 (1.00×) |
| WFD | 14.92 (2.53×) | 24.56 (2.21×) | 39.76 (2.09×) | 60.81 (1.84×) | OOM |
| LoopSpec | 19.65 (3.34×) | 33.89 (3.05×) | 51.03 (2.69×) | 74.24 (2.25×) | 89.65 (1.85×) |
| Suffix-only | 9.45 (1.60×) | 17.34 (1.56×) | 30.17 (1.59×) | 52.01 (1.58×) | 74.45 (1.54×) |
| LoopFill-online | 26.17 (4.44×) | 40.73 (3.67×) | 59.50 (3.13×) | 88.18 (2.67×) | 117.52 (2.43×) |

## ouro-1.4b

| Method | B1 | B2 | B4 | B8 | B16 |
| --- | --- | --- | --- | --- | --- |
| AR | 10.77 (1.00×) | 20.75 (1.00×) | 40.06 (1.00×) | 76.25 (1.00×) | 135.65 (1.00×) |
| WFD | 26.86 (2.49×) | 51.00 (2.46×) | 94.74 (2.36×) | 168.89 (2.22×) | 272.37 (2.01×) |
| LoopSpec | 29.77 (2.76×) | 56.83 (2.74×) | 105.23 (2.63×) | 182.51 (2.39×) | 287.90 (2.12×) |
| Suffix-only | 17.55 (1.63×) | 31.54 (1.52×) | 61.70 (1.54×) | 113.74 (1.49×) | 193.24 (1.42×) |
| LoopFill-online | 47.49 (4.41×) | 84.42 (4.07×) | 143.27 (3.58×) | 214.71 (2.82×) | 289.28 (2.13×) |

## ouro-2.6b

| Method | B1 | B2 | B4 | B8 | B16 |
| --- | --- | --- | --- | --- | --- |
| AR | 5.53 (1.00×) | 10.65 (1.00×) | 20.75 (1.00×) | 38.72 (1.00×) | 67.48 (1.00×) |
| WFD | 15.19 (2.75×) | 29.02 (2.72×) | 53.96 (2.60×) | 98.75 (2.55×) | 158.45 (2.35×) |
| LoopSpec | 16.77 (3.03×) | 32.02 (3.01×) | 58.78 (2.83×) | 104.06 (2.69×) | 166.52 (2.47×) |
| Suffix-only | 8.89 (1.61×) | 17.03 (1.60×) | 32.48 (1.57×) | 60.47 (1.56×) | 102.13 (1.51×) |
| LoopFill-online | 27.12 (4.91×) | 51.17 (4.80×) | 86.22 (4.16×) | 140.94 (3.64×) | 189.56 (2.81×) |
BF16, greedy, CUDA Graph off, same-engine baselines. Outputs can differ between methods: speedup is not proof of equal output or task quality. No confidence intervals are inferred from this single timing repeat. The two tables use different output caps and timing boundaries and must not be pooled.
