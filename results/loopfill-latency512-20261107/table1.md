# Table 1 — B=1 decode latency speedup and acceptance length

90/90 cells complete; 31,380 requests. Each entry is latency speedup vs AR × / loop-normalized γ. Up to 512 new tokens; decode-only timing; one repeat. GM is the unweighted geometric mean of the six dataset speedups.

## huginn-3.5b

| Method | GSM8K | MATH-500 | BBH | HumanEval+ | MBPP+ | SpecBench | GM speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AR | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 |
| WFD | 2.32 / 3.05 | 2.86 / 4.00 | 2.95 / 3.96 | 3.48 / 4.62 | 2.83 / 3.69 | 2.36 / 3.11 | 2.78 |
| LoopSpec | 3.47 / 4.88 | 3.99 / 5.58 | 4.22 / 5.79 | 4.33 / 6.08 | 3.96 / 5.33 | 3.46 / 4.65 | 3.89 |
| Suffix-only | 1.58 / 1.75 | 1.97 / 2.18 | 2.50 / 2.70 | 2.09 / 2.36 | 1.79 / 1.87 | 1.51 / 1.57 | 1.88 |
| LoopFill-online | 4.35 / 6.50 | 5.30 / 7.72 | 6.01 / 8.76 | 5.85 / 8.53 | 4.92 / 7.15 | 4.08 / 5.91 | 5.03 |

## ouro-1.4b

| Method | GSM8K | MATH-500 | BBH | HumanEval+ | MBPP+ | SpecBench | GM speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AR | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 |
| WFD | 2.59 / 2.82 | 2.96 / 3.20 | 3.01 / 3.22 | 3.37 / 3.57 | 2.97 / 3.27 | 2.47 / 2.66 | 2.88 |
| LoopSpec | 2.94 / 3.25 | 3.15 / 3.50 | 3.22 / 3.52 | 3.42 / 3.75 | 3.17 / 3.56 | 2.84 / 3.09 | 3.12 |
| Suffix-only | 1.55 / 1.70 | 1.87 / 2.06 | 2.41 / 2.62 | 2.39 / 2.54 | 1.72 / 1.95 | 1.59 / 1.73 | 1.89 |
| LoopFill-online | 4.54 / 5.54 | 5.44 / 6.84 | 6.65 / 8.07 | 7.52 / 8.99 | 5.31 / 6.72 | 4.31 / 5.17 | 5.52 |

## ouro-2.6b

| Method | GSM8K | MATH-500 | BBH | HumanEval+ | MBPP+ | SpecBench | GM speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AR | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 |
| WFD | 3.11 / 3.32 | 2.86 / 3.06 | 3.37 / 3.51 | 3.18 / 3.47 | 3.17 / 3.46 | 2.79 / 2.94 | 3.07 |
| LoopSpec | 3.23 / 3.59 | 3.10 / 3.45 | 3.47 / 3.73 | 3.33 / 3.68 | 3.33 / 3.66 | 3.09 / 3.34 | 3.26 |
| Suffix-only | 1.79 / 1.91 | 1.44 / 1.61 | 2.62 / 2.75 | 2.08 / 2.31 | 1.74 / 1.90 | 1.60 / 1.74 | 1.84 |
| LoopFill-online | 5.76 / 6.79 | 4.77 / 5.70 | 8.23 / 9.38 | 6.99 / 8.25 | 5.70 / 6.72 | 4.84 / 5.71 | 5.93 |
BF16, greedy, CUDA Graph off, same-engine baselines. Outputs can differ between methods: speedup is not proof of equal output or task quality. No confidence intervals are inferred from this single timing repeat. The two tables use different output caps and timing boundaries and must not be pooled.
