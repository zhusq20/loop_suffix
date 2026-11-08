# Temperature=1 candidate coverage

Each entry is the pooled percentage of sampled full-depth decode tokens that hit existing draft candidates. Prefill tokens are excluded. AR has no drafts. This is not accepted nodes / created nodes, a proposal acceptance probability, or a measurement of saved computation.

## huginn-3.5b

| Method | GSM8K | MATH-500 | BBH | HumanEval+ | MBPP+ | SpecBench |
| --- | --- | --- | --- | --- | --- | --- |
| AR | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| WFD | 60.06% | 68.11% | 72.54% | 69.33% | 67.89% | 55.93% |
| LoopSpec | 74.44% | 78.40% | 82.60% | 80.06% | 78.08% | 65.35% |
| Suffix-only | 26.96% | 38.09% | 48.43% | 34.34% | 31.96% | 20.83% |
| LoopFill-online | 79.07% | 81.49% | 84.39% | 82.01% | 81.95% | 70.52% |

## ouro-1.4b

| Method | GSM8K | MATH-500 | BBH | HumanEval+ | MBPP+ | SpecBench |
| --- | --- | --- | --- | --- | --- | --- |
| AR | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| WFD | 77.39% | 80.13% | 77.23% | 84.05% | 78.64% | 65.74% |
| LoopSpec | 84.49% | 85.08% | 83.23% | 88.16% | 83.67% | 73.28% |
| Suffix-only | 36.03% | 39.10% | 45.89% | 44.80% | 36.60% | 28.37% |
| LoopFill-online | 90.69% | 91.21% | 87.74% | 93.44% | 91.04% | 81.01% |

## ouro-2.6b

| Method | GSM8K | MATH-500 | BBH | HumanEval+ | MBPP+ | SpecBench |
| --- | --- | --- | --- | --- | --- | --- |
| AR | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| WFD | 81.45% | 81.23% | 85.03% | 80.28% | 80.48% | 69.40% |
| LoopSpec | 85.53% | 86.26% | 87.62% | 84.88% | 84.40% | 74.84% |
| Suffix-only | 39.75% | 36.05% | 51.94% | 36.89% | 36.48% | 28.18% |
| LoopFill-online | 90.29% | 91.94% | 91.93% | 91.09% | 91.33% | 83.15% |
BF16, CUDA Graph off, same-engine baselines. Shared request/position random numbers do not guarantee identical outputs across different BF16 compute shapes. Single timing repeat and one sampling seed: no latency confidence intervals or task-quality claims. This batch is separate from temperature=0 results.
