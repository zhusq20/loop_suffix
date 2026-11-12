# Stage timing pilot

32/32 measurement streams complete.

24 stratified SpecBench requests, max1024. Normal timings and diagnostic spans are separate. Host stacks partition diagnostic wall time and include waits; CUDA event spans are shown separately, may include submission gaps, and cannot be added to host times. Readout combines shallow/full-depth coda and LM head; recurrent includes attention KV work.

| Model | B | Method | Normal output token/s | Gamma | Rows/token |
| --- | ---: | --- | ---: | ---: | ---: |
| huginn-3.5b | 1 | ar | 6.18 | 1.000 | 32.00 |
| huginn-3.5b | 1 | budget-online | 14.04 | 6.503 | 72.13 |
| huginn-3.5b | 1 | budget-suffix | 9.18 | 1.624 | 47.23 |
| huginn-3.5b | 1 | loopspec-dual | 19.85 | 4.618 | 75.58 |
| huginn-3.5b | 16 | ar | 25.54 | 1.000 | 32.00 |
| huginn-3.5b | 16 | budget-online | 73.81 | 5.869 | 80.45 |
| huginn-3.5b | 16 | budget-suffix | 40.13 | 1.584 | 46.83 |
| huginn-3.5b | 16 | loopspec-dual | 67.71 | 4.772 | 70.91 |
| ouro-2.6b | 1 | ar | 5.63 | 1.000 | 4.00 |
| ouro-2.6b | 1 | budget-online | 28.44 | 6.112 | 10.48 |
| ouro-2.6b | 1 | budget-suffix | 8.97 | 1.776 | 5.63 |
| ouro-2.6b | 1 | loopspec-dual | 15.97 | 3.333 | 4.81 |
| ouro-2.6b | 16 | ar | 24.80 | 1.000 | 4.00 |
| ouro-2.6b | 16 | budget-online | 70.07 | 5.830 | 10.89 |
| ouro-2.6b | 16 | budget-suffix | 50.81 | 1.975 | 5.29 |
| ouro-2.6b | 16 | loopspec-dual | 61.71 | 3.355 | 4.81 |

This short, single-run pilot does not replace full480 performance results. Output agreement and completion-history order are recorded in pairs.csv; equality does not establish causal stage-specific speedups. Profiler/event instrumentation overhead is retained, not subtracted from normal timings.
