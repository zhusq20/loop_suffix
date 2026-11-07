# Table 2 — SpecBench end-to-end output throughput (token/s)

480 requests; up to 1024 new tokens; prefill + decode + scheduling; BF16; CUDA Graph off; one timing repeat.

## huginn-3.5b

| Method | B1 | B2 | B4 | B8 | B16 |
|---|---:|---:|---:|---:|---:|
| AR | 5.89 | 11.11 | 19.00 | 32.99 | pending |
| WFD | 14.92 | 24.56 | 39.76 | pending | pending |
| LoopSpec | 19.65 | pending | pending | pending | pending |
| Suffix-only | 9.45 | 17.34 | pending | pending | pending |
| LoopFill-online | 26.17 | pending | pending | pending | pending |

## ouro-1.4b

| Method | B1 | B2 | B4 | B8 | B16 |
|---|---:|---:|---:|---:|---:|
| AR | pending | pending | pending | pending | pending |
| WFD | pending | pending | pending | pending | pending |
| LoopSpec | pending | pending | pending | pending | pending |
| Suffix-only | pending | pending | pending | pending | pending |
| LoopFill-online | pending | pending | pending | pending | pending |

## ouro-2.6b

| Method | B1 | B2 | B4 | B8 | B16 |
|---|---:|---:|---:|---:|---:|
| AR | 5.53 | 10.65 | pending | pending | pending |
| WFD | 15.19 | pending | pending | pending | pending |
| LoopSpec | pending | pending | pending | pending | pending |
| Suffix-only | pending | pending | pending | pending | pending |
| LoopFill-online | pending | pending | pending | pending | pending |
