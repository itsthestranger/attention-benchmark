| variant | params | d_ff | KV bytes/token/layer | train ms/step | decode tok/s (batch 32, cache 4k) | val loss, every seed | gap vs MHA [95% CI] | effective-param multiplier | verdict |
|---|---|---|---|---|---|---|---|---|---|
| mha | 17,308,032 | 1024 | 1,536 | — | — | — | — | — | — |
| gqa4 | 17,308,032 | 1152 | 768 | — | — | — | — | — | — |
| mqa | 17,308,032 | 1248 | 192 | — | — | — | — | — | — |
| mla192 | 17,311,104 | 1048 | 432 | — | — | — | — | — | — |
| gqa2 | 17,308,032 | 1216 | 384 | — | — | — | — | — | — |
| mla96 | 17,309,568 | 1224 | 240 | — | — | — | — | — | — |
| swa256 | 17,308,032 | 1024 | 1,536 (capped at 256 tokens) | — | — | — | — | — | — |

Notes:

- test selection: MHA 100M branches for seeds [1, 2, 3, 4, 5] are missing
- test selection: MHA 100M reruns for seeds [1, 2, 3] are missing
- test selection skipped: see the missing MHA 100M runs above
- null control: null 100M runs at MHA's peak learning rate for seeds [11, 12, 13, 14, 15] are missing
- ladder fit: MHA 300M results are missing
- positive control: mha_L6 300M results for seeds [1, 2, 3, 4, 5] and mha 300M results for seeds [1, 2, 3, 4, 5] are missing
- ladder fit at 100M: MHA 100M results are missing
- ladder fit at 200M: MHA 200M results are missing
- sigma_data: data-order runs for data seeds [1, 2, 3] are missing
- seed count rule skipped: no test selection
- seed count rule skipped: no ladder fit
- comparisons skipped: null control not computed
- comparisons skipped: positive control not computed
