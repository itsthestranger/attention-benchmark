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
- ladder fit: MHA 300M results are missing
- seed count rule skipped: no test selection
- seed count rule skipped: no ladder fit
- primary comparisons skipped: no seed count decision
- exploratory comparisons skipped: no seed count decision
