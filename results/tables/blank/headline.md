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

Disclosures:

- MLA bundles two changes, low-rank KV compression and decoupled RoPE; they cannot be fully separated, and compression alone was not measured (plan §13.2).
- Kernel maturity is not architecture: MHA, GQA and MQA use a mature fused kernel, while SWA's block-skipping path and MLA's absorbed path are newer or hand-written, so any SWA or MLA timing gap is partly implementation (plan §13.3).
- Hardware: every number here is from a T4 (sm75) with the memory-efficient kernel, not FlashAttention-2 and no fp8; rankings may differ on Ampere and later (plan §13.4).
- Scale: 17M parameters, context 1,024, 300M tokens, one corpus and one tokenizer; nothing here extrapolates to production scale (plan §13.5).
- Parameter matching moves capacity into the MLP: iso-parameter variants spend their KV-projection savings on a wider MLP, so iso-parameter does not isolate attention; see the Arm B table (plan §13.1).
