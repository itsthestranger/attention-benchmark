## Noise floor at 100M (Gate 2)

| item | value |
|---|---|
| sigma_init, MHA seeds 1-5 | — |
| sigma_init, null-control seeds | — |
| sigma_data, MHA seed 1 over data seeds 1-3 | — |
| s_rep | — |
| s_rep / sigma_init | — |
| pairing threshold | — |
| test chosen | — |

## Null control (Gate 2)

| item | value |
|---|---|
| null seeds | — |
| MHA seeds | — |
| mean gap, nats | — |
| 95% interval | — |
| p | — |
| verdict | — |

## Positive control: L6 against L8 at 300M

| item | value |
|---|---|
| seeds | — |
| mean gap, nats | — |
| 95% interval | — |
| p | — |
| verdict | — |

## Seed count and detectable effect

The ladder slope b at 300M in this section is fitted on the registered seeds only (plan §1.4), while the ladder slope by budget below uses every run present, so the two values at 300M differ once MHA has eight seeds.

| item | value |
|---|---|
| sigma_init at 300M, MHA seeds 1-5 | — |
| ladder slope b at 300M | — |
| 5-seed detectable gap, nats | — |
| undetectable parameter fraction at 5 seeds | — |
| threshold | — |
| seeds per primary | — |
| variants leaving Arm A | — |
| detectable gap of the chosen design, nats | — |

## Ladder slope b by budget

| item | value |
|---|---|
| 100M | — |
| 200M | — |
| 300M | — |

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
