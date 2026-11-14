# Tuned against shared learning rate (plan §7.4 item 5)

For each Arm A variant that runs at its own learning rate, three further standalone 100M runs on init seeds 1 to 3 are made at MHA's rate (decision 224). The two sides are on the same init seeds 1 to 3. The gap column is the variant's 100M branches at its own rate minus those runs, and its interval is computed with the test chosen at Gate 2 (paired by seed under the paired test, two groups of three runs under Welch's test); a negative value means the tuned rate gave the lower loss. The sweep's own difference between the two rates was selected on, so these are fresh seeds. No detection claim is made on this comparison (decision 224).

Not computed: comparisons wait for the controls (plan §8.1, §8.4): the null control must be computed and not detected, and the positive control computed (see gates.md).

| variant | own rate | MHA's rate | tuned 100M loss, seeds 1-3 | shared-rate 100M loss, seeds 1-3 | tuned minus shared, nats | 95% interval |
|---|---|---|---|---|---|---|
| gqa4 | — | — | — | — | — | — |
| mqa | — | — | — | — | — | — |
| mla192 | — | — | — | — | — | — |
| gqa2 | — | — | — | — | — | — |
| mla96 | — | — | — | — | — | — |
| swa256 | — | — | — | — | — | — |
