# Context use (plan §10.3)

Context use is a secondary metric (plan §10.3): gaps against MHA with their 95% intervals. No detection claim is made on them.

SWA at context 1,024 is nearly uninformative (plan §13.11): a 256-token window drops little at this length, so its line may top a loss plot and mean nothing. It is captioned, never headlined.

Not computed: comparisons wait for the controls (plan §8.1, §8.4): the null control must be computed and not detected, and the positive control computed (see gates.md).

| variant | late-context loss, every seed | gap vs MHA [95% CI], late-context | verdict (late-context) | ICL score, every seed | gap vs MHA [95% CI], ICL | verdict (ICL) |
|---|---|---|---|---|---|---|
| gqa4 | — | — | — | — | — | — |
| mqa | — | — | — | — | — | — |
| mla192 | — | — | — | — | — | — |
| gqa2 | — | — | — | — | — | — |
| mla96 | — | — | — | — | — | — |
| swa256 | — | — | — | — | — | — |

Not computed: no 300M result carries the context-use fields yet, or the controls have not passed.
