# Training timing (plan §8.5)

Source: GPU —; backend —; torch —; CUDA —; driver —; timing pass —; — steps, first — discarded, — repetitions.

| track | ms/step, mean | ms/step, each repetition | spread over repetitions, % of mean | ratio to MHA opt | attention share, % | peak allocated, MiB |
|---|---|---|---|---|---|---|
| mha | — | — | — | — | — | — |
| gqa4 | — | — | — | — | — | — |
| mqa | — | — | — | — | — | — |
| mla192 | — | — | — | — | — | — |
| mla192/ref | — | — | — | — | — | — |
| gqa2 | — | — | — | — | — | — |
| mla96 | — | — | — | — | — | — |
| mla96/ref | — | — | — | — | — | — |
| swa256 | — | — | — | — | — | — |
| swa256/ref | — | — | — | — | — | — |

SWA attention time: — ms/step; MHA: — ms/step (attention share × ms/step, plan §2.3 item 2).

Track `opt` is the variant on the path it trains with: opt mode for the grouped variants and SWA, and the naive path for MLA, whose absorbed path exists at inference only (plan §8.5).
Track `ref` is the reference implementation of MLA and SWA: plain PyTorch ops, explicit masks, no fusion (plan §2.3 item 3, §4.4).
MLA's and SWA's kernels are newer or hand-written, so their gap to MHA is partly implementation, not architecture (plan §13.3).
Spread is (max - min) / mean over the track's own repetitions, in percent.
Plan §8.5 reads: "Prediction: MHA, GQA-4, GQA-2 and MQA within ±3% of each other." Their step times are shown in the ms/step and ratio to MHA opt columns of the rows for MHA, gqa4, gqa2 and mqa.
Ratio is each track's ms/step over MHA's opt track. Attention share is the measured attention time (forward and backward) over the whole step's time, in percent (plan §8.5).
Peak memory is torch.cuda.max_memory_allocated; the timing document records no reserved figure.
