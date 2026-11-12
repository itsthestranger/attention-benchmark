# Attention Mechanisms at Small Scale — Implementation Plan v5

Nov 12, 2026 · @Phil

## 0. What changed in v5

v5 keeps the v4 backbone, variants, statistics and failure-mode review. It fixes four problems that would have cost reruns or produced a misleading result (rows 1–4), makes seven changes to power, metrics and scheduling (rows 5–11), and adds what an unattended build needs (rows 12–15). Supersedes v4; v4's own change table stays in the v4 document. Sections 3–8 are specification (freeze before writing code); 9–15 are execution; 16 is the decision log (append only); 17 defines the autonomous build loop and its handover.

**Read §13 before writing code.** It lists the failure modes that corrupt results silently. §13.6 (GradScaler step skipping) can still invalidate the iso-token arm while every curve looks healthy.

**Two thresholds are proposals.** The pairing switch (§1.2) and the seed-count rule (§1.4) live in `configs/prereg.yaml`. Phil confirms or changes them before the pre-registration commit. Neither blocks the build.

| # | Change | Where | Why |
| --- | --- | --- | --- |
| 1 | Naive-path MLA and `ref` SWA exist and pass smoke runs **before** `recipe.yaml` freezes | §7, §7.3, §15 | v4 froze the recipe and GradScaler settings in week 2 but built MLA and SWA in week 3. §13.6 says a later change means rerunning everything, including Arm N and the published floor post. |
| 2 | GPU work respread over four weeks, none above 30 GPU-h. Kaggle environment pinned; `make_tables.py` refuses runs that differ in torch, CUDA or driver version. | §10.6, §12, §15 | v4 put about 50 GPU-h in week 4 against a \~30 h quota. MHA and its comparators run weeks apart, so an image update between them is an uncontrolled change. |
| 3 | GQA/MQA `opt` decode groups queries instead of expanding K/V to 8 heads; new test | §2.3, §4.4, §5, §13.16 | Repeating KV heads keeps the cache small and the kernel's reads full-size, so the decode savings vanish. Same trap as fused but dense SWA, and RQ-I depends on it. |
| 4 | All LR sweeps run at the 100M budget instead of 50M | §7.4, §12.1 | Short horizons favour higher learning rates. The tie rule also compared 50M gaps against a σ\_init measured at 100M. |
| 5 | Three reproducibility pairs instead of one; a pre-registered switch between the paired test and Welch's test | §1.2, §8.1, §13.17 | Pairing rests on an untested assumption. If it gives nothing, the paired test wastes degrees of freedom: 3.2 × σ\_init detectable at 5 seeds against 2.5 for Welch. |
| 6 | Seed count for the primaries (5 or 8) set by a pre-registered rule from MHA data only | §1.4, §8.2 | v4 fixed 5 seeds without knowing σ\_init. Eight seeds lower the detectable gap to about 2.1 × σ\_init paired worst case, 1.8 with Welch. |
| 7 | L6 gets 5 seeds; the positive control is tested at the 300M result | §8.4, §9.4 | The one test that can end the project was the weakest: 3 seeds need a gap of 3.3 × σ\_d, 5 seeds need 1.7. v4 also scheduled it before MHA's 300M runs existed. |
| 8 | Context-use metrics bucket by same-document context, not absolute position | §6, §10.3 | With packing and no cross-document masking, position 512 is often not 512 tokens of context. |
| 9 | RQ-W answered first by a computed estimate; Arm C is optional and first on the cut list | §8.5, §12.2 | Arm C is largely predictable from numbers the plan already produces, and it is the most fragile arm. |
| 10 | Gate 1 passes on the recurrence anchor alone; the SWA prediction is restated | §9.2 | Eight stacked 256-token windows can relay information across about 2,040 tokens, so SWA may not fail at the window boundary. |
| 11 | Detectable-effect multipliers recomputed exactly | §1.3 | v4's values came from an approximation. Exact noncentral-t values are slightly larger: 2.3 instead of 2.2 at 5 seeds, 5.7 instead of 5.0 at 3. |
| 12 | Test tiers: `cpu`, `gpu`, `t4`; every §5 test names its tier | §5.1 | The build loop has no GPU. fp16, backend, memory and timing checks cannot run there and must not be skipped silently. |
| 13 | The harness is built by an autonomous loop that stops at a tested harness and a runbook. It launches no GPU job and reads no result. | §15, §17, `CLAUDE.md`, `TASKS.md` | An agent that writes the analysis and also sees the results is the risk §11 exists to prevent (§13.18). |
| 14 | Schedule restructured: build stage, verification on the 1070 and one T4 session, then four GPU weeks | §15 | The loop replaces week 1. The quota, not coding speed, sets the pace afterwards. |
| 15 | Decisions closed: decay shape, micro-batch rule, probe set, base init rule, MLA factor init, anchor architecture, null-control seeds. MLA standard-RoPE ablation deferred. | §3.1, §3.3, §9.2, §16 | An unattended build stalls or guesses on every open item. |

## 1. Research questions and decision rule

The project asks one thing first: at 14M non-embedding parameters, can attention-mechanism differences be told apart from seed noise at all? Everything else is conditional on that answer.

**Primary**

- **RQ-N, noise floor.** At 14M non-embedding params and 300M tokens, what is the seed-to-seed spread in validation loss? How much is init noise and how much is data-order noise?
- **RQ-D, detectability.** Which of the three primary gaps (§1.2) exceed that floor? For those that don't, what is the largest gap the data still rules out?
- **RQ-I, inference frontier.** At matched quality, how much decode throughput and KV memory does each mechanism buy on fixed hardware?

**Secondary**

- **RQ-E, effect size in parameters.** Expressed as an effective-parameter multiplier against the MHA ladder (§8.4), how large is each gap?
- **RQ-W, training wall-clock.** At a fixed training time budget, do SWA's savings or MLA's overheads change the ranking? Answered first by a computed estimate; Arm C is an optional confirmation (§8.5).
- **RQ-P** — At a fixed parameter budget, are KV heads or MLP width worth more?
- **RQ-M** — How far can KV bytes/token be cut before quality degrades measurably?
- **RQ-C, context use.** Do mechanisms differ in late-context loss even when aggregate loss ties?
- **RQ-R** — Does long-context retrieval degrade faster than perplexity as KV heads are removed?
- **RQ-X** — At what context length does each mechanism beat MHA in real wall-clock?

**Out of scope:** any claim about which mechanism is best at production scale. Say so in the first paragraph of anything published, not the conclusion.

### 1.1 Why v3's rule fails

v3 declared a gap detected when two variants' min–max seed ranges did not overlap. With 3 seeds each and no real difference, all three lowest runs come from one variant in 2 of the 20 equally likely orderings: a 10% false-positive rate per pair. Seven variants give 21 pairs, so about 2 spurious detections are expected even when nothing is real, and the null control itself is flagged 1 time in 10.

### 1.2 Pre-registered rule (commit to the README with a timestamp before any run)

> **Primary comparisons.** Exactly three, each against MHA: GQA-4, MQA, MLA-192. Each uses n init seeds with the data seed fixed. n is 5 or 8, set by the seed-count rule (§1.4) before any variant result at 300M is read.
>
> **Test selection.** Decided from MHA runs only, after Arm N. Let σ\_init be the SD of MHA seeds 1–5 at 100M. Let s\_rep = √(mean(δ²) / 2), where δ is the loss difference within each of the three reproducibility pairs (§8.1). If s\_rep < 0.6 × σ\_init, use the paired test. Otherwise use Welch's test.
>
> **Paired test.** For each seed k, compute d\_k = loss(variant, seed k) − loss(MHA, seed k) on the final validation loss at 300M tokens. Run a two-sided paired t-test on the n values of d\_k.
>
> **Welch's test.** A two-sided unequal-variance t-test on the n variant losses against the n MHA losses, same metric.
>
> **Correction.** Holm across the three comparisons at family-wise α = 0.05, for either test.
>
> **Reporting.** A gap is *detected* only if it survives Holm. Otherwise it is *not detected at this scale*. Either way, report the mean gap, its 95% confidence interval and every per-seed loss. For an undetected gap, the upper end of the interval is the finding: "gaps larger than X nats are ruled out".
>
> **Exploratory comparisons.** GQA-2, MLA-96 and SWA-256 run with 3 seeds. They are reported with intervals and labelled exploratory. No detection claim is made for them. MLA-96 and SWA-256 leave Arm A if the seed-count rule picks 8.
>
> **Middle-case rule.** If the floor lands near the observed gaps, report not-detected with the interval. Do not add seeds after any variant result at 300M has been read.

**Pairing is an assumption, not a given.** Data order is bit-identical across variants, and per-parameter seeding (§4.5) gives matching tensors identical values. But d\_ff differs across Arm A variants, so the MLP tensors, two thirds of the parameters, do not match. fp16 nondeterminism also makes identical configurations drift apart. If reruns of one configuration differ nearly as much as different seeds do, no common-mode noise is left to cancel, and the paired test only loses degrees of freedom.

The break-even is s\_rep ≈ 0.77 × σ\_init at 5 seeds and 0.88 at 8. The rule uses 0.6 because variants pair worse than identical reruns. The choice reads no variant data, so it cannot bias a comparison. `test_primary_stats.py` simulates the whole procedure, switch included.

### 1.3 Detectable effect, fixed before Gate 3

The smallest mean gap detected with 80% power, at Holm's strictest step (α = 0.05/3). σ\_d is the SD of the paired differences; its worst case, with pairing giving no benefit, is √2 × σ\_init.

| Seeds per variant | Test | Detectable gap | Worst case, in σ\_init |
| --- | --- | --- | --- |
| 3 | Paired | 5.7 × σ\_d | 8.0 |
| 5 | Paired | 2.3 × σ\_d | 3.2 |
| 5 | Welch | 2.5 × σ\_init | 2.5 |
| 8 | Paired | 1.5 × σ\_d | 2.1 |
| 8 | Welch | 1.8 × σ\_init | 1.8 |

With six primaries instead of three, the 5-seed paired multiplier would be 2.8. These are exact noncentral-t values; `test_primary_stats.py` reproduces the table.

The test is fixed at Gate 2. The bound in nats needs σ\_init at 300M, so it is committed when MHA's 300M runs finish (§9.4), together with the seed count, before any variant result at 300M is read. Report the realized σ\_d or group SDs afterwards.

### 1.4 Seed-count rule

Applied once, after Arm N and the ladder (§8.4), from MHA runs only:

1. Take the 5-seed detectable gap from §1.3 for the chosen test, in nats, using σ\_init at 300M.
2. Convert it with the ladder slope b at 300M: m = exp(−gap / b).
3. If 1 − m > 0.05, the 5-seed design cannot see a gap worth 5% of the parameters. Run 8 seeds on MHA and the three primaries (seeds 6–8 added) and drop MLA-96 and SWA-256 from Arm A. Otherwise run 5.

This is not adding seeds after seeing results: the rule is fixed in advance and reads no variant data.

The range-overlap scatter stays as a visual, never as a decision.

## 2. Hardware

All reported numbers come from Kaggle T4s; the GTX 1070 is for tests and debugging only.

### 2.1 Roles

| Machine | Role | Not used for |
| --- | --- | --- |
| GTX 1070 (local) | Shape debugging, the `gpu` tier of the correctness suite (§5.1), tiny-scale synthetics, plot iteration, config solver | Any timing number; any training run whose loss is reported |
| Kaggle T4 ×2 | All training, all timing, all inference benchmarks | — |
| Kaggle P100 | Fallback only | Timing runs; results are not comparable to T4 |

The autonomous build loop (§17) runs in a CPU-only session and executes the `cpu` test tier only (§5.1). Phil runs the `gpu` tier on the 1070 and the `t4` tier on Kaggle.

Kaggle free tier: \~30 GPU-h/week, 9–12 h session cap, 20 GB persistent storage. Check the quota counter at the start of each week.

**Open question, resolve at the verification step (§15):** does a T4 ×2 session draw quota at the single-GPU rate? If yes, concurrent runs nearly halve the quota cost in §12.

### 2.2 Two T4s: independent jobs, no DDP

At 17M params, DDP is all overhead and a new class of bugs. Run two independent single-GPU jobs with `CUDA_VISIBLE_DEVICES=0` and `=1` instead.

Concurrent jobs share dataloader CPU, host RAM bandwidth and PCIe. That is harmless when comparing on tokens and fatal when comparing on time.

> **Rule: every run that contributes a timing number runs alone on the box.** Arms A, B, N and the ladder may run concurrently. Arm C, the timing pass and `bench.py` may not.

### 2.3 Kernels on Turing

T4 is sm75. PyTorch's FlashAttention backend needs sm80+, so `scaled_dot_product_attention` uses the **memory-efficient (cutlass) backend**. It is fused, but it is not FlashAttention-2; don't call it that in the writeup. MHA, GQA and MQA route into it; MLA's absorbed path is hand-written, so its timings reflect kernel maturity as well as the algorithm.

**SWA correction to v3.** v3 warned that an explicit mask forces the unfused math backend. On recent PyTorch, the memory-efficient backend accepts an `attn_mask` (verify on the installed version). The real trap is different: **a masked dense kernel still computes every score and only zeroes some.** SWA implemented that way is fused, correct, and no faster than MHA.

The `opt` track for SWA must skip work outside the window:

1. **First choice:** FlexAttention with a sliding-window block mask. Check at the verification step (§15) that it compiles and runs on sm75, since Triton support for Turing has had gaps.
2. **Fallback:** a chunked implementation. Split the sequence into 256-token query chunks; each attends to its own chunk and the previous one via SDPA, with one fixed 256 × 512 mask.

Expected ceiling at ctx 1024: a causal kernel already skips the upper triangle, about 512 keys per query on average. Ideal SWA-256 processes about 224. That is ≤2.3× on attention alone, and attention is 15–25% of step time (§3.1), so expect at most 10–15% wall-clock savings. The chunked fallback works at 256-token block granularity: about 448 keys per query, only \~13% less than the causal kernel at ctx 1024, though \~4× less at ctx 4096. At ctx 1024, treat the fallback's speedup as near zero.

**GQA and MQA decode: the same trap in another place.** The common implementation repeats each KV head across its query heads before calling the kernel. The cache stays small, but the kernel reads a full-size copy, so decode is no faster than MHA and the savings RQ-I is meant to measure vanish. The `opt` decode path groups queries instead: with G KV heads, queries are shaped (B, G, H/G, d\_h) against keys and values of (B, G, T, d\_h). One new token needs no mask, so this is a plain SDPA call. Check on the installed PyTorch that SDPA does not copy its inputs; if it does, use an explicit matmul for decode and disclose it.

**Required, all four:**

1. Log the selected backend per run and assert it **inside the training loop**, not once at startup. Fail the run if it differs from the variant's expected backend.
2. For SWA, assert measured attention time is below MHA's in the timing pass. A fused-but-dense SWA passes the backend check and fails this one.
3. Report MLA and SWA timings in both `ref` and `opt` tracks (§4.4), with a disclosure line in the table caption (§13.3).
4. For GQA and MQA, assert that `opt` decode never holds K/V expanded to 8 heads: peak memory and decode time at cache 8k are below MHA's (§5).

## 3. Frozen architecture

The backbone is unchanged from v3: 8 layers, d\_model 384, 8 query heads of dim 48, 17,308,032 params. v4 corrects the FLOP-share rationale and adds MLA latent norms.

### 3.1 Backbone (identical across every variant)

| Component | Setting |
| --- | --- |
| Architecture | Decoder-only transformer, pre-norm |
| Layers | 8 (6 and 10 for the ladder only, §8.4) |
| d\_model | 384 |
| Query heads × head dim | 8 × 48 |
| Vocab | 8,192 |
| Context | 1,024 |
| Position | RoPE, θ = 10,000 |
| Norm | RMSNorm |
| Activation | SwiGLU |
| Embeddings | Tied |
| Biases | None |
| Init | Normal, std 0.02, per-parameter seeded (§4.5); residual output projections (W\_O, MLP out) scaled by 1/√(2 × layers); norm gains 1; MLA factor pairs per §3.3 |
| Precision | fp16 autocast + GradScaler, fp32 master weights |

**Not bf16.** Turing has fp16 tensor cores but no bf16 support; `torch.cuda.is_bf16_supported()` returns False on T4. Assert the dtype at startup and fail loudly on mismatch. The fp16 choice brings the confound in §13.6.

**Attention share of forward FLOPs, corrected.** v3 quoted 18% / 31% / 47% at ctx 512 / 1024 / 2048. Those figures omit the LM head (6.3M FLOPs/token, as large as two layers) and count the masked upper triangle that fused causal kernels skip.

| Context | Share, dense count | Share, causal (what the kernel does) |
| --- | --- | --- |
| 512 | 15% | 8% |
| 1,024 | 27% | 15% |
| 2,048 | 42% | 27% |

Ctx 1024 stays. Attention's share of *wall-clock* is higher than its FLOP share, because attention kernels at head dim 48 run far below matmul efficiency. Measure the real share in the timing pass (§8.5) and quote that, not the FLOP count. Whether to add ctx 2048 points is open decision 8.

### 3.2 Variant table

Target: **17,308,032 params** (14,162,304 non-embedding). Savings in the KV projections move into d\_ff.

```latex
P_{\text{attn}} = 2d^2 + 2d\,(h_{kv}\, d_h) \qquad P_{\text{FFN}} = 3d\,d_{\text{ff}}
```

| Variant | KV heads | d\_ff | Attn params/layer | KV bytes/token/layer (fp16) | Total params | Role |
| --- | --- | --- | --- | --- | --- | --- |
| MHA | 8 | 1,024 | 589,824 | 1,536 | 17,308,032 | Reference, 5 or 8 seeds (§1.4) |
| GQA-4 | 4 | 1,152 | 442,368 | 768 | 17,308,032 | Primary, 5 or 8 seeds |
| MQA | 1 | 1,248 | 331,776 | 192 | 17,308,032 | Primary, 5 or 8 seeds |
| MLA-192 | — | 1,048 | 562,560 | 432 | 17,311,104 | Primary, 5 or 8 seeds |
| GQA-2 | 2 | 1,216 | 368,640 | 384 | 17,308,032 | Exploratory, 3 seeds |
| MLA-96 | — | 1,224 | 359,616 | 240 | 17,309,568 | Exploratory, 3 seeds; leaves Arm A if §1.4 picks 8 |
| SWA-256 | 8 | 1,024 | 589,824 | 1,536 (capped at 256 tokens) | 17,308,032 | Exploratory, 3 seeds; leaves Arm A if §1.4 picks 8 |
| Null control | 8 | 1,024 | 589,824 | 1,536 | 17,308,032 | MHA under another name, seeds 11–15 |

All non-MLA rows hit the target exactly with integer d\_ff divisible by 8, a lucky property of d\_model 384 at these head counts. It will not survive a change of d\_model; the solver (§4.3) checks.

**MLA rows now include latent RMSNorms** on c\_KV and c\_Q, as in DeepSeek-V2. That adds 2 × d\_c params per layer: +3,072 total for MLA-192 (+0.018%) and +1,536 for MLA-96 (+0.009%). One unit of d\_ff costs 1,152 params, so the offset can't be absorbed. Accept it and print it in the table. The solver asserts exact match for non-MLA variants and |offset| < 0.05% for MLA.

Ordering by KV bytes: MQA 192 < MLA-96 240 < GQA-2 384 < MLA-192 432 < GQA-4 768 < MHA 1,536. MLA and GQA points interleave, which keeps the frontier plot readable.

MLA counts assume d\_c = d\_c^Q, d\_h^R = 24 and a RoPE key shared across heads:

```latex
P_{\text{MLA}} = \underbrace{d\,d_c + 2\,d_c(h\,d_h)}_{\text{KV down/up}} + \underbrace{d\,d_h^R}_{\text{shared }k_R} + \underbrace{d\,d_c^Q + d_c^Q(h\,d_h) + d_c^Q(h\,d_h^R)}_{\text{query side}} + \underbrace{(h\,d_h)\,d}_{W_O} + \underbrace{d_c + d_c^Q}_{\text{latent norms}}
```

### 3.3 MLA implementation notes

**Decoupled RoPE is required.** RoPE applied to k\_C = W\_UK c\_KV would sit between W\_UQ and W\_UK, so the absorbed product could not be precomputed and the efficiency argument collapses.

```
c_KV = RMSNorm(W_DKV · h)        (d_c = 192)
k_R  = RoPE(W_KR · h)            (d_h^R = 24, shared across heads)
cache = [c_KV ; k_R]             → 216 values/token/layer
c_Q  = RMSNorm(W_DQ · h)
q = [q_C ; q_R],  k = [k_C ; k_R]   (k_R broadcast over heads)
```

**Softmax scale is 1/√(d\_h + d\_h^R) = 1/√72, not 1/√48.** The model trains either way, slightly worse when wrong. Assert it in a unit test.

**Initialize the factorized pairs deliberately.** If W\_DKV and W\_UK both use the backbone's default std, their product's variance is roughly squared and MLA starts with much smaller logits than MHA. The latent RMSNorm fixes the scale of c\_KV but not of the up-projections. v5 fixes the choice: scale each up-projection so that its output variance matches the dense projection it replaces. Because the latent is RMS-normalized, the rule is exact: an up-projection from a latent of width d\_c uses std 0.02 × √(d / d\_c), which is 0.02 × √2 for MLA-192 and 0.04 for MLA-96. Down-projections keep std 0.02, since the norm erases their scale. The SVD alternative is dropped; it adds a second init path for no measurable benefit.

This applies to all four pairs: W\_UK·W\_DKV, W\_UV·W\_DKV, W\_UQ·W\_DQ, W\_QR·W\_DQ. The step-0 logit-scale test (§5) catches this class of bug for every variant.

Build the naive path first (materialize k\_C and v\_C). Write the absorbed path only for `bench.py`, gated by a test asserting agreement to 1e-4.

## 4. Code structure

One skeleton, one swappable attention module, and every reported number generated by code rather than typed by hand.

### 4.1 The invariant

Tokenizer, embeddings, MLP shape rule, RMSNorm, init rule, optimizer and data order are byte-identical across variants. **The only code that differs between variants lives under `src/attention/`.** A CI check diffs the config-resolved model graph and fails if anything outside the attention subtree differs, except d\_ff in Arm A and layer count in the ladder.

### 4.2 Interface

```python
class AttnBase(nn.Module):
    def __init__(self, cfg: AttnConfig): ...

    def forward(self, x, freqs, cache=None, *, mode="ref") -> tuple[Tensor, Cache]:
        """mode: 'ref' (naive, readable) | 'opt' (fused / block-skipping / absorbed)."""

    @property
    def n_params(self) -> int: ...

    @property
    def kv_bytes_per_token(self) -> int:
        """Per layer, at cfg.kv_dtype. Analytic."""

    def flops_per_token(self, T: int, causal_skip: bool = True) -> int: ...

    @property
    def sdpa_backend(self) -> str | None:
        """Which fused kernel this variant actually used, or None."""
```

The results table generates from these properties. Never compute KV bytes in a spreadsheet. `bench.py` checks the analytic value against a measured `torch.cuda.memory_allocated` delta during decode and asserts agreement within 5%.

### 4.3 Config solver

`scripts/solve_config.py` takes a variant and a target, solves d\_ff in exact integer arithmetic, builds the model and asserts `sum(p.numel() for p in model.parameters())`. Non-MLA variants must match exactly; MLA must be within 0.05% with the offset recorded (§3.2). It emits resolved YAML, runs in CI, and fails loudly rather than rounding.

### 4.4 Reference and optimized tracks

- **`ref`**: plain PyTorch ops, explicit masks, no fusion. Used for correctness, gradient checks, and the algorithmic speed comparison where every variant is equally unoptimized.
- **`opt`**: best reasonable implementation. Fused SDPA for MHA/GQA/MQA with grouped-query decode (§2.3), block-skipping SWA (§2.3), absorbed MLA at inference. Used for practical numbers.

Report both. A mechanism can look slow because its implementation launches many small kernels; conflating that with an algorithmic property is the most common way these posts mislead.

### 4.5 Making "same init" true

Seeding one global RNG fails because d\_ff differs across variants. The stream diverges at the first differing tensor, and every later tensor differs even where shapes match. Seed per parameter instead:

```python
def init_param(name: str, tensor: Tensor, run_seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(zlib.crc32(name.encode()) ^ run_seed)
    # ... init from g
```

`layers.3.mlp.gate.weight` now gets identical values across variants whenever its shape matches. In Arm A the MLP shapes differ with d\_ff, so across variants only the embeddings and the unchanged attention projections match. That gives pairing its best chance; whether it helps is measured in Arm N (§1.2). Data order uses an independent generator seeded by `data_seed`, so the token stream is bit-identical across all variants and init seeds.

Per-parameter seeding controls which random numbers a tensor gets, not their scale. It does nothing for the low-rank variance problem in §3.3.

### 4.6 Repo layout

```
attention-benchmark/
├── configs/
│   ├── base.yaml                     # backbone, frozen
│   ├── recipe.yaml                   # optimizer, batch, WSD, scaler — frozen after §7
│   ├── prereg.yaml                   # primaries, α, test and seed-rule thresholds; frozen by Phil
│   ├── variants/{mha,gqa4,gqa2,mqa,swa256,mla192,mla96,null}.yaml
│   ├── ladder/{mha_L6,mha_L10}.yaml
│   └── arms/{iso_token,iso_backbone,iso_wallclock}.yaml
├── src/
│   ├── model.py  transformer.py
│   ├── attention/
│   │   ├── base.py                   # AttnBase + registry
│   │   ├── grouped.py                # MHA/GQA/MQA, one file, n_kv_heads knob
│   │   ├── sliding_window.py         # flex + chunked opt paths
│   │   └── mla.py
│   ├── training/{trainer,optimizer,checkpoint,schedule}.py   # schedule = WSD + branch
│   ├── data/{tokenizer,packing,loader}.py
│   ├── eval/{lm,positions,synthetic,retrieval,inference}.py
│   └── utils/{manifest,seeding,flops}.py
├── analysis/
│   ├── primary_test.py                # §1.2, the only code allowed to say "detected"
│   ├── ladder_fit.py                 # §8.4 multipliers, §1.4 seed-count rule
│   └── rqw_estimate.py               # §8.5, computed wall-clock estimate
├── tests/
│   ├── test_causality.py  test_reference.py  test_kv_cache.py
│   ├── test_limits.py                # GQA→MHA, GQA→MQA
│   ├── test_grouped_decode.py        # no expanded K/V in opt decode
│   ├── tolerances.py                 # every tolerance in one frozen file
│   ├── test_params.py                # exact counts, all variants
│   ├── test_mla_absorbed.py
│   ├── test_wsd_branch.py
│   └── test_primary_stats.py          # false-positive rate on simulated nulls
├── scripts/
│   └── solve_config.py  train.py  branch.py  bench.py  timing_pass.py  evaluate.py  make_tables.py
├── results/
│   ├── manifests/                    # one JSON per run and per branch
│   └── tables/                       # auto-generated, never hand-edited
├── docs/
│   ├── plan_v5.md                    # this document, frozen
│   └── decisions.md                  # build-loop decision log, append only
├── CLAUDE.md  TASKS.md  STATUS.md  RUNBOOK.md
└── README.md                         # pre-registration lives here
```

`test_primary_stats.py` simulates thousands of null experiments with realistic noise. It asserts the §1.2 procedure's family-wise false-positive rate is at most 5%, with the test switch and both seed counts included. That test is the guard against a statistics bug, the one error no loss curve reveals.

## 5. Correctness suite

Every test is written, and every `cpu` test passes, before the handover (§17.4). The `gpu` and `t4` tests pass before the first reported Kaggle hour is spent. v5 adds two rows and the tier column.

| Test | Assertion | Tier |
| --- | --- | --- |
| Causal leak | Perturb token t+1; output at position t is bit-identical. Every variant, every mode. | cpu; gpu for fp16 |
| Reference agreement | `opt` matches `ref` on output, loss, input grads and every parameter grad: 1e-6 (fp32), 1e-3 (fp16). | cpu (fp32); gpu (fp16) |
| GQA limits | n\_kv\_heads = n\_q\_heads ≡ MHA; n\_kv\_heads = 1 ≡ MQA. Same code path, so this catches repeat/indexing bugs. | cpu |
| KV-cache decode | Cached token-by-token logits match the full forward. Run at ctx 1024; cache bugs hide at short lengths. | cpu |
| MLA absorbed | Absorbed path matches naive path to 1e-4. | cpu |
| MLA scale | Softmax scale is 1/√(d\_h + d\_h^R). | cpu |
| Param count | Non-MLA variants hit 17,308,032 exactly; MLA within 0.05% with the recorded offset. Ladder sizes match §8.4. | cpu |
| SWA window | Weights outside the window are exactly zero, including at sequence boundaries. | cpu |
| SWA work skipping | In `opt` mode, SWA attention time at ctx 1024 is below MHA's. Catches a fused but dense SWA. On cpu, a structural check on the chunked path: no query is scored against more than 512 keys. | cpu (structure); t4 (timing) |
| GQA/MQA decode (new) | In `opt` decode, K/V reach the kernel with their own head count, never expanded to 8. Peak memory and decode time at cache 8k are below MHA's. | cpu (shapes); gpu (memory); t4 (timing) |
| Backend selection | Each variant selects its expected backend; any fallback to math fails. | gpu, t4 |
| Init logit scale | Step-0 attention-logit std agrees within 10% across all variants (§3.3). | cpu |
| WSD branch fidelity | A decay branched from a trunk checkpoint matches an uninterrupted WSD run of the same length within `opt` tolerance. | cpu; gpu for fp16 |
| Resume fidelity | Kill at step 500, resume, diff the curve against an uninterrupted run, within `opt` tolerance. | cpu; gpu for scaler state |
| Determinism (`ref` only) | Same config and seeds give bit-identical loss for 100 steps, twice. | cpu, gpu |
| Reproducibility (`opt`) | Two identical runs land within a measured, recorded tolerance. | gpu, t4 |
| Context-bucketed loss | Loss bucketed by same-document context, weighted by bucket size, equals aggregate loss. Hand-built windows with mid-window document starts give the expected context values (§6). | cpu |
| Primary statistics | Simulated nulls give family-wise false-positive rate ≤ 5% under the full §1.2 procedure: test switch, Holm correction, both seed counts. The §1.3 table is reproduced. | cpu |
| Manifest gates (new) | `make_tables.py` refuses fake manifests that differ by more than 0.5% in steps taken, in backend, or in torch, CUDA or driver version, and refuses trunk checkpoints. | cpu |

### 5.1 Test tiers

- **`cpu`**: fp32, tiny configs, no CUDA. Runs in CI on every pull request and in the build loop.
- **`gpu`**: needs any CUDA card. fp16 tolerances, GradScaler state, backend selection, memory. Phil runs `pytest -m gpu` on the 1070.
- **`t4`**: needs the Kaggle T4. Anything specific to sm75 or based on timing. Phil runs `pytest -m t4` in one short session.

The build loop writes the tests of all three tiers. `gpu` and `t4` tests skip cleanly without the hardware. A skipped test is never counted as passed: the handover lists every test that has not yet run (§17.4). All tolerances live in one file, `tests/tolerances.py`, which is frozen (§17.3).

**Determinism is only achievable in `ref` mode.** `torch.use_deterministic_algorithms(True)` disables fused attention backward and throws on some SDPA paths. In `opt` mode, measure the run-to-run tolerance once and record it. That tolerance is a floor beneath the noise floor and belongs in the writeup.

**Resume and branch fidelity are the tests people skip.** On Kaggle you will resume constantly, and every WSD branch is a resume from a trunk checkpoint. A resume must restore data position, schedule position, optimizer state, GradScaler state and RNG state. Most hand-rolled trainers get the data position or the schedule step wrong first time, and the curve looks plausible rather than broken.

## 6. Data

Tokenize once, hash everything, and make `train.py` refuse to start on a hash mismatch.

- **Corpus:** FineWeb-Edu, deterministic subset. Fix the shard list; record a SHA-256 of the concatenated token array.
- **Tokenizer:** BPE, vocab 8,192, trained once on a fixed shard with a fixed seed. Commit the files and their hash. Never retrain.
- **Splits:** 400M train / 4M val / 4M test tokens, disjoint at document level. Freeze once.
- **Packing:** documents joined with an EOS separator, chunked to 1,024. No cross-document masking. Store a sidecar array of document start offsets with the tokens; the context-use metrics need it (§10.3).
- **Smoke slice:** first 5M train tokens, for implementation smoke tests only.
- **Token use:** every run reads the train stream from token 0 in the same order. The longest run reads 300M tokens; WSD branches read the same tokens the trunk would read next.

Keep synthetic-task data out of the LM corpus.

Store the tokenized corpus as a versioned Kaggle Dataset (\~800 MB as `uint16`) and mount it read-only. 400M tokens on the 1070's host CPU take hours, so this happens exactly once.

**Why the hash matters.** Re-tokenizing mid-project (a tokenizer tweak, a shard change, new packing) makes every earlier run incomparable to every later one, and no metric shows it. The dataset hash in the run manifest (§10.5) is the only defence.

**Context-use evaluation needs no new data, but absolute position is the wrong axis.** With packing and no cross-document masking, a window often starts mid-document and often holds a second document later on, so position 512 is not 512 tokens of context. For every validation token compute c, the number of tokens of the same document that precede it inside the window, from the document offsets. All context-use metrics bucket by c (§10.3).

## 7. Training recipe

One frozen recipe for every run: AdamW, 65,536 tokens per step, and a warmup-stable-decay schedule whose decay branches give several token budgets per run. Freeze `recipe.yaml` only after all seven variants have passed their smoke runs (§15); changing it later restarts every arm.

### 7.1 Optimizer and batch

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, β = (0.9, 0.95), ε = 1e-8 |
| Weight decay | 0.1 on 2D weight matrices; none on norm gains or the tied embedding |
| Gradient clipping | Global norm 1.0 |
| Global batch | 64 sequences × 1,024 = 65,536 tokens/step |
| Micro-batch | Largest that fits the most memory-hungry variant (MLA naive path) in `ref` mode; identical for all variants, gradient accumulation makes up the rest |
| Dropout | 0 |
| Budgets in steps | 50M = 763 steps · 100M = 1,526 · 200M = 3,052 · 300M = 4,578 |

Budgets are defined in steps, so token counts are exact multiples of the batch. Small models are token-efficient at small batches; 65,536 keeps 4,578 optimizer steps at 300M, enough for the 0.5% step-parity rule (§13.6) to mean something.

### 7.2 WSD schedule with decay branches

Warmup-stable-decay matches cosine at equal budget when the decay is about 20% of the run ([Hägele et al., 2024](https://arxiv.org/abs/2405.18392)). Its advantage here: the stable trunk is shared, so a short decay branched from a trunk checkpoint gives the result for a shorter budget at a fraction of the cost.

- **Warmup:** linear over a fixed 100 steps, identical for every budget.
- **Stable:** constant peak LR.
- **Decay:** the last 20% of the budget's steps, to zero, with the 1−√ shape. Decay starts at step 0.8 × S for budget S.
- **Branches:** checkpoint the trunk at each branch point. The 100M result decays from step 1,221 to 1,526. The 300M result is the main run, decaying from step 3,662 to 4,578.
- **Cost:** a run with 100M and 300M results costs 4,883 steps instead of 4,578, +6.7%. Adding a 200M branch costs another +13%; the ladder (§8.4) takes it, other runs don't.

This replaces v3's separate 100M screening runs and its fragile Arm C schedule. **Never report a trunk checkpoint as a result.** Like a truncated cosine, it is still at peak LR and reports a falsely high loss.

### 7.3 GradScaler

Set `init_scale` and `growth_interval` once, globally, from the smoke runs: at most 3 skipped steps in the first 100, for every variant. All seven variants, MLA and SWA included, must exist and run before these are set. Call `scaler.unscale_` before gradient clipping, once per optimizer step. Never tune per variant. Steps actually taken must agree within 0.5% across any compared runs; at 300M that is 23 steps.

### 7.4 Learning-rate sweep

v3's grid (1.5e-4 to 6e-4) is likely below the optimum for a 17M-parameter model, where AdamW optima usually sit near 1e-3 to 3e-3. A grid whose winner sits at its edge has not found the optimum.

1. **MHA first, 5 points:** 3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3, at the 100M budget (1,526 steps), one seed. v4 swept at 50M; short horizons favour higher rates, and 100M is where Arm N measures σ\_init. This precedes Arm N, because the noise floor must be measured at MHA's tuned LR.
2. **Every other variant and both ladder sizes, 3 points:** ½×, 1× and 2× MHA's winner.
3. **Edge rule:** if a winner is at the edge of its grid, add one point beyond it. Repeat until it isn't.
4. **Tie rule:** if a variant's best LR beats MHA's LR by less than σ\_init at 100M from Arm N, use MHA's LR. A single-seed sweep otherwise selects on noise.
5. **Record everything,** including divergences. Report shared-recipe and tuned results side by side; the gap between them is itself a finding.

## 8. Arms, controls and the calibration ladder

Six pieces: the noise floor with a null control, the main iso-token arm, an iso-backbone check, a ladder that serves as positive control and unit of measurement, a wall-clock estimate with an optional confirming arm, and an inference benchmark where GQA, MQA and MLA actually pay off.

### 8.1 Arm N — noise floor and null control (gates every comparison)

No cross-variant number is computed or viewed until this exists.

- **Init noise:** MHA, init seeds 1–5, data seed 1. These are also Arm A's MHA runs, so σ\_init at 100M (branch) and 300M (main) costs nothing extra.
- **Data-order noise:** MHA, init seed 1, data seeds 2 and 3, 100M budget. Data seed 1 is the Arm A run's 100M branch.
- **Null control:** MHA under the name `null`, init seeds 11–15, data seed 1, 100M budget. It goes through `primary_test.py` against MHA's 100M branches exactly like a variant. Its seeds are unrelated to MHA's, so pairing gives it no benefit: a conservative check that the procedure does not invent findings. It also gives a second estimate of σ\_init. Seeds 11–15 leave 6–8 free for the seed-count rule (§1.4).
- **Reproducibility pairs:** MHA init seeds 1–3 rerun at the 100M budget with identical config and seeds. The three within-pair differences give s\_rep, the floor beneath σ\_init and the input to the test selection (§1.2).

**Deliverable at Gate 2:** σ\_init, σ\_data and s\_rep at 100M, the null-control verdict, the test choice, and a scatter of every run. This is the standalone post (§14.4). σ\_init at 300M and the detectable-effect bound in nats follow when MHA's 300M runs finish (§9.4).

### 8.2 Arm A — iso-parameter, iso-token

All variants at \~17.3M params, 300M tokens, each with a 100M branch, data seed 1.

- **Primary, 5 or 8 init seeds (§1.4):** MHA, GQA-4, MQA, MLA-192.
- **Exploratory, 3 init seeds:** GQA-2, MLA-96, SWA-256. If §1.4 picks 8 seeds, MLA-96 and SWA-256 leave this arm.

The validity check: does the tiny setup recover the ordering of [DeepSeek-V2](https://arxiv.org/abs/2405.04434) Appendix D at 7B? If it recovers nothing, and the ladder shows the pipeline can detect real effects, that is the result. Answers RQ-D, RQ-E, RQ-P, RQ-M, RQ-C.

### 8.3 Arm B — iso-backbone

d\_ff fixed at 1,024 for every variant; total params drift down as KV heads are removed. Only GQA-4, GQA-2 and MQA run, because the MLP-width confound (§13.1) is largest for them; MLA-192's iso-param d\_ff is already 1,048. 3 seeds each, 100M budget. MHA's reference is Arm A's 100M branches, which are the identical config.

The question is whether the ranking flips against Arm A's 100M branches. Report Arm A minus Arm B for MQA as "what the 22% wider MLP bought". Exploratory; no detection claims.

### 8.4 Calibration ladder — positive control and unit of measurement

MHA at 6 and 10 layers, everything else identical, 300M tokens with 100M and 200M branches. L6 runs 5 seeds; L10 runs 3. With the 8-layer Arm A runs, this gives three model sizes.

| Ladder point | Layers | Non-embedding params | vs. 8-layer | Seeds |
| --- | --- | --- | --- | --- |
| L6 | 6 | 10,621,824 | 0.75× | 5 |
| L8 (Arm A MHA) | 8 | 14,162,304 | 1.00× | 5 or 8 |
| L10 | 10 | 17,702,784 | 1.25× | 3 |

**As a positive control.** Pre-register L6 vs. L8 as a single test on seeds 1–5 at the 300M result, α = 0.05, using the test chosen in §1.2. It therefore runs after MHA's 300M runs finish. Layers 0–5 share parameter names, so pairing is partial. With 5 seeds the paired test needs a gap of 1.7 × σ\_d and Welch's needs 2.0 × σ\_init; v4's 3 seeds needed 3.3 × σ\_d. If a 25% capacity cut is not detected, the pipeline cannot see effects of that size. Every null is then reported as uninformative, and the project ships as the noise-floor post.

**As a unit.** Fit final loss against ln(non-embedding params) across the three points: L ≈ a − b ln N. A variant gap Δ then converts to an effective-parameter multiplier exp(−Δ / b): "MQA performs like MHA with 0.94× the parameters". Propagate the interval on Δ and the fit uncertainty on b into the multiplier. The 100M/200M/300M branches give the same conversion along the token axis. The slope b at 300M also feeds the seed-count rule (§1.4).

### 8.5 RQ-W — a computed estimate first, Arm C as optional confirmation

At iso-param, GQA and MQA compute the same attention scores as MHA, because all keep 8 query heads; their KV-projection savings went into d\_ff. Their training step times should therefore match MHA's. Only SWA and MLA can differ.

**Step 1 — timing pass (all 7 variants).** Dedicated GPU, 220 steps each with the first 20 discarded, three round-robin repetitions. Each variant is timed on the path it trains with: `opt` for the grouped variants and SWA, the naive path for MLA, whose absorbed path exists at inference only. Records ms/step, attention's share of step time, and peak memory. **Prediction:** MHA, GQA-4, GQA-2 and MQA within ±3% of each other. If confirmed, the timing table is the evidence that wall-clock cannot separate them.

**Step 2 — computed estimate (`rqw_estimate.py`).** Budget B is MHA's measured time for 1,526 steps (100M tokens). SWA-256 and MLA-192 each afford S\_v = round(B / t\_v) steps in that time. Read each variant's loss at S\_v steps from its own Arm A results, interpolating between its 100M and 300M branches linearly in ln(tokens). Compare with MHA's 100M result and report it as an estimate, with the interpolation stated. If SWA-256 has left Arm A (§1.4, §12.2), report its step-time ratio only.

**Step 3 — Arm C runs (optional, first on the cut list).** MHA, SWA-256 and MLA-192, 3 seeds each, to confirm the estimate.

1. Each variant gets S\_v steps as above. Its WSD decay covers the last 20% of S\_v.
2. Report actual elapsed time beside B; flag any run more than 3% off.

All 9 runs (\~4.5 h) execute in **one session**, round-robin: MHA s1, SWA s1, MLA s1, MHA s2, and so on. Measure step time at three points and report the spread. If the session dies, restart the whole arm; never stitch sessions. MLA trains on the naive path, so its Arm C number is the materialized-key cost, not the absorbed one.

### 8.6 Inference frontier — where KV savings show up (new emphasis)

`bench.py`, dedicated GPU, `opt` mode, absorbed MLA, grouped-query decode for GQA and MQA (§2.3). Decode tok/s at batch 1, 8 and 32 × cache 1k, 4k and 8k; prefill latency and peak memory; maximum batch before OOM at 8k.

MHA's cache is 1,536 B × 8 layers = 12 KB per token, about 100 MB per sequence at 8k. The fp16 weights are \~35 MB. At long cache lengths KV reads dominate decode bandwidth even at batch 1, so real, measurable gaps are expected here. Throughput past the trained context of 1,024 measures speed only, never quality.

The frontier plot: x = decode tok/s at batch 32 and cache 4k, y = 300M validation loss with every seed shown. Answers RQ-I.

### 8.7 Held constant

Corpus, splits, tokenizer, token order, packing, context, the whole of `recipe.yaml` (§7), init rule, eval and checkpoint frequency. Varying: attention module; d\_ff (Arm A); layer count (ladder only); init seed; data seed (Arm N only).

Verify after the fact: optimizer steps actually taken (§13.6) and backend per variant (§2.3). `make_tables.py` refuses to compare runs that disagree on either, or on torch, CUDA or driver version.

## 9. Phases and gates

Seven phases and five checkpoints. Two of them can end the project early: a null control that gets flagged means the pipeline is broken, and a positive control that isn't detected means every null is uninformative. Gate 1 can no longer cancel the LM arms.

&#91;embedded content: roadmap · 7 phases, 5 checkpoints\]

Every gate except Gate 4 lets the project continue on both outcomes; the right-hand lines say what changes.

### 9.1 Phase 0 — harness

Backbone, grouped attention, config solver, correctness suite, WSD branching, manifest logging, statistics code, and the results table and plots built with every cell blank. **Building the table before any run is what stops a story being found in noise afterwards.** The autonomous loop builds all of Phase 0, including the MLA, SWA, probe and benchmark code that v4 scheduled later, and stops at the handover (§17).

### 9.2 Phase 1 — synthetic probes, and why Gate 1 changed

[Zoology](https://arxiv.org/abs/2312.04927) introduced MQAR and showed that softmax attention solves it at sizes where gated convolutions and recurrences fail. Every attention variant here, MQA included, keeps a full per-token KV cache, so none is state-limited the way a recurrence is. v3's Gate 1 assumed MQA would "fall off a cliff". If instead all attention variants saturate, as is likely, v3 would have cancelled the LM study on a probe that was never able to separate them.

**Probe set** (decided in v5, decision 30):

1. **Standard MQAR**, 4 → 256 pairs. A sanity check: every softmax variant should solve it; the recurrence anchor should degrade as pairs grow.
2. **Multi-hop recall:** a value is reused as the key of a second lookup.
3. **High-cardinality MQAR at head dim 16**, to stress what a single shared KV head can separate.
4. **Fuzzy recall:** keys perturbed by noise tokens.
5. **Selective copy and induction**, as in v3.

Probe models keep the backbone's depth, head count and norm, and shrink width only as the task forces. Run 10+ seeds per point; these runs take seconds.

**Recurrence anchor.** The attention module is replaced by a per-channel gated linear recurrence: s\_t = g\_t ⊙ s\_{t−1} + (1 − g\_t) ⊙ W\_i x\_t, with g\_t = sigmoid(W\_g x\_t) and output W\_o s\_t. Its state is a single d\_model vector, which is why it must fail as the number of pairs grows.

**Predictions, committed before running:** all softmax variants saturate probe 1; the anchor degrades with pair count; SWA degrades once dependencies exceed its window but fails outright only beyond its relay range, about layers × (window − 1) ≈ 2,040 tokens, because stacked windows can pass information along. Differences among MHA, GQA, MQA and MLA, if any, appear first in probe 3.

**Gate 1** passes when the anchor fails where predicted, which proves the probes can discriminate. SWA's behaviour is reported, not gating. Attention variants tying is then a reported result. If even the anchor doesn't fail, the probes are broken and get redesigned. Either way the LM arms run.

### 9.3 Phase 2 — MHA learning rate, then the noise floor

MHA's LR sweep comes first, because the floor must be measured at MHA's tuned LR (§7.4). Then Arm N (§8.1).

**Gate 2:** σ\_init, σ\_data and s\_rep reported at 100M; the null control *not* detected; the test choice (§1.2) committed; the floor post drafted with a date. The detectable-effect bound in nats and the seed count are committed in Phase 3, when MHA's 300M runs and the ladder exist (§1.3, §1.4). A detected null control means a bug in pairing, seeding, the statistics code or determinism. Stop and fix it; nothing downstream is valid until it passes.

### 9.4 Phase 3 — sweeps, ladder and timing; the positive control

LR sweeps for every other variant and both ladder sizes, the L6 and L10 runs, MHA's 300M runs, and the timing pass. **Positive control:** L6 vs. L8, a single test on seeds 1–5 at the 300M result, α = 0.05 (§8.4). Then the detectable-effect bound and the seed count are committed (§1.3, §1.4), before any variant result at 300M is read. If it fails, the project ships as the noise-floor post with the finding "even a 25% capacity cut was invisible at this budget". That is still a useful result.

### 9.5 Phase 4 — Arm A, and Gate 3

Before reading any loss comparison, `make_tables.py` confirms steps-taken parity within 0.5% and backend parity. Then `primary_test.py` runs the three primary tests. Gate 3 has no failure branch: a detection and a ruled-out bound are both results. What it decides is whether Gate 4 has anything to work with.

### 9.6 Phase 5 — Arm B, Arm C and the inference frontier

Arm B at 100M (§8.3), the RQ-W estimate and, if kept, Arm C (§8.5), `bench.py` for the decode frontier (§8.6).

### 9.7 Phase 6 — long context and retrieval

Passkey and needle-in-a-haystack at 1k → 8k with distractor sweeps, in two modes. **Zero-shot** evaluates the LM-trained models; **trained** trains fresh models on the synthetic task.

**Gate 4** runs Phase 6 only if a primary gap was detected or a late-context gap's interval excludes zero (§10.3). RoPE beyond 1,024 is its own confound: train a ctx-4,096 variant or use a documented extension method, and say which (open decision 5).

### 9.8 DSA — deferred

Not in v1. Its machinery pays off at ≥4k context; a meaningful experiment compares dense MLA, SWA-MLA and DSA at long context on modern hardware. Implement it on the 1070 for correctness only, as a follow-up post.

## 10. Metrics, plots and manifest

The headline is a paired gap with its interval and its parameter equivalent, never a bare mean. Every number is generated by `make_tables.py` from manifests.

### 10.1 Headline table (one row per variant)

| variant | params | d\_ff | KV bytes/token | train ms/step | decode tok/s (batch 32, cache 4k) | val loss, every seed | gap vs MHA \[95% CI\] | effective-param multiplier | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Verdict is one of *detected*, *not detected* (with the ruled-out bound) or *exploratory*, and is written by `primary_test.py` only. KV bytes and decode throughput sit in the headline because they are what GQA, MQA and MLA exist to improve.

### 10.2 Recorded per run

Train, val and test loss; loss against tokens, wall-clock and estimated FLOPs; tokens/s; time-to-loss-X; peak allocated and reserved memory; GPU utilization; context-bucketed loss (§10.3) at every eval.

### 10.3 Context-bucketed loss and context use

- **Axis:** c, the number of tokens of the same document that precede a token inside the window (§6). Not absolute position.
- **Buckets:** c in \[0, 64), \[64, 256), \[256, 512), \[512, 1024).
- **Late-context loss:** mean loss at c ≥ 512. SWA-256 should diverge from MHA here first.
- **In-context-learning score:** mean loss at c 450–550 minus mean loss at c 40–60, after [Olsson et al., 2022](https://arxiv.org/abs/2209.11895).

Both are defined now and reported as gaps against MHA with intervals for every variant. They are secondary: no detection claim is made on them. A late-context gap with a tied aggregate is still a reportable pattern.

### 10.4 Inference benchmark

- **Prefill:** latency, tok/s and peak VRAM at ctx 128 → 8,192.
- **Decode:** ms/token at batch 1, 8 and 32, cache 1k → 8k.
- **KV cache:** analytic and measured bytes/token (assert within 5%), maximum batch before OOM at 8k.

### 10.5 Plots, in priority order

1. **Primary gaps:** per primary variant, every per-seed value as a dot (d\_k under the paired test, both groups under Welch's), the 95% interval, the pre-registered detectable-effect band, and a second axis in effective-parameter multiplier. The scientific headline; design it first.
2. **Inference frontier:** 300M val loss (every seed) against decode tok/s. The plot practitioners will screenshot.
3. **Memory frontier:** final val loss against KV bytes/token.
4. **Noise floor:** every Arm N and null-control run as a dot.
5. **Ladder:** loss against ln(non-embedding params) with the fit, at 100M, 200M and 300M.
6. **Context gap:** loss gap vs. MHA per context bucket, per variant.
7. **Training curves:** val loss against tokens, faded trunks, branch results as dots.
8. **Latency:** prefill and decode vs. context, `ref` and `opt` overlaid.
9. **Synthetic probes:** accuracy vs. difficulty per variant, with the recurrent anchor.
10. **Retrieval:** accuracy vs. context length and distractor count.

### 10.6 Run manifest

Every run and every branch emits JSON: git SHA, config hash, resolved config including `recipe.yaml`, all seeds, dataset hash, tokenizer hash, torch/CUDA/driver versions, GPU name and compute capability, selected SDPA backend, GradScaler skipped steps and final scale, optimizer steps actually taken, whether the GPU was shared, session ID, resume count, step-time series, final metrics.

New in v4: parent trunk checkpoint and branch step for every branch; the LR-sweep run that chose the LR; the timing-pass ID behind any Arm C step count.

A run without a manifest doesn't exist. `make_tables.py` refuses to compare runs that disagree on backend or differ by more than 0.5% in steps taken. It also refuses runs that differ in torch, CUDA or driver version; pin the Kaggle environment so they cannot (§12.3).

## 11. Protocol discipline

These rules exist so that the results cannot be shaped after they are seen.

- **Pre-register before any variant run:** the §1.2 rule with its test-selection threshold, the three primary comparisons, the seed-count rule (§1.4), the positive-control test, the context-use metric definitions, and the detectable-effect bound once MHA's 300M runs report. Timestamped commit to the README.
- **Only `primary_test.py` says "detected".** No hand-run test, no different correction, no subset of seeds.
- **Five or eight seeds on primary comparisons, by the pre-registered rule; five on L6; three on everything else.** Plot individual runs, always.
- **No seeds added after any variant result is seen,** in any case, including the ambiguous middle (§13.10). The seed-count rule (§1.4) runs before that and reads MHA data only.
- **Fix data order; vary init seed.** The exception is Arm N, where varying data order is the point.
- **Results table and plots built, blank, before running.**
- **Noise floor before any cross-variant number** is computed or looked at.
- **LR sweep per variant** under the edge and tie rules (§7.4).
- **Check steps-taken parity before reading any loss comparison;** check backend parity before reading any timing comparison.
- **Report decay branches only,** never trunk checkpoints.
- **Never re-tokenize. Never tune GradScaler per variant. Never stitch a timing arm across sessions.**
- **Naming:** `{variant}-{arm}-{tokens}-lr{lr}-s{init}-d{data}[-b{branch}]`, for example `mla192-isoparam-300M-lr1e3-s2-d1-b100M`. Full config stored with every checkpoint.
- **Publication decided in advance,** independent of outcome (§14).
- **Frozen files change only with Phil's approval** (§17.3). The build loop launches no GPU run and reads no result.

## 12. Budget and Kaggle operations

v5 costs about 89 single-GPU hours against v4's 78, and runs in about 49 session-hours on T4 ×2 because all but 8.5 hours can run two at a time. The extra hours buy the 100M learning-rate sweeps, two more L6 seeds and the reproducibility pairs.

### 12.1 Estimate

Assumes \~80 min per 4,578-step (300M-token) run on one T4, the midpoint of v3's 60–100 min. Re-measure on the smoke runs and rescale.

| Item | Runs | GPU-h | Concurrent? |
| --- | --- | --- | --- |
| Smoke tests | \~10 × 5M | 0.5 | Yes |
| Phase 1 probes + recurrence anchor | \~250 short | 3.0 | Yes |
| LR sweep | \~33 × 100M | 14.7 | Yes |
| Arm N extras (2 data seeds, 5 null control, 3 reproducibility reruns) | 10 × 100M | 4.4 | Yes |
| Arm A primary | 4 variants × 5 seeds × 300M + 100M branch | 28.4 | Yes |
| Arm A exploratory | 3 variants × 3 seeds × 300M + 100M branch | 12.8 | Yes |
| Arm B | 3 variants × 3 seeds × 100M | 4.0 | Yes |
| Ladder (L6 × 5, L10 × 3, plus 200M branches on L8) | 8 × 300M + branches | 12.6 | Yes |
| Timing pass | 7 variants × 220 steps × 3 | 1.5 | **No** |
| Arm C (optional) | 3 variants × 3 seeds × 100M-equivalent | 4.0 | **No** |
| `bench.py` + retrieval | — | 3.0 | **No** |
| **Total** |  | **88.9** | \~49 session-h |

Two variations: without Arm C the total is about 85 GPU-h. If the seed-count rule picks 8, add 17.1 GPU-h for 12 more runs and remove 8.5 for MLA-96 and SWA-256, about 97.5 in all.

At \~30 quota-hours a week, the base case is about 1.6 weeks of quota if T4 ×2 draws at the single-GPU rate, and about 3 weeks if it draws double (§2.1). The schedule in §15 spreads it over four GPU weeks so that no week exceeds 30 GPU-h either way, which leaves slack for crashes.

### 12.2 Cut list, in order

1. Arm C (−4.0 h). RQ-W keeps its computed estimate (§8.5).
2. SWA-256 out of Arm A (−4.3 h). It stays in the timing pass and Phase 6, where it is informative.
3. MLA-96 out (−4.3 h).
4. Ladder down to L6 and L8 (−5.8 h). The positive control survives; the parameter conversion becomes a two-point slope with a wider interval.
5. Arm B to 1 seed (−2.7 h).

Cuts 2 and 3 are also what the seed-count rule removes when it picks 8 seeds.

**Never cut:** Arm N with the null control and the reproducibility pairs, the LR sweep, the seed count the rule picked for the primaries, the L6 positive control at 5 seeds. Without them nothing else is interpretable.

### 12.3 Kaggle operations

**The commit flow is not the interactive session.** Background execution re-runs the committed notebook from the top in a fresh container; no interactive state carries over.

- `train.py` and `branch.py` are real scripts in the repo. The notebook is a five-line shim: clone, mount the dataset, invoke with a config path.
- The notebook is idempotent from a cold start. Never develop inside it.

**Pin the environment.** MHA and its comparators run weeks apart. Pin the notebook to its original Kaggle environment so that an image update cannot change torch, CUDA or the driver between them. The manifest records all three and `make_tables.py` refuses mismatches.

**Quota:** idle interactive sessions burn quota, so stop them manually. Check the counter before opening a session. Sessions cap at 9–12 h and can die without warning; assume every run is interrupted.

**Checkpoints:**

- Write to `/kaggle/working` often enough to lose at most \~10 minutes.
- Resume automatically from the config hash, with no human decision.
- Keep the last two checkpoints; one written during a crash can be truncated.
- Keep every WSD branch-point checkpoint until its branch has finished and been verified.
- 20 GB persistent storage holds \~100 fp32 checkpoints at \~200 MB each. Prune finished runs.

**Notebooks are public by default on the free tier.** Fine for a project you are publishing, but be deliberate about it.

## 13. Confounds, traps and failure modes

Four categories: confounds to **disclose** (13.1–13.5), **silent corrupters** that produce clean, wrong results (13.6–13.9, 13.16), **project-level risks** (13.10–13.13, 13.18), and **statistical traps** (13.14–13.15, 13.17). Entries new in v5 are numbered after the old ones so that existing references stay valid.

### Confounds to disclose

**13.1 Parameter matching moves capacity into the MLP.** Iso-param MQA has a 22% wider MLP than MHA, so iso-param does not isolate attention. Arm B measures how much that matters; state the tension rather than picking a favourite.

**13.2 MLA bundles two changes:** low-rank KV compression and decoupled RoPE. They can't be fully separated; don't imply compression alone was measured. A partial ablation (MLA with standard RoPE on materialized keys) is deferred (decision 31); the build loop does not implement it.

**13.3 Kernel maturity is not architecture.** MHA, GQA and MQA use a mature fused kernel. SWA's block-skipping path and MLA's absorbed path are newer or hand-written. Any SWA or MLA timing gap is partly implementation; report `ref` and `opt` so readers can separate the two.

**13.4 Hardware specificity.** sm75, no FlashAttention-2, no fp8. Rankings may differ on Ampere and later; that is RQ-X and a follow-up, not a flaw to hide.

**13.5 Scale.** 17M params, ctx 1,024, 300M tokens, one corpus, one tokenizer. Nothing here extrapolates to 7B. First paragraph, not the conclusion.

### Silent corrupters

**13.6 GradScaler makes "iso-token" not iso-anything — highest priority.** `GradScaler` skips optimizer steps on inf/nan gradients, and variants overflow at different rates. Two variants can see identical tokens while taking different numbers of steps, and no curve looks wrong.

- Log skipped steps, final scale and steps actually taken for every run and branch.
- Compare steps taken before comparing losses; a difference above 0.5% (23 steps at 300M) invalidates the comparison.
- If any variant skips often, change the global setting until skips are rare for all, then rerun everything. Never tune per variant.
- A variant that can't be made stable in fp16 is a reportable finding. Don't quietly move it to fp32.

**13.7 Low-rank initialization variance.** See §3.3. Makes MLA look worse than it is by more than any effect being measured. The latent norms reduce but do not remove it.

**13.8 Fused but dense SWA (corrected from v3).** A masked SWA through a fused kernel passes the backend check, is numerically correct, and computes every score anyway. It makes SWA look no faster than MHA. Caught by the work-skipping test (§5).

**13.9 Resume, branch and trunk contamination.** A resume or branch that restores data position or schedule step wrongly produces a plausible curve that is not comparable. Reporting a trunk checkpoint as a result reports a falsely high loss. Caught by the resume and branch fidelity tests and the `-b{branch}` naming rule.

**13.16 Expanded K/V in GQA and MQA decode (new in v5).** Repeating KV heads before the kernel keeps the cache small and the reads full-size. Decode throughput then matches MHA, and the inference frontier shows no benefit where there is one. Caught by the grouped-decode test (§5).

### Project-level risks

**13.10 The noise floor may land in the ambiguous middle.** A floor near the size of the real gaps makes nothing conclusive, and seeds only help as √n. v4 answers this in advance: report not-detected with the confidence interval and the ruled-out bound, plus the effective-parameter interval. An interval saying "MQA costs between −1% and +8% of parameters" is useful even with no detection. Do not add seeds.

**13.11 SWA at ctx 1,024 is nearly uninformative.** A 256-token window drops little at this length; SWA's cost and benefit appear at far longer contexts. It may top Arm C and mean nothing. Caption it heavily, never let it headline, and see open decision 8 for a ctx 2,048 point.

**13.12 Synthetic-to-LM transfer is an assumption.** Probe results at small width say little about the 8-layer LM. Keep probe models as close to the backbone as the task allows (same depth, heads and norm) and shrink only what the task forces. Phase 1 can never cancel the LM arms (§9).

**13.13 Scope is the most likely failure.** Seven variants, a ladder, four arms, probes and long context is a lot for one person on a free tier. The most probable failure is stalling in week 3 with nothing published. The noise-floor post (§14.4) is a hard deliverable with a date.

**13.18 An agent that builds the analysis and sees the results (new in v5).** The harness is built by an autonomous loop. If the same loop also ran the experiments, it could adjust code, tolerances or analysis after seeing numbers, without intent and without a trace. The loop therefore stops at the handover, works on synthetic data only, and cannot change frozen files (§17).

### Statistical traps (new)

**13.14 Many comparisons, weak tests.** Uncorrected tests across 21 pairs find something by chance. v4 limits detection claims to three pre-registered primaries with Holm correction. Exploratory variants and per-position metrics are reported with intervals and no verdict. Simulated nulls in `test_primary_stats.py` guard the code.

**13.15 Selecting on noise.** A single-seed LR sweep picks whichever LR got a lucky seed. Choosing which results to emphasise after seeing them does the same thing at the writeup stage. The tie rule (§7.4) and the fixed plot order (§10.5) remove both choices.

**13.17 Pairing that does not pair (new in v5).** A paired test on runs that share no common-mode noise is still valid, but it spends half its degrees of freedom for nothing. Arm N's reproducibility pairs measure this before any variant is read, and the pre-registered switch picks Welch's test when pairing cannot help (§1.2).

## 14. Publication

Publish regardless of outcome, noise-floor post first. v4's statistics make a null result informative: "gaps above X nats, about Y% of parameters, are ruled out" is a finding, not an absence of one.

### 14.1 Publish regardless of outcome

Conditioning publication on promising results is the mechanism that fills the literature with noise. Committing in advance is what lets you trust your own numbers later.

### 14.2 Credibility levers (scale isn't available, so these are what you have)

- **Seed dots, not error bars.** Every run visible. A reader who sees MHA's spread can read any gap.
- **Positive and null control side by side.** The ladder shows the pipeline sees a 25% capacity cut; the null control shows it doesn't invent differences. Few small-scale ablations show both.
- **Gaps in parameter units.** "Like MHA with 0.94× the parameters" is readable by anyone; "0.012 nats" isn't.
- **Full protocol disclosure,** including every diverged LR and the pre-registration commit hash.
- **Scope in paragraph one.** Readers expect a small ablation dressed as an architecture verdict; disarm that immediately.
- **Repo with exact configs and one-command repro.** A clean, tested, swappable-attention testbed is something people fork.

### 14.3 The null result is the better post

"90 GPU-hours, the mechanism didn't matter at 17M params, here is the floor that swamped it, and here is the smallest gap we could have seen" tells every small-scale experimenter how much compute their ablation needs. Almost nobody publishes this.

### 14.4 Staging

Publish the **noise-floor post** the moment Arm N exists: σ\_init, σ\_data, the null control, and the detectable-effect table. Add the ladder's positive control if it is ready, otherwise as an update. It is independently useful, and feedback can still change the protocol.

### 14.5 Post structure

1. The question
2. Why existing comparisons don't answer it: proponent authorship; tokenizer, data, LR and context confounds across papers
3. The noise floor, the null control and the positive control
4. Synthetic probes, and why standard MQAR doesn't separate attention variants
5. LM results as paired gaps and parameter equivalents
6. The inference frontier
7. Training wall-clock (SWA, MLA)
8. What doesn't transfer

Lead with the negative space.

### 14.6 Prior work to read before writing

- [DeepSeek-V2](https://arxiv.org/abs/2405.04434), Appendix D: MHA/GQA/MQA at 7B, layer count adjusted for parameter parity. The comparison being reproduced at \~1/400th the scale. Also the source of MLA and its latent norms.
- Tensor Product Attention: the same protocol at 353M–1.5B on FineWeb-Edu.
- [Zoology](https://arxiv.org/abs/2312.04927) (Arora et al., 2023): introduces MQAR; shows it separates attention from gated-convolution and recurrent models. Basis for §9's Gate 1 change.
- [MAD](https://arxiv.org/abs/2403.17844): synthetic unit tests correlated with compute-optimal perplexity across 500+ models. Methodological backbone for Phase 1.
- [Hägele et al., 2024](https://arxiv.org/abs/2405.18392): WSD with cooldown matches cosine; basis for §7.2.
- [Olsson et al., 2022](https://arxiv.org/abs/2209.11895): induction heads and the in-context-learning score used in §10.3.
- Speed Always Wins (arXiv 2508.09834): survey for situating MQA/GQA/MLA and sparse attention.

The Zoology and Hägele entries were checked against their arXiv listings on 2026-10-20. Check every citation against the paper itself before publishing; don't take them from a planning document.

## 15. Schedule

Three stages. An autonomous loop builds and tests the harness and stops (§17). Phil then verifies it on real hardware and commits the pre-registration. After that the weekly Kaggle quota sets the pace: four GPU weeks, none above 30 GPU-h.

**Build stage — autonomous loop, CPU only, 0 GPU-h**

Tasks, order and acceptance tests are in `TASKS.md`. The stage covers:

- [ ] Backbone: RMSNorm, SwiGLU, RoPE, tied embeddings, causal masking
- [ ] `grouped.py`: MHA/GQA/MQA in one file with an `n_kv_heads` knob and grouped-query decode
- [ ] `mla.py`: naive path, absorbed path, latent norms, factor init
- [ ] `sliding_window.py`: `ref` path, chunked path, FlexAttention path
- [ ] Per-parameter seeded init (§4.5) and the step-0 logit-scale test across all variants
- [ ] fp16 + GradScaler trainer with skip-count logging, checkpointing, resume
- [ ] WSD schedule, `branch.py`, branch-fidelity test (§7.2)
- [ ] `solve_config.py` + `test_params.py` for all variants and ladder sizes
- [ ] Data pipeline, tested on synthetic text; the real corpus is tokenized by Phil
- [ ] Evaluation: LM loss, context-bucketed loss, probes and the recurrence anchor, retrieval, `bench.py`, `timing_pass.py`
- [ ] `primary_test.py`, `ladder_fit.py`, `rqw_estimate.py` and `test_primary_stats.py`, run against simulated data
- [ ] `make_tables.py` and every plot, generated blank and from fake manifests
- [ ] Kaggle shim notebook
- [ ] Correctness suite (§5): `cpu` tier green in CI; `gpu` and `t4` tiers written
- [ ] `RUNBOOK.md` and the handover report (§17.4)

**Verification — Phil, the 1070 and one short T4 session**

- [ ] `pytest -m gpu` on the 1070
- [ ] `pytest -m t4` on Kaggle: FlexAttention on sm75, backends, SWA and GQA/MQA timing assertions
- [ ] Check how T4 ×2 sessions draw quota (§2.1)
- [ ] Cold-start dry run of the shim on 5M tokens
- [ ] Pin the Kaggle environment (§12.3)
- [ ] Confirm the two thresholds in `configs/prereg.yaml`
- [ ] Pre-registration committed to the README with a timestamp: §1.2 rule, primaries, seed-count rule, positive-control test, context-use metrics

**GPU week 1 — data, recipe, noise floor (≈25 GPU-h)**

- [ ] Tokenizer trained; corpus tokenized once; versioned Kaggle Dataset; hash recorded
- [ ] Smoke run per variant, all seven, on 5M tokens: backend, scaler-skip parity, `init_scale` and micro-batch chosen
- [ ] Freeze `recipe.yaml`
- [ ] Phase 1 probes → **Gate 1**
- [ ] MHA LR sweep at 100M, 5 points plus the edge rule
- [ ] Arm N: MHA seeds 1–5 to their 100M branches, data seeds 2–3, null control, reproducibility pairs → **Gate 2**
- [ ] Commit the test choice (§1.2)
- [ ] LR sweeps: remaining variants and both ladder sizes
- [ ] Draft the noise-floor post; commit a publication date

**GPU week 2 — ladder and positive control (≈19 GPU-h)**

- [ ] Timing pass, dedicated GPU (§8.5)
- [ ] Ladder: L6 at 5 seeds, L10 at 3
- [ ] MHA seeds 1–5 continue to 300M
- [ ] Positive-control test (§8.4)
- [ ] Commit the detectable-effect bound and the seed count (§1.3, §1.4)
- [ ] **Publish the noise-floor post**

**GPU week 3 — primaries (≈21 GPU-h)**

- [ ] Arm A primaries: GQA-4, MQA, MLA-192, and MHA seeds 6–8 if the rule picked 8 → **Gate 3**

**GPU week 4 — everything else (≈24 GPU-h)**

- [ ] Arm A exploratory
- [ ] Arm B
- [ ] `bench.py` runs for the inference frontier
- [ ] RQ-W estimate; Arm C if kept: one uninterrupted session, dedicated GPU, round-robin
- [ ] Generate all tables and plots → **Gate 4**

If the seed-count rule picks 8, week 3 grows to about 38 GPU-h and spills into week 4, where the dropped exploratory runs free the room.

## 16. Decision log

Append rows with the date; never rewrite a past row. Rows 1–9 come from v3; 10–17 were added in v4 on 2026-10-20. Rows 18–34 were added in v5 on 2026-11-12; where one closes an earlier row it says so, and the earlier row stays as written.

| # | Decision | Status | Resolution or leaning | Logged | Resolve by |
| --- | --- | --- | --- | --- | --- |
| 1 | DSA in v1? | Frozen | No; follow-up post on modern hardware | v3 | — |
| 2 | Non-attention anchor (gated linear recurrence)? | Decided | Yes: it is the positive control that proves Phase 1 probes discriminate | v4 | — |
| 3 | Headline framing: iso-param or iso-backbone | Leaning | Report both; lead with iso-param, commensurable with DeepSeek | v3 | End of week 4 |
| 4 | MLA-with-standard-RoPE ablation (§13.2) | Open | Only with ≥5 spare GPU-h | v3 | End of week 3 |
| 5 | RoPE extension method for Phase 6 | Open | Undecided | v3 | Gate 4 |
| 6 | One long post or a series | Leaning | Series, noise floor first | v3 | After the floor post's reception |
| 7 | GradScaler `init_scale` and `growth_interval` | Leaning | Set once, globally, from smoke runs (§7.3) | v3 | End of week 2 |
| 8 | SWA at ctx 2,048 as a separate point (§13.11) | Open | Yes with ≥3 spare GPU-h, otherwise caption heavily | v3 | Gate 3 |
| 9 | Response if the floor lands in the ambiguous middle | Decided | Not detected, with interval and ruled-out bound; no added seeds (§13.10) | v4 | — |
| 10 | Detection rule | Decided | Paired t-test, Holm over 3 primaries, family-wise α = 0.05 (§1.2) | v4 | Freeze at pre-registration |
| 11 | Seeds per comparison | Decided | 5 for primaries, 3 elsewhere | v4 | — |
| 12 | LR schedule | Decided | WSD, 20% decay, branches at 100M (and 200M on the ladder) | v4 | — |
| 13 | WSD decay shape | Leaning | 1−√; linear is the alternative | v4 | Freeze with `recipe.yaml`, week 2 |
| 14 | Ladder axis | Decided | Depth (6/8/10 layers): attention per layer stays identical | v4 | — |
| 15 | Arm C scope | Decided | MHA, SWA-256, MLA-192; others rejoin only if the timing pass shows >±3% | v4 | After the timing pass |
| 16 | Micro-batch size | Open | Largest that fits MLA's naive path | v4 | Week 2 smoke runs |
| 17 | Phase 1 probe set | Open | Multi-hop recall, high-cardinality MQAR at small head dim, fuzzy keys (§9) | v4 | End of week 1 |
| 18 | Recipe freeze order | Decided | Freeze only after all 7 variants pass smoke runs (§7) | v5 | — |
| 19 | LR sweep budget | Decided | 100M for every sweep (§7.4) | v5 | — |
| 20 | Paired or unpaired test | Proposed | Paired only if s\_rep < 0.6 × σ\_init, otherwise Welch (§1.2); Phil confirms the threshold | v5 | Pre-registration |
| 21 | Seed-count rule | Proposed | 8 seeds if the 5-seed detectable gap exceeds 5% effective parameters, otherwise 5 (§1.4); Phil confirms the threshold | v5 | Pre-registration |
| 22 | Positive-control seeds and budget | Decided | L6 at 5 seeds, tested at the 300M result (§8.4) | v5 | — |
| 23 | Arm C | Leaning | Computed estimate first; the runs are optional and first on the cut list (§8.5) | v5 | After the timing pass |
| 24 | Gate 1 criterion | Decided | Anchor only; SWA's behaviour is reported, not gating (§9.2) | v5 | — |
| 25 | Context-use metric axis | Decided | Same-document context inside the window, not absolute position (§6, §10.3) | v5 | — |
| 26 | Environment pinning | Decided | Pin the Kaggle environment; a version mismatch blocks comparison (§12.3) | v5 | — |
| 27 | Null-control seeds | Decided | 11–15, leaving 6–8 free for the seed-count rule (§8.1) | v5 | — |
| 28 | WSD decay shape (closes row 13) | Decided | 1−√ | v5 | — |
| 29 | Micro-batch size (closes row 16) | Decided | The rule stands: largest that fits MLA's naive path in `ref` mode on a T4. The value is measured in the smoke runs and written to `recipe.yaml` before the freeze. | v5 | — |
| 30 | Phase 1 probe set (closes row 17) | Decided | The five probes in §9.2 | v5 | — |
| 31 | MLA-with-standard-RoPE ablation (updates row 4) | Deferred | Not built by the loop; reopen after Gate 3 if GPU-h remain | v5 | Gate 3 |
| 32 | Base init rule and MLA factor init | Decided | Normal, std 0.02, residual outputs scaled by 1/√(2 × layers); MLA up-projections at 0.02 × √(d / d\_c) (§3.1, §3.3) | v5 | — |
| 33 | Recurrence anchor architecture | Decided | Per-channel gated linear recurrence (§9.2) | v5 | — |
| 34 | Build mode | Decided | An autonomous loop builds and tests the harness on CPU and stops at the handover; Phil runs every GPU job from `RUNBOOK.md` (§17) | v5 | — |

**Frozen since v3 (Doc A / Doc B reconciliation):** the 1070 is for dev and tests only, with training on Kaggle T4; 8k BPE tokenizer at ctx 1,024; 17.3M total / 14.2M non-embedding params; TinyStories dropped; the noise floor gates every comparison; fp16 + GradScaler, because T4 lacks bf16.

## 17. Build loop and handover

The harness is built by an autonomous loop: one orchestrating agent that holds this plan, and a team of smaller worker agents that implement single tasks. `CLAUDE.md` holds the operating rules and `TASKS.md` the task graph. This section fixes what the loop may and may not do.

### 17.1 Scope

The loop builds everything in the build stage of §15 and stops at the handover. It launches no GPU job, uses no Kaggle credentials, downloads no corpus and reads no experimental result. Its pipeline tests run on synthetic text, simulated losses and fake manifests only. Phil runs every GPU job from `RUNBOOK.md`.

### 17.2 Roles

- **Orchestrator:** owns the plan, assigns tasks, reviews every diff, merges, keeps `STATUS.md` current. For the tasks where silent bugs hide, it writes the tests itself before any implementation exists: MLA, per-parameter seeding, the trainer's GradScaler handling, resume and branch fidelity, and all statistics code.
- **Workers:** implement one task each against its acceptance test, on their own branch. On the tasks whose tests the orchestrator wrote, a worker may not change the tests.
- **Phil:** answers questions, approves changes to frozen files, runs the `gpu` and `t4` tiers, signs every gate.

### 17.3 Frozen files and human gates

These change only through a pull request that Phil approves:

- `docs/plan_v5.md`
- `configs/base.yaml`, `configs/recipe.yaml`, `configs/prereg.yaml`
- `tests/tolerances.py`
- The pre-registration section of `README.md`

The loop creates each of them once from this plan. `recipe.yaml` carries the §7 values, with `init_scale`, `growth_interval` and the micro-batch left unset for the smoke runs. Loosening a tolerance, skipping a test or marking it as an expected failure in order to pass is forbidden. Gates 1–4, the recipe freeze and the pre-registration commit are Phil's.

### 17.4 Handover

The loop is done when all of the following hold:

1. Every task in `TASKS.md` is merged and the `cpu` tier is green in CI.
2. Every `gpu` and `t4` test is written, skips cleanly without hardware, and is listed in the handover report as not yet run.
3. The headline table and all ten plots generate from fake manifests, and generate blank.
4. `RUNBOOK.md` gives the exact commands for the verification step and for each GPU week of §15, in order, with the gate that follows each block and what to check there.
5. `STATUS.md` lists every decision the loop made (mirrored in `docs/decisions.md`), every open question, and every deviation from this plan.
6. A fresh agent that did not write the code has reviewed the harness against §5 and §13 and its findings are resolved or listed.
