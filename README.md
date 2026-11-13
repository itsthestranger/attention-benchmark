# Attention mechanisms at small scale

A comparison of attention mechanisms (MHA, GQA, MQA, MLA, sliding window) in a 17M-parameter language model trained on up to 300M tokens. The first question is whether differences between mechanisms can be told apart from seed noise at this scale at all. Everything else depends on that answer.

**Scope.** 17M parameters, context 1,024, 300M tokens, one corpus, one tokenizer, Kaggle T4 GPUs. Nothing here says which mechanism is best at production scale.

**State.** The harness is built and tested on CPU. No experiment has run. `STATUS.md` says what is done, what has never executed on a GPU, and which decisions are open.

| Document | What it is |
| --- | --- |
| `docs/plan_v5.md` | The specification. Frozen. |
| `RUNBOOK.md` | Every GPU job in order, with the gate after each block and what to check there. |
| `STATUS.md` | Current state, open questions, deviations from the plan. |
| `docs/decisions.md` | Every implementation decision the build made, with its reason. Append only. |
| `TASKS.md`, `CLAUDE.md` | The build's task list and operating rules. |

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not gpu and not t4"      # the cpu tier, run by CI on every pull request
pytest -m gpu                       # needs any CUDA card
pytest -m t4                        # needs the Kaggle T4
```

A `gpu` or `t4` test that is skipped has not passed.

## Generated outputs

Nothing in a table or a plot is typed by hand.

```bash
python -m scripts.make_tables --blank --out results/tables/blank   # the committed blank tables
python -m scripts.make_plots --blank --out results/plots           # the ten plots, blank
```

## Pre-registration

This section is frozen once committed (plan §17.3). It is committed with a timestamp before any variant run (plan §11). Until then it is a template: every line marked **TO FILL** is empty on purpose, and nothing below has been decided by a result.

- Pre-registration commit: **TO FILL** (commit hash and date)
- Thresholds in `configs/prereg.yaml`, confirmed by Phil on 2026-11-13: `paired_if_s_rep_ratio_below` 0.6, `max_undetectable_param_fraction` 0.05
- Amendments to the plan made before the first GPU run: `docs/plan_v5.md` §18

### The rule (plan §1.2)

> **Primary comparisons.** Exactly three, each against MHA: GQA-4, MQA, MLA-192. Each uses n init seeds with the data seed fixed. n is 5 or 8, set by the seed-count rule before any variant result at 300M is read.
>
> **Test selection.** Decided from MHA runs only, after Arm N. Let σ_init be the SD of MHA seeds 1–5 at 100M. Let s_rep = √(mean(δ²) / 2), where δ is the loss difference within each of the three reproducibility pairs. If s_rep < 0.6 × σ_init, use the paired test. Otherwise use Welch's test.
>
> **Paired test.** For each seed k, compute d_k = loss(variant, seed k) − loss(MHA, seed k) on the final validation loss at 300M tokens. Run a two-sided paired t-test on the n values of d_k.
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

The machine-readable form of this rule is `configs/prereg.yaml`. The only code that emits a verdict is `analysis/primary_test.py`.

### Seed-count rule (plan §1.4)

Applied once, after Arm N and the ladder, from MHA runs only:

1. Take the 5-seed detectable gap for the chosen test, in nats, using σ_init at 300M.
2. Convert it with the ladder slope b at 300M: m = exp(−gap / b).
3. If 1 − m > 0.05, run 8 seeds on MHA and the three primaries and drop MLA-96 and SWA-256 from Arm A. Otherwise run 5.
4. If the slope b is not positive, run 8 seeds as in step 3 and report no effective-parameter multiplier (plan §18, A4).

The detectable gap uses the exact power of the test that is run: 2.63 × σ_init for Welch's test at 5 seeds and 1.83 at 8, not the 2.5 and 1.8 of the plan's §1.3 table. If the paired test is chosen, step 1 uses the worst case σ_d = √2 × σ_init (plan §18, A3).

The rule reads MHA seeds 1–5, L6 seeds 1–5 and L10 seeds 1–3 at 300M and nothing else. It gives no seed count while any of them is missing.

**Declared in `configs/commitments.yaml`, by pull request, each before the results it governs are read** (plan §18, A5):

- `ladder`: `full`, or `l6_l8` if L10 is cut (plan §12.2 item 4). The rule then uses the two-point slope.
- `seed_count`: the number the rule gives, committed before the first primary run of week 3. The analysis tests no primary and no exploratory comparison while it is missing or differs from what the rule computes.

### Controls

- **Null control** (plan §8.1): MHA under the name `null`, init seeds 11–15, at 100M, tested against MHA's 100M branches exactly like a variant. A detected null control means the pipeline is broken; nothing downstream is valid until it passes.
- **Positive control** (plan §8.4): L6 against L8, a single two-sided test on seeds 1–5 at the 300M result, α = 0.05, with the test chosen above. It counts as detected only when L6 is worse than L8. If it is not detected, or detected in the wrong direction, the control has failed and every not-detected verdict is reported as uninformative (plan §18, A6).

### Learning rates (plan §7.4)

- Every sweep runs on init seed 21, which no arm uses (`lr_sweep.init_seed` in `configs/prereg.yaml`).
- Arm A runs each variant at one rate: its own if the tie rule lets it differ from MHA's, otherwise MHA's.
- For each Arm A variant whose rate differs from MHA's, three more runs are made: seeds 1–3, 100M, at MHA's rate. The variant's 100M branches at its own rate minus these runs are reported by seed with an interval. No detection claim is made on that difference (plan §18, A7).

### Context-use metrics (plan §10.3)

Secondary; no detection claim is made on them.

- **Axis:** c, the number of tokens of the same document that precede a token inside the window.
- **Buckets:** c in [0, 64), [64, 256), [256, 512), and 512 and above.
- **Late-context loss:** mean loss at c ≥ 512.
- **In-context-learning score:** mean loss at c from 450 to 550 minus mean loss at c from 40 to 60, end points included.

### Committed later, each before the results it governs are read

- Test chosen at Gate 2 (paired or Welch), with σ_init, σ_data and s_rep at 100M: **TO FILL**
- Detectable-effect bound in nats and the seed count, when MHA's 300M runs and the ladder exist, before any variant result at 300M is read: **TO FILL** (the seed count also goes into `configs/commitments.yaml`)
- Publication date of the noise-floor post: **TO FILL**
