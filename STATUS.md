# STATUS

Last updated: 2026-11-12, session 1.

## Done

- Bootstrap commit: `CLAUDE.md`, `TASKS.md`, `docs/plan_v5.md`.
- T-00 scaffold (#1).
- T-01 config system (#2).
- T-02 backbone and attention registry (#3).
- T-13 data pipeline (#4).
- T-24 primary test and its statistics tests (#5).

## In progress

- Nothing.

## Blocked

- Nothing.

## Open questions

- **Welch multipliers in plan §1.3 (blocks T-25).** The table's Welch rows, 2.5 at 5 seeds and 1.8 at 8, are the values for a pooled-variance t-test. Welch's test as specified in §1.2 needs 2.63 and 1.83 for 80% power; at the plan's 2.5 it has 76%. Both calculations are in `primary_test.py`. Which one feeds the seed-count rule (§1.4) and the reported detectable-effect bound? Recommendation: the exact one.
- **Seed-count rule under the paired test (blocks T-25).** §1.4 step 1 wants the 5-seed detectable gap in nats from sigma_init. For the paired test the multiplier is in units of sigma_d, which is unknown before any variant is read. Use the table's worst case, 3.2 × sigma_init? Recommendation: yes; it is the only value computable from MHA data alone.
- Which PyTorch, Python and CUDA versions does Kaggle's GPU image ship? Needed to pin the environment. Until then: torch 2.15.0, Python 3.12 in CI.

## Readings of the plan for Phil to confirm

These do not block the build. Each is implemented as described and can be changed in one place.

- **Data order (decision 28).** The data seed gives a shuffled order of the 1,024-token windows, also for data seed 1. A run reads a prefix of that order. Plan §6 says "from token 0 in the same order"; §4.5 gives the order its own seeded generator.
- **EOS ownership (decision 31).** The EOS after a document counts as part of that document, so the first token of the next document has context c = 0.
- **In-context-learning windows (decision 20).** End points included: 450 ≤ c ≤ 550 and 40 ≤ c ≤ 60.
- **`configs/data.yaml` (decision 38).** Not frozen. Consider adding it to CODEOWNERS once the hashes are filled in.

## Deviations from the plan

- fp32 `ref`/`opt` agreement tolerance is 1e-5, not 1e-6 (plan §5). Approved by Phil on 2026-11-12. Reason: 1e-6 is at the fp32 rounding floor; correct implementations measured up to 1.8e-6.
- The agreement tests also run in float64 at 1e-12, an addition to "CPU tests use fp32". Approved by Phil on 2026-11-12.

## Decisions

All decisions are in `docs/decisions.md` (rows 1–48 so far).

## Environment

- PyTorch 2.15.0. CI uses the CPU build on Python 3.12.
- The build container cannot reach download.pytorch.org, so it runs the default PyPI build of 2.15.0 on CPU (Python 3.13).

## For Phil

- Confirm or change the two proposed thresholds in `configs/prereg.yaml` before the pre-registration commit: `paired_if_s_rep_ratio_below: 0.6` and `max_undetectable_param_fraction: 0.05`.
- `recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset until the smoke runs.
- After tokenizing the corpus with `python -m scripts.prepare_data`, paste the two hashes it prints into `configs/data.yaml`.

- Protect `main`: pull requests only, the `cpu` check required, and "Require review from Code Owners" enabled, otherwise CODEOWNERS has no effect.
- Create the labels `needs-phil` and `question`.

## Next

1. T-03 seeded init (O+W) and T-04 grouped attention (O+W): tests first.
2. T-14 schedule (O+W), T-12 recurrence anchor (W).
3. T-07 sliding window (O+W).
