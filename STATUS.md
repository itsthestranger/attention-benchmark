# STATUS

Last updated: 2026-11-12, session 2. 21 of 32 tasks merged; `cpu` tier green on `main` (2,284 tests, about two minutes).

## Done

- Bootstrap commit: `CLAUDE.md`, `TASKS.md`, `docs/plan_v5.md`.
- T-00 scaffold (#1).
- T-01 config system (#2).
- T-02 backbone and attention registry (#3).
- T-13 data pipeline (#4).
- T-24 primary test and its statistics tests (#5).
- T-03 per-parameter seeded init (#6).
- T-14 WSD schedule and optimizer parameter groups (#7).
- T-12 recurrence anchor (#8).
- T-04 grouped attention: MHA, GQA, MQA (#9).
- T-15 trainer (#10).
- T-07 sliding-window attention (#11).
- T-05 MLA naive path (#13).
- T-25 ladder fit, effective-parameter multiplier and seed-count rule (#14).
- T-20 evaluation with context-bucketed loss (#15).
- T-19 determinism test and reproducibility measurement (#16).
- T-18 manifest and run naming (#17).
- T-08 attention properties and FLOP counts (#18).
- T-16 checkpoint and resume (#19).
- T-10 step-0 logit-scale test (#21).
- T-09 config solver and exact parameter counts (#22).
- T-26 computed wall-clock estimate (#23).

## In progress

- **T-27 headline table**: implemented by a worker in `/home/user/wt/T-27`, reviewed (three tests added, one rule changed), committed locally on `task/T-27-tables`, not pushed. **Not yet reviewed, not pushed.** T-27's worker raised two points to settle in review: without the ladder there is no seed count and so no primary verdict (the brief contradicted itself; the code follows the plan's order), and it changed the synthetic campaign's default noise seed after seeing an outlier draw.

## Blocked

- Nothing.

## Open questions

- Which PyTorch, Python and CUDA versions does Kaggle's GPU image ship? Needed to pin the environment. Until then: torch 2.15.0, Python 3.12 in CI.

## Readings of the plan for Phil to confirm

These do not block the build. Each is implemented as described and can be changed in one place.

- **Data order (decision 28).** The data seed gives a shuffled order of the 1,024-token windows, also for data seed 1. A run reads a prefix of that order. Plan §6 says "from token 0 in the same order"; §4.5 gives the order its own seeded generator.
- **EOS ownership (decision 31).** The EOS after a document counts as part of that document, so the first token of the next document has context c = 0.
- **In-context-learning windows (decision 20).** End points included: 450 ≤ c ≤ 550 and 40 ≤ c ≤ 60.
- **Top context bucket (decision 95).** c = 1,024 counts in the last bucket, so that bucket is c ≥ 512 and not the half-open [512, 1024) of plan §10.3. Otherwise the bucket-weighted loss cannot equal the aggregate.
- **Multiplier interval (decisions 87, 88, 92).** The interval of the effective-parameter multiplier is the set of gap / slope ratios the data do not reject, with the noise the 8-layer MHA runs contribute to both counted once. It reports no interval when the ladder slope cannot be told from zero. No simulated coverage test is committed, because `tests/tolerances.py` has no allowance for one; say if you want one and with what allowance.
- **Reproducibility in `opt` mode (decision 99).** Plan §5 asks that two identical runs "land within a measured, recorded tolerance". No tolerance exists before the measurement, so the `gpu` and `t4` tests measure and record the difference and apply no threshold.
- **Step-0 logit scale (decision 118).** "Attention-logit std" is pooled over all layers, heads and causal pairs of the full-size model, at two init seeds. The seven variants land between 0.965 and 1.019 of MHA. A single deep layer moves by up to about 10% between seeds of one variant, so the test does not compare layer by layer.
- **Wall-clock estimate below 100M (decision 123).** A variant slower than MHA affords fewer than 1,526 steps. Its loss is then read off the same line in ln(tokens), extended below the 100M branch, and flagged as extrapolated.
- **`configs/data.yaml` (decision 38).** Not frozen. Consider adding it to CODEOWNERS once the hashes are filled in.

## Deviations from the plan

- fp32 `ref`/`opt` agreement tolerance is 1e-5, not 1e-6 (plan §5). Approved by Phil on 2026-11-12. Reason: 1e-6 is at the fp32 rounding floor; correct implementations measured up to 1.8e-6.
- The agreement tests also run in float64 at 1e-12, an addition to "CPU tests use fp32". Approved by Phil on 2026-11-12.
- The seed-count rule and the reported detectable-effect bound use the exact power of Welch's test, 2.63 × σ_init at 5 seeds and 1.83 at 8, not the 2.5 and 1.8 of the plan's §1.3 table (decision 79). Decided by Phil on 2026-11-12. Reason: the table's Welch rows are pooled-variance values, at which Welch's test has 76% power instead of 80%. The plan's table itself is unchanged; correcting it needs a pull request to the frozen plan.

## Decisions

All decisions are in `docs/decisions.md` (rows 1–124 so far).

Answered by Phil in session 2: exact Welch power for the seed-count rule and the bound (decision 79); the √2 × σ_init worst case for the paired test in the seed-count rule (decision 80).

## Hardware tests written, never run

These skip without the hardware. A skip is not a pass. List them with `pytest --collect-only -q -m "gpu or t4"`.

- `gpu`, trainer (`test_trainer.py`): fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusal without scaler settings, the backend restriction. **The fp16, GradScaler and backend code in the trainer has never executed.**
- `gpu`: causal leak under fp16 (`test_causality.py`), `ref`/`opt` agreement under fp16 (`test_reference.py`), `opt` decode peak memory at cache 8k below MHA's (`test_grouped_decode.py`), the tier canary (`test_scaffold.py`).
- `gpu`, determinism (`test_determinism.py`): `ref` mode in fp16 bit-identical over 100 steps twice, for all seven variants. Needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment before pytest starts.
- `gpu` and `t4`, reproducibility (`test_determinism.py`): the `opt` measurement in fp16 for all seven variants; it records differences and applies no threshold.
- `gpu`, checkpoint (`test_checkpoint.py`): GradScaler state after a resume, resume fidelity in fp16 for all seven variants, a checkpoint saved and restored on CUDA.
- `gpu`, evaluation (`test_eval_lm.py`): fp16 evaluation on CUDA against fp32.
- `gpu`, sliding window (`test_sliding_window.py`): both `opt` paths against `ref` in fp32 and fp16, including compiled FlexAttention.
- `t4`: `opt` decode time at cache 8k below MHA's (`test_grouped_decode.py`), the tier canary (`test_scaffold.py`).
- `t4`, sliding window (`test_sliding_window.py`): SWA attention time at ctx 1,024 below MHA's, for the chunked and the FlexAttention path.

## To check on real hardware

- Whether FlexAttention compiles and runs on the T4 (plan §2.3), and which SWA path is faster there.
- Whether the fused kernel copies K and V when given a slice of the cache storage (plan §2.3). If it does, decode needs an explicit matmul, disclosed.

## Environment

- PyTorch 2.15.0. CI uses the CPU build on Python 3.12.
- The build container cannot reach download.pytorch.org, so it runs the default PyPI build of 2.15.0 on CPU (Python 3.13).

## For Phil

- Your two answers (decisions 79, 80) are constants in `analysis/ladder_fit.py`. If you want them in `configs/prereg.yaml` instead, say so: it is a frozen file and needs your approval on the pull request.
- Confirm or change the two proposed thresholds in `configs/prereg.yaml` before the pre-registration commit: `paired_if_s_rep_ratio_below: 0.6` and `max_undetectable_param_fraction: 0.05`.
- `recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset until the smoke runs.
- After tokenizing the corpus with `python -m scripts.prepare_data`, paste the two hashes it prints into `configs/data.yaml`.

- Protect `main`: pull requests only, the `cpu` check required, and "Require review from Code Owners" enabled, otherwise CODEOWNERS has no effect.
- Create the labels `needs-phil` and `question`.

## Next

1. Merge T-27 (see "In progress").
2. T-17 branch (O+W): write `tests/test_wsd_branch.py` first. T-06 MLA absorbed path (O+W).
3. T-11 graph diff (W, after T-09), T-28 plots (W, after T-27), T-21 probes (W), T-22 retrieval generators (W), T-23 benchmark scripts (W, after T-06).

## Notes for the next session

How the loop has been run:

- Workers are Haiku subagents in git worktrees under `/home/user/wt/`, one task each. They commit locally; the orchestrator reviews, pushes and opens the pull request, then squash-merges on green CI.
- Review includes breaking the source on purpose, one change at a time, and checking that a test fails. Every miss so far was a gap in a test, fixed by adding a test.
- Run one pytest process at a time. The container has 4 cores, and two suites at once made runs ten times slower.
- A branch stacked on an unmerged branch conflicts after the squash-merge; resolve with `git checkout --ours` on the files the stacked branch changed.
- The container needs `pip install torch numpy scipy pyyaml tokenizers pyarrow pytest` from PyPI; download.pytorch.org is blocked.

Designs already worked out for tasks not yet started:

- **T-06 absorbed MLA.** Replaces `opt` decode in `mla.py`; the cache already holds only the latent and the positional key.
- **T-17 branch.** A branch restores a `trunk-step*.pt` checkpoint (`src/training/checkpoint.py`) into a trainer whose config has the branch's step count, then decays. `restore` refuses another config hash, so `branch.py` needs its own entry that checks the parent's hash and the branch step (`decay_start` of the branch budget) instead. Its manifest is `kind: branch` with the parent's resolved config (decision 102).
- **T-29 entry points.** Give `ref` and `opt` their own checkpoint roots (decision 115). Record the trainer's enforced backend on the attention modules with `set_sdpa_backend` and write it to the manifest (decision 110).
- **T-23 benchmark.** Compare sliding-window memory with its cap, not with per-token growth (decision 77).
