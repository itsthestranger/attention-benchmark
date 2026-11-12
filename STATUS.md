# STATUS

Last updated: 2026-11-12, session 2. 13 of 32 tasks merged; `cpu` tier green on `main` (1,084 tests).

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

## In progress

- **T-18 manifest**, **T-19 determinism**, **T-20 evaluation**: each implemented by a worker in its worktree under `/home/user/wt/`, committed locally on `task/T-18-manifest`, `task/T-19-determinism`, `task/T-20-eval`. **Not yet reviewed, not pushed.** T-19 was built before MLA merged: rerun it on current `main`, where the determinism test also covers the two MLA variants.

## Blocked

- Nothing.

## Open questions

- Which PyTorch, Python and CUDA versions does Kaggle's GPU image ship? Needed to pin the environment. Until then: torch 2.15.0, Python 3.12 in CI.

## Readings of the plan for Phil to confirm

These do not block the build. Each is implemented as described and can be changed in one place.

- **Data order (decision 28).** The data seed gives a shuffled order of the 1,024-token windows, also for data seed 1. A run reads a prefix of that order. Plan §6 says "from token 0 in the same order"; §4.5 gives the order its own seeded generator.
- **EOS ownership (decision 31).** The EOS after a document counts as part of that document, so the first token of the next document has context c = 0.
- **In-context-learning windows (decision 20).** End points included: 450 ≤ c ≤ 550 and 40 ≤ c ≤ 60.
- **Multiplier interval (decisions 87, 88, 92).** The interval of the effective-parameter multiplier is the set of gap / slope ratios the data do not reject, with the noise the 8-layer MHA runs contribute to both counted once. It reports no interval when the ladder slope cannot be told from zero. No simulated coverage test is committed, because `tests/tolerances.py` has no allowance for one; say if you want one and with what allowance.
- **Reproducibility in `opt` mode (T-19, not yet merged).** Plan §5 asks that two identical runs "land within a measured, recorded tolerance". No tolerance exists before the measurement, so the `gpu` and `t4` tests measure and record the difference and apply no threshold.
- **`configs/data.yaml` (decision 38).** Not frozen. Consider adding it to CODEOWNERS once the hashes are filled in.

## Deviations from the plan

- fp32 `ref`/`opt` agreement tolerance is 1e-5, not 1e-6 (plan §5). Approved by Phil on 2026-11-12. Reason: 1e-6 is at the fp32 rounding floor; correct implementations measured up to 1.8e-6.
- The agreement tests also run in float64 at 1e-12, an addition to "CPU tests use fp32". Approved by Phil on 2026-11-12.
- The seed-count rule and the reported detectable-effect bound use the exact power of Welch's test, 2.63 × σ_init at 5 seeds and 1.83 at 8, not the 2.5 and 1.8 of the plan's §1.3 table (decision 79). Decided by Phil on 2026-11-12. Reason: the table's Welch rows are pooled-variance values, at which Welch's test has 76% power instead of 80%. The plan's table itself is unchanged; correcting it needs a pull request to the frozen plan.

## Decisions

All decisions are in `docs/decisions.md` (rows 1–93 so far).

Answered by Phil in session 2: exact Welch power for the seed-count rule and the bound (decision 79); the √2 × σ_init worst case for the paired test in the seed-count rule (decision 80).

## Hardware tests written, never run

These skip without the hardware. A skip is not a pass. List them with `pytest --collect-only -q -m "gpu or t4"`.

- `gpu`, trainer (`test_trainer.py`): fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusal without scaler settings, the backend restriction. **The fp16, GradScaler and backend code in the trainer has never executed.**
- `gpu`: causal leak under fp16 (`test_causality.py`), `ref`/`opt` agreement under fp16 (`test_reference.py`), `opt` decode peak memory at cache 8k below MHA's (`test_grouped_decode.py`), the tier canary (`test_scaffold.py`).
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

1. Review and merge T-18, T-19 and T-20 (see "In progress").
2. T-06 MLA absorbed path (O+W), T-10 step-0 logit scale (O), T-16 checkpoint and resume (O+W).
3. T-08 properties and FLOPs (W), T-26 wall-clock estimate (W, after T-18), T-27 tables (W, after T-18).

## Notes for the next session

How the loop has been run:

- Workers are Haiku subagents in git worktrees under `/home/user/wt/`, one task each. They commit locally; the orchestrator reviews, pushes and opens the pull request, then squash-merges on green CI.
- Review includes breaking the source on purpose, one change at a time, and checking that a test fails. Every miss so far was a gap in a test, fixed by adding a test.
- Run one pytest process at a time. The container has 4 cores, and two suites at once made runs ten times slower.
- A branch stacked on an unmerged branch conflicts after the squash-merge; resolve with `git checkout --ours` on the files the stacked branch changed.
- The container needs `pip install torch numpy scipy pyyaml tokenizers pyarrow pytest` from PyPI; download.pytorch.org is blocked.

Designs already worked out for tasks not yet started:

- **T-18 manifest.** Run names: `{variant}-{arm tag}-{tokens}-lr{lr}-s{init}-d{data}[-b{branch}]`, with the learning rate in shortest scientific notation and the exponent's minus sign dropped (1e-3 is `lr1e3`, 1.2e-3 is `lr1.2e3`), a ladder name such as `mha_L6` in the variant field, and `{steps}steps` in the tokens field for runs defined by steps. Manifest: one JSON per run or branch with `kind` in run / branch / trunk_checkpoint, a `completed` flag, steps attempted / taken / skipped, scaler scale, versions, GPU, hashes of dataset, tokenizer and data order, the step-time series, `final_metrics` with at least `val_loss`, and the branch's parent run, checkpoint and step. `validate_manifest` reports every problem at once.
- **T-20 evaluation.** The context value c belongs to the predicted token: the number of tokens of its document that precede it inside the window, from the document-offset sidecar. Buckets [0, 64), [64, 256), [256, 512) and c ≥ 512; the last one must include c = 1,024, which the final target of a single-document window reaches, or the bucket-weighted loss cannot equal the aggregate. Sum per-token losses in float64.
- **T-10 logit scale.** Needs a common way to read pre-softmax scores from every attention module; only the sliding-window module exposes weights so far.
- **T-06 absorbed MLA.** Replaces `opt` decode in `mla.py`; the cache already holds only the latent and the positional key.
- **T-16 checkpoint.** The trainer keeps no hidden position: data and schedule are functions of `trainer.step`. A checkpoint needs the model, the optimizer, the scaler, `step`, `steps_taken` and the step records.
- **T-23 benchmark.** Compare sliding-window memory with its cap, not with per-token growth (decision 77).
