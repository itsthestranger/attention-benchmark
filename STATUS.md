# STATUS

Last updated: 2026-11-12, during session 3. 26 of 32 tasks merged; `cpu` tier green on `main` (2,727 tests, about three minutes).

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
- T-27 headline table with refusal gates (#24).
- T-17 WSD decay branches (#25).
- T-11 graph-diff check (#26).
- T-22 retrieval generators and scorer (#27).
- T-06 MLA absorbed path for inference (#28).
- T-23 inference benchmark and training timing pass (#29).
- T-29 entry points and Kaggle shim (#30).
- T-28 plots (#31).
- T-32 controls, noise components and the gate report (#32).
- T-35 plots 4 and 5 read the controls and the per-budget ladder fits (#33).
- T-33 LR-sweep rules, sweep report and diverged runs (#34).
- T-34 Arm B, context use, wall-clock estimate and timing columns (#35).
- T-36 three command-line gaps (#37).
- T-30 `RUNBOOK.md` with `tests/test_runbook.py` (#38).

## In progress

Session 3 (2026-11-12), state when this note was written; check the worktrees under `/home/user/wt/` and the remote branches before trusting it.

- T-31 handover: `README.md` with the unfilled pre-registration template is on branch `task/T-31-handover` (pushed, no pull request yet). The audit by a fresh agent against plan §5 and §13 is running in `/home/user/wt/audit`; it was cut off once by a usage limit and resumed. Its findings are not in yet: until they are fixed or listed here, handover item 6 is open.
- T-21 probes: waits for Phil's two decisions.

## Gaps found in session 3 (tasks T-32 to T-35 added to `TASKS.md`)

The task list reached "analysis done" without these, and the plan needs each of them as generated output (plan §11: nothing typed or computed by hand):

- `primary_test.null_control` and `positive_control` exist, and nothing called them. No report showed σ_data, the test choice, the detectable-effect bound or the seed count. T-32 adds them to `analyse()` and writes `gates.md`; its tests are on `task/T-32-gate-report` (worker briefed). T-32 also makes `analyse()` refuse every cross-variant comparison until the null control exists and is not detected and the positive control exists (plan §8.1, §9.3, §9.4). **For Phil**: that refusal is the loop's reading of the plan.
- No code applies the edge rule and the tie rule of plan §7.4 or reports a sweep, and a diverged run cannot write a manifest today (a completed manifest needs a finite loss), so "record everything, including divergences" is not met. T-33.
- No Arm B table (plan §8.3), no command line for the RQ-W estimate (§8.5), no intervals for the context-use gaps (§10.3), and `make_tables.py` cannot read the timing-pass and benchmark files into the headline table. T-34.
- Plot 5 shows the ladder at 300M only and plot 4 cannot show the null-control verdict, because `analyse()` did not expose them. T-35.
- MHA's Arm A runs need the 200M branch point as well as the 100M one (plan §12.1: "plus 200M branches on L8"), so their run specs must say `branches: [100M, 200M]` from the start. For `RUNBOOK.md`.

## Blocked

- **T-21 probes**, branch `task/T-21-probes` (pushed, one commit marked WIP and UNREVIEWED). The generators, the oracle, the stream and the training script exist, and 115 of 116 tests pass. The one that fails is the task's acceptance test, "a tiny MHA solves 4-pair MQAR on CPU": the worker could not reach a sequence accuracy of 1.0 and did not lower the bar. Measured afterwards on the synthetic task, with a 2-layer model of width 64 (8 heads of 8), 64 sequences per step:
  - 64 keys and values, learning rate 3e-3: 0.87 after 6,000 steps and still rising. At 1e-3: 0.05 after 6,000.
  - 16 keys and values, 3e-3: 1.0 after 4,000 steps. The same at 6e-3: 0.06 after 3,000.
  - 8 keys and values: 0.06 to 0.11 after 3,000 steps at 3e-3, 6e-3 and 1e-2, and for three init seeds.
  - Deeper models (4 and 8 layers) were slower in the worker's runs. Wider ones were not measured; a head dim of 8 gives RoPE only four frequencies, which may be why "attend to the previous token" is slow to form.
  Session 3 measurements (scratch script, 2 layers, 8 heads, 64 sequences per step, learning rate 3e-3, held-out accuracy on 1,024 sequences every 250 steps):
  - Width 128, 64 keys and values, 3,000 steps, 3 seeds each: the layout of the WIP branch (answers fed back as input) sits on the shortcut plateau, token accuracy 0.53, sequence accuracy 0.04 to 0.05. With the answers blanked in the input (the layout of Zoology's own generator, checked in `zoology/data/multiquery_ar.py`: each key queried once, values distinct, the answer only in the labels) it sits at the no-recall level, 0.27. Neither solves. A wider model is not the fix.
  - Width 64, 16 keys and values, 5,000 steps, 3 seeds each. Answers blanked: a sharp jump from 0.33 to 0.98 token accuracy at steps 3,250, 500 and 1,250; sequence accuracy first 1.0 at 4,250, 2,250 and 3,000; 3 of 3 end at 1.0. Answers fed back: a slow climb from the 0.55 plateau; first 1.0 at 4,500 and 5,000, and one seed ends at 0.998.
  - One run costs 150 to 190 s here (25 to 75 ms per step on one or two threads).
  So the criterion can be met only by a test of several thousand steps (half a minute to a few minutes on CPU) whose outcome hangs on a phase transition and may differ on CI's CPU. Options: accept one long test after checking it across seeds and thread counts; change the criterion; or change the task. **Not decided.** See also the first open question.

## Open questions

- **A shortcut in the MQAR probe as specified in the T-21 brief.** Every key is queried exactly once, so "answer with a context value not given yet" scores (1/4 + 1/3 + 1/2 + 1) / 4 = 52% token accuracy at 4 pairs with no recall at all; the plateau is visible in every training curve. Drawing the queries with replacement removes it, but then the tiny model learned nothing in 4,000 steps (0.30). This matters for Gate 1, which reads the anchor's failure off these probes: token accuracy must be compared with that baseline, or sequence accuracy used, or the task changed. The format was the orchestrator's choice (the plan names the probe, not its layout), so it can be changed, but the choice affects what Gate 1 measures. Recommendation: keep one query per key, report sequence accuracy, and state the 52% baseline.
- **A sweep run and a reproducibility rerun can be the same run.** MHA's sweep point at the winning learning rate (100M, seed 1) has exactly the config, name and config hash of the reproducibility rerun of seed 1 (plan §7.4, §8.1). Using the sweep run as the rerun would feed a run that was picked for its low loss into s_rep. Should the rerun of seed 1 be a separate run, and if so how is it told apart: a field in the config, or reruns on seeds other than the sweep's? Until answered, `RUNBOOK.md` cannot give the Arm N commands. Recommendation: sweep on a seed that Arm N does not use.
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
- **Extra comparability gates (decision 125).** `make_tables.py` also refuses runs that differ in dataset hash, tokenizer hash, GPU, mode or precision, beyond the plan's list of steps, backend and versions.
- **MLA cache layout (decision 142).** The MLA cache is one tensor of 216 values per token per layer, and the absorbed matrices travel with the cache. KV bytes per token are unchanged (432).
- **`configs/data.yaml` (decision 38).** Not frozen. Consider adding it to CODEOWNERS once the hashes are filled in.

## Deviations from the plan

- fp32 `ref`/`opt` agreement tolerance is 1e-5, not 1e-6 (plan §5). Approved by Phil on 2026-11-12. Reason: 1e-6 is at the fp32 rounding floor; correct implementations measured up to 1.8e-6.
- The agreement tests also run in float64 at 1e-12, an addition to "CPU tests use fp32". Approved by Phil on 2026-11-12.
- The seed-count rule and the reported detectable-effect bound use the exact power of Welch's test, 2.63 × σ_init at 5 seeds and 1.83 at 8, not the 2.5 and 1.8 of the plan's §1.3 table (decision 79). Decided by Phil on 2026-11-12. Reason: the table's Welch rows are pooled-variance values, at which Welch's test has 76% power instead of 80%. The plan's table itself is unchanged; correcting it needs a pull request to the frozen plan.

## Decisions

All decisions are in `docs/decisions.md` (rows 1–143 so far).

Answered by Phil in session 2: exact Welch power for the seed-count rule and the bound (decision 79); the √2 × σ_init worst case for the paired test in the seed-count rule (decision 80).

## Hardware tests written, never run

These skip without the hardware. A skip is not a pass. List them with `pytest --collect-only -q -m "gpu or t4"`.

- `gpu`, trainer (`test_trainer.py`): fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusal without scaler settings, the backend restriction. **The fp16, GradScaler and backend code in the trainer has never executed.**
- `gpu`: causal leak under fp16 (`test_causality.py`), `ref`/`opt` agreement under fp16 (`test_reference.py`), `opt` decode peak memory at cache 8k below MHA's (`test_grouped_decode.py`), the tier canary (`test_scaffold.py`).
- `gpu`, determinism (`test_determinism.py`): `ref` mode in fp16 bit-identical over 100 steps twice, for all seven variants. Needs `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the environment before pytest starts.
- `gpu` and `t4`, reproducibility (`test_determinism.py`): the `opt` measurement in fp16 for all seven variants; it records differences and applies no threshold.
- `gpu`, absorbed MLA (`test_mla_absorbed.py`): absorbed decode against the naive path in fp16.
- `gpu`, branches (`test_wsd_branch.py`): branch fidelity in fp16 for all seven variants, with the scaler state carried over.
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

1. T-21 probes: settle the acceptance criterion and the shortcut (see "Blocked" and the first open question), then review the WIP branch from scratch with deliberate breaks, as for every task.
2. T-28 plots (W; reuse `tests/fixtures/fake_campaign.py` and `scripts.make_tables.analyse`), T-23 benchmark and timing scripts (W; unblocked by T-06), T-29 entry points and Kaggle shim (W; unblocked by T-17 and T-20).
3. T-30 `RUNBOOK.md` (O; needs the answer on sweep runs and reruns), then T-31 handover audit by a fresh agent.

Remaining: T-31 (audit running), T-21 (waits for Phil).

## Notes for the next session

How the loop has been run:

- Workers are Haiku subagents in git worktrees under `/home/user/wt/`, one task each. They commit locally; the orchestrator reviews, pushes and opens the pull request, then squash-merges on green CI.
- Review includes breaking the source on purpose, one change at a time, and checking that a test fails. Every miss so far was a gap in a test, fixed by adding a test.
- Run one pytest process at a time. The container has 4 cores, and two suites at once made runs ten times slower. Three workers each running the suite, plus the orchestrator's own checks, took the full tier from 2.5 to 13 minutes in session 2: tell workers to run the full suite once, and run deliberate-break checks while no worker is testing.
- If a deliberate-break run is interrupted, the source file it was changing stays changed. Check `git status` in that worktree and restore the file before anything else.
- Acceptance tests for `O+W` tasks are checked against a throwaway implementation before hand-off (kept out of the commit). In session 2 this caught three errors in the orchestrator's own tests and one defect in merged code.
- Workers sometimes call session tools by mistake (a stray message or interrupt call); each was denied and did nothing. Briefs now say not to.
- A branch stacked on an unmerged branch conflicts after the squash-merge; resolve with `git checkout --ours` on the files the stacked branch changed.
- The container needs `pip install torch numpy scipy pyyaml tokenizers pyarrow pytest` from PyPI; download.pytorch.org is blocked.

Designs already worked out for tasks not yet started:

- **T-29 entry points.** `scripts/branch.py` belongs here (decision 133): it calls `branch.start_or_resume_branch` and writes a `kind: branch` manifest with `branch.branch_manifest_block`; `train.py` gives its `Checkpointer` the steps from `branch.branch_steps`. Give `ref` and `opt` their own checkpoint roots (decision 115). Record the trainer's enforced backend on the attention modules with `set_sdpa_backend` and write it to the manifest (decision 110).
- **T-23 benchmark.** Compare sliding-window memory with its cap, not with per-token growth (decision 77). MLA decodes on the absorbed path in `opt` mode and on the naive path in `ref` mode (decision 141); report both (plan §2.3). The timing pass feeds `analysis.rqw_estimate.estimate` a mapping of variant to ms per step, and `make_tables.headline_rows` takes `train_ms_per_step` and `decode_tok_per_s` mappings.
- **T-28 plots.** `scripts.make_tables.analyse` returns everything a plot needs (losses per seed, the test selection, the ladder fit, the seed-count decision, the comparisons and multipliers) and `tests/fixtures/fake_campaign.py` builds a synthetic experiment. Plots must also build blank, from `analyse([])`.
