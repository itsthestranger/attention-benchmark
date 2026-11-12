# STATUS

Last updated: 2026-11-12, session 3. Every task of `TASKS.md` is merged except T-21 (waits for Phil), T-37 (audit fixes, in progress) and T-31 (this handover). The `cpu` tier is green on `main`.

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

Check the worktrees under `/home/user/wt/` and the remote branches before trusting this.

- T-37, fixes for the audit's findings: tests written by the orchestrator (`tests/test_audit_fixes.py`, `tests/test_backend_enforcement.py`), worker briefed, branch `task/T-37-audit-fixes`.
- T-31, handover: `README.md` with the unfilled pre-registration template, this file, and the audit below, on branch `task/T-31-handover`.
- T-21 probes: waits for two decisions (see "Blocked" and "Open questions").

## Handover audit (plan §17.4 item 6)

A fresh agent that wrote none of the code reviewed the harness against plan §5 and §13 at commit `6697a42`. It ran the `cpu` tier (3,103 passed), made 34 deliberate one-line breaks (33 caught) and compared the frozen configs with the plan value by value (they match). Its verdict: the core is sound on CPU; four defects had to be fixed or decided before the handover. Findings and what happened to each:

| # | Finding | Status |
| --- | --- | --- |
| F1 | One run that skipped more than 0.5% of its steps made every table, plot and the sweep report refuse, even when no comparison read it. | Fixed in T-37: step parity is checked among the runs the analysis reads; the sweep lists such a run as diverged. A skippy run that is read is still refused. |
| F2 | The seed-count rule was fed a ladder fit over every MHA seed present, so adding the seeds it asked for could reverse it (reproduced). | Fixed in T-37: the rule reads the registered base seeds only (`Analysis.rule_fit`). **Still open for Phil**: the committed seed count lives in the pre-registration text only; no frozen file holds it, so nothing refuses a run set that disagrees with it. |
| F3 | The FlexAttention path of SWA could not be selected by any entry point. | Fixed in T-37: `opt_path` in `configs/variants/swa256.yaml` (`chunked` by default), part of the config hash. Phil sets it after the T4 verification, before any SWA run. |
| F4 | The test of the trainer's in-loop backend enforcement could not fail. | Fixed in T-37: `tests/test_backend_enforcement.py` observes the restriction during the forward pass; removing the enforcement fails four tests. `gpu` and `t4` versions for all seven variants are written, never run. |
| F5 | With the comparison gate closed, the headline table and three plots still showed the variants' losses. | Fixed in T-37 for the headline table and plots 1 to 3. **Listed, not fixed**: plots 6 and 7 and `arm_b.md` still show variant values while the gate is closed (existing tests pin that with fixtures that have no null control). It needs variant runs before the controls, which the runbook's order prevents. |
| F6 | Nothing refused a sweep on a seed an arm uses. | Fixed in T-37: `select_lr` refuses registered seeds. This builds in the recommendation on the open sweep-seed question; it is removed if Phil decides otherwise. |
| F7 | The runbook's 100-step smoke runs decayed from step 80, a milder schedule than a real run's first 100 steps. | Fixed in `RUNBOOK.md`: smoke runs are real 300M runs stopped with `--until 100`. |
| F8 | The disclosures of plan §13.2, §13.4 and §13.5 were generated nowhere; §13.3 only in one plot caption. | Fixed in T-37: a fixed "Disclosures" block in `headline.md`. `README.md` states the scope. |
| F9 | `STATUS.md` was stale and omitted four `gpu` tests. | Fixed here. |
| F10 | Arm C has no generated report: the 3% elapsed-time flag and the step-time spread of plan §8.5 have no code. | **Listed, not built.** Arm C is optional and first on the cut list. If it runs, those numbers need code first. |
| F11 | The `gpu` memory test of GQA/MQA decode compares with MHA's whole cache and could miss one expanded layer. | **Listed.** It implements the plan's wording; the `cpu` shape test is the real guard and catches the break. |
| F12 | The comment in `configs/prereg.yaml` says the context buckets are half-open; the top bucket is closed (decision 95). | **Listed for Phil**: the file is frozen. |
| — | A `ref`-mode run records the kernel restriction it ran under although no fused kernel ran. | **Listed, not fixed**: the manifest validator requires a backend whenever a GPU is named. The manifest's `mode` says `ref`. |

Suspicions the auditor could not confirm without hardware are under "To check on real hardware".

What the audit did not cover, by its own account: anything on a GPU; CI and branch protection; a full read of the entry points, the benchmark, the plots and the manifest module; the manifest fields against plan §10.6 one by one; the plots visually; about 155 of the 181 decision rows.

## Gaps found in session 3 (tasks T-32 to T-36)

The original task list reached "analysis done" without outputs the plan requires to be generated by code. All five are merged:

- T-32: the null control, the positive control, σ_data and the per-budget ladder fits in `analyse()`, and `gates.md`. No cross-variant comparison is computed before the controls (decision 170).
- T-33: the LR-sweep rules of plan §7.4 as code, `lr_sweep.md`, and a manifest for a diverged run.
- T-34: Arm B, context use, the wall-clock estimate, and the timing columns from the command line.
- T-35: plots 4 and 5 read the controls and the per-budget fits.
- T-36: standalone 100M runs from flags, plots with unfinished runs left out, the timing-pass id of an Arm C run.

Not built, and outside the task list: a run script for Phase 6 (plan §9.7; the RoPE extension is open decision 5), and the Arm C report (F10).

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

Recommendations were given to Phil in session 3; none is answered yet.

- **MQAR layout (blocks T-21).** The layout of the WIP branch feeds each answer back as input, which lets "a context value not given yet" score 52% at 4 pairs with no recall. Zoology's own generator does not feed answers back. Recommendation: never feed answers back, one query per key, values without replacement, token accuracy against the 1/n line with sequence accuracy as a second column; the same for multi-hop and fuzzy recall.
- **T-21 acceptance criterion (blocks T-21).** With identical settings the jump to recall came at steps 500, 1,250 and 3,250, and one 5,000-step run takes 150 to 190 s here. Recommendation: on `cpu`, the property tests plus exact memorisation of one fixed batch; the "solves held-out 4-pair MQAR" assertion as a `gpu` test over 3 seeds with a step budget of three times the slowest CPU solve.
- **The sweep seed.** A sweep on seed 1, 2 or 3 gives the winning point the name and config hash of a reproducibility rerun, and makes that seed's MHA run a selected one. Recommendation: every sweep on one seed no arm uses, for example 21, written into `configs/prereg.yaml` (frozen: Phil's pull request). `select_lr` takes the seed as an argument and, after T-37, refuses registered seeds.
- **Kaggle's torch, Python and CUDA versions.** Needed to pin the environment; `RUNBOOK.md` §1.5 has the cell to run. Until then: torch 2.15.0, Python 3.12 in CI.
- **Where the committed seed count lives** (audit F2): see the audit table.

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
- **No comparison before the controls (decision 170).** `analyse()` computes no primary or exploratory comparison, multiplier, Arm B or context-use gap until the null control exists and is not detected and the positive control exists. An undetected positive control does not block; it adds a note.
- **Sweep rules (decisions 173, 174).** The edge rule is satisfied before the tie rule is read; a diverged point is a tested point that cannot win; "less than σ_init" is strict; a run is diverged when a validation loss is non-finite or it skipped more than 0.5% of its steps.
- **Arm B's learning rate (decision 179).** Arm A minus Arm B is computed only when both arms ran at one rate; the runbook uses each variant's Arm A rate.
- **Operational cadence (decision 155).** Checkpoint every 600 s, keep 2, an in-run validation point every 250 steps on 256 windows, in `configs/ops.yaml`, outside the config hash.
- **Smoke runs (decision 183).** The first 100 steps of a real 300M run, stopped with `--until 100`, at a placeholder rate of 1e-3.
- **The 200M branch of MHA (plan §12.1).** MHA's Arm A runs are configured with the 100M and 200M branch points from the start.

## Deviations from the plan

- fp32 `ref`/`opt` agreement tolerance is 1e-5, not 1e-6 (plan §5). Approved by Phil on 2026-11-12. Reason: 1e-6 is at the fp32 rounding floor; correct implementations measured up to 1.8e-6.
- The agreement tests also run in float64 at 1e-12, an addition to "CPU tests use fp32". Approved by Phil on 2026-11-12.
- The seed-count rule and the reported detectable-effect bound use the exact power of Welch's test, 2.63 × σ_init at 5 seeds and 1.83 at 8, not the 2.5 and 1.8 of the plan's §1.3 table (decision 79). Decided by Phil on 2026-11-12. Reason: the table's Welch rows are pooled-variance values, at which Welch's test has 76% power instead of 80%. The plan's table itself is unchanged; correcting it needs a pull request to the frozen plan.

## Decisions

All decisions are in `docs/decisions.md` (rows 1–183). Rows marked "for Phil" are readings of the plan or numbers the loop chose.

Answered by Phil in session 2: exact Welch power for the seed-count rule and the bound (decision 79); the √2 × σ_init worst case for the paired test in the seed-count rule (decision 80). Approved by Phil in session 1: the fp32 agreement tolerance of 1e-5, the float64 checks, and the Monte-Carlo allowance of the family-wise error test (0.05 plus three standard errors, decision 7).

## Hardware tests written, never run

These skip without the hardware. A skip is not a pass. List them with `pytest --collect-only -q -m "gpu or t4"`. Counts are collected test cases after T-37.

| File | `gpu` | `t4` | What they check |
| --- | --- | --- | --- |
| `test_backend_enforcement.py` | 7 | 7 | One training step per variant under the enforced kernel. |
| `test_causality.py` | 14 | | Causal leak under fp16, every variant and mode. |
| `test_checkpoint.py` | 9 | | GradScaler state after a resume, resume fidelity in fp16, a checkpoint on CUDA. |
| `test_determinism.py` | 14 | 7 | `ref` bit-identical over 100 steps in fp16 (needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`); the `opt` reproducibility measurement, which applies no threshold. |
| `test_entry_points.py` | 1 | | Cold start and resume on CUDA in fp16; the recorded backend and scale. |
| `test_eval_lm.py` | 1 | | fp16 evaluation against fp32. |
| `test_grouped_decode.py` | 3 | 3 | `opt` decode peak memory and decode time at cache 8k below MHA's. |
| `test_inference_bench.py` | 8 | | Measured against analytic KV bytes for every variant; the benchmark on CUDA. |
| `test_mla_absorbed.py` | 2 | | Absorbed decode against the naive path in fp16. |
| `test_reference.py` | 7 | | `ref`/`opt` agreement under fp16. |
| `test_scaffold.py` | 1 | 1 | The tier canaries. |
| `test_sliding_window.py` | 2 | 2 | Both `opt` paths against `ref`, including compiled FlexAttention; SWA attention time below MHA's. |
| `test_timing_pass.py` | 1 | | The timing pass on CUDA. |
| `test_trainer.py` | 5 | | fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusals. |
| `test_wsd_branch.py` | 7 | | Branch fidelity in fp16 with the scaler state carried over. |

**The fp16, GradScaler and backend-enforcement code paths have never executed.** Neither have `scripts/bench.py` and `scripts/timing_pass.py` on a GPU, the shim notebook, or anything on the real corpus.

## To check on real hardware

- Whether FlexAttention compiles and runs on the T4 (plan §2.3), and which SWA path is faster there. Then set `opt_path` in `configs/variants/swa256.yaml` before any SWA run.
- Whether the fused kernel copies K and V when given a slice of the cache storage (plan §2.3). If it does, decode needs an explicit matmul, disclosed.
- The sliding-window cache allocates up to twice its 256-token cap (decision 77), and MLA's absorbed matrices are a fixed cost per cache outside the KV bytes per token (decision 147). Disclose both with the benchmark.
- Suspicions from the audit, not confirmed: the tiny MLA test models have score widths (12, 36) that the memory-efficient kernel on sm75 may not serve in fp16, so those `gpu` tests could fail under an enforced kernel for a reason unrelated to what they check; the `t4` decode-time test at batch 1 may be dominated by launch overhead; one fp16 trainer test asserts an exact overfit after 150 steps.
- Divergence recording was tested with a forced non-finite loss, never with a real diverged run.

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

1. Merge T-37 after review (deliberate breaks, the full tier, `tests/test_runbook.py`), then this handover branch.
2. T-21, once Phil has answered the two questions: change the layout, review the WIP branch from scratch, add the probe commands to `RUNBOOK.md` §2.3.
3. Pin torch to Kaggle's version when Phil reports it, and run the `cpu` tier on it.

Remaining: T-37 (in progress), T-31 (this handover), T-21 (waits for Phil).

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
