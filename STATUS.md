# STATUS

Last updated: 2026-11-13, session 5, after T-51. Phil answered every open question except Kaggle's versions (decisions 215 to 231). T-46 to T-51 apply the answers and are merged; T-52 (the frozen files) is an open pull request waiting for Phil's approval; T-53 (a fresh agent's review of T-39 to T-52) has not started. The `cpu` tier is green on `main` at `d706f0c`: 3,785 tests, about five and a half minutes. The sections below "Session 5" still describe the state at the end of session 4, except where a session-5 row says otherwise; they are rewritten when T-53 closes.

## Session 5 (2026-11-13): Phil's decisions and what follows from them

| Decision | Answer | Task |
| --- | --- | --- |
| 215 | Probe grid: MQAR at 4, 16, 64 pairs, 8 models, 10 seeds (240 runs) decides Gate 1; probes 2 to 5 are an optional second block; a pilot fixes width, steps and rate | T-51 |
| 216 | The positive control fails when L6 is better than L8 | T-47 |
| 217 | GPU utilisation is an `nvidia-smi` side log; time-to-loss-X is not built | T-51 |
| 218 | The timing pass stays at about 8,800 steps, 3 GPU-h | none |
| 219 | Sweeps run on seed 21, a key of `configs/prereg.yaml` | T-52 |
| 220 | `configs/commitments.yaml` holds the seed count and the ladder cut | T-47 |
| 221 | Thresholds 0.6 and 0.05 confirmed | T-52 |
| 222, 228 | Decisions 41, 42, 53, 86, 177, 178, 184, 185 and every listed reading confirmed as implemented | T-52 (the bucket comment) |
| 223 | A ladder slope that is not positive gives 8 seeds, no multipliers | T-47 |
| 224 | Three shared-rate seeds at 100M for an Arm A variant on its own rate; tuned minus shared reported | T-50, T-51 |
| 225 | Benchmark: three repetitions, median and spread | T-49 |
| 226 | A failed SWA timing check marks SWA and keeps the pass | T-48 |
| 227 | Smoke runs at 3e-4 and 4.8e-3 | T-51 |
| 229 | The rule in prose in `README.md`, amendments in the plan, few keys in `prereg.yaml` | T-52 |
| 230 | Arm C cut; Phase 6 deferred to its own build | T-51 |
| 231 | Coverage test of the multiplier interval | T-52 |

Still open: Kaggle's torch, Python and CUDA versions (`RUNBOOK.md` §1.5). Review findings M4, M8, M9 and F11 stay listed, not fixed.

How session 5 is run: the task flow of `CLAUDE.md` (a task branch and a pull request each). T-52 changes frozen files and waits for Phil's approval on GitHub.

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
- T-37 fixes for the handover audit (#39).
- T-38 probe report for Gate 1 and the 1/n line in plot 9 (#41).
- T-21 probe generators and probe training script (#42).
- T-39 analysis refusals: recipe and backbone, the complete ladder, recorded divergences (#44).
- T-40 checkpoint and resume: the stop checkpoint, branch points, the environment (#45).
- T-43 `scripts/prune.py` (#47).
- T-45 plot 7, the plots' timing refusals, the generated hardware line (#48).
- T-42 `timing.md` and `inference.md` (#49).
- T-41 the bucketed loss at every validation point and peak memory (#50).
- T-44 runbook corrections, `make_plots --l10-cut`, CODEOWNERS (#51).
- T-46 the decisions of session 5 and the tasks that apply them (#52).
- T-47 the positive control's direction, a slope that is not positive, `configs/commitments.yaml` (#53).
- T-50 tuned against shared learning rate, `shared_rate.md` (#54).
- T-49 the inference benchmark in three round-robin repetitions, `bench/2` (#55).
- T-48 SWA not below MHA in the timing pass is recorded and marked, not refused (#56).
- T-51 the runbook for the decisions of session 5, with the probe commands of §2.3 (#57).

## In progress

- T-52, branch `task/T-52-frozen`: the sweep seed, the confirmed thresholds and the bucket comment in `configs/prereg.yaml`; the rule in prose in `README.md`; plan §18 (amendments); the coverage allowance in `tests/tolerances.py` with its test; `select_lr` reads the registered seed. A pull request labelled `needs-phil`; Phil approves and merges it.
- T-53: a fresh agent reviews T-39 to T-52. Not started. Its brief is written; it reviews the T-52 branch, which holds everything.
- No worker is running.

## Handover checklist (plan §17.4)

| # | Item | State |
| --- | --- | --- |
| 1 | Every task merged, `cpu` tier green in CI | Met except T-31, the handover itself, which closes when item 4 does. |
| 2 | Every `gpu` and `t4` test written, skipping cleanly, listed as not run | Met; listed below (105 collected). |
| 3 | Headline table and all ten plots generate from fake manifests and blank | Met. `probes.md`, `timing.md` and `inference.md` build blank and from synthetic records too. |
| 4 | `RUNBOOK.md` gives exact commands for verification and each GPU week, with gates | Met since T-51 (#57): §2.3 has the pilot, block 1 and block 2. Every command parses with its script's own parser; none has run on a GPU. |
| 5 | `STATUS.md` lists every decision, open question and deviation | This file, with `docs/decisions.md` rows 1 to 214. |
| 6 | A fresh agent reviewed the harness against plan §5 and §13; findings fixed or listed | Two reviews done (sessions 3 and 4); the second one's findings are below with their status. The fixes for it (T-39 to T-45) were reviewed by the orchestrator with deliberate breaks, not by a fresh agent. The session-3 fixes had the same status, and the second review found defects in them, so a short fresh pass over T-39 to T-45 is recommended before the first GPU hour. |

Items 1 and 6 are open: T-52 waits for Phil, and the fixes of sessions 4 and 5 have not been seen by a fresh agent (T-53).

## Session-4 review (2026-11-13)

Phil asked for a second review of everything built so far. A fresh agent on the full model, which wrote none of the code, reviewed `main` at `384e8b3`; its report is `docs/review_s4.md`. It checked every number it looked at against the plan (all match), made 50 deliberate breaks (44 caught) and ran the runbook's whole command sequence on CPU with tiny budgets. The orchestrator re-ran its reproductions of H1, H2, G1, G6 and G8 and got the same results.

| # | Finding | Status |
| --- | --- | --- |
| H1 | The analysis compares, and gives verdicts on, runs trained under different recipes, init rules, precision or with `--override`; the headline's parameter and `d_ff` columns come from `configs/`, not from the manifests. | **Fixed in T-39** (#44): such runs are refused, and the static columns come from the manifests. |
| H2 | The seed-count rule is computed and printed from an incomplete ladder; in synthetic campaigns near the threshold the decision differed from the complete ladder's. | **Fixed in T-39** (#44): no decision until MHA, L6 and L10 have their registered seeds; `--l10-cut` declares the two-point ladder. **For Phil**: where that declaration should live. |
| G1 | A stop at `--until` leaves no checkpoint to resume from, so an MHA trunk re-runs up to ten minutes of steps and overwrites the branch-point checkpoint its 100M branch used. `RUNBOOK.md` says it resumes from step 1,221. | **Fixed in T-40** (#45): a stop leaves a checkpoint, a trunk resumes from its newest state including the branch point, and a branch point is never overwritten. The runbook's statement is now true. |
| G2 | A manifest records only the last session's environment and GPU, and a resume after an image change is not refused. | **Fixed in T-40** (#45): the environment is in every checkpoint, and a resume or branch under another torch, CUDA, driver or GPU is refused. `gpu_shared` is still the last session's value. |
| G3 | Plot 7 draws every manifest in the directory in the variant's colour, sweep points at rejected learning rates included, and shows variant losses while the gate is closed. The session-3 note that the runbook's order prevents this was wrong: the variant sweeps are such runs. | **Fixed in T-45** (#48): plot 7 draws only the Arm A runs and branches the analysis names, and MHA alone while the gate is closed. |
| G4 | Peak memory, GPU utilisation and the context-bucketed loss at every evaluation (plan §10.2) are not recorded for a training run. | **Fixed in part, T-41** (#50): peak memory and the bucketed loss at every validation point are recorded. **For Phil**: GPU utilisation and time-to-loss-X are not; the plan does not define them. |
| G5 | Attention's measured share, peak memory, the `ref` track, the spread, prefill, maximum batch at 8k and measured KV bytes are written to JSON and read by nothing. | **Fixed in T-42** (#49): `scripts/timing_report.py` writes `timing.md` and `inference.md`, with blank forms. |
| G6 | One recorded diverged run makes `make_tables` and `make_plots` refuse without `--skip-unfinished`, which the runbook's final step forbids. | **Fixed in T-39** (#44): a diverged run is left out and named in a note. |
| G7 | Checkpoints reach about 20 GB in week 1 and nothing prunes them. | **Fixed in T-43** (#47) and the runbook: `scripts/prune.py`, a dry run unless `--apply` is given. |
| G8 | The plots skip the timing-source refusals the tables apply. | **Fixed in T-45** (#48). |
| M1 | The hardware disclosure in `headline.md` is fixed text, printed whatever the manifests say. | **Fixed in T-45** (#48): the line is generated from the manifests. |
| M2 | A non-finite loss on the last training step blocks the manifest for good. | **Fixed in T-40** (#45). |
| M3 | The positive control is two-sided: L6 significantly better than L8 would read as detected. | **For Phil.** |
| M4 | `bench.py --with-ref` repeats the `opt` KV check and the maximum-batch search for each `ref` document and labels the result `ref`. | **Listed, not fixed.** The tables print what the documents hold; the repeated search costs benchmark time only. |
| M5 | The timing pass compares no dataset hash while `configs/data.yaml` is unset, and with `--with-ref` and the share runs it is about 8,800 steps against the plan's 4,620. | **Stated in the runbook** (§3.1): the pass runs after the data hashes are set and takes about twice the plan's 1.5 GPU-h. Whether to cut it is **for Phil**. |
| M6 | A resume is refused when the number of visible CUDA devices changes; the shim does not set `CUDA_VISIBLE_DEVICES`. | **Stated in the runbook** (§0): start a job again with the same `CUDA_VISIBLE_DEVICES`. |
| M7 | The shim can only call `scripts.train`, and its example config collides by name with the runbook's MHA runs. | **Fixed in T-44**: the example run spec carries both branch points. The shim still calls `scripts.train` only; the runbook says how to run another command with it. |
| M8 | `lr_source` and `timing_pass_id` are free text, never checked against an existing run or file. | Listed, not fixed. |
| M9 | A FlexAttention SWA run would record the backend `efficient`, a kernel it never called. | Listed; to settle on the T4. |
| M10 | `gates.md` prints two "slope at 300M" values that differ once MHA has 8 seeds, unexplained. | **Fixed in T-45** (#48): one sentence in `gates.md`. |
| M11 | CODEOWNERS does not cover the files that decide what CI runs: `ci.yml`, `tests/conftest.py`, `tests/hardware.py`, `pyproject.toml`. | **Fixed in T-44**, for Phil to confirm: CODEOWNERS lists the four files. |
| M12 | No smoke run reaches an evaluation, which runs at batch 64 and not at the micro-batch. | **Fixed in T-44**: each smoke run evaluates twice (`--eval-every-steps 50`). |
| M13 | The `opt` reproducibility measurement is built and then dropped; the runbook has no step for `scripts/measure_reproducibility.py`. The session-3 statement that the tests "measure and record" was wrong. | **Fixed in T-44**: `scripts/measure_reproducibility.py` is a runbook step (§2.2) and writes the record. The `gpu` and `t4` tests still keep nothing. |
| M14 | Six breaks no test caught: B10 (Arm B outside the step-parity scope), B22 (`gpu_shared` always false), B36 (reruns not limited to registered seeds), B45 (`skipped_in_first_100` counts step 100), B46 (seed-count threshold `>=`), B47 (in-run validation ignores `eval_curve_windows`). | **Fixed**: each has a test now, checked against its break (#44, #45). |

What the review did not cover, by its own account: anything on a GPU; `scripts/graph_diff.py` and `src/utils/flops.py` beyond running them; `src/eval/retrieval.py`; `src/attention/recurrence.py`; most test files line by line; plots 3, 9 and 10 visually; T-21 and T-38; GitHub (CI, branch protection).

### Decisions the review says are Phil's

Rows of `docs/decisions.md` that change what is measured or how it is judged and were not marked for him. Each is implemented as described and can be changed in one place.

- **41**: under the paired test the null control pairs seeds 11 to 15 with MHA seeds 1 to 5 in list order.
- **42**: exploratory intervals use MHA seeds 1 to 3 when paired, and all 5 or 8 MHA seeds under Welch.
- **53**: the first update runs at learning rate 0, so every run has one effective update fewer than its step count.
- **86**: the ladder fit weights every run, so L8 (5 or 8 seeds) outweighs L10 (3). It sets the slope behind the seed-count rule and every multiplier.
- **126**: no primary verdict without a seed-count decision, so a missing or non-positive ladder slope means the primaries are never tested.
- **127**: one learning rate per variant in Arm A; the "side by side" of plan §7.4 item 5 is two single-seed sweep losses at 100M.
- **145**: benchmark points: prefill at batch 1 only; a decode number is the mean of 10 steps after 3 warm-ups, one repetition, no spread.
- **150**: "SWA attention time below MHA's" is judged as share × ms per step from a separate instrumented run.
- **177, 178**: Arm B and the context-use gaps use the Gate 2 test and seed sets.
- **184, 185**: step parity is checked among the runs the analysis reads; the rule reads the registered-seed fit while the reported fit uses every seed.
- Lower weight: **25**, **32**, **122**, **140**.

### Plan problems the review found

- **§2.3**: SWA's attention time must be below MHA's or the timing run fails, yet the chunked fallback is predicted to save "near zero" at context 1,024. With the default `opt_path: chunked` the timing pass will probably exit 1.
- **§7.4 item 5**: the plan does not say whether Arm A runs each variant at its tuned rate only.
- **§7.3 against §15**: the scaler settings are frozen from smoke runs made before any learning rate is known.
- **§10.6**: PyTorch does not report the kernel it selected; the harness records the one it enforced.
- **§8.4**: the plan does not say that a positive control detected in the wrong direction fails.
- **§12.3**: "20 GB holds about 100 checkpoints" is below what week 1 writes.
- **§10.2**: "GPU utilization" and "time-to-loss-X" are required with no definition.

## Session-3 audit (plan §17.4 item 6)

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

- **`RUNBOOK.md` §2.3, the probe commands** (part of T-44): the plan fixes the five probes but not their run grid, and its numbers disagree. See the first open question.

## Open questions

- **The probe grid (blocks `RUNBOOK.md` §2.3).** Plan §12.1 budgets about 250 short runs in 3 GPU-hours. Plan §9.2 asks for 10+ seeds per point for 7 variants and the anchor, which is 80 runs per difficulty of one probe; MQAR alone at 4 to 256 pairs in powers of two is 560. Nothing fixes width, steps or learning rate per probe, and the runs are not "seconds": a 2-layer, width-64 model needs 2,000 to 4,000 steps for 4 pairs. Options: fit the budget (for example MQAR at 4, 16 and 64 pairs for all 8 models at 10 seeds, 240 runs, and probes 2 to 5 at one size with fewer seeds or not at all), or keep the grid and raise the budget. Recommendation either way: one fixed width, step count and learning rate per point for every model, found by a short pilot on MHA and the anchor only.
- **The sweep seed.** A sweep on seed 1, 2 or 3 gives the winning point the name and config hash of a reproducibility rerun, and makes that seed's MHA run a selected one. Recommendation: every sweep on one seed no arm uses, for example 21, written into `configs/prereg.yaml` (frozen: Phil's pull request). `select_lr` takes the seed as an argument and refuses registered seeds.
- **Kaggle's torch, Python and CUDA versions.** Needed to pin the environment; `RUNBOOK.md` §1.5 has the cell to run. Until then: torch 2.15.0, Python 3.12 in CI.
- **Where the committed seed count lives** (session-3 audit F2): it exists in the pre-registration text only, so nothing refuses a run set that disagrees with it.
- **Where the L10 cut is declared** (decision 203): `--l10-cut` on the command line, or a key in the frozen `configs/prereg.yaml`.
- **The decisions and plan problems of the session-4 review**, listed above. M3 (direction of the positive control), the definition of GPU utilisation and time-to-loss-X (G4), and the timing-pass step count (M5) are the ones a task waits on.

Answered by Phil on 2026-11-13: the MQAR layout and the T-21 acceptance criterion (decisions 189, 190).

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
- Plan §10.2 asks for GPU utilisation and time-to-loss-X per run. Neither is recorded, because the plan does not define them (review G4). Peak memory and the bucketed loss at every validation point are recorded since T-41.
- Plan §5 asks that the `opt` reproducibility tolerance be measured once and recorded. The record is a runbook step (`scripts/measure_reproducibility.py`, §2.2); the `gpu` and `t4` tests measure it and keep nothing, and apply no threshold (decision 99).

## Decisions

All decisions are in `docs/decisions.md` (rows 1–214). Rows marked "for Phil" are readings of the plan or numbers the loop chose.

Answered by Phil in session 2: exact Welch power for the seed-count rule and the bound (decision 79); the √2 × σ_init worst case for the paired test in the seed-count rule (decision 80). Approved by Phil in session 1: the fp32 agreement tolerance of 1e-5, the float64 checks, and the Monte-Carlo allowance of the family-wise error test (0.05 plus three standard errors, decision 7).

Answered by Phil in session 4: the probe layout and reporting (decision 189) and the T-21 acceptance split (decision 190).

## Hardware tests written, never run

These skip without the hardware. A skip is not a pass. List them with `pytest --collect-only -q -m "gpu or t4"`. Counts are collected test cases after T-21.

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
| `test_probes.py` | 3 | | A tiny MHA solves held-out 4-pair MQAR in fp16, seeds 1 to 3, 12,750 steps each. The same settings reach 1.0 on CPU in fp32. |
| `test_reference.py` | 7 | | `ref`/`opt` agreement under fp16. |
| `test_scaffold.py` | 1 | 1 | The tier canaries. |
| `test_sliding_window.py` | 2 | 2 | Both `opt` paths against `ref`, including compiled FlexAttention; SWA attention time below MHA's. |
| `test_timing_pass.py` | 1 | | The timing pass on CUDA. |
| `test_trainer.py` | 5 | | fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusals. |
| `test_wsd_branch.py` | 7 | | Branch fidelity in fp16 with the scaler state carried over. |

**The fp16, GradScaler and backend-enforcement code paths have never executed.** Neither have `scripts/bench.py`, `scripts/timing_pass.py` and the CUDA path of `scripts/train_probe.py` on a GPU, the shim notebook, or anything on the real corpus.

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

1. Phil fixes the probe grid (first open question). Then: a `configs/probes.yaml` or plain loops in `RUNBOOK.md` §2.3 with `scripts/train_probe.py`, `scripts/probe_report.py` and plot 9, a test in `tests/test_runbook.py`, and T-31 closes.
2. A fresh agent reviews T-39 to T-45 (the diffs of pull requests #44, #45 and #47 to #50 and this one) against `docs/review_s4.md`, as the second review did for the first one's fixes.
3. Pin torch to Kaggle's version when Phil reports it, and apply his answers on the open questions and the readings.

Remaining: T-31, which waits for §2.3.

## Notes for the next session

How the loop has been run:

- Session 4: workers are Haiku subagents started with the Agent tool; a review round is the orchestrator reading the diff and running deliberate one-line breaks from a spec file in a separate detached worktree. A worker that hands back is resumed with the findings. T-21 took three rounds, T-38 two.
- A deliberate break that passes is not always a gap: check whether the change is equivalent (session 4 had two) before sending it back.
- The task-flow of `CLAUDE.md` (task branches, a pull request each, squash-merge on green CI) was followed in session 4 although the session was started with a single designated branch.
- The independent review ran on the full model for about 40 minutes beside two workers without slowing them much, as long as every pytest ran with `OMP_NUM_THREADS=1`.
- Agents share the session's scratch directory. A worker overwrote the orchestrator's break runner there; it now lives in its own folder, and briefs name a scratch directory per task.
- Two tasks that pass alone can fail together: T-41 added a file to every run directory and T-43's tests assert exact file sets. Merge `main` into a branch and run the full tier before its pull request, and read the whole summary line, not its tail.
- A worker that stops at a conflict with an existing test is doing the right thing. In session 4 this happened three times (a pinned key set, a fixture without an environment, an old signature); each needed one approved edit outside the task's files.

- **Run every pytest with `OMP_NUM_THREADS=1`.** On this container torch with four threads is far slower on tiny models (a worker measured a factor of about a thousand on a small matmul). With one thread the full `cpu` tier takes about four minutes.
- Deliberate breaks are run in a separate detached worktree of the branch under review, never in the worker's own worktree, so that a worker's test run and the review cannot disturb each other.
- For `O+W` tasks the tests are committed and pushed before the worker is briefed, and checked first against the existing code (which tests fail, which already pass and so guard against over-correcting) or against a throwaway implementation.
- Workers are told to stop and report when an existing test conflicts with the brief. In session 3 this happened three times, and each time stopping was right.
- A test that parses the runbook's commands (`tests/test_runbook.py`) found three missing command-line paths before any human did. Keep it green when a script's flags change.
- Session 3 hit the account's 5-hour usage limit while the audit ran on the full model. Workers are Haiku; the audit is the one place the full model was used.

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
