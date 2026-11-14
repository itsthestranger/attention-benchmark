# STATUS

Last updated: 2026-11-14, end of session 6. Every task of `TASKS.md` is merged. Phil merged T-52 (#59) and confirmed the five readings of session 5 (decision 245), with two additions that are built: plots 2 and 8 draw the benchmark's range as whiskers (T-59), and `scripts/swa_paths.py` records the attention times of both SWA paths at the verification step (T-60). A fourth independent review (`docs/review_s6.md`) looked at both; its findings are fixed in T-61 or listed below. One reading waits for Phil's confirmation: decision 250, what "FlexAttention runs on the T4" means. The `cpu` tier is green: 3,968 tests, about six minutes. The harness is ready for Phil's verification step (`RUNBOOK.md` §1). No GPU job has run.

## Done

Merged, with green CI. Pull request numbers in brackets.

- Bootstrap commit: `CLAUDE.md`, `TASKS.md`, `docs/plan_v5.md`.
- M0 to M5, sessions 1 to 3: T-00 (#1), T-01 (#2), T-02 (#3), T-13 (#4), T-24 (#5), T-03 (#6), T-14 (#7), T-12 (#8), T-04 (#9), T-15 (#10), T-07 (#11), T-05 (#13), T-25 (#14), T-20 (#15), T-19 (#16), T-18 (#17), T-08 (#18), T-16 (#19), T-10 (#21), T-09 (#22), T-26 (#23), T-27 (#24), T-17 (#25), T-11 (#26), T-22 (#27), T-06 (#28), T-23 (#29), T-29 (#30), T-28 (#31), T-32 (#32), T-35 (#33), T-33 (#34), T-34 (#35), T-36 (#37), T-30 (#38), T-37 (#39).
- Session 4: T-38 (#41), T-21 (#42), T-39 (#44), T-40 (#45), T-43 (#47), T-45 (#48), T-42 (#49), T-41 (#50), T-44 (#51).
- Session 5:
  - T-46 Phil's decisions 215 to 231 and the tasks that apply them (#52).
  - T-47 the positive control's direction, a slope that is not positive, `configs/commitments.yaml` (#53).
  - T-50 tuned against shared learning rate, `shared_rate.md` (#54).
  - T-49 the inference benchmark in three round-robin repetitions, `bench/2` (#55).
  - T-48 SWA not below MHA in the timing pass is recorded and marked (#56).
  - T-51 the runbook for the decisions of session 5, with the probe commands of §2.3 (#57).
  - T-53 the third independent review, `docs/review_s5.md`, and the tasks for its findings (#60).
  - T-56 `prune --final-short`, the git checkout checked before the first step, the shim's paths (#61).
  - T-54 declarations from the committed file only; no test pins a file Phil fills in (#62).
  - T-55 what waits for the committed seed count; plot 6 and the gate; the ladder cut in every fit; the benchmark's range in the headline (#63).
  - T-57 runbook and wording fixes of the review, this file (#64).
- Session 6:
  - T-52 the frozen files for the decisions of session 5: merged by Phil (#59).
  - T-58 Phil's decisions of session 6 (245 to 248) and the tasks that apply them (#65).
  - T-59 plots 2 and 8 draw the smallest and the largest repetition as whiskers (#66).
  - T-60 `scripts/swa_paths.py`, its section of `timing.md`, the runbook command (#67).
  - T-61 the fourth independent review, its fixes, this file; closes T-31.

## In progress

- Nothing. No worker is active.

## Blocked

- Nothing.

## Open questions

- **Kaggle's torch, Python and CUDA versions.** Needed to pin the environment; `RUNBOOK.md` §1.5 has the cell to run. Until then: torch 2.15.0, Python 3.12 in CI.
- **Decision 250, for Phil to confirm**: in the rule for SWA's path, "FlexAttention runs on the T4" means that the agreement test passes and that the record's flex track has times, not an error; a `[flex]` timing case that fails at its assertion leaves the path at `flex`. See "Session 6" below. To settle before the `t4` session of `RUNBOOK.md` §1.2.

Everything else is answered: decisions 215 to 231 (2026-11-13) and decision 245 (2026-11-14), which confirms decisions 232, 240, 241, 242 and 244.

## Handover checklist (plan §17.4)

| # | Item | State |
| --- | --- | --- |
| 1 | Every task merged, `cpu` tier green in CI | Met. Every task is merged, T-31 with T-61. |
| 2 | Every `gpu` and `t4` test written, skipping cleanly, listed as not run | Met; listed below (106 collected: 86 `gpu`, 20 `t4`). |
| 3 | Headline table and all ten plots generate from fake manifests and blank | Met. `gates.md`, `arm_b.md`, `shared_rate.md`, `context.md`, `rqw.md`, `lr_sweep.md`, `probes.md`, `timing.md` and `inference.md` build blank and from synthetic records too. |
| 4 | `RUNBOOK.md` gives exact commands for verification and each GPU week, with gates | Met since T-51. Every command parses with its script's own parser; none has run on a GPU. |
| 5 | `STATUS.md` lists every decision, open question and deviation | This file, with `docs/decisions.md` rows 1 to 251 and plan §18. Plan §18 does not list decisions 245 to 251: none changes what the plan says is measured. |
| 6 | A fresh agent reviewed the harness against plan §5 and §13; findings fixed or listed | Four reviews: sessions 3, 4, 5 and 6. The third (`docs/review_s5.md`) reviewed T-39 to T-52, and its fixes (T-54 to T-57) were re-checked by another fresh agent (`docs/review_s5/recheck_s5.md`). The fourth (`docs/review_s6.md`) reviewed T-59 and T-60: 86 deliberate breaks, 54 caught; its findings are fixed in T-61 or listed under "Session 6". The fixes of T-61 were checked by the orchestrator's own breaks (40, all caught after one test was strengthened), not by another fresh agent. |

## Session 5: Phil's decisions

Rows of `docs/decisions.md`. Plan §18 (in T-52) states each as an amendment.

| Decision | Answer | Applied in |
| --- | --- | --- |
| 215 | Probe grid: MQAR at 4, 16, 64 pairs, 8 models, 10 seeds (240 runs) decides Gate 1; probes 2 to 5 are an optional second block; a pilot fixes width, steps and rate | T-51 |
| 216 | The positive control fails when L6 is better than L8 | T-47 |
| 217 | GPU utilisation is an `nvidia-smi` side log; time-to-loss-X is not built | T-51 |
| 218 | The timing pass stays at about 8,800 steps, 3 GPU-h | runbook text |
| 219 | Sweeps run on seed 21, a key of `configs/prereg.yaml` | T-52 |
| 220 | `configs/commitments.yaml` holds the seed count and the ladder cut; no comparison without the committed seed count | T-47 |
| 221 | Thresholds 0.6 and 0.05 confirmed | T-52 |
| 222, 228 | Decisions 41, 42, 53, 86, 177, 178, 184, 185 and every listed reading of the plan confirmed as implemented | T-52 (plan §18.3) |
| 223 | A ladder slope that is not positive gives 8 seeds, no multipliers | T-47 |
| 224 | Three shared-rate seeds at 100M for an Arm A variant on its own rate; tuned minus shared reported | T-50, T-51 |
| 225 | Benchmark: three repetitions, median and range | T-49 |
| 226 | A failed SWA timing check marks SWA and keeps the pass | T-48 |
| 227 | Smoke runs at 3e-4 and 4.8e-3 | T-51 |
| 229 | The rule in prose in `README.md`, amendments in the plan, few keys in `prereg.yaml` | T-52 |
| 230 | Arm C cut; Phase 6 deferred to its own build | T-51 |
| 231 | Coverage test of the multiplier interval | T-52 |

## Session-5 review (2026-11-14)

A fresh agent on the full model, which wrote none of the code, reviewed the branch of T-52 at `864f132`: the fixes of the second review (T-39 to T-45) and the session-5 changes (T-47 to T-52). It ran the `cpu` tier (3,797 passed), made 106 deliberate one-line breaks (98 caught; 4 of the 8 misses are equivalent changes), ran stop, resume, branch and prune through the command lines on CPU (the losses equal an uninterrupted run's, and no branch point was rewritten), ran the timing pass, the benchmark and both reports, and applied each edit the runbook asks Phil to commit.

| # | Finding | Status |
| --- | --- | --- |
| H1 | The `cpu` tier pinned the unfilled state of `configs/commitments.yaml`, `configs/recipe.yaml` and the data hashes: each commit the runbook asks for failed 5 to 11 tests. | **Fixed in T-54** (#62). Under each of the three edits the tier is green (3,794 passed each). |
| G1 | `--commitments FILE` gave verdicts from any file. | **Fixed in T-54** (#62): the flag is gone; both scripts read the checkout's file. An uncommitted edit of that file in the working tree is still read like a committed one: build from a clean checkout (runbook §3.4). |
| G2 | Without a committed seed count only the tests waited; the variants' 300M losses were printed and drawn. | **Fixed in T-55** (#63), as a reading **for Phil to confirm** (decision 240): the headline table, `rqw.md`, `context.md` and plots 2, 3, 6 and 7 withhold them until the count is committed. |
| G3 | Plot 6 drew the variants' gaps whatever the gate said. | **Fixed in T-55** (#63). |
| G4 | The runbook required a `t4` test to pass that decision 226 expects to fail for the chunked path. | **Fixed in T-57**: §1.1 and §1.2 name the three hardware test cases that may fail for a reason the plan gives, and what to record. SWA trains on `flex` if FlexAttention runs on the T4 and agrees with the reference, else on `chunked` (plan §2.3; decision 244). Whether the `chunked` timing case stays an assertion is **for Phil** (decision 242). |
| G5 | The final checkpoints pass 20 GB in week 3; `prune --final` was all or nothing. | **Fixed in T-56** (#61) and the runbook: `--final-short`. |
| M1 | The headline's decode cell and plots 2 and 8 showed the median alone and accepted documents `timing_report` refuses. | **Fixed in T-55** (#63): one check for all readers; the headline cell has the range. Plots 2 and 8 draw medians and point to `inference.md` for the ranges; since T-59 they draw the ranges as whiskers. |
| M2 | A declared ladder cut did not reach the reported fit. | **Fixed in T-55** (#63). |
| M3 | Plot 1 drew a multiplier axis when no multiplier is reported. | **Fixed in T-55** (#63). |
| M4 | The runbook said block 2 is 320 runs; it is 400. | **Fixed in T-57**; plan §18 A12 in T-52. |
| M5 | "By seed" for the shared-rate interval holds only under the paired test. | **Fixed** (wording): the caption in T-55, the README and plan §18 in T-52. |
| M6 | Literal tolerances in two new test files. | **Fixed** in T-54 and T-52. |
| M7 | A run without a git checkout trained to the end and then refused. | **Fixed in T-56** (#61). |
| M8 | The SWA mark is on the ms/step and ratio cells of `timing.md`, not on the attention share or the `ref` row; the note says "on this GPU" on CPU too. | **Listed, not fixed.** The note under the table names both attention times. |
| M9 | The shim and the runbook wrote to different directories. | **Fixed in T-56** (#61). |
| M10 | This file was stale. | Rewritten here. |
| M11 | Test gaps B11, B23, B64, B77. | B23, B64 (T-54) and B77 (T-56) have tests. **B11 is listed**: the positive control's verdict differs only when p equals alpha exactly, and no test pins that boundary. |

What the review did not cover, by its own account: anything on a GPU or Kaggle; the probe generators and `train_probe.py` beyond running the pilot commands on CPU; most test files line by line; the plots as images; `measure_reproducibility.py`, `evaluate.py`, `graph_diff.py`, `solve_config.py`, `prepare_data.py` (unchanged since the second review); GitHub.

### Confirmed by Phil on 2026-11-14 (decision 245)

Each is implemented as described. Decision 241 gains whiskers on plots 2 and 8 (T-59); decision 244 gains a record of both paths' attention times (T-60).

- **Decision 240**: a variant's 300M results are shown only once the seed count is committed (plan §1.4 says "before any variant result at 300M is read"). Know what still shows without a committed count: `arm_b.md` prints the variants' Arm A 100M branch losses, a ranking and Arm B's gap to MHA, and `shared_rate.md` prints the tuned 100M branch losses, because those two tables are 100M comparisons and do not wait. The branches are the first 1,221 steps of the same 300M trunks, and plot 7 withholds them.
- **Decision 244**: SWA's `opt` path is `flex` if FlexAttention compiles, runs and agrees with the reference on the T4, else `chunked`, with no timing comparison of the two. The plan states this as an order of preference (§2.3). Since T-60 `scripts/swa_paths.py` records both paths' attention times; the record selects nothing (decisions 245, 250).
- **Decision 242**: the `t4` test of SWA's attention time against MHA's stays as written, although decision 226 expects its `chunked` case to fail.
- **Decision 232**: the probe pilot's level (0.99) and the sizes of probe block 2.
- **Decision 241**: plots 2 and 8 draw the benchmark's medians; since T-59 with whiskers from the smallest to the largest repetition (decision 245). The numbers are in `inference.md` and in the headline's decode cell.

### Re-check of the fixes (2026-11-14)

Because the fixes of each earlier review had defects that only the next review found, another fresh agent re-checked T-54 to T-57 on a branch that combines `main`, T-52 and T-57: `docs/review_s5/recheck_s5.md`. Its result: every finding it could reproduce on CPU is fixed (H1, G1, G2, G3, G5, M1, M2, M3, M7, M9); the `cpu` tier is green under each of the four edits of `docs/review_s5/phil_edits.py` and under every runbook edit applied at once (3,860 passed each time); it searched every table and all ten plots for 474 variant 300M values in six withheld states and found none; 63 deliberate breaks, 55 caught; no test was weakened. What it found is applied in T-57 (the runbook's FlexAttention test ids and the rule for SWA's path, the shim notebook as a template, a test for declarations read from the environment, the script's logs) or listed below.

## Session 6 (2026-11-14)

Phil's decision 245 confirmed decisions 232, 240, 241, 242 and 244 and added two things.

| Addition | What is built | Task |
| --- | --- | --- |
| The benchmark's range on the speed plots | Plot 2: one horizontal whisker per variant, from the smallest to the largest repetition of the decode tokens/s, at the height of the seed mean. Plot 8: one vertical whisker at every point. The caption says a whisker shorter than its marker is hidden by it. | T-59 (#66) |
| A record of both SWA paths | `scripts/swa_paths.py` times one forward and backward of the attention module alone for MHA, SWA `chunked` and SWA `flex`: the measurement of the `t4` test of SWA's attention time. `timing_report --swa-paths` prints the three times as the last section of `timing.md`, with the record's own source line and the sentence that it selected nothing. The runbook makes the record in §1.2 and again in §3.1. | T-60 (#67) |

Found while building T-60: on CPU, torch 2.15.0 has no backward for the uncompiled FlexAttention. A CPU run records the flex track with that error. Whether the compiled backward works on sm75 is for the T4.

### The fourth review (`docs/review_s6.md`)

A fresh agent that wrote none of the code reviewed `main` with T-59 and T-60: the `cpu` tier (3,915 passed), 86 deliberate breaks (54 caught; 27 of the 32 misses were real gaps in the tests), both scripts run on CPU with 35 hand-edited records, plots 2 and 8 looked at as images, the runbook read as Phil would.

| # | Finding | Status |
| --- | --- | --- |
| H1 | This file was stale: the count of hardware tests, the sentence "which SWA path is faster there. Then set `opt_path`", two sentences about plots and scripts. | **Fixed** here. |
| G1 | The runbook's rule for SWA's path and the record could disagree: the agreement test runs the forward only, and a training step needs the backward. A `[flex]` timing case that failed because flex was slower read as "does not run". | **Fixed in T-61** as a reading **for Phil to confirm** (decision 250): flex runs when the agreement test passes and the record's flex track has times; a timing case that fails at its assertion changes nothing about the path. |
| G2 | The record might not exist when §3.1 and §5.3 read it; `timing_report` then ended in a traceback. | **Fixed in T-61**: §3.1 makes the record again after the timing pass; a missing or unreadable record is refused in one line; a test ties every `timing_report` command to the path the script writes. |
| G3 | Every flex failure was recorded as "could not run", with no repetition and no traceback. | **Fixed in T-61**: the error names the repetition, the traceback goes to stderr. Still any exception of the flex track is recorded and the script exits 0: listed below. |
| G4 | `check_document` accepted records that contradict themselves. | **Fixed in T-61**: the median, smallest and largest value must be those of the repetition list; the count, the environment, the header fields, the precision and backend of the device, the error text and the set of tracks are checked. |
| G5 | Gaps in the tests: a line naming the faster path, the synchronisation and the autocast, the defaults, the model that is timed, what the record says was run, a missing FlexAttention, a refusal test that could not fail, a mirrored or invisible whisker, the runbook's sentences. | **Fixed in T-61**: each has a test, and the orchestrator's break for each is caught. |
| M1 | The "ratio to MHA" column is a computed comparison nobody asked for by name. | **Kept** (decision 251): it is the column the timing table has, on unrounded medians; a test refuses any word that ranks the paths. Phil may drop it. |
| M2 | An error of several lines went into `timing.md` raw. | **Fixed in T-61**. |
| M3 | A `--synthetic` record printed like a real one. | **Fixed in T-61**: the record carries `synthetic` and the source line says the numbers mean nothing. |
| M4 | The section did not say which path the timing table's `swa256` rows used. | **Fixed in T-61** by a caption; the timing-pass document itself still does not record the path: listed below. |
| M5 | Differences from the `t4` test: the script enforces the SDPA backend, the test enforces none; the module is rebuilt each repetition; whether flex ran compiled is not recorded. | **Listed, not fixed.** The caption names the enforced backend. |
| M6 | Plot 2's label was struck through by its whisker. Plot 8's whiskers are hidden at a 2% range, and `opt` and `ref` whiskers look alike. | Plot 2 **fixed in T-61**. Plot 8 **listed**; the caption says a short whisker is hidden. |
| M7 | The non-JSON message lacked the file name; one runbook sentence about the failure message. | **Fixed in T-61**. |

Found by the orchestrator while fixing: with `opt_path: flex` committed and FlexAttention missing, the script could not build the chunked track either. The SWA modules are now built on the chunked path and then switched (decision 251).

What the review did not cover, by its own account: anything on CUDA (fp16, synchronisation, the enforced backend, compiled FlexAttention); timing of the real model; Kaggle storage; the other eight plots and the other tables.

## Earlier reviews

- **Session 3 audit** (commit `6697a42`): findings F1 to F12. Fixed in T-37, except those under "Listed, not fixed" below.
- **Session 4 review** (commit `384e8b3`, `docs/review_s4.md`): findings H1, H2, G1 to G8, M1 to M14. Fixed in T-39 to T-45, decided by Phil in session 5, or listed below.

The per-finding tables of both are in this file's history at commit `c1ffb78`.

## Listed, not fixed

- **The fixes of T-61 were not re-checked by a fresh agent.** The orchestrator ran 40 deliberate breaks against them (39 caught; all 40 after one test was strengthened) and the tier under `opt_path: flex`. The earlier reviews each found defects in the fixes of the one before.
- **Any exception of the flex track is recorded as its result**, a harness bug included; the script exits 0. The traceback is printed and the error names the repetition (review S6, G3).
- **`scripts/swa_paths.py` enforces the SDPA backend; the `t4` test of the same measurement enforces none.** The record does not say whether FlexAttention ran compiled (review S6, M5).
- **The timing-pass document does not record SWA's `opt_path`.** The manifests of the runs do (review S6, M4).
- **Plot 8's whiskers** are hidden by the markers when the range is a few percent, and the `opt` and `ref` whiskers of one variant look alike (review S6, M6).
- **The fp16 autocast inside a CUDA run of `scripts/swa_paths.py`** is checked on CPU only through the function that builds it; the `gpu` test reads the record's label (review S6, S08).
- **`arm_b.md` lists the variants' Arm B losses while the comparison gate is closed** (session-3 F5; the headline table, `shared_rate.md` and plots 1, 2, 3, 6 and 7 withhold theirs). It needs Arm B runs before the controls, which the runbook's order prevents.
- **An uncommitted edit of `configs/commitments.yaml` in the working tree is read like a committed one** (review S5, G1, the rest of it). The runbook says to build the tables from a checkout that has the commit.
- **The SWA mark in `timing.md`** is on the ms/step and ratio cells, not on the attention share or the `ref` row, and its note says "on this GPU" on CPU too (review S5, M8).
- **The positive control at p exactly equal to alpha** is pinned by no test (review S5, B11).
- **Plot 6 says "variant runs are withheld" in its axes when no variant run exists yet** (week 2); its caption line is conditional, the message is not (re-check, 10).
- **Test gaps the re-check names**: the caption sentence of plot 1 for a slope that is not positive; the prune tests have a single run at the primary budget; runbook prose beyond the test ids and run counts.
- **`docs/review_s5/phil_edits.py` is run by hand.** Nothing in CI applies the runbook's edits, and since the unfilled state of the templates is no longer pinned, nothing notices a template committed in a filled state by mistake.
- **The `gpu` memory test of GQA/MQA decode** compares with MHA's whole cache and could miss one expanded layer (F11). The `cpu` shape test is the real guard.
- **A `ref`-mode run records the kernel restriction it ran under** although no fused kernel ran. The manifest's `mode` says `ref`.
- **`gpu_shared` in a manifest is the last session's value** (G2, the rest of it).
- **`bench.py --with-ref`** runs the KV check and the maximum-batch search for the `ref` documents too and labels them `ref` (M4). It costs benchmark time only.
- **`lr_source` and `timing_pass_id` are free text**, never checked against an existing run or file (M8).
- **A FlexAttention SWA run would record the backend `efficient`**, a kernel it never called (M9). To settle on the T4.
- **The `gpu` and `t4` reproducibility tests keep nothing**; the record is the runbook step `scripts/measure_reproducibility.py` (M13).
- **The benchmark writes its files after the last repetition**, so an out-of-memory error in any track loses the whole run. The runbook has a one-repetition trial first.
- **Not built**: a run script for Phase 6 (deferred, decision 230); the Arm C report (Arm C is cut).

## Deviations from the plan

Plan §18 (in T-52) is the list: A1 to A19, with the reason and the decision row of each. The ones that change a number or drop something the plan asks for:

- fp32 `ref`/`opt` agreement is 1e-5, not 1e-6, and the tests also run in float64 at 1e-12 (A2).
- The seed-count rule and the reported bound use the exact power of Welch's test, 2.63 and 1.83 × σ_init, not 2.5 and 1.8 (A3).
- The SWA timing assertion marks instead of failing (A15).
- Time-to-loss-X is not computed, and GPU utilisation is a side log, not a per-run record (A17).
- Arm C is cut (A18). Phase 6 has no run script in this build (A19).
- The probe grid is 240 runs for Gate 1, not the plan's full grid (A12).

## Hardware tests written, never run

These skip without the hardware. A skip is not a pass. List them with `pytest --collect-only -q -m "gpu or t4"`.

| File | `gpu` | `t4` | What they check |
| --- | --- | --- | --- |
| `test_backend_enforcement.py` | 7 | 7 | One training step per variant under the enforced kernel. |
| `test_causality.py` | 14 | | Causal leak under fp16, every variant and mode. |
| `test_checkpoint.py` | 9 | | GradScaler state after a resume, resume fidelity in fp16, a checkpoint on CUDA. |
| `test_determinism.py` | 14 | 7 | `ref` bit-identical over 100 steps in fp16 (needs `CUBLAS_WORKSPACE_CONFIG=:4096:8`); the `opt` reproducibility measurement, which applies no threshold. |
| `test_entry_points.py` | 1 | | Cold start and resume on CUDA in fp16; the recorded backend and scale. |
| `test_eval_lm.py` | 1 | | fp16 evaluation against fp32. |
| `test_grouped_decode.py` | 3 | 3 | `opt` decode peak memory and decode time at cache 8k below MHA's. |
| `test_inference_bench.py` | 8 | | Measured against analytic KV bytes for every variant; the benchmark on CUDA with three repetitions. |
| `test_mla_absorbed.py` | 2 | | Absorbed decode against the naive path in fp16. |
| `test_probes.py` | 3 | | A tiny MHA solves held-out 4-pair MQAR in fp16, seeds 1 to 3, 12,750 steps each. |
| `test_reference.py` | 7 | | `ref`/`opt` agreement under fp16. |
| `test_scaffold.py` | 1 | 1 | The tier canaries. |
| `test_sliding_window.py` | 2 | 2 | Both `opt` paths against `ref`, including compiled FlexAttention; SWA attention time below MHA's. |
| `test_swa_paths.py` | 1 | | `scripts/swa_paths.py` on CUDA in fp16 with the enforced backend. |
| `test_timing_pass.py` | 1 | | The timing pass on CUDA. |
| `test_trainer.py` | 5 | | fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusals. |
| `test_wsd_branch.py` | 7 | | Branch fidelity in fp16 with the scaler state carried over. |

**The fp16, GradScaler and backend-enforcement code paths have never executed.** Neither have `scripts/bench.py`, `scripts/timing_pass.py`, `scripts/swa_paths.py` and the CUDA path of `scripts/train_probe.py` on a GPU, the shim notebook with its `nvidia-smi` log, or anything on the real corpus.

## To check on real hardware

- Whether FlexAttention compiles and runs on the T4 (plan §2.3): the agreement test passes, and the flex track of the `scripts.swa_paths` record has times, which needs the compiled backward in fp16 that no test has run. Then set `opt_path` in `configs/variants/swa256.yaml` by the rule of `RUNBOOK.md` §1.2 (decisions 244 and 250) before any SWA run. The times of the two paths do not enter the choice.
- The `t4` test in `test_sliding_window.py` asserts SWA's attention time below MHA's. With the chunked path it may fail for the reason plan §2.3 gives; the timing pass then marks SWA (decision 226). A failure of that one test for that reason is a finding, not a defect.
- Whether the fused kernel copies K and V when given a slice of the cache storage (plan §2.3). If it does, decode needs an explicit matmul, disclosed.
- The sliding-window cache allocates up to twice its 256-token cap (decision 77), and MLA's absorbed matrices are a fixed cost per cache outside the KV bytes per token (decision 147). Disclose both with the benchmark.
- Suspicions from the session-3 audit, not confirmed: the tiny MLA test models have score widths (12, 36) that the memory-efficient kernel on sm75 may not serve in fp16; the `t4` decode-time test at batch 1 may be dominated by launch overhead; one fp16 trainer test asserts an exact overfit after 150 steps.
- Divergence recording was tested with a forced non-finite loss, never with a real diverged run.
- The probe pilot's first guesses for the steps (4,000, 8,000, 16,000) are guesses; the pilot rule corrects them.

## Environment

- PyTorch 2.15.0. CI uses the CPU build on Python 3.12.
- The build container cannot reach download.pytorch.org, so it runs the default PyPI build of 2.15.0 on CPU (Python 3.13). It needs `pip install torch numpy scipy pyyaml tokenizers pyarrow pytest`.

## For Phil

- **Confirm or change decision 250** (the reading of "FlexAttention runs" in `RUNBOOK.md` §1.2) before the `t4` session.
- Protect `main`: pull requests only, the `cpu` check required, and "Require review from Code Owners" enabled, otherwise CODEOWNERS has no effect. `configs/commitments.yaml` is in CODEOWNERS.
- Kaggle's torch, Python and CUDA versions (`RUNBOOK.md` §1.5).
- The shim notebook in the repository stays a template: fill it in on Kaggle, and do not commit a filled-in copy to git (`RUNBOOK.md` §1.4).
- `recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset until the smoke runs (`RUNBOOK.md` §2.2).
- After tokenizing the corpus, paste the two hashes into `configs/data.yaml` (§2.1). Consider adding that file to CODEOWNERS then.
- During the campaign you commit two things by pull request: `seed_count` in `configs/commitments.yaml` after §3.4, and `ladder: l6_l8` only if you cut L10.
- The "ratio to MHA" column of the SWA section in `timing.md` can go if you want only the three times (review S6, M1).

## Next

1. Phil: the verification step, `RUNBOOK.md` §1 (the `gpu` tier on the 1070, the `t4` tier in one Kaggle session with the record of SWA's two paths, the cold start of the shim, pinning the environment). Expect findings: none of the fp16, GradScaler or kernel code has ever run.
2. A build session after the verification step, before week 1: fix what the hardware tests find, pin torch to Kaggle's version, apply Phil's answer on decision 250.
3. Nothing else. That is still before any result exists.

The build loop stops here (plan §17.4).

## Notes for the next session

How the loop has been run:

- Workers are Haiku subagents started with the Agent tool, one task each, in a git worktree under `/home/user/wt/`. They commit locally; the orchestrator reviews, pushes, opens the pull request and squash-merges on green CI. A brief is a file in the scratch directory; the worker is told to read it.
- Review is the orchestrator reading the diff and running deliberate one-line breaks from a JSON spec in a separate detached worktree (the runner is a 30-line script in the scratch directory; rewrite it if the directory is gone: apply one textual replacement, run the named test files with `pytest -x`, restore with `git checkout`). In session 5: 23, 21, 21 and 17 breaks for T-47, T-50, T-49 and T-48, and 8, 14 and 21 for T-54, T-56 and T-55; most reviews found two or three gaps in the tests, fixed by adding tests. A break that passes is not always a gap: check whether the change is equivalent.
- The independent review of session 5 found what the orchestrator's reviews could not: a problem between the tests and the runbook (H1), seen only by doing what the runbook asks and running the tier. `docs/review_s5/phil_edits.py` does that; run it again whenever a test touches `configs/commitments.yaml`, `configs/recipe.yaml` or `configs/data.yaml`.
- For `O+W` tasks the tests are committed before the worker starts and checked against a throwaway implementation that is kept out of the commit.
- Workers still call session tools by mistake now and then (a stray message, an archive call). Each was denied and did nothing. Briefs say not to.
- Run every pytest with `OMP_NUM_THREADS=1`. With one thread the full `cpu` tier takes five to six minutes, also beside two workers.
- Two tasks that pass alone can fail together: merge `main` into a branch and run the full tier before its pull request.
- Tasks that touch the same file are run one after the other. In session 5 that was `scripts/make_tables.py` and `scripts/timing_report.py`.
- Existing analysis tests call the analysis through `tests/fixtures/commitments.py`, which commits the seed count the rule gives for each synthetic campaign. A test of the commitment itself calls `mt.analyse` with explicit commitments. No test may read the unfilled state of a file Phil fills in off the repository: `tests/fixtures/unfilled.py` builds it.
- A change to a frozen file goes into a pull request labelled `needs-phil` that the loop does not merge. Code that depends on it goes into the same pull request.
- `tests/test_runbook.py` parses every runbook command with its script's own parser. Keep it green when a script's flags change.
- The session's worker process restarted several times in session 5, and one worker stopped at the account's 5-hour usage limit and was resumed with a message hours later. Nothing was lost because every branch was pushed and decisions were logged before the work started. Keep doing that.
- Never run `git add -A` in a worktree while a check that edits files temporarily runs there. In session 5 a work-in-progress commit on a task branch captured `opt_path: flex` from such a check; it was restored in the next commit and never reached `main`. Stage files by name, or run the check in its own worktree.
- Session 6: a reviewer's scratch directory must be a new, empty one. The fourth reviewer was given `recheck/`, which still held the session-5 re-check's files, and overwrote a helper script there. Nothing in the repository was affected.
- Session 6: a brief's premise can be wrong. The brief of T-60 said the uncompiled FlexAttention runs on CPU; its backward does not. The worker reported it instead of bending the test; say in every brief that this is what to do.
- Session 6: the check under the runbook's edits (`docs/review_s5/phil_edits.py`, case X4) found a real dependency: with `opt_path: flex` committed, new code built SWA on a path that may be missing. Run the new test files under each edit before the pull request, not only when a test touches the three files named above.
