# STATUS

Last updated: 2026-11-13, end of session 5. REVIEW_PLACEHOLDER_HEADER

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
REVIEW_PLACEHOLDER_DONE

## In progress

REVIEW_PLACEHOLDER_IN_PROGRESS

## Blocked

- T-52 waits for Phil's approval: it changes frozen files. Until it is merged, `main` has no plan §18, the README's rule text is the old one, `select_lr` still takes `--sweep-seed`, and there is no coverage test of the multiplier interval.

## Open questions

- **Kaggle's torch, Python and CUDA versions.** Needed to pin the environment; `RUNBOOK.md` §1.5 has the cell to run. Until then: torch 2.15.0, Python 3.12 in CI.
- **Two proposals of the build loop in `RUNBOOK.md` §2.3 (decision 232)**, Phil's to change before the probe pilot: a size is settled when MHA's held-out token accuracy is at least 0.99 on both pilot seeds; the sizes of the optional block 2 (16 chains or pairs; 1,024 keys and values at width 128 for probe 3; 16 data and 64 blank tokens; length 64).

Everything else that was open at the end of session 4 was answered by Phil on 2026-11-13: decisions 215 to 231.

## Handover checklist (plan §17.4)

| # | Item | State |
| --- | --- | --- |
| 1 | Every task merged, `cpu` tier green in CI | REVIEW_PLACEHOLDER_ITEM1 |
| 2 | Every `gpu` and `t4` test written, skipping cleanly, listed as not run | Met; listed below (105 collected: 85 `gpu`, 20 `t4`). |
| 3 | Headline table and all ten plots generate from fake manifests and blank | Met. `gates.md`, `arm_b.md`, `shared_rate.md`, `context.md`, `rqw.md`, `lr_sweep.md`, `probes.md`, `timing.md` and `inference.md` build blank and from synthetic records too. |
| 4 | `RUNBOOK.md` gives exact commands for verification and each GPU week, with gates | Met since T-51. Every command parses with its script's own parser; none has run on a GPU. |
| 5 | `STATUS.md` lists every decision, open question and deviation | This file, with `docs/decisions.md` rows 1 to REVIEW_PLACEHOLDER_ROWS and plan §18. |
| 6 | A fresh agent reviewed the harness against plan §5 and §13; findings fixed or listed | REVIEW_PLACEHOLDER_ITEM6 |

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

REVIEW_PLACEHOLDER_REVIEW_SECTION

## Earlier reviews

- **Session 3 audit** (commit `6697a42`): findings F1 to F12. Fixed in T-37, except those under "Listed, not fixed" below.
- **Session 4 review** (commit `384e8b3`, `docs/review_s4.md`): findings H1, H2, G1 to G8, M1 to M14. Fixed in T-39 to T-45, decided by Phil in session 5, or listed below.

The per-finding tables of both are in this file's history at commit `c1ffb78`.

## Listed, not fixed

- **Plot 6 and `arm_b.md` show variant values while the comparison gate is closed** (session-3 F5; plot 7, the headline table, plots 1 to 3 and `shared_rate.md` withhold them). It needs variant runs before the controls, which the runbook's order prevents for Arm B.
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
| `test_timing_pass.py` | 1 | | The timing pass on CUDA. |
| `test_trainer.py` | 5 | | fp16 training with unscaled gradient norms, a skipped step with scaler back-off, scaler growth, refusals. |
| `test_wsd_branch.py` | 7 | | Branch fidelity in fp16 with the scaler state carried over. |

**The fp16, GradScaler and backend-enforcement code paths have never executed.** Neither have `scripts/bench.py`, `scripts/timing_pass.py` and the CUDA path of `scripts/train_probe.py` on a GPU, the shim notebook with its `nvidia-smi` log, or anything on the real corpus.

## To check on real hardware

- Whether FlexAttention compiles and runs on the T4 (plan §2.3), and which SWA path is faster there. Then set `opt_path` in `configs/variants/swa256.yaml` before any SWA run.
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

- **Approve and merge pull request #59 (T-52)**, or say what to change in it: plan §18, the README's rule text, `configs/prereg.yaml`, the coverage allowance.
- Protect `main`: pull requests only, the `cpu` check required, and "Require review from Code Owners" enabled, otherwise CODEOWNERS has no effect. `configs/commitments.yaml` is in CODEOWNERS.
- The label `needs-phil` exists now (on #59). Create `question` if you want it.
- `recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset until the smoke runs (`RUNBOOK.md` §2.2).
- After tokenizing the corpus, paste the two hashes into `configs/data.yaml` (§2.1). Consider adding that file to CODEOWNERS then.
- During the campaign you commit two things by pull request: `seed_count` in `configs/commitments.yaml` after §3.4, and `ladder: l6_l8` only if you cut L10.

## Next

REVIEW_PLACEHOLDER_NEXT

## Notes for the next session

How the loop has been run:

- Workers are Haiku subagents started with the Agent tool, one task each, in a git worktree under `/home/user/wt/`. They commit locally; the orchestrator reviews, pushes, opens the pull request and squash-merges on green CI. A brief is a file in the scratch directory; the worker is told to read it.
- Review is the orchestrator reading the diff and running deliberate one-line breaks from a JSON spec in a separate detached worktree (`orch/run_breaks.py` in the scratch directory). In session 5: 23, 21, 21 and 17 breaks for T-47, T-50, T-49 and T-48; each review found two or three gaps in the tests, fixed by adding tests. A break that passes is not always a gap: check whether the change is equivalent.
- For `O+W` tasks the tests are committed before the worker starts and checked against a throwaway implementation that is kept out of the commit.
- Workers still call session tools by mistake now and then (a stray message, an archive call). Each was denied and did nothing. Briefs say not to.
- Run every pytest with `OMP_NUM_THREADS=1`. With one thread the full `cpu` tier takes five to six minutes, also beside two workers.
- Two tasks that pass alone can fail together: merge `main` into a branch and run the full tier before its pull request.
- Tasks that touch the same file are run one after the other. In session 5 that was `scripts/make_tables.py` and `scripts/timing_report.py`.
- Existing analysis tests call the analysis through `tests/fixtures/commitments.py`, which commits the seed count the rule gives for each synthetic campaign. A test of the commitment itself calls `mt.analyse` with explicit commitments.
- A change to a frozen file goes into a pull request labelled `needs-phil` that the loop does not merge. Code that depends on it goes into the same pull request.
- `tests/test_runbook.py` parses every runbook command with its script's own parser. Keep it green when a script's flags change.
- The session's worker process restarted several times in session 5. Nothing was lost because every branch was pushed and decisions were logged before the work started. Keep doing that.
