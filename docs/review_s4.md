# Independent review S4: harness at `384e8b3`

Review of 2026-11-13 by a fresh agent on the full model, which wrote none of the code, in a detached worktree at `384e8b3`. It committed, pushed and posted nothing. The scratch scripts it names (`review/e1_until.py` and the like) were session files and are not in the repository; the orchestrator re-ran those for H1, H2, G1, G6 and G8 and got the same results. Each reproduced finding gets a test in tasks T-39 to T-43 (`TASKS.md`); what happened to each finding is in `STATUS.md`.

Evidence labels used below: **reproduced** (a command I ran shows it), **break** (a one-line source break and whether a test caught it), **read** (concluded from the code, not executed), **suspected** (cannot be settled without a GPU or Kaggle).

## 1. Verdict

The CPU-verifiable core is sound: every number I checked matches the plan, 44 of 50 deliberate breaks were caught, and the runbook's whole command sequence ran end to end on CPU with tiny budgets (about 150 runs, every table and plot built). It is not ready to hand over as "done": two analysis gaps can produce a wrong pre-registered decision or a wrong verdict with no error (H1, H2), and both are cheap to close. A further group must be fixed before the first reported GPU hour, because week 1's MHA runs are final results that are never rerun (G1 to G8). `STATUS.md` overstates two checklist items: the runbook contains one false statement and misses steps (item 4), and the deviation list is incomplete (item 5). Nothing on a GPU has been verified by me either.

## 2. Findings

### Blocks handover

**H1. The analysis compares runs trained under different recipes, and issues verdicts on them.**
- Where: `scripts/make_tables.py:220` (`_comparability_problems`), `:792` (`analyse`).
- Plan: §8.7 holds "the whole of `recipe.yaml`" constant; §7 and §13.6 say a recipe change restarts every arm.
- Code: the refusal covers backend, versions, hashes, GPU, mode, precision and steps. It never compares `resolved.recipe`, `resolved.init`, `resolved.precision`, the backbone fields or `resolved.overrides`. Run names carry none of these, so an `--override` run has a clean run's name.
- Established: **reproduced** (`review/e4_checks.py`). I gave every MQA manifest of the synthetic campaign weight decay 0, another `init_scale`, warmup 500, another micro-batch and a non-empty `overrides` block. `analyse` accepted them with no note and returned `detected` for MQA. The same hole lets SWA runs made under two `opt_path` values mix (F3), since `configs/variants/swa256.yaml` is not frozen.
- Related: the headline's `params` and `d_ff` columns come from `configs/`, not from the manifests (in my end-to-end run a 64-wide model was printed as 17,308,032 parameters).
- Fix: refuse when the runs a comparison reads differ in recipe, init, precision, the non-attention model fields or have non-empty overrides; take the static columns from the manifests.

**H2. The seed-count rule is computed and printed from an incomplete ladder.**
- Where: `scripts/make_tables.py:467` (`_ladder_fit`), `:490` (`_rule_fit`), `:868-887`.
- Plan: §1.4, the rule is "applied once, after Arm N and the ladder", and the count is committed before any variant 300M result is read.
- Code: `_rule_fit` needs only a non-empty L6 and uses "every L10 seed present". `RUNBOOK.md` §3.4 builds the report with `--skip-unfinished` and tells Phil to commit `seeds per primary` from it.
- Established: **reproduced** (`review/e2_partial_ladder.py`, `review/e2b.py`). With L6 at 2 of 5 seeds and no L10 the report still shows a decision. Over 240 synthetic campaigns placed near the 5% threshold, the decision differed from the complete ladder's in 10 (L6 partial), 6 (L10 absent) and 1 (L10 at 2 of 3 seeds: 5 instead of 8, noise seed 33, sigma 0.009). No note marks the decision as provisional. The T-37 fix for F2 covers added MHA seeds only.
- Fix: compute the rule only when MHA and L6 have exactly their registered seeds and L10 has its three (or is declared cut); otherwise print the reason.

### Fix before the GPU weeks

**G1. `--until` writes no checkpoint, so the MHA trunks do not resume from 1,221, and the branch-point checkpoint is overwritten after its branch used it.**
- Where: `src/training/checkpoint.py:424-445`; `RUNBOOK.md:274` ("it resumes from step 1,221").
- Plan: §7.2 (shared trunk), §12.3 (keep every branch-point checkpoint until its branch is verified), §13.9.
- Code: a regular checkpoint is due only on the 600 s clock or at the final step. A stop at `--until` saves only `trunk-step*.pt`, which `resume_or_start` never resumes from. The continuation restarts from the last clock checkpoint, re-runs up to ten minutes of steps, and rewrites `trunk-step00001221.pt` unconditionally.
- Established: **reproduced** on CPU (`review/e1_until.py`): after `--until 19` and the branch, the continuation printed "no checkpoint; starting at step 0" and the trunk-step file's hash changed. On a GPU in `opt` mode the re-run steps are not bit-identical, so MHA's 300M result no longer continues the trunk its 100M branch left, and the branch manifest's `parent_checkpoint` names a file with other contents.
- Fix: save a regular checkpoint when stopping at `--until`; never overwrite an existing branch-point checkpoint (compare or refuse).

**G2. A manifest records only the last session's environment, GPU and `gpu_shared`, and a resume across an image change is not refused.**
- Where: `scripts/_run_common.py:661,667`; `src/training/checkpoint.py:122-141, 313`.
- Plan: §0 row 2, §10.6, §12.3: version mismatches must block comparison; MHA's trunks are resumed weeks apart by design.
- Established: **reproduced** (`review/e5_env.py`): session 1 under one torch version, session 2 under another; the run completed and its manifest shows only the second. The `ops/` record holds no environment either.
- Fix: store the environment in each checkpoint and per session in the manifest; refuse a resume when torch, CUDA, driver or GPU differ.

**G3. Plot 7 draws every manifest in the directory in the variant's colour, including sweep runs at rejected learning rates and Arm B runs, and shows variant losses while the gate is shut.**
- Where: `scripts/make_plots.py:809-829`.
- Plan: §8.1 ("computed or viewed"), §10.5 plot 7, §13.15.
- Established: **reproduced** and read from the image (`review/plots/full/07_training_curves.png`, `review/e9_plot7.py`). An MHA sweep point at 4.8e-3 appears as an MHA dot at loss 4.4 beside the Arm A branches; L6, L10 and null dots have no legend entry. In the state the runbook itself creates between §2.7 and §3.4 (variant sweeps done, positive control not yet computed) the plot drew 3 dots for each of the six variants with `comparisons_allowed` False.
- This is worse than `STATUS.md` F5 describes: it says variant runs before the controls are prevented by the runbook's order. The sweeps are such runs.
- Fix: draw only the runs the analysis reads, per arm, and apply the gate.

**G4. Per-run records the plan requires are not written, and cannot be backfilled once checkpoints are pruned.**
- Where: `scripts/_run_common.py:548-562` (the curve keeps `loss` only), `:627-639`; `src/utils/manifest.py:34-58`.
- Plan: §10.2: peak allocated and reserved memory, GPU utilization, context-bucketed loss at every eval.
- Code: none of the three is recorded for a training run. In-run points are the loss on 256 windows; buckets exist only at the final step.
- Established: **read**, plus the manifests of my end-to-end run. Not listed under "Deviations" in `STATUS.md`.
- Fix: add them to the manifest now, or list the omission as a deviation for Phil.

**G5. Measured timing and benchmark quantities have no generated output.**
- Where: `scripts/timing_pass.py:388`, `scripts/bench.py:216`, `scripts/make_tables.py:1661-1665`; no reader elsewhere.
- Plan: §2.3 item 3 (MLA and SWA training timings in both tracks, with the disclosure in the table caption), §3.1 and §8.5 (attention's measured share, peak memory, the ±3% prediction as "the timing table"), §8.6 and §10.4 (prefill tok/s and peak VRAM, maximum batch at 8k, measured KV bytes), §9.1 and §11 (tables built blank before running; nothing typed by hand).
- Code: only `ms_per_step` of the `opt` track and one decode point reach a table. `attention_share`, `peak_bytes`, the `ref` track, the per-repetition spread, `max_batch_at_8k` and the measured KV bytes are written to JSON and read by nothing. The runbook asks Phil to check ±3% by eye in the JSON.
- Established: **read** (grep for the keys in `make_tables.py` and `make_plots.py` finds none). `STATUS.md` lists only the Arm C report as unbuilt.
- Fix: a `timing.md` and an `inference.md` with committed blank forms.

**G6. The runbook's final commands refuse as soon as one diverged run is on record.**
- Where: `scripts/make_tables.py:145-172`; `RUNBOOK.md:388-394`.
- Plan: §7.4 item 5 (record every divergence).
- Established: **reproduced** (`review/e6_diverged.py`): one valid diverged sweep manifest in the directory makes `make_tables` and `make_plots` exit 1 without `--skip-unfinished`, which §5.5 forbids. The only other way out is deleting the record.
- Fix: treat a recorded divergence as its own class, listed and not refused.

**G7. Checkpoints will exceed Kaggle's 20 GB in week 1, and there is no way to prune.**
- Where: `src/training/checkpoint.py:159`; no prune script; no runbook step.
- Plan: §12.3 ("Prune finished runs").
- Estimate: a checkpoint is 17,308,032 × 4 B × 3 ≈ 208 MB. A finished run keeps two, branch points are never pruned. Week 1 leaves about 98 files, about 20 GB. Directories are named by config hash, and a branch's own directory hash is in no manifest.
- Established: **read** plus arithmetic. It fails loudly (disk full), not silently.
- Fix: a `prune` command that removes checkpoints of completed, verified manifests, and a runbook step.

**G8. The plots skip the timing-source refusals the tables apply.**
- Where: `scripts/make_plots.py:132-143, 1090-1096`.
- Plan: §2.2, §11 (backend and version parity before any timing comparison).
- Established: **reproduced** (`review/e4_checks.py`): bench records with `gpu_shared: true`, another driver and another GPU built all ten plots; `make_tables` refused the same records with 19 problems.
- Fix: call `check_timing_sources` in `make_inputs`.

### Minor

- **M1. A disclosure is a typed claim.** `scripts/make_tables.py:1157` prints "every number here is from a T4 (sm75) with the memory-efficient kernel" whatever the manifests say. **Reproduced** with a P100, `math`-backend campaign. Generate it from the manifests or refuse other hardware.
- **M2. A non-finite loss on the last training step blocks the manifest for good.** `scripts/_run_common.py:635` takes `train_loss` from the last record; the validator rejects NaN; every later invocation exits 2. **Reproduced** (`review/e8_lastnan.py`). Needs a forward overflow on exactly the last step, so rare.
- **M3. The positive control ignores direction.** `analysis/primary_test.py:407, 434`: two-sided, so L6 significantly better than L8 would read `detected`. **Read**; in my toy run the gap was negative with p = 0.053.
- **M4. `bench.py --with-ref` repeats the `opt`-mode KV check and maximum-batch search for each `ref` document, unenforced, and labels the result `ref`.** `scripts/bench.py:145, 176`; `src/eval/inference.py:334`. **Read.** Each try fills an 8k cache 8 tokens at a time (1,024 forwards). **Suspected**: the search alone may exceed the 3 GPU-h of plan §12.1.
- **M5. Timing pass.** It opens the dataset with whatever `configs/data.yaml` holds, so unset hashes compare nothing (`scripts/timing_pass.py:457-462`). With `--with-ref` and the share runs it is about 8,800 steps against the plan's 4,620 (**suspected** 3 h, not 1.5).
- **M6. A resume is refused when the number of visible CUDA devices changes** (`src/training/checkpoint.py:308`). The shim does not set `CUDA_VISIBLE_DEVICES`; the runbook's shell commands do. Loud, but it will happen. **Read.**
- **M7. Shim.** It can only call `scripts.train` (no branch, timing pass or bench). `configs/runs/example_mha_s1.yaml` has `branches: [100M]`, while the runbook's MHA runs use `100M 200M`, so the two collide by name with different hashes. Whether `/kaggle/working` survives a commit run is **suspected** not; the runbook §1.4 does make Phil test it.
- **M8. `lr_source` and `timing_pass_id` are free text**, typed from an `export`, never checked against an existing sweep run or timing file (`src/utils/manifest.py:432-433`).
- **M9. A FlexAttention SWA run would record backend `efficient`** (`scripts/_run_common.py:662`), a kernel it never called. The restriction does not apply to FlexAttention. **Read.**
- **M10. `gates.md` prints two "slope at 300M" values that differ once MHA has 8 seeds** (`scripts/make_tables.py:1266` against `:1275`), unexplained.
- **M11. CODEOWNERS covers every frozen path, but not the files that decide what CI runs or skips:** `.github/workflows/ci.yml`, `tests/conftest.py`, `tests/hardware.py`, `pyproject.toml`. CI itself runs what `CLAUDE.md` says. I could not check that CI is green or that `main` is protected (no network).
- **M12. The smoke runs never reach an evaluation.** The first in-run point (step 250) and the final evaluation run at batch 64, not at the micro-batch. **Suspected** OOM risk on the T4; add `--eval-every-steps 50` to one smoke run.
- **M13. The `opt` reproducibility tolerance is never recorded.** Plan §5 and §5.1 want it "measured once and recorded" for the writeup. The `gpu` and `t4` tests build the record, pass it to `json.dumps` and drop it (`tests/test_determinism.py:347-369`), and the runbook has no step for `scripts/measure_reproducibility.py`. **Read.** `STATUS.md` says these tests "measure and record".
- **M14. Test gaps** found by the breaks in section 3: B10, B22, B36, B45, B46, B47.

### Checked and found right

- Frozen configs and `tests/tolerances.py` against the plan, value by value; none changed since its first commit.
- `solve_config --all` and `graph_diff`: 17,308,032 for non-MLA, +3,072 and +1,536 for MLA, 10,621,824 and 17,702,784 for the ladder.
- Steps 763 / 1,526 / 3,052 / 4,578 and branch points 1,221 / 2,442 / 3,662.
- No global RNG and no variant-specific code outside `src/attention/`.
- End to end on CPU, the trunk-plus-branch and the standalone 100M run gave `s_rep` exactly 0, and SWA equalled MHA to four decimals when the context was shorter than the window.
- Cache allocation equals the analytic KV bytes for all variants on CPU.
- T-37 against F1 to F8: each fix does what `STATUS.md` says, with the limits in H2 (F2), G3 (F5), M9 (F3), M1 and G5 (F8), and the untested Arm B scope (F1, break B10).
- Full `cpu` tier: 3,163 passed, 102 deselected, 4 min 3 s.

## 3. Deliberate breaks

50 breaks, 44 caught. Each was restored with `git checkout` and `git status --short` was empty afterwards.

| # | Break | Test file run | Caught |
| --- | --- | --- | --- |
| B01 | Trainer indexes the batch by steps taken | `test_trainer` | yes |
| B02 | Trainer indexes the LR by steps taken | `test_trainer` | yes |
| B03 | Restore does not restore `steps_taken` | `test_checkpoint` | yes |
| B04 | Restore skips the optimizer state | `test_checkpoint` | yes |
| B05 | Branch config without `run.parent` | `test_wsd_branch` | yes |
| B06 | Train order from the global RNG | `test_data` | yes |
| B07 | Init seed ignores the parameter name | `test_init` | yes |
| B08 | Linear decay instead of 1−√ | `test_schedule` | yes |
| B09 | In-loop backend restriction removed (F4) | `test_backend_enforcement` | yes |
| B10 | Arm B runs dropped from the step-parity scope (F1) | `test_audit_fixes`, `test_make_tables`, `test_secondary_reports` | **no** |
| B11 | Seed-count rule fed the all-seed fit (F2) | `test_audit_fixes` | yes |
| B12 | Headline shows variants with the gate shut (F5) | `test_audit_fixes` | yes |
| B13 | Plots 1 to 3 withhold nothing (F5) | `test_plots` | yes |
| B14 | Registered sweep seed accepted (F6) | `test_audit_fixes` | yes |
| B15 | Gate stays open on a detected null control | `test_gate_report` | yes |
| B16 | Driver version not compared | `test_make_tables` | yes |
| B17 | Tie rule sign reversed | `test_lr_sweep` | yes |
| B18 | Holm multiplier dropped | `test_primary_stats` | yes |
| B19 | `evaluate.py` accepts a trunk checkpoint | `test_entry_points` | yes |
| B20 | Validation curve not truncated on resume | `test_entry_points` | yes |
| B21 | Unpinned dataset accepted | `test_entry_points` | yes |
| B22 | `gpu_shared` always recorded False | `test_entry_points` | **no** (CPU refuses `yes`; the one `gpu` test passes `no`, so `yes` is tested nowhere) |
| B23 | Timing warm-up not discarded | `test_timing_pass` | yes |
| B24 | Variant-major order, not round-robin | `test_timing_pass` | yes |
| B25 | RQ-W interpolation not in ln(steps) | `test_rqw_estimate` | yes |
| B26 | Context value ignores the window start | `test_eval_lm` | yes |
| B27 | Late context is c > 512 | `test_eval_lm` | yes |
| B28 | Branch-point checkpoints not written | `test_checkpoint` | yes |
| B29 | Trunk checkpoint not refused as a result | `test_make_tables` | yes |
| B30 | K/V expanded in `opt` decode | `test_grouped_decode` | yes |
| B31 | MLA scale 1/√48 in `ref` | `test_mla` | yes |
| B32 | MLA absorbed scale 1/√48 | `test_mla_absorbed` | yes |
| B33 | MLA factor init dropped | `test_init_logits` | yes |
| B34 | SWA chunk scores 3W keys | `test_sliding_window` | yes |
| B35 | Weight decay on the tied embedding | `test_schedule` | yes |
| B36 | Reruns not limited to registered seeds | `test_make_tables`, `test_audit_fixes`, `test_gate_report` | **no** |
| B37 | Shared-GPU timing source accepted | `test_secondary_reports` | yes |
| B38 | Results not compared with timing sources | `test_secondary_reports` | yes |
| B39 | Paired worst case σ_d = σ_init | `test_ladder_fit` | yes |
| B40 | Data seed ignored by the entry point | `test_entry_points` | yes |
| B41 | `lr_source` dropped from the manifest | `test_entry_points` | yes |
| B42 | Ladder runs named after the variant | `test_manifest` | yes |
| B43 | Paired test with n degrees of freedom | `test_primary_stats` | yes |
| B44 | Test switch inverted | `test_primary_stats` | yes |
| B45 | `skipped_in_first_100` counts step 100 | `test_trainer` | **no** |
| B46 | Seed-count threshold `>=` instead of `>` | `test_ladder_fit` | **no** |
| B47 | In-run validation ignores `eval_curve_windows` | `test_entry_points` | **no** |
| B48 | Unfinished checkpoint evaluated without the flag | `test_entry_points` | yes |
| B49 | No EOS after a document | `test_data` | yes |
| B50 | Expected dataset hash not compared | `test_data` | yes |

No test was weakened in the history: frozen files have one commit each; assertion lines removed in later commits were count updates or replaced by stricter forms; T-37's changes to existing tests are additions. The only literal tolerances in test files are `pytest.approx(0.1536)` at `tests/test_init_logits.py:98` and a colour match at `tests/test_plots.py:1008`.

## 4. Decisions that should be Phil's

Rows of `docs/decisions.md` that change what is measured or how it is judged and are not marked for him.

- **41**: the null control under the paired test pairs seeds 11–15 with MHA seeds 1–5 in list order. It decides how Gate 2 is judged.
- **42**: exploratory intervals use MHA seeds 1–3 when paired, and all 5 or 8 MHA seeds under Welch.
- **53**: the first update runs at learning rate 0, so every run has one update fewer than its step count.
- **86**: the ladder fit weights every run, so L8 (5 or 8 seeds) outweighs L10 (3). It sets the slope behind the seed-count rule and every multiplier. Rows 87, 88 and 92 are listed; this one is not.
- **126**: no primary verdict without a seed-count decision, so a non-positive or missing ladder slope means the primaries are never tested.
- **127**: one learning rate per variant in Arm A; a second raises. It settles the reading of plan §7.4 item 5 (see section 5).
- **145**: benchmark points: prefill at batch 1 only; a decode number is the mean of 10 steps after 3 warm-ups, one repetition, no spread. This is the headline's decode column.
- **150**: "SWA attention time below MHA" is judged as share × ms per step from a separate instrumented run.
- **177, 178**: Arm B and context-use gaps use the Gate 2 test and seed sets (Welch compares 3 variant seeds with 5 or 8 MHA seeds).
- **184**: the 0.5% step-parity refusal is narrowed to the runs the analysis reads.
- **185**: the rule reads the registered-seed fit while the reported fit and multipliers use every seed.
- Lower weight: **25** (RMSNorm epsilon and fp32 normalisation under autocast), **32** (split fill order val, test, train), **122** (RQ-W gap at the variant's own seeds), **140** (Phase 6 depths and distractor counts).

## 5. Plan problems

Items `STATUS.md` already lists (data order, bucket top edge, Welch table rows, smoke-run length, extrapolation below 100M) are left out.

- **§2.3 item 2 against §2.3's own prediction.** SWA's attention time must be below MHA's or the timing run fails, yet the chunked fallback "treat as near zero" at context 1,024. With the default `opt_path: chunked`, `timing_pass.py` will probably exit 1 and call the file "not a valid timing result" while still writing it, and the tables read it anyway.
- **§7.4 item 5.** "Report shared-recipe and tuned results side by side": the plan does not say whether Arm A runs each variant at its tuned rate only. The code allows one rate per variant; the side-by-side is two single-seed sweep losses at 100M. If a variant's tuned rate differs from MHA's, the primary comparison is tuned against tuned.
- **§7.3 against §15.** The scaler settings are frozen from smoke runs made before any learning rate is known (placeholder 1e-3), although skips in the first 100 steps depend on the peak rate through the warmup. The runbook only says to look again later.
- **§10.6 "selected SDPA backend".** PyTorch does not report the kernel it selected; the harness records the one it enforced. For a FlexAttention run no SDPA kernel is called at all.
- **§8.4.** The positive control is two-sided; the plan does not say that a detected effect in the wrong direction fails it.
- **§12.3.** "Resume automatically" from `/kaggle/working` assumes storage that survives a commit run; "20 GB holds ~100 checkpoints" is below what week 1 alone writes (G7).
- **§10.2.** "GPU utilization" and "time-to-loss-X" are required per run with no definition (which X, sampled how).

## 6. Coverage

**Read in full:** `docs/plan_v5.md`, `CLAUDE.md`, `TASKS.md`, `STATUS.md`, `RUNBOOK.md`, `README.md`, all 188 rows of `docs/decisions.md`, every file under `configs/`, `.github/`, the shim notebook, `pyproject.toml`. `scripts/`: `_run_common`, `train`, `branch`, `evaluate`, `select_lr`, `prepare_data`, `bench`, `timing_pass`, `make_plots`, `make_tables`. `src/`: `config`, `model`, `transformer`, `utils/manifest`, `utils/seeding`, `training/*`, `data/*`, `eval/lm`, `eval/positions`, `eval/inference`, `attention/base`, `grouped`, `mla`, `sliding_window`, `token_cache`. `analysis/*`. `tests/tolerances.py`, `conftest.py`, `hardware.py`, `test_runbook.py`, `test_backend_enforcement.py`.

**Run:** the full `cpu` tier once; 50 breaks; `solve_config --all`; `graph_diff`; the runbook sequence §2.4 to §5.5 on CPU with a 64-wide model and budgets of 20/40/60 steps (`review/e2e.py`, `review/e2e_stage2.py`); ten plots blank, from the synthetic campaign and gate-shut, of which I looked at plots 1, 2, 4, 5, 6, 7, 8 and blank plot 1.

**Sampled:** `tests/test_audit_fixes.py`, `test_entry_points.py`, `test_grouped_decode.py`, the fixtures, `scripts/measure_reproducibility.py`, `scripts/solve_config.py`, the T-37 diff to existing tests, and the test diffs of commits that removed assertion lines. T-32 to T-36 were reviewed as the current code, not commit by commit.

**Not looked at:** `scripts/graph_diff.py` and `src/utils/flops.py` beyond running them, `src/eval/retrieval.py`, `src/attention/recurrence.py`, `src/utils/reproducibility.py` beyond a skim, most other test files line by line, plots 3, 9 and 10 visually, anything of T-21, GitHub (CI status, branch protection, issues), and anything on a GPU. Every statement above about fp16, GradScaler, kernels, memory or timing on a T4 is read or suspected, never executed.
