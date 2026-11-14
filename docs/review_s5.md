# Independent review S5: harness at `864f132` (branch `task/T-52-frozen`)

Review of 2026-11-13/14 by a fresh agent that wrote none of the code, in a detached worktree of `task/T-52-frozen` (everything on `main` plus the frozen-file changes of pull request 59). Scope: the changes since `384e8b3`, that is T-39 to T-45 (the fixes for `docs/review_s4.md`) and T-47 to T-52 (decisions 215 to 236, plan §18). It committed, pushed and posted nothing, and changed no file outside its scratch directory and its own worktree, which was clean after every break (`git status --short` empty) and is removed.

Evidence labels: **reproduced** (a command I ran shows it), **break** (a one-line source break and whether a test caught it), **read** (concluded from the code or text, not executed), **suspected** (cannot be settled without a GPU or Kaggle).

Scratch directory, called `S` below: `/tmp/claude-0/-home-user-attention-benchmark/b1914e3e-0c3b-517e-b90e-f6c7b7b248ed/scratchpad/review_s5`. The scripts named below are in it; the ones that import the project run from `S/exp`, a clean export of the reviewed commit (`cd S/exp && PYTHONPATH=. OMP_NUM_THREADS=1 python S/<script>`).

Full `cpu` tier at the start: `3797 passed, 105 deselected in 342.60s (0:05:42)`.

## 1. Verdict

The fixes for the session-4 review do what `STATUS.md` says, and so do the session-5 changes: 98 of 106 deliberate breaks were caught, four of the eight misses are equivalent changes, and every end-to-end reproduction on CPU behaved as documented (stop, resume, branch and prune through the command lines gave the same losses as an uninterrupted run and never rewrote a branch point; the timing pass, the benchmark and both reports ran; the tables followed four different commitments files correctly). Every number I checked in the changed code matches the plan and the decisions; the runbook has one wrong count (M4). The harness is not ready to hand over as it stands, for one reason that is cheap to fix now and expensive later: the `cpu` tier pins the files Phil must fill in. Each of the runbook's own commits (the data hashes of §2.1, the recipe freeze of §2.2, the seed count or the ladder cut of §3.4) turns the required CI check red, 5 to 11 tests each (H1), and by then the loop that could repair the tests has stopped. The first of these commits is inside the verification step (§1.4 needs the hashes). Before the GPU weeks, five more things should be settled: the `--commitments FILE` flag undoes the reason for decision 220 (G1); an uncommitted seed count withholds the tests but not the variant losses (G2); plot 6 still shows what `context.md` withholds (G3); runbook §1.2 demands a `t4` pass that decision 226 expects to fail (G4); and the kept final checkpoints pass 20 GB in week 3 with no selective way to remove them (G5). Nothing on a GPU was verified by me either.

## 2. Findings

### Blocks the handover

**H1. The `cpu` tier turns red at every commit the runbook asks Phil to make.**
- Where: `tests/test_session5_rule.py:250` (`test_the_committed_file_declares_nothing_yet`), `:420` (its last assertion reads the repository's file), `:448`, `:472`; `tests/test_gate_report.py::test_main_blank_writes_both_tables`, `tests/test_secondary_reports.py::test_main_blank_writes_all_five_files_equal_to_the_committed_ones`, `tests/test_make_tables.py::test_main_blank_matches_the_committed_file` (the blank forms are generated with the repository's `configs/commitments.yaml`); `tests/test_config.py:309-311` and four more tests there, `tests/test_data.py::test_data_config_holds_plan_values`, `tests/test_entry_points.py` (two), `tests/test_probes.py`, `tests/test_timing_pass.py`, `tests/test_trainer.py` (one each).
- Runbook: §2.1 "Paste them into `configs/data.yaml` ... and commit"; §2.2 "write the three values into `configs/recipe.yaml` (a pull request you approve)"; §3.4 "commit that number as `seed_count` in `configs/commitments.yaml` by pull request"; §1.5 "see the `cpu` tier pass". `CLAUDE.md`: red CI is never merged, and a test is never weakened to make it pass.
- Established: **reproduced** with `S/phil_edits.py`, which applies the edit in the worktree, runs the whole tier and restores the file:
  - `seed_count: 5`: `6 failed, 3791 passed`.
  - `ladder: l6_l8`: `5 failed, 3792 passed`.
  - `micro_batch: 32`, `init_scale: 65536.0`, `growth_interval: 2000` and both hashes set: `11 failed, 3786 passed` (two of them for the hashes alone: `test_data_config_holds_plan_values` and `test_the_repository_data_config_refuses_train_branch_and_evaluate_and_writes_nothing`).
  The failing names are in `S/phil_X1.log`, `S/phil_X2.log`, `S/phil_X3.log`.
- Why it matters: the tests assert the unfilled state of the repository instead of building it. At the gate where the seed count is committed, the only ways forward are to merge red or to edit tests, in the same pull request as the declaration and with results on the table. The commitments file of session 5 added six of these failures (four new tests, and two older blank-form tests that now read it); the recipe and data ones are older, and neither earlier review ran the runbook's edits against the suite.
- Fix: each such test builds its own unfilled recipe, data config or commitments file (the `--commitments`, `--data-config` and override paths exist), and the blank forms are generated with an explicit uncommitted declaration. Then run the three edits above against the tier once more.

### Fix before the GPU weeks

**G1. `--commitments FILE` gives primary verdicts from a file nobody committed, and the tables call it `configs/commitments.yaml`.**
- Where: `scripts/make_tables.py:2149-2152, 2166`; `scripts/make_plots.py:1156-1162`; the heading `DECLARATIONS = "Declarations (configs/commitments.yaml)"` at `scripts/make_tables.py:1593` and the two notes at `:1163-1171`.
- Decision 220: "A flag can be set after results exist; a committed file has a timestamp before the runs it governs." `tests/test_runbook.py:277` says the same about `--l10-cut`.
- Code: the flag that replaced `--l10-cut` takes any path. Nothing in the output says which file was read or whether it is the repository's, and an uncommitted edit of `configs/commitments.yaml` in the working tree is read the same way.
- Established: **reproduced** (`S/e4.py`). `python -m scripts.make_tables --manifests S/e4/manifests --commitments S/e4/commit_five.yaml ...` printed `detected` for MQA and two `not detected` verdicts; `S/e4/tables_five/gates.md` shows "## Declarations (configs/commitments.yaml) ... committed seed count | 5" although the repository's file holds `null`. The same directory with `commit_cut5.yaml` (`ladder: l6_l8`) gives verdicts too.
- Fix: drop the flag from the two command lines (tests can pass `commitments=` to `analyse` and `make_inputs`, as most already do), or print the path and the git state of the file read in `gates.md` and refuse a path outside the repository unless a test-only switch is given.

**G2. With the gate open and no committed seed count, the variants' 300M losses are printed and drawn; only the tests wait.**
- Where: `scripts/make_tables.py:1439, 1447` (`withheld` depends on the gate alone), `:2053` (`_rqw_report`); `scripts/make_plots.py:314-316` (`_withheld`), used by plots 2, 3 and 7.
- Plan §1.4 and §18 A5, README: the seed count is committed "before any variant result at 300M is read".
- Established: **reproduced** (`S/e1.py`, `S/e4.py`). With `seed_count: null`, and also with a count that disagrees with the rule's, `headline.md` lists every variant's loss per seed (`gqa4 | s1 3.2016, s2 3.1986, ...`), `rqw.md` is byte-identical to the committed case and prints "gap, estimate minus MHA", plot 3 draws 29 points, plot 7 draws 54 runs and plot 6 draws 96 gaps. The paired differences can be read off the table by eye.
- The runbook's order (no primary run before the commit) prevents this, and the count is computed from MHA and the ladder alone. What remains open is the ladder declaration: `full` against `l6_l8` can change the count (review S4, H2), and it could be chosen with the losses in view. This is a decision for Phil: either the missing commitment withholds like a shut gate, or the plan's sentence is accepted as a rule for the reader only.

**G3. Plot 6 draws every variant's per-bucket gap to MHA whatever the gate and the commitment say.**
- Where: `scripts/make_plots.py:753-803` (`plot_context_gap`); no call of `_withheld`, no look at the commitment.
- Established: **reproduced** (`S/e1.py`): without the null control (`comparisons_allowed` False) and with no committed seed count, `context.md` is blank and `06_context_gap.png` holds 96 gap points. It also draws seeds outside the registered sets (**read**).
- Status: `STATUS.md` lists this since session 3 (F5, "plots 6 and 7 ... listed, not fixed"). T-45 fixed plot 7 and its pull request is titled "plots refuse what the tables refuse"; plot 6 was left, and T-47 added a second state in which the table waits and the plot does not.
- Fix: the two lines plot 7 got.

**G4. Runbook §1.2 requires a `t4` pass that decision 226 expects to fail, and says nothing about what to do then.**
- Where: `RUNBOOK.md:85` ("every test passes on the T4, in particular ... SWA attention time below MHA's"); `tests/test_sliding_window.py:418-440` (`test_swa_attention_time_at_ctx_1024_is_below_mha[chunked]` and `[flex]`); `RUNBOOK.md` §3.1 ("this is a likely outcome").
- Decision 226 and plan §18 A15 changed the timing pass because plan §2.3 predicts savings near zero for the chunked path. The `t4` test asserts the same inequality on the attention module alone, and its own docstring says a failure for `chunked` is "a result to report, not a test to loosen". The `flex` case asserts that FlexAttention compiles, which the runbook's same sentence treats as open.
- Established: **read**. The outcome is **suspected**.
- Fix: one sentence in §1.2 naming the two test cases that may fail on the T4, what to record and that the session continues. Whether the test itself stays as written is Phil's (it is a plan §5 test).

**G5. The final checkpoints that the runbook says to keep pass Kaggle's 20 GB in week 3, and `prune --final` is all or nothing.**
- Where: `RUNBOOK.md` §0 "Disk"; `scripts/prune.py:145-148`.
- Arithmetic (**read**, with the 208 MB per checkpoint of review S4 and the runbook's run list for 5 seeds): one final checkpoint per completed run and branch gives 44 files after week 1, 78 after week 2, 108 after week 3 (22.5 GB) and up to 153 after week 4. On CPU (`S/e3.py`, **reproduced**) `--final` removed the final checkpoint of every run and branch at once; there is no way to drop only those nothing reads again (sweep points, null and data-order runs, reruns, 100M branches).
- It fails loudly (disk full), in the middle of the primaries.
- Fix: a selector for `--final` (by budget or by name pattern), or a runbook line that says when `--final` is safe and what is lost.

### Minor

- **M1. The benchmark's spread reaches `inference.md` only.** `headline.md`'s decode column (`scripts/bench.py:353-372`, `scripts/make_tables.py:2121-2136`) and plots 2 and 8 (`scripts/make_plots.py:132-143`) read the median alone, check neither the repetition count nor that the median belongs to its list, and accept documents that `timing_report` refuses. **Reproduced** (`S/e7.py`): one-repetition documents, a mix of 1 and 3 repetitions, and a document whose `tok_per_s` is three times its largest repetition all built `headline.md` and the plots with exit 0 (the last printed 3672.0 for MQA); `timing_report` refused the last two. Decision 225 asks for the median "with the smallest and largest value"; the headline cell has no range.
- **M2. A declared ladder cut does not reach the reported fit.** `scripts/make_tables.py:662` reads `mha_L10` "when present" for the fit behind the multipliers, the "ladder slope b by budget" rows and plots 1 and 5, also under `ladder: l6_l8`. **Reproduced** (`S/e5.py`): with the cut declared and one L10 seed on disk, the rule's slope is 0.34888 and the reported 300M slope 0.35911, under the note "mha_L10 is not read, by declared cut". `SLOPE_NOTE` explains a difference only "once MHA has eight seeds".
- **M3. Plot 1 draws a multiplier axis when the tables report no multiplier.** `scripts/make_plots.py:310-311, 412-416, 459-460` key the axis on the all-seed fit; decision 223 keys "no multiplier" on the rule's fit. **Reproduced** (`S/e8.py`, image `S/e8_plot1.png`): fake campaign seed 35 with slope 0 and 8 seeds has a rule slope of -0.00036 and an all-seed slope of +0.0016; `headline.md` has no multiplier and the note of decision 223, while the plot's caption says "Top axis: effective-parameter multiplier exp(−gap / b), b = 0.0016" over ticks of 1e11. Rare (it needs the case of decision 223 first).
- **M4. Runbook numbers and pointers.** §2.3 says block 2 is 320 runs; its loop holds five `train_probe` commands per model and seed, 400 runs (probe 5 of plan §9.2 is two tasks; `tests/test_runbook.py` expects the five). §5.3 says "(the command below)" for the trial command printed above it. **Read.**
- **M5. "By seed" holds only under the paired test.** `scripts/make_tables.py:948-955` computes tuned minus shared with the Gate 2 test; under Welch the interval is that of two independent groups of three, although both sides share init seeds 1 to 3 and the caption (`:1874-1880`), the README and decision 224 say "by seed". **Read**; see section 4.
- **M6. Literal tolerances in new tests**, against `CLAUDE.md` ("No literal tolerances in test files"): `tests/test_shared_rate.py:174, 182` (`abs=1e-9`; there was none at `384e8b3`), and the bounds `0.01`, `0.8` and `3.0` in `tests/test_multiplier_coverage.py:85, 103`. **Read.**
- **M7. A run without a git checkout trains to its last step and then refuses to write its manifest.** `scripts/_run_common.py:772` reads the commit hash only when the manifest is written (`src/utils/manifest.py:269-285`). **Reproduced** by accident: my export had no `.git`, and `scripts.train` ran all steps and exited 2 with "git rev-parse HEAD failed"; every rerun resumed from the last checkpoint and refused again. Older than this review's scope. Read it before the first step.
- **M8. The SWA mark sits on two cells.** In `timing.md` the `swa256` row's ms/step and ratio carry †; its attention share, which is the number the check is made of, and the `swa256/ref` row do not (`scripts/timing_report.py:399-403`). The note under the table names both attention times, so nothing is hidden. On CPU the note still says "on this GPU". **Reproduced** (`S/e2/tables/timing.md`).
- **M9. The shim and the runbook write to different places.** The notebook passes `--out /kaggle/working` and logs to `/kaggle/working/logs`; the runbook's `OUT` is `/kaggle/working/out`. A run started by one is not resumed by the other. The notebook's `log.terminate()` is skipped when the run exits non-zero (`check=True` raises first). **Read.**
- **M10. `STATUS.md` at this commit still describes session 4** below its first table (§2.3 "not written", 3,622 tests, checklist item 4 "not met"). It says it will be rewritten when T-53 closes.
- **M11. Test gaps** from section 3: B11, B23, B64, B77.

### Checked and found right

- T-39: every refusal named in H1, H2 and G6 of the last review is in place and tested (B01 to B08, B94 to B96, B106). The rule waits for MHA 1 to 5, L6 1 to 5 and L10 1 to 3, and reads only those.
- T-40 and T-43, end to end through the command lines (`S/e3.py`): `--until 10`, `--until 19`, branch, prune (nothing of the unfinished trunk is touched), continue, prune, `--final`. The continuation printed "resumed from step00000019.pt at step 19", the hash of `trunk-step00000019.pt` never changed, and the trunk's and the branch's validation losses equal those of an uninterrupted run exactly. The branch point was deleted only after the trunk and its branch were complete.
- T-41: the six keys of every validation point and `peak_memory` are in the manifests of that run.
- T-42, T-48, T-49: `scripts.bench --synthetic`, `scripts.timing_pass --synthetic --with-ref` and `scripts.timing_report` ran; the median, the smallest and the largest value of each point are those of its repetition list; the pass exits 0 when SWA is not below MHA and prints the note; the mark † and the note appear in `timing.md`, `headline.md` and `rqw.md` and nowhere when the check holds. No plot prints a training timing.
- T-47: no verdict without a commitment, none with a wrong one in either direction, none with a commitment and no rule output; `l6_l8` is read; a slope that is not positive gives 8 seeds, a note and no multiplier in the tables; the positive control reads `detected in the wrong direction` only for a significant negative gap (`S/e1.py`, `S/e4.py`, B09 to B22, B97, B98, B100, B104).
- T-50: sweep points on seed 21, runs of a variant on MHA's rate, ladder runs and Arm B runs are not read as shared-rate runs; shared-rate runs are inside the steps-parity check; `select_lr`, the wall-clock estimate and plots 6 and 7 do not read them (B24 to B32, `S/e4.py`).
- T-52: `select_lr` has no seed flag and reads 21 from `configs/prereg.yaml`; a registered seed there is refused. The coverage test fails when the slope's uncertainty is dropped (B91) and when the shared term is dropped (B90).
- T-51: the pilot commands of §2.3 ran on CPU with 60 steps, and `probes.md` prints the `min` column the pilot rule needs; `make_plots --probes DIR` reads the record directory.
- Numbers: sweep seed 21; null seeds 11 to 15; pilot seeds 101 and 102; smoke rates 3e-4 and 4.8e-3, 14 runs, `--until 100`; steps 1,526 / 3,052 / 4,578 and branch points 1,221 / 2,442; 240 runs in block 1; 2.63 and 1.83 for Welch (computed: 2.6298, 1.8318); thresholds 0.6 and 0.05; three repetitions; 5% for KV bytes; 2,500 simulations, seed 20261113, three standard errors.
- Frozen files: `docs/plan_v5.md` gained 62 lines and lost none; `tests/tolerances.py` gained the three coverage constants and nothing else; no tolerance was loosened; no `skip` or `xfail` was added (`tests/test_scaffold.py` forbids them). The README's rule matches `primary_test.py`, `ladder_fit.py` and `make_tables.py` except for M5.

## 3. Deliberate breaks

106 breaks, 98 caught. Of the 8 not caught, 4 are equivalent changes (B30, B69, B78, B101) and 4 are test gaps (B11, B23, B64, B77). Each was a single replacement applied by `S/breaks.py` from `S/specs1.json` to `S/specs4.json`, run with `pytest -x` on the named files, and restored with `git checkout`; results are in `S/breaks_results.jsonl`.

| # | Task | Break | Test files run | Caught by |
| --- | --- | --- | --- | --- |
| B01 | T-39 | recipe block no longer compared across runs (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report | `test_a_run_trained_under_another_setting_is_refused_and_named[recipe.optimizer.weight_decay]` |
| B02 | T-39 | a run made with --override is accepted (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report | `test_a_run_made_with_an_override_is_refused_even_when_the_recipe_agrees` |
| B03 | T-39 | the rule no longer waits for L10 (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report, test_session5_rule | `test_without_all_of_l10_there_is_no_decision[absent]` |
| B04 | T-39 | the rule fit reads every seed present, not the registered ones (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report, test_audit_fixes | `test_an_unregistered_l10_seed_does_not_move_the_rule` |
| B05 | T-39/G6 | a recorded divergence is refused again (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report, test_divergence | `test_a_diverged_run_is_left_out_and_listed_without_the_skip_flag` |
| B06 | T-39 | model difference inside one variant and arm accepted (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report | `test_runs_of_one_variant_must_have_one_model[d_ff of one gqa4 run]` |
| B07 | T-39/H1 | static columns from configs again (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report | `test_the_static_columns_come_from_the_manifests` |
| B08 | T-39 | backbone check ignores d_model and vocab_size (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report | `test_a_run_trained_under_another_setting_is_refused_and_named[model.d_model]` |
| B09 | T-47 | positive control detected in either direction (`analysis/primary_test.py`) | test_session5_rule, test_gate_report, test_make_tables, test_primary_stats | `test_the_positive_control_is_detected_only_when_the_small_model_is_worse[paired]` |
| B10 | T-47 | positive control direction flipped (`analysis/primary_test.py`) | test_session5_rule, test_gate_report, test_make_tables, test_primary_stats | `test_the_positive_control_is_detected_only_when_the_small_model_is_worse[paired]` |
| B11 | T-47 | positive control p == alpha reads not detected (was detected) (`analysis/primary_test.py`) | test_session5_rule, test_gate_report, test_make_tables, test_primary_stats | **missed**; near-equivalent: only p exactly equal to alpha differs, and no test pins that boundary for either control |
| B12 | T-47 | a non-positive slope gives 5 seeds (`analysis/ladder_fit.py`) | test_session5_rule, test_gate_report, test_make_tables, test_ladder_fit | `test_a_slope_that_is_not_positive_gives_eight_seeds[-0.1-paired]` |
| B13 | T-47 | slope exactly 0 treated as positive (`analysis/ladder_fit.py`) | test_session5_rule, test_gate_report, test_make_tables, test_ladder_fit | `test_a_slope_that_is_not_positive_gives_eight_seeds[0.0-paired]` |
| B14 | T-47 | an uncommitted seed count still gives verdicts (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_without_a_committed_seed_count_no_comparison_is_computed` |
| B15 | T-47 | a disagreeing commitment still gives verdicts (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_a_committed_seed_count_that_disagrees_with_the_rule_computes_no_comparison` |
| B16 | T-47 | _effective_decision ignores the commitment (LEFT_ARM shows through) (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables, test_secondary_reports | `test_an_uncommitted_eight_seed_rule_does_not_show_through_the_tables` |
| B17 | T-47 | the ladder declaration is ignored (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables, test_review_s4_analysis, test_make_tables, test_gate_report | `test_the_ladder_key_is_the_declaration_of_the_l10_cut` |
| B18 | T-47 | multipliers attempted with a non-positive slope (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_the_analysis_runs_the_primaries_on_eight_seeds_without_multipliers` |
| B19 | T-47 | any integer accepted as committed seed count (`src/config.py`) | test_session5_rule, test_gate_report, test_make_tables, test_config | `test_a_malformed_file_is_refused[held3]` |
| B20 | T-47 | make_plots main does not pass --commitments on (`scripts/make_plots.py`) | test_session5_rule, test_gate_report, test_make_tables, test_plots | `test_the_command_line_passes_the_commitments_file_to_the_analysis` |
| B21 | T-47 | make_tables main ignores --commitments (reads the repository file) (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_make_tables_reads_the_commitments_file` |
| B22 | T-47 | wrong-direction control gets the plain not-detected note (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_the_analysis_reports_a_control_in_the_wrong_direction_as_failed` |
| B23 | T-47 (control) | gate report hides sigma_init until committed (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | **missed**; nothing pins that the rule's inputs are shown before the commitment (only `seeds per primary` is pinned) |
| B24 | T-50 | any seed at MHA's rate (the sweep point on seed 21) is read as a shared-rate run (`scripts/make_tables.py`) | test_shared_rate, test_select_lr | `test_a_skippy_shared_rate_run_is_refused_and_the_same_run_on_seed_21_is_not` |
| B25 | T-50 | shared-rate runs escape the steps-parity check (`scripts/make_tables.py`) | test_shared_rate | `test_a_skippy_shared_rate_run_is_refused_and_the_same_run_on_seed_21_is_not` |
| B26 | T-50 | tuned minus shared sign flipped (`scripts/make_tables.py`) | test_shared_rate | `test_the_comparison_is_pt_secondary_on_the_same_numbers[moved]` |
| B27 | T-50 | shared-rate comparison computed with the gate shut (`scripts/make_tables.py`) | test_shared_rate | `test_with_the_gate_shut_no_comparison_and_no_loss_is_shown` |
| B28 | T-50 | shared-rate losses printed with the gate shut (`scripts/make_tables.py`) | test_shared_rate | `test_with_the_gate_shut_no_comparison_and_no_loss_is_shown` |
| B29 | T-50 | ladder sizes' runs at MHA's rate are read (`scripts/make_tables.py`) | test_shared_rate | `test_a_ladder_size_on_its_own_rate_gets_no_shared_rate_result_and_its_runs_are_not_read` |
| B30 | T-50 | branch check dropped from the shared-rate selection (equivalent? a branch has tokens 300M) (`scripts/make_tables.py`) | test_shared_rate | missed; equivalent (a branch's name carries tokens 300M, so `tokens == 100M` already excludes it) |
| B31 | T-50 | tuned side reads every seed of the variant (5 or 8), not seeds 1-3 (`scripts/make_tables.py`) | test_shared_rate | `test_result_fields_of_the_moved_variant` |
| B32 | T-50 | a variant on MHA's rate has its standalone runs read (`scripts/make_tables.py`) | test_shared_rate | `test_runs_that_no_comparison_reads_stay_outside_the_steps_parity_check` |
| B33 | T-48 | mark inverted (`scripts/timing_pass.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_the_check_holds_when_swa_attention_is_smaller` |
| B34 | T-48 | equal attention time counts as below (`scripts/timing_pass.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_equal_attention_time_is_not_below_so_it_does_not_hold` |
| B35 | T-48 | headline SWA cell unmarked (`scripts/make_tables.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_headline_and_rqw_are_marked_and_noted_when_swa_is_not_below` |
| B36 | T-48 | rqw SWA cell unmarked (`scripts/make_tables.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_headline_and_rqw_are_marked_and_noted_when_swa_is_not_below` |
| B37 | T-48 | timing.md ratio cell unmarked (`scripts/timing_report.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_timing_md_marks_only_the_swa_opt_row_and_adds_the_note_when_not_below` |
| B38 | T-48 | the pass exits 1 again when SWA is not below (`scripts/timing_pass.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_the_pass_returns_zero_on_cpu_and_cuda_whether_or_not_swa_is_below[not-holds-cpu]` |
| B39 | T-48 | headline note dropped (mark stays) (`scripts/make_tables.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_headline_and_rqw_are_marked_and_noted_when_swa_is_not_below` |
| B40 | T-48 | rqw report built without the check (`scripts/make_tables.py`) | test_swa_check, test_timing_report, test_timing_pass | `test_headline_and_rqw_are_marked_and_noted_when_swa_is_not_below` |
| B41 | T-49 | reported value is the mean, not the median (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report | `test_bench_runs_end_to_end_for_every_variant_in_both_tracks` |
| B42 | T-49 | min and max swapped (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report | `test_median_min_max_and_list_are_the_injected_repetition_values` |
| B43 | T-49 | repetitions back to back per variant, not round-robin (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report | `test_repetitions_run_round_robin_over_variants_and_modes` |
| B44 | T-49 | KV check and max-batch search in every repetition (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report | `test_the_kv_check_runs_once_per_variant_and_mode` |
| B45 | T-49 | peak bytes is the minimum over repetitions (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report | `test_the_aggregates_of_an_even_number_of_repetitions_are_the_mean_of_the_middle_two` |
| B46 | T-49 | headline decode column reads the fastest repetition (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report, test_secondary_reports, test_make_tables | `test_decode_reader_feeds_the_headline_column` |
| B47 | T-49 | default of one repetition (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report, test_runbook | `test_the_default_number_of_repetitions_is_three` |
| B48 | T-49 | prefill aggregated from the first repetition only (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench, test_timing_report | `test_median_min_max_and_list_are_the_injected_repetition_values` |
| B49 | T-42 | spread over the minimum, not the mean (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_the_mha_row_has_every_cell_hand_computed` |
| B50 | T-42 | ratio inverted (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_a_track_without_a_peak_memory_has_a_blank_peak_cell_and_its_spread_is_zero` |
| B51 | T-42/49 | prefill range printed max–min (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_a_prefill_row_has_every_cell_hand_computed` |
| B52 | T-42/49 | decode ms/token column prints the fastest repetition (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_a_decode_row_has_every_cell_hand_computed` |
| B53 | T-42/49 | documents with different repetition counts accepted (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_documents_with_different_repetition_counts_are_refused` |
| B54 | T-42 | timing report skips the timing-source refusals (shared GPU, environment) (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_a_timing_document_recorded_on_a_shared_gpu_is_refused` |
| B55 | T-49 | median/min/max not checked against the repetition list (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_a_median_that_is_not_the_median_of_its_repetitions_is_refused` |
| B56 | T-42 | attention share not in percent (`scripts/timing_report.py`) | test_timing_report, test_swa_check | `test_the_mha_row_has_every_cell_hand_computed` |
| B57 | T-40 | no checkpoint at an --until stop (`scripts/_run_common.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | `test_a_stop_at_until_leaves_a_checkpoint_at_that_step` |
| B58 | T-40 | an existing branch point is overwritten (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | `test_an_existing_branch_point_is_never_overwritten` |
| B59 | T-40 | a resume under another driver is accepted (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | `test_a_resume_under_another_environment_is_refused_and_changes_nothing[driver]` |
| B60 | T-40 | resume ignores branch points (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | `test_a_branch_point_newer_than_every_regular_checkpoint_is_resumed_from` |
| B61 | T-40 | resume takes the oldest checkpoint (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | `test_a_branch_point_newer_than_every_regular_checkpoint_is_resumed_from` |
| B62 | T-40 | a refused checkpoint (other environment) is skipped silently, older one or step 0 used (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | `test_a_resume_under_another_environment_is_refused_and_changes_nothing[torch]` |
| B63 | T-40/M2 | train_loss from the last record even when not finite (`scripts/_run_common.py`) | test_review_s4_resume, test_checkpoint, test_entry_points, test_run_records | `test_a_non_finite_loss_on_the_last_step_still_gives_a_valid_manifest` |
| B64 | T-40 | checkpoints store no driver (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint, test_entry_points | **missed**; on CPU the driver is `None` in both sessions, so storing `None` is the same value. On a GPU every resume would be refused; only the unrun `gpu` tier can see it |
| B65 | T-41 | validation points carry empty buckets (`scripts/_run_common.py`) | test_run_records, test_entry_points, test_manifest | `test_every_validation_point_carries_the_context_blocks` |
| B66 | T-41 | peak memory is the last session's, not the maximum (`scripts/_run_common.py`) | test_run_records, test_entry_points, test_manifest | `test_the_peak_is_the_maximum_over_three_sessions` |
| B67 | T-41 | no peak reading at checkpoints (`scripts/_run_common.py`) | test_run_records, test_entry_points, test_manifest | `test_a_session_killed_after_a_checkpoint_still_counts` |
| B68 | T-41 | peak memory not written to the manifest (`scripts/_run_common.py`) | test_run_records, test_entry_points, test_manifest | `test_a_branch_manifest_carries_the_same_records` |
| B69 | T-41 | a session's later reading replaces its earlier one (equivalent on real CUDA: the peak is monotone) (`scripts/_run_common.py`) | test_run_records, test_entry_points, test_manifest | missed; equivalent on real CUDA (the peak statistic is monotone within a session) |
| B70 | T-43 | a branch point is deleted while its branch is unfinished (`scripts/prune.py`) | test_prune | `test_branch_point_stays_when_its_branch_is_unfinished` |
| B71 | T-43 | checkpoints of an unfinished run are planned for deletion (`scripts/prune.py`) | test_prune | `test_an_unfinished_run_loses_nothing` |
| B72 | T-43 | final checkpoints deleted without --final (`scripts/prune.py`) | test_prune | `test_dry_run_deletes_nothing_and_apply_deletes_exactly_the_listed_files` |
| B73 | T-43 | branch manifest's parent_run not compared (equivalent? the file name already holds the parent's name) (`scripts/prune.py`) | test_prune | `test_a_branch_manifest_naming_another_parent_does_not_authorise_its_branch_point` |
| B74 | T-43 | branch manifest of another configuration with the same name releases the branch point (`scripts/prune.py`) | test_prune | `test_a_branch_manifest_with_another_config_hash_does_not_authorise_its_branch_point` |
| B75 | T-43 | with the final checkpoint missing, the newest regular one is deleted too (`scripts/prune.py`) | test_prune | `test_a_completed_run_without_its_final_checkpoint_keeps_the_newest_regular_one` |
| B76 | T-43 | the dry run deletes (`scripts/prune.py`) | test_prune | `test_dry_run_deletes_nothing_and_apply_deletes_exactly_the_listed_files` |
| B77 | T-43 | apply does not check that a path is a checkpoint under the root (`scripts/prune.py`) | test_prune | **missed**; `apply`'s own path check has no test (the plan never produces such a path) |
| B78 | T-43 | unreadable branch manifest not named (still kept as 'not run'; equivalent in effect) (`scripts/prune.py`) | test_prune | missed; equivalent in effect (the branch point is still kept, with the reason "has not been run") |
| B79 | T-45 | plot 7 draws variant runs with the gate shut (`scripts/make_plots.py`) | test_plots | `test_plot_7_never_draws_a_trunk_point_as_a_result` |
| B80 | T-45 | plot 7 draws every manifest again (sweep points, other arms) (`scripts/make_plots.py`) | test_plots | `test_plot_7_draws_one_final_dot_per_arm_a_headline_run_and_branch` |
| B81 | T-45 | plots skip the timing-source refusals (`scripts/make_plots.py`) | test_plots | `test_make_inputs_refuses_bench_records_the_table_refuses[shared GPU]` |
| B82 | T-45 | 100M branches missing from plot 7 (`scripts/make_tables.py`) | test_plots, test_make_tables | `test_plot_7_draws_one_final_dot_per_arm_a_headline_run_and_branch` |
| B83 | T-45 | 'not FlashAttention-2' printed for a flash-backend campaign (`scripts/make_tables.py`) | test_make_tables, test_review_s4_analysis, test_audit_fixes | `test_a_result_on_the_flash_backend_makes_no_flash_attention_claim` |
| B84 | T-45/M1 | hardware line names the T4 whatever the manifests say (`scripts/make_tables.py`) | test_make_tables, test_review_s4_analysis, test_audit_fixes | `test_the_hardware_line_names_the_gpu_and_backend_the_manifests_record` |
| B85 | T-38/45 | duplicate probe records accepted by the plots (`scripts/make_plots.py`) | test_plots, test_probe_report | `test_duplicate_probe_records_are_refused_by_make_inputs` |
| B86 | T-45/M10 | slope note missing from gates.md (`scripts/make_tables.py`) | test_gate_report, test_make_tables | `test_the_slope_sentence_sits_in_the_seed_count_section_of_every_report` |
| B87 | T-52 | the registered sweep seed is not checked against the arms' seeds (`analysis/lr_sweep.py`) | test_select_lr, test_lr_sweep, test_config, test_runbook | `test_a_prereg_without_a_usable_sweep_seed_refuses_the_report[registered]` |
| B88 | T-52 | select_lr reads sweep runs on seed 1 (`scripts/select_lr.py`) | test_select_lr, test_lr_sweep, test_config, test_runbook | `test_the_report_prints_each_choice_rate_and_source` |
| B89 | T-47 | a NaN slope is not an error (`analysis/ladder_fit.py`) | test_session5_rule, test_ladder_fit | `test_a_slope_that_is_not_a_number_is_still_an_error[nan]` |
| B90 | T-52 | multiplier interval drops the MHA runs' shared term (Welch) (`analysis/ladder_fit.py`) | test_multiplier_coverage, test_ladder_fit | `test_interval_end_points_satisfy_their_definition[0.0-0.0-welch]` |
| B91 | T-52 | multiplier interval drops the slope's uncertainty from the other ladder runs (coverage test alone) (`analysis/ladder_fit.py`) | test_multiplier_coverage | `test_the_multiplier_interval_holds_the_true_ratio_as_often_as_it_says[welch, 5 seeds]` |
| B92 | T-41 | peak_memory with other keys accepted (`src/utils/manifest.py`) | test_manifest, test_run_records | `test_peak_memory_with_missing_or_extra_keys_is_reported[peak0]` |
| B93 | T-41 | train does not reset the peak statistics at session start (`scripts/train.py`) | test_run_records, test_entry_points | `test_the_peak_statistics_start_before_the_model_is_built` |
| B94 | T-39/B10 | Arm B runs outside the steps-parity scope (`scripts/make_tables.py`) | test_review_s4_analysis, test_secondary_reports, test_make_tables | `test_a_skippy_arm_b_run_is_refused[gqa4-1]` |
| B95 | T-39/B36 | reruns not limited to registered seeds (`scripts/make_tables.py`) | test_review_s4_analysis, test_make_tables, test_gate_report | `test_a_rerun_on_an_unregistered_seed_is_not_a_reproducibility_pair` |
| B96 | T-39/B46 | seed-count threshold >= (`analysis/ladder_fit.py`) | test_review_s4_analysis, test_ladder_fit, test_session5_rule | `test_the_seed_count_threshold_is_exceeded_strictly` |
| B97 | T-47 | context use computed without the commitment (on seeds 1-5) (`scripts/make_tables.py`) | test_session5_rule, test_secondary_reports | `test_context_use_waits_for_the_commitment` |
| B98 | T-47 | exploratory comparisons computed from the uncommitted rule output (`scripts/make_tables.py`) | test_session5_rule, test_secondary_reports, test_make_tables | `test_without_a_committed_seed_count_no_comparison_is_computed` |
| B99 | T-49 | maximum-batch search in every repetition (`scripts/bench.py`) | test_bench_repetitions, test_inference_bench | `test_the_max_batch_search_runs_once_per_variant_and_mode_in_the_first_repetition` |
| B100 | T-47 | an unknown key in commitments.yaml is accepted (`src/config.py`) | test_session5_rule, test_config | `test_a_malformed_file_is_refused[held10]` |
| B101 | T-50 | comparison computed with a tuned branch missing (`scripts/make_tables.py`) | test_shared_rate | missed; equivalent in effect (`primary_test.secondary` refuses the missing seed and the note is written) |
| B102 | T-43 | checkpoints of an unfinished branch are planned for deletion (`scripts/prune.py`) | test_prune | `test_an_unfinished_branch_loses_nothing_in_its_own_directory[False]` |
| B103 | T-40 | a damaged branch point is not replaced (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint | `test_a_damaged_branch_point_is_replaced` |
| B104 | T-47 | gates.md prints the rule's count as the committed one (`scripts/make_tables.py`) | test_session5_rule, test_gate_report | `test_without_a_committed_seed_count_no_comparison_is_computed` |
| B105 | T-48 | timing.md ms/step cell unmarked (`scripts/timing_report.py`) | test_swa_check, test_timing_report | `test_timing_md_marks_only_the_swa_opt_row_and_adds_the_note_when_not_below` |
| B106 | T-39 | the rule is content with two L10 seeds (`scripts/make_tables.py`) | test_review_s4_analysis, test_session5_rule, test_gate_report | `test_without_all_of_l10_there_is_no_decision[absent]` |

## 4. Plan or decision problems

- **Decision 220 against its own command line.** The decision replaces a flag by a committed file because a flag can be set after results exist; the implementation keeps a flag that names the file (G1).
- **Plan §1.4, §18 A5 and decision 220 on "read".** The plan wants the count committed before any variant 300M result is read; the decision withholds verdicts only. The code follows the decision (G2). Arm B and the shared-rate comparison deliberately do not wait for the commitment (they are 100M); `rqw.md` and plot 6 do not wait either and are built from 300M results, which no decision covers.
- **Decision 224, "by seed".** Under the paired test the difference is by seed. Under Welch it is not, although the two groups share init seeds. Plan §18 A8 gives the Gate 2 test to the exploratory, Arm B and context-use intervals and does not name this one (M5).
- **Decision 223 and the two fits.** "No multiplier is reported" is tied to the rule's fit on the registered seeds; the multipliers and plot 1 use the fit over every run. The two can disagree in sign (M3). The same pair of fits disagrees under a declared cut when L10 runs exist (M2); plan §18 A5 and A9 speak of the rule only.
- **Decision 226 against the `t4` test of plan §5** (G4): the timing pass no longer fails, the test still asserts.
- **Decision 215, "probes 2 to 5 ... 320 runs".** Probe 5 of plan §9.2 is two tasks, so one size each is 400 runs (M4).
- **Decision 225, "reports the median with the smallest and largest value".** True of `inference.md`; the headline column and plots 2 and 8 show the median only (M1).
- **Decision 229, "gains only".** `configs/prereg.yaml` also gained two comments, on decisions 216 and 223. They are true and change no value.
- **Decision 226, "wherever they are printed".** Read strictly it covers SWA's attention share and its `ref` row in `timing.md` (M8).

## 5. Coverage: what I did not look at

- Anything on a GPU or on Kaggle: fp16, the GradScaler, the kernel enforcement, the CUDA peak statistics, the driver and GPU name in the environment check, `nvidia-smi`, the shim notebook as a notebook.
- T-21 and T-38 (probe generators, `train_probe.py`, `probe_report.py`) beyond running the runbook's pilot commands on CPU and one break; `src/eval/synthetic.py` and `tests/test_probes.py` were not read.
- Most test files line by line. I read `tests/test_session5_rule.py`, `tests/test_multiplier_coverage.py`, `tests/test_runbook.py` and parts of `tests/test_shared_rate.py`, `tests/test_review_s4_resume.py` and `tests/test_entry_points.py`; the others were judged by the breaks.
- The plots as images, except plot 1 in one case. Plots were checked by counting drawn points and reading captions through the figure objects.
- `scripts/measure_reproducibility.py`, `scripts/evaluate.py`, `scripts/graph_diff.py`, `scripts/solve_config.py` and `scripts/prepare_data.py`: unchanged since `384e8b3`, not run.
- The runbook's GPU-hour figures per week and the 208 MB per checkpoint behind G5 (taken from review S4).
- GitHub: CI runs, branch protection, labels and pull request 59 itself. I had no need of the network and did not use it.
- `docs/decisions.md` rows 1 to 214, except where a later row or the plan's §18 cites them.
