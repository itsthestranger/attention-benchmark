# Re-check of the session-5 review fixes (T-54 to T-57)

Reviewed 2026-11-14 by a fresh agent that wrote none of the code. Branch `review/combined-s5` at `523f1d6`, against `864f132`. Work was done in three detached worktrees (`/home/user/wt/recheck`, `recheck2`, `recheck3`), all clean after every break and now removed. Two departures from the brief: I used a third worktree for an edit case of my own, and up to three pytest processes ran at the same time, one per worktree and never two in one, so the timings above 335 s are slower than a run alone. Nothing was committed, pushed or posted, and no file outside the worktrees and the scratch directory was changed.

Scratch directory, called `S` below: `/tmp/claude-0/-home-user-attention-benchmark/b1914e3e-0c3b-517e-b90e-f6c7b7b248ed/scratchpad/recheck`.

Evidence labels: **reproduced** (a command I ran shows it), **break** (a one-line source break and whether a test caught it), **read** (concluded from code or text), **suspected** (needs a GPU to settle).

Full `cpu` tier on the branch as it is: `3860 passed, 105 deselected in 335.15s (0:05:35)` (`S/full_cpu.log`).

## 1. Verdict

The fixes do what decisions 237 to 243 say. H1 is fixed: all four cases of `docs/review_s5/phil_edits.py` leave the tier at 3860 passed, and so does a case of my own that applies every edit the runbook asks for at once, with other values. G1, G2, G3, G5, M1, M2, M3, M7 and M9 are fixed, each reproduced on CPU with synthetic data; M4, M5 and M6 are fixed by reading. 55 of 63 deliberate breaks were caught; of the 8 misses one is equivalent, four are prose or caption text that no test reads, and three are test gaps (two of them on G1's own property). No test was weakened. Two things are not finished. G4 is fixed only in part: the new runbook paragraph names one test that fails when FlexAttention does not run, and a second test fails for the same cause, on the T4 and on the 1070, under a sentence that says to stop (new finding 1); the same paragraph asks Phil to pick the faster SWA path and no command shows which is faster when both tests pass (finding 2). M10 is not fixed on this branch: `STATUS.md` is unchanged since T-53 although `TASKS.md` ticks T-57 (finding 6). The notebook has a conditional case of the H1 class (finding 3). Nothing on a GPU was verified.

## 2. The review's findings

| # | State | Evidence |
| --- | --- | --- |
| H1 | fixed | **Reproduced.** `python docs/review_s5/phil_edits.py X1 X2 X3 X4` in my worktree: X1 `3860 passed, 105 deselected in 339.34s`, X2 `3860 passed ... 340.22s`, X3 `3860 passed ... 334.80s`, X4 `3860 passed, 105 deselected, 1 warning in 343.35s` (`S/phil_edits.out`). My combined case (`seed_count: 8`, `ladder: l6_l8`, `micro_batch: 16`, `init_scale: 32768.0`, `growth_interval: 1000`, both hashes, `opt_path: flex`, the notebook's placeholders filled, a second run spec, the README's TO FILL lines filled, a row added to `docs/decisions.md`, torch pinned in `pyproject.toml` and `ci.yml`): `3860 passed, 105 deselected, 1 warning in 349.70s` (`S/combined_edits.log`). The same class elsewhere: findings 3 and 7. |
| G1 | fixed | **Reproduced** through the real command lines in a clean export: `--commitments FILE`, `--comm FILE` and `--seed-count 5` are refused by both scripts; with the checkout's file at `null`, `5` and `8` the tables withhold, give verdicts, and withhold with "seed count disagrees". No environment variable or run-spec key is read (`grep` for `environ`, `load_commitments`). `--blank` gives the committed blank forms with the file at 5. Known and listed in decision 237: an uncommitted edit of the file is read like a committed one. Test gap: finding 5. |
| G2 | fixed as decided (decision 240, for Phil to confirm) | **Reproduced** (`S/g2g3.py`, output in `S/g2g3_out`). 474 values (each variant's 300M loss per seed, bucket losses, curve points, ICL scores, and gaps and means made of them) were searched in every table `make_tables` writes and in the coordinates and texts of all ten figures, in six states: gate open with no count, with a wrong count, with the cut declared and no count, with a count and no rule output; gate shut with the right count and with none. None appears in `headline.md`, `gates.md`, `context.md`, `rqw.md` or any plot. The script's raw matches in those states (15 to 20 per state) are not 300M values: the 100M branch losses in `arm_b.md` and `shared_rate.md`, which the fixture sets to the 300M loss plus 0.3, the value of one of its curve points, and three noise-floor sigmas in plot 4's caption that equal a gap to four decimals. Plots 2 and 3 draw 5 points (MHA), plot 7 draws 15 runs (MHA), plots 1 and 6 draw 0. With the right count committed the same search finds 606. Residual: finding 8. |
| G3 | fixed | **Reproduced** (same script): plot 6 draws 0 gaps with the gate shut and with the count missing or wrong, 96 when shown. |
| G4 | partly | **Read.** `RUNBOOK.md` §1.2 names both cases of `test_swa_attention_time_at_ctx_1024_is_below_mha`; both ids exist (`pytest --collect-only`), and the quoted plan sentence is in plan §2.3. Not covered: findings 1 and 2. |
| G5 | fixed | **Reproduced** through the command lines (`S/g5.py`): a 32-step run named 300M with a 100M branch, a finished and an unfinished standalone 100M run. `--final-short` listed and deleted the finals of the branch and of the finished short run only; the 300M final stayed with its own reason; the unfinished run lost nothing and resumed; rerunning the finished runs exits 0 ("its final checkpoint has been pruned"); `evaluate` still loads the 300M run. Arithmetic checked against the runbook's run list: 28 files after week 3 and 37 after week 4 with 5 seeds, 40 and 43 with 8 (finding 9). |
| M1 | fixed | **Reproduced** (`S/m1.py`): mixed repetition counts, a median or a minimum that is not of its list, an unknown variant and a NaN repetition are refused by `make_tables`, `make_plots` and `timing_report` with the same message; the decode cell reads `1200.0 (1176.0–1224.0)`. One-repetition documents are accepted by all three, as `timing_report` did before. |
| M2 | fixed | **Reproduced** (the reviewer's `e5.py`): under `l6_l8` the reported fit has the points `mha_L6` and `mha` and b 0.34888, equal to the rule's, also with one L10 seed on disk; the slopes by budget are the same number. |
| M3 | fixed | **Reproduced** (the reviewer's `e8.py`, campaign 35): rule slope -0.00036, all-seed slope +0.0016; the caption now reads "No top axis: the ladder slope is not positive, so no multiplier is reported and 8 seeds are run (decision 223)" and no axis carries a multiplier. |
| M4 | fixed | **Read.** 400 runs in `RUNBOOK.md` §2.3 and plan §18 A12; §5.3 says "the trial above", and the trial is above. Not under test (break R49). |
| M5 | fixed | **Read**, and break R53. The caption, the blank form, `README.md` and plan §18 A8 say the same. |
| M6 | fixed | **Read.** `FLOAT64_IDENTITY` (1e-12, tighter than the 1e-9 it replaces) in `tests/test_shared_rate.py`; the three bounds of the coverage test are in `tests/tolerances.py` with their old values. No literal tolerance was added in the diff. |
| M7 | fixed | **Reproduced** (`S/g5.py`): from an export without `.git`, `train` and `branch` exit 2 with "git rev-parse HEAD failed" before any file exists under OUT; an unfinished run is left byte for byte. `train_probe` already read the hash before training. |
| M8 | listed, not fixed | As `STATUS.md` says. |
| M9 | fixed | **Reproduced**: the cell executed with `subprocess` replaced. `--out /kaggle/working/out`, the log under `/kaggle/working/out/logs`, and `log.terminate()` called once when the training call raises, with the error still raised. A stale pointer in the runbook: finding 4. |
| M10 | not fixed on this branch | Finding 6. |
| M11 | fixed (B23, B64, B77); B11 listed | Breaks R09 to R12 are caught by the three new tests. |

## 3. New findings

### Fix before the GPU weeks

**1. The runbook's new paragraph on FlexAttention names one of the two tests that fail when it does not run.**
- Where: `RUNBOOK.md:84` (§1.1, "every test passes; none is skipped"), `:92-97` (§1.2); `tests/test_sliding_window.py:381-396` (`test_opt_paths_match_ref_on_cuda[flex]`, marked `gpu`); `src/attention/sliding_window.py:65-70, 270`.
- On CUDA the flex path is `torch.compile(flex_attention)`. If that does not compile or run on sm75, the case the paragraph is written for, `test_opt_paths_match_ref_on_cuda[flex]` fails too, and the next sentence says "Any other failure is a finding about the harness: stop and fix it". The same test is in the `-m gpu` run of §1.1 on the 1070 (sm61), which has no exception at all.
- Established: **read**; the outcome is **suspected**.
- Fix: name that case beside the other in §1.2, and say in §1.1 what a failure of it on the 1070 means.

**2. "If the `flex` case passes and is the faster one" cannot be answered from anything the runbook runs.**
- Where: `RUNBOOK.md:97`; `tests/test_sliding_window.py:440`.
- The two attention times are in the assertion message only, so they are printed for a failing case and never for a passing one. No script compares the two paths (`grep flex scripts/` is empty), and the timing pass measures the configured path alone. The choice is part of the configuration hash and must be made before any SWA run.
- Established: **read**.
- Fix: a command that prints both times, or a rule that needs no comparison (for example: flex whenever its case passes).

**3. The notebook has two conditional cases of H1.**
- Where: `tests/test_entry_points.py:592, 599-601`; `tests/test_review_s5_ops.py:129-140` (new in T-56); `RUNBOOK.md:44, 105`.
- (a) The test bans the substring `token` anywhere in the notebook. With the placeholder `DATASET_NAME` set to `fineweb-edu-tokenized`: `1 failed` (`test_the_notebook_is_a_one_cell_shim_without_credentials`).
- (b) `RUNBOOK.md:44` says to replace the module on the notebook's training line "in a committed run" for a branch, a timing pass or the benchmark. With `scripts.branch` there: `2 failed, 79 passed` (the test above, and the new `test_the_notebook_terminates_the_log_in_a_finally_that_wraps_the_training_call`, which looks for `scripts.train` in the `try` body).
- Established: **reproduced**. With the placeholders filled and a name without `token`, the three test files pass (81 passed).
- It bites only if the edited notebook is committed to the repository. §1.4 says "commit the notebook", which can mean Kaggle's commit or git's; the runbook should say which.

### Minor

- **4. `RUNBOOK.md:44` points at the wrong line of the notebook.** "replace the module and its arguments on the notebook's last line": since T-56 the last line is `finally: log and log.terminate()` and the call is on the `try:` line above it. T-57 edited this bullet and kept the pointer. **Read.**
- **5. No test pins that the command lines read the declarations from the checkout's file only.** Breaks R54 and R55 (the path taken from an environment variable when set, in `make_tables.main` and in `config.load_commitments`) pass 208 and 407 tests. The tests replace the loader by functions that accept any path (`tests/test_session5_rule.py:476, 516`, `tests/test_plots.py:1364`). The code is right today; G1 could come back unseen. **Break.**
- **6. `STATUS.md` is the one of T-53.** Its last commit is `21d06ab`. It says the fixes are not merged, counts 3,785 tests, lists decisions "1 to 214", and says the rewrite "lands with T-57" (`STATUS.md:3, 109-110, 123`). `TASKS.md:130` ticks T-57, whose task text includes "Final `STATUS.md`". Decisions 237 to 243 and the two items for Phil (240, 242) are not in it, which handover item 5 asks for. **Read.**
- **7. `phil_edits.py` leaves its logs in the checkout.** After a run `git status` shows `docs/review_s5/phil_X1.log` to `phil_X4.log` untracked (not ignored), so the script's own last line is never empty and a `git add -A` would commit them. The guard against H1 is this manual script; nothing in CI stands in for it, and since the unfilled state is no longer pinned, nothing notices a template committed in a filled state either (commit `9463e1c` restored one such slip). I compared: no file under `configs/` differs from `864f132`. **Reproduced.**
- **8. What still shows without a committed count, by decision 240.** `arm_b.md` prints each Arm B variant's Arm A 100M branch losses per seed, a ranking of the variants "Arm A at 100M" and Arm B's gap to MHA; `shared_rate.md` prints the tuned 100M branch losses. These branches are the first 1,221 steps of the same 300M trunks, and plot 7 withholds them in the same state. The decision says the two tables do not wait; Phil should see this when he confirms it. **Reproduced** (`S/g2g3_out/open_uncommitted/tables`).
- **9. Runbook numbers in the new disk paragraph.** "28 files after week 3" is the 5-seed case; with 8 seeds it is 40 (43 after week 4, as written). Both stay near 9 GB. The "Open items" table does not list `opt_path` (§1.2). **Read.**
- **10. Captions.** Plots 2 and 8 say "the range of each is in inference.md"; that table gives the range of ms/token and of prefill latency, and the range of plot 2's tokens/s is in `headline.md`. Plot 6 says "variant runs are withheld" in week 2, when no variant run exists yet (`scripts/make_plots.py:816`; the caption line beside it is conditional, the message in the axes is not). **Reproduced.**
- **11. Test gaps from section 4.** R27 (the caption sentence for a slope that is not positive), R61 (the prune tests have one run at the primary budget), R49, R50 and R52 (runbook prose: the run count, the test ids of §1.2, where the declarations are read).

### Changed or removed assertions (step 6)

No assertion was loosened, no `skip` or `xfail` added, no literal tolerance added.

| Test | What changed | Still pinned? |
| --- | --- | --- |
| `test_config`: 20 YAML files | 19 outside `configs/runs/`, and the example spec present | Yes, except the number of run specs (decision 243). |
| `test_config`: the three recipe values are `None` | unset or usable, all or none; a new test checks the fixture unsets exactly these three | The refusals of an unfilled recipe are pinned through the fixture (`test_config`, `test_entry_points`, `test_trainer`, `test_timing_pass`). The repository's own state is not, by design. |
| `test_config`, `test_probes`, `test_sliding_window`: `opt_path == "chunked"` | the file's value, one of the two | Resolved config from the file: `test_config`. Build from the config: by override, both paths. With `flex` in the file the default build is not asserted; the two halves are. |
| `test_data`: both hashes `None` | unset or SHA-256, both or neither | The refusal of an unpinned config is pinned on a built copy (`test_entry_points`). |
| `test_session5_rule`: `test_the_committed_file_declares_nothing_yet` removed | the file must load; `uncommitted()` is the full ladder and no count | The behaviour with no count is pinned with explicit mappings. |
| `test_session5_rule`, `test_plots`, `test_review_s4_analysis`: `--commitments FILE` cases | the loader replaced, or a mapping passed | Yes; and the flag is now refused (R04, R05). See finding 5. |
| `test_runbook`: prune commands | four commands, two pairs, `--final` never a token | Stronger. |
| `test_secondary_reports`: decode cell | median with range | Changed with decision 241. |
| `test_shared_rate`: `abs=1e-9` | `FLOAT64_IDENTITY` (1e-12) | Tighter. |
| `test_multiplier_coverage`: 0.01, 0.8, 3.0 | constants of `tests/tolerances.py`, same values | Yes. |
| `test_review_s4_resume`: environment | every compared field given its own value | Stronger (R10). |

## 4. Deliberate breaks

63 breaks, 55 caught. Of the 8 not caught: 1 equivalent (R62), 4 prose or caption text (R27, R49, R50, R52), 3 test gaps (R54, R55, R61). By task: T-54 12 of 14, T-55 27 of 29, T-56 13 of 14, T-57 3 of 6. Each was one replacement applied by `S/breaks.py` or `S/breaks2.py`, run with `pytest -x` on the named files and restored with `git checkout`; results are in `S/breaks_results.jsonl`.

| # | Task | Break | Test files run | Caught by |
| --- | --- | --- | --- | --- |
| R01 | T-54 | make_tables --blank reads the repository's file again (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_the_blank_tables_and_plots_do_not_depend_on_what_the_file_holds[held0]` |
| R02 | T-54 | make_plots --blank reads the repository's file again (`scripts/make_plots.py`) | test_session5_rule, test_plots | `test_the_blank_tables_and_plots_do_not_depend_on_what_the_file_holds[held0]` |
| R03 | T-54 | make_plots --blank does not pass the declarations on to build_all (`scripts/make_plots.py`) | test_session5_rule, test_plots | `test_the_blank_tables_and_plots_do_not_depend_on_what_the_file_holds[held0]` |
| R04 | T-54 | make_tables accepts --commitments FILE again (and ignores it) (`scripts/make_tables.py`) | test_session5_rule, test_review_s4_analysis | `test_no_command_line_flag_declares_anything[flag1-scripts.make_tables]` |
| R05 | T-54 | make_plots accepts --commitments FILE again (and ignores it) (`scripts/make_plots.py`) | test_session5_rule, test_plots | `test_no_command_line_flag_declares_anything[flag1-scripts.make_plots]` |
| R06 | T-54 | make_tables main ignores the mapping given in Python and reads the file (`scripts/make_tables.py`) | test_session5_rule | `test_a_mapping_given_in_python_takes_the_place_of_the_file` |
| R07 | T-54 | make_plots main ignores the mapping given in Python and reads the file (`scripts/make_plots.py`) | test_session5_rule, test_plots | `test_a_mapping_given_in_python_takes_the_place_of_the_file` |
| R08 | T-54 | config.uncommitted() declares the cut ladder (`src/config.py`) | test_session5_rule, test_gate_report, test_make_tables, test_config | `test_the_repositorys_file_is_a_valid_declaration_whatever_it_holds` |
| R09 | T-54 (B23) | gate report hides sigma_init until the count is committed (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables | `test_the_rules_inputs_and_output_are_shown_before_anything_is_committed` |
| R10 | T-54 (B64) | checkpoints store no driver (`src/training/checkpoint.py`) | test_review_s4_resume, test_checkpoint | `test_a_checkpoint_stores_the_environment_it_was_written_in` |
| R11 | T-54 (B77) | prune.apply does not check that a path is a checkpoint under the root (`scripts/prune.py`) | test_prune | `test_apply_refuses_a_planned_path_that_is_not_a_checkpoint_under_out` |
| R12 | T-54 (B77) | prune.apply checks the file name only, not that it lies under the root (`scripts/prune.py`) | test_prune | `test_apply_refuses_a_planned_path_that_is_not_a_checkpoint_under_out` |
| R13 | T-55 | variant_results_shown ignores the commitment (the gate alone, as before) (`scripts/make_tables.py`) | test_review_s5_display, test_plots, test_session5_rule | `test_variant_results_are_shown_only_with_the_gate_open_and_the_seed_count_committed` |
| R14 | T-55 | variant_results_shown ignores the gate (the commitment alone) (`scripts/make_tables.py`) | test_review_s5_display, test_plots, test_make_tables, test_gate_report | `test_plot_6_draws_no_gap_and_says_so_with_the_gate_shut_or_uncommitted[gate shut]` |
| R15 | T-55 | headline withholds on the gate alone (`scripts/make_tables.py`) | test_review_s5_display, test_make_tables, test_session5_rule | `test_headline_withholds_every_variant_cell_but_mha_until_the_seed_count_is_committed[uncommitted]` |
| R16 | T-55 | rqw.md is computed without the committed seed count (`scripts/make_tables.py`) | test_review_s5_display, test_swa_check, test_secondary_reports | `test_rqw_and_context_say_the_commitment_is_missing_and_hold_no_variant_value` |
| R17 | T-55 | context.md does not say that it waits for the commitment (`scripts/make_tables.py`) | test_review_s5_display, test_secondary_reports | `test_rqw_and_context_say_the_commitment_is_missing_and_hold_no_variant_value` |
| R18 | T-55 | plots 2, 3 and 7 withhold on the gate alone (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plots_2_3_and_7_draw_no_variant_point_while_the_commitment_is_missing` |
| R19 | T-55 | plot 6 draws the gaps of withheld variants (G3 undone) (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plot_6_draws_no_gap_and_says_so_with_the_gate_shut_or_uncommitted[gate shut]` |
| R20 | T-55 | the withheld caption is written on the gate alone (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plots_2_3_and_7_draw_no_variant_point_while_the_commitment_is_missing` |
| R21 | T-55 | the commitment note is also written with the gate shut (`scripts/make_tables.py`) | test_review_s5_display, test_make_tables, test_gate_report | `test_the_commitment_note_is_written_once_and_only_while_a_variant_loss_waits` |
| R22 | T-55 | the 300M reported fit ignores the declared cut (M2 undone) (`scripts/make_tables.py`) | test_review_s5_display, test_session5_rule, test_review_s4_analysis | `test_a_declared_cut_leaves_mha_l10_out_of_every_fit_and_multiplier` |
| R23 | T-55 | the 100M fit ignores the declared cut (`scripts/make_tables.py`) | test_review_s5_display, test_session5_rule | `test_a_declared_cut_leaves_mha_l10_out_of_every_fit_and_multiplier` |
| R24 | T-55 | the 200M fit ignores the declared cut (`scripts/make_tables.py`) | test_review_s5_display, test_session5_rule | `test_a_declared_cut_leaves_mha_l10_out_of_every_fit_and_multiplier` |
| R25 | T-55 | plot 5 draws L10 under a declared cut (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plot_5_draws_no_l10_under_a_declared_cut_and_says_so` |
| R26 | T-55 | plot 1 keys the multiplier axis on the all-seed fit alone (M3 undone) (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plot_1_has_no_multiplier_when_only_the_rules_slope_is_not_positive` |
| R27 | T-55 | plot 1's caption says 'the ladder fit does not exist yet' when the slope is not positive (`scripts/make_plots.py`) | test_review_s5_display, test_plots | **missed**; the caption sentence for a slope that is not positive is not pinned (the axis itself is: R26) |
| R28 | T-55 | make_tables does not check the bench documents (M1 undone) (`scripts/make_tables.py`) | test_review_s5_display, test_secondary_reports, test_make_tables | `test_the_table_command_refuses_a_bad_bench_document_and_writes_nothing` |
| R29 | T-55 | make_plots does not check the bench documents (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_the_plot_inputs_refuse_a_bad_bench_document` |
| R30 | T-55 | read_decode_range returns (largest, smallest) (`scripts/bench.py`) | test_review_s5_display, test_secondary_reports, test_bench_repetitions | `test_read_decode_range_is_the_smallest_and_largest_of_the_headline_point` |
| R31 | T-55 | main does not pass the decode range to the headline (`scripts/make_tables.py`) | test_review_s5_display, test_secondary_reports, test_make_tables | `test_main_fills_the_timing_and_decode_columns_from_the_readers` |
| R32 | T-55 | the decode-range note is not written (`scripts/make_tables.py`) | test_review_s5_display, test_secondary_reports, test_make_tables | `test_main_fills_the_timing_and_decode_columns_from_the_readers` |
| R33 | T-55 | check_documents accepts documents with different repetition counts (`scripts/bench.py`) | test_review_s5_display, test_timing_report | `test_documents_with_different_repetition_counts_are_refused` |
| R34 | T-55 | check_documents does not compare median, min and max with the repetition list (`scripts/bench.py`) | test_review_s5_display, test_timing_report | `test_check_documents_refuses_what_timing_report_refuses[a median that is not the median]` |
| R35 | T-55 | plot 8 does not say its speed values are medians (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plots_2_and_8_say_the_speed_values_are_medians` |
| R36 | T-55 | the decode cell prints the median alone although a range is given (`scripts/make_tables.py`) | test_review_s5_display, test_secondary_reports | `test_the_decode_cell_reads_the_median_with_the_range_in_brackets` |
| R37 | T-55 | plot 6 with every variant withheld draws nothing but the caption line is dropped (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plot_6_draws_no_gap_and_says_so_with_the_gate_shut_or_uncommitted[gate shut]` |
| R38 | T-56 | --final-short also removes the final checkpoint of a run at the primary budget (`scripts/prune.py`) | test_prune | `test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final` |
| R39 | T-56 | --final-short keeps the final checkpoint of a branch (`scripts/prune.py`) | test_prune | `test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final` |
| R40 | T-56 | --final-short keeps short runs and removes primary runs (comparison inverted) (`scripts/prune.py`) | test_prune | `test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final` |
| R41 | T-56 | prune's command line does not pass --final-short on (`scripts/prune.py`) | test_prune | `test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final` |
| R42 | T-56 | --final-short reads the primary budget as a fixed '300M', not from prereg (`scripts/prune.py`) | test_prune | `test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final` |
| R43 | T-56 | train does not read the commit hash before the first step (M7 undone) (`scripts/train.py`) | test_review_s5_ops, test_entry_points | `test_train_without_a_git_checkout_is_refused_before_any_step` |
| R44 | T-56 | branch does not read the commit hash before any work (`scripts/branch.py`) | test_review_s5_ops, test_entry_points | `test_branch_without_a_git_checkout_is_refused_and_writes_nothing` |
| R45 | T-56 | the shim passes --out /kaggle/working again (M9 undone) (`notebooks/kaggle_shim.ipynb`) | test_review_s5_ops, test_entry_points, test_runbook | `test_the_notebook_passes_the_runbook_out_directory_to_train` |
| R46 | T-56 | the shim stops its log only when the run succeeds (no finally) (`notebooks/kaggle_shim.ipynb`) | test_review_s5_ops, test_entry_points, test_runbook | `test_the_notebook_terminates_the_log_in_a_finally_that_wraps_the_training_call` |
| R47 | T-56 | the shim writes its utilisation log under /kaggle/working/logs again (`notebooks/kaggle_shim.ipynb`) | test_review_s5_ops, test_entry_points, test_runbook | `test_the_notebook_writes_its_log_under_the_runbook_out_directory` |
| R48 | T-57 | RUNBOOK: the --final-short commands use --final (`RUNBOOK.md`) | test_runbook | `test_pruning_is_shown_as_a_dry_run_before_it_is_applied` |
| R49 | T-57 | RUNBOOK: block 2 is 320 runs again (M4 undone) (`RUNBOOK.md`) | test_runbook | **missed**; runbook prose, no test reads the count |
| R50 | T-57 | RUNBOOK §1.2 names a t4 test that does not exist (`RUNBOOK.md`) | test_runbook | **missed**; no test checks that a test id named in the runbook exists |
| R51 | T-57 | RUNBOOK: OUT is /kaggle/working (differs from the shim) (`RUNBOOK.md`) | test_runbook, test_review_s5_ops | `test_the_notebook_passes_the_runbook_out_directory_to_train` |
| R52 | T-57 | RUNBOOK §3.4 no longer says the tables read the checkout's file / withhold the 300M losses (`RUNBOOK.md`) | test_runbook | **missed**; runbook prose |
| R53 | T-57/T-55 | the shared-rate caption says 'by seed' unconditionally again (M5 undone) (`scripts/make_tables.py`) | test_review_s5_display, test_shared_rate | `test_the_shared_rate_caption_names_both_tests_and_qualifies_by_seed` |
| R54 | T-54 | make_tables' command line reads the declarations from a file named by an environment variable when it is set (`scripts/make_tables.py`) | test_session5_rule, test_gate_report, test_make_tables, test_review_s4_analysis | **missed**; test gap: nothing pins that the command line reads only the checkout's file (new finding 5) |
| R55 | T-54 | config.load_commitments() reads a file named by an environment variable when it is set (`src/config.py`) | test_session5_rule, test_gate_report, test_make_tables, test_config, test_plots | **missed**; the same gap, in the loader |
| R56 | T-55 | plot 2 does not say its speed values are medians (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plots_2_and_8_say_the_speed_values_are_medians` |
| R57 | T-55 | plot 5 does not say the ladder is cut by declaration (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plot_5_draws_no_l10_under_a_declared_cut_and_says_so` |
| R58 | T-56 | a branch is judged like a run under --final-short (its name's budget is the trunk's) (`scripts/prune.py`) | test_prune | `test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final` |
| R59 | T-56 | --final with --final-short keeps the final_short state (plan differs from --final alone) (`scripts/prune.py`) | test_prune | `test_final_then_final_short_is_the_same_plan_as_final` |
| R60 | T-56 | the shim creates /kaggle/working/logs but logs under out/logs (`notebooks/kaggle_shim.ipynb`) | test_review_s5_ops, test_entry_points, test_runbook | `test_the_notebook_writes_its_log_under_the_runbook_out_directory` |
| R61 | T-56 | --final-short treats a 300M run of another arm or ladder size as short (compares the whole name, not the budget) (`scripts/prune.py`) | test_prune | **missed**; the prune tests have one run at the primary budget, an MHA trunk; a break keyed on the variant passes |
| R62 | T-55 | the headline's decode cell shows the range also for a variant without a median (no change expected: equivalent guard) (`scripts/make_tables.py`) | test_review_s5_display, test_secondary_reports, test_make_tables | missed; equivalent in effect (the median and the range are read from the same rows, so a variant with a range always has a median) |
| R63 | T-55 | withheld plots: plot 3 (memory frontier) loses the withheld caption (`scripts/make_plots.py`) | test_review_s5_display, test_plots | `test_plots_2_3_and_7_draw_no_variant_point_while_the_commitment_is_missing` |

## 5. What I did not look at

- Anything on a GPU or on Kaggle: both `t4` cases, FlexAttention, fp16, the notebook as a notebook.
- Code outside the diff of `864f132..523f1d6`, and tasks before T-54.
- `STATUS.md` beyond its head, its "In progress" section and the checklist; `docs/decisions.md` rows other than 237 to 243.
- The effect of another torch version on the tier (`RUNBOOK.md` §1.5). Decision 243 lists `tests/test_data.py::test_golden_order_regression` as known; I only pinned the version strings.
- The plots as images. They were checked through the figure objects: points, line data, bars and texts.
- The 208 MB per checkpoint and the 44, 78 and 108 files of the old disk estimate (taken from review S5).
- GitHub: CI runs, labels and the pull requests themselves. I did not use the network.
