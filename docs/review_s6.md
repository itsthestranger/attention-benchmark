# Re-check of T-59 and T-60 (session 6)

Reviewer: a fresh agent that wrote none of the code. Nothing was fixed, committed or pushed.

## 1. What was run

**Commit reviewed.** Worktree `/home/user/wt/recheck`, branch `review/s6-combined`, HEAD `803b636`.
When the review started `main` was `c2b1156` and `git diff main` showed the ten changed files
(1,184 insertions, 27 deletions). During the review `main` moved to `d866b1c` (T-59 as #66 and T-60
as #67, squash-merged by someone else). The tree of `main` is now identical to the tree reviewed
(`git rev-parse main^{tree} HEAD^{tree}` gives `251c976…` twice), so every finding below applies to
`main` as it stands.

**The `cpu` tier**, once, before any break:
`OMP_NUM_THREADS=1 python3 -m pytest -m "not gpu and not t4" -q -p no:cacheprovider`
gave **3915 passed, 106 deselected in 351.64 s**.

**The hardware tiers.** `pytest --collect-only -q -m "gpu or t4"` collects 106 of 4021 (86 `gpu`,
20 `t4`). The new test `tests/test_swa_paths.py::test_the_script_runs_on_cuda_in_fp16_and_records_the_backend`
is collected. `pytest -m "gpu or t4"` gives 106 skipped, each with "no CUDA device (plan §5.1); not
run, not passed". None of them has run.

**Deliberate breaks.** 86 one-line breaks of the new code and text, each followed by the relevant test
files and `git checkout -- <file>`: 54 caught, 32 missed (table in section 3). One background batch
was stopped at its time limit in the middle of break P11 and left `scripts/make_plots.py` modified;
I restored it with `git checkout`, checked `git status`, and ran P11 to P16 again.

**Real runs on CPU**: `scripts.swa_paths --synthetic`, `scripts.timing_pass --synthetic`,
`scripts.timing_report` with the record, with the record and a timing pass, and with `--blank`;
35 hand-edited or wrong records; plots 2 and 8 rendered to PNG from the test fixtures (full, gate
shut, blank, and a copy with a 25 to 40% range) and looked at.

**Worktree at the end**: `git status --short` is empty. All my files are in the scratch directory
`/tmp/claude-0/-home-user-attention-benchmark/b1914e3e-0c3b-517e-b90e-f6c7b7b248ed/scratchpad/recheck/`.

**One thing I overwrote in the scratch directory.** That directory already held the files of the
session-5 re-check (written 08:35 to 08:57). I wrote my break harness as `breaks.py` without looking
first. The session-5 files include a `breaks.log` and a `breaks2.py`, so a session-5 `breaks.py` very
probably existed and mine replaced it; I cannot restore it. Its outputs (`breaks.log`,
`breaks_results.jsonl`, `break_table.md`, `recheck_s5.md`) are untouched, and none of my other files
or directories collided with an older one.

## 2. Findings

Ids: H blocks the handover, G fix before the GPU weeks, M minor.

### H1. `STATUS.md` states things the merged code makes false

Handover items 2 and 5 are not met until `STATUS.md` is updated. This is bookkeeping that T-61
presumably plans, but four of the statements matter:

- `STATUS.md:46` says "105 collected: 85 `gpu`, 20 `t4`". It is 106 (86 and 20). The table of
  hardware tests by file has no row for `test_swa_paths.py`, so the new `gpu` test is not listed as
  not yet run.
- `STATUS.md:179`: "Whether FlexAttention compiles and runs on the T4 (plan §2.3), **and which SWA
  path is faster there. Then set `opt_path`**". Read with the new table, which shows exactly which
  path is faster, this tells Phil to choose by time. That contradicts decision 244 and the runbook.
  It is the one place I found where the record could influence the path.
- `STATUS.md:106`: "no script times the two paths side by side". One now does.
- `STATUS.md:109` (and the M1 row at line 87): plots 2 and 8 "without error bars". They have whiskers.

Fix: update the count and the table, reword line 179 to the rule of decision 244, correct lines 106
and 109, move T-59 and T-60 from "in progress" to done.

### G1. The runbook's rule for SWA's path and the new record can disagree, and the runbook does not say what then

`RUNBOOK.md` §1.2 decides `flex` by one test: "if `test_opt_paths_match_ref_on_cuda[flex]` passes on
the T4 … set `opt_path: flex`". That test runs the forward only, under `torch.no_grad()`, on a tiny
model (`tests/test_sliding_window.py:383-396`). The record is the first place where FlexAttention is
asked for a forward and a backward in fp16 at batch 16 and context 1,024, which is what training
needs. That the two can differ is not hypothetical: on CPU with torch 2.15.0 the forward-only test
`test_flex_path_matches_ref_uncompiled_on_cpu` passes and the record says
`NotImplementedError: FlexAttention does not support backward on CPU`.

Two cases the text does not settle:

1. The gate test passes and the record carries an error for `swa256/flex`. The rule says set `flex`;
   the first bullet of §1.2 says the two `[flex]` cases fail "if FlexAttention does not compile or
   run … Then the chunked path is SWA's `opt` path". The new paragraph only says the script "still
   exits 0". Followed literally, the rule produces a config on which SWA cannot take a training step.
2. `test_swa_attention_time_at_ctx_1024_is_below_mha[flex]` fails because flex ran and was slower
   than MHA. The first bullet reads this as "does not compile or run" and sends Phil to `chunked`,
   which is a choice by timing. The record tells the two apart (times, or an error), but the runbook
   does not tell Phil to use it for that.

Fix (needs Phil, because it is the wording of decision 244's rule): say that "FlexAttention runs"
means the gate test passes and the record's flex track has times, not an error; and that a `[flex]`
timing case that fails while the record has times is a result to report, not a reason for `chunked`.
The record's times still select nothing.

### G2. The record may not exist when §3.1 and §5.3 read it, and then no table is written

- §1.2 writes `$OUT/timing/swa_paths.json` in the short verification session. Whether `$OUT`
  survives between sessions is checked only in §1.4 ("If `$OUT` does not survive between sessions,
  fix the storage before anything else"). No earlier step of §1 writes to `$OUT`.
- §3.1 and §5.3 pass `--swa-paths $OUT/timing/swa_paths.json`. With the file missing,
  `timing_report` ends with an uncaught `FileNotFoundError` traceback, exit 1, and writes neither
  `timing.md` nor `inference.md` (tried; the output directory was not even created). The timing pass
  itself is not lost.
- The instruction to run the script again is in §1.2 only, and only for a re-pinned image. §3.1,
  where Phil is weeks later, does not repeat it, and its "Check" paragraph does not mention the new
  section or its source line.
- No test would notice if the commands drifted: dropping `--swa-paths` from the §5.3 command (B01),
  writing to another path in §1.2 (B03) and a misspelt file name in §3.1 (B08) all pass.

Fix: in §3.1, before the report command: "if `$OUT/timing/swa_paths.json` is missing, or its source
line does not name this session's torch, CUDA and driver, run the §1.2 command now, alone on the
box"; add the SWA section to the §3.1 check; let `timing_report` refuse a missing file in one line;
a runbook test that every `timing_report` command reads the path the `swa_paths` command writes.

### G3. A flex failure is recorded as "could not run" whatever it was and whenever it happened

`scripts/swa_paths.py:188-191` catches every `Exception` of the flex track, including the build
(`_build` is inside the `try`). Tried with an injected `time_fn`:

- Flex succeeds in repetitions 1 and 2 and raises in repetition 3: the entry has null for every
  number and `timing.md` says "swa256/flex could not run here". It ran twice; the two measured
  repetitions are thrown away and the record does not say in which repetition it failed.
- An `AttributeError` (a harness bug) in the flex track is recorded the same way and the script
  exits 0. Only one line of text is kept, no traceback.
- `KeyboardInterrupt` is not swallowed. A failure of MHA or of the chunked track stops the script
  and nothing is written (correct).

Fix: record the repetition that failed and keep the completed repetitions in a field of their own
(or say "failed in repetition 3 of 3" in the note); keep the traceback in the JSON or print it to
stderr; do not print "could not run" for a track that ran.

### G4. `check_document` accepts records whose numbers contradict each other

`bench.check_documents` refuses a point whose median, min and max are not the statistics of its
repetitions, and a repetition list of the wrong length (`scripts/bench.py:436-448`).
`swa_paths.check_document` checks none of that. Each of these was accepted by
`timing_report --swa-paths` and printed:

| hand edit | printed |
|---|---|
| chunked median 0.001 with repetitions near 1.07 | `0.001 \| 1.068–1.309 \| 0.001` |
| MHA min 9.0, max 1.0 | `1.006 \| 9.000–1.000` |
| MHA `attention_ms_by_repetition: [100.0]` | unchanged row; the list is never compared |
| `repetitions: 30` with three values | "30 repetitions of 3 calls" |
| no `environment` | "GPU —; … torch —" |
| no `backend`, `device`, `precision`, `batch`, `context`, `repetitions`, `repeats`, `warmup`, `order` | a source line of dashes above real-looking numbers |
| `context: "1,024 (real)"`, `batch: "16"`, `precision: "fp16"` on the synthetic CPU record | printed as given |
| flex `error: ""` | "swa256/flex could not run here: " |
| a fourth track `swa256/dense` | ignored silently |

Refused, as they should be: shared GPU (also the string "false"), a missing track, an error on `mha`
or on `swa256/chunked`, a number as a string, NaN, Infinity, a bool, a flex track with an error and
numbers (also with an empty list), an error that is not text, another schema, a `timing_pass/1`
file, a JSON list, `null`, `tracks` as a list.

Fix: the three checks `bench` already makes (length equals `repetitions`, min ≤ median ≤ max, the
statistics of the list), required and typed header fields and environment, a non-empty error text.

### G5. Test gaps: breaks that pass the whole suite

From the 32 missed breaks, the ones that matter (ids refer to section 3):

- **A verdict can be added to `timing.md`.** X01 adds the line "The faster path is swa256/flex."
  under the filled table and every test passes. `JUDGEMENT_WORDS` in `tests/test_timing_report.py:38`
  has no `faster`, `slower`, `better`, and no test pins the lines of the filled section. S36 adds
  "(better)" to the script's printed lines and passes too (the list in `tests/test_swa_paths.py` has
  `faster` and `slower`, not `better`, `worse`, `best`, `quicker`).
- **What makes the record "the same measurement as the test" is not tested in any tier.** Removing
  the synchronisation before the clock starts (S06) or before it stops (S07), and removing the fp16
  autocast (S08), all pass. The `gpu` test would not catch S08 either: it asserts
  `doc["precision"] == "fp16"`, and that label is derived from the device type
  (`swa_paths.py:202`), not from what ran. S06 and S07 can be tested on CPU by replacing `sp._sync`
  and recording the order of events.
- **The numbers of decision 247 are not pinned.** Defaults of 8 for the batch, 20 calls, 3 warm-up
  calls and 1 repetition (S25 to S28) pass; so does `--repetitions 1` in the runbook command (B07).
- **The script may always time the tiny model.** X02 makes `--synthetic` permanent; every test
  passes, because no test builds without it. Building the three real modules at batch 1 takes
  seconds on CPU (I did it: 1024 × 384, `rope_dim` 48, θ 10000, the right path on each).
- **The header of the record is not checked against what ran.** A constant `context` (X03), the
  default `batch` instead of the one used (X05) and `warmup` written from `repeats` (X06) pass.
  `backend: None` (S21) is caught only by the `gpu` test.
- **The "FlexAttention is missing" case of decision 247 has no test.** Narrowing the catch to
  `RuntimeError` (S18) passes; a missing module raises `ValueError` from `set_opt_path`.
- **`check_document`**: NaN and Infinity (S29), numbers on the other three fields of an unavailable
  track (S32), bad values in the repetition list (S34), an error that is not text (X04) are refused
  by the code and by no test.
- **A refusal test that cannot fail.** The case `no-gpu-shared` passes `["--device", "cuda"]`, which
  also lacks `--expected-backend`, so removing the `--gpu-shared` requirement (S38) still exits 2.
- **Whiskers.** A whisker drawn symmetrically about the median, `2·median − max` to `max` (P04),
  passes, because the fixture's range is symmetric (`BENCH_SPREAD` 2% either side). A whisker of
  width 0, which is invisible (P10), passes.
- **Runbook.** The sentence that the script "selects nothing" can be replaced by "set `opt_path` to
  the faster one" (B04) without a failing test.

### M1. The "ratio to MHA" column is a comparison the decision did not ask for

Decisions 245 and 247 ask for the three times. `timing_report.py:418` adds each median divided by
MHA's. It carries no verdict word and the timing table above it has a ratio column too, but it was
added by the worker and logged in decision 249 as a detail. It is the only computed comparison in the
change. Keep it knowingly or drop it. If kept: "unrounded medians" (decision 249) has no test (R02
passes).

### M2. The error text goes into `timing.md` raw

`timing_report.py:411` prints `entry['error']` as it is. A multi-line error becomes several lines of
the document; a record whose error contained a table row printed that row under the table (tried). A
compile failure of `torch.compile` is typically many lines. Fix: print the first line, or collapse
white space; keep the full text in the JSON.

### M3. A synthetic CPU record is printed like a real one

The record has no `synthetic` field, no commit and no time. `timing_report` printed my
`--synthetic` CPU record (batch 4, context 32, fp32, no GPU) under the captions that speak of the
T4; only the source line shows what it is. `timing_pass` and `bench` have no such field either, but
they are cross-checked against the campaign's environment and this record, by decision 247, is not.

### M4. The section does not say which of the two paths SWA's rows above it used

The timing table's `swa256` row is "the variant on the path it trains with", and the
`timing_pass/1` document does not record `opt_path`. The new section states the rule, not the
outcome, so a reader of `timing.md` cannot tell which of the two rows corresponds to the table.
Decision 245's purpose ("the other path's time is known") needs that link.

### M5. Remaining differences from the `t4` test's measurement

The same in both: the clock starts after a synchronise and stops after one; the autocast encloses
the forward only and the backward runs outside it; `zero_grad` is outside the timed region; the
input's gradient accumulates; median of 30 calls after 10; three round-robin repetitions; 16 × 1024
× 384; RoPE dimension 48 and θ 10000 (checked by building the real modules on CPU).

Different:

- The script runs inside `backend_scope($BACKEND)`; the test enforces no backend. They agree only if
  the default choice on sm75 is the efficient kernel, as plan §2.3 says.
- The script builds the module and the input anew in every repetition; the test builds them once. For
  flex this means a new block mask and a new `mask_mod` closure each repetition, which may recompile
  inside the warm-up calls. Not verifiable on CPU.
- The script draws the input from a seeded CPU generator; the test from the global CUDA generator.
  Immaterial for time.
- The test asserts `flex_compiled_available()` first; the script does not, and does not record
  whether the flex track ran compiled.

### M6. Legibility of the whiskers

Looked at the PNGs.

- Plot 2: the whiskers are there, at the height of the seed mean, in the variant's colour, and the
  caption says what they are. The variant's label sits to the right of the diamond at the same
  height, so the right half of each whisker runs through its own label ("SWA-256", "GQA-4", "MHA"
  are struck through), and with nearby variants a whisker crosses a neighbour's label.
- Plot 8: with the fixture's 2% range no whisker is visible; the caption says a short whisker is
  hidden by its marker. With a wide range the `opt` and `ref` whiskers of a variant share the x
  position, the colour and the line style, and all 14 series share each x, so only the outermost
  ends can be read.
- Blank plots 2 and 8 draw no whisker and say "no data yet".
- Axis limits widen to include the whiskers (plot 2 x: 13140–17073 becomes 12810–17468; plot 8 y
  slightly). Dot counts (29 and 0) and line counts (7 and 56) are unchanged against `c2b1156`.

### M7. Smaller points

- A file that is not JSON gives "Expecting property name …" without the file's name.
- `--blank --swa-paths FILE` ignores the file, as documented.
- `RUNBOOK.md:96` still says "Record both attention times from the failure message"; the script now
  does that.
- `scripts/swa_paths.py` is the first file outside `src/attention/` that calls `set_opt_path`. It is
  a measurement script like `timing_pass.py`, so I do not read it as a breach of plan §4.1.

### Classes in which I found nothing

- Frozen files: `git diff c2b1156 HEAD` touches none of `docs/plan_v5.md`, `configs/`,
  `tests/tolerances.py`, `README.md`.
- No literal tolerance in the new tests (`FLOAT64_IDENTITY`, `AGREEMENT_FP64` from
  `tests/tolerances.py`); no `skip`, `xfail` or loosened assertion in the diff.
- No code reads the record to choose a path: the only reader of `swa_paths` is `timing_report`, and
  `opt_path` is read nowhere in `scripts/` or `src/` outside `src/attention/sliding_window.py` and
  the new script.
- No judgement word in the generated output as it stands, the ratio column apart.
- The blank `timing.md` generated here is byte-identical to the committed one, and a test compares
  them (B05, B06 caught). `scripts.prune` leaves the record alone.
- Whiskers: right field, right row (batch 32, cache 4096), right panel; with the gate shut plot 2
  draws MHA's whisker only. Plot 8 draws every variant's range with the gate shut, exactly as it
  already drew every variant's median there.
- `docs/decisions.md`: one row appended (249).

## 3. Deliberate breaks

86 breaks: **54 caught, 32 missed**. Of the misses, 4 are equivalent or cosmetic (S24, S40, P09,
P16), 1 is caught only by the unrun `gpu` test (S21), 27 are real gaps (G2, G5, M1).

Tests run after each: S, R, X: `test_swa_paths.py`, `test_timing_report.py`, `test_runbook.py`;
P: `test_plots.py`; B: `test_runbook.py`, `test_timing_report.py`, `test_swa_paths.py`.

### `scripts/swa_paths.py` (42: 26 caught, 16 missed)

| id | break | result | note on a miss |
|---|---|---|---|
| S01 | repetition value is the mean of the timed calls | caught | |
| S02 | entry is the mean of the repetitions | caught | |
| S03 | `attention_ms_min` is the max | caught | |
| S04 | min over all timed calls, not over repetition medians | caught | |
| S05 | no warm-up calls | caught | |
| S06 | no synchronise before the clock starts | **missed** | real gap; testable on CPU |
| S07 | no synchronise before the clock stops | **missed** | real gap; testable on CPU |
| S08 | no fp16 autocast on CUDA, document still says fp16 | **missed** | real gap; the `gpu` test would pass too |
| S09 | no backward | caught | |
| S10 | `ref` mode timed | caught | |
| S11 | flex track runs chunked | caught | |
| S12 | `--batch` ignored | caught | |
| S13 | input needs no gradient | caught | |
| S14 | seconds written as milliseconds | caught | |
| S15 | failed flex track tried again every repetition | caught | |
| S16 | chunked failure swallowed | caught | |
| S17 | MHA failure swallowed | caught | by the document-check test, not by a run |
| S18 | only `RuntimeError` recorded | **missed** | real gap; a missing FlexAttention raises `ValueError` |
| S19 | error without its type | caught | |
| S20 | precision label swapped | caught | |
| S21 | backend not recorded | **missed** | `gpu` tier only |
| S22 | backend not enforced | caught | |
| S23 | shared GPU not refused | caught | |
| S24 | document not self-checked before writing | **missed** | equivalent for a well-formed document |
| S25 | default batch 8 | **missed** | real gap |
| S26 | default 20 timed calls | **missed** | real gap |
| S27 | default 3 warm-up calls | **missed** | real gap |
| S28 | default 1 repetition | **missed** | real gap |
| S29 | NaN and Infinity accepted by the check | **missed** | real gap in the tests; the code refuses them |
| S30 | zero accepted | caught | |
| S31 | error accepted on MHA and chunked | caught | |
| S32 | unavailable track: only `attention_ms` must be null | **missed** | real gap |
| S33 | missing or null `gpu_shared` accepted | caught | |
| S34 | values of the repetition list not checked | **missed** | real gap |
| S35 | reader accepts any schema | caught | |
| S36 | printed line says "(better)" | **missed** | real gap in the word list |
| S37 | track order swapped | caught | |
| S38 | `--gpu-shared` not required on CUDA | **missed** | real gap; the test case also lacks `--expected-backend` |
| S39 | `attention_ms_min` not checked | caught | |
| S40 | another input seed | **missed** | equivalent |
| S41 | `gpu_shared` written as null | caught | |
| S42 | track-major order instead of round-robin | caught | |

### `scripts/timing_report.py` (14: 13 caught, 1 missed)

| id | break | result | note on a miss |
|---|---|---|---|
| R01 | ratio inverted | caught | |
| R02 | ratio from rounded medians | **missed** | gap; the hand document's values are exact |
| R03 | range printed max–min | caught | |
| R04 | report does not check the record | caught | |
| R05 | error text dropped from the note | caught | |
| R06 | ratio to chunked, not to MHA | caught | |
| R07 | repeats and warm-up swapped in the source line | caught | |
| R08 | `SwaPathsError` not caught by `main` | caught | |
| R09 | `main` never passes the record to the table | caught | by the refusal test only |
| R10 | median column shows the smallest repetition | caught | |
| R11 | the captions "selected nothing" and "own source line" dropped | caught | |
| R12 | blank form lacks the flex row | caught | |
| R13 | `--swa-paths` ignored by `main` | caught | by the refusal test only |
| R14 | source line prints python's version as torch's | caught | |

### `scripts/make_plots.py` (16: 12 caught, 4 missed)

| id | break | result | note on a miss |
|---|---|---|---|
| P01 | plot 2 whisker starts at the median | caught | |
| P02 | plot 2 whisker at the seed median's height | caught | |
| P03 | plot 2 whisker computed from ms per token | caught | |
| P04 | plot 2 whisker symmetric about the median | **missed** | real gap; the fixture's range is symmetric |
| P05 | plot 8 prefill whisker starts at the median | caught | |
| P06 | plot 8 decode whisker ends at the median | caught | |
| P07 | plot 8 decode whiskers all on the batch-1 panel | caught | |
| P08 | plot 8 whiskers grey | caught | |
| P09 | plot 2 whisker drawn over the markers | **missed** | cosmetic |
| P10 | whiskers of width 0 | **missed** | real gap; invisible whiskers pass |
| P11 | caption says "first" for "smallest" | caught | |
| P12 | plot 8 `ref` whiskers tagged `opt` | caught | |
| P13 | plot 8 whiskers fully transparent | caught | |
| P14 | plot 8 prefill whisker at the first context only | caught | |
| P15 | plot 2 whiskers all in MHA's colour | caught | |
| P16 | plot 8 whisker line style changed | **missed** | cosmetic |

### `RUNBOOK.md` and the blank form (8: 3 caught, 5 missed)

| id | break | result | note on a miss |
|---|---|---|---|
| B01 | §5.3 report command without `--swa-paths` | **missed** | real gap; the section would be silently blank |
| B02 | §1.2 command without `--gpu-shared` | caught | |
| B03 | §1.2 writes to a path §3.1 does not read | **missed** | real gap |
| B04 | "selects nothing" replaced by "set `opt_path` to the faster one" | **missed** | real gap |
| B05 | a number typed into the blank `timing.md` | caught | |
| B06 | the blank form's caption edited by hand | caught | |
| B07 | §1.2 command with `--repetitions 1` | **missed** | real gap |
| B08 | §3.1 reads a misspelt file name | **missed** | real gap |

### Further breaks chosen after reading the tests (6: 0 caught)

| id | file | break | result | note |
|---|---|---|---|---|
| X01 | `timing_report.py` | a line "The faster path is …" under the filled table | **missed** | real gap; the most serious one |
| X02 | `swa_paths.py` | the script always times the synthetic model | **missed** | real gap |
| X03 | `swa_paths.py` | the document says context 1024 whatever was timed | **missed** | real gap |
| X04 | `swa_paths.py` | an error that is not text accepted by the check | **missed** | real gap in the tests |
| X05 | `swa_paths.py` | the document records the default batch, not the one used | **missed** | real gap |
| X06 | `swa_paths.py` | the document records `repeats` as `warmup` | **missed** | real gap |

## 4. What I did not check

- Nothing on CUDA. The fp16 autocast, the synchronisation, the enforced backend, compiled
  FlexAttention, whether rebuilding the module each repetition recompiles it, and memory are read
  from the code only. The 106 hardware tests are skipped, not passed.
- I did not time the real model. I built the three real modules on CPU at batch 1 to check shapes and
  paths; the timed runs used `--synthetic` (context 32, where the 256-token window covers everything).
- G1 is an argument from the tests' code and from the CPU behaviour of torch 2.15.0. Whether
  FlexAttention's backward works on sm75 is unknown.
- Whether `$OUT` on Kaggle survives between sessions, and how Phil's sessions are laid out.
- The `cpu` tier ran once, before the breaks. After the breaks I ran only the relevant files and
  checked that the worktree is clean; I did not run the full tier again.
- CI, the pull requests #66 and #67, and how `main` came to move during the review.
- The other eight plots, the other tables, and `README.md` beyond the fact that the diff does not
  touch it.
- Plan sections other than §2.2, §2.3, §4.1, §5, §10.4, §10.5 and §13.
