# RUNBOOK

Every GPU job of plan §15, in order, with the gate after each block and what to check there. The build loop ran none of this. The commands are checked against the scripts' argument parsers by `tests/test_runbook.py`; nothing else about them has run on a GPU.

Read plan §11 and §13 before the first run. Three rules that no command enforces:

- Do not open a variant's manifest, loss curve or table before the gate that allows it. `make_tables.py` refuses to compute comparisons before the controls, but it cannot stop you reading a JSON file.
- Never change `configs/recipe.yaml`, `configs/prereg.yaml` or `tests/tolerances.py` after their freeze. A later change restarts every arm (plan §7, §13.6).
- Never tune anything per variant, never add seeds after a variant result at 300M has been read, never stitch a timing run across sessions.

## Open items

Phil answered the questions this runbook used to list here on 2026-11-13; the answers are `docs/decisions.md` rows 215 to 231 and plan §18. What is left is yours to fill in while running:

| Item | Where | When |
| --- | --- | --- |
| The Kaggle image's torch, Python and CUDA versions. | §1.5 | Before the first run |
| `opt_path` in `configs/variants/swa256.yaml`: `flex` if FlexAttention runs on the T4, else it stays `chunked`. | §1.2 | Before any SWA run |
| `micro_batch`, `init_scale` and `growth_interval` in `configs/recipe.yaml`. | §2.2 | After the smoke runs |
| The dataset and tokenizer hashes in `configs/data.yaml`. | §2.1 | After tokenizing |
| The probe settings the pilot fixes (width, rate, steps per size), and the sizes of block 2 if you run it. | §2.3 | Before block 1 |
| `ladder` in `configs/commitments.yaml`, only if you cut L10. | §3.4 | Before any L10 or seed-count result is read |
| `seed_count` in `configs/commitments.yaml`. | §3.4 | Before the first primary run of week 3 |

## 0. Conventions

Set these once per shell. Paths are examples; `OUT` must be on storage that survives the session, because every run resumes from its checkpoints there (plan §12.3).

```bash
export DATA=/kaggle/input/attention-benchmark-tokens   # the prepared dataset, read-only
export OUT=/kaggle/working/out                         # checkpoints, manifests, tables, plots
export BACKEND=efficient                               # the fused kernel on sm75 (plan §2.3)
export SHARED=no                                       # yes while a second job uses the same GPU
export SWEEP_SEED=21                                   # lr_sweep.init_seed of configs/prereg.yaml: the seed of every sweep run, used by no arm
```

- A T4 ×2 session runs two independent jobs, one per GPU, each started with `CUDA_VISIBLE_DEVICES=0` or `=1` in front of the command. No DDP (plan §2.2). While two jobs run on one GPU, or when Kaggle co-locates them, set `SHARED=yes`: the manifest records it. Jobs that produce a timing number run alone on the box and refuse a shared GPU.
- Every `train` and `branch` command is idempotent: run the identical command again after a session dies and it resumes from the newest state the run has. A run stopped with `--until` leaves a checkpoint at that step, so nothing is run twice. A finished run exits 0 and changes nothing. A run recorded as diverged exits 3 and is not retried until you delete its manifest.
- Start a job again with the same `CUDA_VISIBLE_DEVICES` setting it started with: a checkpoint holds the random state of every visible GPU, and a resume under another number of visible devices is refused.
- Exit status 2 is a refusal (a hash mismatch, an incomplete recipe, a directory that is not a git checkout, a checkpoint of another mode, a checkpoint written under another torch, CUDA, driver or GPU). Read the message; do not work around it. A run cannot continue after the image changed: pin the environment (§1.5) before the first run.
- Budgets are in steps: 100M = 1,526, 200M = 3,052, 300M = 4,578. The 100M branch leaves the trunk at step 1,221 and the 200M branch at step 2,442.
- A learning rate appears in run names without the minus sign: `1.2e3` means 1.2e-3. Type it on the command line as `1.2e-3`.
- Manifests land in `$OUT/manifests/opt/`. Tables and plots read that directory.
- After any block: `python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables --skip-unfinished` rebuilds every table from what exists. It is safe at any time: it computes only what its inputs and the gates allow, and says in the notes what it skipped and why.
- The shim notebook writes to `/kaggle/working/out`, the `OUT` of this runbook, so a run started by either is resumed by the other. It calls `scripts.train` with a run spec and nothing else. For a `branch`, a timing pass or the benchmark in a committed run, replace the module and its arguments on the notebook's `try:` line, in your copy on Kaggle, and change nothing else; or run them in an interactive session.

### GPU utilisation log

Start this once per session, before the first job (plan §18, A17). It appends one line per GPU every 10 seconds to a file of its own and stops with the session:

```bash
mkdir -p $OUT/logs
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 10 \
  >> $OUT/logs/gpu_util_$(date -u +%Y%m%dT%H%M%SZ).csv &
```

It is a raw record for the writeup: nothing in the harness reads it, no manifest holds it, and a run does not depend on it. The shim notebook starts the same log in its own cell. Keep the files; they cannot be made again after the runs.

### Disk

A checkpoint of the real model is about 208 MB. A run keeps its newest two and every branch point, so week 1 alone approaches Kaggle's 20 GB. Prune after every block:

```bash
python -m scripts.prune --out $OUT
python -m scripts.prune --out $OUT --apply
```

The first command deletes nothing: it lists every file it would delete and every file or directory it leaves alone, with the reason. Read it, then run the second. It removes the non-final checkpoints of completed runs and branches, and a branch-point checkpoint once its branch has a completed, valid manifest and the trunk itself is complete. It never touches an unfinished or diverged run, a directory no manifest accounts for, or a manifest. Final checkpoints stay by default, and they add up: one per completed run and branch is about 44 files after week 1, 78 after week 2 and 108 after week 3, which is 22 GB. After each block's tables are built, also run:

```bash
python -m scripts.prune --out $OUT --final-short
python -m scripts.prune --out $OUT --final-short --apply
```

`--final-short` also removes the final checkpoint of every completed branch and of every completed run shorter than 300M (sweep points, the null control, data-order runs, reruns, Arm B, shared-rate runs). Nothing reads those again: their results are in their manifests. The final checkpoints of the 300M runs stay: 28 files after week 3 (40 if the rule picked 8 seeds) and at most 43 after week 4, about 9 GB; `evaluate.py` and a later Phase 6 load them. `--final` removes those too: use it only when the tables are final and Phase 6 is decided.

## 1. Verification (the 1070 and one short T4 session, before any reported GPU hour)

### 1.1 The `gpu` tier, on the 1070

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -m gpu -q
```

Check: every test passes; none is skipped. These tests have never run. A failure here is a finding about the harness, not a flake: stop and fix it. `STATUS.md` lists them by file. The one exception is `tests/test_sliding_window.py::test_opt_paths_match_ref_on_cuda[flex]`: it fails if FlexAttention does not compile or run on this GPU. On the 1070 that is a result to record, not a defect, and it decides nothing; what counts is the T4 (§1.2).

### 1.2 The `t4` tier, in one short Kaggle session

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -m "gpu or t4" -q
```

Check: every test passes on the T4, with these exceptions, which are results, not defects, and do not stop the session:

- `tests/test_sliding_window.py::test_opt_paths_match_ref_on_cuda[flex]` fails, or `tests/test_sliding_window.py::test_swa_attention_time_at_ctx_1024_is_below_mha[flex]` ends in an error raised inside FlexAttention (not at the test's last `assert`): FlexAttention does not compile or run on sm75. Record it.
- `tests/test_sliding_window.py::test_swa_attention_time_at_ctx_1024_is_below_mha[chunked]` or `[flex]` fails at its last `assert`: that path ran and its attention time was not below MHA's at context 1,024. For the chunked path plan §2.3 itself expects it ("treat the fallback's speedup as near zero"). Both attention times are in the failure message and in the record below. This changes nothing about the path; the timing pass of §3.1 will then mark SWA's timings (plan §18, A15).

Any other failure is a finding about the harness: stop and fix it. In particular the backend selection and GQA/MQA decode time below MHA's at cache 8k must pass. Record whether the fused kernel copies K and V in decode (plan §2.3); if it does, decode needs the explicit matmul, disclosed.

Then record the attention times of both SWA paths in the same session, alone on the box (decision 245):

```bash
python -m scripts.swa_paths --device cuda --gpu-shared false --expected-backend $BACKEND --out $OUT/timing/swa_paths.json
```

Check: the script exits 0 and prints one line per track: MHA, SWA `chunked`, SWA `flex`, each with its median and range, or `not available` with the error. A flex track that cannot run is recorded with its error and the repetition it failed in, the traceback is printed, and the script still exits 0. `$OUT` has to survive the session (§1.4); the command is run again in §3.1, so nothing is lost if it does not.

Which path SWA trains with follows plan §2.3, where FlexAttention is the first choice and the chunked path the fallback (decisions 244 and 250). FlexAttention runs on the T4 when both of these hold:

1. `test_opt_paths_match_ref_on_cuda[flex]` passes: it compiles, and its forward agrees with the reference in fp32 and fp16.
2. The record's `swa256/flex` track has times, not an error: a forward and a backward in fp16 at the batch and context of training went through. A training step needs the backward, and the test of item 1 runs none.

If both hold, set `opt_path: flex` in `configs/variants/swa256.yaml` by pull request before any SWA run; otherwise leave `chunked` and record why. The value is part of the configuration hash, so it does not change afterwards. The record selects nothing beyond item 2: its times, their order, and a timing case that fails at its last `assert` do not enter the choice.

### 1.3 Quota

Start a T4 ×2 session, note the quota counter, wait ten minutes, note it again. Check: whether two GPUs draw quota at the single or the double rate (plan §2.1). The schedule fits either way; the answer sets how much slack a week has.

### 1.4 Cold-start dry run of the shim

Upload `notebooks/kaggle_shim.ipynb` to Kaggle, fill its placeholders there (repository URL, pinned commit, dataset name, run spec, `GPU_SHARED`, `EXPECTED_BACKEND`) and commit it on Kaggle ("Save & Run All") so that it runs from the top in a fresh container. The notebook in the repository stays the template with its placeholders: do not commit a filled-in copy to git. Its tests pin the template (the call of `scripts.train`, and no word that looks like a credential, which a dataset name such as `...-tokenized` would trip). A run spec for a job is a new file under `configs/runs/`, committed to git like any config. It can only run after §2.1, because nothing starts without the dataset hashes.

Check: the run starts from step 0, writes checkpoints under `$OUT`, and a second commit of the same notebook resumes without input. If `$OUT` does not survive between sessions, fix the storage before anything else.

### 1.5 Pin the environment

Record the image's versions and pin the notebook to that environment (plan §12.3):

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

Check: send the output to the build loop (or pin `torch` in `pyproject.toml` and `.github/workflows/ci.yml` to it and see the `cpu` tier pass on that version). The tables refuse runs whose torch, CUDA or driver versions differ.

### 1.6 Structure checks (CPU, a minute)

```bash
python -m scripts.solve_config --all
python -m scripts.graph_diff
```

Check: 17,308,032 parameters for every non-MLA variant, the two MLA offsets as in plan §3.2, and no difference outside the attention modules.

### 1.7 Pre-registration

The two thresholds in `configs/prereg.yaml` are confirmed (0.6 and 0.05), and the amendments to the plan are its §18. Commit the pre-registration section of `README.md` with a timestamp and write the commit hash into its first line: the §1.2 rule, the three primaries, the seed-count rule, the positive-control test, the learning-rate rules, the context-use metrics. `configs/prereg.yaml` is frozen from that commit on. The detectable-effect bound and the seed count are added in §3.4; the seed count also goes into `configs/commitments.yaml`.

## 2. GPU week 1: data, recipe, noise floor (about 25 GPU-h)

### 2.1 Tokenize once

On a CPU machine (hours for 400M tokens; plan §6):

```bash
python -m scripts.prepare_data --inputs $SHARDS --tokenizer-inputs $TOKENIZER_SHARD --out $DATA_BUILD
```

`$SHARDS` is the fixed, ordered list of corpus files and `$TOKENIZER_SHARD` the one the tokenizer is trained on. The script prints two hashes. Paste them into `configs/data.yaml` (`dataset_sha256`, `tokenizer.sha256`) and commit. Upload `$DATA_BUILD` as a versioned Kaggle Dataset and mount it read-only as `$DATA`.

Check: `train`, `branch` and `evaluate` refuse to start until both hashes are set, and refuse any dataset whose hashes differ. Never re-tokenize.

### 2.2 Smoke runs, all seven variants, then the recipe freeze

`configs/recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset. Find them here, with overrides. A smoke run is the first 100 steps of a real 300M run, stopped with `--until 100`, so its warmup is that of the real run. (A separate 100-step run would start its decay at step 80.) No learning rate is known yet, and skipped steps during the warmup depend on the peak rate, so the smoke runs are made at both ends of MHA's sweep grid, 3e-4 and 4.8e-3 (plan §18, A13).

First the micro-batch: the largest that fits MLA-192 in `ref` mode (plan §7.1). Try 64, then 32, 16, 8; twenty steps are enough to see whether it fits (memory does not depend on the rate; 1e-3 is a placeholder here):

```bash
python -m scripts.train --variant mla192 --init-seed 1 --peak-lr 1e-3 --until 20 \
  --mode ref --data $DATA --out $OUT/smoke_ref --gpu-shared $SHARED --expected-backend $BACKEND \
  --override recipe.batch.micro_batch=$MICRO \
  --override recipe.grad_scaler.init_scale=$INIT_SCALE \
  --override recipe.grad_scaler.growth_interval=$GROWTH
```

Then every variant in `opt` mode with that micro-batch and one global pair of scaler settings (start from PyTorch's defaults, `INIT_SCALE=65536`, `GROWTH=2000`):

```bash
for SMOKE_LR in 3e-4 4.8e-3; do
  for v in mha gqa4 gqa2 mqa mla192 mla96 swa256; do
    python -m scripts.train --variant $v --init-seed 1 --peak-lr $SMOKE_LR --until 100 --eval-every-steps 50 \
      --data $DATA --out $OUT/smoke --gpu-shared $SHARED --expected-backend $BACKEND \
      --override recipe.batch.micro_batch=$MICRO \
      --override recipe.grad_scaler.init_scale=$INIT_SCALE \
      --override recipe.grad_scaler.growth_interval=$GROWTH
  done
done
```

`--eval-every-steps 50` makes each smoke run evaluate twice (the real cadence is 250 steps, which a smoke run never reaches): the evaluation runs at batch 64, not at the micro-batch, so this is where an out-of-memory in evaluation shows. It is an operational setting and does not enter the configuration.

Each of the 14 runs stops at step 100 and writes an unfinished manifest. Check, in each manifest under `$OUT/smoke/manifests/opt/`: `steps.attempted` is 100 and `steps.skipped_in_first_100` is at most 3, for every variant at both rates (plan §7.3); `backend` is the expected one; `precision` is `fp16`; no run exited with a backend or dtype error. If any variant skips more at either rate, change the global scaler settings and rerun all fourteen into a fresh directory. Never tune per variant. A variant that cannot be made stable in fp16 is a reportable finding (plan §13.6). Also read `step_seconds` and rescale the budget of plan §12.1 if a 300M run will not take about 80 minutes.

After each sweep (§2.4, §2.7), look at `steps.skipped_in_first_100` in the manifest of every run at a chosen rate; the same limit applies. This matters most when the edge rule adds a rate above 4.8e-3, which no smoke run covered. A run over the limit is a question to settle before the arms start: the settings are global and frozen, and are never changed for one variant.

Then write the three values into `configs/recipe.yaml` (a pull request you approve) and freeze it. From here on no command carries an `--override`: the tables refuse any run made with one. The smoke directories are not results; do not point the tables at them, and do not resume them.

Record the run-to-run reproducibility of `opt` mode once, for the writeup (plan §5):

```bash
python -m scripts.measure_reproducibility --variant mha --steps 200 --pairs 3 --device cuda --mode opt \
  --data-dir $DATA --peak-lr 1e-3 --micro-batch $MICRO --init-scale $INIT_SCALE --growth-interval $GROWTH \
  --out $OUT/reproducibility/mha_opt.json
```

It trains four identical 200-step runs and writes how far they differ. It judges nothing; keep the file.

### 2.3 Phase 1 probes, Gate 1 (about 3 GPU-h; block 2 only if quota allows)

The grid (plan §18, A12). Block 1 decides Gate 1: MQAR at 4, 16 and 64 pairs, the seven variants and the recurrence anchor, init seeds 1 to 10, 240 runs. Block 2 is probes 2 to 5 at one size each, the same eight models and seeds, 400 runs (probe 5 is two tasks, selective copy and induction); run it only if the week's quota allows. Every model of a point gets the same width, step count and learning rate. `scripts/train_probe.py` trains one model and writes one record; a record that exists is not trained again, so every loop below can be restarted.

**Pilot, MHA and the anchor only.** It fixes the width, the rate and the steps per size, and it times a run. Its seeds, 101 and 102, are not part of the grid. Start from the settings that solved 4 pairs on CPU in the test suite (width 64, rate 3e-3) and from a first guess for the steps:

```bash
export PROBE_D=64 PROBE_LR=3e-3
export STEPS_4=4000 STEPS_16=8000 STEPS_64=16000        # first guesses; the pilot corrects them
for n in 4 16 64; do
  eval PROBE_STEPS=\$STEPS_$n
  for m in mha recurrence; do
    for s in 101 102; do
      python -m scripts.train_probe --variant $m --task mqar --param n_pairs=$n --d-model $PROBE_D --seed $s \
        --steps $PROBE_STEPS --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes_pilot
    done
  done
done
python -m scripts.probe_report --probes $OUT/probes_pilot --out $OUT/tables_pilot
```

The rule for the pilot, fixed here before any probe result exists:

- A size is settled when MHA's held-out token accuracy in `$OUT/tables_pilot/probes.md` is at least 0.99 on both pilot seeds. (The 0.99 is the build loop's proposal; change it here, before the pilot, if you want another level.)
- If a size is not settled, double its `STEPS_` value and run the pilot again for that size. After two doublings without success, set `PROBE_D=128` for every size and start the pilot again from the first guesses.
- The anchor's pilot runs decide nothing. They show its level and its run time.
- Budget: from the run times, 240 runs must fit about 3 GPU-h, a mean of 45 s per run. If they do not, that is your decision to make before block 1: raise the budget or drop the 64-pair point. Do not shorten the steps below what the rule settled.

Write the settled `PROBE_D`, `PROBE_LR`, `STEPS_4`, `STEPS_16` and `STEPS_64` into `docs/decisions.md` and commit before block 1. The pilot directory is not part of any report.

**Block 1.**

```bash
for n in 4 16 64; do
  eval PROBE_STEPS=\$STEPS_$n
  for m in mha gqa4 gqa2 mqa mla192 mla96 swa256 recurrence; do
    for s in 1 2 3 4 5 6 7 8 9 10; do
      python -m scripts.train_probe --variant $m --task mqar --param n_pairs=$n --d-model $PROBE_D --seed $s \
        --steps $PROBE_STEPS --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes
    done
  done
done
python -m scripts.probe_report --probes $OUT/probes --out $OUT/tables
python -m scripts.make_plots --manifests $OUT/manifests/opt --probes $OUT/probes --out $OUT/plots --skip-unfinished
```

Check `$OUT/tables/probes.md` and `09_probes.png`: every model has 10 seeds at each size; the token accuracy of each softmax variant; the anchor's token accuracy at 4, 16 and 64 pairs against the no-recall level 1/n drawn in the plot. The report prints numbers and no verdict.

**Gate 1** (plan §9.2): passes when the recurrence anchor fails as the number of pairs grows, which shows the probes can discriminate. You judge that from the numbers; the plan gives no threshold. SWA's behaviour is reported, not gating. If even the anchor does not fail, the probes are broken and are redesigned. Either way the LM arms run.

**Block 2, optional.** One size per probe. The sizes below are the build loop's proposal (16, the middle size of block 1, wherever a probe has pairs); they are yours to change before the first block-2 run, and not after. Probe 3 is MQAR with 1,024 keys and values at width 128, which is head dimension 16. Settle the steps for each with the same pilot rule (MHA and the anchor, seeds 101 and 102) before the loop:

```bash
for m in mha gqa4 gqa2 mqa mla192 mla96 swa256 recurrence; do
  for s in 1 2 3 4 5 6 7 8 9 10; do
    python -m scripts.train_probe --variant $m --task multihop --param n_chains=16 --d-model $PROBE_D --seed $s \
      --steps $STEPS_HOP --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes
    python -m scripts.train_probe --variant $m --task mqar --panel mqar_high_cardinality \
      --param n_pairs=16 --param n_keys=1024 --param n_values=1024 --d-model 128 --seed $s \
      --steps $STEPS_CARD --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes
    python -m scripts.train_probe --variant $m --task fuzzy --param n_pairs=16 --d-model $PROBE_D --seed $s \
      --steps $STEPS_FUZZY --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes
    python -m scripts.train_probe --variant $m --task selective_copy --param n_data=16 --param n_blank=64 \
      --d-model $PROBE_D --seed $s \
      --steps $STEPS_COPY --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes
    python -m scripts.train_probe --variant $m --task induction --param length=64 --d-model $PROBE_D --seed $s \
      --steps $STEPS_INDUCTION --peak-lr $PROBE_LR --device cuda --expected-backend $BACKEND --out $OUT/probes
  done
done
python -m scripts.probe_report --probes $OUT/probes --out $OUT/tables
```

### 2.4 MHA learning-rate sweep (5 runs at 100M, plus edge points)

```bash
for LR in 3e-4 6e-4 1.2e-3 2.4e-3 4.8e-3; do
  python -m scripts.train --variant mha --init-seed $SWEEP_SEED --peak-lr $LR --budget 100M --branches \
    --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
python -m scripts.select_lr --manifests $OUT/manifests/opt --out $OUT/tables
```

`select_lr` reads the sweep runs on the registered seed, `lr_sweep.init_seed` of `configs/prereg.yaml`; `SWEEP_SEED` must be that number. Check `$OUT/tables/lr_sweep.md`, section `mha`:

- Status `incomplete`: the listed rates are missing; run them.
- Status `extend`: the winner is at the edge of the grid. Run the one rate given as the next point and build the report again. Repeat until the status is `chosen` (plan §7.4).
- Status `chosen`: note the rate and the `lr_source` run name.

```bash
export MHA_LR=1.2e-3                                    # the rate lr_sweep.md chose, as a number
export MHA_SRC=mha-isoparam-100M-lr1.2e3-s21-d1         # its lr_source line
```

A diverged point is recorded and listed with its reason; it is not an error. Do not pick a rate by eye: the report is the choice.

### 2.5 Arm N (plan §8.1)

MHA seeds 1 to 5 up to their 100M branch point, then their 100M branches. The trunks are configured for both branch points now, because the ladder needs the 200M branch of the same runs later (plan §12.1). A trunk stopped at step 1,221 leaves its branch-point file `trunk-step00001221.pt` and a regular checkpoint at the same step; the branch starts from the first, and §3.3 continues from there. A branch-point file is never overwritten:

```bash
for s in 1 2 3 4 5; do
  python -m scripts.train --variant mha --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --branches 100M 200M --until 1221 \
    --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  python -m scripts.branch --variant mha --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --branches 100M 200M --budget 100M \
    --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
```

Reproducibility reruns (seeds 1 to 3), data-order runs (init seed 1, data seeds 2 and 3) and the null control (seeds 11 to 15), all standalone 100M runs at MHA's rate:

```bash
for s in 1 2 3; do
  python -m scripts.train --variant mha --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --budget 100M --branches --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
for d in 2 3; do
  python -m scripts.train --variant mha --init-seed 1 --data-seed $d --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --budget 100M --branches --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
for s in 11 12 13 14 15; do
  python -m scripts.train --variant null --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --budget 100M --branches --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
```

### 2.6 Gate 2

```bash
python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables --skip-unfinished
python -m scripts.make_plots --manifests $OUT/manifests/opt --out $OUT/plots --skip-unfinished
```

Check `$OUT/tables/gates.md`:

- "Noise floor at 100M": `sigma_init`, the second `sigma_init` from the null seeds, `sigma_data`, `s_rep`, the ratio against the threshold, and the test chosen. None is blank.
- "Null control": the verdict is `not detected`. If the report opens with **NULL CONTROL DETECTED**, stop: there is a bug in pairing, seeding, the statistics or determinism, and nothing downstream is valid until it passes (plan §9.3).
- The notes: no run is reported missing for the test selection or the null control. Steps taken agree within 0.5% and the backend and versions agree, or the script would have refused.

Check `$OUT/plots/04_noise_floor.png`: every Arm N and null-control run is a dot.

**Gate 2** (plan §9.3): commit the test choice to the pre-registration, draft the noise-floor post, and commit its publication date.

### 2.7 Sweeps for the other variants and both ladder sizes (3 runs each at 100M, plus edge points)

For each of `gqa4 gqa2 mqa mla192 mla96 swa256`, at half, one and two times `MHA_LR`. With `MHA_LR=1.2e-3` the three rates are `6e-4 1.2e-3 2.4e-3`:

```bash
for v in gqa4 gqa2 mqa mla192 mla96 swa256; do
  for LR in $LR_HALF $MHA_LR $LR_DOUBLE; do
    python -m scripts.train --variant $v --init-seed $SWEEP_SEED --peak-lr $LR --budget 100M --branches \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
for l in mha_L6 mha_L10; do
  for LR in $LR_HALF $MHA_LR $LR_DOUBLE; do
    python -m scripts.train --variant mha --ladder $l --init-seed $SWEEP_SEED --peak-lr $LR --budget 100M --branches \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
python -m scripts.select_lr --manifests $OUT/manifests/opt --out $OUT/tables
```

Check `lr_sweep.md`, one section per name: follow `incomplete` and `extend` as in §2.4 until every section says `chosen`. The rule that decided is printed: "tie rule" means the variant's best rate beat MHA's by less than `sigma_init` and MHA's rate is used (plan §7.4). Record, for every name, the rate and the `lr_source` run. The shared-recipe and tuned losses are printed side by side; that gap is itself a finding to report.

This is the only place a variant's loss is read before Gate 3, and only at 100M on the sweep seed.

## 3. GPU week 2: ladder and positive control (about 19 GPU-h)

### 3.1 Timing pass (alone on the box, one session, never stitched)

```bash
python -m scripts.timing_pass --peak-lr $MHA_LR --data-dir $DATA --device cuda --gpu-shared false \
  --expected-backend $BACKEND --with-ref --out $OUT/timing/timing_pass.json
```

With `--with-ref` and the attention-share runs the pass is about 8,800 steps, not the 4,620 behind the 1.5 GPU-h of plan §12.1; budget about 3 GPU-h (plan §18, A14). It reads the dataset, so it runs after §2.1.

Record SWA's two paths again in this session, after the pass and alone on the box (a minute). The record of §1.2 may be gone, or may come from an image that was pinned afterwards (§1.5); this run replaces it, so that the record and the timing pass name the same environment:

```bash
python -m scripts.swa_paths --device cuda --gpu-shared false --expected-backend $BACKEND --out $OUT/timing/swa_paths.json
python -m scripts.timing_report --timing-pass $OUT/timing/timing_pass.json --swa-paths $OUT/timing/swa_paths.json --out $OUT/tables
```

Check: the record's `swa256/flex` track is as in §1.2, times or an error. If it was an error then and has times now, or the reverse, stop: `opt_path` was set on a record that no longer holds, and that is a question for the build loop before any further SWA run. `timing_report` refuses a missing or unreadable record in one line and writes no table. The last section of `$OUT/tables/timing.md` shows the three attention times with the record's own source line; it selects nothing.

Check: `timing_pass` exits 0. It prints SWA's attention time per step beside MHA's (plan §2.3 item 2). If SWA's is not below MHA's, the pass is still valid for every variant: SWA's timings are kept and carry the mark † with the note "not faster than a dense kernel on this GPU" in `timing.md`, `headline.md` and `rqw.md` (plan §18, A15). With SWA's chunked path the plan itself expects savings near zero at context 1,024, so this is a likely outcome and a finding to report. In `$OUT/tables/timing.md`, read the ms/step and ratio columns of MHA, GQA-4, GQA-2 and MQA against the plan's prediction that they are within ±3% of each other; the table prints numbers and no verdict. Note the `timing_pass_id` in its source line.

### 3.2 The ladder: L6 at 5 seeds, L10 at 3, each with 100M and 200M branches

```bash
for s in 1 2 3 4 5; do
  python -m scripts.train --variant mha --ladder mha_L6 --init-seed $s --peak-lr $L6_LR --lr-source $L6_SRC \
    --branches 100M 200M --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  for b in 100M 200M; do
    python -m scripts.branch --variant mha --ladder mha_L6 --init-seed $s --peak-lr $L6_LR --lr-source $L6_SRC \
      --branches 100M 200M --budget $b --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
for s in 1 2 3; do
  python -m scripts.train --variant mha --ladder mha_L10 --init-seed $s --peak-lr $L10_LR --lr-source $L10_SRC \
    --branches 100M 200M --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  for b in 100M 200M; do
    python -m scripts.branch --variant mha --ladder mha_L10 --init-seed $s --peak-lr $L10_LR --lr-source $L10_SRC \
      --branches 100M 200M --budget $b --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
```

### 3.3 MHA seeds 1 to 5 continue to 300M, and their 200M branches

The same `train` command as in §2.5 without `--until`: it resumes from step 1,221.

```bash
for s in 1 2 3 4 5; do
  python -m scripts.train --variant mha --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --branches 100M 200M --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  python -m scripts.branch --variant mha --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
    --branches 100M 200M --budget 200M \
    --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
```

### 3.4 Positive control, detectable effect and seed count

```bash
python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables --skip-unfinished \
  --timing-pass $OUT/timing/timing_pass.json
python -m scripts.make_plots --manifests $OUT/manifests/opt --out $OUT/plots --skip-unfinished
```

Check `gates.md`:

- "Positive control: L6 against L8 at 300M": the verdict. `detected` means L6 is worse than L8 and the pipeline can see a 25% capacity cut. `not detected`, or `detected in the wrong direction` (L6 significantly better than L8), means the control has failed: the project ships as the noise-floor post with that finding, and every later not-detected verdict is uninformative (plan §8.4, §9.4, §18 A6); the notes say so.
- "Seed count and detectable effect": `seeds per primary` is 5 or 8. If 8, `variants leaving Arm A` names MLA-96 and SWA-256, and seeds 6 to 8 are added to MHA and the three primaries in §4. It stays blank until MHA seeds 1 to 5, L6 seeds 1 to 5 and L10 seeds 1 to 3 all have their 300M results; the notes name the missing seeds. If you cut L10 (plan §12.2 item 4), declare it by setting `ladder: l6_l8` in `configs/commitments.yaml` by pull request before you look at either result: the rule then uses the two-point slope. After the rule prints `seeds per primary`, commit that number as `seed_count` in `configs/commitments.yaml` by pull request before week 3; until then the tables test no primary.
- "Ladder slope b by budget": three values.

Check `05_ladder.png`: three panels with every seed and a fit line.

- "Declarations (configs/commitments.yaml)": `ladder` and `committed seed count` as the committed file has them. If `ladder slope b at 300M` is not positive, `seeds per primary` is 8, the fraction is blank and no multiplier will be reported (plan §18, A4).

Commit, **before any variant result at 300M is read** and before the first run of §4:

1. To the pre-registration section of `README.md`: the detectable gap of the chosen design in nats and the seed count (plan §1.3, §1.4).
2. To `configs/commitments.yaml`, by pull request: `seed_count: 5` or `seed_count: 8`, the number `seeds per primary` shows.

Then build the tables again from a checkout that has that commit: the "seed count not committed" note is gone. The tables and plots read `configs/commitments.yaml` of the checkout they run from; no flag names another file. Until the seed count is committed, and whenever the committed number differs from the rule's, they compute no primary and no exploratory comparison and withhold every variant's 300M losses (the headline table, `rqw.md`, `context.md` and plots 2, 3, 6 and 7), and say so. Arm B and the shared-rate table are 100M comparisons and do not wait. Publish the noise-floor post.

## 4. GPU week 3: the primaries (about 21 GPU-h, about 38 if the rule picked 8 seeds)

Not before `seed_count` is committed (§3.4). For each primary, at the rate and `lr_source` its sweep section gave. Seeds 1 to 5, or 1 to 8 if the rule picked 8:

```bash
for v in gqa4 mqa mla192; do
  for s in 1 2 3 4 5; do
    python -m scripts.train --variant $v --init-seed $s --peak-lr $V_LR --lr-source $V_SRC \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
    python -m scripts.branch --variant $v --init-seed $s --peak-lr $V_LR --lr-source $V_SRC --budget 100M \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
```

Set `V_LR` and `V_SRC` per variant inside the loop. If the rule picked 8, also run MHA seeds 6, 7 and 8 exactly as in §2.5 and §3.3.

### Gate 3

```bash
python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables --skip-unfinished \
  --timing-pass $OUT/timing/timing_pass.json
python -m scripts.make_plots --manifests $OUT/manifests/opt --out $OUT/plots --skip-unfinished
```

Check, in this order (plan §9.5):

1. The script did not refuse. A refusal names the runs whose steps taken differ by more than 0.5%, whose backend, versions, recipe, init rule, precision settings or backbone differ, or that were made with `--override`. Fix that before reading anything.
2. `headline.md`: the three primaries carry a verdict, `detected` or `not detected`, with the gap, its interval and, for an undetected gap, the bound it rules out. Only `primary_test.py` wrote those words. The notes list nothing missing for the primaries.
3. `01_primary_gaps.png`: every seed is a dot; the band is the bound committed in §3.4.

**Gate 3** has no failure branch: a detection and a ruled-out bound are both results. Do not add seeds, whatever it shows (plan §13.10).

## 5. GPU week 4: everything else (about 24 GPU-h)

### 5.1 Arm A exploratory (skip MLA-96 and SWA-256 if the rule picked 8 seeds)

```bash
for v in gqa2 mla96 swa256; do
  for s in 1 2 3; do
    python -m scripts.train --variant $v --init-seed $s --peak-lr $V_LR --lr-source $V_SRC \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
    python -m scripts.branch --variant $v --init-seed $s --peak-lr $V_LR --lr-source $V_SRC --budget 100M \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
```

### 5.2 Arm B, iso-backbone (3 variants × 3 seeds at 100M)

At each variant's Arm A rate: the table refuses "Arm A minus Arm B" when the two rates differ.

```bash
for v in gqa4 gqa2 mqa; do
  for s in 1 2 3; do
    python -m scripts.train --variant $v --arm iso_backbone --init-seed $s --peak-lr $V_LR --lr-source $V_SRC \
      --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
```

### 5.3 Inference benchmark (alone on the box)

First a trial of one repetition for two variants, to see that every point fits in memory in both modes; its files are not results:

```bash
python -m scripts.bench --device cuda --gpu-shared false --expected-backend $BACKEND --with-ref \
  --repetitions 1 --variants mha mla192 --out-dir $OUT/bench_trial
```

Then the benchmark:

```bash
python -m scripts.bench --device cuda --gpu-shared false --expected-backend $BACKEND --with-ref \
  --out-dir $OUT/bench
```

```bash
python -m scripts.timing_report --timing-pass $OUT/timing/timing_pass.json --bench $OUT/bench --swa-paths $OUT/timing/swa_paths.json --out $OUT/tables
```

The benchmark measures every point three times, round-robin over the variants, and reports the median with the smallest and largest value (plan §18, A16); allow three times the time of a single pass.

Check: one file per variant and mode, and `$OUT/tables/inference.md` with a prefill table, a decode table and a KV line for each, every timing with its range over the three repetitions. The benchmark fails if measured and analytic KV bytes differ by more than 5%. The files are written after the last repetition, so an out-of-memory error in any track ends the benchmark with nothing written. `ref` mode at batch 32 and cache 8k may run out of memory: the trial above is there to find that out, and if it does not fit, narrow `--decode-batches` or `--decode-caches` for the real run and say so. Two things to disclose with the numbers: the sliding-window cache allocates up to twice its 256-token cap, and MLA's absorbed matrices are a fixed cost per cache outside the KV bytes per token.

### 5.4 Shared-rate runs (3 seeds at 100M for each Arm A variant on its own rate)

`lr_sweep.md` says, for every variant, whether it runs at MHA's rate ("tie rule") or at its own. For each variant in Arm A whose rate differs from MHA's, run seeds 1 to 3 standalone at 100M at MHA's rate (plan §18, A7). Set `OWN_RATE_VARIANTS` to those variants; if there is none, skip this section. A variant that left Arm A under the seed-count rule gets no such runs.

```bash
for v in $OWN_RATE_VARIANTS; do
  for s in 1 2 3; do
    python -m scripts.train --variant $v --init-seed $s --peak-lr $MHA_LR --lr-source $MHA_SRC \
      --budget 100M --branches --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
  done
done
```

About 1.3 GPU-h per variant. Check after §5.5: `shared_rate.md` has, for each such variant, its three 100M branches at its own rate, the three runs at MHA's rate, and the difference with its interval; a variant on MHA's rate says so; the notes name any run still missing. The table makes no detection claim.

Arm C is cut (plan §18, A18): the wall-clock question keeps the computed estimate in `rqw.md`.

### 5.5 All tables and plots, Gate 4

```bash
python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables \
  --timing-pass $OUT/timing/timing_pass.json --bench $OUT/bench
python -m scripts.make_plots --manifests $OUT/manifests/opt --bench $OUT/bench --probes $OUT/probes --out $OUT/plots
```

No `--skip-unfinished` here: an unfinished run at this point is a problem to resolve, not to leave out. A run recorded as diverged is the exception: it is left out and named in the notes of `headline.md` and `gates.md` (plan §7.4 item 5).

Check: `headline.md` has no blank cell and its notes explain every omission; its decode cell is the benchmark's median with the smallest and largest repetition in brackets; `gates.md` shows the committed seed count equal to `seeds per primary`; `arm_b.md` has both rankings; `context.md` has both metrics for every variant that ran; `shared_rate.md` is complete for every variant on its own rate; `rqw.md` states the estimate and its interpolation; all ten plots built, with the retrieval plot blank.

**Gate 4** (plan §9.7): Phase 6 runs only if a primary gap was detected or a late-context gap's interval in `context.md` excludes zero. Phase 6 has generators and a scorer (`src/eval/retrieval.py`) and no run script. If the gate opens, Phase 6 is its own build with its own review, and settles the RoPE extension beyond 1,024 tokens (the plan's open decision 5) then (plan §18, A19).

## What the build loop did not verify

- Nothing here ran on a GPU, on Kaggle or on the real corpus. The `gpu` and `t4` tests are written and have never run; list them with `pytest --collect-only -q -m "gpu or t4"`.
- The fp16, GradScaler and backend-enforcement code paths have never executed.
- Divergence recording was tested with a forced non-finite loss, not a real diverged run.
- The environment check on resume and the peak-memory record were tested with replaced readings on the CPU. Reading the driver, the GPU name and the CUDA memory statistics has never run.
- `scripts/prune.py` was tested on tiny synthetic runs. Read its dry run before the first `--apply`.
- Phase 6 has no commands and no run script (plan §18, A19). Arm C is cut (A18).
- The probe commands of §2.3 were parsed, not run: the pilot's first guesses for the steps are guesses, and `scripts/train_probe.py` has never run on a GPU.
- The `nvidia-smi` log line of §0 was not run.
