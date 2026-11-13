# RUNBOOK

Every GPU job of plan §15, in order, with the gate after each block and what to check there. The build loop ran none of this. The commands are checked against the scripts' argument parsers by `tests/test_runbook.py`; nothing else about them has run on a GPU.

Read plan §11 and §13 before the first run. Three rules that no command enforces:

- Do not open a variant's manifest, loss curve or table before the gate that allows it. `make_tables.py` refuses to compute comparisons before the controls, but it cannot stop you reading a JSON file.
- Never change `configs/recipe.yaml`, `configs/prereg.yaml` or `tests/tolerances.py` after their freeze. A later change restarts every arm (plan §7, §13.6).
- Never tune anything per variant, never add seeds after a variant result at 300M has been read, never stitch a timing run across sessions.

## Open items that block parts of this runbook

| Item | Blocks | Who |
| --- | --- | --- |
| The probe grid: which probes at which sizes, for which models, with how many seeds, and one width, step count and learning rate per point (`STATUS.md`, open questions). The scripts exist: `scripts/train_probe.py`, `scripts/probe_report.py`. | §2.3, Gate 1 | Phil |
| The Kaggle image's torch, Python and CUDA versions. | Pinning (§1.5) | Phil |
| Two proposed thresholds in `configs/prereg.yaml` (`paired_if_s_rep_ratio_below: 0.6`, `max_undetectable_param_fraction: 0.05`). | Pre-registration (§1.7) | Phil |
| Whether a positive control detected in the wrong direction (L6 better than L8) fails it. The test is two-sided and would print `detected`. | Reading §3.4 | Phil |
| What "GPU utilization" and "time-to-loss-X" of plan §10.2 mean. Neither is recorded; they cannot be backfilled after the runs. | Nothing runs differently; decide before week 1 if you want them | Phil |
| Where a cut of L10 is declared: the flag `--l10-cut` on the table and plot commands (§3.4), or a key in the frozen `configs/prereg.yaml`. | Only if L10 is cut | Phil |
| Readings of the plan listed in `STATUS.md` (sweep rule order, the comparison gate, Arm B's learning rate, and others). | Nothing; confirm or change before the freeze | Phil |

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
- Exit status 2 is a refusal (a hash mismatch, an incomplete recipe, a checkpoint of another mode, a checkpoint written under another torch, CUDA, driver or GPU). Read the message; do not work around it. A run cannot continue after the image changed: pin the environment (§1.5) before the first run.
- Budgets are in steps: 100M = 1,526, 200M = 3,052, 300M = 4,578. The 100M branch leaves the trunk at step 1,221 and the 200M branch at step 2,442.
- A learning rate appears in run names without the minus sign: `1.2e3` means 1.2e-3. Type it on the command line as `1.2e-3`.
- Manifests land in `$OUT/manifests/opt/`. Tables and plots read that directory.
- After any block: `python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables --skip-unfinished` rebuilds every table from what exists. It is safe at any time: it computes only what its inputs and the gates allow, and says in the notes what it skipped and why.
- The shim notebook calls `scripts.train` with a run spec and nothing else. For a `branch`, a timing pass or the benchmark in a committed run, replace the module and its arguments on the notebook's last line and change nothing else; or run them in an interactive session.

### Disk

A checkpoint of the real model is about 208 MB. A run keeps its newest two and every branch point, so week 1 alone approaches Kaggle's 20 GB. Prune after every block:

```bash
python -m scripts.prune --out $OUT
python -m scripts.prune --out $OUT --apply
```

The first command deletes nothing: it lists every file it would delete and every file or directory it leaves alone, with the reason. Read it, then run the second. It removes the non-final checkpoints of completed runs and branches, and a branch-point checkpoint once its branch has a completed, valid manifest and the trunk itself is complete. It never touches an unfinished or diverged run, a directory no manifest accounts for, or a manifest. Final checkpoints stay unless you add `--final`; keep them until the tables are final and Phase 6 is decided, because `evaluate.py` and zero-shot retrieval load them.

## 1. Verification (the 1070 and one short T4 session, before any reported GPU hour)

### 1.1 The `gpu` tier, on the 1070

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -m gpu -q
```

Check: every test passes; none is skipped. These tests have never run. A failure here is a finding about the harness, not a flake: stop and fix it. `STATUS.md` lists them by file.

### 1.2 The `t4` tier, in one short Kaggle session

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 pytest -m "gpu or t4" -q
```

Check: every test passes on the T4, in particular FlexAttention on sm75 (if it does not compile, the chunked path is SWA's `opt` path; record which), the backend selection, SWA attention time below MHA's, and GQA/MQA decode time below MHA's at cache 8k. Record whether the fused kernel copies K and V in decode (plan §2.3); if it does, decode needs the explicit matmul, disclosed.

### 1.3 Quota

Start a T4 ×2 session, note the quota counter, wait ten minutes, note it again. Check: whether two GPUs draw quota at the single or the double rate (plan §2.1). The schedule fits either way; the answer sets how much slack a week has.

### 1.4 Cold-start dry run of the shim

Fill the placeholders of `notebooks/kaggle_shim.ipynb` (repository URL, pinned commit, dataset name, run spec, `GPU_SHARED`, `EXPECTED_BACKEND`) and commit the notebook so that it runs from the top in a fresh container. It can only run after §2.1, because nothing starts without the dataset hashes.

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

Confirm or change the two thresholds in `configs/prereg.yaml` (a pull request you approve; the file is frozen afterwards). Then commit the pre-registration section of `README.md` with a timestamp: the §1.2 rule, the three primaries, the seed-count rule, the positive-control test, the context-use metrics. The detectable-effect bound and the seed count are added in §3.4.

## 2. GPU week 1: data, recipe, noise floor (about 25 GPU-h)

### 2.1 Tokenize once

On a CPU machine (hours for 400M tokens; plan §6):

```bash
python -m scripts.prepare_data --inputs $SHARDS --tokenizer-inputs $TOKENIZER_SHARD --out $DATA_BUILD
```

`$SHARDS` is the fixed, ordered list of corpus files and `$TOKENIZER_SHARD` the one the tokenizer is trained on. The script prints two hashes. Paste them into `configs/data.yaml` (`dataset_sha256`, `tokenizer.sha256`) and commit. Upload `$DATA_BUILD` as a versioned Kaggle Dataset and mount it read-only as `$DATA`.

Check: `train`, `branch` and `evaluate` refuse to start until both hashes are set, and refuse any dataset whose hashes differ. Never re-tokenize.

### 2.2 Smoke runs, all seven variants, then the recipe freeze

`configs/recipe.yaml` leaves `micro_batch`, `init_scale` and `growth_interval` unset. Find them here, with overrides. A smoke run is the first 100 steps of a real 300M run, stopped with `--until 100`, so its warmup and learning rate are those of the real run. (A separate 100-step run would start its decay at step 80.)

First the micro-batch: the largest that fits MLA-192 in `ref` mode (plan §7.1). Try 64, then 32, 16, 8; twenty steps are enough to see whether it fits:

```bash
python -m scripts.train --variant mla192 --init-seed 1 --peak-lr 1e-3 --until 20 \
  --mode ref --data $DATA --out $OUT/smoke_ref --gpu-shared $SHARED --expected-backend $BACKEND \
  --override recipe.batch.micro_batch=$MICRO \
  --override recipe.grad_scaler.init_scale=$INIT_SCALE \
  --override recipe.grad_scaler.growth_interval=$GROWTH
```

Then every variant in `opt` mode with that micro-batch and one global pair of scaler settings (start from PyTorch's defaults, `INIT_SCALE=65536`, `GROWTH=2000`):

```bash
for v in mha gqa4 gqa2 mqa mla192 mla96 swa256; do
  python -m scripts.train --variant $v --init-seed 1 --peak-lr 1e-3 --until 100 --eval-every-steps 50 \
    --data $DATA --out $OUT/smoke --gpu-shared $SHARED --expected-backend $BACKEND \
    --override recipe.batch.micro_batch=$MICRO \
    --override recipe.grad_scaler.init_scale=$INIT_SCALE \
    --override recipe.grad_scaler.growth_interval=$GROWTH
done
```

`--eval-every-steps 50` makes each smoke run evaluate twice (the real cadence is 250 steps, which a smoke run never reaches): the evaluation runs at batch 64, not at the micro-batch, so this is where an out-of-memory in evaluation shows. It is an operational setting and does not enter the configuration.

Each run stops at step 100 and writes an unfinished manifest. Check, in each manifest under `$OUT/smoke/manifests/opt/`: `steps.attempted` is 100 and `steps.skipped_in_first_100` is at most 3, for every variant (plan §7.3); `backend` is the expected one; `precision` is `fp16`; no run exited with a backend or dtype error. If any variant skips more, change the global scaler settings and rerun all seven into a fresh directory. Never tune per variant. A variant that cannot be made stable in fp16 is a reportable finding (plan §13.6). Also read `step_seconds` and rescale the budget of plan §12.1 if a 300M run will not take about 80 minutes.

The learning rate here is a placeholder, because the sweep comes later. After the sweep, look at `steps.skipped_in_first_100` again in the real runs' manifests; the same limit applies.

Then write the three values into `configs/recipe.yaml` (a pull request you approve) and freeze it. From here on no command carries an `--override`: the tables refuse any run made with one. The smoke directories are not results; do not point the tables at them, and do not resume them.

Record the run-to-run reproducibility of `opt` mode once, for the writeup (plan §5):

```bash
python -m scripts.measure_reproducibility --variant mha --steps 200 --pairs 3 --device cuda --mode opt \
  --data-dir $DATA --peak-lr 1e-3 --micro-batch $MICRO --init-scale $INIT_SCALE --growth-interval $GROWTH \
  --out $OUT/reproducibility/mha_opt.json
```

It trains four identical 200-step runs and writes how far they differ. It judges nothing; keep the file.

### 2.3 Phase 1 probes, Gate 1

No commands yet: the probe grid is not fixed (see the open items). `scripts/train_probe.py` trains one probe model and writes one record, `scripts/probe_report.py` writes `probes.md` from the records, and plot 9 draws them. This section gets the exact loops when the grid is fixed.

**Gate 1** (plan §9.2): the recurrence anchor fails as the number of pairs grows, which shows the probes can discriminate. SWA's behaviour is reported, not gating. Either way the LM arms run.

### 2.4 MHA learning-rate sweep (5 runs at 100M, plus edge points)

```bash
for LR in 3e-4 6e-4 1.2e-3 2.4e-3 4.8e-3; do
  python -m scripts.train --variant mha --init-seed $SWEEP_SEED --peak-lr $LR --budget 100M --branches \
    --data $DATA --out $OUT --gpu-shared $SHARED --expected-backend $BACKEND
done
python -m scripts.select_lr --manifests $OUT/manifests/opt --out $OUT/tables
```

Check `$OUT/tables/lr_sweep.md`, section `mha`:

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

With `--with-ref` and the attention-share runs the pass is about 8,800 steps, not the 4,620 behind the 1.5 GPU-h of plan §12.1; budget about twice that. It reads the dataset, so it runs after §2.1.

```bash
python -m scripts.timing_report --timing-pass $OUT/timing/timing_pass.json --out $OUT/tables
```

Check: `timing_pass` exits 0 (it exits 1 if SWA's attention time is not below MHA's, which means a fused but dense SWA; plan §13.8). With SWA's chunked path the plan itself expects savings near zero at context 1,024, so an exit 1 here is possible and is a finding to report, not a file to use. In `$OUT/tables/timing.md`, read the ms/step and ratio columns of MHA, GQA-4, GQA-2 and MQA against the plan's prediction that they are within ±3% of each other (if not, they rejoin Arm C); the table prints numbers and no verdict. Note the `timing_pass_id` in its source line.

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

- "Positive control: L6 against L8 at 300M": the verdict. `detected` means the pipeline can see a 25% capacity cut. If it is `not detected`, the project ships as the noise-floor post with that finding, and every later not-detected verdict is uninformative (plan §8.4, §9.4); the notes say so.
- "Seed count and detectable effect": `seeds per primary` is 5 or 8. If 8, `variants leaving Arm A` names MLA-96 and SWA-256, and seeds 6 to 8 are added to MHA and the three primaries in §4. It stays blank until MHA seeds 1 to 5, L6 seeds 1 to 5 and L10 seeds 1 to 3 all have their 300M results; the notes name the missing seeds. If you cut L10 (plan §12.2 item 4), add `--l10-cut` to every `make_tables` and `make_plots` command from here on and say so in the pre-registration: the rule then uses the two-point slope. Decide that before you look at the result of either.
- "Ladder slope b by budget": three values.

Check `05_ladder.png`: three panels with every seed and a fit line.

Commit to the pre-registration, **before any variant result at 300M is read**: the detectable gap of the chosen design in nats and the seed count (plan §1.3, §1.4). Publish the noise-floor post.

## 4. GPU week 3: the primaries (about 21 GPU-h, about 38 if the rule picked 8 seeds)

For each primary, at the rate and `lr_source` its sweep section gave. Seeds 1 to 5, or 1 to 8 if the rule picked 8:

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

```bash
python -m scripts.bench --device cuda --gpu-shared false --expected-backend $BACKEND --with-ref \
  --out-dir $OUT/bench
```

```bash
python -m scripts.timing_report --timing-pass $OUT/timing/timing_pass.json --bench $OUT/bench --out $OUT/tables
```

Check: one file per variant and mode, and `$OUT/tables/inference.md` with a prefill table, a decode table and a KV line for each. The benchmark fails if measured and analytic KV bytes differ by more than 5%. `ref` mode at batch 32 and cache 8k may run out of memory; narrow `--decode-batches` or `--decode-caches` for a second `ref` pass and say so. Two things to disclose with the numbers: the sliding-window cache allocates up to twice its 256-token cap, and MLA's absorbed matrices are a fixed cost per cache outside the KV bytes per token.

### 5.4 Arm C, optional and first on the cut list (alone on the box, one session, round-robin)

`rqw.md` gives each variant's afforded steps `S_v`. Run MHA, SWA-256 and MLA-192 in the order MHA s1, SWA s1, MLA s1, MHA s2, and so on, in one session; if it dies, restart the whole arm (plan §8.5):

```bash
python -m scripts.train --variant $v --arm iso_wallclock --init-seed $s --peak-lr $V_LR --lr-source $V_SRC \
  --steps $S_V --timing-pass-id $TP_ID \
  --data $DATA --out $OUT/arm_c --gpu-shared no --expected-backend $BACKEND
```

Check: elapsed time against the budget B in `rqw.md`; flag any run more than 3% off. The tables do not read `$OUT/arm_c`; report Arm C beside the estimate.

### 5.5 All tables and plots, Gate 4

```bash
python -m scripts.make_tables --manifests $OUT/manifests/opt --out $OUT/tables \
  --timing-pass $OUT/timing/timing_pass.json --bench $OUT/bench
python -m scripts.make_plots --manifests $OUT/manifests/opt --bench $OUT/bench --out $OUT/plots
```

No `--skip-unfinished` here: an unfinished run at this point is a problem to resolve, not to leave out. A run recorded as diverged is the exception: it is left out and named in the notes of `headline.md` and `gates.md` (plan §7.4 item 5).

Check: `headline.md` has no blank cell and its notes explain every omission; `arm_b.md` has both rankings; `context.md` has both metrics for every variant that ran; `rqw.md` states the estimate and its interpolation; all ten plots built, with the probe and retrieval plots blank until those records exist.

**Gate 4** (plan §9.7): Phase 6 runs only if a primary gap was detected or a late-context gap's interval in `context.md` excludes zero. Phase 6 has generators and a scorer (`src/eval/retrieval.py`) and no run script: the RoPE extension beyond 1,024 tokens is open decision 5, and the build loop did not close it.

## What the build loop did not verify

- Nothing here ran on a GPU, on Kaggle or on the real corpus. The `gpu` and `t4` tests are written and have never run; list them with `pytest --collect-only -q -m "gpu or t4"`.
- The fp16, GradScaler and backend-enforcement code paths have never executed.
- Divergence recording was tested with a forced non-finite loss, not a real diverged run.
- The environment check on resume and the peak-memory record were tested with replaced readings on the CPU. Reading the driver, the GPU name and the CUDA memory statistics has never run.
- `scripts/prune.py` was tested on tiny synthetic runs. Read its dry run before the first `--apply`.
- Probes (§2.3) and Phase 6 have no commands.
