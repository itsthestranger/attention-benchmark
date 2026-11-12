# TASKS.md — build stage

Task graph for the autonomous build loop. The specification is `docs/plan_v5.md`; the operating rules are in `CLAUDE.md`. The loop ends at the handover (plan §17.4). Tick a task only when its pull request is merged with green CI.

**Owner codes**

- `O`: the orchestrator implements it.
- `O+W`: the orchestrator writes the tests first; a worker implements against them and may not change them.
- `W`: a worker writes code and tests; the orchestrator reviews both.

**Before the first session (Phil)**

- [ ] Create the repository and add `CLAUDE.md`, `TASKS.md` and the plan as `docs/plan_v5.md`.
- [ ] Protect `main`: pull requests only, CI required.
- [ ] Tell the session which PyTorch version Kaggle's GPU image ships, if known.

## M0 — Scaffold

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-00 ✅ | Repo scaffold: layout, `pyproject`, pytest markers, CI for the `cpu` tier, CODEOWNERS for frozen paths, `docs/decisions.md`, `STATUS.md`, `tests/tolerances.py` | O | — | §4.6, §5.1, §17.3 | CI is green on an empty suite; every frozen path is in CODEOWNERS |
| T-01 ✅ | Config system: `base.yaml`, variants, ladder, arms, `recipe.yaml` with smoke-run values unset, `prereg.yaml`; loader with resolved config and hash | W | T-00 | §3, §7, §1.2 | Every config loads; the hash is stable; values match the plan's tables |

## M1 — Model

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-02 ✅ | Backbone: RMSNorm, SwiGLU, RoPE, tied embeddings, pre-norm blocks, attention registry | W | T-01 | §3.1, §4.2 | Shape tests pass; non-attention parameter counts match the formula |
| T-03 ✅ | Per-parameter seeded init and the init rule | O+W | T-02 | §3.1, §4.5 | Same-name, same-shape tensors are identical across variants; different seeds differ; stds and depth scaling match the rule |
| T-04 ✅ | `grouped.py`: MHA/GQA/MQA, `ref` and `opt`, KV cache, grouped-query decode | O+W | T-02 | §2.3, §4.4, §5 | Causal leak, GQA limits, `ref`/`opt` agreement and KV-cache decode at ctx 1024 pass; K/V never expanded in `opt` decode; `gpu` and `t4` tests written |
| T-05 ✅ | `mla.py` naive path: decoupled RoPE, latent norms, softmax scale, factor init | O+W | T-03 | §3.2, §3.3 | 562,560 params per layer at d_c 192; scale is 1/√72; causal leak and KV-cache decode pass; cache holds 216 values per token per layer |
| T-06 ✅ | MLA absorbed path for inference | O+W | T-05 | §3.3 | Matches the naive path to 1e-4 |
| T-07 ✅ | `sliding_window.py`: `ref`, chunked `opt`, FlexAttention path behind an availability check | O+W | T-02 | §2.3 | Weights outside the window are exactly zero, boundaries included; `ref`/`opt` agree; chunked path scores at most 512 keys per query; `t4` timing test written |
| T-08 ✅ | `AttnBase` properties and `utils/flops.py` | W | T-04, T-05, T-07 | §3.1, §3.2, §4.2 | KV bytes match §3.2 for every variant; attention FLOP shares at ctx 512/1,024/2,048 are 8/15/27% causal and 15/27/42% dense, within one point |
| T-09 ✅ | `solve_config.py` and `test_params.py` | W | T-08 | §3.2, §4.3, §8.4 | 17,308,032 exactly for non-MLA variants; MLA offsets +3,072 and +1,536; ladder at 10,621,824 and 17,702,784 non-embedding; fails loudly instead of rounding |
| T-10 ✅ | Step-0 logit-scale test across all variants | O | T-04, T-05, T-07 | §3.3, §5 | Attention-logit std agrees within 10% across the seven variants |
| T-11 ✅ | CI graph-diff check | W | T-09 | §4.1 | Fails when anything outside the attention subtree differs, except d_ff in Arm A and layer count in the ladder |
| T-12 ✅ | Recurrence anchor module | W | T-02 | §9.2 | Causal and shape tests pass; the state is one d_model vector |

## M2 — Training

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-13 ✅ | Data pipeline: tokenizer script, packing with EOS and document-offset sidecar, `uint16` loader with its own data-seed generator, hashing, splits | W | T-01 | §6 | On a synthetic corpus: token order is bit-identical across variants and init seeds; splits are disjoint by document; a wrong hash is refused |
| T-14 ✅ | WSD schedule and optimizer parameter groups | O+W | T-01 | §7.1, §7.2 | LR is as specified at steps 0, 100, 1,221, 1,526, 3,662 and 4,578; decay has the 1−√ shape; weight decay covers 2D matrices only, not the tied embedding |
| T-15 ✅ | Trainer: autocast and GradScaler with unscale before clipping, accumulation, skip and steps-taken logging, in-loop backend and dtype assertions | O+W | T-04, T-13, T-14 | §2.3, §7, §13.6 | A tiny model overfits on CPU; an injected inf gradient is counted as a skipped step; accumulation matches the large batch; `gpu` tests written |
| T-16 ✅ | Checkpoint and resume | O+W | T-15 | §5, §12.3 | Kill at step 500 and resume gives the uninterrupted curve; the last two checkpoints are kept; resume needs no human input |
| T-17 ✅ | `branch.py` | O+W | T-16 | §5, §7.2 | A branch from a trunk checkpoint matches an uninterrupted run; its manifest records parent and branch step |
| T-18 ✅ | Manifest and run naming | W | T-15 | §10.6, §11 | Schema test covers every field; names parse and round-trip |
| T-19 ✅ | Determinism test for `ref` and the reproducibility measurement script | W | T-15 | §5 | 100 steps are bit-identical twice on CPU; the `gpu` measurement is written |

## M3 — Evaluation

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-20 ✅ | `eval/lm.py` and context-bucketed loss with the in-context-learning score | W | T-13, T-15 | §6, §10.3 | Bucket-weighted loss equals aggregate loss; hand-built windows with mid-window document starts give the expected context values |
| T-21 | Probe generators and probe training script | W | T-12, T-15 | §9.2 | Property tests pass for all five probes; a tiny MHA solves 4-pair MQAR on CPU |
| T-22 ✅ | Retrieval evaluation: passkey and needle generators | W | T-15 | §9.7 | Generator tests pass; the RoPE extension stays open (decision 5) |
| T-23 ✅ | `bench.py`, `eval/inference.py`, `timing_pass.py` | W | T-04, T-06, T-07 | §8.5, §8.6, §10.4 | Each runs end to end on CPU with a tiny model; round-robin order and warm-up discard are tested; the analytic-vs-measured KV assertion is written for `gpu` |

## M4 — Analysis

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-24 ✅ | `primary_test.py` reading `prereg.yaml`, and `test_primary_stats.py` | O | T-01 | §1.2, §1.3 | Simulated nulls give family-wise error ≤ 5% with the test switch and both seed counts; the §1.3 table is reproduced within 0.05; no other code emits a verdict |
| T-25 ✅ | `ladder_fit.py` with the seed-count rule | O | T-24 | §1.4, §8.4 | A known slope is recovered; the multiplier interval carries both uncertainties; the rule returns 5 or 8 on constructed cases |
| T-26 ✅ | `rqw_estimate.py` | W | T-18 | §8.5 | Interpolation is correct on constructed manifests; a variant missing from Arm A reports its step-time ratio only |
| T-27 ✅ | `make_tables.py` with its refusal gates | W | T-18, T-24 | §8.7, §10.1 | The headline table builds from fake manifests; each refusal (steps, backend, versions, trunk checkpoint) has a test |
| T-28 ✅ | Plots | W | T-27 | §10.5 | All ten build from fake manifests and build blank; every seed is a visible dot |

## M4b — Reporting gaps found in session 3

The plan requires these outputs, and no task above produces them. Added by the orchestrator on 2026-11-12; see `STATUS.md`.

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-32 ✅ | Controls, σ_data, ladder fits by budget and the gate report in `make_tables.py` | O+W | T-27 | §8.1, §8.4, §9.3, §9.4 | `tests/test_gate_report.py` passes; no comparison is computed before the controls; the blank `gates.md` is committed |
| T-33 | LR-sweep report and selection: edge rule, tie rule, divergences recorded | O+W | T-29, T-32 | §7.4, §13.15 | Constructed sweeps give the pre-registered choice; a winner at the grid edge asks for another point; a diverged run is recorded, not lost |
| T-34 | Secondary reports and command-line wiring: Arm B table, RQ-W estimate from the timing-pass file, context-use gaps with intervals, the timing and decode columns of the headline table | O+W | T-23, T-32 | §8.3, §8.5, §10.1, §10.3 | Each builds blank and from fake manifests; no verdict outside `primary_test.py` |
| T-35 | Plots 4 and 5 read the controls and the per-budget ladder fits | W | T-28, T-32 | §10.5 | The null-control verdict is shown verbatim; the ladder is drawn and fitted at 100M, 200M and 300M |

## M5 — Operations and handover

| ID | Task | Owner | Depends on | Plan | Done when |
| --- | --- | --- | --- | --- | --- |
| T-29 ✅ | Kaggle shim notebook and the `train.py` / `evaluate.py` entry points | W | T-17, T-20 | §12.3 | A cold start runs on CPU against a synthetic dataset and resumes without human input |
| T-30 | `RUNBOOK.md` | O | T-00 to T-35 | §15 | Exact commands for the verification step and each GPU week, in order, with the gate after each block and what Phil checks |
| T-31 | Handover: audit by a fresh agent against §5 and §13, final `STATUS.md`, `README.md` with an unfilled pre-registration template | O | T-30 | §17.4 | The handover checklist in `CLAUDE.md` is complete |
