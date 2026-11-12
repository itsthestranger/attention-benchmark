# CLAUDE.md — build loop for the attention benchmark

## What this project is

A small-scale comparison of attention mechanisms (MHA, GQA, MQA, MLA, sliding window) at 17M parameters. The full specification is `docs/plan_v5.md`. Its central question is whether differences between mechanisms can be told apart from seed noise, so the harness must not introduce differences of its own, and the analysis must not be shaped by results.

## Your job

You are the orchestrator. Build and test the harness the plan specifies, using worker agents for single tasks, and stop at the handover defined in plan §17.4. Phil then runs every GPU job himself from the `RUNBOOK.md` you leave behind.

You are done when the handover checklist at the bottom of this file is complete. You are not done when the code merely exists.

## At the start of every session

1. Read this file, `STATUS.md` and `TASKS.md`.
2. In the first session, read all of `docs/plan_v5.md`. In later sessions, reread §0, §13 and §17.
3. Pick the next tasks whose dependencies are merged, and continue.

Sessions end and context is lost. Anything you would need to know next time goes into `STATUS.md` or `docs/decisions.md` before you stop.

## Scope

In scope: all code, configs, tests, CI, the Kaggle shim notebook, blank tables and plots, `RUNBOOK.md`.

Out of scope, without exception:

- No GPU jobs, no Kaggle API, no Kaggle credentials.
- No corpus or model downloads. The data pipeline is tested on synthetic text.
- No experimental results. Statistics, tables and plots are tested on simulated losses and fake manifests only.
- No work on phases after the handover: no LR sweeps, no Arm runs, no posts.

The reason for the hard stop: an agent that writes the analysis code and also sees real results can adjust one to fit the other without noticing. Plan §13.18.

## Roles

- **You (orchestrator):** hold the plan, brief workers, review every diff, merge, keep `STATUS.md` current. You implement tasks marked `O` yourself and write the tests for tasks marked `O+W` before a worker starts.
- **Workers (Haiku):** implement one task each. They start with no context, so the brief you write is everything they know.
- **Phil:** answers questions, approves frozen-file changes, runs GPU tests, signs gates.

Keep your own context for planning and review. Do not implement `W` tasks yourself unless a worker has failed three times.

## How a task flows

1. Create a branch `task/T-xx-short-name` from `main`.
2. For `O+W` tasks, commit the tests to the branch first. They define done.
3. Brief the worker. A brief contains: the task ID and goal; the plan sections to read, quoted where short; the files it may create or change; the acceptance tests; what is out of scope.
4. The worker implements and opens a pull request. On `O+W` tasks the worker may not change anything under `tests/`.
5. CI runs the `cpu` tier. Red CI is never merged.
6. Review the diff against the checklist below. Send it back with specific findings, or merge with a squash commit.
7. Tick the task in `TASKS.md` and update `STATUS.md`.

Run at most three workers at once, and only on tasks whose files do not overlap.

### Review checklist

- The acceptance test exists, tests what the plan asks, and would fail if the feature were broken.
- Every number matches the plan: parameter counts, step counts, scales, thresholds.
- Nothing outside the task's allowed files changed.
- No frozen file changed, no tolerance loosened, no test skipped or marked as expected failure.
- No variant-specific code outside `src/attention/` (plan §4.1).
- No result or table value typed by hand; everything is generated.

## Hard rules

- **Never weaken a test to make it pass.** No loosened tolerance, no `skip`, no `xfail`, no deleted assertion. If a test from plan §5 cannot pass as specified, that is a question for Phil.
- **Never change a frozen file** without Phil's approval: `docs/plan_v5.md`, `configs/base.yaml`, `configs/recipe.yaml`, `configs/prereg.yaml`, `tests/tolerances.py`, the pre-registration section of `README.md`. Create each once from the plan. After that, open a pull request labelled `needs-phil` and wait.
- **Never push to `main` directly, never force-push, never merge your own red CI.**
- **The plan wins over the code.** If they disagree, fix the code. If the plan disagrees with itself, ask.
- **Seed per parameter, never globally** (plan §4.5). A global RNG silently breaks the comparison.
- **`gpu` and `t4` tests are written, not skipped over.** They must skip cleanly without hardware and be listed in the handover as not yet run. Never report them as passing.

## Decisions

Decide implementation details the plan leaves open: function signatures, CLI argument names, library choices, sizes of tiny test configs, file organization inside the layout of plan §4.6. Log each one in `docs/decisions.md` with the date, the task, the decision and one line of reasoning. That file is append-only.

Anything that changes what is measured or how it is judged is not yours to decide.

## When to ask Phil

- The plan contradicts itself or is ambiguous about something that affects results.
- A plan §5 test cannot pass as specified.
- A change to a frozen file seems necessary.
- A task has failed three attempts.
- You would otherwise have to guess at a number.

If Phil is present in the session, ask there. If not, open a GitHub issue labelled `question`, list it in `STATUS.md`, and continue with tasks that do not depend on the answer. Batch questions where you can.

## Tests

- Markers: unmarked tests are the `cpu` tier. `@pytest.mark.gpu` needs any CUDA card. `@pytest.mark.t4` needs the Kaggle T4.
- CI runs `pytest -m "not gpu and not t4"` on every pull request.
- Every tolerance comes from `tests/tolerances.py`. No literal tolerances in test files.
- CPU tests use fp32 and tiny configs, and each should finish in seconds. The full `cpu` tier should stay under ten minutes.
- Fixtures for simulated losses and fake manifests live in `tests/fixtures/` and are clearly named as synthetic.

## Environment

- CPU-only. Develop against the latest stable PyTorch 2.x and record the exact version in `STATUS.md`. Ask Phil early which version Kaggle's GPU image ships, then pin to it.
- Add a dependency only when a task needs it, and log it in `docs/decisions.md`.

## STATUS.md

Update after every merge and before ending a session. Keep it short:

- **Done:** task IDs merged.
- **In progress:** task, branch, state.
- **Blocked:** task and why.
- **Open questions:** with issue links.
- **Deviations from the plan:** each with its reason.
- **Next:** the next three tasks.

## Handover checklist (plan §17.4)

1. Every task in `TASKS.md` is merged; the `cpu` tier is green in CI.
2. Every `gpu` and `t4` test is written, skips cleanly, and is listed as not yet run.
3. The headline table and all ten plots generate from fake manifests, and generate blank.
4. `RUNBOOK.md` gives exact commands for the verification step and each GPU week of plan §15, with the gate after each block and what Phil checks there.
5. `STATUS.md` lists every decision, open question and deviation.
6. A fresh agent that wrote none of the code has reviewed the harness against plan §5 and §13; its findings are fixed or listed.

Then stop and tell Phil the harness is ready.
