# STATUS

Last updated: 2026-11-12, session 1.

## Done

- Bootstrap commit: `CLAUDE.md`, `TASKS.md`, `docs/plan_v5.md`.

## In progress

- T-00, branch `task/T-00-scaffold`: complete, waiting for the pull request to be merged.

## Blocked

- Nothing.

## Open questions

- Which PyTorch, Python and CUDA versions does Kaggle's GPU image ship? Needed to pin the environment. Until then: torch 2.15.0, Python 3.12 in CI.

## Deviations from the plan

- fp32 `ref`/`opt` agreement tolerance is 1e-5, not 1e-6 (plan §5). Approved by Phil on 2026-11-12. Reason: 1e-6 is at the fp32 rounding floor; correct implementations measured up to 1.8e-6.
- The agreement tests also run in float64 at 1e-12, an addition to "CPU tests use fp32". Approved by Phil on 2026-11-12.

## Decisions

All decisions are in `docs/decisions.md` (rows 1–13 so far).

## Environment

- PyTorch 2.15.0. CI uses the CPU build on Python 3.12.
- The build container cannot reach download.pytorch.org, so it runs the default PyPI build of 2.15.0 on CPU (Python 3.13).

## For Phil

- Protect `main`: pull requests only, the `cpu` check required, and "Require review from Code Owners" enabled, otherwise CODEOWNERS has no effect.
- Create the labels `needs-phil` and `question`.

## Next

1. T-01 config system (W).
2. T-24 `primary_test.py` and `test_primary_stats.py` (O), after T-01.
3. T-02 backbone (W), then T-13 data pipeline and T-14 schedule in parallel.
