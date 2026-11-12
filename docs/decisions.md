# Build-loop decision log

Append only. Each row: date, task, decision, one line of reasoning. Decisions
here are implementation details the plan leaves open (CLAUDE.md, Decisions).
Anything that changes what is measured or how it is judged is Phil's, and is
marked "Phil" in the task column with the date he decided.

| # | Date | Task | Decision | Reasoning |
| --- | --- | --- | --- | --- |
| 1 | 2026-11-12 | Phil | The bootstrap commit (CLAUDE.md, TASKS.md, docs/plan_v5.md) goes to `main` directly, once. | The repository was empty, so there was no branch to open a pull request against. |
| 2 | 2026-11-12 | Phil | Develop and run CI on the latest stable PyTorch 2.x: 2.15.0. Pin to Kaggle's version once it is known. | The Kaggle image version is not known yet. |
| 3 | 2026-11-12 | Phil | A numerical tolerance means max\|a − b\| / max\|b\| ≤ tol, with b the reference. No absolute term. | Plan §5 does not say absolute or relative. Absolute bounds are vacuous for small gradients and too strict for trained logits. |
| 4 | 2026-11-12 | Phil | fp32 `ref`/`opt` agreement is 1e-5, not the 1e-6 of plan §5. **Deviation from the plan.** | 1e-6 is about 8 units of fp32 precision; a correct implementation measured 0.2e-6 to 1.8e-6 on CPU with torch 2.15.0. |
| 5 | 2026-11-12 | Phil | KV-cache decode against the full forward: 1e-5. Resume and branch fidelity on CPU: 1e-5. | Plan §5 gives no number for the first and says "within `opt` tolerance" for the others. |
| 6 | 2026-11-12 | Phil | The agreement tests also run in float64 at 1e-12. | Catches subtle logic errors that fp32 rounding hides. Departs from "CPU tests use fp32" by adding a check, not replacing one. |
| 7 | 2026-11-12 | Phil | The simulated family-wise error check uses a fixed seed and 20,000 simulations, and fails above 0.05 plus three standard errors. | A Monte-Carlo estimate of a rate near 5% needs a stated allowance. |
| 8 | 2026-11-12 | T-00 | `tests/tolerances.py` also holds two values that are neither in the plan nor discussed: gradient accumulation at 1e-5 (fp32) and float64 identities at 1e-12. Flagged in the T-00 pull request for Phil. | Later tasks need them, and the file is frozen once merged. |
| 9 | 2026-11-12 | T-00 | `src` is the import package, as in the layout of plan §4.6. Commands run from the repository root; `pip install -e .` also works. | Keeps the paths the plan and CLAUDE.md refer to, such as `src/attention/`. |
| 10 | 2026-11-12 | T-00 | Dependencies: `torch` (runtime), `pytest` (dev). Python ≥ 3.11. | Nothing else is needed yet. 3.11 gives `tomllib` in the standard library. |
| 11 | 2026-11-12 | T-00 | CI: one job named `cpu`, Python 3.12, CPU build of torch 2.15.0, `pytest -m "not gpu and not t4"`. | The job name is what branch protection requires. |
| 12 | 2026-11-12 | T-00 | Hardware tiers are applied in one place, `tests/conftest.py`: `gpu` needs CUDA, `t4` needs a device whose name contains "T4". A scaffold test fails if any other test file skips or expects failure. | Makes "never skip a test to pass" checkable. |
| 13 | 2026-11-12 | T-00 | CODEOWNERS names `@itsthestranger` for every frozen path, for the whole of `README.md`, and for CODEOWNERS itself. | CODEOWNERS works per file, not per section; protecting the file itself stops the protection being removed unnoticed. |
| 14 | 2026-11-12 | T-01 | A task's pull request ticks the task in `TASKS.md` (✅ after the ID) and updates `STATUS.md` itself, so both become true on `main` at the moment of the merge. | `main` takes pull requests only; a separate bookkeeping pull request per task would double the count. |
| 15 | 2026-11-12 | T-01 | The orchestrator writes the YAML config files from the plan's tables; the worker writes the loader and tests whose expected values are literals from the plan, not read back from the YAML. | Two independent transcriptions of the plan's numbers check each other. |
| 16 | 2026-11-12 | T-01 | Config schema: `resolve()` merges base, recipe, variant, optional ladder and arm into one plain dict; its SHA-256 over canonical JSON is the config hash. The loader lives in `src/config.py`. | The hash must be stable across processes because resume keys on it (plan §12.3). |
| 17 | 2026-11-12 | T-01 | The peak learning rate, seeds, budget and branches are per-run inputs to `resolve()`, not recipe values. Unset recipe values stay `None`; `require_complete()` refuses them and overrides can supply them. | The LR comes from the sweeps (plan §7.4); the three unset values come from the smoke runs (plan §7.3). |
| 18 | 2026-11-12 | T-01 | Arm tags in run names: `isoparam` (Arm A, as in the plan §11 example), `isobackbone` (Arm B), `isowallclock` (Arm C). Arm N and the ladder run under `iso_token`. | They are MHA runs under Arm A's rules (plan §8.1, §8.4); the layout of §4.6 has three arm files. |
| 19 | 2026-11-12 | T-01 | Ladder runs default to branches at 100M and 200M; every other run to 100M. A branch must be shorter than its run. | Plan §7.2: the ladder takes the 200M branch, other runs do not. |
| 20 | 2026-11-12 | T-01 | The in-context-learning windows include their end points (450 ≤ c ≤ 550, 40 ≤ c ≤ 60); the four context buckets are half-open. Flagged for Phil in the T-01 pull request. | Plan §10.3 writes "c 450–550" without saying; the buckets are written as half-open intervals. |
| 21 | 2026-11-12 | T-01 | Dependency added: `pyyaml`. | The plan's configs are YAML. |
