"""SYNTHETIC manifests for the tests. Nothing here comes from a real run.

Every value is made up: the git SHA, the data hashes, the environment (torch
"2.15.0", CUDA "13.2", driver "999.0", GPU "Synthetic T4"), the session IDs and
the validation loss. The configs themselves are real resolved configs from
configs/, so names and branch steps are the ones the plan specifies. Nothing
here reads or writes results.
"""

from __future__ import annotations

from src import config
from src.training.schedule import decay_start
from src.utils import manifest as mf

SYNTHETIC_GIT_SHA = "f" * 40
SYNTHETIC_DATA = {
    "dataset_sha256": "a" * 64,
    "tokenizer_sha256": "b" * 64,
    "order_sha256": "c" * 64,
}
SYNTHETIC_TIMING_PASS_ID = "SYNTHETIC-timing-pass"


def fake_manifest(
    variant: str = "mha",
    *,
    arm: str = "iso_token",
    ladder: str | None = None,
    init_seed: int = 1,
    data_seed: int | None = None,
    peak_lr: float = 1e-3,
    budget: str | None = None,
    steps: int | None = None,
    branch: str | None = None,
    branches: list[str] | None = None,
    kind: str | None = None,
    completed: bool = True,
    val_loss: float = 3.0,
    gpu: bool = True,
    **changes,
) -> dict:
    """A VALID synthetic manifest, built through build_manifest.

    `branch` is a budget label; it makes a branch manifest. `branches` are the
    budget labels the run branches at (default: the arm's). A run at a short
    budget has none of its own, so a standalone 100M run (a reproducibility
    rerun, the null control, a data-seed run, a sweep point) is
    `budget="100M", branches=[]`. `kind` may be
    "trunk_checkpoint" (pass completed=False). An iso_wallclock run needs
    `steps`. Any keyword in `changes` overrides a build_manifest argument and is
    validated like the rest, so an invalid change raises ManifestError.
    """
    if arm == "iso_wallclock" and steps is None:
        raise ValueError("iso_wallclock runs take their step count from the timing pass: pass steps")

    resolved = config.resolve(
        variant,
        arm,
        init_seed=init_seed,
        peak_lr=peak_lr,
        ladder=ladder,
        data_seed=data_seed,
        budget=budget,
        steps=steps,
        branches=branches,
    )
    run = resolved["run"]
    recipe = resolved["recipe"]
    fraction = recipe["schedule"]["decay_fraction"]
    parent = mf.run_name(resolved)

    if branch is not None:
        if kind not in (None, "branch"):
            raise ValueError(f"a branch manifest has kind 'branch', not {kind!r}")
        kind = "branch"
    kind = kind or "run"

    branch_block = None
    if kind == "branch":
        if branch not in run["branches"]:
            raise ValueError(f"branch {branch!r} is not among the run's branches {run['branches']}")
        total = recipe["budgets"][branch]
        branch_step = decay_start(total, fraction)
        branch_block = {
            "budget": branch,
            "steps": total,
            "branch_step": branch_step,
            "parent_run": parent,
            "parent_checkpoint": f"{parent}-trunk-step{branch_step}.pt",
        }
    else:
        total = run["steps"]

    if kind == "trunk_checkpoint":
        attempted = decay_start(total, fraction)
    elif completed:
        attempted = total
    else:
        attempted = total // 2  # an unfinished run
    step_block = {
        "total": total,
        "attempted": attempted,
        "taken": attempted,
        "skipped": 0,
        "skipped_in_first_100": 0,
    }

    if gpu:
        environment = {
            "python": "3.11.9",
            "torch": "2.15.0",
            "cuda": "13.2",
            "driver": "999.0",
            "gpu_name": "Synthetic T4",
            "compute_capability": [7, 5],
        }
        backend, precision, final_scale = "efficient", "fp16", 65536.0
    else:
        environment = {
            "python": "3.11.9",
            "torch": "2.15.0",
            "cuda": None,
            "driver": None,
            "gpu_name": None,
            "compute_capability": None,
        }
        backend, precision, final_scale = None, "fp32", 1.0

    args = {
        "kind": kind,
        "resolved": resolved,
        "completed": completed,
        "git_sha": SYNTHETIC_GIT_SHA,
        "data": dict(SYNTHETIC_DATA),
        "environment": environment,
        "backend": backend,
        "mode": "opt",
        "precision": precision,
        "steps": step_block,
        "final_scale": final_scale,
        "gpu_shared": False,
        "session_ids": ["SYNTHETIC-session-1"],
        "step_seconds": [0.5] * attempted,
        "final_metrics": {"val_loss": val_loss} if completed else None,
        "branch": branch_block,
        "lr_source": None,
        "timing_pass_id": SYNTHETIC_TIMING_PASS_ID if arm == "iso_wallclock" else None,
    }
    args.update(changes)
    return mf.build_manifest(**args)
