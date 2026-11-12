"""WSD decay branches from a trunk checkpoint (plan §5, §7.2, §13.9).

Vocabulary
----------
- The **trunk** is an ordinary long run of a warmup-stable-decay schedule. It trains
  at its constant peak learning rate for most of its steps.
- The **branch point** is a step of the trunk at which a checkpoint is written
  (``trunk-step{step:08d}.pt``). For each budget B the trunk writes one at
  ``decay_start(B, decay_fraction)``, the step where a run of budget B starts to decay.
- A **branch** starts from the trunk's branch-point checkpoint and decays over the last
  20% of its own short budget. Its result equals an uninterrupted run of that budget
  (plan §5), at the cost of only the steps after the branch point. A trunk checkpoint
  is never a result (plan §7.2).

Why a branch has its own configuration
--------------------------------------
``branch_resolved`` gives the branch the parent's configuration with its own step
count, budget and ``run.parent`` entry naming the parent. Because the config hash is
taken over the whole configuration, the branch gets a checkpoint directory of its own.
It can therefore never share a directory with a standalone run of the same budget
(for example a reproducibility rerun, plan §8.1), and neither can resume the other.

Nothing here reads or seeds the global RNG, prompts, or reads input.
"""

from __future__ import annotations

import copy
from pathlib import Path

from src import config
from src.model import build_model
from src.training import checkpoint as ck
from src.training.schedule import decay_start
from src.training.trainer import Trainer
from src.utils import manifest as mf


class BranchError(RuntimeError):
    """A branch cannot be built, started or resumed from the given parent or state."""


def branch_steps(resolved: dict) -> dict[str, int]:
    """{budget label: trunk step at which that branch leaves}, in run.branches order."""
    return {label: branch_step(resolved, label) for label in resolved["run"]["branches"]}


def branch_step(resolved: dict, budget: str) -> int:
    """The trunk step at which the branch of `budget` leaves the trunk."""
    if budget not in resolved["run"]["branches"]:
        raise BranchError(f"{budget!r} is not a branch of this run (branches: {resolved['run']['branches']})")
    recipe = resolved["recipe"]
    return decay_start(recipe["budgets"][budget], recipe["schedule"]["decay_fraction"])


def branch_resolved(parent_resolved: dict, budget: str) -> dict:
    """The branch's own configuration: the parent's, with the short budget and the parent named."""
    step = branch_step(parent_resolved, budget)
    cfg = copy.deepcopy(parent_resolved)
    run = cfg["run"]
    run["parent"] = {
        "config_hash": config.config_hash(parent_resolved),
        "budget": parent_resolved["run"]["budget"],
        "steps": parent_resolved["run"]["steps"],
        "branch_step": step,
    }
    run["steps"] = parent_resolved["recipe"]["budgets"][budget]
    run["budget"] = budget
    run["branches"] = []
    return cfg


def start_branch(trunk_state: dict, budget: str, stream, *, device="cpu", mode="opt",
                 precision=None, expected_backend=None, on_step=None):
    """A trainer for the branch of `budget`, holding the trunk's state at its branch point."""
    if trunk_state["branch_point"] is not True:
        raise BranchError("the state is not a branch point")
    step = branch_step(trunk_state["resolved"], budget)
    if trunk_state["step"] != step:
        raise BranchError(f"the state is at step {trunk_state['step']}, the branch of {budget!r} leaves at {step}")

    cfg = branch_resolved(trunk_state["resolved"], budget)
    trainer = Trainer(
        build_model(cfg),
        cfg,
        stream,
        device=device,
        mode=mode,
        precision=precision,
        expected_backend=expected_backend,
        on_step=on_step,
    )
    # A shallow copy: the caller's state is not changed. Restored as the branch's own state.
    as_branch = dict(trunk_state)
    as_branch["resolved"] = cfg
    as_branch["config_hash"] = config.config_hash(cfg)
    as_branch["branch_point"] = False
    ck.restore(trainer, as_branch)
    return trainer


def start_or_resume_branch(parent_resolved: dict, budget: str, stream, root, **trainer_kwargs):
    """Resume the branch from its own checkpoints, or start it from the trunk's branch point."""
    cfg = branch_resolved(parent_resolved, budget)
    if ck.latest_valid_checkpoint(ck.checkpoint_dir(root, cfg)) is not None:
        trainer = Trainer(build_model(cfg), cfg, stream, **trainer_kwargs)
        return trainer, ck.resume_or_start(trainer, root)

    step = branch_step(parent_resolved, budget)
    path = ck.checkpoint_dir(root, parent_resolved) / f"trunk-step{step:08d}.pt"
    if not path.exists():
        raise BranchError(f"no trunk checkpoint for the branch of {budget!r}: {path} does not exist")
    try:
        state = ck.load_checkpoint(path)
    except ck.CheckpointError as exc:
        raise BranchError(f"the trunk checkpoint for the branch of {budget!r} does not load: {exc}") from exc
    trainer = start_branch(state, budget, stream, **trainer_kwargs)
    return trainer, ck.ResumeInfo(step=step, path=path, resume_count=0)


def branch_manifest_block(parent_resolved: dict, budget: str, parent_checkpoint) -> dict:
    """The `branch` block of a branch's manifest; the manifest carries the parent's config."""
    step = branch_step(parent_resolved, budget)
    return {
        "budget": budget,
        "steps": parent_resolved["recipe"]["budgets"][budget],
        "branch_step": step,
        "parent_run": mf.run_name(parent_resolved),
        "parent_checkpoint": str(parent_checkpoint),
    }
