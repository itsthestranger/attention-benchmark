"""Checkpoint, restore and automatic resume of a training run (plan §5, §12.3, §13.9).

File layout
-----------
A checkpoint directory is ``checkpoint_dir(root, resolved)``, that is
``root / <config hash>``: one directory per configuration, so two configurations
never share one. Inside it:

    step{step:08d}.pt          regular checkpoints; the newest ``keep`` (default 2) survive
    trunk-step{step:08d}.pt    branch-point checkpoints; never pruned here
    .ckpt-*.tmp                a write in progress; removed on failure, ignored by listings

Guarantees
----------
- Atomic: a checkpoint is written to a temporary file in the same directory, flushed to
  disk and renamed onto its final name with ``os.replace``. A crash leaves the previous
  files untouched, plus at worst a stray temporary file that listings ignore. The last two
  regular checkpoints are kept so that a truncated newest file still leaves a fallback.
- Complete: a checkpoint holds the model weights (fp32 master weights, moved to the CPU),
  the optimizer state, the GradScaler state (None when scaling is off), the step records,
  the torch RNG state, the CUDA RNG states when the machine has CUDA, and the full
  resolved configuration.
- Position is ``trainer.step`` and nothing else. The data stream and the learning-rate
  schedule are pure functions of the step, so restoring the step restores both. No other
  position is stored.
- Checked before changed: ``restore`` refuses a state whose config hash, mode or precision
  differ from the trainer's, or a trainer that has already taken a step, before it touches
  the trainer.
- Automatic: ``resume_or_start`` finds the newest checkpoint that loads, from the
  configuration alone. Nothing here asks a question, reads input, seeds the global RNG or
  draws from it.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from src import config
from src.training.trainer import StepRecord

SCHEMA_VERSION = 1

_STEP_FILE = re.compile(r"step([0-9]+)\.pt")
_BRANCH_FILE = re.compile(r"trunk-step([0-9]+)\.pt")
_REQUIRED_KEYS = (
    "schema_version",
    "config_hash",
    "resolved",
    "step",
    "steps_taken",
    "mode",
    "precision",
    "model",
    "optimizer",
    "scaler",
    "records",
    "torch_rng_state",
    "resume_count",
    "branch_point",
)
_RECORD_FIELDS = len(dataclasses.fields(StepRecord))


class CheckpointError(RuntimeError):
    """A checkpoint cannot be read, or cannot be restored into this trainer."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --- Paths and listings ------------------------------------------------------------------------


def checkpoint_dir(root, resolved) -> Path:
    """The directory of one configuration: ``root / config_hash(resolved)``."""
    return Path(root) / config.config_hash(resolved)


def _file_name(step: int, branch_point: bool) -> str:
    prefix = "trunk-" if branch_point else ""
    return f"{prefix}step{step:08d}.pt"


def _listing(directory, pattern: re.Pattern) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    for entry in directory.iterdir():
        match = pattern.fullmatch(entry.name)
        if match and entry.is_file():
            found.append((int(match.group(1)), entry))
    found.sort(key=lambda pair: pair[0])
    return [path for _, path in found]


def list_checkpoints(directory) -> list[Path]:
    """The regular ``step*.pt`` files, oldest first. A missing directory gives []."""
    return _listing(directory, _STEP_FILE)


def list_branch_points(directory) -> list[Path]:
    """The ``trunk-step*.pt`` files, oldest first. A missing directory gives []."""
    return _listing(directory, _BRANCH_FILE)


# --- Saving ------------------------------------------------------------------------------------


def _capture(trainer, *, branch_point: bool, resume_count: int) -> dict:
    """The trainer's whole state as a plain dict. Reads the trainer; changes nothing in it."""
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": config.config_hash(trainer.resolved),
        "resolved": trainer.resolved,
        "step": trainer.step,
        "steps_taken": trainer.steps_taken,
        "mode": trainer.mode,
        "precision": trainer.precision,
        "model": {name: t.detach().cpu().clone() for name, t in trainer.model.state_dict().items()},
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": None if trainer.scaler is None else trainer.scaler.state_dict(),
        "records": [dataclasses.astuple(r) for r in trainer.records],
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": cuda_rng,
        "resume_count": resume_count,
        "branch_point": branch_point,
    }


def _write_atomic(state: dict, final: Path) -> None:
    """Write to a temporary file beside ``final``, then rename it into place."""
    fd, tmp_name = tempfile.mkstemp(dir=final.parent, prefix=".ckpt-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, final)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _prune(directory: Path, keep: int) -> None:
    """Delete all but the newest ``keep`` regular checkpoints. Branch points are not touched."""
    for old in list_checkpoints(directory)[:-keep]:
        old.unlink(missing_ok=True)


def save_checkpoint(trainer, directory, *, branch_point: bool = False, keep: int = 2, resume_count: int = 0) -> Path:
    """Write the trainer's state to ``directory`` and return the file's path.

    Regular checkpoints are named ``step{step:08d}.pt``; branch points are named
    ``trunk-step{step:08d}.pt``. Afterwards only the newest ``keep`` regular checkpoints
    remain.
    """
    if not _is_int(keep) or keep < 1:
        raise ValueError(f"keep must be an int >= 1, got {keep!r}")
    if not _is_int(resume_count) or resume_count < 0:
        raise ValueError(f"resume_count must be an int >= 0, got {resume_count!r}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / _file_name(trainer.step, branch_point)
    state = _capture(trainer, branch_point=branch_point, resume_count=resume_count)
    _write_atomic(state, final)
    _prune(directory, keep)
    return final


# --- Loading and validation --------------------------------------------------------------------


def _records_from(rows: object, step: int) -> list[StepRecord]:
    """Rebuild StepRecord objects from their stored tuples; one per attempted step."""
    if not isinstance(rows, list):
        raise CheckpointError(f"records must be a list, got {type(rows).__name__}")
    if len(rows) != step:
        raise CheckpointError(f"expected {step} step records for step {step}, found {len(rows)}")
    records = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != _RECORD_FIELDS:
            raise CheckpointError(f"malformed step record of type {type(row).__name__}")
        records.append(StepRecord(*row))
    return records


def _validate(state: object) -> None:
    """Raise CheckpointError unless ``state`` has the shape of a schema-1 checkpoint."""
    if not isinstance(state, dict):
        raise CheckpointError(f"not a checkpoint: expected a dict, got {type(state).__name__}")
    missing = [key for key in _REQUIRED_KEYS if key not in state]
    if missing:
        raise CheckpointError(f"not a checkpoint: missing keys {missing}")
    if state["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError(f"schema version {state['schema_version']!r}, expected {SCHEMA_VERSION}")
    for key in ("step", "steps_taken", "resume_count"):
        if not _is_int(state[key]) or state[key] < 0:
            raise CheckpointError(f"{key} must be a non-negative int, got {state[key]!r}")
    if state["steps_taken"] > state["step"]:
        raise CheckpointError("steps_taken is larger than step")
    if not isinstance(state["resolved"], dict):
        raise CheckpointError("the stored configuration is not a dict")
    try:
        actual_hash = config.config_hash(state["resolved"])
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"the stored configuration is not usable: {exc}") from exc
    if state["config_hash"] != actual_hash:
        raise CheckpointError("the stored config hash does not match the stored configuration")
    if not isinstance(state["mode"], str) or not isinstance(state["precision"], str):
        raise CheckpointError("mode and precision must be strings")
    if not isinstance(state["branch_point"], bool):
        raise CheckpointError("branch_point must be a bool")
    if not isinstance(state["model"], dict) or not all(torch.is_tensor(v) for v in state["model"].values()):
        raise CheckpointError("model must be a dict of tensors")
    if not isinstance(state["optimizer"], dict):
        raise CheckpointError("optimizer state must be a dict")
    if state["scaler"] is not None and not isinstance(state["scaler"], dict):
        raise CheckpointError("scaler state must be a dict or None")
    rng = state["torch_rng_state"]
    if not torch.is_tensor(rng) or rng.dtype != torch.uint8:
        raise CheckpointError("torch_rng_state must be a uint8 tensor")
    cuda_rng = state.get("cuda_rng_state")
    if cuda_rng is not None and not isinstance(cuda_rng, list):
        raise CheckpointError("cuda_rng_state must be a list or None")
    _records_from(state["records"], state["step"])


def load_checkpoint(path) -> dict:
    """Read and validate one checkpoint file.

    Raises CheckpointError, with the path in the message, for a missing, truncated,
    unreadable, foreign or other-schema file.
    """
    path = Path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"{path}: cannot read checkpoint: {exc}") from exc
    try:
        _validate(state)
    except CheckpointError as exc:
        raise CheckpointError(f"{path}: {exc}") from exc
    return state


def latest_valid_checkpoint(directory) -> tuple[Path, dict] | None:
    """The newest regular checkpoint that loads, with its state. Damaged files are skipped, not deleted."""
    for path in reversed(list_checkpoints(directory)):
        try:
            return path, load_checkpoint(path)
        except CheckpointError:
            continue
    return None


# --- Restoring ---------------------------------------------------------------------------------


def _check_model(model, saved: dict) -> None:
    expected = model.state_dict()
    if set(saved) != set(expected):
        missing = sorted(set(expected) - set(saved))
        unexpected = sorted(set(saved) - set(expected))
        raise CheckpointError(f"model parameters differ: missing {missing}, unexpected {unexpected}")
    for name, tensor in expected.items():
        if tuple(saved[name].shape) != tuple(tensor.shape):
            raise CheckpointError(
                f"model tensor {name} has shape {tuple(saved[name].shape)}, expected {tuple(tensor.shape)}"
            )


def _check_optimizer(optimizer, model, saved: dict) -> None:
    saved_groups = saved.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(saved.get("state"), dict):
        raise CheckpointError("optimizer state is malformed")
    if len(saved_groups) != len(optimizer.param_groups):
        raise CheckpointError(
            f"optimizer has {len(saved_groups)} parameter groups, the trainer's has {len(optimizer.param_groups)}"
        )
    for saved_group, group in zip(saved_groups, optimizer.param_groups):
        if not isinstance(saved_group, dict) or len(saved_group.get("params", ())) != len(group["params"]):
            raise CheckpointError("optimizer parameter groups differ in size")
    owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if owned != {id(p) for p in model.parameters()}:
        raise CheckpointError("the optimizer does not own the model's parameters")


def _cuda_rng_for(trainer, state: dict) -> list | None:
    """The CUDA RNG states to restore, or None when the trainer does not run on CUDA."""
    if trainer.device.type != "cuda":
        return None
    saved = state.get("cuda_rng_state")
    if not isinstance(saved, list) or len(saved) != torch.cuda.device_count():
        raise CheckpointError("the checkpoint has no CUDA RNG state for this machine's devices")
    return saved


def restore(trainer, state: dict) -> None:
    """Put a loaded state into a fresh trainer built from the same resolved configuration.

    Every check runs before the trainer is changed. A refused restore leaves it untouched.
    """
    _validate(state)
    if state["config_hash"] != config.config_hash(trainer.resolved):
        raise CheckpointError("the checkpoint is for another configuration (config hash differs)")
    if state["mode"] != trainer.mode:
        raise CheckpointError(f"the checkpoint is mode {state['mode']!r}, the trainer is {trainer.mode!r}")
    if state["precision"] != trainer.precision:
        raise CheckpointError(
            f"the checkpoint is precision {state['precision']!r}, the trainer is {trainer.precision!r}"
        )
    if trainer.step != 0 or trainer.steps_taken != 0 or trainer.records:
        raise CheckpointError("the trainer has already taken a step; restore into a fresh trainer")
    if state["step"] > trainer.total_steps:
        raise CheckpointError(f"the checkpoint is at step {state['step']}, past the run's {trainer.total_steps}")
    if (state["scaler"] is None) != (trainer.scaler is None):
        raise CheckpointError("the checkpoint and the trainer disagree about GradScaler state")
    _check_model(trainer.model, state["model"])
    _check_optimizer(trainer.optimizer, trainer.model, state["optimizer"])
    records = _records_from(state["records"], state["step"])
    cuda_rng = _cuda_rng_for(trainer, state)

    # Deep copies: load_state_dict may keep the tensors it is handed, and the optimizer then
    # updates them in place. The loaded state must stay a value that can be restored again.
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    if trainer.scaler is not None:
        trainer.scaler.load_state_dict(copy.deepcopy(state["scaler"]))
    trainer.step = state["step"]
    trainer.steps_taken = state["steps_taken"]
    trainer.records = records
    torch.set_rng_state(state["torch_rng_state"])
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)


@dataclass(frozen=True)
class ResumeInfo:
    step: int                 # 0 for a fresh start
    path: Path | None         # the checkpoint restored from
    resume_count: int         # how many times this run has been resumed, this time included


def resume_or_start(trainer, root) -> ResumeInfo:
    """Restore the newest valid checkpoint of this trainer's configuration, or start fresh.

    Branch points are not resumed from; they belong to branches.
    """
    found = latest_valid_checkpoint(checkpoint_dir(root, trainer.resolved))
    if found is None:
        return ResumeInfo(step=0, path=None, resume_count=0)
    path, state = found
    restore(trainer, state)
    return ResumeInfo(step=trainer.step, path=path, resume_count=state["resume_count"] + 1)


# --- The on_step hook --------------------------------------------------------------------------


class Checkpointer:
    """An ``on_step`` hook that saves checkpoints on a step cadence, a clock cadence, or both.

    After each step: a branch-point save if ``trainer.step`` is in ``branch_steps``, then a
    regular save if one is due. A regular save is due when ``trainer.step`` is a multiple of
    ``every_steps``, when ``every_seconds`` have passed on ``clock`` since the last regular
    save (or since construction), and always after the run's final step. At most one regular
    save happens per call.
    """

    def __init__(
        self,
        root,
        *,
        every_steps: int | None = None,
        every_seconds: float | None = None,
        branch_steps=(),
        keep: int = 2,
        resume_count: int = 0,
        clock=time.monotonic,
    ):
        if every_steps is None and every_seconds is None:
            raise ValueError("give every_steps, every_seconds or both")
        if every_steps is not None and (not _is_int(every_steps) or every_steps < 1):
            raise ValueError(f"every_steps must be an int >= 1, got {every_steps!r}")
        if every_seconds is not None and (
            isinstance(every_seconds, bool)
            or not isinstance(every_seconds, (int, float))
            or not 0 < every_seconds < math.inf
        ):
            raise ValueError(f"every_seconds must be a positive number, got {every_seconds!r}")
        if not _is_int(keep) or keep < 1:
            raise ValueError(f"keep must be an int >= 1, got {keep!r}")
        if not _is_int(resume_count) or resume_count < 0:
            raise ValueError(f"resume_count must be an int >= 0, got {resume_count!r}")
        branches = frozenset(branch_steps)
        if not all(_is_int(s) and s >= 1 for s in branches):
            raise ValueError(f"branch_steps must be ints >= 1, got {sorted(branches, key=str)!r}")

        self.root = Path(root)
        self.every_steps = every_steps
        self.every_seconds = every_seconds
        self.branch_steps = branches
        self.keep = keep
        self.resume_count = resume_count
        self.clock = clock
        self.saved: list[Path] = []
        self._last_regular = clock()

    def _regular_due(self, trainer) -> bool:
        if trainer.step == trainer.total_steps:
            return True
        if self.every_steps is not None and trainer.step % self.every_steps == 0:
            return True
        return self.every_seconds is not None and self.clock() - self._last_regular >= self.every_seconds

    def __call__(self, trainer, record: StepRecord) -> None:
        directory = checkpoint_dir(self.root, trainer.resolved)
        if trainer.step in self.branch_steps:
            self.saved.append(
                save_checkpoint(
                    trainer, directory, branch_point=True, keep=self.keep, resume_count=self.resume_count
                )
            )
        if self._regular_due(trainer):
            self.saved.append(
                save_checkpoint(
                    trainer, directory, branch_point=False, keep=self.keep, resume_count=self.resume_count
                )
            )
            self._last_regular = self.clock()
