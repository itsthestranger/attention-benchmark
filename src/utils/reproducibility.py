"""Determinism and reproducibility measurement (plan §5.1, §13).

`ref` mode must reproduce bit for bit: `deterministic_algorithms()` runs the
trainer with torch's deterministic algorithms switched on, and the test suite
checks that two runs agree exactly. `opt` mode is not deterministic on a GPU;
`measure_reproducibility` runs identical configurations several times and
records how far the losses move. This module measures and records. It never
compares a measurement against a threshold, and it neither seeds nor reads the
global RNG.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from src.attention.base import MODES
from src.config import config_hash
from src.data.loader import TokenSplit
from src.training.trainer import PRECISIONS

CUBLAS_ENV = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_VALUE = ":4096:8"


@contextlib.contextmanager
def deterministic_algorithms(device=None):
    """Turn on torch's deterministic algorithms for the block, then restore the previous setting.

    The previous setting and its warn_only flag are restored on exit, also on error.
    On CUDA, cuBLAS needs CUBLAS_WORKSPACE_CONFIG set before the first CUDA call.
    If `device` is CUDA and the variable is unset, RuntimeError is raised before
    any setting is changed. `device=None` (the default) means no CUDA check.
    """
    if device is not None and torch.device(device).type == "cuda" and CUBLAS_ENV not in os.environ:
        raise RuntimeError(
            f"deterministic CUDA runs need the environment variable {CUBLAS_ENV}='{CUBLAS_VALUE}' "
            "set before the first CUDA call (for example: "
            f"export {CUBLAS_ENV}='{CUBLAS_VALUE}')"
        )
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


@dataclass(frozen=True)
class RunTrace:
    """What one run produced, one entry per step, plus the applied-update count."""

    losses: list
    lrs: list
    grad_norms: list
    skipped: list
    steps_taken: int


def run_losses(build_run: Callable, n_steps: int) -> RunTrace:
    """Train a fresh trainer for n_steps steps and return its trace.

    `build_run()` must return a trainer that has not started training.
    """
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive int, got {n_steps!r}")
    trainer = build_run()
    if trainer.step != 0:
        raise ValueError(f"build_run must return a fresh trainer, got step {trainer.step}")
    trainer.train(until=n_steps)
    records = trainer.records
    if len(records) != n_steps or [r.step for r in records] != list(range(n_steps)):
        raise RuntimeError(f"expected records for steps 0..{n_steps - 1}, got {len(records)}")
    return RunTrace(
        losses=[r.loss for r in records],
        lrs=[r.lr for r in records],
        grad_norms=[r.grad_norm for r in records],
        skipped=[r.skipped for r in records],
        steps_taken=trainer.steps_taken,
    )


def _abs_diff(x: float, y: float) -> float:
    if x == y:
        return 0.0
    d = abs(x - y)
    return math.inf if math.isnan(d) else float(d)


def _rel_diff(x: float, y: float) -> float:
    """|x - y| / |y|, with y the reference. Non-finite or zero reference gives inf when x != y."""
    if x == y:
        return 0.0
    d = _abs_diff(x, y)
    if not math.isfinite(d) or y == 0 or not math.isfinite(y):
        return math.inf
    return d / abs(y)


def compare_runs(a: RunTrace, b: RunTrace) -> dict:
    """Compare two traces. `b` is the reference for the relative difference.

    Raises ValueError if the traces differ in step count, learning-rate series,
    skipped flags or steps_taken: those must match exactly whatever the numerics.
    Losses are compared with ==, so bit_identical means every loss is equal as a float.
    A difference that is NaN is reported as inf.
    """
    n = len(a.losses)
    if n == 0:
        raise ValueError("cannot compare empty runs")
    if len(b.losses) != n:
        raise ValueError(f"runs have different numbers of steps: {n} and {len(b.losses)}")
    if len(a.lrs) != n or len(b.lrs) != n:
        raise ValueError("runs have different learning-rate series")
    if a.lrs != b.lrs:
        raise ValueError("runs have different learning-rate series")
    if len(a.skipped) != n or len(b.skipped) != n or list(a.skipped) != list(b.skipped):
        raise ValueError("runs have different skipped flags")
    if a.steps_taken != b.steps_taken:
        raise ValueError(f"runs took different numbers of steps: {a.steps_taken} and {b.steps_taken}")

    abs_diffs = [_abs_diff(x, y) for x, y in zip(a.losses, b.losses)]
    rel_diffs = [_rel_diff(x, y) for x, y in zip(a.losses, b.losses)]
    first = next((i for i, (x, y) in enumerate(zip(a.losses, b.losses)) if x != y), None)
    return {
        "n_steps": n,
        "bit_identical": first is None,
        "max_abs_loss_diff": max(abs_diffs),
        "max_rel_loss_diff": max(rel_diffs),
        "final_abs_loss_diff": abs_diffs[-1],
        "first_differing_step": first,
    }


def measure_reproducibility(build_run: Callable, n_steps: int, n_pairs: int) -> dict:
    """Run n_pairs + 1 identical runs and compare each of runs 1..n_pairs with run 0 (the reference)."""
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 1:
        raise ValueError(f"n_pairs must be a positive int, got {n_pairs!r}")
    runs = [run_losses(build_run, n_steps) for _ in range(n_pairs + 1)]
    pairs = [compare_runs(runs[i], runs[0]) for i in range(1, n_pairs + 1)]
    return {
        "n_steps": n_steps,
        "n_pairs": n_pairs,
        "pairs": pairs,
        "max_abs_loss_diff": max(p["max_abs_loss_diff"] for p in pairs),
        "max_rel_loss_diff": max(p["max_rel_loss_diff"] for p in pairs),
        "bit_identical": all(p["bit_identical"] for p in pairs),
    }


def reproducibility_record(measurement: dict, *, resolved: dict, mode: str, precision: str, device: str) -> dict:
    """The measurement, with what identifies it. JSON-serialisable."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    dev = torch.device(device)
    on_cuda = dev.type == "cuda"
    gpu_name = None
    cuda_version = None
    if on_cuda:
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(index)
        cuda_version = torch.version.cuda
    return {
        "measurement": measurement,
        "config_hash": config_hash(resolved),
        "mode": mode,
        "precision": precision,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
    }


def synthetic_split(n_tokens: int, vocab: int, seed: int) -> TokenSplit:
    """SYNTHETIC uniform random tokens for exercising the pipeline. Not a corpus.

    Uses its own numpy generator seeded by `seed`; the global RNG is not touched.
    """
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    return TokenSplit(tokens, np.array([0, n_tokens], dtype=np.int64))
