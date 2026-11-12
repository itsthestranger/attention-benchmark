"""The warmup-stable-decay learning-rate schedule (plan §7.2).

The learning rate is a pure function of the optimizer step. Nothing here keeps
a position, so a resume or a branch from a trunk checkpoint is on schedule by
construction.
"""

from __future__ import annotations

import math


def decay_start(total_steps: int, decay_fraction: float) -> int:
    """The step at which decay begins: round(total_steps * (1 - decay_fraction))."""
    return int(round(total_steps * (1 - decay_fraction)))


def wsd_lr(
    step: int,
    *,
    peak_lr: float,
    total_steps: int,
    warmup_steps: int,
    decay_fraction: float,
    decay_shape: str,
    final_lr: float = 0.0,
) -> float:
    """The learning rate of optimizer step `step`, counted from 0 (0 <= step <= total_steps)."""
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps!r}")
    if not 0 <= step <= total_steps:
        raise ValueError(f"step must be in [0, {total_steps}], got {step!r}")
    if warmup_steps < 1:
        raise ValueError(f"warmup_steps must be at least 1, got {warmup_steps!r}")
    if not 0 < decay_fraction < 1:
        raise ValueError(f"decay_fraction must be strictly between 0 and 1, got {decay_fraction!r}")
    if decay_shape != "one_minus_sqrt":
        raise ValueError(f"unsupported decay_shape {decay_shape!r}; only 'one_minus_sqrt'")

    warm = min(step / warmup_steps, 1.0)
    d = decay_start(total_steps, decay_fraction)
    prog = 0.0 if step <= d else (step - d) / (total_steps - d)
    decay = 1 - math.sqrt(prog)
    return float(final_lr + (peak_lr - final_lr) * warm * decay)


def lr_at(step: int, resolved: dict) -> float:
    """The learning rate of step `step` for a resolved config's run and recipe schedule."""
    sched = resolved["recipe"]["schedule"]
    if sched.get("name") != "wsd":
        raise ValueError(f"schedule must be 'wsd', got {sched.get('name')!r}")
    return wsd_lr(
        step,
        peak_lr=resolved["run"]["peak_lr"],
        total_steps=resolved["run"]["steps"],
        warmup_steps=sched["warmup_steps"],
        decay_fraction=sched["decay_fraction"],
        decay_shape=sched["decay_shape"],
        final_lr=sched["final_lr"],
    )


def set_lr(optimizer, lr: float) -> None:
    """Set the learning rate of every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr
