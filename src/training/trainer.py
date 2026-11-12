"""The trainer: one optimizer step at a time, with skip counting (plan §2.3, §7, §13.6).

Attempted steps (`step`) index the data stream and the learning-rate schedule.
Applied updates (`steps_taken`) are counted separately, and every skipped step is
recorded, so a run in which variants overflow at different rates cannot hide it.
The contract is the docstring of tests/test_trainer.py.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.attention.base import MODES
from src.training.optimizer import build_optimizer
from src.training.schedule import lr_at, set_lr

PRECISIONS = ("fp32", "fp16")

_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}
if hasattr(SDPBackend, "CUDNN_ATTENTION"):
    _BACKENDS["cudnn"] = SDPBackend.CUDNN_ATTENTION


@dataclass(frozen=True)
class StepRecord:
    step: int          # index of this step: steps attempted before it
    lr: float          # lr_at(step, resolved)
    loss: float        # mean cross-entropy of the step's tokens, in nats
    grad_norm: float   # global gradient norm before clipping, unscaled; inf or nan if not finite
    skipped: bool      # True if no optimizer update was applied
    scale: float       # GradScaler scale after the step; 1.0 when scaling is off
    seconds: float


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class Trainer:
    def __init__(
        self,
        model,
        resolved: dict,
        stream,
        *,
        device="cpu",
        mode="opt",
        precision=None,
        expected_backend=None,
        on_step: Callable[["Trainer", StepRecord], None] | None = None,
    ):
        self.device = torch.device(device)
        on_cuda = self.device.type == "cuda"

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if precision is None:
            precision = "fp16" if on_cuda else "fp32"
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        if precision == "fp16" and not on_cuda:
            raise ValueError("precision 'fp16' requires a CUDA device")

        batch = resolved["recipe"]["batch"]
        micro_batch = batch["micro_batch"]
        sequences_per_step = batch["sequences_per_step"]
        if micro_batch is None:
            raise ValueError("recipe.batch.micro_batch is unset; set it before starting a run")
        if (
            not _is_int(micro_batch)
            or micro_batch <= 0
            or sequences_per_step % micro_batch != 0
        ):
            raise ValueError(
                f"micro_batch must be a positive divisor of sequences_per_step "
                f"({sequences_per_step}), got {micro_batch!r}"
            )

        scaler_cfg = resolved["recipe"]["grad_scaler"]
        if precision == "fp16" and (scaler_cfg["init_scale"] is None or scaler_cfg["growth_interval"] is None):
            raise ValueError("fp16 needs recipe.grad_scaler init_scale and growth_interval set")

        if expected_backend is not None:
            if not on_cuda:
                raise ValueError("expected_backend is CUDA only")
            if expected_backend not in _BACKENDS:
                raise ValueError(
                    f"expected_backend must be one of {sorted(_BACKENDS)} or None, got {expected_backend!r}"
                )

        total_steps = resolved["run"]["steps"]
        if stream.max_steps < total_steps:
            raise ValueError(
                f"stream has {stream.max_steps} steps, fewer than the run's {total_steps}"
            )

        self.resolved = resolved
        self.stream = stream
        self.mode = mode
        self.precision = precision
        self.expected_backend = expected_backend
        self.on_step = on_step
        self.micro_batch = micro_batch
        self.sequences_per_step = sequences_per_step
        self.total_steps = total_steps
        self.grad_clip_norm = resolved["recipe"]["optimizer"]["grad_clip_norm"]

        self.model = model.to(self.device)
        self.model.train()
        self.optimizer = build_optimizer(self.model, resolved)
        if precision == "fp16":
            self.scaler = torch.amp.GradScaler(
                "cuda",
                init_scale=scaler_cfg["init_scale"],
                growth_interval=scaler_cfg["growth_interval"],
            )
        else:
            self.scaler = None

        self.step = 0
        self.steps_taken = 0
        self.records: list[StepRecord] = []

    @property
    def skipped_steps(self) -> int:
        return self.step - self.steps_taken

    def _forward(self, xb: torch.Tensor) -> torch.Tensor:
        with contextlib.ExitStack() as stack:
            if self.expected_backend is not None:
                stack.enter_context(sdpa_kernel([_BACKENDS[self.expected_backend]]))
            if self.precision == "fp16":
                stack.enter_context(torch.autocast("cuda", dtype=torch.float16))
            logits, _ = self.model(xb, mode=self.mode)
        return logits

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def train_step(self) -> StepRecord:
        if self.step >= self.total_steps:
            raise ValueError(f"the run is over: step {self.step} of {self.total_steps}")

        self._sync()
        t0 = time.perf_counter()
        index = self.step

        x, y = self.stream.batch(index)
        lr = lr_at(index, self.resolved)
        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad(set_to_none=True)

        m = self.micro_batch
        n = self.sequences_per_step
        expected_dtype = torch.float16 if self.precision == "fp16" else torch.float32
        loss_total = 0.0
        for i in range(n // m):
            xb = x[i * m : (i + 1) * m].to(self.device)
            yb = y[i * m : (i + 1) * m].to(self.device)
            logits = self._forward(xb)
            if logits.dtype != expected_dtype:
                raise RuntimeError(
                    f"logits dtype is {logits.dtype}, expected {expected_dtype} for precision "
                    f"{self.precision!r}: dtype check failed (plan §3.1)"
                )
            loss = F.cross_entropy(logits.float().flatten(0, 1), yb.flatten())
            weighted = loss * (m / n)
            if self.scaler is not None:
                self.scaler.scale(weighted).backward()
            else:
                weighted.backward()
            loss_total += loss.item() * m / n

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        finite = math.isfinite(float(grad_norm))

        if self.scaler is None:
            if finite:
                self.optimizer.step()
            scale = 1.0
        else:
            before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale = float(self.scaler.get_scale())
            backed_off = scale < before
            if backed_off != (not finite):
                raise RuntimeError(
                    f"the GradScaler and the trainer disagree about a skipped step at step {index}: "
                    f"grad_norm {float(grad_norm)} finite={finite}, scale {before} -> {scale}"
                )

        skipped = not finite
        self.step += 1
        if not skipped:
            self.steps_taken += 1

        self._sync()
        record = StepRecord(
            step=index,
            lr=float(lr),
            loss=float(loss_total),
            grad_norm=float(grad_norm),
            skipped=bool(skipped),
            scale=float(scale),
            seconds=float(time.perf_counter() - t0),
        )
        self.records.append(record)
        if self.on_step is not None:
            self.on_step(self, record)
        return record

    def train(self, until: int | None = None) -> None:
        target = self.total_steps if until is None else until
        if not _is_int(target) or target < self.step or target > self.total_steps:
            raise ValueError(
                f"until must be an int in [{self.step}, {self.total_steps}], got {target!r}"
            )
        while self.step < target:
            self.train_step()

    def summary(self) -> dict:
        return {
            "steps_attempted": self.step,
            "steps_taken": self.steps_taken,
            "skipped_steps": self.skipped_steps,
            "skipped_in_first_100": sum(1 for r in self.records if r.skipped and r.step < 100),
            "final_scale": self.records[-1].scale if self.records else 1.0,
            "precision": self.precision,
            "micro_batch": self.micro_batch,
            "mode": self.mode,
            "expected_backend": self.expected_backend,
            "total_steps": self.total_steps,
        }
