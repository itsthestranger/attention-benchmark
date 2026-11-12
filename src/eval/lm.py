"""Validation loss, and the same loss split by context value (plan §8.4, prereg `context_use`).

The loss is computed as the trainer computes it: the forward pass in `mode`, under
fp16 autocast on CUDA when the precision is fp16, and cross-entropy in float32 from
`logits.float()`. Per-token losses are summed in float64.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.data.loader import TokenSplit, eval_windows
from src.eval.positions import bucket_index, context_values, validate_buckets

PRECISIONS = ("fp16", "fp32")


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


@dataclass(frozen=True)
class ContextSpec:
    buckets: tuple[tuple[int, int], ...]
    late_context_min: int
    icl_late: tuple[int, int]
    icl_early: tuple[int, int]

    def __post_init__(self) -> None:
        validate_buckets(self.buckets)
        if isinstance(self.late_context_min, bool) or not isinstance(self.late_context_min, int):
            raise ValueError(f"late_context_min must be an int, got {self.late_context_min!r}")
        for name, window in (("icl_late", self.icl_late), ("icl_early", self.icl_early)):
            if len(window) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in window):
                raise ValueError(f"{name} must be a (low, high) pair of ints, got {window!r}")
            if window[0] > window[1]:
                raise ValueError(f"{name} = {window!r} is reversed")

    @classmethod
    def from_prereg(cls, prereg: dict) -> "ContextSpec":
        block = prereg["context_use"]
        icl = block["icl_score"]
        return cls(
            buckets=tuple((int(lo), int(hi)) for lo, hi in block["buckets"]),
            late_context_min=int(block["late_context_min"]),
            icl_late=(int(icl["late"][0]), int(icl["late"][1])),
            icl_early=(int(icl["early"][0]), int(icl["early"][1])),
        )


def _add(acc: list, values: np.ndarray, mask: np.ndarray) -> None:
    """acc = [sum, count] over the masked per-token losses."""
    acc[0] += float(values[mask].sum())
    acc[1] += int(mask.sum())


def _mean(acc: list) -> float | None:
    return acc[0] / acc[1] if acc[1] else None


def _window_block(low: int, high: int, acc: list) -> dict:
    return {"low": int(low), "high": int(high), "n_tokens": acc[1], "loss": _mean(acc)}


def evaluate(
    model,
    split: TokenSplit,
    *,
    context_length: int,
    batch_size: int,
    spec: ContextSpec,
    device="cpu",
    mode="opt",
    precision=None,
    max_windows: int | None = None,
) -> dict:
    """Validation loss of `model` on `split`, overall and split by context value.

    The model is put in eval mode and its previous training/eval state of every
    submodule is restored afterwards, also when an exception is raised.
    """
    T = _positive_int("context_length", context_length)
    _positive_int("batch_size", batch_size)
    dev = torch.device(device)
    if precision is None:
        precision = "fp16" if dev.type == "cuda" else "fp32"
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "fp16" and dev.type != "cuda":
        raise ValueError("fp16 evaluation needs a CUDA device; use fp32 on CPU")
    if max_windows is not None:
        _positive_int("max_windows", max_windows)

    buckets = validate_buckets(spec.buckets)
    total_windows = split.n_windows(T)
    if total_windows == 0:
        raise ValueError(f"split has no window of length {T}")
    limit = total_windows if max_windows is None else min(total_windows, max_windows)

    total = [0.0, 0]
    per_bucket = [[0.0, 0] for _ in buckets]
    late = [0.0, 0]
    icl_late = [0.0, 0]
    icl_early = [0.0, 0]
    late_min = spec.late_context_min
    n_windows = 0

    autocast = (
        torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else contextlib.nullcontext()
    )
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for first, x, y in eval_windows(split, T, batch_size):
                if first >= limit:
                    break
                keep = min(x.shape[0], limit - first)
                x, y = x[:keep], y[:keep]
                with autocast:
                    logits, _ = model(x.to(dev), mode=mode)
                loss = F.cross_entropy(
                    logits.float().flatten(0, 1), y.flatten().to(dev), reduction="none"
                ).view(keep, T)
                finite = torch.isfinite(loss)
                if not bool(finite.all()):
                    bad_rows = (~finite).any(dim=1).nonzero().flatten()
                    raise ValueError(f"non-finite loss in window {first + int(bad_rows[0])}")

                l64 = loss.detach().to(torch.float64).cpu().numpy()
                c = context_values(split.doc_starts, first, keep, T)
                idx = bucket_index(c, buckets)

                total[0] += float(l64.sum())
                total[1] += l64.size
                for b in range(len(buckets)):
                    _add(per_bucket[b], l64, idx == b)
                _add(late, l64, c >= late_min)
                _add(icl_late, l64, (c >= spec.icl_late[0]) & (c <= spec.icl_late[1]))
                _add(icl_early, l64, (c >= spec.icl_early[0]) & (c <= spec.icl_early[1]))
                n_windows += keep
    finally:
        for m, flag in modes:
            m.training = flag

    late_loss = _mean(late)
    early_loss = _mean(icl_early)
    late_icl_loss = _mean(icl_late)
    score = None if late_icl_loss is None or early_loss is None else late_icl_loss - early_loss
    return {
        "loss": total[0] / total[1],
        "n_tokens": total[1],
        "n_windows": n_windows,
        "buckets": [
            _window_block(low, high, per_bucket[b]) for b, (low, high) in enumerate(buckets)
        ],
        "late_context": {"min_c": late_min, "n_tokens": late[1], "loss": late_loss},
        "icl": {
            "late": _window_block(spec.icl_late[0], spec.icl_late[1], icl_late),
            "early": _window_block(spec.icl_early[0], spec.icl_early[1], icl_early),
            "score": score,
        },
    }
