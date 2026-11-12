"""Computed wall-clock estimate: the loss each variant would reach in MHA's time (plan §8.5, step 2).

Budget B is MHA's measured time for `reference_steps` (1,526 steps, 100M tokens).
Variant v affords S_v = round(B / t_v) steps in that time, where t_v is its
measured time per step from the timing pass. Its loss at S_v steps is read from
its own Arm A results, interpolated linearly in ln(tokens) between its 100M
branch and its 300M result. Tokens are proportional to steps, so the same line
is linear in ln(steps).

This module computes the estimate and nothing else. It emits no verdict, no
p-value and no interval (plan §11: that is `primary_test.py` alone). Timings are
passed in as a mapping; reading them from disk is another task's job.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Mapping

from src import config
from src.utils import manifest as mf

REFERENCE = "mha"
ARM_C = "iso_wallclock"
ARM_A = "iso_token"
INTERPOLATION = "linear in ln(tokens) between the variant's own 100M branch and its 300M result"

_FULL = "full"  # the 300M result
_LOW = "low"  # the 100M branch


class EstimateError(ValueError):
    """The estimate cannot be computed from the manifests or timings offered."""


# --- Scalar pieces -------------------------------------------------------------


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EstimateError(f"{name} must be a number, got {value!r}")
    x = float(value)
    if not (math.isfinite(x) and x > 0):
        raise EstimateError(f"{name} must be finite and positive, got {value!r}")
    return x


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EstimateError(f"{name} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise EstimateError(f"{name} must be finite, got {value!r}")
    return x


def steps_afforded(ms_reference: float, ms_variant: float, reference_steps: int) -> int:
    """S_v = round(B / t_v) with B = ms_reference * reference_steps.

    Rounds half up (floor(x + 0.5)), not to even.
    """
    t_ref = _positive_finite("ms_reference", ms_reference)
    t_var = _positive_finite("ms_variant", ms_variant)
    if isinstance(reference_steps, bool) or not isinstance(reference_steps, int) or reference_steps < 1:
        raise EstimateError(f"reference_steps must be a positive int, got {reference_steps!r}")
    budget = t_ref * reference_steps
    return int(math.floor(budget / t_var + 0.5))


def interpolate_loss(steps: int, low: tuple[int, float], high: tuple[int, float]) -> float:
    """Loss at `steps` on the line through (steps_low, loss_low) and (steps_high, loss_high), in ln(steps).

    Values outside [steps_low, steps_high] lie on the same line (extrapolation).
    """
    s_lo, l_lo = low
    s_hi, l_hi = high
    for name, s in (("steps_low", s_lo), ("steps_high", s_hi)):
        if isinstance(s, bool) or not isinstance(s, int) or s < 1:
            raise EstimateError(f"{name} must be a positive int, got {s!r}")
    if not s_lo < s_hi:
        raise EstimateError(f"need steps_low < steps_high, got {s_lo} and {s_hi}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise EstimateError(f"steps must be an int >= 1, got {steps!r}")
    l_lo = _finite("loss_low", l_lo)
    l_hi = _finite("loss_high", l_hi)
    x_lo, x_hi, x = math.log(s_lo), math.log(s_hi), math.log(steps)
    fraction = (x - x_lo) / (x_hi - x_lo)
    return l_lo + (l_hi - l_lo) * fraction


# --- Result types --------------------------------------------------------------


@dataclass(frozen=True)
class VariantEstimate:
    variant: str
    status: str  # "estimated" or "ratio_only"
    ms_per_step: float
    step_time_ratio: float  # t_v / t_mha
    steps: int  # S_v
    tokens: int  # S_v * tokens_per_step
    seeds: tuple[int, ...]  # init seeds used; () when ratio_only
    loss_per_seed: tuple[float, ...]  # estimated loss at S_v steps, per seed
    reference_loss_per_seed: tuple[float, ...]  # MHA's 100M-branch loss for the same seeds
    loss: float | None  # mean of loss_per_seed
    reference_loss: float | None  # mean of reference_loss_per_seed
    gap: float | None  # loss - reference_loss; positive = the variant is worse at equal time
    extrapolated: bool  # S_v lies outside [100M steps, 300M steps]
    interpolation: str | None  # INTERPOLATION when estimated, None when ratio_only
    reason: str | None  # for ratio_only: why


@dataclass(frozen=True)
class WallclockEstimate:
    reference_variant: str
    reference_steps: int
    reference_ms_per_step: float
    budget_ms: float  # B
    variants: tuple[VariantEstimate, ...]


# --- Manifests -----------------------------------------------------------------


def _index_arm_a(manifests: list[dict], variants: tuple[str, ...], tag: str, tokens: str,
                 low_label: str, data_seed: int) -> dict:
    """{(variant, init_seed, stage): (peak_lr, val_loss)} for Arm A results of the listed variants."""
    table: dict[tuple[str, int, str], tuple[float, float]] = {}
    for manifest in manifests:
        if not mf.is_result(manifest):
            continue
        name = mf.parse_run_name(manifest["name"])
        if name.variant not in variants or name.arm != tag:
            continue
        if name.tokens != tokens or name.data_seed != data_seed:
            continue
        if name.branch is None:
            stage = _FULL
        elif name.branch == low_label:
            stage = _LOW
        else:
            continue
        key = (name.variant, name.init_seed, stage)
        if key in table:
            raise EstimateError(
                f"two manifests for variant {name.variant!r}, seed {name.init_seed}, "
                f"stage {stage!r} (the {name.branch or tokens} run)"
            )
        table[key] = (name.peak_lr, _finite("val_loss", manifest["final_metrics"]["val_loss"]))
    return table


# --- The estimate --------------------------------------------------------------


def estimate(manifests: list[dict], ms_per_step: dict[str, float]) -> WallclockEstimate:
    arm_c = config.load_arm(ARM_C)
    arm_a = config.load_arm(ARM_A)
    recipe = config.load_recipe()

    variants = tuple(arm_c["variants"])
    if not variants or variants[0] != REFERENCE:
        raise EstimateError(f"the first variant of {ARM_C} must be {REFERENCE!r}, got {variants}")
    reference_steps = arm_c["reference_steps"]
    tokens_per_step = recipe["batch"]["tokens_per_step"]

    full_label = arm_a["budget"]
    if len(arm_a["branches"]) != 1:
        raise EstimateError(f"{ARM_A} must have exactly one branch, got {arm_a['branches']}")
    low_label = arm_a["branches"][0]
    steps_full = recipe["budgets"][full_label]
    steps_low = recipe["budgets"][low_label]

    for variant in variants:
        if variant not in ms_per_step:
            raise EstimateError(f"no timing for variant {variant!r}")
        _positive_finite(f"ms_per_step[{variant!r}]", ms_per_step[variant])
    t_ref = float(ms_per_step[REFERENCE])
    budget_ms = t_ref * reference_steps

    table = _index_arm_a(manifests, variants, arm_a["tag"], full_label, low_label, arm_a["data_seed"])

    def lr_of(variant: str) -> float | None:
        lrs = {lr for (v, _, stage), (lr, _) in table.items() if v == variant and stage == _FULL}
        if len(lrs) > 1:
            raise EstimateError(
                f"variant {variant!r} has 300M results at several learning rates {sorted(lrs)}"
            )
        return next(iter(lrs)) if lrs else None

    def loss_at(variant: str, seed: int, stage: str, lr: float | None) -> float | None:
        entry = table.get((variant, seed, stage))
        if entry is None or lr is None or entry[0] != lr:
            return None
        return entry[1]

    lrs = {v: lr_of(v) for v in variants}
    ref_lr = lrs[REFERENCE]
    # MHA's 100M-branch loss per seed, at MHA's own 300M learning rate.
    ref_loss = {
        s: loss_at(REFERENCE, s, _LOW, ref_lr)
        for (v, s, st) in table
        if v == REFERENCE and st == _LOW and loss_at(REFERENCE, s, _LOW, ref_lr) is not None
    }

    results: list[VariantEstimate] = []
    for variant in variants[1:]:
        ms_v = float(ms_per_step[variant])
        steps = steps_afforded(t_ref, ms_v, reference_steps)
        base = dict(
            variant=variant,
            ms_per_step=ms_v,
            step_time_ratio=ms_v / t_ref,
            steps=steps,
            tokens=steps * tokens_per_step,
            extrapolated=not (steps_low <= steps <= steps_full),
        )

        candidates = sorted({s for (v, s, st) in table if v == variant and st == _FULL})
        usable = []
        for seed in candidates:
            full_loss = loss_at(variant, seed, _FULL, lrs[variant])
            low_loss = loss_at(variant, seed, _LOW, lrs[variant])
            if full_loss is None or low_loss is None:
                continue
            if seed not in ref_loss:
                continue
            usable.append((seed, low_loss, full_loss))

        if not usable:
            if not any(v == variant for (v, _, _) in table):
                reason = f"no Arm A results for {variant}"
            elif not any(
                loss_at(variant, s, _FULL, lrs[variant]) is not None
                and loss_at(variant, s, _LOW, lrs[variant]) is not None
                for s in candidates
            ):
                reason = (
                    f"no seed of {variant} has both a {full_label} result and a {low_label} "
                    f"branch at its {full_label} learning rate"
                )
            elif ref_lr is None or not ref_loss:
                reason = f"no {REFERENCE} {low_label} branch to compare with"
            else:
                reason = (
                    f"no seed has both a {variant} {low_label} branch and a {REFERENCE} "
                    f"{low_label} branch at the matching learning rate"
                )
            results.append(
                VariantEstimate(
                    status="ratio_only",
                    seeds=(),
                    loss_per_seed=(),
                    reference_loss_per_seed=(),
                    loss=None,
                    reference_loss=None,
                    gap=None,
                    interpolation=None,
                    reason=reason,
                    **base,
                )
            )
            continue

        seeds = tuple(seed for seed, _, _ in usable)
        loss_per_seed = tuple(
            interpolate_loss(steps, (steps_low, low), (steps_full, full))
            for _, low, full in usable
        )
        reference_per_seed = tuple(ref_loss[seed] for seed in seeds)
        mean_loss = sum(loss_per_seed) / len(loss_per_seed)
        mean_ref = sum(reference_per_seed) / len(reference_per_seed)
        results.append(
            VariantEstimate(
                status="estimated",
                seeds=seeds,
                loss_per_seed=loss_per_seed,
                reference_loss_per_seed=reference_per_seed,
                loss=mean_loss,
                reference_loss=mean_ref,
                gap=mean_loss - mean_ref,
                interpolation=INTERPOLATION,
                reason=None,
                **base,
            )
        )

    return WallclockEstimate(
        reference_variant=REFERENCE,
        reference_steps=reference_steps,
        reference_ms_per_step=t_ref,
        budget_ms=budget_ms,
        variants=tuple(results),
    )


def as_dict(result: WallclockEstimate) -> dict:
    """A plain, JSON-serialisable dict of the estimate."""
    return dataclasses.asdict(result)
