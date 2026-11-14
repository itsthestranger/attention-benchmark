"""The learning-rate sweep rules of plan §7.4: the grid, the edge rule and the tie rule.

Every choice is made from the sweep runs alone, by the fixed rules below, never by hand:

- A grid is the factor-2 lattice anchored at its lowest required rate (MHA: 3e-4; a
  variant: its reference rate halved). Its tested points must lie on that lattice and
  must include every required rate, with no gap between the lowest and highest tested
  rate; otherwise the status is INCOMPLETE and nothing is chosen.
- Rates are matched by their run-name form (`manifest.format_lr`), never by float tolerance.
- A diverged point is a tested grid point that can never win. If every point of a
  complete grid diverged, `SweepError` is raised.
- The winner has the lowest loss; exactly equal losses take the lower rate.
- Edge rule: a winner at the lowest or highest tested rate gives EXTEND, with the next
  point one factor-2 step beyond it. Nothing is chosen.
- Tie rule (variants only, after the edge rule): if the reference rate did not diverge
  and the winner beats it by strictly less than sigma_init, the reference rate is chosen.

`points_from_manifests` reads only the standalone sweep runs of one name at the sweep
budget, at the sweep seed, in the iso-token arm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from src import config
from src.utils import manifest as mf

# Plan §7.3 and §13.6: a difference above 0.5% of the steps invalidates the comparison.
# Defined here so that the tables and the sweep read the same value.
STEPS_TAKEN_PARITY = 0.005

MHA_GRID = (3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3)
REFERENCE = "mha"
SWEEP_ARM = "iso_token"

CHOSEN = "chosen"
EXTEND = "extend"
INCOMPLETE = "incomplete"
RULE_WINNER = "lowest loss"
RULE_TIE = "tie rule: reference rate kept"


class SweepError(RuntimeError):
    """A refusal: the sweep cannot be read by the rules. The message says why."""


@dataclass(frozen=True)
class Point:
    lr: float
    loss: float | None  # None for a diverged point
    diverged: bool
    run: str
    reason: str | None = None


@dataclass(frozen=True)
class Choice:
    status: str  # CHOSEN, EXTEND or INCOMPLETE
    lr: float | None  # the rate to use, when CHOSEN
    source_run: str | None  # the sweep run at that rate: the lr_source of later runs
    best_lr: float | None  # the lowest-loss rate, when the grid is complete
    next_lr: float | None  # the point to run next, when EXTEND
    missing: tuple  # the required or gap rates, when INCOMPLETE
    rule: str | None  # RULE_WINNER or RULE_TIE, when CHOSEN
    shared_loss: float | None  # loss at the reference rate (None if it diverged or is not a variant)
    tuned_loss: float | None  # loss at the lowest-loss rate, when CHOSEN
    points: tuple  # every point, sorted by rate
    name: str = ""


# --- Rates ----------------------------------------------------------------------------------


def _key(lr: float) -> str:
    try:
        return mf.format_lr(lr)
    except mf.ManifestError as exc:
        raise SweepError(f"learning rate {lr!r} has no run-name form: {exc}") from exc


def _canonical(lr: float) -> float:
    """The float that the run-name form of `lr` denotes. New rates are always this."""
    return mf.parse_lr(_key(lr))


def variant_grid(reference_lr: float) -> tuple:
    """The three points a variant needs: half, one and two times the reference rate."""
    return (_canonical(reference_lr / 2), _canonical(reference_lr), _canonical(reference_lr * 2))


def _lattice_index(lr: float, anchor: float) -> int:
    """The integer k with lr = anchor * 2**k in run-name form. Refuses rates off the lattice."""
    if not (isinstance(lr, float) and math.isfinite(lr) and lr > 0):
        raise SweepError(f"learning rate {lr!r} is not a positive finite number")
    k = round(math.log2(lr / anchor))
    if _key(anchor * 2.0**k) != _key(lr):
        raise SweepError(
            f"learning rate {_key(lr)} is not on the factor-2 grid from {_key(anchor)}"
        )
    return k


# --- The rules ------------------------------------------------------------------------------


def _indexed(points: Sequence[Point], anchor: float) -> dict:
    """{lattice index: point}. Two runs at one rate are refused."""
    indexed: dict[int, Point] = {}
    for point in points:
        if not isinstance(point, Point):
            raise SweepError(f"not a sweep point: {point!r}")
        if not point.diverged and (
            point.loss is None or not math.isfinite(point.loss)
        ):
            raise SweepError(f"{point.run}: a point that did not diverge needs a finite loss")
        k = _lattice_index(point.lr, anchor)
        if k in indexed:
            other = indexed[k].run
            raise SweepError(
                f"two runs at learning rate {_key(point.lr)}: {other} and {point.run}"
            )
        indexed[k] = point
    return indexed


def _assess(points: Sequence[Point], required: Sequence[float], anchor: float):
    """Completeness, then the winner and the edge rule. Returns (status, indexed, best, missing, next)."""
    indexed = _indexed(points, anchor)
    required_k = [_lattice_index(lr, anchor) for lr in required]
    if indexed:
        low = min(min(indexed), min(required_k))
        high = max(max(indexed), max(required_k))
    else:
        low, high = min(required_k), max(required_k)
    missing = tuple(
        _canonical(anchor * 2.0**k) for k in range(low, high + 1) if k not in indexed
    )
    if missing:
        return INCOMPLETE, indexed, None, missing, None

    live = [p for p in indexed.values() if not p.diverged]
    if not live:
        raise SweepError("every point of the complete grid diverged: no learning rate can be chosen")
    best = min(live, key=lambda p: (p.loss, p.lr))
    best_k = _lattice_index(best.lr, anchor)
    if best_k == max(indexed):
        return EXTEND, indexed, best, (), _canonical(best.lr * 2)
    if best_k == min(indexed):
        return EXTEND, indexed, best, (), _canonical(best.lr / 2)
    return CHOSEN, indexed, best, (), None


def _choice(name, status, points, *, best=None, missing=(), next_lr=None,
            chosen=None, rule=None, shared=None, tuned=None) -> Choice:
    return Choice(
        status=status,
        lr=chosen.lr if chosen is not None else None,
        source_run=chosen.run if chosen is not None else None,
        best_lr=best.lr if best is not None else None,
        next_lr=next_lr,
        missing=missing,
        rule=rule,
        shared_loss=shared,
        tuned_loss=tuned,
        points=tuple(sorted(points, key=lambda p: p.lr)),
        name=name,
    )


def choose_reference(points: Sequence[Point], name: str = REFERENCE) -> Choice:
    """MHA's rate: the lowest-loss point of the grid, after the edge rule. No tie rule."""
    status, indexed, best, missing, next_lr = _assess(points, MHA_GRID, MHA_GRID[0])
    if status == INCOMPLETE:
        return _choice(name, INCOMPLETE, points, missing=missing)
    if status == EXTEND:
        return _choice(name, EXTEND, points, best=best, next_lr=next_lr)
    return _choice(name, CHOSEN, points, best=best, chosen=best, rule=RULE_WINNER,
                   tuned=best.loss)


def choose_variant(
    points: Sequence[Point], *, reference_lr: float, sigma_init: float, name: str = ""
) -> Choice:
    """A variant's or ladder size's rate: the edge rule, then the tie rule against MHA's rate."""
    if (
        isinstance(sigma_init, bool)
        or not isinstance(sigma_init, (int, float))
        or not math.isfinite(sigma_init)
        or sigma_init <= 0
    ):
        raise SweepError(f"sigma_init must be a positive finite number, got {sigma_init!r}")
    grid = variant_grid(reference_lr)
    status, indexed, best, missing, next_lr = _assess(points, grid, grid[0])
    if status == INCOMPLETE:
        return _choice(name, INCOMPLETE, points, missing=missing)
    if status == EXTEND:
        return _choice(name, EXTEND, points, best=best, next_lr=next_lr)

    ref_k = _lattice_index(grid[1], grid[0])
    ref = indexed[ref_k]
    if best is ref:
        return _choice(name, CHOSEN, points, best=best, chosen=best, rule=RULE_WINNER,
                       shared=best.loss, tuned=best.loss)
    if not ref.diverged and (ref.loss - best.loss) < sigma_init:
        return _choice(name, CHOSEN, points, best=best, chosen=ref, rule=RULE_TIE,
                       shared=ref.loss, tuned=best.loss)
    return _choice(name, CHOSEN, points, best=best, chosen=best, rule=RULE_WINNER,
                   shared=None if ref.diverged else ref.loss, tuned=best.loss)


# --- Reading the points from manifests ------------------------------------------------------


def points_from_manifests(
    manifests: Sequence[Mapping], name: str, *, sweep_seed: int, prereg: Mapping | None = None
) -> list[Point]:
    """The sweep points of `name`: standalone runs at the sweep budget and seed, iso-token arm.

    A completed run is a point, diverged if it skipped more than STEPS_TAKEN_PARITY of its
    steps. An unfinished run marked diverged is a diverged point. Anything else is no point.
    """
    if prereg is None:
        prereg = config.load_prereg()
    budget = prereg["test_selection"]["sigma_init"]["budget"]
    arm = config.load_arm(SWEEP_ARM)
    points: list[Point] = []
    for manifest in manifests:
        parsed = mf.parse_run_name(manifest["name"])
        if (
            parsed.variant != name
            or parsed.arm != arm["tag"]
            or parsed.tokens != budget
            or parsed.branch is not None
            or parsed.init_seed != sweep_seed
            or parsed.data_seed != arm["data_seed"]
            or manifest.get("kind") != "run"
        ):
            continue
        if manifest["completed"] is True:
            steps = manifest["steps"]
            if steps["skipped"] > STEPS_TAKEN_PARITY * steps["total"]:
                points.append(Point(
                    lr=parsed.peak_lr, loss=None, diverged=True, run=manifest["name"],
                    reason=(f"skipped {steps['skipped']} of {steps['total']} steps, more than "
                            f"{STEPS_TAKEN_PARITY:g} of the steps"),
                ))
            else:
                points.append(Point(
                    lr=parsed.peak_lr, loss=float(manifest["final_metrics"]["val_loss"]),
                    diverged=False, run=manifest["name"],
                ))
        else:
            metrics = manifest.get("final_metrics") or {}
            if metrics.get("diverged") is True:
                points.append(Point(
                    lr=parsed.peak_lr, loss=None, diverged=True, run=manifest["name"],
                    reason=str(metrics.get("reason") or "diverged, no reason recorded"),
                ))
    return sorted(points, key=lambda p: p.lr)


# --- The sweep seed (plan §7.4, §8.1) ----------------------------------------------------------


def check_sweep_seed(seed: int, prereg: Mapping) -> int:
    """The seed of the sweep runs, or SweepError if an arm uses that seed.

    A sweep run on a registered seed would also be an arm's run (plan §7.4, §8.1), so every
    registered seed of every arm is refused.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SweepError(f"sweep seed must be a non-negative int, got {seed!r}")
    registered = (
        ("seed_count.base_seeds", prereg["seed_count"]["base_seeds"]),
        ("seed_count.added_seeds", prereg["seed_count"]["added_seeds"]),
        ("null_control.init_seeds", prereg["null_control"]["init_seeds"]),
        ("exploratory.init_seeds", prereg["exploratory"]["init_seeds"]),
        ("positive_control.init_seeds", prereg["positive_control"]["init_seeds"]),
    )
    for where, seeds in registered:
        if seed in seeds:
            raise SweepError(
                f"sweep seed {seed} is a registered seed ({where} in prereg.yaml); an arm uses it, "
                "so a sweep run on it would also be an arm's run: choose an unregistered seed"
            )
    return seed


def registered_sweep_seed(prereg: Mapping) -> int:
    """`lr_sweep.init_seed` of prereg.yaml (decision 219), checked by `check_sweep_seed`."""
    try:
        seed = prereg["lr_sweep"]["init_seed"]
    except (KeyError, TypeError) as exc:
        raise SweepError("prereg.yaml has no lr_sweep.init_seed: the sweep seed is registered there") from exc
    return check_sweep_seed(seed, prereg)
