"""The calibration ladder as a unit of measurement (plan §8.4) and the seed-count rule (plan §1.4).

The ladder is MHA at three sizes. Final loss is fitted against the log of the
non-embedding parameter count,

    L = a - b ln N,

and a loss gap D between a variant and MHA then reads as an effective-parameter
multiplier exp(-D / b): "performs like MHA with 0.94x the parameters".

Nothing here emits a verdict; that is `primary_test.py` alone (plan §11). This
module only converts gaps that `primary_test.py` has already computed, and
applies the seed-count rule, which reads MHA runs only.

Two choices in the seed-count rule are Phil's (docs/decisions.md, rows 79 and
80), fixed below as constants:

- the power of Welch's test is the exact one, not the pooled-variance value of
  the plan's §1.3 table;
- under the paired test the rule uses the worst case sigma_d = sqrt(2) *
  sigma_init, because sigma_d is unknown until a variant has been read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import stats

from analysis import primary_test as pt

# Phil, 2026-11-12 (decision 79): the power of the test that is actually run.
WELCH_POWER_METHOD = pt.WELCH_EXACT
# Phil, 2026-11-12 (decision 80): sigma_d at its worst case, in units of sigma_init.
PAIRED_WORST_CASE_SIGMA_D = math.sqrt(2.0)


class LadderError(ValueError):
    """The ladder cannot be fitted, or cannot convert the gap offered."""


# --- The fit (plan §8.4, "as a unit") ----------------------------------------


@dataclass(frozen=True)
class LadderPoint:
    name: str
    n_params: int  # non-embedding parameters
    seeds: tuple[int, ...]
    losses: tuple[float, ...]  # in the order of `seeds`
    mean: float
    # The slope is linear in the run losses: b = sum over points of
    # slope_weight * (sum of the point's losses).
    slope_weight: float


@dataclass(frozen=True)
class LadderFit:
    a: float
    b: float  # nats per e-fold of parameters; positive when more parameters lower the loss
    se_b: float
    df: int  # residual degrees of freedom: runs - 2
    residual_sd: float
    n_runs: int
    points: tuple[LadderPoint, ...]

    def predict(self, n_params: float) -> float:
        return self.a - self.b * math.log(n_params)

    def point(self, name: str) -> LadderPoint:
        for p in self.points:
            if p.name == name:
                return p
        raise LadderError(f"the ladder has no point {name!r}; it has {[p.name for p in self.points]}")

    def slope_interval(self, confidence: float = pt.CONFIDENCE) -> tuple[float, float]:
        half = float(stats.t.ppf(0.5 + confidence / 2.0, self.df)) * self.se_b
        return self.b - half, self.b + half


def fit_ladder(points: Mapping[str, tuple[int, Mapping[int, float]]]) -> LadderFit:
    """Least squares of loss on ln(non-embedding params) over every run.

    `points[name] = (n_params, {init_seed: final loss})`. Every run counts
    once, so a point with more seeds weighs more. The slope's standard error
    comes from the residuals of all runs around the line, which therefore
    include both seed noise and any curvature across the three sizes.
    """
    if len(points) < 2:
        raise LadderError("a ladder needs at least two sizes")
    names, sizes, seeds, losses = [], [], [], []
    for name, (n_params, by_seed) in points.items():
        if isinstance(n_params, bool) or not isinstance(n_params, (int, np.integer)) or n_params <= 0:
            raise LadderError(f"{name}: n_params must be a positive int, got {n_params!r}")
        if not by_seed:
            raise LadderError(f"{name}: no runs")
        point_seeds = tuple(sorted(by_seed))
        values = tuple(float(by_seed[s]) for s in point_seeds)
        if not all(math.isfinite(v) for v in values):
            raise LadderError(f"{name}: a loss is not finite")
        names.append(name)
        sizes.append(int(n_params))
        seeds.append(point_seeds)
        losses.append(values)
    if len(set(sizes)) != len(sizes):
        raise LadderError(f"two ladder points have the same size: {sorted(sizes)}")

    counts = np.array([len(v) for v in losses], dtype=np.float64)
    n_runs = int(counts.sum())
    if n_runs < 3:
        raise LadderError("the fit needs at least three runs to estimate its own uncertainty")
    x = np.log(np.array(sizes, dtype=np.float64))
    means = np.array([np.mean(v) for v in losses], dtype=np.float64)
    x_bar = float(np.sum(counts * x) / n_runs)
    y_bar = float(np.sum(counts * means) / n_runs)
    sxx = float(np.sum(counts * (x - x_bar) ** 2))
    # L = a - b ln N: b is minus the regression slope.
    weights = -(x - x_bar) / sxx
    # Centred losses: the slope comes from differences that are small next to
    # the losses themselves.
    b = float(np.sum(weights * counts * (means - y_bar)))
    a = y_bar + b * x_bar
    rss = 0.0
    for xi, values in zip(x, losses):
        rss += float(np.sum(((np.array(values) - y_bar) + b * (xi - x_bar)) ** 2))
    df = n_runs - 2
    residual_sd = math.sqrt(rss / df)
    se_b = residual_sd / math.sqrt(sxx)
    order = np.argsort(sizes)
    return LadderFit(
        a=a,
        b=b,
        se_b=se_b,
        df=df,
        residual_sd=residual_sd,
        n_runs=n_runs,
        points=tuple(
            LadderPoint(names[i], sizes[i], seeds[i], losses[i], float(means[i]), float(weights[i]))
            for i in order
        ),
    )


# --- Effective-parameter multiplier (plan §8.4) --------------------------------
#
# The multiplier is exp(-rho) with rho = gap / b, a ratio of two noisy numbers.
# Its interval is the set of rho that the data do not reject (Fieller's
# construction): all rho with
#
#     (gap - rho * b)^2 <= t^2 * Var(gap - rho * b).
#
# For a given rho, gap - rho * b is linear in the run losses, so its variance
# is a sum of independent components, each a quadratic in rho with its own
# degrees of freedom. The t quantile uses their Satterthwaite combination. At
# rho = 0 this is exactly the gap's own test, so the interval excludes a
# multiplier of 1 exactly when the gap's interval excludes 0. When the slope
# cannot be told from zero the set is unbounded, and no interval is reported.

_MAX_ITERATIONS = 200


@dataclass(frozen=True)
class VarianceComponent:
    """c0 - 2 rho c1 + rho^2 c2, estimated with `df` degrees of freedom."""

    c0: float
    c1: float
    c2: float
    df: float

    def at(self, rho: float) -> float:
        return max(self.c0 - 2.0 * rho * self.c1 + rho * rho * self.c2, 0.0)


def _endpoint(gap: float, b: float, comps: tuple[VarianceComponent, ...], sign: int, confidence: float):
    """One end of the set, or None if the set is unbounded on that side."""
    c0 = sum(c.c0 for c in comps)
    c1 = sum(c.c1 for c in comps)
    c2 = sum(c.c2 for c in comps)
    rho = gap / b
    for _ in range(_MAX_ITERATIONS):
        parts = [(c.at(rho), c.df) for c in comps]
        total = sum(v for v, _ in parts)
        if total == 0.0:
            return gap / b  # no uncertainty at all: the set is a point
        df = total**2 / sum(v**2 / d for v, d in parts if v > 0.0)
        t2 = float(stats.t.ppf(0.5 + confidence / 2.0, df)) ** 2
        qa = b * b - t2 * c2
        qb = gap * b - t2 * c1
        qc = gap * gap - t2 * c0
        disc = qb * qb - qa * qc
        if qa <= 0.0 or disc < 0.0:
            return None
        new = (qb + sign * math.sqrt(disc)) / qa
        if abs(new - rho) <= 4.0 * math.ulp(max(abs(new), abs(rho), 1.0)):
            return new
        rho = new
    return rho


@dataclass(frozen=True)
class Multiplier:
    multiplier: float  # exp(-gap / b); below 1 means "like MHA with fewer parameters"
    ci_low: float  # 0.0 when not bounded
    ci_high: float  # inf when not bounded
    confidence: float
    bounded: bool  # False: the slope is too uncertain for any interval
    log_ratio: float  # gap / b, so multiplier = exp(-log_ratio)
    log_ratio_low: float | None
    log_ratio_high: float | None
    gap: float
    slope: float
    slope_se: float
    components: tuple[VarianceComponent, ...]


def _multiplier(gap: float, fit: LadderFit, comps, confidence: float) -> Multiplier:
    if not math.isfinite(gap):
        raise LadderError(f"the gap must be finite, got {gap!r}")
    if not fit.b > 0.0:
        raise LadderError(
            f"the ladder slope is {fit.b:.6g}: loss does not fall with parameters, "
            "so a gap has no parameter equivalent"
        )
    comps = tuple(comps)
    low = _endpoint(gap, fit.b, comps, -1, confidence)
    high = _endpoint(gap, fit.b, comps, +1, confidence)
    bounded = low is not None and high is not None
    ratio = gap / fit.b
    return Multiplier(
        multiplier=math.exp(-ratio),
        ci_low=math.exp(-high) if bounded else 0.0,
        ci_high=math.exp(-low) if bounded else math.inf,
        confidence=confidence,
        bounded=bounded,
        log_ratio=ratio,
        log_ratio_low=low if bounded else None,
        log_ratio_high=high if bounded else None,
        gap=float(gap),
        slope=fit.b,
        slope_se=fit.se_b,
        components=comps,
    )


def effective_multiplier(
    gap: float,
    gap_se: float,
    gap_df: float,
    fit: LadderFit,
    *,
    confidence: float = pt.CONFIDENCE,
) -> Multiplier:
    """exp(-gap / b) for a gap measured on runs that are NOT in the ladder.

    The interval carries the gap's uncertainty and the slope's. For a gap
    against runs that are also a ladder point, use `comparison_multiplier`,
    which accounts for the noise the two share.
    """
    if not (math.isfinite(gap_se) and gap_se >= 0.0):
        raise LadderError(f"gap_se must be finite and non-negative, got {gap_se!r}")
    if not gap_df > 0:
        raise LadderError(f"gap_df must be positive, got {gap_df!r}")
    comps = (
        VarianceComponent(gap_se**2, 0.0, 0.0, float(gap_df)),
        VarianceComponent(0.0, 0.0, fit.se_b**2, float(fit.df)),
    )
    return _multiplier(float(gap), fit, comps, confidence)


def comparison_multiplier(comparison: pt.Comparison, fit: LadderFit) -> Multiplier:
    """The multiplier for a gap that `primary_test.py` computed.

    The comparison's reference runs must be runs of the ladder point with the
    reference's name: the 8-layer MHA runs are both. Their noise moves the gap
    and the slope together, so they enter with the comparison's own sample
    variances, exactly as in its test; every other ladder run enters with the
    fit's residual variance. With w the weight of one reference run in the
    slope and n runs per group:

        paired:  n * var_k(d_k / n - rho * w * r_k)                   df n - 1
        Welch:   var(v) / n_v                                         df n_v - 1
                 n_r * (1 / n_r + rho * w)^2 * var(r)                 df n_r - 1
        both:    rho^2 * residual variance * (sum of the other runs' squared weights)
    """
    if any(p.name == comparison.name for p in fit.points):
        raise LadderError(
            f"{comparison.name!r} is a ladder point: its runs are part of the fit, "
            "so its gap has no multiplier"
        )
    ref_point = fit.point(comparison.reference)
    in_ladder = dict(zip(ref_point.seeds, ref_point.losses))
    for seed, loss in zip(comparison.reference_seeds, comparison.reference_losses):
        if seed not in in_ladder or in_ladder[seed] != loss:
            raise LadderError(
                f"reference run {comparison.reference} seed {seed} of the comparison is not "
                f"a run of the ladder point {ref_point.name!r}"
            )
    v = np.array(comparison.variant_losses, dtype=np.float64)
    r = np.array(comparison.reference_losses, dtype=np.float64)
    w = ref_point.slope_weight
    n_r = len(r)
    all_weights_sq = sum(len(p.seeds) * p.slope_weight**2 for p in fit.points)
    other = VarianceComponent(
        0.0, 0.0, fit.residual_sd**2 * (all_weights_sq - n_r * w * w), float(fit.df)
    )
    var_r = float(np.var(r, ddof=1))
    if comparison.test == pt.PAIRED:
        d = v - r
        n = len(d)
        gap = float(d.mean())
        shared = VarianceComponent(
            float(np.var(d, ddof=1)) / n,
            w * float(np.cov(d, r, ddof=1)[0, 1]),
            n * w * w * var_r,
            n - 1.0,
        )
        comps = (shared, other)
    elif comparison.test == pt.WELCH:
        gap = float(v.mean() - r.mean())
        comps = (
            VarianceComponent(float(np.var(v, ddof=1)) / len(v), 0.0, 0.0, len(v) - 1.0),
            VarianceComponent(var_r / n_r, -w * var_r, n_r * w * w * var_r, n_r - 1.0),
            other,
        )
    else:
        raise pt.PreregError(f"unknown test {comparison.test!r}; expected one of {pt.TESTS}")
    return _multiplier(gap, fit, comps, comparison.confidence)


# --- Detectable effect in nats and the seed-count rule (plan §1.3, §1.4) -------


def detectable_multiplier(n_seeds: int, test: str, prereg: Mapping) -> float:
    """The detectable gap for the primaries in units of sigma_init.

    Welch's test: the exact power (decision 79). Paired test: the multiplier in
    sigma_d times the worst case sigma_d = sqrt(2) * sigma_init (decision 80).
    """
    pt.registered_seeds(n_seeds, prereg)  # 5 or 8 only
    in_test_units = pt.detectable_gap_primary(
        n_seeds, test, prereg, welch_method=WELCH_POWER_METHOD
    )
    if test == pt.PAIRED:
        return in_test_units * PAIRED_WORST_CASE_SIGMA_D
    return in_test_units


def sigma_init_at_test_budget(mha_losses: Mapping[int, float], prereg: Mapping) -> float:
    """SD of the MHA base seeds' final losses at the primary budget (300M).

    Exactly the registered base seeds: no subset and no extra runs.
    """
    seeds = list(prereg["seed_count"]["base_seeds"])
    return float(pt.sample_sd(pt.ordered_losses(mha_losses, seeds, "sigma_init at the primary budget")))


@dataclass(frozen=True)
class SeedCountDecision:
    n_seeds: int  # 5 or 8
    seeds: tuple[int, ...]  # the primary seed set that follows
    extended: bool
    leaves_arm_a: tuple[str, ...]  # exploratory variants dropped when extended
    test: str
    sigma_init: float  # MHA base seeds at the primary budget, nats
    detectable_in_sigma_init: float  # 5-seed multiplier
    detectable_gap: float  # nats
    slope: float  # ladder b at the primary budget
    param_multiplier: float | None  # m = exp(-gap / b); None when the slope is not positive
    undetectable_param_fraction: float | None  # 1 - m; None when the slope is not positive
    threshold: float
    # What the design that was picked can detect, for the pre-registered bound.
    bound_in_sigma_init: float
    bound_gap: float
    # Decision 223: a slope that is not positive gives 8 seeds and no parameter equivalent.
    slope_not_positive: bool = False


def seed_count_rule(
    mha_losses: Mapping[int, float],
    fit: LadderFit,
    *,
    test: str,
    prereg: Mapping,
) -> SeedCountDecision:
    """Plan §1.4, applied once, from MHA runs only.

    `mha_losses`: final validation loss of the MHA base seeds at the primary
    budget. `fit`: the ladder fit at the same budget. `test`: the test chosen
    by `primary_test.select_test`.

    1. The 5-seed detectable gap in nats: multiplier * sigma_init.
    2. m = exp(-gap / b).
    3. If 1 - m exceeds the registered fraction, run 8 seeds; otherwise 5.
    """
    if test not in pt.TESTS:
        raise pt.PreregError(f"unknown test {test!r}; expected one of {pt.TESTS}")
    sc = prereg["seed_count"]
    base = list(sc["base_seeds"])
    extended_seeds = base + list(sc["added_seeds"])
    if not math.isfinite(fit.b):
        raise LadderError(f"the ladder slope is {fit.b!r}: the rule needs a finite slope")
    # Decision 223: a slope that is not positive cannot convert a gap into parameters, and
    # the rule gives 8 seeds (plan §1.4 gives 8 as the slope falls toward zero).
    slope_not_positive = fit.b <= 0.0
    sigma_init = sigma_init_at_test_budget(mha_losses, prereg)
    in_sigma = detectable_multiplier(len(base), test, prereg)
    gap = in_sigma * sigma_init
    threshold = sc["max_undetectable_param_fraction"]
    if slope_not_positive:
        m = None
        fraction = None
        extended = True
    else:
        m = math.exp(-gap / fit.b)
        fraction = 1.0 - m
        extended = fraction > threshold
    n_seeds = len(extended_seeds) if extended else len(base)
    bound_in_sigma = detectable_multiplier(n_seeds, test, prereg)
    return SeedCountDecision(
        n_seeds=n_seeds,
        seeds=tuple(extended_seeds if extended else base),
        extended=extended,
        leaves_arm_a=tuple(sc["leave_arm_a_if_extended"]) if extended else (),
        test=test,
        sigma_init=sigma_init,
        detectable_in_sigma_init=in_sigma,
        detectable_gap=gap,
        slope=fit.b,
        param_multiplier=m,
        undetectable_param_fraction=fraction,
        threshold=threshold,
        bound_in_sigma_init=bound_in_sigma,
        bound_gap=bound_in_sigma * sigma_init,
        slope_not_positive=slope_not_positive,
    )
