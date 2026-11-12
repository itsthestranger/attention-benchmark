"""The pre-registered decision rule of plan §1.2. The only code allowed to emit a verdict.

Everything here is fixed by `configs/prereg.yaml` and the plan:

- test selection from MHA runs only (§1.2): paired if s_rep < threshold * sigma_init,
  otherwise Welch;
- the three primary comparisons against MHA with Holm correction at family-wise
  alpha (§1.2);
- the null control and the positive control as single tests (§8.1, §8.4);
- exploratory comparisons, reported with an interval and no verdict (§1.2);
- the detectable-effect multipliers of §1.3.

The numerical core (`paired_t`, `welch_t`, `holm`, `choose_paired`) broadcasts
over leading axes, so `tests/test_primary_stats.py` runs thousands of simulated
experiments through exactly the code that judges a real one.

No function here takes a subset of seeds, another correction or another alpha
from its caller: those come from the pre-registration, and a seed set that
differs from the registered one is an error.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence

import numpy as np
from scipy import optimize, special, stats

PAIRED = "paired"
WELCH = "welch"
TESTS = (PAIRED, WELCH)

# The three verdicts of plan §10.1. Other code imports these; it never spells
# them out (a repository test enforces that).
VERDICT_DETECTED = "detected"
VERDICT_NOT_DETECTED = "not detected"
VERDICT_EXPLORATORY = "exploratory"

CONFIDENCE = 0.95


class PreregError(ValueError):
    """The data offered does not match what was pre-registered."""


# --- Numerical core (broadcasts over leading axes) ---------------------------


class TResult(NamedTuple):
    mean_gap: np.ndarray  # variant minus reference; positive means the variant is worse
    se: np.ndarray
    df: np.ndarray
    t: np.ndarray
    p: np.ndarray  # two-sided
    ci_low: np.ndarray
    ci_high: np.ndarray


def sample_sd(x, axis: int = -1) -> np.ndarray:
    """Sample standard deviation (n - 1 in the denominator)."""
    return np.std(np.asarray(x, dtype=np.float64), axis=axis, ddof=1)


def s_rep(deltas) -> np.ndarray:
    """sqrt(mean(delta^2) / 2) over the reproducibility pairs (plan §1.2)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    return np.sqrt(np.mean(deltas**2, axis=-1) / 2.0)


def choose_paired(sigma_init, s_rep_value, threshold: float) -> np.ndarray:
    """True where the paired test is selected: s_rep < threshold * sigma_init."""
    return np.asarray(s_rep_value) < threshold * np.asarray(sigma_init)


def _finish(mean_gap, se, df, confidence: float) -> TResult:
    mean_gap, se, df = np.broadcast_arrays(
        np.asarray(mean_gap, dtype=np.float64),
        np.asarray(se, dtype=np.float64),
        np.asarray(df, dtype=np.float64),
    )
    zero = se == 0.0
    safe_se = np.where(zero, 1.0, se)
    safe_df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    # With no spread at all the statistic is 0 for a zero gap and infinite otherwise.
    degenerate = np.where(mean_gap == 0.0, 0.0, np.copysign(np.inf, mean_gap))
    t = np.where(zero, degenerate, mean_gap / safe_se)
    p = 2.0 * stats.t.sf(np.abs(t), safe_df)
    half = stats.t.ppf(0.5 + confidence / 2.0, safe_df) * se
    return TResult(mean_gap, se, safe_df, t, p, mean_gap - half, mean_gap + half)


def paired_t(variant, reference, confidence: float = CONFIDENCE) -> TResult:
    """Two-sided paired t-test on d_k = variant_k - reference_k along the last axis."""
    variant = np.asarray(variant, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if variant.shape[-1] != reference.shape[-1]:
        raise PreregError("paired test needs the same number of runs on both sides")
    d = variant - reference
    n = d.shape[-1]
    if n < 2:
        raise PreregError("paired test needs at least two pairs")
    return _finish(d.mean(axis=-1), sample_sd(d) / math.sqrt(n), n - 1, confidence)


def welch_t(variant, reference, confidence: float = CONFIDENCE) -> TResult:
    """Two-sided unequal-variance t-test along the last axis (Welch-Satterthwaite df)."""
    variant = np.asarray(variant, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    nv, nr = variant.shape[-1], reference.shape[-1]
    if nv < 2 or nr < 2:
        raise PreregError("Welch's test needs at least two runs on each side")
    qv = np.var(variant, axis=-1, ddof=1) / nv
    qr = np.var(reference, axis=-1, ddof=1) / nr
    se = np.sqrt(qv + qr)
    denom = qv**2 / (nv - 1) + qr**2 / (nr - 1)
    df = np.where(denom > 0.0, (qv + qr) ** 2 / np.where(denom > 0.0, denom, 1.0), nv + nr - 2.0)
    return _finish(variant.mean(axis=-1) - reference.mean(axis=-1), se, df, confidence)


def run_test(test: str, variant, reference, confidence: float = CONFIDENCE) -> TResult:
    if test == PAIRED:
        return paired_t(variant, reference, confidence)
    if test == WELCH:
        return welch_t(variant, reference, confidence)
    raise PreregError(f"unknown test {test!r}; expected one of {TESTS}")


def holm(p, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Holm's step-down procedure over the last axis.

    Returns (reject, adjusted_p). A hypothesis is rejected when its adjusted
    p-value is at most alpha, which is the step-down rule: the i-th smallest of
    m p-values is compared with alpha / (m - i + 1), stopping at the first
    failure.
    """
    p = np.asarray(p, dtype=np.float64)
    m = p.shape[-1]
    order = np.argsort(p, axis=-1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=-1)
    scaled = ranked * (m - np.arange(m))
    adjusted_ranked = np.minimum(np.maximum.accumulate(scaled, axis=-1), 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    np.put_along_axis(adjusted, order, adjusted_ranked, axis=-1)
    return adjusted <= alpha, adjusted


# --- Detectable effect (plan §1.3) -------------------------------------------

# Two ways to compute the power of Welch's test with n runs per group and equal
# true variances. They differ at small n, and which one feeds the seed-count
# rule and the reported bound is Phil's decision (see STATUS.md):
#
# - WELCH_POOLED_DF: a noncentral t with 2n - 2 degrees of freedom. This is the
#   power of the pooled-variance two-sample test, and it is what reproduces the
#   plan's §1.3 table (2.5 at 5 seeds, 1.8 at 8).
# - WELCH_EXACT: the power of the test `welch_t` actually runs, with its
#   estimated degrees of freedom, by quadrature over the two sample variances
#   (2.63 at 5 seeds, 1.83 at 8).
WELCH_POOLED_DF = "pooled_df"
WELCH_EXACT = "exact"
WELCH_POWER_METHODS = (WELCH_POOLED_DF, WELCH_EXACT)

_QUADRATURE_NODES = 96


def _nct_power(ncp: float, df: float, alpha: float) -> float:
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    # The lower tail is written through the mirrored distribution: the direct
    # noncentral cdf returns NaN for large noncentrality in some scipy versions.
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.sf(crit, df, -ncp))


@functools.lru_cache(maxsize=None)
def _welch_grid(n: int, alpha: float):
    """Quadrature nodes for the two sample variances, with the critical
    half-width of Welch's test at each node (in units of sigma)."""
    nu = n - 1
    x, w = special.roots_genlaguerre(_QUADRATURE_NODES, nu / 2.0 - 1.0)
    w = w / special.gamma(nu / 2.0)
    var = 2.0 * x / nu  # sample variance at each node, sigma = 1
    v1, v2 = var[:, None], var[None, :]
    df = (v1 + v2) ** 2 / ((v1**2 + v2**2) / nu)
    half_width = stats.t.ppf(1.0 - alpha / 2.0, df) * np.sqrt((v1 + v2) / n)
    return half_width, w[:, None] * w[None, :]


def _welch_exact_power(effect: float, n: int, alpha: float) -> float:
    half_width, weight = _welch_grid(n, alpha)
    sd_diff = math.sqrt(2.0 / n)  # SD of the difference of the two group means
    reject = stats.norm.sf((half_width - effect) / sd_diff) + stats.norm.cdf(
        (-half_width - effect) / sd_diff
    )
    return float(np.sum(weight * reject))


def power(effect: float, n: int, test: str, alpha: float, *, welch_method: str) -> float:
    """Power of the two-sided test at level alpha for a true mean gap `effect`.

    `effect` is in units of sigma_d (SD of the paired differences) for the
    paired test, and of sigma_init (per-group SD, equal in both groups) for
    Welch's test, with n runs per group. `welch_method` is ignored for the
    paired test, whose power is an exact noncentral t.
    """
    if test == PAIRED:
        return _nct_power(effect * math.sqrt(n), n - 1, alpha)
    if test != WELCH:
        raise PreregError(f"unknown test {test!r}; expected one of {TESTS}")
    if welch_method == WELCH_POOLED_DF:
        return _nct_power(effect * math.sqrt(n / 2.0), 2 * n - 2, alpha)
    if welch_method == WELCH_EXACT:
        return _welch_exact_power(effect, n, alpha)
    raise PreregError(f"unknown welch_method {welch_method!r}; expected one of {WELCH_POWER_METHODS}")


def detectable_gap(
    n: int, test: str, alpha: float, target_power: float, *, welch_method: str
) -> float:
    """Smallest mean gap detected with `target_power`, in the units of `power`."""
    return float(
        optimize.brentq(
            lambda e: power(e, n, test, alpha, welch_method=welch_method) - target_power,
            1e-6,
            50.0,
            xtol=1e-12,
        )
    )


def primary_alpha_strictest(prereg: Mapping) -> float:
    """Holm's strictest step: family alpha over the number of primary comparisons."""
    primary = prereg["primary"]
    return primary["family_alpha"] / len(primary["comparisons"])


def detectable_gap_primary(n: int, test: str, prereg: Mapping, *, welch_method: str) -> float:
    """The §1.3 multiplier for the primaries: 80% power at Holm's strictest step."""
    return detectable_gap(
        n,
        test,
        primary_alpha_strictest(prereg),
        prereg["seed_count"]["power"],
        welch_method=welch_method,
    )


# --- Test selection (plan §1.2) ----------------------------------------------


@dataclass(frozen=True)
class TestSelection:
    test: str
    sigma_init: float
    s_rep: float
    ratio: float  # s_rep / sigma_init
    threshold: float


def ordered_losses(losses: Mapping[int, float], seeds: Sequence[int], what: str) -> np.ndarray:
    """Losses in the order of the registered seeds; any other seed set is an error."""
    if sorted(losses) != sorted(seeds):
        raise PreregError(
            f"{what}: runs for seeds {sorted(losses)} offered, but seeds {sorted(seeds)} "
            "are pre-registered; no subset and no extra seeds"
        )
    values = np.array([float(losses[s]) for s in seeds], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise PreregError(f"{what}: a loss is not finite")
    return values


def select_test(
    mha_losses: Mapping[int, float],
    repro_pairs: Mapping[int, tuple[float, float]],
    prereg: Mapping,
) -> TestSelection:
    """Choose the test from MHA runs only.

    `mha_losses`: final validation loss per init seed at the registered budget.
    `repro_pairs`: per init seed, the losses of the two identical runs.
    """
    sel = prereg["test_selection"]
    sigma_init = float(sample_sd(ordered_losses(mha_losses, sel["sigma_init"]["init_seeds"], "sigma_init")))
    pair_seeds = sel["reproducibility_pairs"]["init_seeds"]
    if sorted(repro_pairs) != sorted(pair_seeds):
        raise PreregError(
            f"reproducibility pairs for seeds {sorted(repro_pairs)} offered, "
            f"but seeds {sorted(pair_seeds)} are pre-registered"
        )
    deltas = np.array([float(repro_pairs[s][0]) - float(repro_pairs[s][1]) for s in pair_seeds])
    if not np.all(np.isfinite(deltas)):
        raise PreregError("reproducibility pairs: a loss is not finite")
    rep = float(s_rep(deltas))
    threshold = sel["paired_if_s_rep_ratio_below"]
    if sel["otherwise"] != WELCH:
        raise PreregError(f"unsupported fallback test {sel['otherwise']!r}")
    paired = bool(choose_paired(sigma_init, rep, threshold))
    ratio = rep / sigma_init if sigma_init > 0 else math.inf
    return TestSelection(PAIRED if paired else WELCH, sigma_init, rep, ratio, threshold)


# --- Comparisons and verdicts ------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    name: str
    reference: str
    kind: str  # "primary" | "null_control" | "positive_control" | "exploratory"
    test: str
    seeds: tuple[int, ...]  # variant seeds, in order
    reference_seeds: tuple[int, ...]
    variant_losses: tuple[float, ...]
    reference_losses: tuple[float, ...]
    differences: tuple[float, ...] | None  # d_k under the paired test
    mean_gap: float  # variant minus reference, nats
    ci_low: float
    ci_high: float
    confidence: float
    df: float
    t: float | None
    p: float | None  # unadjusted, two-sided; None for exploratory
    p_adjusted: float | None  # Holm-adjusted within the family; equals p for a single test
    alpha: float | None  # family-wise alpha the verdict was judged at
    verdict: str
    ruled_out_above: float | None  # upper end of the interval when not detected


def _comparison(name, reference, kind, test, seeds, ref_seeds, v, r, res, p_adj, alpha, verdict):
    return Comparison(
        name=name,
        reference=reference,
        kind=kind,
        test=test,
        seeds=tuple(seeds),
        reference_seeds=tuple(ref_seeds),
        variant_losses=tuple(float(x) for x in v),
        reference_losses=tuple(float(x) for x in r),
        differences=tuple(float(x) for x in (v - r)) if test == PAIRED else None,
        mean_gap=float(res.mean_gap),
        ci_low=float(res.ci_low),
        ci_high=float(res.ci_high),
        confidence=CONFIDENCE,
        df=float(res.df),
        t=None if verdict == VERDICT_EXPLORATORY else float(res.t),
        p=None if verdict == VERDICT_EXPLORATORY else float(res.p),
        p_adjusted=None if p_adj is None else float(p_adj),
        alpha=alpha,
        verdict=verdict,
        ruled_out_above=float(res.ci_high) if verdict == VERDICT_NOT_DETECTED else None,
    )


def registered_seeds(n_seeds: int, prereg: Mapping) -> list[int]:
    """The primary seed set for the count the seed-count rule picked (plan §1.4)."""
    base = list(prereg["seed_count"]["base_seeds"])
    extended = base + list(prereg["seed_count"]["added_seeds"])
    if n_seeds == len(base):
        return base
    if n_seeds == len(extended):
        return extended
    raise PreregError(f"n_seeds must be {len(base)} or {len(extended)}, got {n_seeds!r}")


def primary_tests(
    losses: Mapping[str, Mapping[int, float]],
    *,
    test: str,
    n_seeds: int,
    prereg: Mapping,
) -> list[Comparison]:
    """The pre-registered primary comparisons, Holm-corrected, in registered order.

    `losses[variant][seed]` is the final validation loss at the registered
    budget. `losses` must hold the reference and every primary variant, each
    with exactly the registered seeds.
    """
    if test not in TESTS:
        raise PreregError(f"unknown test {test!r}; expected one of {TESTS}")
    primary = prereg["primary"]
    if primary["correction"] != "holm" or primary["two_sided"] is not True:
        raise PreregError("only the two-sided test with Holm correction is implemented")
    seeds = registered_seeds(n_seeds, prereg)
    reference = primary["reference"]
    names = list(primary["comparisons"])
    missing = [v for v in [reference, *names] if v not in losses]
    if missing:
        raise PreregError(f"missing losses for {missing}; every primary comparison is required")
    ref = ordered_losses(losses[reference], seeds, reference)
    rows = [ordered_losses(losses[name], seeds, name) for name in names]
    results = [run_test(test, row, ref) for row in rows]
    alpha = primary["family_alpha"]
    reject, adjusted = holm(np.array([float(r.p) for r in results]), alpha)
    return [
        _comparison(
            name, reference, "primary", test, seeds, seeds, row, ref, res, adj, alpha,
            VERDICT_DETECTED if rej else VERDICT_NOT_DETECTED,
        )
        for name, row, res, rej, adj in zip(names, rows, results, reject, adjusted)
    ]


def _single(name, reference, kind, test, seeds, ref_seeds, v, r, alpha) -> Comparison:
    res = run_test(test, v, r)
    verdict = VERDICT_DETECTED if float(res.p) <= alpha else VERDICT_NOT_DETECTED
    return _comparison(name, reference, kind, test, seeds, ref_seeds, v, r, res, res.p, alpha, verdict)


def null_control(
    null_losses: Mapping[int, float],
    mha_losses: Mapping[int, float],
    *,
    test: str,
    prereg: Mapping,
) -> Comparison:
    """MHA under another name against MHA, judged like a variant (plan §8.1).

    Its seeds are unrelated to MHA's, so under the paired test the runs are
    paired in seed order: the i-th null seed with the i-th MHA seed.
    """
    nc = prereg["null_control"]
    ref_seeds = list(prereg["test_selection"]["sigma_init"]["init_seeds"])
    seeds = list(nc["init_seeds"])
    if len(seeds) != len(ref_seeds):
        raise PreregError("null control and reference seed counts differ")
    v = ordered_losses(null_losses, seeds, nc["variant"])
    r = ordered_losses(mha_losses, ref_seeds, nc["reference"])
    alpha = prereg["primary"]["family_alpha"]
    return _single(nc["variant"], nc["reference"], "null_control", test, seeds, ref_seeds, v, r, alpha)


def positive_control(
    small_losses: Mapping[int, float],
    mha_losses: Mapping[int, float],
    *,
    test: str,
    prereg: Mapping,
) -> Comparison:
    """L6 against L8 as a single test (plan §8.4)."""
    pc = prereg["positive_control"]
    seeds = list(pc["init_seeds"])
    v = ordered_losses(small_losses, seeds, pc["small"])
    r = ordered_losses(mha_losses, seeds, pc["reference"])
    return _single(pc["small"], pc["reference"], "positive_control", test, seeds, seeds, v, r, pc["alpha"])


def exploratory(
    name: str,
    variant_losses: Mapping[int, float],
    mha_losses: Mapping[int, float],
    *,
    test: str,
    prereg: Mapping,
) -> Comparison:
    """An exploratory comparison: mean gap and interval, no p-value, no detection claim.

    Under the paired test the variant's seeds are paired with the same MHA
    seeds; under Welch's test every offered MHA run is used.
    """
    ex = prereg["exploratory"]
    if name not in ex["variants"]:
        raise PreregError(f"{name!r} is not a pre-registered exploratory variant")
    reference = prereg["primary"]["reference"]
    seeds = list(ex["init_seeds"])
    v = ordered_losses(variant_losses, seeds, name)
    if test == PAIRED:
        if not set(seeds) <= set(mha_losses):
            raise PreregError(f"{reference}: runs for seeds {seeds} are needed to pair with {name}")
        ref_seeds = seeds
    elif test == WELCH:
        ref_seeds = sorted(mha_losses)
    else:
        raise PreregError(f"unknown test {test!r}; expected one of {TESTS}")
    r = ordered_losses({s: mha_losses[s] for s in ref_seeds}, ref_seeds, reference)
    res = run_test(test, v, r)
    return _comparison(
        name, reference, "exploratory", test, seeds, ref_seeds, v, r, res, None, None,
        VERDICT_EXPLORATORY,
    )
