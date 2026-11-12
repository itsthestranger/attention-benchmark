"""T-25: the ladder fit, the effective-parameter multiplier and the seed-count rule
(plan §8.4, §1.3, §1.4). SYNTHETIC losses only.

Phil's two answers (docs/decisions.md, rows 79 and 80) are pinned here: the
seed-count rule uses the exact power of Welch's test, and under the paired test
the worst case sigma_d = sqrt(2) * sigma_init.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from src import config
from tests.fixtures.synthetic_losses import LADDER_SEEDS, LADDER_SIZES, synthetic_ladder
from tests.tolerances import DETECTABLE_EFFECT_TABLE_ABS, FLOAT64_IDENTITY, scale_relative_error

PREREG = config.load_prereg()
SEED = 20261112


def close(a, b) -> bool:
    return scale_relative_error(a, b) <= FLOAT64_IDENTITY


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng([SEED, *key])


def noisy_fit(key=0, *, a=20.0, b=0.4, sigma=0.01, seeds=None):
    return lf.fit_ladder(synthetic_ladder(rng_for(key), a=a, b=b, sigma=sigma, seeds=seeds))


def variant_runs(fit, gap, key, *, sigma=0.01, n=5, common=0.0):
    """SYNTHETIC variant losses: the reference's loss per seed times `common`, plus noise and a gap."""
    ref = fit.point("mha")
    rng = rng_for(key, 99)
    by_seed = dict(zip(ref.seeds, ref.losses))
    return {
        s: ref.mean + common * (by_seed[s] - ref.mean) + gap + sigma * float(rng.standard_normal())
        for s in ref.seeds[:n]
    }


def comparison(fit, variant, test, *, name="gqa2"):
    """A Comparison built by primary_test itself (exploratory: seeds 1-3, no verdict needed)."""
    ref = fit.point("mha")
    mha = dict(zip(ref.seeds, ref.losses))
    return pt.exploratory(name, {s: variant[s] for s in (1, 2, 3)}, mha, test=test, prereg=PREREG)


# --- The fit --------------------------------------------------------------------------------


def test_ladder_sizes_are_the_plans():
    assert LADDER_SIZES == {"mha_L6": 10_621_824, "mha": 14_162_304, "mha_L10": 17_702_784}
    assert {k: len(v) for k, v in LADDER_SEEDS.items()} == {"mha_L6": 5, "mha": 5, "mha_L10": 3}


@pytest.mark.parametrize("a, b", [(20.0, 0.4), (3.5, 0.05), (10.0, 1.7)])
def test_a_known_slope_is_recovered_exactly_without_noise(a, b):
    fit = lf.fit_ladder(synthetic_ladder(None, a=a, b=b))
    assert close(fit.b, b) and close(fit.a, a)
    assert fit.n_runs == 13 and fit.df == 11
    for name, size in LADDER_SIZES.items():
        assert close(fit.predict(size), a - b * math.log(size))
    assert [p.name for p in fit.points] == ["mha_L6", "mha", "mha_L10"]  # by size


def test_the_fit_is_ordinary_least_squares_over_every_run():
    points = synthetic_ladder(rng_for(1), a=20.0, b=0.4, sigma=0.02)
    fit = lf.fit_ladder(points)
    x = [math.log(size) for size, runs in points.values() for _ in runs]
    y = [loss for _, runs in points.values() for loss in runs.values()]
    ref = stats.linregress(x, y)
    assert close(fit.b, -ref.slope)
    assert close(fit.a, ref.intercept)
    assert close(fit.se_b, ref.stderr)
    assert fit.df == len(y) - 2
    low, high = fit.slope_interval()
    half = stats.t.ppf(0.975, fit.df) * ref.stderr
    assert close(low, -ref.slope - half) and close(high, -ref.slope + half)


def test_a_point_with_more_seeds_weighs_more():
    seeds = dict(LADDER_SEEDS, mha=(1, 2, 3, 4, 5, 6, 7, 8))
    points = synthetic_ladder(rng_for(2), a=20.0, b=0.4, sigma=0.02, seeds=seeds)
    fit = lf.fit_ladder(points)
    assert fit.n_runs == 16 and fit.df == 14
    x = [math.log(size) for size, runs in points.values() for _ in runs]
    y = [loss for _, runs in points.values() for loss in runs.values()]
    assert close(fit.b, -stats.linregress(x, y).slope)


def test_the_slope_is_the_weighted_sum_of_run_losses():
    """The weights are what ties a gap against MHA to the slope."""
    fit = noisy_fit(3)
    # Written with centred losses: the weights sum to zero, so the centre drops out,
    # and the sum is then not a small difference of large numbers.
    centre = sum(sum(p.losses) for p in fit.points) / fit.n_runs
    assert close(sum(p.slope_weight * sum(x - centre for x in p.losses) for p in fit.points), fit.b)
    assert sum(len(p.seeds) * p.slope_weight for p in fit.points) == pytest.approx(0.0, abs=FLOAT64_IDENTITY)
    # Raising one run of the smallest model raises the slope; of the largest, lowers it.
    assert fit.points[0].slope_weight > 0 > fit.points[-1].slope_weight


def test_two_sizes_are_enough_and_give_a_wider_slope_interval():
    """Plan §12.2, cut 4: the ladder down to L6 and L8."""
    three = synthetic_ladder(rng_for(4), a=20.0, b=0.4, sigma=0.02)
    two = {k: v for k, v in three.items() if k != "mha_L10"}
    fit3, fit2 = lf.fit_ladder(three), lf.fit_ladder(two)
    assert fit2.df == 8
    assert fit2.se_b / fit2.residual_sd > fit3.se_b / fit3.residual_sd


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: [p.pop("mha_L6"), p.pop("mha_L10")],  # one size
        lambda p: p.update(mha_L10=(p["mha"][0], p["mha_L10"][1])),  # two points of one size
        lambda p: p.update(mha=(p["mha"][0], {})),  # a point without runs
        lambda p: p.update(mha=(0, p["mha"][1])),  # impossible size
        lambda p: p.update(mha=(14_162_304.0, p["mha"][1])),  # size not an int
        lambda p: p["mha"][1].update({1: float("nan")}),  # non-finite loss
    ],
)
def test_malformed_ladders_raise(mutate):
    points = synthetic_ladder(rng_for(5), a=20.0, b=0.4, sigma=0.02)
    mutate(points)
    with pytest.raises(lf.LadderError):
        lf.fit_ladder(points)


def test_too_few_runs_to_estimate_the_uncertainty_raise():
    with pytest.raises(lf.LadderError):
        lf.fit_ladder({"mha_L6": (10_621_824, {1: 3.1}), "mha": (14_162_304, {1: 3.0})})
    with pytest.raises(lf.LadderError):
        noisy_fit(6).point("mha_L12")


# --- The multiplier: point value --------------------------------------------------------------


def test_multiplier_is_exp_of_minus_gap_over_slope():
    fit = lf.fit_ladder(synthetic_ladder(None, a=20.0, b=0.4))
    # A gap equal to what the ladder predicts for 0.94x the parameters reads as 0.94.
    gap = fit.predict(0.94 * 14_162_304) - fit.predict(14_162_304)
    m = lf.effective_multiplier(gap, 0.001, 4, fit)
    assert close(m.multiplier, 0.94)
    assert close(lf.effective_multiplier(0.0, 0.001, 4, fit).multiplier, 1.0)
    # L6 against L8 reads as the ratio of their sizes, the plan's 0.75x.
    step = fit.predict(10_621_824) - fit.predict(14_162_304)
    ratio = lf.effective_multiplier(step, 0.001, 4, fit).multiplier
    assert close(ratio, 10_621_824 / 14_162_304)
    assert round(ratio, 2) == 0.75
    # A variant that is better than MHA has a multiplier above 1.
    assert lf.effective_multiplier(-0.01, 0.001, 4, fit).multiplier > 1.0


# --- The multiplier: the interval carries both uncertainties -------------------------------------


def test_with_an_exact_slope_the_interval_is_the_gaps_own_interval():
    fit = lf.fit_ladder(synthetic_ladder(None, a=20.0, b=0.4))
    assert fit.se_b <= FLOAT64_IDENTITY
    gap, se, df = 0.02, 0.008, 4
    m = lf.effective_multiplier(gap, se, df, fit)
    half = stats.t.ppf(0.975, df) * se
    assert m.bounded
    assert close(m.ci_low, math.exp(-(gap + half) / fit.b))
    assert close(m.ci_high, math.exp(-(gap - half) / fit.b))
    assert m.ci_low < m.multiplier < m.ci_high


def test_with_an_exact_gap_the_interval_is_the_slopes_own_interval():
    fit = noisy_fit(7)
    gap = 0.03
    m = lf.effective_multiplier(gap, 0.0, 4, fit)
    b_low, b_high = fit.slope_interval()
    assert b_low > 0
    assert close(m.ci_low, math.exp(-gap / b_low))
    assert close(m.ci_high, math.exp(-gap / b_high))
    # With no gap at all the slope's uncertainty cannot matter.
    zero = lf.effective_multiplier(0.0, 0.0, 4, fit)
    assert zero.ci_low == zero.multiplier == zero.ci_high == 1.0


@pytest.mark.parametrize("gap", [0.03, -0.02, 0.1])
def test_the_interval_carries_both_uncertainties(gap):
    """Each end rho solves (gap - rho b)^2 = t(df)^2 (se_gap^2 + rho^2 se_b^2), with the
    Satterthwaite df of the two terms. Leaving either term out breaks the equation."""
    fit = noisy_fit(8)
    se, gap_df = 0.006, 4
    m = lf.effective_multiplier(gap, se, gap_df, fit)
    assert m.bounded and m.ci_low < m.multiplier < m.ci_high
    for rho in (m.log_ratio_low, m.log_ratio_high):
        from_gap, from_slope = se**2, rho**2 * fit.se_b**2
        both = from_gap + from_slope
        df = both**2 / (from_gap**2 / gap_df + from_slope**2 / fit.df)
        lhs = (gap - rho * fit.b) ** 2
        assert close(lhs, stats.t.ppf(0.975, df) ** 2 * both)
        assert not close(lhs, stats.t.ppf(0.975, gap_df) ** 2 * from_gap)
        assert not close(lhs, stats.t.ppf(0.975, fit.df) ** 2 * from_slope)


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("gap", [0.0, 0.004, 0.03, -0.02])
@pytest.mark.parametrize("common", [0.0, 1.0])
def test_interval_end_points_satisfy_their_definition(test, gap, common):
    """Each end rho solves (gap - rho b)^2 = t(df(rho))^2 Var(gap - rho b), with the
    variance rebuilt here from explicit run weights, independently of the module."""
    fit = noisy_fit(9)
    cmp_ = comparison(fit, variant_runs(fit, gap, 9, common=common), test)
    m = lf.comparison_multiplier(cmp_, fit)
    assert m.bounded
    v = np.array(cmp_.variant_losses)
    r = np.array(cmp_.reference_losses)
    w = fit.point("mha").slope_weight
    other_sq = sum(len(p.seeds) * p.slope_weight**2 for p in fit.points) - len(r) * w * w
    for rho in (m.log_ratio_low, m.log_ratio_high):
        if test == pt.PAIRED:
            z = (v - r) / len(r) - rho * w * r
            parts = [(len(r) * np.var(z, ddof=1), len(r) - 1)]
        else:
            parts = [
                (np.var(v, ddof=1) / len(v), len(v) - 1),
                (len(r) * (1 / len(r) + rho * w) ** 2 * np.var(r, ddof=1), len(r) - 1),
            ]
        parts.append((rho**2 * fit.residual_sd**2 * other_sq, fit.df))
        total = sum(p for p, _ in parts)
        df = total**2 / sum(p**2 / d for p, d in parts)
        assert close((m.gap - rho * fit.b) ** 2, stats.t.ppf(0.975, df) ** 2 * total)
    assert m.log_ratio_low < m.log_ratio < m.log_ratio_high
    assert close(m.ci_low, math.exp(-m.log_ratio_high)) and close(m.ci_high, math.exp(-m.log_ratio_low))


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("key", range(40))
def test_the_interval_excludes_one_exactly_when_the_gap_interval_excludes_zero(test, key):
    """At a multiplier of 1 the slope plays no part, so the multiplier can never
    suggest a difference that the gap's own interval does not."""
    fit = noisy_fit(key)
    gap = [0.0, 0.012, -0.012, 0.03][key % 4]
    cmp_ = comparison(fit, variant_runs(fit, gap, key, common=key % 2), test)
    m = lf.comparison_multiplier(cmp_, fit)
    assert m.bounded
    gap_excludes_zero = not (cmp_.ci_low <= 0.0 <= cmp_.ci_high)
    multiplier_excludes_one = not (m.ci_low <= 1.0 <= m.ci_high)
    assert gap_excludes_zero == multiplier_excludes_one
    assert close(m.gap, cmp_.mean_gap) if cmp_.mean_gap != 0 else m.gap == 0


def test_both_outcomes_of_the_previous_test_occur():
    seen = set()
    for test in pt.TESTS:
        for key in range(40):
            fit = noisy_fit(key)
            gap = [0.0, 0.012, -0.012, 0.03][key % 4]
            cmp_ = comparison(fit, variant_runs(fit, gap, key, common=key % 2), test)
            seen.add((test, cmp_.ci_low <= 0.0 <= cmp_.ci_high))
    assert seen == {(t, flag) for t in pt.TESTS for flag in (True, False)}


def test_shared_reference_noise_changes_the_interval():
    """The reference runs are in the ladder. Treating the gap as independent of the
    slope gives a different interval, so the shared noise is really accounted for."""
    fit = noisy_fit(10, sigma=0.02)
    cmp_ = comparison(fit, variant_runs(fit, 0.05, 10, sigma=0.02), pt.WELCH)
    shared = lf.comparison_multiplier(cmp_, fit)
    res = pt.welch_t(np.array(cmp_.variant_losses), np.array(cmp_.reference_losses))
    independent = lf.effective_multiplier(float(res.mean_gap), float(res.se), float(res.df), fit)
    assert shared.multiplier == independent.multiplier
    assert not close(shared.ci_low, independent.ci_low)
    assert not close(shared.ci_high, independent.ci_high)


def test_an_unresolved_slope_gives_no_interval():
    """Runs so noisy that the slope cannot be told from zero: any ratio is possible."""
    fit = noisy_fit(11, b=0.01, sigma=0.5)
    low, high = fit.slope_interval()
    assert low < 0 < high
    if fit.b > 0:
        m = lf.effective_multiplier(0.02, 0.01, 4, fit)
        assert not m.bounded
        assert m.ci_low == 0.0 and m.ci_high == math.inf
        assert m.log_ratio_low is None and m.log_ratio_high is None
    else:
        with pytest.raises(lf.LadderError):
            lf.effective_multiplier(0.02, 0.01, 4, fit)


def test_unresolved_slope_case_with_a_positive_estimate_exists():
    """The branch above that returns an unbounded interval is really exercised."""
    hits = 0
    for key in range(20):
        fit = noisy_fit(100 + key, b=0.01, sigma=0.5)
        if fit.b > 0:
            hits += not lf.effective_multiplier(0.02, 0.01, 4, fit).bounded
    assert hits > 0


def test_a_slope_of_the_wrong_sign_has_no_multiplier():
    fit = lf.fit_ladder(synthetic_ladder(None, a=3.0, b=-0.1))
    assert fit.b < 0
    with pytest.raises(lf.LadderError):
        lf.effective_multiplier(0.01, 0.001, 4, fit)
    with pytest.raises(lf.LadderError):
        lf.seed_count_rule({s: 3.0 + 0.01 * s for s in range(1, 6)}, fit, test=pt.WELCH, prereg=PREREG)


@pytest.mark.parametrize("bad", [dict(gap=float("nan")), dict(gap_se=-1.0), dict(gap_se=float("inf")), dict(gap_df=0)])
def test_multiplier_refuses_malformed_inputs(bad):
    fit = noisy_fit(12)
    args = dict(gap=0.01, gap_se=0.005, gap_df=4)
    args.update(bad)
    with pytest.raises(lf.LadderError):
        lf.effective_multiplier(args["gap"], args["gap_se"], args["gap_df"], fit)


def test_comparison_multiplier_needs_the_reference_runs_in_the_ladder():
    fit = noisy_fit(13)
    variant = variant_runs(fit, 0.02, 13)
    good = comparison(fit, variant, pt.PAIRED)
    lf.comparison_multiplier(good, fit)
    # The same comparison against a ladder built from other MHA runs.
    with pytest.raises(lf.LadderError):
        lf.comparison_multiplier(good, noisy_fit(14))
    # A ladder with no point for the reference.
    no_ref = lf.fit_ladder({k: v for k, v in synthetic_ladder(rng_for(13), a=20.0, b=0.4, sigma=0.01).items() if k != "mha"})
    with pytest.raises(lf.LadderError):
        lf.comparison_multiplier(good, no_ref)


def test_a_ladder_point_has_no_multiplier_of_its_own():
    """L6 against L8 is the positive control; its runs are part of the fit."""
    fit = noisy_fit(15)
    l6, l8 = fit.point("mha_L6"), fit.point("mha")
    control = pt.positive_control(
        dict(zip(l6.seeds, l6.losses)), dict(zip(l8.seeds, l8.losses)), test=pt.WELCH, prereg=PREREG
    )
    with pytest.raises(lf.LadderError):
        lf.comparison_multiplier(control, fit)


# --- Detectable effect and the seed-count rule (plan §1.3, §1.4) ----------------------------------


def test_welch_multiplier_is_the_exact_power_not_the_plans_table():
    """Phil, decision 79: 2.63 at 5 seeds and 1.83 at 8, not the table's 2.5 and 1.8."""
    assert lf.WELCH_POWER_METHOD == pt.WELCH_EXACT
    for n, exact_value, table_value in ((5, 2.63, 2.5), (8, 1.83, 1.8)):
        got = lf.detectable_multiplier(n, pt.WELCH, PREREG)
        assert got == pt.detectable_gap_primary(n, pt.WELCH, PREREG, welch_method=pt.WELCH_EXACT)
        assert got > pt.detectable_gap_primary(n, pt.WELCH, PREREG, welch_method=pt.WELCH_POOLED_DF)
        assert abs(got - exact_value) <= DETECTABLE_EFFECT_TABLE_ABS
        assert got > table_value


def test_paired_multiplier_is_the_worst_case_in_sigma_init():
    """Phil, decision 80: sqrt(2) * sigma_init for sigma_d, the plan table's 3.2 and 2.1."""
    assert lf.PAIRED_WORST_CASE_SIGMA_D == math.sqrt(2.0)
    for n, table_value in ((5, 3.2), (8, 2.1)):
        got = lf.detectable_multiplier(n, pt.PAIRED, PREREG)
        in_sigma_d = pt.detectable_gap_primary(n, pt.PAIRED, PREREG, welch_method=pt.WELCH_EXACT)
        assert got == in_sigma_d * math.sqrt(2.0)
        assert abs(got - table_value) <= DETECTABLE_EFFECT_TABLE_ABS


def test_detectable_multiplier_takes_five_or_eight_seeds_only():
    for n in (3, 6, 7, 10):
        with pytest.raises(pt.PreregError):
            lf.detectable_multiplier(n, pt.WELCH, PREREG)


def mha_with_sd(sd: float) -> dict[int, float]:
    """SYNTHETIC MHA losses for seeds 1-5 whose sample SD is exactly `sd`."""
    base = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    return {s: 3.0 + float(x) * sd / float(np.std(base, ddof=1)) for s, x in zip(range(1, 6), base)}


def slope_for_fraction(fraction: float, sd: float, test: str) -> float:
    """The slope at which the 5-seed detectable gap is worth `fraction` of the parameters."""
    gap = lf.detectable_multiplier(5, test, PREREG) * sd
    return -gap / math.log(1.0 - fraction)


def exact_fit(b: float):
    return lf.fit_ladder(synthetic_ladder(None, a=20.0, b=b))


@pytest.mark.parametrize("test", pt.TESTS)
def test_rule_returns_five_when_the_design_sees_five_percent(test):
    sd = 0.004
    d = lf.seed_count_rule(mha_with_sd(sd), exact_fit(slope_for_fraction(0.04, sd, test)), test=test, prereg=PREREG)
    assert d.n_seeds == 5 and d.seeds == (1, 2, 3, 4, 5)
    assert not d.extended and d.leaves_arm_a == ()
    assert close(d.undetectable_param_fraction, 0.04)
    assert d.threshold == 0.05
    assert d.bound_in_sigma_init == d.detectable_in_sigma_init


@pytest.mark.parametrize("test", pt.TESTS)
def test_rule_returns_eight_when_it_does_not(test):
    sd = 0.004
    d = lf.seed_count_rule(mha_with_sd(sd), exact_fit(slope_for_fraction(0.06, sd, test)), test=test, prereg=PREREG)
    assert d.n_seeds == 8 and d.seeds == (1, 2, 3, 4, 5, 6, 7, 8)
    assert d.extended and d.leaves_arm_a == ("mla96", "swa256")
    assert close(d.undetectable_param_fraction, 0.06)
    # The bound that is then registered is what 8 seeds can detect.
    assert d.bound_in_sigma_init == lf.detectable_multiplier(8, test, PREREG)
    assert close(d.bound_gap, d.bound_in_sigma_init * sd)
    assert d.bound_gap < d.detectable_gap


@pytest.mark.parametrize("test", pt.TESTS)
def test_rule_follows_the_plans_three_steps(test):
    sd, b = 0.0061, 0.37
    d = lf.seed_count_rule(mha_with_sd(sd), exact_fit(b), test=test, prereg=PREREG)
    assert close(d.sigma_init, sd)
    multiplier = {
        pt.WELCH: pt.detectable_gap_primary(5, pt.WELCH, PREREG, welch_method=pt.WELCH_EXACT),
        pt.PAIRED: pt.detectable_gap_primary(5, pt.PAIRED, PREREG, welch_method=pt.WELCH_EXACT) * math.sqrt(2.0),
    }[test]
    assert close(d.detectable_gap, multiplier * sd)  # step 1
    assert close(d.param_multiplier, math.exp(-multiplier * sd / b))  # step 2
    assert close(d.undetectable_param_fraction, 1.0 - d.param_multiplier)
    assert d.extended == (1.0 - math.exp(-multiplier * sd / b) > 0.05)  # step 3
    assert d.test == test and close(d.slope, b)


def test_the_two_tests_can_disagree_on_the_seed_count():
    """The paired worst case (3.24) asks for more than Welch (2.63): a slope between
    the two thresholds gives 8 seeds under the paired test and 5 under Welch's."""
    sd = 0.004
    b = 0.5 * (slope_for_fraction(0.05, sd, pt.PAIRED) + slope_for_fraction(0.05, sd, pt.WELCH))
    fit = exact_fit(b)
    assert lf.seed_count_rule(mha_with_sd(sd), fit, test=pt.PAIRED, prereg=PREREG).n_seeds == 8
    assert lf.seed_count_rule(mha_with_sd(sd), fit, test=pt.WELCH, prereg=PREREG).n_seeds == 5


def test_rule_follows_the_registered_threshold():
    sd = 0.004
    fit = exact_fit(slope_for_fraction(0.04, sd, pt.WELCH))
    stricter = {**PREREG, "seed_count": {**PREREG["seed_count"], "max_undetectable_param_fraction": 0.03}}
    assert lf.seed_count_rule(mha_with_sd(sd), fit, test=pt.WELCH, prereg=PREREG).n_seeds == 5
    assert lf.seed_count_rule(mha_with_sd(sd), fit, test=pt.WELCH, prereg=stricter).n_seeds == 8


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.pop(5),  # a base seed missing
        lambda m: m.update({6: 3.0}),  # an extra seed
        lambda m: m.update({3: float("nan")}),
    ],
)
def test_rule_reads_exactly_the_registered_mha_seeds(mutate):
    mha = mha_with_sd(0.004)
    mutate(mha)
    with pytest.raises(pt.PreregError):
        lf.seed_count_rule(mha, exact_fit(0.4), test=pt.WELCH, prereg=PREREG)


def test_rule_refuses_an_unknown_test():
    with pytest.raises(pt.PreregError):
        lf.seed_count_rule(mha_with_sd(0.004), exact_fit(0.4), test="student", prereg=PREREG)


def test_rule_takes_no_variant_data():
    """Plan §1.4: from MHA runs only. The signature has no place for anything else."""
    import inspect

    assert list(inspect.signature(lf.seed_count_rule).parameters) == ["mha_losses", "fit", "test", "prereg"]


def test_ladder_fit_emits_no_verdict():
    source = (config.CONFIG_DIR.parent / "analysis" / "ladder_fit.py").read_text()
    for verdict in (pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED):
        assert f'"{verdict}"' not in source and f"'{verdict}'" not in source
