"""T-24: the pre-registered decision rule (plan §1.2, §1.3) on SYNTHETIC losses only.

The guard against a statistics bug, the one error no loss curve reveals
(plan §4.6). Thousands of simulated null experiments go through the same
functions that will judge the real runs: test selection, the chosen test, Holm.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from analysis import primary_test as pt
from src import config
from tests.fixtures.synthetic_losses import REGIMES, simulate_experiments
from tests.tolerances import (
    DETECTABLE_EFFECT_TABLE_ABS,
    FLOAT64_IDENTITY,
    FWER_MAX,
    FWER_N_SIMULATIONS,
    FWER_SIMULATION_SEED,
    FWER_STANDARD_ERRORS,
    scale_relative_error,
)

ROOT = Path(__file__).resolve().parent.parent
PREREG = config.load_prereg()
SEED_COUNTS = (5, 8)  # plan §1.4
RHOS = (0.0, 1.0)  # selection data independent of, or identical in noise to, the test data


def monte_carlo_bound(rate: float) -> float:
    """Largest estimate still consistent with a true rate of `rate` (frozen allowance)."""
    return rate + FWER_STANDARD_ERRORS * math.sqrt(rate * (1.0 - rate) / FWER_N_SIMULATIONS)


def close(a, b) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return scale_relative_error(a, b) <= FLOAT64_IDENTITY


# --- The whole procedure on simulated nulls (plan §5, "Primary statistics") ---


def simulated_procedure(n_seeds: int, regime_index: int, rho: float, effects=None):
    """Run the §1.2 procedure on FWER_N_SIMULATIONS synthetic experiments.

    Returns (experiments, paired_chosen, p_values, holm_reject).
    """
    sel = PREREG["test_selection"]
    primary = PREREG["primary"]
    rng = np.random.default_rng([FWER_SIMULATION_SEED, n_seeds, regime_index, int(rho * 100)])
    ex = simulate_experiments(
        rng,
        n_experiments=FWER_N_SIMULATIONS,
        n_seeds=n_seeds,
        n_variants=len(primary["comparisons"]),
        regime=REGIMES[regime_index],
        rho=rho,
        n_select=len(sel["sigma_init"]["init_seeds"]),
        n_pairs=len(sel["reproducibility_pairs"]["init_seeds"]),
        effects=effects,
    )
    paired = pt.choose_paired(
        pt.sample_sd(ex.mha_select), pt.s_rep(ex.repro_deltas), sel["paired_if_s_rep_ratio_below"]
    )
    reference = ex.reference[:, None, :]
    p = np.where(
        paired[:, None], pt.paired_t(ex.variants, reference).p, pt.welch_t(ex.variants, reference).p
    )
    reject, _ = pt.holm(p, primary["family_alpha"])
    return ex, paired, p, reject


@pytest.mark.parametrize("rho", RHOS)
@pytest.mark.parametrize("regime_index", range(len(REGIMES)), ids=[r.name for r in REGIMES])
@pytest.mark.parametrize("n_seeds", SEED_COUNTS)
def test_family_wise_error_on_simulated_nulls(n_seeds, regime_index, rho):
    _, _, _, reject = simulated_procedure(n_seeds, regime_index, rho)
    family_wise_error = reject.any(axis=-1).mean()
    assert family_wise_error <= monte_carlo_bound(FWER_MAX)


@pytest.mark.parametrize("n_seeds", SEED_COUNTS)
def test_simulation_can_fail_without_the_correction(n_seeds):
    """The same simulation flags a broken procedure: three uncorrected tests."""
    _, _, p, reject = simulated_procedure(n_seeds, 0, 0.0)
    alpha = PREREG["primary"]["family_alpha"]
    uncorrected = (p <= alpha).any(axis=-1).mean()
    assert uncorrected > monte_carlo_bound(FWER_MAX)
    assert reject.any(axis=-1).mean() < uncorrected


@pytest.mark.parametrize("n_seeds", SEED_COUNTS)
def test_the_switch_is_exercised(n_seeds):
    by_name = {r.name: i for i, r in enumerate(REGIMES)}
    _, strong, _, _ = simulated_procedure(n_seeds, by_name["pairing_strong"], 0.0)
    _, absent, _, _ = simulated_procedure(n_seeds, by_name["pairing_absent"], 0.0)
    _, boundary, _, _ = simulated_procedure(n_seeds, by_name["boundary_mixed"], 0.0)
    assert strong.mean() > 0.5 > absent.mean()  # a majority each way
    assert boundary.any() and not boundary.all()


@pytest.mark.parametrize("n_seeds", SEED_COUNTS)
def test_public_api_agrees_with_the_simulated_procedure(n_seeds):
    """`select_test` and `primary_tests` give the verdicts the simulation counted."""
    by_name = {r.name: i for i, r in enumerate(REGIMES)}
    effects = (0.0, 1.5, 4.0)  # a mix of verdicts, not only nulls
    ex, paired, _, reject = simulated_procedure(n_seeds, by_name["boundary_mixed"], 0.0, effects)
    sel = PREREG["test_selection"]
    select_seeds = sel["sigma_init"]["init_seeds"]
    pair_seeds = sel["reproducibility_pairs"]["init_seeds"]
    seeds = pt.registered_seeds(n_seeds, PREREG)
    names = PREREG["primary"]["comparisons"]
    seen = set()
    for i in range(300):
        selection = pt.select_test(
            dict(zip(select_seeds, ex.mha_select[i])),
            {s: (ex.mha_select[i, k], ex.repro_reruns[i, k]) for k, s in enumerate(pair_seeds)},
            PREREG,
        )
        assert (selection.test == pt.PAIRED) == bool(paired[i])
        losses = {PREREG["primary"]["reference"]: dict(zip(seeds, ex.reference[i]))}
        for j, name in enumerate(names):
            losses[name] = dict(zip(seeds, ex.variants[i, j]))
        results = pt.primary_tests(losses, test=selection.test, n_seeds=n_seeds, prereg=PREREG)
        assert [c.name for c in results] == list(names)
        assert [c.verdict == pt.VERDICT_DETECTED for c in results] == list(reject[i])
        seen.update((selection.test, c.verdict) for c in results)
    assert len(seen) == 4  # both tests, both verdicts


# --- The numerical core against independent references -------------------------


def _samples(seed: int, n: int, shift: float):
    rng = np.random.default_rng(seed)
    return rng.normal(3.0 + shift, 0.05, n), rng.normal(3.0, 0.02, n)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_paired_t_matches_scipy_and_the_formula(n):
    v, r = _samples(n, n, 0.03)
    res = pt.paired_t(v, r)
    ref = stats.ttest_rel(v, r)
    d = v - r
    se = d.std(ddof=1) / math.sqrt(n)
    assert close(res.mean_gap, d.mean())
    assert close(res.t, d.mean() / se)
    assert res.df == n - 1
    assert close(res.t, ref.statistic) and close(res.p, ref.pvalue)
    ci = ref.confidence_interval(pt.CONFIDENCE)
    assert close(res.ci_low, ci.low) and close(res.ci_high, ci.high)


@pytest.mark.parametrize("nv, nr", [(5, 5), (8, 8), (3, 5), (3, 8)])
def test_welch_t_matches_scipy_and_the_formula(nv, nr):
    v, _ = _samples(nv, nv, 0.03)
    _, r = _samples(nr + 100, nr, 0.0)
    res = pt.welch_t(v, r)
    ref = stats.ttest_ind(v, r, equal_var=False)
    qv, qr = v.var(ddof=1) / nv, r.var(ddof=1) / nr
    df = (qv + qr) ** 2 / (qv**2 / (nv - 1) + qr**2 / (nr - 1))
    assert close(res.mean_gap, v.mean() - r.mean())
    assert close(res.se, math.sqrt(qv + qr))
    assert close(res.df, df)
    assert close(res.t, ref.statistic) and close(res.p, ref.pvalue)
    ci = ref.confidence_interval(pt.CONFIDENCE)
    assert close(res.ci_low, ci.low) and close(res.ci_high, ci.high)


def test_gap_sign_is_variant_minus_reference():
    v, r = np.array([3.2, 3.3, 3.1]), np.array([3.0, 3.1, 2.95])
    assert pt.paired_t(v, r).mean_gap > 0 and pt.welch_t(v, r).mean_gap > 0
    assert pt.paired_t(r, v).mean_gap < 0 and pt.welch_t(r, v).mean_gap < 0


def test_core_broadcasts_like_the_scalar_case():
    rng = np.random.default_rng(7)
    v, r = rng.normal(size=(4, 3, 5)), rng.normal(size=(4, 1, 5))
    for fn in (pt.paired_t, pt.welch_t):
        batch = fn(v, r)
        assert batch.p.shape == (4, 3)
        one = fn(v[2, 1], r[2, 0])
        for field in pt.TResult._fields:
            assert getattr(batch, field)[2, 1] == getattr(one, field)


def test_no_spread_at_all():
    same = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    for fn in (pt.paired_t, pt.welch_t):
        null = fn(same, same)
        assert null.t == 0.0 and null.p == 1.0
        shifted = fn(same + 0.5, same)
        assert shifted.p == 0.0 and shifted.t == math.inf


def test_too_few_runs_raise():
    with pytest.raises(pt.PreregError):
        pt.paired_t([1.0], [2.0])
    with pytest.raises(pt.PreregError):
        pt.paired_t([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(pt.PreregError):
        pt.welch_t([1.0], [2.0, 3.0])
    with pytest.raises(pt.PreregError):
        pt.run_test("student", [1.0, 2.0], [2.0, 3.0])


def test_s_rep_and_sample_sd_by_hand():
    assert close(pt.s_rep([0.02, -0.02, 0.02]), math.sqrt(2e-4))
    assert close(pt.s_rep([0.3, 0.0, 0.0]), math.sqrt(0.09 / 3 / 2))
    assert close(pt.sample_sd([1.0, 2.0, 3.0, 4.0, 5.0]), math.sqrt(2.5))


def test_paired_needs_s_rep_strictly_below_the_threshold():
    assert bool(pt.choose_paired(1.0, 0.59, 0.6)) is True
    assert bool(pt.choose_paired(1.0, 0.6, 0.6)) is False  # equal is not below
    assert bool(pt.choose_paired(1.0, 0.61, 0.6)) is False
    assert list(pt.choose_paired([1.0, 2.0], [0.7, 0.7], 0.6)) == [False, True]


# --- Holm ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, reject, adjusted",
    [
        ([0.01, 0.04, 0.03], [True, False, False], [0.03, 0.06, 0.06]),
        ([0.01, 0.02, 0.03], [True, True, True], [0.03, 0.04, 0.04]),
        ([0.02, 0.02, 0.02], [False, False, False], [0.06, 0.06, 0.06]),
        ([0.03, 0.001, 0.5], [False, True, False], [0.06, 0.003, 0.5]),
        ([0.9, 0.8, 0.7], [False, False, False], [1.0, 1.0, 1.0]),
        ([0.04], [True], [0.04]),
        ([0.06], [False], [0.06]),
    ],
)
def test_holm_known_cases(p, reject, adjusted):
    got_reject, got_adjusted = pt.holm(p, 0.05)
    assert list(got_reject) == reject
    assert close(got_adjusted, np.array(adjusted))


def test_holm_matches_the_step_down_definition():
    rng = np.random.default_rng(11)
    alpha = PREREG["primary"]["family_alpha"]
    for p in rng.uniform(0.0, 0.1, size=(500, 3)):
        expected = np.zeros(3, dtype=bool)
        for rank, index in enumerate(np.argsort(p, kind="stable")):
            if p[index] > alpha / (3 - rank):
                break
            expected[index] = True
        assert list(pt.holm(p, alpha)[0]) == list(expected)


def test_holm_does_not_depend_on_order():
    p = np.array([0.012, 0.03, 0.02])
    reject, adjusted = pt.holm(p, 0.05)
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        r2, a2 = pt.holm(p[perm], 0.05)
        assert list(r2) == list(reject[perm]) and list(a2) == list(adjusted[perm])


# --- Test selection ------------------------------------------------------------

MHA_100M = {1: 3.300, 2: 3.320, 3: 3.280, 4: 3.310, 5: 3.290}


def _pairs(delta: float):
    return {s: (MHA_100M[s], MHA_100M[s] - delta) for s in (1, 2, 3)}


def test_select_test_reports_its_inputs():
    sel = pt.select_test(MHA_100M, _pairs(0.004), PREREG)
    assert close(sel.sigma_init, np.std(list(MHA_100M.values()), ddof=1))
    assert close(sel.s_rep, 0.004 / math.sqrt(2.0))
    assert close(sel.ratio, sel.s_rep / sel.sigma_init)
    assert sel.threshold == 0.6  # plan §1.2, proposed (decision 20)
    assert sel.test == pt.PAIRED


def test_select_test_switches_to_welch_when_reruns_differ_like_seeds():
    assert pt.select_test(MHA_100M, _pairs(0.02), PREREG).test == pt.WELCH


def test_select_test_follows_the_registered_threshold():
    sigma = float(np.std(list(MHA_100M.values()), ddof=1))
    for factor, expected in ((0.59, pt.PAIRED), (0.61, pt.WELCH)):
        delta = factor * sigma * math.sqrt(2.0)  # gives s_rep = factor * sigma_init
        assert pt.select_test(MHA_100M, _pairs(delta), PREREG).test == expected


@pytest.mark.parametrize(
    "mha, pairs",
    [
        ({k: v for k, v in MHA_100M.items() if k != 5}, _pairs(0.004)),  # a seed missing
        ({**MHA_100M, 6: 3.3}, _pairs(0.004)),  # an extra seed
        (MHA_100M, {1: (3.3, 3.3), 2: (3.3, 3.3)}),  # a pair missing
        (MHA_100M, {**_pairs(0.004), 4: (3.3, 3.3)}),  # an extra pair
        ({**MHA_100M, 3: float("nan")}, _pairs(0.004)),
        (MHA_100M, {**_pairs(0.004), 2: (3.3, float("inf"))}),
    ],
)
def test_select_test_refuses_anything_but_the_registered_runs(mha, pairs):
    with pytest.raises(pt.PreregError):
        pt.select_test(mha, pairs, PREREG)


# --- Primary comparisons ---------------------------------------------------------

NOISE = (0.004, -0.003, 0.001, -0.002, 0.0025, -0.001, 0.003, -0.0035)
MHA_SPREAD = (0.00, 0.02, -0.02, 0.01, -0.01, 0.015, -0.015, 0.005)


def primary_losses(n_seeds: int, gaps: dict[str, float]) -> dict:
    seeds = pt.registered_seeds(n_seeds, PREREG)
    mha = {s: 3.0 + MHA_SPREAD[i] for i, s in enumerate(seeds)}
    out = {"mha": mha}
    for shift, (name, gap) in enumerate(gaps.items()):
        out[name] = {
            s: mha[s] + gap + NOISE[(i + shift) % len(NOISE)] for i, s in enumerate(seeds)
        }
    return out


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("n_seeds", SEED_COUNTS)
def test_primary_verdicts_on_constructed_gaps(n_seeds, test):
    losses = primary_losses(n_seeds, {"gqa4": 0.0, "mqa": 0.5, "mla192": -0.5})
    results = pt.primary_tests(losses, test=test, n_seeds=n_seeds, prereg=PREREG)
    assert [c.name for c in results] == ["gqa4", "mqa", "mla192"]  # registered order
    by_name = {c.name: c for c in results}
    assert by_name["gqa4"].verdict == pt.VERDICT_NOT_DETECTED
    assert by_name["mqa"].verdict == pt.VERDICT_DETECTED
    assert by_name["mla192"].verdict == pt.VERDICT_DETECTED
    assert by_name["mqa"].mean_gap > 0 > by_name["mla192"].mean_gap
    for c in results:
        assert c.kind == "primary" and c.reference == "mha" and c.test == test
        assert c.alpha == 0.05 and c.confidence == 0.95
        assert c.seeds == c.reference_seeds == tuple(pt.registered_seeds(n_seeds, PREREG))
        assert len(c.variant_losses) == len(c.reference_losses) == n_seeds
        assert c.ci_low < c.mean_gap < c.ci_high
        assert (c.verdict == pt.VERDICT_DETECTED) == (c.p_adjusted <= c.alpha)
    # Every per-seed loss is reported, and the paired differences when paired.
    assert by_name["mqa"].variant_losses == tuple(losses["mqa"][s] for s in by_name["mqa"].seeds)
    if test == pt.PAIRED:
        assert close(np.array(by_name["mqa"].differences), np.array(by_name["mqa"].variant_losses)
                     - np.array(by_name["mqa"].reference_losses))
    else:
        assert by_name["mqa"].differences is None
    # The ruled-out bound is the upper end of the interval, only when not detected.
    assert by_name["gqa4"].ruled_out_above == by_name["gqa4"].ci_high
    assert by_name["mqa"].ruled_out_above is None


def test_primary_adjusted_p_is_holm_over_the_three():
    losses = primary_losses(5, {"gqa4": 0.004, "mqa": 0.008, "mla192": 0.02})
    results = pt.primary_tests(losses, test=pt.PAIRED, n_seeds=5, prereg=PREREG)
    raw = np.array([c.p for c in results])
    reject, adjusted = pt.holm(raw, 0.05)
    assert close(np.array([c.p_adjusted for c in results]), adjusted)
    assert [c.verdict == pt.VERDICT_DETECTED for c in results] == list(reject)
    ref = np.array([losses["mha"][s] for s in (1, 2, 3, 4, 5)])
    for c in results:
        own = stats.ttest_rel(np.array([losses[c.name][s] for s in (1, 2, 3, 4, 5)]), ref)
        assert close(c.p, own.pvalue)


def test_holm_changes_a_verdict_that_an_uncorrected_test_would_give():
    """A gap with 0.05/3 < p <= 0.05 next to two nulls is not detected."""
    for gap in np.linspace(0.002, 0.02, 400):
        losses = primary_losses(5, {"gqa4": float(gap), "mqa": 0.0, "mla192": 0.0})
        c = pt.primary_tests(losses, test=pt.PAIRED, n_seeds=5, prereg=PREREG)[0]
        if 0.05 / 3 < c.p <= 0.05:
            assert c.verdict == pt.VERDICT_NOT_DETECTED and c.p_adjusted > 0.05
            return
    raise AssertionError("no constructed gap landed between alpha/3 and alpha")


@pytest.mark.parametrize("test", pt.TESTS)
def test_identical_runs_are_not_detected(test):
    losses = primary_losses(5, {"gqa4": 0.0, "mqa": 0.0, "mla192": 0.0})
    for name in ("gqa4", "mqa", "mla192"):
        losses[name] = dict(losses["mha"])
    for c in pt.primary_tests(losses, test=test, n_seeds=5, prereg=PREREG):
        assert c.verdict == pt.VERDICT_NOT_DETECTED and c.mean_gap == 0.0


def _drop(d, key):
    return {k: v for k, v in d.items() if k != key}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda L: L.update(mqa=_drop(L["mqa"], 5)),  # a subset of seeds
        lambda L: L.update(mqa={**L["mqa"], 6: 3.0}),  # an extra seed
        lambda L: L.update(mha=_drop(L["mha"], 1)),
        lambda L: L.pop("mla192"),  # a primary missing
        lambda L: L.pop("mha"),
        lambda L: L.update(gqa4={**L["gqa4"], 2: float("nan")}),
    ],
)
def test_primary_tests_refuse_anything_but_the_registered_runs(mutate):
    losses = primary_losses(5, {"gqa4": 0.0, "mqa": 0.5, "mla192": -0.5})
    mutate(losses)
    with pytest.raises(pt.PreregError):
        pt.primary_tests(losses, test=pt.PAIRED, n_seeds=5, prereg=PREREG)


def test_seed_count_is_five_or_eight_only():
    assert pt.registered_seeds(5, PREREG) == [1, 2, 3, 4, 5]
    assert pt.registered_seeds(8, PREREG) == [1, 2, 3, 4, 5, 6, 7, 8]
    for bad in (3, 6, 7, 9):
        with pytest.raises(pt.PreregError):
            pt.registered_seeds(bad, PREREG)
    losses = primary_losses(8, {"gqa4": 0.0, "mqa": 0.5, "mla192": -0.5})
    with pytest.raises(pt.PreregError):  # eight seeds offered, five registered
        pt.primary_tests(losses, test=pt.PAIRED, n_seeds=5, prereg=PREREG)
    with pytest.raises(pt.PreregError):
        pt.primary_tests(losses, test="student", n_seeds=8, prereg=PREREG)


def test_extra_variants_do_not_join_the_family():
    """Exploratory losses in the same mapping change nothing (plan §13.14)."""
    losses = primary_losses(5, {"gqa4": 0.004, "mqa": 0.008, "mla192": 0.02})
    base = pt.primary_tests(losses, test=pt.PAIRED, n_seeds=5, prereg=PREREG)
    losses["gqa2"] = {1: 9.0, 2: 9.1, 3: 9.2}
    assert pt.primary_tests(losses, test=pt.PAIRED, n_seeds=5, prereg=PREREG) == base


# --- Null control, positive control, exploratory --------------------------------


@pytest.mark.parametrize("test", pt.TESTS)
def test_null_control_pairs_in_seed_order_and_is_a_single_test(test):
    null = {11: 3.305, 12: 3.318, 13: 3.284, 14: 3.307, 15: 3.292}
    c = pt.null_control(null, MHA_100M, test=test, prereg=PREREG)
    assert (c.name, c.reference, c.kind) == ("null", "mha", "null_control")
    assert c.seeds == (11, 12, 13, 14, 15) and c.reference_seeds == (1, 2, 3, 4, 5)
    assert c.alpha == 0.05 and c.p_adjusted == c.p
    assert c.verdict == pt.VERDICT_NOT_DETECTED
    if test == pt.PAIRED:
        assert close(np.array(c.differences), np.array([null[10 + k] - MHA_100M[k] for k in range(1, 6)]))
    shifted = {s: v + 0.5 for s, v in null.items()}
    assert pt.null_control(shifted, MHA_100M, test=test, prereg=PREREG).verdict == pt.VERDICT_DETECTED
    with pytest.raises(pt.PreregError):  # MHA's own seeds are not the null control's
        pt.null_control(MHA_100M, MHA_100M, test=test, prereg=PREREG)


@pytest.mark.parametrize("test", pt.TESTS)
def test_positive_control_is_a_single_test_at_its_own_alpha(test):
    mha = {s: 3.0 + MHA_SPREAD[i] for i, s in enumerate((1, 2, 3, 4, 5))}
    l6 = {s: v + 0.2 + NOISE[i] for i, (s, v) in enumerate(mha.items())}
    c = pt.positive_control(l6, mha, test=test, prereg=PREREG)
    assert (c.name, c.reference, c.kind) == ("mha_L6", "mha", "positive_control")
    assert c.alpha == 0.05 and c.p_adjusted == c.p and c.verdict == pt.VERDICT_DETECTED
    same = pt.positive_control(dict(mha), mha, test=test, prereg=PREREG)
    assert same.verdict == pt.VERDICT_NOT_DETECTED
    with pytest.raises(pt.PreregError):  # three seeds are not the registered five
        pt.positive_control(_drop(_drop(l6, 4), 5), mha, test=test, prereg=PREREG)


def test_single_test_uses_alpha_not_alpha_over_three():
    """A gap with 0.05/3 < p <= 0.05 is detected as a single test."""
    mha = {s: 3.0 + MHA_SPREAD[i] for i, s in enumerate((1, 2, 3, 4, 5))}
    for gap in np.linspace(0.002, 0.02, 400):
        l6 = {s: v + float(gap) + NOISE[i] for i, (s, v) in enumerate(mha.items())}
        c = pt.positive_control(l6, mha, test=pt.PAIRED, prereg=PREREG)
        if 0.05 / 3 < c.p <= 0.05:
            assert c.verdict == pt.VERDICT_DETECTED
            return
    raise AssertionError("no constructed gap landed between alpha/3 and alpha")


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("n_mha", SEED_COUNTS)
def test_exploratory_has_an_interval_and_no_claim(test, n_mha):
    mha = {s: 3.0 + MHA_SPREAD[i] for i, s in enumerate(range(1, n_mha + 1))}
    gqa2 = {1: 3.9, 2: 3.95, 3: 3.85}  # a gap no test would miss
    c = pt.exploratory("gqa2", gqa2, mha, test=test, prereg=PREREG)
    assert c.verdict == pt.VERDICT_EXPLORATORY and c.kind == "exploratory"
    assert c.p is None and c.p_adjusted is None and c.t is None and c.alpha is None
    assert c.ruled_out_above is None
    assert c.ci_low < c.mean_gap < c.ci_high and c.seeds == (1, 2, 3)
    assert c.reference_seeds == ((1, 2, 3) if test == pt.PAIRED else tuple(range(1, n_mha + 1)))
    with pytest.raises(pt.PreregError):
        pt.exploratory("mqa", gqa2, mha, test=test, prereg=PREREG)  # a primary is not exploratory
    with pytest.raises(pt.PreregError):
        pt.exploratory("gqa2", {1: 3.9, 2: 3.95}, mha, test=test, prereg=PREREG)


# --- Detectable effect (plan §1.3) -------------------------------------------------

POOLED = pt.WELCH_POOLED_DF
EXACT = pt.WELCH_EXACT


def table_close(value: float, plan_value: float) -> bool:
    return abs(value - plan_value) <= DETECTABLE_EFFECT_TABLE_ABS


@pytest.mark.parametrize(
    "n, test, plan_value, plan_worst_case",
    [
        (3, pt.PAIRED, 5.7, 8.0),
        (5, pt.PAIRED, 2.3, 3.2),
        (5, pt.WELCH, 2.5, 2.5),
        (8, pt.PAIRED, 1.5, 2.1),
        (8, pt.WELCH, 1.8, 1.8),
    ],
)
def test_plan_1_3_table_is_reproduced(n, test, plan_value, plan_worst_case):
    gap = pt.detectable_gap_primary(n, test, PREREG, welch_method=POOLED)
    assert table_close(gap, plan_value)
    worst = gap * math.sqrt(2.0) if test == pt.PAIRED else gap  # sigma_d <= sqrt(2) sigma_init
    assert table_close(worst, plan_worst_case)


def test_plan_1_3_other_quoted_values():
    power = PREREG["seed_count"]["power"]
    # "With six primaries instead of three, the 5-seed paired multiplier would be 2.8."
    assert table_close(pt.detectable_gap(5, pt.PAIRED, 0.05 / 6, power, welch_method=POOLED), 2.8)
    # Positive control, a single test at alpha 0.05 (plan §8.4).
    alpha = PREREG["positive_control"]["alpha"]
    assert table_close(pt.detectable_gap(5, pt.PAIRED, alpha, power, welch_method=POOLED), 1.7)
    assert table_close(pt.detectable_gap(5, pt.WELCH, alpha, power, welch_method=POOLED), 2.0)
    assert table_close(pt.detectable_gap(3, pt.PAIRED, alpha, power, welch_method=POOLED), 3.3)
    # Break-even s_rep / sigma_init: 0.77 at 5 seeds and 0.88 at 8 (plan §1.2).
    for n, plan_value in ((5, 0.77), (8, 0.88)):
        paired = pt.detectable_gap_primary(n, pt.PAIRED, PREREG, welch_method=POOLED)
        welch = pt.detectable_gap_primary(n, pt.WELCH, PREREG, welch_method=POOLED)
        assert table_close(welch / paired / math.sqrt(2.0), plan_value)


def test_primary_alpha_is_holms_strictest_step():
    assert pt.primary_alpha_strictest(PREREG) == 0.05 / 3


@pytest.mark.parametrize("n", SEED_COUNTS)
@pytest.mark.parametrize("test", pt.TESTS)
def test_detectable_gap_has_the_target_power_when_simulated(n, test):
    """The exact calculation is the power of the test the code really runs."""
    alpha = pt.primary_alpha_strictest(PREREG)
    target = PREREG["seed_count"]["power"]
    gap = pt.detectable_gap_primary(n, test, PREREG, welch_method=EXACT)
    rng = np.random.default_rng([FWER_SIMULATION_SEED, n, pt.TESTS.index(test)])
    variant = rng.standard_normal((FWER_N_SIMULATIONS, n)) + gap
    reference = (
        np.zeros_like(variant) if test == pt.PAIRED else rng.standard_normal(variant.shape)
    )
    simulated = (pt.run_test(test, variant, reference).p <= alpha).mean()
    allowance = FWER_STANDARD_ERRORS * math.sqrt(target * (1.0 - target) / FWER_N_SIMULATIONS)
    assert abs(simulated - target) <= allowance


@pytest.mark.parametrize("n", SEED_COUNTS)
def test_pooled_df_formula_overstates_welch_power(n):
    """Why two methods exist: at the plan's Welch multiplier, the test that is
    actually run has less than the target power. Open question for Phil."""
    target = PREREG["seed_count"]["power"]
    alpha = pt.primary_alpha_strictest(PREREG)
    pooled_gap = pt.detectable_gap_primary(n, pt.WELCH, PREREG, welch_method=POOLED)
    exact_gap = pt.detectable_gap_primary(n, pt.WELCH, PREREG, welch_method=EXACT)
    assert exact_gap > pooled_gap
    assert pt.power(pooled_gap, n, pt.WELCH, alpha, welch_method=EXACT) < target
    assert close(pt.power(exact_gap, n, pt.WELCH, alpha, welch_method=EXACT), target)
    assert close(pt.power(pooled_gap, n, pt.WELCH, alpha, welch_method=POOLED), target)


def test_power_basics():
    for test in pt.TESTS:
        for method in pt.WELCH_POWER_METHODS:
            low = pt.power(0.5, 5, test, 0.05, welch_method=method)
            high = pt.power(2.0, 5, test, 0.05, welch_method=method)
            assert 0.0 < low < high < 1.0
            assert close(
                pt.power(-2.0, 5, test, 0.05, welch_method=method), high
            )  # two-sided
    # With no effect the paired test rejects at exactly alpha.
    assert close(pt.power(0.0, 5, pt.PAIRED, 0.05, welch_method=EXACT), 0.05)
    with pytest.raises(pt.PreregError):
        pt.power(1.0, 5, pt.WELCH, 0.05, welch_method="guess")
    with pytest.raises(pt.PreregError):
        pt.power(1.0, 5, "student", 0.05, welch_method=EXACT)


# --- Only primary_test.py emits a verdict (plan §11) -------------------------------

VERDICT_WORDS = (pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


def verdict_literals(source: str) -> list[tuple[int, str]]:
    """String constants, outside docstrings, that spell out a detection verdict."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            text = " ".join(node.value.lower().split())
            if any(text == w or text.startswith(w + " ") for w in VERDICT_WORDS):
                hits.append((node.lineno, node.value))
    return hits


def test_verdict_strings():
    assert pt.VERDICT_DETECTED == "detected"
    assert pt.VERDICT_NOT_DETECTED == "not detected"
    assert pt.VERDICT_EXPLORATORY == "exploratory"


def test_the_verdict_scan_finds_a_planted_literal():
    planted = 'def f():\n    """not detected is fine in a docstring"""\n    return "Not  detected at this scale"\n'
    assert verdict_literals(planted) == [(3, "Not  detected at this scale")]
    assert verdict_literals('x = "undetected"\ny = "detector"\n') == []
    own = (ROOT / "analysis" / "primary_test.py").read_text()
    assert len(verdict_literals(own)) == 2  # the two constants, defined once each


def test_no_other_code_emits_a_verdict():
    offenders = []
    for folder in ("src", "analysis", "scripts"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path == ROOT / "analysis" / "primary_test.py":
                continue
            for line, text in verdict_literals(path.read_text()):
                offenders.append(f"{path.relative_to(ROOT)}:{line}: {text!r}")
    assert not offenders, "import the verdict constants from analysis.primary_test:\n" + "\n".join(offenders)
