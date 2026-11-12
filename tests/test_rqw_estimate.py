"""The computed wall-clock estimate (plan §8.5, step 2) on SYNTHETIC manifests only.

Every manifest here is built by tests/fixtures/fake_manifests.py and carries made-up
losses. Nothing reads or writes a real result.
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from analysis import rqw_estimate as rq
from src import config
from tests.fixtures.fake_manifests import fake_manifest
from tests.tolerances import FLOAT64_IDENTITY, scale_relative_error

RECIPE = config.load_recipe()
STEPS_100M = RECIPE["budgets"]["100M"]
STEPS_300M = RECIPE["budgets"]["300M"]
TOKENS_PER_STEP = RECIPE["batch"]["tokens_per_step"]
REFERENCE_STEPS = config.load_arm("iso_wallclock")["reference_steps"]


def line(steps: int, low: tuple[int, float], high: tuple[int, float]) -> float:
    """The interpolation formula, written out in the test."""
    (s_lo, l_lo), (s_hi, l_hi) = low, high
    t = (math.log(steps) - math.log(s_lo)) / (math.log(s_hi) - math.log(s_lo))
    return l_lo + (l_hi - l_lo) * t


def arm_a(variant: str, seed: int, loss: float, lr: float = 1e-3, **kw) -> dict:
    return fake_manifest(variant, init_seed=seed, peak_lr=lr, val_loss=loss, **kw)


def branch(variant: str, seed: int, loss: float, lr: float = 1e-3, **kw) -> dict:
    return fake_manifest(variant, init_seed=seed, peak_lr=lr, branch="100M", val_loss=loss, **kw)


def constructed_manifests() -> list[dict]:
    """MHA, mla192 and swa256 for seeds 1-3, each with a 300M result and a 100M branch."""
    mha_low = {1: 3.50, 2: 3.60, 3: 3.70}
    mla_low = {1: 3.30, 2: 3.40, 3: 3.50}
    mla_full = {1: 2.90, 2: 3.00, 3: 3.10}
    swa_low = {1: 3.80, 2: 3.90, 3: 4.00}
    swa_full = {1: 3.20, 2: 3.30, 3: 3.40}
    out = []
    for s in (1, 2, 3):
        out += [
            arm_a("mha", s, 3.0 + 0.1 * s),
            branch("mha", s, mha_low[s]),
            arm_a("mla192", s, mla_full[s]),
            branch("mla192", s, mla_low[s]),
            arm_a("swa256", s, swa_full[s]),
            branch("swa256", s, swa_low[s]),
        ]
    return out


TIMINGS = {"mha": 1.0, "mla192": 0.5, "swa256": 2.0}


@pytest.fixture(scope="module")
def constructed() -> rq.WallclockEstimate:
    return rq.estimate(constructed_manifests(), TIMINGS)


def by_name(result: rq.WallclockEstimate, variant: str) -> rq.VariantEstimate:
    (found,) = [v for v in result.variants if v.variant == variant]
    return found


# --- steps_afforded -------------------------------------------------------------


def test_steps_afforded_equal_timings_give_reference_steps():
    assert rq.steps_afforded(1.0, 1.0, REFERENCE_STEPS) == REFERENCE_STEPS == 1526


def test_steps_afforded_ten_percent_faster_gives_1679():
    assert rq.steps_afforded(1.1, 1.0, 1526) == 1679


def test_steps_afforded_twice_as_slow_gives_763():
    assert rq.steps_afforded(1.0, 2.0, 1526) == 763


def test_steps_afforded_rounds_half_up():
    # B = 4.5 exactly: half rounds up to 5, not to even (4).
    assert rq.steps_afforded(1.5, 1.0, 3) == 5
    # B = 2.5 exactly: rounds up to 3.
    assert rq.steps_afforded(2.5, 1.0, 1) == 3


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_steps_afforded_rejects_bad_timings(bad):
    with pytest.raises(rq.EstimateError):
        rq.steps_afforded(bad, 1.0, 1526)
    with pytest.raises(rq.EstimateError):
        rq.steps_afforded(1.0, bad, 1526)


# --- interpolate_loss -------------------------------------------------------------


def test_interpolate_loss_returns_end_values_exactly():
    assert rq.interpolate_loss(1000, (1000, 4.0), (4000, 3.0)) == 4.0
    assert rq.interpolate_loss(4000, (1000, 4.0), (4000, 3.0)) == 3.0


def test_interpolate_loss_at_geometric_mean_is_arithmetic_mean():
    # 2000 is the geometric mean of 1000 and 4000, so the midpoint in ln(steps).
    value = rq.interpolate_loss(2000, (1000, 4.0), (4000, 3.0))
    assert scale_relative_error(value, 3.5) <= FLOAT64_IDENTITY


def test_interpolate_loss_hand_computed_case():
    assert scale_relative_error(rq.interpolate_loss(2000, (1000, 4.0), (4000, 3.0)), 3.5) <= FLOAT64_IDENTITY


def test_interpolate_loss_extrapolates_below_on_the_same_line():
    # 500 is one octave below 1000, so the loss is one step beyond 4.0 on the same slope.
    assert scale_relative_error(rq.interpolate_loss(500, (1000, 4.0), (4000, 3.0)), 4.5) <= FLOAT64_IDENTITY


def test_interpolate_loss_extrapolates_above_on_the_same_line():
    assert scale_relative_error(rq.interpolate_loss(8000, (1000, 4.0), (4000, 3.0)), 2.5) <= FLOAT64_IDENTITY


@pytest.mark.parametrize(
    "steps, low, high",
    [
        (2000, (4000, 4.0), (1000, 3.0)),  # wrong order
        (2000, (1000, 4.0), (1000, 3.0)),  # equal
        (2000, (0, 4.0), (4000, 3.0)),  # non-positive
        (0, (1000, 4.0), (4000, 3.0)),  # steps below 1
        (2000, (1000, math.nan), (4000, 3.0)),  # non-finite loss
    ],
)
def test_interpolate_loss_rejects_bad_inputs(steps, low, high):
    with pytest.raises(rq.EstimateError):
        rq.interpolate_loss(steps, low, high)


# --- estimate on constructed manifests --------------------------------------------


def test_interpolation_is_correct_on_constructed_manifests(constructed):
    r = constructed
    ms_mha = TIMINGS["mha"]
    assert r.reference_variant == "mha"
    assert r.reference_steps == REFERENCE_STEPS
    assert r.reference_ms_per_step == ms_mha
    assert r.budget_ms == ms_mha * REFERENCE_STEPS
    assert [v.variant for v in r.variants] == ["swa256", "mla192"]

    mla = by_name(r, "mla192")
    steps = rq.steps_afforded(ms_mha, TIMINGS["mla192"], REFERENCE_STEPS)
    assert steps == 3052
    assert mla.status == "estimated"
    assert mla.ms_per_step == TIMINGS["mla192"]
    assert scale_relative_error(mla.step_time_ratio, 0.5) <= FLOAT64_IDENTITY
    assert mla.steps == steps
    assert mla.tokens == steps * TOKENS_PER_STEP
    assert mla.seeds == (1, 2, 3)

    mla_low = {1: 3.30, 2: 3.40, 3: 3.50}
    mla_full = {1: 2.90, 2: 3.00, 3: 3.10}
    mha_low = {1: 3.50, 2: 3.60, 3: 3.70}
    expected = [line(steps, (STEPS_100M, mla_low[s]), (STEPS_300M, mla_full[s])) for s in (1, 2, 3)]
    expected_ref = [mha_low[s] for s in (1, 2, 3)]
    for got, want in zip(mla.loss_per_seed, expected):
        assert scale_relative_error(got, want) <= FLOAT64_IDENTITY
    assert mla.reference_loss_per_seed == tuple(expected_ref)
    mean = sum(expected) / 3
    mean_ref = sum(expected_ref) / 3
    assert scale_relative_error(mla.loss, mean) <= FLOAT64_IDENTITY
    assert scale_relative_error(mla.reference_loss, mean_ref) <= FLOAT64_IDENTITY
    assert scale_relative_error(mla.gap, mean - mean_ref) <= FLOAT64_IDENTITY
    assert mla.interpolation == rq.INTERPOLATION
    assert mla.reason is None


def test_swa256_slower_than_mha_is_extrapolated_on_the_same_line(constructed):
    swa = by_name(constructed, "swa256")
    assert swa.status == "estimated"
    assert swa.steps == 763 == rq.steps_afforded(1.0, 2.0, REFERENCE_STEPS)
    assert swa.step_time_ratio == 2.0
    assert swa.tokens == 763 * TOKENS_PER_STEP
    assert swa.extrapolated is True

    swa_low = {1: 3.80, 2: 3.90, 3: 4.00}
    swa_full = {1: 3.20, 2: 3.30, 3: 3.40}
    for got, s in zip(swa.loss_per_seed, (1, 2, 3)):
        want = line(763, (STEPS_100M, swa_low[s]), (STEPS_300M, swa_full[s]))
        assert scale_relative_error(got, want) <= FLOAT64_IDENTITY


def test_faster_within_range_is_not_extrapolated(constructed):
    assert by_name(constructed, "mla192").extrapolated is False


def test_variant_faster_than_the_range_is_extrapolated():
    # 0.25 ms per step: 6104 steps, above the 300M step count.
    manifests = [arm_a("mha", 1, 3.0), branch("mha", 1, 3.5)]
    manifests += [arm_a("mla192", 1, 2.9), branch("mla192", 1, 3.3)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.25, "swa256": 1.0})
    mla = by_name(r, "mla192")
    assert mla.steps == 6104 > STEPS_300M
    assert mla.extrapolated is True
    assert mla.status == "estimated"
    want = line(6104, (STEPS_100M, 3.3), (STEPS_300M, 2.9))
    assert scale_relative_error(mla.loss_per_seed[0], want) <= FLOAT64_IDENTITY


# --- ratio only ---------------------------------------------------------------------


def test_variant_missing_from_arm_a_reports_ratio_only():
    manifests = [arm_a("mha", s, 3.0) for s in (1, 2)]
    manifests += [branch("mha", s, 3.5) for s in (1, 2)]
    manifests += [arm_a("mla192", s, 2.9) for s in (1, 2)]
    manifests += [branch("mla192", s, 3.3) for s in (1, 2)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.5, "swa256": 2.0})

    swa = by_name(r, "swa256")
    assert swa.status == "ratio_only"
    assert swa.step_time_ratio == 2.0
    assert swa.ms_per_step == 2.0
    assert swa.steps == 763
    assert swa.tokens == 763 * TOKENS_PER_STEP
    assert swa.loss is None and swa.reference_loss is None and swa.gap is None
    assert swa.seeds == () and swa.loss_per_seed == () and swa.reference_loss_per_seed == ()
    assert swa.interpolation is None
    assert swa.reason and "swa256" in swa.reason

    mla = by_name(r, "mla192")
    assert mla.status == "estimated"
    assert mla.seeds == (1, 2)


# --- partial data -------------------------------------------------------------------


def test_variant_seed_without_100m_branch_is_not_used():
    manifests = [arm_a("mha", s, 3.0) for s in (1, 2)]
    manifests += [branch("mha", s, 3.5) for s in (1, 2)]
    manifests += [arm_a("mla192", s, 2.9) for s in (1, 2, 3)]
    manifests += [branch("mla192", s, 3.3) for s in (1, 2)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.5, "swa256": 2.0})
    assert by_name(r, "mla192").seeds == (1, 2)


def test_seed_without_mha_branch_is_not_used():
    manifests = [arm_a("mha", s, 3.0) for s in (1, 2, 3)]
    manifests += [branch("mha", s, 3.5) for s in (1, 2)]
    manifests += [arm_a("mla192", s, 2.9) for s in (1, 2, 3)]
    manifests += [branch("mla192", s, 3.3) for s in (1, 2, 3)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.5, "swa256": 2.0})
    mla = by_name(r, "mla192")
    assert mla.seeds == (1, 2)
    assert mla.reference_loss_per_seed == (3.5, 3.5)


def test_seeds_one_to_three_with_mha_branches_for_one_and_two_gives_one_and_two():
    manifests = [arm_a("mha", s, 3.0) for s in (1, 2, 3)]
    manifests += [branch("mha", s, 3.5) for s in (1, 2)]
    manifests += [arm_a("mla192", s, 2.9) for s in (1, 2, 3)]
    manifests += [branch("mla192", s, 3.3) for s in (1, 2, 3)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.5, "swa256": 2.0})
    assert by_name(r, "mla192").seeds == (1, 2)


def test_seeds_are_sorted_ascending():
    manifests = [arm_a("mha", s, 3.0) for s in (3, 1, 2)]
    manifests += [branch("mha", s, 3.5) for s in (3, 1, 2)]
    manifests += [arm_a("mla192", s, 2.9) for s in (3, 1, 2)]
    manifests += [branch("mla192", s, 3.3) for s in (3, 1, 2)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.5, "swa256": 2.0})
    assert by_name(r, "mla192").seeds == (1, 2, 3)


def test_branch_at_a_different_learning_rate_is_not_used():
    manifests = [arm_a("mha", 1, 3.0), branch("mha", 1, 3.5)]
    manifests += [arm_a("mla192", 1, 2.9, lr=1e-3), branch("mla192", 1, 3.3, lr=2e-3)]
    r = rq.estimate(manifests, {"mha": 1.0, "mla192": 0.5, "swa256": 2.0})
    mla = by_name(r, "mla192")
    assert mla.status == "ratio_only"
    assert mla.reason


# --- ignored manifests ----------------------------------------------------------------


def _base_manifests() -> list[dict]:
    manifests = [arm_a("mha", s, 3.0 + 0.1 * s) for s in (1, 2)]
    manifests += [branch("mha", s, 3.5 + 0.1 * s) for s in (1, 2)]
    manifests += [arm_a("mla192", s, 2.9 + 0.1 * s) for s in (1, 2)]
    manifests += [branch("mla192", s, 3.3 + 0.1 * s) for s in (1, 2)]
    return manifests


TIMINGS_BASE = {"mha": 1.0, "mla192": 0.5, "swa256": 2.0}


def test_trunk_checkpoints_and_unfinished_runs_are_ignored():
    baseline = rq.as_dict(rq.estimate(_base_manifests(), TIMINGS_BASE))
    extra = [
        fake_manifest("mha", init_seed=1, kind="trunk_checkpoint", completed=False),
        fake_manifest("mla192", init_seed=1, completed=False, val_loss=0.1),
        fake_manifest("mla192", init_seed=2, peak_lr=1e-3, completed=False, val_loss=0.1),
    ]
    assert rq.as_dict(rq.estimate(_base_manifests() + extra, TIMINGS_BASE)) == baseline


def test_other_arms_budgets_ladders_null_and_data_seed_are_ignored():
    baseline = rq.as_dict(rq.estimate(_base_manifests(), TIMINGS_BASE))
    extra = [
        fake_manifest("gqa4", arm="iso_backbone", budget="100M", branches=[], init_seed=1, val_loss=0.1),
        fake_manifest("mha", budget="100M", branches=[], init_seed=1, val_loss=0.1),
        fake_manifest("mha", ladder="mha_L6", init_seed=1, val_loss=0.1),
        fake_manifest("null", init_seed=1, val_loss=0.1),
        fake_manifest("mla192", init_seed=1, data_seed=2, val_loss=0.1),
        fake_manifest("mha", init_seed=1, data_seed=2, branch="100M", val_loss=0.1),
    ]
    assert rq.as_dict(rq.estimate(_base_manifests() + extra, TIMINGS_BASE)) == baseline


# --- errors ---------------------------------------------------------------------------


def test_mixed_learning_rates_among_300m_runs_raise():
    manifests = _base_manifests()
    manifests.append(arm_a("mla192", 3, 2.9, lr=2e-3))
    with pytest.raises(rq.EstimateError):
        rq.estimate(manifests, TIMINGS_BASE)


def test_duplicate_300m_result_raises():
    manifests = _base_manifests() + [arm_a("mla192", 1, 2.95)]
    with pytest.raises(rq.EstimateError):
        rq.estimate(manifests, TIMINGS_BASE)


def test_duplicate_100m_branch_raises():
    manifests = _base_manifests() + [branch("mha", 1, 3.6)]
    with pytest.raises(rq.EstimateError):
        rq.estimate(manifests, TIMINGS_BASE)


@pytest.mark.parametrize("bad", [None, 0.0, -0.5, math.nan, math.inf])
def test_missing_or_non_positive_timing_raises(bad):
    timings = dict(TIMINGS_BASE)
    if bad is None:
        del timings["mla192"]
    else:
        timings["mla192"] = bad
    with pytest.raises(rq.EstimateError):
        rq.estimate(_base_manifests(), timings)


def test_missing_reference_timing_raises():
    timings = {"mla192": 0.5, "swa256": 2.0}
    with pytest.raises(rq.EstimateError):
        rq.estimate(_base_manifests(), timings)


def test_extra_timing_entries_are_ignored():
    baseline = rq.as_dict(rq.estimate(_base_manifests(), TIMINGS_BASE))
    timings = dict(TIMINGS_BASE, gqa4=3.0)
    assert rq.as_dict(rq.estimate(_base_manifests(), timings)) == baseline


# --- output shape ---------------------------------------------------------------------


def test_as_dict_round_trips_through_json(constructed):
    data = rq.as_dict(constructed)
    text = json.dumps(data)
    assert json.loads(text)["variants"][1]["variant"] == "mla192"


def test_result_has_no_verdict_or_p_value_fields(constructed):
    forbidden = {"verdict", "p", "p_value"}
    for cls in (rq.WallclockEstimate, rq.VariantEstimate):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not names & forbidden
    data = rq.as_dict(constructed)
    assert not _keys(data) & forbidden


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        out = set(value)
        for v in value.values():
            out |= _keys(v)
        return out
    if isinstance(value, list):
        out: set[str] = set()
        for v in value:
            out |= _keys(v)
        return out
    return set()
