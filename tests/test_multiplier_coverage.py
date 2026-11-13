"""Simulated coverage of the effective-parameter multiplier's interval (plan §8.4, decision 231).

"MQA performs like MHA with 0.94x the parameters" is a headline number. Its interval is the set
of gap / slope ratios the data do not reject (`analysis/ladder_fit.py`), carrying the gap's
uncertainty and the slope's, with the noise of the 8-layer MHA runs, which move both, counted
once. The other tests check that construction against its definition. This one checks what the
definition is for: over many SIMULATED campaigns with a known gap and a known slope, the 95%
interval holds the true ratio about 95% of the time.

Every loss here is made up. Allowance and seed: tests/tolerances.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from src import config
from tests.fixtures.synthetic_losses import LADDER_SIZES, synthetic_ladder
from tests.tolerances import (
    MULTIPLIER_COVERAGE_N_SIMULATIONS,
    MULTIPLIER_COVERAGE_SIMULATION_SEED,
    MULTIPLIER_COVERAGE_STANDARD_ERRORS,
)

PREREG = config.load_prereg()
PRIMARY = PREREG["primary"]["comparisons"][0]
INTERCEPT = 6.0
SIGMA = 0.01  # SD of seed noise in nats, every run

# (name, test, primary seeds, slope b, true gap in nats, share of MHA's seed effect a variant run keeps).
# The first four have a gap large enough that the slope's uncertainty is a visible part of the
# interval: an interval that left it out would cover too rarely. `shared` 0.9 is the case the
# paired test is chosen for. The last has a shallow ladder, where part of the intervals are
# unbounded; an unbounded interval holds every value.
SCENARIOS = (
    ("welch, 5 seeds", pt.WELCH, 5, 0.14, 0.06, 0.0),
    ("welch, 8 seeds", pt.WELCH, 8, 0.14, 0.06, 0.0),
    ("paired, 5 seeds", pt.PAIRED, 5, 0.14, 0.06, 0.9),
    ("paired, 8 seeds", pt.PAIRED, 8, 0.14, 0.06, 0.9),
    ("welch, 5 seeds, shallow ladder", pt.WELCH, 5, 0.05, 0.02, 0.0),
)


def simulated_multiplier(rng: np.random.Generator, test: str, n: int, b: float, gap: float, shared: float):
    """One SYNTHETIC campaign: the ladder, one primary family on `n` seeds, and the first primary's multiplier."""
    seeds = {"mha_L6": tuple(range(1, 6)), "mha": tuple(range(1, n + 1)), "mha_L10": (1, 2, 3)}
    ladder = synthetic_ladder(rng, a=INTERCEPT, b=b, sigma=SIGMA, seeds=seeds)
    fit = lf.fit_ladder(ladder)
    mha = ladder["mha"][1]
    centre = INTERCEPT - b * math.log(LADDER_SIZES["mha"])
    own = SIGMA * math.sqrt(1.0 - shared * shared)
    losses = {"mha": mha}
    for name in PREREG["primary"]["comparisons"]:
        losses[name] = {
            s: centre + gap + shared * (mha[s] - centre) + own * float(rng.standard_normal()) for s in mha
        }
    comparison = next(c for c in pt.primary_tests(losses, test=test, n_seeds=n, prereg=PREREG) if c.name == PRIMARY)
    if not fit.b > 0:
        return None  # no multiplier exists (decision 223); counted apart
    return lf.comparison_multiplier(comparison, fit)


@pytest.mark.parametrize("index", range(len(SCENARIOS)), ids=[s[0] for s in SCENARIOS])
def test_the_multiplier_interval_holds_the_true_ratio_as_often_as_it_says(index):
    _, test, n, b, gap, shared = SCENARIOS[index]
    rng = np.random.default_rng([MULTIPLIER_COVERAGE_SIMULATION_SEED, index])
    truth = gap / b
    held = total = no_slope = 0
    confidence = None
    for _ in range(MULTIPLIER_COVERAGE_N_SIMULATIONS):
        m = simulated_multiplier(rng, test, n, b, gap, shared)
        if m is None:
            no_slope += 1
            continue
        confidence = m.confidence
        total += 1
        held += (not m.bounded) or (m.log_ratio_low <= truth <= m.log_ratio_high)
    assert confidence == pt.CONFIDENCE
    # A slope that is not positive is rare on these ladders; if it were common the check would say little.
    assert no_slope <= 0.01 * MULTIPLIER_COVERAGE_N_SIMULATIONS
    standard_error = math.sqrt(confidence * (1.0 - confidence) / total)
    assert held / total >= confidence - MULTIPLIER_COVERAGE_STANDARD_ERRORS * standard_error


def test_the_interval_is_not_just_wide(  # the check above passes for an interval that holds everything
):
    """In the first scenario every interval is bounded and its width is of the size of the noise."""
    _, test, n, b, gap, shared = SCENARIOS[0]
    rng = np.random.default_rng([MULTIPLIER_COVERAGE_SIMULATION_SEED, len(SCENARIOS)])
    widths = []
    for _ in range(200):
        m = simulated_multiplier(rng, test, n, b, gap, shared)
        assert m is not None and m.bounded
        widths.append(m.log_ratio_high - m.log_ratio_low)
    # Without any slope uncertainty the interval of gap / b would be about 4 SE of the gap wide,
    # 4 * SIGMA * sqrt(2 / n) / b. The real one is wider than that and of the same order.
    floor = 4.0 * SIGMA * math.sqrt(2.0 / n) / b
    assert 0.8 * floor < float(np.median(widths)) < 3.0 * floor
