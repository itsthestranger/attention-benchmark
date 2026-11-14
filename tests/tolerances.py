"""Every tolerance used by the test suite. FROZEN (plan §17.3).

This file changes only through a pull request that Phil approves. Loosening a
value here in order to make a test pass is forbidden (CLAUDE.md, hard rules).
Test files import their tolerances from here and contain no literal ones.

Definition
----------
Unless a constant says otherwise, a numerical tolerance bounds the error
relative to the scale of the reference tensor:

    max|a - b| / max|b| <= tol

where ``b`` is the reference. ``scale_relative_error`` below is that
definition; tests use it instead of re-deriving it. There is no absolute term:
an absolute bound is vacuous for small tensors such as parameter gradients and
too strict for large ones such as trained logits.

Provenance
----------
Each constant names its source. "Phil 2026-11-12" marks values agreed in the
first build session, including one deviation from the plan (REF_OPT_FP32).
"""

from typing import Final


def scale_relative_error(a, b) -> float:
    """max|a - b| / max|b|, with ``b`` the reference.

    Accepts tensors, arrays or plain numbers. If the reference is all zeros the
    result is 0.0 when ``a`` is too, and infinity otherwise.
    """
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff, scale = abs(a - b), abs(b)
    else:
        diff, scale = float(abs(a - b).max()), float(abs(b).max())
    if scale == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / scale


# --- Agreement between implementations (plan §5) ---------------------------

# `opt` against `ref` on output, loss, input grads and every parameter grad.
# Plan §5 says 1e-6 for fp32. Deviation, Phil 2026-11-12: 1e-6 is about 8 units
# of fp32 precision and correct implementations measure 0.2e-6 to 1.8e-6.
REF_OPT_FP32: Final = 1e-5
# Plan §5, unchanged. `gpu` tier.
REF_OPT_FP16: Final = 1e-3

# The same agreement checks run once more in float64, where a correct
# implementation agrees to rounding. Addition, Phil 2026-11-12.
AGREEMENT_FP64: Final = 1e-12

# Cached token-by-token logits against the full forward, at ctx 1024.
# Plan §5 gives no number. Phil 2026-11-12.
KV_CACHE_DECODE_FP32: Final = 1e-5

# MLA absorbed path against the naive path. Plan §3.3 and §5.
MLA_ABSORBED: Final = 1e-4

# --- Resume and branch fidelity (plan §5: "within `opt` tolerance") ---------

RESUME_FIDELITY_FP32: Final = REF_OPT_FP32
RESUME_FIDELITY_FP16: Final = REF_OPT_FP16
BRANCH_FIDELITY_FP32: Final = REF_OPT_FP32
BRANCH_FIDELITY_FP16: Final = REF_OPT_FP16

# --- Trainer -----------------------------------------------------------------

# Gradient accumulation over micro-batches against the same batch in one pass
# (TASKS.md T-15). Not in the plan; same fp32 bound as REF_OPT_FP32.
GRAD_ACCUMULATION_FP32: Final = REF_OPT_FP32

# --- Model properties --------------------------------------------------------

# Step-0 attention-logit std agrees across all variants within 10% (plan §5).
INIT_LOGIT_STD_REL: Final = 0.10

# MLA total parameters against the 17,308,032 target: |offset| < 0.05%
# (plan §3.2, §4.3). Non-MLA variants must match exactly; no tolerance.
MLA_PARAM_OFFSET_REL: Final = 0.0005

# Attention FLOP shares against the plan §3.1 table, in percentage points
# (TASKS.md T-08: "within one point").
FLOP_SHARE_POINTS: Final = 1.0

# Analytic KV bytes/token against the measured allocation delta during decode
# (plan §4.2, §10.4). `gpu` tier.
KV_BYTES_MEASURED_REL: Final = 0.05

# --- Manifest gates ----------------------------------------------------------

# Optimizer steps actually taken must agree within 0.5% across compared runs
# (plan §7.3, §13.6). `make_tables.py` refuses anything above this.
STEPS_TAKEN_PARITY_REL: Final = 0.005

# --- Statistics --------------------------------------------------------------

# Family-wise false-positive rate of the full §1.2 procedure on simulated nulls
# (plan §5).
FWER_MAX: Final = 0.05
# Monte-Carlo allowance for that check, Phil 2026-11-12: a fixed seed, this
# many simulated experiments, and a failure when the estimate exceeds FWER_MAX
# by more than this many standard errors of a binomial proportion at FWER_MAX.
FWER_N_SIMULATIONS: Final = 20_000
FWER_SIMULATION_SEED: Final = 20261112
FWER_STANDARD_ERRORS: Final = 3.0

# Coverage of the effective-parameter multiplier's interval on simulated ladders
# (plan §8.4; docs/decisions.md row 231, Phil 2026-11-13): a fixed seed, this
# many simulated campaigns per scenario, and a failure when the share of
# intervals that hold the true value falls below the interval's own confidence
# by more than this many standard errors of a binomial proportion at that
# confidence. The convention of the family-wise error check above.
MULTIPLIER_COVERAGE_N_SIMULATIONS: Final = 2_500
MULTIPLIER_COVERAGE_SIMULATION_SEED: Final = 20261113
MULTIPLIER_COVERAGE_STANDARD_ERRORS: Final = 3.0
# In the same check: the largest share of simulated campaigns whose ladder slope is not positive
# (they have no multiplier and are counted apart), and the band, in units of the width the
# interval would have without any slope uncertainty, that the median width must lie in. The
# second guards against an interval that covers because it is too wide.
MULTIPLIER_COVERAGE_MAX_NO_SLOPE_SHARE: Final = 0.01
MULTIPLIER_WIDTH_BAND: Final = (0.8, 3.0)

# The detectable-effect multipliers of plan §1.3 are reproduced within this
# absolute amount (TASKS.md T-24).
DETECTABLE_EFFECT_TABLE_ABS: Final = 0.05

# --- Exact arithmetic --------------------------------------------------------

# Quantities that are equal up to float64 rounding: schedule values, bucket
# weighting, interpolation on constructed inputs. Not in the plan.
FLOAT64_IDENTITY: Final = 1e-12
