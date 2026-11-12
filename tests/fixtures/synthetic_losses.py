"""SYNTHETIC losses for the statistics tests. Nothing here comes from a real run.

The noise model for one run of variant v with init seed k:

    loss = common[k] + specific[v, k] + rerun[v, k]

- `common` is shared by every variant with the same init seed. It is what a
  paired test cancels.
- `specific` belongs to one variant and seed.
- `rerun` changes every time the same configuration is run again. The
  reproducibility pairs measure it.

Each component exists at two budgets (the one used to choose the test and the
one the test is run on), correlated by `rho`. Under the null every variant has
the same mean.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class NoiseRegime(NamedTuple):
    name: str
    sigma_common: float
    sigma_specific: float
    sigma_rerun: float


# s_rep / sigma_init is about sigma_rerun / sqrt(sum of squares).
REGIMES = (
    NoiseRegime("pairing_strong", 1.0, 0.2, 0.1),  # ratio 0.10: paired almost always
    NoiseRegime("pairing_absent", 0.0, 0.3, 1.0),  # ratio 0.96: Welch mostly
    NoiseRegime("boundary_mixed", 0.7, 0.374, 0.6),  # ratio 0.60: the switch flips
    NoiseRegime("boundary_no_common", 0.0, 0.8, 0.6),  # ratio 0.60, nothing to cancel
    NoiseRegime("boundary_all_common", 0.8, 0.0, 0.6),  # ratio 0.60, pairing helps most
)


class SyntheticExperiments(NamedTuple):
    mha_select: np.ndarray  # (N, n_select) MHA losses at the selection budget
    repro_deltas: np.ndarray  # (N, n_pairs) differences within the reproducibility pairs
    repro_reruns: np.ndarray  # (N, n_pairs) the rerun losses themselves
    reference: np.ndarray  # (N, n) MHA losses at the test budget
    variants: np.ndarray  # (N, n_variants, n) variant losses at the test budget


def simulate_experiments(
    rng: np.random.Generator,
    *,
    n_experiments: int,
    n_seeds: int,
    n_variants: int,
    regime: NoiseRegime,
    rho: float,
    n_select: int,
    n_pairs: int,
    effects: tuple[float, ...] | None = None,
) -> SyntheticExperiments:
    """Draw `n_experiments` independent synthetic experiments.

    `effects[i]` is added to variant i's mean (default: none, the null).
    """
    if n_select > n_seeds or n_pairs > n_select:
        raise ValueError("selection seeds must be a prefix of the run seeds")

    def components(shape):
        a = rng.standard_normal(shape)
        b = rng.standard_normal(shape)
        return a, rho * a + np.sqrt(1.0 - rho**2) * b  # (test budget, selection budget)

    common_t, common_s = components((n_experiments, 1, n_seeds))
    specific_t, specific_s = components((n_experiments, n_variants + 1, n_seeds))
    rerun_t, rerun_s = components((n_experiments, n_variants + 1, n_seeds))
    sc, ss, sr = regime.sigma_common, regime.sigma_specific, regime.sigma_rerun
    test = sc * common_t + ss * specific_t + sr * rerun_t
    select = sc * common_s + ss * specific_s + sr * rerun_s
    if effects is not None:
        test[:, 1:, :] += np.asarray(effects, dtype=np.float64)[None, :, None]

    mha_select = select[:, 0, :n_select]
    # A rerun keeps the seed's common and specific parts and redraws the rest.
    stable = sc * common_s[:, 0, :n_pairs] + ss * specific_s[:, 0, :n_pairs]
    reruns = stable + sr * rng.standard_normal((n_experiments, n_pairs))
    deltas = mha_select[:, :n_pairs] - reruns
    return SyntheticExperiments(mha_select, deltas, reruns, test[:, 0, :], test[:, 1:, :])
