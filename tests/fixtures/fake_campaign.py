"""A SYNTHETIC campaign of run manifests for tests. Nothing here comes from a real run.

`fake_campaign` builds a whole made-up experiment: MHA at 300M and its 100M
branches, MHA's standalone 100M reruns, the three primary variants, the three
exploratory variants and the two ladder points. Every validation loss is a
formula, not a measurement:

    loss = base - slope * ln(n_non_embedding / MHA_N) + gaps.get(variant, 0) + sigma * noise

The noise comes from `numpy.random.default_rng([seed, <variant index>, init_seed])`,
so the global RNG is never touched and each run has its own stream. A 100M
branch is its run's 300M loss plus a constant. A rerun is the 100M branch plus
`rerun_sigma` times its own noise.

Parts that can be left out with `without=(...)`:

- "mha": MHA at 300M, seeds 1..n_primary_seeds
- "branches": the 100M branches of MHA and the primaries
- "reruns": MHA's standalone 100M reruns, seeds 1-3
- "primaries": gqa4, mqa, mla192 at 300M
- "exploratory": gqa2, mla96, swa256 at 300M, seeds 1-3
- "ladder": mha_L6 at 300M, seeds 1-5
- "ladder_l10": mha_L10 at 300M, seeds 1-3
- "null": the null control, MHA under the name `null`, standalone 100M runs, seeds 11-15
- "data_order": MHA init seed 1 with data seeds 2 and 3, standalone 100M runs

With `ladder_branches=True` MHA and the ladder points run with 100M and 200M
branches (plan §7.2, §12.1) and the campaign holds those branch manifests too. A
200M branch is its run's 300M loss plus a constant, like a 100M branch. A null
run is a 100M MHA loss with its own noise stream plus `null_gap`; a data-order
run is MHA seed 1's 100M branch loss plus `data_sigma` times its own noise.

The returned manifests are unwritten dicts built by `fake_manifest`; write them
with `manifest.write_manifest` or edit a deep copy.
"""

from __future__ import annotations

import functools
import math
from typing import Iterable

import numpy as np

from src import config
from src.model import build_model
from tests.fixtures.fake_manifests import fake_manifest

PARTS = (
    "mha", "branches", "reruns", "primaries", "exploratory", "ladder", "ladder_l10", "null", "data_order",
)
PRIMARY_VARIANTS = ("gqa4", "mqa", "mla192")
EXPLORATORY_VARIANTS = ("gqa2", "mla96", "swa256")
BASE_LOSS = 3.2
BRANCH_OFFSET = 0.3  # 100M-branch loss minus 300M loss, a constant
BRANCH_OFFSET_200M = 0.1  # 200M-branch loss minus 300M loss, a constant
NULL_SEEDS = (11, 12, 13, 14, 15)  # plan §8.1
DATA_ORDER_SEEDS = (2, 3)  # plan §8.1: MHA init seed 1 under two other data orders
# Non-embedding parameter counts of the calibration ladder (plan §8.4), synthetic
# copies of the values build_model gives; the tests check they agree.
LADDER_PARAMS = {"mha_L6": 10_621_824, "mha": 14_162_304, "mha_L10": 17_702_784}
# A stable index per variant for the noise streams; new variants go at the end.
VARIANT_INDEX = {
    "mha": 0,
    "gqa4": 1,
    "mqa": 2,
    "mla192": 3,
    "gqa2": 4,
    "mla96": 5,
    "swa256": 6,
    "mha_L6": 7,
    "mha_L10": 8,
    "null": 9,
    "data_order": 10,
}


@functools.lru_cache(maxsize=None)
def _non_embedding(variant: str) -> int:
    """Non-embedding parameters of a 300M variant at its default configuration."""
    return build_model(config.resolve(variant, init_seed=1, peak_lr=1e-3)).n_non_embedding_params


def _loss(variant: str, n_params: int, init_seed: int, *, seed, base, slope, gaps, sigma) -> float:
    rng = np.random.default_rng([seed, VARIANT_INDEX[variant], init_seed])
    noise = float(rng.standard_normal())
    return (
        base
        - slope * math.log(n_params / LADDER_PARAMS["mha"])
        + gaps.get(variant, 0.0)
        + sigma * noise
    )


def fake_campaign(
    *,
    sigma: float = 0.004,
    slope: float = 0.4,
    gaps: dict[str, float] | None = None,
    n_primary_seeds: int = 5,
    seed: int = 1,
    rerun_sigma: float | None = None,
    without: Iterable[str] = (),
    null_gap: float = 0.0,
    data_sigma: float | None = None,
    ladder_branches: bool = False,
) -> list[dict]:
    """A synthetic campaign as a list of manifests. See the module docstring.

    `sigma`: SD of seed noise in nats. `slope`: the ladder's b (loss falls by
    slope per e-fold of parameters). `gaps`: loss added to a variant at 300M
    (its 100M branch carries it too). `n_primary_seeds`: 5 or 8 seeds for MHA
    and the primaries. `rerun_sigma` defaults to sigma / 10. `without`: parts to
    leave out, from PARTS. `null_gap`: loss added to every null-control run (0 for
    a true null). `data_sigma`: SD of data-order noise, default sigma.
    `ladder_branches`: MHA and the ladder points also branch at 200M.
    """
    unknown = set(without) - set(PARTS)
    if unknown:
        raise ValueError(f"unknown parts {sorted(unknown)}; expected some of {PARTS}")
    skip = set(without)
    gaps = dict(gaps or {})
    rerun_sigma = sigma / 10.0 if rerun_sigma is None else rerun_sigma
    data_sigma = sigma if data_sigma is None else data_sigma
    ladder_run_branches = ["100M", "200M"] if ladder_branches else None
    common = dict(seed=seed, base=BASE_LOSS, slope=slope, gaps=gaps, sigma=sigma)
    manifests: list[dict] = []
    seeds = range(1, n_primary_seeds + 1)

    def loss_at_300m(key: str, init_seed: int) -> float:
        # `key` is a variant, or a ladder name: each has its own noise stream.
        n_params = LADDER_PARAMS[key] if key in LADDER_PARAMS and key != "mha" else _non_embedding(key)
        return _loss(key, n_params, init_seed, **common)

    def run(key: str, init_seed: int, loss: float, **kwargs) -> dict:
        variant = "mha" if key.startswith("mha") else key
        ladder = key if key.startswith("mha_L") else None
        if key.startswith("mha") and ladder_run_branches is not None:
            kwargs.setdefault("branches", ladder_run_branches)
        return fake_manifest(variant, ladder=ladder, init_seed=init_seed, val_loss=loss, **kwargs)

    # The 300M runs of each part, and whether that part has 100M branches.
    main_parts = []
    if "mha" not in skip:
        main_parts.append(("mha", seeds))
    if "primaries" not in skip:
        main_parts.extend((variant, seeds) for variant in PRIMARY_VARIANTS)
    for key, key_seeds in main_parts:
        for s in key_seeds:
            manifests.append(run(key, s, loss_at_300m(key, s)))
            if "branches" not in skip:
                manifests.append(run(key, s, loss_at_300m(key, s) + BRANCH_OFFSET, branch="100M"))
                if key == "mha" and ladder_branches:
                    manifests.append(run(key, s, loss_at_300m(key, s) + BRANCH_OFFSET_200M, branch="200M"))
    if "reruns" not in skip:
        for s in range(1, 4):
            branch_loss = loss_at_300m("mha", s) + BRANCH_OFFSET
            rng = np.random.default_rng([seed, VARIANT_INDEX["mha"], s, 1])
            rerun = branch_loss + rerun_sigma * float(rng.standard_normal())
            manifests.append(
                fake_manifest("mha", budget="100M", branches=[], init_seed=s, val_loss=rerun)
            )
    if "exploratory" not in skip:
        for variant in EXPLORATORY_VARIANTS:
            for s in range(1, 4):
                manifests.append(run(variant, s, loss_at_300m(variant, s)))
    for part, key, key_seeds in (("ladder", "mha_L6", range(1, 6)), ("ladder_l10", "mha_L10", range(1, 4))):
        if part in skip:
            continue
        for s in key_seeds:
            manifests.append(run(key, s, loss_at_300m(key, s)))
            if ladder_branches and "branches" not in skip:
                manifests.append(run(key, s, loss_at_300m(key, s) + BRANCH_OFFSET, branch="100M"))
                manifests.append(run(key, s, loss_at_300m(key, s) + BRANCH_OFFSET_200M, branch="200M"))
    if "null" not in skip:
        for s in NULL_SEEDS:
            loss = _loss("mha", _non_embedding("mha"), s, **{**common, "gaps": {}}) + BRANCH_OFFSET + null_gap
            manifests.append(fake_manifest("null", budget="100M", branches=[], init_seed=s, val_loss=loss))
    if "data_order" not in skip:
        branch_loss = loss_at_300m("mha", 1) + BRANCH_OFFSET
        for d in DATA_ORDER_SEEDS:
            rng = np.random.default_rng([seed, VARIANT_INDEX["data_order"], d])
            loss = branch_loss + data_sigma * float(rng.standard_normal())
            manifests.append(
                fake_manifest("mha", budget="100M", branches=[], init_seed=1, data_seed=d, val_loss=loss)
            )
    return manifests
