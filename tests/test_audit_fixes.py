"""Fixes for the handover audit of plan §17.4 (session 3). Written before the fixes (TASKS.md T-37, O+W).

A worker implements against this file and may not change it. Every number is made up.

F1. A run that skipped more than 0.5% of its steps made every table and the sweep
    report refuse, even when no comparison reads that run (plan §13.6: the rule
    invalidates a COMPARISON). The sweep is meant to list such a run as diverged.
F2. The seed-count rule (plan §1.4: "applied once ... from MHA runs only") was fed a
    ladder fit over every MHA seed present, so adding the seeds the rule asked for
    could reverse its own decision.
F5. With the comparison gate closed, the headline table still printed the variants'
    losses beside MHA's (plan §8.1: "No cross-variant number is computed or viewed").
F6. Nothing refused a sweep on a seed that an arm uses (plan §7.4, §8.1): the sweep
    run at the winning rate is then also a reproducibility rerun.
"""

from __future__ import annotations

import pytest

from analysis import ladder_fit as lf
from analysis import lr_sweep as sw
from scripts import make_tables as mt
from scripts import select_lr
from src import config
from src.utils import manifest as mf
from tests.fixtures.fake_campaign import LADDER_PARAMS, fake_campaign
from tests.fixtures.fake_manifests import fake_manifest
from tests.test_gate_report import losses_of
from tests.test_lr_sweep import with_skips
from tests.fixtures.commitments import analyse_committed

PREREG = config.load_prereg()
BASE_SEEDS = tuple(PREREG["seed_count"]["base_seeds"])
SWEEP_SEED = 21
GRID = sw.MHA_GRID
# A synthetic campaign that sits on the seed-count threshold: found by the audit.
THRESHOLD_CASE = dict(seed=3, slope=0.3748)


def sweep_run(lr, loss=3.5, variant="mha", seed=SWEEP_SEED):
    return fake_manifest(variant, budget="100M", branches=[], init_seed=seed, peak_lr=lr, val_loss=loss)


def mha_sweep(skipped_at_top: int) -> list[dict]:
    """MHA's five sweep points at 100M; the highest rate skipped `skipped_at_top` steps."""
    runs = [sweep_run(lr, 3.60 - 0.02 * i) for i, lr in enumerate(GRID[:-1])]
    runs.append(with_skips(sweep_run(GRID[-1], 3.40), skipped_at_top))
    return runs


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    return fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True)


@pytest.fixture(scope="module")
def analysis(campaign) -> mt.Analysis:
    return analyse_committed(campaign)


# --- F1: step parity belongs to the runs a comparison reads ------------------------------------


def test_a_skippy_sweep_run_does_not_stop_the_analysis(campaign, analysis):
    with_sweep = analyse_committed(campaign + mha_sweep(skipped_at_top=8))
    assert with_sweep.comparisons == analysis.comparisons
    assert with_sweep.decision == analysis.decision
    assert with_sweep.null_control == analysis.null_control


def test_the_sweep_report_lists_a_skippy_run_as_diverged(campaign, tmp_path):
    directory = tmp_path / "manifests"
    for manifest in campaign + mha_sweep(skipped_at_top=8):
        mf.write_manifest(manifest, directory)
    out = tmp_path / "tables"
    assert select_lr.main(["--manifests", str(directory), "--sweep-seed", str(SWEEP_SEED), "--out", str(out)]) == 0
    text = (out / "lr_sweep.md").read_text(encoding="utf-8")
    top = mf.format_lr(GRID[-1])
    line = next(l for l in text.splitlines() if l.startswith(f"| {top} |"))
    assert "DIVERGED" in line and "skipped" in line
    # The diverged top point bounds the grid, so the best finite point, one below it, is chosen.
    points = sw.points_from_manifests(mha_sweep(skipped_at_top=8), "mha", sweep_seed=SWEEP_SEED)
    choice = sw.choose_reference(points)
    assert choice.status == sw.CHOSEN and choice.lr == GRID[-2]
    assert f"**Rate to use:** {mf.format_lr(choice.lr)}" in text


def _replace(campaign, predicate, change):
    out, hit = [], 0
    for m in campaign:
        if predicate(mf.parse_run_name(m["name"])):
            out.append(change(m))
            hit += 1
        else:
            out.append(m)
    assert hit == 1, hit
    return out


@pytest.mark.parametrize(
    "what, predicate",
    [
        ("an MHA 100M branch", lambda p: (p.variant, p.init_seed, p.branch) == ("mha", 2, "100M")),
        ("a primary's 300M run", lambda p: (p.variant, p.init_seed, p.branch, p.tokens) == ("mqa", 3, None, "300M")),
        ("a null-control run", lambda p: (p.variant, p.init_seed) == ("null", 12)),
        ("a reproducibility rerun", lambda p: (p.variant, p.init_seed, p.branch, p.tokens, p.data_seed) == ("mha", 1, None, "100M", 1)),
        ("a data-order run", lambda p: (p.variant, p.data_seed) == ("mha", 2)),
        ("a ladder run", lambda p: (p.variant, p.init_seed, p.branch, p.tokens) == ("mha_L6", 4, None, "300M")),
    ],
)
def test_a_skippy_run_that_the_analysis_reads_is_still_refused(campaign, what, predicate):
    total = {"100M": 1526, "300M": 4578}
    def skippy(m):
        return with_skips(m, 8 if m["steps"]["total"] == total["100M"] else 24)
    with pytest.raises(mt.TableError, match="steps taken"):
        analyse_committed(_replace(campaign, predicate, skippy))


def test_versions_and_backend_are_still_compared_over_every_run(campaign):
    odd = fake_manifest("mha", budget="100M", branches=[], init_seed=SWEEP_SEED, peak_lr=4.8e-3, val_loss=3.4)
    odd["environment"] = {**odd["environment"], "torch": "9.9.9"}
    with pytest.raises(mt.TableError, match="torch"):
        analyse_committed(campaign + [odd])


# --- F2: the seed-count rule reads the registered base seeds only -------------------------------


def test_the_seed_count_does_not_change_when_the_seeds_it_asked_for_arrive():
    five = analyse_committed(fake_campaign(n_primary_seeds=5, **THRESHOLD_CASE))
    eight = analyse_committed(fake_campaign(n_primary_seeds=8, **THRESHOLD_CASE))
    assert five.decision.n_seeds == 8, "this synthetic campaign is meant to ask for 8 seeds"
    assert eight.decision == five.decision
    assert set(eight.comparisons) == set(PREREG["primary"]["comparisons"]) | {"gqa2"}
    assert eight.comparisons["mqa"].seeds == tuple(range(1, 9))


def test_the_rule_is_fed_the_ladder_fit_of_the_registered_seeds():
    c = fake_campaign(n_primary_seeds=8, **THRESHOLD_CASE)
    a = analyse_committed(c)
    points = {
        "mha": (LADDER_PARAMS["mha"], {s: v for s, v in losses_of(c, variant="mha", tokens="300M", branch=None).items() if s in BASE_SEEDS}),
        "mha_L6": (LADDER_PARAMS["mha_L6"], losses_of(c, variant="mha_L6", tokens="300M", branch=None)),
        "mha_L10": (LADDER_PARAMS["mha_L10"], losses_of(c, variant="mha_L10", tokens="300M", branch=None)),
    }
    rule_fit = lf.fit_ladder(points)
    assert a.rule_fit == rule_fit
    assert a.decision.slope == rule_fit.b
    # The reported fit and the multipliers keep using every registered MHA run, as before.
    assert len(a.fit.point("mha").seeds) == 8 and len(a.rule_fit.point("mha").seeds) == len(BASE_SEEDS)
    assert a.fit.n_runs == rule_fit.n_runs + 3
    assert mt.gate_values(a)["Seed count and detectable effect"]["ladder slope b at 300M"] == rule_fit.b


def test_with_five_seeds_the_two_fits_are_the_same(analysis):
    assert analysis.rule_fit == analysis.fit


def test_the_blank_analysis_has_no_rule_fit():
    assert analyse_committed([]).rule_fit is None


# --- F5: nothing about a variant is shown while the gate is closed -------------------------------


def _rows(analysis):
    return {row["variant"]: row for row in mt.headline_rows(analysis)}


@pytest.mark.parametrize("closed_by", ["no null control", "a detected null control"])
def test_with_the_gate_closed_the_headline_shows_only_mha(closed_by):
    kwargs = {"without": ("null",)} if closed_by == "no null control" else {"null_gap": 0.5}
    a = analyse_committed(fake_campaign(gaps={"mqa": 0.05}, **kwargs))
    assert a.comparisons_allowed is False
    rows = _rows(a)
    assert rows["mha"]["val loss, every seed"] != mt.BLANK
    for variant in mt.HEADLINE_VARIANTS:
        if variant == "mha":
            continue
        for column in ("val loss, every seed", "gap vs MHA [95% CI]", "effective-param multiplier", "verdict"):
            assert rows[variant][column] == mt.BLANK, (variant, column)
        assert rows[variant]["params"] != mt.BLANK  # the static columns stay
    assert any("withheld" in note and "controls" in note for note in a.notes)


def test_with_the_gate_open_the_headline_shows_every_variant(analysis):
    assert analysis.comparisons_allowed is True
    rows = _rows(analysis)
    for variant in mt.HEADLINE_VARIANTS:
        assert rows[variant]["val loss, every seed"] != mt.BLANK, variant
    assert not any("withheld" in note for note in analysis.notes)


# --- F6: a sweep seed that an arm uses is refused ------------------------------------------------


REGISTERED = sorted(
    set(PREREG["seed_count"]["base_seeds"]) | set(PREREG["seed_count"]["added_seeds"])
    | set(PREREG["null_control"]["init_seeds"]) | set(PREREG["exploratory"]["init_seeds"])
    | set(PREREG["positive_control"]["init_seeds"])
)


@pytest.mark.parametrize("seed", REGISTERED)
def test_a_registered_seed_is_refused_as_the_sweep_seed(seed):
    with pytest.raises(sw.SweepError, match="sweep seed"):
        sw.check_sweep_seed(seed, PREREG)


@pytest.mark.parametrize("seed", [0, 9, 10, 16, 21, 100])
def test_a_seed_no_arm_uses_is_accepted(seed):
    assert sw.check_sweep_seed(seed, PREREG) == seed


def test_the_sweep_report_refuses_a_registered_seed(campaign, tmp_path, capsys):
    directory = tmp_path / "manifests"
    for manifest in campaign:
        mf.write_manifest(manifest, directory)
    out = tmp_path / "tables"
    assert select_lr.main(["--manifests", str(directory), "--sweep-seed", "1", "--out", str(out)]) != 0
    assert "sweep seed" in capsys.readouterr().err
    assert not (out / "lr_sweep.md").exists()
