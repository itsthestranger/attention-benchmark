"""Phil's decisions of session 5 that change the rule (docs/decisions.md 216, 220, 223).

Written before the code (TASKS.md T-47, O+W). A worker implements against this file and may
not change it. Every number is made up.

216. The positive control (L6 against L8, plan §8.4) stays a two-sided test at its alpha and
     counts as detected only when the smaller model is WORSE. A significant gap the other way
     is its own verdict and a failed control. The null control and the primaries keep reading
     both directions.
223. A ladder slope that is not positive gives 8 seeds (plan §1.4 gives 8 as the slope falls
     toward zero; for a negative slope its formula read literally would give 5, and the code
     gave no decision at all). No multiplier is reported then.
220. `configs/commitments.yaml` holds the two declarations made during the campaign: the
     ladder (`full`, or `l6_l8` for plan §12.2 item 4) and the committed seed count. The
     analysis tests no primary and no exploratory comparison while the seed count is missing
     or disagrees with what the rule computes. The key replaces the flag `--l10-cut`.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from scripts import make_plots as mp
from scripts import make_tables as mt
from src import config
from src.utils import manifest as mf
from tests.fixtures import fake_plot_inputs as fpi
from tests.fixtures.commitments import committed, write_commitments
from tests.fixtures.fake_campaign import fake_campaign
from tests.fixtures.synthetic_losses import synthetic_ladder

ROOT = Path(__file__).resolve().parent.parent
PREREG = config.load_prereg()
BASE_SEEDS = tuple(PREREG["seed_count"]["base_seeds"])
EXTENDED_SEEDS = BASE_SEEDS + tuple(PREREG["seed_count"]["added_seeds"])
CONTROL_SEEDS = tuple(PREREG["positive_control"]["init_seeds"])
PRIMARIES = tuple(PREREG["primary"]["comparisons"])
EXPLORATORY = tuple(PREREG["exploratory"]["variants"])
CONTROL_SECTION = "Positive control: L6 against L8 at 300M"
SEED_SECTION = "Seed count and detectable effect"
DECLARATIONS = "Declarations (configs/commitments.yaml)"
FULL, CUT = "full", "l6_l8"
REAL_LOAD_COMMITMENTS = config.load_commitments  # the loader itself, before any test replaces it


def control_losses(shift: float, *, noise: float = 0.002):
    """SYNTHETIC: MHA losses per control seed, and the small model's, `shift` nats above them."""
    rng = np.random.default_rng([20261113, 216])
    mha = {s: 3.2 + noise * float(rng.standard_normal()) for s in CONTROL_SEEDS}
    small = {s: mha[s] + shift + noise * float(rng.standard_normal()) for s in CONTROL_SEEDS}
    return small, mha


# --- 216: the direction of the positive control ---------------------------------------------------


def test_the_wrong_direction_has_its_own_verdict_string():
    assert pt.VERDICT_WRONG_DIRECTION == "detected in the wrong direction"
    assert pt.VERDICT_WRONG_DIRECTION not in (pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED, pt.VERDICT_EXPLORATORY)


@pytest.mark.parametrize("test", pt.TESTS)
def test_the_positive_control_is_detected_only_when_the_small_model_is_worse(test):
    small, mha = control_losses(+0.05)
    worse = pt.positive_control(small, mha, test=test, prereg=PREREG)
    assert worse.mean_gap > 0 and worse.p <= PREREG["positive_control"]["alpha"]
    assert worse.verdict == pt.VERDICT_DETECTED

    small, mha = control_losses(-0.05)
    better = pt.positive_control(small, mha, test=test, prereg=PREREG)
    assert better.mean_gap < 0 and better.p <= PREREG["positive_control"]["alpha"]
    assert better.verdict == pt.VERDICT_WRONG_DIRECTION


@pytest.mark.parametrize("test", pt.TESTS)
def test_the_test_itself_stays_two_sided(test):
    """Only the reading of the verdict changes: p, t and the interval are the two-sided test's."""
    small, mha = control_losses(-0.05)
    control = pt.positive_control(small, mha, test=test, prereg=PREREG)
    plain = pt.run_test(test, [small[s] for s in CONTROL_SEEDS], [mha[s] for s in CONTROL_SEEDS])
    assert control.p == float(plain.p) and control.t == float(plain.t)
    assert (control.ci_low, control.ci_high) == (float(plain.ci_low), float(plain.ci_high))
    assert control.alpha == PREREG["positive_control"]["alpha"] and control.p_adjusted == control.p
    assert control.ruled_out_above is None  # only an undetected gap rules anything out


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("shift", [+0.0005, -0.0005])
def test_an_insignificant_gap_is_not_detected_in_either_direction(test, shift):
    small, mha = control_losses(shift, noise=0.01)
    control = pt.positive_control(small, mha, test=test, prereg=PREREG)
    assert control.p > PREREG["positive_control"]["alpha"]
    assert control.verdict == pt.VERDICT_NOT_DETECTED
    assert control.ruled_out_above == control.ci_high


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("shift", [+0.05, -0.05])
def test_the_null_control_still_reads_both_directions(test, shift):
    """A null that differs from MHA either way is a bug in the pipeline (plan §9.3)."""
    rng = np.random.default_rng([20261113, 217])
    mha_seeds = PREREG["test_selection"]["sigma_init"]["init_seeds"]
    mha = {s: 3.5 + 0.002 * float(rng.standard_normal()) for s in mha_seeds}
    null = {s: 3.5 + shift + 0.002 * float(rng.standard_normal()) for s in PREREG["null_control"]["init_seeds"]}
    assert pt.null_control(null, mha, test=test, prereg=PREREG).verdict == pt.VERDICT_DETECTED


@pytest.mark.parametrize("test", pt.TESTS)
def test_a_primary_better_than_mha_is_still_detected(test):
    rng = np.random.default_rng([20261113, 218])
    losses = {"mha": {s: 3.2 + 0.002 * float(rng.standard_normal()) for s in BASE_SEEDS}}
    for name in PRIMARIES:
        losses[name] = {s: losses["mha"][s] - 0.05 + 0.002 * float(rng.standard_normal()) for s in BASE_SEEDS}
    for comparison in pt.primary_tests(losses, test=test, n_seeds=len(BASE_SEEDS), prereg=PREREG):
        assert comparison.mean_gap < 0 and comparison.verdict == pt.VERDICT_DETECTED


@pytest.fixture(scope="module")
def upside_down() -> list[dict]:
    """SYNTHETIC: loss RISES with parameters, so L6 is better than L8 and the ladder slope is negative."""
    return fake_campaign(slope=-0.4, n_primary_seeds=8, gaps={"mqa": 0.05})


@pytest.fixture(scope="module")
def upside_down_analysis(upside_down) -> mt.Analysis:
    return mt.analyse(upside_down, commitments=committed(upside_down))


def test_the_analysis_reports_a_control_in_the_wrong_direction_as_failed(upside_down_analysis):
    a = upside_down_analysis
    assert a.positive_control.verdict == pt.VERDICT_WRONG_DIRECTION
    wrong = [note for note in a.notes if "positive control" in note and "wrong direction" in note]
    assert len(wrong) == 1
    assert "uninformative" in wrong[0]  # the consequence of a failed control (plan §8.4)
    # It is not also reported as merely undetected.
    assert not any(note.startswith("positive control not detected") for note in a.notes)
    # As for an undetected control (decision 170), the gate stays open: the note is the finding.
    assert a.comparisons_allowed


def test_the_gate_report_prints_the_wrong_direction_verbatim(upside_down_analysis):
    assert mt.gate_values(upside_down_analysis)[CONTROL_SECTION]["verdict"] == pt.VERDICT_WRONG_DIRECTION
    assert pt.VERDICT_WRONG_DIRECTION in mt.render_gates(upside_down_analysis)


def test_a_control_in_the_right_direction_has_no_such_note():
    campaign = fake_campaign(gaps={"mqa": 0.05})
    a = mt.analyse(campaign, commitments=committed(campaign))
    assert a.positive_control.verdict == pt.VERDICT_DETECTED
    assert not any("wrong direction" in note for note in a.notes)


# --- 223: a ladder slope that is not positive -----------------------------------------------------


def mha_base(sd: float = 0.01) -> dict[int, float]:
    return {s: 3.0 + sd * (s - 3) for s in BASE_SEEDS}


def fit_with_slope(b: float) -> lf.LadderFit:
    return dataclasses.replace(lf.fit_ladder(synthetic_ladder(None, a=3.0, b=0.2)), b=b)


@pytest.mark.parametrize("test", pt.TESTS)
@pytest.mark.parametrize("b", [-0.1, -1e-9, 0.0])
def test_a_slope_that_is_not_positive_gives_eight_seeds(test, b):
    d = lf.seed_count_rule(mha_base(), fit_with_slope(b), test=test, prereg=PREREG)
    assert d.slope_not_positive is True
    assert d.n_seeds == len(EXTENDED_SEEDS) == 8 and d.extended is True
    assert d.seeds == EXTENDED_SEEDS
    assert d.leaves_arm_a == tuple(PREREG["seed_count"]["leave_arm_a_if_extended"])
    assert d.slope == b and d.test == test
    # No parameter equivalent exists, so none is invented.
    assert d.param_multiplier is None and d.undetectable_param_fraction is None
    # Everything that does not need the slope is what it is for any other decision.
    assert d.sigma_init == lf.sigma_init_at_test_budget(mha_base(), PREREG)
    assert d.detectable_in_sigma_init == lf.detectable_multiplier(5, test, PREREG)
    assert d.detectable_gap == d.detectable_in_sigma_init * d.sigma_init
    assert d.bound_in_sigma_init == lf.detectable_multiplier(8, test, PREREG)
    assert d.bound_gap == d.bound_in_sigma_init * d.sigma_init
    assert d.threshold == PREREG["seed_count"]["max_undetectable_param_fraction"]


@pytest.mark.parametrize("test", pt.TESTS)
def test_a_positive_slope_is_decided_as_before(test):
    for b, sd in ((0.4, 0.001), (0.05, 0.02)):
        d = lf.seed_count_rule(mha_base(sd), fit_with_slope(b), test=test, prereg=PREREG)
        assert d.slope_not_positive is False
        gap = lf.detectable_multiplier(5, test, PREREG) * lf.sigma_init_at_test_budget(mha_base(sd), PREREG)
        assert d.param_multiplier == math.exp(-gap / b)
        assert d.undetectable_param_fraction == 1.0 - d.param_multiplier
        assert d.extended == (d.undetectable_param_fraction > d.threshold)
        assert d.n_seeds == (8 if d.extended else 5)
    # The two cases straddle the threshold, so the positive branch is exercised both ways.
    small = lf.seed_count_rule(mha_base(0.001), fit_with_slope(0.4), test=test, prereg=PREREG)
    large = lf.seed_count_rule(mha_base(0.02), fit_with_slope(0.05), test=test, prereg=PREREG)
    assert (small.n_seeds, large.n_seeds) == (5, 8)


@pytest.mark.parametrize("b", [float("nan"), float("inf"), float("-inf")])
def test_a_slope_that_is_not_a_number_is_still_an_error(b):
    with pytest.raises(lf.LadderError):
        lf.seed_count_rule(mha_base(), fit_with_slope(b), test=pt.WELCH, prereg=PREREG)


def test_the_analysis_runs_the_primaries_on_eight_seeds_without_multipliers(upside_down_analysis):
    a = upside_down_analysis
    assert a.rule_fit.b < 0
    assert a.decision.slope_not_positive and a.decision.n_seeds == 8
    assert a.committed_seed_count == 8
    # The primaries are tested on seeds 1 to 8; MLA-96 and SWA-256 have left Arm A.
    assert set(a.comparisons) == set(PRIMARIES) | {"gqa2"}
    for name in PRIMARIES:
        assert a.comparisons[name].seeds == EXTENDED_SEEDS
    assert a.multipliers == {}
    notes = [note for note in a.notes if "not positive" in note]
    assert len(notes) == 1 and "8 seeds" in notes[0] and "multiplier" in notes[0]
    # One reason, said once: no "multiplier <name>: ..." line per comparison.
    assert not any(note.startswith("multiplier ") for note in a.notes)


def test_the_gate_report_of_a_slope_that_is_not_positive(upside_down_analysis):
    values = mt.gate_values(upside_down_analysis)[SEED_SECTION]
    assert values["seeds per primary"] == 8
    assert values["ladder slope b at 300M"] == upside_down_analysis.rule_fit.b < 0
    assert values["undetectable parameter fraction at 5 seeds"] is None
    assert values["variants leaving Arm A"] == tuple(PREREG["seed_count"]["leave_arm_a_if_extended"])
    text = mt.render_gates(upside_down_analysis)
    row = next(line for line in text.splitlines() if line.startswith("| undetectable parameter fraction at 5 seeds"))
    assert row.split("|")[2].strip() == mt.BLANK


def test_with_a_positive_slope_the_multipliers_are_still_reported():
    campaign = fake_campaign(gaps={"mqa": 0.05})
    a = mt.analyse(campaign, commitments=committed(campaign))
    assert a.decision.slope_not_positive is False
    assert set(a.multipliers) == set(a.comparisons) == set(PRIMARIES) | set(EXPLORATORY)
    assert not any("not positive" in note for note in a.notes)


# --- 220: configs/commitments.yaml ----------------------------------------------------------------


def test_the_repositorys_file_is_a_valid_declaration_whatever_it_holds():
    """Phil fills this file in during the campaign (RUNBOOK §3.4), so no test pins what it holds
    (review S5, H1): it must load. What a campaign that has declared nothing holds is `uncommitted()`,
    and that is what the blank tables and plots are built with."""
    held = config.load_commitments()
    assert set(held) == {"ladder", "seed_count"}
    assert held["ladder"] in (FULL, CUT) and held["seed_count"] in (None, 5, 8)
    assert (config.CONFIG_DIR / "commitments.yaml").is_file()
    assert config.uncommitted() == {"ladder": FULL, "seed_count": None}
    assert config.uncommitted() is not config.uncommitted()  # a fresh mapping each time


def test_the_file_is_protected_like_the_frozen_files():
    rules = [line.split()[0] for line in (ROOT / ".github" / "CODEOWNERS").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    assert "/configs/commitments.yaml" in rules


@pytest.mark.parametrize("held", [
    {"ladder": FULL, "seed_count": None},
    {"ladder": FULL, "seed_count": 5},
    {"ladder": CUT, "seed_count": 8},
])
def test_a_valid_file_loads_as_it_is(tmp_path, held):
    assert config.load_commitments(write_commitments(tmp_path / "c.yaml", held)) == held


@pytest.mark.parametrize("held", [
    {"ladder": "L6_L8", "seed_count": None},  # not one of the two names
    {"ladder": None, "seed_count": None},
    {"ladder": True, "seed_count": None},
    {"ladder": FULL, "seed_count": 6},  # the rule gives 5 or 8 and nothing else
    {"ladder": FULL, "seed_count": 3},
    {"ladder": FULL, "seed_count": "5"},
    {"ladder": FULL, "seed_count": 5.0},
    {"ladder": FULL, "seed_count": True},
    {"ladder": FULL},  # a missing key is not "nothing declared"
    {"seed_count": 5},
    {"ladder": FULL, "seed_count": 5, "l10_cut": True},  # an unknown key is a typo
])
def test_a_malformed_file_is_refused(tmp_path, held):
    with pytest.raises(config.ConfigError):
        config.load_commitments(write_commitments(tmp_path / "c.yaml", held))


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises((config.ConfigError, OSError)):
        config.load_commitments(tmp_path / "absent.yaml")


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    return fake_campaign(gaps={"mqa": 0.05})


def test_without_a_committed_seed_count_no_comparison_is_computed(campaign):
    a = mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": None})
    # The rule is still computed and printed: it is what Phil commits.
    assert a.decision is not None and a.decision.n_seeds == 5
    assert a.committed_seed_count is None and a.ladder == FULL
    assert a.comparisons == {} and a.multipliers == {}
    assert all(use.late_context is None and use.icl_score is None for use in a.context_use.values())
    notes = [note for note in a.notes if "not committed" in note]
    assert len(notes) == 1
    assert "configs/commitments.yaml" in notes[0] and "5" in notes[0]
    # The controls and the noise floor do not wait for it.
    assert a.null_control is not None and a.positive_control is not None and a.comparisons_allowed


def test_with_the_rules_seed_count_committed_every_comparison_is_computed(campaign):
    a = mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": 5})
    assert a.committed_seed_count == 5
    assert set(a.comparisons) == set(PRIMARIES) | set(EXPLORATORY)
    assert set(a.multipliers) == set(a.comparisons)
    assert not any("not committed" in note or "disagrees" in note for note in a.notes)


def test_a_committed_seed_count_that_disagrees_with_the_rule_computes_no_comparison(campaign):
    a = mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": 8})
    assert a.decision.n_seeds == 5 and a.committed_seed_count == 8
    assert a.comparisons == {} and a.multipliers == {}
    notes = [note for note in a.notes if "disagrees" in note]
    assert len(notes) == 1 and "8" in notes[0] and "5" in notes[0]
    assert not any("not committed" in note for note in a.notes)


def test_a_disagreement_is_caught_in_the_other_direction_too():
    eight = fake_campaign(sigma=0.01, n_primary_seeds=8, gaps={"mqa": 0.05})
    assert committed(eight)["seed_count"] == 8, "this synthetic campaign is meant to ask for 8 seeds"
    a = mt.analyse(eight, commitments={"ladder": FULL, "seed_count": 5})
    assert a.comparisons == {} and any("disagrees" in note for note in a.notes)
    assert set(mt.analyse(eight, commitments=committed(eight)).comparisons) == set(PRIMARIES) | {"gqa2"}


def test_the_headline_table_has_no_verdict_without_the_commitment(campaign):
    uncommitted = mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": None})
    done = mt.analyse(campaign, commitments=committed(campaign))

    def verdicts(a):
        """The verdict cells of the variant rows; MHA's own cell says "reference"."""
        return [row["verdict"] for row in mt.headline_rows(a)][1:]

    assert mt.BLANK not in verdicts(done) and pt.VERDICT_DETECTED in verdicts(done)
    assert set(verdicts(uncommitted)) == {mt.BLANK}


@pytest.fixture(scope="module")
def eight_seed_campaign() -> list[dict]:
    campaign_ = fpi.with_eval_fields(fake_campaign(sigma=0.01, n_primary_seeds=8, gaps={"mqa": 0.05}))
    assert committed(campaign_)["seed_count"] == 8, "this synthetic campaign is meant to ask for 8 seeds"
    return campaign_


def test_an_uncommitted_eight_seed_rule_does_not_show_through_the_tables(eight_seed_campaign):
    """Break 13 and 14 of the T-47 review: "left Arm A" is a consequence of the seed count, so the
    headline and the context table say it only once the count is committed."""
    done = mt.analyse(eight_seed_campaign, commitments=committed(eight_seed_campaign))
    uncommitted = mt.analyse(eight_seed_campaign, commitments={"ladder": FULL, "seed_count": None})
    wrong = mt.analyse(eight_seed_campaign, commitments={"ladder": FULL, "seed_count": 5})
    assert uncommitted.decision == done.decision and done.decision.extended

    def headline_verdicts_of(a):
        return [row["verdict"] for row in mt.headline_rows(a)]

    assert headline_verdicts_of(done).count(mt.LEFT_ARM) == len(done.decision.leaves_arm_a) == 2
    assert mt.LEFT_ARM in mt.render_context(done)
    for a in (uncommitted, wrong):
        assert mt.LEFT_ARM not in headline_verdicts_of(a)
        assert mt.LEFT_ARM not in mt.render_context(a)


def test_context_use_waits_for_the_commitment():
    """Break 15 of the T-47 review, on runs that carry context values."""
    campaign_ = fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}))
    done = mt.analyse(campaign_, commitments=committed(campaign_))
    assert all(use.late_context is not None and use.icl_score is not None for use in done.context_use.values())
    for seed_count in (None, 8):
        a = mt.analyse(campaign_, commitments={"ladder": FULL, "seed_count": seed_count})
        # As without any decision: no context-use gap against MHA is computed.
        assert all(use.late_context is None and use.icl_score is None for use in a.context_use.values())
        assert a.context_use == mt.analyse(
            [m for m in campaign_ if "mha_L" not in m["name"]], commitments={"ladder": FULL, "seed_count": None}
        ).context_use


def test_a_seed_count_committed_before_the_rule_has_a_decision_computes_nothing():
    early = fake_campaign(gaps={"mqa": 0.05}, without=("ladder",))
    a = mt.analyse(early, commitments={"ladder": FULL, "seed_count": 5})
    assert a.decision is None and a.comparisons == {}


def without_l10(campaign_):
    return [x for x in campaign_ if mf.parse_run_name(x["name"]).variant != "mha_L10"]


def test_the_ladder_key_is_the_declaration_of_the_l10_cut(campaign):
    no_l10 = without_l10(campaign)
    assert len(no_l10) < len(campaign)
    waiting = mt.analyse(no_l10, commitments={"ladder": FULL, "seed_count": None})
    declared = mt.analyse(no_l10, commitments={"ladder": CUT, "seed_count": None})
    assert waiting.decision is None  # absent L10 runs are not a decision to cut them (review S4, H2)
    assert declared.decision is not None and declared.ladder == CUT
    assert set(p.name for p in declared.rule_fit.points) == {"mha", "mha_L6"}
    assert any("mha_L10" in note and "declared cut" in note for note in declared.notes)
    # Declared with the L10 runs present, the rule gives what it gives without them.
    present = mt.analyse(campaign, commitments={"ladder": CUT, "seed_count": None})
    assert present.decision == declared.decision


def test_the_flag_is_gone_from_the_analysis():
    with pytest.raises(TypeError):
        mt.analyse([], l10_cut=True)
    with pytest.raises(TypeError):
        mp.make_inputs([], l10_cut=True)


def test_without_an_argument_the_analysis_reads_the_committed_file(campaign, monkeypatch):
    calls = []

    def fake_load(path=None):
        calls.append(path)
        return {"ladder": FULL, "seed_count": 5}

    monkeypatch.setattr(config, "load_commitments", fake_load)
    a = mt.analyse(campaign)
    assert calls == [None]
    assert a.committed_seed_count == 5 and set(a.comparisons) == set(PRIMARIES) | set(EXPLORATORY)
    # With a file that declares no seed count, nothing is compared.
    monkeypatch.setattr(config, "load_commitments", lambda path=None: config.uncommitted())
    assert mt.analyse(campaign).comparisons == {}


def test_the_rules_inputs_and_output_are_shown_before_anything_is_committed(campaign):
    """Break B23 of review S5: Phil commits the seed count from this section, so every row of it is
    filled as soon as the rule can be computed, committed or not."""
    uncommitted = mt.gate_values(mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": None}))
    done = mt.gate_values(mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": 5}))
    assert all(value is not None for value in uncommitted[SEED_SECTION].values())
    assert uncommitted[SEED_SECTION] == done[SEED_SECTION]


def test_the_gate_report_shows_the_declarations_beside_the_rule(campaign):
    uncommitted = mt.gate_values(mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": None}))
    assert uncommitted[DECLARATIONS] == {"ladder": FULL, "committed seed count": None}
    assert uncommitted[SEED_SECTION]["seeds per primary"] == 5  # what the rule gives, to be committed
    done = mt.analyse(campaign, commitments={"ladder": FULL, "seed_count": 5})
    assert mt.gate_values(done)[DECLARATIONS] == {"ladder": FULL, "committed seed count": 5}
    text = mt.render_gates(done)
    assert f"## {DECLARATIONS}" in text or DECLARATIONS in text
    row = next(line for line in text.splitlines() if line.startswith("| committed seed count"))
    assert row.split("|")[2].strip() == "5"


def test_the_blank_gate_report_is_the_committed_one(tmp_path, capsys):
    assert mt.main(["--blank", "--out", str(tmp_path)]) == 0
    capsys.readouterr()
    blank = ROOT / "results" / "tables" / "blank" / "gates.md"
    assert (tmp_path / "gates.md").read_text(encoding="utf-8") == blank.read_text(encoding="utf-8")
    assert DECLARATIONS in blank.read_text(encoding="utf-8")


@pytest.mark.parametrize("held", [{"ladder": FULL, "seed_count": 5}, {"ladder": CUT, "seed_count": 8}])
def test_the_blank_tables_and_plots_do_not_depend_on_what_the_file_holds(tmp_path, monkeypatch, capsys, held):
    """Review S5, H1: after Phil's commit of §3.4 the committed blank forms must still be the generated ones."""
    assert mt.main(["--blank", "--out", str(tmp_path / "before")]) == 0
    monkeypatch.setattr(config, "load_commitments", lambda path=None: dict(held))
    assert mt.main(["--blank", "--out", str(tmp_path / "after")]) == 0
    before = sorted(p.name for p in (tmp_path / "before").iterdir())
    assert before and before == sorted(p.name for p in (tmp_path / "after").iterdir())
    for name in before:
        assert (tmp_path / "after" / name).read_bytes() == (tmp_path / "before" / name).read_bytes(), name
        committed_blank = ROOT / "results" / "tables" / "blank" / name
        assert (tmp_path / "after" / name).read_bytes() == committed_blank.read_bytes(), name
    seen = []
    real = mt.analyse

    def spy(results, prereg=None, **kwargs):
        seen.append(kwargs.get("commitments"))
        return real(results, prereg, **kwargs)

    monkeypatch.setattr(mt, "analyse", spy)
    assert mp.main(["--blank", "--out", str(tmp_path / "plots")]) == 0
    capsys.readouterr()
    assert seen == [config.uncommitted()]


# --- 220: the command lines ------------------------------------------------------------------------


@pytest.fixture()
def manifest_dir(campaign, tmp_path):
    directory = tmp_path / "manifests"
    for manifest in campaign:
        mf.write_manifest(manifest, directory)
    return directory


def headline_verdicts(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [word for word in (pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED) if f"| {word} |" in text or f" {word} " in text]


def file_holds(monkeypatch, tmp_path, held) -> None:
    """Make `config.load_commitments()` read a file that holds `held`, through the real loader and its checks."""
    path = write_commitments(tmp_path / f"held_{len(list(tmp_path.glob('held_*')))}.yaml", held)
    monkeypatch.setattr(config, "load_commitments", lambda _=None: REAL_LOAD_COMMITMENTS(path))


def test_make_tables_reads_the_repositorys_commitments_file(manifest_dir, tmp_path, monkeypatch, capsys):
    common = ["--manifests", str(manifest_dir)]
    for name, held in (
        ("none", {"ladder": FULL, "seed_count": None}),
        ("done", {"ladder": FULL, "seed_count": 5}),
        ("wrong", {"ladder": FULL, "seed_count": 8}),
    ):
        file_holds(monkeypatch, tmp_path, held)
        assert mt.main([*common, "--out", str(tmp_path / name)]) == 0
    capsys.readouterr()
    assert headline_verdicts(tmp_path / "done" / "headline.md")
    assert not headline_verdicts(tmp_path / "none" / "headline.md")
    assert not headline_verdicts(tmp_path / "wrong" / "headline.md")
    assert "not committed" in (tmp_path / "none" / "headline.md").read_text(encoding="utf-8")
    assert "disagrees" in (tmp_path / "wrong" / "headline.md").read_text(encoding="utf-8")


def test_a_mapping_given_in_python_takes_the_place_of_the_file(manifest_dir, tmp_path, monkeypatch, capsys):
    """`main(argv, commitments=...)` is how the tests of other files declare; it does not read the file."""
    def must_not_be_read(path=None):
        raise AssertionError("the repository's file was read although a mapping was given")

    monkeypatch.setattr(config, "load_commitments", must_not_be_read)
    done = {"ladder": FULL, "seed_count": 5}
    assert mt.main(["--manifests", str(manifest_dir), "--out", str(tmp_path / "t")], commitments=done) == 0
    none = tmp_path / "none"
    assert mp.main(["--manifests", str(manifest_dir), "--bench", str(none), "--probes", str(none / "p.json"),
                    "--retrieval", str(none / "r.json"), "--out", str(tmp_path / "p")], commitments=done) == 0
    capsys.readouterr()
    assert headline_verdicts(tmp_path / "t" / "headline.md")


def test_make_tables_refuses_a_malformed_commitments_file(manifest_dir, tmp_path, monkeypatch, capsys):
    file_holds(monkeypatch, tmp_path, {"ladder": FULL, "seed_count": 6})
    assert mt.main(["--manifests", str(manifest_dir), "--out", str(tmp_path / "out")]) == 1
    assert "seed_count" in capsys.readouterr().err
    assert not (tmp_path / "out" / "headline.md").exists()


def test_make_plots_refuses_a_malformed_commitments_file(manifest_dir, tmp_path, monkeypatch, capsys):
    file_holds(monkeypatch, tmp_path, {"ladder": "cut", "seed_count": None})
    none = tmp_path / "none"
    code = mp.main(["--manifests", str(manifest_dir), "--bench", str(none), "--probes", str(none / "p.json"),
                    "--retrieval", str(none / "r.json"), "--out", str(tmp_path / "out")])
    assert code == 1
    assert "ladder" in capsys.readouterr().err


@pytest.mark.parametrize("module", [mt, mp])
@pytest.mark.parametrize("flag", [["--l10-cut"], ["--commitments", "some.yaml"], ["--seed-count", "5"]])
def test_no_command_line_flag_declares_anything(module, flag, tmp_path, capsys):
    """Decision 220; review S5, G1: a flag can be set after the results exist, and so can a flag that
    names a file. The declarations are what the repository's committed file holds."""
    with pytest.raises(SystemExit) as exit_:
        module.main(["--blank", "--out", str(tmp_path), *flag])
    assert exit_.value.code == 2
    capsys.readouterr()
