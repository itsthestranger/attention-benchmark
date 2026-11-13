"""The gate report: controls, noise components and per-budget ladder fits (plan §8.1, §8.4, §9.3, §9.4).

Written by the orchestrator before the implementation (TASKS.md T-32, O+W). A
worker implements against this file and may not change it.

What is specified here
----------------------
`scripts.make_tables.analyse` also computes, in this order and from the runs
named in the plan:

- `null_100m`: {init seed: loss} of the null control's standalone 100M runs at
  MHA's learning rate.
- `data_order_100m`: {data seed: loss} of MHA init seed 1 at 100M: data seed 1
  is the Arm A run's 100M branch, the others are standalone runs.
- `sigma_data`, `sigma_init_null`: sample SDs of those two sets, when complete.
- `null_control`: `primary_test.null_control` against MHA's 100M BRANCHES.
- `positive_control`: `primary_test.positive_control`, L6 against L8 at 300M.
- `losses_200m` and `fits`: the ladder fit at 100M, 200M and 300M.

and it refuses to compute any cross-variant comparison until the null control
exists and is not detected, and the positive control exists (plan §8.1: "No
cross-variant number is computed or viewed until this exists"; §9.3: "nothing
downstream is valid until it passes"; §9.4).

`gate_values` and `render_gates` turn that into the report Phil reads at Gates
2 and 3. Every verdict in it is the string `primary_test` wrote.

All data here are synthetic (tests/fixtures/fake_campaign.py).
"""

from __future__ import annotations

import dataclasses
import statistics
from pathlib import Path

import pytest

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from scripts import make_tables as mt
from src import config
from src.utils import manifest as mf
from tests.fixtures.fake_campaign import (
    DATA_ORDER_SEEDS,
    LADDER_PARAMS,
    NULL_SEEDS,
    fake_campaign,
)
from tests.fixtures.fake_manifests import fake_manifest
from tests.tolerances import FLOAT64_IDENTITY, scale_relative_error
from tests.fixtures.commitments import analyse_committed, tables_main

ROOT = Path(__file__).resolve().parents[1]
BLANK_GATES = ROOT / "results" / "tables" / "blank" / "gates.md"
DASH = "—"

# Section headings and row labels of the report, in order. They are the contract
# between the report, RUNBOOK.md and the plots.
EXPECTED_SECTIONS = {
    "Noise floor at 100M (Gate 2)": (
        "sigma_init, MHA seeds 1-5",
        "sigma_init, null-control seeds",
        "sigma_data, MHA seed 1 over data seeds 1-3",
        "s_rep",
        "s_rep / sigma_init",
        "pairing threshold",
        "test chosen",
    ),
    "Null control (Gate 2)": (
        "null seeds",
        "MHA seeds",
        "mean gap, nats",
        "95% interval",
        "p",
        "verdict",
    ),
    "Positive control: L6 against L8 at 300M": (
        "seeds",
        "mean gap, nats",
        "95% interval",
        "p",
        "verdict",
    ),
    "Seed count and detectable effect": (
        "sigma_init at 300M, MHA seeds 1-5",
        "ladder slope b at 300M",
        "5-seed detectable gap, nats",
        "undetectable parameter fraction at 5 seeds",
        "threshold",
        "seeds per primary",
        "variants leaving Arm A",
        "detectable gap of the chosen design, nats",
    ),
    # Decision 220: what configs/commitments.yaml declares, beside what the rule computes.
    "Declarations (configs/commitments.yaml)": ("ladder", "committed seed count"),
    "Ladder slope b by budget": ("100M", "200M", "300M"),
}

DECLARATIONS = "Declarations (configs/commitments.yaml)"

PREREG = config.load_prereg()
SELECTION_SEEDS = tuple(PREREG["test_selection"]["sigma_init"]["init_seeds"])
CONTROL_SEEDS = tuple(PREREG["positive_control"]["init_seeds"])


# --- Helpers: read the synthetic campaign independently of make_tables ---------------------


def _parsed(manifest: dict) -> mf.RunName:
    return mf.parse_run_name(manifest["name"])


def _loss(manifest: dict) -> float:
    return float(manifest["final_metrics"]["val_loss"])


def losses_of(manifests, *, variant: str, tokens: str, branch: str | None, data_seed: int = 1) -> dict[int, float]:
    """{init seed: loss} of the manifests with exactly this variant, token label, branch and data seed."""
    out = {}
    for m in manifests:
        p = _parsed(m)
        if (p.variant, p.tokens, p.branch, p.data_seed) == (variant, tokens, branch, data_seed):
            assert p.init_seed not in out, m["name"]
            out[p.init_seed] = _loss(m)
    return out


def without_manifest(manifests, predicate) -> list[dict]:
    kept = [m for m in manifests if not predicate(_parsed(m))]
    assert len(kept) < len(manifests), "the predicate matched nothing"
    return kept


def gate_two_stage(manifests) -> list[dict]:
    """What exists at Gate 2: no 300M result of anything, only 100M branches and standalone 100M runs."""
    return without_manifest(manifests, lambda p: p.tokens == "300M" and p.branch is None)


# --- Module fixtures ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    return fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True)


@pytest.fixture(scope="module")
def analysis(campaign) -> mt.Analysis:
    return analyse_committed(campaign)


@pytest.fixture(scope="module")
def blank() -> mt.Analysis:
    return analyse_committed([])


@pytest.fixture(scope="module")
def mha_branches(campaign) -> dict[int, float]:
    return losses_of(campaign, variant="mha", tokens="300M", branch="100M")


@pytest.fixture(scope="module")
def mha_300(campaign) -> dict[int, float]:
    return losses_of(campaign, variant="mha", tokens="300M", branch=None)


# --- 1. What is collected ----------------------------------------------------------------------


def test_the_fixture_has_the_runs_this_file_needs(campaign):
    assert set(losses_of(campaign, variant="null", tokens="100M", branch=None)) == set(NULL_SEEDS)
    for d in DATA_ORDER_SEEDS:
        assert set(losses_of(campaign, variant="mha", tokens="100M", branch=None, data_seed=d)) == {1}
    for name in ("mha", "mha_L6", "mha_L10"):
        assert losses_of(campaign, variant=name, tokens="300M", branch="200M")


def test_null_runs_are_collected_by_seed(campaign, analysis):
    assert analysis.null_100m == losses_of(campaign, variant="null", tokens="100M", branch=None)


def test_data_order_runs_are_seed_one_under_each_data_seed(campaign, analysis, mha_branches):
    expected = {1: mha_branches[1]}
    for d in DATA_ORDER_SEEDS:
        expected[d] = losses_of(campaign, variant="mha", tokens="100M", branch=None, data_seed=d)[1]
    assert analysis.data_order_100m == expected


def test_a_data_order_run_of_another_init_seed_is_not_read(campaign, analysis):
    stray = fake_manifest("mha", budget="100M", branches=[], init_seed=2, data_seed=2, val_loss=9.0)
    assert analyse_committed(campaign + [stray]).data_order_100m == analysis.data_order_100m


def test_a_null_run_at_another_learning_rate_is_not_read(campaign):
    moved = without_manifest(campaign, lambda p: p.variant == "null" and p.init_seed == NULL_SEEDS[0])
    moved.append(
        fake_manifest("null", budget="100M", branches=[], init_seed=NULL_SEEDS[0], peak_lr=2e-3, val_loss=3.5)
    )
    a = analyse_committed(moved)
    assert NULL_SEEDS[0] not in a.null_100m
    assert a.null_control is None
    assert any("null control" in note and str(NULL_SEEDS[0]) in note for note in a.notes)


def test_two_hundred_million_branches_are_collected(campaign, analysis):
    for name in ("mha", "mha_L6", "mha_L10"):
        assert analysis.losses_200m[name] == losses_of(campaign, variant=name, tokens="300M", branch="200M")


# --- 2. Noise components -----------------------------------------------------------------------


def test_sigma_data_is_the_sample_sd_over_the_three_data_seeds(analysis):
    values = [analysis.data_order_100m[d] for d in (1, *DATA_ORDER_SEEDS)]
    assert scale_relative_error(analysis.sigma_data, statistics.stdev(values)) <= FLOAT64_IDENTITY


def test_sigma_init_from_the_null_seeds_is_their_sample_sd(analysis):
    values = [analysis.null_100m[s] for s in NULL_SEEDS]
    assert scale_relative_error(analysis.sigma_init_null, statistics.stdev(values)) <= FLOAT64_IDENTITY


def test_sigma_data_needs_all_three_data_seeds(campaign):
    a = analyse_committed(without_manifest(campaign, lambda p: p.variant == "mha" and p.data_seed == DATA_ORDER_SEEDS[-1]))
    assert a.sigma_data is None
    assert any("sigma_data" in note and str(DATA_ORDER_SEEDS[-1]) in note for note in a.notes)


# --- 3. The two controls come from primary_test, on the registered runs ------------------------


def test_null_control_is_primary_tests_own_on_the_hundred_million_branches(analysis, mha_branches):
    expected = pt.null_control(
        analysis.null_100m,
        {s: mha_branches[s] for s in SELECTION_SEEDS},
        test=analysis.selection.test,
        prereg=PREREG,
    )
    assert analysis.null_control == expected
    assert analysis.null_control.kind == "null_control"
    # The reference is the Arm A runs' 100M branches, never the standalone reruns.
    assert analysis.null_control.reference_losses == tuple(mha_branches[s] for s in SELECTION_SEEDS)
    assert analysis.null_control.reference_losses[:3] != tuple(analysis.reruns_100m[s] for s in (1, 2, 3))


def test_positive_control_is_primary_tests_own_on_seeds_one_to_five(analysis, campaign, mha_300):
    l6 = losses_of(campaign, variant="mha_L6", tokens="300M", branch=None)
    expected = pt.positive_control(
        {s: l6[s] for s in CONTROL_SEEDS},
        {s: mha_300[s] for s in CONTROL_SEEDS},
        test=analysis.selection.test,
        prereg=PREREG,
    )
    assert analysis.positive_control == expected
    assert analysis.positive_control.kind == "positive_control"


def test_positive_control_ignores_seeds_six_to_eight():
    eight = fake_campaign(sigma=0.01, n_primary_seeds=8, gaps={"mqa": 0.05})
    a = analyse_committed(eight)
    assert a.decision.n_seeds == 8, "this synthetic campaign is meant to pick 8 seeds"
    assert a.positive_control.seeds == CONTROL_SEEDS
    assert a.positive_control.reference_seeds == CONTROL_SEEDS


def test_a_missing_null_seed_leaves_the_null_control_uncomputed(campaign):
    a = analyse_committed(without_manifest(campaign, lambda p: p.variant == "null" and p.init_seed == NULL_SEEDS[2]))
    assert a.null_control is None
    assert a.sigma_init_null is None
    assert any("null control" in note and str(NULL_SEEDS[2]) in note for note in a.notes)


# --- 4. Ladder fits by budget (plan §8.4) -------------------------------------------------------


def test_fits_has_the_three_budgets_and_the_primary_one_is_the_fit(analysis):
    assert tuple(analysis.fits) == ("100M", "200M", "300M")
    assert analysis.fits["300M"] == analysis.fit


@pytest.mark.parametrize("budget", ["100M", "200M"])
def test_branch_fits_are_ladder_fits_of_the_branch_losses(campaign, analysis, budget):
    points = {
        name: (LADDER_PARAMS[name], losses_of(campaign, variant=name, tokens="300M", branch=budget))
        for name in ("mha_L6", "mha", "mha_L10")
    }
    assert analysis.fits[budget] == lf.fit_ladder(points)


def test_branch_fits_are_absent_without_ladder_branches():
    a = analyse_committed(fake_campaign())
    assert a.fits["300M"] == a.fit and a.fit is not None
    assert a.fits["100M"] is None and a.fits["200M"] is None
    assert any("ladder fit at 200M" in note for note in a.notes)


# --- 5. No cross-variant number before the controls --------------------------------------------


def test_with_both_controls_passing_every_comparison_is_computed(analysis):
    assert analysis.null_control.verdict == pt.VERDICT_NOT_DETECTED
    assert analysis.positive_control.verdict == pt.VERDICT_DETECTED
    assert set(analysis.comparisons) == set(PREREG["primary"]["comparisons"]) | set(PREREG["exploratory"]["variants"])
    assert set(analysis.multipliers) == set(analysis.comparisons)
    assert analysis.comparisons["mqa"].verdict == pt.VERDICT_DETECTED


def test_gate_two_stage_has_the_floor_and_the_null_control_and_nothing_downstream(campaign):
    a = analyse_committed(gate_two_stage(campaign))
    assert a.selection is not None
    assert a.null_control is not None
    assert a.sigma_data is not None and a.sigma_init_null is not None
    assert a.positive_control is None
    assert a.decision is None
    assert a.comparisons == {} and a.multipliers == {}
    assert any("positive control" in note for note in a.notes)


def test_no_comparison_without_the_null_control():
    a = analyse_committed(fake_campaign(gaps={"mqa": 0.05}, without=("null",)))
    assert a.null_control is None
    assert a.decision is not None, "the seed-count rule reads MHA and ladder runs only and still applies"
    assert a.comparisons == {} and a.multipliers == {}
    assert any("comparisons skipped" in note and "null control" in note for note in a.notes)


def test_a_detected_null_control_stops_every_comparison():
    a = analyse_committed(fake_campaign(gaps={"mqa": 0.05}, null_gap=0.5))
    assert a.null_control.verdict == pt.VERDICT_DETECTED
    assert a.comparisons == {} and a.multipliers == {}
    assert any("null control" in note and "plan §9.3" in note for note in a.notes)


def test_no_comparison_without_the_positive_control(campaign):
    # L6 seed 5 is missing: the reported ladder still fits, but the registered positive control cannot
    # run, and the seed-count rule waits for the complete ladder (plan §1.4; review S4, H2).
    a = analyse_committed(without_manifest(campaign, lambda p: p.variant == "mha_L6" and p.init_seed == CONTROL_SEEDS[-1]))
    assert a.fit is not None and a.decision is None
    assert a.positive_control is None
    assert a.comparisons == {} and a.multipliers == {}
    assert any("comparisons skipped" in note and "positive control" in note for note in a.notes)


def test_an_undetected_positive_control_is_flagged_and_comparisons_still_run(campaign, analysis, monkeypatch):
    undetected = dataclasses.replace(analysis.positive_control, verdict=pt.VERDICT_NOT_DETECTED)
    monkeypatch.setattr(mt.pt, "positive_control", lambda *args, **kwargs: undetected)
    a = analyse_committed(campaign)
    assert a.positive_control.verdict == pt.VERDICT_NOT_DETECTED
    assert set(a.comparisons) == set(analysis.comparisons)
    assert any("positive control" in note and "uninformative" in note and "plan §8.4" in note for note in a.notes)


# --- 6. The report ------------------------------------------------------------------------------


def test_report_sections_and_labels_are_fixed(analysis, blank):
    for a in (analysis, blank):
        values = mt.gate_values(a)
        assert list(values) == list(EXPECTED_SECTIONS)
        for section, labels in EXPECTED_SECTIONS.items():
            assert tuple(values[section]) == labels, section


def test_blank_report_has_no_value(blank):
    values = mt.gate_values(blank)
    # The one value a campaign without runs has: the ladder the committed file declares.
    assert values[DECLARATIONS].pop("ladder") == "full"
    assert all(value is None for section in values.values() for value in section.values())
    assert blank.null_control is None and blank.positive_control is None
    assert blank.sigma_data is None and blank.sigma_init_null is None
    assert blank.null_100m == {} and blank.data_order_100m == {} and blank.losses_200m == {}
    assert blank.fits == {"100M": None, "200M": None, "300M": None}


def test_report_values_are_the_analysis_objects_own(analysis):
    v = mt.gate_values(analysis)
    sel, nc, pc, d = analysis.selection, analysis.null_control, analysis.positive_control, analysis.decision

    floor = v["Noise floor at 100M (Gate 2)"]
    assert floor["sigma_init, MHA seeds 1-5"] == sel.sigma_init
    assert floor["sigma_init, null-control seeds"] == analysis.sigma_init_null
    assert floor["sigma_data, MHA seed 1 over data seeds 1-3"] == analysis.sigma_data
    assert floor["s_rep"] == sel.s_rep
    assert floor["s_rep / sigma_init"] == sel.ratio
    assert floor["pairing threshold"] == sel.threshold
    assert floor["test chosen"] == sel.test

    null = v["Null control (Gate 2)"]
    assert null["null seeds"] == nc.seeds and null["MHA seeds"] == nc.reference_seeds
    assert null["mean gap, nats"] == nc.mean_gap
    assert null["95% interval"] == (nc.ci_low, nc.ci_high)
    assert null["p"] == nc.p
    assert null["verdict"] == nc.verdict

    positive = v["Positive control: L6 against L8 at 300M"]
    assert positive["seeds"] == pc.seeds
    assert positive["mean gap, nats"] == pc.mean_gap
    assert positive["95% interval"] == (pc.ci_low, pc.ci_high)
    assert positive["p"] == pc.p
    assert positive["verdict"] == pc.verdict

    count = v["Seed count and detectable effect"]
    assert count["sigma_init at 300M, MHA seeds 1-5"] == d.sigma_init
    assert count["ladder slope b at 300M"] == analysis.fit.b
    assert count["5-seed detectable gap, nats"] == d.detectable_gap
    assert count["undetectable parameter fraction at 5 seeds"] == d.undetectable_param_fraction
    assert count["threshold"] == d.threshold
    assert count["seeds per primary"] == d.n_seeds
    assert count["variants leaving Arm A"] == (d.leaves_arm_a if d.extended else ())
    assert count["detectable gap of the chosen design, nats"] == d.bound_gap

    slopes = v["Ladder slope b by budget"]
    assert slopes == {budget: analysis.fits[budget].b for budget in ("100M", "200M", "300M")}


def test_an_extended_design_names_the_variants_that_leave():
    a = analyse_committed(fake_campaign(sigma=0.01, n_primary_seeds=8, gaps={"mqa": 0.05}))
    count = mt.gate_values(a)["Seed count and detectable effect"]
    assert count["seeds per primary"] == 8
    assert count["variants leaving Arm A"] == tuple(PREREG["seed_count"]["leave_arm_a_if_extended"])


def test_rendered_report_has_every_heading_and_label(analysis, blank):
    for a in (analysis, blank):
        text = mt.render_gates(a)
        position = -1
        for section, labels in EXPECTED_SECTIONS.items():
            here = text.index(section, position + 1)
            assert here > position, section
            position = here
            for label in labels:
                assert f"| {label} |" in text, (section, label)


def test_rendered_blank_report_shows_a_dash_in_every_row(blank):
    text = mt.render_gates(blank)
    n_rows = sum(len(labels) for labels in EXPECTED_SECTIONS.values())
    rows = [line for line in text.splitlines() if line.endswith(f"| {DASH} |")]
    assert len(rows) == n_rows - 1  # every row but the declared ladder, which the committed file always has
    assert "| ladder | full |" in text
    for note in blank.notes:
        assert note in text


def test_rendered_verdicts_are_primary_tests_words(analysis):
    text = mt.render_gates(analysis)
    assert f"| verdict | {analysis.null_control.verdict} |" in text
    assert f"| verdict | {analysis.positive_control.verdict} |" in text
    assert f"| test chosen | {analysis.selection.test} |" in text
    assert f"| seeds per primary | {analysis.decision.n_seeds} |" in text


def test_a_detected_null_control_is_said_at_the_top_of_the_report():
    a = analyse_committed(fake_campaign(null_gap=0.5))
    text = mt.render_gates(a)
    first_section = text.index(next(iter(EXPECTED_SECTIONS)))
    warning = text.index("NULL CONTROL")
    assert warning < first_section
    assert "plan §9.3" in text[:first_section]
    assert "NULL CONTROL" not in mt.render_gates(analyse_committed(fake_campaign()))


# --- 7. The command line and the committed blank report -----------------------------------------

SLOPE_SENTENCE = (
    "The ladder slope b at 300M in this section is fitted on the registered seeds only (plan §1.4), "
    "while the ladder slope by budget below uses every run present, so the two values at 300M differ "
    "once MHA has eight seeds."
)


def test_the_slope_sentence_sits_in_the_seed_count_section_of_every_report(blank, analysis):
    for a in (blank, analysis):
        text = mt.render_gates(a)
        assert text.count(SLOPE_SENTENCE) == 1
        start = text.index("## Seed count and detectable effect")
        end = text.index("## Ladder slope b by budget")
        assert start < text.index(SLOPE_SENTENCE) < text.index("| item | value |", start) < end


def test_the_two_300m_slopes_are_both_labelled_in_the_blank_report(blank):
    text = mt.render_gates(blank)
    assert "| ladder slope b at 300M | — |" in text  # the rule's fit, in the seed-count section
    assert "| 300M | — |" in text  # the fit by budget


def test_committed_blank_report_is_generated_from_no_manifests(blank):
    assert BLANK_GATES.read_text(encoding="utf-8") == mt.render_gates(blank)


def test_main_blank_writes_both_tables(tmp_path):
    assert tables_main(["--blank", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "gates.md").read_bytes() == BLANK_GATES.read_bytes()
    assert (tmp_path / "headline.md").is_file()


def test_main_writes_the_gate_report_from_manifests(tmp_path, campaign, analysis):
    directory = tmp_path / "manifests"
    for manifest in campaign:
        mf.write_manifest(manifest, directory)
    out = tmp_path / "tables"
    assert tables_main(["--manifests", str(directory), "--out", str(out)]) == 0
    assert (out / "gates.md").read_text(encoding="utf-8") == mt.render_gates(analysis)
