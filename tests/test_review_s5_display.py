"""T-55 (review S5, G2, G3, M1, M2, M3, M5): what the tables and plots show, and when.

Every campaign here is SYNTHETIC (tests/fixtures/fake_campaign.py, fake_plot_inputs.py,
fake_secondary.py). The seed count is committed as the rule gives it, or deliberately not, through
the `commitments` argument of make_tables.analyse and make_plots.make_inputs.
"""

from __future__ import annotations

import dataclasses

import pytest

from analysis import ladder_fit as lf
from scripts import bench as bd
from scripts import make_plots as m
from scripts import make_tables as mt
from scripts import timing_report as report
from src.utils import manifest as mf
from tests.fixtures import fake_plot_inputs as fpi
from tests.fixtures.commitments import committed, plots_main, tables_main
from tests.fixtures.fake_campaign import fake_campaign
from tests.fixtures.fake_secondary import bench_document, write_bench_documents
from tests.test_plots import all_texts, dot_count
from tests.test_timing_report import hand_bench_doc

FULL, CUT = "full", "l6_l8"
WITHHELD = " ".join(m.WITHHELD_CAPTION.split())
COMMIT_NOTE = "variant runs are withheld until the seed count is committed"
CONTROLS_NOTE = "variant runs are withheld until the controls pass"


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}))


@pytest.fixture(scope="module")
def shut_campaign() -> list[dict]:
    """No null control: the gate is shut."""
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, without=("null",)))


@pytest.fixture(scope="module")
def right(campaign) -> int:
    n = committed(campaign)["seed_count"]
    assert n in (5, 8)
    return n


def _wrong(n: int) -> int:
    return 5 if n == 8 else 8


def _analyse(results, seed_count, ladder: str = FULL) -> mt.Analysis:
    return mt.analyse(results, commitments={"ladder": ladder, "seed_count": seed_count})


def _inputs(results, seed_count, *, bench=(), ladder: str = FULL) -> m.PlotInputs:
    return m.make_inputs(
        results, bench, fpi.probe_records(), fpi.retrieval_records(),
        commitments={"ladder": ladder, "seed_count": seed_count},
    )


def _has(fig, text: str) -> bool:
    """True when some text of the figure contains `text`."""
    return any(text in t for t in all_texts(fig))


def _headline(analysis, variant: str, column: str, **kwargs) -> str:
    rows = {row["variant"]: row for row in mt.headline_rows(analysis, **kwargs)}
    return rows[variant][column]


# --- Fix 1 (G2): the variants' 300M results wait for the committed seed count -----------------------


def test_variant_results_are_shown_only_with_the_gate_open_and_the_seed_count_committed(campaign, right):
    assert not mt.variant_results_shown(_analyse(campaign, None))
    assert not mt.variant_results_shown(_analyse(campaign, _wrong(right)))
    shown = _analyse(campaign, right)
    assert shown.comparisons_allowed and mt.variant_results_shown(shown)


@pytest.mark.parametrize("which", ["uncommitted", "wrong"])
def test_headline_withholds_every_variant_cell_but_mha_until_the_seed_count_is_committed(campaign, right, which):
    seed = None if which == "uncommitted" else _wrong(right)
    a = _analyse(campaign, seed)
    assert a.comparisons_allowed and a.decision is not None
    assert _headline(a, "mha", "val loss, every seed") != mt.BLANK
    for variant in mt.HEADLINE_VARIANTS:
        if variant == "mha":
            continue
        for column in ("val loss, every seed", "gap vs MHA [95% CI]", "effective-param multiplier", "verdict"):
            assert _headline(a, variant, column) == mt.BLANK, (variant, column)


def test_rqw_and_context_say_the_commitment_is_missing_and_hold_no_variant_value(campaign):
    a = _analyse(campaign, None)
    ms_per_step = {variant: 1.0 for variant in mt.HEADLINE_VARIANTS}
    rqw = mt._rqw_report(a, campaign, ms_per_step)
    assert f"Not computed: {mt.COMMITMENT_WAIT}." in rqw
    assert "estimated loss, every seed" not in rqw  # no table: no variant value
    context = mt.render_context(a)
    assert f"Not computed: {mt.COMMITMENT_WAIT} (see gates.md)." in context
    assert mt.GATE_WAIT not in context
    assert not a.context_use


def test_the_gate_wait_text_stays_for_a_shut_gate(shut_campaign):
    a = _analyse(shut_campaign, None)
    assert not a.comparisons_allowed
    assert f"Not computed: {mt.GATE_WAIT}" in mt.render_context(a)
    assert f"Not computed: {mt.GATE_WAIT}." in mt._rqw_report(a, shut_campaign, {"mha": 1.0})


def test_plots_2_3_and_7_draw_no_variant_point_while_the_commitment_is_missing(campaign):
    inp = _inputs(campaign, None, bench=fpi.bench_records())
    assert dot_count(m.plot_inference_frontier(inp)) == len(inp.analysis.losses["mha"])
    assert dot_count(m.plot_memory_frontier(inp)) == len(inp.analysis.losses["mha"])
    runs, withheld = m._curve_runs(inp)
    assert withheld
    assert {mf.parse_run_name(x["name"]).variant for x, _ in runs} == {"mha"}
    assert dot_count(m.plot_training_curves(inp)) == len(runs)
    for fig in (m.plot_inference_frontier(inp), m.plot_memory_frontier(inp), m.plot_training_curves(inp)):
        assert _has(fig, WITHHELD)


def test_plots_2_3_and_7_show_every_variant_once_the_seed_count_is_committed(campaign, right):
    inp = _inputs(campaign, right, bench=fpi.bench_records())
    expected = sum(len(inp.analysis.losses[v]) for v in mt.HEADLINE_VARIANTS)
    assert dot_count(m.plot_memory_frontier(inp)) == expected
    runs, withheld = m._curve_runs(inp)
    assert not withheld
    assert dot_count(m.plot_training_curves(inp)) == len(runs)
    assert not _has(m.plot_memory_frontier(inp), WITHHELD)


def test_the_commitment_note_is_written_once_and_only_while_a_variant_loss_waits(campaign, shut_campaign, right):
    a = _analyse(campaign, None)
    notes = [n for n in a.notes if n.startswith(COMMIT_NOTE)]
    assert len(notes) == 1
    assert "mqa" in notes[0]
    assert not [n for n in a.notes if n.startswith(CONTROLS_NOTE)]
    # Gate shut: the old note, and not the new one.
    shut = _analyse(shut_campaign, None)
    assert [n for n in shut.notes if n.startswith(CONTROLS_NOTE)]
    assert not [n for n in shut.notes if n.startswith(COMMIT_NOTE)]
    # Committed: no note.
    assert not [n for n in _analyse(campaign, right).notes if COMMIT_NOTE in n]


def test_with_the_commitment_the_headline_shows_every_variant_as_before(campaign, right):
    a = _analyse(campaign, right)
    for variant in mt.HEADLINE_VARIANTS:
        if variant != "mha":
            assert _headline(a, variant, "val loss, every seed") != mt.BLANK
    assert _headline(a, "mqa", "verdict") != mt.BLANK


def test_arm_b_and_the_shared_rate_table_do_not_wait_for_the_commitment(campaign, right):
    uncommitted, committed_ = _analyse(campaign, None), _analyse(campaign, right)
    assert mt.render_arm_b(uncommitted) == mt.render_arm_b(committed_)
    assert mt.render_shared_rate(uncommitted) == mt.render_shared_rate(committed_)


# --- Fix 2 (G3): plot 6 draws no gap of a withheld variant ----------------------------------------


def test_plot_6_draws_the_gaps_when_the_variants_are_shown(campaign, right):
    inp = _inputs(campaign, right)
    fig = m.plot_context_gap(inp)
    mha_seeds = {
        mf.parse_run_name(x["name"]).init_seed for x in campaign
        if mf.parse_run_name(x["name"]).variant == "mha" and mf.parse_run_name(x["name"]).tokens == "300M"
        and mf.parse_run_name(x["name"]).branch is None and mf.parse_run_name(x["name"]).data_seed == 1
    }
    pairs = sum(len([s for s in inp.analysis.losses.get(v, {}) if s in mha_seeds]) for v in mt.HEADLINE_VARIANTS if v != "mha")
    assert dot_count(fig) == pairs * len(fpi.BUCKETS) > 0
    assert not _has(fig, WITHHELD)


@pytest.mark.parametrize("case", ["gate shut", "no commitment"])
def test_plot_6_draws_no_gap_and_says_so_with_the_gate_shut_or_uncommitted(campaign, shut_campaign, right, case):
    inp = _inputs(shut_campaign, right) if case == "gate shut" else _inputs(campaign, None)
    fig = m.plot_context_gap(inp)
    assert dot_count(fig) == 0
    assert _has(fig, WITHHELD)


# --- Fix 3 (M2): a declared cut reaches every fit, the multipliers and plot 5 ---------------------


@pytest.fixture(scope="module")
def branched_campaign() -> list[dict]:
    """With L10 runs at 300M and the 100M and 200M branches of the ladder."""
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True))


def test_a_declared_cut_leaves_mha_l10_out_of_every_fit_and_multiplier(branched_campaign):
    n = committed(branched_campaign, ladder=CUT)["seed_count"]
    a = _analyse(branched_campaign, n, ladder=CUT)
    assert sorted(p.name for p in a.fit.points) == ["mha", "mha_L6"]
    for budget in ("100M", "200M", "300M"):
        assert sorted(p.name for p in a.fits[budget].points) == ["mha", "mha_L6"], budget
    assert a.fit.b == a.rule_fit.b
    assert a.multipliers
    for name, multiplier in a.multipliers.items():
        assert multiplier == lf.comparison_multiplier(a.comparisons[name], a.fit)


def test_without_a_cut_the_fits_have_three_points(branched_campaign, right):
    full = _analyse(branched_campaign, committed(branched_campaign)["seed_count"])
    assert sorted(p.name for p in full.fit.points) == ["mha", "mha_L10", "mha_L6"]
    assert sorted(p.name for p in full.fits["200M"].points) == ["mha", "mha_L10", "mha_L6"]


def test_plot_5_draws_no_l10_under_a_declared_cut_and_says_so(branched_campaign):
    n = committed(branched_campaign, ladder=CUT)["seed_count"]
    inp = _inputs(branched_campaign, n, ladder=CUT)
    fig = m.plot_ladder(inp)
    a = inp.analysis
    expected = sum(
        len(by_budget.get(v, {})) for by_budget in (a.losses_100m, a.losses_200m, a.losses)
        for v in ("mha", "mha_L6")
    )
    assert dot_count(fig) == expected > 0
    assert not any(t.get_text() == "L10" for ax in fig.axes for t in ax.texts)
    assert "cut to L6 and L8 by declaration" in " ".join(all_texts(fig))


def test_plot_5_keeps_l10_without_a_cut(branched_campaign):
    inp = _inputs(branched_campaign, committed(branched_campaign)["seed_count"])
    fig = m.plot_ladder(inp)
    assert any(t.get_text() == "L10" for ax in fig.axes for t in ax.texts)
    assert "cut to L6 and L8" not in " ".join(all_texts(fig))


# --- Fix 4 (M3): no multiplier when the ladder slope is not positive ------------------------------


# The multiplier axis, its label and the caption line that describes it. The analysis note that no
# multiplier is reported (decision 223) is not an axis and is not checked here.
MULTIPLIER_AXIS_TEXT = ("effective-parameter multiplier", "multiplier exp(")


def _multiplier_mentions(fig) -> list[str]:
    texts = list(all_texts(fig))
    for ax in fig.axes:
        texts += [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]
    return [t for t in texts if any(phrase in t for phrase in MULTIPLIER_AXIS_TEXT)]


def test_plot_1_has_no_multiplier_when_the_slope_is_not_positive():
    campaign = fpi.with_eval_fields(fake_campaign(slope=-0.4, n_primary_seeds=8, gaps={"mqa": 0.05}))
    n = committed(campaign)["seed_count"]
    assert n == 8
    inp = _inputs(campaign, n)
    assert inp.analysis.decision.slope_not_positive
    fig = m.plot_primary_gaps(inp)
    assert _multiplier_mentions(fig) == []


def test_plot_1_has_its_multiplier_axis_when_the_slope_is_positive(campaign, right):
    fig = m.plot_primary_gaps(_inputs(campaign, right))
    assert _multiplier_mentions(fig)


# --- Fix 5 (M1): the benchmark's ranges, and documents the report refuses -------------------------


def _bad_documents() -> dict[str, list]:
    cases = {}
    doc = hand_bench_doc()
    doc["schema"] = "bench/1"
    cases["another schema"] = [doc]
    cases["unknown variant"] = [hand_bench_doc("foo")]
    cases["mode other than opt or ref"] = [hand_bench_doc(mode="train")]
    doc = hand_bench_doc()
    del doc["decode"]
    cases["a missing field"] = [doc]
    doc = hand_bench_doc()
    doc["prefill"][0]["latency_ms_by_repetition"] = [3.5, 4.0]
    cases["a list of the wrong length"] = [doc]
    doc = hand_bench_doc()
    doc["decode"][0]["ms_per_token_min"] = 2.6
    cases["a minimum above the median"] = [doc]
    doc = hand_bench_doc()
    doc["prefill"][0]["latency_ms"] = 3.9
    cases["a median that is not the median"] = [doc]
    cases["two documents for one variant and mode"] = [hand_bench_doc(), hand_bench_doc()]
    return cases


BAD_DOCUMENTS = _bad_documents()


@pytest.mark.parametrize("case", sorted(BAD_DOCUMENTS))
def test_check_documents_refuses_what_timing_report_refuses(case):
    docs = BAD_DOCUMENTS[case]
    with pytest.raises(bd.BenchError) as bench_error:
        bd.check_documents(docs)
    with pytest.raises(report.TimingReportError) as report_error:
        report.check_bench_documents(docs)
    assert str(report_error.value) == str(bench_error.value)


def test_check_documents_accepts_a_valid_document():
    bd.check_documents([hand_bench_doc(), hand_bench_doc("gqa4")])


def test_the_table_command_refuses_a_bad_bench_document_and_writes_nothing(tmp_path, campaign):
    manifests = tmp_path / "manifests"
    for manifest in fake_campaign(gaps={"mqa": 0.05}):
        mf.write_manifest(manifest, manifests)
    bench_dir = tmp_path / "bench"
    write_bench_documents(bench_dir, [BAD_DOCUMENTS["a median that is not the median"][0]])
    out = tmp_path / "tables"
    assert tables_main(["--manifests", str(manifests), "--out", str(out), "--bench", str(bench_dir)]) == 1
    assert not out.exists() or not any(out.iterdir())


def test_the_plot_inputs_refuse_a_bad_bench_document(campaign):
    with pytest.raises(m.PlotError):
        m.make_inputs(campaign, [BAD_DOCUMENTS["a median that is not the median"][0]], commitments={"ladder": FULL, "seed_count": None})


def test_read_decode_range_is_the_smallest_and_largest_of_the_headline_point(tmp_path):
    bench_dir = tmp_path / "bench"
    write_bench_documents(bench_dir, [hand_bench_doc()])
    assert bd.read_decode_range(bench_dir) == {"mha": (11000.0, 14600.0)}
    assert bd.read_decode_tok_per_s(bench_dir) == {"mha": 12800.0}


def test_the_decode_cell_reads_the_median_with_the_range_in_brackets(campaign, right):
    a = _analyse(campaign, right)
    column = "decode tok/s (batch 32, cache 4k)"
    span = {"mha": (11000.0, 14600.0)}
    assert _headline(a, "mha", column, decode_tok_per_s={"mha": 12800.0}, decode_range=span) == "12800.0 (11000.0–14600.0)"
    assert _headline(a, "mha", column, decode_tok_per_s={"mha": 12800.0}) == "12800.0"
    assert _headline(a, "mha", column) == mt.BLANK


def test_plots_2_and_8_say_the_speed_values_are_medians(campaign, right):
    inp = _inputs(campaign, right, bench=fpi.bench_records())
    for fig in (m.plot_inference_frontier(inp), m.plot_latency(inp)):
        assert _has(fig, " ".join(m.SPEED_MEDIAN_CAPTION.split()))


# --- Fix 6 (M5): the shared-rate caption is true under both tests ---------------------------------


def test_the_shared_rate_caption_names_both_tests_and_qualifies_by_seed():
    caption = mt.SHARED_RATE_CAPTION
    assert "paired by seed under the paired test" in caption
    assert "two groups of three runs under Welch's test" in caption
    assert caption.count("by seed") == 1
    assert "on the same init seeds 1 to 3" in caption


def test_the_shared_rate_table_carries_the_caption(campaign, right):
    assert mt.SHARED_RATE_CAPTION in mt.render_shared_rate(_analyse(campaign, right))
