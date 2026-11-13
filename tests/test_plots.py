"""T-28: the ten plots, built blank and from synthetic inputs (plan §10.5, §13.15, §14.2).

Every manifest and record here is SYNTHETIC (tests/fixtures/fake_campaign.py and
tests/fixtures/fake_plot_inputs.py). The tests inspect the Figure objects: which
points each axes draws, which legend entries and captions it carries. Expected
counts come from the manifests and from `analyse`, never from the plot code.
"""

from __future__ import annotations

import ast
import copy
import inspect
import math
import os
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure

from analysis import primary_test as pt
from scripts import make_plots as m
from scripts import make_tables as mt
from src.utils import manifest as mf
from tests.fixtures import fake_plot_inputs as fpi
from tests.fixtures import fake_secondary as fsec
from src import config
from tests.fixtures.fake_campaign import DATA_ORDER_SEEDS, LADDER_PARAMS, NULL_SEEDS, fake_campaign
from tests.fixtures.fake_manifests import fake_manifest
from tests.tolerances import FLOAT64_IDENTITY
from tests.fixtures.commitments import (
    analyse_committed, committed, make_inputs_committed, plots_main, write_commitments,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# The brief's requirement, not a numerical tolerance: dots at alpha 0.6 or more stay visible when they overlap.
MIN_DOT_ALPHA = 0.6
EXPECTED_FILES = (
    "01_primary_gaps.png",
    "02_inference_frontier.png",
    "03_memory_frontier.png",
    "04_noise_floor.png",
    "05_ladder.png",
    "06_context_gap.png",
    "07_training_curves.png",
    "08_latency.png",
    "09_probes.png",
    "10_retrieval.png",
)
PRIMARY = ["gqa4", "mqa", "mla192"]
EXPLORATORY = ["gqa2", "mla96", "swa256"]
COMPARED_IN_GAP_PLOTS = [v for v in mt.HEADLINE_VARIANTS if v != "mha"]
PARSE = mf.parse_run_name


# --- Helpers that read Figure objects -----------------------------------------------------


def dot_collections(ax) -> list:
    """The seed dots of an axes: every scatter collection. The only way a plot draws a seed."""
    return [c for c in ax.collections if isinstance(c, PathCollection)]


def plotted_points(ax) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for collection in dot_collections(ax):
        points.extend((float(x), float(y)) for x, y in np.asarray(collection.get_offsets()))
    return points


def all_points(fig) -> list[tuple[float, float]]:
    return [p for ax in fig.axes for p in plotted_points(ax)]


def dot_count(fig) -> int:
    return len(all_points(fig))


def all_texts(fig) -> list[str]:
    """Every text of the figure, with its line breaks (from wrapping) turned into spaces."""
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    texts += [t.get_text() for t in fig.texts]
    return [" ".join(text.split()) for text in texts]


def legend_labels(fig) -> list[str]:
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            return [t.get_text() for t in legend.get_texts()]
    return []


def assert_visible_dots(ax) -> None:
    """Every dot has a non-zero size and an alpha of at least MIN_DOT_ALPHA, and lies inside the axes."""
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    for collection in dot_collections(ax):
        sizes = np.asarray(collection.get_sizes())
        assert sizes.size and np.all(sizes > 0)
        assert collection.get_alpha() is not None and collection.get_alpha() >= MIN_DOT_ALPHA
    for x, y in plotted_points(ax):
        assert x0 <= x <= x1 and y0 <= y <= y1, f"dot ({x}, {y}) lies outside the axes"


def expected_legend(variants, prereg) -> list[str]:
    return [m._label(v, prereg) for v in variants]


def canvas_draw(fig) -> None:
    FigureCanvasAgg(fig)
    fig.canvas.draw()


def _primary_panel(figure: Figure, name: str, prereg):
    label = m.DISPLAY.get(name, name)
    matches = [ax for ax in figure.axes if ax.get_title().split("\n")[0] == label]
    assert len(matches) == 1, f"no single panel titled {label!r}"
    return matches[0]


def _arm_n_losses(manifests) -> list[float]:
    """Standalone 100M MHA runs (reproducibility reruns, data-order runs) and the null control."""
    out = []
    for manifest in manifests:
        parsed = PARSE(manifest["name"])
        if parsed.tokens != "100M" or parsed.branch is not None:
            continue
        if parsed.variant in ("mha", "null"):
            out.append(float(manifest["final_metrics"]["val_loss"]))
    return out


def _tokens_of(name: str) -> int:
    parsed = PARSE(name)
    return m._token_count(parsed.branch if parsed.branch is not None else parsed.tokens)


# --- Module fixtures ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    """Paired test selected (the default rerun noise), every synthetic input, Arm N extras."""
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}))


@pytest.fixture(scope="module")
def welch_campaign() -> list[dict]:
    """Welch's test selected: the reproducibility reruns are noisier than the threshold allows."""
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, rerun_sigma=0.008))


@pytest.fixture(scope="module")
def inputs(campaign):
    return make_inputs_committed(campaign, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records())


@pytest.fixture(scope="module")
def welch_inputs(welch_campaign):
    return make_inputs_committed(welch_campaign, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records())


@pytest.fixture(scope="module")
def figures(inputs) -> dict[str, Figure]:
    return {name: builder(inputs) for name, builder in m.PLOTS}


@pytest.fixture(scope="module")
def blank_figures() -> dict[str, Figure]:
    blank = make_inputs_committed([])
    return {name: builder(blank) for name, builder in m.PLOTS}


# --- The ten plots, built and written ----------------------------------------------------------


def test_the_plot_list_is_the_ten_plots_in_plan_order():
    assert tuple(name for name, _ in m.PLOTS) == EXPECTED_FILES


def test_build_all_writes_exactly_the_ten_files_in_order_as_png(tmp_path, campaign):
    paths = m.build_all(campaign, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records(), tmp_path)
    assert [p.name for p in paths] == list(EXPECTED_FILES)
    assert sorted(os.listdir(tmp_path)) == sorted(EXPECTED_FILES)
    for path in paths:
        data = path.read_bytes()
        assert data[:8] == PNG_SIGNATURE
        assert len(data) > 1000


def test_each_builder_returns_a_figure(figures):
    for figure in figures.values():
        assert isinstance(figure, Figure)


# --- Blank plots -------------------------------------------------------------------------------


def test_every_blank_plot_has_zero_points_a_no_data_note_a_title_and_axis_labels(blank_figures):
    for name, figure in blank_figures.items():
        assert dot_count(figure) == 0, name
        assert any("no data yet" in text for text in all_texts(figure)), name
        assert figure.get_suptitle(), name
        for ax in figure.axes:
            if not ax.axison:
                continue
            assert ax.get_xlabel(), f"{name}: an axes has no x label"
            assert ax.get_ylabel(), f"{name}: an axes has no y label"


def test_blank_plots_list_their_variants_in_the_fixed_order(blank_figures, inputs):
    prereg = inputs.prereg
    headline = list(mt.HEADLINE_VARIANTS)
    assert legend_labels(blank_figures["01_primary_gaps.png"]) == expected_legend(headline, prereg)
    assert legend_labels(blank_figures["02_inference_frontier.png"]) == expected_legend(headline, prereg)
    assert legend_labels(blank_figures["03_memory_frontier.png"]) == expected_legend(headline, prereg)


# --- Every seed is a visible dot ------------------------------------------------------------------


def _primary_expected(analysis) -> int:
    """One dot per seed: d_k under the paired test, both groups under Welch's."""
    total = 0
    for comp in analysis.comparisons.values():
        if comp.test == pt.PAIRED:
            total += len(comp.differences)
        else:
            total += len(comp.variant_losses) + len(comp.reference_losses)
    return total


def test_plot_1_draws_every_seed_and_nothing_else(figures, inputs):
    assert len(inputs.analysis.comparisons) == 6
    assert inputs.analysis.selection.test == pt.PAIRED
    assert dot_count(figures["01_primary_gaps.png"]) == _primary_expected(inputs.analysis)
    for ax in figures["01_primary_gaps.png"].axes:
        if ax.get_title():
            assert_visible_dots(ax)


def test_plot_1_welch_draws_both_groups_for_every_seed(welch_inputs):
    assert welch_inputs.analysis.selection.test == pt.WELCH
    figure = m.plot_primary_gaps(welch_inputs)
    assert dot_count(figure) == _primary_expected(welch_inputs.analysis)


def test_plot_2_draws_every_seed_of_every_variant_with_a_benchmark(figures, inputs):
    bench = {(r["variant"], r["mode"]) for r in inputs.bench}
    expected = sum(len(inputs.analysis.losses[v]) for v in mt.HEADLINE_VARIANTS if (v, "opt") in bench)
    assert dot_count(figures["02_inference_frontier.png"]) == expected == 29
    for ax in figures["02_inference_frontier.png"].axes:
        assert_visible_dots(ax)


def test_plot_2_x_is_the_opt_decode_rate_at_batch_32_and_cache_4096(figures, inputs):
    """Plot 2 is at batch 32 and cache 4k (plan §8.6): x is that record's tok/s, opt path only."""
    xs = {round(x, 9) for x, _ in all_points(figures["02_inference_frontier.png"])}
    expected = set()
    for record in inputs.bench:
        if record["mode"] == "opt" and record["variant"] in inputs.analysis.losses:
            rows = [r for r in record["decode"] if r["batch"] == 32 and r["cache"] == 4096]
            expected.add(round(float(rows[0]["tok_per_s"]), 9))
    assert xs == expected and len(expected) == 7


def test_plot_3_x_is_the_analytic_kv_bytes_per_token_of_each_variant(figures, inputs):
    xs = sorted({round(x, 9) for x, _ in all_points(figures["03_memory_frontier.png"])})
    expected = sorted({
        float(mt.static_columns(v)["KV bytes/token/layer"].split(" ")[0].replace(",", ""))
        for v in mt.HEADLINE_VARIANTS
    })
    assert xs == [round(x, 9) for x in expected]


def test_plot_3_draws_every_seed_of_every_variant(figures, inputs):
    expected = sum(len(inputs.analysis.losses[v]) for v in mt.HEADLINE_VARIANTS)
    assert dot_count(figures["03_memory_frontier.png"]) == expected == 29


def test_plot_4_draws_every_arm_n_and_null_run_and_nothing_else(figures, inputs, campaign):
    a = inputs.analysis
    arm_n = _arm_n_losses(campaign)
    # 5 MHA branches (init seeds 1-5), 3 reruns, 2 data-order runs and 5 null-control runs in the
    # manifests; the data-order row also shows data seed 1, which is init seed 1's branch: 16 dots.
    assert len(a.losses_100m["mha"]) == 5 and len(a.reruns_100m) == 3 and len(arm_n) == 10
    figure = figures["04_noise_floor.png"]
    assert dot_count(figure) == 16
    drawn = sorted(x for x, _ in all_points(figure))
    wanted = sorted(list(a.losses_100m["mha"].values()) + arm_n + [a.data_order_100m[1]])
    np.testing.assert_allclose(drawn, wanted, rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_5_draws_every_ladder_seed_at_300m_and_every_100m_branch(figures, inputs):
    a = inputs.analysis
    at_300 = sum(len(a.losses.get(name, {})) for name in m.LADDER)
    at_100 = sum(len(a.losses_100m.get(name, {})) for name in m.LADDER)
    # 300M: L6 (5 seeds), MHA (5), L10 (3). 100M: the MHA branches (5); the fixture has no ladder branches.
    assert (at_300, at_100) == (13, 5)
    assert dot_count(figures["05_ladder.png"]) == at_300 + at_100 == 18
    for ax in _ladder_panels(figures["05_ladder.png"]):
        assert_visible_dots(ax)


def test_plot_6_draws_every_seed_gap_in_every_bucket(figures, inputs, campaign):
    mha_seeds = {
        PARSE(x["name"]).init_seed
        for x in campaign
        if PARSE(x["name"]).variant == "mha" and PARSE(x["name"]).tokens == "300M"
        and PARSE(x["name"]).branch is None and PARSE(x["name"]).data_seed == 1
    }
    pairs = sum(len([s for s in inputs.analysis.losses.get(v, {}) if s in mha_seeds]) for v in COMPARED_IN_GAP_PLOTS)
    assert pairs == 24
    assert dot_count(figures["06_context_gap.png"]) == pairs * len(fpi.BUCKETS) == 96


def test_plot_7_draws_one_final_dot_per_arm_a_headline_run_and_branch(campaign):
    # T-45 (G3): the expected count is read off the run names, not from the plot code.
    assert dot_count(m.plot_training_curves(make_inputs_committed(campaign))) == len(_plot_7_names(campaign)) > 0


def test_plot_8_draws_lines_and_no_dots(figures):
    assert dot_count(figures["08_latency.png"]) == 0


def test_plot_9_draws_every_probe_record_as_a_dot(figures, inputs):
    assert dot_count(figures["09_probes.png"]) == len(inputs.probes)
    for ax in figures["09_probes.png"].axes:
        if ax.get_title().startswith("probe:"):
            assert_visible_dots(ax)


def test_plot_10_draws_every_retrieval_record_as_a_dot(figures, inputs):
    assert dot_count(figures["10_retrieval.png"]) == len(inputs.retrieval)


# --- Plots 4 and 5 read the controls and the per-budget ladder fits -----------------------------

# Plot 4's rows, from the top: branches (y = 3), reruns (2), data order (1), null control (0).
# Dots are staggered by 0.14 around their row, so rounding y recovers the row.
ROW_DATA_ORDER, ROW_NULL = 1, 0
LADDER_BUDGET_TITLES = ["100M tokens", "200M tokens", "300M tokens"]


def _noise_row_points(figure, row: int) -> list[float]:
    return sorted(x for x, y in all_points(figure) if round(y) == row)


def _ladder_panels(figure) -> list:
    """Plot 5's three panels: the axes that carry a title (the legend and caption rows have none)."""
    return [ax for ax in figure.axes if ax.get_title()]


def _fit_lines(ax) -> list:
    return [line for line in ax.lines if line.get_linestyle() == "--"]


def _panel_text(ax) -> str:
    return " ".join(" ".join(t.get_text().split()) for t in ax.texts)


def _normalised(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def ladder_inputs():
    """Every ladder budget: MHA and the ladder points also branch at 200M, so all three fits exist."""
    campaign_ = fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True))
    return make_inputs_committed(campaign_, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records())


def test_plot_4_null_and_data_order_rows_are_the_analysis_values(figures, inputs):
    a = inputs.analysis
    assert len(a.null_100m) == len(NULL_SEEDS) and set(a.data_order_100m) == {1, *DATA_ORDER_SEEDS}
    figure = figures["04_noise_floor.png"]
    for row, values in ((ROW_NULL, a.null_100m), (ROW_DATA_ORDER, a.data_order_100m)):
        drawn = _noise_row_points(figure, row)
        assert len(drawn) == len(values)
        np.testing.assert_allclose(drawn, sorted(values.values()), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_4_labels_the_data_order_row_and_says_data_seed_1_is_not_a_second_run(figures):
    figure = figures["04_noise_floor.png"]
    labels = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert any("data seeds 1-3" in label for label in labels)
    assert any("Data seed 1 is the same run as init seed 1's dot" in _normalised(text) for text in all_texts(figure))


def test_plot_4_at_gate_two_draws_the_null_and_data_order_rows_from_the_analysis():
    campaign_ = fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True))
    # Gate 2 (test_gate_report.gate_two_stage): no 300M result of anything, only 100M and 200M branches.
    stage = [x for x in campaign_ if not (PARSE(x["name"]).tokens == "300M" and PARSE(x["name"]).branch is None)]
    inp = make_inputs_committed(stage)
    a = inp.analysis
    assert a.null_control is not None and a.fit is None and a.positive_control is None
    figure = m.plot_noise_floor(inp)
    for row, values in ((ROW_NULL, a.null_100m), (ROW_DATA_ORDER, a.data_order_100m)):
        drawn = _noise_row_points(figure, row)
        assert len(drawn) == len(values) > 0
        np.testing.assert_allclose(drawn, sorted(values.values()), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_4_shows_the_null_control_verdict_mean_gap_interval_and_test(figures, inputs):
    comp = inputs.analysis.null_control
    assert comp is not None
    texts = _normalised(" ".join(all_texts(figures["04_noise_floor.png"])))
    assert f"verdict {comp.verdict};" in texts
    assert f"mean gap {comp.mean_gap:+.4f} nats" in texts
    assert f"[{comp.ci_low:+.4f}, {comp.ci_high:+.4f}]" in texts
    assert f"{comp.test} test" in texts


def test_plot_4_warns_when_the_null_control_verdict_is_the_detected_one():
    bad = make_inputs_committed(fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, null_gap=0.5)))
    assert bad.analysis.null_control.verdict == pt.VERDICT_DETECTED
    figure = m.plot_noise_floor(bad)
    titles = [_normalised(ax.get_title()) for ax in figure.axes if ax.get_title()]
    assert len(titles) == 1
    assert "NULL CONTROL" in titles[0] and "plan §9.3" in titles[0]
    assert f"verdict {pt.VERDICT_DETECTED}" in titles[0]


def test_plot_4_has_no_null_warning_when_the_null_control_is_not_detected(figures, inputs):
    assert inputs.analysis.null_control.verdict != pt.VERDICT_DETECTED
    figure = figures["04_noise_floor.png"]
    assert all("NULL CONTROL" not in ax.get_title() for ax in figure.axes)
    assert not any("NULL CONTROL" in text for text in all_texts(figure))


def test_plot_4_says_the_null_control_is_not_computed_and_draws_no_null_dots():
    no_null = make_inputs_committed(fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, without=("null",))))
    a = no_null.analysis
    assert a.null_control is None and a.null_100m == {}
    figure = m.plot_noise_floor(no_null)
    assert _noise_row_points(figure, ROW_NULL) == []
    reason = m._reason(a.notes, "null control")
    assert reason is not None
    texts = _normalised(" ".join(all_texts(figure)))
    assert "Null control (plan §8.1): not computed." in texts
    assert _normalised(reason) in texts


def test_plot_4_reads_the_analysis_and_no_longer_parses_run_names():
    assert not hasattr(m, "_standalone_100m") and not hasattr(m, "DATA_ORDER_SEEDS")
    source = inspect.getsource(m.plot_noise_floor)
    assert "parse_run_name" not in source and '["name"]' not in source


def test_plot_5_has_three_panels_for_100m_200m_and_300m_in_that_order(ladder_inputs):
    panels = _ladder_panels(m.plot_ladder(ladder_inputs))
    assert [ax.get_title().split("\n")[0] for ax in panels] == LADDER_BUDGET_TITLES


def test_plot_5_each_panel_draws_every_ladder_seed_at_its_own_budget(ladder_inputs):
    a = ladder_inputs.analysis
    by_budget = {"100M": a.losses_100m, "200M": a.losses_200m, "300M": a.losses}
    panels = _ladder_panels(m.plot_ladder(ladder_inputs))
    for ax, budget in zip(panels, ("100M", "200M", "300M")):
        expected = sorted(
            (math.log(LADDER_PARAMS[name]), loss)
            for name in m.LADDER
            for loss in by_budget[budget].get(name, {}).values()
        )
        drawn = sorted(plotted_points(ax))
        assert len(drawn) == len(expected) == 13, budget
        np.testing.assert_allclose(np.asarray(drawn), np.asarray(expected), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
        assert_visible_dots(ax)


def test_plot_5_fit_line_of_each_panel_passes_through_the_analysis_fit(ladder_inputs):
    a = ladder_inputs.analysis
    panels = _ladder_panels(m.plot_ladder(ladder_inputs))
    for ax, budget in zip(panels, ("100M", "200M", "300M")):
        fit = a.fits[budget]
        assert fit is not None
        lines = _fit_lines(ax)
        assert len(lines) == 1
        xs_line = np.asarray(lines[0].get_xdata(), dtype=float)
        ys_line = np.asarray(lines[0].get_ydata(), dtype=float)
        for name in m.LADDER:
            n = LADDER_PARAMS[name]
            on_line = np.interp(math.log(n), xs_line, ys_line)
            np.testing.assert_allclose(on_line, fit.predict(n), rtol=FLOAT64_IDENTITY, atol=0.0)
        assert f"{fit.b:.4f}" in ax.get_title()


def test_plot_5_without_ladder_branches_200m_has_no_data_and_100m_has_dots_but_no_fit(inputs):
    panels = _ladder_panels(m.plot_ladder(inputs))
    assert [len(plotted_points(ax)) for ax in panels] == [5, 0, 13]
    assert _fit_lines(panels[0]) == [] and _fit_lines(panels[1]) == [] and len(_fit_lines(panels[2])) == 1
    assert "no data yet" in _panel_text(panels[1])
    reason = m._reason(inputs.analysis.notes, "ladder fit at 100M")
    assert reason is not None and "mha_L6" in reason
    assert _normalised("No fit line: " + reason) in _panel_text(panels[0])


def test_plot_5_shows_the_positive_control_verdict_on_the_300m_panel_only(ladder_inputs):
    a = ladder_inputs.analysis
    assert a.positive_control is not None
    panels = _ladder_panels(m.plot_ladder(ladder_inputs))
    text_300 = _panel_text(panels[2])
    assert f"verdict {a.positive_control.verdict};" in text_300
    assert f"mean gap {a.positive_control.mean_gap:+.4f} nats" in text_300
    assert "Positive control" not in _panel_text(panels[0]) and "Positive control" not in _panel_text(panels[1])


def test_plot_5_says_the_positive_control_is_not_computed_when_a_seed_is_missing():
    campaign_ = fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True)
    dropped = [
        x for x in campaign_
        if not (PARSE(x["name"]).variant == "mha_L6" and PARSE(x["name"]).tokens == "300M"
                and PARSE(x["name"]).branch is None and PARSE(x["name"]).init_seed == 5)
    ]
    assert len(dropped) == len(campaign_) - 1
    inp = make_inputs_committed(fpi.with_eval_fields(dropped))
    assert inp.analysis.positive_control is None
    reason = m._reason(inp.analysis.notes, "positive control")
    assert reason is not None
    text_300 = _panel_text(_ladder_panels(m.plot_ladder(inp))[2])
    assert "Positive control (plan §8.4): not computed." in text_300
    assert _normalised(reason) in text_300


def test_blank_plot_5_has_three_empty_panels_titled_by_budget(blank_figures):
    panels = _ladder_panels(blank_figures["05_ladder.png"])
    assert [ax.get_title().split("\n")[0] for ax in panels] == LADDER_BUDGET_TITLES
    for ax in panels:
        assert plotted_points(ax) == []
        assert "no data yet" in _panel_text(ax)
        assert ax.get_xlabel() and ax.get_ylabel()


def test_plot_5_caption_no_longer_describes_the_200m_branches_as_missing(figures):
    texts = _normalised(" ".join(all_texts(figures["05_ladder.png"])))
    assert "not in analyse() output" not in texts and "drawn but not fitted" not in texts


def test_every_axes_of_every_plot_keeps_its_dots_visible(figures):
    for figure in figures.values():
        for ax in figure.axes:
            assert_visible_dots(ax)


# --- Plot 1: the numbers, the verdict, the band and the second axis ----------------------------------


def test_plot_1_paired_dots_are_the_comparison_differences(figures, inputs):
    figure = figures["01_primary_gaps.png"]
    for name in PRIMARY:
        comp = inputs.analysis.comparisons[name]
        assert comp.test == pt.PAIRED
        ax = _primary_panel(figure, name, inputs.prereg)
        xs = sorted(x for x, _ in plotted_points(ax))
        np.testing.assert_allclose(xs, sorted(comp.differences), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_1_shows_each_comparisons_verdict_text_exactly(figures, inputs):
    figure = figures["01_primary_gaps.png"]
    verdicts = set()
    for name in PRIMARY + EXPLORATORY:
        comp = inputs.analysis.comparisons[name]
        ax = _primary_panel(figure, name, inputs.prereg)
        assert ax.get_title().split("\n")[-1] == comp.verdict
        verdicts.add(comp.verdict)
    assert verdicts <= {pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED, pt.VERDICT_EXPLORATORY}
    assert {pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED, pt.VERDICT_EXPLORATORY} == verdicts


def test_plot_1_welch_panels_centre_both_groups_on_the_mha_mean(welch_inputs):
    figure = m.plot_primary_gaps(welch_inputs)
    for name in PRIMARY:
        comp = welch_inputs.analysis.comparisons[name]
        assert comp.test == pt.WELCH
        ax = _primary_panel(figure, name, welch_inputs.prereg)
        mean_ref = float(np.mean(comp.reference_losses))
        # Each dot belongs to the nearer of the two rows drawn: MHA at y = 1, the variant at y = 0.
        mha_row = sorted(x for x, y in plotted_points(ax) if abs(y - 1.0) < abs(y))
        variant_row = sorted(x for x, y in plotted_points(ax) if abs(y) <= abs(y - 1.0))
        np.testing.assert_allclose(sorted(x - mean_ref for x in comp.reference_losses), mha_row,
                                   rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
        np.testing.assert_allclose(sorted(x - mean_ref for x in comp.variant_losses), variant_row,
                                   rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
        # The interval is drawn from the comparison's own bounds.
        bounds = [line.get_xdata() for line in ax.lines if len(line.get_xdata()) == 2]
        assert any(np.allclose(b, [comp.ci_low, comp.ci_high], rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
                   for b in bounds), name


def _band_half_width(patch) -> float:
    """The half width of an axvspan band: a Rectangle from x0 to x0 + width (matplotlib's axvspan)."""
    assert isinstance(patch, Rectangle)
    return max(abs(patch.get_x()), abs(patch.get_x() + patch.get_width()))


def test_plot_1_band_is_the_pre_registered_detectable_gap(figures, inputs):
    decision = inputs.analysis.decision
    ax = _primary_panel(figures["01_primary_gaps.png"], "mqa", inputs.prereg)
    assert len(ax.patches) == 1, "the only patch of a primary panel is the detectable-effect band"
    np.testing.assert_allclose(_band_half_width(ax.patches[0]), decision.bound_gap,
                               rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_1_second_axis_maps_a_gap_to_its_multiplier(figures, inputs):
    b = inputs.analysis.fit.b
    figure = figures["01_primary_gaps.png"]
    ax = _primary_panel(figure, "mqa", inputs.prereg)
    canvas_draw(figure)
    top = ax.child_axes[0]
    ticks = [t for t in top.xaxis.get_ticklocs() if t > 0][:2]
    assert len(ticks) == 2
    for multiplier in ticks:
        gap = -b * np.log(multiplier)
        np.testing.assert_allclose(np.exp(-gap / b), multiplier, rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
        on_top = top.transData.transform((multiplier, 0.0))[0]
        on_bottom = ax.transData.transform((gap, 0.0))[0]
        np.testing.assert_allclose(on_top, on_bottom, rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_1_without_a_fit_has_no_second_axis_and_says_why(campaign):
    no_ladder = make_inputs_committed([x for x in campaign if "mha_L" not in x["name"]])
    assert no_ladder.analysis.fit is None
    figure = m.plot_primary_gaps(no_ladder)
    assert all(not ax.child_axes for ax in figure.axes)
    assert any("No top axis" in text for text in all_texts(figure))


def test_plot_1_without_a_decision_has_no_band_and_says_why(campaign):
    no_reruns = make_inputs_committed([
        x for x in campaign
        if not (PARSE(x["name"]).variant == "mha" and PARSE(x["name"]).tokens == "100M"
                and PARSE(x["name"]).branch is None and PARSE(x["name"]).data_seed == 1)
    ])
    assert no_reruns.analysis.decision is None
    figure = m.plot_primary_gaps(no_reruns)
    assert any("No detectable-effect band" in text for text in all_texts(figure))
    assert all(not ax.patches for ax in figure.axes)


# --- No statistics in the plot module ----------------------------------------------------------------


def test_the_no_data_note_is_the_literal_text():
    assert m.NO_DATA == "no data yet"


def test_every_primary_panel_has_one_band_and_no_exploratory_panel_has_one(figures, inputs):
    """Plan §1.2: exploratory variants make no detection claim, so they carry no detectable-effect band."""
    decision = inputs.analysis.decision
    figure = figures["01_primary_gaps.png"]
    for name in PRIMARY:
        ax = _primary_panel(figure, name, inputs.prereg)
        assert len(ax.patches) == 1, name
        np.testing.assert_allclose(_band_half_width(ax.patches[0]), decision.bound_gap,
                                   rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
    for name in EXPLORATORY:
        ax = _primary_panel(figure, name, inputs.prereg)
        assert len(ax.patches) == 0, name


def _constructed_300m_and_100m(order: str) -> list[dict]:
    """MHA and MQA at 300M (seeds 1, 2) and their 100M branches. The branches' bucket losses sit at
    different offsets (MQA +1, MHA -1 nat), so a branch standing in for a 300M run changes the gap."""
    manifests = fpi.bucket_manifests()
    for (variant, seed), losses in fpi.CONSTRUCTED_BUCKETS.items():
        offset = 1.0 if variant == "mqa" else -1.0
        shifted = tuple(x + offset for x in losses)
        final = float(np.mean(shifted))
        manifests.append(
            fake_manifest(variant, init_seed=seed, branch="100M", val_loss=final,
                          final_metrics={"val_loss": final, "val": fpi.val_block(final, shifted)})
        )
    return manifests if order == "forward" else list(reversed(manifests))


@pytest.mark.parametrize("order", ["forward", "reversed"])
def test_plot_6_uses_the_300m_runs_only_whatever_the_order(order):
    figure = m.plot_context_gap(make_inputs_committed(_constructed_300m_and_100m(order)))
    expected = []
    for seed in (1, 2):
        mha = fpi.CONSTRUCTED_BUCKETS[("mha", seed)]
        mqa = fpi.CONSTRUCTED_BUCKETS[("mqa", seed)]
        expected += [q - h for h, q in zip(mha, mqa)]
    gaps = sorted(y for ax in figure.axes for _, y in plotted_points(ax))
    assert len(gaps) == len(expected) == 8
    np.testing.assert_allclose(gaps, sorted(expected), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_arm_a_runs_refuses_two_300m_runs_for_one_variant_and_seed():
    prereg = config.load_prereg()
    mqa = fpi.bucket_manifests()[2]
    assert mqa["name"].startswith("mqa-")
    with pytest.raises(m.PlotError, match="claim mqa seed 1"):
        m._arm_a_runs([mqa, mqa], prereg)


def test_latency_legend_and_lines_agree_on_ref_dashed_and_opt_solid(figures, inputs):
    figure = figures["08_latency.png"]
    legend = next(ax.get_legend() for ax in figure.axes if ax.get_legend() is not None)
    styles = {t.get_text(): h.get_linestyle() for h, t in zip(legend.get_lines(), legend.get_texts())}
    assert styles["ref path"] == "--"
    assert styles["opt path"] == "-"
    panel = figure.axes[0]  # the prefill panel: one line per (variant, mode)
    tagged = [line for line in panel.lines if line.get_gid()]
    assert len(tagged) == 2 * len(mt.HEADLINE_VARIANTS)
    bench = {(r["variant"], r["mode"]): r for r in inputs.bench}
    for line in tagged:
        mode, variant = line.get_gid().split(":")
        assert line.get_linestyle() == ("--" if mode == "ref" else "-"), line.get_gid()
        expected = sorted((r["context"], r["latency_ms"]) for r in bench[(variant, mode)]["prefill"] if r["batch"] == 1)
        np.testing.assert_allclose(line.get_xdata(), [x for x, _ in expected], rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)
        np.testing.assert_allclose(line.get_ydata(), [y for _, y in expected], rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_the_plot_module_contains_no_statistics_of_its_own():
    source = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [name for name in imported if "scipy" in name]
    assert "analysis.ladder_fit" not in imported and "analysis.rqw_estimate" not in imported
    for word in ("ttest", "pvalue", "p_value", "welch_t", "paired_t", "holm(", "run_test("):
        assert word not in source, word
    # No verdict string is typed out: no string literal of the module names a verdict. The
    # constants are compared by name (pt.VERDICT_DETECTED), which is not a string literal.
    literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not [text for text in literals if "detected" in text.lower()]
    assert ".verdict" in source


def test_the_verdict_text_on_the_figure_is_the_comparisons_own(figures, inputs):
    """The only verdict text a plot prints is Comparison.verdict, one of the constants of primary_test.py."""
    source = Path(m.__file__).read_text(encoding="utf-8")
    for constant in (pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED):
        assert constant not in source
    titles = [ax.get_title().split("\n")[-1] for ax in figures["01_primary_gaps.png"].axes if ax.get_title()]
    comparisons = inputs.analysis.comparisons.values()
    assert {c.verdict for c in comparisons} <= set(titles) | {""}
    assert {c.verdict for c in comparisons} <= {pt.VERDICT_DETECTED, pt.VERDICT_NOT_DETECTED, pt.VERDICT_EXPLORATORY}


# --- Fixed order, labels and captions ---------------------------------------------------------------


def test_legends_keep_the_fixed_variant_order(figures, inputs):
    prereg = inputs.prereg
    headline = list(mt.HEADLINE_VARIANTS)
    assert legend_labels(figures["01_primary_gaps.png"]) == expected_legend(headline, prereg)
    assert legend_labels(figures["02_inference_frontier.png"]) == expected_legend(headline, prereg)
    assert legend_labels(figures["03_memory_frontier.png"]) == expected_legend(headline, prereg)
    assert legend_labels(figures["06_context_gap.png"]) == expected_legend(COMPARED_IN_GAP_PLOTS, prereg)
    assert legend_labels(figures["07_training_curves.png"]) == expected_legend(headline, prereg)
    assert legend_labels(figures["08_latency.png"])[:7] == expected_legend(headline, prereg)
    assert legend_labels(figures["09_probes.png"]) == expected_legend([*headline, m.PROBE_ANCHOR], prereg)
    assert legend_labels(figures["10_retrieval.png"])[:7] == expected_legend(headline, prereg)


def test_exploratory_variants_are_labelled_wherever_they_carry_a_gap(figures, inputs):
    for name in EXPLORATORY:
        ax = _primary_panel(figures["01_primary_gaps.png"], name, inputs.prereg)
        assert "exploratory" in ax.get_title()
    labels = legend_labels(figures["06_context_gap.png"])
    for name in EXPLORATORY:
        assert any(label.startswith(m.DISPLAY[name]) and "(exploratory)" in label for label in labels), name


@pytest.mark.parametrize(
    "name",
    ["01_primary_gaps.png", "02_inference_frontier.png", "03_memory_frontier.png",
     "06_context_gap.png", "07_training_curves.png"],
)
def test_loss_plots_carry_the_swa_caveat(figures, name):
    assert any(m.SWA_CAVEAT in text for text in all_texts(figures[name])), name


def test_latency_plot_carries_the_kernel_maturity_disclosure(figures):
    texts = all_texts(figures["08_latency.png"])
    assert any(m.KERNEL_DISCLOSURE in text for text in texts)
    assert any("newer or hand-written kernels" in text for text in texts)


def test_latency_plot_draws_ref_dashed_and_opt_solid(figures):
    ax = figures["08_latency.png"].axes[0]
    dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
    solid = [line for line in ax.lines if line.get_linestyle() == "-"]
    # One prefill line per variant and mode: seven dashed (ref), seven solid (opt).
    assert len(dashed) == len(solid) == len(mt.HEADLINE_VARIANTS)


def test_probe_plot_draws_the_anchor_black_and_dashed(figures):
    for ax in figures["09_probes.png"].axes:
        if not ax.get_title().startswith("probe:"):
            continue
        anchor = [
            line for line in ax.lines
            if line.get_linestyle() == "--" and to_rgb(line.get_color()) == (0.0, 0.0, 0.0)
        ]
        assert anchor, ax.get_title()


# --- Refusals and partial data ----------------------------------------------------------------------


def test_plots_refuse_incomparable_manifests(tmp_path, campaign):
    bad = copy.deepcopy(campaign)
    bad[0]["environment"].update(torch="2.16.0")
    with pytest.raises(mt.TableError, match="torch version"):
        m.build_all(bad, [], [], [], tmp_path)
    with pytest.raises(mt.TableError, match="torch version"):
        make_inputs_committed(bad)


def test_an_arm_n_only_stage_builds_every_plot_and_plot_4_shows_only_arm_n(tmp_path):
    base = fake_campaign(without=("primaries", "exploratory", "ladder", "ladder_l10"))
    stage = [x for x in base if PARSE(x["name"]).tokens == "100M" or PARSE(x["name"]).branch == "100M"]
    stage = fpi.with_eval_fields(stage)
    paths = m.build_all(stage, [], [], [], tmp_path)
    assert [p.name for p in paths] == list(EXPECTED_FILES)
    inp = make_inputs_committed(stage)
    figure = m.plot_noise_floor(inp)
    mha_branches = [float(x["final_metrics"]["val_loss"]) for x in stage
                    if PARSE(x["name"]).variant == "mha" and PARSE(x["name"]).branch == "100M"]
    # The data-order row also shows data seed 1: init seed 1's branch, a second time (16 dots).
    seed_1_branch = [float(x["final_metrics"]["val_loss"]) for x in stage
                     if PARSE(x["name"]).variant == "mha" and PARSE(x["name"]).branch == "100M"
                     and PARSE(x["name"]).init_seed == 1]
    wanted = sorted(mha_branches + _arm_n_losses(stage) + seed_1_branch)
    assert len(wanted) == 16
    drawn = sorted(x for x, _ in all_points(figure))
    np.testing.assert_allclose(drawn, wanted, rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_without_per_bucket_fields_plot_6_says_no_data_and_plot_7_says_no_curves():
    inp = make_inputs_committed(fake_campaign(gaps={"mqa": 0.05}))
    assert any("lack final_metrics.val" in text for text in all_texts(m.plot_context_gap(inp)))
    assert any("lack final_metrics.val_curve" in text for text in all_texts(m.plot_training_curves(inp)))


def test_plot_6_gaps_are_per_seed_against_mha_of_the_same_seed():
    figure = m.plot_context_gap(make_inputs_committed(fpi.bucket_manifests()))
    expected = []
    for seed in (1, 2):
        mha = fpi.CONSTRUCTED_BUCKETS[("mha", seed)]
        mqa = fpi.CONSTRUCTED_BUCKETS[("mqa", seed)]
        expected += [q - h for h, q in zip(mha, mqa)]
    gaps = sorted(y for ax in figure.axes for _, y in plotted_points(ax))
    np.testing.assert_allclose(gaps, sorted(expected), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_7_never_draws_a_trunk_point_as_a_result():
    inp = make_inputs_committed(fpi.curve_manifests())
    ax = m.plot_training_curves(inp).axes[0]
    # T-45 (G3): these constructed runs have no null control, so the gate is shut and MQA's run is
    # withheld; only MHA's two runs are drawn.
    results = sorted(
        (float(_tokens_of(x["name"])), float(x["final_metrics"]["val_loss"]))
        for x in inp.results if PARSE(x["name"]).variant == "mha"
    )
    dots = sorted((float(x), float(y)) for x, y in plotted_points(ax))
    assert dots == results
    # The faded trunk curves are the lines drawn with alpha below one; the dots are never lines.
    curve_points = [(float(x), float(y)) for line in ax.lines
                    if line.get_alpha() is not None and line.get_alpha() < 1.0
                    for x, y in zip(line.get_xdata(), line.get_ydata())]
    assert len(curve_points) == 3 + 2  # the curves of the 300M MHA and the 100M MHA branch; MQA's is withheld
    for point in curve_points:
        if point not in results:
            assert point not in dots


def test_a_partial_campaign_without_the_ladder_builds_and_says_the_fit_is_missing(tmp_path):
    partial = fpi.with_eval_fields(fake_campaign(without=("ladder", "ladder_l10")))
    assert len(m.build_all(partial, [], [], [], tmp_path)) == 10
    inp = make_inputs_committed(partial)
    assert inp.analysis.fit is None
    assert any("ladder" in note for note in inp.analysis.notes)


def test_missing_benchmark_is_no_data_not_an_error(campaign):
    figure = m.plot_inference_frontier(make_inputs_committed(campaign))
    assert dot_count(figure) == 0
    assert any("no data yet" in text for text in all_texts(figure))


# --- Command line -------------------------------------------------------------------------------------


def test_the_blank_command_line_writes_ten_files(tmp_path):
    assert plots_main(["--blank", "--out", str(tmp_path)]) == 0
    assert sorted(os.listdir(tmp_path)) == sorted(EXPECTED_FILES)


def test_the_command_line_with_missing_inputs_builds_blank_plots(tmp_path):
    code = plots_main([
        "--manifests", str(tmp_path / "none"), "--bench", str(tmp_path / "none"),
        "--probes", str(tmp_path / "none.json"), "--retrieval", str(tmp_path / "none.json"),
        "--out", str(tmp_path / "out"),
    ])
    assert code == 0
    assert sorted(os.listdir(tmp_path / "out")) == sorted(EXPECTED_FILES)


def test_the_command_line_refuses_incomparable_manifests(tmp_path, campaign, capsys):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    bad = copy.deepcopy(campaign)
    bad[0]["environment"].update(torch="2.16.0")
    for manifest in bad:
        mf.write_manifest(manifest, manifests)
    assert plots_main(["--manifests", str(manifests), "--out", str(tmp_path / "out")]) == 1
    assert "torch version" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# --- T-36: unfinished runs left out of the plots, by name (plan §8.1, §10.6) -------------------------


def _unfinished_run() -> dict:
    """A SYNTHETIC standalone 100M MHA run that has not finished: it would be a dot on plot 4."""
    return fake_manifest("mha", budget="100M", branches=[], init_seed=9, completed=False)


def _trunk_checkpoint() -> dict:
    """A SYNTHETIC trunk checkpoint, which no plot may report as a result."""
    return fake_manifest("mha", kind="trunk_checkpoint", completed=False, init_seed=9)


def test_an_unfinished_run_refuses_the_plots_without_the_flag(campaign, tmp_path):
    with pytest.raises(mt.TableError, match="unfinished"):
        m.build_all([*campaign, _unfinished_run()], fpi.bench_records(), fpi.probe_records(),
                    fpi.retrieval_records(), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_with_the_flag_all_ten_plots_build_and_the_unfinished_run_draws_no_dot(campaign, tmp_path):
    unfinished = _unfinished_run()
    bench, probes, retrieval = fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records()
    paths = m.build_all([*campaign, unfinished], bench, probes, retrieval, tmp_path / "out", skip_unfinished=True)
    assert [p.name for p in paths] == list(EXPECTED_FILES)
    assert all(p.exists() for p in paths)
    base = make_inputs_committed(campaign, bench, probes, retrieval)
    skipping = make_inputs_committed([*campaign, unfinished], bench, probes, retrieval, skip_unfinished=True)
    assert skipping.skipped == (unfinished["name"],)
    for name, builder in m.PLOTS:
        assert dot_count(builder(skipping)) == dot_count(builder(base)), name


def test_plot_4_names_the_unfinished_runs_left_out_and_only_when_some_were(campaign):
    unfinished = _unfinished_run()
    builder = dict(m.PLOTS)["04_noise_floor.png"]
    fig = builder(make_inputs_committed([*campaign, unfinished], skip_unfinished=True))
    assert any(f"unfinished runs left out: {unfinished['name']}" in text for text in all_texts(fig))
    plain = builder(make_inputs_committed(campaign))
    assert not any("unfinished runs left out" in text for text in all_texts(plain))


def test_the_command_line_refuses_unfinished_runs_without_the_flag_and_builds_with_it(campaign, tmp_path, capsys):
    manifests = tmp_path / "manifests"
    for manifest in [*campaign, _unfinished_run()]:
        mf.write_manifest(manifest, manifests)
    none = tmp_path / "none"
    common = ["--manifests", str(manifests), "--bench", str(none), "--probes", str(none / "p.json"),
              "--retrieval", str(none / "r.json")]
    assert plots_main([*common, "--out", str(tmp_path / "refused")]) == 1
    assert "unfinished" in capsys.readouterr().err
    assert not (tmp_path / "refused").exists()
    assert plots_main([*common, "--out", str(tmp_path / "built"), "--skip-unfinished"]) == 0
    assert sorted(os.listdir(tmp_path / "built")) == sorted(EXPECTED_FILES)


def test_a_trunk_checkpoint_is_refused_with_and_without_the_flag(campaign, tmp_path, capsys):
    manifests = [*campaign, _trunk_checkpoint()]
    bench, probes, retrieval = fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records()
    for skip in (False, True):
        with pytest.raises(mt.TableError, match="trunk checkpoint"):
            m.build_all(manifests, bench, probes, retrieval, tmp_path / "out", skip_unfinished=skip)
    assert not (tmp_path / "out").exists()
    trunk_dir = tmp_path / "trunk_manifests"
    for manifest in manifests:
        mf.write_manifest(manifest, trunk_dir)
    none = tmp_path / "none"
    common = ["--manifests", str(trunk_dir), "--bench", str(none), "--probes", str(none / "p.json"),
              "--retrieval", str(none / "r.json"), "--out", str(tmp_path / "cli_out")]
    assert plots_main(common) == 1
    assert plots_main([*common, "--skip-unfinished"]) == 1
    assert "trunk checkpoint" in capsys.readouterr().err
    assert not (tmp_path / "cli_out").exists()


# --- F5 (T-37): with the gate shut, plots 1, 2 and 3 draw no variant dot ----------------------

WITHHELD = " ".join(m.WITHHELD_CAPTION.split())
SHUT_CASES = {
    "no null control": dict(gaps={"mqa": 0.05}, without=("null",)),
    "a detected null control": dict(gaps={"mqa": 0.05}, null_gap=0.5),
}


def dots_by_variant(ax) -> dict[str, int]:
    """The seed dots of an axes, counted by the colour each variant is drawn in (the colours are fixed)."""
    names = [*mt.HEADLINE_VARIANTS, *config.ladder_names(), "null"]
    counts = {name: 0 for name in names}
    for collection in dot_collections(ax):
        colours = collection.get_facecolor()
        if len(colours) == 0:
            continue
        for name in names:
            if np.allclose(colours[0][:3], to_rgb(m._colour(name)), atol=1e-6):
                counts[name] += len(collection.get_offsets())
    return counts


def _caption_says_withheld(figure) -> bool:
    return any(WITHHELD in text for text in all_texts(figure))


@pytest.fixture(params=sorted(SHUT_CASES))
def shut(request):
    inputs_ = make_inputs_committed(
        fpi.with_eval_fields(fake_campaign(**SHUT_CASES[request.param])),
        fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records(),
    )
    assert inputs_.analysis.comparisons_allowed is False
    return inputs_


def test_the_dot_counter_sees_variant_dots_when_the_gate_is_open(inputs):
    counts = dots_by_variant(m.plot_inference_frontier(inputs).axes[0])
    assert counts["mqa"] == len(inputs.analysis.losses["mqa"]) > 0
    assert not _caption_says_withheld(m.plot_inference_frontier(inputs))


def test_with_the_gate_shut_plot_2_draws_mha_and_no_other_variant(shut):
    figure = m.plot_inference_frontier(shut)
    counts = dots_by_variant(figure.axes[0])
    assert counts["mha"] == len(shut.analysis.losses["mha"]) > 0
    assert {name: c for name, c in counts.items() if name != "mha"} == {n: 0 for n in counts if n != "mha"}
    assert _caption_says_withheld(figure)


def test_with_the_gate_shut_plot_3_draws_mha_and_no_other_variant(shut):
    figure = m.plot_memory_frontier(shut)
    counts = dots_by_variant(figure.axes[0])
    assert counts["mha"] == len(shut.analysis.losses["mha"]) > 0
    assert {name: c for name, c in counts.items() if name != "mha"} == {n: 0 for n in counts if n != "mha"}
    assert _caption_says_withheld(figure)


def test_with_the_gate_shut_plot_1_draws_no_dot_and_says_the_variant_runs_are_withheld(shut):
    figure = m.plot_primary_gaps(shut)
    assert dot_count(figure) == 0
    assert _caption_says_withheld(figure)


def test_plot_1_draws_the_variant_dots_when_the_gate_is_open(inputs):
    figure = m.plot_primary_gaps(inputs)
    assert dot_count(figure) > 0
    assert not _caption_says_withheld(figure)


def test_the_withheld_caption_is_absent_when_no_variant_run_exists():
    blank = make_inputs_committed([])
    for builder in (m.plot_primary_gaps, m.plot_inference_frontier, m.plot_memory_frontier):
        assert not _caption_says_withheld(builder(blank)), builder.__name__


# --- T-38: plot 9 draws the no-recall level 1/n, labelled as token accuracy -----------------------

CHANCE_GREY = (0.5, 0.5, 0.5)
CHANCE_CAPTION = "The grey dotted line is the no-recall level 1/n."


def probe_panels(figure: Figure) -> list:
    return [ax for ax in figure.axes if ax.get_title().startswith("probe:")]


def chance_lines(ax) -> list:
    """The grey dotted lines of a panel: the chance level, drawn by plot 9 and nothing else."""
    return [
        line for line in ax.lines
        if line.get_linestyle() == ":" and to_rgb(line.get_color()) == CHANCE_GREY
    ]


def test_every_probe_panel_draws_one_chance_line_at_the_true_difficulties(figures):
    panels = probe_panels(figures["09_probes.png"])
    assert len(panels) == len(fpi.PROBE_TASKS)
    for ax in panels:
        (line,) = chance_lines(ax)
        points = [(float(x), float(y)) for x, y in line.get_xydata()]
        assert points == [(float(d), 1 / d) for d in fpi.PROBE_DIFFICULTIES], ax.get_title()


def test_probe_panels_draw_no_chance_line_when_no_record_carries_chance(campaign):
    records = [{key: value for key, value in r.items() if key != "chance"} for r in fpi.probe_records()]
    figure = m.plot_probes(make_inputs_committed(campaign, fpi.bench_records(), records, fpi.retrieval_records()))
    panels = probe_panels(figure)
    assert len(panels) == len(fpi.PROBE_TASKS)
    for ax in panels:
        assert chance_lines(ax) == [], ax.get_title()


@pytest.mark.parametrize("other", [0.5, None])
def test_records_at_one_point_that_disagree_on_chance_are_refused(campaign, other):
    records = fpi.probe_records()
    records[0] = {**records[0], "chance": other}  # the first mqar record, at difficulty 4
    inputs_ = make_inputs_committed(campaign, fpi.bench_records(), records, fpi.retrieval_records())
    with pytest.raises(m.PlotError, match="disagree on chance"):
        m.plot_probes(inputs_)


def test_duplicate_probe_records_are_refused_by_make_inputs(campaign):
    records = fpi.probe_records()
    records.append(dict(records[0]))  # same variant, task, difficulty and init seed
    with pytest.raises(m.PlotError, match="two records"):
        make_inputs_committed(campaign, fpi.bench_records(), records, fpi.retrieval_records())


def test_plot_9_is_labelled_as_token_accuracy_on_every_panel(figures, blank_figures):
    panels = probe_panels(figures["09_probes.png"])
    assert panels
    for ax in panels:
        assert ax.get_ylabel() == "token accuracy (held-out)", ax.get_title()
    assert "token accuracy (held-out)" in [ax.get_ylabel() for ax in blank_figures["09_probes.png"].axes]


def test_plot_9_caption_explains_the_grey_dotted_line(figures, blank_figures):
    assert any(CHANCE_CAPTION in text for text in all_texts(figures["09_probes.png"]))
    assert any(CHANCE_CAPTION in text for text in all_texts(blank_figures["09_probes.png"]))


# --- T-45 (G3): plot 7 draws the Arm A runs of the headline variants and nothing else -------------
# Every manifest here is synthetic. The expected dots come from the run names (see _plot_7_names),
# never from the plot code.

def _plot_7_names(manifests) -> list[str]:
    """The names plot 7 must draw, read from the run names alone: the headline variants' 300M results
    and their 100M and 200M branches, in Arm A, at each variant's own learning rate (plan §8.1)."""
    arm = config.load_arm(mt.ARM)
    parsed = [(x["name"], PARSE(x["name"])) for x in manifests]
    arm_a = [
        (name, p) for name, p in parsed
        if p.arm == arm["tag"] and p.data_seed == arm["data_seed"] and p.variant in mt.HEADLINE_VARIANTS
        and p.tokens == "300M" and p.branch in (None, "100M", "200M")
    ]
    rates: dict[str, set] = {}
    for _, p in arm_a:
        if p.branch is None:
            rates.setdefault(p.variant, set()).add(p.peak_lr)
    assert all(len(r) == 1 for r in rates.values()), rates
    return [name for name, p in arm_a if p.peak_lr in rates[p.variant]]


def _plotted_curve_lines(ax) -> list[tuple[float, float, float]]:
    """(colour as RGB, x, y) of every faded curve line of the axes (alpha below one)."""
    return sorted(
        (tuple(round(c, 6) for c in to_rgb(line.get_color())), float(x), float(y))
        for line in ax.lines if line.get_alpha() is not None and line.get_alpha() < 1.0
        for x, y in zip(line.get_xdata(), line.get_ydata())
    )


def _plot_7_points(manifests_by_name, names) -> list[tuple[float, float]]:
    return sorted((float(_tokens_of(n)), float(manifests_by_name[n]["final_metrics"]["val_loss"])) for n in names)


def _mha_names(manifests) -> list[str]:
    return [n for n in _plot_7_names(manifests) if PARSE(n).variant == "mha"]


def test_plot_7_draws_exactly_the_arm_a_headline_runs_and_branches(campaign):
    inp = make_inputs_committed(campaign)
    ax = m.plot_training_curves(inp).axes[0]
    names = _plot_7_names(campaign)
    by_name = {x["name"]: x for x in campaign}
    assert names and len(names) == len(set(names))
    assert sorted((float(x), float(y)) for x, y in plotted_points(ax)) == _plot_7_points(by_name, names)


def test_sweep_arm_b_rerun_null_and_ladder_runs_add_no_dot_and_no_line():
    base = fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}))  # the gate is open: controls pass
    assert make_inputs_committed(base).analysis.comparisons_allowed
    arm_b = fsec.arm_b_manifests(base, variants=("gqa4", "mqa"), seeds=(1,))  # Arm B needs 100M branches
    extras = [
        fake_manifest("mha", budget="100M", branches=[], init_seed=9, val_loss=3.3),  # a rerun, unregistered seed
        fake_manifest("mha", budget="100M", branches=[], init_seed=1, data_seed=4, val_loss=3.3),  # data order
        fake_manifest("null", budget="100M", branches=[], init_seed=16, val_loss=3.5),  # a null run, unregistered seed
        fake_manifest("mha", ladder="mha_L6", init_seed=9, val_loss=3.1),  # a ladder run
        fake_manifest("mqa", init_seed=1, budget="100M", branches=[], peak_lr=3e-4),  # a learning-rate sweep point
        *arm_b,
    ]
    parsed = [PARSE(x["name"]) for x in extras]
    assert any(p.arm == config.load_arm(mt.ARM_B)["tag"] for p in parsed)  # Arm B
    assert any(p.variant == "null" for p in parsed) and any(p.variant.startswith("mha_L") for p in parsed)
    assert any(p.peak_lr == 3e-4 for p in parsed)  # the sweep point
    assert sum(p.variant == "mha" and p.tokens == "100M" and p.branch is None for p in parsed) == 2  # reruns
    names = {x["name"] for x in base}
    assert all(x["name"] not in names for x in extras)  # every addition is a new run

    extended = fpi.with_eval_fields(extras)
    base_inputs = make_inputs_committed(base)
    extended_inputs = make_inputs_committed([*base, *extended])
    assert _plot_7_names(extended_inputs.results) == _plot_7_names(base_inputs.results)
    base_ax = m.plot_training_curves(base_inputs).axes[0]
    extended_ax = m.plot_training_curves(extended_inputs).axes[0]
    assert len(plotted_points(base_ax)) == len(_plot_7_names(base)) > 0
    assert sorted(plotted_points(extended_ax)) == sorted(plotted_points(base_ax))
    assert _plotted_curve_lines(extended_ax) == _plotted_curve_lines(base_ax)


@pytest.mark.parametrize("case", sorted(SHUT_CASES))
def test_with_the_gate_shut_plot_7_draws_mha_only_and_says_the_variants_are_withheld(case):
    shut_inputs = make_inputs_committed(fpi.with_eval_fields(fake_campaign(**SHUT_CASES[case])))
    assert shut_inputs.analysis.comparisons_allowed is False
    figure = m.plot_training_curves(shut_inputs)
    ax = figure.axes[0]
    expected_mha = _mha_names(shut_inputs.results)
    assert expected_mha
    counts = dots_by_variant(ax)
    assert counts["mha"] == len(expected_mha) == dot_count(figure)
    assert {name: c for name, c in counts.items() if name != "mha"} == {n: 0 for n in counts if n != "mha"}
    mha_colour = to_rgb(m._colour("mha"))
    curve_colours = {to_rgb(line.get_color()) for line in ax.lines if line.get_alpha() is not None and line.get_alpha() < 1.0}
    assert curve_colours == {mha_colour}
    assert _caption_says_withheld(figure)


def test_with_the_gate_open_plot_7_draws_the_variants_and_no_withheld_caption(campaign):
    inp = make_inputs_committed(campaign)
    assert inp.analysis.comparisons_allowed
    figure = m.plot_training_curves(inp)
    counts = dots_by_variant(figure.axes[0])
    assert counts["mqa"] > 0 and counts["mha"] > 0
    assert not _caption_says_withheld(figure)


def test_plot_7_caption_states_the_drawing_rule_in_one_sentence(campaign, blank_figures):
    rule = (
        "Drawn: the Arm A runs and branches of the headline variants at each variant's learning rate; "
        "learning-rate sweep points and other arms are not drawn."
    )
    assert any(rule in text for text in all_texts(m.plot_training_curves(make_inputs_committed(campaign))))
    assert any(rule in text for text in all_texts(blank_figures["07_training_curves.png"]))


# --- T-45 (G8): the plots refuse bench records that the table refuses ----------------------------


def _agreeing_bench() -> list[dict]:
    """bench/1 documents of every headline variant in both modes, recorded as the synthetic campaign's environment."""
    return [fsec.bench_document(v, mode) for v in fpi.BENCH_VARIANTS for mode in ("opt", "ref")]


def _bench_with(variant: str, mode: str = "opt", **changes) -> list[dict]:
    """The agreeing documents, with the one of (variant, mode) recorded with `changes`."""
    first = fsec.bench_document(variant, mode, **changes)
    return [first] + [d for d in _agreeing_bench() if (d["variant"], d["mode"]) != (variant, mode)]


@pytest.mark.parametrize(
    "bench, reason",
    [
        (_bench_with("mha", shared=True), "gpu_shared True"),
        (_bench_with("mha", environment={**fsec.ENVIRONMENT, "driver": "998.0"}), "driver version"),
        (_bench_with("mha", environment={**fsec.ENVIRONMENT, "gpu_name": "Tesla P100-PCIE-16GB"}), "GPU name"),
        (_bench_with("gqa4", backend="math"), "backend"),
    ],
    ids=["shared GPU", "other driver", "other GPU", "other backend"],
)
def test_make_inputs_refuses_bench_records_the_table_refuses(campaign, bench, reason):
    with pytest.raises(mt.TableError, match=reason):
        make_inputs_committed(campaign, bench)


def test_the_command_line_refuses_a_shared_gpu_bench_record_and_exits_one(tmp_path, campaign, capsys):
    manifests, bench_dir = tmp_path / "manifests", tmp_path / "bench"
    for manifest in campaign:
        mf.write_manifest(manifest, manifests)
    fsec.write_bench_documents(bench_dir, _bench_with("mqa", shared=True))
    code = plots_main([
        "--manifests", str(manifests), "--bench", str(bench_dir), "--out", str(tmp_path / "plots"),
        "--probes", str(tmp_path / "none.json"), "--retrieval", str(tmp_path / "none.json"),
    ])
    assert code == 1
    assert "gpu_shared True" in capsys.readouterr().err
    assert not (tmp_path / "plots").exists()


def test_agreeing_bench_records_still_build_all_ten_plots(tmp_path, campaign):
    paths = m.build_all(campaign, _agreeing_bench(), fpi.probe_records(), fpi.retrieval_records(), tmp_path)
    assert [p.name for p in paths] == list(EXPECTED_FILES)
    for path in paths:
        assert path.read_bytes()[:8] == PNG_SIGNATURE


def test_the_synthetic_bench_records_agree_with_the_synthetic_campaign(campaign):
    # T-45 (G8): the fixture's records carry the campaign's environment and backend, so the table's
    # timing-source checks accept them. A disagreement raises TableError here.
    built = make_inputs_committed(campaign, fpi.bench_records())
    assert len(built.bench) == len(fpi.bench_records()) > 0


# --- T-44: the plots take the tables' declaration that L10 is cut ---------------------------------


def _without_l10(campaign_):
    return [x for x in campaign_ if mf.parse_run_name(x["name"]).variant != "mha_L10"]


def test_without_l10_the_plots_have_no_seed_count_until_the_cut_is_declared(campaign):
    """Plan §12.2 item 4, review S4 H2: absent L10 runs are not a decision to cut them."""
    runs = _without_l10(campaign)
    assert len(runs) < len(campaign)
    waiting = make_inputs_committed(runs, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records())
    cut = committed(runs, ladder="l6_l8")
    declared = make_inputs_committed(
        runs, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records(), commitments=cut
    )
    assert waiting.analysis.decision is None and waiting.analysis.comparisons == {}
    assert declared.analysis.decision is not None
    assert declared.analysis == mt.analyse(list(runs), commitments=cut)


def test_the_command_line_passes_the_commitments_file_to_the_analysis(campaign, tmp_path, monkeypatch, capsys):
    directory = tmp_path / "manifests"
    for manifest in _without_l10(campaign):
        mf.write_manifest(manifest, directory)
    seen = []
    real = mt.analyse

    def spy(results, prereg=None, **kwargs):
        seen.append(kwargs.get("commitments"))
        return real(results, prereg, **kwargs)

    monkeypatch.setattr(mt, "analyse", spy)
    none = tmp_path / "none"
    common = ["--manifests", str(directory), "--bench", str(none), "--probes", str(none / "p.json"),
              "--retrieval", str(none / "r.json")]
    cut = {"ladder": "l6_l8", "seed_count": 5}
    file = write_commitments(tmp_path / "commitments.yaml", cut)
    assert m.main([*common, "--out", str(tmp_path / "a")]) == 0
    assert m.main([*common, "--out", str(tmp_path / "b"), "--commitments", str(file)]) == 0
    capsys.readouterr()
    # Without the flag the plots read configs/commitments.yaml, like the tables.
    assert seen == [config.load_commitments(), cut]
    with pytest.raises(SystemExit):
        m.main([*common, "--out", str(tmp_path / "c"), "--l10-cut"])  # the flag is gone (decision 220)
    capsys.readouterr()
