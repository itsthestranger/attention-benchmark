"""T-28: the ten plots, built blank and from synthetic inputs (plan §10.5, §13.15, §14.2).

Every manifest and record here is SYNTHETIC (tests/fixtures/fake_campaign.py and
tests/fixtures/fake_plot_inputs.py). The tests inspect the Figure objects: which
points each axes draws, which legend entries and captions it carries. Expected
counts come from the manifests and from `analyse`, never from the plot code.
"""

from __future__ import annotations

import ast
import copy
import os
import re
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
from src import config
from tests.fixtures.fake_campaign import fake_campaign
from tests.fixtures.fake_manifests import fake_manifest
from tests.tolerances import FLOAT64_IDENTITY

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
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}) + fpi.arm_n_extras())


@pytest.fixture(scope="module")
def welch_campaign() -> list[dict]:
    """Welch's test selected: the reproducibility reruns are noisier than the threshold allows."""
    return fpi.with_eval_fields(fake_campaign(gaps={"mqa": 0.05}, rerun_sigma=0.008) + fpi.arm_n_extras())


@pytest.fixture(scope="module")
def inputs(campaign):
    return m.make_inputs(campaign, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records())


@pytest.fixture(scope="module")
def welch_inputs(welch_campaign):
    return m.make_inputs(welch_campaign, fpi.bench_records(), fpi.probe_records(), fpi.retrieval_records())


@pytest.fixture(scope="module")
def figures(inputs) -> dict[str, Figure]:
    return {name: builder(inputs) for name, builder in m.PLOTS}


@pytest.fixture(scope="module")
def blank_figures() -> dict[str, Figure]:
    blank = m.make_inputs([])
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
    # 5 MHA branches (init seeds 1-5), 3 reruns, 2 data-order runs, 5 null-control runs.
    assert len(a.losses_100m["mha"]) == 5 and len(a.reruns_100m) == 3 and len(arm_n) == 10
    figure = figures["04_noise_floor.png"]
    assert dot_count(figure) == 15
    drawn = sorted(x for x, _ in all_points(figure))
    wanted = sorted(list(a.losses_100m["mha"].values()) + arm_n)
    np.testing.assert_allclose(drawn, wanted, rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_5_draws_every_ladder_seed_at_300m_and_every_100m_branch(figures, inputs):
    a = inputs.analysis
    at_300 = sum(len(a.losses.get(name, {})) for name in m.LADDER)
    at_100 = sum(len(a.losses_100m.get(name, {})) for name in m.LADDER)
    # 300M: L6 (5 seeds), MHA (5), L10 (3). 100M: the MHA branches (5); the fixture has no ladder branches.
    assert (at_300, at_100) == (13, 5)
    assert dot_count(figures["05_ladder.png"]) == at_300 + at_100 == 18
    assert_visible_dots(figures["05_ladder.png"].axes[0])


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


def test_plot_7_draws_every_result_as_a_final_dot(figures, inputs):
    assert dot_count(figures["07_training_curves.png"]) == len(inputs.results) == 67


def test_plot_8_draws_lines_and_no_dots(figures):
    assert dot_count(figures["08_latency.png"]) == 0


def test_plot_9_draws_every_probe_record_as_a_dot(figures, inputs):
    assert dot_count(figures["09_probes.png"]) == len(inputs.probes)
    for ax in figures["09_probes.png"].axes:
        if ax.get_title().startswith("probe:"):
            assert_visible_dots(ax)


def test_plot_10_draws_every_retrieval_record_as_a_dot(figures, inputs):
    assert dot_count(figures["10_retrieval.png"]) == len(inputs.retrieval)


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
    no_ladder = m.make_inputs([x for x in campaign if "mha_L" not in x["name"]])
    assert no_ladder.analysis.fit is None
    figure = m.plot_primary_gaps(no_ladder)
    assert all(not ax.child_axes for ax in figure.axes)
    assert any("No top axis" in text for text in all_texts(figure))


def test_plot_1_without_a_decision_has_no_band_and_says_why(campaign):
    no_reruns = m.make_inputs([
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
    figure = m.plot_context_gap(m.make_inputs(_constructed_300m_and_100m(order)))
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
    assert re.search(r"detected", source, re.IGNORECASE) is None
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
        m.make_inputs(bad)


def test_an_arm_n_only_stage_builds_every_plot_and_plot_4_shows_only_arm_n(tmp_path):
    base = fake_campaign(without=("primaries", "exploratory", "ladder", "ladder_l10"))
    stage = [x for x in base if PARSE(x["name"]).tokens == "100M" or PARSE(x["name"]).branch == "100M"]
    stage = fpi.with_eval_fields(stage + fpi.arm_n_extras())
    paths = m.build_all(stage, [], [], [], tmp_path)
    assert [p.name for p in paths] == list(EXPECTED_FILES)
    inp = m.make_inputs(stage)
    figure = m.plot_noise_floor(inp)
    mha_branches = [float(x["final_metrics"]["val_loss"]) for x in stage
                    if PARSE(x["name"]).variant == "mha" and PARSE(x["name"]).branch == "100M"]
    wanted = sorted(mha_branches + _arm_n_losses(stage))
    assert len(wanted) == 15
    drawn = sorted(x for x, _ in all_points(figure))
    np.testing.assert_allclose(drawn, wanted, rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_without_per_bucket_fields_plot_6_says_no_data_and_plot_7_says_no_curves():
    inp = m.make_inputs(fake_campaign(gaps={"mqa": 0.05}))
    assert any("lack final_metrics.val" in text for text in all_texts(m.plot_context_gap(inp)))
    assert any("lack final_metrics.val_curve" in text for text in all_texts(m.plot_training_curves(inp)))


def test_plot_6_gaps_are_per_seed_against_mha_of_the_same_seed():
    figure = m.plot_context_gap(m.make_inputs(fpi.bucket_manifests()))
    expected = []
    for seed in (1, 2):
        mha = fpi.CONSTRUCTED_BUCKETS[("mha", seed)]
        mqa = fpi.CONSTRUCTED_BUCKETS[("mqa", seed)]
        expected += [q - h for h, q in zip(mha, mqa)]
    gaps = sorted(y for ax in figure.axes for _, y in plotted_points(ax))
    np.testing.assert_allclose(gaps, sorted(expected), rtol=FLOAT64_IDENTITY, atol=FLOAT64_IDENTITY)


def test_plot_7_never_draws_a_trunk_point_as_a_result():
    inp = m.make_inputs(fpi.curve_manifests())
    ax = m.plot_training_curves(inp).axes[0]
    results = sorted((float(_tokens_of(x["name"])), float(x["final_metrics"]["val_loss"])) for x in inp.results)
    dots = sorted((float(x), float(y)) for x, y in plotted_points(ax))
    assert dots == results
    # The faded trunk curves are the lines drawn with alpha below one; the dots are never lines.
    curve_points = [(float(x), float(y)) for line in ax.lines
                    if line.get_alpha() is not None and line.get_alpha() < 1.0
                    for x, y in zip(line.get_xdata(), line.get_ydata())]
    assert len(curve_points) == 3 + 2 + 2  # the curves of the 300M MHA, the 100M MHA branch and 300M MQA
    for point in curve_points:
        if point not in results:
            assert point not in dots


def test_a_partial_campaign_without_the_ladder_builds_and_says_the_fit_is_missing(tmp_path):
    partial = fpi.with_eval_fields(fake_campaign(without=("ladder", "ladder_l10")) + fpi.arm_n_extras())
    assert len(m.build_all(partial, [], [], [], tmp_path)) == 10
    inp = m.make_inputs(partial)
    assert inp.analysis.fit is None
    assert any("ladder" in note for note in inp.analysis.notes)


def test_missing_benchmark_is_no_data_not_an_error(campaign):
    figure = m.plot_inference_frontier(m.make_inputs(campaign))
    assert dot_count(figure) == 0
    assert any("no data yet" in text for text in all_texts(figure))


# --- Command line -------------------------------------------------------------------------------------


def test_the_blank_command_line_writes_ten_files(tmp_path):
    assert m.main(["--blank", "--out", str(tmp_path)]) == 0
    assert sorted(os.listdir(tmp_path)) == sorted(EXPECTED_FILES)


def test_the_command_line_with_missing_inputs_builds_blank_plots(tmp_path):
    code = m.main([
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
    assert m.main(["--manifests", str(manifests), "--out", str(tmp_path / "out")]) == 1
    assert "torch version" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
