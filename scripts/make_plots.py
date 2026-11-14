"""The ten plots of plan §10.5, built from the same manifests as the headline table.

    python -m scripts.make_plots --manifests results/manifests --bench results/bench \
        --probes results/probes.json --retrieval results/retrieval.json --out results/plots
    python -m scripts.make_plots --blank --out results/plots/blank

Every number on a plot comes from `scripts.make_tables.analyse` (for the
statistics) or from the inputs (for everything else). This module computes no
statistic and decides no verdict: the verdict text on plot 1 is
`Comparison.verdict` from `analysis/primary_test.py`, printed as it is. Every
seed is a dot; means and intervals are added to the dots, never used in their
place. Blank inputs give blank plots with the same titles, axes and legends.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator, NullLocator

from analysis import primary_test as pt
from scripts import bench as bd
from scripts import make_tables as mt
from src import config
from src.model import build_model
from src.utils import manifest as mf

DPI = 120
BENCH_SCHEMA = "bench/2"
FRONTIER_BATCH = 32
FRONTIER_CACHE = 4096
RETRIEVAL_MODES = ("zero_shot", "trained")
# The lengths of a blank retrieval panel's axis only; with data, the ticks are the lengths present.
RETRIEVAL_LENGTHS = (256, 512, 1024, 2048)
RETRIEVAL_TASKS = ("passkey", "needle")
PROBE_ANCHOR = "recurrence"
# The ladder points (plan §8.4): L6, the 8-layer MHA reference and L10.
LADDER = ("mha_L6", "mha", "mha_L10")

SCATTER_SIZE = 42  # points squared; every dot in every plot
SCATTER_ALPHA = 0.8
MEAN_MARKER_SIZE = math.sqrt(SCATTER_SIZE)
LINE_WIDTH = 1.8
NO_DATA = "no data yet"
CHARS_PER_INCH = 13
LEGEND_COLUMNS = 10

SWA_CAVEAT = (
    "SWA-256 at context 1,024 is nearly uninformative: a 256-token window drops little at "
    "this length, and its cost and benefit appear at far longer contexts (plan §13.11)."
)
KERNEL_DISCLOSURE = (
    "MLA's and SWA's opt paths are newer or hand-written kernels, not architecture; ref and "
    "opt are both drawn so readers can separate the implementation share (plan §13.3)."
)
# Decisions 225 and 245: the speed values are medians; the whiskers on plots 2 and 8 span the
# smallest and the largest repetition. The numbers are in inference.md and the headline table.
SPEED_MEDIAN_CAPTION = (
    "Speed values are medians of the inference benchmark's repetitions; the whiskers span the smallest and the "
    "largest repetition, and a whisker shorter than its marker is hidden by it. The numbers are in inference.md "
    "and the headline table (decisions 225 and 245)."
)
WHISKER_WIDTH = 1.2
WITHHELD_CAPTION = (
    "Variant runs are withheld until the controls pass and the seed count is committed (plan §1.4, "
    "§8.1): only MHA's runs are drawn."
)

DISPLAY = {
    "mha": "MHA",
    "gqa4": "GQA-4",
    "mqa": "MQA",
    "mla192": "MLA-192",
    "gqa2": "GQA-2",
    "mla96": "MLA-96",
    "swa256": "SWA-256",
    PROBE_ANCHOR: "recurrent anchor",
    "mha_L6": "MHA, 6 layers (L6)",
    "mha_L10": "MHA, 10 layers (L10)",
    "null": "null control (MHA renamed)",
}
# One fixed colour per variant, colour-blind safe (Okabe-Ito), MHA neutral. Used by every plot.
COLOURS = {
    "mha": "#333333",
    "gqa4": "#0072B2",
    "mqa": "#E69F00",
    "mla192": "#009E73",
    "gqa2": "#56B4E9",
    "mla96": "#CC79A7",
    "swa256": "#D55E00",
    PROBE_ANCHOR: "#000000",
    "mha_L6": "#9E9E9E",
    "mha_L10": "#616161",
    "null": "#7B3294",
}
FALLBACK_COLOUR = "#777777"


class PlotError(ValueError):
    """An input cannot be plotted. Refusals from the table (mt.TableError) propagate unchanged."""


@dataclass(frozen=True)
class PlotInputs:
    results: tuple  # completed runs and branches, from select_results
    analysis: mt.Analysis  # from analyse(results): the only source of statistics
    bench: tuple  # bench/2 records
    probes: tuple  # {variant, task, difficulty, init_seed, accuracy}
    retrieval: tuple  # {variant, task, mode, length, n_distractors, init_seed, accuracy}
    prereg: Mapping
    skipped: tuple = ()  # names of unfinished runs left out (--skip-unfinished); empty when none


# --- Inputs ------------------------------------------------------------------------


def _checked(records, keys: Sequence[str], what: str) -> tuple:
    out = []
    for record in records:
        if not isinstance(record, Mapping):
            raise PlotError(f"{what}: a record must be an object, got {record!r}")
        missing = [key for key in keys if key not in record]
        if missing:
            raise PlotError(f"{what}: a record lacks {missing}: {dict(record)!r}")
        out.append(dict(record))
    return tuple(out)


def _checked_bench(records) -> tuple:
    out = []
    seen = set()
    for record in _checked(records, ("schema", "variant", "mode"), "bench"):
        if record["schema"] != BENCH_SCHEMA:
            raise PlotError(f"bench: schema {record['schema']!r}, expected {BENCH_SCHEMA!r}")
        key = (record["variant"], record["mode"])
        if key in seen:
            raise PlotError(f"bench: two records for variant {key[0]!r} in mode {key[1]!r}")
        seen.add(key)
        out.append(record)
    # The same refusals as the table's bench documents (bench.check_documents): a document the report
    # refuses draws no point.
    try:
        bd.check_documents(out)
    except bd.BenchError as exc:
        raise PlotError(str(exc)) from exc
    return tuple(out)


def _checked_probes(records) -> tuple:
    out = _checked(records, ("variant", "task", "difficulty", "init_seed", "accuracy"), "probes")
    seen = set()
    for record in out:
        key = (record["variant"], record["task"], record["difficulty"], record["init_seed"])
        if key in seen:
            raise PlotError(
                f"probes: two records for variant {key[0]!r}, task {key[1]!r}, difficulty {key[2]!r}, "
                f"init seed {key[3]!r}"
            )
        seen.add(key)
    return out


def make_inputs(manifests, bench=(), probes=(), retrieval=(), *, prereg: Mapping | None = None,
                skip_unfinished: bool = False, commitments: Mapping | None = None) -> PlotInputs:
    """Select, analyse and check the inputs. TableError from the table propagates.

    With skip_unfinished, unfinished runs are left out by name (as in make_tables); the names
    are kept on the inputs so that the plots that draw manifests can say which were left out.
    The bench records go through the same refusals as the table's bench documents
    (make_tables.check_timing_sources): a shared GPU, or another torch, CUDA, driver, GPU or backend.
    `commitments` is configs/commitments.yaml's {"ladder", "seed_count"} (decision 220; None reads the
    repository's file): the plots must be built with the same declarations as the tables, or they
    show no seed count.
    """
    prereg = config.load_prereg() if prereg is None else prereg
    results, skipped = mt.select_results(list(manifests), skip_unfinished=skip_unfinished)
    analysis = mt.analyse(results, prereg, commitments=commitments)
    bench = _checked_bench(bench)
    mt.check_timing_sources(results, None, list(bench))
    return PlotInputs(
        results=tuple(results),
        analysis=analysis,
        bench=bench,
        probes=_checked_probes(probes),
        retrieval=_checked(
            retrieval,
            ("variant", "task", "mode", "length", "n_distractors", "init_seed", "accuracy"),
            "retrieval",
        ),
        prereg=prereg,
        skipped=tuple(skipped),
    )


# --- Shared drawing helpers --------------------------------------------------------


def _colour(variant: str) -> str:
    return COLOURS.get(variant, FALLBACK_COLOUR)


def _label(variant: str, prereg: Mapping) -> str:
    name = DISPLAY.get(variant, variant)
    if variant in prereg["exploratory"]["variants"]:
        name += " (exploratory)"
    return name


def _handles(prereg: Mapping, variants: Sequence[str], *, line: bool = False) -> list[Line2D]:
    if line:
        return [Line2D([], [], color=_colour(v), lw=LINE_WIDTH, label=_label(v, prereg)) for v in variants]
    return [
        Line2D([], [], color=_colour(v), marker="o", linestyle="None", markersize=MEAN_MARKER_SIZE,
               label=_label(v, prereg))
        for v in variants
    ]


def _wrapped(parts: Sequence[str | None], width_in: float) -> list[str]:
    """Caption lines wrapped to the figure's width: about 13 characters per inch at 8 pt."""
    lines: list[str] = []
    for part in parts:
        if part:
            lines.extend(textwrap.wrap(part, width=max(20, int(width_in * CHARS_PER_INCH))))
    return lines


def _figure(size, *, title: str, rows: int = 1, cols: int = 1, handles=(), caption=()):
    """A figure with a legend row, the panel grid and a caption row. Returns (fig, panels)."""
    fig = Figure(figsize=size, layout="constrained")
    fig.suptitle(title, fontsize=12)
    lines = _wrapped(caption, size[0])
    ratios = []
    # Legend entries per row: as many as fit the width at 1.9 in each, never more than LEGEND_COLUMNS.
    ncol = max(1, min(len(handles), LEGEND_COLUMNS, int(size[0] // 1.9)))
    if handles:
        ratios.append(0.22 * math.ceil(len(handles) / ncol))
    ratios += [1.0] * rows
    if lines:
        ratios.append(0.06 + 0.13 * len(lines))
    grid = fig.add_gridspec(len(ratios), cols, height_ratios=ratios)
    top = 1 if handles else 0
    panels = [[fig.add_subplot(grid[top + r, c]) for c in range(cols)] for r in range(rows)]
    if handles:
        leg = fig.add_subplot(grid[0, :])
        leg.axis("off")
        leg.legend(handles=list(handles), loc="center", ncol=ncol, frameon=False, fontsize=8)
    if lines:
        cap = fig.add_subplot(grid[-1, :])
        cap.axis("off")
        # The text starts at the figure's left edge, whatever the panels' tick labels take, and
        # is wrapped to the figure's width; the row only reserves its height.
        left_of_figure = blended_transform_factory(fig.transFigure, cap.transAxes)
        cap.text(0.01, 1.0, "\n".join(lines), transform=left_of_figure, va="top", ha="left", fontsize=8)
    return fig, panels


def _no_data(ax, reason: str | None = None) -> None:
    text = NO_DATA if not reason else f"{NO_DATA}\n{textwrap.fill(reason, 48)}"
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=9, color="0.3")


def _dots(ax, xs, ys, colour: str, marker: str = "o"):
    """Seed dots. The only scatter artists on a plot: every dot is one seed."""
    return ax.scatter(
        list(xs), list(ys), s=SCATTER_SIZE, color=colour, alpha=SCATTER_ALPHA, marker=marker,
        edgecolors="white", linewidths=0.5, zorder=3,
    )


def _mean_mark(ax, x: float, y: float, colour: str) -> None:
    ax.plot([x], [y], marker="D", markersize=MEAN_MARKER_SIZE, linestyle="None", color=colour,
            markeredgecolor="black", markeredgewidth=0.6, zorder=4)


def _stagger(n: int, centre: float, spread: float = 0.12) -> list[float]:
    """Deterministic vertical offsets so that seeds at nearly the same value stay apart."""
    return [centre + spread * ((i % 3) - 1) for i in range(n)]


def _multiplier_axis(ax, b: float):
    """The top axis: gap in nats along the bottom, multiplier exp(-gap / b) along the top (plan §8.4)."""

    def forward(gap):
        return np.exp(-np.asarray(gap, dtype=float) / b)

    def inverse(multiplier):
        # Matplotlib asks for the axis limits too, which can reach a multiplier of 0: clip, do not warn.
        return -b * np.log(np.maximum(np.asarray(multiplier, dtype=float), 1e-300))

    top = ax.secondary_xaxis("top", functions=(forward, inverse))
    top.set_xlabel("multiplier exp(−gap / b)", fontsize=8)
    top.tick_params(labelsize=7)
    return top


def _reason(notes: Sequence[str], name: str) -> str | None:
    for note in notes:
        if name in note:
            return note
    return None


def _notes_part(notes: Sequence[str]) -> str | None:
    return ("Notes from the analysis: " + "; ".join(notes)) if notes else None


def _left_out_part(skipped: Sequence[str]) -> str | None:
    """One caption line naming the unfinished runs left out by --skip-unfinished, or None."""
    return ("unfinished runs left out: " + ", ".join(skipped)) if skipped else None


def _ladder_fit_usable(analysis: mt.Analysis) -> bool:
    """A multiplier axis may be drawn: the 300M fit exists with a positive b, and the rule's ladder
    slope is positive. Decision 223: with a slope that is not positive no multiplier is reported."""
    if analysis.decision is not None and analysis.decision.slope_not_positive:
        return False
    return analysis.fit is not None and analysis.fit.b > 0.0


def _withheld(analysis: mt.Analysis, variant: str) -> bool:
    """Plan §8.1, §1.4: while the gate is shut or the seed count is not committed, no variant's loss is
    drawn. MHA's loss is the reference, so it is always drawn."""
    return variant != "mha" and not mt.variant_results_shown(analysis)


def _withheld_part(analysis: mt.Analysis) -> str | None:
    """The caption line saying variant runs are withheld, when there is a variant run to withhold."""
    if mt.variant_results_shown(analysis):
        return None
    if any(analysis.losses.get(v) for v in mt.HEADLINE_VARIANTS if v != "mha"):
        return WITHHELD_CAPTION
    return None


# --- Model facts (no results) ------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _non_embedding(name: str) -> int:
    """Non-embedding parameters of a ladder point or of a headline variant, from its config."""
    if name in config.ladder_names():
        resolved = config.resolve("mha", ladder=name, init_seed=1, peak_lr=1e-3)
    else:
        resolved = config.resolve(name, init_seed=1, peak_lr=1e-3)
    return build_model(resolved).n_non_embedding_params


_KV_RE = re.compile(r"^([\d,]+)")


def _kv_bytes_per_token_per_layer(variant: str) -> int:
    """The analytic cache size from make_tables.static_columns (the table's own column)."""
    text = mt.static_columns(variant)["KV bytes/token/layer"]
    match = _KV_RE.match(text)
    if match is None:
        raise PlotError(f"cannot read the KV bytes of {variant!r} from {text!r}")
    return int(match.group(1).replace(",", ""))


def _token_count(label: str) -> int:
    if label.endswith("M") and label[:-1].isdigit():
        return int(label[:-1]) * 1_000_000
    raise PlotError(f"cannot place budget {label!r} on a token axis")


def _bench_index(records) -> dict:
    return {(r["variant"], r["mode"]): r for r in records}


# --- Plot 1: primary gaps ------------------------------------------------------------


def _comparison_summary(comp: pt.Comparison) -> str:
    text = f"gap {comp.mean_gap:+.4f} nats, 95% CI [{comp.ci_low:+.4f}, {comp.ci_high:+.4f}]"
    text += f"\n{comp.test} test, {len(comp.variant_losses)} variant seeds"
    if comp.ruled_out_above is not None:
        text += f"\ngaps above {comp.ruled_out_above:+.4f} nats ruled out"
    return text


def _draw_comparison(ax, comp, prereg: Mapping, colour: str) -> None:
    if comp.test == pt.PAIRED:
        diffs = list(comp.differences)
        _dots(ax, diffs, _stagger(len(diffs), 0.0), colour)
        ax.set_yticks([0.0])
        ax.set_yticklabels(["d_k"])
        ax.set_ylim(-0.75, 0.6)
        ax.set_xlabel("gap vs MHA, nats: d_k = variant − MHA, same seed\n" + _comparison_summary(comp), fontsize=8)
        ci_y = -0.45
    else:
        mean_ref = float(np.mean(comp.reference_losses))
        ref = [x - mean_ref for x in comp.reference_losses]
        var = [x - mean_ref for x in comp.variant_losses]
        _dots(ax, ref, _stagger(len(ref), 1.0), _colour("mha"))
        _dots(ax, var, _stagger(len(var), 0.0), colour)
        ax.set_yticks([1.0, 0.0])
        ax.set_yticklabels(["MHA", DISPLAY.get(comp.name, comp.name)])
        ax.set_ylim(-0.75, 1.6)
        ax.set_xlabel("loss, nats, centred on the MHA mean (Welch: each group's seeds)\n" + _comparison_summary(comp), fontsize=8)
        ci_y = -0.45
    ax.plot([comp.ci_low, comp.ci_high], [ci_y, ci_y], color=colour, lw=2.2, solid_capstyle="butt", zorder=2)
    _mean_mark(ax, comp.mean_gap, ci_y, colour)


def plot_primary_gaps(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    primary = list(prereg["primary"]["comparisons"])
    exploratory = list(prereg["exploratory"]["variants"])
    decision = a.decision
    band = None if decision is None else decision.bound_gap
    if decision is None:
        band_text = "No detectable-effect band: there is no seed-count decision yet (plan §1.3, §1.4)."
    else:
        band_text = (
            f"Shaded band: the detectable gap, ±{band:.4f} nats for {decision.n_seeds} seeds with the "
            f"{decision.test} test at 80% power and Holm's strictest step (plan §1.3). Drawn on the top row only."
        )
    if _ladder_fit_usable(a):
        axis_text = (
            f"Top axis: effective-parameter multiplier exp(−gap / b), b = {a.fit.b:.4f} nats per e-fold of "
            "parameters, from the 300M ladder fit (plan §8.4)."
        )
    elif a.decision is not None and a.decision.slope_not_positive:
        axis_text = (
            "No top axis: the ladder slope is not positive, so no multiplier is reported and 8 seeds are "
            "run (decision 223)."
        )
    else:
        axis_text = "No top axis: the ladder fit does not exist yet, so a gap has no parameter equivalent."
    cols = max(len(primary), len(exploratory), 1)
    fig, panels = _figure(
        (4.3 * cols, 7.6),
        title="Primary gaps against MHA: every seed, with interval and verdict (plan §10.5, plot 1)",
        rows=2,
        cols=cols,
        handles=_handles(prereg, mt.HEADLINE_VARIANTS),
        caption=[
            "Dots are seeds. Paired panels plot d_k = variant − MHA on the same seed. Welch panels "
            "centre each group on the MHA mean, so the MHA dots and the variant dots share one axis. "
            "Bar: 95% confidence interval; diamond: mean gap. Verdict and interval text come from primary_test.py.",
            band_text,
            axis_text,
            "Bottom row, exploratory (plan §1.2): intervals only, no detection claim, no band.",
            _withheld_part(a),
            SWA_CAVEAT,
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    for row, names in enumerate([primary, exploratory]):
        for col in range(cols):
            ax = panels[row][col]
            if col >= len(names):
                ax.axis("off")
                continue
            name = names[col]
            comp = a.comparisons.get(name)
            ax.set_xlabel("gap vs MHA, nats (paired) or loss centred on MHA (Welch)", fontsize=8)
            ax.set_ylabel("seed rows", fontsize=8)
            if comp is None:
                ax.set_title(_label(name, prereg), fontsize=9)
                _no_data(ax, _reason(a.notes, name))
                continue
            # The exploratory suffix is in the legend; the title's verdict line already says "exploratory".
            title = DISPLAY.get(name, name)
            ax.set_title(f"{title}\n{comp.verdict}", fontsize=9, pad=24 if _ladder_fit_usable(a) else 6)
            ax.axvline(0.0, color="black", lw=0.8, zorder=1)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            if row == 0 and band is not None:
                ax.axvspan(-band, band, color="0.88", zorder=0)
            _draw_comparison(ax, comp, prereg, _colour(name))
            if _ladder_fit_usable(a):
                _multiplier_axis(ax, a.fit.b)
    return fig


# --- Plot 2: inference frontier --------------------------------------------------------


def _decode_row(record):
    """The decode row of a bench record at the headline point (FRONTIER_BATCH, FRONTIER_CACHE), or None."""
    if record is None:
        return None
    rows = [r for r in record["decode"] if r["batch"] == FRONTIER_BATCH and r["cache"] == FRONTIER_CACHE]
    return rows[0] if rows else None


def plot_inference_frontier(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    bench = _bench_index(inp.bench)
    rows, missing = [], []
    for variant in mt.HEADLINE_VARIANTS:
        losses = a.losses.get(variant, {})
        if not losses or _withheld(a, variant):
            continue
        row = _decode_row(bench.get((variant, "opt")))
        if row is None:
            missing.append(f"{variant} has no opt benchmark record at this point")
            continue
        rows.append((variant, row, losses))
    fig, panels = _figure(
        (9.0, 6.4),
        title="Inference frontier: 300M validation loss against decode throughput (plan §8.6, plot 2)",
        handles=_handles(prereg, mt.HEADLINE_VARIANTS),
        caption=[
            f"Decode throughput from the opt path at batch {FRONTIER_BATCH} and cache {FRONTIER_CACHE} tokens. "
            "Every seed is a dot; the diamond is the seed mean. Right is faster; lower is better on loss.",
            SPEED_MEDIAN_CAPTION,
            SWA_CAVEAT,
            ("Missing: " + "; ".join(missing) + ".") if missing else None,
            _withheld_part(a),
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    ax = panels[0][0]
    ax.set_xlabel(f"decode tok/s, opt, batch {FRONTIER_BATCH}, cache {FRONTIER_CACHE}")
    ax.set_ylabel("300M validation loss, nats")
    for variant, row, losses in rows:
        x = float(row["tok_per_s"])
        seeds = sorted(losses)
        ys = [losses[s] for s in seeds]
        mean = float(np.mean(ys))
        # The range of the repetitions (decision 245): one horizontal whisker at the seed mean's height.
        ax.hlines(mean, float(row["tok_per_s_min"]), float(row["tok_per_s_max"]), colors=_colour(variant),
                  linewidth=WHISKER_WIDTH, zorder=2).set_gid(f"range:{variant}")
        _dots(ax, [x] * len(ys), ys, _colour(variant))
        _mean_mark(ax, x, mean, _colour(variant))
        ax.annotate(DISPLAY[variant], (x, mean), xytext=(7, 0), textcoords="offset points",
                    fontsize=8, va="center")
    ax.margins(x=0.1)
    if not rows:
        _no_data(ax)
    return fig


# --- Plot 3: memory frontier -----------------------------------------------------------


def plot_memory_frontier(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    fig, panels = _figure(
        (9.0, 6.4),
        title="Memory frontier: 300M validation loss against KV cache bytes per token (plan §10.5, plot 3)",
        handles=_handles(prereg, mt.HEADLINE_VARIANTS),
        caption=[
            "KV bytes per token per layer, analytic, from the model through make_tables.static_columns "
            "(the table's own column). SWA-256's cache is capped at its window. Every seed is a dot; "
            "the diamond is the seed mean.",
            SWA_CAVEAT,
            _withheld_part(a),
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    ax = panels[0][0]
    ax.set_xlabel("KV cache bytes per token per layer (analytic)")
    ax.set_ylabel("300M validation loss, nats")
    drawn = 0
    for variant in mt.HEADLINE_VARIANTS:
        losses = a.losses.get(variant, {})
        if not losses or _withheld(a, variant):
            continue
        x = _kv_bytes_per_token_per_layer(variant)
        seeds = sorted(losses)
        ys = [losses[s] for s in seeds]
        _dots(ax, [x] * len(ys), ys, _colour(variant))
        _mean_mark(ax, x, float(np.mean(ys)), _colour(variant))
        ax.annotate(DISPLAY[variant], (x, float(np.mean(ys))), xytext=(7, 0), textcoords="offset points",
                    fontsize=8, va="center")
        drawn += len(ys)
    if drawn == 0:
        _no_data(ax)
    return fig


# --- Plot 4: noise floor ---------------------------------------------------------------


NULL_WARNING = (
    "NULL CONTROL: verdict {verdict}. The null control must show no effect; if it "
    "does, pairing, seeding or the pipeline has a bug (plan §9.3). Read no comparison until it is fixed."
)


def _sd_text(value: float | None) -> str:
    return "not available yet" if value is None else f"{value:.4f} nats"


def _verdict_line(comp: pt.Comparison) -> str:
    return (
        f"verdict {comp.verdict}; mean gap {comp.mean_gap:+.4f} nats, {comp.confidence:.0%} CI "
        f"[{comp.ci_low:+.4f}, {comp.ci_high:+.4f}]; {comp.test} test"
    )


def _null_control_text(a: mt.Analysis) -> str:
    if a.null_control is not None:
        return f"Null control (plan §8.1): {_verdict_line(a.null_control)}."
    reason = _reason(a.notes, "null control")
    tail = f" Reason from the analysis notes: {reason}" if reason else " The analysis notes give no reason."
    return "Null control (plan §8.1): not computed." + tail


def plot_noise_floor(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    rows = [
        ("MHA 100M branches, init seeds (Arm N, data seed 1)", list(a.losses_100m.get("mha", {}).values()), "mha"),
        ("MHA 100M reruns, reproducibility pairs", list(a.reruns_100m.values()), "mha"),
        ("MHA init seed 1 at 100M, data seeds 1-3 (data-order noise)", list(a.data_order_100m.values()), "mha"),
        ("null control, 100M (MHA under another name)", list(a.null_100m.values()), "null"),
    ]
    sel = a.selection
    sel_text = (
        f"At 100M: sigma_init = {sel.sigma_init:.4f} nats (MHA seeds 1-5), s_rep = {sel.s_rep:.4f}, "
        f"ratio {sel.ratio:.3f} against threshold {sel.threshold}; test chosen: {sel.test}."
        if sel is not None
        else "Test selection not available yet; see the notes."
    )
    sigma_text = (
        f"sigma_init_null (null control seeds) = {_sd_text(a.sigma_init_null)}; "
        f"sigma_data (MHA init seed 1, data seeds 1-3) = {_sd_text(a.sigma_data)}."
    )
    warning = None
    if a.null_control is not None and a.null_control.verdict == pt.VERDICT_DETECTED:
        warning = textwrap.fill(NULL_WARNING.format(verdict=a.null_control.verdict), 110)
    fig, panels = _figure(
        (9.5, 5.6),
        title="Noise floor: every Arm N and null-control run (plan §8.1, plot 4)",
        caption=[
            "Every dot is one standalone or branch run at 100M tokens. Diamonds: row means. Nothing else is drawn.",
            "The data-order row is MHA init seed 1 under data seeds 1, 2 and 3. Data seed 1 is the same run as "
            "init seed 1's dot in the first row, shown in both rows: it is one run, not two.",
            f"{sel_text} {sigma_text}",
            _null_control_text(a),
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    ax = panels[0][0]
    if warning is not None:
        ax.set_title(warning, fontsize=9, color="firebrick", fontweight="bold")
    ax.set_xlabel("validation loss at 100M tokens, nats")
    ax.set_ylabel("run group")
    ax.set_yticks([len(rows) - 1 - i for i in range(len(rows))])
    ax.set_yticklabels([label for label, _, _ in rows], fontsize=8)
    drawn = 0
    for i, (label, values, variant) in enumerate(rows):
        y = len(rows) - 1 - i
        if not values:
            ax.text(0.99, y, "no runs", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7, color="0.4")
            continue
        _dots(ax, values, _stagger(len(values), float(y), 0.14), _colour(variant))
        _mean_mark(ax, float(np.mean(values)), float(y), _colour(variant))
        drawn += len(values)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    if drawn == 0:
        _no_data(ax)
    return fig


# --- Plot 5: ladder ---------------------------------------------------------------------

# The analysis reason that explains a missing fit at each budget (its notes start with these).
FIT_NOTE = {"100M": "ladder fit at 100M", "200M": "ladder fit at 200M", "300M": "ladder fit:"}
STATUS_WIDTH = 34
# Short point labels inside the panels; the legend carries the full names.
LADDER_SHORT = {"mha_L6": "L6", "mha": "MHA", "mha_L10": "L10"}


def _positive_control_text(a: mt.Analysis) -> str:
    if a.positive_control is not None:
        return f"Positive control (plan §8.4): {_verdict_line(a.positive_control)}."
    reason = _reason(a.notes, "positive control")
    tail = f" Reason from the analysis notes: {reason}" if reason else " The analysis notes give no reason."
    return "Positive control (plan §8.4): not computed." + tail


def _fit_text(a: mt.Analysis, budget: str) -> str | None:
    if a.fits.get(budget) is not None:
        return None
    reason = _reason(a.notes, FIT_NOTE[budget])
    return "No fit line: " + (reason if reason else "the analysis notes give no reason.")


def plot_ladder(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    # Plan §12.2 item 4: a declared cut ("l6_l8") leaves L10 out of the ladder: no point, no tick, and
    # no part of the fit line (the fits of analyse() do not read mha_L10 under the cut).
    points = tuple(v for v in LADDER if not (a.ladder == "l6_l8" and v == "mha_L10"))
    xs = {name: math.log(_non_embedding(name)) for name in points}
    by_budget = {"100M": a.losses_100m, "200M": a.losses_200m, "300M": a.losses}
    fig, panels = _figure(
        (12.0, 6.0),
        title="Calibration ladder: loss against ln(non-embedding parameters), one panel per budget (plan §8.4, plot 5)",
        cols=len(mt.LADDER_BUDGETS),
        handles=[
            Line2D([], [], color=_colour(v), marker="o", linestyle="None", markersize=MEAN_MARKER_SIZE,
                   label=DISPLAY[v])
            for v in points
        ],
        caption=[
            "Every dot is one seed's final validation loss at that panel's budget, at x = ln of its "
            "non-embedding parameter count (ticks show the counts). Each panel has its own y-limits.",
            "The ladder is cut to L6 and L8 by declaration (configs/commitments.yaml, plan §12.2 item 4): "
            "no L10 point is drawn or fitted." if a.ladder == "l6_l8" else None,
            "Dashed line: the fit of analyse() at that panel's budget, a − b ln N, drawn from the fit object; "
            "b is in the panel title. The 300M fit is the primary one; the 100M and 200M fits are the same "
            "conversion along the token axis (plan §8.4). Nothing is refitted here.",
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    for col, budget in enumerate(mt.LADDER_BUDGETS):
        ax = panels[0][col]
        losses = by_budget[budget]
        fit = a.fits.get(budget)
        title = f"{budget} tokens"
        title += f"\nfit slope b = {fit.b:.4f} nats per e-fold" if fit is not None else "\nno fit line"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ln(non-embedding parameters); ticks show the count", fontsize=8)
        ax.set_ylabel("validation loss, nats", fontsize=8)
        ax.set_xticks([xs[v] for v in points])
        ax.set_xticklabels([f"{_non_embedding(v) / 1e6:.2f}M" for v in points], fontsize=7)
        # The same x range in every panel, with or without runs: the ladder points, and room either side.
        ax.set_xlim(min(xs.values()) - 0.1, max(xs.values()) + 0.1)
        drawn = 0
        for variant in points:
            at_budget = losses.get(variant, {})
            if not at_budget:
                continue
            seeds = sorted(at_budget)
            _dots(ax, [xs[variant]] * len(seeds), [at_budget[s] for s in seeds], _colour(variant))
            ax.annotate(LADDER_SHORT[variant], (xs[variant], max(at_budget.values())), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7)
            drawn += len(seeds)
        if fit is not None:
            grid = np.linspace(min(xs.values()) - 0.05, max(xs.values()) + 0.05, 50)
            ax.plot(grid, [fit.predict(math.exp(g)) for g in grid], color="black", lw=1.0, ls="--", zorder=1)
        status = []
        if drawn == 0:
            _no_data(ax, _reason(a.notes, FIT_NOTE[budget]))
        else:
            fit_text = _fit_text(a, budget)
            if fit_text is not None:
                status.append(fit_text)
        if budget == "300M":
            status.append(_positive_control_text(a))
        if status:
            text = "\n".join(textwrap.fill(s, STATUS_WIDTH) for s in status)
            ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top", fontsize=7, zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    return fig


# --- Plot 6: context gap ----------------------------------------------------------------


def _arm_a_runs(results: Sequence[dict], prereg: Mapping) -> dict:
    """The 300M Arm A runs of the headline variants, by (variant, init seed)."""
    arm = config.load_arm(mt.ARM)
    budget = prereg["primary"]["budget"]
    runs = {}
    for manifest in results:
        parsed = mf.parse_run_name(manifest["name"])
        if parsed.arm != arm["tag"] or parsed.data_seed != arm["data_seed"]:
            continue
        if parsed.branch is not None or parsed.tokens != budget or parsed.variant not in mt.HEADLINE_VARIANTS:
            continue
        key = (parsed.variant, parsed.init_seed)
        if key in runs:
            raise PlotError(f"two 300M runs claim {key[0]} seed {key[1]}: {runs[key]['name']} and {manifest['name']}")
        runs[key] = manifest
    return runs


def plot_context_gap(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    buckets = prereg["context_use"]["buckets"]
    labels = [f"[{lo}, {hi})" for lo, hi in buckets]
    compared = [v for v in mt.HEADLINE_VARIANTS if v != "mha"]
    fig, panels = _figure(
        (10.0, 6.0),
        title="Context gap: loss gap against MHA per context bucket, same seed (plan §10.3, plot 6)",
        handles=_handles(prereg, compared),
        caption=[
            "Context c is the number of tokens of the same document before a token in the window (plan §10.3). "
            "Each dot is one seed's gap to MHA of the same seed in that bucket; bars are the means. "
            "Positive means worse than MHA. Secondary metrics: no detection claim.",
            _withheld_part(a),
            SWA_CAVEAT,
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    ax = panels[0][0]
    ax.set_xlabel("context bucket c (tokens)")
    ax.set_ylabel("val loss gap vs MHA, same seed, nats")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    runs = _arm_a_runs(inp.results, prereg)
    have_val = any("val" in m["final_metrics"] for m in runs.values())
    if not have_val:
        _no_data(ax, "manifests lack final_metrics.val, the per-bucket validation loss")
        return fig
    width = 0.7
    drawn = 0
    withheld = [v for v in compared if _withheld(a, v)]
    if len(withheld) == len(compared):
        _no_data(ax, "variant runs are withheld until the controls pass and the seed count is committed (plan §1.4, §8.1)")
        return fig
    for vi, variant in enumerate(compared):
        if variant in withheld:
            continue  # plan §8.1, §1.4: no gap of a withheld variant is drawn
        offset = (vi - (len(compared) - 1) / 2) * (width / len(compared))
        seeds = sorted(s for (v, s) in runs if v == variant and (("mha", s) in runs))
        for bi in range(len(buckets)):
            gaps = []
            for seed in seeds:
                v_bucket = runs[(variant, seed)]["final_metrics"]["val"]["buckets"][bi]["loss"]
                m_bucket = runs[("mha", seed)]["final_metrics"]["val"]["buckets"][bi]["loss"]
                if v_bucket is None or m_bucket is None:
                    continue
                gaps.append(float(v_bucket) - float(m_bucket))
            if not gaps:
                continue
            _dots(ax, [bi + offset] * len(gaps), gaps, _colour(variant))
            _mean_mark(ax, bi + offset, float(np.mean(gaps)), _colour(variant))
            drawn += len(gaps)
    if drawn == 0:
        _no_data(ax, "no variant has the same seed as MHA in the validation buckets")
    return fig


# --- Plot 7: training curves ------------------------------------------------------------


def _curve_runs(inp: PlotInputs) -> tuple[list[tuple[dict, mf.RunName]], bool]:
    """The manifests plot 7 draws, and whether any variant's run is withheld (plan §8.1).

    The analysis names them (its `curve_runs`): the Arm A runs and branches of the headline variants
    at each variant's learning rate. While the gate is shut, a variant's runs are not drawn; MHA's are.
    """
    a = inp.analysis
    names = set(a.curve_runs)
    chosen: list[tuple[dict, mf.RunName]] = []
    withheld = False
    for manifest in inp.results:
        if manifest["name"] not in names:
            continue
        parsed = mf.parse_run_name(manifest["name"])
        if _withheld(a, parsed.variant):
            withheld = True
            continue
        chosen.append((manifest, parsed))
    return chosen, withheld


def plot_training_curves(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    runs, withheld = _curve_runs(inp)
    fig, panels = _figure(
        (9.5, 6.0),
        title="Training curves: validation loss against tokens (plan §10.5, plot 7)",
        handles=_handles(prereg, mt.HEADLINE_VARIANTS, line=True),
        caption=[
            "Faded lines: the val_curve of each run, its evaluations during training. Solid dots: the final "
            "validation loss of each drawn run and branch, at its token count. Drawn: the Arm A runs and "
            "branches of the headline variants at each variant's learning rate; learning-rate sweep points "
            "and other arms are not drawn. Trunk checkpoints are never results and are never drawn as dots.",
            WITHHELD_CAPTION if withheld else None,
            SWA_CAVEAT,
            _notes_part(a.notes),
            _left_out_part(inp.skipped),
        ],
    )
    ax = panels[0][0]
    ax.set_xscale("log")
    ax.set_xlabel("tokens (log scale)")
    ax.set_ylabel("validation loss, nats")
    by_variant: dict[str, tuple[list, list]] = {}
    curves = 0
    skipped = []
    for manifest, parsed in runs:
        metrics = manifest["final_metrics"]
        curve = metrics.get("val_curve")
        if curve:
            curves += 1
            ax.plot([p["tokens"] for p in curve], [p["val_loss"] for p in curve],
                    color=_colour(parsed.variant), alpha=0.3, lw=1.0, zorder=1)
        label = parsed.branch if parsed.branch is not None else parsed.tokens
        try:
            tokens = _token_count(label)
        except PlotError:
            tokens = curve[-1]["tokens"] if curve else None
        if tokens is None:
            skipped.append(manifest["name"])
            continue
        xs, ys = by_variant.setdefault(parsed.variant, ([], []))
        xs.append(tokens)
        ys.append(float(metrics["val_loss"]))
    for variant, (xs, ys) in by_variant.items():
        _dots(ax, xs, ys, _colour(variant))
    if not by_variant:
        _no_data(ax)
    elif curves == 0:
        ax.text(0.5, 0.97, "no data yet for the curves: manifests lack final_metrics.val_curve",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color="0.3")
    if skipped:
        ax.text(0.01, 0.01, f"not placed on the token axis: {', '.join(skipped)}", transform=ax.transAxes,
                fontsize=7, color="0.3")
    return fig


# --- Plot 8: latency ---------------------------------------------------------------------


def _ranges(ax, xs, lows, highs, colour: str, tag: str) -> None:
    """Plot 8's range whiskers (decision 245): one vertical whisker per point, from the smallest to the
    largest repetition, as one artist with gid "range:<tag>" (tag is "<mode>:<variant>")."""
    ax.vlines(list(xs), list(lows), list(highs), colors=colour, linewidth=WHISKER_WIDTH,
              zorder=2).set_gid(f"range:{tag}")


def plot_latency(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    bench = _bench_index(inp.bench)
    batches = (1, 8, 32)
    fig, panels = _figure(
        (13.0, 4.8),
        title="Latency: prefill and decode against context, ref and opt (plan §8.5 and §10.4, plot 8)",
        cols=1 + len(batches),
        handles=_handles(prereg, mt.HEADLINE_VARIANTS, line=True)
        + [Line2D([], [], color="0.2", lw=LINE_WIDTH, linestyle="-", label="opt path"),
           Line2D([], [], color="0.2", lw=LINE_WIDTH, linestyle="--", label="ref path")],
        caption=[
            KERNEL_DISCLOSURE,
            "Prefill: batch 1, the smallest batch in the records, against context length. Decode: ms per token "
            "against cache length, one panel per batch. Log axes.",
            "Latency is a benchmark on one fixed device; it measures speed, never quality.",
            SPEED_MEDIAN_CAPTION,
            SWA_CAVEAT,
            _notes_part(a.notes),
        ],
    )
    row = panels[0]
    titles = ["prefill, batch 1: latency (ms)"] + [f"decode, batch {b}: ms per token" for b in batches]
    has_any = False
    for ax, title in zip(row, titles):
        ax.set_title(title, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("context (tokens)" if title.startswith("prefill") else "cache length (tokens)", fontsize=8)
        ax.set_ylabel("latency, ms" if title.startswith("prefill") else "ms per token", fontsize=8)
    for variant in mt.HEADLINE_VARIANTS:
        for mode, style in (("opt", "-"), ("ref", "--")):
            record = bench.get((variant, mode))
            if record is None:
                continue
            tag = f"{mode}:{variant}"
            prefill = sorted((r["context"], r["latency_ms"], r["latency_ms_min"], r["latency_ms_max"])
                             for r in record["prefill"] if r["batch"] == 1)
            if prefill:
                for line in row[0].plot([p[0] for p in prefill], [p[1] for p in prefill], color=_colour(variant),
                                        ls=style, lw=LINE_WIDTH, marker="o", markersize=3):
                    line.set_gid(tag)
                _ranges(row[0], [p[0] for p in prefill], [p[2] for p in prefill], [p[3] for p in prefill],
                        _colour(variant), tag)
                has_any = True
            for ax, batch in zip(row[1:], batches):
                decode = sorted((r["cache"], r["ms_per_token"], r["ms_per_token_min"], r["ms_per_token_max"])
                                for r in record["decode"] if r["batch"] == batch)
                if decode:
                    for line in ax.plot([p[0] for p in decode], [p[1] for p in decode], color=_colour(variant),
                                        ls=style, lw=LINE_WIDTH, marker="o", markersize=3):
                        line.set_gid(tag)
                    _ranges(ax, [p[0] for p in decode], [p[2] for p in decode], [p[3] for p in decode],
                            _colour(variant), tag)
                    has_any = True
    if not has_any:
        for ax in row:
            _no_data(ax)
    return fig


# --- Plot 9: synthetic probes -----------------------------------------------------------


def _chance_points(rows) -> list[tuple]:
    """(difficulty, chance) for each difficulty of one task. Records at one point must agree on chance."""
    by_difficulty: dict = {}
    for row in rows:
        by_difficulty.setdefault(row["difficulty"], set()).add(row.get("chance"))
    points = []
    for difficulty in sorted(by_difficulty):
        values = by_difficulty[difficulty]
        if len(values) > 1:
            raise PlotError(
                f"probes: records at difficulty {difficulty} disagree on chance: "
                f"{sorted(values, key=repr)}"
            )
        (value,) = values
        if value is not None:
            points.append((difficulty, value))
    return points


def plot_probes(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    tasks = sorted({r["task"] for r in inp.probes})
    variants = [*mt.HEADLINE_VARIANTS, PROBE_ANCHOR]
    cols = max(len(tasks), 1)
    fig, panels = _figure(
        (max(9.0, 4.6 * cols), 4.9),
        title="Synthetic probes: accuracy against difficulty, one panel per task (plan §9.2, plot 9)",
        cols=cols,
        handles=_handles(prereg, variants, line=True),
        caption=[
            "Every init seed is a dot; the line through each variant's per-difficulty mean. The recurrent anchor is "
            "black and dashed (plan §9.2). Difficulty is the task's own size, on a log2 axis. "
            "The grey dotted line is the no-recall level 1/n.",
            _notes_part(a.notes),
        ],
    )
    for col in range(cols):
        ax = panels[0][col]
        ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xlabel("difficulty (task size)")
        ax.set_ylabel("token accuracy (held-out)")
        ax.set_ylim(-0.02, 1.02)
        if not tasks:
            ax.set_title("probe", fontsize=9)
            _no_data(ax)
            continue
        task = tasks[col]
        ax.set_title(f"probe: {task}", fontsize=9)
        rows = [r for r in inp.probes if r["task"] == task]
        difficulties = sorted({r["difficulty"] for r in rows})
        ax.set_xticks(difficulties)
        ax.set_xticklabels([str(d) for d in difficulties])
        chance = _chance_points(rows)
        if chance:
            # One grey dotted line per panel, at the true difficulties: the no-recall level 1/n.
            ax.plot([d for d, _ in chance], [c for _, c in chance], color="0.5", ls=":", lw=LINE_WIDTH, zorder=1)
        present = [v for v in variants if any(r["variant"] == v for r in rows)]
        mid = (len(present) - 1) / 2
        for i, variant in enumerate(present):
            vrows = [r for r in rows if r["variant"] == variant]
            factor = 2.0 ** (0.10 * (i - mid))
            colour = _colour(variant)
            _dots(ax, [r["difficulty"] * factor for r in vrows], [r["accuracy"] for r in vrows], colour)
            means = []
            for d in difficulties:
                accs = [r["accuracy"] for r in vrows if r["difficulty"] == d]
                if accs:
                    means.append((d * factor, float(np.mean(accs))))
            style = "--" if variant == PROBE_ANCHOR else "-"
            ax.plot([x for x, _ in means], [y for _, y in means], color=colour, ls=style, lw=LINE_WIDTH, zorder=2)
    return fig


# --- Plot 10: retrieval -------------------------------------------------------------------


def plot_retrieval(inp: PlotInputs) -> Figure:
    prereg, a = inp.prereg, inp.analysis
    counts = sorted({r["n_distractors"] for r in inp.retrieval})
    cols = max(len(counts), 1)
    variants = list(mt.HEADLINE_VARIANTS)
    markers = {"passkey": "o", "needle": "s"}
    handles = _handles(prereg, variants) + [
        Line2D([], [], color="0.55", marker=markers["passkey"], linestyle="None", markersize=MEAN_MARKER_SIZE,
               label="passkey (circles)"),
        Line2D([], [], color="0.55", marker=markers["needle"], linestyle="None", markersize=MEAN_MARKER_SIZE,
               label="needle (squares)"),
    ]
    fig, panels = _figure(
        (max(9.0, 4.4 * cols + 0.8), 7.0),
        title="Retrieval: accuracy against context length, one panel per distractor count (plan §9.7, plot 10)",
        rows=len(RETRIEVAL_MODES),
        cols=cols,
        handles=handles,
        caption=[
            "Rows: zero-shot and trained. Columns: distractor count. Every init seed is a dot; the line through each "
            "variant's per-length mean. Passkey circles, needle squares. Length on a log2 axis.",
            SWA_CAVEAT,
            _notes_part(a.notes),
        ],
    )
    for row, mode in enumerate(RETRIEVAL_MODES):
        for col in range(cols):
            ax = panels[row][col]
            ax.set_xscale("log", base=2)
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("context length (tokens)", fontsize=8)
            ax.set_ylabel(f"accuracy, {mode.replace('_', '-')}", fontsize=8)
            if not counts:
                ax.set_title("retrieval", fontsize=9)
                ax.set_xlim(RETRIEVAL_LENGTHS[0] / 2, RETRIEVAL_LENGTHS[-1] * 2)
                ax.set_xticks(RETRIEVAL_LENGTHS)
                ax.set_xticklabels([str(n) for n in RETRIEVAL_LENGTHS], fontsize=7)
                _no_data(ax)
                continue
            count = counts[col]
            ax.set_title(f"{count} distractors", fontsize=9)
            rows = [r for r in inp.retrieval if r["mode"] == mode and r["n_distractors"] == count]
            if not rows:
                _no_data(ax)
                continue
            lengths = sorted({r["length"] for r in rows})
            ax.set_xticks(lengths)
            ax.set_xticklabels([str(n) for n in lengths], fontsize=7)
            present = [v for v in variants if any(r["variant"] == v for r in rows)]
            # One horizontal slot per (variant, task), so that no two groups sit on the same length.
            slots = [(v, t) for v in present for t in RETRIEVAL_TASKS]
            mid = (len(slots) - 1) / 2
            for slot, (variant, task) in enumerate(slots):
                trows = [r for r in rows if r["variant"] == variant and r["task"] == task]
                if not trows:
                    continue
                factor = 2.0 ** (0.06 * (slot - mid))
                colour = _colour(variant)
                _dots(ax, [r["length"] * factor for r in trows], [r["accuracy"] for r in trows], colour,
                      marker=markers[task])
                means = []
                for n in lengths:
                    accs = [r["accuracy"] for r in trows if r["length"] == n]
                    if accs:
                        means.append((n * factor, float(np.mean(accs))))
                style = "-" if task == "passkey" else ":"
                ax.plot([x for x, _ in means], [y for _, y in means], color=colour, ls=style, lw=1.2, zorder=2)
    return fig


# --- The ten plots, in plan order -------------------------------------------------------

PLOTS: tuple[tuple[str, Callable[[PlotInputs], Figure]], ...] = (
    ("01_primary_gaps.png", plot_primary_gaps),
    ("02_inference_frontier.png", plot_inference_frontier),
    ("03_memory_frontier.png", plot_memory_frontier),
    ("04_noise_floor.png", plot_noise_floor),
    ("05_ladder.png", plot_ladder),
    ("06_context_gap.png", plot_context_gap),
    ("07_training_curves.png", plot_training_curves),
    ("08_latency.png", plot_latency),
    ("09_probes.png", plot_probes),
    ("10_retrieval.png", plot_retrieval),
)


def build_all(manifests, bench, probes, retrieval, out_dir, *, skip_unfinished: bool = False,
              commitments: Mapping | None = None) -> list[Path]:
    """Write the ten PNG files, in plan order, into `out_dir`. Returns their paths.

    Unfinished runs are refused unless skip_unfinished leaves them out, by name (plan §10.6).
    """
    inputs = make_inputs(
        manifests, bench, probes, retrieval, skip_unfinished=skip_unfinished, commitments=commitments
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, builder in PLOTS:
        path = out / filename
        builder(inputs).savefig(path, dpi=DPI, format="png")
        paths.append(path)
    return paths


# --- Command line ----------------------------------------------------------------------


def load_records(path) -> list:
    """A JSON file holding one record or a list of them, or every such file in a directory. Missing gives []."""
    target = Path(path)
    if not target.exists():
        return []
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    records: list = []
    for file in files:
        data = json.loads(file.read_text(encoding="utf-8"))
        records.extend(data if isinstance(data, list) else [data])
    return records


def main(argv: Sequence[str] | None = None, *, commitments: Mapping | None = None) -> int:
    """The command line. The declarations are read from the repository's configs/commitments.yaml.

    As for make_tables: no flag names another file (decision 220; review S5, G1), `commitments` is
    for callers in Python, and --blank builds the plots of a campaign that has declared nothing.
    """
    parser = argparse.ArgumentParser(description="Build the ten plots of plan §10.5.")
    parser.add_argument("--manifests", default="results/manifests", help="directory of manifests")
    parser.add_argument("--bench", default="results/bench", help="directory (or file) of bench/2 records")
    parser.add_argument("--probes", default="results/probes.json", help="probe records (JSON list)")
    parser.add_argument("--retrieval", default="results/retrieval.json", help="retrieval records (JSON list)")
    parser.add_argument("--out", default="results/plots", help="directory to write the PNG files to")
    parser.add_argument("--blank", action="store_true", help="ignore every input; build the blank plots")
    parser.add_argument(
        "--skip-unfinished", action="store_true", help="leave unfinished runs out, by name"
    )
    args = parser.parse_args(argv)
    try:
        if commitments is None:
            commitments = config.uncommitted() if args.blank else config.load_commitments()
        if args.blank:
            paths = build_all([], [], [], [], args.out, commitments=commitments)
        else:
            paths = build_all(
                mt.load_manifests(args.manifests),
                load_records(args.bench),
                load_records(args.probes),
                load_records(args.retrieval),
                args.out,
                skip_unfinished=args.skip_unfinished,
                commitments=commitments,
            )
    except (mt.TableError, mf.ManifestError, PlotError, config.ConfigError, OSError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
