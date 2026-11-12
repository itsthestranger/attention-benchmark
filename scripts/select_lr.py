"""The learning-rate sweep report (plan §7.4), built from run manifests and nothing else.

    python -m scripts.select_lr --sweep-seed 21 --manifests results/manifests --out results/tables
    python -m scripts.select_lr --blank --out results/tables/blank

MHA is swept first with the five points of plan §7.4; every other headline variant
and both ladder sizes are then swept with the three points around MHA's rate. Each
name's rate is chosen by the rules of `analysis/lr_sweep.py`. Nothing is typed by hand.

The exit status is 0 when the report was written, whatever the statuses. It is
non-zero only on a refusal: unreadable manifests, or a sweep the rules cannot read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

from analysis import lr_sweep as sw
from scripts import make_tables as mt
from src import config
from src.utils import manifest as mf

REPORT_NAME = "lr_sweep.md"
WAIT_FOR_MHA = (
    "waiting for MHA's sweep: MHA has no chosen rate yet (plan §9.3: MHA's sweep first, "
    "then Arm N, then the other sweeps)"
)
WAIT_FOR_SIGMA = (
    "waiting for the noise floor: the test selection, and so sigma_init, does not exist yet "
    "(it needs MHA's 100M branches and reruns; plan §9.3)"
)
STATUS_TEXT = {sw.CHOSEN: "chosen", sw.EXTEND: "extend", sw.INCOMPLETE: "incomplete"}


def names_in_order() -> list[str]:
    """MHA, then the other headline variants, then both ladder sizes: the fixed section order."""
    headline = [name for name in mt.HEADLINE_VARIANTS if name != sw.REFERENCE]
    return [sw.REFERENCE, *headline, *config.ladder_names()]


def _rate(lr: float | None) -> str:
    return mt.BLANK if lr is None else mf.format_lr(lr)


def _loss(loss: float | None) -> str:
    return mt.BLANK if loss is None else f"{loss:.6f}"


def _points_table(points: Sequence[sw.Point]) -> list[str]:
    if not points:
        return ["No sweep runs read for this name."]
    lines = ["| rate | loss | run |", "|---|---|---|"]
    for point in points:
        if point.diverged:
            loss = f"DIVERGED: {point.reason or 'no reason recorded'}"
        else:
            loss = _loss(point.loss)
        lines.append(f"| {_rate(point.lr)} | {loss} | {point.run} |")
    return lines


def _side_by_side(choice: sw.Choice, reference_lr: float | None) -> list[str]:
    """The shared-recipe and tuned rates and losses, side by side (plan §7.4, item 5).

    The shared-recipe loss is this name's own run at MHA's rate, so it is read from the
    name's points, not from MHA's.
    """
    shared_rate = mt.BLANK
    shared_loss = mt.BLANK
    if choice.name != sw.REFERENCE and reference_lr is not None:
        shared_rate = _rate(reference_lr)
        at_reference = next(
            (p for p in choice.points if mf.format_lr(p.lr) == mf.format_lr(reference_lr)), None
        )
        if at_reference is not None and at_reference.diverged:
            shared_loss = "DIVERGED"
        else:
            shared_loss = _loss(choice.shared_loss)
    tuned_rate = mt.BLANK
    tuned_loss = mt.BLANK
    if choice.status == sw.CHOSEN:
        tuned_rate = _rate(choice.best_lr)
        tuned_loss = _loss(choice.tuned_loss)
    return [
        "| | shared recipe | tuned |",
        "|---|---|---|",
        f"| rate | {shared_rate} | {tuned_rate} |",
        f"| loss | {shared_loss} | {tuned_loss} |",
    ]


def _decision_lines(choice: sw.Choice, reference_lr: float | None) -> list[str]:
    lines = [
        f"- **Status:** {STATUS_TEXT[choice.status]}",
        f"- **Rate to use:** {_rate(choice.lr)}",
        f"- **Rule that decided:** {choice.rule or mt.BLANK}",
        "",
    ]
    lines.extend(_side_by_side(choice, reference_lr))
    lines.append("")
    next_rate = _rate(choice.next_lr) if choice.status == sw.EXTEND else mt.BLANK
    missing = ", ".join(_rate(lr) for lr in choice.missing) if choice.status == sw.INCOMPLETE else ""
    lines.append(f"- **Next point to run (extend):** {next_rate}")
    lines.append(f"- **Missing rates (incomplete):** {missing or mt.BLANK}")
    lines.append(f"- **lr_source:** {choice.source_run or mt.BLANK}")
    return lines


def _section(name: str, points: Sequence[sw.Point], choice: sw.Choice | None, waiting: str,
             reference_lr: float | None, notes: Sequence[str]) -> list[str]:
    lines = [f"## {name}", ""]
    lines.extend(_points_table(points))
    lines.append("")
    if choice is None:
        lines.append(f"- **Status:** {waiting}")
        lines.append(f"- **Rate to use:** {mt.BLANK}")
        lines.append(f"- **lr_source:** {mt.BLANK}")
    else:
        lines.extend(_decision_lines(choice, reference_lr))
    lines.extend(f"- Note: {note}" for note in notes)
    lines.append("")
    return lines


def build(manifests: Sequence[Mapping], sweep_seed: int | None):
    """Choices by name, points by name, and the notes on sigma_init, from the manifests.

    Trunk checkpoints are not results and carry no sweep point, so they are left out before
    `select_results`, which refuses them. Unfinished runs stay in, for their diverged points.
    Raises SweepError, TableError or ManifestError on a refusal.
    """
    prereg = config.load_prereg()
    results_input = [m for m in manifests if m.get("kind") != "trunk_checkpoint"]
    results, _skipped = mt.select_results(results_input, skip_unfinished=True)
    points = {
        name: (sw.points_from_manifests(manifests, name, sweep_seed=sweep_seed, prereg=prereg)
               if sweep_seed is not None else [])
        for name in names_in_order()
    }
    mha = sw.choose_reference(points[sw.REFERENCE], name=sw.REFERENCE)
    choices: dict[str, sw.Choice] = {sw.REFERENCE: mha}
    sigma_notes: list[str] = []
    if mha.status == sw.CHOSEN:
        analysis = mt.analyse(results)
        # Only the reasons sigma_init is missing matter here; the 300M notes are not about the sweep.
        sigma_notes = [note for note in analysis.notes if note.startswith("test selection")]
        if analysis.selection is not None:
            sigma = analysis.selection.sigma_init
            for name in names_in_order()[1:]:
                choices[name] = sw.choose_variant(
                    points[name], reference_lr=mha.lr, sigma_init=sigma, name=name
                )
    return choices, points, sigma_notes


def render(choices: Mapping[str, sw.Choice], points: Mapping[str, Sequence[sw.Point]],
           sigma_notes: Sequence[str], sweep_seed: int | None, manifest_count: int) -> str:
    mha = choices[sw.REFERENCE]
    lines = [
        "# Learning-rate sweep (plan §7.4)",
        "",
        "Generated by `python -m scripts.select_lr` from the run manifests. Every value below is "
        "computed by the rules of `analysis/lr_sweep.py`; none is typed by hand.",
        "",
        f"- Sweep seed: {sweep_seed if sweep_seed is not None else 'none (no manifests read)'}",
        f"- Manifests read: {manifest_count}",
        "",
    ]
    for name in names_in_order():
        choice = choices.get(name)
        if name == sw.REFERENCE:
            waiting, notes = "", []
        elif mha.status != sw.CHOSEN:
            waiting, notes = WAIT_FOR_MHA, []
        else:
            waiting, notes = WAIT_FOR_SIGMA, list(sigma_notes)
        lines.extend(_section(
            name, points[name], choice, waiting,
            mha.lr if name != sw.REFERENCE else None, notes,
        ))
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the learning-rate sweep report.")
    parser.add_argument("--manifests", default="results/manifests", help="directory of manifests")
    parser.add_argument("--sweep-seed", type=int, default=None,
                        help="the init seed of the sweep runs (required unless --blank)")
    parser.add_argument("--out", default="results/tables", help="directory to write lr_sweep.md to")
    parser.add_argument("--blank", action="store_true",
                        help="ignore --manifests; write the report for no runs")
    args = parser.parse_args(argv)
    if not args.blank and args.sweep_seed is None:
        parser.error("--sweep-seed is required unless --blank is given")

    sweep_seed = None if args.blank else args.sweep_seed
    try:
        manifests = [] if args.blank else mt.load_manifests(args.manifests)
        choices, points, sigma_notes = build(manifests, sweep_seed)
    except (sw.SweepError, mt.TableError, mf.ManifestError) as exc:
        print(exc, file=sys.stderr)
        return 1

    text = render(choices, points, sigma_notes, sweep_seed, len(manifests))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / REPORT_NAME
    path.write_text(text, encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
