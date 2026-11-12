"""The headline table (plan §10.1), built from run manifests and nothing else.

    python -m scripts.make_tables --manifests results/manifests --out results/tables
    python -m scripts.make_tables --blank --out results/tables/blank

The script refuses to build the table from runs that are not comparable (plan
§7.2, §10.6, §13.6) and lists every violation instead of building a partial
table. Every value comes from `analysis/` or from the configs; nothing is typed
in by hand. It judges nothing itself: verdicts are `Comparison.verdict` from
`analysis/primary_test.py`.
"""

from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from analysis.lr_sweep import STEPS_TAKEN_PARITY  # plan §7.3, §13.6: defined once, there
from src import config
from src.model import build_model
from src.utils import manifest as mf

HEADLINE_VARIANTS = ("mha", "gqa4", "mqa", "mla192", "gqa2", "mla96", "swa256")
BLANK = "—"
ARM = "iso_token"
LEFT_ARM = "left Arm A (seed-count rule)"
COLUMNS = (
    "variant",
    "params",
    "d_ff",
    "KV bytes/token/layer",
    "train ms/step",
    "decode tok/s (batch 32, cache 4k)",
    "val loss, every seed",
    "gap vs MHA [95% CI]",
    "effective-param multiplier",
    "verdict",
)


class TableError(RuntimeError):
    """A refusal: the table cannot be built. The message lists every violation."""


def _listing(head: str, problems: Sequence[str]) -> str:
    return "\n".join([f"{head}; {len(problems)} problem(s):"] + [f"  - {p}" for p in problems])


# --- Loading and selecting ----------------------------------------------------


def load_manifests(directory: str | Path) -> list[dict]:
    """Every *.json manifest in `directory`, sorted by file name. A missing directory gives []."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("*.json"), key=lambda p: p.name)
    return [mf.read_manifest(path) for path in paths]


def select_results(
    manifests: Sequence[dict], *, skip_unfinished: bool = False
) -> tuple[list[dict], list[str]]:
    """The completed runs and branches. Returns (results, names_skipped).

    A trunk checkpoint is never a result, with or without `skip_unfinished`: it
    is still at peak learning rate and its loss is falsely high. An unfinished
    run is refused unless `skip_unfinished` leaves it out, by name.
    """
    results: list[dict] = []
    skipped: list[str] = []
    problems: list[str] = []
    for manifest in manifests:
        name = manifest.get("name")
        if manifest.get("kind") == "trunk_checkpoint":
            problems.append(
                f"{name}: a trunk checkpoint is never reported as a result; it is still at "
                "peak learning rate and its loss is falsely high"
            )
        elif mf.is_result(manifest):
            results.append(manifest)
        elif skip_unfinished:
            skipped.append(name)
        else:
            problems.append(f"{name}: unfinished, so it is not a result (skip_unfinished leaves it out)")
    if problems:
        raise TableError(_listing("cannot select results", problems))
    return results, skipped


# --- Comparability (plan §7.2, §10.6, §13.6) ----------------------------------


def _distinct(results: Sequence[dict], label: str, getter) -> str | None:
    """One violation line when the runs disagree about the value `getter` reads."""
    groups: dict[str, tuple[object, list[str]]] = {}
    for manifest in results:
        value = getter(manifest)
        groups.setdefault(repr(value), (value, []))[1].append(manifest["name"])
    if len(groups) <= 1:
        return None
    parts = "; ".join(f"{value!r} in {', '.join(names)}" for value, names in groups.values())
    return f"{label} differs across runs: {parts}"


def _field(*path: str):
    def read(manifest: dict):
        node = manifest
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
        return node

    return read


def check_comparable(results: Sequence[dict]) -> None:
    """Raise TableError listing every reason these runs cannot be compared together."""
    problems: list[str] = []

    not_results = [m["name"] for m in results if not mf.is_result(m)]
    if not_results:
        problems.append(f"not a finished run or branch: {', '.join(not_results)}")

    for label, getter in (
        ("backend", _field("backend")),
        ("torch version", _field("environment", "torch")),
        ("CUDA version", _field("environment", "cuda")),
        ("driver version", _field("environment", "driver")),
        ("dataset hash", _field("data", "dataset_sha256")),
        ("tokenizer hash", _field("data", "tokenizer_sha256")),
        ("GPU name", _field("environment", "gpu_name")),
        ("mode", _field("mode")),
        ("precision", _field("precision")),
    ):
        line = _distinct(results, label, getter)
        if line is not None:
            problems.append(line)

    by_total: dict[int, list[dict]] = {}
    for manifest in results:
        by_total.setdefault(manifest["steps"]["total"], []).append(manifest)
    for total in sorted(by_total):
        group = by_total[total]
        low = min(group, key=lambda m: m["steps"]["taken"])
        high = max(group, key=lambda m: m["steps"]["taken"])
        diff = high["steps"]["taken"] - low["steps"]["taken"]
        if diff > STEPS_TAKEN_PARITY * total:
            problems.append(
                f"steps taken differ by {diff} at {total} total steps: {low['name']} took "
                f"{low['steps']['taken']}, {high['name']} took {high['steps']['taken']}; "
                f"the limit is {STEPS_TAKEN_PARITY * total:g}"
            )

    if problems:
        raise TableError(_listing("the runs are not comparable", problems))


# --- The analysis --------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    losses: dict  # {variant or ladder name: {init_seed: val_loss}}, 300M results
    losses_100m: dict  # {name: {init_seed: val_loss}}, 100M branches
    reruns_100m: dict  # {init_seed: val_loss}, MHA standalone 100M reruns
    selection: object  # primary_test.TestSelection or None
    fit: object  # ladder_fit.LadderFit or None
    decision: object  # ladder_fit.SeedCountDecision or None
    comparisons: dict  # {variant: primary_test.Comparison}
    multipliers: dict  # {variant: ladder_fit.Multiplier}
    notes: tuple  # reasons for everything not computed, in a fixed order


# The number of non-embedding parameters of each manifest's configuration, by
# config hash. Building a model is slow; the same configuration never changes.
_PARAM_CACHE: dict[str, int] = {}


def _non_embedding_params(manifest: dict) -> int:
    key = manifest["config_hash"]
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = build_model(manifest["resolved"]).n_non_embedding_params
    return _PARAM_CACHE[key]


def _collect(results: Sequence[dict], prereg: Mapping):
    """Losses by name and seed, from the runs this table reads. Raises on mixed learning rates."""
    seen: set[str] = set()
    for manifest in results:
        if manifest["name"] in seen:
            raise TableError(f"two manifests have the name {manifest['name']!r}")
        seen.add(manifest["name"])

    arm = config.load_arm(ARM)
    budget = prereg["primary"]["budget"]
    small = prereg["test_selection"]["sigma_init"]["budget"]
    wanted = set(HEADLINE_VARIANTS) | set(config.ladder_names())

    selected = []
    for manifest in results:
        parsed = mf.parse_run_name(manifest["name"])
        if parsed.arm != arm["tag"] or parsed.data_seed != arm["data_seed"]:
            continue
        if parsed.variant not in wanted:
            continue
        selected.append((manifest, parsed))

    # One learning rate per variant: that of its runs at the primary budget, results and
    # branches alike. The branches finish first, and the test selection needs them (and the
    # reruns at the same rate) before any 300M result exists (plan §8.1, §9.3).
    lr_runs: dict[str, dict[float, list[str]]] = {}
    for manifest, parsed in selected:
        if parsed.tokens == budget:
            lr_runs.setdefault(parsed.variant, {}).setdefault(parsed.peak_lr, []).append(manifest["name"])
    problems = []
    for variant, by_lr in sorted(lr_runs.items()):
        if len(by_lr) > 1:
            listed = "; ".join(
                f"{mf.format_lr(lr)} in {', '.join(names_)}" for lr, names_ in sorted(by_lr.items())
            )
            problems.append(f"{variant} has {budget} runs at more than one peak learning rate: {listed}")
    if problems:
        raise TableError(_listing("two learning rates of one variant cannot be mixed", problems))
    lr = {variant: next(iter(by_lr)) for variant, by_lr in lr_runs.items()}

    losses: dict[str, dict[int, float]] = {}
    losses_100m: dict[str, dict[int, float]] = {}
    reruns: dict[int, float] = {}
    examples: dict[str, dict] = {}
    for manifest, parsed in selected:
        loss = float(manifest["final_metrics"]["val_loss"])
        variant = parsed.variant
        if parsed.branch is None and parsed.tokens == budget:
            losses.setdefault(variant, {})[parsed.init_seed] = loss
            examples.setdefault(variant, manifest)
        elif parsed.branch == small and parsed.tokens == budget and parsed.peak_lr == lr.get(variant):
            losses_100m.setdefault(variant, {})[parsed.init_seed] = loss
        elif (
            variant == "mha"
            and parsed.branch is None
            and parsed.tokens == small
            and parsed.peak_lr == lr.get("mha")
        ):
            reruns[parsed.init_seed] = loss
    return losses, losses_100m, reruns, examples


def _missing(seeds, have: Mapping) -> list[int]:
    return [s for s in seeds if s not in have]


def analyse(results: Sequence[dict], prereg: Mapping | None = None) -> Analysis:
    """Every number of the table, computed in the pre-registered order. Refuses incomparable runs."""
    check_comparable(results)
    if prereg is None:
        prereg = config.load_prereg()
    losses, losses_100m, reruns, examples = _collect(results, prereg)
    notes: list[str] = []

    # 1. Test selection, from MHA runs only (plan §1.2).
    sel_seeds = list(prereg["test_selection"]["sigma_init"]["init_seeds"])
    pair_seeds = list(prereg["test_selection"]["reproducibility_pairs"]["init_seeds"])
    mha_100 = losses_100m.get("mha", {})
    selection = None
    miss_branch = _missing(sorted(set(sel_seeds) | set(pair_seeds)), mha_100)
    miss_rerun = _missing(pair_seeds, reruns)
    if miss_branch:
        notes.append(f"test selection: MHA 100M branches for seeds {miss_branch} are missing")
    if miss_rerun:
        notes.append(f"test selection: MHA 100M reruns for seeds {miss_rerun} are missing")
    if miss_branch or miss_rerun:
        notes.append("test selection skipped: see the missing MHA 100M runs above")
    else:
        try:
            selection = pt.select_test(
                {s: mha_100[s] for s in sel_seeds},
                {s: (mha_100[s], reruns[s]) for s in pair_seeds},
                prereg,
            )
        except pt.PreregError as exc:
            notes.append(f"test selection: {exc}")

    # 2. Ladder fit at 300M (plan §8.4).
    fit = None
    mha_300 = losses.get("mha", {})
    if not mha_300:
        notes.append("ladder fit: MHA 300M results are missing")
    elif not losses.get("mha_L6"):
        notes.append("ladder fit: mha_L6 300M results are missing")
    else:
        points = {"mha": (_non_embedding_params(examples["mha"]), mha_300)}
        for name in ("mha_L6", "mha_L10"):
            if losses.get(name):
                points[name] = (_non_embedding_params(examples[name]), losses[name])
        try:
            fit = lf.fit_ladder(points)
        except lf.LadderError as exc:
            notes.append(f"ladder fit: {exc}")

    # 3. Seed count (plan §1.4).
    decision = None
    base = list(prereg["seed_count"]["base_seeds"])
    if selection is None:
        notes.append("seed count rule skipped: no test selection")
    if fit is None:
        notes.append("seed count rule skipped: no ladder fit")
    if selection is not None and fit is not None:
        miss = _missing(base, mha_300)
        if miss:
            notes.append(f"seed count rule: MHA 300M results for seeds {miss} are missing")
        else:
            try:
                decision = lf.seed_count_rule(
                    {s: mha_300[s] for s in base}, fit, test=selection.test, prereg=prereg
                )
            except lf.LadderError as exc:
                notes.append(f"seed count rule: {exc}")

    # 4. The three primary comparisons, Holm-corrected together (plan §1.2).
    comparisons: dict[str, pt.Comparison] = {}
    primary_names = [prereg["primary"]["reference"], *prereg["primary"]["comparisons"]]
    if decision is None:
        notes.append("primary comparisons skipped: no seed count decision")
    else:
        missing = {
            name: _missing(decision.seeds, losses.get(name, {}))
            for name in primary_names
        }
        missing = {name: seeds for name, seeds in missing.items() if seeds}
        if missing:
            listed = "; ".join(f"{name} seeds {seeds}" for name, seeds in missing.items())
            notes.append(f"primary comparisons skipped: 300M results missing for {listed}")
        else:
            try:
                primaries = pt.primary_tests(
                    {
                        name: {s: losses[name][s] for s in decision.seeds}
                        for name in primary_names
                    },
                    test=selection.test,
                    n_seeds=decision.n_seeds,
                    prereg=prereg,
                )
                comparisons.update({c.name: c for c in primaries})
            except pt.PreregError as exc:
                notes.append(f"primary comparisons: {exc}")

    # 5. Exploratory comparisons (plan §1.2, §1.4).
    ex_seeds = list(prereg["exploratory"]["init_seeds"])
    if decision is None:
        notes.append("exploratory comparisons skipped: no seed count decision")
    else:
        for name in prereg["exploratory"]["variants"]:
            if decision.extended and name in decision.leaves_arm_a:
                notes.append(f"{name} left Arm A under the seed-count rule: not computed")
                continue
            miss_own = _missing(ex_seeds, losses.get(name, {}))
            miss_mha = _missing(decision.seeds, mha_300)
            if miss_own or miss_mha:
                parts = []
                if miss_own:
                    parts.append(f"{name} 300M results for seeds {miss_own}")
                if miss_mha:
                    parts.append(f"MHA 300M results for seeds {miss_mha}")
                notes.append(f"exploratory {name} skipped: {' and '.join(parts)} are missing")
                continue
            try:
                comparisons[name] = pt.exploratory(
                    name,
                    {s: losses[name][s] for s in ex_seeds},
                    {s: mha_300[s] for s in decision.seeds},
                    test=selection.test,
                    prereg=prereg,
                )
            except pt.PreregError as exc:
                notes.append(f"exploratory {name}: {exc}")

    # Runs with seeds outside the registered sets are shown in the table but never tested.
    if decision is not None:
        registered = {name: set(decision.seeds) for name in primary_names}
        registered.update({name: set(ex_seeds) for name in prereg["exploratory"]["variants"]})
        for name in sorted(registered):
            extra = sorted(set(losses.get(name, {})) - registered[name])
            if extra and name in comparisons:
                notes.append(
                    f"{name}: 300M results for seeds {extra} are outside the registered seeds "
                    "and are not part of its comparison"
                )

    # 6. Effective-parameter multipliers (plan §8.4).
    multipliers: dict[str, lf.Multiplier] = {}
    if comparisons and fit is None:
        notes.append("multipliers skipped: no ladder fit")
    elif fit is not None:
        for name, comparison in comparisons.items():
            try:
                multipliers[name] = lf.comparison_multiplier(comparison, fit)
            except lf.LadderError as exc:
                notes.append(f"multiplier {name}: {exc}")

    return Analysis(
        losses=losses,
        losses_100m=losses_100m,
        reruns_100m=reruns,
        selection=selection,
        fit=fit,
        decision=decision,
        comparisons=comparisons,
        multipliers=multipliers,
        notes=tuple(notes),
    )


# --- The table -------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _model_facts(variant: str) -> tuple[int, int, int, int | None]:
    resolved = config.resolve(variant, init_seed=1, peak_lr=1e-3)
    model = build_model(resolved)
    attn = model.layers[0].attn
    return model.n_params, resolved["model"]["d_ff"], attn.kv_bytes_per_token, attn.kv_cache_max_tokens


def static_columns(variant: str) -> dict:
    """Parameters, d_ff and the decode cache, from the configs. No results needed."""
    n_params, d_ff, kv_bytes, cap = _model_facts(variant)
    kv_text = f"{kv_bytes:,}"
    if cap is not None:
        kv_text += f" (capped at {cap} tokens)"
    return {"params": f"{n_params:,}", "d_ff": str(d_ff), "KV bytes/token/layer": kv_text}


def _timing(values: Mapping[str, float] | None, variant: str) -> str:
    if values is None or values.get(variant) is None:
        return BLANK
    return f"{values[variant]:.1f}"


def _losses_cell(by_seed: Mapping[int, float]) -> str:
    if not by_seed:
        return BLANK
    return ", ".join(f"s{seed} {loss:.4f}" for seed, loss in sorted(by_seed.items()))


def _multiplier_cell(m) -> str:
    if m is None:
        return BLANK
    if not m.bounded:
        return f"{m.multiplier:.3f}× [no interval: ladder slope unresolved]"
    return f"{m.multiplier:.3f}× [{m.ci_low:.3f}, {m.ci_high:.3f}]"


def _verdict_cell(comparison) -> str:
    text = comparison.verdict
    if comparison.ruled_out_above is not None:
        text += f" (gaps above {comparison.ruled_out_above:+.4f} nats ruled out)"
    return text


def headline_rows(
    analysis: Analysis,
    *,
    train_ms_per_step: Mapping[str, float] | None = None,
    decode_tok_per_s: Mapping[str, float] | None = None,
) -> list[dict]:
    """One row per headline variant, keyed by COLUMNS, every value a string."""
    reference = "mha"
    reference_present = bool(analysis.losses.get(reference))
    rows = []
    for variant in HEADLINE_VARIANTS:
        static = static_columns(variant)
        cells = {
            "variant": variant,
            "params": static["params"],
            "d_ff": static["d_ff"],
            "KV bytes/token/layer": static["KV bytes/token/layer"],
            "train ms/step": _timing(train_ms_per_step, variant),
            "decode tok/s (batch 32, cache 4k)": _timing(decode_tok_per_s, variant),
            "val loss, every seed": _losses_cell(analysis.losses.get(variant, {})),
        }
        if variant == reference:
            if reference_present:
                cells["gap vs MHA [95% CI]"] = "reference"
                cells["effective-param multiplier"] = "1 (reference)"
                cells["verdict"] = "reference"
            else:
                cells["gap vs MHA [95% CI]"] = BLANK
                cells["effective-param multiplier"] = BLANK
                cells["verdict"] = BLANK
        else:
            decision = analysis.decision
            comparison = analysis.comparisons.get(variant)
            if decision is not None and decision.extended and variant in decision.leaves_arm_a:
                cells["verdict"] = LEFT_ARM
            else:
                cells["verdict"] = BLANK if comparison is None else _verdict_cell(comparison)
            if comparison is None:
                cells["gap vs MHA [95% CI]"] = BLANK
            else:
                cells["gap vs MHA [95% CI]"] = (
                    f"{comparison.mean_gap:+.4f} [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]"
                )
            cells["effective-param multiplier"] = _multiplier_cell(analysis.multipliers.get(variant))
        rows.append({column: cells[column] for column in COLUMNS})
    return rows


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def render_markdown(rows: Sequence[Mapping[str, str]], notes: Sequence[str] = ()) -> str:
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[column]) for column in COLUMNS) + " |")
    text = "\n".join(lines) + "\n"
    if notes:
        text += "\nNotes:\n\n" + "".join(f"- {_cell(note)}\n" for note in notes)
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the headline table from run manifests.")
    parser.add_argument("--manifests", default="results/manifests", help="directory of manifests")
    parser.add_argument("--out", default="results/tables", help="directory to write headline.md to")
    parser.add_argument(
        "--skip-unfinished", action="store_true", help="leave unfinished runs out, by name"
    )
    parser.add_argument(
        "--blank", action="store_true", help="ignore --manifests; write the table for no runs"
    )
    args = parser.parse_args(argv)

    try:
        manifests = [] if args.blank else load_manifests(args.manifests)
        results, skipped = select_results(manifests, skip_unfinished=args.skip_unfinished)
        analysis = analyse(results)
    except (TableError, mf.ManifestError) as exc:
        print(exc, file=sys.stderr)
        return 1

    notes = list(analysis.notes)
    if skipped:
        notes.append("unfinished runs left out: " + ", ".join(skipped))
    text = render_markdown(headline_rows(analysis), notes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "headline.md"
    path.write_text(text, encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
