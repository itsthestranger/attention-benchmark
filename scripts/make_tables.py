"""The headline table (plan §10.1) and the secondary reports, built from run manifests and nothing else.

    python -m scripts.make_tables --manifests results/manifests --out results/tables
    python -m scripts.make_tables --manifests results/manifests --out results/tables \\
        --timing-pass results/timing_pass.json --bench results/bench
    python -m scripts.make_tables --blank --out results/tables/blank

Writes headline.md, gates.md, arm_b.md (plan §8.3), context.md (plan §10.3) and
rqw.md (plan §8.5). The script refuses to build the tables from runs that are not
comparable (plan §7.2, §10.6, §13.6), and refuses timing and bench sources that
were not measured alone or that differ in environment, and lists every violation
instead of building a partial table. Every value comes from `analysis/` or from
the configs; nothing is typed in by hand. It judges nothing itself: verdicts are
`Comparison.verdict` from `analysis/primary_test.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import numbers
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from analysis import rqw_estimate as rq
from scripts import bench as bd
from scripts import timing_pass as tp
from src import config
from src.model import build_model
from src.utils import manifest as mf

HEADLINE_VARIANTS = ("mha", "gqa4", "mqa", "mla192", "gqa2", "mla96", "swa256")
BLANK = "—"
ARM = "iso_token"
# Budget labels of the calibration ladder's branches (plan §7.2, §8.4). The 100M
# budget is the test-selection budget in prereg.yaml; 200M is the ladder's middle branch.
LADDER_BUDGETS = ("100M", "200M", "300M")
MID_BUDGET = "200M"
# Plan §8.1: the data-order noise is MHA at init seed 1 under three data orders.
# Data seed 1 is the Arm A data seed (its run is the 100M branch of init seed 1);
# seeds 2 and 3 are standalone 100M runs at MHA's learning rate. Not in prereg.yaml.
DATA_ORDER_INIT_SEED = 1
DATA_ORDER_EXTRA_SEEDS = (2, 3)
# Plan §7.3 and §13.6: a difference above 0.5% of the steps invalidates the comparison.
STEPS_TAKEN_PARITY = 0.005
LEFT_ARM = "left Arm A (seed-count rule)"
ARM_B = "iso_backbone"
# Plan §10.3: the two context-use metrics, (key in the manifest's val block, label).
CONTEXT_METRICS = (("late_context", "late-context loss"), ("icl", "ICL score"))
GATE_WAIT = (
    "comparisons wait for the controls (plan §8.1, §8.4): the null control must be computed and "
    "not detected, and the positive control computed"
)
NO_TIMING = "no timing pass was given (--timing-pass)"
ARM_B_CAPTION = (
    "Arm B (iso-backbone, plan §8.3) is exploratory: no detection claim is made on any gap in this table."
)
ARM_B_MQA_LINE = (
    'Plan §8.3 reports the Arm A minus Arm B gap for MQA as "what the wider MLP bought"; the '
    "Arm A minus Arm B column below is that gap for every variant of the arm."
)
CONTEXT_CAPTION = (
    "Context use is a secondary metric (plan §10.3): gaps against MHA with their 95% intervals. "
    "No detection claim is made on them."
)
SWA_CAPTION = (
    "SWA at context 1,024 is nearly uninformative (plan §13.11): a 256-token window drops little at "
    "this length, so its line may top a loss plot and mean nothing. It is captioned, never headlined."
)
RQW_CAPTION = (
    "This is an estimate, not a measurement (plan §8.5, step 2): each variant's loss at the steps it "
    "affords in MHA's time is interpolated from its own Arm A results. It carries no interval and no "
    "verdict."
)
ARM_B_COLUMNS = (
    "variant",
    "Arm B val loss, every seed",
    "gap vs MHA [95% CI]",
    "verdict (vs MHA)",
    "Arm A 100M val loss, every seed",
    "Arm A minus Arm B [95% CI]",
    "verdict (Arm A minus Arm B)",
)
CONTEXT_COLUMNS = (
    "variant",
    "late-context loss, every seed",
    "gap vs MHA [95% CI], late-context",
    "verdict (late-context)",
    "ICL score, every seed",
    "gap vs MHA [95% CI], ICL",
    "verdict (ICL)",
)
RQW_COLUMNS = (
    "variant",
    "ms/step",
    "step-time ratio to MHA",
    "steps afforded S_v",
    "extrapolated",
    "estimated loss, every seed",
    "MHA 100M loss, every seed",
    "gap, estimate minus MHA (nats)",
    "status",
)
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
class ArmBResult:
    """One variant of Arm B (plan §8.3). Comparisons are None when they are not computed."""

    variant: str
    losses: dict  # {init_seed: val_loss}, the variant's standalone Arm B runs (registered seeds)
    arm_a_losses: dict  # {init_seed: val_loss}, its Arm A 100M branches (the arm's seeds)
    vs_mha: object  # primary_test.Comparison or None: Arm B against MHA's Arm A 100M branches
    arm_a_minus_arm_b: object  # primary_test.Comparison or None: Arm A 100M minus Arm B


@dataclass(frozen=True)
class ContextUse:
    """One variant's context-use metrics at 300M (plan §10.3). Values are None where missing."""

    variant: str
    seeds: tuple  # the variant's registered seeds
    late_context_values: dict  # {init_seed: late-context loss or None}
    icl_values: dict  # {init_seed: ICL score or None}
    late_context: object  # primary_test.Comparison or None, against MHA
    icl_score: object  # primary_test.Comparison or None, against MHA


@dataclass(frozen=True)
class Analysis:
    losses: dict  # {variant or ladder name: {init_seed: val_loss}}, 300M results
    losses_100m: dict  # {name: {init_seed: val_loss}}, 100M branches
    reruns_100m: dict  # {init_seed: val_loss}, MHA standalone 100M reruns
    selection: object  # primary_test.TestSelection or None
    fit: object  # ladder_fit.LadderFit or None: the 300M fit, the primary one
    decision: object  # ladder_fit.SeedCountDecision or None
    comparisons: dict  # {variant: primary_test.Comparison}
    multipliers: dict  # {variant: ladder_fit.Multiplier}
    notes: tuple  # reasons for everything not computed, in a fixed order
    losses_200m: dict  # {name: {init_seed: val_loss}}, 200M branches
    fits: dict  # {"100M" | "200M" | "300M": ladder_fit.LadderFit or None}
    null_100m: dict  # {init_seed: val_loss}, the null control's standalone 100M runs
    data_order_100m: dict  # {data_seed: val_loss}, MHA init seed 1 at 100M
    sigma_data: float | None  # SD over the three data orders, when complete
    sigma_init_null: float | None  # SD over the registered null seeds, when complete
    null_control: object  # primary_test.Comparison or None
    positive_control: object  # primary_test.Comparison or None
    comparisons_allowed: bool  # the gate: both controls computed, the null not detected
    arm_b: dict  # {variant: ArmBResult}, every variant of the arm in its order (plan §8.3)
    context_use: dict  # {variant: ContextUse}, headline order, no MHA (plan §10.3)


class _Collected(NamedTuple):
    losses: dict
    losses_100m: dict
    losses_200m: dict
    reruns: dict
    examples: dict
    null_100m: dict
    data_order_100m: dict
    context: dict  # {variant: {init_seed: {"late_context": x or None, "icl": y or None}}}, 300M


# The number of non-embedding parameters of each manifest's configuration, by
# config hash. Building a model is slow; the same configuration never changes.
_PARAM_CACHE: dict[str, int] = {}


def _non_embedding_params(manifest: dict) -> int:
    key = manifest["config_hash"]
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = build_model(manifest["resolved"]).n_non_embedding_params
    return _PARAM_CACHE[key]


def _collect(results: Sequence[dict], prereg: Mapping) -> _Collected:
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
    losses_200m: dict[str, dict[int, float]] = {}
    reruns: dict[int, float] = {}
    context: dict[str, dict[int, dict]] = {}
    # One manifest per variant, for its parameter count. Every run of a variant has the
    # same model, so the first one read will do.
    examples: dict[str, dict] = {}
    for manifest, parsed in selected:
        loss = float(manifest["final_metrics"]["val_loss"])
        variant = parsed.variant
        examples.setdefault(variant, manifest)
        if parsed.branch is None and parsed.tokens == budget:
            losses.setdefault(variant, {})[parsed.init_seed] = loss
            context.setdefault(variant, {})[parsed.init_seed] = _context_values(manifest)
        elif parsed.branch == small and parsed.tokens == budget and parsed.peak_lr == lr.get(variant):
            losses_100m.setdefault(variant, {})[parsed.init_seed] = loss
        elif parsed.branch == MID_BUDGET and parsed.tokens == budget and parsed.peak_lr == lr.get(variant):
            losses_200m.setdefault(variant, {})[parsed.init_seed] = loss
        elif (
            variant == "mha"
            and parsed.branch is None
            and parsed.tokens == small
            and parsed.peak_lr == lr.get("mha")
        ):
            reruns[parsed.init_seed] = loss

    # The null control and the data-order runs are not in `selected`: the null variant is
    # not a headline variant, and the data-order runs carry other data seeds (plan §8.1).
    null_spec = prereg["null_control"]
    null_seeds = set(null_spec["init_seeds"])
    null_100m: dict[int, float] = {}
    data_order: dict[int, float] = {}
    for manifest in results:
        parsed = mf.parse_run_name(manifest["name"])
        if parsed.arm != arm["tag"] or parsed.branch is not None:
            continue
        loss = float(manifest["final_metrics"]["val_loss"])
        if (
            parsed.variant == null_spec["variant"]
            and parsed.tokens == null_spec["budget"]
            and parsed.data_seed == null_spec["data_seed"]
            and parsed.init_seed in null_seeds
            and parsed.peak_lr == lr.get("mha")
        ):
            null_100m[parsed.init_seed] = loss
        elif (
            parsed.variant == "mha"
            and parsed.tokens == small
            and parsed.init_seed == DATA_ORDER_INIT_SEED
            and parsed.data_seed in DATA_ORDER_EXTRA_SEEDS
            and parsed.peak_lr == lr.get("mha")
        ):
            data_order[parsed.data_seed] = loss
    # Data seed 1 is the Arm A data seed: its value is the 100M branch of the same init seed.
    branch_100m = losses_100m.get("mha", {})
    if DATA_ORDER_INIT_SEED in branch_100m:
        data_order[arm["data_seed"]] = branch_100m[DATA_ORDER_INIT_SEED]
    return _Collected(
        losses, losses_100m, losses_200m, reruns, examples, null_100m, data_order, context
    )


def _context_values(manifest: dict) -> dict:
    """The late-context loss and the ICL score of one 300M result (plan §10.3); None where absent."""
    block = manifest["final_metrics"].get("val")
    late = icl = None
    if isinstance(block, dict):
        late_block = block.get("late_context")
        if isinstance(late_block, dict) and late_block.get("loss") is not None:
            late = float(late_block["loss"])
        icl_block = block.get("icl")
        if isinstance(icl_block, dict) and icl_block.get("score") is not None:
            icl = float(icl_block["score"])
    return {"late_context": late, "icl": icl}


def _missing(seeds, have: Mapping) -> list[int]:
    return [s for s in seeds if s not in have]


def _ladder_fit(by_name: Mapping, examples: Mapping, *, budget: str, prefix: str, notes: list[str]):
    """The ladder fit of one budget (plan §8.4), or None with the reason in `notes`.

    `by_name[name] = {init_seed: loss}` for MHA and the ladder points at that budget.
    MHA and mha_L6 are required; mha_L10 is used when present.
    """
    if not by_name.get("mha"):
        notes.append(f"{prefix}: MHA {budget} results are missing")
        return None
    if not by_name.get("mha_L6"):
        notes.append(f"{prefix}: mha_L6 {budget} results are missing")
        return None
    points = {"mha": (_non_embedding_params(examples["mha"]), by_name["mha"])}
    for name in ("mha_L6", "mha_L10"):
        if by_name.get(name):
            points[name] = (_non_embedding_params(examples[name]), by_name[name])
    try:
        return lf.fit_ladder(points)
    except lf.LadderError as exc:
        notes.append(f"{prefix}: {exc}")
        return None


def _null_control(null_100m: Mapping, mha_100: Mapping, selection, prereg: Mapping, notes: list[str]):
    """The null control (plan §8.1), against MHA's 100M branches of the test-selection seeds."""
    spec = prereg["null_control"]
    missing = _missing(spec["init_seeds"], null_100m)
    if missing:
        notes.append(
            f"null control: {spec['variant']} 100M runs at MHA's peak learning rate for seeds "
            f"{missing} are missing"
        )
        return None
    if selection is None:
        notes.append("null control skipped: no test selection")
        return None
    sel_seeds = list(prereg["test_selection"]["sigma_init"]["init_seeds"])
    missing_ref = _missing(sel_seeds, mha_100)
    if missing_ref:
        notes.append(f"null control: MHA 100M branches for seeds {missing_ref} are missing")
        return None
    try:
        return pt.null_control(
            null_100m,
            {s: mha_100[s] for s in sel_seeds},
            test=selection.test,
            prereg=prereg,
        )
    except pt.PreregError as exc:
        notes.append(f"null control: {exc}")
        return None


def _positive_control(losses: Mapping, selection, prereg: Mapping, notes: list[str]):
    """L6 against L8 at 300M on exactly the registered seeds (plan §8.4)."""
    spec = prereg["positive_control"]
    seeds = list(spec["init_seeds"])
    small = losses.get(spec["small"], {})
    reference = losses.get(spec["reference"], {})
    miss_small = _missing(seeds, small)
    miss_ref = _missing(seeds, reference)
    if miss_small or miss_ref:
        parts = []
        if miss_small:
            parts.append(f"{spec['small']} 300M results for seeds {miss_small}")
        if miss_ref:
            parts.append(f"{spec['reference']} 300M results for seeds {miss_ref}")
        notes.append(f"positive control: {' and '.join(parts)} are missing")
        return None
    if selection is None:
        notes.append("positive control skipped: no test selection")
        return None
    try:
        return pt.positive_control(
            {s: small[s] for s in seeds},
            {s: reference[s] for s in seeds},
            test=selection.test,
            prereg=prereg,
        )
    except pt.PreregError as exc:
        notes.append(f"positive control: {exc}")
        return None


def _collect_arm_b(results: Sequence[dict], arm: Mapping) -> tuple[dict, dict]:
    """Arm B's standalone runs: ({variant: {init_seed: loss}} on the registered seeds, {variant: [seeds]} outside them).

    Raises TableError when one variant's Arm B runs were trained at two peak learning rates.
    """
    variants = set(arm["variants"])
    seeds = set(arm["init_seeds"])
    lr_runs: dict[str, dict[float, list[str]]] = {}
    losses: dict[str, dict[int, float]] = {}
    outside: dict[str, list[int]] = {}
    for manifest in results:
        parsed = mf.parse_run_name(manifest["name"])
        if parsed.arm != arm["tag"] or parsed.data_seed != arm["data_seed"] or parsed.variant not in variants:
            continue
        if parsed.branch is not None or parsed.tokens != arm["budget"]:
            continue
        lr_runs.setdefault(parsed.variant, {}).setdefault(parsed.peak_lr, []).append(manifest["name"])
        if parsed.init_seed in seeds:
            losses.setdefault(parsed.variant, {})[parsed.init_seed] = float(manifest["final_metrics"]["val_loss"])
        else:
            outside.setdefault(parsed.variant, []).append(parsed.init_seed)
    problems = []
    for variant, by_lr in sorted(lr_runs.items()):
        if len(by_lr) > 1:
            listed = "; ".join(
                f"{mf.format_lr(lr)} in {', '.join(names_)}" for lr, names_ in sorted(by_lr.items())
            )
            problems.append(f"{variant} has {arm['budget']} Arm B runs at more than one peak learning rate: {listed}")
    if problems:
        raise TableError(_listing("two learning rates of one variant cannot be mixed", problems))
    return losses, {variant: sorted(found) for variant, found in outside.items()}


def _arm_b_comparisons(
    variant: str, own: Mapping, arm_a: Mapping, mha_100: Mapping, *, arm: Mapping, selection, prereg: Mapping,
    notes: list[str],
) -> tuple[object, object]:
    """(Arm B against MHA, Arm A minus Arm B) for one variant, both None with the reasons in `notes`."""
    seeds = list(arm["init_seeds"])
    test = selection.test
    # Paired: MHA's branches for the arm's seeds. Welch: the test-selection seeds, all offered.
    mha_seeds = seeds if test == pt.PAIRED else list(prereg["test_selection"]["sigma_init"]["init_seeds"])
    vs_mha = arm_minus = None

    missing_own = _missing(seeds, own)
    if missing_own:
        notes.append(f"Arm B {variant}: Arm B runs for seeds {missing_own} are missing; its comparisons are not computed")
        return None, None
    missing_mha = _missing(mha_seeds, mha_100)
    if missing_mha:
        notes.append(f"Arm B {variant}: MHA 100M branches for seeds {missing_mha} are missing; not compared with MHA")
    else:
        try:
            vs_mha = pt.secondary(
                variant,
                {s: own[s] for s in seeds},
                {s: mha_100[s] for s in mha_seeds},
                reference="mha",
                seeds=seeds,
                test=test,
            )
        except pt.PreregError as exc:
            notes.append(f"Arm B {variant}: {exc}")
    missing_a = _missing(seeds, arm_a)
    if missing_a:
        notes.append(f"Arm B {variant}: Arm A 100M branches for seeds {missing_a} are missing; Arm A minus Arm B is not computed")
    else:
        try:
            arm_minus = pt.secondary(
                variant,
                {s: arm_a[s] for s in seeds},
                {s: own[s] for s in seeds},
                reference=f"{variant} at d_ff {arm['d_ff']}",
                seeds=seeds,
                test=test,
            )
        except pt.PreregError as exc:
            notes.append(f"Arm B {variant}: {exc}")
    return vs_mha, arm_minus


def _arm_b_results(
    results: Sequence[dict], losses_100m: Mapping, *, selection, comparing: bool, prereg: Mapping, notes: list[str]
) -> dict:
    """{variant: ArmBResult} in the arm file's order (plan §8.3). Comparisons only when `comparing`."""
    arm = config.load_arm(ARM_B)
    seeds = tuple(arm["init_seeds"])
    runs, outside = _collect_arm_b(results, arm)
    # An arm that has not been run at all is not a missing input of this analysis: arm_b.md says
    # so by its blanks, and the shared notes stay as they are. Once any run exists, a variant
    # without one is noted.
    started = bool(runs)
    mha_100 = losses_100m.get("mha", {})
    out: dict[str, ArmBResult] = {}
    for variant in arm["variants"]:
        own_all = runs.get(variant, {})
        own = {s: own_all[s] for s in seeds if s in own_all}
        if variant in outside:
            notes.append(
                f"Arm B {variant}: runs for seeds {outside[variant]} are outside the registered seeds "
                "and are not used"
            )
        if started and not own_all:
            notes.append(f"Arm B {variant}: no standalone {arm['budget']} run of the arm (seeds {list(seeds)})")
        arm_a_all = losses_100m.get(variant, {})
        arm_a = {s: arm_a_all[s] for s in seeds if s in arm_a_all}
        vs_mha = arm_minus = None
        if comparing and own_all:
            vs_mha, arm_minus = _arm_b_comparisons(
                variant, own, arm_a, mha_100, arm=arm, selection=selection, prereg=prereg, notes=notes
            )
        out[variant] = ArmBResult(variant, own, arm_a, vs_mha, arm_minus)
    return out


def _context_use(
    context: Mapping, *, selection, decision, prereg: Mapping, comparing: bool, notes: list[str]
) -> dict:
    """{variant: ContextUse} in headline order, without MHA (plan §10.3). Empty unless `comparing`."""
    if not comparing or decision is None:
        return {}
    # Until some 300M result carries the context-use fields, nothing is measured and nothing is
    # missing; once one does, every absent value is noted (the same rule as Arm B's runs).
    measured = any(
        per_seed.get(key) is not None
        for per_variant in context.values()
        for per_seed in per_variant.values()
        for key, _ in CONTEXT_METRICS
    )
    if not measured:
        return {}
    test = selection.test
    primaries = tuple(prereg["primary"]["comparisons"])
    exploratory = tuple(prereg["exploratory"]["variants"])
    ex_seeds = tuple(prereg["exploratory"]["init_seeds"])
    out: dict[str, ContextUse] = {}
    for variant in HEADLINE_VARIANTS:
        if variant == "mha":
            continue
        if decision.extended and variant in decision.leaves_arm_a:
            continue  # skipped exactly as its loss comparison is
        if variant in primaries:
            seeds = tuple(decision.seeds)
        elif variant in exploratory:
            seeds = ex_seeds
        else:
            continue
        # Paired: MHA for the variant's seeds. Welch: MHA for the registered decision seeds.
        ref_seeds = seeds if test == pt.PAIRED else tuple(decision.seeds)
        values = {
            key: {s: context.get(variant, {}).get(s, {}).get(key) for s in seeds} for key, _ in CONTEXT_METRICS
        }
        ref_values = {
            key: {s: context.get("mha", {}).get(s, {}).get(key) for s in ref_seeds} for key, _ in CONTEXT_METRICS
        }
        comparisons: dict[str, object] = {}
        for key, label in CONTEXT_METRICS:
            missing_own = [s for s in seeds if values[key][s] is None]
            missing_ref = [s for s in ref_seeds if ref_values[key][s] is None]
            comparisons[key] = None
            if missing_own:
                notes.append(
                    f"context use: {variant} {label} is missing for seeds {missing_own}; its comparison is not computed"
                )
            if missing_ref:
                notes.append(
                    f"context use: mha {label} is missing for seeds {missing_ref}; "
                    f"the comparison of {variant} with it is not computed"
                )
            if not missing_own and not missing_ref:
                comparisons[key] = pt.secondary(
                    variant,
                    {s: values[key][s] for s in seeds},
                    {s: ref_values[key][s] for s in ref_seeds},
                    reference="mha",
                    seeds=seeds,
                    test=test,
                )
        out[variant] = ContextUse(
            variant=variant,
            seeds=seeds,
            late_context_values=values["late_context"],
            icl_values=values["icl"],
            late_context=comparisons["late_context"],
            icl_score=comparisons["icl"],
        )
    return out


def analyse(results: Sequence[dict], prereg: Mapping | None = None) -> Analysis:
    """Every number of the table, computed in the pre-registered order. Refuses incomparable runs."""
    check_comparable(results)
    if prereg is None:
        prereg = config.load_prereg()
    collected = _collect(results, prereg)
    losses = collected.losses
    losses_100m = collected.losses_100m
    reruns = collected.reruns
    examples = collected.examples
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

    # 1b. The null control (plan §8.1): before any ladder or cross-variant number is read.
    null_control = _null_control(collected.null_100m, mha_100, selection, prereg, notes)

    # 2. Ladder fit at 300M (plan §8.4).
    mha_300 = losses.get("mha", {})
    fit = _ladder_fit(losses, examples, budget="300M", prefix="ladder fit", notes=notes)

    # 2b. The positive control (plan §8.4, §9.4): L6 against L8 at 300M.
    positive_control = _positive_control(losses, selection, prereg, notes)

    # 2c. The same ladder at the two branch budgets (plan §8.4): a fit per budget, read only.
    fit_100m = _ladder_fit(
        losses_100m, examples, budget="100M", prefix="ladder fit at 100M", notes=notes
    )
    fit_200m = _ladder_fit(
        collected.losses_200m, examples, budget=MID_BUDGET, prefix="ladder fit at 200M", notes=notes
    )

    # The noise components of plan §8.1, from their registered runs only.
    null_100m = collected.null_100m
    data_order = collected.data_order_100m
    sigma_init_null = None
    if all(s in null_100m for s in prereg["null_control"]["init_seeds"]):
        sigma_init_null = float(pt.sample_sd([null_100m[s] for s in prereg["null_control"]["init_seeds"]]))
    needed_data = (config.load_arm(ARM)["data_seed"], *DATA_ORDER_EXTRA_SEEDS)
    sigma_data = None
    missing_data = [d for d in needed_data if d not in data_order]
    if missing_data:
        notes.append(f"sigma_data: data-order runs for data seeds {missing_data} are missing")
    else:
        sigma_data = float(pt.sample_sd([data_order[d] for d in needed_data]))

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

    # The gate (plan §8.1, §8.4, §9.3, §9.4): no cross-variant number before the controls.
    gate_open = True
    if null_control is None:
        gate_open = False
        notes.append("comparisons skipped: null control not computed")
    elif null_control.verdict == pt.VERDICT_DETECTED:
        gate_open = False
        notes.append(
            "comparisons skipped: the null control is detected, so a bug in pairing, seeding, the "
            "statistics code or determinism is possible; nothing downstream is valid until it passes "
            "(plan §9.3)"
        )
    if positive_control is None:
        gate_open = False
        notes.append("comparisons skipped: positive control not computed")
    elif positive_control.verdict != pt.VERDICT_DETECTED:
        notes.append(
            "positive control not detected: the pipeline cannot see an effect of the size of a 25% "
            "capacity cut, so every null in this report is uninformative (plan §8.4)"
        )

    # 4. The three primary comparisons, Holm-corrected together (plan §1.2).
    comparisons: dict[str, pt.Comparison] = {}
    primary_names = [prereg["primary"]["reference"], *prereg["primary"]["comparisons"]]
    if not gate_open:
        pass  # the gate notes above say why
    elif decision is None:
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
    if not gate_open:
        pass  # the gate notes above say why
    elif decision is None:
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
    if gate_open and decision is not None:
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

    # 7. Arm B (plan §8.3): exploratory, its own standalone 100M runs. Losses are listed even
    # while the gate is shut; the comparisons wait for it.
    comparing = gate_open and selection is not None
    arm_b = _arm_b_results(
        results, losses_100m, selection=selection, comparing=comparing, prereg=prereg, notes=notes
    )

    # 8. Context use (plan §10.3): secondary, at 300M, every headline variant but MHA.
    context_use = _context_use(
        collected.context, selection=selection, decision=decision, prereg=prereg, comparing=comparing, notes=notes
    )

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
        losses_200m=collected.losses_200m,
        fits={"100M": fit_100m, "200M": fit_200m, "300M": fit},
        null_100m=null_100m,
        data_order_100m=data_order,
        sigma_data=sigma_data,
        sigma_init_null=sigma_init_null,
        null_control=null_control,
        positive_control=positive_control,
        comparisons_allowed=gate_open,
        arm_b=arm_b,
        context_use=context_use,
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


# --- The gate report (plan §9.3, §9.4) ----------------------------------------------

# Section headings and row labels, in order. The report, RUNBOOK.md and the plots use them.
GATE_SECTIONS: dict[str, tuple[str, ...]] = {
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
    "Ladder slope b by budget": ("100M", "200M", "300M"),
}

NULL_WARNING = (
    "**NULL CONTROL DETECTED (plan §9.3).** A detected null control means a bug in pairing, "
    "seeding, the statistics code or determinism. Nothing downstream is valid until it passes. "
    "No cross-variant comparison is computed in this report."
)


def _attr(obj, name: str):
    return None if obj is None else getattr(obj, name)


def gate_values(analysis: Analysis) -> dict[str, dict[str, object]]:
    """The raw value of every row of the gate report, from the analysis objects. No rounding."""
    sel, nc, pc, d = analysis.selection, analysis.null_control, analysis.positive_control, analysis.decision
    fit = analysis.fit
    null_interval = None if nc is None else (nc.ci_low, nc.ci_high)
    control_interval = None if pc is None else (pc.ci_low, pc.ci_high)
    leaves = None if d is None else (d.leaves_arm_a if d.extended else ())
    return {
        "Noise floor at 100M (Gate 2)": {
            "sigma_init, MHA seeds 1-5": _attr(sel, "sigma_init"),
            "sigma_init, null-control seeds": analysis.sigma_init_null,
            "sigma_data, MHA seed 1 over data seeds 1-3": analysis.sigma_data,
            "s_rep": _attr(sel, "s_rep"),
            "s_rep / sigma_init": _attr(sel, "ratio"),
            "pairing threshold": _attr(sel, "threshold"),
            "test chosen": _attr(sel, "test"),
        },
        "Null control (Gate 2)": {
            "null seeds": _attr(nc, "seeds"),
            "MHA seeds": _attr(nc, "reference_seeds"),
            "mean gap, nats": _attr(nc, "mean_gap"),
            "95% interval": null_interval,
            "p": _attr(nc, "p"),
            "verdict": _attr(nc, "verdict"),
        },
        "Positive control: L6 against L8 at 300M": {
            "seeds": _attr(pc, "seeds"),
            "mean gap, nats": _attr(pc, "mean_gap"),
            "95% interval": control_interval,
            "p": _attr(pc, "p"),
            "verdict": _attr(pc, "verdict"),
        },
        "Seed count and detectable effect": {
            "sigma_init at 300M, MHA seeds 1-5": _attr(d, "sigma_init"),
            "ladder slope b at 300M": _attr(fit, "b"),
            "5-seed detectable gap, nats": _attr(d, "detectable_gap"),
            "undetectable parameter fraction at 5 seeds": _attr(d, "undetectable_param_fraction"),
            "threshold": _attr(d, "threshold"),
            "seeds per primary": _attr(d, "n_seeds"),
            "variants leaving Arm A": leaves,
            "detectable gap of the chosen design, nats": _attr(d, "bound_gap"),
        },
        "Ladder slope b by budget": {
            budget: _attr(analysis.fits.get(budget), "b") for budget in LADDER_BUDGETS
        },
    }


def _format_value(value) -> str:
    if value is None:
        return BLANK
    if isinstance(value, str):
        return _cell(value)
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        return f"{float(value):.6g}"
    if isinstance(value, tuple):
        if not value:
            return "none"
        if all(isinstance(x, float) for x in value):
            return "[" + ", ".join(_format_value(x) for x in value) + "]"
        return ", ".join(_format_value(x) for x in value)
    raise TypeError(f"no rendering for {value!r}")


def render_gates(analysis: Analysis) -> str:
    """The gate report as Markdown: one table per section, then the analysis notes verbatim."""
    values = gate_values(analysis)
    lines: list[str] = []
    nc = analysis.null_control
    if nc is not None and nc.verdict == pt.VERDICT_DETECTED:
        lines += [f"> {NULL_WARNING}", ""]
    for section, labels in GATE_SECTIONS.items():
        lines += [f"## {section}", "", "| item | value |", "|---|---|"]
        for label in labels:
            lines.append(f"| {_cell(label)} | {_format_value(values[section][label])} |")
        lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines += [f"- {note}" for note in analysis.notes] or ["none"]
    return "\n".join(lines) + "\n"


# --- Arm B and context use: raw values, then rendering (plan §8.3, §10.3) ------------


def _seed_cell(seeds: Sequence[int], values: Mapping[int, float | None]) -> str:
    """\"s1 3.1234, s2 —, s3 3.0000\" over the registered seeds; a dash when no seed has a value."""
    if all(values.get(s) is None for s in seeds):
        return BLANK
    parts = []
    for s in seeds:
        value = values.get(s)
        parts.append(f"s{s} {BLANK}" if value is None else f"s{s} {value:.4f}")
    return ", ".join(parts)


def _gap_text(comparison) -> str:
    if comparison is None:
        return BLANK
    return f"{comparison.mean_gap:+.4f} [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]"


def _verdict_text(comparison) -> str:
    return BLANK if comparison is None else comparison.verdict


def _md_table(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[column]) for column in columns) + " |")
    return lines


def _ranking(by_variant: Mapping[str, Mapping[int, float]], variants: Sequence[str], seeds: Sequence[int]) -> dict:
    """Variants with every registered seed, ranked by the plain mean of their losses, lowest first."""
    ranked: list[tuple[str, float]] = []
    unranked: list[str] = []
    for variant in variants:
        found = by_variant.get(variant, {})
        if all(s in found for s in seeds):
            ranked.append((variant, sum(found[s] for s in seeds) / len(seeds)))
        else:
            unranked.append(variant)
    ranked.sort(key=lambda item: item[1])  # a stable sort: ties keep the arm's order
    return {"ranked": ranked, "unranked": unranked}


def arm_b_values(analysis: Analysis) -> dict:
    """The raw Arm B values: losses by seed, the two comparison objects, and the two rankings.

    `rankings` is None while the gate is shut: a ranking is a comparison across arms.
    `same order` is None unless both rankings name every variant.
    """
    arm = config.load_arm(ARM_B)
    variants = tuple(arm["variants"])
    seeds = tuple(arm["init_seeds"])
    rows: dict[str, dict] = {}
    for variant in variants:
        result = analysis.arm_b.get(variant)
        rows[variant] = {
            "Arm B losses": {} if result is None else result.losses,
            "vs MHA": None if result is None else result.vs_mha,
            "Arm A 100M losses": {} if result is None else result.arm_a_losses,
            "Arm A minus Arm B": None if result is None else result.arm_a_minus_arm_b,
        }
    rankings = None
    same = None
    if analysis.comparisons_allowed:
        rankings = {
            "Arm A at 100M": _ranking({v: rows[v]["Arm A 100M losses"] for v in variants}, variants, seeds),
            "Arm B": _ranking({v: rows[v]["Arm B losses"] for v in variants}, variants, seeds),
        }
        a, b = rankings["Arm A at 100M"], rankings["Arm B"]
        if not a["unranked"] and not b["unranked"]:
            same = [v for v, _ in a["ranked"]] == [v for v, _ in b["ranked"]]
    return {"variants": rows, "rankings": rankings, "same order": same}


def render_arm_b(analysis: Analysis) -> str:
    """arm_b.md: Arm B against MHA and Arm A minus Arm B, with the two rankings (plan §8.3)."""
    values = arm_b_values(analysis)
    arm = config.load_arm(ARM_B)
    seeds = list(arm["init_seeds"])
    lines = ["# Arm B: iso-backbone (plan §8.3)", "", ARM_B_CAPTION, "", ARM_B_MQA_LINE, ""]
    if not analysis.comparisons_allowed:
        lines += [f"Not computed: {GATE_WAIT} (see gates.md).", ""]
    rows = []
    for variant, row in values["variants"].items():
        rows.append(
            {
                "variant": variant,
                "Arm B val loss, every seed": _seed_cell(seeds, row["Arm B losses"]),
                "gap vs MHA [95% CI]": _gap_text(row["vs MHA"]),
                "verdict (vs MHA)": _verdict_text(row["vs MHA"]),
                "Arm A 100M val loss, every seed": _seed_cell(seeds, row["Arm A 100M losses"]),
                "Arm A minus Arm B [95% CI]": _gap_text(row["Arm A minus Arm B"]),
                "verdict (Arm A minus Arm B)": _verdict_text(row["Arm A minus Arm B"]),
            }
        )
    lines += _md_table(ARM_B_COLUMNS, rows)
    lines += ["", "Rankings by mean loss, lowest first (variants of the arm only; a description, not a test):", ""]
    rankings = values["rankings"]
    if rankings is None:
        lines.append(f"Not stated: {GATE_WAIT}.")
    else:
        for label in ("Arm A at 100M", "Arm B"):
            ranking = rankings[label]
            lines.append(f"{label}:")
            lines.append("")
            if ranking["ranked"]:
                lines += [f"{i}. {variant}, mean {mean:.4f}" for i, (variant, mean) in enumerate(ranking["ranked"], 1)]
            else:
                lines.append("Not stated: no variant has a complete set of seeds.")
            if ranking["unranked"]:
                lines.append(f"Not ranked, seeds missing: {', '.join(ranking['unranked'])}.")
            lines.append("")
        same = values["same order"]
        if same is None:
            lines.append("Not stated: a variant is not ranked in both lists.")
        elif same:
            lines.append("The two orders are the same.")
        else:
            lines.append("The two orders differ.")
    notes = [note for note in analysis.notes if note.startswith("Arm B")]
    if notes:
        lines += ["", "Notes:", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def context_values(analysis: Analysis) -> dict:
    """The raw context-use values per variant: losses and ICL scores by seed, and both comparisons."""
    return {
        variant: {
            "late-context losses": cu.late_context_values,
            "late-context vs MHA": cu.late_context,
            "ICL scores": cu.icl_values,
            "ICL vs MHA": cu.icl_score,
        }
        for variant, cu in analysis.context_use.items()
    }


def _left_arm(analysis: Analysis, variant: str) -> bool:
    decision = analysis.decision
    return decision is not None and decision.extended and variant in decision.leaves_arm_a


def render_context(analysis: Analysis) -> str:
    """context.md: late-context loss and ICL score against MHA, with intervals (plan §10.3)."""
    values = context_values(analysis)
    lines = ["# Context use (plan §10.3)", "", CONTEXT_CAPTION, "", SWA_CAPTION, ""]
    if not analysis.comparisons_allowed:
        lines += [f"Not computed: {GATE_WAIT} (see gates.md).", ""]
    rows = []
    for variant in HEADLINE_VARIANTS:
        if variant == "mha":
            continue
        row = values.get(variant)
        if row is None:
            verdict = LEFT_ARM if _left_arm(analysis, variant) else BLANK
            rows.append(
                {
                    "variant": variant,
                    "late-context loss, every seed": BLANK,
                    "gap vs MHA [95% CI], late-context": BLANK,
                    "verdict (late-context)": verdict,
                    "ICL score, every seed": BLANK,
                    "gap vs MHA [95% CI], ICL": BLANK,
                    "verdict (ICL)": verdict,
                }
            )
            continue
        cu = analysis.context_use[variant]
        rows.append(
            {
                "variant": variant,
                "late-context loss, every seed": _seed_cell(cu.seeds, row["late-context losses"]),
                "gap vs MHA [95% CI], late-context": _gap_text(row["late-context vs MHA"]),
                "verdict (late-context)": _verdict_text(row["late-context vs MHA"]),
                "ICL score, every seed": _seed_cell(cu.seeds, row["ICL scores"]),
                "gap vs MHA [95% CI], ICL": _gap_text(row["ICL vs MHA"]),
                "verdict (ICL)": _verdict_text(row["ICL vs MHA"]),
            }
        )
    lines += _md_table(CONTEXT_COLUMNS, rows)
    if not analysis.context_use:
        lines += ["", "Not computed: no 300M result carries the context-use fields yet, or the controls have not passed."]
    notes = [note for note in analysis.notes if note.startswith("context use")]
    if notes:
        lines += ["", "Notes:", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


# --- RQ-W: the computed wall-clock estimate (plan §8.5, step 2) ----------------------


def render_rqw(estimate: rq.WallclockEstimate | None, reason: str | None) -> str:
    """rqw.md. With no estimate, the blank form with `reason` (default: no timing pass was given)."""
    lines = ["# RQ-W: wall-clock estimate (plan §8.5)", ""]
    if estimate is None:
        lines += [f"Budget B: {BLANK}", "", RQW_CAPTION, "", f"Not computed: {reason or NO_TIMING}.", ""]
        return "\n".join(lines) + "\n"
    lines += [
        f"Budget B: {estimate.budget_ms:.1f} ms, MHA's measured time for {estimate.reference_steps} steps "
        f"at {estimate.reference_ms_per_step:.3f} ms per step.",
        "",
        RQW_CAPTION,
        "",
        f"Interpolation: {rq.INTERPOLATION}.",
        "",
    ]
    rows = []
    for v in estimate.variants:
        rows.append(
            {
                "variant": v.variant,
                "ms/step": f"{v.ms_per_step:.3f}",
                "step-time ratio to MHA": f"{v.step_time_ratio:.3f}",
                "steps afforded S_v": str(v.steps),
                "extrapolated": "yes" if v.extrapolated else "no",
                "estimated loss, every seed": _seed_cell(v.seeds, dict(zip(v.seeds, v.loss_per_seed))),
                "MHA 100M loss, every seed": _seed_cell(v.seeds, dict(zip(v.seeds, v.reference_loss_per_seed))),
                "gap, estimate minus MHA (nats)": BLANK if v.gap is None else f"{v.gap:+.4f}",
                "status": "estimated" if v.status == "estimated" else f"ratio only: {v.reason}",
            }
        )
    lines += _md_table(RQW_COLUMNS, rows)
    return "\n".join(lines) + "\n"


def _rqw_report(analysis: Analysis, results: Sequence[dict], ms_per_step: Mapping[str, float] | None) -> str:
    if ms_per_step is None:
        return render_rqw(None, NO_TIMING)
    if not analysis.comparisons_allowed:
        return render_rqw(None, GATE_WAIT)
    try:
        estimate = rq.estimate(list(results), dict(ms_per_step))
    except rq.EstimateError as exc:
        return render_rqw(None, f"the estimate cannot be computed: {exc}")
    return render_rqw(estimate, None)


# --- Timing and bench sources: refusals (plan §2.2, §10.6, §11) ----------------------

ENVIRONMENT_FIELDS = (("torch version", "torch"), ("CUDA version", "cuda"), ("driver version", "driver"), ("GPU name", "gpu_name"))
TIMING_LABEL = "timing pass"


def _disagreement(what: str, items: Sequence[tuple[str, object]]) -> str | None:
    """One violation line when the labelled values of `what` are not all equal."""
    groups: dict[str, tuple[object, list[str]]] = {}
    for label, value in items:
        groups.setdefault(repr(value), (value, []))[1].append(label)
    if len(groups) <= 1:
        return None
    parts = "; ".join(f"{value!r} in {', '.join(names)}" for value, names in groups.values())
    return f"{what} differs: {parts}"


def check_timing_sources(results: Sequence[dict], timing_doc: dict | None, bench_docs: Sequence[dict]) -> None:
    """Raise TableError listing every reason the timing and bench sources cannot be used with `results`.

    Both the timing pass and every bench document must have been recorded alone on the GPU (plan §2.2).
    Every timing source must agree with the others, and with each result, on torch, CUDA, driver, GPU
    name and backend (plan §10.6, §11). The backend is compared on the opt documents only: a ref
    document records no backend. Results are compared only when a timing or bench source is given.
    """
    sources: list[tuple[str, dict, str]] = []  # (label, document, mode)
    if timing_doc is not None:
        sources.append((TIMING_LABEL, timing_doc, "opt"))
    for doc in bench_docs:
        sources.append((f"bench {doc.get('variant')}/{doc.get('mode')}", doc, doc.get("mode")))
    if not sources:
        return
    problems: list[str] = []
    for label, doc, _ in sources:
        if doc.get("gpu_shared") is not False:
            problems.append(
                f"{label}: recorded with gpu_shared {doc.get('gpu_shared')!r}; it must run alone on the GPU (plan §2.2)"
            )
    fields = [(what, _field("environment", key)) for what, key in ENVIRONMENT_FIELDS]
    fields.append(("backend", _field("backend")))
    for what, getter in fields:
        scope = [(label, doc) for label, doc, mode in sources if what != "backend" or mode == "opt"]
        if not scope:
            continue
        line = _disagreement(what, [(label, getter(doc)) for label, doc in scope])
        if line is not None:
            problems.append(line)
        if results:
            reference = getter(scope[0][1])
            items = [("the timing and bench documents", reference)] + [
                (manifest["name"], getter(manifest)) for manifest in results
            ]
            line = _disagreement(f"{what} of the results against the timing and bench documents", items)
            if line is not None:
                problems.append(line)
    if problems:
        raise TableError(_listing("the timing and bench sources cannot be used with these runs", problems))


def _load_bench_documents(directory: str | Path) -> list[dict]:
    """Every bench_*.json document in `directory`, schema-checked. The readers check gpu_shared."""
    paths = sorted(Path(directory).glob("bench_*.json"))
    if not paths:
        raise TableError(f"{directory}: no bench documents (bench_*.json)")
    docs = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TableError(f"{path.name}: not JSON: {exc}") from exc
        if not isinstance(doc, dict) or doc.get("schema") != bd.SCHEMA:
            found = doc.get("schema") if isinstance(doc, dict) else None
            raise TableError(f"{path.name}: schema {found!r}, expected {bd.SCHEMA!r}")
        docs.append(doc)
    return docs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the headline table and the secondary reports from run manifests.")
    parser.add_argument("--manifests", default="results/manifests", help="directory of manifests")
    parser.add_argument("--out", default="results/tables", help="directory to write the tables to")
    parser.add_argument(
        "--skip-unfinished", action="store_true", help="leave unfinished runs out, by name"
    )
    parser.add_argument(
        "--blank", action="store_true", help="ignore --manifests; write the tables for no runs"
    )
    parser.add_argument(
        "--timing-pass", default=None, metavar="FILE",
        help="a timing_pass/1 document: fills train ms/step and writes the RQ-W estimate",
    )
    parser.add_argument(
        "--bench", default=None, metavar="DIR",
        help="a directory of bench_*.json documents: fills decode tok/s",
    )
    args = parser.parse_args(argv)

    try:
        manifests = [] if args.blank else load_manifests(args.manifests)
        results, skipped = select_results(manifests, skip_unfinished=args.skip_unfinished)
        analysis = analyse(results)
        timing_doc = tp.read_timing_pass(args.timing_pass) if args.timing_pass else None
        bench_docs = _load_bench_documents(args.bench) if args.bench else []
        check_timing_sources(results, timing_doc, bench_docs)
        ms_per_step = tp.read_ms_per_step(args.timing_pass) if timing_doc is not None else None
        decode = bd.read_decode_tok_per_s(args.bench) if args.bench else None
    except (TableError, mf.ManifestError, tp.TimingError, bd.BenchError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    notes = list(analysis.notes)
    if skipped:
        notes.append("unfinished runs left out: " + ", ".join(skipped))
    noted = dataclasses.replace(analysis, notes=tuple(notes))
    texts = {
        "headline.md": render_markdown(
            headline_rows(analysis, train_ms_per_step=ms_per_step, decode_tok_per_s=decode), notes
        ),
        "gates.md": render_gates(noted),
        "arm_b.md": render_arm_b(noted),
        "context.md": render_context(noted),
        "rqw.md": _rqw_report(analysis, results, ms_per_step),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
