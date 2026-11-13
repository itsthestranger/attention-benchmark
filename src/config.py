"""Loads the frozen YAML configs and resolves one run's configuration (plan §3, §7, §8).

Every public function returns fresh objects: the YAML files are re-read on each
call and anything derived from them is copied, so callers may mutate results.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_VARIANT_DIR = CONFIG_DIR / "variants"
_LADDER_DIR = CONFIG_DIR / "ladder"
_ARM_DIR = CONFIG_DIR / "arms"

# Plan §7.2: the ladder adds the 200M branch to the default 100M branch.
_LADDER_BRANCHES = ["100M", "200M"]


class ConfigError(ValueError):
    """A config file is malformed, or a request names something unknown or inconsistent."""


def _read(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _stems(folder: Path) -> list[str]:
    return sorted(p.stem for p in folder.glob("*.yaml"))


def _named(kind: str, name: object, folder: Path) -> dict:
    names = _stems(folder)
    if not isinstance(name, str) or name not in names:
        raise ConfigError(f"unknown {kind} {name!r}; known: {names}")
    return _read(folder / f"{name}.yaml")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# --- Loaders -----------------------------------------------------------------


def load_base() -> dict:
    return _read(CONFIG_DIR / "base.yaml")


def load_recipe() -> dict:
    return _read(CONFIG_DIR / "recipe.yaml")


def load_prereg() -> dict:
    return _read(CONFIG_DIR / "prereg.yaml")


# Decision 220: the two declarations Phil makes during the campaign, by pull request at the gate.
_COMMITMENT_KEYS = ("ladder", "seed_count")
_COMMITTED_LADDERS = ("full", "l6_l8")


def load_commitments(path: str | Path | None = None) -> dict:
    """configs/commitments.yaml (or `path`): the ladder and the committed seed count, validated."""
    if path is None:
        path = CONFIG_DIR / "commitments.yaml"
    try:
        held = _read(Path(path))
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read commitments: {exc}") from exc
    if set(held) != set(_COMMITMENT_KEYS):
        raise ConfigError(f"{path}: the keys must be exactly {list(_COMMITMENT_KEYS)}, got {sorted(held)}")
    ladder = held["ladder"]
    if not isinstance(ladder, str) or ladder not in _COMMITTED_LADDERS:
        raise ConfigError(f"{path}: ladder must be one of {list(_COMMITTED_LADDERS)}, got {ladder!r}")
    seed_count = held["seed_count"]
    if seed_count is not None:
        allowed = _seed_counts()
        if not _is_int(seed_count) or seed_count not in allowed:
            raise ConfigError(f"{path}: seed_count must be null or one of {allowed} (the rule's counts), got {seed_count!r}")
    return {"ladder": ladder, "seed_count": seed_count}


def _seed_counts() -> list[int]:
    sc = load_prereg()["seed_count"]
    return [len(sc["base_seeds"]), len(sc["base_seeds"]) + len(sc["added_seeds"])]


def load_variant(name: str) -> dict:
    return _named("variant", name, _VARIANT_DIR)


def load_ladder(name: str) -> dict:
    return _named("ladder", name, _LADDER_DIR)


def load_arm(name: str) -> dict:
    return _named("arm", name, _ARM_DIR)


def variant_names() -> list[str]:
    return _stems(_VARIANT_DIR)


def ladder_names() -> list[str]:
    return _stems(_LADDER_DIR)


def arm_names() -> list[str]:
    return _stems(_ARM_DIR)


# --- Resolution --------------------------------------------------------------


def resolve(
    variant: str,
    arm: str = "iso_token",
    *,
    init_seed: int,
    peak_lr: float,
    ladder: str | None = None,
    data_seed: int | None = None,
    budget: str | None = None,
    steps: int | None = None,
    branches: list[str] | None = None,
    overrides: dict | None = None,
) -> dict:
    """One run's full configuration as a plain dict (see module docstring)."""
    if not _is_int(init_seed) or init_seed < 0:
        raise ConfigError(f"init_seed must be a non-negative int, got {init_seed!r}")
    if not _is_number(peak_lr) or not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ConfigError(f"peak_lr must be a positive float, got {peak_lr!r}")
    if overrides is not None and not isinstance(overrides, dict):
        raise ConfigError(f"overrides must be a dict, got {type(overrides).__name__}")

    base = load_base()
    recipe = load_recipe()
    var = load_variant(variant)
    arm_doc = load_arm(arm)
    budgets = recipe["budgets"]

    # Rule 2: an arm may restrict which variants it runs.
    if "variants" in arm_doc and variant not in arm_doc["variants"]:
        raise ConfigError(
            f"variant {variant!r} is not run in arm {arm!r} (allowed: {arm_doc['variants']})"
        )

    # Rule 1: d_ff comes from the variant, or is fixed by the arm.
    model = copy.deepcopy(base["model"])
    d_ff = arm_doc["d_ff"]
    if d_ff == "from_variant":
        d_ff = var["d_ff"]
    elif not _is_int(d_ff):
        raise ConfigError(f"arm {arm!r}: d_ff must be an int or 'from_variant', got {d_ff!r}")
    model["d_ff"] = d_ff
    model["attention"] = copy.deepcopy(var["attention"])

    # Rule 3: a ladder run is an MHA run with a different layer count.
    if ladder is not None:
        ladder_doc = _named("ladder", ladder, _LADDER_DIR)
        if arm != "iso_token":
            raise ConfigError(f"ladder {ladder!r} requires arm 'iso_token', not {arm!r}")
        if variant != ladder_doc.get("variant"):
            raise ConfigError(
                f"ladder {ladder!r} runs variant {ladder_doc.get('variant')!r}, not {variant!r}"
            )
        model.update(copy.deepcopy(ladder_doc.get("model", {})))

    # Rule 4.
    if data_seed is None:
        data_seed = arm_doc["data_seed"]
    if not _is_int(data_seed):
        raise ConfigError(f"data_seed must be an int, got {data_seed!r}")

    # Rule 5: explicit steps, or the budget label mapped through the recipe.
    if steps is not None:
        if not _is_int(steps) or steps <= 0:
            raise ConfigError(f"steps must be a positive int, got {steps!r}")
        run_steps = steps
        run_budget = budget
    else:
        label = budget if budget is not None else arm_doc.get("budget")
        if not isinstance(label, str) or label not in budgets:
            raise ConfigError(
                f"budget {label!r} is not a label in recipe budgets {list(budgets)}; "
                "steps must be given explicitly"
            )
        run_budget = label
        run_steps = budgets[label]

    # Rule 6: branches must exist and be shorter than the run.
    if branches is not None:
        if not isinstance(branches, (list, tuple)):
            raise ConfigError(f"branches must be a list of budget labels, got {branches!r}")
        run_branches = list(branches)
    elif ladder is not None:
        run_branches = list(_LADDER_BRANCHES)
    else:
        run_branches = list(arm_doc.get("branches", []))
    for label in run_branches:
        if not isinstance(label, str) or label not in budgets:
            raise ConfigError(f"branch {label!r} is not a label in recipe budgets {list(budgets)}")
        if budgets[label] >= run_steps:
            raise ConfigError(
                f"branch {label!r} ({budgets[label]} steps) must be shorter than the run "
                f"({run_steps} steps)"
            )

    resolved = {
        "variant": variant,
        "arm": arm,
        "ladder": ladder,
        "model": model,
        "init": copy.deepcopy(base["init"]),
        "precision": copy.deepcopy(base["precision"]),
        "params": copy.deepcopy(base["params"]),
        "recipe": recipe,
        "run": {
            "init_seed": init_seed,
            "data_seed": data_seed,
            "peak_lr": float(peak_lr),
            "budget": run_budget,
            "steps": run_steps,
            "branches": run_branches,
        },
        "overrides": copy.deepcopy(overrides) if overrides else {},
    }

    # Rule 8: overrides are applied last, and only to paths that already exist.
    for path, value in (overrides or {}).items():
        _apply_override(resolved, path, value)
    return resolved


def _apply_override(resolved: dict, path: object, value: object) -> None:
    if not isinstance(path, str) or not path or "" in path.split("."):
        raise ConfigError(f"override path must be a dotted string, got {path!r}")
    parts = path.split(".")
    if parts[0] == "overrides":
        raise ConfigError(f"override path {path!r} may not start with 'overrides'")
    node = resolved
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"override path {path!r} does not exist in the resolved config")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ConfigError(f"override path {path!r} does not exist in the resolved config")
    node[parts[-1]] = copy.deepcopy(value)


def require_complete(resolved: dict) -> None:
    """Raise ConfigError if any value inside resolved["recipe"] is still None."""
    unset: list[str] = []

    def walk(node: object, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{prefix}.{index}")
        elif node is None:
            unset.append(prefix)

    walk(resolved["recipe"], "recipe")
    if unset:
        raise ConfigError("unset recipe values: " + ", ".join(unset))


def config_hash(resolved: dict) -> str:
    """SHA-256 hex digest of the canonical JSON form of a resolved config."""
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()
