"""Run names (plan §11) and run manifests (plan §10.6).

Every training run and every branch writes one JSON manifest. A run without a
manifest does not exist, and the analysis refuses to compare runs whose
manifests disagree. This module names runs, builds and validates manifests,
and reads and writes them. It trains nothing and reads no result.
"""

from __future__ import annotations

import contextlib
import copy
import decimal
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.training.schedule import decay_start

SCHEMA_VERSION = 1
KINDS = ("run", "branch", "trunk_checkpoint")
MODES = ("ref", "opt")
PRECISIONS = ("fp32", "fp16")
BACKENDS = ("flash", "efficient", "math", "cudnn")
WALLCLOCK_ARM = "iso_wallclock"

TOP_KEYS = (
    "schema_version",
    "name",
    "kind",
    "completed",
    "git_sha",
    "config_hash",
    "resolved",
    "seeds",
    "data",
    "environment",
    "backend",
    "mode",
    "precision",
    "steps",
    "scaler",
    "gpu_shared",
    "session_ids",
    "resume_count",
    "step_seconds",
    "final_metrics",
    "branch",
    "lr_source",
    "timing_pass_id",
)
SEED_KEYS = ("init_seed", "data_seed")
DATA_KEYS = ("dataset_sha256", "tokenizer_sha256", "order_sha256")
ENVIRONMENT_KEYS = ("python", "torch", "cuda", "driver", "gpu_name", "compute_capability")
STEP_KEYS = ("total", "attempted", "taken", "skipped", "skipped_in_first_100")
SCALER_KEYS = ("final_scale",)
BRANCH_KEYS = ("budget", "steps", "branch_step", "parent_run", "parent_checkpoint")

_VARIANT = r"[A-Za-z0-9_]+"
_ARM = r"[a-z][a-z0-9]*"
_TOKENS = r"[1-9][0-9]*(?:M|steps)"
_BUDGET = r"[1-9][0-9]*M"
_SEED = r"0|[1-9][0-9]*"
_NAME_RE = re.compile(
    rf"(?P<variant>{_VARIANT})-(?P<arm>{_ARM})-(?P<tokens>{_TOKENS})"
    rf"-lr(?P<lr>[^-]+)-s(?P<init>{_SEED})-d(?P<data>{_SEED})"
    rf"(?:-b(?P<branch>{_BUDGET}))?"
)
_VARIANT_RE = re.compile(_VARIANT)
_ARM_RE = re.compile(_ARM)
_TOKENS_RE = re.compile(_TOKENS)
_BUDGET_RE = re.compile(_BUDGET)
# Learning rate 1.2e3 means 1.2e-3: the exponent is the negative power of ten,
# with its minus sign dropped, so every representable value is below 1.
_LR_RE = re.compile(r"([1-9](?:\.[0-9]+)?)e([1-9][0-9]*)")
_HEX = "0123456789abcdef"
_DERIVE_ERRORS = (KeyError, TypeError, ValueError, AttributeError)
_MISSING = object()


class ManifestError(ValueError):
    """A run name or manifest is malformed, or a manifest cannot be written or read."""


# --- Run names ----------------------------------------------------------------


@dataclass(frozen=True)
class RunName:
    variant: str
    arm: str  # the tag, e.g. "isoparam"
    tokens: str  # "300M" or "1400steps"
    peak_lr: float
    init_seed: int
    data_seed: int
    branch: str | None = None


def format_lr(lr: float) -> str:
    """The peak learning rate in shortest scientific form, without the 'lr' prefix.

    1e-3 -> "1e3", 1.2e-3 -> "1.2e3". Raises rather than rounding.
    """
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise ManifestError(f"learning rate must be a number, got {lr!r}")
    if not (math.isfinite(lr) and 0 < lr < 1):
        raise ManifestError(f"learning rate must be strictly between 0 and 1, got {lr!r}")
    _, digits, exponent = decimal.Decimal(repr(float(lr))).as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    text = "".join(str(d) for d in digits)
    if len(text) > 6:
        raise ManifestError(
            f"learning rate {lr!r} needs {len(text)} significant digits; use at most 6"
        )
    scientific_exponent = exponent + len(digits) - 1
    mantissa = text[0] + (f".{text[1:]}" if len(text) > 1 else "")
    return f"{mantissa}e{-scientific_exponent}"


def parse_lr(text: str) -> float:
    """Inverse of format_lr. Only the canonical spelling is accepted."""
    if not isinstance(text, str):
        raise ManifestError(f"learning rate text must be a string, got {text!r}")
    match = _LR_RE.fullmatch(text)
    if match is None:
        raise ManifestError(f"learning rate {text!r} is not of the form 1.2e3 (meaning 1.2e-3)")
    value = float(f"{match.group(1)}e-{match.group(2)}")
    if format_lr(value) != text:
        raise ManifestError(f"learning rate {text!r} is not in canonical form")
    return value


def _check_name_fields(name: RunName) -> None:
    def bad(field: str, value: object, shape: str) -> ManifestError:
        return ManifestError(f"{field} must be {shape}, got {value!r}")

    if not isinstance(name.variant, str) or not _VARIANT_RE.fullmatch(name.variant):
        raise bad("variant", name.variant, "letters, digits and underscores")
    if not isinstance(name.arm, str) or not _ARM_RE.fullmatch(name.arm):
        raise bad("arm", name.arm, "a lowercase tag")
    if not isinstance(name.tokens, str) or not _TOKENS_RE.fullmatch(name.tokens):
        raise bad("tokens", name.tokens, "a budget label such as 300M, or 1400steps")
    for field, value in (("init_seed", name.init_seed), ("data_seed", name.data_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise bad(field, value, "a non-negative int")
    if name.branch is not None and (
        not isinstance(name.branch, str) or not _BUDGET_RE.fullmatch(name.branch)
    ):
        raise bad("branch", name.branch, "None or a budget label such as 100M")


def format_run_name(name: RunName) -> str:
    """`{variant}-{arm}-{tokens}-lr{lr}-s{init}-d{data}[-b{branch}]`."""
    _check_name_fields(name)
    text = (
        f"{name.variant}-{name.arm}-{name.tokens}-lr{format_lr(name.peak_lr)}"
        f"-s{name.init_seed}-d{name.data_seed}"
    )
    if name.branch is not None:
        text += f"-b{name.branch}"
    return text


def parse_run_name(text: str) -> RunName:
    if not isinstance(text, str):
        raise ManifestError(f"run name must be a string, got {text!r}")
    match = _NAME_RE.fullmatch(text)
    if match is None:
        raise ManifestError(
            f"{text!r} is not a run name of the form "
            "variant-arm-tokens-lrLR-sINIT-dDATA[-bBRANCH]"
        )
    return RunName(
        variant=match["variant"],
        arm=match["arm"],
        tokens=match["tokens"],
        peak_lr=parse_lr(match["lr"]),
        init_seed=int(match["init"]),
        data_seed=int(match["data"]),
        branch=match["branch"],
    )


def _run_name_parts(resolved: dict, branch: str | None) -> RunName:
    """The name of a run, without checking that `branch` is one of its branches."""
    run = resolved["run"]
    variant = resolved["ladder"] if resolved["ladder"] is not None else resolved["variant"]
    arm = config.load_arm(resolved["arm"])["tag"]
    budget, steps = run["budget"], run["steps"]
    if isinstance(budget, str) and resolved["recipe"]["budgets"].get(budget) == steps:
        tokens = budget
    else:
        tokens = f"{steps}steps"
    return RunName(
        variant=variant,
        arm=arm,
        tokens=tokens,
        peak_lr=run["peak_lr"],
        init_seed=run["init_seed"],
        data_seed=run["data_seed"],
        branch=branch,
    )


def run_name(resolved: dict, branch: str | None = None) -> str:
    """The name of a run, or of one of its branches when `branch` is a budget label."""
    if branch is not None:
        branches = resolved["run"]["branches"]
        if branch not in branches:
            raise ManifestError(f"branch {branch!r} is not among this run's branches {branches}")
    return format_run_name(_run_name_parts(resolved, branch))


# --- Environment -------------------------------------------------------------


def collect_environment() -> dict:
    """Python, torch and CUDA versions, and the GPU if there is one."""
    import torch  # deferred so that validation needs no torch

    env = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": None,
        "driver": None,
        "gpu_name": None,
        "compute_capability": None,
    }
    if torch.cuda.is_available():
        env["cuda"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        env["compute_capability"] = [int(major), int(minor)]
        env["driver"] = _driver_version()
    return env


def _driver_version() -> str:
    # A guessed driver version would let make_tables compare the wrong thing, so fail instead.
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManifestError(f"CUDA is available but nvidia-smi cannot be run: {exc}") from exc
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        raise ManifestError(
            "CUDA is available but the driver version cannot be read from nvidia-smi: "
            f"{proc.stderr.strip() or 'no output'}"
        )
    return lines[0]


def git_sha(repo_dir: str | Path | None = None) -> str:
    """The full SHA of HEAD. Defaults to this project's repository."""
    cwd = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManifestError(f"git cannot be run in {cwd}: {exc}") from exc
    sha = proc.stdout.strip()
    if proc.returncode != 0 or not _is_hex(sha, 40):
        raise ManifestError(f"git rev-parse HEAD failed in {cwd}: {proc.stderr.strip()}")
    return sha


# --- Manifests ---------------------------------------------------------------


def build_manifest(
    *,
    kind,
    resolved,
    completed,
    git_sha,
    data,
    environment,
    backend,
    mode,
    precision,
    steps,
    final_scale,
    gpu_shared,
    session_ids,
    step_seconds,
    final_metrics,
    branch=None,
    lr_source=None,
    timing_pass_id=None,
) -> dict:
    """Assemble and validate a manifest. Raises ManifestError listing every problem.

    `branch` is the branch block (budget, steps, branch_step, parent_run,
    parent_checkpoint) or None. The name, config hash, seeds and resume count
    are computed here, not passed in.
    """
    resolved, data, environment, steps, session_ids, step_seconds, final_metrics, branch = (
        copy.deepcopy(
            (resolved, data, environment, steps, session_ids, step_seconds, final_metrics, branch)
        )
    )
    problems: list[str] = []

    budget = branch.get("budget") if isinstance(branch, dict) else None
    try:
        name = format_run_name(_run_name_parts(resolved, budget))
    except _DERIVE_ERRORS as exc:
        name = None
        problems.append(f"name: cannot be derived from the resolved config ({exc})")

    try:
        config_hash = config.config_hash(resolved)
    except _DERIVE_ERRORS as exc:
        config_hash = None
        problems.append(f"config_hash: cannot be computed ({exc})")

    run = resolved.get("run") if isinstance(resolved, dict) else None
    seeds = None
    if isinstance(run, dict):
        seeds = {"init_seed": run.get("init_seed"), "data_seed": run.get("data_seed")}
    resume_count = len(session_ids) - 1 if isinstance(session_ids, list) else None

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "kind": kind,
        "completed": completed,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "resolved": resolved,
        "seeds": seeds,
        "data": data,
        "environment": environment,
        "backend": backend,
        "mode": mode,
        "precision": precision,
        "steps": steps,
        "scaler": {"final_scale": final_scale},
        "gpu_shared": gpu_shared,
        "session_ids": session_ids,
        "resume_count": resume_count,
        "step_seconds": step_seconds,
        "final_metrics": final_metrics,
        "branch": branch,
        "lr_source": lr_source,
        "timing_pass_id": timing_pass_id,
    }
    problems.extend(validate_manifest(manifest))
    problems = list(dict.fromkeys(problems))
    if problems:
        raise ManifestError(_listing("invalid manifest", problems))
    return manifest


def validate_manifest(manifest: object) -> list[str]:
    """Every problem found, as messages that start with the dotted path of the field.

    An empty list means valid. Never raises on malformed input.
    """
    if not isinstance(manifest, dict):
        return [f"manifest: must be a JSON object, got {type(manifest).__name__}"]
    c = _Checks()
    c.expect_keys(manifest, "", TOP_KEYS)
    _walk_json(manifest, "", c)
    take = c.take

    take(manifest, "", "schema_version", lambda v: _is_int(v) and v == SCHEMA_VERSION,
         f"the integer {SCHEMA_VERSION}")
    take(manifest, "", "name", _is_str, "a string")
    take(manifest, "", "kind", lambda v: v in KINDS, "one of " + ", ".join(KINDS))
    take(manifest, "", "completed", _is_bool, "true or false")
    take(manifest, "", "git_sha", lambda v: _is_hex(v, 40), "40 lowercase hex characters")
    take(manifest, "", "config_hash", lambda v: _is_hex(v, 64), "64 lowercase hex characters")
    take(manifest, "", "resolved", _is_dict, "an object")

    seeds = c.block(manifest, "", "seeds", SEED_KEYS)
    for key in SEED_KEYS:
        take(seeds, "seeds", key, _is_nonneg_int, "a non-negative int")

    data = c.block(manifest, "", "data", DATA_KEYS)
    for key in DATA_KEYS:
        take(data, "data", key, lambda v: _is_hex(v, 64), "64 lowercase hex characters")

    env = c.block(manifest, "", "environment", ENVIRONMENT_KEYS)
    take(env, "environment", "python", _is_str, "a string")
    take(env, "environment", "torch", _is_str, "a string")
    for key in ("cuda", "driver", "gpu_name"):
        take(env, "environment", key, _optional(_is_str), "null or a string")
    take(env, "environment", "compute_capability", _optional(_is_capability),
         "null or a list of two non-negative ints")

    take(manifest, "", "backend", _optional(lambda v: v in BACKENDS),
         "null or one of " + ", ".join(BACKENDS))
    take(manifest, "", "mode", lambda v: v in MODES, "one of " + ", ".join(MODES))
    take(manifest, "", "precision", lambda v: v in PRECISIONS,
         "one of " + ", ".join(PRECISIONS))

    steps = c.block(manifest, "", "steps", STEP_KEYS)
    for key in STEP_KEYS:
        take(steps, "steps", key, _is_nonneg_int, "a non-negative int")

    scaler = c.block(manifest, "", "scaler", SCALER_KEYS)
    take(scaler, "scaler", "final_scale", _is_float, "a float")

    take(manifest, "", "gpu_shared", _is_bool, "true or false")
    take(manifest, "", "session_ids", _is_str_list, "a non-empty list of strings")
    take(manifest, "", "resume_count", _is_nonneg_int, "a non-negative int")
    take(manifest, "", "step_seconds", _is_float_list, "a list of floats")
    take(manifest, "", "final_metrics", _optional(_is_dict), "null or an object")
    take(manifest, "", "lr_source", _optional(_is_str), "null or a string")
    take(manifest, "", "timing_pass_id", _optional(_is_str), "null or a string")

    if "branch" in manifest:
        raw = manifest["branch"]
        if raw is None:
            c.good["branch"] = None
        elif isinstance(raw, dict):
            c.good["branch"] = raw
            c.expect_keys(raw, "branch", BRANCH_KEYS)
            take(raw, "branch", "budget", _is_str, "a budget label such as 100M")
            take(raw, "branch", "steps", _is_nonneg_int, "a non-negative int")
            take(raw, "branch", "branch_step", _is_nonneg_int, "a non-negative int")
            take(raw, "branch", "parent_run", _is_str, "a string")
            take(raw, "branch", "parent_checkpoint", _is_str, "a string")
        else:
            c.fail("branch: must be null or an object")

    _cross_check(c)
    return c.problems


# --- Validation internals ----------------------------------------------------


class _Checks:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.good: dict[str, object] = {}  # validated values, keyed by dotted path

    def fail(self, message: str) -> None:
        self.problems.append(message)

    def expect_keys(self, block: dict, path: str, keys: tuple) -> None:
        for key in keys:
            if key not in block:
                self.fail(f"{_join(path, key)}: missing")
        for key in block:
            if key not in keys:
                self.fail(f"{_join(path, str(key))}: unknown key")

    def block(self, container: dict, path: str, key: str, keys: tuple) -> dict | None:
        if key not in container:
            return None  # reported as missing at the top level
        value = container[key]
        if not isinstance(value, dict):
            self.fail(f"{_join(path, key)}: must be an object")
            return None
        self.expect_keys(value, _join(path, key), keys)
        return value

    def take(self, block: dict | None, path: str, key: str, check, what: str) -> object:
        if block is None or key not in block:
            return _MISSING
        value = block[key]
        full = _join(path, key)
        if not check(value):
            self.fail(f"{full}: must be {what}")
            return _MISSING
        self.good[full] = value
        return value


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else str(key)


def _walk_json(value: object, path: str, c: _Checks) -> None:
    """Every number is finite, every key a string, every value a JSON type."""
    where = path or "manifest"
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                c.fail(f"{where}: key {key!r} is not a string")
            _walk_json(item, _join(path, str(key)), c)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_json(item, f"{path}[{index}]", c)
    elif value is None or isinstance(value, (bool, int, str)):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            c.fail(f"{where}: {value!r} is not a finite number")
    else:
        c.fail(f"{where}: {type(value).__name__} is not a JSON value")


def _cross_check(c: _Checks) -> None:
    """The consistency rules of plan §10.6, each over values that already passed type checks."""
    g = c.good

    resolved = g.get("resolved", _MISSING)
    run = resolved.get("run") if isinstance(resolved, dict) else None
    recipe = resolved.get("recipe") if isinstance(resolved, dict) else None
    branch = g.get("branch", _MISSING)
    budget = g.get("branch.budget", _MISSING)

    # Rule 2: the name is the name the resolved config gives.
    if "name" in g and resolved is not _MISSING and branch is not _MISSING:
        if branch is None or budget is not _MISSING:
            try:
                expected = format_run_name(
                    _run_name_parts(resolved, None if branch is None else budget)
                )
            except _DERIVE_ERRORS as exc:
                c.fail(f"name: cannot be checked, the resolved config is malformed ({exc})")
            else:
                if g["name"] != expected:
                    c.fail(f"name: {g['name']!r} should be {expected!r} for this resolved config")

    # Rule 3: the hash and the seeds follow the resolved config.
    if "config_hash" in g and resolved is not _MISSING:
        try:
            computed = config.config_hash(resolved)
        except _DERIVE_ERRORS as exc:
            c.fail(f"resolved: cannot be hashed ({exc})")
        else:
            if g["config_hash"] != computed:
                c.fail("config_hash: does not match the resolved config")
    if isinstance(run, dict):
        for key in SEED_KEYS:
            if f"seeds.{key}" in g and key in run and g[f"seeds.{key}"] != run[key]:
                c.fail(f"seeds.{key}: {g[f'seeds.{key}']!r} differs from resolved.run.{key} "
                       f"({run[key]!r})")

    # Rule 4: kind and branch agree; steps.total matches the run or the branch.
    if "kind" in g and branch is not _MISSING:
        if g["kind"] == "branch" and branch is None:
            c.fail("kind: is 'branch' but branch is null")
        if g["kind"] != "branch" and branch is not None:
            c.fail(f"branch: is set but kind is {g['kind']!r}")
    if isinstance(branch, dict):
        _check_branch(c, resolved, run, recipe)
    if "kind" in g and g["kind"] in ("run", "trunk_checkpoint") and isinstance(run, dict):
        if "steps.total" in g and "steps" in run and g["steps.total"] != run["steps"]:
            c.fail(f"steps.total: {g['steps.total']} must equal resolved.run.steps ({run['steps']})")

    # Rule 5: step counts add up.
    if all(f"steps.{key}" in g for key in STEP_KEYS):
        s = {key: g[f"steps.{key}"] for key in STEP_KEYS}
        if s["attempted"] != s["taken"] + s["skipped"]:
            c.fail(f"steps.attempted: {s['attempted']} != taken {s['taken']} + skipped {s['skipped']}")
        if s["attempted"] > s["total"]:
            c.fail(f"steps.attempted: {s['attempted']} exceeds steps.total {s['total']}")
        if s["skipped_in_first_100"] > min(s["skipped"], 100):
            c.fail("steps.skipped_in_first_100: exceeds min(skipped, 100)")
        if "step_seconds" in g and len(g["step_seconds"]) != s["attempted"]:
            c.fail(f"step_seconds: has {len(g['step_seconds'])} entries for "
                   f"{s['attempted']} attempted steps")

    # Rule 6: completed means finished; a trunk checkpoint is never completed.
    if g.get("completed") is True:
        if g.get("kind") == "trunk_checkpoint":
            c.fail("completed: a trunk_checkpoint is never completed; it is a trunk still at "
                   "peak learning rate")
        if "steps.attempted" in g and "steps.total" in g and (
            g["steps.attempted"] != g["steps.total"]
        ):
            c.fail(f"steps.attempted: a completed run must have attempted == total "
                   f"({g['steps.attempted']} != {g['steps.total']})")
        if "final_metrics" in g:
            metrics = g["final_metrics"]
            if not isinstance(metrics, dict):
                c.fail("final_metrics: a completed manifest needs an object with a finite val_loss")
            elif "val_loss" not in metrics:
                c.fail("final_metrics.val_loss: missing from a completed manifest")
            elif not _is_float(metrics["val_loss"]):
                c.fail("final_metrics.val_loss: must be a float")

    # Rule 7: resume count.
    if "session_ids" in g and "resume_count" in g:
        expected = len(g["session_ids"]) - 1
        if g["resume_count"] != expected:
            c.fail(f"resume_count: must be len(session_ids) - 1 = {expected}, "
                   f"got {g['resume_count']}")

    # Rule 8: GPU fields agree with whether a GPU was used.
    if "environment.gpu_name" in g:
        if g["environment.gpu_name"] is None:
            for key in ("cuda", "driver", "compute_capability"):
                if f"environment.{key}" in g and g[f"environment.{key}"] is not None:
                    c.fail(f"environment.{key}: must be null when no GPU is named")
            if "backend" in g and g["backend"] is not None:
                c.fail("backend: must be null when no GPU is named")
            if "precision" in g and g["precision"] != "fp32":
                c.fail("precision: must be fp32 when no GPU is named")
        else:
            for key in ("cuda", "driver", "compute_capability"):
                if f"environment.{key}" in g and g[f"environment.{key}"] is None:
                    c.fail(f"environment.{key}: must be set when a GPU is named")
            if "backend" in g and g["backend"] is None:
                c.fail("backend: must be set when a GPU is named")

    # Rule 9: a timing pass is named exactly for iso_wallclock runs.
    if "timing_pass_id" in g and isinstance(resolved, dict):
        wallclock = resolved.get("arm") == WALLCLOCK_ARM
        if wallclock and g["timing_pass_id"] is None:
            c.fail(f"timing_pass_id: required when resolved.arm is {WALLCLOCK_ARM}")
        if not wallclock and g["timing_pass_id"] is not None:
            c.fail(f"timing_pass_id: must be null unless resolved.arm is {WALLCLOCK_ARM}")


def _check_branch(c: _Checks, resolved, run, recipe) -> None:
    g = c.good
    budgets = recipe.get("budgets") if isinstance(recipe, dict) else None
    schedule = recipe.get("schedule") if isinstance(recipe, dict) else None
    fraction = schedule.get("decay_fraction") if isinstance(schedule, dict) else None

    if "branch.budget" in g:
        budget = g["branch.budget"]
        branches = run.get("branches") if isinstance(run, dict) else None
        if isinstance(branches, list) and budget not in branches:
            c.fail(f"branch.budget: {budget!r} is not among the run's branches {branches}")
        if "branch.steps" in g and isinstance(budgets, dict) and budget in budgets:
            if g["branch.steps"] != budgets[budget]:
                c.fail(f"branch.steps: must be {budgets[budget]} for budget {budget!r}, "
                       f"got {g['branch.steps']}")

    if "branch.steps" in g and fraction is not None:
        try:
            expected_step = decay_start(g["branch.steps"], fraction)
        except _DERIVE_ERRORS:
            expected_step = None
        if expected_step is not None and "branch.branch_step" in g:
            if g["branch.branch_step"] != expected_step:
                c.fail(f"branch.branch_step: must be decay_start(steps, decay_fraction) = "
                       f"{expected_step}, got {g['branch.branch_step']}")

    if "branch.parent_run" in g and isinstance(resolved, dict):
        try:
            expected_parent = format_run_name(_run_name_parts(resolved, None))
        except _DERIVE_ERRORS:
            expected_parent = None
        if expected_parent is not None and g["branch.parent_run"] != expected_parent:
            c.fail(f"branch.parent_run: {g['branch.parent_run']!r} should be {expected_parent!r}")

    if "steps.total" in g and "branch.steps" in g and g["steps.total"] != g["branch.steps"]:
        c.fail(f"steps.total: a branch's total must equal branch.steps "
               f"({g['steps.total']} != {g['branch.steps']})")


def _listing(head: str, problems: list[str]) -> str:
    return head + "".join(f"\n  - {problem}" for problem in problems)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonneg_int(value: object) -> bool:
    return _is_int(value) and value >= 0


def _is_float(value: object) -> bool:
    return isinstance(value, float)


def _is_float_list(value: object) -> bool:
    return isinstance(value, list) and all(_is_float(item) for item in value)


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_dict(value: object) -> bool:
    return isinstance(value, dict)


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(_is_str(item) for item in value)


def _is_capability(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_nonneg_int(v) for v in value)


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(ch in _HEX for ch in value)


def _optional(check):
    return lambda value: value is None or check(value)


# --- Files -------------------------------------------------------------------


def write_manifest(manifest: dict, directory: str | Path) -> Path:
    """Write `{name}.json` atomically. Refuses to replace a completed manifest."""
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError(_listing("refusing to write an invalid manifest", problems))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest['name']}.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ManifestError(f"{path} exists and cannot be read; not overwriting ({exc})") from exc
        if isinstance(existing, dict) and existing.get("completed") is True:
            raise ManifestError(f"{path} already holds a completed manifest; refusing to overwrite")

    text = json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
    return path


def _reject_constant(token: str) -> float:
    raise ValueError(f"{token} is not allowed in a manifest")


def read_manifest(path: str | Path) -> dict:
    """Load and validate a manifest. The file must be named `{name}.json`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"{path} cannot be read: {exc}") from exc
    try:
        manifest = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ManifestError(f"{path} is not a valid JSON manifest: {exc}") from exc
    problems = validate_manifest(manifest)
    if isinstance(manifest, dict) and isinstance(manifest.get("name"), str):
        if path.name != f"{manifest['name']}.json":
            problems.append(f"file name {path.name!r} is not '{manifest['name']}.json'")
    if problems:
        raise ManifestError(_listing(f"{path} is invalid", problems))
    return manifest


def is_result(manifest: dict) -> bool:
    """True only for a completed run or branch. Trunk checkpoints and unfinished runs are not results."""
    return (
        isinstance(manifest, dict)
        and manifest.get("completed") is True
        and manifest.get("kind") in ("run", "branch")
    )
