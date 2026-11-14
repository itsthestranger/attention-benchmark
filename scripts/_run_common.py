"""Plumbing shared by the entry points train.py, branch.py and evaluate.py (plan §6, §7, §10.6, §12.3).

It reads the run spec and the operational settings, resolves the configuration,
opens the dataset, lays out the output directory, keeps the sessions and the
in-run validation curve, and writes manifests. It decides nothing the plan
measures or judges: every number that enters a manifest comes from the trainer,
the evaluator or the libraries the entry points call.

Layout under OUT (one working directory per experiment)
    checkpoints/{mode}/{config hash}/   step*.pt, trunk-step*.pt (branch points),
                                        sessions.json, val_curve.json, peak_memory.json
    manifests/{mode}/{name}.json        one manifest per run and per branch, and nothing else
    ops/{mode}/{name}.json              the operational settings of each session (kept out of
                                        the manifest directory: scripts/make_tables.py reads
                                        every file there as a manifest)

The mode is part of both paths (decision 115). A run's name does not carry the
mode, so without it a `ref` and an `opt` run of one spec would write one manifest.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

from src import config
from src.attention.base import SDPA_BACKENDS, AttnBase
from src.data.loader import DataError, TrainStream, open_dataset
from src.eval.lm import ContextSpec, NonFiniteLoss, evaluate
from src.training import branch as br
from src.training import checkpoint as ck
from src.utils import manifest as mf

RUN_SPEC_KEYS = (
    "variant", "arm", "ladder", "init_seed", "data_seed", "peak_lr",
    "budget", "steps", "branches", "lr_source", "overrides", "timing_pass_id",
)
# Keys a command line may give or override, by the name the spec file uses.
SPEC_FLAGS = ("variant", "arm", "ladder", "init_seed", "data_seed", "peak_lr", "steps", "lr_source")
# The other keys a flag of the same name gives (train.py gives timing_pass_id; the others take it from the spec).
OTHER_SPEC_FLAGS = ("budget", "branches", "timing_pass_id")
OPS_KEYS = (
    "checkpoint_every_seconds", "checkpoint_every_steps", "eval_every_steps",
    "eval_curve_windows", "eval_batch_size", "keep_checkpoints",
)
MODES = ("ref", "opt")
CURVE_FILE = "val_curve.json"
PEAK_FILE = "peak_memory.json"  # the peak memory of each session of a run or branch (T-41)
SESSIONS_FILE = "sessions.json"
PEAK_KEYS = ("allocated_bytes", "reserved_bytes")
CURVE_KEYS = ("step", "tokens", "val_loss", "buckets", "late_context", "icl")
DATA_CONFIG = config.CONFIG_DIR / "data.yaml"  # read at call time; tests may replace it
OPS_CONFIG = config.CONFIG_DIR / "ops.yaml"
DEFAULT_ARM = "iso_token"

_SDPA = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}
if hasattr(SDPBackend, "CUDNN_ATTENTION"):
    _SDPA["cudnn"] = SDPBackend.CUDNN_ATTENTION


class EntryError(ValueError):
    """An entry point refuses: a bad argument, an inconsistent output directory, or a missing requirement."""


# --- Arguments -------------------------------------------------------------------------------------


def spec_arguments(parser: argparse.ArgumentParser, *, parent_budget: bool) -> None:
    """The run spec (positional or --run-spec) and the flags that give or override its keys."""
    parser.add_argument("spec", nargs="?", metavar="RUN_SPEC", help="run-spec YAML file")
    parser.add_argument("--run-spec", dest="spec_flag", metavar="PATH", help="the same as RUN_SPEC")
    parser.add_argument("--variant", help="variant name, e.g. mha")
    parser.add_argument("--arm", help=f"arm name (default {DEFAULT_ARM})")
    parser.add_argument("--ladder", help="ladder name, e.g. mha_L6")
    parser.add_argument("--init-seed", dest="init_seed", type=int)
    parser.add_argument("--data-seed", dest="data_seed", type=int)
    parser.add_argument("--peak-lr", dest="peak_lr", type=float)
    if parent_budget:
        parser.add_argument("--budget", dest="budget", metavar="LABEL", help="budget label of the run")
    parser.add_argument("--steps", type=int, help="step count, instead of a budget label")
    parser.add_argument("--branches", nargs="*", metavar="LABEL",
                        help="branch budget labels. Give the flag with no label (`--branches`) for a run with "
                             "no branch, e.g. a standalone 100M run. Not given, the run spec's branches or the "
                             "arm's default apply.")
    parser.add_argument("--lr-source", dest="lr_source", metavar="RUN_NAME",
                        help="name of the LR-sweep run that chose the peak LR")
    parser.add_argument("--override", action="append", metavar="PATH=VALUE",
                        help="override one resolved-config value, e.g. recipe.batch.micro_batch=2 (YAML value)")


def ops_arguments(parser: argparse.ArgumentParser) -> None:
    """Operational settings (configs/ops.yaml). They never enter the configuration or its hash."""
    parser.add_argument("--checkpoint-every-seconds", dest="checkpoint_every_seconds", type=float)
    parser.add_argument("--checkpoint-every-steps", dest="checkpoint_every_steps", type=int)
    parser.add_argument("--eval-every-steps", dest="eval_every_steps", type=int)
    parser.add_argument("--eval-curve-windows", dest="eval_curve_windows", type=int)
    parser.add_argument("--eval-batch-size", dest="eval_batch_size", type=int)
    parser.add_argument("--keep-checkpoints", dest="keep_checkpoints", type=int)


def _spec_path(args) -> str | None:
    positional, flag = getattr(args, "spec", None), getattr(args, "spec_flag", None)
    if positional is not None and flag is not None:
        raise EntryError("give the run spec once: positionally or with --run-spec, not both")
    return positional if positional is not None else flag


def load_run_spec(path) -> dict:
    p = Path(path)
    doc = _read_yaml(p)
    if doc is None:
        doc = {}
    if not isinstance(doc, dict):
        raise EntryError(f"{p}: the top level must be a mapping")
    unknown = sorted(str(k) for k in doc if k not in RUN_SPEC_KEYS)
    if unknown:
        raise EntryError(f"{p}: unknown keys {unknown}; allowed: {list(RUN_SPEC_KEYS)}")
    spec = {key: doc.get(key) for key in RUN_SPEC_KEYS}
    if spec["overrides"] is None:
        spec["overrides"] = {}
    if not isinstance(spec["overrides"], dict):
        raise EntryError(f"{p}: overrides must be a mapping of dotted paths to values")
    return spec


def build_spec(args) -> dict:
    """The run spec of a command: the file, with every flag given on the command line replacing its key."""
    path = _spec_path(args)
    spec = load_run_spec(path) if path is not None else {key: None for key in RUN_SPEC_KEYS}
    spec["overrides"] = dict(spec["overrides"] or {})
    for key in SPEC_FLAGS + OTHER_SPEC_FLAGS:
        value = getattr(args, key, None)
        if value is not None:
            spec[key] = value
    for item in getattr(args, "override", None) or []:
        path_text, sep, text = item.partition("=")
        if not sep or not path_text:
            raise EntryError(f"--override expects PATH=VALUE, got {item!r}")
        try:
            spec["overrides"][path_text] = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise EntryError(f"--override {item!r}: the value is not valid YAML: {exc}") from exc
    return spec


def check_timing_pass(resolved: dict, timing_pass_id) -> str | None:
    """The timing pass a run takes its step count from (plan §8.5, §10.6), checked before any work.

    Required for the wall-clock arm, refused for every other arm. The manifest validator checks the
    same rule at the end of a run; this refuses at the start, so an Arm C run cannot train and then fail.
    """
    if timing_pass_id is not None and (not isinstance(timing_pass_id, str) or not timing_pass_id):
        raise EntryError(f"--timing-pass-id must be a non-empty string, got {timing_pass_id!r}")
    wallclock = resolved["arm"] == mf.WALLCLOCK_ARM
    if wallclock and timing_pass_id is None:
        raise EntryError(
            f"--timing-pass-id is required: arm {mf.WALLCLOCK_ARM!r} takes its step count from a timing "
            "pass (plan §8.5). Give the timing_pass_id that the timing pass wrote (or the run-spec key "
            "timing_pass_id)."
        )
    if not wallclock and timing_pass_id is not None:
        raise EntryError(
            f"--timing-pass-id is refused: only arm {mf.WALLCLOCK_ARM!r} has a timing pass, and this run's "
            f"arm is {resolved['arm']!r}"
        )
    return timing_pass_id


def resolve_spec(spec: dict) -> dict:
    """The full resolved configuration of a spec. Library refusals propagate unchanged."""
    missing = [key for key in ("variant", "init_seed", "peak_lr") if spec[key] is None]
    if missing:
        raise EntryError(f"the run needs {missing}: give them in the run spec or as flags")
    return config.resolve(
        spec["variant"],
        spec["arm"] or DEFAULT_ARM,
        init_seed=spec["init_seed"],
        peak_lr=spec["peak_lr"],
        ladder=spec["ladder"],
        data_seed=spec["data_seed"],
        budget=spec["budget"],
        steps=spec["steps"],
        branches=spec["branches"],
        overrides=spec["overrides"] or None,
    )


def load_ops(args) -> dict:
    """configs/ops.yaml with every flag that was given replacing its value."""
    doc = _read_yaml(OPS_CONFIG)
    if not isinstance(doc, dict) or set(doc) != set(OPS_KEYS):
        raise EntryError(f"{OPS_CONFIG}: the keys must be exactly {sorted(OPS_KEYS)}")
    ops = dict(doc)
    for key in OPS_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            ops[key] = value
    _check_ops(ops)
    return ops


def _check_ops(ops: dict) -> None:
    def positive_int(key: str) -> None:
        value = ops[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EntryError(f"ops {key} must be a positive int, got {value!r}")

    for key in ("eval_every_steps", "eval_curve_windows", "eval_batch_size", "keep_checkpoints"):
        positive_int(key)
    if ops["checkpoint_every_steps"] is not None:
        positive_int("checkpoint_every_steps")
    seconds = ops["checkpoint_every_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds < float("inf"):
        raise EntryError(f"ops checkpoint_every_seconds must be a positive number, got {seconds!r}")


# --- Devices and backends --------------------------------------------------------------------------


def choose_device(name: str | None) -> torch.device:
    if name is None:
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise EntryError(f"--device {name!r}: {exc}") from exc
    if device.type not in ("cpu", "cuda"):
        raise EntryError(f"--device must be cpu or cuda, got {name!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EntryError("--device cuda, but CUDA is not available here")
    return device


def backend_for(device: torch.device, expected: str | None, *, required: bool) -> str | None:
    """The fused kernel the trainer enforces. Required on CUDA (plan §2.3); None on the CPU."""
    if expected is not None and expected not in SDPA_BACKENDS:
        raise EntryError(f"--expected-backend must be one of {list(SDPA_BACKENDS)}, got {expected!r}")
    if device.type == "cuda":
        if expected is None and required:
            raise EntryError(
                "--expected-backend is required on CUDA: the trainer enforces that kernel inside "
                f"the loop (plan §2.3); one of {list(SDPA_BACKENDS)}"
            )
        return expected
    if expected is not None:
        raise EntryError("--expected-backend is CUDA only: the CPU has no fused kernel to enforce")
    return None


def gpu_shared_for(device: torch.device, value: str | None) -> bool:
    """Whether another job shares the GPU, as the human states it (plan §10.6). Required on CUDA."""
    if device.type == "cuda":
        if value is None:
            raise EntryError("--gpu-shared yes|no is required on CUDA: state whether another job shares the GPU")
        return value == "yes"
    if value == "yes":
        raise EntryError("--gpu-shared yes is meaningless on the CPU")
    return False


@contextlib.contextmanager
def enforced_backend(name: str | None):
    """Restrict every fused attention call inside the block to the named kernel (plan §2.3)."""
    if name is None:
        yield
        return
    with sdpa_kernel([_SDPA[name]]):
        yield


def enforce_backend(model, backend: str | None) -> None:
    """Record the enforced kernel on every attention module, where the manifest reads it (decision 110)."""
    for module in model.modules():
        if isinstance(module, AttnBase):
            module.set_sdpa_backend(backend)


# --- Data ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    splits: dict
    dataset_sha256: str
    tokenizer_sha256: str


def open_data(data_dir, resolved: dict, data_config=None) -> Dataset:
    """Open the dataset, refusing a hash mismatch and an unpinned dataset (plan §6).

    The expected hashes come from the data config (configs/data.yaml by default). A config
    whose hashes are null refuses before anything else: nothing else can say which data a
    run is about.
    """
    path = Path(data_config) if data_config is not None else DATA_CONFIG
    data_cfg = _read_yaml(path)
    tokenizer_cfg = data_cfg.get("tokenizer") or {}
    dataset_pin = data_cfg.get("dataset_sha256")
    tokenizer_pin = tokenizer_cfg.get("sha256")
    if dataset_pin is None or tokenizer_pin is None:
        raise EntryError(
            f"{path} does not pin the data: dataset_sha256 and tokenizer.sha256 must be set. "
            "Run `python -m scripts.prepare_data` once and paste the two hashes it prints into "
            "configs/data.yaml. Nothing skips this check."
        )
    splits = open_dataset(
        data_dir,
        expected_dataset_sha256=dataset_pin,
        expected_tokenizer_sha256=tokenizer_pin,
    )
    # open_dataset has just verified meta.json against the files, so its values are the verified ones.
    meta = _read_json(Path(data_dir) / "meta.json")
    context = resolved["model"]["context_length"]
    if meta["context_length"] != context:
        raise EntryError(f"the dataset is packed at context {meta['context_length']}, the model's is {context}")
    if resolved["recipe"]["batch"]["sequence_length"] != context:
        raise EntryError("recipe.batch.sequence_length must equal model.context_length")
    if meta["vocab_size"] != resolved["model"]["vocab_size"]:
        raise EntryError(
            f"the dataset's vocabulary is {meta['vocab_size']}, the model's is {resolved['model']['vocab_size']}"
        )
    return Dataset(
        splits=splits,
        dataset_sha256=meta["dataset_sha256"],
        tokenizer_sha256=meta["tokenizer_sha256"],
    )


def data_config_argument(parser: argparse.ArgumentParser) -> None:
    """--data-config: the file that pins the dataset's hashes (default configs/data.yaml)."""
    parser.add_argument("--data-config", dest="data_config", metavar="PATH", default=None,
                        help="data config that pins the dataset hashes (default configs/data.yaml)")


def make_stream(ds: Dataset, resolved: dict) -> TrainStream:
    return TrainStream(
        ds.splits["train"],
        context_length=resolved["model"]["context_length"],
        sequences_per_step=resolved["recipe"]["batch"]["sequences_per_step"],
        data_seed=resolved["run"]["data_seed"],
    )


# --- Layout, sessions, completion ------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    out: Path
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise EntryError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def checkpoints(self) -> Path:
        return self.out / "checkpoints" / self.mode

    @property
    def manifests(self) -> Path:
        return self.out / "manifests" / self.mode

    @property
    def ops(self) -> Path:
        return self.out / "ops" / self.mode


def register_session(directory: Path) -> list[str]:
    """Append this process's session id to the configuration's session list and return the list.

    Called after the restore has succeeded, so a refused start leaves no session behind.
    Every process after the first counts as a resume (`resume_count` = len - 1).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SESSIONS_FILE
    ids = _read_json(path) if path.exists() else []
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise EntryError(f"{path} is not a session list")
    ids.append(uuid.uuid4().hex)
    _write_json_atomic(path, ids)
    return ids


def record_ops(ops_dir: Path, name: str, session_id: str, ops: dict, device: torch.device, until) -> None:
    """Keep the operational values each session actually used (plan §8.7), beside the manifests."""
    path = ops_dir / f"{name}.json"
    entries = _read_json(path) if path.exists() else []
    if not isinstance(entries, list):
        raise EntryError(f"{path} is not an ops record")
    entries.append({"session_id": session_id, "device": str(device), "until": until, "ops": ops})
    _write_json_atomic(path, entries)


def already_complete(layout: Layout, name: str, ckpt_dir: Path, manifest_cfg: dict, total: int) -> bool:
    """True when a completed manifest of this configuration exists: nothing is left to do.

    The final checkpoint may have been pruned (plan §12.3); that is reported, not an error.
    A manifest of another configuration under the same name is refused, not overwritten.
    """
    path = layout.manifests / f"{name}.json"
    if not path.exists():
        return False
    existing = mf.read_manifest(path)
    if existing["config_hash"] != config.config_hash(manifest_cfg):
        raise EntryError(f"{path} belongs to another configuration; refusing to write this run under its name")
    if existing["completed"] is not True:
        return False
    if not (ckpt_dir / f"step{total:08d}.pt").exists():
        print(f"{path.name}: complete; its final checkpoint has been pruned (plan §12.3)")
    return True


def already_diverged(layout: Layout, name: str, manifest_cfg: dict) -> dict | None:
    """The `final_metrics` of a recorded divergence of this configuration, or None.

    A diverged run is never retried by a later invocation: its manifest stays, and a person
    who wants another attempt deletes it. A manifest of another configuration is refused.
    """
    path = layout.manifests / f"{name}.json"
    if not path.exists():
        return None
    existing = mf.read_manifest(path)
    if existing["config_hash"] != config.config_hash(manifest_cfg):
        raise EntryError(f"{path} belongs to another configuration; refusing to write this run under its name")
    metrics = existing["final_metrics"]
    if existing["completed"] is False and isinstance(metrics, dict) and metrics.get("diverged") is True:
        return metrics
    return None


def record_divergence(trainer, job: "Job", sessions: list[str], exc: NonFiniteLoss) -> int:
    """Write the run's manifest as unfinished and diverged, and report it. Exit status DIVERGED_EXIT.

    The checkpoints stay where they are. The manifest's final_metrics say that the run diverged,
    why, and at which step (plan §10.6); mf.is_result is False for it.
    """
    final = {"diverged": True, "reason": str(exc), "step": int(trainer.step)}
    path = write_manifest(job, trainer, sessions, completed=False, final=final)
    print(f"{path}: diverged at step {trainer.step} of {trainer.total_steps}: {exc}")
    return DIVERGED_EXIT


# --- The validation curve (plan §10.2) -------------------------------------------------------------


class ValCurve:
    """The in-run validation points of one configuration, persisted beside its checkpoints.

    An entry is written before the checkpoint of the same step, so a checkpoint never
    exists without its point. On resume, entries beyond the restored step are dropped and
    recomputed, so an interrupted run and an uninterrupted one produce the same curve.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[dict] = []

    def load(self) -> None:
        data = _read_json(self.path) if self.path.exists() else []
        if not isinstance(data, list) or not all(_is_curve_entry(e) for e in data):
            raise EntryError(f"{self.path} is not a validation curve")
        self.entries = data

    def truncate_after(self, step: int) -> int:
        kept = [e for e in self.entries if e["step"] <= step]
        dropped = len(self.entries) - len(kept)
        self.entries = kept
        return dropped

    def append(self, *, step: int, tokens: int, val_loss: float, buckets: list, late_context: dict,
               icl: dict) -> None:
        """One point: the loss, and the context-bucketed blocks of the same evaluation (plan §10.3)."""
        self.entries.append({
            "step": int(step), "tokens": int(tokens), "val_loss": float(val_loss),
            "buckets": buckets, "late_context": late_context, "icl": icl,
        })

    def save(self) -> None:
        _write_json_atomic(self.path, self.entries)


def _is_curve_entry(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and set(entry) == set(CURVE_KEYS)
        and isinstance(entry["step"], int)
        and isinstance(entry["tokens"], int)
        and isinstance(entry["val_loss"], float)
        and isinstance(entry["buckets"], list)
        and isinstance(entry["late_context"], dict)
        and isinstance(entry["icl"], dict)
    )


# --- Peak memory (T-41) ----------------------------------------------------------------------------


def peak_memory(device) -> dict:
    """The peak memory this process has reached on `device`: {"allocated_bytes", "reserved_bytes"}.

    On CUDA, the torch statistics since the last reset_peak_memory. On the CPU both are None. Every
    read of the peak goes through this function, so a test can replace it.
    """
    if torch.device(device).type != "cuda":
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_peak_memory(device) -> None:
    """Start a session's peak statistics from zero. CUDA only; a no-op on the CPU."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


class PeakMemory:
    """The peak memory of each session of one run or branch, kept beside its curve file.

    A session records its reading when it writes a checkpoint, when it stops at --until and
    when it finishes, so a session killed after a checkpoint still counts. The file maps each
    session id to its peak; the manifest reports the maximum over the sessions in it.
    """

    def __init__(self, path: Path, session_id: str, device: torch.device):
        self.path = Path(path)
        self.session_id = session_id
        self.device = device

    def record(self) -> None:
        data = self._read()
        reading = peak_memory(self.device)
        previous = data.get(self.session_id, {})
        data[self.session_id] = {
            key: _larger(previous.get(key), reading[key]) for key in PEAK_KEYS
        }
        _write_json_atomic(self.path, data)

    def maximum(self) -> dict:
        data = self._read()
        return {key: _larger_of(entry[key] for entry in data.values()) for key in PEAK_KEYS}

    def _read(self) -> dict:
        data = _read_json(self.path) if self.path.exists() else {}
        if not isinstance(data, dict) or not all(_is_peak_entry(v) for v in data.values()):
            raise EntryError(f"{self.path} is not a peak-memory record")
        return data


def _is_peak_entry(entry: object) -> bool:
    return (
        isinstance(entry, dict)
        and set(entry) == set(PEAK_KEYS)
        and all(v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0)
                for v in entry.values())
    )


def _larger(a, b):
    """The larger of two readings, where None means no reading."""
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _larger_of(values) -> int | None:
    result = None
    for value in values:
        result = _larger(result, value)
    return result


# --- Evaluation ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSettings:
    spec: ContextSpec
    context_length: int
    batch_size: int
    backend: str | None


def eval_settings(resolved: dict, ops: dict, backend: str | None) -> EvalSettings:
    return EvalSettings(
        spec=ContextSpec.from_prereg(config.load_prereg()),
        context_length=resolved["model"]["context_length"],
        batch_size=ops["eval_batch_size"],
        backend=backend,
    )


def evaluate_split(model, split, settings: EvalSettings, *, device, mode, precision, max_windows=None) -> dict:
    """plan §10: the evaluator of src/eval/lm.py, with the fused kernel enforced as in training."""
    with enforced_backend(settings.backend):
        return evaluate(
            model,
            split,
            context_length=settings.context_length,
            batch_size=settings.batch_size,
            spec=settings.spec,
            device=device,
            mode=mode,
            precision=precision,
            max_windows=max_windows,
        )


class RunHook:
    """The trainer's on_step hook: the in-run validation point, then the checkpoint cadence."""

    def __init__(self, *, checkpointer, curve: ValCurve, val_split, settings: EvalSettings,
                 every: int, windows: int, peaks: PeakMemory):
        self.checkpointer = checkpointer
        self.curve = curve
        self.val_split = val_split
        self.settings = settings
        self.every = every
        self.windows = windows
        self.peaks = peaks

    def __call__(self, trainer, record) -> None:
        if trainer.step % self.every == 0:
            result = evaluate_split(
                trainer.model,
                self.val_split,
                self.settings,
                device=trainer.device,
                mode=trainer.mode,
                precision=trainer.precision,
                max_windows=self.windows,
            )
            tokens = trainer.step * trainer.resolved["recipe"]["batch"]["tokens_per_step"]
            self.curve.append(
                step=trainer.step, tokens=tokens, val_loss=result["loss"],
                buckets=result["buckets"], late_context=result["late_context"], icl=result["icl"],
            )
            self.curve.save()
        saved = len(self.checkpointer.saved)
        self.checkpointer(trainer, record)
        if len(self.checkpointer.saved) != saved:
            self.peaks.record()  # a session killed after this checkpoint still counts (T-41)


# --- Jobs and manifests ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    kind: str                   # "run" or "branch"
    manifest_cfg: dict          # the configuration the manifest carries (the parent's for a branch)
    name: str
    layout: Layout
    ckpt_dir: Path              # the directory of the configuration that trains
    dataset: Dataset
    settings: EvalSettings
    ops: dict
    device: torch.device
    backend: str | None
    gpu_shared: bool
    lr_source: str | None
    branch: dict | None
    until: int | None
    branch_points: tuple[int, ...] = ()
    timing_pass_id: str | None = None  # required for the wall-clock arm, null otherwise (check_timing_pass)


def run_job(trainer, job: Job) -> int:
    """Train to `job.until` (or the end), then write the manifest: completed at the end, unfinished before it."""
    total = trainer.total_steps
    if job.until is not None and not trainer.step <= job.until <= total:
        raise EntryError(f"--until must be in [{trainer.step}, {total}] (the restored step is {trainer.step})")
    sessions = register_session(job.ckpt_dir)
    curve = ValCurve(job.ckpt_dir / CURVE_FILE)
    curve.load()
    if curve.truncate_after(trainer.step):
        curve.save()
    record_ops(job.layout.ops, job.name, sessions[-1], job.ops, job.device, job.until)
    peaks = PeakMemory(job.ckpt_dir / PEAK_FILE, sessions[-1], job.device)
    checkpointer = ck.Checkpointer(
        job.layout.checkpoints,
        every_steps=job.ops["checkpoint_every_steps"],
        every_seconds=job.ops["checkpoint_every_seconds"],
        branch_steps=job.branch_points,
        keep=job.ops["keep_checkpoints"],
        resume_count=len(sessions) - 1,
    )
    trainer.on_step = RunHook(
        checkpointer=checkpointer,
        curve=curve,
        val_split=job.dataset.splits["val"],
        settings=job.settings,
        every=job.ops["eval_every_steps"],
        windows=job.ops["eval_curve_windows"],
        peaks=peaks,
    )
    try:
        trainer.train(until=job.until)
        if trainer.step < total:
            # A stop before the end leaves a regular checkpoint at the stop step, so that the next
            # invocation resumes from here and not from an older cadence checkpoint, which would run
            # these steps again (review S4, G1; plan §7.2, §12.3).
            ck.save_checkpoint(
                trainer, job.ckpt_dir, keep=job.ops["keep_checkpoints"], resume_count=len(sessions) - 1
            )
        finished = trainer.step == total
        final = final_metrics(trainer, job, curve.entries) if finished else None
    except NonFiniteLoss as exc:
        peaks.record()
        return record_divergence(trainer, job, sessions, exc)
    peaks.record()  # after the final evaluation, which is part of this session (T-41)
    if final is not None:
        final["peak_memory"] = peaks.maximum()
    path = write_manifest(job, trainer, sessions, completed=finished, final=final)
    state = "completed" if finished else f"unfinished at step {trainer.step} of {total}"
    print(f"{path}: {state}; resumes so far: {len(sessions) - 1}")
    return 0


def final_metrics(trainer, job: Job, curve_entries: list[dict]) -> dict:
    """The full validation and test splits, at the final step (plan §10.2; key names are a contract)."""
    kwargs = dict(device=trainer.device, mode=trainer.mode, precision=trainer.precision)
    val = evaluate_split(trainer.model, job.dataset.splits["val"], job.settings, **kwargs)
    test = evaluate_split(trainer.model, job.dataset.splits["test"], job.settings, **kwargs)
    # The loss of the last step whose recorded loss is finite. A non-finite loss on the last step is
    # skipped like any other non-finite step (the weights are intact), and must not make the manifest
    # invalid: JSON cannot hold NaN (review S4, M2).
    finite_losses = [r.loss for r in trainer.records if math.isfinite(r.loss)]
    return {
        "val_loss": val["loss"],
        "test_loss": test["loss"],
        "train_loss": finite_losses[-1] if finite_losses else None,
        "val": val,
        "test": test,
        "val_curve": [dict(e) for e in curve_entries],
    }


def require_git_checkout() -> None:
    """Read the commit hash before any work (review M7): a run or branch that cannot record it is refused here,
    with no step taken and no file written under OUT, not after its last step. write_manifest reads it again
    at the end, so the manifest holds the hash as it always did."""
    mf.git_sha()


def write_manifest(job: Job, trainer, sessions: list[str], *, completed: bool, final: dict | None) -> Path:
    s = trainer.summary()
    steps = {
        "total": trainer.total_steps,
        "attempted": s["steps_attempted"],
        "taken": s["steps_taken"],
        "skipped": s["skipped_steps"],
        "skipped_in_first_100": s["skipped_in_first_100"],
    }
    manifest = mf.build_manifest(
        kind=job.kind,
        resolved=job.manifest_cfg,
        completed=completed,
        git_sha=mf.git_sha(),
        data={
            "dataset_sha256": job.dataset.dataset_sha256,
            "tokenizer_sha256": job.dataset.tokenizer_sha256,
            "order_sha256": trainer.stream.order_sha256,
        },
        environment=mf.collect_environment(),
        backend=job.backend,
        mode=trainer.mode,
        precision=trainer.precision,
        steps=steps,
        final_scale=s["final_scale"],
        gpu_shared=job.gpu_shared,
        session_ids=sessions,
        step_seconds=[r.seconds for r in trainer.records],
        final_metrics=final,
        branch=job.branch,
        lr_source=job.lr_source,
        timing_pass_id=job.timing_pass_id,
    )
    return mf.write_manifest(manifest, job.layout.manifests)


# --- Small helpers ---------------------------------------------------------------------------------


REFUSALS = (ValueError, DataError, ck.CheckpointError, br.BranchError)
# The exit status of a run that diverged: its manifest is recorded, and it is not a result.
DIVERGED_EXIT = 3


def guarded(body) -> int:
    """Run an entry point's body. A library refusal prints its message and gives exit status 2."""
    try:
        return body()
    except REFUSALS as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


def _read_yaml(path: Path):
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except OSError as exc:
        raise EntryError(f"{path} cannot be read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EntryError(f"{path}: invalid YAML: {exc}") from exc


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EntryError(f"{path} cannot be read as JSON: {exc}") from exc


def _write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, sort_keys=True, indent=1, allow_nan=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise
