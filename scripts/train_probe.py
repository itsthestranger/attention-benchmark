"""Train one probe model on one synthetic task and write one result record (plan §9.2).

The probe model keeps the backbone's depth and head count and shrinks width only as
the task forces (see `probe_resolved`). Training runs through the same `Trainer` as
the language-model runs, and on CUDA the held-out evaluation runs under the same
enforced kernel and autocast setting as an LM evaluation. Held-out sequences come
from a different stream seed.

    python -m scripts.train_probe --variant V --task T --param K=V ... --d-model N --seed S
        --steps N --peak-lr X --out DIR [--data-seed N] [--sequences-per-step N]
        [--eval-sequences N] [--n-layers N] [--mode opt|ref] [--device cpu|cuda]
        [--precision fp32|fp16] [--expected-backend NAME] [--panel NAME]
        [--difficulty-param NAME] [--override PATH=VALUE ...] [--force]

The record is one JSON object with schema "probe/1" (see `run_probe`). It is written
to DIR under a file name that is known before training:
    {task}-{variant}-n{difficulty}-s{init_seed}-{h}.json
where h is the first 8 hex digits of the SHA-256 of the canonical JSON of the
configuration hash, the spec, the data seed, the eval size and the mode. If that file
exists, nothing is trained and the script exits 0; --force retrains and replaces it.
A refusal exits 2 and a training error or interrupt writes no file.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import src.attention.recurrence  # noqa: F401  (registers "recurrence")
import torch

from scripts import _run_common as rc
from src import config
from src.eval.synthetic import ProbeSpec, ProbeStream, chance_level, counts, difficulty, make_spec
from src.model import build_model
from src.training.trainer import Trainer
from src.utils import manifest as mf

SCHEMA = "probe/1"
RECURRENCE = "recurrence"
BACKBONE_HEADS = 8
BACKBONE_D_MODEL = 384
HELD_OUT_OFFSET = 1_000_003
DEFAULT_EVAL_SEQUENCES = 1024
_LABEL = re.compile(r"[A-Za-z0-9_-]+")
# The resolved-config keys probe_resolved sets itself. An extra override of one of them is refused.
PROBE_OWNED_KEYS = frozenset({
    "model.n_heads", "model.d_model", "model.head_dim", "model.d_ff", "model.vocab_size",
    "model.context_length", "model.attention", "model.n_layers",
    "recipe.batch.sequences_per_step", "recipe.batch.sequence_length",
    "recipe.batch.tokens_per_step", "recipe.batch.micro_batch",
})


def probe_resolved(
    variant: str,
    spec: ProbeSpec,
    *,
    d_model: int,
    init_seed: int,
    steps: int,
    peak_lr: float,
    sequences_per_step: int = 32,
    micro_batch: int | None = None,
    n_layers: int | None = None,
    overrides: Mapping | None = None,
) -> dict:
    """The resolved config of one probe run. Raises ValueError for widths the rules cannot shrink.

    `overrides` maps dotted paths to values and is applied after the probe's own settings,
    as scripts/train.py applies --override. A path the probe sets itself is refused.
    """
    if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0 or d_model % 16:
        raise ValueError(f"d_model must be a positive multiple of 16, got {d_model!r}")
    head_dim = d_model // BACKBONE_HEADS
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    if n_layers is not None and (isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1):
        raise ValueError(f"n_layers must be a positive int, got {n_layers!r}")
    if isinstance(sequences_per_step, bool) or not isinstance(sequences_per_step, int) or sequences_per_step < 1:
        raise ValueError(f"sequences_per_step must be a positive int, got {sequences_per_step!r}")
    micro = sequences_per_step if micro_batch is None else micro_batch
    if isinstance(micro, bool) or not isinstance(micro, int) or micro < 1 or sequences_per_step % micro:
        raise ValueError(f"micro_batch must divide sequences_per_step ({sequences_per_step}), got {micro_batch!r}")

    if variant == RECURRENCE:
        base_name = "mha"
    elif variant in config.variant_names():
        base_name = variant
    else:
        raise ValueError(f"unknown variant {variant!r}; known: {config.variant_names() + [RECURRENCE]}")
    base = config.load_variant(base_name)

    probe_settings: dict = {
        "model.n_heads": BACKBONE_HEADS,
        "model.d_model": d_model,
        "model.head_dim": head_dim,
        "model.d_ff": round(base["d_ff"] * d_model / BACKBONE_D_MODEL),
        "model.vocab_size": spec.vocab_size,
        "model.context_length": spec.seq_len,
        "recipe.batch.sequences_per_step": sequences_per_step,
        "recipe.batch.sequence_length": spec.seq_len,
        "recipe.batch.tokens_per_step": sequences_per_step * spec.seq_len,
        "recipe.batch.micro_batch": micro,
    }
    if n_layers is not None:
        probe_settings["model.n_layers"] = n_layers

    attention = dict(base["attention"])
    if variant == RECURRENCE:
        attention = {"type": RECURRENCE}
    elif attention["type"] == "mla":
        for key in ("d_c", "d_c_q"):
            scaled = base["attention"][key] * d_model / BACKBONE_D_MODEL
            if scaled != int(scaled) or int(scaled) < 1:
                raise ValueError(f"MLA {key} must scale to a positive integer, got {scaled!r} at d_model {d_model}")
        rope_dim = head_dim // 2
        if rope_dim % 2:
            raise ValueError(f"MLA rope_dim (head_dim // 2 = {rope_dim}) must be even")
        attention = {
            "type": "mla",
            "d_c": int(base["attention"]["d_c"] * d_model / BACKBONE_D_MODEL),
            "d_c_q": int(base["attention"]["d_c_q"] * d_model / BACKBONE_D_MODEL),
            "rope_dim": rope_dim,
        }
    probe_settings["model.attention"] = attention

    extra = dict(overrides or {})
    clash = sorted(set(extra) & PROBE_OWNED_KEYS)
    if clash:
        raise ValueError(f"these keys are set by the probe itself and cannot be overridden: {clash}")
    return config.resolve(
        base_name,
        init_seed=init_seed,
        peak_lr=peak_lr,
        steps=steps,
        branches=[],
        overrides={**probe_settings, **extra},
    )


def _pool(total: dict, part: dict) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _label(spec: ProbeSpec, panel: str | None) -> str:
    """The task label of the record: the panel name if given, else the generator's task."""
    if panel is None:
        return spec.task
    if not isinstance(panel, str) or not _LABEL.fullmatch(panel):
        raise ValueError(f"panel must be letters, digits, '_' or '-', got {panel!r}")
    return panel


def _difficulty(spec: ProbeSpec, difficulty_param: str | None) -> int:
    if difficulty_param is None:
        return difficulty(spec)
    params = spec.as_dict()
    if difficulty_param not in params:
        raise ValueError(f"{spec.task} has no parameter {difficulty_param!r}; known: {sorted(params)}")
    value = params[difficulty_param]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"difficulty parameter {difficulty_param!r} must be an int, got {value!r}")
    return value


def _plan(
    variant: str,
    spec: ProbeSpec,
    *,
    d_model: int,
    init_seed: int,
    steps: int,
    peak_lr: float,
    data_seed: int = 0,
    eval_sequences: int = DEFAULT_EVAL_SEQUENCES,
    sequences_per_step: int = 32,
    n_layers: int | None = None,
    mode: str = "opt",
    panel: str | None = None,
    difficulty_param: str | None = None,
    overrides: Mapping | None = None,
) -> dict:
    """Everything a run's record and file name depend on, computed before any training.

    The defaults are those of run_probe.
    """
    _check_int("eval_sequences", eval_sequences, 1)
    _check_int("data_seed", data_seed, 0)
    if mode not in rc.MODES:
        raise ValueError(f"mode must be one of {rc.MODES}, got {mode!r}")
    resolved = probe_resolved(
        variant,
        spec,
        d_model=d_model,
        init_seed=init_seed,
        steps=steps,
        peak_lr=peak_lr,
        sequences_per_step=sequences_per_step,
        n_layers=n_layers,
        overrides=overrides,
    )
    spec_record = {"seq_len": spec.seq_len, "vocab_size": spec.vocab_size, "params": spec.as_dict()}
    task = _label(spec, panel)
    level = _difficulty(spec, difficulty_param)
    identity = {
        "config_hash": config.config_hash(resolved),
        "spec": spec_record,
        "data_seed": data_seed,
        "eval_sequences": eval_sequences,
        "mode": mode,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "resolved": resolved,
        "spec": spec_record,
        "task": task,
        "difficulty": level,
        "file_name": f"{task}-{variant}-n{level}-s{init_seed}-{digest[:8]}.json",
    }


def probe_path(out_dir, variant: str, spec: ProbeSpec, **plan_args) -> Path:
    """The file a run with these arguments writes to, known before training."""
    return Path(out_dir) / _plan(variant, spec, **plan_args)["file_name"]


def _held_out_counts(trainer: Trainer, spec: ProbeSpec, *, data_seed: int, eval_sequences: int,
                     sequences_per_step: int, mode: str, backend: str | None) -> dict:
    """Pooled counts over `eval_sequences` held-out sequences, under the run's kernel and autocast."""
    n_batches = -(-eval_sequences // sequences_per_step)
    held_out = ProbeStream(
        spec,
        sequences_per_step=sequences_per_step,
        seed=data_seed + HELD_OUT_OFFSET,
        max_steps=n_batches,
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.float16) if trainer.precision == "fp16" else contextlib.nullcontext()
    )
    pooled: dict = {}
    with rc.enforced_backend(backend), autocast:
        for i in range(n_batches):
            x, y = held_out.batch(i)
            keep = min(sequences_per_step, eval_sequences - i * sequences_per_step)
            _pool(pooled, counts(trainer.model, x[:keep].to(trainer.device), y[:keep].to(trainer.device), mode=mode))
    if pooled.get("n_scored", 0) == 0:
        raise ValueError("held-out sequences have no scored positions")
    return pooled


def run_probe(
    variant: str,
    spec: ProbeSpec,
    *,
    d_model: int,
    init_seed: int,
    steps: int,
    peak_lr: float,
    data_seed: int = 0,
    eval_sequences: int = DEFAULT_EVAL_SEQUENCES,
    sequences_per_step: int = 32,
    n_layers: int | None = None,
    mode: str = "opt",
    device: str = "cpu",
    precision: str | None = None,
    expected_backend: str | None = None,
    panel: str | None = None,
    difficulty_param: str | None = None,
    overrides: Mapping | None = None,
) -> dict:
    """Train one probe model, evaluate it on held-out sequences, and return the record (schema probe/1).

    `overrides` are extra resolved-config values (dotted paths), as --override gives them to train.py.

    Keys: schema, variant, task (panel label or generator task), generator, difficulty,
    init_seed, accuracy (held-out token accuracy), sequence_accuracy, chance (no-recall
    level, or None), n_scored, n_sequences, spec {seq_len, vocab_size, params}, d_model,
    n_layers, data_seed, steps, steps_taken, skipped_steps, peak_lr, sequences_per_step,
    eval_sequences, final_train_loss, config_hash, mode, precision, backend (CUDA only,
    else None), device, environment, git_sha.
    """
    plan = _plan(
        variant,
        spec,
        d_model=d_model,
        init_seed=init_seed,
        steps=steps,
        peak_lr=peak_lr,
        data_seed=data_seed,
        eval_sequences=eval_sequences,
        sequences_per_step=sequences_per_step,
        n_layers=n_layers,
        mode=mode,
        panel=panel,
        difficulty_param=difficulty_param,
        overrides=overrides,
    )
    resolved = plan["resolved"]
    dev = rc.choose_device(device)
    backend = rc.backend_for(dev, expected_backend, required=True)
    environment = mf.collect_environment()
    git_sha = mf.git_sha()

    train_stream = ProbeStream(spec, sequences_per_step=sequences_per_step, seed=data_seed, max_steps=steps)
    trainer = Trainer(
        build_model(resolved),
        resolved,
        train_stream,
        device=dev,
        mode=mode,
        precision=precision,
        expected_backend=backend,
    )
    rc.enforce_backend(trainer.model, backend)
    trainer.train()
    final_loss = trainer.records[-1].loss

    pooled = _held_out_counts(
        trainer,
        spec,
        data_seed=data_seed,
        eval_sequences=eval_sequences,
        sequences_per_step=sequences_per_step,
        mode=mode,
        backend=backend,
    )
    return {
        "schema": SCHEMA,
        "variant": variant,
        "task": plan["task"],
        "generator": spec.task,
        "difficulty": plan["difficulty"],
        "init_seed": init_seed,
        "accuracy": pooled["n_correct_tokens"] / pooled["n_scored"],
        "sequence_accuracy": pooled["n_correct_sequences"] / pooled["n_sequences"],
        "chance": chance_level(spec),
        "n_scored": pooled["n_scored"],
        "n_sequences": pooled["n_sequences"],
        "spec": plan["spec"],
        "d_model": d_model,
        "n_layers": resolved["model"]["n_layers"],
        "data_seed": data_seed,
        "steps": steps,
        "steps_taken": trainer.steps_taken,
        "skipped_steps": trainer.skipped_steps,
        "peak_lr": float(peak_lr),
        "sequences_per_step": sequences_per_step,
        "eval_sequences": eval_sequences,
        "final_train_loss": float(final_loss),
        "config_hash": config.config_hash(resolved),
        "mode": mode,
        "precision": trainer.precision,
        "backend": backend,
        "device": dev.type,
        "environment": environment,
        "git_sha": git_sha,
    }


def run_to_dir(
    out_dir,
    variant: str,
    spec: ProbeSpec,
    *,
    force: bool = False,
    device: str = "cpu",
    precision: str | None = None,
    expected_backend: str | None = None,
    **plan_args,
) -> tuple[Path, bool]:
    """Write the record of one run to out_dir unless its file exists. Returns (path, trained).

    `plan_args` are the keyword arguments of `run_probe` other than device, precision and
    expected_backend. An existing file is left untouched unless force is true.
    """
    dev = rc.choose_device(device)
    rc.backend_for(dev, expected_backend, required=True)  # refuse before anything is read or written
    path = probe_path(out_dir, variant, spec, **plan_args)
    if path.exists() and not force:
        return path, False
    record = run_probe(
        variant,
        spec,
        device=device,
        precision=precision,
        expected_backend=expected_backend,
        **plan_args,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rc._write_json_atomic(path, record)
    return path, True


def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_params(items: list[str]) -> dict:
    params = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--param expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        params[key] = _parse_value(value)
    return params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one probe model and write its result record as JSON.")
    parser.add_argument("--variant", required=True, help="a variant name, or 'recurrence'")
    parser.add_argument("--task", required=True, help="one of: mqar, multihop, fuzzy, selective_copy, induction")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="task parameter, repeatable")
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="init seed")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--peak-lr", type=float, required=True)
    parser.add_argument("--sequences-per-step", type=int, default=32)
    parser.add_argument("--eval-sequences", type=int, default=DEFAULT_EVAL_SEQUENCES,
                        help="held-out sequences evaluated, in batches of --sequences-per-step")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--mode", choices=rc.MODES, default="opt")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default=None,
                        help="default fp32 on CPU, fp16 on CUDA")
    parser.add_argument("--expected-backend", dest="expected_backend", choices=sorted(rc.SDPA_BACKENDS),
                        help="required on CUDA: the fused kernel the trainer enforces; refused on CPU")
    parser.add_argument("--panel", default=None, metavar="NAME", help="task label of the record (default: the task)")
    parser.add_argument("--difficulty-param", dest="difficulty_param", default=None, metavar="NAME",
                        help="the spec parameter that is the record's difficulty (default: the task's difficulty)")
    parser.add_argument("--override", action="append", default=[], metavar="PATH=VALUE",
                        help="override one resolved-config value, e.g. recipe.grad_scaler.init_scale=65536.0 "
                             "(YAML value), repeatable")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR", help="directory for the record file")
    parser.add_argument("--force", action="store_true", help="retrain even if the record file exists")
    return parser


def _parse_overrides(items: list[str]) -> dict:
    """--override items, parsed by scripts/train.py's own parser (rc.build_spec), not a copy of it."""
    return rc.build_spec(argparse.Namespace(override=list(items)))["overrides"]


def _main(args) -> int:
    spec = make_spec(args.task, **_parse_params(args.param))
    path, trained = run_to_dir(
        args.out,
        args.variant,
        spec,
        force=args.force,
        device=args.device,
        precision=args.precision,
        expected_backend=args.expected_backend,
        d_model=args.d_model,
        init_seed=args.seed,
        steps=args.steps,
        peak_lr=args.peak_lr,
        data_seed=args.data_seed,
        eval_sequences=args.eval_sequences,
        sequences_per_step=args.sequences_per_step,
        n_layers=args.n_layers,
        mode=args.mode,
        panel=args.panel,
        difficulty_param=args.difficulty_param,
        overrides=_parse_overrides(args.override),
    )
    if trained:
        print(f"wrote {path}")
    else:
        print(f"{path} exists; not retrained (pass --force to retrain)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return rc.guarded(lambda: _main(args))


if __name__ == "__main__":
    sys.exit(main())
