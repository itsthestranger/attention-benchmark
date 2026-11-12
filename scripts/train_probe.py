"""Train one probe model on one synthetic task and report its held-out accuracy (plan §9.2).

The probe model keeps the backbone's depth and head count and shrinks width only as
the task forces (see `probe_resolved`). Training runs through the same `Trainer` as
the language-model runs. Held-out batches come from a different stream seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import src.attention.recurrence  # noqa: F401  (registers "recurrence")
from src import config
from src.eval.synthetic import ProbeSpec, ProbeStream, counts, make_spec
from src.model import build_model
from src.training.trainer import Trainer

RECURRENCE = "recurrence"
BACKBONE_HEADS = 8
BACKBONE_D_MODEL = 384
HELD_OUT_OFFSET = 1_000_003


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
) -> dict:
    """The resolved config of one probe run. Raises ValueError for widths the rules cannot shrink."""
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

    overrides: dict = {
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
        overrides["model.n_layers"] = n_layers

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
    overrides["model.attention"] = attention

    return config.resolve(
        base_name,
        init_seed=init_seed,
        peak_lr=peak_lr,
        steps=steps,
        branches=[],
        overrides=overrides,
    )


def _pool(total: dict, part: dict) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def run_probe(
    variant: str,
    spec: ProbeSpec,
    *,
    d_model: int,
    init_seed: int,
    steps: int,
    peak_lr: float,
    data_seed: int = 0,
    eval_batches: int = 4,
    sequences_per_step: int = 32,
    n_layers: int | None = None,
    mode: str = "opt",
    device: str = "cpu",
) -> dict:
    """Train one probe model and evaluate it on held-out batches. Returns a JSON-serialisable dict."""
    if isinstance(eval_batches, bool) or not isinstance(eval_batches, int) or eval_batches < 1:
        raise ValueError(f"eval_batches must be a positive int, got {eval_batches!r}")
    resolved = probe_resolved(
        variant,
        spec,
        d_model=d_model,
        init_seed=init_seed,
        steps=steps,
        peak_lr=peak_lr,
        sequences_per_step=sequences_per_step,
        n_layers=n_layers,
    )
    model = build_model(resolved)
    train_stream = ProbeStream(spec, sequences_per_step=sequences_per_step, seed=data_seed, max_steps=steps)
    trainer = Trainer(model, resolved, train_stream, device=device, mode=mode)
    trainer.train()
    final_loss = trainer.records[-1].loss

    held_out = ProbeStream(
        spec,
        sequences_per_step=sequences_per_step,
        seed=data_seed + HELD_OUT_OFFSET,
        max_steps=eval_batches,
    )
    pooled: dict = {}
    for i in range(eval_batches):
        x, y = held_out.batch(i)
        _pool(pooled, counts(trainer.model, x.to(device), y.to(device), mode=mode))
    if pooled["n_scored"] == 0:
        raise ValueError("held-out batches have no scored positions")

    return {
        "variant": variant,
        "task": spec.task,
        "spec": {"seq_len": spec.seq_len, "vocab_size": spec.vocab_size, "params": spec.as_dict()},
        "d_model": d_model,
        "n_layers": resolved["model"]["n_layers"],
        "init_seed": init_seed,
        "data_seed": data_seed,
        "steps": steps,
        "steps_taken": trainer.steps_taken,
        "skipped_steps": trainer.skipped_steps,
        "peak_lr": float(peak_lr),
        "sequences_per_step": sequences_per_step,
        "eval_batches": eval_batches,
        "final_train_loss": float(final_loss),
        "token_accuracy": pooled["n_correct_tokens"] / pooled["n_scored"],
        "sequence_accuracy": pooled["n_correct_sequences"] / pooled["n_sequences"],
        "n_scored": pooled["n_scored"],
        "config_hash": config.config_hash(resolved),
    }


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
    parser = argparse.ArgumentParser(description="Train one probe model and write its result as JSON.")
    parser.add_argument("--variant", required=True, help="a variant name, or 'recurrence'")
    parser.add_argument("--task", required=True, help="one of: mqar, multihop, fuzzy, selective_copy, induction")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="task parameter, repeatable")
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, help="init seed")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--peak-lr", type=float, required=True)
    parser.add_argument("--sequences-per-step", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON here (default: stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = make_spec(args.task, **_parse_params(args.param))
    result = run_probe(
        args.variant,
        spec,
        d_model=args.d_model,
        init_seed=args.seed,
        steps=args.steps,
        peak_lr=args.peak_lr,
        data_seed=args.data_seed,
        eval_batches=args.eval_batches,
        sequences_per_step=args.sequences_per_step,
        n_layers=args.n_layers,
        device=args.device,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
