"""Measure run-to-run reproducibility of one configuration (plan §5.1, §13).

Runs n_pairs + 1 identical training runs, compares each with the first, and
writes the measurement as JSON. It records numbers and judges nothing.

    python -m scripts.measure_reproducibility --variant mha --steps 200 --pairs 3 \
        --device cuda --mode opt --data-dir data/prepared --peak-lr 1e-3 \
        --micro-batch 16 --out results/reproducibility/mha_opt.json \
        [--init-scale 65536 --growth-interval 2000] [--synthetic]

`--synthetic` replaces the dataset and the model by a tiny synthetic setup
(uniform random tokens, a small model, micro-batch 4 of 8 sequences); it is for
exercising the script end to end, and its numbers mean nothing about the study.
`ref` mode runs inside deterministic_algorithms(); `opt` mode does not.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import yaml

from src import config
from src.data.loader import TrainStream, open_dataset
from src.model import build_model
from src.training.trainer import Trainer
from src.utils.reproducibility import (
    deterministic_algorithms,
    measure_reproducibility,
    reproducibility_record,
    synthetic_split,
)

SYNTHETIC_VOCAB = 128
SYNTHETIC_CONTEXT = 32
SYNTHETIC_SEQUENCES = 8
SYNTHETIC_MICRO_BATCH = 4
SYNTHETIC_DIMS = {
    "model.d_model": 64,
    "model.n_heads": 8,
    "model.head_dim": 8,
    "model.n_layers": 2,
    "model.d_ff": 96,
    "model.vocab_size": SYNTHETIC_VOCAB,
    "model.context_length": SYNTHETIC_CONTEXT,
    "recipe.batch.sequences_per_step": SYNTHETIC_SEQUENCES,
    "recipe.batch.sequence_length": SYNTHETIC_CONTEXT,
    "recipe.batch.tokens_per_step": SYNTHETIC_SEQUENCES * SYNTHETIC_CONTEXT,
    "recipe.batch.micro_batch": SYNTHETIC_MICRO_BATCH,
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="mha", help="variant file name under configs/variants/")
    p.add_argument("--steps", type=int, required=True, help="steps per run")
    p.add_argument("--pairs", type=int, required=True, help="comparisons against the first run")
    p.add_argument("--device", default="cpu", help="torch device, e.g. cuda or cpu")
    p.add_argument("--mode", default="opt", choices=["ref", "opt"])
    p.add_argument("--data-dir", default=None, help="prepared dataset directory (not needed with --synthetic)")
    p.add_argument("--peak-lr", type=float, default=1e-3)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--init-scale", type=float, default=None, help="GradScaler init_scale (fp16 runs)")
    p.add_argument("--growth-interval", type=int, default=None, help="GradScaler growth_interval (fp16 runs)")
    p.add_argument("--out", required=True, help="output JSON path; parent directories are created")
    p.add_argument("--synthetic", action="store_true", help="tiny synthetic model and data, for testing the script")
    return p


def _overrides(args) -> dict:
    if args.synthetic:
        return dict(SYNTHETIC_DIMS)
    overrides = {"recipe.batch.micro_batch": args.micro_batch}
    if args.init_scale is not None:
        overrides["recipe.grad_scaler.init_scale"] = args.init_scale
    if args.growth_interval is not None:
        overrides["recipe.grad_scaler.growth_interval"] = args.growth_interval
    return overrides


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.synthetic and args.data_dir is None:
        parser.error("--data-dir is required unless --synthetic is given")

    resolved = config.resolve(
        args.variant,
        init_seed=1,
        peak_lr=args.peak_lr,
        steps=args.steps,
        branches=[],  # branch checkpoints are not part of a reproducibility run
        overrides=_overrides(args),
    )
    batch = resolved["recipe"]["batch"]
    context = resolved["model"]["context_length"]
    if args.synthetic:
        n_tokens = args.steps * batch["sequences_per_step"] * context + 1
        split = synthetic_split(n_tokens, SYNTHETIC_VOCAB, seed=0)
    else:
        # The hashes registered in configs/data.yaml are enforced, as for a training run (plan §6).
        data_cfg = yaml.safe_load((config.CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
        split = open_dataset(
            args.data_dir,
            expected_dataset_sha256=data_cfg["dataset_sha256"],
            expected_tokenizer_sha256=data_cfg["tokenizer"]["sha256"],
        )["train"]
    stream = TrainStream(
        split,
        context_length=context,
        sequences_per_step=batch["sequences_per_step"],
        data_seed=resolved["run"]["data_seed"],
    )

    ran: dict = {}  # what the trainers actually ran with, for the record

    def build_run() -> Trainer:
        trainer = Trainer(build_model(resolved), resolved, stream, device=args.device, mode=args.mode)
        ran["mode"], ran["precision"] = trainer.mode, trainer.precision
        return trainer

    scope = deterministic_algorithms(device=args.device) if args.mode == "ref" else contextlib.nullcontext()
    with scope:
        measurement = measure_reproducibility(build_run, args.steps, args.pairs)
    record = reproducibility_record(
        measurement,
        resolved=resolved,
        mode=ran["mode"],
        precision=ran["precision"],
        device=args.device,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a run whose loss is not finite has no reproducibility to record.
    out.write_text(json.dumps(record, sort_keys=True, indent=1, allow_nan=False) + "\n", encoding="utf-8")

    print(
        f"steps={measurement['n_steps']} pairs={measurement['n_pairs']} "
        f"bit_identical={measurement['bit_identical']} "
        f"max_abs_loss_diff={measurement['max_abs_loss_diff']!r} "
        f"max_rel_loss_diff={measurement['max_rel_loss_diff']!r}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
