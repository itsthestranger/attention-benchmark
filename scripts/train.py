"""Train one configuration, or resume it, in one process (plan §6, §7, §10.6, §12.3).

    python -m scripts.train RUN_SPEC --data DIR --out DIR [--mode opt|ref] [--device cpu|cuda]
        [--gpu-shared yes|no] [--expected-backend NAME] [--until STEP]
        [--variant V] [--arm A] [--init-seed N] [--data-seed N] [--peak-lr X] [--budget L]
        [--steps N] [--branches L ...] [--lr-source RUN] [--override PATH=VALUE ...]
        [--data-config PATH] [--checkpoint-every-seconds S] [--checkpoint-every-steps N] [--eval-every-steps N]
        [--eval-curve-windows N] [--eval-batch-size N] [--keep-checkpoints N]

One invocation is one run (a trunk with its final decay), started or resumed. A
second invocation after completion changes nothing and exits 0. A refusal (hash
mismatch, incomplete recipe, another configuration's checkpoint) exits 2 with the
library's message. Nothing here prompts or reads input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.model import build_model
from src.training import branch as br
from src.training import checkpoint as ck
from src.training.trainer import Trainer
from src.utils import manifest as mf
from scripts import _run_common as rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.train",
        description="Train or resume one run; see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rc.spec_arguments(p, parent_budget=True)
    p.add_argument("--data", required=True, metavar="DIR", help="prepared dataset directory")
    rc.data_config_argument(p)
    p.add_argument("--out", required=True, metavar="DIR", help="working directory: checkpoints and manifests")
    p.add_argument("--mode", choices=rc.MODES, default="opt")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--gpu-shared", dest="gpu_shared", choices=("yes", "no"),
                   help="required on CUDA: whether another job shares the GPU")
    p.add_argument("--expected-backend", dest="expected_backend", choices=sorted(rc.SDPA_BACKENDS),
                   help="required on CUDA: the fused kernel the trainer enforces inside the loop")
    p.add_argument("--until", type=int, metavar="STEP", help="stop after this step (for session limits and tests)")
    rc.ops_arguments(p)
    return p


def _train(args) -> int:
    spec = rc.build_spec(args)
    resolved = rc.resolve_spec(spec)
    config.require_complete(resolved)
    ops = rc.load_ops(args)
    device = rc.choose_device(args.device)
    backend = rc.backend_for(device, args.expected_backend, required=True)
    gpu_shared = rc.gpu_shared_for(device, args.gpu_shared)

    layout = rc.Layout(Path(args.out), args.mode)
    name = mf.run_name(resolved)
    ckpt_dir = ck.checkpoint_dir(layout.checkpoints, resolved)
    total = resolved["run"]["steps"]
    ds = rc.open_data(args.data, resolved, args.data_config)
    if rc.already_complete(layout, name, ckpt_dir, resolved, total):
        print(f"{name}: already complete ({total} steps, mode {args.mode}); nothing to do")
        return 0

    stream = rc.make_stream(ds, resolved)
    trainer = Trainer(
        build_model(resolved),
        resolved,
        stream,
        device=device,
        mode=args.mode,
        expected_backend=backend,
    )
    rc.enforce_backend(trainer.model, backend)
    info = ck.resume_or_start(trainer, layout.checkpoints)
    if info.path is None:
        print(f"{name}: no checkpoint; starting at step 0")
    else:
        print(f"{name}: resumed from {info.path.name} at step {info.step}")

    job = rc.Job(
        kind="run",
        manifest_cfg=resolved,
        name=name,
        layout=layout,
        ckpt_dir=ckpt_dir,
        dataset=ds,
        settings=rc.eval_settings(resolved, ops, backend),
        ops=ops,
        device=device,
        backend=backend,
        gpu_shared=gpu_shared,
        lr_source=spec["lr_source"],
        branch=None,
        until=args.until,
        branch_points=tuple(br.branch_steps(resolved).values()),
    )
    return rc.run_job(trainer, job)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return rc.guarded(lambda: _train(args))


if __name__ == "__main__":
    raise SystemExit(main())
