"""Train one WSD decay branch of an existing trunk, or resume it (plan §7.2, §10.6, §12.3).

    python -m scripts.branch RUN_SPEC --budget LABEL --data DIR --out DIR [--mode opt|ref]
        [--device cpu|cuda] [--gpu-shared yes|no] [--expected-backend NAME] [--until STEP]
        [--variant V] [--arm A] [--init-seed N] [--data-seed N] [--peak-lr X] [--steps N]
        [--branches L ...] [--lr-source RUN] [--override PATH=VALUE ...]
        [--data-config PATH] [--checkpoint-every-seconds S] [--checkpoint-every-steps N] [--eval-every-steps N]
        [--eval-curve-windows N] [--eval-batch-size N] [--keep-checkpoints N]

RUN_SPEC identifies the parent trunk. --budget names the branch (for example 100M).
The branch starts from the trunk's branch-point checkpoint of that budget, which
must exist; once the branch has its own checkpoints it resumes from them. The
manifest is kind `branch`, carries the parent's configuration and the `branch`
block, and is named `...-b{budget}`. A branch whose loss is not finite is recorded as
diverged and exits 3, as scripts.train does; a later invocation exits 3 without training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.training import branch as br
from src.training import checkpoint as ck
from src.utils import manifest as mf
from scripts import _run_common as rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.branch",
        description="Train or resume one decay branch; see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rc.spec_arguments(p, parent_budget=False)
    p.add_argument("--budget", dest="branch_budget", required=True, metavar="LABEL",
                   help="the branch's budget label, e.g. 100M; one of the trunk's branches")
    p.add_argument("--data", required=True, metavar="DIR", help="prepared dataset directory")
    rc.data_config_argument(p)
    p.add_argument("--out", required=True, metavar="DIR", help="the trunk's working directory")
    p.add_argument("--mode", choices=rc.MODES, default="opt")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--gpu-shared", dest="gpu_shared", choices=("yes", "no"),
                   help="required on CUDA: whether another job shares the GPU")
    p.add_argument("--expected-backend", dest="expected_backend", choices=sorted(rc.SDPA_BACKENDS),
                   help="required on CUDA: the fused kernel the trainer enforces inside the loop")
    p.add_argument("--until", type=int, metavar="STEP", help="stop after this step (for session limits and tests)")
    rc.ops_arguments(p)
    return p


def _branch(args) -> int:
    spec = rc.build_spec(args)
    parent = rc.resolve_spec(spec)
    config.require_complete(parent)
    budget = args.branch_budget
    branch_cfg = br.branch_resolved(parent, budget)  # BranchError unless budget is one of the trunk's branches
    rc.require_git_checkout()  # before any work: no step is taken without a commit hash (review M7)
    ops = rc.load_ops(args)
    device = rc.choose_device(args.device)
    rc.reset_peak_memory(device)  # this session's peak starts here, before the model exists (T-41)
    backend = rc.backend_for(device, args.expected_backend, required=True)
    gpu_shared = rc.gpu_shared_for(device, args.gpu_shared)

    layout = rc.Layout(Path(args.out), args.mode)
    name = mf.run_name(parent, budget)
    ckpt_dir = ck.checkpoint_dir(layout.checkpoints, branch_cfg)
    total = branch_cfg["run"]["steps"]
    ds = rc.open_data(args.data, parent, args.data_config)
    if rc.already_complete(layout, name, ckpt_dir, parent, total):
        print(f"{name}: already complete ({total} steps, mode {args.mode}); nothing to do")
        return 0
    diverged = rc.already_diverged(layout, name, parent)
    if diverged is not None:
        print(f"{name}: recorded as diverged at step {diverged['step']} ({diverged['reason']}); "
              f"nothing to do (delete its manifest to retry)")
        return rc.DIVERGED_EXIT

    stream = rc.make_stream(ds, parent)
    trainer, info = br.start_or_resume_branch(
        parent,
        budget,
        stream,
        layout.checkpoints,
        device=device,
        mode=args.mode,
        expected_backend=backend,
    )
    rc.enforce_backend(trainer.model, backend)
    print(f"{name}: from {info.path} at step {info.step}")

    trunk_point = ck.checkpoint_dir(layout.checkpoints, parent) / f"trunk-step{br.branch_step(parent, budget):08d}.pt"
    block = br.branch_manifest_block(parent, budget, trunk_point.relative_to(layout.out).as_posix())
    job = rc.Job(
        kind="branch",
        manifest_cfg=parent,
        name=name,
        layout=layout,
        ckpt_dir=ckpt_dir,
        dataset=ds,
        settings=rc.eval_settings(parent, ops, backend),
        ops=ops,
        device=device,
        backend=backend,
        gpu_shared=gpu_shared,
        lr_source=spec["lr_source"],
        branch=block,
        until=args.until,
        branch_points=(),
    )
    return rc.run_job(trainer, job)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return rc.guarded(lambda: _branch(args))


if __name__ == "__main__":
    raise SystemExit(main())
