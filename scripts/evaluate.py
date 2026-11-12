"""Evaluate a checkpoint on the validation or the test split (plan §6, §10).

    python -m scripts.evaluate RUN_SPEC --out DIR --data DIR --split val|test --json PATH
        [--branch LABEL] [--mode opt|ref] [--device cpu|cuda] [--expected-backend NAME]
        [--eval-batch-size N]
    python -m scripts.evaluate --checkpoint PATH --data DIR --split val|test --json PATH
        [--mode opt|ref] [--device cpu|cuda] [--expected-backend NAME]

With a run spec, the latest valid checkpoint of that run (or of its branch LABEL)
in OUT/checkpoints/{mode}/ is used. With --checkpoint, that file is used. The
result dict of src/eval/lm.evaluate is written as JSON. The script writes no
manifest and changes no checkpoint. A trunk (branch-point) checkpoint is refused:
it is still at peak learning rate and is not a result (plan §7.2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model import build_model
from src.training import branch as br
from src.training import checkpoint as ck
from scripts import _run_common as rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts.evaluate",
        description="Evaluate a checkpoint on one split; see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rc.spec_arguments(p, parent_budget=True)
    p.add_argument("--branch", dest="branch_budget", metavar="LABEL",
                   help="evaluate the latest checkpoint of this branch of the run")
    p.add_argument("--checkpoint", metavar="PATH", help="evaluate this checkpoint file instead")
    p.add_argument("--out", metavar="DIR", help="the run's working directory (with a run spec)")
    p.add_argument("--data", required=True, metavar="DIR", help="prepared dataset directory")
    p.add_argument("--split", required=True, choices=("val", "test"))
    p.add_argument("--json", required=True, metavar="PATH", help="where to write the result dict")
    p.add_argument("--mode", choices=rc.MODES, default=None,
                   help="evaluation mode; with a run spec also selects the run's checkpoints (default opt)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--expected-backend", dest="expected_backend", choices=sorted(rc.SDPA_BACKENDS),
                   help="optional on CUDA: enforce this fused kernel during evaluation")
    p.add_argument("--eval-batch-size", dest="eval_batch_size", type=int)
    return p


def _find_checkpoint(args):
    """(path, state) of the checkpoint to evaluate."""
    if args.checkpoint is not None:
        if args.spec is not None or args.spec_flag is not None or args.branch_budget is not None:
            raise rc.EntryError("give --checkpoint or a run spec, not both")
        path = Path(args.checkpoint)
        return path, ck.load_checkpoint(path)
    if args.spec is None and args.spec_flag is None:
        raise rc.EntryError("give a run spec, or --checkpoint PATH")
    if args.out is None:
        raise rc.EntryError("--out is required to find a run's checkpoint from its run spec")
    resolved = rc.resolve_spec(rc.build_spec(args))
    cfg = resolved if args.branch_budget is None else br.branch_resolved(resolved, args.branch_budget)
    layout = rc.Layout(Path(args.out), args.mode or "opt")
    found = ck.latest_valid_checkpoint(ck.checkpoint_dir(layout.checkpoints, cfg))
    if found is None:
        raise rc.EntryError(f"no valid checkpoint for this configuration under {layout.checkpoints}")
    return found


def _evaluate(args) -> int:
    path, state = _find_checkpoint(args)
    if state["branch_point"]:
        raise rc.EntryError(f"{path} is a trunk (branch-point) checkpoint: not a result (plan §7.2)")
    resolved = state["resolved"]
    mode = args.mode or state["mode"]

    device = rc.choose_device(args.device)
    backend = rc.backend_for(device, args.expected_backend, required=False)
    ds = rc.open_data(args.data, resolved)
    model = build_model(resolved)
    model.load_state_dict(state["model"])
    model.to(device)
    settings = rc.eval_settings(resolved, rc.load_ops(args), backend)
    result = rc.evaluate_split(
        model,
        ds.splits[args.split],
        settings,
        device=device,
        mode=mode,
        precision=state["precision"],
    )
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{path.name} (step {state['step']}, {mode}, {state['precision']}) {args.split} loss {result['loss']:.6f} "
          f"over {result['n_tokens']} tokens -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return rc.guarded(lambda: _evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
