"""Solves d_ff so that every attention variant has the same parameter budget (plan §3.2).

Total parameters = embedding (counted once, tied with the head)
                   + n_layers x (attention + 3 x d_model x d_ff + 2 x d_model)
                   + d_model (final norm).

Attention parameters are the matrices A plus the 1-D RMSNorm gains g. Only MLA
has gains (its latent norms), and they cannot be absorbed into d_ff, so MLA is
allowed an offset of n_layers x g, which must stay under the bound in
configs/base.yaml. Everything else is exact integer arithmetic: the solver
raises rather than rounds.

    python -m scripts.solve_config --variant mha
    python -m scripts.solve_config --all
"""

from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass

import yaml

from src import config
from src.attention.base import AttnConfig, build_attention
from src.model import build_model

INIT_SEED = 1      # parameter counts do not depend on these two
PEAK_LR = 1e-3
ISO_ARM = "iso_token"


class SolveError(ValueError):
    """The target cannot be hit exactly, or a config file disagrees with the solver."""


@dataclass(frozen=True)
class Solution:
    variant: str
    ladder: str | None
    arm: str
    n_layers: int
    d_ff: int
    attention_params_per_layer: int      # matrices + gains
    attention_gain_params_per_layer: int # g
    total_params: int
    non_embedding_params: int
    target_total: int
    offset: int                          # total_params - target_total
    relative_offset: float               # offset / target_total


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _params_config() -> dict:
    return config.load_base()["params"]


def _target(target_total: int | None) -> int:
    if target_total is None:
        return _params_config()["target_total"]
    if not _is_int(target_total):
        raise SolveError(f"target_total must be an int, got {target_total!r}")
    return target_total


@functools.lru_cache(maxsize=None)
def _attention_split(variant: str) -> tuple[int, int, int, int, int]:
    """(n_layers, d_model, vocab_size, matrix params per layer, gain params per layer)."""
    resolved = config.resolve(variant, ISO_ARM, init_seed=INIT_SEED, peak_lr=PEAK_LR)
    model = resolved["model"]
    attn = build_attention(AttnConfig.from_resolved(resolved))
    matrices = gains = 0
    for p in attn.parameters():
        if p.dim() == 2:
            matrices += p.numel()
        elif p.dim() == 1:
            gains += p.numel()
        else:
            raise SolveError(f"{variant}: attention parameter of shape {tuple(p.shape)} is neither a matrix nor a gain")
    return model["n_layers"], model["d_model"], model["vocab_size"], matrices, gains


def solve_d_ff(variant: str, target_total: int | None = None) -> int:
    """d_ff that hits the target with the attention matrices counted and the gains left out.

    Exact: raises SolveError if no integer d_ff gives the target.
    """
    target = _target(target_total)
    n_layers, d, vocab, matrices, _gains = _attention_split(variant)
    numerator = target - vocab * d - d - n_layers * (matrices + 2 * d)
    denominator = 3 * d * n_layers
    if numerator <= 0:
        raise SolveError(
            f"{variant}: target {target} is too small: even d_ff = 0 leaves "
            f"{numerator} parameters over the non-MLP budget"
        )
    if numerator % denominator:
        raise SolveError(
            f"{variant}: target {target} cannot be hit exactly: the MLP budget {numerator} is not a "
            f"multiple of {denominator} (one d_ff step); the solver does not round"
        )
    return numerator // denominator


def _check_offset(*, total: int, target_total: int, n_layers: int, gains: int, max_relative: float) -> int:
    """The offset of a solved model, or SolveError if it is not the allowed gain offset."""
    offset = total - target_total
    if offset != n_layers * gains:
        raise SolveError(
            f"parameter count {total} is {offset} off the target {target_total}; "
            f"the gains account for {n_layers * gains}"
        )
    if gains == 0:
        return offset
    relative = abs(offset) / target_total
    if not relative < max_relative:
        raise SolveError(
            f"gain offset {offset} is {100 * relative:.4f}% of the target, not below "
            f"{100 * max_relative:.4f}%"
        )
    return offset


@dataclass(frozen=True)
class _Measured:
    n_layers: int
    d_ff: int
    attention_params_per_layer: int
    gains_per_layer: int
    total: int
    non_embedding: int


@functools.lru_cache(maxsize=None)
def _measure(variant: str, arm: str, ladder: str | None, d_ff: int | None) -> _Measured:
    """Build the model and count it. d_ff None keeps the configured value."""
    overrides = None if d_ff is None else {"model.d_ff": d_ff}
    resolved = config.resolve(
        variant, arm, init_seed=INIT_SEED, peak_lr=PEAK_LR, ladder=ladder, overrides=overrides
    )
    model = build_model(resolved)
    attn = model.layers[0].attn
    return _Measured(
        n_layers=resolved["model"]["n_layers"],
        d_ff=resolved["model"]["d_ff"],
        attention_params_per_layer=attn.n_params,
        gains_per_layer=sum(p.numel() for p in attn.parameters() if p.dim() == 1),
        total=sum(p.numel() for p in model.parameters()),
        non_embedding=model.n_non_embedding_params,
    )


def solve(variant: str, target_total: int | None = None) -> Solution:
    """The iso-parameter model of a variant. Raises SolveError rather than miss the target."""
    target = _target(target_total)
    d_ff = solve_d_ff(variant, target)
    if target_total is None:
        configured = config.load_variant(variant)["d_ff"]
        if configured != d_ff:
            raise SolveError(
                f"{variant}: the solver gives d_ff {d_ff} but configs/variants/{variant}.yaml "
                f"says {configured}; the config files and the solver must agree"
            )
    measured = _measure(variant, ISO_ARM, None, d_ff)
    offset = _check_offset(
        total=measured.total,
        target_total=target,
        n_layers=measured.n_layers,
        gains=measured.gains_per_layer,
        max_relative=_params_config()["mla_max_relative_offset"],
    )
    return Solution(
        variant=variant,
        ladder=None,
        arm=ISO_ARM,
        n_layers=measured.n_layers,
        d_ff=d_ff,
        attention_params_per_layer=measured.attention_params_per_layer,
        attention_gain_params_per_layer=measured.gains_per_layer,
        total_params=measured.total,
        non_embedding_params=measured.non_embedding,
        target_total=target,
        offset=offset,
        relative_offset=offset / target,
    )


def describe(variant: str, *, arm: str = "iso_token", ladder: str | None = None) -> Solution:
    """Counts of the model as configured for this arm or ladder. Nothing is solved."""
    target = _target(None)
    measured = _measure(variant, arm, ladder, None)
    offset = measured.total - target
    return Solution(
        variant=variant,
        ladder=ladder,
        arm=arm,
        n_layers=measured.n_layers,
        d_ff=measured.d_ff,
        attention_params_per_layer=measured.attention_params_per_layer,
        attention_gain_params_per_layer=measured.gains_per_layer,
        total_params=measured.total,
        non_embedding_params=measured.non_embedding,
        target_total=target,
        offset=offset,
        relative_offset=offset / target,
    )


def resolved_yaml(solution: Solution) -> str:
    """The resolved config of the solved model, with a comment header stating its counts."""
    resolved = config.resolve(
        solution.variant, solution.arm, init_seed=INIT_SEED, peak_lr=PEAK_LR, ladder=solution.ladder
    )
    if resolved["model"]["d_ff"] != solution.d_ff:
        resolved = config.resolve(
            solution.variant,
            solution.arm,
            init_seed=INIT_SEED,
            peak_lr=PEAK_LR,
            ladder=solution.ladder,
            overrides={"model.d_ff": solution.d_ff},
        )
    header = [
        f"# variant: {solution.variant}",
        f"# ladder: {solution.ladder}",
        f"# arm: {solution.arm}",
        f"# d_ff: {solution.d_ff}",
        f"# total_params: {solution.total_params}",
        f"# target_total: {solution.target_total}",
        f"# offset: {solution.offset}",
    ]
    body = yaml.safe_dump(resolved, sort_keys=True)
    return "\n".join(header) + "\n" + body


def _summary_line(label: str, s: Solution) -> str:
    return (
        f"{label}: d_ff={s.d_ff} layers={s.n_layers} "
        f"attention_per_layer={s.attention_params_per_layer} total={s.total_params} "
        f"non_embedding={s.non_embedding_params} offset={s.offset}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.solve_config",
        description="Solve d_ff for each attention variant and print the resolved config.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", metavar="NAME", help="solve one variant and print its resolved YAML")
    group.add_argument("--all", action="store_true", help="solve every variant and describe every ladder")
    parser.add_argument("--target", type=int, metavar="N", help="total-parameter target for --variant")
    args = parser.parse_args(argv)
    if args.all and args.target is not None:
        parser.error("--target applies to --variant only")

    try:
        if args.all:
            for name in config.variant_names():
                print(_summary_line(f"variant {name}", solve(name)))
            for name in config.ladder_names():
                ladder_variant = config.load_ladder(name)["variant"]
                print(_summary_line(f"ladder {name}", describe(ladder_variant, ladder=name)))
        else:
            sys.stdout.write(resolved_yaml(solve(args.variant, target_total=args.target)))
    except (SolveError, config.ConfigError) as exc:
        print(f"solve_config: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
