"""Shared helpers for the correctness suite of plan §5: tiny models of each variant.

Every test of the suite that says "every variant, every mode" parametrises over
`CASES`. Tasks that add an attention mechanism add their cases here.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from src import config
from src.model import Transformer, build_model

MODES = ("ref", "opt")


class Case(NamedTuple):
    name: str  # a variant file under configs/variants/
    n_kv_heads: int | None  # KV heads for grouped attention, None otherwise


# T-04: the grouped family. MLA and sliding-window cases join in T-05 and T-07.
GROUPED_CASES = (
    Case("mha", 8),
    Case("gqa4", 4),
    Case("gqa2", 2),
    Case("mqa", 1),
)
CASES = GROUPED_CASES

# Tiny dimensions that keep the plan's 8 query heads.
D_MODEL, N_HEADS, HEAD_DIM, N_LAYERS, D_FF, VOCAB = 64, 8, 8, 2, 96, 128


def tiny_resolved(case: Case, *, init_seed: int = 1, context_length: int = 64, **overrides) -> dict:
    """The variant's real config file, shrunk in width and depth only."""
    dims = {
        "model.d_model": D_MODEL,
        "model.n_heads": N_HEADS,
        "model.head_dim": HEAD_DIM,
        "model.n_layers": N_LAYERS,
        "model.d_ff": D_FF,
        "model.vocab_size": VOCAB,
        "model.context_length": context_length,
    }
    dims.update(overrides)
    return config.resolve(case.name, init_seed=init_seed, peak_lr=1e-3, overrides=dims)


def tiny_model(
    case: Case,
    *,
    init_seed: int = 1,
    sharpen: float = 1.0,
    dtype: torch.dtype = torch.float32,
    **overrides,
) -> Transformer:
    """A tiny model with the project's own initialisation.

    `sharpen` multiplies every attention parameter whose local name starts with
    "w_q" or "w_k". At initialisation attention is nearly uniform, which hides
    masking and indexing errors; a factor of a few makes it peaked.
    """
    model = build_model(tiny_resolved(case, init_seed=init_seed, **overrides))
    if sharpen != 1.0:
        with torch.no_grad():
            for layer in model.layers:
                for name, p in layer.attn.named_parameters():
                    if name.startswith(("w_q", "w_k")):
                        p.mul_(sharpen)
    return model.to(dtype).eval()


def tokens(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=g)
