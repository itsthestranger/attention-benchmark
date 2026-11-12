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
    window: int | None = None  # sliding-window size used in the tiny model; None if dense
    latent: int | None = None  # MLA latent width (d_c = d_c_q) used in the tiny model

    def held(self, n_tokens: int) -> int:
        """Tokens a KV cache holds after `n_tokens`: all of them, or the last window."""
        return n_tokens if self.window is None else min(n_tokens, self.window)


# T-04: the grouped family. T-07: sliding window. T-05: MLA.
GROUPED_CASES = (
    Case("mha", 8),
    Case("gqa4", 4),
    Case("gqa2", 2),
    Case("mqa", 1),
)
# The variant file says window 256. Tiny models use 16 so that short sequences
# cross several windows; tests that need the real window replace it.
SWA_CASES = (Case("swa256", 8, 16),)
REAL_WINDOW = 256  # plan §3.2
# The plan's MLA has latents of d_model / 2 and d_model / 4 and a RoPE key of
# head_dim / 2; the tiny models keep those proportions.
MLA_CASES = (Case("mla192", None, latent=32), Case("mla96", None, latent=16))
MLA_ROPE_DIM = 4
CASES = GROUPED_CASES + SWA_CASES + MLA_CASES

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
    if case.window is not None:
        dims["model.attention"] = {
            "type": "sliding_window",
            "n_kv_heads": case.n_kv_heads,
            "window": case.window,
        }
    if case.latent is not None:
        dims["model.attention"] = {
            "type": "mla",
            "d_c": case.latent,
            "d_c_q": case.latent,
            "rope_dim": MLA_ROPE_DIM,
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

    `sharpen` multiplies every attention parameter that produces queries or
    keys: local names starting with "w_q", "w_k", "w_uq" or "w_uk". At
    initialisation attention is nearly uniform, which hides masking and
    indexing errors; a factor of a few makes it peaked.
    """
    model = build_model(tiny_resolved(case, init_seed=init_seed, **overrides))
    if sharpen != 1.0:
        with torch.no_grad():
            for layer in model.layers:
                for name, p in layer.attn.named_parameters():
                    if name.startswith(("w_q", "w_k", "w_uq", "w_uk")):
                        p.mul_(sharpen)
    return model.to(dtype).eval()


def tokens(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, length), generator=g)
