"""The attention interface and the registry that maps config names to classes.

Every attention mechanism is a subclass of ``AttnBase`` registered under its
``model.attention.type`` name. The backbone (``src/transformer.py``) builds one
per layer through ``build_attention`` and never inspects the variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

MODES = ("ref", "opt")


@dataclass(frozen=True)
class AttnConfig:
    type: str                 # registry key, e.g. "grouped"
    d_model: int
    n_heads: int              # query heads
    head_dim: int
    n_layers: int
    context_length: int
    rope_theta: float
    kv_dtype: str             # "float16" from precision.autocast_dtype
    options: Mapping[str, Any]  # the rest of model["attention"], without "type"

    @classmethod
    def from_resolved(cls, resolved: dict) -> "AttnConfig":
        model = resolved["model"]
        attention = dict(model["attention"])
        attn_type = attention.pop("type")
        return cls(
            type=attn_type,
            d_model=model["d_model"],
            n_heads=model["n_heads"],
            head_dim=model["head_dim"],
            n_layers=model["n_layers"],
            context_length=model["context_length"],
            rope_theta=model["rope_theta"],
            kv_dtype=resolved["precision"]["autocast_dtype"],
            options=attention,
        )


class AttnBase(nn.Module):
    """Base class for attention mechanisms. Subclasses override the methods below."""

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        self._sdpa_backend = None  # plain attribute: not a parameter, not in state_dict()

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        """x: (B, T, d_model). freqs: RoPE table for exactly these T positions.

        cache None = full causal forward, returns (y, None).
        cache given = decode: attend to cached tokens plus x, return (y, new_cache).
        mode: 'ref' (plain ops) | 'opt' (fused / block-skipping / absorbed).
        """
        raise NotImplementedError

    def init_cache(self, capacity: int | None = None) -> Any:
        """An empty cache for decode."""
        raise NotImplementedError

    @property
    def rope_dim(self) -> int:
        """Width RoPE is applied over. Even."""
        raise NotImplementedError

    def residual_output_params(self) -> set[str]:
        """Local names (e.g. "o.weight") of the weights that write to the residual stream."""
        raise NotImplementedError

    def init_std_multipliers(self) -> dict[str, float]:
        """Local name -> multiplier on the base init std. Empty by default."""
        return {}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- decode cache -------------------------------------------------------------------

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes one token adds to the decode cache of one layer, at ``cfg.kv_dtype``."""
        raise NotImplementedError

    @property
    def kv_cache_max_tokens(self) -> int | None:
        """The most tokens the decode cache ever holds, or None if it grows without bound."""
        return None

    def kv_cache_bytes(self, n_tokens: int) -> int:
        """Bytes the decode cache of one layer holds after ``n_tokens`` tokens."""
        if isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 0:
            raise ValueError(f"n_tokens must be a non-negative int, got {n_tokens!r}")
        cap = self.kv_cache_max_tokens
        held = n_tokens if cap is None else min(n_tokens, cap)
        return self.kv_bytes_per_token * held

    # --- FLOPs (one multiply or one add is one FLOP; see plan §3.1) ------------------------

    def projection_flops_per_token(self) -> int:
        """Linear maps applied once per token: 2 x the weights of the module's 2-D matrices."""
        return 2 * sum(p.numel() for p in self.parameters() if p.dim() == 2)

    def mixing_flops_per_token(self, T: int, causal_skip: bool = True) -> int:
        """Score and value-mixing FLOPs per token, averaged over the T positions of a sequence."""
        raise NotImplementedError

    def flops_per_token(self, T: int, causal_skip: bool = True) -> int:
        """Forward attention FLOPs per token at context T: projection plus mixing."""
        check_context_length(T)
        return self.projection_flops_per_token() + self.mixing_flops_per_token(T, causal_skip)

    # --- fused kernel record ----------------------------------------------------------------

    @property
    def sdpa_backend(self) -> str | None:
        """The fused kernel recorded for this module, or None if none was enforced."""
        return self._sdpa_backend

    def set_sdpa_backend(self, name: str | None) -> None:
        """Record the fused kernel an experiment enforced. Never called from a forward pass."""
        if name is not None and name not in SDPA_BACKENDS:
            raise ValueError(f"sdpa backend must be one of {SDPA_BACKENDS} or None, got {name!r}")
        self._sdpa_backend = name


SDPA_BACKENDS = ("flash", "efficient", "math", "cudnn")


def bytes_per_value(dtype_name: str) -> int:
    """Bytes of one value of the floating-point dtype called ``dtype_name`` (e.g. "float16")."""
    dtype = getattr(torch, dtype_name, None) if isinstance(dtype_name, str) else None
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"unknown floating-point dtype name {dtype_name!r}")
    return torch.finfo(dtype).bits // 8


def check_context_length(T: int) -> None:
    if isinstance(T, bool) or not isinstance(T, int) or T < 1:
        raise ValueError(f"T must be an int >= 1, got {T!r}")


def mixing_flops(
    T: int,
    n_heads: int,
    score_width: int,
    value_width: int,
    *,
    causal_skip: bool,
    window: int | None = None,
) -> int:
    """Score and value-mixing FLOPs per token, rounded to the nearest integer.

    One query against one key costs 2 * n_heads * score_width for the scores and
    2 * n_heads * value_width for the weighted values. ``causal_skip=False`` scores
    every query against all T keys (a masked dense kernel, sliding window included).
    ``causal_skip=True`` counts the keys a kernel that skips masked keys reads:
    query t sees t + 1 keys, or min(t + 1, window) when a window is given.
    """
    check_context_length(T)
    if not isinstance(causal_skip, bool):
        raise ValueError(f"causal_skip must be a bool, got {causal_skip!r}")
    if not causal_skip:
        keys_total = T * T
    elif window is None or window >= T:
        keys_total = T * (T + 1) // 2
    else:
        keys_total = window * (window + 1) // 2 + (T - window) * window
    total = 2 * n_heads * (score_width + value_width) * keys_total  # FLOPs over the sequence
    return (2 * total + T) // (2 * T)  # total / T, rounded half up, in integers


_REGISTRY: dict[str, type[AttnBase]] = {}


def register(name: str):
    """Class decorator: make ``cls`` available as ``model.attention.type == name``."""

    def decorator(cls):
        if not (isinstance(cls, type) and issubclass(cls, AttnBase)):
            raise ValueError(f"{cls!r} is not an AttnBase subclass; cannot register it as {name!r}")
        if name in _REGISTRY:
            raise ValueError(f"attention type {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def unregister(name: str) -> None:
    if name not in _REGISTRY:
        raise KeyError(f"attention type {name!r} is not registered")
    del _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)


def build_attention(cfg: AttnConfig) -> AttnBase:
    if cfg.type not in _REGISTRY:
        raise ValueError(f"unknown attention type {cfg.type!r}; registered: {registered()}")
    return _REGISTRY[cfg.type](cfg)
