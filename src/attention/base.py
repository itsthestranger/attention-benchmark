"""The attention interface and the registry that maps config names to classes.

Every attention mechanism is a subclass of ``AttnBase`` registered under its
``model.attention.type`` name. The backbone (``src/transformer.py``) builds one
per layer through ``build_attention`` and never inspects the variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        """x: (B, T, d_model). freqs: RoPE table for exactly these T positions.

        cache None = full causal forward, returns (y, None).
        cache given = decode: attend to cached tokens plus x, return (y, new_cache).
        mode: 'ref' (plain ops) | 'opt' (fused / block-skipping / absorbed).
        """
        raise NotImplementedError

    def init_cache(self) -> Any:
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

    @property
    def kv_bytes_per_token(self) -> int:
        raise NotImplementedError

    def flops_per_token(self, T: int, causal_skip: bool = True) -> int:
        raise NotImplementedError

    @property
    def sdpa_backend(self) -> str | None:
        raise NotImplementedError


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
