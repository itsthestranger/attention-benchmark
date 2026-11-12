"""The transformer backbone, identical for every attention variant (plan §3.1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.base import MODES, AttnConfig
from src.transformer import Block, RMSNorm, rope_freqs
from src.utils.seeding import init_model

REQUIRED_INIT = {
    "distribution": "normal",
    "residual_output_scale": "inverse_sqrt_two_layers",
    "norm_gain": 1.0,
    "seeding": "per_parameter",
}


@dataclass
class DecodeState:
    caches: list     # one per layer
    pos: int         # tokens already consumed


class Transformer(nn.Module):
    def __init__(self, resolved: dict):
        super().__init__()
        m = resolved["model"]
        if m["n_heads"] * m["head_dim"] != m["d_model"]:
            raise ValueError(
                f"n_heads * head_dim must equal d_model: "
                f"{m['n_heads']} * {m['head_dim']} != {m['d_model']}"
            )
        if m["tie_embeddings"] is not True:
            raise ValueError(f"only tie_embeddings=True is supported, got {m['tie_embeddings']!r}")
        if m["bias"] is not False:
            raise ValueError(f"only bias=False is supported, got {m['bias']!r}")
        if m["norm"] != "rmsnorm":
            raise ValueError(f"only norm='rmsnorm' is supported, got {m['norm']!r}")
        if m["activation"] != "swiglu":
            raise ValueError(f"only activation='swiglu' is supported, got {m['activation']!r}")
        if m["n_layers"] < 1:
            raise ValueError(f"n_layers must be at least 1, got {m['n_layers']!r}")

        attn_cfg = AttnConfig.from_resolved(resolved)
        self.n_layers = m["n_layers"]
        self.rope_theta = m["rope_theta"]
        self.init_std = resolved["init"]["std"]
        self.embed = nn.Embedding(m["vocab_size"], m["d_model"])
        self.layers = nn.ModuleList(Block(attn_cfg, m["d_ff"]) for _ in range(self.n_layers))
        self.norm_f = RMSNorm(m["d_model"])

    def forward(
        self,
        idx: torch.Tensor,
        *,
        mode: str = "ref",
        state: DecodeState | None = None,
    ) -> tuple[torch.Tensor, DecodeState | None]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if state is not None and len(state.caches) != self.n_layers:
            raise ValueError(f"state has {len(state.caches)} caches for {self.n_layers} layers")

        T = idx.shape[-1]
        start = 0 if state is None else state.pos
        rope_dim = self.layers[0].attn.rope_dim
        freqs = rope_freqs(rope_dim, start, T, self.rope_theta, device=idx.device)

        x = self.embed(idx)
        if state is None:
            for layer in self.layers:
                x, _ = layer(x, freqs, None, mode=mode)
            new_state = None
        else:
            new_caches = []
            for layer, cache in zip(self.layers, state.caches):
                x, cache = layer(x, freqs, cache, mode=mode)
                new_caches.append(cache)
            new_state = DecodeState(caches=new_caches, pos=state.pos + T)

        logits = F.linear(self.norm_f(x), self.embed.weight)
        return logits, new_state

    def init_state(self) -> DecodeState:
        return DecodeState(caches=[layer.attn.init_cache() for layer in self.layers], pos=0)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_non_embedding_params(self) -> int:
        return self.n_params - self.embed.weight.numel()

    @property
    def n_non_attention_params(self) -> int:
        attn_params = sum(layer.attn.n_params for layer in self.layers)
        return self.n_params - attn_params


def build_model(resolved: dict) -> Transformer:
    """The only supported way to get a model for training or evaluation."""
    for key, expected in REQUIRED_INIT.items():
        value = resolved["init"][key]
        if value != expected:
            raise ValueError(f"init.{key} must be {expected!r} to match the init rule, got {value!r}")
    model = Transformer(resolved)
    init_model(model, resolved["run"]["init_seed"])
    return model
