"""Grouped attention: multi-head (G = H), grouped-query (1 < G < H) and multi-query (G = 1).

All variants keep H query heads and differ only in the number G of key/value
heads. Query head h reads KV head h // (H // G).

`ref` is plain tensor code with an explicit mask. `opt` calls the fused
`F.scaled_dot_product_attention`; for decode through a KV cache it groups the
queries so that K and V reach the kernel with their own head count, never
repeated to H.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.base import MODES, AttnBase, AttnConfig, register
from src.attention.token_cache import TokenStorage, append_tokens
from src.transformer import apply_rope


class KVCache:
    """A value: the first `length` tokens of a storage. Never modified by appending."""

    def __init__(self, storage: TokenStorage | None, length: int, capacity_hint: int = 0):
        self._storage = storage
        self.length = length
        self._capacity_hint = capacity_hint

    @property
    def capacity(self) -> int:
        """Tokens the current storage has room for (0 if none yet)."""
        return 0 if self._storage is None else self._storage.tensors[0].shape[2]

    @property
    def k(self) -> torch.Tensor | None:
        return None if self._storage is None else self._storage.tensors[0][:, :, : self.length]

    @property
    def v(self) -> torch.Tensor | None:
        return None if self._storage is None else self._storage.tensors[1][:, :, : self.length]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "KVCache":
        """A new cache holding this cache's tokens followed by k_new, v_new (B, G, T, dh)."""
        if k_new.shape != v_new.shape or k_new.dim() != 4:
            raise ValueError(f"k and v must be equal (B, G, T, dh) tensors, got {tuple(k_new.shape)}")
        if self._storage is not None:
            expected = (self.k.shape[0], self.k.shape[1], self.k.shape[3])
            if (k_new.shape[0], k_new.shape[1], k_new.shape[3]) != expected:
                raise ValueError(
                    f"new tokens have shape (B, G, dh) = {(k_new.shape[0], k_new.shape[1], k_new.shape[3])}, "
                    f"cache holds {expected}: batch size and heads must match"
                )
        storage, length = append_tokens(
            self._storage, self.length, self._capacity_hint, (k_new, v_new), axis=2
        )
        return KVCache(storage, length, self._capacity_hint)


def _allowed(past: int, T: int, S: int, device) -> torch.Tensor:
    """(T, S) boolean mask: new token t may see keys j with j < past + t + 1."""
    keys = torch.arange(S, device=device)
    limit = past + torch.arange(T, device=device) + 1
    return keys[None, :] < limit[:, None]


@register("grouped")
class GroupedAttention(AttnBase):
    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        if "n_kv_heads" not in cfg.options:
            raise ValueError("grouped attention needs the option n_kv_heads")
        G = cfg.options["n_kv_heads"]
        H, dh, d = cfg.n_heads, cfg.head_dim, cfg.d_model
        if isinstance(G, bool) or not isinstance(G, int) or G < 1 or H % G:
            raise ValueError(f"n_kv_heads must be a positive divisor of n_heads={H}, got {G!r}")
        self.n_heads = H
        self.n_kv_heads = G
        self.share = H // G
        self.head_dim = dh
        self.w_q = nn.Linear(d, H * dh, bias=False)
        self.w_k = nn.Linear(d, G * dh, bias=False)
        self.w_v = nn.Linear(d, G * dh, bias=False)
        self.w_o = nn.Linear(H * dh, d, bias=False)

    @property
    def rope_dim(self) -> int:
        return self.head_dim

    def residual_output_params(self) -> set[str]:
        return {"w_o.weight"}

    def init_cache(self, capacity: int | None = None) -> KVCache:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
            raise ValueError(f"capacity must be a non-negative int or None, got {capacity!r}")
        return KVCache(None, 0, capacity or 0)

    def _expand(self, t: torch.Tensor) -> torch.Tensor:
        """(B, G, S, dh) -> (B, H, S, dh), each KV head repeated for its query heads."""
        return t.repeat_interleave(self.share, dim=1)

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if cache is not None and x.requires_grad:
            raise RuntimeError(
                "decode through a KV cache is inference-only: run it under torch.no_grad()"
            )
        B, T, _ = x.shape
        H, G, share, dh = self.n_heads, self.n_kv_heads, self.share, self.head_dim

        q = self.w_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.w_k(x).view(B, T, G, dh).transpose(1, 2)
        v = self.w_v(x).view(B, T, G, dh).transpose(1, 2)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if cache is None:
            if mode == "ref":
                y = self._ref(q, k, v, _allowed(0, T, T, x.device))
            else:
                y = F.scaled_dot_product_attention(
                    q, self._expand(k), self._expand(v), is_causal=True
                )
            new_cache = None
        else:
            past = cache.length
            new_cache = cache.append(k, v)
            S = new_cache.length
            allowed = _allowed(past, T, S, x.device)
            if mode == "ref":
                y = self._ref(q, new_cache.k, new_cache.v, allowed)
            else:
                # Group the queries: (B, H, T, dh) -> (B, G, share * T, dh), so that
                # K and V keep their G heads. Row s * T + t is query head g * share + s, token t.
                qg = q.reshape(B, G, share, T, dh).reshape(B, G, share * T, dh)
                if T == 1:
                    out = F.scaled_dot_product_attention(qg, new_cache.k, new_cache.v)
                else:
                    out = F.scaled_dot_product_attention(
                        qg, new_cache.k, new_cache.v, attn_mask=allowed.repeat(share, 1)
                    )
                y = out.reshape(B, G, share, T, dh).reshape(B, H, T, dh)

        y = y.transpose(1, 2).reshape(B, T, H * dh)
        return self.w_o(y), new_cache

    def _ref(self, q, k, v, allowed):
        k = self._expand(k)
        v = self._expand(v)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v
