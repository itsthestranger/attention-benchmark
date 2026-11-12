"""Multi-head latent attention (DeepSeek-V2), naive path: keys and values are
materialised per head from the cached latent on every call.

Keys and values are compressed into one latent c_KV per token (normalised), and
the cache holds only that latent plus a shared positional key k_R (already
rotated). Position is carried by decoupled RoPE: it touches only q_R and k_R,
never the content parts q_C, k_C, v.

`ref` is plain tensor code with an explicit mask. `opt` makes one call to
`F.scaled_dot_product_attention` on queries [q_C ; q_R] and keys [k_C ; k_R].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (tests wrap this module's F attribute)

from src.attention.base import MODES, AttnBase, AttnConfig, register
from src.attention.grouped import _allowed
from src.attention.token_cache import TokenStorage, append_tokens
from src.transformer import RMSNorm, apply_rope


def _int_option(options, name: str, *, even: bool = False) -> int:
    if name not in options:
        raise ValueError(f"mla needs the option {name}")
    value = options[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or (even and value % 2):
        want = "a positive even int" if even else "a positive int"
        raise ValueError(f"{name} must be {want}, got {value!r}")
    return value


class LatentCache:
    """A value: the first `length` tokens of a storage holding c_KV (B, S, d_c) and k_R (B, S, r).

    Appending never changes this object or the tokens it describes.
    """

    def __init__(self, storage: TokenStorage | None, length: int, values_per_token: int, capacity_hint: int = 0):
        self._storage = storage
        self.length = length
        self.values_per_token = values_per_token
        self._capacity_hint = capacity_hint

    @property
    def capacity(self) -> int:
        """Tokens the current storage has room for (0 if none yet)."""
        return 0 if self._storage is None else self._storage.tensors[0].shape[1]

    @property
    def c_kv(self) -> torch.Tensor | None:
        return None if self._storage is None else self._storage.tensors[0][:, : self.length]

    @property
    def k_r(self) -> torch.Tensor | None:
        return None if self._storage is None else self._storage.tensors[1][:, : self.length]

    def append(self, c_kv: torch.Tensor, k_r: torch.Tensor) -> "LatentCache":
        """A new cache holding this cache's tokens followed by c_kv (B, T, d_c) and k_r (B, T, r)."""
        if c_kv.dim() != 3 or k_r.dim() != 3 or c_kv.shape[:2] != k_r.shape[:2]:
            raise ValueError(
                f"c_kv and k_r must be (B, T, width) tensors with equal B and T, "
                f"got {tuple(c_kv.shape)} and {tuple(k_r.shape)}"
            )
        if self._storage is not None:
            held_c, held_r = self._storage.tensors
            if (c_kv.shape[0], c_kv.shape[2], k_r.shape[2]) != (held_c.shape[0], held_c.shape[2], held_r.shape[2]):
                raise ValueError(
                    f"new tokens have shape (B, d_c, r) = {(c_kv.shape[0], c_kv.shape[2], k_r.shape[2])}, "
                    f"cache holds {(held_c.shape[0], held_c.shape[2], held_r.shape[2])}: "
                    "batch size and widths must match"
                )
        storage, length = append_tokens(
            self._storage, self.length, self._capacity_hint, (c_kv, k_r), axis=1
        )
        return LatentCache(storage, length, self.values_per_token, self._capacity_hint)


@register("mla")
class LatentAttention(AttnBase):
    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        opts = cfg.options
        self.d_c = _int_option(opts, "d_c")
        self.d_c_q = _int_option(opts, "d_c_q")
        self._rope_dim = _int_option(opts, "rope_dim", even=True)
        d, H, dh, r = cfg.d_model, cfg.n_heads, cfg.head_dim, self._rope_dim
        self.n_heads = H
        self.head_dim = dh

        self.w_dkv = nn.Linear(d, self.d_c, bias=False)
        self.norm_kv = RMSNorm(self.d_c)
        self.w_uk = nn.Linear(self.d_c, H * dh, bias=False)
        self.w_uv = nn.Linear(self.d_c, H * dh, bias=False)
        self.w_kr = nn.Linear(d, r, bias=False)
        self.w_dq = nn.Linear(d, self.d_c_q, bias=False)
        self.norm_q = RMSNorm(self.d_c_q)
        self.w_uq = nn.Linear(self.d_c_q, H * dh, bias=False)
        self.w_qr = nn.Linear(self.d_c_q, H * r, bias=False)
        self.w_o = nn.Linear(H * dh, d, bias=False)

    @property
    def rope_dim(self) -> int:
        return self._rope_dim

    def residual_output_params(self) -> set[str]:
        return {"w_o.weight"}

    def init_std_multipliers(self) -> dict[str, float]:
        d = self.cfg.d_model
        up = math.sqrt(d / self.d_c)
        up_q = math.sqrt(d / self.d_c_q)
        return {"w_uk.weight": up, "w_uv.weight": up, "w_uq.weight": up_q, "w_qr.weight": up_q}

    def init_cache(self, capacity: int | None = None) -> LatentCache:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
            raise ValueError(f"capacity must be a non-negative int or None, got {capacity!r}")
        return LatentCache(None, 0, self.d_c + self._rope_dim, capacity or 0)

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if cache is not None and x.requires_grad:
            raise RuntimeError(
                "decode through a KV cache is inference-only: run it under torch.no_grad()"
            )
        B, T, _ = x.shape
        H, dh, r = self.n_heads, self.head_dim, self._rope_dim

        c_kv = self.norm_kv(self.w_dkv(x))                    # (B, T, d_c), cached
        k_r = apply_rope(self.w_kr(x), freqs)                 # (B, T, r), cached, shared by heads
        c_q = self.norm_q(self.w_dq(x))
        q_c = self.w_uq(c_q).view(B, T, H, dh).transpose(1, 2)
        q_r = apply_rope(self.w_qr(c_q).view(B, T, H, r).transpose(1, 2), freqs)

        if cache is None:
            past, new_cache = 0, None
            latents, positional = c_kv, k_r
        else:
            past = cache.length
            new_cache = cache.append(c_kv, k_r)
            latents, positional = new_cache.c_kv, new_cache.k_r
        S = latents.shape[1]

        k_c = self.w_uk(latents).view(B, S, H, dh).transpose(1, 2)
        v = self.w_uv(latents).view(B, S, H, dh).transpose(1, 2)
        allowed = _allowed(past, T, S, x.device)

        if mode == "ref":
            score = (q_c @ k_c.transpose(-1, -2) + q_r @ positional[:, None].transpose(-1, -2))
            score = score / math.sqrt(dh + r)
            score = score.masked_fill(~allowed, float("-inf"))
            y = torch.softmax(score, dim=-1) @ v
        else:
            q = torch.cat([q_c, q_r], dim=-1)
            k = torch.cat([k_c, positional[:, None].expand(B, H, S, r)], dim=-1)
            if cache is None:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif T == 1:
                y = F.scaled_dot_product_attention(q, k, v)  # every cached key is visible
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)

        y = y.transpose(1, 2).reshape(B, T, H * dh)
        return self.w_o(y), new_cache
