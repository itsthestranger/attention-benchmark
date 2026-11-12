"""Multi-head latent attention (DeepSeek-V2), with two decode paths.

Keys and values are compressed into one latent c_KV per token (normalised), and
the cache holds that latent plus a shared positional key k_R (already
rotated). Position is carried by decoupled RoPE: it touches only q_R and k_R,
never the content parts q_C, k_C, v. The cache stores one tensor per token,
[c_KV ; k_R], of width d_c + r.

Without a cache (training, and full-sequence evaluation) both modes materialise
per-head keys and values from the latent: `ref` is plain tensor code with an
explicit mask, `opt` makes one call to `F.scaled_dot_product_attention` on
queries [q_C ; q_R] and keys [k_C ; k_R].

Through a cache, `ref` is the naive path: it rebuilds k_C and v for every cached
token on every call. `opt` is the absorbed path (inference only): the
up-projections are folded into the query and the output once, so attention runs
over the cached latents directly. Its query is [W_UK^T W_UQ c_Q ; q_R] per head,
its key is the cache itself, its value is the c_KV part of the cache, and the
output is mapped back by W_O W_UV per head. Neither w_uk, w_uv, w_uq nor w_o is
applied on that path. The absorbed matrices are computed from the weights when
the cache is created, so a cache belongs to the weights it was created with.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (tests wrap this module's F attribute)

from src.attention.base import MODES, AttnBase, AttnConfig, bytes_per_value, mixing_flops, register
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
    """A value: the first `length` tokens of a storage holding [c_KV ; k_R] per token.

    `d_c` is the width of the c_KV part; the rest of the width is k_R. `absorbed` is
    the (q_absorb, o_absorb) pair the cache was created with; caches appended from
    it carry the same tuple. Appending never changes this object or the tokens it
    describes.
    """

    def __init__(
        self,
        storage: TokenStorage | None,
        length: int,
        values_per_token: int,
        capacity_hint: int = 0,
        *,
        d_c: int,
        absorbed: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        self._storage = storage
        self.length = length
        self.values_per_token = values_per_token
        self._capacity_hint = capacity_hint
        self._d_c = d_c
        self.absorbed = absorbed

    @property
    def capacity(self) -> int:
        """Tokens the current storage has room for (0 if none yet)."""
        return 0 if self._storage is None else self._storage.tensors[0].shape[1]

    @property
    def latent(self) -> torch.Tensor | None:
        """The first `length` tokens, each [c_KV ; k_R], shape (B, length, d_c + r)."""
        return None if self._storage is None else self._storage.tensors[0][:, : self.length]

    @property
    def c_kv(self) -> torch.Tensor | None:
        """A view of the latent part, shape (B, length, d_c)."""
        return None if self._storage is None else self.latent[..., : self._d_c]

    @property
    def k_r(self) -> torch.Tensor | None:
        """A view of the positional part, shape (B, length, r)."""
        return None if self._storage is None else self.latent[..., self._d_c :]

    def append(self, c_kv: torch.Tensor, k_r: torch.Tensor) -> "LatentCache":
        """A new cache holding this cache's tokens followed by c_kv (B, T, d_c) and k_r (B, T, r)."""
        if c_kv.dim() != 3 or k_r.dim() != 3 or c_kv.shape[:2] != k_r.shape[:2]:
            raise ValueError(
                f"c_kv and k_r must be (B, T, width) tensors with equal B and T, "
                f"got {tuple(c_kv.shape)} and {tuple(k_r.shape)}"
            )
        r = self.values_per_token - self._d_c
        if (c_kv.shape[2], k_r.shape[2]) != (self._d_c, r):
            raise ValueError(
                f"new tokens have widths (d_c, r) = {(c_kv.shape[2], k_r.shape[2])}, "
                f"cache holds {(self._d_c, r)}"
            )
        if self._storage is not None and c_kv.shape[0] != self._storage.tensors[0].shape[0]:
            raise ValueError(
                f"new tokens have batch size {c_kv.shape[0]}, "
                f"cache holds {self._storage.tensors[0].shape[0]}: batch size must match"
            )
        storage, length = append_tokens(
            self._storage, self.length, self._capacity_hint, (torch.cat([c_kv, k_r], dim=-1),), axis=1
        )
        return LatentCache(
            storage,
            length,
            self.values_per_token,
            self._capacity_hint,
            d_c=self._d_c,
            absorbed=self.absorbed,
        )


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

    @property
    def kv_bytes_per_token(self) -> int:
        # The latent c_KV and the shared positional key k_R.
        return (self.d_c + self._rope_dim) * bytes_per_value(self.cfg.kv_dtype)

    @property
    def sdpa_backend(self) -> str | None:
        # The fused kernel recorded by whoever enforced one (set_sdpa_backend), or None.
        return self._sdpa_backend

    def mixing_flops_per_token(self, T: int, causal_skip: bool = True) -> int:
        # Scores are (head_dim + rope_dim) wide: content and positional parts; values are head_dim wide.
        return mixing_flops(
            T,
            self.n_heads,
            self.head_dim + self._rope_dim,
            self.head_dim,
            causal_skip=causal_skip,
        )

    def init_std_multipliers(self) -> dict[str, float]:
        d = self.cfg.d_model
        up = math.sqrt(d / self.d_c)
        up_q = math.sqrt(d / self.d_c_q)
        return {"w_uk.weight": up, "w_uv.weight": up, "w_uq.weight": up_q, "w_qr.weight": up_q}

    def absorbed_matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(q_absorb, o_absorb) from the current weights, detached.

        q_absorb[h] = W_UK[h]^T W_UQ[h], shape (H, d_c, d_c_q): the query's content part
        becomes q_absorb[h] c_Q, which dots directly with the cached c_KV.
        o_absorb[h] = W_O[:, h] W_UV[h], shape (H, d, d_c): maps the attention-weighted
        c_KV of head h to the model dimension.
        """
        H, dh, d = self.n_heads, self.head_dim, self.cfg.d_model
        with torch.no_grad():
            w_uk = self.w_uk.weight.view(H, dh, self.d_c)
            w_uq = self.w_uq.weight.view(H, dh, self.d_c_q)
            w_uv = self.w_uv.weight.view(H, dh, self.d_c)
            w_o = self.w_o.weight.view(d, H, dh).permute(1, 0, 2)  # (H, d, dh)
            return w_uk.transpose(1, 2) @ w_uq, w_o @ w_uv

    def init_cache(self, capacity: int | None = None) -> LatentCache:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
            raise ValueError(f"capacity must be a non-negative int or None, got {capacity!r}")
        return LatentCache(
            None,
            0,
            self.d_c + self._rope_dim,
            capacity or 0,
            d_c=self.d_c,
            absorbed=self.absorbed_matrices(),
        )

    def _rope_queries(self, c_q, freqs):
        """q_R (B, H, T, r): per-head positional queries from c_Q (B, T, d_c_q), rotated."""
        B, T, _ = c_q.shape
        H, r = self.n_heads, self._rope_dim
        return apply_rope(self.w_qr(c_q).view(B, T, H, r).transpose(1, 2), freqs)

    def _project(self, x, freqs):
        """The cached pair and the queries: c_KV (B, T, d_c), k_R (B, T, r) rotated and shared
        by the heads, q_C (B, H, T, dh), q_R (B, H, T, r) rotated."""
        B, T, _ = x.shape
        H, dh = self.n_heads, self.head_dim
        c_kv = self.norm_kv(self.w_dkv(x))
        k_r = apply_rope(self.w_kr(x), freqs)
        c_q = self.norm_q(self.w_dq(x))
        q_c = self.w_uq(c_q).view(B, T, H, dh).transpose(1, 2)
        q_r = self._rope_queries(c_q, freqs)
        return c_kv, k_r, q_c, q_r

    def _content_keys(self, latents):
        """k_C (B, H, S, dh) from the latents (B, S, d_c)."""
        B, S, _ = latents.shape
        return self.w_uk(latents).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _logits(self, q_c, q_r, k_c, k_r):
        """Content plus positional scores (B, H, n, S), divided by sqrt(d_h + d_h^R)."""
        score = q_c @ k_c.transpose(-1, -2) + q_r @ k_r[:, None].transpose(-1, -2)
        return score / math.sqrt(self.head_dim + self._rope_dim)

    def attention_logits(self, x, freqs):
        c_kv, k_r, q_c, q_r = self._project(x, freqs)
        return self._logits(q_c, q_r, self._content_keys(c_kv), k_r)

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if cache is not None and x.requires_grad:
            raise RuntimeError(
                "decode through a KV cache is inference-only: run it under torch.no_grad()"
            )
        if cache is not None and mode == "opt":
            return self._absorbed_forward(x, freqs, cache)

        B, T, _ = x.shape
        H, dh, r = self.n_heads, self.head_dim, self._rope_dim

        c_kv, k_r, q_c, q_r = self._project(x, freqs)

        if cache is None:
            past, new_cache = 0, None
            latents, positional = c_kv, k_r
        else:
            past = cache.length
            new_cache = cache.append(c_kv, k_r)
            latents, positional = new_cache.c_kv, new_cache.k_r
        S = latents.shape[1]

        k_c = self._content_keys(latents)
        v = self.w_uv(latents).view(B, S, H, dh).transpose(1, 2)
        allowed = _allowed(past, T, S, x.device)

        if mode == "ref":
            score = self._logits(q_c, q_r, k_c, positional).masked_fill(~allowed, float("-inf"))
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

    def _absorbed_forward(self, x, freqs, cache):
        """Decode through a cache over the cached latents, with no per-head keys or values.

        Score for head h, token t, cached token j:
          (W_UK[h]^T W_UQ[h] c_Q) . c_KV[j] + q_R[h, t] . k_R[j]
        which is one dot product of width d_c + r between the query [qt ; q_R] and the
        cache entry [c_KV ; k_R]. The attention output is the c_KV-weighted sum, mapped
        back by o_absorb[h] = W_O[:, h] W_UV[h] and summed over heads.
        """
        if cache.absorbed is None:
            raise ValueError("this cache was not created by init_cache(); it has no absorbed matrices")
        q_absorb, o_absorb = cache.absorbed
        B, T, _ = x.shape
        H, r, d_c = self.n_heads, self._rope_dim, self.d_c

        c_kv = self.norm_kv(self.w_dkv(x))
        k_r = apply_rope(self.w_kr(x), freqs)
        c_q = self.norm_q(self.w_dq(x))
        q_r = self._rope_queries(c_q, freqs)
        qt = torch.einsum("hcq,btq->bhtc", q_absorb.to(c_q.dtype), c_q)  # (B, H, T, d_c)

        past = cache.length
        new_cache = cache.append(c_kv, k_r)
        latent = new_cache.latent  # (B, S, d_c + r), the cache's own memory
        S = latent.shape[1]

        # Row h * T + t of the single query group is head h, token t.
        query = torch.cat([qt, q_r], dim=-1).reshape(B, 1, H * T, d_c + r)
        key = latent[:, None]  # (B, 1, S, d_c + r)
        value = latent[:, None, :, :d_c]  # (B, 1, S, d_c), a view
        scale = 1.0 / math.sqrt(self.head_dim + r)  # the per-head score scale, not 1 / sqrt(d_c + r)
        if T == 1:
            out = F.scaled_dot_product_attention(query, key, value, scale=scale)
        else:
            mask = _allowed(past, T, S, x.device).repeat(H, 1)
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)

        out = out.reshape(B, H, T, d_c)
        y = torch.einsum("hdc,bhtc->btd", o_absorb.to(out.dtype), out)
        return y, new_cache
