"""Backbone pieces shared by every attention variant: RMSNorm, SwiGLU, RoPE, Block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.attention.base import AttnConfig, build_attention

RMSNORM_EPS = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise in at least float32 (float16 and bfloat16 are upcast), then
        # return to the input dtype. float32 and float64 use their own dtype.
        compute = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
        xc = x.to(compute)
        y = xc * torch.rsqrt(xc.pow(2).mean(dim=-1, keepdim=True) + RMSNORM_EPS)
        return (y * self.gain.to(compute)).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def rope_freqs(
    rope_dim: int,
    start: int,
    length: int,
    theta: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """RoPE table of shape (length, rope_dim // 2, 2) for positions start .. start+length-1.

    ``[..., 0]`` holds cos and ``[..., 1]`` holds sin of ``p * theta ** (-2i / rope_dim)``.
    """
    if isinstance(rope_dim, bool) or not isinstance(rope_dim, int) or rope_dim <= 0 or rope_dim % 2:
        raise ValueError(f"rope_dim must be a positive even int, got {rope_dim!r}")
    pairs = torch.arange(rope_dim // 2, dtype=torch.float32, device=device)
    inv_freq = theta ** (-2.0 * pairs / rope_dim)
    positions = torch.arange(start, start + length, dtype=torch.float32, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs (x[2i], x[2i+1]) of x by the angles in freqs.

    x: (..., T, rope_dim). freqs: (T, rope_dim // 2, 2), cast to x.dtype.
    """
    if x.dim() < 2:
        raise ValueError(f"x must have shape (..., T, rope_dim), got {tuple(x.shape)}")
    t, rope_dim = x.shape[-2], x.shape[-1]
    if rope_dim % 2:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if tuple(freqs.shape) != (t, rope_dim // 2, 2):
        raise ValueError(
            f"freqs shape {tuple(freqs.shape)} does not match x: expected {(t, rope_dim // 2, 2)}"
        )
    f = freqs.to(x.dtype)
    cos, sin = f[..., 0], f[..., 1]
    even, odd = x[..., 0::2], x[..., 1::2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class Block(nn.Module):
    """Pre-norm residual block: attention then SwiGLU."""

    def __init__(self, cfg: AttnConfig, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = build_attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, d_ff)

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        a, cache = self.attn(self.norm1(x), freqs, cache, mode=mode)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, cache
