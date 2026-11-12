"""T-12: a gated linear recurrence whose whole memory is one d_model vector (plan §9.2).

This is the control module. It is deliberately weak and is not attention: it
must fail where attention succeeds, which shows that the synthetic memory tasks
can tell models apart.

Per channel, with positions t = 0 .. T-1 and s_{-1} = 0:

    g_t = sigmoid(w_g x_t)
    s_t = g_t * s_{t-1} + (1 - g_t) * (w_i x_t)
    y_t = w_o s_t
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.attention.base import MODES, AttnBase, register


@dataclass(frozen=True)
class RecurrenceState:
    s: torch.Tensor | None     # (B, d_model), or None before any token


@register("recurrence")
class GatedRecurrence(AttnBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.d_model
        self.w_i = nn.Linear(d, d, bias=False)
        self.w_g = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        # freqs is ignored: a recurrence carries position in its state.
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        B, T, d = x.shape

        # The gate and the input projection for every position at once. These
        # are the only batched matrix products in this module.
        g = torch.sigmoid(self.w_g(x))
        u = self.w_i(x)

        # Starting state. A cache with s=None is the same as no cache.
        if cache is None or cache.s is None:
            s = x.new_zeros(B, d)
        else:
            s = cache.s

        # The state update is sequential. There is no faster exact form: a closed
        # form needs cumulative products of the gates, which underflow for long
        # sequences. So "ref" and "opt" run the same loop and give identical output.
        states = []
        for t in range(T):
            gt = g[:, t]
            s = gt * s + (1 - gt) * u[:, t]
            states.append(s)

        S = torch.stack(states, dim=1) if states else x.new_zeros(B, 0, d)
        y = self.w_o(S)

        # A new object every time: the cache passed in is never mutated.
        new_cache = None if cache is None else RecurrenceState(s)
        return y, new_cache

    def init_cache(self) -> RecurrenceState:
        return RecurrenceState(None)

    @property
    def rope_dim(self) -> int:
        # Not used by the recurrence; the model needs a value to build its table.
        return self.cfg.head_dim

    def residual_output_params(self) -> set[str]:
        return {"w_o.weight"}

    @property
    def kv_bytes_per_token(self) -> int:
        # Nothing is stored per token: the whole memory is state_values floats.
        return 0

    @property
    def state_values(self) -> int:
        return self.cfg.d_model

    @property
    def sdpa_backend(self) -> None:
        return None
