"""Sliding-window attention: query t sees keys j with t - W < j <= t (W keys including itself).

Projections, RoPE, scale, head order and KV grouping are those of
`GroupedAttention`, which this class subclasses. Only the set of visible keys
differs.

`ref` is plain tensor code with an explicit mask. `opt` skips the work outside
the window, in one of two ways. The path is the option `opt_path` of the variant's
`attention` block (set at construction; `set_opt_path` changes it afterwards):

* "chunked" (default): queries in chunks of at most W tokens; a chunk's keys are
  the W tokens before it plus its own. One `F.scaled_dot_product_attention` call
  per chunk.
* "flex": FlexAttention with a sliding-window block mask (compiled on CUDA).

Decode through a KV cache always uses the chunked rules. The cache holds the
last W tokens and is a value; see `WindowCache`.
"""

import torch.nn.functional as F  # noqa: N812  (tests wrap this module's F attribute)

import math

import torch

from src.attention.base import MODES, mixing_flops, register
from src.attention.grouped import GroupedAttention
from src.transformer import apply_rope

OPT_PATHS = ("chunked", "flex")
DEFAULT_OPT_PATH = "chunked"
_COMPILED_FLEX = None


def flex_compiled_available() -> bool:
    """True when FlexAttention can be imported and a CUDA device is present."""
    try:
        import torch.nn.attention.flex_attention  # noqa: F401
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _flex_importable() -> bool:
    """True when this PyTorch provides FlexAttention (CUDA is not needed to build the path)."""
    try:
        import torch.nn.attention.flex_attention  # noqa: F401
    except ImportError:
        return False
    return True


def _checked_opt_path(name) -> str:
    """The name of an opt path this module knows and can build here; ValueError otherwise."""
    if name not in OPT_PATHS:
        raise ValueError(f"opt_path must be one of {OPT_PATHS}, got {name!r}")
    if name == "flex" and not _flex_importable():
        raise ValueError(
            "opt_path 'flex' needs torch.nn.attention.flex_attention, which this PyTorch does not "
            "provide; use opt_path 'chunked'"
        )
    return name


def _compiled_flex_attention():
    global _COMPILED_FLEX
    if _COMPILED_FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX


def _window_mask(queries: torch.Tensor, keys: torch.Tensor, window: int) -> torch.Tensor:
    """(len(queries), len(keys)) bool, True where query t may see key j: t - W < j <= t."""
    t = queries[:, None]
    j = keys[None, :]
    return (j <= t) & (t - j < window)


class _Buffers:
    """One allocation, possibly shared by several WindowCache objects.

    Slots below `filled` have been written by some cache and are never overwritten.
    """

    __slots__ = ("k", "v", "filled")

    def __init__(self, k: torch.Tensor, v: torch.Tensor, filled: int):
        self.k = k
        self.v = v
        self.filled = filled


class WindowCache:
    """A value: the last `length` (at most W) tokens of a storage, in order.

    Appending never changes this object or the tokens it describes. It writes in
    place only where no other cache has written, and copies otherwise.
    """

    def __init__(self, buffers, start: int, length: int, window: int, capacity_hint: int = 0):
        self._buf = buffers
        self._start = start
        self.length = length
        self.window = window
        self._capacity_hint = capacity_hint

    @property
    def capacity(self) -> int:
        """Tokens the current storage has room for (0 if none yet); at most 2 * window."""
        return 0 if self._buf is None else self._buf.k.shape[2]

    @property
    def k(self) -> torch.Tensor | None:
        if self._buf is None:
            return None
        return self._buf.k[:, :, self._start : self._start + self.length]

    @property
    def v(self) -> torch.Tensor | None:
        if self._buf is None:
            return None
        return self._buf.v[:, :, self._start : self._start + self.length]

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "WindowCache":
        """A new cache holding the last `window` tokens of this cache followed by k_new, v_new."""
        if k_new.shape != v_new.shape or k_new.dim() != 4:
            raise ValueError(f"k and v must be equal (B, G, T, dh) tensors, got {tuple(k_new.shape)}")
        W = self.window
        if k_new.shape[2] > W:  # only the last W new tokens can be held
            k_new, v_new = k_new[:, :, -W:], v_new[:, :, -W:]
        n = k_new.shape[2]
        buf = self._buf
        if buf is not None:
            expected = (buf.k.shape[0], buf.k.shape[1], buf.k.shape[3])
            if (k_new.shape[0], k_new.shape[1], k_new.shape[3]) != expected:
                raise ValueError(
                    f"new tokens have shape (B, G, dh) = {(k_new.shape[0], k_new.shape[1], k_new.shape[3])}, "
                    f"cache holds {expected}: batch size and heads must match"
                )
        L = self.length
        held = min(L + n, W)
        kept = held - n  # old tokens that stay in the window
        end = self._start + L
        with torch.no_grad():
            if buf is not None and buf.filled == end and end + n <= buf.k.shape[2]:
                storage, start = buf, end - kept  # room, and nobody wrote past `end`
            else:
                if buf is None:
                    cap = min(2 * W, max(n, self._capacity_hint))
                else:
                    cap = 2 * W
                shape = (k_new.shape[0], k_new.shape[1], cap, k_new.shape[3])
                k_buf = torch.empty(shape, dtype=k_new.dtype, device=k_new.device)
                v_buf = torch.empty(shape, dtype=v_new.dtype, device=v_new.device)
                if kept:
                    k_buf[:, :, :kept] = buf.k[:, :, end - kept : end]
                    v_buf[:, :, :kept] = buf.v[:, :, end - kept : end]
                storage, start = _Buffers(k_buf, v_buf, 0), 0
            write = start + kept
            storage.k[:, :, write : write + n] = k_new
            storage.v[:, :, write : write + n] = v_new
        storage.filled = write + n
        return WindowCache(storage, start, held, W, self._capacity_hint)


@register("sliding_window")
class SlidingWindowAttention(GroupedAttention):
    def __init__(self, cfg):
        super().__init__(cfg)
        W = cfg.options.get("window")
        if isinstance(W, bool) or not isinstance(W, int) or W < 1:
            raise ValueError(f"window must be an int >= 1, got {W!r}")
        self.window = W
        # The variant's `attention.opt_path` (plan §2.3): part of the resolved config and the hash.
        self.opt_path = _checked_opt_path(cfg.options.get("opt_path", DEFAULT_OPT_PATH))
        self._block_masks = {}

    @property
    def kv_cache_max_tokens(self) -> int:
        # The cache holds the last `window` tokens; its bytes per token are GroupedAttention's.
        return self.window

    def mixing_flops_per_token(self, T: int, causal_skip: bool = True) -> int:
        return mixing_flops(
            T, self.n_heads, self.head_dim, self.head_dim, causal_skip=causal_skip, window=self.window
        )

    def set_opt_path(self, name):
        self.opt_path = _checked_opt_path(name)

    def init_cache(self, capacity: int | None = None) -> WindowCache:
        if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
            raise ValueError(f"capacity must be a non-negative int or None, got {capacity!r}")
        return WindowCache(None, 0, 0, self.window, capacity or 0)

    # --- shared pieces -------------------------------------------------------------------

    def _project(self, x, freqs):
        B, T, _ = x.shape
        H, G, dh = self.n_heads, self.n_kv_heads, self.head_dim
        q = self.w_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.w_k(x).view(B, T, G, dh).transpose(1, 2)
        v = self.w_v(x).view(B, T, G, dh).transpose(1, 2)
        return apply_rope(q, freqs), apply_rope(k, freqs), v

    def _queries_keys(self, x, freqs):
        q, k, _ = self._project(x, freqs)
        return q, k

    def _weights(self, q, k, allowed):
        """Softmax weights (B, H, n, S) of queries q against keys k (B, G, S, dh)."""
        scores = self._logits(q, k).masked_fill(~allowed, float("-inf"))
        return torch.softmax(scores, dim=-1)

    def _fused(self, q, k, v, allowed):
        """Fused kernel on (B, H, n, dh) queries and (B, G, S, dh) keys, grouped as in GroupedAttention."""
        B, H, n, dh = q.shape
        G, share = self.n_kv_heads, self.share
        qg = q.reshape(B, G, share, n, dh).reshape(B, G, share * n, dh)
        if allowed is None:
            out = F.scaled_dot_product_attention(qg, k, v)
        else:
            out = F.scaled_dot_product_attention(qg, k, v, attn_mask=allowed.repeat(share, 1))
        return out.reshape(B, G, share, n, dh).reshape(B, H, n, dh)

    # --- full forward (no cache) -----------------------------------------------------------

    def _full_weights(self, q, k):
        T = q.shape[2]
        pos = torch.arange(T, device=q.device)
        return self._weights(q, k, _window_mask(pos, pos, self.window))

    def attention_weights(self, x, freqs):
        q, k, _ = self._project(x, freqs)
        return self._full_weights(q, k)

    def _full_ref(self, q, k, v):
        return self._full_weights(q, k) @ self._expand(v)

    def _chunked(self, q, k, v):
        W, T, dev = self.window, q.shape[2], q.device
        k, v = self._expand(k), self._expand(v)
        pieces = []
        for a in range(0, T, W):
            b = min(a + W, T)
            s = max(0, a - W)
            allowed = _window_mask(torch.arange(a, b, device=dev), torch.arange(s, b, device=dev), W)
            pieces.append(
                F.scaled_dot_product_attention(
                    q[:, :, a:b], k[:, :, s:b], v[:, :, s:b], attn_mask=allowed
                )
            )
        return torch.cat(pieces, dim=2)

    def _flex(self, x, q, k, v):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        W, T, dev = self.window, q.shape[2], x.device
        key = (T, str(dev))
        block_mask = self._block_masks.get(key)
        if block_mask is None:

            def mask_mod(b, h, q_idx, kv_idx):
                return (kv_idx <= q_idx) & (q_idx - kv_idx < W)

            block_mask = create_block_mask(mask_mod, None, None, T, T, device=dev)
            self._block_masks[key] = block_mask
        fn = _compiled_flex_attention() if x.is_cuda else flex_attention
        return fn(q, self._expand(k), self._expand(v), block_mask=block_mask)

    # --- decode through a cache ---------------------------------------------------------------

    def _decode(self, q, k, v, cache, mode):
        W, T, dev = self.window, q.shape[2], q.device
        pieces, c = [], cache
        for a in range(0, T, W):
            b = min(a + W, T)
            n = b - a
            L = c.length
            qc, kc, vc = q[:, :, a:b], k[:, :, a:b], v[:, :, a:b]
            s = max(0, L - W + 1)  # the earliest key the chunk's first query may see
            if n == 1:
                # The new cache already holds exactly the keys this token may see (the
                # last min(L + 1, W) tokens, itself included, in order): no copy.
                c = c.append(kc, vc)
                keys, vals = c.k, c.v
            else:
                # Prefill chunk: the cache holds W tokens but up to 2W - 1 are needed.
                held_k, held_v = c.k, c.v
                c = c.append(kc, vc)  # validates the batch size before anything is concatenated
                if held_k is None:
                    keys, vals = kc, vc
                else:
                    keys = torch.cat((held_k, kc), dim=2)
                    vals = torch.cat((held_v, vc), dim=2)
                keys, vals = keys[:, :, s:], vals[:, :, s:]
            q_pos = torch.arange(L, L + n, device=dev)
            k_pos = torch.arange(s, L + n, device=dev)
            if mode == "ref":
                pieces.append(self._weights(qc, keys, _window_mask(q_pos, k_pos, W)) @ self._expand(vals))
            elif n == 1:
                # Every key held is visible to the single query: no mask.
                pieces.append(self._fused(qc, keys, vals, None))
            else:
                pieces.append(self._fused(qc, keys, vals, _window_mask(q_pos, k_pos, W)))
        return torch.cat(pieces, dim=2), c

    # --- module interface ------------------------------------------------------------------

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if cache is not None and x.requires_grad:
            raise RuntimeError(
                "decode through a KV cache is inference-only: run it under torch.no_grad()"
            )
        B, T, _ = x.shape
        H, dh = self.n_heads, self.head_dim
        q, k, v = self._project(x, freqs)

        if cache is None:
            if mode == "ref":
                y = self._full_ref(q, k, v)
            elif self.opt_path == "chunked":
                y = self._chunked(q, k, v)
            else:
                y = self._flex(x, q, k, v)
            new_cache = None
        else:
            y, new_cache = self._decode(q, k, v, cache, mode)

        y = y.transpose(1, 2).reshape(B, T, H * dh)
        return self.w_o(y), new_cache
