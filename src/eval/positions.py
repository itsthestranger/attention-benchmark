"""Context values and context buckets for the evaluation (plan §8.4, prereg `context_use`).

For a target token at global offset g in window w (window start ws = w * T), the
context value is the number of tokens of the same document that precede it inside
the window:

    c = g - max(s, ws)

where s is the start offset of the document containing g. A document may start
anywhere, and may span several windows; there is no cross-document masking.
"""

from __future__ import annotations

import numpy as np


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def validate_buckets(buckets) -> list[tuple[int, int]]:
    """Check a bucket list and return it as a list of (low, high) int pairs.

    Buckets are half-open [low, high) except the last, which is closed at its
    top. They must be non-empty, start at 0, and be contiguous and increasing.
    """
    try:
        pairs = [tuple(b) for b in buckets]
    except TypeError as exc:
        raise ValueError(f"buckets must be a sequence of (low, high) pairs: {exc}") from exc
    if not pairs:
        raise ValueError("buckets must not be empty")
    out: list[tuple[int, int]] = []
    for i, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"bucket {i} must be a (low, high) pair, got {pair!r}")
        low = _check_int(f"bucket {i} low", pair[0], -(10**18))
        high = _check_int(f"bucket {i} high", pair[1], -(10**18))
        out.append((low, high))
    if out[0][0] != 0:
        raise ValueError(f"the first bucket must start at 0, starts at {out[0][0]}")
    for i, (low, high) in enumerate(out):
        if low >= high:
            raise ValueError(f"bucket {i} = [{low}, {high}) is empty or reversed")
        if i and low != out[i - 1][1]:
            raise ValueError(
                f"bucket {i} starts at {low} but the previous bucket ends at {out[i - 1][1]}: "
                "buckets must be contiguous"
            )
    return out


def context_values(doc_starts, first_window: int, n_windows: int, context_length: int) -> np.ndarray:
    """int64 array (n_windows, context_length): c for every target of windows
    first_window .. first_window + n_windows - 1. Vectorised (np.searchsorted), no Python loop over tokens."""
    T = _check_int("context_length", context_length, 1)
    first = _check_int("first_window", first_window, 0)
    n = _check_int("n_windows", n_windows, 0)
    starts = np.asarray(doc_starts, dtype=np.int64)
    if starts.ndim != 1 or len(starts) < 2 or starts[0] != 0 or np.any(np.diff(starts) <= 0):
        raise ValueError(
            "doc_starts must be 1-D, start at 0, be strictly increasing, and end with a sentinel"
        )
    begins = starts[:-1]  # the final sentinel is not a document start

    ws = (np.arange(first, first + n, dtype=np.int64) * T)[:, None]  # (n, 1) window starts
    g = ws + np.arange(1, T + 1, dtype=np.int64)[None, :]  # (n, T) global offsets of targets
    if n and g[-1, -1] >= starts[-1]:
        raise ValueError(
            f"window {first + n - 1} reads past the end of the split ({starts[-1]} tokens)"
        )
    doc = np.searchsorted(begins, g, side="right") - 1
    s = begins[doc]
    return (g - np.maximum(s, ws)).astype(np.int64)


def bucket_index(c: np.ndarray, buckets) -> np.ndarray:
    """Index of the bucket of each c (same shape). Raises ValueError if any c is in no bucket."""
    pairs = validate_buckets(buckets)
    edges = np.array([pairs[0][0]] + [high for _, high in pairs], dtype=np.int64)
    c = np.asarray(c)
    if c.size:
        bad = (c < edges[0]) | (c > edges[-1])
        if bad.any():
            raise ValueError(
                f"context value {c[bad].flat[0]} lies in no bucket; buckets cover [{edges[0]}, {edges[-1]}]"
            )
    idx = np.searchsorted(edges, c, side="right") - 1
    # c equal to the top edge falls past the last edge; the last bucket is closed at its top.
    return np.minimum(idx, len(pairs) - 1).astype(np.int64)
