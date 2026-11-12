"""SYNTHETIC token splits with chosen document boundaries. Nothing here comes from a corpus.

The evaluation tests need splits whose document boundaries they control. The
token values are random (or constant, where the test wants every target to be
the same class), and the document starts are either drawn at random or written
by hand in the test. A document's last token is EOS (token 0), as in packing.
"""

from __future__ import annotations

import numpy as np

from src.data.loader import TokenSplit

EOS = 0


def document_starts(n_tokens: int, mean_doc_len: float, rng: np.random.Generator) -> np.ndarray:
    """SYNTHETIC document starts over n_tokens tokens, geometric lengths with the given mean.
    Returns int64 with a final sentinel equal to n_tokens. Starts are strictly increasing."""
    cuts = np.flatnonzero(rng.random(n_tokens) < 1.0 / mean_doc_len)
    cuts = cuts[cuts > 0]
    return np.concatenate([[0], cuts, [n_tokens]]).astype(np.int64)


def synthetic_split(doc_starts, vocab: int, rng: np.random.Generator) -> TokenSplit:
    """SYNTHETIC split: random tokens in [1, vocab), EOS at the last token of each document."""
    starts = np.asarray(doc_starts, dtype=np.int64)
    n = int(starts[-1])
    tokens = rng.integers(1, vocab, size=n).astype(np.uint16)
    tokens[starts[1:] - 1] = EOS
    return TokenSplit(tokens, starts)


def constant_split(doc_starts, token: int) -> TokenSplit:
    """SYNTHETIC split in which every token equals `token`, with the given document starts."""
    starts = np.asarray(doc_starts, dtype=np.int64)
    return TokenSplit(np.full(int(starts[-1]), token, dtype=np.uint16), starts)
