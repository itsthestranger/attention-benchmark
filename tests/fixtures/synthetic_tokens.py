"""SYNTHETIC token data for trainer tests. Nothing here comes from a corpus.

`rule_split` builds a token stream in which every token is a fixed function of
the one before it, so the next token is perfectly predictable. A model that
trains at all drives its loss towards zero and predicts every token correctly.
"""

from __future__ import annotations

import numpy as np

from src.data.loader import TokenSplit


def rule_split(n_tokens: int, vocab: int, seed: int = 0) -> TokenSplit:
    """A split of `n_tokens` tokens with tokens[i + 1] = rule[tokens[i]]."""
    rng = np.random.default_rng(seed)
    rule = rng.permutation(vocab)
    tokens = np.empty(n_tokens, dtype=np.uint16)
    tokens[0] = 0
    for i in range(1, n_tokens):
        tokens[i] = rule[tokens[i - 1]]
    return TokenSplit(tokens, np.array([0, n_tokens], dtype=np.int64))
