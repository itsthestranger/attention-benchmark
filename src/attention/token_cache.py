"""Growable per-token storage shared by the attention caches (grouped, MLA).

A storage holds one or more tensors that all carry the token axis at the same
position. Appending writes new tokens in place when the storage is not shared
with a later write, and otherwise copies into a new allocation. Growth is
geometric unless a capacity hint allocates room once.
"""

from __future__ import annotations

import torch


class TokenStorage:
    """One allocation, possibly shared by several cache objects.

    Slots below `filled` are final: no cache may overwrite them.
    """

    __slots__ = ("tensors", "filled")

    def __init__(self, tensors: list[torch.Tensor], filled: int):
        self.tensors = tensors
        self.filled = filled


def _allocate(like: torch.Tensor, capacity: int, axis: int) -> torch.Tensor:
    shape = list(like.shape)
    shape[axis] = capacity
    return torch.empty(shape, dtype=like.dtype, device=like.device)


def append_tokens(
    storage: TokenStorage | None,
    length: int,
    capacity_hint: int,
    news: tuple[torch.Tensor, ...],
    axis: int,
) -> tuple[TokenStorage, int]:
    """Write the new tokens after the first `length` tokens of `storage`.

    `news` holds one tensor per storage tensor, all with the same token count
    along `axis`, and equal shapes in every other dimension (the caller checks
    that against the storage). Returns the storage now holding the tokens and
    the new length. The storage passed in is never written past `length` unless
    nobody else has written there.
    """
    T = news[0].shape[axis]
    L = length
    if storage is None:
        capacity = max(capacity_hint, T)
        storage = TokenStorage([_allocate(n, capacity, axis) for n in news], 0)
    elif not (storage.filled == L and L + T <= storage.tensors[0].shape[axis]):
        cap = storage.tensors[0].shape[axis]
        new_cap = cap if L + T <= cap else max(L + T, 2 * cap)
        bufs = [_allocate(n, new_cap, axis) for n in news]
        with torch.no_grad():
            for buf, old in zip(bufs, storage.tensors):
                buf.narrow(axis, 0, L).copy_(old.narrow(axis, 0, L))
        storage = TokenStorage(bufs, L)
    with torch.no_grad():
        for buf, n in zip(storage.tensors, news):
            buf.narrow(axis, L, T).copy_(n)
    storage.filled = L + T
    return storage, L + T
