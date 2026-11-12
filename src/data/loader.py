"""Reading a prepared dataset, and the token order every run shares (plan §6).

Two properties carry the comparison and are tested directly:

* The train stream is a pure function of (dataset, context_length,
  sequences_per_step, data_seed). It reads no model, variant or init seed, and
  it uses its own torch Generator, so the global RNG is never touched.
* open_dataset refuses data whose hashes differ from the expected ones.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.data.packing import _hash_bins
from src.data.tokenizer import file_sha256

SPLITS = ("train", "val", "test")


class DataError(RuntimeError):
    """The dataset on disk is not the one the caller expects, or is malformed."""


class TokenSplit:
    """One split, read-only. ``tokens`` is uint16; ``doc_starts`` is int64 with
    n_docs + 1 entries, the last being len(tokens)."""

    def __init__(self, tokens: np.memmap, doc_starts: np.ndarray):
        self.tokens = tokens
        self.doc_starts = doc_starts

    def n_windows(self, context_length: int) -> int:
        return max(0, (len(self.tokens) - 1) // context_length)

    def window(self, index: int, context_length: int) -> tuple[np.ndarray, np.ndarray]:
        n = self.n_windows(context_length)
        if not 0 <= index < n:
            raise IndexError(f"window {index} out of range [0, {n})")
        i = index * context_length
        x = np.array(self.tokens[i : i + context_length], dtype=np.int64)
        y = np.array(self.tokens[i + 1 : i + context_length + 1], dtype=np.int64)
        return x, y


def _read_meta(root: Path) -> dict:
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        raise DataError(f"meta.json missing in {root}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for name in SPLITS:
            if not all(k in meta["splits"][name] for k in ("n_tokens", "n_docs", "sha256", "docs_sha256")):
                raise KeyError(name)
        for key in ("tokenizer_sha256", "dataset_sha256"):
            if key not in meta:
                raise KeyError(key)
    except (ValueError, KeyError, TypeError) as exc:
        raise DataError(f"meta.json is malformed: {exc!r}") from exc
    return meta


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise DataError(f"missing file: {path.name}")


def open_dataset(
    directory: str | Path,
    *,
    expected_dataset_sha256: str | None,
    expected_tokenizer_sha256: str | None,
    verify: bool = True,
) -> dict[str, TokenSplit]:
    """Open a prepared dataset and return its splits.

    Structure is always checked: files present, lengths consistent with meta.json,
    offsets valid. With ``verify=True`` every file is also hashed against
    meta.json. With ``verify=False`` the hashes recorded in meta.json are trusted
    and a flipped token inside a file is NOT detected. An expected hash, when
    given, is always compared.
    """
    root = Path(directory)
    meta = _read_meta(root)
    _require_file(root / "tokenizer.json")
    for name in SPLITS:
        _require_file(root / f"{name}.bin")
        _require_file(root / f"{name}.docs.bin")

    splits: dict[str, TokenSplit] = {}
    for name in SPLITS:
        sm = meta["splits"][name]
        bin_path = root / f"{name}.bin"
        docs_path = root / f"{name}.docs.bin"
        size = bin_path.stat().st_size
        if size % 2:
            raise DataError(f"{bin_path.name}: length {size} is not a multiple of 2")
        n_tokens = size // 2
        if n_tokens != sm["n_tokens"]:
            raise DataError(f"{bin_path.name}: {n_tokens} tokens, meta.json says {sm['n_tokens']}")
        if docs_path.stat().st_size % 8:
            raise DataError(f"{docs_path.name}: length is not a multiple of 8")
        starts = np.fromfile(docs_path, dtype="<i8").astype(np.int64)
        if len(starts) != sm["n_docs"] + 1:
            raise DataError(f"{docs_path.name}: {len(starts)} offsets, expected {sm['n_docs'] + 1}")
        if starts[0] != 0:
            raise DataError(f"{docs_path.name}: first offset is {starts[0]}, not 0")
        if np.any(np.diff(starts) <= 0):
            raise DataError(f"{docs_path.name}: offsets are not strictly increasing")
        if starts[-1] != n_tokens:
            raise DataError(f"{docs_path.name}: last offset {starts[-1]} != {n_tokens} tokens")
        if n_tokens:
            tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        else:
            tokens = np.zeros(0, dtype=np.uint16)
        splits[name] = TokenSplit(tokens, starts)

    if verify:
        bin_hashes, dataset_hash = _hash_bins(root)
        for name in SPLITS:
            sm = meta["splits"][name]
            if bin_hashes[f"{name}.bin"] != sm["sha256"]:
                raise DataError(f"{name}.bin does not match meta.json")
            if file_sha256(root / f"{name}.docs.bin") != sm["docs_sha256"]:
                raise DataError(f"{name}.docs.bin does not match meta.json")
        if file_sha256(root / "tokenizer.json") != meta["tokenizer_sha256"]:
            raise DataError("tokenizer.json does not match meta.json")
        if dataset_hash != meta["dataset_sha256"]:
            raise DataError("dataset hash over train.bin, val.bin, test.bin does not match meta.json")
    else:
        dataset_hash = meta["dataset_sha256"]
    tokenizer_hash = meta["tokenizer_sha256"]

    if expected_dataset_sha256 is not None and expected_dataset_sha256 != dataset_hash:
        raise DataError(
            f"dataset hash {dataset_hash} does not match expected {expected_dataset_sha256}"
        )
    if expected_tokenizer_sha256 is not None and expected_tokenizer_sha256 != tokenizer_hash:
        raise DataError(
            f"tokenizer hash {tokenizer_hash} does not match expected {expected_tokenizer_sha256}"
        )
    return splits


def train_order(n_windows: int, data_seed: int) -> torch.Tensor:
    """Permutation of range(n_windows) from a private generator seeded by data_seed.
    Does not read or change the global RNG state."""
    g = torch.Generator(device="cpu")
    g.manual_seed(data_seed)
    return torch.randperm(n_windows, generator=g)


def order_sha256(order: torch.Tensor) -> str:
    return hashlib.sha256(order.detach().cpu().numpy().astype("<i8").tobytes()).hexdigest()


class TrainStream:
    """Batches of the train split in a fixed order. Stateless: batch(k) depends
    only on k and the constructor arguments."""

    def __init__(self, split: TokenSplit, *, context_length: int, sequences_per_step: int,
                 data_seed: int):
        if sequences_per_step <= 0:
            raise ValueError("sequences_per_step must be positive")
        self._split = split
        self._context_length = context_length
        self._batch = sequences_per_step
        self.order = train_order(split.n_windows(context_length), data_seed)
        self.order_sha256 = order_sha256(self.order)
        self.max_steps = len(self.order) // sequences_per_step

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= step < self.max_steps:
            raise IndexError(f"step {step} out of range [0, {self.max_steps})")
        b = self._batch
        indices = self.order[step * b : (step + 1) * b].tolist()
        pairs = [self._split.window(i, self._context_length) for i in indices]
        x = np.stack([p[0] for p in pairs])
        y = np.stack([p[1] for p in pairs])
        return torch.from_numpy(x), torch.from_numpy(y)


def eval_windows(
    split: TokenSplit, context_length: int, batch_size: int
) -> Iterator[tuple[int, torch.Tensor, torch.Tensor]]:
    """Every window once, in index order. Yields (first_window_index, x, y)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    t = context_length
    n = split.n_windows(t)
    for start in range(0, n, batch_size):
        count = min(batch_size, n - start)
        block = np.asarray(split.tokens[start * t : (start + count) * t + 1], dtype=np.int64)
        x = np.ascontiguousarray(block[: count * t].reshape(count, t))
        y = np.ascontiguousarray(block[1 : count * t + 1].reshape(count, t))
        yield start, torch.from_numpy(x), torch.from_numpy(y)
