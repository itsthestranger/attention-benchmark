"""Packing documents into the on-disk dataset format (plan §6).

Each document is its token ids followed by one EOS id. Splits are filled in the
order val, test, train, by whole documents, and each split is truncated to its
exact target. See the module tests for the invariants.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import tokenizers

from src.data.tokenizer import encode_document, eos_id, file_sha256, save_tokenizer

SPLIT_ORDER = ("val", "test", "train")
_UINT16_MAX = 65535
_CHUNK = 1 << 20


def _split_files(name: str) -> tuple[str, str]:
    return f"{name}.bin", f"{name}.docs.bin"


def pack_dataset(
    texts: Iterable[str],
    tok,
    out_dir: str | Path,
    *,
    targets: dict[str, int],
    context_length: int,
    sources: list[dict] | None = None,
    allow_short_train: bool = False,
    tokenizer_file: str | Path | None = None,
    tokenizer_sources: list[dict] | None = None,
) -> dict:
    """Write tokenizer.json, {split}.bin, {split}.docs.bin and meta.json to out_dir.

    ``tokenizer_file``, if given, is copied byte for byte as tokenizer.json (it
    must be the file ``tok`` was loaded from). Otherwise ``tok`` is saved, which
    is deterministic for a tokenizer trained by ``train_tokenizer``.
    """
    vocab_size = tok.get_vocab_size()
    if vocab_size > 65536:
        raise ValueError(f"vocabulary has {vocab_size} entries; uint16 holds at most 65536")
    if set(targets) != set(SPLIT_ORDER):
        raise ValueError(f"targets must name exactly {SPLIT_ORDER}, got {sorted(targets)}")
    for name, target in targets.items():
        if not isinstance(target, int) or target <= 0:
            raise ValueError(f"target for {name} must be a positive integer, got {target!r}")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    eos = eos_id(tok)
    if eos > _UINT16_MAX:
        raise ValueError(f"EOS id {eos} does not fit uint16")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"
    if tokenizer_file is not None:
        shutil.copyfile(tokenizer_file, tok_path)
    else:
        save_tokenizer(tok, tok_path)

    docs = iter(texts)
    splits_meta: dict[str, dict] = {}
    for name in SPLIT_ORDER:
        target = targets[name]
        bin_name, docs_name = _split_files(name)
        starts: list[int] = []
        written = 0
        with open(out / bin_name, "wb") as f:
            while written < target:
                text = next(docs, None)
                if text is None:
                    break
                ids = encode_document(tok, text)
                if not ids:
                    continue
                ids.append(eos)
                if max(ids) > _UINT16_MAX:
                    raise ValueError("token id does not fit uint16")
                take = ids[: target - written]  # the last document may be cut here
                starts.append(written)
                f.write(np.asarray(take, dtype="<u2").tobytes())
                written += len(take)
        if written < target and not (name == "train" and allow_short_train):
            raise ValueError(
                f"split {name!r} needs {target} tokens but the input ran out at {written}"
            )
        with open(out / docs_name, "wb") as f:
            f.write(np.asarray(starts + [written], dtype="<i8").tobytes())
        splits_meta[name] = {
            "n_tokens": written,
            "n_docs": len(starts),
            "sha256": file_sha256(out / bin_name),
            "docs_sha256": file_sha256(out / docs_name),
        }

    dataset_hash = dataset_sha256(out)
    meta = {
        "vocab_size": vocab_size,
        "eos_id": eos,
        "context_length": context_length,
        "tokenizer_sha256": file_sha256(tok_path),
        "dataset_sha256": dataset_hash,
        "splits": splits_meta,
        "sources": list(sources or []),
        "tokenizer_sources": list(tokenizer_sources or []),
        "versions": {
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return meta


def dataset_sha256(directory: str | Path) -> str:
    """SHA-256 over the bytes of train.bin, val.bin, test.bin, in that order."""
    return hash_bins(Path(directory))[1]


def hash_bins(directory: Path) -> tuple[dict[str, str], str]:
    """Per-file SHA-256 of train/val/test .bin, and the combined dataset hash,
    computed in one streaming pass."""
    combined = hashlib.sha256()
    per_file: dict[str, str] = {}
    for name in ("train", "val", "test"):
        h = hashlib.sha256()
        with open(directory / f"{name}.bin", "rb") as f:
            while chunk := f.read(_CHUNK):
                h.update(chunk)
                combined.update(chunk)
        per_file[f"{name}.bin"] = h.hexdigest()
    return per_file, combined.hexdigest()
