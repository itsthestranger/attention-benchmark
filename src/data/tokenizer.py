"""Byte-level BPE tokenizer and corpus readers (plan §6).

The tokenizer is trained once, then its file and hash are committed. Nothing in
this module retrains a tokenizer that already exists; that decision belongs to
the caller (scripts/prepare_data.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"

_READ_CHUNK = 1 << 20


def iter_documents(paths: Iterable[str | Path], text_column: str = "text") -> Iterator[str]:
    """Yield non-blank document texts: files in the given order, rows in file order.

    An unsupported extension is an error, raised before any document is read.
    """
    paths = [Path(p) for p in paths]
    for path in paths:
        if path.suffix.lower() not in (".parquet", ".jsonl"):
            raise ValueError(f"unsupported input extension: {path}")
    return _iter_documents(paths, text_column)


def _iter_documents(paths: list[Path], text_column: str) -> Iterator[str]:
    for path in paths:
        if path.suffix.lower() == ".parquet":
            texts = _iter_parquet(path, text_column)
        else:
            texts = _iter_jsonl(path, text_column)
        for text in texts:
            if text is None or not text.strip():
                continue
            yield text


def _iter_parquet(path: Path, text_column: str) -> Iterator[str | None]:
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as pf:
        if text_column not in pf.schema_arrow.names:
            raise ValueError(f"{path}: no column named {text_column!r}")
        for batch in pf.iter_batches(batch_size=1024, columns=[text_column]):
            yield from batch.column(0).to_pylist()


def _iter_jsonl(path: Path, text_column: str) -> Iterator[str | None]:
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if text_column not in row:
                raise ValueError(f"{path}:{number}: no field {text_column!r}")
            yield row[text_column]


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    """Train a byte-level BPE tokenizer. EOS gets id 0. Same texts, same order,
    same file bytes (checked by the tests)."""
    tok = Tokenizer(models.BPE())  # no unknown token: every byte is in the alphabet
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    if eos_id(tok) != 0:
        raise RuntimeError("EOS token did not get id 0")
    return tok


def save_tokenizer(tok: Tokenizer, path: str | Path) -> None:
    tok.save(str(path))


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_READ_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def encode_document(tok: Tokenizer, text: str) -> list[int]:
    """Token ids of one document, without EOS. Any literal EOS string in the text
    is removed first, so it can never produce the EOS id."""
    while EOS_TOKEN in text:
        text = text.replace(EOS_TOKEN, "")
    if not text:
        return []
    return tok.encode(text, add_special_tokens=False).ids


def eos_id(tok: Tokenizer) -> int:
    idx = tok.token_to_id(EOS_TOKEN)
    if idx is None:
        raise ValueError(f"tokenizer has no {EOS_TOKEN!r} token")
    return idx
