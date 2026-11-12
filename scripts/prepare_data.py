"""Tokenize a corpus once and freeze it (plan §6).

Run from the repository root:

    python -m scripts.prepare_data --inputs shard1.parquet shard2.parquet \
        --tokenizer-inputs shard1.parquet --out data/prepared

Pass --tokenizer PATH to reuse an existing tokenizer.json instead of training one.
This script never edits configs/data.yaml; it prints the two lines to paste there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.data.packing import pack_dataset
from src.data.tokenizer import file_sha256, iter_documents, load_tokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "data.yaml"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="parquet/jsonl files, in the order to use")
    p.add_argument("--tokenizer-inputs", nargs="+", default=None,
                   help="files to train the tokenizer on (ignored with --tokenizer)")
    p.add_argument("--tokenizer", default=None, help="existing tokenizer.json to reuse (never retrained)")
    p.add_argument("--out", required=True, help="output directory; must not already hold a meta.json")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--train-tokens", type=int, default=None)
    p.add_argument("--val-tokens", type=int, default=None)
    p.add_argument("--test-tokens", type=int, default=None)
    p.add_argument("--allow-short-train", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out = Path(args.out)
    if (out / "meta.json").exists():
        print(f"refusing to overwrite an existing dataset: {out / 'meta.json'}", file=sys.stderr)
        return 2
    for path in args.inputs:
        if not Path(path).is_file():
            print(f"input not found: {path}", file=sys.stderr)
            return 2

    text_column = cfg["text_column"]
    context_length = args.context_length or cfg["context_length"]
    targets = {
        "train": args.train_tokens or cfg["splits"]["train"],
        "val": args.val_tokens or cfg["splits"]["val"],
        "test": args.test_tokens or cfg["splits"]["test"],
    }

    if args.tokenizer is not None:
        if not Path(args.tokenizer).is_file():
            print(f"tokenizer not found: {args.tokenizer}", file=sys.stderr)
            return 2
        tok = load_tokenizer(args.tokenizer)
        tokenizer_file = args.tokenizer
        print(f"reusing tokenizer {args.tokenizer}")
    else:
        if not args.tokenizer_inputs:
            print("give --tokenizer-inputs to train a tokenizer, or --tokenizer to reuse one", file=sys.stderr)
            return 2
        vocab_size = args.vocab_size or cfg["tokenizer"]["vocab_size"]
        tok = train_tokenizer(
            iter_documents(args.tokenizer_inputs, text_column), vocab_size
        )
        tokenizer_file = None
        print(f"trained tokenizer, vocab {tok.get_vocab_size()} (requested {vocab_size})")

    sources = [{"path": str(Path(p)), "sha256": file_sha256(p)} for p in args.inputs]
    meta = pack_dataset(
        iter_documents(args.inputs, text_column),
        tok,
        out,
        targets=targets,
        context_length=context_length,
        sources=sources,
        allow_short_train=args.allow_short_train,
        tokenizer_file=tokenizer_file,
    )

    print(f"dataset written to {out}")
    print(f"tokens: " + ", ".join(f"{k}={v['n_tokens']}" for k, v in meta["splits"].items()))
    print(f"tokenizer_sha256: {meta['tokenizer_sha256']}")
    print(f"dataset_sha256: {meta['dataset_sha256']}")
    print("paste into configs/data.yaml (the script does not edit it):")
    print(f"  sha256: {meta['tokenizer_sha256']}        # under tokenizer:")
    print(f"dataset_sha256: {meta['dataset_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
