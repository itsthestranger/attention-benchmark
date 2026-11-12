"""T-13: data pipeline. Everything runs on synthetic text generated here.

No skips and no tolerances: every check is exact equality.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from scripts.prepare_data import main
from src.data.loader import (
    DataError,
    TrainStream,
    eval_windows,
    open_dataset,
    order_sha256,
    train_order,
)
from src.data.packing import dataset_sha256, pack_dataset
from src.data.tokenizer import (
    EOS_TOKEN,
    encode_document,
    eos_id,
    file_sha256,
    iter_documents,
    save_tokenizer,
    train_tokenizer,
)

ROOT = Path(__file__).resolve().parent.parent

CONTEXT = 16
VOCAB = 300
TARGETS = {"val": 300, "test": 300, "train": 4000}
SPLITS = ("train", "val", "test")

WORDS = [
    "alpha", "beta", "gamma", "delta", "the", "of", "data", "model", "token",
    "naïve", "café", "日本語", "Ωμέγα", "emoji🙂", "x", "yy", "zzz", "tab\there",
]


def _doc(rng: random.Random) -> str:
    text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40)))
    if rng.random() < 0.2:
        text += "\n"
    return text


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("corpus")
    rng = random.Random(20261112)
    rows: list[str] = []
    for i in range(400):
        if i % 50 == 7:
            rows.append("")
        elif i % 50 == 19:
            rows.append(" \n\t  ")
        else:
            rows.append(_doc(rng))

    jsonl = base / "shard.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"id": i, "text": row}, ensure_ascii=False) + "\n")
    parquet = base / "shard.parquet"
    pq.write_table(pa.table({"text": pa.array(rows, pa.string())}), parquet, row_group_size=50)

    texts = [r for r in rows if r.strip()]
    tok = train_tokenizer(iter_documents([jsonl]), VOCAB)
    tok_path = base / "tokenizer.json"
    save_tokenizer(tok, tok_path)

    dataset = base / "dataset"
    sources = [{"path": str(jsonl), "sha256": file_sha256(jsonl)}]
    meta = pack_dataset(iter_documents([jsonl]), tok, dataset, targets=TARGETS,
                        context_length=CONTEXT, sources=sources)
    splits = open_dataset(dataset, expected_dataset_sha256=None, expected_tokenizer_sha256=None)
    return types.SimpleNamespace(
        base=base, rows=rows, texts=texts, jsonl=jsonl, parquet=parquet,
        tok=tok, tok_path=tok_path, dataset=dataset, meta=meta, splits=splits,
    )


def _reconstruct(corpus) -> dict[str, tuple[list[int], int | None]]:
    """Walk val, test, train in order and decode every document, matching it to the
    source text at one running cursor. Returns, per split, the source indices it
    holds and the index of its truncated last document (None if not truncated)."""
    tok = corpus.tok
    cursor = 0
    result = {}
    for name in ("val", "test", "train"):
        arr = np.fromfile(corpus.dataset / f"{name}.bin", dtype=np.uint16)
        starts = np.fromfile(corpus.dataset / f"{name}.docs.bin", dtype="<i8")
        indices: list[int] = []
        truncated = None
        n_docs = len(starts) - 1
        for k in range(n_docs):
            ids = arr[starts[k]:starts[k + 1]].tolist()
            if ids[-1] == eos_id(tok):
                body = ids[:-1]
                assert eos_id(tok) not in body
                assert tok.decode(body) == corpus.texts[cursor]
            else:
                assert k == n_docs - 1, "only the last document of a split may be truncated"
                truncated = cursor
                full = encode_document(tok, corpus.texts[cursor])
                assert ids == full[: len(ids)]
                assert len(ids) <= len(full)
            indices.append(cursor)
            cursor += 1
        result[name] = (indices, truncated)
    return result


# --- 1. Readers ------------------------------------------------------------

def test_parquet_and_jsonl_yield_same_documents(corpus):
    from_parquet = list(iter_documents([corpus.parquet]))
    from_jsonl = list(iter_documents([corpus.jsonl]))
    assert from_parquet == from_jsonl == corpus.texts
    assert all(text.strip() for text in from_parquet)
    assert list(iter_documents([corpus.jsonl, corpus.parquet])) == corpus.texts * 2


def test_blank_documents_are_skipped_in_both_formats(corpus):
    assert len(corpus.rows) > len(corpus.texts)  # the fixture does contain blank rows
    assert "" in corpus.rows and " \n\t  " in corpus.rows
    assert "" not in list(iter_documents([corpus.jsonl]))


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "shard.csv"
    bad.write_text("text\nhello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_documents([bad]))


# --- 2. Tokenizer ----------------------------------------------------------

def test_eos_has_id_zero(corpus):
    assert eos_id(corpus.tok) == 0
    assert corpus.tok.token_to_id(EOS_TOKEN) == 0


def test_round_trip_including_non_ascii(corpus):
    extra = ["naïve café — 日本語 🙂 Ωμέγα", "tabs\tand  double  spaces\n", "ünïcödé ✓ 𝔘𝔫𝔦"]
    for text in corpus.texts[:60] + extra:
        assert corpus.tok.decode(encode_document(corpus.tok, text)) == text


def test_training_twice_gives_identical_bytes(corpus, tmp_path):
    first = train_tokenizer(iter(corpus.texts), VOCAB)
    second = train_tokenizer(iter(corpus.texts), VOCAB)
    save_tokenizer(first, tmp_path / "a.json")
    save_tokenizer(second, tmp_path / "b.json")
    assert file_sha256(tmp_path / "a.json") == file_sha256(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_literal_eos_string_never_produces_eos_id(corpus):
    tok = corpus.tok
    for text in ["a <|endoftext|> b", "<|endoftext|>", "x<|endoftext<|endoftext|>|>y"]:
        assert eos_id(tok) not in encode_document(tok, text)
    assert encode_document(tok, "a<|endoftext|>b") == tok.encode("ab", add_special_tokens=False).ids
    assert encode_document(tok, "<|endoftext|>") == []


# --- 3. Packing ------------------------------------------------------------

def test_bin_lengths_dtype_and_range(corpus):
    vocab = corpus.tok.get_vocab_size()
    for name in SPLITS:
        arr = np.fromfile(corpus.dataset / f"{name}.bin", dtype=np.uint16)
        assert arr.dtype == np.uint16
        assert len(arr) == TARGETS[name]
        assert int(arr.max()) < vocab


def test_offsets_are_well_formed(corpus):
    for name in SPLITS:
        arr = np.fromfile(corpus.dataset / f"{name}.bin", dtype=np.uint16)
        starts = np.fromfile(corpus.dataset / f"{name}.docs.bin", dtype="<i8")
        assert starts[0] == 0
        assert np.all(np.diff(starts) > 0)
        assert starts[-1] == len(arr)
        assert len(starts) == corpus.meta["splits"][name]["n_docs"] + 1


def test_eos_placement_and_counts(corpus):
    eos = eos_id(corpus.tok)
    for name in SPLITS:
        arr = np.fromfile(corpus.dataset / f"{name}.bin", dtype=np.uint16)
        starts = np.fromfile(corpus.dataset / f"{name}.docs.bin", dtype="<i8")
        n_docs = len(starts) - 1
        ends = arr[starts[1:] - 1]
        # Every document but possibly the last ends with EOS.
        assert np.all(ends[:-1] == eos)
        # The token before every later start is EOS.
        assert np.all(arr[starts[1:-1] - 1] == eos)
        # The EOS count equals the number of complete documents.
        truncated = int(arr[-1] != eos)
        assert int(np.sum(arr == eos)) == n_docs - truncated


def test_decoding_reproduces_source_text(corpus):
    recon = _reconstruct(corpus)
    total = sum(len(indices) for indices, _ in recon.values())
    assert total == sum(
        len(np.fromfile(corpus.dataset / f"{n}.docs.bin", dtype="<i8")) - 1 for n in SPLITS
    )


# --- 4. Splits are disjoint by document ------------------------------------

def test_splits_are_disjoint_ordered_and_contiguous(corpus):
    recon = _reconstruct(corpus)
    val, test, train = (recon[n][0] for n in ("val", "test", "train"))
    assert not set(val) & set(test)
    assert not set(test) & set(train)
    assert not set(val) & set(train)
    assert max(val) < min(test) and max(test) < min(train)
    # Each split starts with the document after the previous split's last one.
    assert test[0] == val[-1] + 1
    assert train[0] == test[-1] + 1
    # Val and test are truncated mid-document, so their boundaries are exercised.
    assert recon["val"][1] == val[-1]
    assert recon["test"][1] == test[-1]


# --- 5. Running out of input -----------------------------------------------

def test_short_train_raises_unless_allowed(corpus, tmp_path):
    too_long = dict(TARGETS, train=10**6)
    with pytest.raises(ValueError):
        pack_dataset(iter_documents([corpus.jsonl]), corpus.tok, tmp_path / "a",
                     targets=too_long, context_length=CONTEXT)
    meta = pack_dataset(iter_documents([corpus.jsonl]), corpus.tok, tmp_path / "b",
                        targets=too_long, context_length=CONTEXT, allow_short_train=True)
    n_train = meta["splits"]["train"]["n_tokens"]
    assert 0 < n_train < 10**6
    arr = np.fromfile(tmp_path / "b" / "train.bin", dtype=np.uint16)
    assert len(arr) == n_train
    assert int(arr[-1]) == eos_id(corpus.tok)  # input ended on a document boundary


def test_short_val_or_test_always_raises(corpus, tmp_path):
    for name in ("val", "test"):
        targets = dict(TARGETS, **{name: 10**6})
        with pytest.raises(ValueError):
            pack_dataset(iter_documents([corpus.jsonl]), corpus.tok, tmp_path / name,
                         targets=targets, context_length=CONTEXT, allow_short_train=True)


# --- 6. Hashes -------------------------------------------------------------

def test_dataset_hash_matches_independent_computation(corpus):
    h = hashlib.sha256()
    for name in ("train", "val", "test"):
        h.update((corpus.dataset / f"{name}.bin").read_bytes())
    assert dataset_sha256(corpus.dataset) == h.hexdigest()
    assert corpus.meta["dataset_sha256"] == h.hexdigest()


def test_meta_records_matching_hashes(corpus):
    meta = json.loads((corpus.dataset / "meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizer_sha256"] == hashlib.sha256((corpus.dataset / "tokenizer.json").read_bytes()).hexdigest()
    for name in SPLITS:
        sm = meta["splits"][name]
        assert sm["sha256"] == hashlib.sha256((corpus.dataset / f"{name}.bin").read_bytes()).hexdigest()
        assert sm["docs_sha256"] == hashlib.sha256((corpus.dataset / f"{name}.docs.bin").read_bytes()).hexdigest()
        assert sm["n_tokens"] == TARGETS[name]
    assert meta["vocab_size"] == corpus.tok.get_vocab_size()
    assert meta["eos_id"] == 0
    assert meta["context_length"] == CONTEXT
    assert meta["sources"] == corpus.meta["sources"]


# --- 7. Refusal ------------------------------------------------------------

def _other(hex_digest: str) -> str:
    return ("0" if hex_digest[0] != "0" else "1") + hex_digest[1:]


def _copy(corpus, tmp_path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(corpus.dataset, dst)
    return dst


def _flip_token(path: Path, index: int) -> None:
    arr = np.fromfile(path, dtype=np.uint16)
    arr[index] = arr[index] ^ 1
    arr.tofile(path)


def _flip_byte(path: Path, index: int) -> None:
    data = bytearray(path.read_bytes())
    data[index] ^= 1
    path.write_bytes(bytes(data))


def _meta(corpus) -> dict:
    return json.loads((corpus.dataset / "meta.json").read_text(encoding="utf-8"))


def test_open_accepts_intact_dataset(corpus):
    meta = _meta(corpus)
    splits = open_dataset(corpus.dataset, expected_dataset_sha256=meta["dataset_sha256"],
                          expected_tokenizer_sha256=meta["tokenizer_sha256"])
    assert set(splits) == set(SPLITS)


def test_wrong_expected_dataset_hash_refused(corpus):
    meta = _meta(corpus)
    with pytest.raises(DataError):
        open_dataset(corpus.dataset, expected_dataset_sha256=_other(meta["dataset_sha256"]),
                     expected_tokenizer_sha256=None)


def test_wrong_expected_tokenizer_hash_refused(corpus):
    meta = _meta(corpus)
    with pytest.raises(DataError):
        open_dataset(corpus.dataset, expected_dataset_sha256=None,
                     expected_tokenizer_sha256=_other(meta["tokenizer_sha256"]))


def test_flipped_token_in_train_refused(corpus, tmp_path):
    copy = _copy(corpus, tmp_path)
    _flip_token(copy / "train.bin", 1234)
    with pytest.raises(DataError, match="train.bin"):
        open_dataset(copy, expected_dataset_sha256=None, expected_tokenizer_sha256=None)


def test_flipped_token_in_val_refused(corpus, tmp_path):
    copy = _copy(corpus, tmp_path)
    _flip_token(copy / "val.bin", 17)
    with pytest.raises(DataError, match="val.bin"):
        open_dataset(copy, expected_dataset_sha256=None, expected_tokenizer_sha256=None)


def test_flipped_offset_in_docs_file_refused(corpus, tmp_path):
    copy = _copy(corpus, tmp_path)
    _flip_byte(copy / "test.docs.bin", 8)  # low byte of the second offset
    with pytest.raises(DataError, match="test.docs.bin"):
        open_dataset(copy, expected_dataset_sha256=None, expected_tokenizer_sha256=None)


def test_changed_tokenizer_file_refused(corpus, tmp_path):
    copy = _copy(corpus, tmp_path)
    _flip_byte(copy / "tokenizer.json", len((copy / "tokenizer.json").read_bytes()) // 2)
    with pytest.raises(DataError, match="tokenizer.json"):
        open_dataset(copy, expected_dataset_sha256=None, expected_tokenizer_sha256=None)


def test_missing_file_refused(corpus, tmp_path):
    copy = _copy(corpus, tmp_path)
    (copy / "train.bin").unlink()
    with pytest.raises(DataError, match="train.bin"):
        open_dataset(copy, expected_dataset_sha256=None, expected_tokenizer_sha256=None)


def test_edited_n_tokens_in_meta_refused(corpus, tmp_path):
    copy = _copy(corpus, tmp_path)
    meta = json.loads((copy / "meta.json").read_text(encoding="utf-8"))
    meta["splits"]["val"]["n_tokens"] += 1
    (copy / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(DataError, match="val.bin"):
        open_dataset(copy, expected_dataset_sha256=None, expected_tokenizer_sha256=None)


def test_without_verify_a_flipped_token_is_not_detected_but_wrong_hash_is(corpus, tmp_path):
    # Documented behaviour: verify=False trusts the hashes recorded in meta.json.
    copy = _copy(corpus, tmp_path)
    meta = _meta(corpus)
    _flip_token(copy / "train.bin", 1234)
    splits = open_dataset(copy, expected_dataset_sha256=meta["dataset_sha256"],
                          expected_tokenizer_sha256=meta["tokenizer_sha256"], verify=False)
    assert int(splits["train"].tokens[1234]) == int(corpus.splits["train"].tokens[1234]) ^ 1
    with pytest.raises(DataError):
        open_dataset(copy, expected_dataset_sha256=_other(meta["dataset_sha256"]),
                     expected_tokenizer_sha256=None, verify=False)


# --- 8. Windows ------------------------------------------------------------

def test_windows_are_contiguous_and_shifted(corpus):
    split = corpus.splits["train"]
    tokens = np.asarray(split.tokens)
    n = split.n_windows(CONTEXT)
    assert n == (len(tokens) - 1) // CONTEXT
    xs = []
    for i in range(n):
        x, y = split.window(i, CONTEXT)
        assert x.shape == (CONTEXT,) and y.shape == (CONTEXT,)
        assert np.array_equal(y[:-1], x[1:])
        assert y[-1] == tokens[(i + 1) * CONTEXT]  # the token right after the window
        xs.append(x)
    assert np.array_equal(np.concatenate(xs), tokens[: n * CONTEXT])  # no overlap, no gap
    with pytest.raises(IndexError):
        split.window(n, CONTEXT)
    with pytest.raises(IndexError):
        split.window(-1, CONTEXT)


# --- 9. Order --------------------------------------------------------------

def test_train_order_is_a_permutation_and_reproducible():
    order = train_order(50, 7)
    assert order.dtype == torch.int64
    assert sorted(order.tolist()) == list(range(50))
    assert torch.equal(order, train_order(50, 7))


def test_different_seeds_give_different_orders():
    orders = [train_order(100, seed).tolist() for seed in (1, 2, 3)]
    assert len({tuple(o) for o in orders}) == 3


def test_golden_order_regression():
    # Recorded from torch 2.15.0+cu132. A change here means the torch version
    # changed the randperm stream, and every recorded run's data order changed with it.
    assert train_order(10, 1).tolist() == [5, 6, 1, 2, 0, 8, 9, 3, 7, 4]


# --- 10. Independence ------------------------------------------------------

def test_stream_is_independent_of_global_rng_and_model_construction(corpus):
    split = corpus.splits["train"]
    torch.manual_seed(1)
    before = torch.get_rng_state()
    a = TrainStream(split, context_length=CONTEXT, sequences_per_step=4, data_seed=7)
    b = TrainStream(split, context_length=CONTEXT, sequences_per_step=4, data_seed=7)
    order = train_order(split.n_windows(CONTEXT), 7)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(a.order, order) and torch.equal(b.order, order)
    assert a.order_sha256 == order_sha256(order) == b.order_sha256

    batches_a, batches_b = [], []
    for step in range(a.max_steps):
        torch.manual_seed(100 + step)
        torch.rand(step + 3)
        torch.nn.Linear(3 + step, 5 + step)  # imitates a variant with a different init
        batches_a.append(a.batch(step))
    torch.manual_seed(999)
    torch.nn.Linear(64, 64)
    for step in range(b.max_steps):
        torch.manual_seed(7 * step)
        torch.nn.Linear(2, 2 * step + 1)
        batches_b.append(b.batch(step))
    for (xa, ya), (xb, yb) in zip(batches_a, batches_b):
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_global_rng_untouched_by_stream_and_batches(corpus):
    torch.manual_seed(5)
    before = torch.get_rng_state()
    train_order(1000, 3)
    stream = TrainStream(corpus.splits["train"], context_length=CONTEXT,
                         sequences_per_step=8, data_seed=3)
    stream.batch(0)
    assert torch.equal(before, torch.get_rng_state())


# --- 11. TrainStream.batch ---------------------------------------------------

def test_batch_shape_dtype_and_contents(corpus):
    split = corpus.splits["train"]
    tokens = np.asarray(split.tokens)
    stream = TrainStream(split, context_length=CONTEXT, sequences_per_step=8, data_seed=3)
    assert stream.max_steps == split.n_windows(CONTEXT) // 8
    for step in range(stream.max_steps):
        x, y = stream.batch(step)
        assert x.shape == (8, CONTEXT) and y.shape == (8, CONTEXT)
        assert x.dtype == torch.int64 and y.dtype == torch.int64
        for j, index in enumerate(stream.order[step * 8:(step + 1) * 8].tolist()):
            start = index * CONTEXT
            assert torch.equal(x[j], torch.from_numpy(tokens[start:start + CONTEXT].astype(np.int64)))
            assert torch.equal(y[j], torch.from_numpy(tokens[start + 1:start + CONTEXT + 1].astype(np.int64)))


def test_shorter_run_is_prefix_of_longer_and_reopened_data_agrees(corpus):
    reopened = open_dataset(corpus.dataset, expected_dataset_sha256=None, expected_tokenizer_sha256=None)
    short = TrainStream(corpus.splits["train"], context_length=CONTEXT, sequences_per_step=8, data_seed=3)
    long_ = TrainStream(reopened["train"], context_length=CONTEXT, sequences_per_step=8, data_seed=3)
    for k in range(short.max_steps):
        xa, ya = short.batch(k)
        xb, yb = long_.batch(k)
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_out_of_range_steps_raise(corpus):
    stream = TrainStream(corpus.splits["train"], context_length=CONTEXT, sequences_per_step=8, data_seed=3)
    with pytest.raises(IndexError):
        stream.batch(stream.max_steps)
    with pytest.raises(IndexError):
        stream.batch(-1)


def test_batches_are_stateless(corpus):
    stream = TrainStream(corpus.splits["train"], context_length=CONTEXT, sequences_per_step=8, data_seed=3)
    in_order = [stream.batch(k) for k in range(stream.max_steps)]
    for k in reversed(range(stream.max_steps)):
        x, y = stream.batch(k)
        assert torch.equal(x, in_order[k][0]) and torch.equal(y, in_order[k][1])


# --- 12. Eval windows --------------------------------------------------------

def test_eval_windows_cover_every_window_once_in_order(corpus):
    split = corpus.splits["val"]
    n = split.n_windows(CONTEXT)
    seen = []
    starts = []
    sizes = []
    for start, x, y in eval_windows(split, CONTEXT, batch_size=5):
        starts.append(start)
        sizes.append(x.shape[0])
        assert x.shape == y.shape and x.dtype == torch.int64
        seen.append(x)
        for j in range(x.shape[0]):
            xr, yr = split.window(start + j, CONTEXT)
            assert torch.equal(x[j], torch.from_numpy(xr)) and torch.equal(y[j], torch.from_numpy(yr))
    assert starts == list(range(0, n, 5))
    assert sizes[-1] == n - starts[-1] and sizes[-1] < 5
    assert sum(sizes) == n
    assert torch.equal(torch.cat(seen), torch.cat([torch.from_numpy(split.window(i, CONTEXT)[0])[None] for i in range(n)]))


# --- 13. Command line --------------------------------------------------------

def _prepare_args(corpus, out: Path, *extra: str) -> list[str]:
    return [
        "--inputs", str(corpus.jsonl),
        "--out", str(out),
        "--vocab-size", str(VOCAB),
        "--context-length", str(CONTEXT),
        "--train-tokens", str(TARGETS["train"]),
        "--val-tokens", str(TARGETS["val"]),
        "--test-tokens", str(TARGETS["test"]),
        *extra,
    ]


def test_prepare_script_end_to_end_and_refuses_overwrite(corpus, tmp_path, capsys):
    out = tmp_path / "run1"
    rc = main(_prepare_args(corpus, out, "--tokenizer-inputs", str(corpus.jsonl)))
    assert rc == 0
    for name in ("tokenizer.json", "meta.json", "train.bin", "train.docs.bin",
                 "val.bin", "val.docs.bin", "test.bin", "test.docs.bin"):
        assert (out / name).is_file()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["sources"][0]["sha256"] == file_sha256(corpus.jsonl)
    printed = capsys.readouterr().out
    assert meta["dataset_sha256"] in printed and meta["tokenizer_sha256"] in printed
    splits = open_dataset(out, expected_dataset_sha256=meta["dataset_sha256"],
                          expected_tokenizer_sha256=meta["tokenizer_sha256"])
    assert len(splits["train"].tokens) == TARGETS["train"]

    before = (out / "meta.json").read_bytes()
    assert main(_prepare_args(corpus, out, "--tokenizer-inputs", str(corpus.jsonl))) != 0
    assert (out / "meta.json").read_bytes() == before


def test_prepare_script_reuses_given_tokenizer(corpus, tmp_path):
    out = tmp_path / "given"
    rc = main(_prepare_args(corpus, out, "--tokenizer", str(corpus.tok_path)))
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizer_sha256"] == file_sha256(corpus.tok_path)
    assert (out / "tokenizer.json").read_bytes() == corpus.tok_path.read_bytes()


def test_same_inputs_give_same_dataset_hash_in_different_directories(corpus, tmp_path):
    hashes = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert main(_prepare_args(corpus, out, "--tokenizer-inputs", str(corpus.jsonl))) == 0
        hashes.append(json.loads((out / "meta.json").read_text(encoding="utf-8"))["dataset_sha256"])
    assert hashes[0] == hashes[1]
    out = tmp_path / "c"
    assert main(_prepare_args(corpus, out, "--tokenizer", str(corpus.tok_path))) == 0
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["dataset_sha256"] == hashes[0]


# --- 14. Vocabulary limit ----------------------------------------------------

class _FakeTokenizer:
    def __init__(self, size: int):
        self._size = size

    def get_vocab_size(self) -> int:
        return self._size


def test_vocabulary_above_uint16_range_refused(tmp_path):
    with pytest.raises(ValueError):
        pack_dataset(iter(["some text"]), _FakeTokenizer(65537), tmp_path / "big",
                     targets=TARGETS, context_length=CONTEXT)
    assert not (tmp_path / "big" / "train.bin").exists()


# --- 15. Config --------------------------------------------------------------

def test_data_config_holds_plan_values():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["tokenizer"]["vocab_size"] == 8192
    assert cfg["tokenizer"]["eos_token"] == EOS_TOKEN
    assert cfg["context_length"] == 1024
    assert cfg["splits"] == {"train": 400_000_000, "val": 4_000_000, "test": 4_000_000}
    assert cfg["smoke_tokens"] == 5_000_000
    assert cfg["tokenizer"]["sha256"] is None
    assert cfg["dataset_sha256"] is None
