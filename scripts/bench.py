"""Inference benchmark: prefill, decode, KV bytes and maximum batch (plan §8.6, §10.4).

One JSON file per (variant, mode) is written to --out-dir, named bench_<variant>_<mode>.json,
in the "bench/2" schema (see `write_bench_file`). `read_decode_tok_per_s` reads the headline
column that scripts/make_tables.py takes.

Every point is measured in `--repetitions` repetitions (default 3, decision 225). The repetitions
run round-robin over the (variant, mode) pairs: repetition 1 of every pair, then repetition 2 of
every pair, and so on. Each repetition rebuilds the model (untimed) and measures every prefill
and decode point with the warm-ups and timed calls of the single-measurement discipline. The
document reports the median of the repetitions with the smallest and largest value. The KV-bytes
check and the maximum-batch search are not timings: they run once per pair, in repetition 1.

    python -m scripts.bench --device cuda --gpu-shared false --expected-backend efficient \
        --out-dir results/bench
    python -m scripts.bench --device cuda --gpu-shared false --expected-backend efficient \
        --with-ref --out-dir results/bench
    python -m scripts.bench --synthetic --contexts 16 32 --decode-batches 1 2 \
        --decode-caches 8 16 --out-dir /tmp/bench_check

`--synthetic` replaces the model by a tiny one and is for exercising the script; its
numbers mean nothing about the study. Plan §2.2: the benchmark must run alone on the
box. On CUDA `--gpu-shared` is required and the script refuses `true`.

Plan §2.3 item 1: on CUDA `--expected-backend` is required (the SDPA kernel the variants
must use). Opt-mode forwards run with only that kernel enabled, so a fallback fails.
It is refused on CPU. `ref` documents make no SDPA call and record a null backend.

The benchmark runs on the GPU in fp16 (the model is cast with `.half()`); on CPU it
runs in fp32, and the JSON `precision` field says which.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

from src import config
from src.eval.inference import (
    BenchError,
    WARMUP,
    REPS,
    analytic_kv_bytes_per_token,
    analytic_kv_cache_bytes,
    build_model_for_benchmark,
    check_backend_flag,
    max_batch_before_oom,
    measure_decode,
    measure_kv_allocation,
    measure_prefill,
    try_decode_at,
)
from src.utils.manifest import collect_environment

SCHEMA = "bench/2"
# Plan §4.2: analytic and measured KV bytes must agree within 5%. Written here, not imported
# from tests/ (the script must run without the test tree). tests/tolerances.py holds the same
# value as KV_BYTES_MEASURED_REL, and a cpu test asserts the two are equal.
KV_BYTES_TOLERANCE_REL = 0.05

ALL_VARIANTS = ("mha", "gqa4", "gqa2", "mqa", "mla192", "mla96", "swa256")  # never "null"
DEFAULT_CONTEXTS = (128, 256, 512, 1024, 2048, 4096, 8192)  # plan §10.4: 128 to 8,192, powers of two
DEFAULT_DECODE_BATCHES = (1, 8, 32)  # plan §10.4
DEFAULT_DECODE_CACHES = (1024, 4096, 8192)  # plan §10.4: 1k, 4k, 8k
DEFAULT_PREFILL_BATCH = 1  # not in the plan; one sequence per prefill keeps 8k within memory
DEFAULT_REPETITIONS = 3  # decision 225: every point in three round-robin repetitions
MAX_BATCH_CACHE = 8192  # plan §8.6: maximum batch before OOM at cache 8k
HEADLINE_BATCH = 32  # plan §10.1 column: decode tok/s at batch 32 ...
HEADLINE_CACHE = 4096  # ... and cache 4,096, opt mode
KV_CHECK_TOKENS = 1024  # tokens the KV allocation check holds (capped to the window for SWA)
MODES = ("opt", "ref")

SYNTHETIC_DIMS = {
    "model.d_model": 64,
    "model.n_heads": 8,
    "model.head_dim": 8,
    "model.n_layers": 2,
    "model.d_ff": 96,
    "model.vocab_size": 128,
    "recipe.batch.sequences_per_step": 8,
    "recipe.batch.sequence_length": 32,
    "recipe.batch.tokens_per_step": 8 * 32,
    "recipe.batch.micro_batch": 4,
}


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1"):
        return True
    if text.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def check_gpu_shared(value: bool | None, device: torch.device) -> bool:
    """The value recorded in the output. Refuses a shared GPU (plan §2.2); requires the flag on CUDA."""
    if value is None:
        if device.type == "cuda":
            raise BenchError(
                "--gpu-shared true|false is required on CUDA: state whether any other process "
                "uses the GPU (plan §2.2)"
            )
        value = False
    if value:
        raise BenchError("refusing to run: gpu_shared is true; the benchmark must run alone (plan §2.2)")
    return value


def _overrides(synthetic: bool) -> dict:
    return dict(SYNTHETIC_DIMS) if synthetic else {}


def _release(model) -> None:
    """Drop the model and return its memory before the next repetition is built."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _aggregate_row(rows: Sequence[dict], identity: Sequence[str], timed: Sequence[str]) -> dict:
    """One output point from the same point measured in every repetition.

    `timed` fields get the median (the reported value), the list in repetition order, and the
    smallest and largest repetition. `peak_bytes` is the maximum over the repetitions, None if
    any repetition has none (off CUDA every repetition has none).
    """
    out: dict = {key: rows[0][key] for key in identity}
    for field in timed:
        values = [float(row[field]) for row in rows]
        out[field] = float(statistics.median(values))
        out[f"{field}_by_repetition"] = values
        out[f"{field}_min"] = min(values)
        out[f"{field}_max"] = max(values)
    peaks = [row["peak_bytes"] for row in rows]
    out["peak_bytes"] = None if any(p is None for p in peaks) else max(peaks)
    return out


def aggregate_prefill(rows: Sequence[dict]) -> dict:
    """The prefill point of a document, from its row in each repetition (the same context and batch)."""
    return _aggregate_row(rows, ("context", "batch"), ("latency_ms", "tok_per_s"))


def aggregate_decode(rows: Sequence[dict]) -> dict:
    """The decode point of a document, from its row in each repetition (the same batch and cache)."""
    return _aggregate_row(rows, ("batch", "cache"), ("ms_per_token", "tok_per_s"))


class _Track:
    """One (variant, mode) across its repetitions: what the round-robin accumulates.

    The model is built fresh for each repetition by the caller and not kept here. The KV-bytes
    check and the maximum-batch search run in the first repetition only.
    """

    def __init__(
        self,
        variant: str,
        mode: str,
        *,
        device: torch.device,
        contexts=DEFAULT_CONTEXTS,
        prefill_batch: int = DEFAULT_PREFILL_BATCH,
        decode_batches=DEFAULT_DECODE_BATCHES,
        decode_caches=DEFAULT_DECODE_CACHES,
        max_batch_cache: int = MAX_BATCH_CACHE,
        with_max_batch: bool = True,
        overrides: dict | None = None,
        gpu_shared: bool = False,
        expected_backend: str | None = None,
        warmup: int = WARMUP,
        reps: int = REPS,
        clock=None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.variant = variant
        self.mode = mode
        self.device = device
        self.cuda = device.type == "cuda"
        self.backend = expected_backend if mode == "opt" else None
        self.gpu_shared = bool(gpu_shared)
        self.contexts = tuple(contexts)
        self.prefill_batch = prefill_batch
        self.decode_batches = tuple(decode_batches)
        self.decode_caches = tuple(decode_caches)
        self.max_batch_cache = max_batch_cache
        self.with_max_batch = with_max_batch
        self.overrides = overrides
        self.warmup = warmup
        self.reps = reps
        self.clock_kwargs = {} if clock is None else {"clock": clock}
        self.longest = max(
            list(self.contexts) + list(self.decode_caches)
            + ([max_batch_cache] if with_max_batch and self.cuda else [])
        )
        self.kv: dict | None = None
        self.max_batch: int | None = None
        self.prefill_reps: list[list[dict]] = []
        self.decode_reps: list[list[dict]] = []

    def build(self):
        """A fresh model for one repetition (untimed). Half precision on CUDA."""
        model = build_model_for_benchmark(self.variant, context_length=self.longest, overrides=self.overrides).to(self.device)
        if self.cuda:
            model = model.half()
        return model

    def repetition(self, model) -> None:
        """Measure every prefill and decode point once on `model`. The first call also runs the
        KV-bytes check and the maximum-batch search."""
        first = not self.prefill_reps
        if first:
            self._check_kv(model)
        prefill = [
            measure_prefill(
                model, mode=self.mode, context=c, batch=self.prefill_batch, device=self.device,
                backend=self.backend, warmup=self.warmup, reps=self.reps, **self.clock_kwargs,
            )
            for c in self.contexts
        ]
        decode = [
            measure_decode(
                model, mode=self.mode, batch=b, cache=c, device=self.device,
                backend=self.backend, warmup=self.warmup, reps=self.reps, **self.clock_kwargs,
            )
            for b in self.decode_batches
            for c in self.decode_caches
        ]
        self.prefill_reps.append(prefill)
        self.decode_reps.append(decode)
        if first and self.with_max_batch and self.cuda:
            self.max_batch = max_batch_before_oom(
                lambda b: try_decode_at(
                    model, batch=b, cache=self.max_batch_cache, device=self.device, backend=self.backend
                )
            )
            torch.cuda.empty_cache()

    def _check_kv(self, model) -> None:
        analytic = analytic_kv_bytes_per_token(model)
        measured = None
        measured_tokens = None
        allocation = measure_kv_allocation(model, device=self.device, n_tokens=KV_CHECK_TOKENS, backend=self.backend)
        if allocation is not None:
            total, n = allocation
            expected = analytic_kv_cache_bytes(model, n)
            rel = abs(total - expected) / expected
            if rel > KV_BYTES_TOLERANCE_REL:
                raise BenchError(
                    f"{self.variant}/{self.mode}: measured KV bytes {total} for {n} tokens differ from the analytic "
                    f"{expected} by {rel:.2%} (limit {KV_BYTES_TOLERANCE_REL:.0%}, plan §4.2)"
                )
            measured = total / n
            measured_tokens = n
        self.kv = {"analytic": analytic, "measured": measured, "measured_tokens": measured_tokens}

    def document(self) -> dict:
        """The schema-"bench/2" document, from every repetition measured so far."""
        if not self.prefill_reps:
            raise BenchError(f"{self.variant}/{self.mode}: no repetition was measured")
        prefill = [aggregate_prefill(rows) for rows in zip(*self.prefill_reps)]
        decode = [aggregate_decode(rows) for rows in zip(*self.decode_reps)]
        return {
            "schema": SCHEMA,
            "variant": self.variant,
            "mode": self.mode,
            "precision": "fp16" if self.cuda else "fp32",
            "backend": self.backend,
            "environment": collect_environment(),
            "gpu_shared": self.gpu_shared,
            "repetitions": len(self.prefill_reps),
            "kv_bytes_per_token": self.kv,
            "prefill": prefill,
            "decode": decode,
            "max_batch_at_8k": self.max_batch,
        }


def bench_variant_mode(
    variant: str,
    mode: str,
    *,
    device: torch.device,
    contexts=DEFAULT_CONTEXTS,
    prefill_batch: int = DEFAULT_PREFILL_BATCH,
    decode_batches=DEFAULT_DECODE_BATCHES,
    decode_caches=DEFAULT_DECODE_CACHES,
    max_batch_cache: int = MAX_BATCH_CACHE,
    with_max_batch: bool = True,
    overrides: dict | None = None,
    gpu_shared: bool = False,
    expected_backend: str | None = None,
    warmup: int = WARMUP,
    reps: int = REPS,
    repetitions: int = 1,
    clock=None,
) -> dict:
    """The schema-"bench/2" document for one variant and mode, from `repetitions` repetitions.

    `expected_backend` is enforced on the opt track only. Its name is recorded in the
    document's "backend" key; ref documents record null. The command line runs the
    repetitions round-robin over all pairs instead (see `main`); this function runs them
    one pair at a time.
    """
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError(f"repetitions must be an int >= 1, got {repetitions!r}")
    track = _Track(
        variant, mode, device=device, contexts=contexts, prefill_batch=prefill_batch,
        decode_batches=decode_batches, decode_caches=decode_caches, max_batch_cache=max_batch_cache,
        with_max_batch=with_max_batch, overrides=overrides, gpu_shared=gpu_shared,
        expected_backend=expected_backend, warmup=warmup, reps=reps, clock=clock,
    )
    for _ in range(repetitions):
        model = track.build()
        track.repetition(model)
        _release(model)
    return track.document()


def write_bench_file(out_dir: Path, document: dict) -> Path:
    """Write one "bench/2" document as bench_<variant>_<mode>.json.

    Keys: schema, variant, mode, precision, backend (enforced SDPA name on opt documents
    on CUDA, else null), environment, gpu_shared, repetitions (N), kv_bytes_per_token
    {analytic, measured, measured_tokens}, prefill, decode, max_batch_at_8k.

    prefill: [{context, batch, latency_ms, tok_per_s, latency_ms_by_repetition,
    tok_per_s_by_repetition, latency_ms_min, latency_ms_max, tok_per_s_min, tok_per_s_max,
    peak_bytes}]. latency_ms and tok_per_s are the medians over the N repetitions; the
    *_by_repetition lists hold the N values in repetition order; *_min and *_max are the
    smallest and largest of the N values; peak_bytes is the maximum over the repetitions.

    decode: [{batch, cache, ms_per_token, tok_per_s, ms_per_token_by_repetition,
    tok_per_s_by_repetition, ms_per_token_min, ms_per_token_max, tok_per_s_min, tok_per_s_max,
    peak_bytes}], with the same meanings for ms_per_token and tok_per_s.

    Off CUDA, measured, measured_tokens, peak_bytes and max_batch_at_8k are null.
    """
    path = out_dir / f"bench_{document['variant']}_{document['mode']}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _headline_decode_rows(directory: str | Path):
    """(variant, decode row) of the headline point, opt mode, of every document in `directory`.

    A bench/1 file, or one recorded with gpu_shared true, is refused.
    """
    for path in sorted(Path(directory).glob("bench_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("schema") != SCHEMA:
            raise BenchError(f"{path.name}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
        if doc.get("gpu_shared") is not False:
            raise BenchError(f"{path.name}: recorded with gpu_shared true; not usable (plan §2.2)")
        if doc["mode"] != "opt":
            continue
        for row in doc["decode"]:
            if row["batch"] == HEADLINE_BATCH and row["cache"] == HEADLINE_CACHE:
                yield doc["variant"], row


def read_decode_tok_per_s(directory: str | Path) -> dict[str, float]:
    """{variant: decode tok/s at batch 32, cache 4,096, opt mode}, the headline column of make_tables.

    The value is the median over the file's repetitions. Variants whose file lacks that point are
    omitted, and make_tables shows a blank cell. A bench/1 file, or one recorded with gpu_shared
    true, is refused.
    """
    return {variant: row["tok_per_s"] for variant, row in _headline_decode_rows(directory)}


def read_decode_range(directory: str | Path) -> dict[str, tuple[float, float]]:
    """{variant: (smallest, largest) decode tok/s at batch 32, cache 4,096, opt mode} (decision 225).

    The same point and refusals as read_decode_tok_per_s; the median is that function's value.
    """
    return {variant: (row["tok_per_s_min"], row["tok_per_s_max"]) for variant, row in _headline_decode_rows(directory)}


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_positive(value) -> bool:
    return is_number(value) and value > 0


NULL_VARIANT = "null"


def known_variant(variant) -> bool:
    """A variant the project knows, other than the "null" placeholder of the timing documents."""
    return isinstance(variant, str) and variant != NULL_VARIANT and variant in config.variant_names()


def _listing(head: str, problems: Sequence[str]) -> str:
    return "\n".join([f"{head}; {len(problems)} problem(s):"] + [f"  - {p}" for p in problems])


# --- bench/2 document checks (plan §10.4, decision 225): shared by make_tables, make_plots and timing_report

_BENCH_KEYS = ("repetitions", "kv_bytes_per_token", "prefill", "decode", "max_batch_at_8k")
_PREFILL_IDENTITY = ("context", "batch")
_DECODE_IDENTITY = ("batch", "cache")
_PREFILL_TIMED = ("latency_ms", "tok_per_s")
_DECODE_TIMED = ("ms_per_token", "tok_per_s")
_KV_KEYS = ("analytic", "measured", "measured_tokens")


def _timed_keys(identity: tuple, timed: tuple) -> list[str]:
    """The keys of a bench/2 point: identity, then for each timed field its median, its repetitions, min and max, then peak_bytes."""
    keys = list(identity)
    for field in timed:
        keys += [field, f"{field}_by_repetition", f"{field}_min", f"{field}_max"]
    return keys + ["peak_bytes"]


_PREFILL_KEYS = tuple(_timed_keys(_PREFILL_IDENTITY, _PREFILL_TIMED))
_DECODE_KEYS = tuple(_timed_keys(_DECODE_IDENTITY, _DECODE_TIMED))


def _point_problems(label: str, row, repetitions, timed: tuple) -> list[str]:
    """Problems with the timed fields of one point: lists of `repetitions` positive numbers, and
    median, min and max that are the statistics of that list with min <= median <= max."""
    problems = []
    for field in timed:
        name = f"{label}: {field}"
        values = row.get(f"{field}_by_repetition")
        if not isinstance(values, list) or len(values) != repetitions or not all(is_positive(v) for v in values):
            problems.append(f"{name}_by_repetition {values!r}, expected {repetitions} positive numbers")
            continue
        lo, median, hi = row.get(f"{field}_min"), row.get(field), row.get(f"{field}_max")
        if not all(is_positive(v) for v in (lo, median, hi)):
            problems.append(f"{name}: median, min and max {median!r}, {lo!r}, {hi!r}, expected positive numbers")
        elif not lo <= median <= hi:
            problems.append(f"{name}: min {lo}, median {median}, max {hi}; expected min <= median <= max")
        elif lo != min(values) or hi != max(values) or median != statistics.median(values):
            problems.append(f"{name}: median, min and max are not the statistics of {name}_by_repetition")
    return problems


def check_documents(docs: Sequence) -> None:
    """Refuse bench documents no report can read: another schema, an unknown variant or mode,
    a missing field, or a second document for one (variant, mode)."""
    problems = []
    seen: set[tuple] = set()
    counts = set()
    for index, doc in enumerate(docs):
        label = f"bench document {index}"
        if not isinstance(doc, Mapping):
            problems.append(f"{label} is not an object")
            continue
        variant, mode = doc.get("variant"), doc.get("mode")
        label = f"bench {variant!r}/{mode!r}"
        if doc.get("schema") != SCHEMA:
            problems.append(f"{label}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
        if not known_variant(variant):
            problems.append(f"{label}: variant {variant!r} is not one the project knows")
        if mode not in MODES:
            problems.append(f"{label}: mode {mode!r}, expected one of {list(MODES)}")
        missing = [key for key in _BENCH_KEYS if key not in doc]
        if missing:
            problems.append(f"{label} lacks {missing}")
            continue
        repetitions = doc["repetitions"]
        counts.add(repetitions)
        valid = isinstance(repetitions, int) and not isinstance(repetitions, bool) and repetitions >= 1
        if not valid:
            problems.append(f"{label}: repetitions {repetitions!r}, expected an integer of at least 1")
        for row in doc["prefill"]:
            if not all(key in row for key in _PREFILL_KEYS):
                problems.append(f"{label}: a prefill row lacks one of {list(_PREFILL_KEYS)}")
            elif valid:
                problems += _point_problems(f"{label}, prefill context {row['context']}", row, repetitions, _PREFILL_TIMED)
        for row in doc["decode"]:
            if not all(key in row for key in _DECODE_KEYS):
                problems.append(f"{label}: a decode row lacks one of {list(_DECODE_KEYS)}")
            elif valid:
                problems += _point_problems(
                    f"{label}, decode batch {row['batch']} cache {row['cache']}", row, repetitions, _DECODE_TIMED
                )
        if not all(key in doc["kv_bytes_per_token"] for key in _KV_KEYS):
            problems.append(f"{label}: kv_bytes_per_token lacks one of {list(_KV_KEYS)}")
        key = (variant, mode)
        if key in seen:
            problems.append(f"two bench documents for {variant}/{mode}: each (variant, mode) must appear once")
        seen.add(key)
    if len(counts) > 1:
        problems.append(f"the bench documents disagree on the number of repetitions: {sorted(map(str, counts))}")
    if problems:
        raise BenchError(_listing("the bench documents cannot be reported", problems))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), help="variant names; default all seven")
    p.add_argument("--with-ref", action="store_true", help="also benchmark the ref track (plan §4.4)")
    p.add_argument("--device", default="cpu", help="torch device, cpu or cuda")
    p.add_argument("--gpu-shared", type=_bool, default=None, help="required on CUDA: true or false (plan §2.2)")
    p.add_argument(
        "--expected-backend", default=None,
        help="required on CUDA, refused on CPU: the SDPA backend the opt track must use "
             "(flash, efficient, math or cudnn; plan §2.3 item 1)",
    )
    p.add_argument("--contexts", nargs="+", type=int, default=list(DEFAULT_CONTEXTS))
    p.add_argument("--prefill-batch", type=int, default=DEFAULT_PREFILL_BATCH)
    p.add_argument("--decode-batches", nargs="+", type=int, default=list(DEFAULT_DECODE_BATCHES))
    p.add_argument("--decode-caches", nargs="+", type=int, default=list(DEFAULT_DECODE_CACHES))
    p.add_argument("--max-batch-cache", type=int, default=MAX_BATCH_CACHE)
    p.add_argument("--skip-max-batch", action="store_true", help="do not run the maximum-batch search")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument(
        "--repetitions", type=int, default=DEFAULT_REPETITIONS,
        help="repetitions of every point, round-robin over the variants (default 3, decision 225; at least 1)",
    )
    p.add_argument("--synthetic", action="store_true", help="tiny synthetic model, for testing the script only")
    p.add_argument("--out-dir", required=True, help="directory for the JSON files; created if missing")
    return p


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    known = config.variant_names()
    for name in args.variants:
        if name == "null" or name not in known:
            parser.error(f"unknown or disallowed variant {name!r}; choose from {list(ALL_VARIANTS)}")
    device = torch.device(args.device)
    try:
        gpu_shared = check_gpu_shared(args.gpu_shared, device)
        expected_backend = check_backend_flag(args.expected_backend, device)
    except BenchError as exc:
        parser.error(str(exc))

    if args.repetitions < 1:
        parser.error(f"--repetitions must be at least 1, got {args.repetitions}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ("opt", "ref") if args.with_ref else ("opt",)
    tracks = [
        _Track(
            variant,
            mode,
            device=device,
            contexts=args.contexts,
            prefill_batch=args.prefill_batch,
            decode_batches=args.decode_batches,
            decode_caches=args.decode_caches,
            max_batch_cache=args.max_batch_cache,
            with_max_batch=not args.skip_max_batch,
            overrides=_overrides(args.synthetic),
            gpu_shared=gpu_shared,
            expected_backend=expected_backend,
            warmup=args.warmup,
            reps=args.reps,
        )
        for variant in args.variants
        for mode in modes
    ]
    # Round-robin: repetition 1 of every (variant, mode), then repetition 2 of every one, and so on.
    for repetition in range(args.repetitions):
        for track in tracks:
            print(f"repetition {repetition + 1} of {args.repetitions}: {track.variant}/{track.mode}", flush=True)
            model = track.build()
            track.repetition(model)
            _release(model)
    for track in tracks:
        path = write_bench_file(out_dir, track.document())
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
