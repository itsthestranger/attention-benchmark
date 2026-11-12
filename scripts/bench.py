"""Inference benchmark: prefill, decode, KV bytes and maximum batch (plan §8.6, §10.4).

One JSON file per (variant, mode) is written to --out-dir, named bench_<variant>_<mode>.json,
in the "bench/1" schema (see `write_bench_file`). `read_decode_tok_per_s` reads the headline
column that scripts/make_tables.py takes.

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
import json
import sys
from pathlib import Path

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

SCHEMA = "bench/1"
# Plan §4.2: analytic and measured KV bytes must agree within 5%. Written here, not imported
# from tests/ (the script must run without the test tree). tests/tolerances.py holds the same
# value as KV_BYTES_MEASURED_REL, and a cpu test asserts the two are equal.
KV_BYTES_TOLERANCE_REL = 0.05

ALL_VARIANTS = ("mha", "gqa4", "gqa2", "mqa", "mla192", "mla96", "swa256")  # never "null"
DEFAULT_CONTEXTS = (128, 256, 512, 1024, 2048, 4096, 8192)  # plan §10.4: 128 to 8,192, powers of two
DEFAULT_DECODE_BATCHES = (1, 8, 32)  # plan §10.4
DEFAULT_DECODE_CACHES = (1024, 4096, 8192)  # plan §10.4: 1k, 4k, 8k
DEFAULT_PREFILL_BATCH = 1  # not in the plan; one sequence per prefill keeps 8k within memory
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
    clock=None,
) -> dict:
    """The schema-"bench/1" document for one variant and mode.

    `expected_backend` is enforced on the opt track only. Its name is recorded in the
    document's "backend" key; ref documents record null.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    backend = expected_backend if mode == "opt" else None
    kwargs = {} if clock is None else {"clock": clock}
    cuda = device.type == "cuda"
    longest = max(list(contexts) + list(decode_caches) + ([max_batch_cache] if with_max_batch and cuda else []))
    model = build_model_for_benchmark(variant, context_length=longest, overrides=overrides).to(device)
    if cuda:
        model = model.half()

    analytic = analytic_kv_bytes_per_token(model)
    measured = None
    measured_tokens = None
    allocation = measure_kv_allocation(model, device=device, n_tokens=KV_CHECK_TOKENS, backend=backend)
    if allocation is not None:
        total, n = allocation
        expected = analytic_kv_cache_bytes(model, n)
        rel = abs(total - expected) / expected
        if rel > KV_BYTES_TOLERANCE_REL:
            raise BenchError(
                f"{variant}/{mode}: measured KV bytes {total} for {n} tokens differ from the analytic "
                f"{expected} by {rel:.2%} (limit {KV_BYTES_TOLERANCE_REL:.0%}, plan §4.2)"
            )
        measured = total / n
        measured_tokens = n

    prefill = [
        measure_prefill(
            model, mode=mode, context=c, batch=prefill_batch, device=device,
            backend=backend, warmup=warmup, reps=reps, **kwargs,
        )
        for c in contexts
    ]
    decode = [
        measure_decode(
            model, mode=mode, batch=b, cache=c, device=device,
            backend=backend, warmup=warmup, reps=reps, **kwargs,
        )
        for b in decode_batches
        for c in decode_caches
    ]

    max_batch = None
    if with_max_batch and cuda:
        max_batch = max_batch_before_oom(
            lambda b: try_decode_at(
                model, batch=b, cache=max_batch_cache, device=device, backend=backend
            )
        )
        torch.cuda.empty_cache()

    return {
        "schema": SCHEMA,
        "variant": variant,
        "mode": mode,
        "precision": "fp16" if cuda else "fp32",
        "backend": backend,
        "environment": collect_environment(),
        "gpu_shared": bool(gpu_shared),
        "kv_bytes_per_token": {
            "analytic": analytic,
            "measured": measured,
            "measured_tokens": measured_tokens,
        },
        "prefill": prefill,
        "decode": decode,
        "max_batch_at_8k": max_batch,
    }


def write_bench_file(out_dir: Path, document: dict) -> Path:
    """Write one "bench/1" document as bench_<variant>_<mode>.json.

    Keys: schema, variant, mode, precision, backend (enforced SDPA name on opt documents
    on CUDA, else null), environment, gpu_shared, kv_bytes_per_token {analytic, measured,
    measured_tokens}, prefill [{context, batch, latency_ms, tok_per_s, peak_bytes}],
    decode [{batch, cache, ms_per_token, tok_per_s, peak_bytes}], max_batch_at_8k.
    Off CUDA, measured, measured_tokens, peak_bytes and max_batch_at_8k are null.
    """
    path = out_dir / f"bench_{document['variant']}_{document['mode']}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def read_decode_tok_per_s(directory: str | Path) -> dict[str, float]:
    """{variant: decode tok/s at batch 32, cache 4,096, opt mode}, the headline column of make_tables.

    Variants whose file lacks that point are omitted, and make_tables shows a blank cell.
    A file recorded with gpu_shared true is refused.
    """
    result: dict[str, float] = {}
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
                result[doc["variant"]] = row["tok_per_s"]
    return result


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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ("opt", "ref") if args.with_ref else ("opt",)
    for variant in args.variants:
        for mode in modes:
            document = bench_variant_mode(
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
            path = write_bench_file(out_dir, document)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
