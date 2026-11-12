"""Inference measurements for the benchmark (plan §8.6, §10.4). Library only, no command line.

What is measured
----------------
* Prefill: one full forward over `batch x context` tokens, no cache. Latency is the
  mean over the measured repetitions; `peak_bytes` is the peak of
  `torch.cuda.max_memory_allocated` over those repetitions, resident weights included.
* Decode: the cache is filled to `cache` tokens (untimed), then single-token steps
  are timed. The steps extend the cache by one token each, so the measured steps read
  between `cache` and `cache + warmup + reps` tokens. The cache is sized for that
  growth up front, so no reallocation happens inside the timed region.
* KV bytes: on CUDA, the `memory_allocated` growth of a cache that holds `n` tokens,
  sized to exactly those tokens. The cache is filled by one forward over `n` tokens,
  the same code path as decode. Memory allocated by `init_state` is excluded: MLA's
  absorbed matrices are per-cache constants, not per-token bytes. On CPU the
  measurement is None.
* Max batch: the largest batch for which a decode step at the given cache does not
  raise `torch.OutOfMemoryError`. Only that error is caught. The cache is filled in
  steps of `MAX_BATCH_FILL_CHUNK` tokens, so the fill's transient does not dominate the
  cache; the reported maximum still includes the transient of one 8-token fill step.

SDPA backend (plan §2.3 item 1): on CUDA the opt-mode forwards run inside
`sdpa_kernel` with only the named backend enabled, so a fallback to another kernel
raises instead of being timed. `ref` forwards make no SDPA call and take no backend.

Timing discipline: `warmup` calls are timed and discarded, then `reps` calls are
measured. On CUDA every timed region is bracketed by `torch.cuda.synchronize`. The
clock is injectable (seconds, like `time.perf_counter`).

Lengths past the trained context of 1,024 measure speed only, never quality (plan §8.6).
The weights are random, so no number here is a model quality result.

No global RNG is seeded or read: token ids come from an explicit `torch.Generator`.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from src import config
from src.attention.base import SDPA_BACKENDS
from src.model import DecodeState, Transformer, build_model

DEFAULT_CLOCK = time.perf_counter
WARMUP = 3
REPS = 10
FILL_CHUNK = 128  # tokens per forward when filling a cache for timing; untimed, bounds transient memory
# Tokens per forward when the max-batch search fills a cache. A 128-token chunk at batch b
# and an 8k cache can need b * heads * 128 * 8192 score elements transiently, which is more
# than MQA's whole cache per sequence (about 12.6 MB at 8k). Eight tokens keep the transient
# small against the cache, so the search measures the cache rather than the fill.
MAX_BATCH_FILL_CHUNK = 8
MAX_BATCH_LIMIT = 4096
TOKEN_SEED = 1  # seed of the explicit generator that draws prompt and decode tokens
INIT_SEED = 1  # seed of the per-parameter init used for every benchmark model
PEAK_LR = 1e-3  # config.resolve needs a learning rate; weights do not affect speed


class BenchError(RuntimeError):
    """A benchmark cannot be run, or a check it makes fails."""


def _sdpa_members() -> dict[str, SDPBackend]:
    members = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        members["cudnn"] = SDPBackend.CUDNN_ATTENTION
    return members


# The one place where names meet torch's SDPBackend members.
SDPA_MEMBERS: dict[str, SDPBackend] = _sdpa_members()


@dataclass(frozen=True)
class Timed:
    ms: tuple[float, ...]  # the measured repetitions, in milliseconds
    peak_bytes: int | None  # None off CUDA

    @property
    def mean_ms(self) -> float:
        return sum(self.ms) / len(self.ms)


def _check_count(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def _sync(device: torch.device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(device)


def check_backend_name(name: object) -> str | None:
    """`name` if it is a known SDPA backend name (or None), else BenchError."""
    if name is None:
        return None
    if not isinstance(name, str) or name not in SDPA_BACKENDS or name not in SDPA_MEMBERS:
        raise BenchError(
            f"unknown SDPA backend {name!r}; choose from {sorted(SDPA_MEMBERS)}"
        )
    return name


def check_backend_flag(value: object, device: torch.device) -> str | None:
    """The enforced SDPA backend for a run on `device`: required on CUDA, refused elsewhere (plan §2.3)."""
    name = check_backend_name(value)
    if _is_cuda(device):
        if name is None:
            raise BenchError(
                "--expected-backend is required on CUDA: the SDPA kernel the variants must use "
                "(plan §2.3 item 1); it is a fact about the hardware, so state it"
            )
    elif name is not None:
        raise BenchError("--expected-backend is CUDA only; refused on this device")
    return name


def backend_scope(name: str | None, device: torch.device):
    """A context in which only the named SDPA backend may run. A no-op when `name` is None."""
    if name is None:
        return contextlib.nullcontext()
    if not _is_cuda(device):
        raise BenchError("an SDPA backend can only be enforced on CUDA")
    return sdpa_kernel([SDPA_MEMBERS[check_backend_name(name)]])


def _timed(fn: Callable[[], object], device: torch.device, clock: Callable[[], float]) -> float:
    _sync(device)
    t0 = clock()
    fn()
    _sync(device)
    return (clock() - t0) * 1000.0


def time_calls(
    fn: Callable[[], object],
    *,
    warmup: int,
    reps: int,
    device: torch.device,
    clock: Callable[[], float] = DEFAULT_CLOCK,
) -> Timed:
    """Call `fn` `warmup` times with its timings discarded, then `reps` times and measure those."""
    _check_count("warmup", warmup, 0)
    _check_count("reps", reps, 1)
    for _ in range(warmup):
        _timed(fn, device, clock)  # discarded: first calls pay for allocation and kernel selection
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)
    ms = tuple(_timed(fn, device, clock) for _ in range(reps))
    peak = torch.cuda.max_memory_allocated(device) if _is_cuda(device) else None
    return Timed(ms=ms, peak_bytes=peak)


# --- Analytic quantities (no device needed) -------------------------------------------


def analytic_kv_bytes_per_token(model: Transformer) -> int:
    """Bytes one token adds to the decode cache, summed over layers (plan §4.2)."""
    return sum(layer.attn.kv_bytes_per_token for layer in model.layers)


def analytic_kv_cache_bytes(model: Transformer, n_tokens: int) -> int:
    """Bytes the decode cache holds after `n_tokens`, summed over layers. Honours the window cap."""
    return sum(layer.attn.kv_cache_bytes(n_tokens) for layer in model.layers)


def max_cache_tokens(model: Transformer) -> int | None:
    """The most tokens any layer's cache holds, or None if the cache is unbounded."""
    caps = [layer.attn.kv_cache_max_tokens for layer in model.layers]
    if any(c is None for c in caps):
        return None
    return min(caps)


def build_model_for_benchmark(
    variant: str, *, context_length: int, overrides: dict | None = None
) -> Transformer:
    """The variant's model from its config, with the project's seeded init.

    `model.context_length` is set to `context_length`. It does not change any weight:
    RoPE is computed for the positions it is given, and the attention modules accept
    any length. Weights are random, so the numbers are speed only (plan §8.6).
    """
    merged = dict(overrides or {})
    merged["model.context_length"] = context_length
    resolved = config.resolve(
        variant, init_seed=INIT_SEED, peak_lr=PEAK_LR, steps=1, branches=[], overrides=merged
    )
    return build_model(resolved).eval()


def random_tokens(batch: int, length: int, vocab: int, device: torch.device, seed: int = TOKEN_SEED) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab, (batch, length), generator=g).to(device)


# --- Prefill and decode -----------------------------------------------------------------


def measure_prefill(
    model: Transformer,
    *,
    mode: str,
    context: int,
    batch: int,
    device: torch.device,
    backend: str | None = None,
    warmup: int = WARMUP,
    reps: int = REPS,
    clock: Callable[[], float] = DEFAULT_CLOCK,
) -> dict:
    _check_count("context", context, 1)
    _check_count("batch", batch, 1)
    idx = random_tokens(batch, context, model.embed.num_embeddings, device)

    def run() -> None:
        with torch.no_grad(), backend_scope(backend, device):
            model(idx, mode=mode)

    timed = time_calls(run, warmup=warmup, reps=reps, device=device, clock=clock)
    latency = timed.mean_ms
    if latency <= 0:
        raise BenchError(f"prefill latency is not positive ({latency} ms); the clock is too coarse")
    return {
        "context": context,
        "batch": batch,
        "latency_ms": latency,
        "tok_per_s": batch * context * 1000.0 / latency,
        "peak_bytes": timed.peak_bytes,
    }


def fill_state(
    model: Transformer,
    *,
    mode: str,
    batch: int,
    length: int,
    device: torch.device,
    capacity: int | None = None,
    chunk: int = FILL_CHUNK,
    backend: str | None = None,
) -> DecodeState:
    """A decode state holding `length` tokens of random prompt, fed in chunks of at most `chunk`."""
    state = model.init_state(capacity=capacity)
    idx = random_tokens(batch, length, model.embed.num_embeddings, device)
    with torch.no_grad(), backend_scope(backend, device):
        for start in range(0, length, chunk):
            _, state = model(idx[:, start : start + chunk], mode=mode, state=state)
    return state


def measure_decode(
    model: Transformer,
    *,
    mode: str,
    batch: int,
    cache: int,
    device: torch.device,
    backend: str | None = None,
    warmup: int = WARMUP,
    reps: int = REPS,
    clock: Callable[[], float] = DEFAULT_CLOCK,
) -> dict:
    """Single-token decode steps on a cache of `cache` tokens (see the module docstring)."""
    _check_count("batch", batch, 1)
    _check_count("cache", cache, 1)
    capacity = cache + warmup + reps  # room for every step, so storage is never reallocated while timing
    holder = {
        "state": fill_state(
            model, mode=mode, batch=batch, length=cache, device=device, capacity=capacity, backend=backend
        )
    }
    step_token = random_tokens(batch, 1, model.embed.num_embeddings, device, seed=TOKEN_SEED + 1)

    def run() -> None:
        with torch.no_grad(), backend_scope(backend, device):
            _, holder["state"] = model(step_token, mode=mode, state=holder["state"])

    timed = time_calls(run, warmup=warmup, reps=reps, device=device, clock=clock)
    ms = timed.mean_ms
    if ms <= 0:
        raise BenchError(f"decode latency is not positive ({ms} ms); the clock is too coarse")
    return {
        "batch": batch,
        "cache": cache,
        "ms_per_token": ms,
        "tok_per_s": batch * 1000.0 / ms,
        "peak_bytes": timed.peak_bytes,
    }


def measure_kv_allocation(
    model: Transformer,
    *,
    device: torch.device,
    n_tokens: int,
    backend: str | None = None,
    seed: int = TOKEN_SEED,
) -> tuple[int, int] | None:
    """(bytes allocated by a cache holding n tokens, n), summed over layers; None off CUDA.

    `n` is `n_tokens`, or the window if the cache is capped below it. The cache is
    created first, so per-cache constants are excluded, and the forward allocates
    storage sized to exactly `n` tokens (no capacity slack, no growth).
    """
    if not _is_cuda(device):
        return None
    _check_count("n_tokens", n_tokens, 1)
    cap = max_cache_tokens(model)
    n = n_tokens if cap is None else min(n_tokens, cap)
    state = model.init_state()
    idx = random_tokens(1, n, model.embed.num_embeddings, device, seed=seed)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    with torch.no_grad(), backend_scope(backend, device):
        logits, state = model(idx, mode="opt", state=state)
    del logits
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    return after - before, n


# --- Maximum batch before OOM -----------------------------------------------------------


def try_decode_at(
    model: Transformer, *, batch: int, cache: int, device: torch.device, backend: str | None = None
) -> None:
    """One opt-mode decode step on `batch` sequences holding `cache` tokens: the body of the max-batch search.

    The cache is filled in steps of MAX_BATCH_FILL_CHUNK tokens.
    """
    state = fill_state(
        model,
        mode="opt",
        batch=batch,
        length=cache,
        device=device,
        capacity=cache + 1,
        chunk=MAX_BATCH_FILL_CHUNK,
        backend=backend,
    )
    step = random_tokens(batch, 1, model.embed.num_embeddings, device, seed=TOKEN_SEED + 2)
    with torch.no_grad(), backend_scope(backend, device):
        model(step, mode="opt", state=state)
    _sync(device)


def max_batch_before_oom(try_batch: Callable[[int], object], *, limit: int = MAX_BATCH_LIMIT) -> int:
    """The largest batch b for which try_batch(b) does not raise torch.OutOfMemoryError.

    Assumes that fitting is monotone in b. Doubles until a batch fails, then bisects.
    Returns 0 if batch 1 already fails. Any other exception propagates. Raises
    BenchError if nothing up to `limit` fails, because the answer is then not known.
    """
    _check_count("limit", limit, 1)

    def fits(b: int) -> bool:
        try:
            try_batch(b)
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # release the failed try's blocks before the next one
            return False
        return True

    if not fits(1):
        return 0
    lo, hi = 1, None
    while lo < limit:
        b = min(2 * lo, limit)
        if fits(b):
            lo = b
        else:
            hi = b
            break
    if hi is None:
        raise BenchError(f"batch {lo} still fits at the search limit {limit}; raise the limit")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo
