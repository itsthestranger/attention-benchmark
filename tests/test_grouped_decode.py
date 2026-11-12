"""Plan §5, GQA/MQA decode: in `opt` decode, K and V reach the kernel with
their own head count, never expanded to 8 (plan §2.3, §13.16).

The common implementation repeats each KV head across its query heads before
the kernel. The cache stays small, but the kernel reads a full-size copy, so
decode is no faster than MHA and the savings the inference benchmark is meant
to measure vanish. The `opt` decode path groups the queries instead.

cpu tier: shapes. gpu tier: peak memory. t4 tier: decode time.

Written by the orchestrator before the implementation; the worker may not
change this file.

The contract for `opt` decode in `src/attention/grouped.py`
-----------------------------------------------------------
With G KV heads, H query heads, S tokens in the cache after appending and T new
tokens, the one call to `F.scaled_dot_product_attention` per layer receives

    query: (B, G, (H // G) * T, head_dim)
    key:   (B, G, S, head_dim)
    value: (B, G, S, head_dim)

For T = 1 no mask is passed. The cache holds keys (already rotated) and values
as (B, G, S, head_dim) and exposes them as `cache.k` and `cache.v`.
"""

import time

import pytest
import torch

from src import config
from src.model import build_model
from tests.variants import GROUPED_CASES, HEAD_DIM, N_HEADS, Case, tiny_model, tokens

MHA = Case("mha", 8)
REDUCED = [c for c in GROUPED_CASES if c.n_kv_heads < N_HEADS]
CACHE_8K = 8192  # plan §8.6


class Recorder:
    """Wraps `scaled_dot_product_attention` inside the grouped module."""

    def __init__(self):
        import src.attention.grouped as grouped

        self.module = grouped
        self.calls = []

    def __enter__(self):
        self.original = self.module.F.scaled_dot_product_attention

        def recording(query, key, value, *args, **kwargs):
            mask = kwargs.get("attn_mask", args[0] if args else None)
            self.calls.append(
                {
                    "query": tuple(query.shape),
                    "key": tuple(key.shape),
                    "value": tuple(value.shape),
                    "mask": None if mask is None else tuple(mask.shape),
                    "is_causal": kwargs.get("is_causal", False),
                }
            )
            return self.original(query, key, value, *args, **kwargs)

        self.module.F.scaled_dot_product_attention = recording
        return self

    def __exit__(self, *exc):
        self.module.F.scaled_dot_product_attention = self.original


@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_opt_decode_passes_kv_with_their_own_head_count(case):
    G, share = case.n_kv_heads, N_HEADS // case.n_kv_heads
    model = tiny_model(case)
    idx = tokens(3, 12, seed=20)
    state = model.init_state()
    with torch.no_grad(), Recorder() as rec:
        for t in range(12):
            _, state = model(idx[:, t : t + 1], mode="opt", state=state)
    assert len(rec.calls) == 12 * len(model.layers)  # one kernel call per layer per token
    for i, call in enumerate(rec.calls):
        cached = i // len(model.layers) + 1
        assert call["query"] == (3, G, share, HEAD_DIM), call
        assert call["key"] == (3, G, cached, HEAD_DIM), call
        assert call["value"] == (3, G, cached, HEAD_DIM), call
        assert call["mask"] is None and call["is_causal"] is False, call


@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_opt_prefill_through_the_cache_is_grouped_too(case):
    G, share = case.n_kv_heads, N_HEADS // case.n_kv_heads
    model = tiny_model(case)
    idx = tokens(2, 12, seed=21)
    with torch.no_grad(), Recorder() as rec:
        _, state = model(idx[:, :5], mode="opt", state=model.init_state())
        model(idx[:, 5:], mode="opt", state=state)
    first, second = rec.calls[0], rec.calls[len(model.layers)]
    assert first["query"] == (2, G, share * 5, HEAD_DIM) and first["key"] == (2, G, 5, HEAD_DIM)
    assert second["query"] == (2, G, share * 7, HEAD_DIM) and second["key"] == (2, G, 12, HEAD_DIM)
    assert second["value"] == (2, G, 12, HEAD_DIM)


@pytest.mark.parametrize("mode", ["ref", "opt"])
@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_cache_holds_kv_heads_only(case, mode):
    model = tiny_model(case)
    with torch.no_grad():
        _, state = model(tokens(3, 9, seed=22), mode=mode, state=model.init_state())
    for cache in state.caches:
        assert tuple(cache.k.shape) == (3, case.n_kv_heads, 9, HEAD_DIM)
        assert tuple(cache.v.shape) == (3, case.n_kv_heads, 9, HEAD_DIM)
        assert cache.length == 9


def test_cached_values_per_token_shrink_with_kv_heads():
    """What the KV bytes of plan §3.2 count: 2 * G * head_dim values per token per layer."""
    for case in GROUPED_CASES:
        model = tiny_model(case)
        with torch.no_grad():
            _, state = model(tokens(1, 4), mode="opt", state=model.init_state())
        cache = state.caches[0]
        per_token = (cache.k.numel() + cache.v.numel()) // cache.length
        assert per_token == 2 * case.n_kv_heads * HEAD_DIM


# --- gpu tier: peak memory at cache 8k is below MHA's (plan §5) --------------------------


def full_size_model(case: Case):
    return build_model(config.resolve(case.name, init_seed=1, peak_lr=1e-3)).eval()


def decode_at_cache_8k(case: Case, n_tokens: int):
    """Prefill a cache of 8k tokens, then decode `n_tokens` one at a time in fp16.
    Returns (peak bytes allocated during decode, seconds per token)."""
    model = full_size_model(case).cuda()
    prompt = torch.zeros(1, CACHE_8K, dtype=torch.long, device="cuda")
    token = torch.zeros(1, 1, dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        state = model.init_state(capacity=CACHE_8K + n_tokens + 8)
        for start in range(0, CACHE_8K, 1024):
            _, state = model(prompt[:, start : start + 1024], mode="opt", state=state)
        for _ in range(8):  # warm-up, discarded
            _, state = model(token, mode="opt", state=state)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        for _ in range(n_tokens):
            _, state = model(token, mode="opt", state=state)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - begin) / n_tokens
    peak = torch.cuda.max_memory_allocated()
    del model, state
    torch.cuda.empty_cache()
    return peak, seconds


@pytest.mark.gpu
@pytest.mark.parametrize("case", REDUCED, ids=lambda c: c.name)
def test_opt_decode_peak_memory_at_cache_8k_is_below_mha(case):
    mha_peak, _ = decode_at_cache_8k(MHA, 32)
    peak, _ = decode_at_cache_8k(case, 32)
    assert peak < mha_peak, (case.name, peak, mha_peak)


# --- t4 tier: decode time at cache 8k is below MHA's (plan §5) -----------------------------


@pytest.mark.t4
@pytest.mark.parametrize("case", REDUCED, ids=lambda c: c.name)
def test_opt_decode_time_at_cache_8k_is_below_mha(case):
    """Run alone on the box (plan §2.2). Three round-robin repetitions; the medians compare."""
    mha_times, times = [], []
    for _ in range(3):
        mha_times.append(decode_at_cache_8k(MHA, 256)[1])
        times.append(decode_at_cache_8k(case, 256)[1])
    assert sorted(times)[1] < sorted(mha_times)[1], (case.name, times, mha_times)
