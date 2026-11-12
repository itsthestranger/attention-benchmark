"""Plan §5, KV-cache decode: cached token-by-token logits match the full
forward. Run at ctx 1024, because cache bugs hide at short lengths.

Written by the orchestrator before the implementation; the worker may not
change this file.

The cache contract (every attention module with a per-token cache)
------------------------------------------------------------------
- `attn.init_cache(capacity=None)` returns an empty cache. `capacity`, in
  tokens, is a hint to allocate room once instead of growing.
- `Transformer.init_state(capacity=None)` passes the hint to every layer.
- `attn(x, freqs, cache, mode=...)` with a cache attends to the cached tokens
  and, causally, to the tokens of `x`; it returns the output for `x` and a NEW
  cache object that also holds the tokens of `x`. `x` may hold one token or
  many (a prefill).
- Caches are values: the cache passed in still describes the same tokens after
  the call, and may be extended again (a second branch from the same state).
  An implementation may share storage between caches, but must copy before it
  would overwrite a token that another cache still describes.
- Without a capacity hint, storage grows geometrically, so decoding N tokens
  allocates O(log N) buffers, not N.
- A cache exposes `length` (tokens held) and `capacity` (tokens it has room for).
- A sliding-window cache holds the last `window` tokens only: `length` never
  exceeds `window`, `capacity` never exceeds 2 * `window` whatever the hint, and
  `k` and `v` are the held tokens in order. It is a value like any other cache.
"""

import math

import pytest
import torch

from tests.tolerances import AGREEMENT_FP64, KV_CACHE_DECODE_FP32, scale_relative_error
from tests.variants import CASES, MODES, REAL_WINDOW, VOCAB, tiny_model, tokens

CTX = 1024  # plan §3.1


def decode(model, idx, mode, pieces, capacity=None):
    """Feed `idx` through the cache in pieces of the given lengths; return all logits."""
    state = model.init_state() if capacity is None else model.init_state(capacity=capacity)
    out, start = [], 0
    for n in pieces:
        logits, state = model(idx[:, start : start + n], mode=mode, state=state)
        out.append(logits)
        start += n
    assert start == idx.shape[1] and state.pos == start
    return torch.cat(out, dim=1), state


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_token_by_token_decode_matches_the_full_forward_at_ctx_1024(case, mode):
    if case.window is not None:
        case = case._replace(window=REAL_WINDOW)  # the plan's window at the plan's context
    model = tiny_model(case, sharpen=6.0, context_length=CTX)
    idx = tokens(1, CTX, seed=11)
    with torch.no_grad():
        full, _ = model(idx, mode="ref")
        cached, state = decode(model, idx, mode, [1] * CTX)
    assert cached.shape == full.shape
    assert scale_relative_error(cached, full) <= KV_CACHE_DECODE_FP32
    # Not only on average: the last position, which reads the longest cache.
    assert scale_relative_error(cached[:, -1], full[:, -1]) <= KV_CACHE_DECODE_FP32
    assert all(cache.length == case.held(CTX) for cache in state.caches)


@pytest.mark.parametrize("decode_mode", MODES)
@pytest.mark.parametrize("full_mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_decode_matches_the_full_forward_in_float64(case, full_mode, decode_mode):
    model = tiny_model(case, sharpen=6.0, dtype=torch.float64)
    idx = tokens(2, 96, seed=12)
    with torch.no_grad():
        full, _ = model(idx, mode=full_mode)
        cached, _ = decode(model, idx, decode_mode, [1] * 96)
    assert scale_relative_error(cached, full) <= AGREEMENT_FP64


@pytest.mark.parametrize(
    "pieces",
    [[96], [1, 95], [95, 1], [40, 1, 1, 30, 24], [7] * 13 + [5]],
    ids=["all", "one-then-rest", "rest-then-one", "mixed", "sevens"],
)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_prefill_in_pieces_matches_the_full_forward(case, mode, pieces):
    """A piece of several tokens must be causal inside itself and see the whole cache."""
    model = tiny_model(case, sharpen=6.0, dtype=torch.float64)
    idx = tokens(2, 96, seed=13)
    with torch.no_grad():
        full, _ = model(idx, mode=mode)
        cached, _ = decode(model, idx, mode, pieces)
    assert scale_relative_error(cached, full) <= AGREEMENT_FP64


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_a_piece_is_causal_inside_itself(case, mode):
    model = tiny_model(case, sharpen=6.0)
    idx = tokens(2, 40, seed=14)
    changed = idx.clone()
    changed[:, 30:] = (changed[:, 30:] + 1) % VOCAB
    with torch.no_grad():
        a, _ = decode(model, idx, mode, [20, 20])
        b, _ = decode(model, changed, mode, [20, 20])
    assert torch.equal(a[:, :30], b[:, :30])
    assert not torch.equal(a[:, 30:], b[:, 30:])


@pytest.mark.parametrize("capacity", [None, 64], ids=["grown", "roomy"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_caches_are_values(case, mode, capacity):
    """Branching twice from one state: neither branch disturbs the other or the state.

    With spare capacity both branches could write the same slot of shared
    storage; the second one must copy instead.
    """
    model = tiny_model(case, sharpen=6.0, dtype=torch.float64)
    idx = tokens(2, 24, seed=15)
    other = (idx + 3) % VOCAB
    with torch.no_grad():
        full_a, _ = model(idx, mode=mode)
        full_b, _ = model(torch.cat([idx[:, :10], other[:, 10:]], dim=1), mode=mode)
        start = model.init_state() if capacity is None else model.init_state(capacity=capacity)
        _, shared = model(idx[:, :10], mode=mode, state=start)
        assert shared.pos == 10 and all(c.length == 10 for c in shared.caches)

        a1, state_a = model(idx[:, 10:11], mode=mode, state=shared)
        b1, state_b = model(other[:, 10:11], mode=mode, state=shared)  # second branch
        assert shared.pos == 10 and all(c.length == 10 for c in shared.caches)
        a_rest, _ = model(idx[:, 11:], mode=mode, state=state_a)
        b_rest, _ = model(other[:, 11:], mode=mode, state=state_b)
        a_again, _ = model(idx[:, 10:11], mode=mode, state=shared)  # the state is still intact

    assert scale_relative_error(torch.cat([a1, a_rest], 1), full_a[:, 10:]) <= AGREEMENT_FP64
    assert scale_relative_error(torch.cat([b1, b_rest], 1), full_b[:, 10:]) <= AGREEMENT_FP64
    assert torch.equal(a_again, a1)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_storage_grows_geometrically_without_a_hint(case):
    model = tiny_model(case)
    n = 300
    idx = tokens(1, n, seed=16)
    state = model.init_state()
    seen, states = set(), []
    with torch.no_grad():
        for t in range(n):
            _, state = model(idx[:, t : t + 1], mode="opt", state=state)
            states.append(state)  # keep every cache alive so addresses are not reused
            cache = state.caches[0]
            assert cache.length == case.held(t + 1) and cache.capacity >= cache.length
            seen.add(cache.capacity)
    assert len(seen) <= math.ceil(math.log2(n)) + 1
    if case.window is not None:
        assert max(seen) <= 2 * case.window


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_capacity_hint_allocates_once(case):
    model = tiny_model(case)
    n = 300
    idx = tokens(1, n, seed=16)
    state = model.init_state(capacity=n)
    seen = set()
    with torch.no_grad():
        for t in range(n):
            _, state = model(idx[:, t : t + 1], mode="opt", state=state)
            seen.add(state.caches[0].capacity)
        hinted, _ = decode(model, idx, "opt", [1] * n, capacity=n)
        plain, _ = decode(model, idx, "opt", [1] * n)
    if case.window is None:
        assert seen == {n}
    else:
        assert len(seen) == 1 and max(seen) <= 2 * case.window  # a window needs no more
    assert torch.equal(hinted, plain)  # the hint changes storage, not results


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_a_cache_built_for_one_batch_size_refuses_another(case, mode):
    model = tiny_model(case)
    with torch.no_grad():
        _, state = model(tokens(2, 5), mode=mode, state=model.init_state())
        with pytest.raises(ValueError):
            model(tokens(3, 1), mode=mode, state=state)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_decode_through_a_cache_is_inference_only(case, mode):
    """Cache storage is written in place, so it cannot carry gradients. Asking
    for them is an error, not a silently wrong gradient."""
    model = tiny_model(case)
    with pytest.raises(RuntimeError):
        model(tokens(1, 3), mode=mode, state=model.init_state())  # grad enabled
    with torch.no_grad():
        logits, state = model(tokens(1, 3), mode=mode, state=model.init_state())
    assert not logits.requires_grad and state.pos == 3
