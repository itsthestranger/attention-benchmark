"""T-07: sliding-window attention (plan §2.3, §5 "SWA window" and "SWA work skipping").

Written by the orchestrator before the implementation; the worker may not
change this file. The causal-leak, reference-agreement and KV-cache tests of
plan §5 cover this variant through `tests/variants.py`.

The contract for `src/attention/sliding_window.py`
--------------------------------------------------
    @register("sliding_window")
    class SlidingWindowAttention(AttnBase)
    def flex_compiled_available() -> bool

- Options: `n_kv_heads` (must divide `n_heads`) and `window` (W, a positive int;
  ValueError otherwise).
- The same four weights as grouped attention, with the same names and shapes:
  `w_q`, `w_k`, `w_v`, `w_o`. RoPE, scale, head order and grouping are the same.
  So a sliding-window model has exactly MHA's parameters when it has 8 KV heads.
- The window: query t attends to keys j with t - W < j <= t. That is W keys
  including itself, fewer near the start.
- `attention_weights(x, freqs)` returns the weights of the `ref` computation for
  a full forward, shape (B, H, T, T).
- `mode="ref"`: plain tensor operations with an explicit mask; no fused kernel.
- `mode="opt"` skips work outside the window (plan §13.8: a masked dense kernel
  is fused, correct and no faster). Two paths, chosen with `set_opt_path(name)`
  and readable as `opt_path`:
    * "chunked" (the default): queries are processed in chunks of at most W
      consecutive tokens; a chunk's keys are its own tokens and the W tokens
      before it. Every call to `F.scaled_dot_product_attention` therefore has
      at most W queries and at most 2 * W keys. This holds for a full forward
      and for a prefill of several tokens through a cache.
    * "flex": `torch.nn.attention.flex_attention.flex_attention` with a
      sliding-window block mask; compiled with `torch.compile` when the input
      is on a CUDA device, uncompiled otherwise (which is dense, and only good
      for checking the mask on CPU). Used for the full forward; decode through
      a cache always uses the chunked rules.
  `set_opt_path` raises ValueError for an unknown name.
- `flex_compiled_available()` is True when FlexAttention can be imported and a
  CUDA device is present. Whether it compiles on the T4 is checked on hardware.
- Decode: with one new token, `opt` passes the held keys and values (at most W,
  the new token included) to the fused kernel with no mask.
- The cache: see tests/test_kv_cache.py.
"""

import math
import statistics
import time

import pytest
import torch

from src import config
from src.attention.base import AttnConfig, build_attention, registered
from src.transformer import rope_freqs
from tests.tolerances import AGREEMENT_FP64, REF_OPT_FP16, REF_OPT_FP32, scale_relative_error
from tests.variants import HEAD_DIM, MODES, N_HEADS, REAL_WINDOW, Case, tiny_model, tokens

MHA = Case("mha", 8)
W = 16
SWA = Case("swa256", 8, W)
FLEX_WARNING = "ignore:flex_attention called without torch.compile"


def module(case=SWA, *, dtype=torch.float64, sharpen=6.0, layer=0):
    return tiny_model(case, sharpen=sharpen, dtype=dtype).layers[layer].attn


def inputs(T, *, B=2, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    d = N_HEADS * HEAD_DIM
    return torch.randn(B, T, d, generator=g, dtype=dtype), rope_freqs(HEAD_DIM, 0, T, 10000.0)


class Recorder:
    """Wraps `scaled_dot_product_attention` inside the sliding-window module."""

    def __init__(self):
        import src.attention.sliding_window as sw

        self.module, self.calls = sw, []

    def __enter__(self):
        self.original = self.module.F.scaled_dot_product_attention

        def recording(query, key, value, *args, **kwargs):
            mask = kwargs.get("attn_mask", args[0] if args else None)
            self.calls.append(
                {"queries": query.shape[-2], "keys": key.shape[-2], "masked": mask is not None}
            )
            return self.original(query, key, value, *args, **kwargs)

        self.module.F.scaled_dot_product_attention = recording
        return self

    def __exit__(self, *exc):
        self.module.F.scaled_dot_product_attention = self.original


# --- Parameters: the same as MHA ---------------------------------------------------------


def test_registered_with_mha_parameters():
    assert "sliding_window" in registered()
    swa, mha = module(), module(MHA)
    assert {n: tuple(p.shape) for n, p in swa.named_parameters()} == {
        n: tuple(p.shape) for n, p in mha.named_parameters()
    }
    assert swa.rope_dim == HEAD_DIM
    assert swa.residual_output_params() == {"w_o.weight"}
    assert swa.init_std_multipliers() == {}


def test_same_seed_gives_bit_identical_weights_to_mha():
    """Same names and shapes, so per-parameter seeding makes SWA start as MHA (plan §4.5)."""
    swa = dict(tiny_model(SWA, init_seed=3).named_parameters())
    mha = dict(tiny_model(MHA, init_seed=3).named_parameters())
    assert swa.keys() == mha.keys()
    for name in swa:
        assert torch.equal(swa[name], mha[name]), name


def test_full_size_matches_the_plan_table():
    cfg = AttnConfig.from_resolved(config.resolve("swa256", init_seed=1, peak_lr=1e-3))
    attn = build_attention(cfg)
    assert attn.n_params == 589_824  # plan §3.2
    assert attn.cfg.options["window"] == 256


@pytest.mark.parametrize("window", [0, -1, 2.5, True, None])
def test_invalid_window_raises(window):
    with pytest.raises(ValueError):
        tiny_model(SWA, **{"model.attention": {"type": "sliding_window", "n_kv_heads": 8, "window": window}})


def test_missing_window_and_bad_kv_heads_raise():
    with pytest.raises(ValueError):
        tiny_model(SWA, **{"model.attention": {"type": "sliding_window", "n_kv_heads": 8}})
    with pytest.raises(ValueError):
        tiny_model(SWA, **{"model.attention": {"type": "sliding_window", "n_kv_heads": 3, "window": 16}})


# --- The window (plan §5: weights outside are exactly zero, boundaries included) -----------


@pytest.mark.parametrize("T", [1, W - 1, W, W + 1, 3 * W + 5])
def test_weights_outside_the_window_are_exactly_zero(T):
    attn = module()
    x, freqs = inputs(T)
    with torch.no_grad():
        weights = attn.attention_weights(x, freqs)
    assert weights.shape == (2, N_HEADS, T, T)
    t = torch.arange(T)[:, None]
    j = torch.arange(T)[None, :]
    inside = (j <= t) & (t - j < W)
    assert torch.equal(weights[..., ~inside], torch.zeros_like(weights[..., ~inside]))
    assert (weights[..., inside] > 0).all()  # boundaries included: j = t and j = t - W + 1
    assert scale_relative_error(weights.sum(-1), torch.ones_like(weights.sum(-1))) <= AGREEMENT_FP64
    if T > W:
        last = T - 1
        assert (weights[..., last, last - W + 1] > 0).all()  # the oldest key inside
        assert (weights[..., last, last - W] == 0).all()  # the first key outside


def test_keys_per_query_average_224_at_the_plans_sizes():
    """Plan §2.3: ideal SWA-256 processes about 224 keys per query at ctx 1024."""
    t = torch.arange(1024)[:, None]
    j = torch.arange(1024)[None, :]
    inside = (j <= t) & (t - j < 256)
    assert inside.sum().item() / 1024 == 224.125


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("j", [0, 1, W - 1, W, W + 1, 2 * W - 1, 2 * W, 37])
def test_a_token_influences_exactly_the_next_window_of_outputs(mode, j):
    """Output level, so it holds for `opt` too: bit-identical outside, changed inside."""
    attn = module(dtype=torch.float32)
    T = 4 * W + 3
    x, freqs = inputs(T, dtype=torch.float32, seed=j)
    changed = x.clone()
    changed[:, j] += 1.0
    with torch.no_grad():
        base, _ = attn(x, freqs, mode=mode)
        out, _ = attn(changed, freqs, mode=mode)
    for t in range(T):
        same = torch.equal(out[:, t], base[:, t])
        assert same == (not j <= t <= j + W - 1), (j, t)


@pytest.mark.parametrize("mode", MODES)
def test_a_window_at_least_as_long_as_the_sequence_is_mha(mode):
    wide = Case("swa256", 8, 64)
    for T in (5, 63, 64):
        swa_model = tiny_model(wide, sharpen=6.0, dtype=torch.float64)
        mha_model = tiny_model(MHA, sharpen=6.0, dtype=torch.float64)
        idx = tokens(2, T, seed=T)
        with torch.no_grad():
            got, _ = swa_model(idx, mode=mode)
            expected, _ = mha_model(idx, mode=mode)
        assert scale_relative_error(got, expected) <= AGREEMENT_FP64, T


@pytest.mark.parametrize("mode", MODES)
def test_a_shorter_window_is_not_mha(mode):
    swa_model = tiny_model(SWA, sharpen=6.0, dtype=torch.float64)
    mha_model = tiny_model(MHA, sharpen=6.0, dtype=torch.float64)
    idx = tokens(2, 3 * W, seed=1)
    with torch.no_grad():
        got, _ = swa_model(idx, mode=mode)
        expected, _ = mha_model(idx, mode=mode)
    assert scale_relative_error(got[:, :W], expected[:, :W]) <= AGREEMENT_FP64  # before it slides
    assert scale_relative_error(got[:, W:], expected[:, W:]) > REF_OPT_FP32


@pytest.mark.parametrize("mode", MODES)
def test_output_equals_the_weights_applied_by_hand(mode):
    """`attention_weights` is what the module really uses."""
    attn = module()
    T = 2 * W + 7
    x, freqs = inputs(T, seed=5)
    with torch.no_grad():
        got, _ = attn(x, freqs, mode=mode)
        weights = attn.attention_weights(x, freqs)
        v = attn.w_v(x).view(2, T, N_HEADS, HEAD_DIM).transpose(1, 2)  # 8 KV heads here
        expected = attn.w_o((weights @ v).transpose(1, 2).reshape(2, T, N_HEADS * HEAD_DIM))
    assert scale_relative_error(got, expected) <= AGREEMENT_FP64


# --- Work skipping, structure (plan §5, cpu tier) --------------------------------------------


def test_ref_does_not_call_the_fused_kernel():
    attn = module()
    x, freqs = inputs(40)
    with torch.no_grad(), Recorder() as rec:
        attn(x, freqs, mode="ref")
    assert rec.calls == []


@pytest.mark.parametrize("T", [1, W - 1, W, W + 1, 2 * W, 5 * W, 5 * W + 3])
def test_chunked_path_never_scores_more_than_two_windows_of_keys(T):
    attn = module()
    assert attn.opt_path == "chunked"
    x, freqs = inputs(T)
    with torch.no_grad(), Recorder() as rec:
        attn(x, freqs, mode="opt")
    assert len(rec.calls) == math.ceil(T / W)
    assert sum(c["queries"] for c in rec.calls) == T
    for c in rec.calls:
        assert c["queries"] <= W and c["keys"] <= 2 * W, c
    # The first chunk has no predecessor; later full chunks see their own W and the W before.
    assert rec.calls[0]["keys"] == min(T, W)
    for c in rec.calls[1:]:
        assert c["keys"] == W + c["queries"]


def test_chunked_path_at_the_plans_sizes_scores_at_most_512_keys_per_query():
    """Plan §5: no query is scored against more than 512 keys. Plan §2.3: about 448 on average."""
    real = Case("swa256", 8, REAL_WINDOW)
    attn = tiny_model(real, context_length=1024).layers[0].attn
    x, freqs = inputs(1024, B=1, dtype=torch.float32)
    with torch.no_grad(), Recorder() as rec:
        attn(x, freqs, mode="opt")
    assert [(c["queries"], c["keys"]) for c in rec.calls] == [(256, 256), (256, 512), (256, 512), (256, 512)]
    assert max(c["keys"] for c in rec.calls) == 512
    assert sum(c["queries"] * c["keys"] for c in rec.calls) / 1024 == 448


@pytest.mark.parametrize("pieces", [[5 * W + 3], [3, 4 * W, W], [W + 1] * 4])
def test_prefill_through_a_cache_is_chunked_too(pieces):
    model = tiny_model(SWA)
    idx = tokens(1, sum(pieces), seed=2)
    state, start = model.init_state(), 0
    with torch.no_grad(), Recorder() as rec:
        for n in pieces:
            _, state = model(idx[:, start : start + n], mode="opt", state=state)
            start += n
    assert rec.calls
    for c in rec.calls:
        assert c["queries"] <= W and c["keys"] <= 2 * W, c


def test_single_token_decode_reads_at_most_one_window_without_a_mask():
    model = tiny_model(SWA)
    n = 3 * W
    idx = tokens(2, n, seed=3)
    state = model.init_state()
    with torch.no_grad(), Recorder() as rec:
        for t in range(n):
            _, state = model(idx[:, t : t + 1], mode="opt", state=state)
    layers = len(model.layers)
    assert len(rec.calls) == n * layers
    for i, c in enumerate(rec.calls):
        assert c["keys"] == min(i // layers + 1, W) and not c["masked"], (i, c)
    for cache in state.caches:
        assert cache.length == W and cache.capacity <= 2 * W
        assert tuple(cache.k.shape) == (2, N_HEADS, W, HEAD_DIM)


# --- The FlexAttention path -------------------------------------------------------------------


def test_opt_path_selection():
    attn = module()
    assert attn.opt_path == "chunked"
    attn.set_opt_path("flex")
    assert attn.opt_path == "flex"
    attn.set_opt_path("chunked")
    assert attn.opt_path == "chunked"
    with pytest.raises(ValueError):
        attn.set_opt_path("dense")


def test_flex_compiled_available_needs_cuda():
    from src.attention.sliding_window import flex_compiled_available

    assert flex_compiled_available() is torch.cuda.is_available()


@pytest.mark.filterwarnings(FLEX_WARNING)
@pytest.mark.parametrize("dtype, tol", [(torch.float32, REF_OPT_FP32), (torch.float64, AGREEMENT_FP64)], ids=["fp32", "fp64"])
@pytest.mark.parametrize("T", [W - 1, W, 3 * W + 5])
def test_flex_path_matches_ref_uncompiled_on_cpu(T, dtype, tol):
    """Uncompiled FlexAttention is dense, but it checks the window mask."""
    attn = module(dtype=dtype)
    x, freqs = inputs(T, dtype=dtype, seed=7)
    with torch.no_grad():
        ref, _ = attn(x, freqs, mode="ref")
        attn.set_opt_path("flex")
        flex, _ = attn(x, freqs, mode="opt")
    assert attn.opt_path == "flex"
    assert scale_relative_error(flex, ref) <= tol


@pytest.mark.filterwarnings(FLEX_WARNING)
def test_flex_path_respects_the_window_exactly():
    attn = module(dtype=torch.float32)
    attn.set_opt_path("flex")
    T, j = 4 * W, W + 3
    x, freqs = inputs(T, dtype=torch.float32, seed=8)
    changed = x.clone()
    changed[:, j] += 1.0
    with torch.no_grad():
        base, _ = attn(x, freqs, mode="opt")
        out, _ = attn(changed, freqs, mode="opt")
    for t in range(T):
        assert torch.equal(out[:, t], base[:, t]) == (not j <= t <= j + W - 1), t


# --- gpu tier -----------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("path", ["chunked", "flex"])
def test_opt_paths_match_ref_on_cuda(path):
    """The compiled FlexAttention path, and both paths under fp16."""
    model = tiny_model(SWA, sharpen=6.0).cuda()
    for layer in model.layers:
        layer.attn.set_opt_path(path)
    idx = tokens(3, 4 * W + 5, seed=1).cuda()
    with torch.no_grad():
        ref, _ = model(idx, mode="ref")
        opt, _ = model(idx, mode="opt")
        assert scale_relative_error(opt, ref) <= REF_OPT_FP32
        with torch.autocast("cuda", dtype=torch.float16):
            ref16, _ = model(idx, mode="ref")
            opt16, _ = model(idx, mode="opt")
    assert scale_relative_error(opt16.float(), ref16.float()) <= REF_OPT_FP16


# --- t4 tier: SWA attention time at ctx 1024 is below MHA's (plan §5) ----------------------------


def attention_seconds(attn, x, freqs, repeats=30, warmup=10):
    """Median seconds for one forward and backward of an attention module, fp16."""
    times = []
    for i in range(warmup + repeats):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            y, _ = attn(x, freqs, mode="opt")
        y.float().sum().backward()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - begin)
        attn.zero_grad(set_to_none=True)
    return statistics.median(times)


@pytest.mark.t4
@pytest.mark.parametrize("path", ["chunked", "flex"])
def test_swa_attention_time_at_ctx_1024_is_below_mha(path):
    """Catches a fused but dense SWA (plan §13.8). Run alone on the box (plan §2.2).

    Plan §2.3 expects the chunked path to gain little at ctx 1024; if this
    fails for "chunked" only, that is the plan's prediction and a result to
    report, not a test to loosen.
    """
    from src.attention.sliding_window import flex_compiled_available

    if path == "flex":
        assert flex_compiled_available(), "FlexAttention is not usable here; report it (plan §2.3)"
    mha = build_attention(AttnConfig.from_resolved(config.resolve("mha", init_seed=1, peak_lr=1e-3))).cuda()
    swa = build_attention(AttnConfig.from_resolved(config.resolve("swa256", init_seed=1, peak_lr=1e-3))).cuda()
    swa.set_opt_path(path)
    x = torch.randn(16, 1024, 384, device="cuda", requires_grad=True)
    freqs = rope_freqs(48, 0, 1024, 10000.0, device="cuda")
    mha_times, swa_times = [], []
    for _ in range(3):  # round-robin
        mha_times.append(attention_seconds(mha, x, freqs))
        swa_times.append(attention_seconds(swa, x, freqs))
    assert statistics.median(swa_times) < statistics.median(mha_times), (path, swa_times, mha_times)
