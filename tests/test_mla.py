"""T-05: multi-head latent attention, naive path (plan §3.2, §3.3; §5 "MLA scale").

Written by the orchestrator before the implementation; the worker may not
change this file. The causal-leak, reference-agreement and KV-cache tests of
plan §5 cover this variant through `tests/variants.py`.

The contract for `src/attention/mla.py`
---------------------------------------
    @register("mla")
    class LatentAttention(AttnBase)

Options: `d_c` (KV latent width), `d_c_q` (query latent width), `rope_dim`
(d_h^R, even). With d = d_model, H = n_heads, dh = head_dim, r = rope_dim:

    w_dkv:   Linear(d, d_c)            norm_kv: RMSNorm(d_c)
    w_uk:    Linear(d_c, H * dh)       w_uv:    Linear(d_c, H * dh)
    w_kr:    Linear(d, r)
    w_dq:    Linear(d, d_c_q)          norm_q:  RMSNorm(d_c_q)
    w_uq:    Linear(d_c_q, H * dh)     w_qr:    Linear(d_c_q, H * r)
    w_o:     Linear(H * dh, d)

All bias-free; the norms are `src.transformer.RMSNorm`. Plan §3.3:

    c_KV = norm_kv(w_dkv(h))                     the cached latent
    k_R  = RoPE(w_kr(h))                         r wide, shared by all heads
    c_Q  = norm_q(w_dq(h))
    q_C  = w_uq(c_Q)    per head, dh wide        k_C = w_uk(c_KV)   per head, dh wide
    q_R  = RoPE(w_qr(c_Q))  per head, r wide     v   = w_uv(c_KV)   per head, dh wide

    score[h, t, j] = (q_C[h, t] . k_C[h, j] + q_R[h, t] . k_R[j]) / sqrt(dh + r)

Causal softmax over j; the head outputs (weights times v) are concatenated in
head order and passed through `w_o`. Projection outputs split into heads in
order, as in grouped attention. RoPE touches only q_R and k_R (decoupled RoPE),
with the `freqs` passed in; `rope_dim` is r.

- `residual_output_params()` is {"w_o.weight"}.
- `init_std_multipliers()` (plan §3.3): each up-projection out of a normalised
  latent gets sqrt(d / latent width): `w_uk.weight` and `w_uv.weight` with d_c,
  `w_uq.weight` and `w_qr.weight` with d_c_q. Down-projections and `w_kr` are
  not listed.
- `mode="ref"`: plain tensor operations, explicit mask, no fused kernel.
  `mode="opt"` (this task: the naive path): one call per layer to
  `F.scaled_dot_product_attention` with queries [q_C ; q_R] and keys
  [k_C ; k_R], both dh + r wide, and values dh wide; the kernel's own scale is
  then 1 / sqrt(dh + r).
- The cache holds c_KV and k_R only: d_c + r values per token per layer, as
  `cache.c_kv` (B, S, d_c) and `cache.k_r` (B, S, r), with `length`,
  `capacity` and `values_per_token`. It is a value, grows geometrically and
  takes a capacity hint, like every cache (tests/test_kv_cache.py). Decode
  materialises k_C and v from the cached latents (the naive path); the absorbed
  path is T-06.
"""

import math

import pytest
import torch

from src import config
from src.attention.base import AttnConfig, build_attention, registered
from src.model import Transformer, build_model
from src.transformer import RMSNorm, apply_rope, rope_freqs
from tests.tolerances import AGREEMENT_FP64, INIT_LOGIT_STD_REL, REF_OPT_FP32, scale_relative_error
from tests.variants import HEAD_DIM, MLA_CASES, MODES, N_HEADS, tiny_model, tokens

D = N_HEADS * HEAD_DIM
R = 4  # rope_dim of the tiny MLA models
MLA192, MLA96 = MLA_CASES


def module(case=MLA192, *, dtype=torch.float64, sharpen=6.0, layer=1):
    return tiny_model(case, sharpen=sharpen, dtype=dtype).layers[layer].attn


def inputs(T, *, B=2, dtype=torch.float64, seed=0, start=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g, dtype=dtype), rope_freqs(R, start, T, 10000.0)


def naive_mla(attn, x, freqs, scale=None):
    """Plan §3.3 written out one head and one query at a time."""
    B, T, _ = x.shape
    H, dh, r = N_HEADS, HEAD_DIM, R
    scale = math.sqrt(dh + r) if scale is None else scale
    c_kv = attn.norm_kv(x @ attn.w_dkv.weight.T)
    c_q = attn.norm_q(x @ attn.w_dq.weight.T)
    k_r = apply_rope(x @ attn.w_kr.weight.T, freqs)
    q_c_all, k_c_all = c_q @ attn.w_uq.weight.T, c_kv @ attn.w_uk.weight.T
    q_r_all, v_all = c_q @ attn.w_qr.weight.T, c_kv @ attn.w_uv.weight.T
    heads = []
    for h in range(H):
        q_c, k_c = q_c_all[..., h * dh : (h + 1) * dh], k_c_all[..., h * dh : (h + 1) * dh]
        q_r = apply_rope(q_r_all[..., h * r : (h + 1) * r], freqs)
        v = v_all[..., h * dh : (h + 1) * dh]
        out = torch.zeros(B, T, dh, dtype=x.dtype)
        for t in range(T):
            content = (k_c[:, : t + 1] @ q_c[:, t].unsqueeze(-1)).squeeze(-1)
            position = (k_r[:, : t + 1] @ q_r[:, t].unsqueeze(-1)).squeeze(-1)
            weights = torch.softmax((content + position) / scale, dim=-1)
            out[:, t] = (weights.unsqueeze(-1) * v[:, : t + 1]).sum(dim=1)
        heads.append(out)
    return torch.cat(heads, dim=-1) @ attn.w_o.weight.T


# --- Parameters (plan §3.2) ---------------------------------------------------------------


@pytest.mark.parametrize("name, per_layer, total", [("mla192", 562_560, 17_311_104), ("mla96", 359_616, 17_309_568)])
def test_full_size_parameter_counts_match_the_plan(name, per_layer, total):
    resolved = config.resolve(name, init_seed=1, peak_lr=1e-3)
    attn = build_attention(AttnConfig.from_resolved(resolved))
    assert attn.n_params == per_layer
    model = Transformer(resolved)
    assert model.n_params == total
    assert total - 17_308_032 == 16 * resolved["model"]["attention"]["d_c"]  # the latent norms


@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_parameter_names_and_shapes(case):
    assert "mla" in registered()
    attn = module(case)
    c = case.latent
    assert {n: tuple(p.shape) for n, p in attn.named_parameters()} == {
        "w_dkv.weight": (c, D),
        "norm_kv.gain": (c,),
        "w_uk.weight": (D, c),
        "w_uv.weight": (D, c),
        "w_kr.weight": (R, D),
        "w_dq.weight": (c, D),
        "norm_q.gain": (c,),
        "w_uq.weight": (D, c),
        "w_qr.weight": (N_HEADS * R, c),
        "w_o.weight": (D, D),
    }
    assert isinstance(attn.norm_kv, RMSNorm) and isinstance(attn.norm_q, RMSNorm)
    assert attn.rope_dim == R
    assert attn.residual_output_params() == {"w_o.weight"}


@pytest.mark.parametrize(
    "attention",
    [
        {"type": "mla", "d_c": 32, "d_c_q": 32},
        {"type": "mla", "d_c": 32, "rope_dim": 4},
        {"type": "mla", "d_c": 32, "d_c_q": 32, "rope_dim": 3},
        {"type": "mla", "d_c": 0, "d_c_q": 32, "rope_dim": 4},
        {"type": "mla", "d_c": 32, "d_c_q": -1, "rope_dim": 4},
    ],
)
def test_invalid_options_raise(attention):
    with pytest.raises(ValueError):
        tiny_model(MLA192, **{"model.attention": attention})


# --- The computation (plan §3.3) -------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_matches_the_written_out_formula(case, mode):
    attn = module(case)
    x, freqs = inputs(20, seed=1)
    with torch.no_grad():
        got, cache = attn(x, freqs, mode=mode)
        expected = naive_mla(attn, x, freqs)
    assert cache is None and got.shape == x.shape
    assert scale_relative_error(got, expected) <= AGREEMENT_FP64


@pytest.mark.parametrize("mode", MODES)
def test_softmax_scale_is_one_over_sqrt_of_head_dim_plus_rope_dim(mode):
    """Plan §5 "MLA scale": 1 / sqrt(d_h + d_h^R), not 1 / sqrt(d_h)."""
    attn = module()
    x, freqs = inputs(12, seed=2)
    with torch.no_grad():
        got, _ = attn(x, freqs, mode=mode)
        right = naive_mla(attn, x, freqs, scale=math.sqrt(HEAD_DIM + R))
        wrong = naive_mla(attn, x, freqs, scale=math.sqrt(HEAD_DIM))
    assert scale_relative_error(got, right) <= AGREEMENT_FP64
    assert scale_relative_error(got, wrong) > REF_OPT_FP32


def test_full_size_scale_is_one_over_sqrt_72():
    attn = build_attention(AttnConfig.from_resolved(config.resolve("mla192", init_seed=1, peak_lr=1e-3)))
    assert attn.cfg.head_dim + attn.rope_dim == 72


@pytest.mark.parametrize("mode", MODES)
def test_opt_path_structure(mode):
    """`ref` uses no fused kernel; `opt` makes one call with dh + r wide queries and keys."""
    import src.attention.mla as mla

    attn = module()
    x, freqs = inputs(10)
    calls = []
    original = mla.F.scaled_dot_product_attention

    def recording(query, key, value, *args, **kwargs):
        calls.append((tuple(query.shape), tuple(key.shape), tuple(value.shape), kwargs.get("is_causal", False)))
        return original(query, key, value, *args, **kwargs)

    mla.F.scaled_dot_product_attention = recording
    try:
        with torch.no_grad():
            attn(x, freqs, mode=mode)
    finally:
        mla.F.scaled_dot_product_attention = original
    if mode == "ref":
        assert calls == []
    else:
        wide = (2, N_HEADS, 10, HEAD_DIM + R)
        assert calls == [(wide, wide, (2, N_HEADS, 10, HEAD_DIM), True)]


# --- Decoupled RoPE ------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_output_depends_on_relative_position_only(mode):
    attn = module()
    x, _ = inputs(16, seed=3)
    with torch.no_grad():
        at_zero, _ = attn(x, rope_freqs(R, 0, 16, 10000.0), mode=mode)
        shifted, _ = attn(x, rope_freqs(R, 9, 16, 10000.0), mode=mode)
    assert scale_relative_error(shifted, at_zero) <= REF_OPT_FP32  # the table itself is float32


@pytest.mark.parametrize("mode", MODES)
def test_content_scores_carry_no_position(mode):
    """With the positional query silenced, positions cannot matter at all:
    RoPE must not touch q_C or k_C (plan §3.3)."""
    attn = module()
    x, _ = inputs(16, seed=4)
    with torch.no_grad():
        attn.w_qr.weight.zero_()
        a, _ = attn(x, rope_freqs(R, 0, 16, 10000.0), mode=mode)
        b, _ = attn(x, rope_freqs(R, 500, 16, 10000.0), mode=mode)
        c, _ = attn(x, rope_freqs(R, 0, 16, 3.0), mode=mode)  # another theta altogether
    assert scale_relative_error(b, a) <= AGREEMENT_FP64
    assert scale_relative_error(c, a) <= AGREEMENT_FP64


@pytest.mark.parametrize("mode", MODES)
def test_the_positional_key_is_shared_by_all_heads(mode):
    """One k_R for every head: with identical positional queries and no content
    scores, every head has the same attention weights."""
    attn = module()
    x, freqs = inputs(12, seed=5)
    captured = {}
    handle = attn.w_o.register_forward_hook(lambda _m, inp, _o: captured.update(heads=inp[0]))
    with torch.no_grad():
        attn.w_uq.weight.zero_()  # no content scores
        attn.w_qr.weight.copy_(attn.w_qr.weight[:R].repeat(N_HEADS, 1))  # same q_R in every head
        attn.w_uv.weight.copy_(attn.w_uv.weight[:HEAD_DIM].repeat(N_HEADS, 1))  # same values too
        attn(x, freqs, mode=mode)
    handle.remove()
    heads = captured["heads"].view(2, 12, N_HEADS, HEAD_DIM)
    for h in range(1, N_HEADS):
        assert scale_relative_error(heads[:, :, h], heads[:, :, 0]) <= AGREEMENT_FP64, h


# --- Latent norms and factor initialisation (plan §3.3) -----------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("down", ["w_dkv", "w_dq"])
def test_the_latent_norm_erases_the_scale_of_its_down_projection(mode, down):
    attn = module()
    x, freqs = inputs(12, seed=6)
    weight = getattr(attn, down).weight
    with torch.no_grad():
        weight.mul_(100.0)  # far above the norm's epsilon
        a, _ = attn(x, freqs, mode=mode)
        weight.mul_(3.0)
        b, _ = attn(x, freqs, mode=mode)
    assert scale_relative_error(b, a) <= REF_OPT_FP32


def test_the_latents_are_normalised():
    attn = module()
    x, freqs = inputs(12, seed=7)
    seen = {}
    hooks = [
        attn.norm_kv.register_forward_hook(lambda _m, _i, out: seen.update(kv=out)),
        attn.norm_q.register_forward_hook(lambda _m, _i, out: seen.update(q=out)),
    ]
    with torch.no_grad():
        attn.w_dkv.weight.mul_(100.0)
        attn.w_dq.weight.mul_(100.0)
        attn(x, freqs, mode="ref")
    for hook in hooks:
        hook.remove()
    for latent in (seen["kv"], seen["q"]):
        rms = latent.pow(2).mean(-1).sqrt()
        assert scale_relative_error(rms, torch.ones_like(rms)) <= REF_OPT_FP32


@pytest.mark.parametrize("name, d_c", [("mla192", 192), ("mla96", 96)])
def test_factor_init_multipliers_at_full_size(name, d_c):
    """Plan §3.3: up-projections use std 0.02 * sqrt(d / d_c): 0.02 * sqrt(2) and 0.04."""
    resolved = config.resolve(name, init_seed=1, peak_lr=1e-3)
    attn = build_attention(AttnConfig.from_resolved(resolved))
    m = math.sqrt(384 / d_c)
    assert attn.init_std_multipliers() == {
        "w_uk.weight": m, "w_uv.weight": m, "w_uq.weight": m, "w_qr.weight": m,
    }
    assert {192: 0.02 * math.sqrt(2), 96: 0.04}[d_c] == 0.02 * m


def test_each_factor_init_multiplier_uses_its_own_latent_width():
    """The plan sets d_c = d_c_q, which would hide a mix-up of the two widths."""
    resolved = config.resolve(
        "mla192", init_seed=1, peak_lr=1e-3,
        overrides={"model.attention": {"type": "mla", "d_c": 192, "d_c_q": 96, "rope_dim": 24}},
    )
    attn = build_attention(AttnConfig.from_resolved(resolved))
    kv, q = math.sqrt(384 / 192), math.sqrt(384 / 96)
    assert attn.init_std_multipliers() == {
        "w_uk.weight": kv, "w_uv.weight": kv, "w_uq.weight": q, "w_qr.weight": q,
    }
    assert attn.w_uk.weight.shape == (384, 192) and attn.w_uq.weight.shape == (384, 96)
    assert attn.w_qr.weight.shape == (8 * 24, 96)
    assert attn.init_cache().values_per_token == 192 + 24


@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_built_model_follows_the_factor_init(case):
    import zlib

    model = build_model(config.resolve(case.name, init_seed=4, peak_lr=1e-3))
    d_c = model.layers[0].attn.cfg.options["d_c"]
    up, base, out = 0.02 * math.sqrt(384 / d_c), 0.02, 0.02 / 4.0
    expected = {
        "w_uk.weight": up, "w_uv.weight": up, "w_uq.weight": up, "w_qr.weight": up,
        "w_dkv.weight": base, "w_dq.weight": base, "w_kr.weight": base, "w_o.weight": out,
    }
    params = dict(model.named_parameters())
    for local, std in expected.items():
        name = f"layers.5.attn.{local}"
        g = torch.Generator(device="cpu")
        g.manual_seed(zlib.crc32(name.encode()) ^ 4)
        draw = torch.empty(tuple(params[name].shape), dtype=torch.float32).normal_(generator=g)
        assert scale_relative_error(params[name].detach(), draw * std) <= REF_OPT_FP32, name
    for local in ("norm_kv.gain", "norm_q.gain"):
        gain = params[f"layers.5.attn.{local}"]
        assert torch.equal(gain, torch.ones_like(gain))


def test_up_projection_output_has_the_variance_of_a_dense_projection():
    """Why the multiplier: w_uk(c_KV) should look like a dense key projection of a
    normalised input, whose entries have variance d * 0.02^2."""
    model = build_model(config.resolve("mla192", init_seed=1, peak_lr=1e-3))
    attn = model.layers[0].attn
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 64, 384, generator=g)
    x = x / x.pow(2).mean(-1, keepdim=True).sqrt()  # what the block's norm hands to attention
    with torch.no_grad():
        k_c = attn.w_uk(attn.norm_kv(attn.w_dkv(x)))
        dense = x @ (torch.randn(384, 384, generator=g) * 0.02)
    assert abs(k_c.std().item() / dense.std().item() - 1.0) <= INIT_LOGIT_STD_REL


# --- The cache holds the latent and the positional key only (plan §3.2) --------------------------


def test_full_size_cache_holds_216_values_per_token_per_layer():
    model = build_model(config.resolve("mla192", init_seed=1, peak_lr=1e-3)).eval()
    with torch.no_grad():
        _, state = model(tokens(2, 5), mode="opt", state=model.init_state())
    assert len(state.caches) == 8
    for cache in state.caches:
        assert cache.length == 5 and cache.values_per_token == 216
        assert tuple(cache.c_kv.shape) == (2, 5, 192)
        assert tuple(cache.k_r.shape) == (2, 5, 24)
    # 216 values in fp16 are the 432 bytes of the plan's table.
    assert 216 * 2 == 432


@pytest.mark.parametrize("name, values", [("mla192", 216), ("mla96", 120)])
def test_cache_values_per_token(name, values):
    attn = build_attention(AttnConfig.from_resolved(config.resolve(name, init_seed=1, peak_lr=1e-3)))
    x = torch.zeros(1, 2, 384)
    with torch.no_grad():
        _, cache = attn(x, rope_freqs(24, 0, 2, 10000.0), attn.init_cache(), mode="ref")
    assert cache.values_per_token == values
    assert cache.c_kv.numel() + cache.k_r.numel() == 2 * values


@pytest.mark.parametrize("mode", MODES)
def test_cached_positional_keys_are_already_rotated(mode):
    """Decode at position p must not need the table for earlier positions."""
    attn = module()
    x, freqs = inputs(9, seed=8)
    with torch.no_grad():
        full, _ = attn(x, freqs, mode=mode)
        cache, outs = attn.init_cache(), []
        for t in range(9):
            y, cache = attn(x[:, t : t + 1], freqs[t : t + 1], cache, mode=mode)
            outs.append(y)
    assert scale_relative_error(torch.cat(outs, dim=1), full) <= AGREEMENT_FP64
