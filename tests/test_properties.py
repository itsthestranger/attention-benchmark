"""T-08: analytic properties of every attention module, and the FLOP counts built from them.

What is checked
---------------
* The decode-cache bytes per token of every variant at full size, against the
  plan's table, and against the bytes a real decode cache holds (tiny models).
* The cache cap of the sliding window and the cache-size helper.
* The FLOP counts of plan §3.1 (one FLOP is one multiply or one add): the
  projection and the score/value mixing of each module, the LM head, and the
  attention share of the forward FLOPs at 512, 1,024 and 2,048 tokens.
* The recorded fused-kernel name, which must not change the module's state.

Full-size models are built once per test session. Nothing here trains a model
or reads a real result: every number is analytic or measured on random tiny
models.
"""

from __future__ import annotations

import dataclasses
import functools
from fractions import Fraction

import pytest
import torch

from src import config
from src.attention.base import SDPA_BACKENDS, AttnBase, AttnConfig, build_attention, bytes_per_value
from src.attention.recurrence import GatedRecurrence  # registers "recurrence"
from src.model import build_model
from src.utils.flops import attention_share, model_flops_per_token
from tests.tolerances import FLOP_SHARE_POINTS
from tests.variants import CASES, MLA_ROPE_DIM, tiny_model, tokens

VARIANTS = config.variant_names()

# Plan §4.2 table, fp16 cache, bytes per token per layer.
PLAN_KV_BYTES = {
    "mha": 1536,
    "gqa4": 768,
    "gqa2": 384,
    "mqa": 192,
    "mla192": 432,
    "mla96": 240,
    "swa256": 1536,
    "null": 1536,
}
PLAN_WINDOW = 256  # plan §3.2, the sliding window at full size

# Plan §3.1: attention share of the forward FLOPs, percent, by (context, causal_skip).
PLAN_SHARE = {
    (512, False): 15,
    (512, True): 8,
    (1024, False): 27,
    (1024, True): 15,
    (2048, False): 42,
    (2048, True): 27,
}

# Full-size literal values (plan §3.1, §3.2).
MHA_PROJECTION = 1_179_648          # 2 * 589,824
MLA192_PROJECTION = 1_124_352       # 2 * (562,560 - 384): the two latent norm gains are not matrices
MHA_DENSE_1024 = 1_572_864          # 4 * 384 * 1024
MHA_CAUSAL_1024 = 787_200           # 2 * 384 * 1025
SWA_CAUSAL_1024 = 344_256           # round(4 * 384 * 224.125)
MLA192_DENSE_1024 = 1_966_080       # 2 * 8 * 72 * 1024 + 2 * 8 * 48 * 1024
LM_HEAD = 6_291_456                 # 2 * 384 * 8192
MHA_MLP = 8 * 2 * 3 * 384 * 1024    # 18,874,368 over 8 layers


@functools.lru_cache(maxsize=None)
def _full_model(name: str):
    return build_model(config.resolve(name, init_seed=1, peak_lr=1e-3))


@functools.lru_cache(maxsize=None)
def _full_attn(name: str):
    """The layer-0 attention of a full-size model. Shared: tests must not change it.

    Only the module is kept, so the other layers and the embedding are freed.
    """
    return build_model(config.resolve(name, init_seed=1, peak_lr=1e-3)).layers[0].attn


@functools.lru_cache(maxsize=None)
def _tiny_attn(case):
    return tiny_model(case).layers[0].attn


def _fresh(name: str):
    """A new attention module with the same config, for tests that change it."""
    return build_attention(_full_attn(name).cfg)


# --- Plan table: KV bytes, cache cap and cache size ------------------------------------------


def test_plan_table_covers_every_variant():
    assert set(PLAN_KV_BYTES) == set(VARIANTS)


@pytest.mark.parametrize("name", VARIANTS)
def test_kv_bytes_per_token_matches_the_plan_table(name):
    assert _full_attn(name).kv_bytes_per_token == PLAN_KV_BYTES[name]


@pytest.mark.parametrize("name", VARIANTS)
def test_kv_cache_max_tokens(name):
    expected = PLAN_WINDOW if name == "swa256" else None
    assert _full_attn(name).kv_cache_max_tokens == expected


@pytest.mark.parametrize("name", VARIANTS)
def test_kv_cache_bytes_at_long_and_short_contexts(name):
    attn = _full_attn(name)
    per_token = PLAN_KV_BYTES[name]
    if name == "swa256":
        assert attn.kv_cache_bytes(8192) == PLAN_WINDOW * 1536
    else:
        assert attn.kv_cache_bytes(8192) == 8192 * per_token
    # Below the window nothing is capped, for any variant.
    assert attn.kv_cache_bytes(100) == 100 * per_token


def test_mha_cache_is_twelve_kilobytes_per_token_over_eight_layers():
    assert _full_attn("mha").kv_bytes_per_token * 8 == 12_288


def test_kv_bytes_ordering_matches_the_plan():
    order = sorted(["mha", "gqa4", "gqa2", "mqa", "mla192", "mla96"], key=lambda n: _full_attn(n).kv_bytes_per_token)
    assert order == ["mqa", "mla96", "gqa2", "mla192", "gqa4", "mha"]
    assert [_full_attn(n).kv_bytes_per_token for n in order] == [192, 240, 384, 432, 768, 1536]


@pytest.mark.parametrize("n_tokens", [-1, 1.5, True])
def test_kv_cache_bytes_rejects_a_bad_token_count(n_tokens):
    with pytest.raises(ValueError):
        _full_attn("mha").kv_cache_bytes(n_tokens)


@pytest.mark.parametrize("name", VARIANTS)
def test_float32_cache_doubles_the_bytes(name):
    attn = _full_attn(name)
    fp32 = build_attention(dataclasses.replace(attn.cfg, kv_dtype="float32"))
    assert fp32.kv_bytes_per_token == 2 * attn.kv_bytes_per_token
    assert fp32.kv_cache_bytes(8192) == 2 * attn.kv_cache_bytes(8192)


@pytest.mark.parametrize("name", VARIANTS)
def test_unknown_kv_dtype_name_raises(name):
    attn = build_attention(dataclasses.replace(_full_attn(name).cfg, kv_dtype="float7"))
    with pytest.raises(ValueError):
        _ = attn.kv_bytes_per_token


@pytest.mark.parametrize(
    "dtype_name, width", [("float16", 2), ("bfloat16", 2), ("float32", 4), ("float64", 8)]
)
def test_bytes_per_value_of_floating_point_names(dtype_name, width):
    assert bytes_per_value(dtype_name) == width


@pytest.mark.parametrize("dtype_name", ["int8", "half_precision", "", None])
def test_bytes_per_value_rejects_other_names(dtype_name):
    with pytest.raises(ValueError):
        bytes_per_value(dtype_name)


# --- The cache holds what the bytes say it holds ---------------------------------------------


@pytest.mark.parametrize("n_tokens", [6, 20])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_kv_bytes_per_token_match_the_real_decode_cache(case, n_tokens):
    # fp16 tiny model, so the cache's own tensors give the bytes. Batch of 2: bytes scale with it.
    model = tiny_model(case, dtype=torch.float16)
    attn = model.layers[0].attn
    with torch.no_grad():
        _, state = model(tokens(2, n_tokens), state=model.init_state())
    cache = state.caches[0]
    held = case.held(n_tokens)  # 20 > the tiny window 16 only for swa256
    assert cache.length == held
    stored = [cache.k, cache.v] if case.latent is None else [cache.c_kv, cache.k_r]
    measured = sum(t.numel() * t.element_size() for t in stored)
    assert measured == 2 * held * attn.kv_bytes_per_token


# --- Projection FLOPs --------------------------------------------------------------------------


@pytest.mark.parametrize("name", VARIANTS)
def test_projection_flops_match_the_weight_shapes(name):
    attn = _full_attn(name)
    d, H, dh = attn.cfg.d_model, attn.cfg.n_heads, attn.cfg.head_dim
    if attn.cfg.type == "mla":
        dc, dcq, r = attn.d_c, attn.d_c_q, attn.rope_dim
        matrices = (
            d * dc            # w_dkv
            + 2 * dc * H * dh  # w_uk, w_uv
            + d * r           # w_kr
            + d * dcq         # w_dq
            + dcq * H * dh    # w_uq
            + dcq * H * r     # w_qr
            + H * dh * d     # w_o
        )
    else:
        G = attn.n_kv_heads
        matrices = 2 * d * H * dh + 2 * d * G * dh  # w_q, w_o and w_k, w_v
    assert attn.projection_flops_per_token() == 2 * matrices


def test_projection_flops_literal_values():
    assert _full_attn("mha").projection_flops_per_token() == MHA_PROJECTION
    assert _full_attn("mla192").projection_flops_per_token() == MLA192_PROJECTION


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("T", [1, 17, 1024])
@pytest.mark.parametrize("causal_skip", [True, False])
def test_flops_per_token_is_projection_plus_mixing(name, T, causal_skip):
    attn = _full_attn(name)
    assert attn.flops_per_token(T, causal_skip) == (
        attn.projection_flops_per_token() + attn.mixing_flops_per_token(T, causal_skip)
    )


# --- Mixing FLOPs: literal values and identities -----------------------------------------------


def test_mha_mixing_literals_at_1024():
    mha = _full_attn("mha")
    assert mha.mixing_flops_per_token(1024, causal_skip=False) == 4 * 384 * 1024 == MHA_DENSE_1024
    assert mha.mixing_flops_per_token(1024, causal_skip=True) == 2 * 384 * 1025 == MHA_CAUSAL_1024


def test_sliding_window_mixing_literal_at_1024():
    swa = _full_attn("swa256")
    assert swa.mixing_flops_per_token(1024, causal_skip=True) == SWA_CAUSAL_1024
    # Dense: every query is scored against every key, the window does not change the count.
    assert swa.mixing_flops_per_token(1024, causal_skip=False) == MHA_DENSE_1024


def test_mla_mixing_literal_at_1024():
    mla = _full_attn("mla192")
    assert mla.mixing_flops_per_token(1024, causal_skip=False) == (
        2 * 8 * 72 * 1024 + 2 * 8 * 48 * 1024
    ) == MLA192_DENSE_1024


@pytest.mark.parametrize("name", ["gqa4", "gqa2", "mqa"])
@pytest.mark.parametrize("T", [1, 5, 1024])
@pytest.mark.parametrize("causal_skip", [True, False])
def test_fewer_kv_heads_do_not_change_the_mixing(name, T, causal_skip):
    assert _full_attn(name).mixing_flops_per_token(T, causal_skip) == (
        _full_attn("mha").mixing_flops_per_token(T, causal_skip)
    )


_MIXING_T = [1, 2, 3, 64, 255, 256, 257, 512, 1023, 1024, 2048]


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("causal_skip", [True, False])
def test_mixing_is_nondecreasing_in_T(name, causal_skip):
    attn = _full_attn(name)
    values = [attn.mixing_flops_per_token(T, causal_skip) for T in _MIXING_T]
    assert values == sorted(values)


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("T", _MIXING_T)
def test_dense_count_is_at_least_the_causal_count(name, T):
    attn = _full_attn(name)
    assert attn.mixing_flops_per_token(T, causal_skip=False) >= attn.mixing_flops_per_token(T, True)


@pytest.mark.parametrize("name", VARIANTS)
def test_dense_and_causal_agree_at_one_token(name):
    attn = _full_attn(name)
    assert attn.mixing_flops_per_token(1, causal_skip=False) == attn.mixing_flops_per_token(1, True)


@pytest.mark.parametrize("T", [1, 2, 255, 256])
def test_sliding_window_causal_equals_mha_up_to_the_window(T):
    assert _full_attn("swa256").mixing_flops_per_token(T, True) == _full_attn("mha").mixing_flops_per_token(T, True)


@pytest.mark.parametrize("T", [257, 1024])
def test_sliding_window_causal_is_smaller_beyond_the_window(T):
    assert _full_attn("swa256").mixing_flops_per_token(T, True) < _full_attn("mha").mixing_flops_per_token(T, True)


def _slow_mixing(attn, case, T: int, causal_skip: bool) -> int:
    """One query at a time: the same count, written as the plain sum it stands for."""
    score_width = attn.head_dim + (MLA_ROPE_DIM if case.latent is not None else 0)
    per_key = 2 * attn.n_heads * (score_width + attn.head_dim)
    total = 0
    for t in range(T):
        if not causal_skip:
            keys = T
        elif case.window is None:
            keys = t + 1
        else:
            keys = min(t + 1, case.window)
        total += per_key * keys
    return int(Fraction(total, T) + Fraction(1, 2))  # total / T, rounded half up


@pytest.mark.parametrize("T", [1, 2, 5, 15, 16, 17, 33, 50])
@pytest.mark.parametrize("causal_skip", [True, False])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_mixing_matches_a_slow_loop_over_queries(case, T, causal_skip):
    attn = _tiny_attn(case)
    assert attn.mixing_flops_per_token(T, causal_skip) == _slow_mixing(attn, case, T, causal_skip)


# --- Model-level FLOPs and the attention share -------------------------------------------------


def test_model_flops_keys_and_total():
    counts = model_flops_per_token(_full_model("mha"), 1024)
    assert set(counts) == {"attention_mixing", "attention_projection", "mlp", "lm_head", "total"}
    assert all(isinstance(v, int) for v in counts.values())
    assert counts["total"] == (
        counts["attention_mixing"] + counts["attention_projection"] + counts["mlp"] + counts["lm_head"]
    )


def test_mha_model_flops_literals_at_1024():
    counts = model_flops_per_token(_full_model("mha"), 1024)
    assert counts["lm_head"] == LM_HEAD == 6_291_456
    assert counts["mlp"] == MHA_MLP == 18_874_368
    assert counts["attention_projection"] == 8 * MHA_PROJECTION
    assert counts["attention_mixing"] == 8 * MHA_CAUSAL_1024
    dense = model_flops_per_token(_full_model("mha"), 1024, causal_skip=False)
    assert dense["attention_mixing"] == 8 * MHA_DENSE_1024


def test_model_flops_on_a_six_layer_ladder_model():
    model = build_model(config.resolve("mha", ladder="mha_L6", init_seed=1, peak_lr=1e-3))
    assert len(model.layers) == 6
    counts = model_flops_per_token(model, 1024)
    assert counts["mlp"] == 6 * 2 * 3 * 384 * 1024
    assert counts["attention_mixing"] == 6 * MHA_CAUSAL_1024
    assert counts["attention_projection"] == 6 * MHA_PROJECTION
    assert counts["lm_head"] == LM_HEAD


@pytest.mark.parametrize(("T", "causal_skip"), sorted(PLAN_SHARE))
def test_attention_share_of_forward_flops_matches_the_plan(T, causal_skip):
    share = attention_share(_full_model("mha"), T, causal_skip)
    plan_value = PLAN_SHARE[(T, causal_skip)]
    assert abs(100 * share - plan_value) <= FLOP_SHARE_POINTS


def test_attention_share_is_mixing_over_total():
    model = _full_model("mha")
    counts = model_flops_per_token(model, 512, causal_skip=True)
    assert attention_share(model, 512, causal_skip=True) == counts["attention_mixing"] / counts["total"]


# --- Fused-kernel record -----------------------------------------------------------------------


@pytest.mark.parametrize("name", VARIANTS)
def test_sdpa_backend_is_none_until_recorded(name):
    assert _fresh(name).sdpa_backend is None


@pytest.mark.parametrize("backend", [*SDPA_BACKENDS, None])
def test_set_sdpa_backend_accepts_the_known_names(backend):
    attn = _fresh("mha")
    attn.set_sdpa_backend(backend)
    assert attn.sdpa_backend == backend


@pytest.mark.parametrize("backend", ["triton", "Flash", "", 1])
def test_set_sdpa_backend_rejects_other_names(backend):
    attn = _fresh("mha")
    with pytest.raises(ValueError):
        attn.set_sdpa_backend(backend)
    assert attn.sdpa_backend is None


def test_sdpa_backend_leaves_the_state_dict_unchanged():
    attn = _fresh("mha")
    keys_before = list(attn.state_dict().keys())
    buffers_before = [name for name, _ in attn.named_buffers()]
    attn.set_sdpa_backend("flash")
    assert list(attn.state_dict().keys()) == keys_before
    assert [name for name, _ in attn.named_buffers()] == buffers_before


def _recurrence_module():
    resolved = config.resolve(
        "mha", init_seed=1, peak_lr=1e-3, overrides={"model.attention": {"type": "recurrence"}}
    )
    rec = build_attention(AttnConfig.from_resolved(resolved))
    assert isinstance(rec, GatedRecurrence)
    return rec


def test_recurrence_has_no_cache_flops_or_kernel():
    rec = _recurrence_module()
    assert rec.kv_bytes_per_token == 0
    assert rec.kv_cache_max_tokens == 0
    assert rec.kv_cache_bytes(10) == 0
    with pytest.raises(NotImplementedError):
        rec.flops_per_token(16)
    assert rec.sdpa_backend is None


def test_recurrence_accepts_none_and_rejects_a_kernel_name():
    rec = _recurrence_module()
    rec.set_sdpa_backend(None)
    with pytest.raises(ValueError):
        rec.set_sdpa_backend("flash")


class _Bare(AttnBase):
    """An unregistered subclass that overrides nothing: the base-class defaults."""


def test_base_class_defaults():
    bare = _Bare(_full_attn("mha").cfg)
    assert bare.kv_cache_max_tokens is None
    assert bare.sdpa_backend is None
    with pytest.raises(NotImplementedError):
        _ = bare.kv_bytes_per_token


# --- Invalid context lengths -------------------------------------------------------------------


@pytest.mark.parametrize("T", [0, -1, 1.0, 2.5, True])
@pytest.mark.parametrize("name", ["mha", "swa256", "mla192"])
def test_invalid_context_length_raises(name, T):
    attn = _full_attn(name)
    with pytest.raises(ValueError):
        attn.flops_per_token(T)
    with pytest.raises(ValueError):
        attn.mixing_flops_per_token(T)
