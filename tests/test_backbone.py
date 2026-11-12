"""T-02: the backbone is identical for every variant; only the attention module is swapped.

Real attention types are not registered yet, so the toy attentions below are
registered for this module only and removed again afterwards.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.attention.base import AttnBase, AttnConfig, build_attention, register, registered, unregister
from src.model import DecodeState, Transformer
from src.transformer import RMSNORM_EPS, Block, RMSNorm, SwiGLU, apply_rope, rope_freqs
from tests.tolerances import FLOAT64_IDENTITY, REF_OPT_FP32, scale_relative_error


# --- Toy attentions ----------------------------------------------------------


class ToyCausal(AttnBase):
    """y[t] = proj(mean of x[0..t]); keeps a running sum in its cache."""

    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.last_freqs = None
        self.last_mode = None

    @property
    def rope_dim(self) -> int:
        return self.cfg.head_dim

    def init_cache(self):
        return {"sum": None, "n": 0}

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        self.last_freqs = freqs
        self.last_mode = mode
        T = x.shape[1]
        prev_sum, n = (None, 0) if cache is None else (cache["sum"], cache["n"])
        cs = x.cumsum(dim=1)
        if prev_sum is not None:
            cs = cs + prev_sum.unsqueeze(1)
        counts = torch.arange(n + 1, n + T + 1, device=x.device, dtype=x.dtype)
        y = self.proj(cs / counts.view(1, T, 1))
        if cache is None:
            return y, None
        # A new dict, so the cache the caller passed in is never mutated.
        return y, {"sum": cs[:, -1].clone(), "n": n + T}


class ToyZero(AttnBase):
    """No parameters; returns zeros and the cache unchanged."""

    @property
    def rope_dim(self) -> int:
        return self.cfg.head_dim

    def init_cache(self):
        return None

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        return torch.zeros_like(x), cache


@pytest.fixture(scope="module", autouse=True)
def _toy_attentions_registered():
    register("toy_causal")(ToyCausal)
    register("toy_zero")(ToyZero)
    yield
    unregister("toy_causal")
    unregister("toy_zero")


# --- Helpers -----------------------------------------------------------------


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(shape, seed=0, dtype=torch.float32, scale=1.0):
    return torch.randn(shape, generator=_gen(seed), dtype=dtype) * scale


def _resolved(attention="toy_causal", n_layers=2, **model_overrides):
    """A tiny resolved config; keyword arguments override fields of ``model``.

    ``attention`` is a registered type name, or a full attention dict.
    """
    attention_dict = attention if isinstance(attention, dict) else {"type": attention}
    overrides = {
        "model.attention": attention_dict,
        "model.d_model": 16,
        "model.n_heads": 2,
        "model.head_dim": 8,
        "model.n_layers": n_layers,
        "model.vocab_size": 32,
        "model.d_ff": 24,
        "model.context_length": 64,
    }
    overrides.update({f"model.{k}": v for k, v in model_overrides.items()})
    return config.resolve("mha", init_seed=0, peak_lr=1e-3, overrides=overrides)


def _cfg(**kwargs) -> AttnConfig:
    fields = dict(
        type="toy_causal",
        d_model=16,
        n_heads=2,
        head_dim=8,
        n_layers=2,
        context_length=64,
        rope_theta=10000.0,
        kv_dtype="float16",
        options={},
    )
    fields.update(kwargs)
    return AttnConfig(**fields)


def _rope64(rope_dim, start, length, theta):
    """A float64 RoPE table, built here so that the float64 identities hold to rounding."""
    inv_freq = theta ** (-2.0 * torch.arange(rope_dim // 2, dtype=torch.float64) / rope_dim)
    positions = torch.arange(start, start + length, dtype=torch.float64)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


# --- 1. RMSNorm --------------------------------------------------------------


def test_rmsnorm_has_one_gain_parameter_of_ones():
    norm = RMSNorm(16)
    assert [n for n, _ in norm.named_parameters()] == ["gain"]
    assert tuple(norm.gain.shape) == (16,)
    assert torch.equal(norm.gain, torch.ones(16))


def test_rmsnorm_output_rms_is_one_at_large_scale():
    # RMSNorm adds RMSNORM_EPS inside the square root, so the output RMS is
    # sqrt(ms / (ms + eps)). At unit input scale that is about 1 - 5e-7; at input
    # scale 1e4 the mean square is 1e8 and the deviation drops below the float64 bound.
    norm = RMSNorm(16).double().requires_grad_(False)
    x = _randn((4, 5, 16), seed=1, dtype=torch.float64, scale=1e4)
    rms = norm(x).pow(2).mean(dim=-1).sqrt()
    assert scale_relative_error(rms, torch.ones_like(rms)) <= FLOAT64_IDENTITY


def test_rmsnorm_matches_hand_formula():
    norm = RMSNorm(16).double().requires_grad_(False)
    with torch.no_grad():
        norm.gain.copy_(_randn((16,), seed=2, dtype=torch.float64))
    x = _randn((3, 4, 16), seed=3, dtype=torch.float64)
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    expected = x / torch.sqrt(ms + RMSNORM_EPS) * norm.gain
    assert scale_relative_error(norm(x), expected) <= FLOAT64_IDENTITY


def test_rmsnorm_is_scale_invariant_to_tolerance():
    norm = RMSNorm(16).double().requires_grad_(False)
    x = _randn((2, 3, 16), seed=4, dtype=torch.float64, scale=1e4)
    assert scale_relative_error(norm(3.0 * x), norm(x)) <= FLOAT64_IDENTITY


def test_rmsnorm_dtypes():
    norm = RMSNorm(16)
    assert norm(_randn((2, 16), seed=5, dtype=torch.float32)).dtype == torch.float32
    assert norm(_randn((2, 16), seed=5, dtype=torch.float64)).dtype == torch.float64
    half = RMSNorm(16).half()
    assert half(_randn((2, 16), seed=5, dtype=torch.float16)).dtype == torch.float16


# --- 2. SwiGLU ---------------------------------------------------------------


def test_swiglu_equals_hand_computation_exactly():
    mlp = SwiGLU(6, 10)
    x = _randn((2, 3, 6), seed=6)
    expected = F.linear(
        F.silu(F.linear(x, mlp.gate.weight)) * F.linear(x, mlp.up.weight),
        mlp.down.weight,
    )
    assert torch.equal(mlp(x), expected)


def test_swiglu_parameter_names_shapes_and_count():
    d, d_ff = 6, 10
    mlp = SwiGLU(d, d_ff)
    shapes = {n: tuple(p.shape) for n, p in mlp.named_parameters()}
    assert shapes == {
        "gate.weight": (d_ff, d),
        "up.weight": (d_ff, d),
        "down.weight": (d, d_ff),
    }
    assert sum(p.numel() for p in mlp.parameters()) == 3 * d * d_ff


# --- 3. rope_freqs -----------------------------------------------------------


def test_rope_freqs_shape_and_dtype():
    f = rope_freqs(8, 0, 5, 10000.0)
    assert tuple(f.shape) == (5, 4, 2)
    assert f.dtype == torch.float32


def test_rope_freqs_position_zero_is_cos_one_sin_zero():
    f = rope_freqs(8, 0, 1, 10000.0)
    assert torch.equal(f[0, :, 0], torch.ones(4))
    assert torch.equal(f[0, :, 1], torch.zeros(4))


def test_rope_freqs_pair_zero_angle_is_position():
    f = rope_freqs(8, 3, 1, 10000.0)
    assert scale_relative_error(float(f[0, 0, 0]), math.cos(3.0)) <= REF_OPT_FP32
    assert scale_relative_error(float(f[0, 0, 1]), math.sin(3.0)) <= REF_OPT_FP32


def test_rope_freqs_pair_one_at_position_ten():
    # theta = 1e4, rope_dim = 8: pair 1 has frequency 1e4 ** (-0.25) = 0.1, so position 10 gives angle 1.0.
    f = rope_freqs(8, 10, 1, 10000.0)
    assert scale_relative_error(float(f[0, 1, 0]), math.cos(1.0)) <= REF_OPT_FP32
    assert scale_relative_error(float(f[0, 1, 1]), math.sin(1.0)) <= REF_OPT_FP32


def test_rope_freqs_offset_is_bit_identical_to_slice():
    full = rope_freqs(8, 0, 8, 10000.0)
    assert torch.equal(rope_freqs(8, 5, 3, 10000.0), full[5:8])


@pytest.mark.parametrize("rope_dim", [7, 0, -2])
def test_rope_freqs_rejects_odd_or_non_positive_dim(rope_dim):
    with pytest.raises(ValueError):
        rope_freqs(rope_dim, 0, 4, 10000.0)


# --- 4. apply_rope -----------------------------------------------------------


def test_apply_rope_position_zero_is_identity():
    x = _randn((2, 1, 8), seed=7, dtype=torch.float64)
    assert torch.equal(apply_rope(x, rope_freqs(8, 0, 1, 10000.0)), x)


def test_apply_rope_preserves_each_pair_norm():
    x = _randn((2, 5, 8), seed=8, dtype=torch.float64)
    y = apply_rope(x, _rope64(8, 2, 5, 10000.0))
    pair_norm = lambda t: t.reshape(*t.shape[:-1], -1, 2).pow(2).sum(dim=-1)
    assert scale_relative_error(pair_norm(y), pair_norm(x)) <= FLOAT64_IDENTITY


def test_apply_rope_depends_only_on_relative_position():
    theta, dim = 10000.0, 8
    q = _randn((1, 1, dim), seed=9, dtype=torch.float64)
    k = _randn((1, 1, dim), seed=10, dtype=torch.float64)

    def dot(a, b):
        rq = apply_rope(q, _rope64(dim, a, 1, theta))
        rk = apply_rope(k, _rope64(dim, b, 1, theta))
        return (rq * rk).sum()

    base = dot(3, 7)
    assert scale_relative_error(dot(8, 12), base) <= FLOAT64_IDENTITY
    assert scale_relative_error(dot(50, 54), base) <= FLOAT64_IDENTITY


def test_apply_rope_hand_computed_rotation():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    # Quarter turn: cos 0, sin 1. y0 = 1*0 - 2*1 = -2, y1 = 1*1 + 2*0 = 1.
    quarter = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    assert torch.equal(apply_rope(x, quarter), torch.tensor([[-2.0, 1.0]], dtype=torch.float64))
    # cos 0.6, sin 0.8: y0 = 0.6 - 1.6 = -1.0, y1 = 0.8 + 1.2 = 2.0.
    angle = torch.tensor([[[0.6, 0.8]]], dtype=torch.float64)
    expected = torch.tensor([[-1.0, 2.0]], dtype=torch.float64)
    assert scale_relative_error(apply_rope(x, angle), expected) <= FLOAT64_IDENTITY


def test_apply_rope_output_dtype_follows_input():
    x = _randn((2, 3, 8), seed=11, dtype=torch.float16)
    y = apply_rope(x, rope_freqs(8, 0, 3, 10000.0))
    assert y.dtype == torch.float16
    assert y.shape == x.shape


def test_apply_rope_shape_mismatch_raises():
    x = _randn((1, 4), seed=12, dtype=torch.float64)
    with pytest.raises(ValueError):
        apply_rope(x, rope_freqs(4, 0, 2, 10000.0))  # two positions for one token
    with pytest.raises(ValueError):
        apply_rope(x, rope_freqs(8, 0, 1, 10000.0))  # wrong rope width


# --- 5. Registry and config --------------------------------------------------


def test_build_attention_returns_registered_class():
    cfg = AttnConfig.from_resolved(_resolved("toy_causal"))
    attn = build_attention(cfg)
    assert isinstance(attn, ToyCausal)
    assert attn.cfg is cfg


def test_build_attention_unknown_type_names_it():
    with pytest.raises(ValueError, match="no_such_attention"):
        build_attention(_cfg(type="no_such_attention"))


def test_register_duplicate_name_raises():
    with pytest.raises(ValueError):
        register("toy_causal")(ToyZero)
    assert build_attention(_cfg(type="toy_causal")).__class__ is ToyCausal


def test_register_non_attn_class_raises():
    with pytest.raises(ValueError):
        register("toy_plain")(nn.Linear)
    assert "toy_plain" not in registered()


def test_unregister_removes_name_and_missing_raises():
    register("toy_temp")(ToyZero)
    assert "toy_temp" in registered()
    unregister("toy_temp")
    assert "toy_temp" not in registered()
    with pytest.raises(KeyError):
        unregister("toy_temp")


def test_from_resolved_fields_and_options_exclude_type():
    resolved = _resolved({"type": "toy_causal", "window": 4})
    cfg = AttnConfig.from_resolved(resolved)
    assert cfg.type == "toy_causal"
    assert cfg.d_model == 16
    assert cfg.n_heads == 2
    assert cfg.head_dim == 8
    assert cfg.n_layers == 2
    assert cfg.context_length == 64
    assert cfg.rope_theta == resolved["model"]["rope_theta"] == 10000.0
    assert cfg.kv_dtype == "float16"
    assert dict(cfg.options) == {"window": 4}
    assert "type" not in cfg.options


# --- 6. AttnBase defaults ----------------------------------------------------


def test_attn_base_unimplemented_methods_raise():
    base = AttnBase(_cfg())
    x = _randn((1, 2, 16), seed=13)
    with pytest.raises(NotImplementedError):
        base.forward(x, rope_freqs(8, 0, 2, 10000.0))
    with pytest.raises(NotImplementedError):
        base.init_cache()
    with pytest.raises(NotImplementedError):
        _ = base.rope_dim
    with pytest.raises(NotImplementedError):
        _ = base.kv_bytes_per_token
    with pytest.raises(NotImplementedError):
        base.flops_per_token(8)
    with pytest.raises(NotImplementedError):
        _ = base.sdpa_backend


def test_attn_base_n_params_counts_parameters():
    assert AttnBase(_cfg()).n_params == 0
    assert ToyCausal(_cfg()).n_params == 16 * 16
    assert ToyZero(_cfg(type="toy_zero")).n_params == 0


# --- 7. Model shape ----------------------------------------------------------


def test_logits_shape_and_no_state_returned():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 5), generator=_gen(14))
    logits, state = model(idx)
    assert tuple(logits.shape) == (2, 5, 32)
    assert state is None


# --- 8. Parameter names ------------------------------------------------------


def test_parameter_names_are_exact():
    model = Transformer(_resolved(n_layers=2))
    expected = {"embed.weight", "norm_f.gain"}
    for i in range(2):
        expected |= {
            f"layers.{i}.norm1.gain",
            f"layers.{i}.attn.proj.weight",
            f"layers.{i}.norm2.gain",
            f"layers.{i}.mlp.gate.weight",
            f"layers.{i}.mlp.up.weight",
            f"layers.{i}.mlp.down.weight",
        }
    names = {n for n, _ in model.named_parameters()}
    assert names == expected
    assert not any("bias" in n for n in names)


# --- 9. Tied embeddings ------------------------------------------------------


def test_embedding_is_the_only_vocab_sized_parameter_and_is_tied():
    V, d = 32, 16
    model = Transformer(_resolved())
    vocab_shaped = [n for n, p in model.named_parameters() if tuple(p.shape) == (V, d)]
    assert vocab_shaped == ["embed.weight"]

    captured = {}
    handle = model.norm_f.register_forward_hook(lambda mod, inp, out: captured.update(h=out.detach()))
    idx = torch.randint(0, V, (2, 4), generator=_gen(15))
    logits, _ = model(idx)
    handle.remove()
    assert torch.equal(logits, F.linear(captured["h"], model.embed.weight))


def test_scaling_embedding_changes_logits():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 4), generator=_gen(16))
    before, _ = model(idx)
    with torch.no_grad():
        model.embed.weight.mul_(2.0)
    after, _ = model(idx)
    assert not torch.equal(before, after)


# --- 10. Non-attention parameter counts, full size --------------------------


@pytest.mark.parametrize("variant", config.variant_names())
def test_non_attention_params_full_size(variant):
    resolved = config.resolve(
        variant, init_seed=0, peak_lr=1e-3, overrides={"model.attention": {"type": "toy_zero"}}
    )
    d_ff = resolved["model"]["d_ff"]
    model = Transformer(resolved)
    # V=8192, d=384, L=8 from the brief, with the variant's own d_ff.
    assert model.n_non_attention_params == 8192 * 384 + 8 * (3 * 384 * d_ff + 2 * 384) + 384
    assert model.n_non_embedding_params == model.n_params - 8192 * 384


@pytest.mark.parametrize("ladder, layers", [("mha_L6", 6), ("mha_L10", 10)])
def test_ladder_non_attention_params_full_size(ladder, layers):
    resolved = config.resolve(
        "mha",
        ladder=ladder,
        init_seed=0,
        peak_lr=1e-3,
        overrides={"model.attention": {"type": "toy_zero"}},
    )
    assert resolved["model"]["n_layers"] == layers
    d_ff = resolved["model"]["d_ff"]
    model = Transformer(resolved)
    assert len(model.layers) == layers
    assert model.n_non_attention_params == 8192 * 384 + layers * (3 * 384 * d_ff + 2 * 384) + 384
    assert model.n_non_embedding_params == model.n_params - 8192 * 384


# --- 11. Pre-norm wiring -----------------------------------------------------


def _block(attention):
    resolved = _resolved(attention, n_layers=1)
    cfg = AttnConfig.from_resolved(resolved)
    m = resolved["model"]
    return Block(cfg, m["d_ff"]), m["rope_theta"], m["head_dim"]


def test_block_with_zero_attention_is_residual_mlp_exactly():
    block, theta, head_dim = _block("toy_zero")
    x = _randn((2, 5, 16), seed=17)
    freqs = rope_freqs(head_dim, 0, 5, theta)
    y, cache = block(x, freqs)
    assert cache is None
    assert torch.equal(y, x + block.mlp(block.norm2(x)))


def test_block_with_causal_attention_follows_pre_norm_order_exactly():
    block, theta, head_dim = _block("toy_causal")
    x = _randn((2, 5, 16), seed=18)
    freqs = rope_freqs(head_dim, 0, 5, theta)
    y, _ = block(x, freqs)
    x1 = x + block.attn(block.norm1(x), freqs)[0]
    assert torch.equal(y, x1 + block.mlp(block.norm2(x1)))


# --- 12. Causality ------------------------------------------------------------


def test_logits_before_a_change_are_unaffected_by_it():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 8), generator=_gen(19))
    changed = idx.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    before, _ = model(idx)
    after, _ = model(changed)
    assert torch.equal(before[:, :4], after[:, :4])
    assert not torch.equal(before[:, 4:], after[:, 4:])


# --- 13. Decode state ---------------------------------------------------------


def test_init_state_has_zero_pos_and_one_cache_per_layer():
    model = Transformer(_resolved(n_layers=3))
    state = model.init_state()
    assert isinstance(state, DecodeState)
    assert state.pos == 0
    assert len(state.caches) == 3


def test_pieced_decode_matches_full_forward_and_leaves_state_intact():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 8), generator=_gen(20))
    full, _ = model(idx)

    start = model.init_state()
    state, pieces = start, []
    for lo, hi, expected_pos in [(0, 3, 3), (3, 4, 4), (4, 8, 8)]:
        logits, new_state = model(idx[:, lo:hi], state=state)
        pieces.append(logits)
        assert new_state.pos == expected_pos
        assert state.pos == expected_pos - (hi - lo)  # the state passed in is untouched
        state = new_state
    assert start.pos == 0
    assert scale_relative_error(torch.cat(pieces, dim=1).detach(), full.detach()) <= REF_OPT_FP32


# --- 14. RoPE positions -------------------------------------------------------


def test_rope_positions_follow_the_state_position():
    model = Transformer(_resolved())
    theta, head_dim = model.rope_theta, 8
    idx = torch.randint(0, 32, (2, 3), generator=_gen(21))
    state = DecodeState(caches=model.init_state().caches, pos=5)
    model(idx, state=state)
    expected = rope_freqs(head_dim, 5, 3, theta)
    for layer in model.layers:
        assert torch.equal(layer.attn.last_freqs, expected)


def test_rope_starts_at_zero_without_state():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 4), generator=_gen(22))
    model(idx)
    expected = rope_freqs(8, 0, 4, model.rope_theta)
    for layer in model.layers:
        assert torch.equal(layer.attn.last_freqs, expected)


# --- 15. Mode -----------------------------------------------------------------


def test_opt_mode_reaches_attention():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 4), generator=_gen(23))
    model(idx, mode="opt")
    assert all(layer.attn.last_mode == "opt" for layer in model.layers)


def test_invalid_mode_raises():
    model = Transformer(_resolved())
    idx = torch.randint(0, 32, (2, 4), generator=_gen(24))
    with pytest.raises(ValueError):
        model(idx, mode="fast")


# --- 16. Constructor validation -----------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_heads": 4},                 # 4 * 8 != 16
        {"tie_embeddings": False},
        {"bias": True},
        {"norm": "layernorm"},
        {"activation": "gelu"},
    ],
)
def test_unsupported_architecture_values_raise(overrides):
    with pytest.raises(ValueError):
        Transformer(_resolved(**overrides))


# --- 17. float64 --------------------------------------------------------------


def test_model_runs_in_float64():
    model = Transformer(_resolved()).double()
    idx = torch.randint(0, 32, (2, 4), generator=_gen(25))
    logits, _ = model(idx)
    assert logits.dtype == torch.float64
    assert tuple(logits.shape) == (2, 4, 32)
