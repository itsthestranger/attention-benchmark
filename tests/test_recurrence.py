"""T-12: the recurrence anchor (plan §9.2), a control that is deliberately not attention.

Module-level claims are checked on GatedRecurrence directly. Claims that concern
the whole model (causality, decoding through the state) are checked through
build_model. Float64 identities use FLOAT64_IDENTITY; float32 agreement uses
REF_OPT_FP32. Every tolerance comes from tests/tolerances.py.
"""

import math

import pytest
import torch

from src import config
from src.attention.base import AttnConfig, build_attention, registered
from src.attention.recurrence import GatedRecurrence, RecurrenceState
from src.model import build_model
from src.transformer import rope_freqs
from tests.tolerances import FLOAT64_IDENTITY, REF_OPT_FP32, scale_relative_error

D = 8
HEADS = 2
HEAD_DIM = 4
VOCAB = 32
D_FF = 16
CTX = 64
B = 2


# --- Helpers -----------------------------------------------------------------


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(shape, seed=0, dtype=torch.float32, scale=1.0):
    return torch.randn(shape, generator=_gen(seed), dtype=dtype) * scale


def _tokens(shape, seed):
    return torch.randint(0, VOCAB, shape, generator=_gen(seed))


def _randomise(module, seed, scale=0.5):
    g = _gen(seed)
    with torch.no_grad():
        for _, p in module.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * scale)


def _resolved(n_layers=2, init_seed=0):
    overrides = {
        "model.attention": {"type": "recurrence"},
        "model.d_model": D,
        "model.n_heads": HEADS,
        "model.head_dim": HEAD_DIM,
        "model.n_layers": n_layers,
        "model.vocab_size": VOCAB,
        "model.d_ff": D_FF,
        "model.context_length": CTX,
    }
    return config.resolve("mha", init_seed=init_seed, peak_lr=1e-3, overrides=overrides)


def _module(dtype=torch.float32, seed=7, scale=0.5):
    cfg = AttnConfig.from_resolved(_resolved(n_layers=1))
    mod = build_attention(cfg).to(dtype)
    _randomise(mod, seed, scale)
    return mod


def _model(n_layers=2, init_seed=0):
    return build_model(_resolved(n_layers=n_layers, init_seed=init_seed))


def _decode(mod, x, pieces):
    """Feed x through the cache in consecutive pieces; return (outputs, final cache)."""
    cache = mod.init_cache()
    outs, start = [], 0
    for n in pieces:
        y, cache = mod(x[:, start:start + n], None, cache)
        outs.append(y)
        start += n
    assert start == x.shape[1]
    return torch.cat(outs, dim=1), cache


def _decode_model(model, idx, pieces):
    """Feed idx through model.init_state() in consecutive pieces; return (logits, final state)."""
    state = model.init_state()
    outs, start = [], 0
    for n in pieces:
        logits, state = model(idx[:, start:start + n], state=state)
        outs.append(logits)
        start += n
    assert start == idx.shape[1]
    return torch.cat(outs, dim=1), state


# --- 1. Registry -------------------------------------------------------------


def test_registry_contains_recurrence_and_builds_it():
    assert "recurrence" in registered()
    mod = build_attention(AttnConfig.from_resolved(_resolved(n_layers=1)))
    assert isinstance(mod, GatedRecurrence)


# --- 2. Parameters -----------------------------------------------------------


def test_parameters_are_three_bias_free_square_matrices():
    mod = _module()
    named = dict(mod.named_parameters())
    assert set(named) == {"w_i.weight", "w_g.weight", "w_o.weight"}
    for p in named.values():
        assert tuple(p.shape) == (D, D)
    assert mod.n_params == 3 * D * D
    assert all(not n.endswith("bias") for n, _ in mod.named_parameters())


# --- 3. Shape and cache -------------------------------------------------------


def test_output_shape_and_no_cache_when_none_given():
    mod = _module()
    x = _randn((B, 5, D), seed=1)
    y, cache = mod(x, None)
    assert tuple(y.shape) == (B, 5, D)
    assert cache is None


def test_freqs_argument_is_ignored():
    mod = _module()
    x = _randn((B, 5, D), seed=2)
    freqs = rope_freqs(mod.rope_dim, 0, 5, 10000.0)
    y_none, _ = mod(x, None)
    y_freqs, _ = mod(x, freqs)
    assert torch.equal(y_none, y_freqs)


# --- 4. The formula, by hand --------------------------------------------------


def test_recurrence_matches_explicit_loop_in_float64():
    mod = _module().double()
    T = 5
    x = _randn((B, T, D), seed=3, dtype=torch.float64)
    y, _ = mod(x, None)

    w_i, w_g, w_o = mod.w_i.weight, mod.w_g.weight, mod.w_o.weight
    with torch.no_grad():
        s = torch.zeros(B, D, dtype=torch.float64)
        expected = []
        for t in range(T):
            xt = x[:, t]
            g = torch.sigmoid(xt @ w_g.T)
            u = xt @ w_i.T
            s = g * s + (1 - g) * u
            expected.append(s @ w_o.T)
        expected = torch.stack(expected, dim=1)
    assert scale_relative_error(y.detach(), expected) <= FLOAT64_IDENTITY


def test_first_position_is_w_o_of_one_minus_gate_times_input():
    mod = _module().double()
    x = _randn((B, 3, D), seed=4, dtype=torch.float64)
    y, _ = mod(x, None)
    x0 = x[:, 0]
    g0 = torch.sigmoid(x0 @ mod.w_g.weight.T)
    u0 = x0 @ mod.w_i.weight.T
    with torch.no_grad():
        expected = ((1 - g0) * u0) @ mod.w_o.weight.T
    assert scale_relative_error(y[:, 0].detach(), expected) <= FLOAT64_IDENTITY


# --- 5. Causality -------------------------------------------------------------


def test_module_output_before_a_change_is_bit_identical():
    mod = _module()
    T, t = 6, 2
    x1 = _randn((B, T, D), seed=5)
    x2 = x1.clone()
    x2[:, t + 1:] = _randn((B, T - t - 1, D), seed=6)
    y1, _ = mod(x1, None)
    y2, _ = mod(x2, None)
    assert torch.equal(y1[:, :t + 1], y2[:, :t + 1])
    assert not torch.equal(y1[:, t + 1:], y2[:, t + 1:])


def test_model_logits_before_a_token_change_are_bit_identical():
    model = _model()
    T, t = 8, 3
    idx1 = _tokens((B, T), seed=7)
    idx2 = idx1.clone()
    idx2[:, t + 1:] = (idx1[:, t + 1:] + 1) % VOCAB
    l1, _ = model(idx1)
    l2, _ = model(idx2)
    assert torch.equal(l1[:, :t + 1], l2[:, :t + 1])
    assert not torch.equal(l1[:, t + 1:], l2[:, t + 1:])


# --- 6. The state is one d_model vector ---------------------------------------


@pytest.mark.parametrize("n_tokens", [1, 5, 50])
def test_state_is_one_d_model_vector_whatever_the_length(n_tokens):
    mod = _module()
    x = _randn((B, n_tokens, D), seed=8)
    _, state = mod(x, None, mod.init_cache())
    assert isinstance(state, RecurrenceState)
    assert tuple(state.s.shape) == (B, D)


def test_state_size_and_storage_properties():
    mod = _module()
    assert mod.state_values == D
    assert mod.kv_bytes_per_token == 0


# --- 7. Decode equals the full forward ----------------------------------------


def test_token_by_token_decode_matches_full_forward():
    mod = _module()
    T = 8
    x = _randn((B, T, D), seed=9)
    with torch.no_grad():
        y_full, s_full = mod(x, None, mod.init_cache())
        y_dec, s_dec = _decode(mod, x, [1] * T)
    assert scale_relative_error(y_dec, y_full) <= REF_OPT_FP32
    assert scale_relative_error(s_dec.s, s_full.s) <= REF_OPT_FP32


def test_uneven_piece_decode_matches_full_forward():
    mod = _module()
    T = 8
    x = _randn((B, T, D), seed=10)
    with torch.no_grad():
        y_full, s_full = mod(x, None, mod.init_cache())
        y_dec, s_dec = _decode(mod, x, [3, 1, 4])
    assert scale_relative_error(y_dec, y_full) <= REF_OPT_FP32
    assert scale_relative_error(s_dec.s, s_full.s) <= REF_OPT_FP32


def test_model_decode_matches_full_forward_in_pieces():
    model = _model()
    T = 8
    idx = _tokens((B, T), seed=11)
    with torch.no_grad():
        logits_full, _ = model(idx)
        logits_dec, state = _decode_model(model, idx, [3, 1, 4])
    assert scale_relative_error(logits_dec, logits_full) <= REF_OPT_FP32
    assert state.pos == T
    assert all(tuple(c.s.shape) == (B, D) for c in state.caches)


# --- 8. The cache passed in is not mutated ------------------------------------


def test_passed_in_cache_is_not_mutated():
    mod = _module()
    s0 = _randn((B, D), seed=12)
    cache = RecurrenceState(s0)
    before = cache.s.clone()
    x = _randn((B, 4, D), seed=13)
    _, new_cache = mod(x, None, cache)
    assert torch.equal(cache.s, before)
    assert new_cache is not cache
    assert new_cache.s is not cache.s


# --- 9. ref and opt agree; invalid mode is refused ----------------------------


def test_ref_and_opt_give_identical_outputs_and_states():
    mod = _module()
    x = _randn((B, 6, D), seed=14)
    cache = RecurrenceState(_randn((B, D), seed=15))
    y_ref, c_ref = mod(x, None, cache, mode="ref")
    y_opt, c_opt = mod(x, None, cache, mode="opt")
    assert torch.equal(y_ref, y_opt)
    assert torch.equal(c_ref.s, c_opt.s)


def test_invalid_mode_raises_value_error():
    mod = _module()
    x = _randn((B, 3, D), seed=16)
    with pytest.raises(ValueError):
        mod(x, None, mode="fast")


# --- 10. Gate extremes --------------------------------------------------------


def test_gate_saturated_at_one_keeps_the_start_state():
    mod = _module()
    with torch.no_grad():
        mod.w_g.weight.fill_(50.0)  # sigmoid(50 * positive sum) is exactly 1
        x = _randn((B, 6, D), seed=17).abs() + 0.1
        y_zero, _ = mod(x, None)
        memoryless = mod.w_o(mod.w_i(x))
        assert y_zero.abs().max() <= REF_OPT_FP32 * memoryless.abs().max()

        s0 = _randn((B, D), seed=18)
        y_s0, cache = mod(x, None, RecurrenceState(s0))
        expected = mod.w_o(s0).unsqueeze(1).expand_as(y_s0)
        assert scale_relative_error(y_s0, expected) <= REF_OPT_FP32
        assert scale_relative_error(cache.s, s0) <= REF_OPT_FP32


def test_gate_saturated_at_zero_remembers_nothing():
    mod = _module()
    with torch.no_grad():
        mod.w_g.weight.fill_(-50.0)  # sigmoid(-50 * positive sum) is numerically 0
    with torch.no_grad():
        x = _randn((B, 6, D), seed=19).abs() + 0.1
        y, _ = mod(x, None, RecurrenceState(_randn((B, D), seed=20)))
        memoryless = mod.w_o(mod.w_i(x))
        assert scale_relative_error(y, memoryless) <= REF_OPT_FP32


# --- 11. Exponential-moving-average memory ------------------------------------


def test_gate_half_gives_exponential_moving_average_in_float64():
    mod = _module().double()
    with torch.no_grad():
        mod.w_g.weight.zero_()  # sigmoid(0) = 0.5 for every input
        mod.w_i.weight.copy_(torch.eye(D, dtype=torch.float64))
        mod.w_o.weight.copy_(torch.eye(D, dtype=torch.float64))
    T = 6
    x = _randn((B, T, D), seed=21, dtype=torch.float64)
    y, _ = mod(x, None)

    s = torch.zeros(B, D, dtype=torch.float64)
    expected = []
    for t in range(T):
        s = 0.5 * s + 0.5 * x[:, t]
        expected.append(s)
    expected = torch.stack(expected, dim=1)
    assert scale_relative_error(y.detach(), expected) <= FLOAT64_IDENTITY


# --- 12. Gradients ------------------------------------------------------------


def test_gradients_reach_all_three_weights_and_are_finite():
    mod = _module()
    x = _randn((B, 5, D), seed=22)
    y, _ = mod(x, None)
    loss = y[:, -1].pow(2).sum()
    loss.backward()
    for name, p in mod.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.norm().item() > 0.0, name


# --- 13. Initialisation through build_model -----------------------------------


def test_residual_scaling_applies_to_w_o_only():
    seed = 3
    m1 = _model(n_layers=1, init_seed=seed)
    m4 = _model(n_layers=4, init_seed=seed)
    a1, a4 = m1.layers[0].attn, m4.layers[0].attn
    # Not residual outputs: identical bits whatever the depth.
    assert torch.equal(a1.w_i.weight, a4.w_i.weight)
    assert torch.equal(a1.w_g.weight, a4.w_g.weight)
    # Residual output: std is scaled by 1 / sqrt(2 * n_layers), so
    # w_o(4 layers) * sqrt(8) equals w_o(1 layer) * sqrt(2).
    assert scale_relative_error(a4.w_o.weight.detach() * math.sqrt(8), a1.w_o.weight.detach() * math.sqrt(2)) <= REF_OPT_FP32


# --- 14. float64 --------------------------------------------------------------


def test_module_runs_in_float64():
    mod = _module().double()
    assert all(p.dtype == torch.float64 for p in mod.parameters())
    x = _randn((B, 4, D), seed=23, dtype=torch.float64)
    y, state = mod(x, None, mod.init_cache())
    assert y.dtype == torch.float64
    assert state.s.dtype == torch.float64


# --- 15. Interface values -----------------------------------------------------


def test_interface_values():
    mod = _module()
    assert mod.rope_dim == HEAD_DIM
    assert mod.sdpa_backend is None
    assert mod.init_cache().s is None
    with pytest.raises(NotImplementedError):
        mod.flops_per_token(16)
