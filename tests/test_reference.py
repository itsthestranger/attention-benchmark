"""Plan §5, reference agreement: `opt` matches `ref` on output, loss, input
gradients and every parameter gradient. fp32 on the cpu tier, fp16 on the gpu
tier, and once more in float64 (decision 6).

Also checks the grouped family against an independent, loop-per-head
implementation written here, so `ref` and `opt` cannot agree on a shared mistake.

Written by the orchestrator before the implementation; the worker may not
change this file.

The contract for `src/attention/grouped.py`
-------------------------------------------
    @register("grouped")
    class GroupedAttention(AttnBase)

- Option `n_kv_heads` (G). ValueError unless it divides `n_heads` (H).
- Four bias-free `nn.Linear`, named exactly:
      w_q: d_model -> H * head_dim
      w_k: d_model -> G * head_dim
      w_v: d_model -> G * head_dim
      w_o: H * head_dim -> d_model
  Projection outputs split into heads in order: head j of a projection is the
  slice [j * head_dim, (j + 1) * head_dim).
- Query head h uses KV head h // (H // G): consecutive query heads share one.
- RoPE (`src.transformer.apply_rope`) is applied to the queries and the keys of
  every head with the `freqs` passed in; `rope_dim` is `head_dim`.
- Scores are q . k / sqrt(head_dim); softmax over keys; causal.
- The head outputs are concatenated in head order and passed through `w_o`.
- `residual_output_params()` is {"w_o.weight"}.
- `mode="ref"`: plain tensor operations with an explicit mask; no call to
  `scaled_dot_product_attention`.
  `mode="opt"`: `torch.nn.functional.scaled_dot_product_attention`, reached as
  `F.scaled_dot_product_attention` inside the module (tests wrap it).
  Any other mode: ValueError.
- Cache and decode: see tests/test_kv_cache.py and tests/test_grouped_decode.py.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.attention.base import AttnConfig, build_attention, registered
from src.transformer import apply_rope, rope_freqs
from tests.tolerances import AGREEMENT_FP64, REF_OPT_FP16, REF_OPT_FP32, scale_relative_error
from tests.variants import (
    CASES,
    GROUPED_CASES,
    HEAD_DIM,
    MODES,
    N_HEADS,
    VOCAB,
    tiny_model,
    tiny_resolved,
    tokens,
)

TOLERANCE = {torch.float32: REF_OPT_FP32, torch.float64: AGREEMENT_FP64}


def forward_and_backward(model, idx, targets, mode):
    """Logits, loss, gradient at the embedding output, and every parameter gradient."""
    captured = {}

    def keep(_module, _inputs, output):
        output.retain_grad()
        captured["embedded"] = output

    handle = model.embed.register_forward_hook(keep)
    try:
        model.zero_grad(set_to_none=True)
        logits, _ = model(idx, mode=mode)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
    finally:
        handle.remove()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    return logits.detach(), loss.detach(), captured["embedded"].grad.detach().clone(), grads


@pytest.mark.parametrize("sharpen", [1.0, 6.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_opt_matches_ref(case, dtype, sharpen):
    tol = TOLERANCE[dtype]
    model = tiny_model(case, sharpen=sharpen, dtype=dtype)
    idx, targets = tokens(3, 64, seed=1), tokens(3, 64, seed=2)
    ref = forward_and_backward(model, idx, targets, "ref")
    opt = forward_and_backward(model, idx, targets, "opt")
    assert opt[0].dtype == dtype
    assert scale_relative_error(opt[0], ref[0]) <= tol, "logits"
    assert scale_relative_error(opt[1], ref[1]) <= tol, "loss"
    assert scale_relative_error(opt[2], ref[2]) <= tol, "input gradient"
    assert ref[3].keys() == opt[3].keys()
    for name in ref[3]:
        assert ref[3][name].abs().max() > 0, name  # every parameter takes part
        assert scale_relative_error(opt[3][name], ref[3][name]) <= tol, name


@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_modes_differ_in_code_path_not_in_result(case):
    """`ref` must not call the fused kernel; `opt` must."""
    import src.attention.grouped as grouped

    model = tiny_model(case)
    calls = []
    original = grouped.F.scaled_dot_product_attention

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    grouped.F.scaled_dot_product_attention = counting
    try:
        with torch.no_grad():
            model(tokens(1, 16), mode="ref")
            assert calls == []
            model(tokens(1, 16), mode="opt")
            assert len(calls) == len(model.layers)
    finally:
        grouped.F.scaled_dot_product_attention = original


def test_invalid_mode_raises_in_the_module():
    attn = tiny_model(GROUPED_CASES[0]).layers[0].attn
    x = torch.zeros(1, 4, attn.cfg.d_model)
    with pytest.raises(ValueError):
        attn(x, rope_freqs(HEAD_DIM, 0, 4, 10000.0), mode="fast")


# --- The grouped family against a loop-per-head implementation ------------------------


def naive_grouped(attn, x, freqs):
    """Textbook grouped-query attention, one query head at a time."""
    H, dh = N_HEADS, HEAD_DIM
    G = attn.cfg.options["n_kv_heads"]
    B, T, _ = x.shape
    q_all, k_all, v_all = x @ attn.w_q.weight.T, x @ attn.w_k.weight.T, x @ attn.w_v.weight.T
    outputs = []
    for h in range(H):
        g = h // (H // G)
        q = apply_rope(q_all[..., h * dh : (h + 1) * dh], freqs)
        k = apply_rope(k_all[..., g * dh : (g + 1) * dh], freqs)
        v = v_all[..., g * dh : (g + 1) * dh]
        out = torch.zeros(B, T, dh, dtype=x.dtype)
        for t in range(T):
            scores = (k[:, : t + 1] @ q[:, t].unsqueeze(-1)).squeeze(-1) / math.sqrt(dh)
            weights = torch.softmax(scores, dim=-1)
            out[:, t] = (weights.unsqueeze(-1) * v[:, : t + 1]).sum(dim=1)
        outputs.append(out)
    return torch.cat(outputs, dim=-1) @ attn.w_o.weight.T


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_grouped_matches_the_loop_per_head_implementation(case, mode):
    attn = tiny_model(case, sharpen=6.0, dtype=torch.float64).layers[1].attn
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 24, attn.cfg.d_model, generator=g, dtype=torch.float64)
    freqs = rope_freqs(HEAD_DIM, 0, 24, 10000.0)
    with torch.no_grad():
        got, cache = attn(x, freqs, mode=mode)
        expected = naive_grouped(attn, x, freqs)
    assert cache is None
    assert got.shape == x.shape
    assert scale_relative_error(got, expected) <= AGREEMENT_FP64


@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_grouped_parameters(case):
    assert "grouped" in registered()
    attn = tiny_model(case).layers[0].attn
    d, G = attn.cfg.d_model, case.n_kv_heads
    shapes = {name: tuple(p.shape) for name, p in attn.named_parameters()}
    assert shapes == {
        "w_q.weight": (N_HEADS * HEAD_DIM, d),
        "w_k.weight": (G * HEAD_DIM, d),
        "w_v.weight": (G * HEAD_DIM, d),
        "w_o.weight": (d, N_HEADS * HEAD_DIM),
    }
    # Plan §3.2: P_attn = 2 d^2 + 2 d (h_kv d_h).
    assert attn.n_params == 2 * d * d + 2 * d * (G * HEAD_DIM)
    assert attn.rope_dim == HEAD_DIM
    assert attn.residual_output_params() == {"w_o.weight"}
    assert attn.init_std_multipliers() == {}


def test_full_size_attention_parameters_match_the_plan_table():
    """Plan §3.2: attention parameters per layer at d_model 384."""
    from src import config

    expected = {"mha": 589_824, "gqa4": 442_368, "gqa2": 368_640, "mqa": 331_776}
    for name, count in expected.items():
        cfg = AttnConfig.from_resolved(config.resolve(name, init_seed=1, peak_lr=1e-3))
        assert build_attention(cfg).n_params == count, name


def test_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError):
        tiny_model(GROUPED_CASES[0], **{"model.attention": {"type": "grouped", "n_kv_heads": 3}})


def test_softmax_scale_is_one_over_sqrt_head_dim():
    """Doubling queries doubles the scores: the result must equal a hand computation
    with the 1 / sqrt(head_dim) scale, and differ from one without it."""
    attn = tiny_model(GROUPED_CASES[0], sharpen=6.0, dtype=torch.float64).layers[0].attn
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, 6, attn.cfg.d_model, generator=g, dtype=torch.float64)
    freqs = rope_freqs(HEAD_DIM, 0, 6, 10000.0)
    with torch.no_grad():
        got, _ = attn(x, freqs, mode="ref")
        expected = naive_grouped(attn, x, freqs)
        attn.w_q.weight.mul_(math.sqrt(HEAD_DIM))  # as if the scale were missing
        unscaled, _ = attn(x, freqs, mode="ref")
    assert scale_relative_error(got, expected) <= AGREEMENT_FP64
    assert scale_relative_error(unscaled, expected) > REF_OPT_FP32


# --- gpu tier: fp16 (plan §5) ---------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_opt_matches_ref_fp16(case):
    model = tiny_model(case, sharpen=6.0).cuda()
    idx, targets = tokens(3, 64, seed=1).cuda(), tokens(3, 64, seed=2).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        ref = forward_and_backward(model, idx, targets, "ref")
        opt = forward_and_backward(model, idx, targets, "opt")
    assert scale_relative_error(opt[0].float(), ref[0].float()) <= REF_OPT_FP16, "logits"
    assert scale_relative_error(opt[1].float(), ref[1].float()) <= REF_OPT_FP16, "loss"
    assert scale_relative_error(opt[2].float(), ref[2].float()) <= REF_OPT_FP16, "input gradient"
    for name in ref[3]:
        assert scale_relative_error(opt[3][name].float(), ref[3][name].float()) <= REF_OPT_FP16, name
