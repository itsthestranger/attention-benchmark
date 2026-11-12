"""Plan §5, GQA limits: n_kv_heads = n_q_heads is MHA; n_kv_heads = 1 is MQA.

One code path serves the whole family, so a grouped model must equal an MHA
model whose K and V projections are its own, repeated for the query heads that
share them. That catches repeat and indexing errors.

Written by the orchestrator before the implementation; the worker may not
change this file.
"""

import pytest
import torch

from tests.tolerances import AGREEMENT_FP64, REF_OPT_FP32, scale_relative_error
from tests.variants import GROUPED_CASES, HEAD_DIM, MODES, N_HEADS, Case, tiny_model, tokens

MHA = Case("mha", 8)
TOLERANCE = {torch.float32: REF_OPT_FP32, torch.float64: AGREEMENT_FP64}


def as_mha(grouped_model, case: Case, dtype):
    """An MHA model computing the same function as `grouped_model`."""
    mha = tiny_model(MHA, dtype=dtype, **{"model.d_ff": grouped_model.layers[0].mlp.gate.out_features})
    share = N_HEADS // case.n_kv_heads
    source = dict(grouped_model.named_parameters())
    with torch.no_grad():
        for name, p in mha.named_parameters():
            if name.endswith(("attn.w_k.weight", "attn.w_v.weight")):
                rows = source[name].view(case.n_kv_heads, HEAD_DIM, -1)
                p.copy_(rows.repeat_interleave(share, dim=0).reshape(p.shape))
            else:
                p.copy_(source[name])
    return mha


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: c.name)
def test_grouped_equals_mha_with_repeated_kv_heads(case, mode, dtype):
    grouped = tiny_model(case, sharpen=6.0, dtype=dtype)
    mha = as_mha(grouped, case, dtype)
    idx = tokens(2, 40, seed=6)
    with torch.no_grad():
        got, _ = grouped(idx, mode=mode)
        expected, _ = mha(idx, mode=mode)
    assert scale_relative_error(got, expected) <= TOLERANCE[dtype]


@pytest.mark.parametrize("mode", MODES)
def test_the_wrong_grouping_is_told_apart(mode):
    """`repeat` instead of `repeat_interleave` gives a different function."""
    case = Case("gqa4", 4)
    grouped = tiny_model(case, sharpen=6.0, dtype=torch.float64)
    mha = as_mha(grouped, case, torch.float64)
    source = dict(grouped.named_parameters())
    with torch.no_grad():
        for name, p in mha.named_parameters():
            if name.endswith(("attn.w_k.weight", "attn.w_v.weight")):
                rows = source[name].view(case.n_kv_heads, HEAD_DIM, -1)
                p.copy_(rows.repeat(N_HEADS // case.n_kv_heads, 1, 1).reshape(p.shape))
        idx = tokens(2, 40, seed=6)
        got, _ = grouped(idx, mode=mode)
        wrong, _ = mha(idx, mode=mode)
    assert scale_relative_error(got, wrong) > REF_OPT_FP32


@pytest.mark.parametrize("mode", MODES)
def test_mqa_shares_one_kv_head_across_all_query_heads(mode):
    """With a single KV head, every query head reads the same keys and values:
    making all query heads identical makes all head outputs identical."""
    model = tiny_model(Case("mqa", 1), sharpen=6.0, dtype=torch.float64)
    attn = model.layers[0].attn
    captured = {}
    handle = attn.w_o.register_forward_hook(lambda _m, inputs, _o: captured.update(heads=inputs[0]))
    with torch.no_grad():
        first = attn.w_q.weight[:HEAD_DIM].clone()
        attn.w_q.weight.copy_(first.repeat(N_HEADS, 1))
        model(tokens(2, 20, seed=8), mode=mode)
    handle.remove()
    heads = captured["heads"].view(2, 20, N_HEADS, HEAD_DIM)
    for h in range(1, N_HEADS):
        assert scale_relative_error(heads[:, :, h], heads[:, :, 0]) <= AGREEMENT_FP64, h


@pytest.mark.parametrize("mode", MODES)
def test_mha_heads_are_independent(mode):
    """With eight KV heads, changing head 3's keys changes head 3's output only."""
    model = tiny_model(MHA, sharpen=6.0, dtype=torch.float64)
    attn = model.layers[0].attn
    captured = []
    handle = attn.w_o.register_forward_hook(lambda _m, inputs, _o: captured.append(inputs[0]))
    idx = tokens(2, 20, seed=8)
    with torch.no_grad():
        model(idx, mode=mode)
        attn.w_k.weight[3 * HEAD_DIM : 4 * HEAD_DIM].mul_(-1.5)
        model(idx, mode=mode)
    handle.remove()
    before, after = (c.view(2, 20, N_HEADS, HEAD_DIM) for c in captured)
    for h in range(N_HEADS):
        same = scale_relative_error(after[:, :, h], before[:, :, h]) <= AGREEMENT_FP64
        assert same == (h != 3), h
