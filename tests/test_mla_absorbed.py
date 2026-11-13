"""T-06: the absorbed MLA path for inference (plan §3.3, §4.4; §5 "MLA absorbed").

Written by the orchestrator before the implementation; the worker may not
change this file.

Plan §3.3: "Build the naive path first (materialize k_C and v_C). Write the
absorbed path only for `bench.py`, gated by a test asserting agreement to
1e-4." The naive path rebuilds per-head keys and values for every cached token
on every decode step. The absorbed path never does: it folds the up-projections
into the query and into the output, and attends over the cached latents
themselves, which is where MLA's decode saving comes from (plan §8.6).

    score[h, t, j] = q_C[h, t] . k_C[h, j] + q_R[h, t] . k_R[j]
                   = (W_UK[h]^T q_C[h, t]) . c_KV[j] + q_R[h, t] . k_R[j]
    y[t]           = sum_h W_O[:, h] (sum_j w[h, t, j] W_UV[h] c_KV[j])
                   = sum_h (W_O[:, h] W_UV[h]) (sum_j w[h, t, j] c_KV[j])

with W_UK[h], W_UV[h], W_UQ[h] the rows h*dh:(h+1)*dh of the weights of w_uk,
w_uv, w_uq, and W_O[:, h] the columns h*dh:(h+1)*dh of w_o's weight.

The contract (changes to `src/attention/mla.py`)
-------------------------------------------------
- `LatentAttention.absorbed_matrices() -> (q_absorb, o_absorb)`, computed under
  no_grad from the current weights, in their dtype:
      q_absorb (H, d_c, d_c_q):  q_absorb[h] = W_UK[h]^T W_UQ[h]
      o_absorb (H, d, d_c):      o_absorb[h] = W_O[:, h] W_UV[h]
- The cache stores ONE tensor per layer, `cache.latent` of shape
  (B, S, d_c + r): each token's [c_KV ; k_R], the plan's "cache = [c_KV ; k_R]".
  `cache.c_kv` and `cache.k_r` stay available as VIEWS of it (no copy).
  `length`, `capacity`, `values_per_token`, `append(c_kv, k_r)`, value semantics,
  geometric growth and the capacity hint are as before.
- `init_cache()` computes the absorbed matrices once; the cache, and every
  cache appended from it, carries them as `cache.absorbed` (the same tuple
  object). A cache therefore belongs to the weights it was created with.
- `forward(x, freqs, cache, mode="opt")` WITH a cache is the absorbed path:
      c_Q = norm_q(w_dq(x));  qt[h] = q_absorb[h] c_Q  (d_c wide)
      q_R = RoPE(w_qr(c_Q)) per head, as before; the new tokens' c_KV and k_R
      are appended to the cache, as before.
      One call to F.scaled_dot_product_attention with
          query (B, 1, H * T, d_c + r):  [qt ; q_R], row h * T + t for head h, token t
          key   (B, 1, S, d_c + r):      the new cache's latent tensor itself, not a copy
          value (B, 1, S, d_c):          its first d_c columns, a view, not a copy
          scale = 1 / sqrt(d_h + d_h^R), passed explicitly: the kernel's own
                  default would be 1 / sqrt(d_c + r), which is wrong
          no mask when T == 1; for T > 1 the causal mask of the T new tokens
          against the S cached ones, repeated for the H query groups.
      The result (B, H, T, d_c) is combined as y = sum_h o_absorb[h] o[h].
  Neither w_uk, w_uv, w_uq nor w_o is applied to anything in this path.
- `mode="ref"` with a cache, and every forward WITHOUT a cache in either mode,
  are unchanged: they are the naive path, which is what MLA trains on.
"""

from __future__ import annotations

import math

import pytest
import torch

from src import config
from src.attention.base import AttnConfig, build_attention
from src.transformer import rope_freqs
from tests.tolerances import AGREEMENT_FP64, MLA_ABSORBED, REF_OPT_FP16, scale_relative_error
from tests.variants import HEAD_DIM, MLA_CASES, MLA_ROPE_DIM, N_HEADS, tiny_model, tokens

D = N_HEADS * HEAD_DIM
H, DH, R = N_HEADS, HEAD_DIM, MLA_ROPE_DIM
MLA192, MLA96 = MLA_CASES


def tiny_attn(case=MLA192, dtype=torch.float64, sharpen=6.0):
    return tiny_model(case, sharpen=sharpen, dtype=dtype).layers[1].attn


def hidden(n, *, batch=2, dtype=torch.float64, seed=0, d=D):
    return torch.randn(batch, n, d, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def module_decode(attn, x, mode, pieces, capacity=None):
    """Feed x to one attention module through a cache, in pieces. Returns (outputs, final cache)."""
    cache = attn.init_cache() if capacity is None else attn.init_cache(capacity=capacity)
    outs, start = [], 0
    with torch.no_grad():
        for n in pieces:
            freqs = rope_freqs(attn.rope_dim, start, n, 10000.0)
            y, cache = attn(x[:, start : start + n], freqs, cache, mode=mode)
            outs.append(y)
            start += n
    assert start == x.shape[1]
    return torch.cat(outs, dim=1), cache


def model_decode(model, idx, mode, pieces):
    state, outs, start = model.init_state(), [], 0
    with torch.no_grad():
        for n in pieces:
            logits, state = model(idx[:, start : start + n], mode=mode, state=state)
            outs.append(logits)
            start += n
    return torch.cat(outs, dim=1)


class Recorder:
    """Wraps `scaled_dot_product_attention` inside the MLA module and keeps what it was given."""

    def __init__(self):
        import src.attention.mla as mla

        self.module, self.calls = mla, []

    def __enter__(self):
        self.original = self.module.F.scaled_dot_product_attention

        def recording(query, key, value, *args, **kwargs):
            self.calls.append({"query": query, "key": key, "value": value, "args": args, "kwargs": kwargs})
            return self.original(query, key, value, *args, **kwargs)

        self.module.F.scaled_dot_product_attention = recording
        return self

    def __exit__(self, *exc):
        self.module.F.scaled_dot_product_attention = self.original


# --- The absorbed matrices ------------------------------------------------------------------


@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_absorbed_matrices_are_the_per_head_products(case):
    attn = tiny_attn(case)
    d_c = attn.d_c
    q_absorb, o_absorb = attn.absorbed_matrices()
    assert q_absorb.shape == (H, d_c, attn.d_c_q) and o_absorb.shape == (H, D, d_c)
    assert q_absorb.dtype == o_absorb.dtype == torch.float64
    assert not q_absorb.requires_grad and not o_absorb.requires_grad
    for h in range(H):
        rows = slice(h * DH, (h + 1) * DH)
        w_uk, w_uq, w_uv = attn.w_uk.weight[rows], attn.w_uq.weight[rows], attn.w_uv.weight[rows]
        w_o = attn.w_o.weight[:, rows]
        assert scale_relative_error(q_absorb[h], (w_uk.T @ w_uq).detach()) <= AGREEMENT_FP64
        assert scale_relative_error(o_absorb[h], (w_o @ w_uv).detach()) <= AGREEMENT_FP64


def test_absorbed_matrices_at_full_size():
    resolved = config.resolve("mla192", init_seed=1, peak_lr=1e-3)
    attn = build_attention(AttnConfig.from_resolved(resolved))
    q_absorb, o_absorb = attn.absorbed_matrices()
    assert q_absorb.shape == (8, 192, 192) and o_absorb.shape == (8, 384, 192)
    assert set(dict(attn.named_parameters())) == {
        "w_dkv.weight", "norm_kv.gain", "w_uk.weight", "w_uv.weight", "w_kr.weight",
        "w_dq.weight", "norm_q.gain", "w_uq.weight", "w_qr.weight", "w_o.weight",
    }  # nothing absorbed is a parameter or a buffer
    assert list(attn.state_dict()) == [n for n, _ in attn.named_parameters()]


# --- Agreement with the naive path (plan §5: 1e-4) ---------------------------------------------


@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_absorbed_decode_matches_the_naive_full_forward_at_ctx_1024(case):
    """The plan's test, token by token at the trained context, in fp32."""
    model = tiny_model(case, sharpen=6.0, context_length=1024)
    idx = tokens(1, 1024)
    with torch.no_grad():
        naive, _ = model(idx, mode="ref")
    absorbed = model_decode(model, idx, "opt", [1] * 1024)
    assert scale_relative_error(absorbed, naive) <= MLA_ABSORBED


@pytest.mark.parametrize("pieces", [[1] * 24, [24], [5, 1, 1, 9, 8], [23, 1]], ids=["tokens", "one", "mixed", "prefill"])
@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_absorbed_decode_matches_naive_decode_in_float64(case, pieces):
    """In float64 the two paths differ by rounding only; a logic error cannot hide in 1e-4."""
    attn = tiny_attn(case)
    x = hidden(24)
    absorbed, _ = module_decode(attn, x, "opt", pieces)
    naive, _ = module_decode(attn, x, "ref", pieces)
    with torch.no_grad():
        full, _ = attn(x, rope_freqs(attn.rope_dim, 0, 24, 10000.0), mode="ref")
    assert scale_relative_error(absorbed, naive) <= AGREEMENT_FP64
    assert scale_relative_error(absorbed, full) <= AGREEMENT_FP64


@pytest.mark.parametrize("name", ["mla192", "mla96"])
def test_absorbed_decode_matches_naive_at_full_size_fp32(name):
    """One full-size module (d_model 384, 8 heads of 48, rope 24) on normalised inputs."""
    resolved = config.resolve(name, init_seed=1, peak_lr=1e-3)
    attn = build_attention(AttnConfig.from_resolved(resolved)).eval()
    with torch.no_grad():
        for p_name, p in attn.named_parameters():  # a usable scale without the model's init code
            if p.dim() == 2:
                p.normal_(0.0, 6.0 / math.sqrt(p.shape[1]), generator=torch.Generator().manual_seed(len(p_name)))
    x = hidden(48, dtype=torch.float32, d=384)
    absorbed, _ = module_decode(attn, x, "opt", [16] + [1] * 32)
    naive, _ = module_decode(attn, x, "ref", [16] + [1] * 32)
    assert scale_relative_error(absorbed, naive) <= MLA_ABSORBED


def test_absorbed_prefill_is_causal_inside_a_piece():
    attn = tiny_attn()
    x = hidden(12)
    changed = x.clone()
    changed[:, 9] += 1.0
    a, _ = module_decode(attn, x, "opt", [4, 8])
    b, _ = module_decode(attn, changed, "opt", [4, 8])
    assert torch.equal(a[:, :9], b[:, :9]) and not torch.equal(a[:, 9:], b[:, 9:])


# --- What reaches the kernel (plan §13.16: the cache must not be expanded per head) ------------


@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_one_token_attends_over_the_cache_itself(case):
    attn = tiny_attn(case)
    d_c = attn.d_c
    x = hidden(9)
    _, cache = module_decode(attn, x[:, :8], "opt", [8])
    with Recorder() as rec, torch.no_grad():
        _, new_cache = attn(x[:, 8:9], rope_freqs(attn.rope_dim, 8, 1, 10000.0), cache, mode="opt")
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert tuple(call["query"].shape) == (2, 1, H, d_c + R)
    assert tuple(call["key"].shape) == (2, 1, 9, d_c + R)
    assert tuple(call["value"].shape) == (2, 1, 9, d_c)
    assert call["args"] == () and set(call["kwargs"]) == {"scale"}  # no mask for one token
    assert call["kwargs"]["scale"] == 1.0 / math.sqrt(DH + R)
    # Keys and values are the cache's own memory: nothing was copied or expanded.
    assert call["key"].data_ptr() == new_cache.latent.data_ptr()
    assert call["value"].data_ptr() == new_cache.latent.data_ptr()
    assert torch.equal(call["key"][:, 0], new_cache.latent)
    assert torch.equal(call["value"][:, 0], new_cache.c_kv)


def test_a_piece_of_several_tokens_is_masked_per_query_group():
    attn = tiny_attn()
    x = hidden(11)
    _, cache = module_decode(attn, x[:, :8], "opt", [8])
    with Recorder() as rec, torch.no_grad():
        attn(x[:, 8:11], rope_freqs(attn.rope_dim, 8, 3, 10000.0), cache, mode="opt")
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert tuple(call["query"].shape) == (2, 1, H * 3, attn.d_c + R)
    assert tuple(call["key"].shape) == (2, 1, 11, attn.d_c + R)
    mask = call["kwargs"]["attn_mask"]
    assert mask.dtype == torch.bool and tuple(mask.shape) == (H * 3, 11)
    one_group = torch.tensor([[j <= 8 + t for j in range(11)] for t in range(3)])
    assert torch.equal(mask, one_group.repeat(H, 1))
    assert call["kwargs"]["scale"] == 1.0 / math.sqrt(DH + R)


def test_the_up_projections_are_not_applied_in_absorbed_decode():
    """The naive path applies w_uk and w_uv to every cached token on every step."""
    attn = tiny_attn()
    x = hidden(10)
    calls = {name: 0 for name in ("w_uk", "w_uv", "w_uq", "w_o")}
    hooks = [
        getattr(attn, name).register_forward_hook(lambda m, a, o, name=name: calls.__setitem__(name, calls[name] + 1))
        for name in calls
    ]
    try:
        module_decode(attn, x, "opt", [4] + [1] * 6)
        assert calls == {"w_uk": 0, "w_uv": 0, "w_uq": 0, "w_o": 0}
        module_decode(attn, x, "ref", [4] + [1] * 6)
        assert all(n == 7 for n in calls.values())  # the naive path, once per call
    finally:
        for hook in hooks:
            hook.remove()


def test_the_training_forward_is_still_the_naive_path():
    """No cache: `opt` is the T-05 path, with per-head keys of width d_h + d_h^R."""
    attn = tiny_attn()
    x = hidden(6)
    with Recorder() as rec, torch.no_grad():
        attn(x, rope_freqs(attn.rope_dim, 0, 6, 10000.0), mode="opt")
    assert len(rec.calls) == 1
    assert tuple(rec.calls[0]["query"].shape) == (2, H, 6, DH + R)
    assert tuple(rec.calls[0]["key"].shape) == (2, H, 6, DH + R)
    assert tuple(rec.calls[0]["value"].shape) == (2, H, 6, DH)


def test_gradients_still_flow_through_the_training_forward():
    attn = tiny_attn()
    x = hidden(6).requires_grad_(True)
    y, _ = attn(x, rope_freqs(attn.rope_dim, 0, 6, 10000.0), mode="opt")
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in attn.parameters())


# --- The cache: one tensor of d_c + r values per token (plan §3.3) --------------------------------


@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_the_cache_is_one_tensor_and_its_parts_are_views(case):
    attn = tiny_attn(case)
    d_c = attn.d_c
    _, cache = module_decode(attn, hidden(7), "opt", [3, 1, 1, 1, 1])
    assert cache.length == 7 and cache.values_per_token == d_c + R
    assert tuple(cache.latent.shape) == (2, 7, d_c + R)
    assert tuple(cache.c_kv.shape) == (2, 7, d_c) and tuple(cache.k_r.shape) == (2, 7, R)
    assert torch.equal(cache.latent[..., :d_c], cache.c_kv) and torch.equal(cache.latent[..., d_c:], cache.k_r)
    assert cache.c_kv.data_ptr() == cache.latent.data_ptr()  # views, not copies
    assert cache.k_r.untyped_storage().data_ptr() == cache.latent.untyped_storage().data_ptr()
    empty = attn.init_cache()
    assert empty.length == 0 and empty.latent is None and empty.c_kv is None and empty.k_r is None


def test_full_size_cache_is_216_values_per_token():
    resolved = config.resolve("mla192", init_seed=1, peak_lr=1e-3)
    attn = build_attention(AttnConfig.from_resolved(resolved)).eval()
    x = hidden(5, dtype=torch.float32, d=384)
    _, cache = module_decode(attn, x, "opt", [5])
    assert tuple(cache.latent.shape) == (2, 5, 216)  # plan §3.3: 216 values/token/layer
    assert cache.latent.numel() // (2 * 5) * 2 == attn.kv_bytes_per_token == 432  # fp16 bytes, plan §3.2


def test_both_modes_fill_the_cache_identically():
    attn = tiny_attn()
    x = hidden(9)
    _, absorbed_cache = module_decode(attn, x, "opt", [4, 1, 1, 1, 1, 1])
    _, naive_cache = module_decode(attn, x, "ref", [4, 1, 1, 1, 1, 1])
    assert torch.equal(absorbed_cache.latent, naive_cache.latent)


def test_a_cache_carries_the_matrices_it_was_created_with():
    attn = tiny_attn()
    x = hidden(6)
    first = attn.init_cache()
    assert isinstance(first.absorbed, tuple) and len(first.absorbed) == 2
    q_absorb, o_absorb = attn.absorbed_matrices()
    assert torch.equal(first.absorbed[0], q_absorb) and torch.equal(first.absorbed[1], o_absorb)
    with torch.no_grad():
        _, second = attn(x[:, :3], rope_freqs(attn.rope_dim, 0, 3, 10000.0), first, mode="opt")
        _, third = attn(x[:, 3:], rope_freqs(attn.rope_dim, 3, 3, 10000.0), second, mode="opt")
    assert second.absorbed is first.absorbed and third.absorbed is first.absorbed  # computed once
    # New weights need a new cache; the old cache keeps the matrices of the old weights.
    with torch.no_grad():
        attn.w_uk.weight.mul_(2.0)
    later = attn.init_cache()
    assert not torch.equal(later.absorbed[0], first.absorbed[0])
    assert torch.equal(first.absorbed[0], q_absorb)
    assert scale_relative_error(later.absorbed[0], 2.0 * q_absorb) <= AGREEMENT_FP64


def test_absorbed_caches_are_values():
    """Appending to one cache twice gives two independent continuations (decision 61)."""
    attn = tiny_attn()
    x = hidden(8)
    _, base = module_decode(attn, x[:, :5], "opt", [5])
    kept = base.latent.clone()
    with torch.no_grad():
        ya, a = attn(x[:, 5:6], rope_freqs(attn.rope_dim, 5, 1, 10000.0), base, mode="opt")
        yb, b = attn(x[:, 6:7], rope_freqs(attn.rope_dim, 5, 1, 10000.0), base, mode="opt")
        ya2, a2 = attn(x[:, 5:6], rope_freqs(attn.rope_dim, 5, 1, 10000.0), base, mode="opt")
    assert base.length == 5 and torch.equal(base.latent, kept)
    assert a.length == b.length == 6 and not torch.equal(a.latent, b.latent)
    assert torch.equal(ya, ya2) and torch.equal(a.latent, a2.latent)


def test_absorbed_decode_is_inference_only():
    attn = tiny_attn()
    x = hidden(3).requires_grad_(True)
    with pytest.raises(RuntimeError):
        attn(x, rope_freqs(attn.rope_dim, 0, 3, 10000.0), attn.init_cache(), mode="opt")


# --- gpu tier --------------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("case", MLA_CASES, ids=lambda c: c.name)
def test_absorbed_decode_matches_naive_in_fp16(case):
    model = tiny_model(case, sharpen=6.0, context_length=256).cuda()
    idx = tokens(2, 256).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        naive, _ = model(idx, mode="ref")
        absorbed = model_decode(model, idx, "opt", [128] + [1] * 128)
    assert absorbed.dtype == torch.float16
    assert scale_relative_error(absorbed.float(), naive.float()) <= REF_OPT_FP16
