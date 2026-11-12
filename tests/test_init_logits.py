"""T-10: step-0 attention-logit scale across all variants (plan §3.3, §5 "Init logit scale").

Plan §3.3: if a factorized pair such as W_UK . W_DKV uses the backbone's default
std for both factors, MLA starts with much smaller logits than MHA, which makes
it look worse than it is by more than any effect being measured. The init rule
scales each up-projection so that MLA's logits start at MHA's scale. Plan §5:
"Step-0 attention-logit std agrees within 10% across all variants."

What is measured: the full-size model at initialisation reads a fixed batch of
SYNTHETIC random tokens in `ref` mode; every attention module's input is
captured, and `attention_logits` gives its scores as they enter the softmax.
The statistic is the standard deviation over every layer, head and causal
(query, key) pair, pooled. Single layers are noisier than the 10% bound: the
same variant with another init seed moves a deep layer's std by up to about
10%, so a per-layer comparison would test the seed, not the init rule.
"""

from __future__ import annotations

import functools
import math

import pytest
import torch

from src import config
from src.attention.mla import LatentAttention
from src.model import build_model
from src.transformer import rope_freqs
from tests.tolerances import AGREEMENT_FP64, INIT_LOGIT_STD_REL, scale_relative_error
from tests.variants import CASES, tiny_model

SEVEN = ("mha", "gqa4", "gqa2", "mqa", "swa256", "mla192", "mla96")
T, B, VOCAB = 128, 2, 8192
SEEDS = (1, 2)


def tokens() -> torch.Tensor:
    return torch.randint(0, VOCAB, (B, T), generator=torch.Generator().manual_seed(0))


def logits_by_layer(model) -> list[torch.Tensor]:
    """Causal attention logits of every layer, for the fixed token batch."""
    seen = []
    hooks = [
        layer.attn.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        for layer in model.layers
    ]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    try:
        with torch.no_grad():
            model(tokens(), mode="ref")
            out = []
            for layer, hidden in zip(model.layers, seen):
                freqs = rope_freqs(layer.attn.rope_dim, 0, T, model.rope_theta)
                out.append(layer.attn.attention_logits(hidden, freqs)[..., causal])
    finally:
        for hook in hooks:
            hook.remove()
    return out


@functools.lru_cache(maxsize=None)
def step0_std(name: str, init_seed: int) -> tuple[float, float]:
    """(std pooled over all layers, std of layer 0) at initialisation."""
    model = build_model(config.resolve(name, init_seed=init_seed, peak_lr=1e-3)).eval()
    per_layer = logits_by_layer(model)
    return torch.cat([x.flatten() for x in per_layer]).std().item(), per_layer[0].std().item()


def test_the_seven_variants_are_the_plans():
    assert set(SEVEN) == set(config.variant_names()) - {"null"}
    assert len(SEVEN) == 7


@pytest.mark.parametrize("init_seed", SEEDS)
@pytest.mark.parametrize("name", SEVEN)
def test_step0_logit_std_agrees_with_mha(name, init_seed):
    """The plan's test: within 10% of MHA, for every variant."""
    pooled, first = step0_std(name, init_seed)
    mha_pooled, mha_first = step0_std("mha", init_seed)
    assert abs(pooled / mha_pooled - 1.0) <= INIT_LOGIT_STD_REL
    # Layer 0 reads the normalised embeddings, the same input in every variant.
    assert abs(first / mha_first - 1.0) <= INIT_LOGIT_STD_REL


@pytest.mark.parametrize("init_seed", SEEDS)
def test_all_pairs_of_variants_agree(init_seed):
    """"Across all variants": the largest against the smallest, not only each against MHA."""
    pooled = [step0_std(name, init_seed)[0] for name in SEVEN]
    assert max(pooled) / min(pooled) - 1.0 <= INIT_LOGIT_STD_REL


def test_the_scale_is_the_one_the_init_rule_predicts():
    """Queries and keys of a normalised input have entries of variance d * 0.02^2, so a
    logit, a sum of d_h products divided by sqrt(d_h), has std d * 0.02^2 = 0.1536."""
    predicted = 384 * 0.02**2
    assert predicted == pytest.approx(0.1536)
    for name in SEVEN:
        assert abs(step0_std(name, 1)[1] / predicted - 1.0) <= INIT_LOGIT_STD_REL


def test_sliding_window_starts_from_mhas_logits():
    """Same parameter names and shapes, so per-parameter seeding gives SWA-256 MHA's weights."""
    assert step0_std("swa256", 1) == step0_std("mha", 1)


@pytest.mark.parametrize("name, d_c", [("mla192", 192), ("mla96", 96)])
def test_the_check_fails_without_the_factor_init(name, d_c, monkeypatch):
    """The bug the test exists for (plan §3.3, §13.7). With the default std on the
    up-projections, q_C, k_C and q_R each lose a factor sqrt(d_c / d) while the dense k_R
    keeps its scale. With r = d_c / d the logit variance falls to (48 r^2 + 24 r) / 72 of
    MHA's: a std of 0.58 for MLA-192 and 0.35 for MLA-96. The 10% bound must not hold."""
    monkeypatch.setattr(LatentAttention, "init_std_multipliers", lambda self: {})
    model = build_model(config.resolve(name, init_seed=1, peak_lr=1e-3)).eval()
    ratio = logits_by_layer(model)[0].std().item() / step0_std("mha", 1)[1]
    assert abs(ratio - 1.0) > INIT_LOGIT_STD_REL
    r = d_c / 384
    predicted = math.sqrt((48 * r * r + 24 * r) / 72)
    assert abs(ratio / predicted - 1.0) <= INIT_LOGIT_STD_REL


def test_the_check_fails_with_the_wrong_softmax_scale(monkeypatch):
    """Dividing MLA's scores by sqrt(48) instead of sqrt(72) raises its logits by 22%."""

    def wrong_scale(self, q_c, q_r, k_c, k_r):
        score = q_c @ k_c.transpose(-1, -2) + q_r @ k_r[:, None].transpose(-1, -2)
        return score / math.sqrt(self.head_dim)

    monkeypatch.setattr(LatentAttention, "_logits", wrong_scale)
    model = build_model(config.resolve("mla192", init_seed=1, peak_lr=1e-3)).eval()
    ratio = logits_by_layer(model)[0].std().item() / step0_std("mha", 1)[1]
    assert abs(ratio - 1.0) > INIT_LOGIT_STD_REL
    assert abs(ratio / math.sqrt(72 / 48) - 1.0) <= INIT_LOGIT_STD_REL


# --- attention_logits is what the forward pass uses -------------------------------------------------


def _values(attn, x):
    """Per-head values (B, H, T, dh), written out for the two families."""
    B_, T_, _ = x.shape
    H, dh = attn.cfg.n_heads, attn.cfg.head_dim
    if isinstance(attn, LatentAttention):
        return attn.w_uv(attn.norm_kv(attn.w_dkv(x))).view(B_, T_, H, dh).transpose(1, 2)
    G = attn.n_kv_heads
    v = attn.w_v(x).view(B_, T_, G, dh).transpose(1, 2)
    return v.repeat_interleave(H // G, dim=1)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_the_ref_forward_is_softmax_of_attention_logits(case):
    """Mask the logits, take the softmax, average the values, project: that must be the
    `ref` output. So the logits the test above reads are the ones training starts from."""
    model = tiny_model(case, sharpen=6.0, dtype=torch.float64)
    attn = model.layers[1].attn
    n = 40  # longer than the tiny sliding window of 16
    d = attn.cfg.d_model
    x = torch.randn(2, n, d, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    freqs = rope_freqs(attn.rope_dim, 0, n, model.rope_theta)
    with torch.no_grad():
        logits = attn.attention_logits(x, freqs)
        assert logits.shape == (2, attn.cfg.n_heads, n, n)
        pos = torch.arange(n)
        allowed = pos[None, :] <= pos[:, None]
        if case.window is not None:
            allowed &= pos[:, None] - pos[None, :] < case.window
        weights = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        mixed = (weights @ _values(attn, x)).transpose(1, 2).reshape(2, n, d)
        expected = attn.w_o(mixed)
        got, _ = attn(x, freqs, mode="ref")
    assert scale_relative_error(got, expected) <= AGREEMENT_FP64


def test_modules_without_scores_do_not_answer():
    from src.attention.base import AttnBase, AttnConfig
    import src.attention.recurrence  # noqa: F401  (registers itself)
    from src.attention.base import build_attention
    import dataclasses

    cfg = AttnConfig.from_resolved(config.resolve("mha", init_seed=1, peak_lr=1e-3))
    with pytest.raises(NotImplementedError):
        AttnBase(cfg).attention_logits(torch.zeros(1, 2, 384), None)
    recurrence = build_attention(dataclasses.replace(cfg, type="recurrence", options={}))
    with pytest.raises(NotImplementedError):
        recurrence.attention_logits(torch.zeros(1, 2, 384), None)
