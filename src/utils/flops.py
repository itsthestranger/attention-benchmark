"""Analytic forward FLOPs per token of a whole model (plan §3.1, §10.4).

Convention: one FLOP is one multiply or one add, so a matrix product costs twice
its multiply-adds. Norms, softmax, RoPE rotation and the embedding lookup count
as zero. Every figure is per token, averaged over the T positions of one
sequence of length T, for one layer unless it says otherwise.

Sizes are read from the model's own parameters, not from the config, so a
ladder model with a different layer count needs no change here.
"""

from __future__ import annotations


def model_flops_per_token(model, T: int, causal_skip: bool = True) -> dict:
    """Forward FLOPs per token of the whole model at context T, as a dict of ints.

    Keys: "attention_mixing", "attention_projection", "mlp", "lm_head" and "total"
    (the sum of the four). The LM head is the tied embedding used as the output
    projection.
    """
    attn = [layer.attn for layer in model.layers]
    mixing = sum(a.mixing_flops_per_token(T, causal_skip) for a in attn)
    projection = sum(a.projection_flops_per_token() for a in attn)
    mlp = sum(
        2 * sum(w.weight.numel() for w in (layer.mlp.gate, layer.mlp.up, layer.mlp.down))
        for layer in model.layers
    )
    vocab, d_model = model.embed.weight.shape
    lm_head = 2 * d_model * vocab
    total = mixing + projection + mlp + lm_head
    return {
        "attention_mixing": mixing,
        "attention_projection": projection,
        "mlp": mlp,
        "lm_head": lm_head,
        "total": total,
    }


def attention_share(model, T: int, causal_skip: bool = True) -> float:
    """Attention mixing FLOPs as a fraction of the forward FLOPs per token (plan §3.1)."""
    counts = model_flops_per_token(model, T, causal_skip)
    return counts["attention_mixing"] / counts["total"]
