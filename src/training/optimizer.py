"""AdamW and its two parameter groups (plan §7.1)."""

from __future__ import annotations

import torch


def param_groups(model, weight_decay: float) -> list[dict]:
    """Decay every 2-D parameter except the tied embedding; no decay for the rest."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.dim() >= 2 and p is not model.embed.weight:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
    ]


def build_optimizer(model, resolved: dict) -> torch.optim.AdamW:
    """AdamW over the two groups, with betas, eps and weight decay from the recipe."""
    opt_cfg = resolved["recipe"]["optimizer"]
    if opt_cfg.get("name") != "adamw":
        raise ValueError(f"optimizer must be 'adamw', got {opt_cfg.get('name')!r}")
    return torch.optim.AdamW(
        param_groups(model, opt_cfg["weight_decay"]),
        lr=resolved["run"]["peak_lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )
