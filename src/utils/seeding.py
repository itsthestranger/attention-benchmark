"""Per-parameter seeded initialisation (plan §3.1, §4.5).

Every parameter draws from its own generator, seeded from its full name and the
run seed, so a parameter with the same name and shape has the same values in
every attention variant. Variant-specific facts come only through the
`AttnBase` methods `residual_output_params` and `init_std_multipliers`.
"""

import math
import zlib

import torch


def param_seed(name: str, run_seed: int) -> int:
    return zlib.crc32(name.encode()) ^ run_seed


def init_model(model, run_seed: int) -> None:
    named = list(model.named_parameters())

    seen: dict[int, str] = {}
    for name, _ in named:
        code = zlib.crc32(name.encode())
        if code in seen:
            raise ValueError(f"crc32 collision between parameters {seen[code]!r} and {name!r}")
        seen[code] = name

    residual: set[str] = set()
    multipliers: dict[str, float] = {}
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        own = {n for n, _ in attn.named_parameters()}
        declared = attn.residual_output_params()
        scaled = attn.init_std_multipliers()
        for local in sorted(set(declared) | set(scaled)):
            if local not in own:
                raise ValueError(f"layers.{i}.attn declares {local!r}, which is not one of its parameters")
        residual.update(f"layers.{i}.attn.{local}" for local in declared)
        multipliers.update({f"layers.{i}.attn.{local}": m for local, m in scaled.items()})
        residual.add(f"layers.{i}.mlp.down.weight")

    residual_scale = 1.0 / math.sqrt(2 * model.n_layers)
    plan = []
    for name, p in named:
        if p.dim() == 1 and name.endswith(".gain"):
            plan.append((name, p, None))
        elif p.dim() == 2:
            std = model.init_std * multipliers.get(name, 1.0)
            if name in residual:
                std *= residual_scale
            plan.append((name, p, std))
        else:
            raise ValueError(f"no init rule for parameter {name!r} with shape {tuple(p.shape)}")

    with torch.no_grad():
        for name, p, std in plan:
            if std is None:
                p.fill_(1.0)
                continue
            g = torch.Generator(device="cpu")
            g.manual_seed(param_seed(name, run_seed))
            z = torch.empty(tuple(p.shape), dtype=torch.float32).normal_(generator=g)
            p.copy_((z * std).to(dtype=p.dtype, device=p.device))
