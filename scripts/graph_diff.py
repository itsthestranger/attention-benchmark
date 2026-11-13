"""Checks that the attention module is the only thing that differs between variants (plan §4.1).

For every comparison the experiment makes, the resolved configs are built into
models and described by `model_graph`: the modules outside the attention
subtrees, their parameters (shape, dtype, weight-decay flag, init rule), the
per-layer interface the backbone sees of attention, the config without
variant-specific keys, and a forward fingerprint with attention stubbed out.
`diff_graphs` then reports every difference that the allowed exceptions do not
cover.

    python -m scripts.graph_diff
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import re
import sys

import torch
import torch.nn as nn

from src import config
from src.model import build_model
from src.training.optimizer import param_groups

ALLOW_D_FF = "d_ff"  # Arm A: KV savings move into the MLP width
ALLOW_N_LAYERS = "n_layers"  # the ladder: MHA at another depth
_ALLOWED = frozenset({ALLOW_D_FF, ALLOW_N_LAYERS})

ATTENTION_MARK = "<attention>"
FORWARD_BATCH, FORWARD_TOKENS = 2, 16  # the fixed integer batch of the fingerprint
INIT_SEED = 1
PEAK_LR = 1e-3

_ABSENT = object()
_LAYER_RE = re.compile(r"^layers\.(\d+)(?:\.|$)")
_ATTN_EXACT_RE = re.compile(r"^layers\.\d+\.attn$")
_ATTN_SUBTREE_RE = re.compile(r"^layers\.\d+\.attn(?:\.|$)")
_MLP_RE = re.compile(r"^layers\.\d+\.mlp\.(gate|up|down)\.weight$")
_DOWN_RE = re.compile(r"^layers\.\d+\.mlp\.down\.weight$")


class GraphDiffError(AssertionError):
    """A difference outside the attention subtree that the allowed exceptions do not cover."""


# --- Describing a built model ------------------------------------------------------


class _AttentionStub(nn.Module):
    """Stands in for an attention module in the forward fingerprint: returns its input unchanged."""

    def __init__(self, rope_dim: int):
        super().__init__()
        self._rope_dim = rope_dim  # Transformer.forward reads it to build RoPE tables

    @property
    def rope_dim(self) -> int:
        return self._rope_dim

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        return x, cache


def _qualname(module: nn.Module) -> str:
    return type(module).__module__ + "." + type(module).__qualname__


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_bytes(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().numpy().tobytes()


def _rule_std(base_std: float, multiplier: float, residual: bool, n_layers: int) -> float:
    """Init std of a 2-D parameter, the arithmetic of `src/utils/seeding.py:init_model`.

    That function does not expose the std of each parameter, so this repeats its
    steps in the same order (base x multiplier, then the residual scale for
    residual output projections), so that the two results are bit-identical.
    """
    std = base_std * multiplier
    if residual:
        std *= 1.0 / math.sqrt(2 * n_layers)
    return std


def _init_rule(name: str, p: torch.Tensor, base_std: float, n_layers: int) -> tuple[str, float | None]:
    if p.dim() == 1 and name.endswith(".gain"):
        return "ones", None
    if p.dim() == 2:
        # Outside the attention subtrees the only multiplier-free residual projection is mlp.down.
        return "normal", _rule_std(base_std, 1.0, _DOWN_RE.match(name) is not None, n_layers)
    return "none", None  # no init rule: the parameter itself is what differs


def _weight_decay_flags(model: nn.Module, weight_decay: float) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    for group in param_groups(model, weight_decay):
        for p in group["params"]:
            flags[id(p)] = group["name"] == "decay"
    return flags


def _config_view(resolved: dict) -> dict:
    """The resolved config without the per-run and per-variant entries."""
    view = copy.deepcopy(resolved)
    for key in ("variant", "ladder", "arm", "overrides", "run"):
        view.pop(key, None)
    view["model"].pop("attention", None)
    return view


def _forward_fingerprint(model: nn.Module, resolved: dict) -> str:
    net = copy.deepcopy(model).double()
    for layer in net.layers:
        layer.attn = _AttentionStub(layer.attn.rope_dim)
    vocab = resolved["model"]["vocab_size"]
    idx = (torch.arange(FORWARD_BATCH * FORWARD_TOKENS, dtype=torch.long) % vocab).reshape(
        FORWARD_BATCH, FORWARD_TOKENS
    )
    with torch.no_grad():
        logits, _ = net(idx, mode="ref")
    return _sha256(_array_bytes(logits))


def _attention_signature(model: nn.Module) -> tuple:
    return tuple(
        (
            _qualname(layer.attn),
            tuple((n, tuple(p.shape)) for n, p in layer.attn.named_parameters()),
        )
        for layer in model.layers
    )


def graph_of(model: nn.Module, resolved: dict) -> dict:
    """The graph of an already-built model. Does not modify `model`."""
    n_layers = model.n_layers
    base_std = model.init_std
    flags = _weight_decay_flags(model, resolved["recipe"]["optimizer"]["weight_decay"])

    modules: dict[str, str] = {}
    for name, module in model.named_modules():
        if _ATTN_EXACT_RE.match(name):
            modules[name] = ATTENTION_MARK
        elif not _ATTN_SUBTREE_RE.match(name):
            modules[name] = _qualname(module)

    parameters: dict[str, dict] = {}
    hashes: dict[str, str] = {}
    for name, p in model.named_parameters():
        if _ATTN_SUBTREE_RE.match(name):
            continue
        init, std = _init_rule(name, p, base_std, n_layers)
        parameters[name] = {
            "shape": list(p.shape),
            "dtype": str(p.dtype),
            "weight_decay": flags[id(p)],
            "init_std": std,
            "init": init,
        }
        hashes[name] = _sha256(_array_bytes(p))

    attention_interface = [
        {"d_model": layer.attn.cfg.d_model, "n_heads": layer.attn.cfg.n_heads, "head_dim": layer.attn.cfg.head_dim}
        for layer in model.layers
    ]

    return {
        "modules": modules,
        "parameters": parameters,
        "attention_interface": attention_interface,
        "config": _config_view(resolved),
        "parameter_hashes": hashes,
        "forward": _forward_fingerprint(model, resolved),
    }


def _canonical(resolved: dict) -> str:
    """What building the model and its graph depends on: everything but the run's budget and seeds."""
    view = copy.deepcopy(resolved)
    for key in ("variant", "ladder", "arm", "overrides"):
        view.pop(key, None)
    view["run"] = {"init_seed": resolved["run"]["init_seed"]}
    return json.dumps(view, sort_keys=True, separators=(",", ":"))


@functools.lru_cache(maxsize=None)
def _describe(canonical: str) -> tuple[dict, tuple]:
    resolved = json.loads(canonical)
    model = build_model(resolved)
    return graph_of(model, resolved), _attention_signature(model)


def model_graph(resolved: dict) -> dict:
    """The graph of the model `resolved` builds. Built once per distinct model; returns a copy."""
    return copy.deepcopy(_describe(_canonical(resolved))[0])


def attention_signature(resolved: dict) -> tuple:
    """Per layer: the attention class and its parameter names and shapes. Used by the ladder check only."""
    return _describe(_canonical(resolved))[1]


# --- Comparing two graphs ----------------------------------------------------------


def _leaves(node: object, prefix: str = "") -> dict[str, object]:
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def _show(value: object) -> str:
    return "<absent>" if value is _ABSENT else repr(value)


def _union(a: dict, b: dict) -> list[str]:
    return sorted(set(a) | set(b))


def _mlp_axes(name: str) -> tuple[int, int] | None:
    """(free axis, fixed axis) of an MLP weight: the axis that carries d_ff, and the one that carries d_model."""
    match = _MLP_RE.match(name)
    if match is None:
        return None
    return (1, 0) if match.group(1) == "down" else (0, 1)


def _mlp_shape_ok(name: str, shape: list, cfg: dict) -> bool:
    free, fixed = _mlp_axes(name)
    return len(shape) == 2 and shape[free] == cfg["model"]["d_ff"] and shape[fixed] == cfg["model"]["d_model"]


def _differences(reference: dict, other: dict, allow: frozenset) -> list[tuple[str, str]]:
    """(kind, line) for every difference. kind is "structure" or "value" (a hash or the forward fingerprint)."""
    unknown = set(allow) - _ALLOWED
    if unknown:
        raise ValueError(f"unknown allowance(s) {sorted(unknown)}; known: {sorted(_ALLOWED)}")
    allow = frozenset(allow)

    ref_cfg, oth_cfg = reference["config"], other["config"]
    ref_d, oth_d = ref_cfg["model"]["d_ff"], oth_cfg["model"]["d_ff"]
    ref_n, oth_n = ref_cfg["model"]["n_layers"], oth_cfg["model"]["n_layers"]
    # An allowance applies only when the value it covers really differs (d_ff equal: nothing may differ).
    d_ff_on = ALLOW_D_FF in allow and ref_d != oth_d
    n_on = ALLOW_N_LAYERS in allow and ref_n != oth_n
    cut = min(ref_n, oth_n) if n_on else None

    out: list[tuple[str, str]] = []

    def structure(line: str) -> None:
        out.append(("structure", line))

    def value(line: str) -> None:
        out.append(("value", line))

    def ignored(name: str) -> bool:
        if cut is None:
            return False
        match = _LAYER_RE.match(name)
        return match is not None and int(match.group(1)) >= cut

    # Config: the only keys allowed to differ are the ones the allowances name.
    ignored_keys = set()
    if d_ff_on:
        ignored_keys.add("model.d_ff")
    if n_on:
        ignored_keys.add("model.n_layers")
    ref_leaves, oth_leaves = _leaves(ref_cfg), _leaves(oth_cfg)
    for path in _union(ref_leaves, oth_leaves):
        if path in ignored_keys:
            continue
        a, b = ref_leaves.get(path, _ABSENT), oth_leaves.get(path, _ABSENT)
        if a != b:
            structure(f"config.{path}: {_show(a)} (reference) != {_show(b)} (other)")

    # Modules outside the attention subtrees. The attention module itself is one entry.
    rm, om = reference["modules"], other["modules"]
    for name in _union(rm, om):
        if ignored(name):
            continue
        if name not in om:
            structure(f"module {name!r}: only in reference ({rm[name]})")
        elif name not in rm:
            structure(f"module {name!r}: only in other ({om[name]})")
        elif rm[name] != om[name]:
            structure(f"module {name!r}: {rm[name]} (reference) != {om[name]} (other)")

    # What the backbone sees of attention.
    ai_r, ai_o = reference["attention_interface"], other["attention_interface"]
    for i in range(max(len(ai_r), len(ai_o))):
        if cut is not None and i >= cut:
            continue
        if i >= len(ai_r) or i >= len(ai_o):
            structure(f"attention interface of layer {i}: present in one model only")
        elif ai_r[i] != ai_o[i]:
            structure(f"attention interface of layer {i}: {ai_r[i]} (reference) != {ai_o[i]} (other)")

    # Parameters outside the attention subtrees: shape, dtype, weight decay, init rule.
    rp, op = reference["parameters"], other["parameters"]
    for name in _union(rp, op):
        if ignored(name):
            continue
        if name not in op:
            structure(f"parameter {name!r}: only in reference")
            continue
        if name not in rp:
            structure(f"parameter {name!r}: only in other")
            continue
        a, b = rp[name], op[name]
        if d_ff_on and _mlp_axes(name) is not None:
            free, fixed = _mlp_axes(name)
            for label, shape, cfg in (("reference", a["shape"], ref_cfg), ("other", b["shape"], oth_cfg)):
                if not _mlp_shape_ok(name, shape, cfg):
                    structure(
                        f"parameter {name!r}: shape {shape} in {label} does not match its model "
                        f"(d_ff={cfg['model']['d_ff']}, d_model={cfg['model']['d_model']}) "
                        f"on axes free={free}, fixed={fixed}"
                    )
            if a["shape"][fixed] != b["shape"][fixed]:
                structure(f"parameter {name!r}: shape {a['shape']} (reference) != {b['shape']} (other)")
        elif a["shape"] != b["shape"]:
            structure(f"parameter {name!r}: shape {a['shape']} (reference) != {b['shape']} (other)")
        for field in ("dtype", "weight_decay", "init"):
            if a[field] != b[field]:
                structure(f"parameter {name!r}: {field} {a[field]!r} (reference) != {b[field]!r} (other)")
        if n_on and _DOWN_RE.match(name):
            for label, entry, cfg in (("reference", a, ref_cfg), ("other", b, oth_cfg)):
                expected = _rule_std(cfg["init"]["std"], 1.0, True, cfg["model"]["n_layers"])
                if entry["init_std"] != expected:
                    structure(
                        f"parameter {name!r}: init_std {entry['init_std']!r} in {label} is not the "
                        f"residual rule value {expected!r} for {cfg['model']['n_layers']} layers"
                    )
        elif a["init_std"] != b["init_std"]:
            structure(f"parameter {name!r}: init_std {a['init_std']!r} (reference) != {b['init_std']!r} (other)")

    # Parameter values. Hashes of the allowed tensors may differ; nothing else may.
    rh, oh = reference["parameter_hashes"], other["parameter_hashes"]
    for name in _union(rh, oh):
        if ignored(name):
            continue
        if rh.get(name, _ABSENT) == oh.get(name, _ABSENT):
            continue
        if d_ff_on and _mlp_axes(name) is not None:
            continue
        if n_on and _DOWN_RE.match(name):
            continue
        value(f"parameter hash {name!r} differs")

    # The forward fingerprint may differ only where the MLP width or the depth differs.
    if reference["forward"] != other["forward"] and not (d_ff_on or n_on):
        value("forward fingerprint differs")

    return out


def diff_graphs(reference: dict, other: dict, *, allow: frozenset = frozenset()) -> list[str]:
    """Every difference outside the attention subtree that `allow` does not cover, as readable lines.
    [] means the two models differ only where they may."""
    return [line for _, line in _differences(reference, other, frozenset(allow))]


def _format(header: str, lines: list[str]) -> str:
    return "\n".join([header, *("  - " + line for line in lines)])


def check_pair(reference_resolved: dict, other_resolved: dict, *, allow: frozenset = frozenset()) -> None:
    """Raise GraphDiffError listing every difference outside attention that `allow` does not cover."""
    allow = frozenset(allow)
    lines = diff_graphs(model_graph(reference_resolved), model_graph(other_resolved), allow=allow)
    if lines:
        raise GraphDiffError(_format(f"{len(lines)} difference(s) outside attention (allow: {_allow_text(allow)})", lines))


def _allow_text(allow: frozenset) -> str:
    return ", ".join(sorted(allow)) or "nothing"


# --- The experiment's comparisons --------------------------------------------------


def _resolve(variant: str, arm: str = "iso_token", *, init_seed: int = INIT_SEED, **kwargs) -> dict:
    return config.resolve(variant, arm, init_seed=init_seed, peak_lr=PEAK_LR, **kwargs)


def _compare(label: str, reference: dict, other: dict, allow: frozenset) -> tuple[str, list[str]]:
    lines = diff_graphs(model_graph(reference), model_graph(other), allow=allow)
    rm, om = reference["model"], other["model"]
    summary = (
        f"{label}: ok (d_ff {om['d_ff']} vs {rm['d_ff']}; n_layers {om['n_layers']} vs {rm['n_layers']}; "
        f"allow: {_allow_text(allow)})"
    )
    return summary, [f"{label}: {line}" for line in lines]


def check_all() -> list[str]:
    """Run every comparison the experiment makes and return one summary line per comparison.
    Raises GraphDiffError (listing everything) if any comparison fails."""
    summaries: list[str] = []
    problems: list[str] = []
    mha = _resolve("mha")
    all_d = frozenset({ALLOW_D_FF})
    all_n = frozenset({ALLOW_N_LAYERS})

    def run(label: str, resolved: dict, reference: dict, allow: frozenset) -> None:
        summary, lines = _compare(label, reference, resolved, allow)
        if lines:
            problems.extend(lines)
        else:
            summaries.append(summary)

    # 1. Arm A: every variant against MHA; d_ff may differ.
    for variant in config.variant_names():
        run(f"arm A {variant} vs mha", _resolve(variant), mha, all_d)

    # 2. Arm B: iso-backbone variants against MHA; nothing may differ (d_ff is 1024 for all).
    for variant in config.load_arm("iso_backbone")["variants"]:
        run(f"arm B {variant} vs mha", _resolve(variant, "iso_backbone"), mha, frozenset())

    # 3. Arm C: iso-wall-clock variants against MHA; d_ff may differ.
    arm_c = config.load_arm("iso_wallclock")
    for variant in arm_c["variants"]:
        run(
            f"arm C {variant} vs mha",
            _resolve(variant, "iso_wallclock", steps=arm_c["reference_steps"]),
            mha,
            all_d,
        )

    # 4. Ladder: MHA at another depth; n_layers may differ. Attention is compared here, and only here.
    mha_sig = attention_signature(mha)
    if any(s != mha_sig[0] for s in mha_sig):
        problems.append("mha: attention layers are not all alike; the ladder check assumes they are")
    for name in config.ladder_names():
        ladder = _resolve("mha", ladder=name)
        run(f"ladder {name} vs mha", ladder, mha, all_n)
        for i, sig in enumerate(attention_signature(ladder)):
            if sig != mha_sig[0]:
                problems.append(f"ladder {name}: attention of layer {i} differs from mha's: {sig} != {mha_sig[0]}")

    # 5. Seeds: changing the init seed may change hashes and the fingerprint, and nothing else.
    seeded = _resolve("mha", init_seed=2)
    diffs = _differences(model_graph(mha), model_graph(seeded), frozenset())
    if not diffs:
        problems.append("seed 2 vs seed 1: no difference at all; the hashes do not react to values")
    elif any(kind != "value" for kind, _ in diffs):
        problems.extend(f"seed 2 vs seed 1: {line}" for kind, line in diffs if kind != "value")
    else:
        summaries.append(f"seed 2 vs seed 1 (mha): ok ({len(diffs)} value differences, nothing structural)")

    if problems:
        raise GraphDiffError(_format(f"{len(problems)} problem(s) outside attention", problems))
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.graph_diff",
        description="Check that only the attention module differs between the experiment's models (plan §4.1).",
    )
    parser.parse_args(argv)
    try:
        lines = check_all()
    except GraphDiffError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
