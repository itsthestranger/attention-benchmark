"""T-11: the graph-diff check (plan §4.1). The only code that differs between variants lives under `src/attention/`.

These tests check the checker itself: that the repository passes it, that the
identical-model guarantees hold, and that each planted difference outside
attention is caught and named. Tampering is done on copies of graphs or on
models built in memory; no file on disk is edited.
"""

import copy
import json
import math
import re

import pytest
import torch
import torch.nn as nn

from scripts import graph_diff as gd
from scripts.graph_diff import ALLOW_D_FF, ALLOW_N_LAYERS, GraphDiffError
from src import config
from src.model import build_model
from src.transformer import RMSNorm
from tests.tolerances import FLOAT64_IDENTITY, scale_relative_error

ZERO_HASH = "0" * 64  # a hash no real tensor has


def res(variant, arm="iso_token", *, init_seed=1, **kwargs):
    return config.resolve(variant, arm, init_seed=init_seed, peak_lr=1e-3, **kwargs)


def graph(variant, **kwargs):
    return gd.model_graph(res(variant, **kwargs))


def gain_names(graph_dict):
    return [n for n in graph_dict["parameter_hashes"] if n.endswith(".gain")]


# --- The repository passes ---------------------------------------------------------


def test_check_all_passes_and_returns_one_line_per_comparison():
    lines = gd.check_all()
    # Arm A 8 (every variant), arm B 3, arm C 3, ladder 2, seeds 1.
    assert len(lines) == 8 + 3 + 3 + 2 + 1
    assert all(line.startswith(("arm ", "ladder ", "seed ")) for line in lines)
    assert all(": ok" in line for line in lines)


def test_main_returns_zero_on_the_repository(capsys):
    assert gd.main([]) == 0
    assert "arm A" in capsys.readouterr().out


def test_main_reports_failure_on_stderr_and_returns_one(monkeypatch, capsys):
    def failing():
        raise GraphDiffError("planted failure line")

    monkeypatch.setattr(gd, "check_all", failing)
    assert gd.main([]) == 1
    captured = capsys.readouterr()
    assert "planted failure line" in captured.err
    assert captured.out == ""


def test_check_all_names_a_planted_difference(monkeypatch):
    real = gd.model_graph

    def tampered(resolved):
        g = real(resolved)
        if resolved["variant"] == "gqa2":
            g["config"]["model"]["rope_theta"] = 5000.0
        return g

    monkeypatch.setattr(gd, "model_graph", tampered)
    with pytest.raises(GraphDiffError, match="rope_theta"):
        gd.check_all()


def test_graph_has_the_documented_shape():
    g = graph("mha")
    assert set(g) == {"modules", "parameters", "attention_interface", "config", "parameter_hashes", "forward"}
    assert g["modules"]["layers.0.attn"] == gd.ATTENTION_MARK
    assert not any(gd._ATTN_SUBTREE_RE.match(n) for n in g["parameters"])
    assert "embed.weight" in g["parameters"] and "embed.weight" in g["parameter_hashes"]
    assert "run" not in g["config"] and "variant" not in g["config"] and "attention" not in g["config"]["model"]
    json.dumps(g)  # JSON-serialisable


def test_init_std_of_each_rule_matches_the_init_rule():
    resolved = res("mha")
    g = gd.model_graph(resolved)
    base = resolved["init"]["std"]
    n = resolved["model"]["n_layers"]
    assert scale_relative_error(g["parameters"]["embed.weight"]["init_std"], base) <= FLOAT64_IDENTITY
    down = g["parameters"]["layers.0.mlp.down.weight"]["init_std"]
    assert scale_relative_error(down, base / math.sqrt(2 * n)) <= FLOAT64_IDENTITY
    assert g["parameters"]["layers.0.mlp.gate.weight"]["init_std"] == base
    assert g["parameters"]["layers.0.norm1.gain"]["init"] == "ones"


# --- Identical-model guarantees ----------------------------------------------------


def test_swa256_and_null_are_identical_to_mha_outside_attention():
    mha = graph("mha")
    for variant in ("swa256", "null"):
        other = graph(variant)
        assert other["forward"] == mha["forward"]
        assert other["parameter_hashes"] == mha["parameter_hashes"]


def test_gqa4_changes_only_the_mlp_and_the_fingerprint():
    mha = graph("mha")
    gqa4 = graph("gqa4")
    assert gqa4["forward"] != mha["forward"]
    assert gqa4["parameter_hashes"]["embed.weight"] == mha["parameter_hashes"]["embed.weight"]
    names = gain_names(mha)
    assert names  # the norm gains are checked, not skipped
    for name in names:
        assert gqa4["parameter_hashes"][name] == mha["parameter_hashes"][name]


def test_stubbed_forward_bypasses_attention_but_not_the_mlp():
    resolved = res("mha")
    model = build_model(resolved)
    before = gd.graph_of(model, resolved)["forward"]
    with torch.no_grad():
        for p in model.layers[0].attn.parameters():
            p.add_(1.0)
    assert gd.graph_of(model, resolved)["forward"] == before
    with torch.no_grad():
        model.layers[0].mlp.gate.weight.add_(1.0)
    assert gd.graph_of(model, resolved)["forward"] != before


def test_graph_of_does_not_modify_the_model():
    resolved = res("mha")
    model = build_model(resolved)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    gd.graph_of(model, resolved)
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


# --- Planted differences outside attention are caught ------------------------------


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"model.rope_theta": 5000.0}, "rope_theta"),
        ({"model.vocab_size": 4096}, "vocab_size"),
        ({"model.context_length": 512}, "context_length"),
        ({"recipe.optimizer.weight_decay": 0.0}, "weight_decay"),
        ({"init.std": 0.03}, "init.std"),
        ({"precision.autocast_dtype": "bfloat16"}, "autocast_dtype"),
    ],
)
def test_config_value_that_differs_is_named(overrides, needle):
    tampered = res("gqa4", overrides=overrides)
    with pytest.raises(GraphDiffError, match=re.escape(needle)):
        gd.check_pair(tampered, res("mha"), allow=frozenset({ALLOW_D_FF}))


def test_d_ff_difference_without_allowance_is_caught():
    with pytest.raises(GraphDiffError, match=r"config\.model\.d_ff") as info:
        gd.check_pair(res("gqa4"), res("mha"), allow=frozenset())
    assert "layers.0.mlp.gate.weight" in str(info.value)


def test_layer_count_difference_under_d_ff_only_is_caught():
    with pytest.raises(GraphDiffError, match="n_layers"):
        gd.check_pair(res("mha", overrides={"model.n_layers": 6}), res("mha"), allow=frozenset({ALLOW_D_FF}))


@pytest.mark.parametrize("name", ["embed.weight", "layers.3.norm2.gain"])
def test_value_change_with_matching_shape_is_caught(name):
    resolved = res("mha")
    model = build_model(resolved)
    param = dict(model.named_parameters())[name]
    with torch.no_grad():
        param.view(-1)[0].add_(1.0)
    lines = gd.diff_graphs(gd.model_graph(resolved), gd.graph_of(model, resolved))
    assert any(name in line for line in lines)


def test_module_class_change_is_caught():
    resolved = res("mha")
    model = build_model(resolved)

    class OtherNorm(RMSNorm):  # same parameters and values, different class
        pass

    model.layers[0].norm2 = OtherNorm(resolved["model"]["d_model"])
    lines = gd.diff_graphs(gd.model_graph(resolved), gd.graph_of(model, resolved))
    assert any("layers.0.norm2" in line and "OtherNorm" in line for line in lines)


def test_module_swapped_for_another_class_is_caught():
    resolved = res("mha")
    model = build_model(resolved)
    model.layers[0].norm2 = nn.LayerNorm(resolved["model"]["d_model"])
    lines = gd.diff_graphs(gd.model_graph(resolved), gd.graph_of(model, resolved))
    assert any("layers.0.norm2" in line for line in lines)


def test_extra_non_attention_parameter_is_caught():
    resolved = res("mha")
    model = build_model(resolved)
    model.layers[0].extra_bias = nn.Parameter(torch.zeros(resolved["model"]["d_model"]))
    lines = gd.diff_graphs(gd.model_graph(resolved), gd.graph_of(model, resolved))
    assert any("layers.0.extra_bias" in line for line in lines)


def test_weight_decay_flag_tampered_is_caught():
    g = graph("mha")
    t = copy.deepcopy(g)
    t["parameters"]["layers.0.norm1.gain"]["weight_decay"] = True
    lines = gd.diff_graphs(g, t)
    assert any("layers.0.norm1.gain" in line and "weight_decay" in line for line in lines)


def test_init_std_tampered_is_caught():
    g = graph("mha")
    t = copy.deepcopy(g)
    t["parameters"]["layers.0.mlp.gate.weight"]["init_std"] += 1.0
    lines = gd.diff_graphs(g, t)
    assert any("layers.0.mlp.gate.weight" in line and "init_std" in line for line in lines)


def test_init_kind_tampered_is_caught():
    g = graph("mha")
    t = copy.deepcopy(g)
    t["parameters"]["layers.0.norm1.gain"]["init"] = "normal"
    assert any("layers.0.norm1.gain" in line for line in gd.diff_graphs(g, t))


def test_every_difference_is_reported_not_only_the_first():
    g = graph("mha")
    t = copy.deepcopy(g)
    t["config"]["model"]["rope_theta"] = 5.0
    t["parameters"]["layers.0.norm1.gain"]["weight_decay"] = True
    t["parameter_hashes"]["embed.weight"] = ZERO_HASH
    lines = gd.diff_graphs(g, t)
    assert len(lines) >= 3
    for needle in ("rope_theta", "layers.0.norm1.gain", "embed.weight"):
        assert any(needle in line for line in lines)


# --- Allowances ---------------------------------------------------------------------


def test_d_ff_allowance_permits_the_mlp_width_and_nothing_else():
    assert gd.diff_graphs(graph("mha"), graph("gqa4"), allow=frozenset({ALLOW_D_FF})) == []


def test_d_ff_allowance_is_void_when_d_ff_is_equal():
    mha = graph("mha")
    t = copy.deepcopy(graph("swa256"))
    t["parameter_hashes"]["embed.weight"] = ZERO_HASH
    lines = gd.diff_graphs(mha, t, allow=frozenset({ALLOW_D_FF}))
    assert any("embed.weight" in line for line in lines)
    t2 = copy.deepcopy(graph("swa256"))
    t2["forward"] = ZERO_HASH
    assert any("forward" in line for line in gd.diff_graphs(mha, t2, allow=frozenset({ALLOW_D_FF})))


def test_mlp_weight_in_the_wrong_dimension_is_caught():
    ref = graph("mha")
    for wrong in ([1152, 385], [384, 1152], [1100, 384]):
        t = copy.deepcopy(graph("gqa4"))
        t["parameters"]["layers.0.mlp.gate.weight"]["shape"] = wrong
        lines = gd.diff_graphs(ref, t, allow=frozenset({ALLOW_D_FF}))
        assert any("layers.0.mlp.gate.weight" in line for line in lines), wrong


def test_mlp_dimension_not_equal_to_d_ff_is_caught():
    t = copy.deepcopy(graph("gqa4"))  # d_ff 1152
    t["parameters"]["layers.2.mlp.down.weight"]["shape"] = [384, 1100]
    lines = gd.diff_graphs(graph("mha"), t, allow=frozenset({ALLOW_D_FF}))
    assert any("layers.2.mlp.down.weight" in line and "d_ff" in line for line in lines)


def test_n_layers_allowance_passes_the_ladder():
    assert gd.diff_graphs(graph("mha"), graph("mha", ladder="mha_L6"), allow=frozenset({ALLOW_N_LAYERS})) == []
    gd.check_pair(res("mha", ladder="mha_L10"), res("mha"), allow=frozenset({ALLOW_N_LAYERS}))


def test_n_layers_allowance_does_not_permit_a_shared_layer_hash_change():
    t = copy.deepcopy(graph("mha", ladder="mha_L6"))
    t["parameter_hashes"]["layers.0.mlp.gate.weight"] = ZERO_HASH
    lines = gd.diff_graphs(graph("mha"), t, allow=frozenset({ALLOW_N_LAYERS}))
    assert any("layers.0.mlp.gate.weight" in line for line in lines)


def test_n_layers_allowance_requires_the_rule_value_for_residual_projections():
    t = copy.deepcopy(graph("mha", ladder="mha_L6"))
    t["parameters"]["layers.0.mlp.down.weight"]["init_std"] += 1.0
    lines = gd.diff_graphs(graph("mha"), t, allow=frozenset({ALLOW_N_LAYERS}))
    assert any("layers.0.mlp.down.weight" in line and "rule" in line for line in lines)


def test_n_layers_allowance_still_reports_embedding_and_norm_changes():
    t = copy.deepcopy(graph("mha", ladder="mha_L6"))
    t["parameter_hashes"]["embed.weight"] = ZERO_HASH
    lines = gd.diff_graphs(graph("mha"), t, allow=frozenset({ALLOW_N_LAYERS}))
    assert any("embed.weight" in line for line in lines)


def test_attention_differences_are_ignored():
    assert gd.diff_graphs(graph("mha"), graph("mla192"), allow=frozenset({ALLOW_D_FF})) == []


def test_extra_attention_parameter_is_ignored():
    resolved = res("mha")
    model = build_model(resolved)
    model.layers[0].attn.extra = nn.Parameter(torch.zeros(3))
    assert gd.diff_graphs(graph("mha"), gd.graph_of(model, resolved)) == []


def test_unknown_allowance_is_rejected():
    with pytest.raises(ValueError, match="unknown allowance"):
        gd.diff_graphs(graph("mha"), graph("mha"), allow=frozenset({"layers"}))


def test_seed_change_moves_only_hashes_and_the_fingerprint():
    lines = gd.diff_graphs(graph("mha"), graph("mha", init_seed=2))
    assert lines
    assert all("parameter hash" in line or "forward" in line for line in lines)
    mha, seeded = graph("mha"), graph("mha", init_seed=2)
    assert seeded["parameter_hashes"]["embed.weight"] != mha["parameter_hashes"]["embed.weight"]
    for name in gain_names(mha):
        assert seeded["parameter_hashes"][name] == mha["parameter_hashes"][name]
