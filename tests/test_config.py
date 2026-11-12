"""T-01: the config loader reads the plan's numbers exactly, and resolve() is strict.

Expected values are literals taken from the plan excerpts (section noted per
test), not read back from the YAML, so these tests check the YAML against the
plan. Equality is exact throughout.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src import config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

# Plan §3.1 shape constants.
V = 8192  # vocabulary
D = 384  # d_model
H = 8  # query heads
DH = 48  # head dim

VARIANT_TABLE = {
    # name: (attention dict, d_ff), plan §3.2
    "mha": ({"type": "grouped", "n_kv_heads": 8}, 1024),
    "gqa4": ({"type": "grouped", "n_kv_heads": 4}, 1152),
    "gqa2": ({"type": "grouped", "n_kv_heads": 2}, 1216),
    "mqa": ({"type": "grouped", "n_kv_heads": 1}, 1248),
    "swa256": ({"type": "sliding_window", "n_kv_heads": 8, "window": 256}, 1024),
    "mla192": ({"type": "mla", "d_c": 192, "d_c_q": 192, "rope_dim": 24}, 1048),
    "mla96": ({"type": "mla", "d_c": 96, "d_c_q": 96, "rope_dim": 24}, 1224),
    "null": ({"type": "grouped", "n_kv_heads": 8}, 1024),
}

MODEL_KEYS = {
    "n_layers",
    "d_model",
    "n_heads",
    "head_dim",
    "vocab_size",
    "context_length",
    "rope_theta",
    "norm",
    "activation",
    "tie_embeddings",
    "bias",
    "d_ff",
    "attention",
}


def res(variant: str = "mha", arm: str = "iso_token", **kwargs) -> dict:
    kwargs.setdefault("init_seed", 1)
    kwargs.setdefault("peak_lr", 1e-3)
    return config.resolve(variant, arm, **kwargs)


def attn_params_per_layer(variant_doc: dict) -> int:
    a = variant_doc["attention"]
    if a["type"] in ("grouped", "sliding_window"):
        return 2 * D * D + 2 * D * (a["n_kv_heads"] * DH)
    assert a["type"] == "mla"
    dc, r = a["d_c"], a["rope_dim"]
    assert a["d_c_q"] == dc
    return (
        D * dc
        + 2 * dc * (H * DH)
        + D * r
        + D * dc
        + dc * (H * DH)
        + dc * (H * r)
        + (H * DH) * D
        + 2 * dc
    )


def total_params(variant_doc: dict, layers: int) -> int:
    return V * D + layers * (attn_params_per_layer(variant_doc) + 3 * D * variant_doc["d_ff"] + 2 * D) + D


def _reverse_keys(node):
    if isinstance(node, dict):
        return {k: _reverse_keys(node[k]) for k in reversed(list(node))}
    if isinstance(node, list):
        return [_reverse_keys(v) for v in node]
    return node


# --- Loading -----------------------------------------------------------------


def test_every_yaml_file_loads_to_a_dict():
    paths = sorted(CONFIGS.rglob("*.yaml"))
    # 5 top-level (ops.yaml is the fifth, T-29), 8 variants, 2 ladder, 3 arms, 1 example run spec
    assert len(paths) == 19
    for path in paths:
        assert isinstance(yaml.safe_load(path.read_text()), dict), path


def test_name_lists_are_exact():
    assert config.variant_names() == ["gqa2", "gqa4", "mha", "mla192", "mla96", "mqa", "null", "swa256"]
    assert config.ladder_names() == ["mha_L10", "mha_L6"]
    assert config.arm_names() == ["iso_backbone", "iso_token", "iso_wallclock"]


def test_loaders_return_fresh_objects():
    cfg = config.load_base()
    cfg["model"]["d_model"] = 0
    assert config.load_base()["model"]["d_model"] == 384

    rec = config.load_recipe()
    rec["budgets"]["50M"] = 0
    assert config.load_recipe()["budgets"]["50M"] == 763

    names = config.variant_names()
    names.append("bogus")
    assert "bogus" not in config.variant_names()

    r = res()
    r["recipe"]["optimizer"]["betas"].append(9)
    r["model"]["attention"]["n_kv_heads"] = 1
    r["run"]["branches"].append("300M")
    fresh = res()
    assert fresh["recipe"]["optimizer"]["betas"] == [0.9, 0.95]
    assert fresh["model"]["attention"]["n_kv_heads"] == 8
    assert fresh["run"]["branches"] == ["100M"]


@pytest.mark.parametrize(
    "loader, bad",
    [
        (config.load_variant, "nope"),
        (config.load_ladder, "nope"),
        (config.load_arm, "nope"),
        (config.load_variant, "../base"),
        (config.load_variant, None),
    ],
)
def test_unknown_names_raise_naming_the_value(loader, bad):
    with pytest.raises(config.ConfigError, match=re.escape(repr(bad))):
        loader(bad)


# --- base.yaml (plan §3.1) ---------------------------------------------------


def test_base_matches_plan_3_1():
    base = config.load_base()
    assert base["model"] == {
        "n_layers": 8,
        "d_model": 384,
        "n_heads": 8,
        "head_dim": 48,
        "vocab_size": 8192,
        "context_length": 1024,
        "rope_theta": 10000.0,
        "norm": "rmsnorm",
        "activation": "swiglu",
        "tie_embeddings": True,
        "bias": False,
    }
    assert base["init"]["distribution"] == "normal"
    assert base["init"]["std"] == 0.02
    assert base["init"]["norm_gain"] == 1.0
    assert base["precision"] == {
        "autocast_dtype": "float16",
        "master_dtype": "float32",
        "grad_scaler": True,
    }
    assert base["params"]["target_total"] == 17308032
    assert base["params"]["target_non_embedding"] == 14162304
    assert base["params"]["mla_max_relative_offset"] == 0.0005


def test_base_heads_times_head_dim_is_d_model():
    m = config.load_base()["model"]
    assert m["n_heads"] * m["head_dim"] == m["d_model"]


# --- Variants (plan §3.2) ----------------------------------------------------


def test_variant_set_is_exactly_the_eight_plan_variants():
    assert set(config.variant_names()) == set(VARIANT_TABLE)
    assert len(VARIANT_TABLE) == 8


@pytest.mark.parametrize("name", sorted(VARIANT_TABLE))
def test_variant_matches_plan_table(name):
    attention, d_ff = VARIANT_TABLE[name]
    doc = config.load_variant(name)
    assert doc["attention"] == attention
    assert doc["d_ff"] == d_ff


@pytest.mark.parametrize("name", sorted(VARIANT_TABLE))
def test_variant_name_equals_file_stem(name):
    assert config.load_variant(name)["name"] == name


def test_null_name_is_the_string_null():
    name = config.load_variant("null")["name"]
    assert isinstance(name, str)
    assert name == "null"


def test_null_equals_mha_except_name():
    null, mha = config.load_variant("null"), config.load_variant("mha")
    null.pop("name")
    mha.pop("name")
    assert null == mha


# --- Parameter totals (plan §3.2, §4.3) --------------------------------------


EXPECTED_ATTN_PER_LAYER = {
    "mha": 589824,
    "gqa4": 442368,
    "gqa2": 368640,
    "mqa": 331776,
    "mla192": 562560,
    "mla96": 359616,
}


@pytest.mark.parametrize("name, expected", sorted(EXPECTED_ATTN_PER_LAYER.items()))
def test_attention_params_per_layer(name, expected):
    assert attn_params_per_layer(config.load_variant(name)) == expected


EXPECTED_TOTAL = {
    "mha": 17308032,
    "gqa4": 17308032,
    "gqa2": 17308032,
    "mqa": 17308032,
    "swa256": 17308032,
    "null": 17308032,
    "mla192": 17311104,
    "mla96": 17309568,
}


@pytest.mark.parametrize("name, expected", sorted(EXPECTED_TOTAL.items()))
def test_total_params_at_8_layers(name, expected):
    assert total_params(config.load_variant(name), 8) == expected


def test_mha_non_embedding_matches_base_target():
    total = total_params(config.load_variant("mha"), 8)
    assert total - V * D == 14162304


def test_ladder_non_embedding_counts():
    mha = config.load_variant("mha")
    assert total_params(mha, 6) - V * D == 10621824
    assert total_params(mha, 10) - V * D == 17702784


# --- recipe.yaml (plan §7) ---------------------------------------------------


def test_recipe_matches_plan_7():
    rec = config.load_recipe()
    assert rec["optimizer"]["name"] == "adamw"
    assert rec["optimizer"]["betas"] == [0.9, 0.95]
    assert rec["optimizer"]["eps"] == 1e-8
    assert isinstance(rec["optimizer"]["eps"], float)
    assert rec["optimizer"]["weight_decay"] == 0.1
    assert rec["optimizer"]["grad_clip_norm"] == 1.0
    assert rec["batch"]["sequences_per_step"] == 64
    assert rec["batch"]["sequence_length"] == 1024
    assert rec["batch"]["tokens_per_step"] == 65536
    assert rec["dropout"] == 0.0
    assert rec["schedule"] == {
        "name": "wsd",
        "warmup_steps": 100,
        "decay_fraction": 0.2,
        "decay_shape": "one_minus_sqrt",
        "final_lr": 0.0,
    }


def test_tokens_per_step_is_sequences_times_length():
    b = config.load_recipe()["batch"]
    assert b["sequences_per_step"] * b["sequence_length"] == b["tokens_per_step"] == 65536


def test_budget_steps_match_plan_and_ceiling_formula():
    plan = {50: 763, 100: 1526, 200: 3052, 300: 4578}
    budgets = config.load_recipe()["budgets"]
    for millions, steps in plan.items():
        assert -(-(millions * 1_000_000) // 65536) == steps
        assert budgets[f"{millions}M"] == steps


def test_unset_recipe_values_are_none():
    rec = config.load_recipe()
    assert rec["batch"]["micro_batch"] is None
    assert rec["grad_scaler"]["init_scale"] is None
    assert rec["grad_scaler"]["growth_interval"] is None


# --- prereg.yaml (plan §1.2, §1.4, §8.1, §8.4, §10.3) ------------------------


def test_prereg_primary_comparisons():
    p = config.load_prereg()["primary"]
    assert p == {
        "reference": "mha",
        "comparisons": ["gqa4", "mqa", "mla192"],
        "metric": "final_val_loss",
        "budget": "300M",
        "two_sided": True,
        "family_alpha": 0.05,
        "correction": "holm",
    }


def test_prereg_test_selection():
    ts = config.load_prereg()["test_selection"]
    assert ts["sigma_init"] == {"variant": "mha", "init_seeds": [1, 2, 3, 4, 5], "budget": "100M"}
    assert ts["reproducibility_pairs"] == {"variant": "mha", "init_seeds": [1, 2, 3], "budget": "100M"}
    assert ts["paired_if_s_rep_ratio_below"] == 0.6
    assert ts["otherwise"] == "welch"


def test_prereg_seed_count():
    sc = config.load_prereg()["seed_count"]
    assert sc["base_seeds"] == [1, 2, 3, 4, 5]
    assert sc["added_seeds"] == [6, 7, 8]
    assert sc["power"] == 0.80
    assert sc["max_undetectable_param_fraction"] == 0.05
    assert sc["leave_arm_a_if_extended"] == ["mla96", "swa256"]
    assert not set(sc["added_seeds"]) & set(config.load_prereg()["null_control"]["init_seeds"])


def test_prereg_exploratory_and_controls():
    p = config.load_prereg()
    assert p["exploratory"] == {"variants": ["gqa2", "mla96", "swa256"], "init_seeds": [1, 2, 3]}
    assert p["positive_control"] == {
        "small": "mha_L6",
        "reference": "mha",
        "init_seeds": [1, 2, 3, 4, 5],
        "budget": "300M",
        "alpha": 0.05,
    }
    assert p["null_control"] == {
        "variant": "null",
        "reference": "mha",
        "init_seeds": [11, 12, 13, 14, 15],
        "data_seed": 1,
        "budget": "100M",
    }


def test_prereg_context_use():
    cu = config.load_prereg()["context_use"]
    assert cu["buckets"] == [[0, 64], [64, 256], [256, 512], [512, 1024]]
    assert cu["late_context_min"] == 512
    assert cu["icl_score"] == {"late": [450, 550], "early": [40, 60]}


def test_prereg_named_variants_exist():
    p = config.load_prereg()
    named = {
        p["primary"]["reference"],
        *p["primary"]["comparisons"],
        *p["exploratory"]["variants"],
        p["positive_control"]["small"],
        p["positive_control"]["reference"],
        p["null_control"]["variant"],
        p["null_control"]["reference"],
        p["test_selection"]["sigma_init"]["variant"],
        p["test_selection"]["reproducibility_pairs"]["variant"],
        *p["seed_count"]["leave_arm_a_if_extended"],
    }
    assert named <= set(config.variant_names()) | set(config.ladder_names())


# --- Ladder and arms (plan §8.1 to §8.5) -------------------------------------


def test_ladder_files():
    l6, l10 = config.load_ladder("mha_L6"), config.load_ladder("mha_L10")
    assert (l6["name"], l6["variant"], l6["model"]["n_layers"]) == ("mha_L6", "mha", 6)
    assert (l10["name"], l10["variant"], l10["model"]["n_layers"]) == ("mha_L10", "mha", 10)


def test_arm_iso_token():
    arm = config.load_arm("iso_token")
    assert arm["tag"] == "isoparam"
    assert arm["d_ff"] == "from_variant"
    assert arm["budget"] == "300M"
    assert arm["branches"] == ["100M"]
    assert arm["data_seed"] == 1
    assert arm["timing"] is False
    assert "variants" not in arm


def test_arm_iso_backbone():
    arm = config.load_arm("iso_backbone")
    assert arm["tag"] == "isobackbone"
    assert arm["d_ff"] == 1024
    assert arm["variants"] == ["gqa4", "gqa2", "mqa"]
    assert arm["init_seeds"] == [1, 2, 3]
    assert arm["budget"] == "100M"
    assert arm["branches"] == []


def test_arm_iso_wallclock():
    arm = config.load_arm("iso_wallclock")
    assert arm["tag"] == "isowallclock"
    assert arm["variants"] == ["mha", "swa256", "mla192"]
    assert arm["init_seeds"] == [1, 2, 3]
    assert arm["budget"] == "from_timing_pass"
    assert arm["reference_steps"] == 1526
    assert arm["timing"] is True
    assert arm["max_elapsed_deviation"] == 0.03


# --- resolve ------------------------------------------------------------------


def test_resolved_top_level_keys_are_exact():
    assert set(res()) == {
        "variant", "arm", "ladder", "model", "init", "precision",
        "params", "recipe", "run", "overrides",
    }
    assert set(res()["model"]) == MODEL_KEYS


@pytest.mark.parametrize("name", sorted(VARIANT_TABLE))
def test_default_iso_token_resolve(name):
    r = res(name)
    attention, d_ff = VARIANT_TABLE[name]
    assert r["variant"] == name
    assert r["arm"] == "iso_token"
    assert r["ladder"] is None
    assert r["model"]["d_ff"] == d_ff
    assert r["model"]["attention"] == attention
    assert r["run"] == {
        "init_seed": 1,
        "data_seed": 1,
        "peak_lr": 1e-3,
        "budget": "300M",
        "steps": 4578,
        "branches": ["100M"],
    }
    assert r["overrides"] == {}


def test_resolve_copies_base_sections():
    r = res()
    base = config.load_base()
    assert r["init"] == base["init"]
    assert r["precision"] == base["precision"]
    assert r["params"] == base["params"]
    assert r["recipe"] == config.load_recipe()


def test_iso_backbone_fixes_d_ff_and_uses_100m():
    for name in ("gqa4", "gqa2", "mqa"):
        r = res(name, "iso_backbone")
        assert r["model"]["d_ff"] == 1024
        assert r["run"]["steps"] == 1526
        assert r["run"]["budget"] == "100M"
        assert r["run"]["branches"] == []


@pytest.mark.parametrize("name", ["mha", "mla192"])
def test_iso_backbone_rejects_variants_not_in_arm(name):
    with pytest.raises(config.ConfigError, match=name):
        res(name, "iso_backbone")


def test_ladder_mha_l6():
    r = res("mha", ladder="mha_L6")
    assert r["model"]["n_layers"] == 6
    assert r["ladder"] == "mha_L6"
    assert r["run"]["branches"] == ["100M", "200M"]
    assert r["run"]["steps"] == 4578
    other = res("mha")["model"]
    other.pop("n_layers")
    r["model"].pop("n_layers")
    assert r["model"] == other


def test_ladder_mha_l10():
    assert res("mha", ladder="mha_L10")["model"]["n_layers"] == 10


def test_ladder_with_non_mha_variant_raises():
    with pytest.raises(config.ConfigError, match="mha"):
        res("gqa4", ladder="mha_L6")


def test_ladder_with_other_arm_raises():
    with pytest.raises(config.ConfigError, match="iso_token"):
        res("mha", "iso_wallclock", ladder="mha_L6", steps=100)


def test_iso_wallclock_without_steps_raises():
    with pytest.raises(config.ConfigError, match="steps"):
        res("mha", "iso_wallclock")


def test_iso_wallclock_with_steps_resolves():
    r = res("mla192", "iso_wallclock", steps=1400)
    assert r["run"]["steps"] == 1400
    assert r["run"]["budget"] is None
    assert r["run"]["branches"] == []
    assert r["model"]["d_ff"] == 1048


def test_budget_100m_default_branch_is_not_shorter_and_raises():
    with pytest.raises(config.ConfigError, match="100M"):
        res("mha", budget="100M")


def test_budget_100m_with_no_branches_resolves():
    r = res("mha", budget="100M", branches=[])
    assert r["run"]["steps"] == 1526
    assert r["run"]["budget"] == "100M"
    assert r["run"]["branches"] == []


def test_smoke_run_with_steps_76_and_no_branches():
    r = res("mha", steps=76, branches=[])
    assert r["run"]["steps"] == 76
    assert r["run"]["branches"] == []


def test_steps_76_with_default_branch_raises():
    with pytest.raises(config.ConfigError, match="100M"):
        res("mha", steps=76)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_non_positive_or_non_int_steps_raise(bad):
    with pytest.raises(config.ConfigError):
        res("mha", steps=bad, branches=[])


def test_unknown_variant_arm_and_ladder_raise():
    with pytest.raises(config.ConfigError, match="nope"):
        res("nope")
    with pytest.raises(config.ConfigError, match="nope"):
        res("mha", "nope")
    with pytest.raises(config.ConfigError, match="nope"):
        res("mha", ladder="nope")


@pytest.mark.parametrize("bad", [0.0, -1e-3, float("nan"), float("inf"), "1e-3", True])
def test_bad_peak_lr_raises(bad):
    with pytest.raises(config.ConfigError, match="peak_lr"):
        res(peak_lr=bad)


def test_peak_lr_is_stored_as_float():
    r = res(peak_lr=1)
    assert r["run"]["peak_lr"] == 1.0
    assert type(r["run"]["peak_lr"]) is float


@pytest.mark.parametrize("bad", [-1, 1.0, "1", True])
def test_bad_init_seed_raises(bad):
    with pytest.raises(config.ConfigError, match="init_seed"):
        res(init_seed=bad)


def test_init_seed_zero_is_allowed():
    assert res(init_seed=0)["run"]["init_seed"] == 0


def test_data_seed_defaults_to_arm_and_can_be_set():
    assert res()["run"]["data_seed"] == 1
    assert res(data_seed=7)["run"]["data_seed"] == 7


def test_unknown_branch_label_raises():
    with pytest.raises(config.ConfigError, match="999M"):
        res(branches=["999M"])


def test_branch_not_shorter_than_run_raises():
    with pytest.raises(config.ConfigError, match="200M"):
        res(branches=["200M"], steps=3052)


def test_overrides_are_applied_and_recorded():
    ov = {"recipe.grad_scaler.init_scale": 1024.0, "model.d_model": 64}
    r = res(overrides=ov)
    assert r["recipe"]["grad_scaler"]["init_scale"] == 1024.0
    assert r["model"]["d_model"] == 64
    assert r["overrides"] == {"recipe.grad_scaler.init_scale": 1024.0, "model.d_model": 64}


def test_overrides_do_not_mutate_base_files_or_later_resolves():
    res(overrides={"recipe.grad_scaler.init_scale": 1024.0, "model.d_model": 64})
    fresh = res()
    assert fresh["recipe"]["grad_scaler"]["init_scale"] is None
    assert fresh["model"]["d_model"] == 384
    assert fresh["overrides"] == {}
    assert config.load_recipe()["grad_scaler"]["init_scale"] is None
    assert config.load_base()["model"]["d_model"] == 384


@pytest.mark.parametrize(
    "path",
    ["recipe.nope", "model.d_model.x", "run.nope", "overrides.x", "overrides", "recipe..batch", ""],
)
def test_bad_override_paths_raise(path):
    with pytest.raises(config.ConfigError):
        res(overrides={path: 1})


def test_null_resolves_as_mha_except_variant():
    null = res("null")
    mha = res("mha")
    assert null["variant"] == "null"
    null["variant"] = "mha"
    assert null == mha


# --- require_complete ----------------------------------------------------------

UNSET_PATHS = {
    "recipe.batch.micro_batch",
    "recipe.grad_scaler.init_scale",
    "recipe.grad_scaler.growth_interval",
}


def test_resolve_never_fails_for_unset_values():
    assert res()["recipe"]["batch"]["micro_batch"] is None


def test_require_complete_raises_and_lists_the_three_unset_paths():
    with pytest.raises(config.ConfigError) as info:
        config.require_complete(res())
    listed = set(re.findall(r"recipe(?:\.\w+)+", str(info.value)))
    assert listed == UNSET_PATHS


def test_require_complete_passes_when_all_three_are_supplied():
    r = res(
        overrides={
            "recipe.batch.micro_batch": 64,
            "recipe.grad_scaler.init_scale": 1024.0,
            "recipe.grad_scaler.growth_interval": 2000,
        }
    )
    assert config.require_complete(r) is None


def test_require_complete_names_only_the_still_unset_path():
    r = res(overrides={"recipe.batch.micro_batch": 64, "recipe.grad_scaler.init_scale": 1024.0})
    with pytest.raises(config.ConfigError) as info:
        config.require_complete(r)
    assert set(re.findall(r"recipe(?:\.\w+)+", str(info.value))) == {"recipe.grad_scaler.growth_interval"}


# --- config_hash --------------------------------------------------------------


def test_hash_equal_for_identical_resolves():
    assert config.config_hash(res()) == config.config_hash(res())


@pytest.mark.parametrize(
    "other",
    [
        dict(init_seed=2),
        dict(peak_lr=2e-3),
        dict(data_seed=2),
        dict(overrides={"model.d_model": 64}),
    ],
)
def test_hash_differs_when_a_run_input_differs(other):
    assert config.config_hash(res()) != config.config_hash(res(**other))


def test_hash_differs_for_another_variant():
    assert config.config_hash(res("mha")) != config.config_hash(res("gqa4"))


def test_hash_unchanged_by_key_order():
    r = res()
    reversed_r = _reverse_keys(r)
    assert list(reversed_r) != list(r)
    assert config.config_hash(reversed_r) == config.config_hash(r)


def test_hash_is_64_char_hex_and_matches_the_definition():
    r = res()
    h = config.config_hash(r)
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    canonical = json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert h == hashlib.sha256(canonical).hexdigest()


def test_hash_stable_across_processes_with_different_hash_seeds():
    expected = config.config_hash(res("mha", init_seed=1, peak_lr=1e-3))
    code = (
        "from src import config\n"
        "print(config.config_hash(config.resolve('mha', 'iso_token', init_seed=1, peak_lr=1e-3)))\n"
    )
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.strip() == expected
