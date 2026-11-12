"""Exact parameter counts of every variant at the iso-parameter target (plan §3.2, §4.6).

The numbers are the plan's table. Each model is built once per process (the
solver and the independent count both cache their builds), so the file stays
fast. Everything here is exact integer arithmetic; the only non-integer bound
is the MLA offset limit, read from configs/base.yaml.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from scripts import solve_config
from scripts.solve_config import SolveError, describe, main, resolved_yaml, solve, solve_d_ff
from src import config
from src.model import build_model

TARGET_TOTAL = 17_308_032
TARGET_NON_EMBEDDING = 14_162_304
VOCAB = 8_192
D_MODEL = 384
N_LAYERS = 8

# name: (d_ff, attention params per layer, total params, offset)
PLAN_TABLE = {
    "mha": (1024, 589_824, 17_308_032, 0),
    "gqa4": (1152, 442_368, 17_308_032, 0),
    "gqa2": (1216, 368_640, 17_308_032, 0),
    "mqa": (1248, 331_776, 17_308_032, 0),
    "swa256": (1024, 589_824, 17_308_032, 0),
    "null": (1024, 589_824, 17_308_032, 0),
    "mla192": (1048, 562_560, 17_311_104, 3_072),
    "mla96": (1224, 359_616, 17_309_568, 1_536),
}
NON_MLA = ["mha", "gqa4", "gqa2", "mqa", "swa256", "null"]
MLA = ["mla192", "mla96"]


@functools.lru_cache(maxsize=None)
def _independent_total(name: str) -> int:
    """Count the parameters of the model built from the configured d_ff, without the solver."""
    model = build_model(config.resolve(name, init_seed=1, peak_lr=1e-3))
    return sum(p.numel() for p in model.parameters())


def test_table_covers_every_variant():
    assert sorted(PLAN_TABLE) == sorted(config.variant_names())


@pytest.mark.parametrize("name", list(PLAN_TABLE))
def test_solver_reproduces_plan_table(name):
    d_ff, attn, total, offset = PLAN_TABLE[name]
    sol = solve(name)
    assert sol.variant == name
    assert sol.ladder is None
    assert sol.arm == "iso_token"
    assert sol.n_layers == N_LAYERS
    assert sol.d_ff == d_ff
    assert sol.attention_params_per_layer == attn
    assert sol.total_params == total
    assert sol.target_total == TARGET_TOTAL
    assert sol.offset == offset
    assert sol.relative_offset == offset / TARGET_TOTAL
    assert sol.non_embedding_params == total - VOCAB * D_MODEL


@pytest.mark.parametrize("name", NON_MLA)
def test_non_mla_rows_hit_target_exactly(name):
    sol = solve(name)
    assert sol.total_params == TARGET_TOTAL
    assert sol.non_embedding_params == TARGET_NON_EMBEDDING
    assert sol.offset == 0
    assert sol.attention_gain_params_per_layer == 0
    assert sol.d_ff % 8 == 0


def test_mla_rows_carry_gains_only():
    assert solve("mla192").attention_gain_params_per_layer == 384
    assert solve("mla96").attention_gain_params_per_layer == 192


@pytest.mark.parametrize("name", list(PLAN_TABLE))
def test_total_matches_independent_count(name):
    assert solve(name).total_params == _independent_total(name)
    assert _independent_total(name) == PLAN_TABLE[name][2]


@pytest.mark.parametrize("name", MLA)
def test_mla_offset_is_two_latent_norms_per_layer(name):
    d_c = config.load_variant(name)["attention"]["d_c"]
    sol = solve(name)
    assert sol.offset == 8 * 2 * d_c
    assert sol.offset == N_LAYERS * sol.attention_gain_params_per_layer


@pytest.mark.parametrize("name", MLA)
def test_mla_offset_is_below_the_allowed_fraction(name):
    bound = config.load_base()["params"]["mla_max_relative_offset"]
    sol = solve(name)
    assert abs(sol.relative_offset) < bound
    assert round(100 * sol.relative_offset, 3) == {"mla192": 0.018, "mla96": 0.009}[name]


@pytest.mark.parametrize("name", list(PLAN_TABLE))
def test_parameter_formula(name):
    sol = solve(name)
    attn = sol.attention_params_per_layer
    d_ff = sol.d_ff
    expected = VOCAB * D_MODEL + N_LAYERS * (attn + 3 * D_MODEL * d_ff + 2 * D_MODEL) + D_MODEL
    assert sol.total_params == expected


def test_non_mla_d_ff_values_are_multiples_of_eight():
    for name in NON_MLA:
        assert solve(name).d_ff % 8 == 0


def test_ladder_counts():
    six = describe("mha", ladder="mha_L6")
    ten = describe("mha", ladder="mha_L10")
    assert six.n_layers == 6
    assert six.non_embedding_params == 10_621_824
    assert ten.n_layers == 10
    assert ten.non_embedding_params == 17_702_784
    assert solve("mha").non_embedding_params == 14_162_304
    assert six.d_ff == ten.d_ff == solve("mha").d_ff == 1024


@pytest.mark.parametrize("name", ["gqa4", "gqa2", "mqa"])
def test_iso_backbone_totals_drift_down(name):
    sol = describe(name, arm="iso_backbone")
    assert sol.arm == "iso_backbone"
    assert sol.d_ff == 1024
    iso_d_ff = solve(name).d_ff
    assert sol.total_params == TARGET_TOTAL - 8 * 3 * 384 * (iso_d_ff - 1024)
    assert sol.total_params < TARGET_TOTAL


def test_iso_backbone_order_is_gqa4_gqa2_mqa():
    totals = {n: describe(n, arm="iso_backbone").total_params for n in ["gqa4", "gqa2", "mqa"]}
    assert totals["gqa4"] > totals["gqa2"] > totals["mqa"]


def test_target_one_unit_off_fails_loudly():
    with pytest.raises(SolveError):
        solve("mha", target_total=17_308_033)


def test_target_not_a_multiple_of_one_d_ff_step_fails_loudly():
    with pytest.raises(SolveError):
        solve("mha", target_total=TARGET_TOTAL + 5)
    with pytest.raises(SolveError):
        solve_d_ff("mha", target_total=TARGET_TOTAL + 5)


def test_target_too_small_fails_loudly():
    with pytest.raises(SolveError):
        solve("mha", target_total=1000)
    with pytest.raises(SolveError):
        solve_d_ff("mha", target_total=1000)


def test_one_d_ff_step_up_solves_to_1025():
    # One unit of d_ff for mha costs 3 x 384 x 8 = 9,216 parameters.
    target = TARGET_TOTAL + 9_216
    assert solve_d_ff("mha", target_total=target) == 1025
    sol = solve("mha", target_total=target)
    assert sol.d_ff == 1025
    assert sol.total_params == target
    assert sol.offset == 0


def test_config_disagreement_names_both_values(monkeypatch):
    real_load_variant = config.load_variant

    def patched(name):
        doc = real_load_variant(name)
        if name == "mha":
            doc = {**doc, "d_ff": 1032}
        return doc

    monkeypatch.setattr(config, "load_variant", patched)
    with pytest.raises(SolveError) as excinfo:
        solve("mha")
    message = str(excinfo.value)
    assert "1032" in message
    assert "1024" in message


def test_gain_offset_at_or_above_bound_raises():
    bound = config.load_base()["params"]["mla_max_relative_offset"]
    # 3,072 on a target of one million is 0.31%, far above the bound.
    with pytest.raises(SolveError):
        solve_config._check_offset(
            total=1_000_000 + 3_072, target_total=1_000_000, n_layers=8, gains=384, max_relative=bound
        )
    # Exactly at the bound is not strictly below it.
    assert 500 / 1_000_000 == bound
    with pytest.raises(SolveError):
        solve_config._check_offset(
            total=1_000_000 + 500, target_total=1_000_000, n_layers=1, gains=500, max_relative=bound
        )


def test_gain_offset_below_bound_passes_and_mismatch_raises():
    bound = config.load_base()["params"]["mla_max_relative_offset"]
    offset = solve_config._check_offset(
        total=TARGET_TOTAL + 3_072, target_total=TARGET_TOTAL, n_layers=8, gains=384, max_relative=bound
    )
    assert offset == 3_072
    with pytest.raises(SolveError):
        solve_config._check_offset(
            total=TARGET_TOTAL + 3_071, target_total=TARGET_TOTAL, n_layers=8, gains=384, max_relative=bound
        )
    with pytest.raises(SolveError):
        solve_config._check_offset(
            total=TARGET_TOTAL + 1, target_total=TARGET_TOTAL, n_layers=8, gains=0, max_relative=bound
        )


def test_resolved_yaml_round_trips_to_resolved_config():
    text = resolved_yaml(solve("mla192"))
    assert yaml.safe_load(text) == config.resolve("mla192", init_seed=1, peak_lr=1e-3)
    assert "17311104" in text
    assert "3072" in text


def test_resolved_yaml_of_a_solved_target_carries_its_d_ff():
    sol = solve("mha", target_total=TARGET_TOTAL + 9_216)
    loaded = yaml.safe_load(resolved_yaml(sol))
    assert loaded["model"]["d_ff"] == 1025


def test_cli_prints_resolved_yaml(capsys):
    assert main(["--variant", "mha"]) == 0
    out = capsys.readouterr().out
    assert yaml.safe_load(out) == config.resolve("mha", init_seed=1, peak_lr=1e-3)


def test_cli_all_prints_one_line_per_variant_and_ladder(capsys):
    assert main(["--all"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(config.variant_names()) + len(config.ladder_names())
    assert sum(line.startswith("variant ") for line in lines) == len(config.variant_names())
    assert sum(line.startswith("ladder ") for line in lines) == len(config.ladder_names())


def test_cli_target_off_by_one_returns_1_and_writes_reason(capsys):
    assert main(["--variant", "mha", "--target", "17308033"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "17308033" in captured.err
