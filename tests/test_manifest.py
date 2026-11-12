"""T-18: run names and run manifests (plan §10.6, §11)."""

import copy
import json
import re

import pytest
import torch

from src import config
from src.training.schedule import decay_start
from src.utils import manifest as mf
from src.utils.manifest import ManifestError, RunName
from tests.fixtures.fake_manifests import fake_manifest

LR_GRID = [3e-4, 6e-4, 1.2e-3, 2.4e-3, 4.8e-3]
LR_GRID_WITH_HALVES_AND_DOUBLES = LR_GRID + [lr / 2 for lr in LR_GRID] + [lr * 2 for lr in LR_GRID]

EXPECTED_TOP_KEYS = {
    "schema_version",
    "name",
    "kind",
    "completed",
    "git_sha",
    "config_hash",
    "resolved",
    "seeds",
    "data",
    "environment",
    "backend",
    "mode",
    "precision",
    "steps",
    "scaler",
    "gpu_shared",
    "session_ids",
    "resume_count",
    "step_seconds",
    "final_metrics",
    "branch",
    "lr_source",
    "timing_pass_id",
}
NESTED_BLOCKS = ("seeds", "data", "environment", "steps", "scaler", "branch")

# Sentinel replacement: a list is the wrong type for every field of a manifest.
WRONG_TYPE = []


# --- helpers -----------------------------------------------------------------


def _has(problems, *words):
    """Some problem mentions every word."""
    return any(all(word in problem for word in words) for problem in problems)


def _at(manifest, path):
    node = manifest
    for part in path.split("."):
        node = node[part]
    return node


def _delete(manifest, path):
    *parents, leaf = path.split(".")
    node = manifest
    for part in parents:
        node = node[part]
    del node[leaf]


def _replace(manifest, path, value):
    *parents, leaf = path.split(".")
    node = manifest
    for part in parents:
        node = node[part]
    node[leaf] = value


def _all_paths(manifest):
    """Every top-level key, and every key of every nested block that is present."""
    paths = list(manifest)
    for block in NESTED_BLOCKS:
        if isinstance(manifest.get(block), dict):
            paths += [f"{block}.{key}" for key in manifest[block]]
    return paths


RUN = fake_manifest("mha")
BRANCH = fake_manifest("mha", branch="100M")
CPU = fake_manifest("mha", gpu=False)
WALLCLOCK = fake_manifest("mla192", arm="iso_wallclock", steps=1400)
MANIFESTS = {"run": RUN, "branch": BRANCH, "cpu": CPU, "wallclock": WALLCLOCK}
SCHEMA_CASES = [(label, path) for label, m in MANIFESTS.items() for path in _all_paths(m)]


# --- run names: the plan's example and resolved configs ------------------------


def test_plan_example_parses_and_formats_back():
    text = "mla192-isoparam-300M-lr1e3-s2-d1-b100M"
    name = mf.parse_run_name(text)
    assert name == RunName(
        variant="mla192",
        arm="isoparam",
        tokens="300M",
        peak_lr=1e-3,
        init_seed=2,
        data_seed=1,
        branch="100M",
    )
    assert mf.format_run_name(name) == text


@pytest.mark.parametrize(
    "variant, arm, kwargs, expected",
    [
        ("mha", "iso_token", dict(init_seed=1, peak_lr=1e-3), "mha-isoparam-300M-lr1e3-s1-d1"),
        ("mla192", "iso_token", dict(init_seed=2, peak_lr=1e-3), "mla192-isoparam-300M-lr1e3-s2-d1"),
        (
            "mha",
            "iso_token",
            dict(init_seed=1, peak_lr=1e-3, ladder="mha_L6"),
            "mha_L6-isoparam-300M-lr1e3-s1-d1",
        ),
        (
            "mha",
            "iso_token",
            dict(init_seed=1, peak_lr=1e-3, ladder="mha_L10"),
            "mha_L10-isoparam-300M-lr1e3-s1-d1",
        ),
        ("null", "iso_token", dict(init_seed=1, peak_lr=1e-3), "null-isoparam-300M-lr1e3-s1-d1"),
        (
            "gqa4",
            "iso_backbone",
            dict(init_seed=3, peak_lr=2.4e-3),
            "gqa4-isobackbone-100M-lr2.4e3-s3-d1",
        ),
        (
            "mla192",
            "iso_wallclock",
            dict(init_seed=1, peak_lr=1e-3, steps=1400),
            "mla192-isowallclock-1400steps-lr1e3-s1-d1",
        ),
    ],
)
def test_run_name_from_resolved_config(variant, arm, kwargs, expected):
    resolved = config.resolve(variant, arm, **kwargs)
    assert mf.run_name(resolved) == expected
    assert mf.format_run_name(mf.parse_run_name(expected)) == expected


@pytest.mark.parametrize(
    "resolved_args, branch, expected",
    [
        (dict(ladder="mha_L6"), "100M", "mha_L6-isoparam-300M-lr1e3-s1-d1-b100M"),
        (dict(ladder="mha_L6"), "200M", "mha_L6-isoparam-300M-lr1e3-s1-d1-b200M"),
        ({}, "100M", "mha-isoparam-300M-lr1e3-s1-d1-b100M"),
    ],
)
def test_branch_run_name(resolved_args, branch, expected):
    resolved = config.resolve("mha", "iso_token", init_seed=1, peak_lr=1e-3, **resolved_args)
    assert mf.run_name(resolved, branch) == expected
    parsed = mf.parse_run_name(expected)
    assert parsed.branch == branch
    assert mf.format_run_name(parsed) == expected


def test_branch_not_among_the_runs_branches_is_refused():
    resolved = config.resolve("mha", "iso_token", init_seed=1, peak_lr=1e-3)
    assert resolved["run"]["branches"] == ["100M"]
    with pytest.raises(ManifestError, match="200M"):
        mf.run_name(resolved, "200M")


def test_run_name_of_an_unknown_arm_is_refused():
    resolved = config.resolve("mha", "iso_token", init_seed=1, peak_lr=1e-3)
    resolved["arm"] = "no_such_arm"
    with pytest.raises(config.ConfigError):
        mf.run_name(resolved)


# --- learning rates ----------------------------------------------------------


@pytest.mark.parametrize(
    "lr, text",
    [
        (1e-3, "1e3"),
        (1.2e-3, "1.2e3"),
        (3e-4, "3e4"),
        (4.8e-3, "4.8e3"),
        (6e-4, "6e4"),
        (2.4e-3, "2.4e3"),
        (1.5e-4, "1.5e4"),
        (9.6e-3, "9.6e3"),
        (0.5, "5e1"),
    ],
)
def test_format_lr_examples(lr, text):
    assert mf.format_lr(lr) == text


@pytest.mark.parametrize("lr", LR_GRID_WITH_HALVES_AND_DOUBLES)
def test_lr_round_trips_in_both_directions(lr):
    text = mf.format_lr(lr)
    assert mf.parse_lr(text) == lr
    assert mf.format_lr(mf.parse_lr(text)) == text


def test_parse_lr_gives_the_float_it_names():
    assert mf.parse_lr("1.2e3") == float("1.2e-3")


@pytest.mark.parametrize(
    "lr", [0.0, 1.0, 1.5, -1e-3, float("nan"), float("inf"), 0.1 + 0.2, 1.2345678e-3]
)
def test_format_lr_refuses_values_it_cannot_write(lr):
    with pytest.raises(ManifestError):
        mf.format_lr(lr)


@pytest.mark.parametrize(
    "text", ["1e0", "1.0e3", "1.2e-3", "12e3", "0.5e1", "1.2", "e3", "1e", "1.2345678e3", ""]
)
def test_parse_lr_refuses_text_it_cannot_read(text):
    with pytest.raises(ManifestError):
        mf.parse_lr(text)


# --- run names: round trips and malformed names --------------------------------


@pytest.mark.parametrize("lr", LR_GRID_WITH_HALVES_AND_DOUBLES)
def test_every_learning_rate_round_trips_in_a_name(lr):
    name = RunName("mha_L6", "isoparam", "300M", lr, 2, 1, "200M")
    text = mf.format_run_name(name)
    assert mf.parse_run_name(text) == name
    assert mf.format_run_name(mf.parse_run_name(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "mha-isoparam-300M-lr1e3-s1-d1",
        "mha-isoparam-1400steps-lr1e3-s1-d1-b100M",
        "null-isobackbone-100M-lr3e4-s0-d0",
        "mha_L6-isoparam-300M-lr1.2e3-s10-d2-b200M",
    ],
)
def test_well_formed_names_round_trip(text):
    assert mf.format_run_name(mf.parse_run_name(text)) == text


MALFORMED_NAMES = [
    "",
    "mha-isoparam-300M-lr1e3-s1",  # missing data seed
    "mha-isoparam-lr1e3-s1-d1",  # missing tokens
    "mha-isoparam-300M-d1",  # missing learning rate and init seed
    "mha-300M-isoparam-lr1e3-s1-d1",  # wrong order
    "mha-isoparam-300M-lr1e0-s1-d1",  # learning rate of 1
    "mha-isoparam-300M-lr2e0-s1-d1",  # learning rate of 2
    "mha-isoparam-300M-lr1.0e3-s1-d1",  # not the canonical spelling
    "mha-isoparam-300M-lr1e3-s-1-d1",  # negative init seed
    "mha-isoparam-300M-lr1e3-s1-d-1",  # negative data seed
    "mha-isoparam-300M-lr1e3-s01-d1",  # leading zero
    "-isoparam-300M-lr1e3-s1-d1",  # empty variant
    "mla-192-isoparam-300M-lr1e3-s1-d1",  # hyphen in the variant
    "mha-isoparam-300M-lr1e3-s1-d1-x",  # trailing junk
    "mha-isoparam-300M-lr1e3-s1-d1-b100M-extra",  # trailing junk after the branch
    "mha-isoparam-300M-lr1e3-s1-d1 ",  # trailing space
    "mha-isoparam-300M-lr1e3-s1-d1-b100",  # branch without the M
    "mha-isoparam-300M-lr1e3-s1-d1-b0M",  # branch of zero steps
    "mha-isoparam-300M-lr1e3-s1-d1-100M",  # branch without the dash-b
    "mha-isoparam-300M-lr1e3-s1-d1-b",  # empty branch
    "mha-isoparam-0M-lr1e3-s1-d1",  # zero tokens
]


@pytest.mark.parametrize("text", MALFORMED_NAMES)
def test_malformed_names_are_refused(text):
    with pytest.raises(ManifestError):
        mf.parse_run_name(text)


@pytest.mark.parametrize(
    "name",
    [
        RunName("mha-x", "isoparam", "300M", 1e-3, 1, 1),
        RunName("mha", "IsoParam", "300M", 1e-3, 1, 1),
        RunName("mha", "isoparam", "300", 1e-3, 1, 1),
        RunName("mha", "isoparam", "300M", 1.0, 1, 1),
        RunName("mha", "isoparam", "300M", 1e-3, -1, 1),
        RunName("mha", "isoparam", "300M", 1e-3, True, 1),
        RunName("mha", "isoparam", "300M", 1e-3, 1, 1, "100"),
    ],
)
def test_invalid_run_name_fields_are_refused(name):
    with pytest.raises(ManifestError):
        mf.format_run_name(name)


# --- the schema --------------------------------------------------------------


def test_valid_fixtures_have_no_problems():
    for manifest in MANIFESTS.values():
        assert mf.validate_manifest(manifest) == []


def test_top_level_keys_are_the_ones_the_brief_names():
    assert set(RUN) == EXPECTED_TOP_KEYS
    assert set(mf.TOP_KEYS) == EXPECTED_TOP_KEYS


@pytest.mark.parametrize("label, path", SCHEMA_CASES)
def test_deleting_any_field_is_reported(label, path):
    manifest = copy.deepcopy(MANIFESTS[label])
    _delete(manifest, path)
    problems = mf.validate_manifest(manifest)
    assert any(problem.startswith(f"{path}:") for problem in problems), problems


@pytest.mark.parametrize("label, path", SCHEMA_CASES)
def test_a_wrong_type_for_any_field_is_reported(label, path):
    manifest = copy.deepcopy(MANIFESTS[label])
    _replace(manifest, path, WRONG_TYPE)
    problems = mf.validate_manifest(manifest)
    assert any(problem.startswith(f"{path}:") for problem in problems), problems


def test_unknown_top_level_key_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["extra"] = 1
    assert _has(mf.validate_manifest(manifest), "extra", "unknown")


def test_unknown_key_inside_a_block_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["extra"] = 1
    assert _has(mf.validate_manifest(manifest), "steps.extra", "unknown")


@pytest.mark.parametrize("value", [None, 1, "x", []])
def test_validate_never_raises_on_a_non_object(value):
    problems = mf.validate_manifest(value)
    assert len(problems) == 1


def test_an_empty_object_reports_every_key_missing():
    problems = mf.validate_manifest({})
    assert len(problems) == len(mf.TOP_KEYS)
    assert all(problem.endswith(": missing") for problem in problems)


def test_validate_never_raises_on_a_mangled_manifest():
    manifest = copy.deepcopy(RUN)
    manifest["resolved"] = {"run": "not a dict", "variant": None}
    manifest["steps"] = None
    problems = mf.validate_manifest(manifest)
    assert problems and all(isinstance(problem, str) for problem in problems)


@pytest.mark.parametrize(
    "path, value",
    [
        ("git_sha", "A" * 40),
        ("git_sha", "a" * 39),
        ("config_hash", "g" * 64),
        ("data.dataset_sha256", "a" * 63),
        ("backend", "flash3"),
        ("mode", "naive"),
        ("precision", "bf16"),
        ("schema_version", 2),
        ("steps.skipped", -1),
        ("resume_count", -1),
        ("environment.compute_capability", (7, 5)),
        ("step_seconds", [0.5, float("nan")]),
    ],
)
def test_out_of_range_or_malformed_values_are_reported(path, value):
    manifest = copy.deepcopy(RUN)
    _replace(manifest, path, value)
    assert _has(mf.validate_manifest(manifest), path)


def test_nan_val_loss_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["final_metrics"]["val_loss"] = float("nan")
    assert _has(mf.validate_manifest(manifest), "val_loss", "finite")


def test_infinite_value_anywhere_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["resolved"]["model"]["rope_theta"] = float("inf")
    assert _has(mf.validate_manifest(manifest), "rope_theta", "finite")


# --- the consistency rules of plan §10.6 -------------------------------------


def test_rule_2_name_must_match_the_resolved_config():
    manifest = copy.deepcopy(RUN)
    manifest["name"] = "mha-isoparam-300M-lr2e3-s1-d1"
    assert _has(mf.validate_manifest(manifest), "name")


def test_rule_3_config_hash_follows_the_resolved_config():
    manifest = copy.deepcopy(RUN)
    manifest["resolved"]["model"]["n_layers"] = 7
    assert _has(mf.validate_manifest(manifest), "config_hash")


def test_rule_3_seeds_follow_the_resolved_config():
    manifest = copy.deepcopy(RUN)
    manifest["seeds"]["data_seed"] = 2
    assert _has(mf.validate_manifest(manifest), "seeds.data_seed")


def test_rule_4_branch_budget_must_be_a_branch_of_the_run():
    manifest = copy.deepcopy(BRANCH)
    manifest["branch"]["budget"] = "200M"
    assert _has(mf.validate_manifest(manifest), "branch.budget", "branches")


def test_rule_4_branch_kind_and_branch_block_agree():
    manifest = copy.deepcopy(RUN)
    manifest["kind"] = "branch"
    assert _has(mf.validate_manifest(manifest), "kind", "branch")

    manifest = copy.deepcopy(BRANCH)
    manifest["kind"] = "run"
    assert _has(mf.validate_manifest(manifest), "branch", "kind")


def test_rule_4_branch_step_for_100M_is_1221():
    assert BRANCH["branch"]["steps"] == 1526
    assert BRANCH["branch"]["branch_step"] == 1221
    manifest = copy.deepcopy(BRANCH)
    manifest["branch"]["branch_step"] = 1222
    assert _has(mf.validate_manifest(manifest), "branch_step")


def test_rule_4_branch_step_for_200M_on_the_ladder_is_2442():
    manifest_200 = fake_manifest("mha", ladder="mha_L6", branch="200M")
    assert manifest_200["branch"]["steps"] == 3052
    assert manifest_200["branch"]["branch_step"] == 2442
    assert manifest_200["name"] == "mha_L6-isoparam-300M-lr1e3-s1-d1-b200M"
    manifest_200["branch"]["branch_step"] = 2441
    assert _has(mf.validate_manifest(manifest_200), "branch_step")


def test_rule_4_branch_steps_must_match_the_budget():
    manifest = copy.deepcopy(BRANCH)
    manifest["branch"]["steps"] = 1500
    assert _has(mf.validate_manifest(manifest), "branch.steps")


def test_rule_4_parent_run_must_match():
    manifest = copy.deepcopy(BRANCH)
    manifest["branch"]["parent_run"] = "mha-isobackbone-100M-lr1e3-s1-d1"
    assert _has(mf.validate_manifest(manifest), "parent_run")


def test_rule_4_run_steps_total_must_match_the_resolved_config():
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["total"] = 4000
    manifest["steps"]["attempted"] = 4000
    manifest["steps"]["taken"] = 4000
    manifest["step_seconds"] = manifest["step_seconds"][:4000]
    assert _has(mf.validate_manifest(manifest), "steps.total", "resolved.run.steps")


def test_rule_5_steps_taken_and_skipped_must_add_up():
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["taken"] = 4570
    problems = mf.validate_manifest(manifest)
    assert _has(problems, "attempted", "taken")


def test_rule_5_skips_in_the_first_100_cannot_exceed_skips():
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["skipped"] = 1
    manifest["steps"]["taken"] = manifest["steps"]["total"] - 1
    manifest["steps"]["skipped_in_first_100"] = 2
    assert _has(mf.validate_manifest(manifest), "skipped_in_first_100")


def test_rule_5_attempted_cannot_exceed_total():
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["total"] = 4000
    manifest["resolved"]["run"]["steps"] = 4000
    assert _has(mf.validate_manifest(manifest), "exceeds", "steps.total")


def test_rule_5_step_seconds_must_have_one_entry_per_attempted_step():
    manifest = copy.deepcopy(RUN)
    manifest["step_seconds"] = manifest["step_seconds"][:-1]
    assert _has(mf.validate_manifest(manifest), "step_seconds", "attempted")


def test_rule_6_completed_run_must_reach_total():
    manifest = copy.deepcopy(RUN)
    manifest["steps"] = {
        "total": 4578,
        "attempted": 4000,
        "taken": 4000,
        "skipped": 0,
        "skipped_in_first_100": 0,
    }
    manifest["step_seconds"] = manifest["step_seconds"][:4000]
    assert _has(mf.validate_manifest(manifest), "completed", "attempted")


def test_rule_6_completed_without_val_loss_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["final_metrics"] = {}
    assert _has(mf.validate_manifest(manifest), "val_loss", "missing")


def test_rule_6_completed_without_final_metrics_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["final_metrics"] = None
    assert _has(mf.validate_manifest(manifest), "final_metrics", "completed")


def test_rule_6_a_completed_trunk_checkpoint_is_reported():
    trunk = fake_manifest("mha", kind="trunk_checkpoint", completed=False)
    assert mf.validate_manifest(trunk) == []
    trunk["completed"] = True
    trunk["final_metrics"] = {"val_loss": 3.0}
    assert _has(mf.validate_manifest(trunk), "trunk_checkpoint", "never completed")


def test_rule_6_an_unfinished_run_is_valid_without_metrics():
    unfinished = fake_manifest("mha", completed=False)
    assert unfinished["final_metrics"] is None
    assert mf.validate_manifest(unfinished) == []


def test_rule_7_resume_count_is_sessions_minus_one():
    manifest = copy.deepcopy(RUN)
    manifest["resume_count"] = 1
    assert _has(mf.validate_manifest(manifest), "resume_count")

    manifest = copy.deepcopy(RUN)
    manifest["session_ids"] = ["s1", "s2"]
    manifest["resume_count"] = 1
    assert mf.validate_manifest(manifest) == []


def test_rule_7_sessions_must_be_at_least_one():
    manifest = copy.deepcopy(RUN)
    manifest["session_ids"] = []
    assert _has(mf.validate_manifest(manifest), "session_ids")


def test_rule_8_cpu_manifest_with_cuda_fields_is_reported():
    manifest = copy.deepcopy(CPU)
    manifest["environment"]["cuda"] = "13.2"
    assert _has(mf.validate_manifest(manifest), "environment.cuda", "null")


def test_rule_8_cpu_manifest_with_a_backend_is_reported():
    manifest = copy.deepcopy(CPU)
    manifest["backend"] = "efficient"
    assert _has(mf.validate_manifest(manifest), "backend", "null")


def test_rule_8_cpu_manifest_must_be_fp32():
    manifest = copy.deepcopy(CPU)
    manifest["precision"] = "fp16"
    assert _has(mf.validate_manifest(manifest), "precision", "fp32")


def test_rule_8_gpu_manifest_without_a_backend_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["backend"] = None
    assert _has(mf.validate_manifest(manifest), "backend", "set")


def test_rule_8_gpu_manifest_without_a_driver_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["environment"]["driver"] = None
    assert _has(mf.validate_manifest(manifest), "environment.driver", "set")


def test_rule_8_gpu_name_removed_but_cuda_kept_is_reported():
    manifest = copy.deepcopy(RUN)
    manifest["environment"]["gpu_name"] = None
    assert _has(mf.validate_manifest(manifest), "environment.cuda", "null")


def test_rule_9_timing_pass_is_required_for_iso_wallclock():
    manifest = copy.deepcopy(WALLCLOCK)
    manifest["timing_pass_id"] = None
    assert _has(mf.validate_manifest(manifest), "timing_pass_id", "required")


def test_rule_9_timing_pass_is_refused_outside_iso_wallclock():
    manifest = copy.deepcopy(RUN)
    manifest["timing_pass_id"] = "SYNTHETIC-timing-pass"
    assert _has(mf.validate_manifest(manifest), "timing_pass_id", "null")


def test_iso_wallclock_fixture_needs_explicit_steps():
    with pytest.raises(ValueError):
        fake_manifest("mla192", arm="iso_wallclock")


def test_several_problems_are_reported_together():
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["taken"] = 4570
    manifest["resume_count"] = 5
    manifest["environment"]["gpu_name"] = None
    problems = mf.validate_manifest(manifest)
    assert len(problems) >= 3
    assert _has(problems, "attempted", "taken")
    assert _has(problems, "resume_count")
    assert _has(problems, "environment.cuda", "null")


def test_build_manifest_lists_every_problem_it_refuses():
    kwargs = dict(
        kind="run",
        resolved=config.resolve("mha", "iso_token", init_seed=1, peak_lr=1e-3),
        completed=True,
        git_sha="f" * 40,
        data=dict(RUN["data"]),
        environment=dict(RUN["environment"]),
        backend="efficient",
        mode="opt",
        precision="fp16",
        steps={"total": 4578, "attempted": 4578, "taken": 4000, "skipped": 0, "skipped_in_first_100": 0},
        final_scale=1.0,
        gpu_shared=False,
        session_ids=["s1"],
        step_seconds=[0.5] * 4578,
        final_metrics={"val_loss": 3.0},
    )
    with pytest.raises(ManifestError) as excinfo:
        mf.build_manifest(**kwargs)
    assert "attempted" in str(excinfo.value)


def test_build_manifest_deep_copies_the_resolved_config():
    resolved = config.resolve("mha", "iso_token", init_seed=1, peak_lr=1e-3)
    manifest = fake_manifest("mha")
    built = mf.build_manifest(
        kind="run",
        resolved=resolved,
        completed=False,
        git_sha="f" * 40,
        data=dict(RUN["data"]),
        environment=dict(RUN["environment"]),
        backend="efficient",
        mode="opt",
        precision="fp16",
        steps={"total": 4578, "attempted": 10, "taken": 10, "skipped": 0, "skipped_in_first_100": 0},
        final_scale=65536.0,
        gpu_shared=False,
        session_ids=["s1"],
        step_seconds=[0.5] * 10,
        final_metrics=None,
    )
    resolved["model"]["n_layers"] = 99
    assert built["resolved"]["model"]["n_layers"] == 8
    assert built["config_hash"] == config.config_hash(built["resolved"])
    assert manifest["config_hash"] == built["config_hash"]


# --- files -------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path):
    manifest = copy.deepcopy(RUN)
    path = mf.write_manifest(manifest, tmp_path)
    assert path == tmp_path / f"{manifest['name']}.json"
    assert mf.read_manifest(path) == manifest
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_branch_file_is_named_with_its_suffix(tmp_path):
    path = mf.write_manifest(BRANCH, tmp_path)
    assert path.name == "mha-isoparam-300M-lr1e3-s1-d1-b100M.json"
    assert mf.read_manifest(path) == BRANCH


def test_file_is_sorted_keys_with_indent_one(tmp_path):
    path = mf.write_manifest(RUN, tmp_path)
    expected = json.dumps(RUN, sort_keys=True, indent=1, allow_nan=False) + "\n"
    assert path.read_text(encoding="utf-8") == expected


def test_a_completed_manifest_is_never_overwritten(tmp_path):
    path = mf.write_manifest(RUN, tmp_path)
    before = path.read_bytes()
    other = fake_manifest("mha", val_loss=2.5)
    assert other["name"] == RUN["name"]
    with pytest.raises(ManifestError, match="completed"):
        mf.write_manifest(other, tmp_path)
    assert path.read_bytes() == before


def test_an_incomplete_manifest_can_be_overwritten(tmp_path):
    unfinished = fake_manifest("mha", completed=False)
    path = mf.write_manifest(unfinished, tmp_path)
    mf.write_manifest(fake_manifest("mha", completed=False, steps=None), tmp_path)
    assert mf.read_manifest(path)["completed"] is False
    mf.write_manifest(RUN, tmp_path)
    assert mf.read_manifest(path) == RUN


def test_an_invalid_manifest_is_not_written(tmp_path):
    manifest = copy.deepcopy(RUN)
    manifest["steps"]["taken"] = 1
    with pytest.raises(ManifestError, match="attempted"):
        mf.write_manifest(manifest, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_read_refuses_a_renamed_file(tmp_path):
    path = mf.write_manifest(RUN, tmp_path)
    renamed = tmp_path / "renamed.json"
    path.rename(renamed)
    with pytest.raises(ManifestError, match="renamed.json"):
        mf.read_manifest(renamed)


def test_read_refuses_an_invalid_file(tmp_path):
    manifest = copy.deepcopy(RUN)
    manifest["resume_count"] = 3
    path = tmp_path / f"{manifest['name']}.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    with pytest.raises(ManifestError, match="resume_count"):
        mf.read_manifest(path)


def test_read_refuses_a_nan_in_the_file(tmp_path):
    text = json.dumps(RUN).replace('"val_loss": 3.0', '"val_loss": NaN')
    path = tmp_path / f"{RUN['name']}.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError):
        mf.read_manifest(path)


def test_read_refuses_a_file_that_is_not_json(tmp_path):
    path = tmp_path / f"{RUN['name']}.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        mf.read_manifest(path)


# --- results -----------------------------------------------------------------


@pytest.mark.parametrize(
    "manifest, expected",
    [
        (RUN, True),
        (BRANCH, True),
        (fake_manifest("mha", completed=False), False),
        (fake_manifest("mha", kind="trunk_checkpoint", completed=False), False),
    ],
)
def test_is_result(manifest, expected):
    assert mf.is_result(manifest) is expected


def test_a_completed_trunk_checkpoint_is_still_not_a_result():
    trunk = fake_manifest("mha", kind="trunk_checkpoint", completed=False)
    trunk["completed"] = True
    assert mf.is_result(trunk) is False


# --- environment and git -----------------------------------------------------


def test_collect_environment_on_this_machine():
    env = mf.collect_environment()
    assert set(env) == {"python", "torch", "cuda", "driver", "gpu_name", "compute_capability"}
    assert env["torch"] == str(torch.__version__)
    if torch.cuda.is_available():
        assert env["cuda"] and env["driver"] and env["gpu_name"]
        assert isinstance(env["compute_capability"], list) and len(env["compute_capability"]) == 2
    else:
        assert env["cuda"] is None
        assert env["driver"] is None
        assert env["gpu_name"] is None
        assert env["compute_capability"] is None


def test_git_sha_of_this_repository_is_forty_hex_characters():
    assert re.fullmatch(r"[0-9a-f]{40}", mf.git_sha())


def test_git_sha_outside_a_repository_is_refused(tmp_path):
    with pytest.raises(ManifestError):
        mf.git_sha(tmp_path)
