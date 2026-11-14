"""Fixes for the session-4 review (docs/review_s4.md), analysis part. Written before the fixes (TASKS.md T-39, O+W).

A worker implements against this file and may not change it. Every number is made up.

H1. The analysis compared, and gave verdicts on, runs trained under different recipes,
    init rules, precision settings or backbones, and runs made with `--override`
    (plan §8.7: the whole of recipe.yaml is held constant; §7 and §13.6: a recipe change
    restarts every arm; §4.1: nothing outside the attention subtree differs, except d_ff
    in Arm A and the layer count in the ladder). The headline's `params` and `d_ff`
    columns came from configs/, whatever model the runs had.
H2. The seed-count rule (plan §1.4: "applied once, after Arm N and the ladder") was
    computed and printed from an incomplete ladder. Near the threshold the decision
    from a partial ladder differed from the complete ladder's. Plan §12.2 item 4
    allows cutting L10; that is a declaration, not an absence.
G6. One recorded diverged run made the tables refuse unless `--skip-unfinished` was
    given, which the final step of the runbook forbids (plan §7.4 item 5: record
    every divergence).
B10, B36, B46. Three deliberate breaks of the review that no test caught.
"""

from __future__ import annotations

import copy
import math

import pytest

from analysis import ladder_fit as lf
from scripts import make_tables as mt
from src import config
from src.model import build_model
from src.utils import manifest as mf
from tests.fixtures import fake_secondary as fs
from tests.fixtures.fake_campaign import LADDER_PARAMS
from tests.fixtures.fake_manifests import fake_manifest
from tests.test_gate_report import losses_of
from tests.test_lr_sweep import with_skips
from tests.fixtures.commitments import analyse_committed, committed, tables_main

PREREG = config.load_prereg()
BASE_SEEDS = tuple(PREREG["seed_count"]["base_seeds"])
L6_SEEDS = tuple(PREREG["positive_control"]["init_seeds"])
SWEEP_SEED = 21


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    # fake_campaign plus the exploratory variants' 100M branches, which Arm B is compared with.
    return list(fs.secondary_campaign(gaps={"mqa": 0.05}, ladder_branches=True))


@pytest.fixture(scope="module")
def analysis(campaign) -> mt.Analysis:
    return analyse_committed(campaign)


def parsed(manifest):
    return mf.parse_run_name(manifest["name"])


def edited(manifest: dict, edit) -> dict:
    """The same manifest with its resolved config changed by `edit`; the hash follows, the name does not."""
    changed = copy.deepcopy(manifest)
    edit(changed["resolved"])
    changed["config_hash"] = config.config_hash(changed["resolved"])
    assert mf.validate_manifest(changed) == [], mf.validate_manifest(changed)
    return changed


def replace(manifests, predicate, change, *, expect: int | None = 1) -> list[dict]:
    out, hit = [], 0
    for m in manifests:
        if predicate(parsed(m)):
            out.append(change(m))
            hit += 1
        else:
            out.append(m)
    assert hit > 0 if expect is None else hit == expect, hit
    return out


def is_run(variant, seed, tokens="300M", branch=None):
    return lambda p: (p.variant, p.init_seed, p.tokens, p.branch, p.data_seed) == (variant, seed, tokens, branch, 1)


# --- H1: one recipe, one init rule, one precision setting, one backbone, no overrides -----------


def _set(path: str, value):
    def edit(resolved):
        node = resolved
        *parents, last = path.split(".")
        for key in parents:
            node = node[key]
        assert last in node, path
        assert node[last] != value, (path, value)
        node[last] = value

    return edit


OTHER_SETTINGS = [
    ("recipe", "recipe.optimizer.weight_decay", 0.0),
    ("recipe", "recipe.schedule.warmup_steps", 500),
    ("recipe", "recipe.grad_scaler.init_scale", 1024.0),
    ("recipe", "recipe.batch.micro_batch", 8),
    ("init", "init.std", 0.5),
    ("precision", "precision.autocast_dtype", "bfloat16"),
    ("backbone", "model.rope_theta", 5000.0),
    ("backbone", "model.norm", "layernorm"),
    ("backbone", "model.d_model", 256),
]


@pytest.mark.parametrize("label, path, value", OTHER_SETTINGS, ids=[p for _, p, _ in OTHER_SETTINGS])
def test_a_run_trained_under_another_setting_is_refused_and_named(campaign, label, path, value):
    target = next(m for m in campaign if is_run("mqa", 3)(parsed(m)))
    mixed = replace(campaign, is_run("mqa", 3), lambda m: edited(m, _set(path, value)))
    with pytest.raises(mt.TableError) as caught:
        analyse_committed(mixed)
    message = str(caught.value)
    assert label in message, message
    assert target["name"] in message, message


def test_a_whole_variant_under_another_recipe_is_refused(campaign):
    """The review's reproduction: every MQA run under another recipe was accepted and judged."""
    def other(m):
        return edited(m, lambda r: (_set("recipe.optimizer.weight_decay", 0.0)(r), _set("recipe.schedule.warmup_steps", 500)(r)))

    mixed = replace(campaign, lambda p: p.variant == "mqa", other, expect=None)
    with pytest.raises(mt.TableError, match="recipe"):
        analyse_committed(mixed)


def test_a_run_nothing_reads_is_still_held_to_the_recipe(campaign):
    """Like versions and backend: a sweep point under another recipe means the recipe moved after the freeze."""
    sweep = fake_manifest("mha", budget="100M", branches=[], init_seed=SWEEP_SEED, peak_lr=4.8e-3, val_loss=3.4)
    odd = edited(sweep, _set("recipe.optimizer.weight_decay", 0.0))
    with pytest.raises(mt.TableError, match="recipe"):
        analyse_committed(campaign + [odd])


def test_a_run_made_with_an_override_is_refused_even_when_the_recipe_agrees(campaign):
    target = next(m for m in campaign if is_run("gqa4", 2)(parsed(m)))
    weight_decay = target["resolved"]["recipe"]["optimizer"]["weight_decay"]

    def with_override(resolved):
        resolved["overrides"] = {"recipe.optimizer.weight_decay": weight_decay}

    mixed = replace(campaign, is_run("gqa4", 2), lambda m: edited(m, with_override))
    with pytest.raises(mt.TableError) as caught:
        analyse_committed(mixed)
    message = str(caught.value)
    assert "override" in message, message
    assert target["name"] in message, message


@pytest.mark.parametrize(
    "variant, seed, path, value",
    [("gqa4", 2, "model.d_ff", 1000), ("mha", 4, "model.n_layers", 6), ("mha_L6", 2, "model.n_layers", 8)],
    ids=["d_ff of one gqa4 run", "depth of one mha run", "depth of one L6 run"],
)
def test_runs_of_one_variant_must_have_one_model(campaign, variant, seed, path, value):
    """d_ff may differ between variants and the layer count along the ladder, never within one name."""
    target = next(m for m in campaign if is_run(variant, seed)(parsed(m)))
    mixed = replace(campaign, is_run(variant, seed), lambda m: edited(m, _set(path, value)))
    with pytest.raises(mt.TableError) as caught:
        analyse_committed(mixed)
    message = str(caught.value)
    assert "model" in message, message
    assert target["name"] in message, message


def test_the_differences_the_plan_allows_are_not_refused(campaign, analysis):
    """Variants (attention, d_ff), the ladder (layers), Arm B (d_ff), a sweep (peak LR), Arm C (steps)."""
    extra = fs.arm_b_manifests(campaign) + [
        fake_manifest("mha", budget="100M", branches=[], init_seed=SWEEP_SEED, peak_lr=4.8e-3, val_loss=3.4),
        fake_manifest("mla192", arm="iso_wallclock", steps=1400, init_seed=1, val_loss=3.5),
    ]
    with_extra = analyse_committed(campaign + extra)
    assert with_extra.comparisons == analysis.comparisons
    assert with_extra.decision == analysis.decision


def test_the_static_columns_come_from_the_manifests(campaign):
    """A campaign whose gqa4 runs all had another d_ff prints that d_ff and that model's parameter count."""
    other_d_ff = 1000
    mixed = replace(campaign, lambda p: p.variant == "gqa4", lambda m: edited(m, _set("model.d_ff", other_d_ff)), expect=None)
    example = next(m for m in mixed if is_run("gqa4", 1)(parsed(m)))
    n_params = build_model(example["resolved"]).n_params
    assert f"{n_params:,}" != mt.static_columns("gqa4")["params"], "the edit must change the count"

    rows = {row["variant"]: row for row in mt.headline_rows(analyse_committed(mixed))}
    assert rows["gqa4"]["d_ff"] == str(other_d_ff)
    assert rows["gqa4"]["params"] == f"{n_params:,}"
    # A variant whose runs have the configs' model prints the configs' numbers, as before.
    assert rows["mqa"]["d_ff"] == mt.static_columns("mqa")["d_ff"]
    assert rows["mqa"]["params"] == mt.static_columns("mqa")["params"]


def test_the_blank_table_keeps_the_configs_static_columns():
    rows = {row["variant"]: row for row in mt.headline_rows(analyse_committed([]))}
    for variant in mt.HEADLINE_VARIANTS:
        static = mt.static_columns(variant)
        assert rows[variant]["params"] == static["params"] and rows[variant]["d_ff"] == static["d_ff"]


# --- H2: the seed-count rule waits for the complete ladder ---------------------------------------


def without(manifests, predicate) -> list[dict]:
    out = [m for m in manifests if not predicate(parsed(m))]
    assert len(out) < len(manifests)
    return out


def ladder_runs(name, seeds):
    return lambda p: p.variant == name and p.init_seed in seeds


def two_point_decision(manifests, analysis_):
    """The rule on MHA's base seeds and L6's registered seeds only, computed here."""
    mha = {s: v for s, v in losses_of(manifests, variant="mha", tokens="300M", branch=None).items() if s in BASE_SEEDS}
    points = {
        "mha": (LADDER_PARAMS["mha"], mha),
        "mha_L6": (LADDER_PARAMS["mha_L6"], losses_of(manifests, variant="mha_L6", tokens="300M", branch=None)),
    }
    return lf.seed_count_rule(mha, lf.fit_ladder(points), test=analysis_.selection.test, prereg=PREREG)


def test_l10_has_three_registered_seeds():
    assert tuple(mt.L10_SEEDS) == (1, 2, 3)  # plan §8.4: "L6 runs 5 seeds; L10 runs 3"


def test_the_complete_ladder_gives_a_decision(analysis):
    assert analysis.decision is not None and analysis.decision.n_seeds in (5, 8)
    assert not any(note.startswith("seed count rule") for note in analysis.notes)


@pytest.mark.parametrize("missing", [(1, 2, 3), (3,), (1, 2)], ids=["absent", "one seed short", "one seed only"])
def test_without_all_of_l10_there_is_no_decision(campaign, missing):
    a = analyse_committed(without(campaign, ladder_runs("mha_L10", missing)))
    assert a.decision is None
    assert f"seed count rule: mha_L10 300M results for seeds {list(missing)} are missing" in a.notes
    assert a.comparisons == {}  # decision 126: no primary verdict without a seed-count decision
    assert mt.gate_values(a)["Seed count and detectable effect"]["seeds per primary"] is None


@pytest.mark.parametrize("missing", [(4, 5), (1,)], ids=["two short", "one short"])
def test_without_all_of_l6_there_is_no_decision(campaign, missing):
    a = analyse_committed(without(campaign, ladder_runs("mha_L6", missing)))
    assert a.decision is None
    assert f"seed count rule: mha_L6 300M results for seeds {list(missing)} are missing" in a.notes


def test_l10_declared_cut_gives_the_two_point_rule(campaign):
    no_l10 = without(campaign, ladder_runs("mha_L10", (1, 2, 3)))
    a = mt.analyse(no_l10, commitments=committed(no_l10, ladder="l6_l8"))
    assert a.decision is not None
    assert a.decision == two_point_decision(no_l10, a)
    assert set(p.name for p in a.rule_fit.points) == {"mha", "mha_L6"}
    assert any("mha_L10" in note and "declared cut" in note for note in a.notes)


def test_l10_declared_cut_is_not_read_by_the_rule_when_its_runs_exist(campaign):
    """A declaration made after the runs exist must give what it gives without them."""
    a = mt.analyse(campaign, commitments=committed(campaign, ladder="l6_l8"))
    assert a.decision == two_point_decision(campaign, a)
    assert set(p.name for p in a.rule_fit.points) == {"mha", "mha_L6"}


def test_the_declaration_does_not_rescue_an_incomplete_l6(campaign):
    a = mt.analyse(without(campaign, ladder_runs("mha_L6", (5,))), commitments={"ladder": "l6_l8", "seed_count": None})
    assert a.decision is None


def test_an_unregistered_l10_seed_does_not_move_the_rule(campaign, analysis):
    extra = fake_manifest("mha", ladder="mha_L10", init_seed=4, branches=["100M", "200M"], val_loss=2.0)
    a = analyse_committed(campaign + [extra])
    assert a.decision == analysis.decision
    assert a.rule_fit == analysis.rule_fit


def test_the_command_line_declares_the_cut(campaign, tmp_path, capsys):
    directory = tmp_path / "manifests"
    for manifest in without(campaign, ladder_runs("mha_L10", (1, 2, 3))):
        mf.write_manifest(manifest, directory)
    waiting, cut = tmp_path / "waiting", tmp_path / "cut"
    assert tables_main(["--manifests", str(directory), "--out", str(waiting)]) == 0
    declared = {"ladder": "l6_l8", "seed_count": None}
    assert mt.main(["--manifests", str(directory), "--out", str(cut)], commitments=declared) == 0
    with pytest.raises(SystemExit):
        mt.main(["--manifests", str(directory), "--out", str(cut), "--l10-cut"])  # the flag is gone (decision 220)
    capsys.readouterr()
    assert "mha_L10 300M results for seeds [1, 2, 3] are missing" in (waiting / "gates.md").read_text(encoding="utf-8")
    assert "declared cut" in (cut / "gates.md").read_text(encoding="utf-8")
    assert "declared cut" not in (waiting / "gates.md").read_text(encoding="utf-8")


# --- G6: a recorded divergence is listed, not refused ---------------------------------------------


def diverged_run(lr=9.6e-3, seed=SWEEP_SEED) -> dict:
    """A sweep point that diverged, as scripts.train records it: unfinished, marked diverged."""
    manifest = fake_manifest(
        "mha", budget="100M", branches=[], init_seed=seed, peak_lr=lr, completed=False,
        final_metrics={"diverged": True, "reason": "non-finite loss in window 3", "step": 250},
    )
    assert mf.validate_manifest(manifest) == []
    assert not mf.is_result(manifest)
    return manifest


def test_a_diverged_run_is_left_out_and_listed_without_the_skip_flag(campaign):
    diverged = diverged_run()
    results, skipped = mt.select_results(campaign + [diverged])
    assert [m["name"] for m in results] == [m["name"] for m in campaign]
    assert skipped == []
    assert mt.diverged_runs(campaign + [diverged]) == [diverged["name"]]
    assert mt.diverged_runs(campaign) == []


def test_an_unfinished_run_that_did_not_diverge_is_still_refused(campaign):
    unfinished = fake_manifest("mha", budget="100M", branches=[], init_seed=SWEEP_SEED, peak_lr=2.4e-3, completed=False)
    with pytest.raises(mt.TableError, match="unfinished"):
        mt.select_results(campaign + [unfinished, diverged_run()])
    assert mt.diverged_runs(campaign + [unfinished]) == []


def test_a_completed_run_is_never_listed_as_diverged(campaign):
    """A completed run that skipped many steps is a result; step parity judges it (plan §13.6)."""
    skippy = with_skips(fake_manifest("mha", budget="100M", branches=[], init_seed=SWEEP_SEED, peak_lr=4.8e-3), 8)
    assert mt.diverged_runs(campaign + [skippy]) == []


def test_the_tables_build_with_a_diverged_run_on_record(campaign, tmp_path, capsys):
    diverged = diverged_run()
    directory = tmp_path / "manifests"
    for manifest in campaign + [diverged]:
        mf.write_manifest(manifest, directory)
    out = tmp_path / "tables"
    assert tables_main(["--manifests", str(directory), "--out", str(out)]) == 0
    capsys.readouterr()
    clean = tmp_path / "clean"
    clean_dir = tmp_path / "clean_manifests"
    for manifest in campaign:
        mf.write_manifest(manifest, clean_dir)
    assert tables_main(["--manifests", str(clean_dir), "--out", str(clean)]) == 0
    capsys.readouterr()
    for name in ("headline.md", "gates.md"):
        text = (out / name).read_text(encoding="utf-8")
        line = next((l for l in text.splitlines() if diverged["name"] in l), None)
        assert line is not None and "diverged" in line, name
        # Nothing else changes: the same file without that line is the clean campaign's file.
        assert diverged["name"] not in (clean / name).read_text(encoding="utf-8")


# --- B10, B36, B46: breaks of the review that no test caught --------------------------------------


@pytest.mark.parametrize("variant, seed", [("gqa4", 1), ("mqa", 3)])
def test_a_skippy_arm_b_run_is_refused(campaign, variant, seed):
    """B10: Arm B's runs are read by a comparison, so step parity covers them (plan §13.6)."""
    arm_b = fs.arm_b_manifests(campaign)
    analyse_committed(campaign + arm_b)  # accepted as built
    def is_target(p):
        return (p.arm, p.variant, p.init_seed) == ("isobackbone", variant, seed)
    assert sum(is_target(parsed(m)) for m in arm_b) == 1, sorted({parsed(m).arm for m in arm_b})
    skippy = replace(arm_b, is_target, lambda m: with_skips(m, 8))
    with pytest.raises(mt.TableError, match="steps taken"):
        analyse_committed(campaign + skippy)


def test_a_rerun_on_an_unregistered_seed_is_not_a_reproducibility_pair(campaign, analysis):
    """B36: plan §8.1 registers reruns on seeds 1 to 3 only."""
    registered = tuple(PREREG["test_selection"]["reproducibility_pairs"]["init_seeds"])
    extra = fake_manifest("mha", budget="100M", branches=[], init_seed=4, val_loss=9.9)
    a = analyse_committed(campaign + [extra])
    assert tuple(sorted(a.reruns_100m)) == registered
    assert a.selection == analysis.selection


def test_the_seed_count_threshold_is_exceeded_strictly(campaign, analysis):
    """B46: plan §1.4, "if 1 − m > 0.05": a fraction equal to the threshold stays at 5 seeds."""
    mha = {s: v for s, v in losses_of(campaign, variant="mha", tokens="300M", branch=None).items() if s in BASE_SEEDS}
    fraction = analysis.decision.undetectable_param_fraction
    assert 0.0 < fraction < 1.0

    def n_seeds(threshold: float) -> int:
        prereg = copy.deepcopy(PREREG)
        prereg["seed_count"]["max_undetectable_param_fraction"] = threshold
        return lf.seed_count_rule(mha, analysis.rule_fit, test=analysis.selection.test, prereg=prereg).n_seeds

    assert n_seeds(fraction) == 5
    assert n_seeds(math.nextafter(fraction, 1.0)) == 5
    assert n_seeds(math.nextafter(fraction, 0.0)) == 8
