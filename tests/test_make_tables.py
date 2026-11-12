"""T-27: the headline table and its refusal gates (plan §7.2, §10.1, §10.6, §13.6).

Every manifest here is SYNTHETIC (tests/fixtures/fake_manifests.py and
fake_campaign.py). The campaign is built once per module; tests that edit a
manifest work on a deep copy.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from analysis import ladder_fit as lf
from analysis import primary_test as pt
from scripts import make_tables as mt
from src import config
from src.utils import manifest as mf
from tests.fixtures.fake_campaign import (
    EXPLORATORY_VARIANTS,
    LADDER_PARAMS,
    PRIMARY_VARIANTS,
    fake_campaign,
)
from tests.fixtures.fake_manifests import fake_manifest
from tests.tolerances import STEPS_TAKEN_PARITY_REL

ROOT = Path(__file__).resolve().parents[1]
BLANK = mt.BLANK
ALL_VARIANTS = mt.HEADLINE_VARIANTS
STATIC = {
    "mha": ("17,308,032", "1024", "1,536"),
    "mqa": ("17,308,032", "1248", "192"),
    "mla192": ("17,311,104", "1048", "432"),
    "swa256": ("17,308,032", "1024", "1,536 (capped at 256 tokens)"),
}


def _valid(manifest: dict) -> dict:
    assert mf.validate_manifest(manifest) == []
    return manifest


def _with_steps(manifest: dict, *, taken: int) -> dict:
    """A completed run that took `taken` of its total steps (the rest are skipped)."""
    edited = copy.deepcopy(manifest)
    total = edited["steps"]["total"]
    edited["steps"].update(attempted=total, taken=taken, skipped=total - taken)
    edited["step_seconds"] = [0.5] * total
    return _valid(edited)


def _violation_lines(error: mt.TableError) -> list[str]:
    return [line for line in str(error).splitlines() if line.startswith("  - ")]


def _by_variant(rows: list[dict]) -> dict[str, dict]:
    return {row["variant"]: row for row in rows}


def _drop(manifests: list[dict], *, variant: str, init_seed: int, tokens: str = "300M", branch=None):
    def matches(m: dict) -> bool:
        parsed = mf.parse_run_name(m["name"])
        return (
            parsed.variant == variant
            and parsed.init_seed == init_seed
            and parsed.tokens == tokens
            and parsed.branch == branch
        )

    kept = [m for m in manifests if not matches(m)]
    assert len(kept) == len(manifests) - 1, f"no manifest for {variant} seed {init_seed}"
    return kept


# --- Module fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def campaign() -> list[dict]:
    # A complete campaign: the ladder and MHA also branch at 200M (plan §7.2, §12.1).
    return fake_campaign(gaps={"mqa": 0.05}, ladder_branches=True)


@pytest.fixture(scope="module")
def analysis(campaign) -> mt.Analysis:
    return mt.analyse(campaign)


@pytest.fixture(scope="module")
def rows(analysis) -> list[dict]:
    return mt.headline_rows(analysis)


@pytest.fixture(scope="module")
def eight_seed_analysis() -> mt.Analysis:
    return mt.analyse(fake_campaign(sigma=0.01, n_primary_seeds=8, gaps={"mqa": 0.05}))


# --- Refusal gates: check_comparable ----------------------------------------------


def test_step_parity_constant_matches_the_tolerance_file():
    assert mt.STEPS_TAKEN_PARITY == STEPS_TAKEN_PARITY_REL


@pytest.mark.parametrize(
    "taken, refused",
    [(4578 - 22, False), (4578 - 23, True)],
    ids=["22-steps-passes", "23-steps-refused"],
)
def test_steps_taken_limit_at_the_300m_budget(taken, refused):
    full = _valid(fake_manifest("mha", init_seed=1))
    assert full["steps"]["total"] == 4578
    fewer = _with_steps(fake_manifest("gqa4", init_seed=1), taken=taken)
    if not refused:
        mt.check_comparable([full, fewer])
        return
    with pytest.raises(mt.TableError) as caught:
        mt.check_comparable([full, fewer])
    message = str(caught.value)
    assert "steps taken" in message and "4578" in message
    assert fewer["name"] in message and full["name"] in message


@pytest.mark.parametrize(
    "taken, refused",
    [(1526 - 7, False), (1526 - 8, True)],
    ids=["7-steps-passes", "8-steps-refused"],
)
def test_steps_taken_limit_at_the_100m_budget(taken, refused):
    full = _valid(fake_manifest("mha", budget="100M", branches=[], init_seed=1))
    assert full["steps"]["total"] == 1526
    fewer = _with_steps(fake_manifest("mha", budget="100M", branches=[], init_seed=2), taken=taken)
    if not refused:
        mt.check_comparable([full, fewer])
        return
    with pytest.raises(mt.TableError) as caught:
        mt.check_comparable([full, fewer])
    assert "steps taken" in str(caught.value) and "1526" in str(caught.value)


def test_runs_with_different_totals_are_not_compared_by_the_step_rule():
    long_run = _valid(fake_manifest("mha", init_seed=1))
    short_run = _with_steps(
        fake_manifest("mha", budget="100M", branches=[], init_seed=2), taken=1000
    )
    mt.check_comparable([long_run, short_run])


MISMATCHES = [
    ("backend", lambda m: m.update(backend="math"), "backend"),
    ("torch version", lambda m: m["environment"].update(torch="2.16.0"), "torch version"),
    ("CUDA version", lambda m: m["environment"].update(cuda="12.8"), "CUDA version"),
    ("driver version", lambda m: m["environment"].update(driver="998.0"), "driver version"),
    ("dataset hash", lambda m: m["data"].update(dataset_sha256="d" * 64), "dataset hash"),
    ("tokenizer hash", lambda m: m["data"].update(tokenizer_sha256="e" * 64), "tokenizer hash"),
    ("GPU name", lambda m: m["environment"].update(gpu_name="Other GPU"), "GPU name"),
    ("mode", lambda m: m.update(mode="ref"), "mode differs"),
    ("precision", lambda m: m.update(precision="fp32"), "precision differs"),
]


@pytest.mark.parametrize("label, edit, needle", MISMATCHES, ids=[m[0] for m in MISMATCHES])
def test_one_mismatch_is_refused_and_named(label, edit, needle):
    base = _valid(fake_manifest("mha", init_seed=1))
    other = copy.deepcopy(_valid(fake_manifest("gqa4", init_seed=1)))
    edit(other)
    other = _valid(other)
    mt.check_comparable([base, _valid(fake_manifest("gqa4", init_seed=1))])  # the valid pair passes
    with pytest.raises(mt.TableError) as caught:
        mt.check_comparable([base, other])
    lines = _violation_lines(caught.value)
    assert len(lines) == 1, lines
    assert needle in lines[0]
    assert other["name"] in lines[0]


def test_several_violations_are_all_listed():
    a = _valid(fake_manifest("mha", init_seed=1))
    b = copy.deepcopy(_valid(fake_manifest("gqa4", init_seed=1)))
    b.update(backend="math")
    b["environment"].update(torch="2.16.0")
    c = _with_steps(fake_manifest("mqa", init_seed=1), taken=4578 - 23)
    with pytest.raises(mt.TableError) as caught:
        mt.check_comparable([a, b, c])
    lines = _violation_lines(caught.value)
    assert len(lines) == 3, lines
    assert any("backend" in line for line in lines)
    assert any("torch version" in line for line in lines)
    assert any("steps taken" in line for line in lines)


def test_an_empty_list_passes():
    mt.check_comparable([])


# --- Selecting results ------------------------------------------------------------


def test_trunk_checkpoint_is_refused_by_select_results_and_check_comparable():
    trunk = fake_manifest("mha", kind="trunk_checkpoint", completed=False)
    with pytest.raises(mt.TableError, match="trunk checkpoint"):
        mt.select_results([trunk])
    with pytest.raises(mt.TableError, match="trunk checkpoint"):
        mt.select_results([trunk], skip_unfinished=True)
    with pytest.raises(mt.TableError):
        mt.check_comparable([trunk])


def test_unfinished_run_is_refused_then_skipped_by_name():
    done = _valid(fake_manifest("mha", init_seed=1))
    unfinished = _valid(fake_manifest("gqa4", init_seed=1, completed=False))
    with pytest.raises(mt.TableError) as caught:
        mt.select_results([done, unfinished])
    assert unfinished["name"] in str(caught.value)
    results, skipped = mt.select_results([done, unfinished], skip_unfinished=True)
    assert results == [done]
    assert skipped == [unfinished["name"]]


# --- Loading --------------------------------------------------------------------------


def test_load_manifests_reads_in_file_name_order(tmp_path):
    written = [
        fake_manifest("mha", init_seed=2),
        fake_manifest("gqa4", init_seed=1),
        fake_manifest("mha", init_seed=1),
    ]
    for manifest in written:
        mf.write_manifest(manifest, tmp_path)
    loaded = mt.load_manifests(tmp_path)
    assert [m["name"] for m in loaded] == sorted(m["name"] for m in written)
    assert loaded == sorted(written, key=lambda m: mf.manifest_filename(m))


def test_load_manifests_of_a_missing_directory_is_empty(tmp_path):
    assert mt.load_manifests(tmp_path / "does-not-exist") == []


def test_load_manifests_raises_on_an_invalid_file(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(mf.ManifestError):
        mt.load_manifests(tmp_path)


# --- The headline table from fake manifests -------------------------------------------


def test_table_has_one_row_per_headline_variant_in_order(rows):
    assert [row["variant"] for row in rows] == list(ALL_VARIANTS)
    for row in rows:
        assert tuple(row) == mt.COLUMNS
        assert all(isinstance(value, str) for value in row.values())


def test_verdict_cells_come_from_the_comparisons(rows, analysis):
    by = _by_variant(rows)
    mqa = analysis.comparisons["mqa"]
    assert mqa.verdict == pt.VERDICT_DETECTED
    assert by["mqa"]["verdict"].startswith(pt.VERDICT_DETECTED)
    assert by["gqa4"]["verdict"].startswith(pt.VERDICT_NOT_DETECTED)
    assert "ruled out" in by["gqa4"]["verdict"]
    assert by["mla192"]["verdict"].startswith(pt.VERDICT_NOT_DETECTED)
    for variant in EXPLORATORY_VARIANTS:
        assert by[variant]["verdict"] == pt.VERDICT_EXPLORATORY
    assert by["mha"]["verdict"] == "reference"


def test_numeric_cells_match_the_modules_called_directly(campaign, rows):
    """Re-derive every number from the manifests with primary_test and ladder_fit."""
    by = _by_variant(rows)
    prereg = config.load_prereg()

    def losses_of(variant: str, *, tokens="300M", branch=None) -> dict[int, float]:
        out = {}
        for manifest in campaign:
            parsed = mf.parse_run_name(manifest["name"])
            if parsed.variant == variant and parsed.tokens == tokens and parsed.branch == branch:
                out[parsed.init_seed] = float(manifest["final_metrics"]["val_loss"])
        return out

    mha_100 = losses_of("mha", tokens="300M", branch="100M")
    reruns = losses_of("mha", tokens="100M")
    selection = pt.select_test(
        {s: mha_100[s] for s in range(1, 6)},
        {s: (mha_100[s], reruns[s]) for s in range(1, 4)},
        prereg,
    )
    mha_300 = losses_of("mha")
    points = {
        "mha": (LADDER_PARAMS["mha"], mha_300),
        "mha_L6": (LADDER_PARAMS["mha_L6"], losses_of("mha_L6")),
        "mha_L10": (LADDER_PARAMS["mha_L10"], losses_of("mha_L10")),
    }
    fit = lf.fit_ladder(points)
    decision = lf.seed_count_rule(
        {s: mha_300[s] for s in range(1, 6)}, fit, test=selection.test, prereg=prereg
    )
    names = prereg["primary"]["comparisons"]
    comparisons = pt.primary_tests(
        {v: losses_of(v) for v in ["mha", *names]},
        test=selection.test,
        n_seeds=decision.n_seeds,
        prereg=prereg,
    )
    for comparison in comparisons:
        row = by[comparison.name]
        gap = f"{comparison.mean_gap:+.4f} [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]"
        assert row["gap vs MHA [95% CI]"] == gap
        multiplier = lf.comparison_multiplier(comparison, fit)
        assert row["effective-param multiplier"] == (
            f"{multiplier.multiplier:.3f}× [{multiplier.ci_low:.3f}, {multiplier.ci_high:.3f}]"
        )
        assert row["verdict"].startswith(comparison.verdict)
    for variant in EXPLORATORY_VARIANTS:
        comparison = pt.exploratory(
            variant,
            losses_of(variant),
            {s: mha_300[s] for s in decision.seeds},
            test=selection.test,
            prereg=prereg,
        )
        assert by[variant]["verdict"] == comparison.verdict
        assert by[variant]["gap vs MHA [95% CI]"] == (
            f"{comparison.mean_gap:+.4f} [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]"
        )
    for variant in ALL_VARIANTS:
        if variant == "mha":
            continue
        expected = ", ".join(f"s{s} {loss:.4f}" for s, loss in sorted(losses_of(variant).items()))
        assert by[variant]["val loss, every seed"] == expected
    assert by["mha"]["val loss, every seed"] == ", ".join(
        f"s{s} {loss:.4f}" for s, loss in sorted(mha_300.items())
    )


def test_analysis_objects_for_a_small_noise_campaign(analysis):
    assert analysis.selection.test == pt.PAIRED
    assert analysis.decision.n_seeds == 5
    assert analysis.decision.seeds == (1, 2, 3, 4, 5)
    assert analysis.notes == ()
    for point in analysis.fit.points:
        assert point.n_params == LADDER_PARAMS[point.name]


def test_eight_seed_rule_extends_the_primaries_and_leaves_arm_a(eight_seed_analysis):
    a = eight_seed_analysis
    assert a.decision.n_seeds == 8
    assert a.decision.seeds == (1, 2, 3, 4, 5, 6, 7, 8)
    for name in PRIMARY_VARIANTS:
        assert a.comparisons[name].seeds == tuple(range(1, 9))
    assert a.comparisons["mqa"].verdict == pt.VERDICT_DETECTED
    rows = _by_variant(mt.headline_rows(a))
    for variant in ("mla96", "swa256"):
        assert rows[variant]["verdict"] == "left Arm A (seed-count rule)"
        assert rows[variant]["gap vs MHA [95% CI]"] == BLANK
        assert rows[variant]["effective-param multiplier"] == BLANK
        assert variant not in a.comparisons
    assert rows["gqa2"]["verdict"] == pt.VERDICT_EXPLORATORY
    assert any("mla96" in note and "left Arm A" in note for note in a.notes)


# --- Partial data never guesses -------------------------------------------------------


def _verdict_cells(rows: list[dict]) -> list[str]:
    return [row["verdict"] for row in rows]


def test_without_reruns_there_is_no_verdict_anywhere():
    a = mt.analyse(fake_campaign(gaps={"mqa": 0.05}, without=("reruns",)))
    assert a.selection is None and a.decision is None and a.comparisons == {}
    rows = mt.headline_rows(a)
    assert set(_verdict_cells(rows)) <= {BLANK, "reference"}
    assert all(row["gap vs MHA [95% CI]"] in (BLANK, "reference") for row in rows)
    assert any("reruns" in note and "seeds [1, 2, 3]" in note for note in a.notes)


def test_with_one_primary_missing_a_seed_none_of_the_three_has_a_verdict(campaign):
    partial = _drop(campaign, variant="mqa", init_seed=5)
    a = mt.analyse(partial)
    assert a.decision is not None and a.decision.n_seeds == 5
    assert not {"gqa4", "mqa", "mla192"} & set(a.comparisons)
    rows = _by_variant(mt.headline_rows(a))
    for name in PRIMARY_VARIANTS:
        assert rows[name]["verdict"] == BLANK
        assert rows[name]["gap vs MHA [95% CI]"] == BLANK
    assert any("mqa seeds [5]" in note for note in a.notes)
    for variant in EXPLORATORY_VARIANTS:
        assert rows[variant]["verdict"] == pt.VERDICT_EXPLORATORY
    assert set(a.multipliers) == set(EXPLORATORY_VARIANTS)


def test_without_the_ladder_no_seed_count_and_no_verdict_is_given():
    """Part B needs the ladder fit for the seed-count rule, so the table gives no verdict."""
    a = mt.analyse(fake_campaign(gaps={"mqa": 0.05}, without=("ladder",)))
    assert a.fit is None and a.decision is None
    assert a.comparisons == {} and a.multipliers == {}
    assert set(_verdict_cells(mt.headline_rows(a))) <= {BLANK, "reference"}
    assert any("mha_L6" in note for note in a.notes)
    assert any("seed count rule skipped: no ladder fit" == note for note in a.notes)


def test_with_only_mha_l6_missing_the_note_names_it():
    """The ladder part is mha_L6 alone; mha_L10 is still there and does not rescue the fit."""
    a = mt.analyse(fake_campaign(gaps={"mqa": 0.05}, without=("ladder",)))
    assert "ladder fit: mha_L6 300M results are missing" in a.notes


def test_without_l10_the_fit_uses_the_two_other_points():
    a = mt.analyse(fake_campaign(without=("ladder_l10",)))
    assert a.fit is not None
    assert set(p.name for p in a.fit.points) == {"mha", "mha_L6"}
    assert a.decision.n_seeds == 5


# --- Learning rates, ignored runs and refusals -----------------------------------------


def test_mixed_learning_rates_of_one_variant_raise(campaign):
    mixed = campaign + [fake_manifest("mha", init_seed=9, peak_lr=3e-3)]
    with pytest.raises(mt.TableError) as caught:
        mt.analyse(mixed)
    message = str(caught.value)
    assert "mha" in message and "more than one peak learning rate" in message
    assert "3e3" in message and "1e3" in message


def test_a_100m_sweep_point_at_another_learning_rate_is_ignored(campaign, analysis):
    # A sweep point is a standalone 100M run (plan §7.4). A branch at another learning rate
    # would mean a 300M run at that rate, which is refused (see the test added in review).
    extra = [
        fake_manifest("mha", budget="100M", branches=[], init_seed=1, peak_lr=3e-3, val_loss=9.0),
        fake_manifest("gqa4", budget="100M", branches=[], init_seed=1, peak_lr=5e-4, val_loss=9.0),
    ]
    a = mt.analyse(campaign + extra)
    assert a.reruns_100m == analysis.reruns_100m
    assert a.losses_100m == analysis.losses_100m
    assert a.losses == analysis.losses


def test_null_other_data_seed_and_other_arm_runs_are_ignored(campaign, analysis):
    extra = [
        fake_manifest("null", init_seed=1, val_loss=9.0),
        fake_manifest("gqa4", init_seed=1, data_seed=2, val_loss=9.0),
        fake_manifest("gqa4", arm="iso_backbone", init_seed=1, val_loss=9.0),
    ]
    a = mt.analyse(campaign + extra)
    assert a.losses == analysis.losses
    assert "null" not in a.losses
    assert a.comparisons.keys() == analysis.comparisons.keys()


def test_analyse_refuses_incomparable_runs(campaign):
    bad = copy.deepcopy(fake_manifest("gqa2", init_seed=1, val_loss=3.0))
    bad["backend"] = "math"
    with pytest.raises(mt.TableError, match="backend"):
        mt.analyse(campaign + [bad])


# --- Timing columns and rendering ------------------------------------------------------


def test_timing_mappings_fill_their_columns_with_one_decimal(analysis):
    rows = _by_variant(
        mt.headline_rows(
            analysis,
            train_ms_per_step={"mha": 12.34, "gqa4": 56.78},
            decode_tok_per_s={"mha": 987.0},
        )
    )
    assert rows["mha"]["train ms/step"] == "12.3"
    assert rows["gqa4"]["train ms/step"] == "56.8"
    assert rows["mqa"]["train ms/step"] == BLANK
    assert rows["mha"]["decode tok/s (batch 32, cache 4k)"] == "987.0"
    assert rows["gqa4"]["decode tok/s (batch 32, cache 4k)"] == BLANK


def test_render_markdown_layout_and_notes(rows):
    text = mt.render_markdown(rows, ["first note", "second note"])
    lines = text.splitlines()
    assert lines[0] == "| " + " | ".join(mt.COLUMNS) + " |"
    assert lines[1].startswith("|---")
    table_lines = [line for line in lines if line.startswith("| ")]
    assert len(table_lines) == 1 + len(rows)
    assert "- first note" in text and "- second note" in text
    assert text.endswith("\n")
    assert "Notes:" in text
    assert "Notes:" not in mt.render_markdown(rows, ())


# --- The command line ----------------------------------------------------------------------


def test_main_writes_headline_md_and_returns_zero(tmp_path, campaign, analysis, capsys):
    manifests = tmp_path / "manifests"
    for manifest in campaign:
        mf.write_manifest(manifest, manifests)
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(manifests), "--out", str(out)]) == 0
    written = (out / "headline.md").read_text(encoding="utf-8")
    assert written == mt.render_markdown(mt.headline_rows(analysis), analysis.notes)
    assert str(out / "headline.md") in capsys.readouterr().out


def test_main_refusal_returns_one_and_writes_nothing(tmp_path, campaign, capsys):
    manifests = tmp_path / "manifests"
    for manifest in campaign[:10]:
        mf.write_manifest(manifest, manifests)
    bad = copy.deepcopy(fake_manifest("gqa2", init_seed=1, val_loss=3.0))
    bad["data"]["dataset_sha256"] = "d" * 64
    mf.write_manifest(bad, manifests)
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(manifests), "--out", str(out)]) == 1
    assert "dataset hash" in capsys.readouterr().err
    assert not (out / "headline.md").exists()


def test_main_leaves_unfinished_runs_out_only_when_asked(tmp_path, campaign):
    manifests = tmp_path / "manifests"
    for manifest in campaign[:10]:
        mf.write_manifest(manifest, manifests)
    unfinished = fake_manifest("gqa2", init_seed=9, completed=False)
    mf.write_manifest(unfinished, manifests)
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(manifests), "--out", str(out)]) == 1
    assert mt.main(["--manifests", str(manifests), "--out", str(out), "--skip-unfinished"]) == 0
    text = (out / "headline.md").read_text(encoding="utf-8")
    assert unfinished["name"] in text


# --- The blank table (no manifests) ----------------------------------------------------------


def test_blank_rows_have_the_static_columns_and_nothing_else():
    rows = _by_variant(mt.headline_rows(mt.analyse([])))
    for variant in ALL_VARIANTS:
        row = rows[variant]
        assert row["params"] != BLANK and row["d_ff"] != BLANK
        assert row["KV bytes/token/layer"] != BLANK
        for column in mt.COLUMNS[4:]:
            assert row[column] == BLANK, (variant, column)
    for variant, (params, d_ff, kv) in STATIC.items():
        assert rows[variant]["params"] == params
        assert rows[variant]["d_ff"] == d_ff
        assert rows[variant]["KV bytes/token/layer"] == kv


def test_blank_analysis_has_notes_for_every_missing_input():
    notes = mt.analyse([]).notes
    assert any("MHA 100M branches" in note for note in notes)
    assert any("reruns" in note for note in notes)
    assert any("ladder fit: MHA 300M results are missing" == note for note in notes)


def test_committed_blank_table_is_generated_from_no_manifests():
    committed = (ROOT / "results" / "tables" / "blank" / "headline.md").read_bytes()
    generated = mt.render_markdown(mt.headline_rows(mt.analyse([])), mt.analyse([]).notes)
    assert committed == generated.encode("utf-8")


def test_main_blank_matches_the_committed_file(tmp_path):
    assert mt.main(["--blank", "--out", str(tmp_path)]) == 0
    committed = (ROOT / "results" / "tables" / "blank" / "headline.md").read_bytes()
    assert (tmp_path / "headline.md").read_bytes() == committed


def test_the_test_is_selected_before_any_300m_result_exists(campaign, analysis):
    """Added in review. At Gate 2 the 300M runs are still training: only their 100M branches
    and MHA's reruns are results. The selection must already work then, and nothing that
    needs a 300M result may be computed."""
    early = [m for m in campaign if mf.parse_run_name(m["name"]).branch == "100M"
             or mf.parse_run_name(m["name"]).tokens == "100M"]
    assert early and all(m["steps"]["total"] == 1526 for m in early)
    result = mt.analyse(early)
    assert result.selection is not None
    assert result.selection == analysis.selection
    assert result.fit is None and result.decision is None
    assert result.comparisons == {} and result.multipliers == {}
    assert all(cell == mt.BLANK for cell in _verdict_cells(mt.headline_rows(result)))


def test_a_branch_at_another_learning_rate_is_refused(campaign):
    """Added in review: a variant's branches and results share one learning rate."""
    other = fake_manifest("mha", init_seed=1, branch="100M", peak_lr=2e-3, val_loss=3.6)
    with pytest.raises(mt.TableError, match="more than one peak learning rate"):
        mt.analyse(list(campaign) + [other])


def test_seeds_outside_the_registered_sets_are_shown_but_never_tested(campaign, analysis, rows):
    """Added in review. An extra run must not change a comparison: the registered seeds are
    the test (plan §11, "no subset of seeds", and none added either)."""
    extra = [
        fake_manifest("gqa4", init_seed=6, val_loss=1.0),  # a primary, far from the others
        fake_manifest("gqa2", init_seed=4, val_loss=1.0),  # an exploratory variant
    ]
    with_extra = mt.analyse(list(campaign) + extra)
    assert with_extra.comparisons == analysis.comparisons
    assert with_extra.multipliers == analysis.multipliers
    assert with_extra.decision == analysis.decision
    new_rows, old_rows = _by_variant(mt.headline_rows(with_extra)), _by_variant(rows)
    for name in ("gqa4", "gqa2"):
        assert new_rows[name]["verdict"] == old_rows[name]["verdict"]
        assert new_rows[name]["gap vs MHA [95% CI]"] == old_rows[name]["gap vs MHA [95% CI]"]
    assert "s6 1.0000" in new_rows["gqa4"]["val loss, every seed"]  # every seed is shown
    assert "s4 1.0000" in new_rows["gqa2"]["val loss, every seed"]
    assert any("gqa4" in n and "[6]" in n for n in with_extra.notes)
    assert any("gqa2" in n and "[4]" in n for n in with_extra.notes)


# --- F8 (T-37): the disclosures of plan §13.1 to §13.5 end every headline table ----------------

DISCLOSURE_SECTIONS = ("§13.2", "§13.3", "§13.4", "§13.5", "§13.1")


def test_each_disclosure_is_one_line_naming_its_plan_section_in_the_fixed_order():
    assert len(mt.DISCLOSURES) == len(DISCLOSURE_SECTIONS)
    found = [line[line.rindex("(plan ") + len("(plan "):].rstrip(".)") for line in mt.DISCLOSURES]
    assert found == list(DISCLOSURE_SECTIONS)


def test_the_disclosures_end_the_headline_after_the_notes_for_blank_full_and_gate_shut_tables(analysis):
    cases = {
        "blank": mt.analyse([]),
        "full campaign": analysis,
        "gate shut": mt.analyse(fake_campaign(gaps={"mqa": 0.05}, without=("null",))),
    }
    assert cases["blank"].notes and cases["gate shut"].notes  # so that "after the notes" is checked
    for label, a in cases.items():
        text = mt.render_markdown(mt.headline_rows(a), a.notes)
        positions = [text.index(f"- {line}\n") for line in mt.DISCLOSURES]
        assert positions == sorted(positions), label
        assert text.count("Disclosures:") == 1, label
        assert text.endswith(f"- {mt.DISCLOSURES[-1]}\n"), label
        if a.notes:
            assert text.index("\nNotes:\n") < positions[0], label
