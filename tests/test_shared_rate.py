"""T-50: tuned against shared learning rate (decision 224, plan §7.4 item 5), in scripts/make_tables.py.

Every manifest here is SYNTHETIC (tests/fixtures/fake_secondary.py, fake_campaign.py, fake_manifests.py):
every loss and every rate is made up. The statistics are never recomputed here except to call
`primary_test.secondary` on the values the analysis read, which the comparison must equal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis import primary_test as pt
from scripts import make_tables as mt
from src.utils import manifest as mf
from tests.fixtures.commitments import analyse_committed, tables_main
from tests.fixtures.fake_manifests import fake_manifest
from tests.fixtures.fake_secondary import find, secondary_campaign
from tests.test_lr_sweep import with_skips
from tests.tolerances import FLOAT64_IDENTITY

ROOT = Path(__file__).resolve().parents[1]
BLANK_FILE = ROOT / "results" / "tables" / "blank" / "shared_rate.md"
MHA_RATE = 1e-3  # the fake manifests' default rate, which MHA runs at
OWN_RATE = 2e-3  # a made-up rate that the moved variant runs at
THIRD_RATE = 4e-3
SEEDS = mt.SHARED_RATE_SEEDS
SHIFT = 0.01  # shared-rate loss minus tuned loss, nats, for the synthetic tests (negative gap)
SPREAD = 0.002  # extra per-seed spread, so the interval is not degenerate
VARIANT = "mqa"  # a primary variant: moved to OWN_RATE in most tests
HEADLINE_NON_MHA = tuple(v for v in mt.HEADLINE_VARIANTS if v != "mha")


# --- Helpers ----------------------------------------------------------------------------------------


def at_rate(manifests, variant: str, peak_lr: float) -> list[dict]:
    """The variant's 300M runs and 100M branches rebuilt at `peak_lr`, with the same seeds and losses."""
    out = []
    for manifest in manifests:
        parsed = mf.parse_run_name(manifest["name"])
        if parsed.variant != variant:
            out.append(manifest)
            continue
        extra = {"branch": "100M"} if parsed.branch else {}
        out.append(
            fake_manifest(
                variant, init_seed=parsed.init_seed, peak_lr=peak_lr,
                val_loss=float(manifest["final_metrics"]["val_loss"]), **extra,
            )
        )
    return out


def branch_loss(manifests, variant: str, seed: int, peak_lr: float) -> float:
    run = find(manifests, variant=variant, arm="isoparam", init_seed=seed, tokens="300M", branch="100M", peak_lr=peak_lr)
    return float(run["final_metrics"]["val_loss"])


def standalone(variant: str, seed: int, peak_lr: float, loss: float, **kwargs) -> dict:
    """A standalone 100M run (budget 100M, no branches) of the arm, as the shared-rate runs are."""
    return fake_manifest(variant, budget="100M", branches=[], init_seed=seed, peak_lr=peak_lr, val_loss=loss, **kwargs)


def shared_runs(manifests, variant: str, *, shift: float = SHIFT, seeds=SEEDS) -> list[dict]:
    """Standalone 100M runs of `variant` at MHA's rate: its tuned branch loss plus `shift` (plus a spread)."""
    runs = []
    for seed in seeds:
        loss = branch_loss(manifests, variant, seed, OWN_RATE) + shift + SPREAD * (seed - 2)
        runs.append(standalone(variant, seed, MHA_RATE, loss))
    return runs


def with_variant_moved(variant: str = VARIANT, *, base=None, rerun_sigma=None):
    """A campaign in which `variant` runs at OWN_RATE, MHA at MHA_RATE, with its shared-rate runs."""
    base = secondary_campaign(gaps={variant: 0.05}, rerun_sigma=rerun_sigma) if base is None else base
    moved = at_rate(base, variant, OWN_RATE)
    return moved + shared_runs(moved, variant)


def row_of(text: str, variant: str) -> list[str]:
    """The cells of the table row of `variant`, stripped."""
    line = next(l for l in text.splitlines() if l.startswith(f"| {variant} |"))
    return [cell.strip() for cell in line.strip("|").split("|")]


def table(analysis) -> str:
    return mt.render_shared_rate(analysis)


@pytest.fixture(scope="module")
def no_variant_differs():
    return analyse_committed(secondary_campaign())


@pytest.fixture(scope="module")
def moved():
    manifests = with_variant_moved()
    return manifests, analyse_committed(manifests)


@pytest.fixture(scope="module")
def moved_welch():
    manifests = with_variant_moved(rerun_sigma=0.02)
    return manifests, analyse_committed(manifests)


# --- The fixtures take the tests they are meant for -------------------------------------------------


def test_fixtures_take_the_tests_they_are_meant_for(moved, moved_welch):
    assert moved[1].selection.test == pt.PAIRED
    assert moved_welch[1].selection.test == pt.WELCH
    assert moved[1].comparisons_allowed and moved_welch[1].comparisons_allowed
    assert moved[1].peak_lr[VARIANT] == OWN_RATE and moved[1].peak_lr["mha"] == MHA_RATE


# --- No variant differs ------------------------------------------------------------------------------


def test_no_variant_differs_gives_no_shared_rate_results(no_variant_differs):
    assert no_variant_differs.shared_rate == {}
    assert not any(note.startswith("shared-rate runs:") for note in no_variant_differs.notes)


def test_no_variant_differs_every_row_says_same_rate(no_variant_differs):
    text = table(no_variant_differs)
    for variant in HEADLINE_NON_MHA:
        cells = row_of(text, variant)
        assert cells[1] == cells[2] == "1e3", variant
        assert cells[3:] == [mt.SAME_RATE] * 4, variant


# --- One variant on its own rate ---------------------------------------------------------------------


def test_result_fields_of_the_moved_variant(moved):
    analysis = moved[1]
    assert set(analysis.shared_rate) == {VARIANT}
    result = analysis.shared_rate[VARIANT]
    assert isinstance(result, mt.SharedRateResult)
    assert result.variant == VARIANT
    assert result.peak_lr == OWN_RATE
    assert result.shared_peak_lr == MHA_RATE
    assert set(result.tuned_losses) == set(SEEDS) == set(result.shared_losses)
    for seed in SEEDS:
        assert result.tuned_losses[seed] == branch_loss(moved[0], VARIANT, seed, OWN_RATE)
        shared = find(moved[0], variant=VARIANT, arm="isoparam", init_seed=seed, tokens="100M", peak_lr=MHA_RATE)
        assert result.shared_losses[seed] == float(shared["final_metrics"]["val_loss"])
    assert result.tuned_minus_shared is not None


@pytest.mark.parametrize("which", ["moved", "moved_welch"])
def test_the_comparison_is_pt_secondary_on_the_same_numbers(which, request):
    manifests, analysis = request.getfixturevalue(which)
    result = analysis.shared_rate[VARIANT]
    direct = pt.secondary(
        VARIANT,
        {s: branch_loss(manifests, VARIANT, s, OWN_RATE) for s in SEEDS},
        dict(result.shared_losses),
        reference=f"{VARIANT} at MHA's rate {mf.format_lr(MHA_RATE)}",
        seeds=list(SEEDS),
        test=analysis.selection.test,
    )
    assert result.tuned_minus_shared == direct
    assert result.tuned_minus_shared.verdict == pt.VERDICT_EXPLORATORY
    assert result.tuned_minus_shared.p is None
    assert analysis.selection.test == (pt.WELCH if which == "moved_welch" else pt.PAIRED)


def test_tuned_better_than_shared_gives_a_negative_gap(moved):
    # The shared-rate losses are the tuned ones plus SHIFT, and the spread averages to zero over seeds 1-3.
    gap = moved[1].shared_rate[VARIANT].tuned_minus_shared
    assert gap.mean_gap == pytest.approx(-SHIFT, abs=FLOAT64_IDENTITY)
    assert gap.mean_gap < 0


def test_tuned_worse_than_shared_gives_a_positive_gap():
    base = at_rate(secondary_campaign(gaps={VARIANT: 0.05}), VARIANT, OWN_RATE)
    manifests = base + shared_runs(base, VARIANT, shift=-SHIFT)
    gap = analyse_committed(manifests).shared_rate[VARIANT].tuned_minus_shared
    assert gap.mean_gap == pytest.approx(SHIFT, abs=FLOAT64_IDENTITY)
    assert gap.mean_gap > 0


# --- Runs that must be ignored -----------------------------------------------------------------------


def test_ignored_runs_change_nothing(moved):
    manifests, analysis = moved
    extra = [
        standalone(VARIANT, 21, MHA_RATE, 3.3),  # a shared-rate run on seed 21: the sweep seed
        standalone(VARIANT, 4, MHA_RATE, 3.3),  # a shared-rate run on seed 4: not a registered seed
        standalone(VARIANT, 1, THIRD_RATE, 3.4),  # a standalone 100M run at a third rate
        standalone("gqa4", 1, MHA_RATE, 3.3),  # gqa4 runs at MHA's rate: no shared-rate runs for it
        standalone("gqa4", 2, MHA_RATE, 3.3),
        fake_manifest("mha", ladder="mha_L6", budget="100M", branches=[], init_seed=1, peak_lr=THIRD_RATE, val_loss=3.4),
    ]
    with_extra = analyse_committed(manifests + extra)
    assert set(with_extra.shared_rate) == {VARIANT}
    result, base = with_extra.shared_rate[VARIANT], analysis.shared_rate[VARIANT]
    assert result.shared_losses == base.shared_losses
    assert result.tuned_losses == base.tuned_losses
    assert result.tuned_minus_shared == base.tuned_minus_shared
    assert with_extra.comparisons == analysis.comparisons
    assert with_extra.multipliers == analysis.multipliers
    assert with_extra.notes == analysis.notes


def test_a_shared_rate_run_of_a_variant_on_mhas_rate_is_ignored(moved):
    manifests, _ = moved
    extra = [standalone("gqa4", s, MHA_RATE, 3.3 + s / 100) for s in SEEDS]
    analysis = analyse_committed(manifests + extra)
    assert "gqa4" not in analysis.shared_rate
    assert row_of(table(analysis), "gqa4")[3:] == [mt.SAME_RATE] * 4


def test_a_100m_branch_at_mhas_rate_is_not_a_shared_rate_run(moved):
    # A branch of a variant at MHA's rate is a second rate of that variant in the 300M sense: the
    # existing refusal of mixed rates applies to it (the rate is taken from runs at the primary budget).
    manifests, _ = moved
    branch = fake_manifest(VARIANT, init_seed=1, branch="100M", peak_lr=MHA_RATE, val_loss=3.3)
    with pytest.raises(mt.TableError, match="more than one peak learning rate"):
        analyse_committed(manifests + [branch])


# --- Missing runs -----------------------------------------------------------------------------------


def test_missing_shared_rate_seeds_give_no_comparison_and_a_note():
    manifests = at_rate(secondary_campaign(gaps={VARIANT: 0.05}), VARIANT, OWN_RATE)
    manifests += shared_runs(manifests, VARIANT, seeds=(1, 2))
    analysis = analyse_committed(manifests)
    result = analysis.shared_rate[VARIANT]
    assert result.tuned_minus_shared is None
    assert set(result.shared_losses) == {1, 2}
    note = f"shared-rate runs: {VARIANT} 100M runs at MHA's rate {mf.format_lr(MHA_RATE)} for seeds [3] are missing (decision 224)"
    assert note in analysis.notes
    assert note in table(analysis)


def test_missing_tuned_branches_give_no_comparison_and_a_note():
    base = at_rate(secondary_campaign(gaps={VARIANT: 0.05}), VARIANT, OWN_RATE)
    shared = shared_runs(base, VARIANT)  # read before seed 2's branch is dropped
    manifests = [m for m in base if m["name"] != f"{VARIANT}-isoparam-300M-lr2e3-s2-d1-b100M"] + shared
    analysis = analyse_committed(manifests)
    result = analysis.shared_rate[VARIANT]
    assert result.tuned_minus_shared is None
    assert set(result.tuned_losses) == {1, 3}
    assert any(
        note.startswith(f"shared-rate runs: {VARIANT} 100M branches at its own rate {mf.format_lr(OWN_RATE)} for seeds [2]")
        for note in analysis.notes
    )


# --- The gate ----------------------------------------------------------------------------------------


def test_with_the_gate_shut_no_comparison_and_no_loss_is_shown():
    manifests = at_rate(secondary_campaign(without=("null",), gaps={VARIANT: 0.05}), VARIANT, OWN_RATE)
    manifests += shared_runs(manifests, VARIANT)
    analysis = analyse_committed(manifests)
    assert analysis.comparisons_allowed is False
    assert analysis.shared_rate[VARIANT].tuned_minus_shared is None
    row = row_of(table(analysis), VARIANT)
    assert row[1:3] == ["2e3", "1e3"]
    assert row[3:5] == [mt.BLANK, mt.BLANK]
    assert row[5:] == [mt.BLANK, mt.BLANK]
    assert mt.GATE_WAIT in table(analysis)


def test_with_the_gate_shut_the_missing_run_notes_are_still_there():
    manifests = at_rate(secondary_campaign(without=("null",), gaps={VARIANT: 0.05}), VARIANT, OWN_RATE)
    analysis = analyse_committed(manifests)
    assert analysis.comparisons_allowed is False
    assert any(note.startswith(f"shared-rate runs: {VARIANT} 100M runs at MHA's rate") for note in analysis.notes)


def test_the_comparison_does_not_wait_for_the_seed_count_commitment(moved):
    analysis = mt.analyse(moved[0], commitments={"ladder": "full", "seed_count": None})
    assert analysis.committed_seed_count is None
    assert analysis.shared_rate[VARIANT].tuned_minus_shared == moved[1].shared_rate[VARIANT].tuned_minus_shared


# --- Steps parity (plan §13.6) -------------------------------------------------------------------------


def test_a_skippy_shared_rate_run_is_refused_and_the_same_run_on_seed_21_is_not(moved):
    manifests, _ = moved
    total = 1526  # the 100M total of the registered recipe, as tests/test_audit_fixes.py uses it
    shared_seed_1 = find(manifests, variant=VARIANT, arm="isoparam", init_seed=1, tokens="100M", peak_lr=MHA_RATE)
    assert shared_seed_1["steps"]["total"] == total
    swapped = [with_skips(m, 8) if m["name"] == shared_seed_1["name"] else m for m in manifests]
    with pytest.raises(mt.TableError, match="steps taken"):
        analyse_committed(swapped)
    seed_21 = standalone(VARIANT, 21, MHA_RATE, 3.3)
    assert seed_21["steps"]["total"] == total
    skippy_21 = with_skips(seed_21, 8)
    analysis = analyse_committed(manifests + [skippy_21])
    assert analysis.shared_rate[VARIANT].tuned_minus_shared == moved[1].shared_rate[VARIANT].tuned_minus_shared


# --- The table ------------------------------------------------------------------------------------------


def test_the_table_has_its_title_columns_and_row_order(moved):
    text = table(moved[1])
    lines = text.splitlines()
    assert lines[0] == mt.SHARED_RATE_TITLE == "# Tuned against shared learning rate (plan §7.4 item 5)"
    header = next(l for l in lines if l.startswith("| variant |"))
    assert header == "| " + " | ".join(mt.SHARED_RATE_COLUMNS) + " |"
    rows = [l.split("|")[1].strip() for l in lines if l.startswith("| ") and not l.startswith("| variant |")]
    assert rows == list(HEADLINE_NON_MHA)


def test_the_moved_variant_row_has_formatted_rates_losses_and_gap(moved):
    manifests, analysis = moved
    text = table(analysis)
    cells = row_of(text, VARIANT)
    result = analysis.shared_rate[VARIANT]
    assert cells[1] == mf.format_lr(OWN_RATE) == "2e3"
    assert cells[2] == mf.format_lr(MHA_RATE) == "1e3"
    tuned = ", ".join(f"s{s} {result.tuned_losses[s]:.4f}" for s in SEEDS)
    shared = ", ".join(f"s{s} {result.shared_losses[s]:.4f}" for s in SEEDS)
    assert cells[3] == tuned
    assert cells[4] == shared
    gap = result.tuned_minus_shared
    assert cells[5] == f"{gap.mean_gap:+.4f}"
    assert cells[6] == f"[{gap.ci_low:+.4f}, {gap.ci_high:+.4f}]"


def test_the_other_rows_are_same_rate_or_blank(moved):
    text = table(moved[1])
    for variant in HEADLINE_NON_MHA:
        if variant == VARIANT:
            continue
        cells = row_of(text, variant)
        assert cells[1] == cells[2] == "1e3", variant
        assert cells[3:] == [mt.SAME_RATE] * 4, variant


def test_a_variant_without_a_known_rate_is_blank(moved):
    manifests, _ = moved
    no_gqa2 = [m for m in manifests if mf.parse_run_name(m["name"]).variant != "gqa2"]
    analysis = analyse_committed(no_gqa2)
    assert "gqa2" not in analysis.peak_lr
    assert row_of(table(analysis), "gqa2")[1:] == [mt.BLANK] * 6


def test_every_row_is_blank_while_mhas_rate_is_unknown():
    manifests = [m for m in secondary_campaign() if mf.parse_run_name(m["name"]).variant != "mha"]
    analysis = analyse_committed(manifests)
    assert "mha" not in analysis.peak_lr
    for variant in HEADLINE_NON_MHA:
        assert row_of(table(analysis), variant)[1:] == [mt.BLANK] * 6, variant


def test_notes_are_listed_under_the_table_and_no_verdict_word_appears(moved):
    manifests = at_rate(secondary_campaign(gaps={VARIANT: 0.05}), VARIANT, OWN_RATE)
    analysis = analyse_committed(manifests)
    text = table(analysis)
    assert "Notes:" in text
    listed = [l for l in text.splitlines() if l.startswith("- shared-rate runs:")]
    assert listed == [f"- {n}" for n in analysis.notes if n.startswith("shared-rate runs:")]
    assert listed
    for candidate in (table(moved[1]), text):
        assert "detected" not in candidate
        assert "not detected" not in candidate


def test_the_numbers_are_formatted_like_arm_b(moved):
    # Arm B's gaps and intervals use four decimals and a signed mean; the new table must match.
    text = table(moved[1])
    gap = moved[1].shared_rate[VARIANT].tuned_minus_shared
    assert f"{gap.mean_gap:+.4f}" in text and f"[{gap.ci_low:+.4f}, {gap.ci_high:+.4f}]" in text


# --- main writes the table, and the committed blank form ----------------------------------------------


def test_main_writes_shared_rate_md_equal_to_the_renderer(tmp_path, moved):
    directory = tmp_path / "manifests"
    for manifest in moved[0]:
        mf.write_manifest(manifest, directory)
    out = tmp_path / "tables"
    assert tables_main(["--manifests", str(directory), "--out", str(out)]) == 0
    text = (out / "shared_rate.md").read_text(encoding="utf-8")
    assert text == mt.render_shared_rate(analyse_committed(moved[0]))
    assert VARIANT in row_of(text, VARIANT)[0]


def test_the_committed_blank_form_is_generated_from_no_manifests(tmp_path):
    assert tables_main(["--blank", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "shared_rate.md").read_bytes() == BLANK_FILE.read_bytes()
    assert BLANK_FILE.read_text(encoding="utf-8") == mt.render_shared_rate(analyse_committed([]))


# --- The existing tables are unchanged by runs that nothing reads -------------------------------------


def test_existing_comparisons_and_notes_do_not_change_with_ignored_runs():
    base = secondary_campaign(gaps={VARIANT: 0.05})
    plain = analyse_committed(base)
    extra = [
        standalone(VARIANT, 21, MHA_RATE, 3.3),
        standalone(VARIANT, 1, THIRD_RATE, 3.4),
        standalone("gqa4", 1, MHA_RATE, 3.3),
    ]
    with_extra = analyse_committed(base + extra)
    assert with_extra.comparisons == plain.comparisons
    assert with_extra.multipliers == plain.multipliers
    assert with_extra.notes == plain.notes
    assert with_extra.shared_rate == {} == plain.shared_rate


# --- Added by the orchestrator in review: three breaks the tests above did not catch ----------------


def test_runs_that_no_comparison_reads_stay_outside_the_steps_parity_check(moved):
    """Only a shared-rate run is read. A standalone 100M run on seeds 1 to 3 at MHA's rate of a variant
    that itself runs at MHA's rate, or of a ladder size, is read by nothing, so a skippy one must not
    stop the tables (plan §13.6: parity is checked among the runs a comparison reads)."""
    manifests, analysis = moved
    same_rate_variant = next(v for v in HEADLINE_NON_MHA if v != VARIANT and analysis.peak_lr.get(v) == MHA_RATE)
    skippy_same_rate = with_skips(standalone(same_rate_variant, 1, MHA_RATE, 3.3), 8)
    skippy_ladder = with_skips(
        fake_manifest("mha", ladder="mha_L6", budget="100M", branches=[], init_seed=1, peak_lr=MHA_RATE, val_loss=3.3), 8
    )
    for extra in (skippy_same_rate, skippy_ladder):
        again = analyse_committed(manifests + [extra])
        assert again.shared_rate == analysis.shared_rate and again.comparisons == analysis.comparisons


def test_a_ladder_size_on_its_own_rate_gets_no_shared_rate_result_and_its_runs_are_not_read():
    """Decision 224: the ladder sizes get no such runs. With mha_L6 moved to its own rate, a skippy
    standalone 100M L6 run at MHA's rate on seed 1 is read by nothing."""
    base = secondary_campaign(gaps={VARIANT: 0.05})
    moved_ladder = []
    for manifest in base:
        parsed = mf.parse_run_name(manifest["name"])
        if parsed.variant != "mha_L6":
            moved_ladder.append(manifest)
            continue
        extra = {"branch": parsed.branch} if parsed.branch else {}
        moved_ladder.append(fake_manifest(
            "mha", ladder="mha_L6", init_seed=parsed.init_seed, peak_lr=OWN_RATE,
            val_loss=float(manifest["final_metrics"]["val_loss"]), **extra,
        ))
    assert moved_ladder != base
    plain = analyse_committed(moved_ladder)
    assert plain.shared_rate == {} and plain.peak_lr["mha_L6"] == OWN_RATE
    skippy = with_skips(
        fake_manifest("mha", ladder="mha_L6", budget="100M", branches=[], init_seed=1, peak_lr=MHA_RATE, val_loss=3.3), 8
    )
    again = analyse_committed(moved_ladder + [skippy])
    assert again.shared_rate == {} and again.comparisons == plain.comparisons
    assert not any(note.startswith("shared-rate runs:") for note in again.notes)


def test_two_variants_on_their_own_rates_come_in_headline_order():
    first, second = "gqa2", "mqa"  # headline order is mqa before gqa2
    assert mt.HEADLINE_VARIANTS.index(second) < mt.HEADLINE_VARIANTS.index(first)
    base = secondary_campaign(gaps={VARIANT: 0.05})
    manifests = at_rate(at_rate(base, first, OWN_RATE), second, OWN_RATE)
    manifests = manifests + shared_runs(manifests, first) + shared_runs(manifests, second)
    analysis = analyse_committed(manifests)
    assert list(analysis.shared_rate) == [second, first]
    assert all(result.tuned_minus_shared is not None for result in analysis.shared_rate.values())
