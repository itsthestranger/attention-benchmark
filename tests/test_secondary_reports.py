"""T-34: Arm B, context use, the timing and bench refusals, and RQ-W in scripts/make_tables.py.

plan §8.3 (Arm B), §8.5 (RQ-W), §10.3 (context use), §10.6 and §11 (refusals).
Every manifest, timing document and bench document here is SYNTHETIC
(tests/fixtures/fake_secondary.py, fake_campaign.py, fake_manifests.py). The statistics are
never recomputed here: the tests call `primary_test.secondary` on values read from the
manifests and compare the result with the object `make_tables` returns.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from analysis import primary_test as pt
from analysis import rqw_estimate as rq
from scripts import bench as bd
from scripts import make_tables as mt
from scripts import timing_pass as tp
from src import config
from src.utils import manifest as mf
from tests.fixtures.fake_secondary import (
    ARM_B_SEEDS,
    ARM_B_VARIANTS,
    DECODE_TOK_PER_S,
    ENVIRONMENT,
    TIMING_MS,
    arm_b_manifests,
    bench_document,
    find,
    secondary_campaign,
    timing_document,
    with_context,
    with_val_removed,
    write_bench_documents,
    write_timing_document,
)
from tests.fixtures.fake_manifests import fake_manifest

ROOT = Path(__file__).resolve().parents[1]
BLANK_DIR = ROOT / "results" / "tables" / "blank"
BLANK_NAMES = ("headline.md", "gates.md", "arm_b.md", "context.md", "rqw.md")
PRE = config.load_prereg()
PRIMARIES = tuple(PRE["primary"]["comparisons"])
EXPLORATORY = tuple(PRE["exploratory"]["variants"])
EXPLORATORY_SEEDS = tuple(PRE["exploratory"]["init_seeds"])
TEST_SELECTION_SEEDS = tuple(PRE["test_selection"]["sigma_init"]["init_seeds"])
CONTEXT_VARIANTS = tuple(v for v in mt.HEADLINE_VARIANTS if v != "mha")
ARM_B_D_FF = config.load_arm("iso_backbone")["d_ff"]


# --- Helpers ------------------------------------------------------------------------------


def _loss(manifests, variant, seed, *, arm="isoparam", tokens="300M", branch=None) -> float:
    return float(find(manifests, variant=variant, arm=arm, init_seed=seed, tokens=tokens, branch=branch)["final_metrics"]["val_loss"])


def _val(manifests, variant, seed, key) -> float:
    block = find(manifests, variant=variant, arm="isoparam", init_seed=seed, tokens="300M")["final_metrics"]["val"]
    return float(block["late_context"]["loss"] if key == "late_context" else block["icl"]["score"])


def _without(manifests, *, variant, arm, seed, tokens, peak_lr=None):
    target = find(manifests, variant=variant, arm=arm, init_seed=seed, tokens=tokens, peak_lr=peak_lr)
    assert target is not None, (variant, arm, seed, tokens)
    return [m for m in manifests if m is not target]


def _cells(text: str, first: str) -> list[str]:
    row = next(line for line in text.splitlines() if line.startswith(f"| {first} |"))
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _first_column(text: str) -> list[str]:
    return [line.split("|")[1].strip() for line in text.splitlines() if line.startswith("| ") and not line.startswith("| variant")]


def _violations(error) -> list[str]:
    return [line for line in str(error).splitlines() if line.startswith("  - ")]


def _write_manifests(directory: Path, manifests) -> Path:
    for manifest in manifests:
        mf.write_manifest(manifest, directory)
    return directory


# --- Module fixtures (each campaign is built once) ----------------------------------------------


@pytest.fixture(scope="module")
def arm_a_campaign() -> list[dict]:
    """Arm A only, with context fields: the paired case (s_rep well under sigma_init)."""
    return with_context(secondary_campaign(gaps={"mqa": 0.05}))


@pytest.fixture(scope="module")
def paired(arm_a_campaign):
    manifests = arm_a_campaign + arm_b_manifests(arm_a_campaign)
    return manifests, mt.analyse(manifests)


@pytest.fixture(scope="module")
def welch(arm_a_campaign):
    """The Welch case: the reruns scatter far more than sigma_init, so s_rep / sigma_init is large."""
    base = with_context(secondary_campaign(gaps={"mqa": 0.05}, rerun_sigma=0.02))
    manifests = base + arm_b_manifests(base)
    return manifests, mt.analyse(manifests)


@pytest.fixture(scope="module")
def no_null(arm_a_campaign):
    """The gate is shut: the null control is missing."""
    base = with_context(secondary_campaign(without=("null",), gaps={"mqa": 0.05}))
    manifests = base + arm_b_manifests(base)
    return manifests, mt.analyse(manifests)


@pytest.fixture(scope="module")
def eight():
    """Eight seeds: the seed-count rule extends the primaries and mla96 and swa256 leave Arm A."""
    manifests = with_context(secondary_campaign(sigma=0.01, n_primary_seeds=8, gaps={"mqa": 0.05}))
    return manifests, mt.analyse(manifests)


@pytest.fixture(params=["paired", "welch"])
def case(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="module")
def results(paired):
    return mt.select_results(paired[0])[0]


# --- The fixtures take the tests they are meant for -------------------------------------------


def test_the_fixtures_take_the_tests_they_are_meant_for(paired, welch, no_null):
    assert paired[1].selection.test == pt.PAIRED
    assert welch[1].selection.test == pt.WELCH
    assert no_null[1].comparisons_allowed is False
    assert paired[1].comparisons_allowed is True
    assert paired[1].decision.seeds == (1, 2, 3, 4, 5)


# --- Arm B (plan §8.3) -----------------------------------------------------------------------


def test_arm_b_has_the_arm_files_variants_in_their_order(case):
    _, analysis = case
    arm = config.load_arm("iso_backbone")
    assert list(analysis.arm_b) == list(arm["variants"]) == list(ARM_B_VARIANTS)
    for variant, result in analysis.arm_b.items():
        assert result.variant == variant


def test_arm_b_losses_are_the_standalone_runs_and_arm_a_the_100m_branches(case):
    manifests, analysis = case
    for variant in ARM_B_VARIANTS:
        result = analysis.arm_b[variant]
        assert result.losses == {s: _loss(manifests, variant, s, arm="isobackbone", tokens="100M") for s in ARM_B_SEEDS}
        assert result.arm_a_losses == {s: _loss(manifests, variant, s, tokens="300M", branch="100M") for s in ARM_B_SEEDS}


def test_vs_mha_is_primary_test_secondary_called_on_the_manifests_values(case):
    manifests, analysis = case
    test = analysis.selection.test
    mha_seeds = ARM_B_SEEDS if test == pt.PAIRED else TEST_SELECTION_SEEDS
    for variant in ARM_B_VARIANTS:
        expected = pt.secondary(
            variant,
            {s: _loss(manifests, variant, s, arm="isobackbone", tokens="100M") for s in ARM_B_SEEDS},
            {s: _loss(manifests, "mha", s, tokens="300M", branch="100M") for s in mha_seeds},
            reference="mha",
            seeds=ARM_B_SEEDS,
            test=test,
        )
        assert analysis.arm_b[variant].vs_mha == expected


def test_arm_a_minus_arm_b_is_primary_test_secondary_with_arm_a_first(case):
    manifests, analysis = case
    test = analysis.selection.test
    for variant in ARM_B_VARIANTS:
        expected = pt.secondary(
            variant,
            {s: _loss(manifests, variant, s, tokens="300M", branch="100M") for s in ARM_B_SEEDS},
            {s: _loss(manifests, variant, s, arm="isobackbone", tokens="100M") for s in ARM_B_SEEDS},
            reference=f"{variant} at d_ff {ARM_B_D_FF}",
            seeds=ARM_B_SEEDS,
            test=test,
        )
        assert analysis.arm_b[variant].arm_a_minus_arm_b == expected


def test_a_wider_mlp_that_helps_gives_a_negative_arm_a_minus_arm_b(arm_a_campaign):
    # Arm B worse by a constant: Arm A minus Arm B is negative. Arm B better: positive.
    worse = mt.analyse(arm_a_campaign + arm_b_manifests(arm_a_campaign, offset=0.25, noise=0.0))
    better = mt.analyse(arm_a_campaign + arm_b_manifests(arm_a_campaign, offset=-0.25, noise=0.0))
    for variant in ARM_B_VARIANTS:
        assert worse.arm_b[variant].arm_a_minus_arm_b.mean_gap < 0
        assert better.arm_b[variant].arm_a_minus_arm_b.mean_gap > 0


def test_a_missing_arm_b_seed_gives_none_and_a_note_naming_it(paired):
    manifests, _ = paired
    kept = _without(manifests, variant="mqa", arm="isobackbone", seed=2, tokens="100M")
    analysis = mt.analyse(kept)
    result = analysis.arm_b["mqa"]
    assert set(result.losses) == {1, 3}
    assert result.vs_mha is None and result.arm_a_minus_arm_b is None
    assert any(n.startswith("Arm B mqa") and "[2]" in n for n in analysis.notes)
    assert analysis.arm_b["gqa4"].vs_mha is not None


def test_a_variant_with_no_arm_b_run_has_empty_losses_and_one_note(paired):
    manifests, _ = paired
    kept = manifests
    for seed in ARM_B_SEEDS:
        kept = _without(kept, variant="mqa", arm="isobackbone", seed=seed, tokens="100M")
    analysis = mt.analyse(kept)
    result = analysis.arm_b["mqa"]
    assert result.losses == {} and result.vs_mha is None and result.arm_a_minus_arm_b is None
    notes = [n for n in analysis.notes if n.startswith("Arm B mqa")]
    assert len(notes) == 1 and "no standalone 100M run" in notes[0]


def test_two_learning_rates_for_one_arm_b_variant_raise(paired):
    manifests, _ = paired
    extra = fake_manifest("mqa", arm="iso_backbone", budget="100M", branches=[], init_seed=1, peak_lr=3e-3, val_loss=3.0)
    with pytest.raises(mt.TableError, match="more than one peak learning rate"):
        mt.analyse(manifests + [extra])


def test_without_the_null_control_the_comparisons_wait_and_the_losses_are_listed(no_null):
    _, analysis = no_null
    assert analysis.comparisons_allowed is False
    assert analysis.context_use == {}
    for variant in ARM_B_VARIANTS:
        result = analysis.arm_b[variant]
        assert set(result.losses) == set(ARM_B_SEEDS)
        assert result.vs_mha is None and result.arm_a_minus_arm_b is None


def test_the_blank_analysis_has_one_empty_result_per_arm_variant_and_no_arm_b_note():
    blank = mt.analyse([])
    assert list(blank.arm_b) == list(ARM_B_VARIANTS)
    for result in blank.arm_b.values():
        assert result.losses == {} and result.arm_a_losses == {}
        assert result.vs_mha is None and result.arm_a_minus_arm_b is None
    assert blank.context_use == {} and blank.comparisons_allowed is False
    assert not any(n.startswith("Arm B") or n.startswith("context use") for n in blank.notes)


# --- Context use (plan §10.3) ---------------------------------------------------------------


def test_context_use_has_the_headline_variants_but_mha_in_order(case):
    _, analysis = case
    assert not analysis.decision.extended
    assert list(analysis.context_use) == list(CONTEXT_VARIANTS)


def test_each_context_comparison_is_primary_test_secondary_called_on_the_manifests(case):
    manifests, analysis = case
    test = analysis.selection.test
    for variant, cu in analysis.context_use.items():
        seeds = tuple(analysis.decision.seeds) if variant in PRIMARIES else EXPLORATORY_SEEDS
        ref_seeds = seeds if test == pt.PAIRED else tuple(analysis.decision.seeds)
        for key, comparison in (("late_context", cu.late_context), ("icl", cu.icl_score)):
            expected = pt.secondary(
                variant,
                {s: _val(manifests, variant, s, key) for s in seeds},
                {s: _val(manifests, "mha", s, key) for s in ref_seeds},
                reference="mha",
                seeds=seeds,
                test=test,
            )
            assert comparison == expected, (variant, key)


def test_primaries_use_the_decision_seeds_and_exploratory_variants_their_three(case):
    _, analysis = case
    for variant, cu in analysis.context_use.items():
        expected = tuple(analysis.decision.seeds) if variant in PRIMARIES else EXPLORATORY_SEEDS
        assert cu.seeds == expected
        for comparison in (cu.late_context, cu.icl_score):
            assert comparison.seeds == expected


def test_a_result_without_a_val_block_gives_none_and_names_variant_metric_and_seed(paired):
    manifests, _ = paired
    analysis = mt.analyse(with_val_removed(manifests, variant="gqa4", init_seed=2))
    cu = analysis.context_use["gqa4"]
    assert cu.late_context is None and cu.icl_score is None
    assert cu.late_context_values[2] is None and cu.icl_values[2] is None
    assert cu.late_context_values[1] is not None
    assert any(n.startswith("context use: gqa4 late-context loss is missing for seeds [2]") for n in analysis.notes)
    assert any(n.startswith("context use: gqa4 ICL score is missing for seeds [2]") for n in analysis.notes)
    assert analysis.context_use["mqa"].late_context is not None


def test_a_missing_mha_value_gives_none_for_every_variant_and_names_mha(paired):
    manifests, _ = paired
    analysis = mt.analyse(with_val_removed(manifests, variant="mha", init_seed=1))
    for cu in analysis.context_use.values():
        assert cu.late_context is None and cu.icl_score is None
    assert any(n.startswith("context use: mha late-context loss is missing for seeds [1]") for n in analysis.notes)


def test_eight_seeds_skip_the_variants_that_left_arm_a(eight):
    _, analysis = eight
    assert analysis.decision.extended
    assert set(analysis.decision.leaves_arm_a) == {"mla96", "swa256"}
    for leaving in analysis.decision.leaves_arm_a:
        assert leaving not in analysis.context_use
    for variant in PRIMARIES:
        assert analysis.context_use[variant].seeds == tuple(range(1, 9))
    assert analysis.context_use["gqa2"].seeds == EXPLORATORY_SEEDS
    assert not any(n.startswith("context use") for n in analysis.notes)


# --- Rendering: Arm B and context use -------------------------------------------------------------


def test_arm_b_md_has_one_row_per_variant_in_order_and_the_captions(paired):
    _, analysis = paired
    text = mt.render_arm_b(analysis)
    assert _first_column(text) == list(ARM_B_VARIANTS)
    assert mt.ARM_B_CAPTION in text and "exploratory" in mt.ARM_B_CAPTION
    assert mt.ARM_B_MQA_LINE in text


def test_each_verdict_cell_is_the_comparisons_own_verdict_and_gaps_are_formatted(paired):
    _, analysis = paired
    text = mt.render_arm_b(analysis)
    for variant in ARM_B_VARIANTS:
        cells = _cells(text, variant)
        vs, diff = analysis.arm_b[variant].vs_mha, analysis.arm_b[variant].arm_a_minus_arm_b
        assert cells[3] == vs.verdict == pt.VERDICT_EXPLORATORY
        assert cells[6] == diff.verdict == pt.VERDICT_EXPLORATORY
        assert cells[2] == f"{vs.mean_gap:+.4f} [{vs.ci_low:+.4f}, {vs.ci_high:+.4f}]"
        assert cells[5] == f"{diff.mean_gap:+.4f} [{diff.ci_low:+.4f}, {diff.ci_high:+.4f}]"


def test_none_renders_as_a_dash_in_the_gap_and_verdict_cells(no_null):
    _, analysis = no_null
    text = mt.render_arm_b(analysis)
    for variant in ARM_B_VARIANTS:
        cells = _cells(text, variant)
        assert cells[1] != mt.BLANK  # the losses are listed while the gate is shut
        assert cells[2] == cells[3] == cells[5] == cells[6] == mt.BLANK
    assert f"Not computed: {mt.GATE_WAIT}" in text


def test_arm_b_rankings_are_the_plain_means_of_the_manifests(case):
    manifests, analysis = case
    text = mt.render_arm_b(analysis)
    arm_a = {v: [_loss(manifests, v, s, tokens="300M", branch="100M") for s in ARM_B_SEEDS] for v in ARM_B_VARIANTS}
    arm_b = {v: [_loss(manifests, v, s, arm="isobackbone", tokens="100M") for s in ARM_B_SEEDS] for v in ARM_B_VARIANTS}
    orders = []
    lines = []
    for losses in (arm_a, arm_b):
        means = {v: sum(losses[v]) / len(ARM_B_SEEDS) for v in ARM_B_VARIANTS}
        order = sorted(ARM_B_VARIANTS, key=lambda v: means[v])
        orders.append(order)
        lines += [f"{i}. {v}, mean {means[v]:.4f}" for i, v in enumerate(order, 1)]
    listed = [line for line in text.splitlines() if re.match(r"^\d\. ", line)]
    assert listed == lines
    same = orders[0] == orders[1]
    assert ("The two orders are the same." in text) == same
    assert ("The two orders differ." in text) == (not same)


def test_arm_b_rankings_are_not_stated_while_the_gate_is_shut(no_null):
    _, analysis = no_null
    text = mt.render_arm_b(analysis)
    assert not re.search(r"^\d\. ", text, re.MULTILINE)
    assert "Not stated: " + mt.GATE_WAIT in text


def test_context_md_has_every_variant_but_mha_in_order_with_captions_and_verdicts(paired):
    _, analysis = paired
    text = mt.render_context(analysis)
    assert _first_column(text) == list(CONTEXT_VARIANTS)
    assert mt.CONTEXT_CAPTION in text and mt.SWA_CAPTION in text
    assert "plan §13.11" in mt.SWA_CAPTION
    for variant in CONTEXT_VARIANTS:
        cells = _cells(text, variant)
        assert cells[3] == analysis.context_use[variant].late_context.verdict
        assert cells[6] == analysis.context_use[variant].icl_score.verdict


def test_context_md_shows_the_left_arm_variants_and_dashes_when_nothing_is_computed(eight):
    _, analysis = eight
    text = mt.render_context(analysis)
    assert _cells(text, "swa256")[3] == mt.LEFT_ARM
    assert _cells(text, "mla96")[6] == mt.LEFT_ARM
    blank_text = mt.render_context(mt.analyse([]))
    assert _first_column(blank_text) == list(CONTEXT_VARIANTS)
    for variant in CONTEXT_VARIANTS:
        assert all(cell == mt.BLANK for cell in _cells(blank_text, variant)[1:])
    assert "Not computed:" in blank_text


# --- Timing and bench: the command line and the refusals (plan §2.2, §10.6, §11) -------------------


def test_main_fills_the_timing_and_decode_columns_from_the_readers(tmp_path, paired):
    manifests, _ = paired
    directory = _write_manifests(tmp_path / "manifests", manifests)
    timing = tmp_path / "timing.json"
    write_timing_document(timing, timing_document())
    bench = tmp_path / "bench"
    write_bench_documents(bench, [bench_document(v) for v in DECODE_TOK_PER_S] + [bench_document("mha", "ref")])
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(directory), "--out", str(out), "--timing-pass", str(timing), "--bench", str(bench)]) == 0
    analysis = mt.analyse(mt.select_results(mt.load_manifests(directory))[0])
    expected = mt.render_markdown(
        mt.headline_rows(
            analysis,
            train_ms_per_step=tp.read_ms_per_step(timing),
            decode_tok_per_s=bd.read_decode_tok_per_s(bench),
        ),
        analysis.notes,
    )
    assert (out / "headline.md").read_text(encoding="utf-8") == expected
    cells = _cells((out / "headline.md").read_text(encoding="utf-8"), "mha")
    assert cells[mt.COLUMNS.index("train ms/step")] == f"{TIMING_MS['mha']:.1f}"
    assert cells[mt.COLUMNS.index("decode tok/s (batch 32, cache 4k)")] == f"{DECODE_TOK_PER_S['mha']:.1f}"


def test_a_shared_gpu_is_refused_for_the_timing_pass(results):
    with pytest.raises(mt.TableError, match="timing pass: recorded with gpu_shared True"):
        mt.check_timing_sources(results, timing_document(shared=True), [bench_document("mha")])


def test_a_shared_gpu_is_refused_for_a_bench_document(results):
    with pytest.raises(mt.TableError, match="bench mqa/opt: recorded with gpu_shared True"):
        mt.check_timing_sources(results, timing_document(), [bench_document("mqa", shared=True)])


def test_a_torch_mismatch_between_bench_and_timing_is_refused(results):
    bench = bench_document("mqa", environment={**ENVIRONMENT, "torch": "2.16.0"})
    with pytest.raises(mt.TableError, match="torch version differs"):
        mt.check_timing_sources(results, timing_document(), [bench])


def test_a_driver_mismatch_with_the_results_is_refused(results):
    timing = timing_document(environment={**ENVIRONMENT, "driver": "998.0"})
    with pytest.raises(mt.TableError, match="driver version of the results against"):
        mt.check_timing_sources(results, timing, [bench_document("mha")])


def test_a_backend_mismatch_between_opt_documents_is_refused(results):
    with pytest.raises(mt.TableError, match="backend differs"):
        mt.check_timing_sources(results, timing_document(), [bench_document("gqa4", backend="math")])


def test_every_problem_is_listed_together(results):
    timing = timing_document(shared=True)
    bench = [bench_document("mqa", environment={**ENVIRONMENT, "torch": "2.16.0"}), bench_document("gqa4", backend="math")]
    with pytest.raises(mt.TableError) as info:
        mt.check_timing_sources(results, timing, bench)
    lines = _violations(info.value)
    assert len(lines) == 3
    assert any("gpu_shared" in line for line in lines)
    assert any("torch version differs" in line for line in lines)
    assert any("backend differs" in line for line in lines)


def test_consistent_sources_pass_including_the_ref_documents(results):
    bench = [bench_document("mha"), bench_document("mha", "ref"), bench_document("mqa")]
    assert mt.check_timing_sources(results, timing_document(), bench) is None
    assert mt.check_timing_sources(results, None, []) is None


def test_main_refusal_exits_one_and_writes_no_file(tmp_path, paired):
    manifests, _ = paired
    directory = _write_manifests(tmp_path / "manifests", manifests)
    timing = tmp_path / "timing.json"
    write_timing_document(timing, timing_document(shared=True))
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(directory), "--out", str(out), "--timing-pass", str(timing)]) == 1
    assert not any((out / name).exists() for name in BLANK_NAMES)


# --- RQ-W (plan §8.5) -------------------------------------------------------------------------


def test_rqw_md_equals_render_rqw_of_the_estimate_called_in_the_test(tmp_path, paired, results):
    manifests, _ = paired
    directory = _write_manifests(tmp_path / "manifests", manifests)
    timing = tmp_path / "timing.json"
    write_timing_document(timing, timing_document())
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(directory), "--out", str(out), "--timing-pass", str(timing)]) == 0
    expected = mt.render_rqw(rq.estimate(results, tp.read_ms_per_step(timing)), None)
    assert (out / "rqw.md").read_text(encoding="utf-8") == expected
    assert "estimated" in expected and rq.INTERPOLATION in expected


def test_rqw_waits_for_the_controls_when_the_gate_is_shut(tmp_path, no_null):
    manifests, _ = no_null
    directory = _write_manifests(tmp_path / "manifests", manifests)
    timing = tmp_path / "timing.json"
    write_timing_document(timing, timing_document())
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(directory), "--out", str(out), "--timing-pass", str(timing)]) == 0
    text = (out / "rqw.md").read_text(encoding="utf-8")
    assert mt.GATE_WAIT in text
    assert "| variant |" not in text


def test_rqw_without_a_timing_pass_is_the_blank_form(tmp_path, paired):
    manifests, _ = paired
    directory = _write_manifests(tmp_path / "manifests", manifests)
    out = tmp_path / "tables"
    assert mt.main(["--manifests", str(directory), "--out", str(out)]) == 0
    text = (out / "rqw.md").read_text(encoding="utf-8")
    assert text == mt.render_rqw(None, None)
    assert mt.NO_TIMING in text


# --- The blank files (generated from no manifests) ------------------------------------------------


def test_committed_blank_files_are_generated_from_no_manifests():
    blank = mt.analyse([])
    assert (BLANK_DIR / "arm_b.md").read_text(encoding="utf-8") == mt.render_arm_b(blank)
    assert (BLANK_DIR / "context.md").read_text(encoding="utf-8") == mt.render_context(blank)
    assert (BLANK_DIR / "rqw.md").read_text(encoding="utf-8") == mt.render_rqw(None, None)


def test_main_blank_writes_all_five_files_equal_to_the_committed_ones(tmp_path):
    assert mt.main(["--blank", "--out", str(tmp_path)]) == 0
    for name in BLANK_NAMES:
        assert (tmp_path / name).read_bytes() == (BLANK_DIR / name).read_bytes(), name
