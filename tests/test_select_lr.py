"""The learning-rate sweep report (scripts/select_lr.py), on SYNTHETIC manifests only.

Every loss, rate and seed here is made up. The sweep rules themselves are tested in
test_lr_sweep.py; these tests check the wiring: which runs feed which name, the order of
the sections, the values printed, sigma_init from MHA's Arm N runs, and the refusals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis import lr_sweep as sw
from scripts import make_tables as mt
from scripts import select_lr as sel
from src import config
from src.utils import manifest as mf
from tests.fixtures.fake_manifests import fake_manifest

ROOT = Path(__file__).resolve().parents[1]
SWEEP_SEED = 21
ARM_N_BASE = 3.2  # made-up base loss of the Arm N runs
REF = 1.2e-3
MHA_POINTS = {3e-4: 3.60, 6e-4: 3.55, 1.2e-3: 3.50, 2.4e-3: 3.52, 4.8e-3: 3.70}


def sweep_run(variant: str, lr: float, loss: float | None, *, ladder: str | None = None):
    """A standalone 100M run at the sweep seed; loss None makes it a diverged run."""
    if loss is None:
        return fake_manifest(
            variant, ladder=ladder, budget="100M", branches=[], init_seed=SWEEP_SEED,
            peak_lr=lr, completed=False, final_metrics={"diverged": True, "reason": "made up"},
        )
    return fake_manifest(
        variant, ladder=ladder, budget="100M", branches=[], init_seed=SWEEP_SEED,
        peak_lr=lr, val_loss=loss,
    )


def arm_n(spread: float, *, rate: float = REF) -> list[dict]:
    """MHA's Arm N runs at `rate`: five 100M branches (test selection) and three reruns.

    The five branch losses are ARM_N_BASE + spread * (0, 1, -1, 0.5, -0.5), so their sample
    SD is spread * sqrt(0.625). The reruns sit at the same losses and feed the pair test.
    """
    offsets = (0.0, 1.0, -1.0, 0.5, -0.5)
    manifests = [
        fake_manifest("mha", init_seed=seed, peak_lr=rate, branch="100M",
                      val_loss=ARM_N_BASE + spread * offsets[seed - 1])
        for seed in range(1, 6)
    ]
    manifests += [
        fake_manifest("mha", budget="100M", branches=[], init_seed=seed, peak_lr=rate,
                      val_loss=ARM_N_BASE + spread * offsets[seed - 1])
        for seed in range(1, 4)
    ]
    return manifests


def mha_sweep(losses: dict) -> list[dict]:
    return [sweep_run("mha", lr, loss) for lr, loss in losses.items()]


def full_sweep(spread: float = 0.01) -> list[dict]:
    """A synthetic campaign that reaches every status: see the expectations in the tests below."""
    manifests = mha_sweep(MHA_POINTS) + arm_n(spread)
    manifests += [sweep_run("gqa4", lr, loss) for lr, loss in
                  {6e-4: 3.56, 1.2e-3: 3.503, 2.4e-3: 3.500, 4.8e-3: 3.60}.items()]  # tie rule
    manifests += [sweep_run("mqa", lr, loss) for lr, loss in
                  {6e-4: 3.56, 1.2e-3: 3.50, 2.4e-3: 3.53}.items()]  # the reference rate wins
    manifests += [sweep_run("mla192", lr, loss) for lr, loss in
                  {6e-4: 3.56, 1.2e-3: 3.51, 2.4e-3: 3.50, 4.8e-3: 3.55}.items()]  # winner
    manifests += [sweep_run("gqa2", lr, loss) for lr, loss in
                  {6e-4: 3.56, 1.2e-3: 3.50, 2.4e-3: 3.48}.items()]  # extend upwards
    manifests += [sweep_run("mla96", lr, loss) for lr, loss in
                  {6e-4: 3.56, 2.4e-3: 3.53}.items()]  # incomplete: 1.2e-3 is missing
    manifests += [sweep_run("swa256", lr, loss) for lr, loss in
                  {6e-4: 3.56, 1.2e-3: 3.50, 2.4e-3: 3.52, 4.8e-3: None}.items()]  # a divergence
    manifests += [sweep_run("mha", lr, loss, ladder="mha_L6") for lr, loss in
                  {6e-4: 3.80, 1.2e-3: 3.70, 2.4e-3: 3.60, 4.8e-3: 3.75}.items()]  # ladder winner
    return manifests


def write(manifests: list[dict], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        mf.write_manifest(manifest, directory)
    return directory


def section(text: str, name: str) -> str:
    """The text of one section of the report, from its heading to the next heading."""
    body = text.split(f"## {name}\n", 1)[1]
    return body.split("\n## ", 1)[0]


# --- The blank report ---------------------------------------------------------------------------


def test_the_blank_report_has_every_section_and_no_decision(tmp_path):
    assert sel.main(["--blank", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "lr_sweep.md").read_text(encoding="utf-8")
    assert "Manifests read: 0" in text
    assert "## mha" in text and "**Status:** incomplete" in text
    assert "3e4, 6e4, 1.2e3, 2.4e3, 4.8e3" in section(text, "mha")
    for name in sel.names_in_order()[1:]:
        body = section(text, name)
        assert "waiting for MHA's sweep" in body
        assert "**Rate to use:** —" in body
        assert "**lr_source:** —" in body


def test_the_blank_report_needs_no_sweep_seed_and_reads_no_manifests(tmp_path):
    assert sel.main(["--blank", "--manifests", str(tmp_path / "missing"), "--out", str(tmp_path)]) == 0


def test_the_committed_blank_report_is_generated_from_no_manifests(tmp_path):
    assert sel.main(["--blank", "--out", str(tmp_path)]) == 0
    committed = (ROOT / "results" / "tables" / "blank" / "lr_sweep.md").read_bytes()
    assert committed == (tmp_path / "lr_sweep.md").read_bytes()


# --- A full synthetic sweep ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    directory = write(full_sweep(), tmp_path_factory.mktemp("full") / "manifests")
    manifests = mt.load_manifests(directory)
    choices, points, notes = sel.build(manifests, SWEEP_SEED)
    out = tmp_path_factory.mktemp("out")
    assert sel.main(["--manifests", str(directory), "--sweep-seed", str(SWEEP_SEED),
                     "--out", str(out)]) == 0
    return choices, points, (out / "lr_sweep.md").read_text(encoding="utf-8")


def test_mha_is_chosen_by_the_winner_rule(full):
    choices, _, _ = full
    mha = choices[sw.REFERENCE]
    assert mha.status == sw.CHOSEN and mha.lr == REF
    assert mha.rule == sw.RULE_WINNER
    assert mha.source_run == f"mha-isoparam-100M-lr1.2e3-s{SWEEP_SEED}-d1"


def test_a_variant_that_beats_mha_by_less_than_sigma_takes_mhas_rate(full):
    choices, _, _ = full
    gqa4 = choices["gqa4"]
    assert gqa4.status == sw.CHOSEN and gqa4.rule == sw.RULE_TIE
    assert gqa4.best_lr == 2.4e-3
    assert gqa4.lr == REF
    # The sweep run of gqa4 itself at the chosen rate, not MHA's run at that rate.
    assert gqa4.source_run == f"gqa4-isoparam-100M-lr1.2e3-s{SWEEP_SEED}-d1"
    assert gqa4.shared_loss == 3.503 and gqa4.tuned_loss == 3.500


def test_a_variant_that_beats_mha_by_more_than_sigma_keeps_its_rate(full):
    choices, _, _ = full
    mla192 = choices["mla192"]
    assert mla192.status == sw.CHOSEN and mla192.rule == sw.RULE_WINNER
    assert mla192.lr == 2.4e-3
    assert mla192.source_run == f"mla192-isoparam-100M-lr2.4e3-s{SWEEP_SEED}-d1"
    assert mla192.shared_loss == 3.51 and mla192.tuned_loss == 3.50


def test_a_variant_whose_winner_is_mha_uses_mhas_rate(full):
    choices, _, _ = full
    mqa = choices["mqa"]
    assert mqa.status == sw.CHOSEN and mqa.lr == REF and mqa.rule == sw.RULE_WINNER


def test_an_extending_variant_gives_the_next_rate_and_no_decision(full):
    choices, _, _ = full
    gqa2 = choices["gqa2"]
    assert gqa2.status == sw.EXTEND and gqa2.lr is None and gqa2.source_run is None
    assert mf.format_lr(gqa2.next_lr) == "4.8e3"


def test_an_incomplete_variant_lists_its_missing_rate(full):
    choices, _, _ = full
    mla96 = choices["mla96"]
    assert mla96.status == sw.INCOMPLETE and mla96.lr is None
    assert mla96.missing == (REF,)


def test_a_diverged_point_is_a_point_that_cannot_win(full):
    choices, _, _ = full
    swa = choices["swa256"]
    assert swa.status == sw.CHOSEN and swa.lr == REF
    (diverged,) = [p for p in swa.points if p.diverged]
    assert diverged.lr == 4.8e-3 and diverged.loss is None and diverged.reason == "made up"


def test_a_ladder_size_is_chosen_like_a_variant_and_a_name_with_no_runs_is_incomplete(full):
    choices, _, _ = full
    assert choices["mha_L6"].status == sw.CHOSEN and choices["mha_L6"].lr == 2.4e-3
    assert choices["mha_L6"].rule == sw.RULE_WINNER
    mha_l10 = choices["mha_L10"]
    assert mha_l10.status == sw.INCOMPLETE and mha_l10.points == ()
    assert mha_l10.missing == (6e-4, REF, 2.4e-3)


def test_the_report_prints_each_choice_rate_and_source(full):
    choices, _, text = full
    for name, choice in choices.items():
        body = section(text, name)
        if choice.lr is not None:
            assert f"- **Rate to use:** {mf.format_lr(choice.lr)}" in body
            assert f"- **lr_source:** {choice.source_run}" in body
        else:
            assert "- **Rate to use:** —" in body
            assert "- **lr_source:** —" in body
        assert f"- **Status:** {sel.STATUS_TEXT[choice.status]}" in body


def test_the_report_prints_the_extension_and_missing_rates_from_the_choices(full):
    choices, _, text = full
    assert f"- **Next point to run (extend):** {mf.format_lr(choices['gqa2'].next_lr)}" in section(text, "gqa2")
    assert f"- **Missing rates (incomplete):** {mf.format_lr(REF)}" in section(text, "mla96")


def test_the_report_prints_every_point_with_its_rate_loss_or_divergence_and_run(full):
    _, points, text = full
    body = section(text, "swa256")
    assert "| 4.8e3 | DIVERGED: made up | swa256-isoparam-100M-lr4.8e3" in body
    assert "| 1.2e3 | 3.500000 | swa256-isoparam-100M-lr1.2e3" in body
    for point in points["swa256"]:
        assert f"{mf.format_lr(point.lr)} | " in body


def test_the_shared_and_tuned_columns_are_the_choice_losses(full):
    choices, _, text = full
    body = section(text, "gqa4")
    assert f"| loss | {choices['gqa4'].shared_loss:.6f} | {choices['gqa4'].tuned_loss:.6f} |" in body
    assert f"| rate | {mf.format_lr(REF)} | {mf.format_lr(2.4e-3)} |" in body


def test_the_sections_are_in_the_fixed_order(full):
    _, _, text = full
    headings = [line[3:] for line in text.splitlines() if line.startswith("## ")]
    assert headings == sel.names_in_order()
    assert headings[0] == "mha"
    assert headings[1:7] == [n for n in mt.HEADLINE_VARIANTS if n != "mha"]
    assert headings[7:] == config.ladder_names()


# --- The tie rule takes sigma_init from MHA's Arm N runs ----------------------------------------


def _tie_sweep(arm_n_spread: float, *, extra_variant_spread: float | None = None):
    manifests = mha_sweep(MHA_POINTS) + arm_n(arm_n_spread)
    if extra_variant_spread is not None:
        # A variant's own 100M branches, with a very different spread: they must not count.
        manifests += [
            fake_manifest("gqa4", init_seed=seed, peak_lr=REF, branch="100M",
                          val_loss=ARM_N_BASE + extra_variant_spread * (seed - 3))
            for seed in range(1, 6)
        ]
    manifests += [sweep_run("gqa4", lr, loss) for lr, loss in
                  {6e-4: 3.56, 1.2e-3: 3.503, 2.4e-3: 3.500, 4.8e-3: 3.60}.items()]
    return manifests


def test_a_gap_below_mha_sigma_is_a_tie_and_one_above_it_is_not(tmp_path):
    # The same sweep, the same 0.003 gap: only MHA's Arm N spread decides the rule.
    # Sample SD = spread * sqrt(0.625): about 0.0079 for 0.01 and about 0.00008 for 0.0001.
    wide = mt.load_manifests(write(_tie_sweep(0.01), tmp_path / "wide"))
    narrow = mt.load_manifests(write(_tie_sweep(0.0001), tmp_path / "narrow"))
    tie, _, _ = sel.build(wide, SWEEP_SEED)
    winner, _, _ = sel.build(narrow, SWEEP_SEED)
    assert tie["gqa4"].rule == sw.RULE_TIE and tie["gqa4"].lr == REF
    assert winner["gqa4"].rule == sw.RULE_WINNER and winner["gqa4"].lr == 2.4e-3


def test_sigma_comes_from_mha_only(tmp_path):
    # Wide spread in gqa4's own branches, narrow in MHA's: the narrow MHA spread decides.
    manifests = _tie_sweep(0.0001, extra_variant_spread=0.5)
    choices, _, _ = sel.build(mt.load_manifests(write(manifests, tmp_path / "m")), SWEEP_SEED)
    assert choices["gqa4"].rule == sw.RULE_WINNER and choices["gqa4"].lr == 2.4e-3


# --- Waiting states -----------------------------------------------------------------------------


def test_variants_wait_for_mha_when_mha_is_not_chosen(tmp_path):
    manifests = mha_sweep({3e-4: 3.60, 6e-4: 3.55}) + arm_n(0.01)
    manifests += [sweep_run("gqa4", 1.2e-3, 3.5)]
    out = tmp_path / "out"
    assert sel.main(["--manifests", str(write(manifests, tmp_path / "m")),
                     "--sweep-seed", str(SWEEP_SEED), "--out", str(out)]) == 0
    text = (out / "lr_sweep.md").read_text(encoding="utf-8")
    assert "**Status:** incomplete" in section(text, "mha")
    assert "waiting for MHA's sweep" in section(text, "gqa4")
    choices, _, _ = sel.build(mt.load_manifests(tmp_path / "m"), SWEEP_SEED)
    assert set(choices) == {sw.REFERENCE}


def test_variants_wait_for_the_noise_floor_when_arm_n_is_missing(tmp_path):
    manifests = mha_sweep(MHA_POINTS) + [sweep_run("gqa4", 1.2e-3, 3.5)]
    directory = write(manifests, tmp_path / "m")
    choices, _, notes = sel.build(mt.load_manifests(directory), SWEEP_SEED)
    assert choices[sw.REFERENCE].status == sw.CHOSEN
    assert "gqa4" not in choices
    assert notes  # the analysis says what is missing
    assert sel.main(["--manifests", str(directory), "--sweep-seed", str(SWEEP_SEED),
                     "--out", str(tmp_path / "out")]) == 0
    text = (tmp_path / "out" / "lr_sweep.md").read_text(encoding="utf-8")
    assert "waiting for the noise floor" in section(text, "gqa4")
    assert "**Status:** chosen" in section(text, "mha")


# --- Refusals -----------------------------------------------------------------------------------


def test_a_sweep_in_which_every_point_diverged_is_refused(tmp_path):
    manifests = mha_sweep({lr: None for lr in sw.MHA_GRID})
    directory = write(manifests, tmp_path / "m")
    out = tmp_path / "out"
    assert sel.main(["--manifests", str(directory), "--sweep-seed", str(SWEEP_SEED),
                     "--out", str(out)]) != 0
    assert not (out / "lr_sweep.md").exists()


def test_unreadable_manifests_are_refused(tmp_path):
    directory = tmp_path / "m"
    directory.mkdir()
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    assert sel.main(["--manifests", str(directory), "--sweep-seed", str(SWEEP_SEED),
                     "--out", str(tmp_path / "out")]) != 0


def test_a_sweep_seed_is_required_unless_blank(tmp_path):
    with pytest.raises(SystemExit) as refused:
        sel.main(["--out", str(tmp_path)])
    assert refused.value.code != 0


def test_a_trunk_checkpoint_in_the_manifests_does_not_refuse_the_report(tmp_path):
    trunk = fake_manifest("mha", init_seed=1, peak_lr=REF, kind="trunk_checkpoint", completed=False)
    manifests = full_sweep() + [trunk]
    choices, _, _ = sel.build(manifests, SWEEP_SEED)
    assert choices[sw.REFERENCE].lr == REF
