"""T-48: SWA not below MHA is recorded, not refused (plan §2.3 item 2, §13.8, decision 226).

Every timing document here is SYNTHETIC (tests/fixtures/fake_secondary.py). The check does not
fail the timing pass: the pass returns 0 on CPU and on CUDA, and the three report files mark SWA's
timings and say that SWA is not faster than a dense kernel on this GPU, only when it is not below MHA's.
The expected cells are computed by hand in the comments, and compared exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import rqw_estimate as rq
from scripts import make_tables as mt
from scripts import timing_pass as tp
from scripts import timing_report as report
from src.utils import manifest as mf
from tests.fixtures.commitments import analyse_committed, tables_main
from tests.fixtures.fake_secondary import (
    TIMING_MS,
    arm_b_manifests,
    secondary_campaign,
    timing_document,
    with_context,
    write_timing_document,
)

ROOT = Path(__file__).resolve().parents[1]
BLANK_DIR = ROOT / "results" / "tables" / "blank"
PHRASE = "not faster than a dense kernel on this GPU"
MARK = tp.SWA_DENSE_MARK

# Attention time per step is attention_share x ms_per_step: MHA 0.25 x 100.0 = 25.0.
MHA = (0.25, 100.0)
HOLDS_SWA = (0.10, 97.0)  # 0.10 x 97.0 = 9.7, below 25.0
NOT_HOLDS_SWA = (0.30, 97.0)  # 0.30 x 97.0 = 29.1, not below 25.0
EQUAL_SWA = (0.25, 100.0)  # 25.0, equal: not below


def swa_doc(*, mha=MHA, swa=HOLDS_SWA, drop: tuple[str, ...] = ()) -> dict:
    """A synthetic timing_pass/1 document: every variant's opt entry (the RQ-W estimate needs all of them),
    MHA's and SWA's set by the arguments, and SWA's and MLA's ref entries."""
    ms = dict(TIMING_MS)
    ms["mha"], ms["swa256"] = mha[1], swa[1]
    doc = timing_document(ms=ms)
    doc["variants"]["mha"]["attention_share"] = mha[0]
    doc["variants"]["swa256"]["attention_share"] = swa[0]
    doc["variants"]["swa256/ref"] = {
        "mode": "ref", "ms_per_step": 200.0, "ms_per_step_by_repetition": [200.0],
        "attention_share": 0.1, "peak_bytes": None,
    }
    for key in drop:
        del doc["variants"][key]
    doc["order"] = list(doc["variants"])
    return doc


NOT_HOLDS_NOTE = tp.swa_dense_note(tp.swa_check_of(swa_doc(swa=NOT_HOLDS_SWA)))  # "SWA is not faster ... 29.100 ... 25.000"
NOT_HOLDS_LINE = f"{MARK} {NOT_HOLDS_NOTE}"


def _cells(text: str, track: str) -> list[str]:
    """The cells of the first table row whose first cell is `track`."""
    for line in text.splitlines():
        if line.startswith(f"| {track} |"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"no row {track!r} in\n{text}")


def _comparison(swa: str, mha: str) -> str:
    return f"SWA attention time: {swa} ms/step; MHA: {mha} ms/step (attention share × ms/step, plan §2.3 item 2)."


def _write(directory: Path, doc: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "timing_pass.json"
    write_timing_document(path, doc)
    return path


# --- The check -----------------------------------------------------------------------------------


def test_the_check_holds_when_swa_attention_is_smaller():
    check = tp.swa_check_of(swa_doc())
    assert check.holds is True
    assert check.mha_attention_ms == 0.25 * 100.0
    assert check.swa_attention_ms == 0.10 * 97.0
    assert tp.swa_not_faster(check) is False


def test_the_check_does_not_hold_when_swa_attention_is_larger():
    check = tp.swa_check_of(swa_doc(swa=NOT_HOLDS_SWA))
    assert check.holds is False
    assert check.swa_attention_ms == 0.30 * 97.0
    assert tp.swa_not_faster(check) is True


def test_equal_attention_time_is_not_below_so_it_does_not_hold():
    check = tp.swa_check_of(swa_doc(swa=EQUAL_SWA))
    assert check.holds is False
    assert check.swa_attention_ms == check.mha_attention_ms
    assert tp.swa_not_faster(check) is True


def test_the_check_is_none_without_the_mha_or_the_swa_opt_entry():
    assert tp.swa_check_of(swa_doc(drop=("mha",))) is None
    assert tp.swa_check_of(swa_doc(drop=("swa256",))) is None
    assert tp.swa_check_of({}) is None
    assert tp.swa_not_faster(None) is False


def test_the_check_takes_a_document_or_its_variants_mapping():
    doc = swa_doc(swa=NOT_HOLDS_SWA)
    assert tp.swa_check_of(doc) == tp.swa_check_of(doc["variants"])


def test_the_existing_assertion_helpers_are_unchanged():
    with pytest.raises(AssertionError, match="not below MHA"):
        tp.require_swa_below_mha(swa_doc(swa=NOT_HOLDS_SWA)["variants"])
    assert tp.require_swa_below_mha(swa_doc()["variants"]).holds is True


def test_the_note_names_the_phrase_the_plan_sections_and_both_times():
    check = tp.swa_check_of(swa_doc(swa=NOT_HOLDS_SWA))
    assert PHRASE in NOT_HOLDS_NOTE
    assert "(plan §2.3 item 2, §13.8)" in NOT_HOLDS_NOTE
    assert f"{check.swa_attention_ms:.3f} ms/step" in NOT_HOLDS_NOTE
    assert f"{check.mha_attention_ms:.3f} ms/step" in NOT_HOLDS_NOTE


# --- The timing pass command line: exit status 0 on CPU and on CUDA ----------------------------

CPU_ARGS: list[str] = []
CUDA_ARGS = ["--device", "cuda", "--gpu-shared", "false", "--expected-backend", "efficient"]


def _pass_args(out: Path, device_args: list[str]) -> list[str]:
    return [
        "--synthetic", "--peak-lr", "1e-3", "--variants", "mha", "swa256", "--steps", "4",
        "--discarded", "1", "--repetitions", "2", *device_args, "--out", str(out),
    ]


@pytest.mark.parametrize("device_args", [CPU_ARGS, CUDA_ARGS], ids=["cpu", "cuda"])
@pytest.mark.parametrize("swa", [HOLDS_SWA, NOT_HOLDS_SWA], ids=["holds", "not-holds"])
def test_the_pass_returns_zero_on_cpu_and_cuda_whether_or_not_swa_is_below(
    tmp_path, monkeypatch, capsys, device_args, swa
):
    # The measurement is replaced by a synthetic document: this tests the exit status and the messages.
    document = swa_doc(swa=swa)
    monkeypatch.setattr(tp, "run_timing_pass", lambda setup, **kwargs: document)
    out = tmp_path / "timing_pass.json"
    assert tp.main(_pass_args(out, device_args)) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["variants"] == document["variants"]
    captured = capsys.readouterr()
    assert "SWA attention" in captured.out
    if swa == NOT_HOLDS_SWA:
        assert "NOT below" in captured.out
        assert NOT_HOLDS_LINE in captured.err
        assert "decision 226" in captured.err
        assert "not a valid timing result" not in captured.err
    else:
        assert "below (holds)" in captured.out
        assert captured.err == ""


def test_the_pass_document_keeps_the_schema_and_gains_no_key(tmp_path, monkeypatch):
    document = swa_doc(swa=NOT_HOLDS_SWA)
    monkeypatch.setattr(tp, "run_timing_pass", lambda setup, **kwargs: document)
    out = tmp_path / "timing_pass.json"
    assert tp.main(_pass_args(out, CPU_ARGS)) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == tp.SCHEMA == "timing_pass/1"
    assert set(written) == set(document)


# --- timing.md ----------------------------------------------------------------------------------


def test_timing_md_states_both_attention_times_whether_or_not_swa_is_below(tmp_path):
    for case, swa_text in ((HOLDS_SWA, "9.700"), (NOT_HOLDS_SWA, "29.100"), (EQUAL_SWA, "25.000")):
        out = tmp_path / swa_text
        path = _write(tmp_path / f"doc_{swa_text}", swa_doc(swa=case))
        assert report.main(["--timing-pass", str(path), "--out", str(out)]) == 0
        text = (out / report.TIMING_NAME).read_text(encoding="utf-8")
        assert _comparison(swa_text, "25.000") in text


def test_timing_md_marks_only_the_swa_opt_row_and_adds_the_note_when_not_below():
    text = report.render_timing(swa_doc(swa=NOT_HOLDS_SWA))
    ms_col = report.TIMING_COLUMNS.index("ms/step, mean")
    ratio_col = report.TIMING_COLUMNS.index(report.RATIO_COLUMN)
    swa_row = _cells(text, "swa256")
    assert swa_row[ms_col] == "97.000" + MARK
    assert swa_row[ratio_col] == "0.970" + MARK
    # The /ref row of the same variant is not marked, nor is MHA.
    assert MARK not in " ".join(_cells(text, "swa256/ref"))
    assert _cells(text, "mha")[ms_col] == "100.000"
    # Three marks in all: the two cells of SWA's opt row, and the note line under the table.
    assert text.count(MARK) == 3
    assert NOT_HOLDS_LINE in text.splitlines()
    assert text.count(PHRASE) == 1


def test_timing_md_has_no_mark_and_no_note_when_swa_is_below():
    text = report.render_timing(swa_doc(swa=HOLDS_SWA))
    ms_col = report.TIMING_COLUMNS.index("ms/step, mean")
    assert _cells(text, "swa256")[ms_col] == "97.000"
    assert MARK not in text and PHRASE not in text
    assert _comparison("9.700", "25.000") in text


def test_timing_md_has_the_comparison_line_directly_under_the_table():
    lines = report.render_timing(swa_doc()).splitlines()
    index = lines.index(_comparison("9.700", "25.000"))
    assert lines[index - 1] == ""
    assert lines[index - 2].endswith("|")  # the last table row


def test_timing_md_has_no_comparison_and_no_mark_without_the_swa_or_mha_entry():
    for drop in (("mha",), ("swa256",)):
        text = report.render_timing(swa_doc(swa=NOT_HOLDS_SWA, drop=drop))
        assert "SWA attention time" not in text
        assert MARK not in text and PHRASE not in text


def test_a_document_without_the_swa_or_the_mha_entry_is_written_without_error(tmp_path):
    for drop in (("mha",), ("swa256",)):
        out = tmp_path / drop[0]
        path = _write(tmp_path / f"doc_{drop[0]}", swa_doc(swa=NOT_HOLDS_SWA, drop=drop))
        assert report.main(["--timing-pass", str(path), "--out", str(out)]) == 0
        assert MARK not in (out / report.TIMING_NAME).read_text(encoding="utf-8")


def test_the_blank_timing_form_has_the_comparison_line_with_blanks_and_no_mark():
    text = report.render_timing(None)
    assert _comparison("—", "—") in text
    assert MARK not in text and PHRASE not in text


def test_the_committed_blank_timing_md_is_the_generated_blank_form():
    assert (BLANK_DIR / report.TIMING_NAME).read_text(encoding="utf-8") == report.render_timing(None)


# --- headline.md and rqw.md ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def campaign():
    base = with_context(secondary_campaign(gaps={"mqa": 0.05}))
    manifests = base + arm_b_manifests(base)
    return manifests, mt.select_results(manifests)[0], analyse_committed(manifests)


def _manifests(tmp_path: Path, manifests) -> Path:
    directory = tmp_path / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    for manifest in manifests:
        mf.write_manifest(manifest, directory)
    return directory


def _run_tables(tmp_path: Path, manifests, timing: Path) -> Path:
    directory = _manifests(tmp_path, manifests)
    out = tmp_path / "tables"
    assert tables_main(["--manifests", str(directory), "--out", str(out), "--timing-pass", str(timing)]) == 0
    return out


def _ms(path: Path) -> dict:
    return tp.read_ms_per_step(path)


def test_headline_and_rqw_are_marked_and_noted_when_swa_is_not_below(tmp_path, campaign):
    manifests, _, _ = campaign
    out = _run_tables(tmp_path, manifests, _write(tmp_path / "timing", swa_doc(swa=NOT_HOLDS_SWA)))

    headline = (out / "headline.md").read_text(encoding="utf-8")
    train_col = mt.COLUMNS.index("train ms/step")
    assert _cells(headline, "swa256")[train_col] == "97.0" + MARK
    assert _cells(headline, "mha")[train_col] == "100.0"
    assert _cells(headline, "gqa4")[train_col] == f"{TIMING_MS['gqa4']:.1f}"
    assert headline.count(f"- {NOT_HOLDS_LINE}") == 1
    assert headline.count(PHRASE) == 1

    rqw = (out / "rqw.md").read_text(encoding="utf-8")
    ms_col = mt.RQW_COLUMNS.index("ms/step")
    assert _cells(rqw, "swa256")[ms_col] == "97.000" + MARK
    assert NOT_HOLDS_LINE in rqw.splitlines()
    assert rqw.count(PHRASE) == 1


def test_the_marked_tables_differ_from_the_unmarked_ones_only_by_the_mark_and_the_note(tmp_path, campaign):
    manifests, _, _ = campaign
    marked = _run_tables(tmp_path / "marked", manifests, _write(tmp_path / "m_timing", swa_doc(swa=NOT_HOLDS_SWA)))
    # The same ms/step values, with SWA's attention share changed so that the check holds: nothing else moves.
    plain = _run_tables(tmp_path / "plain", manifests, _write(tmp_path / "p_timing", swa_doc(swa=HOLDS_SWA)))

    marked_rqw = (marked / "rqw.md").read_text(encoding="utf-8")
    plain_rqw = (plain / "rqw.md").read_text(encoding="utf-8")
    assert MARK not in plain_rqw and PHRASE not in plain_rqw
    assert marked_rqw.replace("\n\n" + NOT_HOLDS_LINE + "\n", "\n").replace(MARK, "") == plain_rqw

    marked_headline = (marked / "headline.md").read_text(encoding="utf-8")
    plain_headline = (plain / "headline.md").read_text(encoding="utf-8")
    assert MARK not in plain_headline and PHRASE not in plain_headline
    # Only the table and the notes differ; the disclosures that follow are the same text.
    assert marked_headline.split("\nDisclosures:")[1] == plain_headline.split("\nDisclosures:")[1]
    assert marked_headline.replace(MARK, "").replace(f"- {NOT_HOLDS_LINE}\n", "").split("\nNotes:")[0] == plain_headline.split("\nNotes:")[0]


def test_holding_swa_gives_the_unmarked_tables_byte_for_byte(tmp_path, campaign):
    manifests, results, analysis = campaign
    timing = _write(tmp_path / "timing", swa_doc())
    out = _run_tables(tmp_path, manifests, timing)
    headline = (out / "headline.md").read_text(encoding="utf-8")
    rqw = (out / "rqw.md").read_text(encoding="utf-8")
    assert MARK not in headline and MARK not in rqw
    assert PHRASE not in headline and PHRASE not in rqw
    assert headline == mt.render_markdown(
        mt.headline_rows(analysis, train_ms_per_step=_ms(timing)), analysis.notes, results=results,
    )
    assert rqw == mt.render_rqw(rq.estimate(list(results), _ms(timing)), None)


def test_without_a_timing_document_no_table_is_marked(tmp_path, campaign):
    manifests, _, _ = campaign
    directory = _manifests(tmp_path, manifests)
    out = tmp_path / "tables"
    assert tables_main(["--manifests", str(directory), "--out", str(out)]) == 0
    for name in ("headline.md", "rqw.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert MARK not in text and PHRASE not in text


def test_a_timing_document_without_the_mha_entry_marks_nothing_and_does_not_fail(tmp_path, campaign):
    manifests, _, _ = campaign
    out = _run_tables(tmp_path, manifests, _write(tmp_path / "timing", swa_doc(swa=NOT_HOLDS_SWA, drop=("mha",))))
    for name in ("headline.md", "rqw.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert MARK not in text and PHRASE not in text


def test_the_committed_blank_rqw_and_timing_carry_no_mark():
    assert MARK not in (BLANK_DIR / report.TIMING_NAME).read_text(encoding="utf-8")
    assert (BLANK_DIR / "rqw.md").read_text(encoding="utf-8") == mt.render_rqw(None, None)
