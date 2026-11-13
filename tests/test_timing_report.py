"""T-42: the timing and inference tables (scripts/timing_report.py), built from timing_pass/1 and bench/1.

Every document here is SYNTHETIC (tests/fixtures/fake_secondary.py), except the end-to-end test, which
runs the two scripts in their --synthetic modes on the CPU. The values are chosen so that the expected
cells can be computed by hand, and the tests compare rendered strings or exact floats: no tolerances.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import bench as bd
from scripts import timing_pass as tp
from scripts import timing_report as report
from tests.fixtures.fake_secondary import (
    ENVIRONMENT,
    bench_document,
    timing_document,
    write_bench_documents,
    write_timing_document,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_PREDICTION = "Prediction: MHA, GQA-4, GQA-2 and MQA within ±3% of each other."
BLANK_DIR = ROOT / "results" / "tables" / "blank"
MIB = 2**20
EXPECTED_TRACKS = [
    "mha", "gqa4", "mqa", "mla192", "mla192/ref", "gqa2", "mla96", "mla96/ref", "swa256", "swa256/ref",
]
JUDGEMENT_WORDS = re.compile(
    r"\b(pass|passes|passed|fail|fails|failed|ok|within|exceeds|exceed|above|below|holds|meets|"
    r"violates|verified|valid|refused)\b",
    re.IGNORECASE,
)


def hand_timing_doc() -> dict:
    """Repetitions 10, 11, 12 for MHA: mean 11.0, spread (12 - 10) / 11, every cell worked by hand."""
    doc = timing_document(ms={"mha": 11.0, "gqa4": 22.0, "gqa2": 11.0, "mqa": 11.0, "mla192": 22.0,
                              "mla96": 22.0, "swa256": 11.0})
    doc["variants"]["mha"].update(
        ms_per_step=11.0, ms_per_step_by_repetition=[10.0, 11.0, 12.0], attention_share=0.25, peak_bytes=100 * MIB
    )
    doc["variants"]["gqa4"].update(ms_per_step_by_repetition=[22.0, 22.0, 22.0], attention_share=0.5, peak_bytes=None)
    doc["variants"]["mla192/ref"].update(
        ms_per_step=33.0, ms_per_step_by_repetition=[33.0, 33.0, 33.0], attention_share=0.125, peak_bytes=3 * MIB
    )
    return doc


def hand_bench_doc(variant: str = "mha", mode: str = "opt") -> dict:
    """A bench document whose prefill, decode and KV cells are hand-chosen."""
    doc = bench_document(variant, mode, backend="efficient" if mode == "opt" else None, tok_per_s=12800.0)
    doc["prefill"] = [{"context": 1024, "batch": 1, "latency_ms": 4.0, "tok_per_s": 256000.0, "peak_bytes": 5 * MIB}]
    doc["decode"] = [{"batch": 32, "cache": 4096, "ms_per_token": 2.5, "tok_per_s": 12800.0, "peak_bytes": 7 * MIB}]
    doc["kv_bytes_per_token"] = {"analytic": 1536, "measured": 1920.0, "measured_tokens": 1024}
    doc["max_batch_at_8k"] = 40
    return doc


def judged(text: str) -> str:
    """The text as a judgement check reads it: the plan's noun "timing pass" (plan §8.5), or its id, is not a verdict."""
    text = text.replace(PLAN_PREDICTION, "")  # the plan's sentence, quoted (it contains the word "within")
    return re.sub(r"timing[ -]pass", "", text, flags=re.IGNORECASE)


def table_rows(block: str, first_header: str) -> list[list[str]]:
    """The data rows of the first table in `block` whose first header cell is `first_header`."""
    lines = block.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {first_header} |"))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def sections(text: str) -> list[tuple[str, str]]:
    """(heading, block) for each `## ` section of an inference report."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    return [(part.splitlines()[0], part) for part in parts]


def row_for(rows: list[dict], track: str) -> dict:
    return next(r for r in rows if r["track"] == track)


def run_report(tmp_path, *args) -> Path:
    out = tmp_path / "out"
    assert report.main([*args, "--out", str(out)]) == 0
    return out


def assert_refused(tmp_path, capsys, *args) -> str:
    """The report refuses: status 1, a message on stderr, and neither file written."""
    out = tmp_path / "out"
    assert report.main([*args, "--out", str(out)]) == 1
    assert not (out / report.TIMING_NAME).exists()
    assert not (out / report.INFERENCE_NAME).exists()
    message = capsys.readouterr().err
    assert message.strip(), "a refusal must say what is wrong"
    return message


def bench_dir(tmp_path, *docs) -> Path:
    directory = tmp_path / "bench"
    write_bench_documents(directory, list(docs))
    return directory


# --- timing.md: hand-computed cells ------------------------------------------------------------


def test_the_mha_row_has_every_cell_hand_computed():
    rows = report.timing_rows(hand_timing_doc())
    assert row_for(rows, "mha") == {
        "track": "mha",
        "ms/step, mean": "11.000",
        "ms/step, each repetition": "10.000, 11.000, 12.000",
        "spread over repetitions, % of mean": "18.18",
        "ratio to MHA opt (plan §8.5 predicts 0.97 to 1.03 for gqa4, gqa2 and mqa)": "1.000",
        "attention share, %": "25.0",
        "peak allocated, MiB": "100.0",
    }


def test_a_track_without_a_peak_memory_has_a_blank_peak_cell_and_its_spread_is_zero():
    rows = report.timing_rows(hand_timing_doc())
    assert row_for(rows, "gqa4") == {
        "track": "gqa4",
        "ms/step, mean": "22.000",
        "ms/step, each repetition": "22.000, 22.000, 22.000",
        "spread over repetitions, % of mean": "0.00",
        "ratio to MHA opt (plan §8.5 predicts 0.97 to 1.03 for gqa4, gqa2 and mqa)": "2.000",
        "attention share, %": "50.0",
        "peak allocated, MiB": "—",
    }


def test_a_ref_row_has_its_ratio_to_the_mha_opt_track_and_its_own_peak():
    rows = report.timing_rows(hand_timing_doc())
    assert row_for(rows, "mla192/ref") == {
        "track": "mla192/ref",
        "ms/step, mean": "33.000",
        "ms/step, each repetition": "33.000, 33.000, 33.000",
        "spread over repetitions, % of mean": "0.00",
        "ratio to MHA opt (plan §8.5 predicts 0.97 to 1.03 for gqa4, gqa2 and mqa)": "3.000",
        "attention share, %": "12.5",
        "peak allocated, MiB": "3.0",
    }


def test_rows_are_in_headline_order_with_each_ref_track_directly_under_its_opt_track():
    rows = report.timing_rows(hand_timing_doc())
    assert [r["track"] for r in rows] == [
        "mha", "gqa4", "mqa", "mla192", "mla192/ref", "gqa2", "mla96", "swa256",
    ]


def test_the_row_order_does_not_depend_on_the_order_of_the_document():
    doc = hand_timing_doc()
    doc["variants"] = dict(reversed(list(doc["variants"].items())))
    assert [r["track"] for r in report.timing_rows(doc)] == [
        "mha", "gqa4", "mqa", "mla192", "mla192/ref", "gqa2", "mla96", "swa256",
    ]


def test_the_ratio_is_blank_when_the_document_has_no_mha_opt_track():
    doc = hand_timing_doc()
    del doc["variants"]["mha"]
    rows = report.timing_rows(doc)
    assert row_for(rows, "gqa4")["ratio to MHA opt (plan §8.5 predicts 0.97 to 1.03 for gqa4, gqa2 and mqa)"] == "—"


def header_cells(text: str) -> list[str]:
    header = next(line for line in text.splitlines() if line.startswith("| track |"))
    return [cell.strip() for cell in header.strip().strip("|").split("|")]


def test_the_plan_three_percent_is_on_the_ratio_column_and_never_on_the_spread_column():
    cells = header_cells(report.render_timing(hand_timing_doc()))
    spread, ratio = cells[3], cells[4]
    assert spread == "spread over repetitions, % of mean"
    assert not any(ch.isdigit() for ch in spread), "no number from the plan may sit on the spread column"
    assert ratio.startswith("ratio to MHA opt (plan §8.5 predicts 0.97 to 1.03 for gqa4, gqa2 and mqa)")


def test_the_prediction_is_quoted_once_as_a_caption_that_points_at_the_ratio_column():
    text = report.render_timing(hand_timing_doc())
    assert text.count(PLAN_PREDICTION) == 1
    caption = next(line for line in text.splitlines() if PLAN_PREDICTION in line)
    assert "ratio to MHA opt column" in caption
    spread_caption = next(line for line in text.splitlines() if line.startswith("Spread is"))
    assert spread_caption == "Spread is (max - min) / mean over the track's own repetitions, in percent."


def test_the_timing_file_has_its_header_source_line_and_captions():
    text = report.render_timing(hand_timing_doc())
    assert text.splitlines()[0] == "# Training timing (plan §8.5)"
    assert (
        "| track | ms/step, mean | ms/step, each repetition | spread over repetitions, % of mean | "
        "ratio to MHA opt (plan §8.5 predicts 0.97 to 1.03 for gqa4, gqa2 and mqa) | attention share, % | "
        "peak allocated, MiB |"
    ) in text
    assert "Source: GPU Synthetic T4; backend efficient; torch 2.15.0; CUDA 13.2; driver 999.0" in text
    assert "timing pass SYNTHETIC-timing-pass; 220 steps, first 20 discarded, 3 repetitions." in text
    assert "naive path for MLA" in text
    assert "partly implementation" in text
    assert "no reserved figure" in text


# --- inference.md: hand-computed cells ----------------------------------------------------------


def test_a_prefill_row_has_every_cell_hand_computed():
    section = report.inference_tables([hand_bench_doc()])[0]
    assert section.prefill == [{
        "context": "1024", "batch": "1", "latency, ms": "4.000", "tokens/s": "256000.0",
        "peak allocated, MiB": "5.0",
    }]


def test_a_decode_row_has_every_cell_hand_computed():
    section = report.inference_tables([hand_bench_doc()])[0]
    assert section.decode == [{
        "batch": "32", "cache": "4096", "ms/token": "2.500", "tokens/s": "12800.0", "peak allocated, MiB": "7.0",
    }]


def test_the_kv_line_has_every_cell_hand_computed():
    section = report.inference_tables([hand_bench_doc()])[0]
    assert section.kv == {
        "analytic bytes/token": "1536.0",
        "measured bytes/token": "1920.0",
        "measured over tokens": "1024",
        "measured / analytic (limit 5%, plan §10.4)": "1.250",
        "max batch at cache 8k": "40",
    }


def test_the_rendered_rows_are_the_hand_computed_lines():
    text = report.render_inference([hand_bench_doc()])
    assert "| 1024 | 1 | 4.000 | 256000.0 | 5.0 |" in text
    assert "| 32 | 4096 | 2.500 | 12800.0 | 7.0 |" in text
    assert "| 1536.0 | 1920.0 | 1024 | 1.250 | 40 |" in text


def test_a_null_cell_of_a_document_is_blank():
    doc = hand_bench_doc()
    doc["prefill"][0]["peak_bytes"] = None
    doc["kv_bytes_per_token"] = {"analytic": 1536, "measured": None, "measured_tokens": None}
    doc["max_batch_at_8k"] = None
    text = report.render_inference([doc])
    assert "| 1024 | 1 | 4.000 | 256000.0 | — |" in text
    assert "| 1536.0 | — | — | — | — |" in text


def test_sections_are_headline_order_with_opt_before_ref():
    docs = [hand_bench_doc("mla192", "ref"), hand_bench_doc("mha", "ref"), hand_bench_doc("mla192", "opt"),
            hand_bench_doc("mha", "opt")]
    headings = [(s.variant, s.mode) for s in report.inference_tables(docs)]
    assert headings == [("mha", "opt"), ("mha", "ref"), ("mla192", "opt"), ("mla192", "ref")]


def test_the_inference_file_has_its_captions_and_the_precision_source_line():
    text = report.render_inference([hand_bench_doc()])
    assert text.splitlines()[0] == "# Inference benchmark (plan §8.6, §10.4)"
    assert "Source: precision fp16; backend efficient; GPU Synthetic T4; torch 2.15.0; CUDA 13.2; driver 999.0." in text
    assert "limit between them at 5%" in text
    assert "| analytic bytes/token | measured bytes/token | measured over tokens | measured / analytic (limit 5%, plan §10.4) | max batch at cache 8k |" in text


# --- Blank forms --------------------------------------------------------------------------------


def test_the_blank_timing_form_equals_the_committed_file_byte_for_byte():
    assert report.render_timing(None) == (BLANK_DIR / "timing.md").read_text(encoding="utf-8")


def test_the_blank_inference_form_equals_the_committed_file_byte_for_byte():
    assert report.render_inference(None) == (BLANK_DIR / "inference.md").read_text(encoding="utf-8")


def test_the_blank_command_line_writes_the_committed_forms(tmp_path):
    out = run_report(tmp_path, "--blank")
    assert (out / "timing.md").read_text(encoding="utf-8") == (BLANK_DIR / "timing.md").read_text(encoding="utf-8")
    assert (out / "inference.md").read_text(encoding="utf-8") == (BLANK_DIR / "inference.md").read_text(encoding="utf-8")


def test_the_blank_timing_form_has_one_row_per_expected_track_and_every_measured_cell_blank():
    rows = table_rows((BLANK_DIR / "timing.md").read_text(encoding="utf-8"), "track")
    assert [r[0] for r in rows] == EXPECTED_TRACKS
    assert all(cell == "—" for r in rows for cell in r[1:])


def test_the_blank_inference_form_has_every_variant_and_mode_with_blank_measured_cells():
    text = (BLANK_DIR / "inference.md").read_text(encoding="utf-8")
    heads = [heading for heading, _ in sections(text)]
    expected = [f"`{variant}`, {mode} mode" for variant in report.HEADLINE for mode in ("opt", "ref")]
    assert heads == expected
    for _, block in sections(text):
        prefill = table_rows(block, "context")
        decode = table_rows(block, "batch")
        kv = table_rows(block, "analytic bytes/token")
        assert [r[0] for r in prefill] == ["128", "256", "512", "1024", "2048", "4096", "8192"]
        assert [(r[0], r[1]) for r in decode] == [
            (str(b), str(c)) for b in bd.DEFAULT_DECODE_BATCHES for c in bd.DEFAULT_DECODE_CACHES
        ]
        assert all(cell == "—" for r in prefill for cell in r[1:])
        assert all(cell == "—" for r in decode for cell in r[2:])
        assert len(kv) == 1 and all(cell == "—" for cell in kv[0])


# --- Refusals ------------------------------------------------------------------------------------


def test_another_timing_schema_is_refused(tmp_path, capsys):
    doc = hand_timing_doc()
    doc["schema"] = "timing_pass/0"
    path = tmp_path / "timing.json"
    write_timing_document(path, doc)
    assert "timing_pass/0" in assert_refused(tmp_path, capsys, "--timing-pass", str(path))


def test_a_track_of_a_variant_the_project_does_not_know_is_refused(tmp_path, capsys):
    doc = timing_document(ms={"mha": 11.0, "foo": 11.0})
    path = tmp_path / "timing.json"
    write_timing_document(path, doc)
    assert "'foo'" in assert_refused(tmp_path, capsys, "--timing-pass", str(path))


def test_the_null_variant_is_refused_as_a_track(tmp_path, capsys):
    doc = timing_document(ms={"mha": 11.0, "null": 11.0})
    path = tmp_path / "timing.json"
    write_timing_document(path, doc)
    assert "'null'" in assert_refused(tmp_path, capsys, "--timing-pass", str(path))


def test_a_ref_track_for_a_variant_without_one_is_refused(tmp_path, capsys):
    doc = timing_document(ms={"mha": 11.0})
    doc["variants"]["mha/ref"] = {"mode": "ref", "ms_per_step": 11.0, "ms_per_step_by_repetition": [11.0],
                                  "attention_share": 0.25, "peak_bytes": None}
    path = tmp_path / "timing.json"
    write_timing_document(path, doc)
    assert "mha/ref" in assert_refused(tmp_path, capsys, "--timing-pass", str(path))


def test_a_track_whose_mode_disagrees_with_its_key_is_refused(tmp_path, capsys):
    doc = timing_document(ms={"mha": 11.0})
    doc["variants"]["mha"]["mode"] = "ref"
    path = tmp_path / "timing.json"
    write_timing_document(path, doc)
    assert "expected 'opt'" in assert_refused(tmp_path, capsys, "--timing-pass", str(path))


def test_a_timing_document_recorded_on_a_shared_gpu_is_refused(tmp_path, capsys):
    path = tmp_path / "timing.json"
    write_timing_document(path, timing_document(shared=True))
    assert "gpu_shared" in assert_refused(tmp_path, capsys, "--timing-pass", str(path))


def test_a_bench_document_recorded_on_a_shared_gpu_is_refused(tmp_path, capsys):
    directory = bench_dir(tmp_path, bench_document("mha", shared=True))
    assert "gpu_shared" in assert_refused(tmp_path, capsys, "--bench", str(directory))


def test_timing_and_bench_documents_that_disagree_on_torch_are_refused(tmp_path, capsys):
    path = tmp_path / "timing.json"
    write_timing_document(path, timing_document(environment={**ENVIRONMENT, "torch": "2.14.0"}))
    directory = bench_dir(tmp_path, bench_document("mha"))
    assert "torch version differs" in assert_refused(tmp_path, capsys, "--timing-pass", str(path), "--bench", str(directory))


def test_timing_and_bench_documents_that_disagree_on_backend_are_refused(tmp_path, capsys):
    path = tmp_path / "timing.json"
    write_timing_document(path, timing_document(backend="flash"))
    directory = bench_dir(tmp_path, bench_document("mha", backend="efficient"))
    assert "backend differs" in assert_refused(tmp_path, capsys, "--timing-pass", str(path), "--bench", str(directory))


def test_a_bench_document_with_another_schema_is_refused(tmp_path, capsys):
    doc = bench_document("mha")
    doc["schema"] = "bench/0"
    directory = bench_dir(tmp_path, doc)
    assert "bench/0" in assert_refused(tmp_path, capsys, "--bench", str(directory))


def test_a_bench_variant_the_project_does_not_know_is_refused(tmp_path, capsys):
    directory = bench_dir(tmp_path, bench_document("foo", tok_per_s=1.0))
    assert "'foo'" in assert_refused(tmp_path, capsys, "--bench", str(directory))


def test_a_bench_mode_other_than_opt_or_ref_is_refused(tmp_path, capsys):
    directory = bench_dir(tmp_path, bench_document("mha", "train"))
    assert "'train'" in assert_refused(tmp_path, capsys, "--bench", str(directory))


def test_two_bench_documents_for_one_variant_and_mode_are_refused(tmp_path, capsys):
    directory = bench_dir(tmp_path, bench_document("mha"))
    (directory / "bench_mha_opt_copy.json").write_text(
        json.dumps(bench_document("mha"), indent=2) + "\n", encoding="utf-8"
    )
    assert "two bench documents for mha/opt" in assert_refused(tmp_path, capsys, "--bench", str(directory))


def test_an_empty_bench_directory_is_refused(tmp_path, capsys):
    directory = tmp_path / "bench"
    directory.mkdir()
    assert "no bench documents" in assert_refused(tmp_path, capsys, "--bench", str(directory))


def test_the_render_functions_refuse_what_the_command_line_refuses():
    doc = hand_timing_doc()
    doc["variants"]["foo"] = dict(doc["variants"]["mha"])
    with pytest.raises(report.TimingReportError, match="'foo'"):
        report.render_timing(doc)
    with pytest.raises(report.TimingReportError, match="two bench documents for mha/opt"):
        report.render_inference([hand_bench_doc(), hand_bench_doc()])


# --- No verdict ----------------------------------------------------------------------------------


def test_no_judgement_word_appears_in_the_filled_or_blank_files(tmp_path):
    texts = [
        report.render_timing(hand_timing_doc()),
        report.render_inference([hand_bench_doc(), hand_bench_doc("mla192", "ref")]),
        report.render_timing(None),
        report.render_inference(None),
    ]
    for text in texts:
        found = JUDGEMENT_WORDS.search(judged(text))
        assert found is None, found.group(0) if found else None


# --- Module surface -------------------------------------------------------------------------------


def test_the_module_imports_without_matplotlib():
    code = "import sys, scripts.timing_report; assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_the_expected_tracks_are_the_timing_tracks_of_a_pass_with_ref_tracks():
    expected = report.expected_timing_tracks()
    assert expected == EXPECTED_TRACKS
    assert set(tp.REF_TRACK_VARIANTS) == {"mla192", "mla96", "swa256"}


# --- End to end: the real scripts, in their synthetic modes -----------------------------------


def test_both_scripts_in_synthetic_mode_feed_both_tables(tmp_path):
    timing_path = tmp_path / "timing" / "timing_pass.json"
    assert tp.main([
        "--synthetic", "--peak-lr", "1e-3", "--steps", "4", "--discarded", "1", "--repetitions", "2",
        "--variants", "mha", "mla192", "swa256", "--with-ref", "--out", str(timing_path),
    ]) == 0
    bench = tmp_path / "bench"
    assert bd.main([
        "--synthetic", "--variants", "mha", "mla192", "--with-ref", "--contexts", "16", "32",
        "--decode-batches", "1", "--decode-caches", "8", "--warmup", "1", "--reps", "1",
        "--skip-max-batch", "--out-dir", str(bench),
    ]) == 0

    out = run_report(tmp_path, "--timing-pass", str(timing_path), "--bench", str(bench))
    timing_doc = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_text = (out / report.TIMING_NAME).read_text(encoding="utf-8")
    rows = table_rows(timing_text, "track")
    assert [r[0] for r in rows] == [
        "mha", "mla192", "mla192/ref", "swa256", "swa256/ref",
    ]
    assert sorted(timing_doc["variants"]) == sorted(r[0] for r in rows)
    assert rows[0][4] == "1.000"

    inference_text = (out / report.INFERENCE_NAME).read_text(encoding="utf-8")
    headings = [heading for heading, _ in sections(inference_text)]
    assert headings == ["`mha`, opt mode", "`mha`, ref mode", "`mla192`, opt mode", "`mla192`, ref mode"]
    for _, block in sections(inference_text):
        assert len(table_rows(block, "context")) == 2  # contexts 16 and 32
        assert len(table_rows(block, "batch")) == 1  # one decode point: batch 1, cache 8

    assert JUDGEMENT_WORDS.search(judged(timing_text)) is None
    assert JUDGEMENT_WORDS.search(judged(inference_text)) is None
