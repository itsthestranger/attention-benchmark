"""T-38: the synthetic-probe report (plan §9.2). Every record here is hand-built and SYNTHETIC.

The values are chosen so that the expected numbers can be computed by hand (0.5, 1.0, 0.25, ...),
and the tests compare exact floats or rendered strings: there are no numerical tolerances.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import probe_report as pr

ROOT = Path(__file__).resolve().parents[1]
BLANK_REPORT = ROOT / "results" / "tables" / "blank" / "probes.md"
JUDGEMENT_WORDS = re.compile(r"\b(pass|passes|passed|fail|fails|failed|detected|solved)\b", re.IGNORECASE)


def environment(**overrides) -> dict:
    env = {"python": "3.13.0", "torch": "2.15.0", "cuda": None, "driver": None, "gpu_name": None,
           "compute_capability": None}
    env.update(overrides)
    return env


def record(**overrides) -> dict:
    """A complete, valid probe/1 record. Override what a test needs."""
    base = {
        "schema": "probe/1", "variant": "mha", "task": "mqar", "generator": "mqar", "difficulty": 4,
        "init_seed": 1, "accuracy": 0.5, "sequence_accuracy": 0.25, "chance": 0.25, "n_scored": 64,
        "n_sequences": 16, "spec": {"seq_len": 64, "vocab_size": 128, "params": {"pairs": 4}},
        "d_model": 32, "n_layers": 2, "data_seed": 7, "steps": 100, "steps_taken": 100, "skipped_steps": 0,
        "peak_lr": 0.001, "sequences_per_step": 8, "eval_sequences": 16, "final_train_loss": 0.5,
        "config_hash": "abc123", "mode": "opt", "precision": "fp32", "backend": None, "device": "cpu",
        "environment": environment(), "git_sha": "deadbeef",
    }
    base.update(overrides)
    return base


def refused(records) -> str:
    with pytest.raises(pr.ProbeReportError) as info:
        pr.check(records)
    return str(info.value)


# --- check: one test per refusal ---------------------------------------------------------------


def test_a_record_lacking_a_key_is_refused_and_named():
    bad = record()
    del bad["config_hash"]
    message = refused([record(), bad])
    assert "lacks" in message and "config_hash" in message
    assert "record 1" in message
    assert "record 0" not in message


def test_a_record_that_is_not_an_object_is_refused():
    message = refused([record(), "not a record"])
    assert "record 1 is not an object" in message


@pytest.mark.parametrize("schema", ["probe/0", "bench/1", None])
def test_a_wrong_schema_is_refused_and_named(schema):
    message = refused([record(schema=schema)])
    assert "record 0" in message and "schema" in message and repr(schema) in message


@pytest.mark.parametrize("key", ["accuracy", "sequence_accuracy"])
@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), float("inf"), True, "0.5"])
def test_an_accuracy_outside_the_unit_interval_or_non_finite_is_refused(key, value):
    message = refused([record(), record(init_seed=2, **{key: value})])
    assert "record 1" in message and key in message and "record 0" not in message


@pytest.mark.parametrize("difficulty", [0, -3, 2.0, True])
def test_a_difficulty_below_one_or_not_an_integer_is_refused(difficulty):
    message = refused([record(difficulty=difficulty)])
    assert "record 0" in message and "difficulty" in message


# Each field is refused alone at its point, so that only the field's own check can refuse it.
MALFORMED_FIELDS = [
    ("init_seed", "3"),
    ("d_model", 2.0),
    ("n_layers", "2"),
    ("steps", None),
    ("steps_taken", "100"),
    ("skipped_steps", "3"),
    ("peak_lr", "0.001"),
]


@pytest.mark.parametrize("field, value", MALFORMED_FIELDS)
def test_a_malformed_numeric_field_is_refused_and_named(field, value):
    message = refused([record(difficulty=9, **{field: value})])
    assert "record 0" in message and field in message


@pytest.mark.parametrize("field", ["init_seed", "d_model", "n_layers", "steps", "steps_taken", "skipped_steps"])
@pytest.mark.parametrize("value", [True, -1])
def test_a_count_field_that_is_a_boolean_or_negative_is_refused(field, value):
    message = refused([record(difficulty=9, **{field: value})])
    assert "record 0" in message and field in message and "non-negative integer" in message


@pytest.mark.parametrize("value", [0.0, -0.001, float("nan"), float("inf"), True])
def test_a_peak_lr_that_is_not_a_finite_positive_number_is_refused(value):
    message = refused([record(difficulty=9, peak_lr=value)])
    assert "record 0" in message and "peak_lr" in message and "above zero" in message


def test_the_command_line_refuses_a_malformed_field_without_a_traceback(tmp_path, capsys):
    probes = _write(tmp_path / "probes", [record(difficulty=9, skipped_steps="3")])
    out = tmp_path / "tables"
    assert pr.main(["--probes", str(probes), "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "record 0" in err and "skipped_steps" in err
    assert not out.exists()


@pytest.mark.parametrize("chance", [0.0, -0.25, 1.5, float("nan")])
def test_a_chance_outside_zero_to_one_is_refused(chance):
    # Alone at its point, so that only the range check can refuse it.
    message = refused([record(difficulty=7, chance=chance)])
    assert "record 0" in message and "chance" in message
    assert "expected None or a number in (0, 1]" in message
    assert "differs" not in message


def test_a_chance_of_none_and_a_chance_of_one_are_accepted():
    pr.check([record(chance=None, task="selective_copy", difficulty=8), record(chance=1.0, difficulty=1)])


@pytest.mark.parametrize("variant", ["gqa8", "MHA", "", None])
def test_a_variant_outside_the_list_is_refused_and_named(variant):
    message = refused([record(), record(variant=variant, init_seed=2)])
    assert "record 1" in message and "variant" in message and repr(variant) in message


def test_two_records_with_the_same_key_are_refused_and_both_named():
    message = refused([record(init_seed=3), record(init_seed=3, accuracy=0.75)])
    assert "repeats" in message
    assert "record 1" in message and "record 0" in message
    assert "init_seed=3" in message


def test_the_same_seed_for_another_variant_or_task_is_not_a_repeat():
    pr.check([record(), record(variant="gqa4"), record(task="multihop"), record(init_seed=2)])


@pytest.mark.parametrize(
    "key, other",
    [
        ("generator", "multihop"),
        ("spec", {"seq_len": 128, "vocab_size": 128, "params": {"pairs": 4}}),
        ("chance", 0.5),
        ("d_model", 64),
        ("n_layers", 4),
        ("steps", 200),
        ("sequences_per_step", 16),
        ("data_seed", 8),
        ("eval_sequences", 32),
        ("peak_lr", 0.002),
        ("mode", "ref"),
        ("precision", "fp16"),
    ],
)
def test_records_of_one_point_must_agree_on_every_setting(key, other):
    message = refused([record(), record(variant="gqa4", **{key: other})])
    # The offender is record 1; record 0 is named only as the point's reference.
    assert message.startswith("record 1 ") and "differs from record 0" in message and key in message


def test_a_setting_may_differ_between_points():
    pr.check([record(difficulty=4), record(difficulty=8, d_model=64, steps=200, peak_lr=0.002)])


@pytest.mark.parametrize("key, other", [("torch", "2.16.0"), ("cuda", "12.8")])
def test_records_of_one_task_must_share_torch_and_cuda(key, other):
    message = refused([record(), record(variant="gqa4", difficulty=8, environment=environment(**{key: other}))])
    assert "record 1" in message and f"environment {key!r}" in message


def test_records_of_different_tasks_may_differ_in_environment():
    pr.check([record(), record(task="multihop", environment=environment(torch="2.16.0"))])


def test_an_environment_without_torch_is_refused():
    env = environment()
    del env["torch"]
    message = refused([record(environment=env)])
    assert "record 0" in message and "environment" in message


# --- rows ---------------------------------------------------------------------------------------


def test_rows_compute_the_hand_worked_values():
    rows = pr.rows([
        record(init_seed=1, accuracy=0.5, sequence_accuracy=0.25, skipped_steps=0),
        record(init_seed=2, accuracy=1.0, sequence_accuracy=0.75, skipped_steps=3),
    ])
    assert rows == [{
        "task": "mqar", "difficulty": 4, "variant": "mha", "seeds": 2,
        "accuracy_mean": 0.75, "accuracy_min": 0.5, "accuracy_max": 1.0,
        "chance": 0.25, "sequence_mean": 0.5, "sequence_min": 0.25,
        "d_model": 32, "steps": 100, "peak_lr": 0.001, "max_skipped_steps": 3,
    }]


def test_rows_keep_chance_none_as_none():
    (row,) = pr.rows([record(chance=None, task="induction", difficulty=8)])
    assert row["chance"] is None


def test_rows_are_sorted_by_task_then_difficulty_then_variant_order():
    shuffled = [
        record(task="multihop", difficulty=4, variant="mha", chance=0.25),
        record(task="mqar", difficulty=8, variant="recurrence", chance=0.125),
        record(task="mqar", difficulty=4, variant="recurrence", chance=0.25),
        record(task="mqar", difficulty=8, variant="mha", chance=0.125),
        record(task="mqar", difficulty=4, variant="gqa4", chance=0.25),
        record(task="mqar", difficulty=4, variant="mha", chance=0.25),
    ]
    order = [(r["task"], r["difficulty"], r["variant"]) for r in pr.rows(shuffled)]
    assert order == [
        ("mqar", 4, "mha"),
        ("mqar", 4, "gqa4"),
        ("mqar", 4, "recurrence"),
        ("mqar", 8, "mha"),
        ("mqar", 8, "recurrence"),
        ("multihop", 4, "mha"),
    ]


def test_rows_refuse_what_check_refuses():
    with pytest.raises(pr.ProbeReportError):
        pr.rows([record(), record()])


# --- render -------------------------------------------------------------------------------------


def test_render_prints_the_hand_worked_numbers_with_four_decimals():
    text = pr.render([
        record(init_seed=1, accuracy=0.5, sequence_accuracy=0.25),
        record(init_seed=2, accuracy=1.0, sequence_accuracy=0.75),
    ])
    assert "| 4 | mha | 2 | 0.7500 | 0.5000 | 1.0000 | 0.2500 | 0.5000 | 0.2500 | 32 | 100 | 0.001 | 0 |" in text


def test_render_prints_a_dash_for_a_missing_chance():
    text = pr.render([record(task="selective_copy", chance=None, accuracy=0.5, sequence_accuracy=0.5)])
    assert "| 4 | mha | 1 | 0.5000 | 0.5000 | 0.5000 | — | 0.5000 | 0.5000 | 32 | 100 | 0.001 | 0 |" in text


def test_render_has_one_section_per_task_with_its_table_header():
    text = pr.render([record(task="mqar"), record(task="multihop"), record(task="multihop", difficulty=8)])
    assert text.count("## Task `") == 2
    assert "## Task `mqar`" in text and "## Task `multihop`" in text
    assert text.count("| difficulty | variant | seeds | token acc. mean | min | max | 1/n | sequence acc. mean | min |"
                      " d_model | steps | peak LR | max skipped steps |") == 2


def test_render_states_the_paragraph_and_the_title():
    text = pr.render([record()])
    assert text.startswith("# Synthetic probes (plan §9.2)\n")
    assert pr.PARAGRAPH in text
    assert "no verdict is computed" in text and "Gate 1 is signed by Phil" in text


@pytest.mark.parametrize("records", [[], [record(), record(task="multihop", chance=None, difficulty=8)]])
def test_render_contains_no_judgement_word(records):
    assert JUDGEMENT_WORDS.search(pr.render(records)) is None


def test_the_blank_report_has_no_records_line_and_no_sections():
    text = pr.render([])
    assert "No probe records yet." in text
    assert "## Task" not in text


def test_the_blank_render_equals_the_committed_blank_report_byte_for_byte():
    assert pr.render([]) == BLANK_REPORT.read_text(encoding="utf-8")


# --- read_records -------------------------------------------------------------------------------


def _write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_read_records_reads_a_file_with_one_record(tmp_path):
    path = _write(tmp_path / "one.json", record(init_seed=5))
    assert pr.read_records(path) == [record(init_seed=5)]


def test_read_records_reads_a_file_with_a_list(tmp_path):
    path = _write(tmp_path / "list.json", [record(init_seed=1), record(init_seed=2)])
    assert [r["init_seed"] for r in pr.read_records(path)] == [1, 2]


def test_read_records_reads_every_json_in_a_directory_by_file_name(tmp_path):
    _write(tmp_path / "b.json", [record(init_seed=2)])
    _write(tmp_path / "a.json", record(init_seed=1))
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert [r["init_seed"] for r in pr.read_records(tmp_path)] == [1, 2]


def test_read_records_of_a_missing_path_is_empty(tmp_path):
    assert pr.read_records(tmp_path / "absent") == []


def test_read_records_refuses_a_file_that_is_not_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(pr.ProbeReportError, match="broken.json"):
        pr.read_records(path)


# --- The command line -----------------------------------------------------------------------------


def test_the_command_line_writes_the_report_and_returns_zero(tmp_path, capsys):
    probes = _write(tmp_path / "probes", [record(init_seed=1), record(init_seed=2, accuracy=1.0)])
    out = tmp_path / "tables"
    assert pr.main(["--probes", str(probes), "--out", str(out)]) == 0
    written = out / "probes.md"
    assert written.read_text(encoding="utf-8") == pr.render(pr.read_records(probes))
    assert capsys.readouterr().out.strip() == str(written)


def test_the_command_line_refuses_a_duplicate_with_status_one_and_writes_nothing(tmp_path, capsys):
    probes = _write(tmp_path / "probes", [record(init_seed=1), record(init_seed=1)])
    out = tmp_path / "tables"
    assert pr.main(["--probes", str(probes), "--out", str(out)]) == 1
    assert "repeats" in capsys.readouterr().err
    assert not out.exists()


def test_blank_ignores_the_probes_argument(tmp_path):
    probes = _write(tmp_path / "probes", [record(init_seed=1), record(init_seed=1)])
    out = tmp_path / "blank"
    assert pr.main(["--blank", "--probes", str(probes), "--out", str(out)]) == 0
    assert (out / "probes.md").read_text(encoding="utf-8") == BLANK_REPORT.read_text(encoding="utf-8")


def test_the_module_imports_without_matplotlib():
    code = "import sys, scripts.probe_report; assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
