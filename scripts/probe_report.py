"""The synthetic-probe report (plan §9.2), built from probe/1 records and nothing else.

    python -m scripts.probe_report --probes results/probes --out results/tables
    python -m scripts.probe_report --blank --out results/tables/blank

Every number in the report is computed from the records here. The report decides no
verdict: Gate 1 is signed by Phil from these numbers (plan §9.2). The module does not
import matplotlib, so the report needs no plotting stack.

The exit status is 0 when the report was written. It is 1 on a refusal, and then no
file is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from scripts.make_tables import HEADLINE_VARIANTS

PROBE_SCHEMA = "probe/1"
PROBE_ANCHOR = "recurrence"
VARIANT_ORDER = (*HEADLINE_VARIANTS, PROBE_ANCHOR)
REPORT_NAME = "probes.md"
DASH = "—"
REQUIRED_KEYS = (
    "schema", "variant", "task", "generator", "difficulty", "init_seed", "accuracy", "sequence_accuracy",
    "chance", "n_scored", "n_sequences", "spec", "d_model", "n_layers", "data_seed", "steps",
    "steps_taken", "skipped_steps", "peak_lr", "sequences_per_step", "eval_sequences",
    "final_train_loss", "config_hash", "mode", "precision", "backend", "device", "environment",
    "git_sha",
)
# Within one (task, difficulty) point every model is trained and evaluated the same way (plan §9.2).
SAME_AT_POINT = (
    "generator", "spec", "chance", "d_model", "n_layers", "steps", "sequences_per_step", "data_seed",
    "eval_sequences", "peak_lr", "mode", "precision",
)
SAME_FOR_TASK_ENVIRONMENT = ("torch", "cuda")
# Integer fields that the report prints or takes a maximum of: non-negative integers only.
COUNT_FIELDS = ("init_seed", "d_model", "n_layers", "steps", "steps_taken", "skipped_steps")
PARAGRAPH = (
    "Accuracies are token accuracy on held-out sequences; sequence accuracy is the second column. "
    "\"1/n\" is the level of a model that knows which values are in the context and recalls nothing, "
    "with n the task's difficulty; it is shown as \"—\" where a task has no such level. The data behind "
    "Gate 1 (plan §9.2) is here, but no verdict is computed in this report: Gate 1 is signed by Phil. "
    "The recurrence anchor is expected to fall towards 1/n as the task grows; SWA's behaviour is "
    "reported, not gating."
)
COLUMNS = (
    "difficulty", "variant", "seeds", "token acc. mean", "min", "max", "1/n", "sequence acc. mean", "min",
    "d_model", "steps", "peak LR", "max skipped steps",
)


class ProbeReportError(ValueError):
    """Records that cannot be reported. The message names the records concerned."""


def _name(index: int, record) -> str:
    if isinstance(record, Mapping):
        parts = [f"{key}={record[key]!r}" for key in ("task", "variant", "difficulty", "init_seed") if key in record]
        if parts:
            return f"record {index} ({', '.join(parts)})"
    return f"record {index}"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_unit(value) -> bool:
    return _is_number(value) and 0 <= value <= 1


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _record_problems(index: int, record) -> list[str]:
    if not isinstance(record, Mapping):
        return [f"record {index} is not an object: {record!r}"]
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        return [f"{_name(index, record)} lacks {missing}"]
    name = _name(index, record)
    problems = []
    if record["schema"] != PROBE_SCHEMA:
        problems.append(f"{name} has schema {record['schema']!r}, expected {PROBE_SCHEMA!r}")
    if record["variant"] not in VARIANT_ORDER:
        problems.append(f"{name} has variant {record['variant']!r}, not one of {list(VARIANT_ORDER)}")
    if not isinstance(record["task"], str) or not record["task"]:
        problems.append(f"{name} has task {record['task']!r}, expected a non-empty name")
    for key in ("accuracy", "sequence_accuracy"):
        if not _is_unit(record[key]):
            problems.append(f"{name} has {key} {record[key]!r}, not a finite number in [0, 1]")
    difficulty = record["difficulty"]
    if not isinstance(difficulty, int) or isinstance(difficulty, bool) or difficulty < 1:
        problems.append(f"{name} has difficulty {difficulty!r}, expected an integer of at least 1")
    for key in COUNT_FIELDS:
        if not _is_count(record[key]):
            problems.append(f"{name} has {key} {record[key]!r}, expected a non-negative integer")
    if not (_is_number(record["peak_lr"]) and record["peak_lr"] > 0):
        problems.append(f"{name} has peak_lr {record['peak_lr']!r}, expected a finite number above zero")
    chance = record["chance"]
    if chance is not None and not (_is_number(chance) and 0 < chance <= 1):
        problems.append(f"{name} has chance {chance!r}, expected None or a number in (0, 1]")
    environment = record["environment"]
    if not isinstance(environment, Mapping) or any(k not in environment for k in SAME_FOR_TASK_ENVIRONMENT):
        problems.append(f"{name} has environment {environment!r}, expected an object with 'torch' and 'cuda'")
    return problems


def check(records: Sequence) -> None:
    """Refuse records that cannot be reported together. Raises ProbeReportError naming the records."""
    problems = []
    for index, record in enumerate(records):
        problems.extend(_record_problems(index, record))
    if problems:
        raise ProbeReportError("\n".join(problems))

    first_seen: dict = {}
    for index, record in enumerate(records):
        key = (record["task"], record["variant"], record["difficulty"], record["init_seed"])
        if key in first_seen:
            earlier = first_seen[key]
            problems.append(
                f"{_name(index, record)} repeats {_name(earlier, records[earlier])}: the same "
                "(task, variant, difficulty, init_seed)"
            )
        else:
            first_seen[key] = index

    points: dict = defaultdict(list)
    for index, record in enumerate(records):
        points[(record["task"], record["difficulty"])].append(index)
    for (task, difficulty), indices in points.items():
        head = indices[0]
        for index in indices[1:]:
            for key in SAME_AT_POINT:
                if records[index][key] != records[head][key]:
                    problems.append(
                        f"{_name(index, records[index])} differs from {_name(head, records[head])} in "
                        f"{key}: {records[index][key]!r} and {records[head][key]!r} at task {task!r}, "
                        f"difficulty {difficulty}"
                    )

    by_task: dict = defaultdict(list)
    for index, record in enumerate(records):
        by_task[record["task"]].append(index)
    for task, indices in by_task.items():
        head = indices[0]
        for index in indices[1:]:
            for key in SAME_FOR_TASK_ENVIRONMENT:
                here = records[index]["environment"][key]
                there = records[head]["environment"][key]
                if here != there:
                    problems.append(
                        f"{_name(index, records[index])} differs from {_name(head, records[head])} in "
                        f"environment {key!r}: {here!r} and {there!r} for task {task!r}"
                    )
    if problems:
        raise ProbeReportError("\n".join(problems))


def read_records(path) -> list[dict]:
    """A file holding one record or a list, or every *.json in a directory, by name. Missing gives []."""
    target = Path(path)
    if not target.exists():
        return []
    files = sorted(target.glob("*.json")) if target.is_dir() else [target]
    records: list = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProbeReportError(f"{file}: not JSON ({exc})") from exc
        records.extend(data if isinstance(data, list) else [data])
    return records


def rows(records: Sequence) -> list[dict]:
    """One row per (task, difficulty, variant). Refuses what `check` refuses."""
    check(records)
    groups: dict = defaultdict(list)
    for record in records:
        groups[(record["task"], record["difficulty"], record["variant"])].append(record)
    order = sorted(groups, key=lambda k: (k[0], k[1], VARIANT_ORDER.index(k[2])))
    out = []
    for task, difficulty, variant in order:
        members = groups[(task, difficulty, variant)]
        accuracy = [r["accuracy"] for r in members]
        sequence = [r["sequence_accuracy"] for r in members]
        head = members[0]
        out.append({
            "task": task,
            "difficulty": difficulty,
            "variant": variant,
            "seeds": len(members),
            "accuracy_mean": sum(accuracy) / len(accuracy),
            "accuracy_min": min(accuracy),
            "accuracy_max": max(accuracy),
            "chance": head["chance"],
            "sequence_mean": sum(sequence) / len(sequence),
            "sequence_min": min(sequence),
            "d_model": head["d_model"],
            "steps": head["steps"],
            "peak_lr": head["peak_lr"],
            "max_skipped_steps": max(r["skipped_steps"] for r in members),
        })
    return out


def _four(value) -> str:
    return DASH if value is None else f"{value:.4f}"


def render(records: Sequence) -> str:
    """Markdown: the title, the fixed paragraph, then one table per task. Numbers only, no verdict."""
    table_rows = rows(records)
    lines = ["# Synthetic probes (plan §9.2)", "", PARAGRAPH, ""]
    if not table_rows:
        lines += ["No probe records yet.", ""]
        return "\n".join(lines)
    for task in sorted({row["task"] for row in table_rows}):
        lines += [f"## Task `{task}`", "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        for row in (r for r in table_rows if r["task"] == task):
            cells = [
                str(row["difficulty"]), row["variant"], str(row["seeds"]),
                _four(row["accuracy_mean"]), _four(row["accuracy_min"]), _four(row["accuracy_max"]),
                _four(row["chance"]), _four(row["sequence_mean"]), _four(row["sequence_min"]),
                str(row["d_model"]), str(row["steps"]), f"{row['peak_lr']:g}", str(row["max_skipped_steps"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic-probe report (plan §9.2) from probe/1 records.")
    parser.add_argument("--probes", default="results/probes", help="a probe/1 file, or a directory of them")
    parser.add_argument("--out", default="results/tables", help="directory to write probes.md to")
    parser.add_argument("--blank", action="store_true", help="ignore --probes; write the report for no records")
    args = parser.parse_args(argv)
    try:
        records = [] if args.blank else read_records(args.probes)
        text = render(records)
    except ProbeReportError as exc:
        print(exc, file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / REPORT_NAME
    path.write_text(text, encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
