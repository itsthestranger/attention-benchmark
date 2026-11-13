"""RUNBOOK.md gives exact commands (plan §17.4). This file checks that they are real.

Every `python -m scripts.<name> ...` command inside a bash block of RUNBOOK.md is
parsed with that script's own argument parser, after the shell variables have
been replaced by placeholder values. A renamed flag, a missing required flag or
a mistyped script name fails here. Nothing is run beyond argument parsing, and
the numbers the runbook states about the schedule are compared with the configs.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
from pathlib import Path

import pytest

from analysis import lr_sweep as sw
from src import config
from src.training.schedule import decay_start
from src.utils import manifest as mf

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (ROOT / "RUNBOOK.md").read_text(encoding="utf-8")

# Placeholder values for the shell variables the runbook uses. A variable that is
# not listed here fails the test, so the runbook cannot use one it never explains.
VARIABLES = {
    "DATA": "/data", "OUT": "/out", "BACKEND": "efficient", "SHARED": "no",
    "SHARDS": "a.parquet b.parquet", "TOKENIZER_SHARD": "a.parquet", "DATA_BUILD": "/build",
    "MICRO": "16", "INIT_SCALE": "65536", "GROWTH": "2000",
    "SWEEP_SEED": "21", "LR": "1.2e-3", "LR_HALF": "6e-4", "LR_DOUBLE": "2.4e-3",
    "MHA_LR": "1.2e-3", "MHA_SRC": "mha-isoparam-100M-lr1.2e3-s21-d1",
    "L6_LR": "1.2e-3", "L6_SRC": "mha_L6-isoparam-100M-lr1.2e3-s21-d1",
    "L10_LR": "1.2e-3", "L10_SRC": "mha_L10-isoparam-100M-lr1.2e3-s21-d1",
    "V_LR": "1.2e-3", "V_SRC": "mqa-isoparam-100M-lr1.2e3-s21-d1",
    "S_V": "1400", "TP_ID": "tp-placeholder",
    "v": "mqa", "s": "1", "d": "2", "b": "100M", "l": "mha_L6",
}
_VARIABLE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def bash_blocks() -> list[str]:
    return re.findall(r"```bash\n(.*?)```", RUNBOOK, flags=re.DOTALL)


def script_commands() -> list[str]:
    """Each `python -m scripts.X ...` command as one logical line, variables not yet replaced."""
    commands = []
    for block in bash_blocks():
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.strip()
            if "python -m scripts." not in line:
                continue
            commands.append(line[line.index("python -m scripts."):])
    return commands


def substitute(command: str) -> str:
    def replace(match: re.Match) -> str:
        name = match.group(1)
        assert name in VARIABLES, f"RUNBOOK.md uses ${name}, which this test does not know: {command}"
        return VARIABLES[name]

    return _VARIABLE.sub(replace, command)


class _Parsed(Exception):
    def __init__(self, namespace):
        self.namespace = namespace


def parse(command: str, monkeypatch) -> argparse.Namespace:
    """The namespace the script's own parser gives for a runbook command. Nothing else runs."""
    tokens = shlex.split(substitute(command))
    assert tokens[:2] == ["python", "-m"], command
    module = importlib.import_module(tokens[2])
    original = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    try:
        module.main(tokens[3:])
    except _Parsed as parsed:
        return parsed.namespace
    except SystemExit as exc:
        pytest.fail(f"the script's parser refuses this runbook command (exit {exc.code}): {command}")
    finally:
        monkeypatch.setattr(argparse.ArgumentParser, "parse_args", original)
    pytest.fail(f"{tokens[2]}.main never parsed its arguments: {command}")


COMMANDS = script_commands()


def test_the_runbook_has_commands_for_every_script_it_should():
    used = {shlex.split(c)[2] for c in COMMANDS}
    assert used >= {
        "scripts.prepare_data", "scripts.train", "scripts.branch", "scripts.select_lr",
        "scripts.make_tables", "scripts.make_plots", "scripts.timing_pass", "scripts.bench",
        "scripts.solve_config", "scripts.graph_diff",
        "scripts.prune", "scripts.timing_report", "scripts.measure_reproducibility",
    }
    assert len(COMMANDS) >= 30


@pytest.mark.parametrize("command", COMMANDS, ids=[f"{i:02d}-{shlex.split(c)[2]}" for i, c in enumerate(COMMANDS)])
def test_every_runbook_command_parses_with_its_scripts_own_parser(command, monkeypatch):
    parse(command, monkeypatch)


def test_no_command_carries_an_override_outside_the_smoke_runs():
    for command in COMMANDS:
        if "--override" in command:
            assert "--until" in command and "/smoke" in command, command


def test_training_commands_state_the_gpu_facts_the_manifest_records():
    for command in COMMANDS:
        if shlex.split(command)[2] in ("scripts.train", "scripts.branch"):
            assert "--gpu-shared" in command and "--expected-backend" in command, command
        if shlex.split(command)[2] in ("scripts.timing_pass", "scripts.bench"):
            assert "--gpu-shared false" in command and "--expected-backend" in command, command


def test_a_standalone_run_in_the_runbook_resolves_to_a_run_without_branches(monkeypatch):
    from scripts import _run_common as rc

    standalone = [c for c in COMMANDS if "--budget 100M --branches" in c and "scripts.train" in c]
    assert len(standalone) >= 6  # sweeps, reruns, data-order runs, the null control
    for command in standalone:
        resolved = rc.resolve_spec(rc.build_spec(parse(command, monkeypatch)))
        assert resolved["run"]["branches"] == [] and resolved["run"]["steps"] == BUDGET_STEPS["100M"]
        assert mf.parse_run_name(mf.run_name(resolved)).tokens == "100M"


# --- The numbers the runbook states -----------------------------------------------------------

RECIPE = config.load_recipe()
BUDGET_STEPS = dict(RECIPE["budgets"])
FRACTION = RECIPE["schedule"]["decay_fraction"]


def _number(text: str) -> int:
    return int(text.replace(",", ""))


def test_the_stated_budgets_and_branch_points_are_the_recipes():
    line = next(l for l in RUNBOOK.splitlines() if l.startswith("- Budgets are in steps"))
    stated = {label: _number(n) for label, n in re.findall(r"(\d+M) = ([\d,]+)", line)}
    assert stated == {label: BUDGET_STEPS[label] for label in ("100M", "200M", "300M")}
    leaves = [_number(n) for n in re.findall(r"at step ([\d,]+)", line)]
    assert leaves == [decay_start(BUDGET_STEPS["100M"], FRACTION), decay_start(BUDGET_STEPS["200M"], FRACTION)]


def test_the_until_steps_are_the_branch_point_and_the_smoke_lengths():
    smoke = {int(n) for c in COMMANDS if "/smoke" in c for n in re.findall(r"--until (\d+)", c)}
    real = {int(n) for c in COMMANDS if "/smoke" not in c for n in re.findall(r"--until (\d+)", c)}
    assert real == {decay_start(BUDGET_STEPS["100M"], FRACTION)}
    # Plan §7.3 judges skipped steps "in the first 100": the smoke run of every variant stops there,
    # still inside the stable phase of a real run (the first decay starts later).
    assert max(smoke) == 100 and all(n <= 100 for n in smoke)
    assert 100 < decay_start(BUDGET_STEPS["300M"], FRACTION)
    assert 100 >= RECIPE["schedule"]["warmup_steps"]


def test_the_sweep_grid_and_the_registered_seeds_are_the_codes():
    grid_line = next(l for l in RUNBOOK.splitlines() if l.strip().startswith("for LR in 3e-4"))
    rates = [float(x) for x in re.findall(r"[\d.]+e-\d+", grid_line)]
    assert tuple(rates) == sw.MHA_GRID
    prereg = config.load_prereg()
    null_line = next(l for l in RUNBOOK.splitlines() if l.strip().startswith("for s in 11"))
    assert [int(x) for x in re.findall(r"\b\d+\b", null_line)] == list(prereg["null_control"]["init_seeds"])
    blocks = "\n".join(bash_blocks())
    assert "for v in mha gqa4 gqa2 mqa mla192 mla96 swa256; do" in blocks
    assert f"for v in {' '.join(prereg['primary']['comparisons'])}; do" in blocks
    assert f"for v in {' '.join(prereg['exploratory']['variants'])}; do" in blocks
    assert f"for v in {' '.join(config.load_arm('iso_backbone')['variants'])}; do" in blocks


def test_the_gate_report_rows_the_runbook_names_exist():
    from scripts import make_tables as mt

    for section in ("Noise floor at 100M", "Null control", "Positive control: L6 against L8 at 300M",
                    "Seed count and detectable effect", "Ladder slope b by budget"):
        assert any(section in heading for heading in mt.GATE_SECTIONS), section
        assert section in RUNBOOK, section
    labels = {label for rows in mt.GATE_SECTIONS.values() for label in rows}
    for label in ("seeds per primary", "variants leaving Arm A"):
        assert label in labels and label in RUNBOOK


def test_pruning_is_shown_as_a_dry_run_before_it_is_applied():
    """T-43: the runbook never shows --apply without the dry run of the same command before it."""
    prune = [c for c in COMMANDS if shlex.split(c)[2] == "scripts.prune"]
    assert len(prune) >= 2
    assert "--apply" not in prune[0] and "--apply" in prune[1]
    assert prune[1].replace(" --apply", "") == prune[0]
    assert all("--final" not in c for c in prune)


def test_one_smoke_loop_reaches_an_in_run_evaluation():
    """Review S4, M12: the real cadence is 250 steps, which a 100-step smoke run never reaches."""
    smoke = [c for c in COMMANDS if "/smoke " in c and "--until 100" in c]
    assert smoke and all("--eval-every-steps 50" in c for c in smoke)


def test_the_declarations_are_in_the_commitments_file_and_not_on_a_command_line():
    """Decision 220: the ladder cut and the committed seed count are keys of configs/commitments.yaml,
    filled by pull request at the gate. A flag could be set after the results exist, and the tables
    and the plots could get different ones."""
    assert "--l10-cut" not in RUNBOOK
    assert "configs/commitments.yaml" in RUNBOOK
    for command in COMMANDS:
        if shlex.split(command)[2] in ("scripts.make_tables", "scripts.make_plots"):
            assert "--commitments" not in command, command  # both read the committed file

