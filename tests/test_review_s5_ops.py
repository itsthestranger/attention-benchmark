"""T-56: the operational findings of review s5 (G5 is in test_prune.py; these are M7 and M9).

M7: a run or branch started without a git checkout is refused before its first training step, with
no file written under OUT, instead of training to its last step and refusing to write its manifest.
M9: the Kaggle shim notebook writes where RUNBOOK.md writes, and it stops its log process whether the
run fails or not.

SYNTHETIC data only, through the entry points' own helpers and fixtures in tests/test_entry_points.py.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import pytest

from scripts import branch as branch_script
from scripts import train as train_script
from src.utils import manifest as mf
from tests.test_entry_points import (  # noqa: F401  (fixtures and helpers of the entry-point tests)
    branch_source,
    run_branch,
    run_train,
    short_resolved,
    snapshot,
    specs,
    synthetic,
    train_argv,
    trunk_resolved,
)

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "kaggle_shim.ipynb"
RUNBOOK = ROOT / "RUNBOOK.md"


# --- M7: the commit hash is read before the first step ---------------------------------------------


@pytest.fixture
def no_git(monkeypatch):
    """Make the function that runs git fail, as it does in a directory without a .git checkout."""
    def fail(repo_dir=None):
        raise mf.ManifestError("git rev-parse HEAD failed in /nowhere: fatal: not a git repository")

    monkeypatch.setattr(mf, "git_sha", fail)


def _files_under(out: Path) -> list[Path]:
    return [p for p in out.rglob("*") if p.is_file()] if out.exists() else []


def test_train_without_a_git_checkout_is_refused_before_any_step(no_git, specs, synthetic, tmp_path):
    out = tmp_path / "out"
    code = train_script.main(train_argv(specs.trunk, out, synthetic.root))
    assert code == 2
    assert _files_under(out) == []  # no checkpoint and no manifest: no step was taken


def test_train_refusal_names_git_and_uses_status_2(no_git, specs, synthetic, tmp_path, capsys):
    code = run_train(specs.short, tmp_path / "out", synthetic.root)
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("refused: ") and "git" in err


def test_branch_without_a_git_checkout_is_refused_and_writes_nothing(
    no_git, branch_source, specs, synthetic, tmp_path, capsys
):
    out = tmp_path / "out"
    shutil.copytree(branch_source, out)  # a completed trunk with its branch point: the branch could run
    before = snapshot(out)
    assert run_branch(specs.trunk, out, synthetic.root) == 2
    assert "git" in capsys.readouterr().err
    assert snapshot(out) == before  # no branch checkpoint and no branch manifest was written
    assert not any("-b24M" in p.name for p in out.rglob("*"))


def test_a_normal_run_with_git_completes_and_its_manifest_holds_the_commit_hash(
    specs, synthetic, short_resolved, tmp_path
):
    out = tmp_path / "out"
    assert run_train(specs.short, out, synthetic.root) == 0
    m = mf.read_manifest(out / "manifests" / "opt" / f"{mf.run_name(short_resolved)}.json")
    assert m["completed"] is True
    assert m["git_sha"] == mf.git_sha()


# --- M9: the shim notebook writes where the runbook writes, and stops its log whatever happens ----


def _notebook_cell() -> tuple[str, ast.Module]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 1
    source = "".join(code[0]["source"])
    return source, ast.parse(source)


def _runbook_out() -> str:
    found = re.findall(r"^export OUT=(\S+)", RUNBOOK.read_text(encoding="utf-8"), re.M)
    assert len(found) == 1, found
    return found[0]


def test_the_notebook_code_cell_compiles_and_has_at_most_ten_lines_of_newlines():
    source, _ = _notebook_cell()
    compile(source, "kaggle_shim.ipynb", "exec")
    assert source.count("\n") <= 10


def test_the_notebook_passes_the_runbook_out_directory_to_train():
    source, _ = _notebook_cell()
    match = re.search(r'"--out",\s*"([^"]+)"', source)
    assert match is not None and match.group(1) == _runbook_out() == "/kaggle/working/out"


def test_the_notebook_writes_its_log_under_the_runbook_out_directory():
    source, _ = _notebook_cell()
    logs = re.findall(r'f?"(/kaggle/working/[^"{]*?)/logs/', source)
    assert logs and all(d == _runbook_out() for d in logs)
    assert f'"{_runbook_out()}/logs"' in source


def test_the_notebook_terminates_the_log_in_a_finally_that_wraps_the_training_call():
    _, tree = _notebook_cell()
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    wrapping = [
        t for t in tries
        if any("scripts.train" in ast.unparse(stmt) for stmt in t.body)
        and any(any(isinstance(a, ast.Attribute) and a.attr == "terminate" for a in ast.walk(stmt))
                for stmt in t.finalbody)
    ]
    assert len(wrapping) == 1
    # The training call is not caught: a failed run still fails the cell (no handler swallows it).
    assert wrapping[0].handlers == []
