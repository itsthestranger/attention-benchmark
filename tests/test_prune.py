"""T-43: scripts/prune.py deletes only the checkpoints no later command needs (plan §12.3).

SYNTHETIC data only, through the entry points' own helpers in tests/test_entry_points.py:
a tiny trunk of 32 steps with a 24-step branch from its branch point at step 19. Every test
works on its own copy of a module-scoped run, so no test sees another test's deletions.

Expected file sets are written out in full: each test names every checkpoint that must
exist afterwards.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path

import pytest

from scripts import _run_common as rc
from scripts import prune
from src import config
from src.training import branch as br
from src.training import checkpoint as ck
from src.utils import manifest as mf
from tests.test_entry_points import (  # noqa: F401  (fixtures and helpers of the entry-point tests)
    SHORT_LABEL,
    TRUNK_LABEL,
    _spec_copy,
    run_branch,
    run_train,
    short_resolved,
    snapshot,
    specs,
    synthetic,
    trunk_resolved,
)

CURVE = rc.CURVE_FILE
SESSIONS = rc.SESSIONS_FILE
PEAK = rc.PEAK_FILE  # the per-session peak-memory record (T-41): bookkeeping, never pruned
LISTED = re.compile(r"^  (?:would delete|deleted) (\S+) \(")


# --- Module-scoped runs ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trunk_only(synthetic, specs, tmp_path_factory):
    """A completed 32-step trunk, its branch point at step 19 kept, no branch run."""
    out = tmp_path_factory.mktemp("prune_trunk")
    assert run_train(specs.trunk, out, synthetic.root) == 0
    return out


@pytest.fixture(scope="module")
def full(trunk_only, specs, synthetic, tmp_path_factory):
    """The trunk above and its completed 24M branch, which saves every 2 steps so it has a non-final checkpoint."""
    out = tmp_path_factory.mktemp("prune_full")
    shutil.copytree(trunk_only, out, dirs_exist_ok=True)
    assert run_branch(specs.trunk, out, synthetic.root, extra=["--checkpoint-every-steps", "2"]) == 0
    return out


@pytest.fixture(scope="module")
def unfinished_trunk(synthetic, specs, tmp_path_factory):
    """A trunk stopped at step 20: unfinished, with regular checkpoints at 8 and 16 and branch point 19."""
    out = tmp_path_factory.mktemp("prune_unfinished")
    assert run_train(specs.trunk, out, synthetic.root, extra=["--until", "20"]) == 0
    return out


@pytest.fixture(scope="module")
def trunk_ref(synthetic, specs, tmp_path_factory):
    """A completed trunk in ref mode, to check that the two modes are pruned separately."""
    out = tmp_path_factory.mktemp("prune_ref")
    assert run_train(specs.trunk, out, synthetic.root, mode="ref") == 0
    return out


# --- Helpers -----------------------------------------------------------------------------------


def _copy(src: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "out"
    shutil.copytree(src, dst)
    return dst


def _rel(out: Path, path: Path) -> str:
    return Path(path).relative_to(out).as_posix()


def _files(out: Path, mode: str = "opt") -> set[str]:
    """Every file under OUT/checkpoints/MODE, relative to OUT."""
    root = Path(out) / "checkpoints" / mode
    return {_rel(out, p) for p in root.rglob("*") if p.is_file()}


def _manifest_files(out: Path, mode: str = "opt") -> set[str]:
    root = Path(out) / "manifests" / mode
    return {_rel(out, p) for p in root.iterdir() if p.is_file()}


def _dir(out: Path, resolved: dict, mode: str = "opt") -> str:
    """The checkpoint directory of a configuration, relative to OUT."""
    return _rel(out, ck.checkpoint_dir(Path(out) / "checkpoints" / mode, resolved))


def _branch_resolved(trunk_resolved: dict) -> dict:
    return br.branch_resolved(trunk_resolved, SHORT_LABEL)


def _listed(text: str) -> set[str]:
    return {m.group(1) for m in (LISTED.match(line) for line in text.splitlines()) if m}


def _checkpoint_bytes(out: Path, mode: str = "opt") -> int:
    return sum(p.stat().st_size for p in (Path(out) / "checkpoints" / mode).rglob("*") if p.is_file())


def _rewrite_manifest(out: Path, name: str, edit, mode: str = "opt") -> Path:
    """Edit a manifest's JSON in place, without the writer's checks (to make a broken copy)."""
    path = Path(out) / "manifests" / mode / f"{name}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return path


def _prune(out: Path, *extra: str, capsys=None) -> tuple[int, str]:
    rc_code = prune.main(["--out", str(out), *extra])
    text = capsys.readouterr().out if capsys is not None else ""
    return rc_code, text


# --- The dry run and the listing ---------------------------------------------------------------


def test_dry_run_deletes_nothing_and_apply_deletes_exactly_the_listed_files(full, trunk_resolved, tmp_path, capsys):
    out = _copy(full, tmp_path)
    before = snapshot(out)
    code, dry = _prune(out, capsys=capsys)
    assert code == 0
    assert snapshot(out) == before
    listed = _listed(dry)
    assert len(listed) == 3
    assert mf.run_name(trunk_resolved) in dry and mf.run_name(trunk_resolved, SHORT_LABEL) in dry

    code, applied = _prune(out, "--apply", capsys=capsys)
    assert code == 0
    after = snapshot(out)
    assert set(after) <= set(before)
    assert set(before) - set(after) == listed == _listed(applied)


def test_completed_trunk_and_branch_keep_only_their_finals_and_curve_and_session_files(full, trunk_resolved, tmp_path):
    out = _copy(full, tmp_path)
    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    assert _files(out) == {
        f"{trunk}/step00000024.pt", f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt",
        f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000022.pt", f"{branch}/step00000024.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }
    assert prune.main(["--out", str(out), "--apply"]) == 0
    assert _files(out) == {
        f"{trunk}/step00000032.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000024.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }


def test_final_flag_removes_the_final_checkpoints_and_nothing_else(full, trunk_resolved, tmp_path):
    out = _copy(full, tmp_path)
    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    assert prune.main(["--out", str(out), "--apply", "--final"]) == 0
    assert _files(out) == {
        f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }


# --- Branch points ------------------------------------------------------------------------------


def test_branch_point_stays_when_its_branch_was_never_run(trunk_only, trunk_resolved, tmp_path):
    out = _copy(trunk_only, tmp_path)
    trunk = _dir(out, trunk_resolved)
    assert prune.main(["--out", str(out), "--apply"]) == 0
    assert _files(out) == {
        f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
    }


def test_branch_point_stays_when_its_branch_is_unfinished(trunk_only, specs, synthetic, trunk_resolved, tmp_path):
    out = _copy(trunk_only, tmp_path)
    assert run_branch(specs.trunk, out, synthetic.root, extra=["--until", "20", "--checkpoint-every-steps", "2"]) == 0
    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    assert mf.read_manifest(out / "manifests/opt" / f"{mf.run_name(trunk_resolved, SHORT_LABEL)}.json")["completed"] is False
    assert prune.main(["--out", str(out), "--apply"]) == 0
    assert _files(out) == {
        f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000020.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }


@pytest.mark.parametrize("damage", ["unparseable", "invalid"])
def test_branch_point_stays_when_the_branch_manifest_is_invalid(full, trunk_resolved, tmp_path, damage):
    out = _copy(full, tmp_path)
    name = mf.run_name(trunk_resolved, SHORT_LABEL)
    path = out / "manifests/opt" / f"{name}.json"
    if damage == "unparseable":
        path.write_text("{ not json", encoding="utf-8")
    else:
        _rewrite_manifest(out, name, lambda doc: doc.__setitem__("completed", "yes"))
    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    assert prune.main(["--out", str(out), "--apply"]) == 0
    # Only the trunk's regular checkpoint at 24 goes: the branch point stays, and so does
    # everything of the branch, whose manifest cannot be trusted.
    assert _files(out) == {
        f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000022.pt", f"{branch}/step00000024.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }


# --- Things that must lose nothing --------------------------------------------------------------


def test_an_unfinished_run_loses_nothing(unfinished_trunk, trunk_resolved, tmp_path):
    out = _copy(unfinished_trunk, tmp_path)
    before = snapshot(out)
    assert prune.main(["--out", str(out), "--apply", "--final"]) == 0
    assert snapshot(out) == before
    trunk = _dir(out, trunk_resolved)
    # Stopped at step 20: the cadence checkpoint at 16 and the one a stop leaves at 20 (T-40).
    assert _files(out) == {
        f"{trunk}/step00000016.pt", f"{trunk}/step00000020.pt", f"{trunk}/trunk-step00000019.pt",
        f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
    }


def test_a_diverged_run_loses_nothing(trunk_only, trunk_resolved, tmp_path):
    out = _copy(trunk_only, tmp_path)
    name = mf.run_name(trunk_resolved)

    def diverge(doc):
        doc["completed"] = False
        doc["steps"]["attempted"] = 20
        doc["steps"]["taken"] = 20
        doc["step_seconds"] = doc["step_seconds"][:20]
        doc["final_metrics"] = {"diverged": True, "reason": "synthetic test", "step": 20}

    _rewrite_manifest(out, name, diverge)
    assert mf.read_manifest(out / "manifests/opt" / f"{name}.json")["final_metrics"]["diverged"] is True
    before = snapshot(out)
    assert prune.main(["--out", str(out), "--apply", "--final"]) == 0
    assert snapshot(out) == before


def test_a_run_with_an_unreadable_manifest_loses_nothing(full, trunk_resolved, tmp_path, capsys):
    out = _copy(full, tmp_path)
    (out / "manifests/opt" / f"{mf.run_name(trunk_resolved)}.json").write_text("garbage", encoding="utf-8")
    trunk = _dir(out, trunk_resolved)
    trunk_files = {f for f in _files(out) if f.startswith(trunk + "/")}
    code, text = _prune(out, "--apply", "--final", capsys=capsys)
    assert code == 0
    assert "unreadable or invalid" in text
    assert {f for f in _files(out) if f.startswith(trunk + "/")} == trunk_files
    # The branch's own manifest is valid and complete: its own directory is still pruned by its own rule.
    branch = _dir(out, _branch_resolved(trunk_resolved))
    assert _listed(text) == {f"{branch}/step00000022.pt", f"{branch}/step00000024.pt"}


def test_a_checkpoint_directory_with_no_manifest_is_unknown_and_loses_nothing(trunk_only, trunk_resolved, tmp_path, capsys):
    out = _copy(trunk_only, tmp_path)
    orphan = out / "checkpoints/opt" / ("0" * 64)
    orphan.mkdir()
    (orphan / "step00000008.pt").write_bytes(b"orphan")
    before = snapshot(out)
    p = prune.plan(out, "opt")
    assert [(k.name, k.path) for k in p.kept if k.path == orphan] == [("", orphan)]
    assert "unknown" in next(k.reason for k in p.kept if k.path == orphan)
    assert prune.main(["--out", str(out), "--apply"]) == 0
    trunk = _dir(out, trunk_resolved)
    # The trunk's own non-final checkpoint goes, as it would without the orphan; the orphan stays.
    assert set(before) - set(snapshot(out)) == {f"{trunk}/step00000024.pt"}
    assert {f: h for f, h in snapshot(out).items() if f.startswith("checkpoints/opt/" + "0" * 64)} == {
        f: h for f, h in before.items() if f.startswith("checkpoints/opt/" + "0" * 64)
    }


def test_curve_session_list_and_manifests_are_never_deleted(full, trunk_resolved, tmp_path):
    out = _copy(full, tmp_path)
    manifests = _manifest_files(out)
    kept_by_name = {f for f in _files(out) if f.endswith(f"/{CURVE}") or f.endswith(f"/{SESSIONS}")}
    assert len(kept_by_name) == 4
    assert prune.main(["--out", str(out), "--apply", "--final"]) == 0
    assert _manifest_files(out) == manifests
    assert {f for f in _files(out) if f.endswith(f"/{CURVE}") or f.endswith(f"/{SESSIONS}")} == kept_by_name


# --- Modes, idempotence, the resumable commands ------------------------------------------------


def test_opt_and_ref_are_pruned_separately(full, trunk_ref, trunk_resolved, tmp_path):
    out = _copy(full, tmp_path)
    shutil.copytree(trunk_ref / "checkpoints/ref", out / "checkpoints/ref")
    shutil.copytree(trunk_ref / "manifests/ref", out / "manifests/ref")
    ref_before = {f: h for f, h in snapshot(out).items() if "/ref/" in f}
    assert prune.main(["--out", str(out), "--mode", "opt", "--apply"]) == 0
    assert {f: h for f, h in snapshot(out).items() if "/ref/" in f} == ref_before
    ref_trunk = _dir(out, trunk_resolved, mode="ref")
    assert prune.main(["--out", str(out), "--mode", "ref", "--apply"]) == 0
    assert _files(out, "ref") == {
        f"{ref_trunk}/step00000032.pt", f"{ref_trunk}/trunk-step00000019.pt",
        f"{ref_trunk}/{CURVE}", f"{ref_trunk}/{SESSIONS}", f"{ref_trunk}/{PEAK}",
    }


@pytest.mark.parametrize("final", [False, True])
def test_a_second_apply_deletes_nothing_and_the_resumable_commands_change_no_file(
    full, specs, synthetic, tmp_path, capsys, final
):
    out = _copy(full, tmp_path)
    flags = ["--final"] if final else []
    assert prune.main(["--out", str(out), "--apply", *flags]) == 0
    capsys.readouterr()
    settled = snapshot(out)
    assert prune.main(["--out", str(out), "--apply", *flags]) == 0
    assert "nothing to delete" in capsys.readouterr().out
    assert snapshot(out) == settled
    assert run_train(specs.trunk, out, synthetic.root) == 0
    assert run_branch(specs.trunk, out, synthetic.root) == 0
    assert snapshot(out) == settled


def test_the_bytes_reported_are_the_bytes_freed(full, tmp_path):
    out = _copy(full, tmp_path)
    p = prune.plan(out, "opt", final=True)
    assert p.total_bytes > 0 and len(p.deletions) == 5  # trunk 24, trunk 19, trunk 32, branch 22, branch 24
    before = _checkpoint_bytes(out)
    freed = prune.apply(p)
    assert freed == p.total_bytes == before - _checkpoint_bytes(out)


def test_a_missing_out_is_refused_with_status_2(tmp_path, capsys):
    assert prune.main(["--out", str(tmp_path / "missing")]) == 2
    assert "refused" in capsys.readouterr().err


def test_an_unknown_mode_is_refused_with_status_2(trunk_only, tmp_path, capsys):
    out = _copy(trunk_only, tmp_path)
    assert prune.main(["--out", str(out), "--mode", "sgd"]) == 2
    assert "refused" in capsys.readouterr().err


# --- Review gaps: unfinished branches, branch ownership, the final fallback, extra trunk-step files ---


@pytest.fixture(scope="module")
def trunk_with_stopped_branch(trunk_only, specs, synthetic, tmp_path_factory):
    """A completed trunk whose 24M branch was stopped at step 22: regular checkpoints at 20 and 22 remain."""
    out = tmp_path_factory.mktemp("prune_stopped_branch")
    shutil.copytree(trunk_only, out, dirs_exist_ok=True)
    assert run_branch(specs.trunk, out, synthetic.root, extra=["--until", "22", "--checkpoint-every-steps", "2"]) == 0
    return out


def _kept_reason(p: prune.Plan, path: Path) -> str:
    return next(k.reason for k in p.kept if k.path == path)


@pytest.mark.parametrize("final", [False, True])
def test_an_unfinished_branch_loses_nothing_in_its_own_directory(
    trunk_with_stopped_branch, trunk_resolved, tmp_path, final
):
    out = _copy(trunk_with_stopped_branch, tmp_path)
    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    assert _files(out) == {
        f"{trunk}/step00000024.pt", f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt",
        f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000020.pt", f"{branch}/step00000022.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }
    flags = ["--final"] if final else []
    assert prune.main(["--out", str(out), "--apply", *flags]) == 0
    expected = {
        f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000020.pt", f"{branch}/step00000022.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }
    if final:
        expected.discard(f"{trunk}/step00000032.pt")
    # The trunk's non-final step 24 goes; nothing of the unfinished branch does, final or not.
    assert _files(out) == expected


def test_a_branch_manifest_with_another_config_hash_does_not_authorise_its_branch_point(full, trunk_resolved, tmp_path):
    out = _copy(full, tmp_path)
    name = mf.run_name(trunk_resolved, SHORT_LABEL)

    def other_config(doc):
        # A value outside the run's name, so the name still matches; the hash is recomputed so the file stays valid.
        doc["resolved"]["model"]["d_ff"] += 1
        doc["config_hash"] = config.config_hash(doc["resolved"])

    _rewrite_manifest(out, name, other_config)
    branch_manifest = mf.read_manifest(out / "manifests/opt" / f"{name}.json")  # still valid
    assert branch_manifest["config_hash"] != config.config_hash(trunk_resolved)

    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    p = prune.plan(out, "opt")
    branch_point = out / "checkpoints/opt" / trunk.split("/")[-1] / "trunk-step00000019.pt"
    assert "does not belong to this run" in _kept_reason(p, branch_point)
    assert prune.main(["--out", str(out), "--apply"]) == 0
    assert _files(out) == {
        f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000019.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000022.pt", f"{branch}/step00000024.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }


def test_a_branch_manifest_naming_another_parent_does_not_authorise_its_branch_point(full, trunk_resolved):
    """A unit check of the parent-name condition on a hand-built manifest.

    read_manifest cannot produce this case: branch.parent_run must equal the name of resolved, and
    config_hash must equal the hash of resolved (manifest.py, validate_manifest), so a valid branch
    manifest with this parent's hash always names this parent. The condition is checked directly.
    """
    name = mf.run_name(trunk_resolved, SHORT_LABEL)
    doc = json.loads((full / "manifests/opt" / f"{name}.json").read_text(encoding="utf-8"))
    doc["branch"]["parent_run"] = "some-other-parent"
    manifests = {f"{name}.json": doc}
    reason = prune._branch_blocker(manifests, {}, trunk_resolved, SHORT_LABEL)
    assert reason is not None and "does not belong to this run" in reason
    assert prune._branch_blocker({f"{name}.json": json.loads((full / 'manifests/opt' / f'{name}.json').read_text())},
                                 {}, trunk_resolved, SHORT_LABEL) is None


def test_a_completed_run_without_its_final_checkpoint_keeps_the_newest_regular_one(trunk_only, trunk_resolved, tmp_path):
    out = _copy(trunk_only, tmp_path)
    trunk = _dir(out, trunk_resolved)
    directory = out / "checkpoints/opt" / trunk.split("/")[-1]
    (directory / "step00000032.pt").unlink()  # the final checkpoint is gone
    (directory / "step00000016.pt").write_bytes(b"older regular checkpoint (stub: only names matter here)")
    assert prune.main(["--out", str(out), "--apply"]) == 0
    # The older regular checkpoint goes; the newest, step 24, is kept as the fallback.
    assert _files(out) == {
        f"{trunk}/step00000024.pt", f"{trunk}/trunk-step00000019.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
    }


def test_a_trunk_step_file_that_is_not_a_branch_point_of_the_run_is_left_alone(full, trunk_resolved, tmp_path):
    out = _copy(full, tmp_path)
    trunk, branch = _dir(out, trunk_resolved), _dir(out, _branch_resolved(trunk_resolved))
    directory = out / "checkpoints/opt" / trunk.split("/")[-1]
    extra = directory / "trunk-step00000010.pt"  # step 10 is none of this run's branch steps
    shutil.copy(directory / "trunk-step00000019.pt", extra)

    p = prune.plan(out, "opt")
    assert "not a branch point of this run" in _kept_reason(p, extra)
    assert prune.main(["--out", str(out), "--apply"]) == 0
    assert _files(out) == {
        f"{trunk}/step00000032.pt", f"{trunk}/trunk-step00000010.pt", f"{trunk}/{CURVE}", f"{trunk}/{SESSIONS}", f"{trunk}/{PEAK}",
        f"{branch}/step00000024.pt", f"{branch}/{CURVE}", f"{branch}/{SESSIONS}", f"{branch}/{PEAK}",
    }


# --- --final-short: branches and short runs lose their finals, the primary budget's final stays ---


@pytest.fixture
def primary_is_the_trunk_budget(monkeypatch):
    """Make the primary budget the tiny trunk's token label. prune reads it with config.load_prereg(), so the
    patch is what the code sees: the tests need no run named 300M."""
    real = config.load_prereg

    def patched():
        doc = copy.deepcopy(real())
        doc["primary"]["budget"] = TRUNK_LABEL
        return doc

    monkeypatch.setattr(config, "load_prereg", patched)


@pytest.fixture(scope="module")
def campaign(full, specs, synthetic, tmp_path_factory):
    """The trunk at 32M with its 24M branch, plus a finished standalone 24M run (a short run, not a branch)."""
    out = tmp_path_factory.mktemp("prune_campaign")
    shutil.copytree(full, out, dirs_exist_ok=True)
    assert run_train(specs.short, out, synthetic.root) == 0
    return out


@pytest.fixture(scope="module")
def campaign_with_unfinished_short(campaign, specs, synthetic, tmp_path_factory):
    """The campaign plus a 24M run with another init seed, stopped at step 12: unfinished, not a result."""
    out = tmp_path_factory.mktemp("prune_campaign_unfinished")
    shutil.copytree(campaign, out, dirs_exist_ok=True)
    spec = _spec_copy(specs.short, out.parent / "short_seed2.yaml", init_seed=2)
    assert run_train(spec, out, synthetic.root, extra=["--until", "12"]) == 0
    return out


def test_the_primary_budget_is_read_from_the_prereg_config(campaign, trunk_resolved):
    p = prune.plan(campaign, "opt", final_short=True)
    assert p.primary == config.load_prereg()["primary"]["budget"] == "300M"
    assert prune.plan(campaign, "opt").primary is None  # not read without the flag


def test_final_short_drops_the_branch_and_short_run_finals_and_keeps_the_primary_final(
    campaign, trunk_resolved, short_resolved, tmp_path, capsys, primary_is_the_trunk_budget
):
    out = _copy(campaign, tmp_path)
    trunk = _dir(out, trunk_resolved)
    branch = _dir(out, _branch_resolved(trunk_resolved))
    short = _dir(out, short_resolved)
    assert short != trunk and short != branch
    trunk_final = f"{trunk}/step00000032.pt"
    assert trunk_final in _files(out)
    code, dry = _prune(out, "--final-short", capsys=capsys)
    assert code == 0
    listed = _listed(dry)
    assert f"{branch}/step00000024.pt" in listed and f"{short}/step00000024.pt" in listed
    assert trunk_final not in listed
    assert f"keep {trunk_final}: final checkpoint kept: a run at the primary budget is loaded by evaluation " \
           "(--final removes it)" in dry
    before_apply = _files(out)
    assert prune.main(["--out", str(out), "--apply", "--final-short"]) == 0
    assert set(before_apply) - _files(out) == listed
    assert trunk_final in _files(out)
    assert f"{branch}/step00000024.pt" not in _files(out) and f"{short}/step00000024.pt" not in _files(out)


def test_final_short_on_the_campaign_removes_exactly_the_branch_and_short_run_finals_and_their_others(
    campaign, trunk_resolved, short_resolved, tmp_path, primary_is_the_trunk_budget
):
    out = _copy(campaign, tmp_path)
    trunk = _dir(out, trunk_resolved)
    branch = _dir(out, _branch_resolved(trunk_resolved))
    short = _dir(out, short_resolved)
    before = _files(out)
    assert prune.main(["--out", str(out), "--apply", "--final-short"]) == 0
    after = _files(out)
    assert before - after == {
        f"{trunk}/step00000024.pt", f"{trunk}/trunk-step00000019.pt",
        f"{branch}/step00000022.pt", f"{branch}/step00000024.pt",
        f"{short}/step00000016.pt", f"{short}/step00000024.pt",
    }
    assert f"{trunk}/step00000032.pt" in after
    assert {f for f in after if "/" + CURVE in f or "/" + SESSIONS in f or "/" + PEAK in f} == {
        f for f in before if "/" + CURVE in f or "/" + SESSIONS in f or "/" + PEAK in f
    }


def test_without_flags_no_final_checkpoint_is_listed_and_the_final_kept_reason_is_unchanged(
    campaign, trunk_resolved, short_resolved, tmp_path, capsys
):
    out = _copy(campaign, tmp_path)
    p = prune.plan(out, "opt")
    finals = [f"{_dir(out, r)}/step{total:08d}.pt" for r, total in ((trunk_resolved, 32), (short_resolved, 24))]
    finals.append(f"{_dir(out, _branch_resolved(trunk_resolved))}/step00000024.pt")
    deleted = {_rel(out, d.path) for d in p.deletions}
    assert not deleted & set(finals)
    for f in finals:
        assert _kept_reason(p, out / f) == "final checkpoint kept (evaluation loads it; --final removes it)"


def test_final_then_final_short_is_the_same_plan_as_final(campaign, tmp_path, primary_is_the_trunk_budget):
    out = _copy(campaign, tmp_path)
    both = prune.plan(out, "opt", final=True, final_short=True)
    assert both == prune.plan(out, "opt", final=True)
    assert both.deletions  # not vacuously empty


def test_final_short_plus_final_equals_final_alone_on_the_listed_output(
    campaign, tmp_path, capsys, primary_is_the_trunk_budget
):
    out = _copy(campaign, tmp_path)
    _, text_both = _prune(out, "--final", "--final-short", capsys=capsys)
    _, text_final = _prune(out, "--final", capsys=capsys)
    assert text_both == text_final


def test_an_unfinished_short_run_loses_nothing_under_final_short(
    campaign_with_unfinished_short, short_resolved, tmp_path, primary_is_the_trunk_budget
):
    out = _copy(campaign_with_unfinished_short, tmp_path)
    seed2 = copy.deepcopy(short_resolved)
    seed2["run"]["init_seed"] = 2
    unfinished = _dir(out, seed2)
    before = {f: h for f, h in snapshot(out).items() if f.startswith(unfinished + "/")}
    assert before, "the unfinished short run has no files"
    p = prune.plan(out, "opt", final_short=True)
    assert not [d for d in p.deletions if _rel(out, d.path).startswith(unfinished + "/")]
    assert prune.main(["--out", str(out), "--apply", "--final-short"]) == 0
    assert {f: h for f, h in snapshot(out).items() if f.startswith(unfinished + "/")} == before


def test_apply_refuses_a_planned_path_that_is_not_a_checkpoint_under_out(full, trunk_resolved, tmp_path):
    """The apply step's own check (prune.apply): a planned path that is no step or trunk-step file under
    OUT/checkpoints/MODE is refused before any file is deleted, even when other planned paths are valid."""
    out = _copy(full, tmp_path)
    p = prune.plan(out, "opt", final=True)
    valid = p.deletions[0]
    foreign = out / "manifests/opt" / f"{mf.run_name(trunk_resolved)}.json"
    outside = tmp_path / "elsewhere" / "step00000008.pt"
    outside.parent.mkdir()
    outside.write_bytes(b"not ours")
    before = snapshot(out)
    for bad in (foreign, outside):
        q = prune.Plan(out=p.out, mode=p.mode, final=p.final,
                       deletions=[valid, prune.Deletion(bad, "x", 1)])
        with pytest.raises(rc.EntryError):
            prune.apply(q)
        assert snapshot(out) == before and outside.read_bytes() == b"not ours"
