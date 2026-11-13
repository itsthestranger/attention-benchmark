"""Fixes for the session-4 review (docs/review_s4.md), checkpoint and resume part.

Written before the fixes (TASKS.md T-40, O+W). A worker implements against this file and
may not change it. SYNTHETIC data only, tiny models, budgets of 24 to 32 steps.

G1. A stop at `--until` left no checkpoint, so a trunk stopped at its branch point did not
    resume from it: it restarted from the last cadence checkpoint, ran the steps again and
    rewrote the branch-point checkpoint its branch had started from (plan §7.2, §12.3,
    §13.9). In `opt` mode on a GPU the steps run again are not bit-identical.
G2. A resume after the environment changed (torch, CUDA, driver, GPU) was not refused, and
    the manifest recorded only the last session's environment (plan §0 row 2, §10.6).
M2. A non-finite loss on the last training step made the manifest invalid for good.
B22, B45, B47. Three deliberate breaks of the review that no test caught.
"""

from __future__ import annotations

import hashlib
import math

import pytest
import torch

from scripts import _run_common as rc
from src.training import checkpoint as ck
from src.training import trainer as trainer_module
from src.utils import manifest as mf
from tests.test_checkpoint import poison_next_gradient, resolved_for, scramble, trainer_for
from tests.test_entry_points import (  # noqa: F401  (fixtures are used by name)
    BRANCH_STEP,
    RESUME_TOL,
    SHORT_LABEL,
    TRUNK_STEPS,
    assert_same_result,
    final_state,
    losses,
    manifest_of,
    reference,
    run_branch,
    run_train,
    snapshot,
    specs,
    synthetic,
    trunk_resolved,
)
from tests.test_trainer import make_trainer

ENV_FIELDS = ("torch", "cuda", "driver", "gpu_name")


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trunk_dir(out, resolved, mode="opt"):
    return ck.checkpoint_dir(out / "checkpoints" / mode, resolved)


@pytest.fixture
def resumes(monkeypatch):
    """Every ResumeInfo that `ck.resume_or_start` returns while the test runs."""
    seen: list[ck.ResumeInfo] = []
    real = ck.resume_or_start

    def spy(trainer, root):
        info = real(trainer, root)
        seen.append(info)
        return info

    monkeypatch.setattr(ck, "resume_or_start", spy)
    return seen


# --- G1, unit level: what a resume starts from, and what a branch-point save may replace --------


def test_a_branch_point_newer_than_every_regular_checkpoint_is_resumed_from(tmp_path):
    cfg = resolved_for(steps=20)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, directory)
    first.train(until=8)
    trunk = ck.save_checkpoint(first, directory, branch_point=True)
    second = trainer_for(cfg)
    info = ck.resume_or_start(second, tmp_path)
    assert info.step == 8 and info.path == trunk
    assert second.step == 8 and len(second.records) == 8
    for name, tensor in first.model.state_dict().items():
        assert torch.equal(second.model.state_dict()[name], tensor), name


def test_a_regular_checkpoint_newer_than_the_branch_point_is_resumed_from(tmp_path):
    cfg = resolved_for(steps=20)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, directory, branch_point=True)
    first.train(until=8)
    regular = ck.save_checkpoint(first, directory)
    info = ck.resume_or_start(trainer_for(cfg), tmp_path)
    assert info.step == 8 and info.path == regular


def test_at_the_same_step_the_regular_checkpoint_is_the_one_resumed_from(tmp_path):
    cfg = resolved_for(steps=20)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, directory, branch_point=True)
    regular = ck.save_checkpoint(first, directory)
    info = ck.resume_or_start(trainer_for(cfg), tmp_path)
    assert info.step == 5 and info.path == regular


def test_a_damaged_branch_point_is_skipped_by_a_resume(tmp_path):
    cfg = resolved_for(steps=20)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    first = trainer_for(cfg)
    first.train(until=5)
    regular = ck.save_checkpoint(first, directory)
    first.train(until=8)
    trunk = ck.save_checkpoint(first, directory, branch_point=True)
    trunk.write_bytes(trunk.read_bytes()[:100])
    info = ck.resume_or_start(trainer_for(cfg), tmp_path)
    assert info.step == 5 and info.path == regular


def test_latest_valid_checkpoint_still_reads_regular_checkpoints_only(tmp_path):
    """Branches and the completion check use it on a directory of regular checkpoints."""
    cfg = resolved_for(steps=20)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, directory, branch_point=True)
    assert ck.latest_valid_checkpoint(directory) is None


def test_an_existing_branch_point_is_never_overwritten(tmp_path):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    trainer.train(until=5)
    trunk = ck.save_checkpoint(trainer, tmp_path, branch_point=True)
    before = sha(trunk)
    # The same state again, and another state at the same step: neither may replace the file.
    with pytest.raises(ck.CheckpointError, match="trunk-step00000005"):
        ck.save_checkpoint(trainer, tmp_path, branch_point=True)
    scramble(trainer)
    with pytest.raises(ck.CheckpointError, match="trunk-step00000005"):
        ck.save_checkpoint(trainer, tmp_path, branch_point=True)
    assert sha(trunk) == before
    assert [p.name for p in tmp_path.iterdir()] == ["trunk-step00000005.pt"]  # no stray temporary file


def test_a_damaged_branch_point_is_replaced(tmp_path):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    trainer.train(until=5)
    trunk = ck.save_checkpoint(trainer, tmp_path, branch_point=True)
    trunk.write_bytes(trunk.read_bytes()[:100])
    again = ck.save_checkpoint(trainer, tmp_path, branch_point=True)
    assert again == trunk and ck.load_checkpoint(trunk)["step"] == 5


def test_a_regular_checkpoint_may_still_be_written_twice(tmp_path):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    trainer.train(until=5)
    ck.save_checkpoint(trainer, tmp_path)
    ck.save_checkpoint(trainer, tmp_path)
    assert [p.name for p in ck.list_checkpoints(tmp_path)] == ["step00000005.pt"]


# --- G1, entry points: a stop at --until can be resumed from --------------------------------------


def test_a_stop_at_until_leaves_a_checkpoint_at_that_step(specs, synthetic, trunk_resolved, reference, resumes, tmp_path):
    out = tmp_path / "stopped"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "13")) == 0  # 13 is on no cadence
    names = [p.name for p in ck.list_checkpoints(trunk_dir(out, trunk_resolved))]
    assert names[-1] == "step00000013.pt"
    assert ck.load_checkpoint(trunk_dir(out, trunk_resolved) / names[-1])["step"] == 13
    assert run_train(specs.trunk, out, synthetic.root) == 0
    assert [info.step for info in resumes] == [0, 13]
    name = mf.run_name(trunk_resolved)
    assert manifest_of(out, "opt", name)["completed"] is True
    assert_same_result(out, reference("opt"), "opt", trunk_resolved, RESUME_TOL["opt"], name)


def test_the_checkpoint_left_at_a_stop_stores_the_resume_count(specs, synthetic, trunk_resolved, tmp_path):
    """Found by a deliberate break in review: the stop checkpoint carries the session's resume count."""
    out = tmp_path / "counted"
    directory = trunk_dir(out, trunk_resolved)
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "13")) == 0
    assert ck.load_checkpoint(directory / "step00000013.pt")["resume_count"] == 0
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "21")) == 0
    assert ck.load_checkpoint(directory / "step00000021.pt")["resume_count"] == 1


@pytest.mark.parametrize("mode", ["ref", "opt"])
def test_a_trunk_stopped_at_its_branch_point_resumes_from_it_and_keeps_it(
    mode, specs, synthetic, trunk_resolved, reference, resumes, tmp_path
):
    """The runbook's order: trunk to the branch point, the branch, then the trunk to the end."""
    out = tmp_path / "trunk"
    assert run_train(specs.trunk, out, synthetic.root, mode=mode, extra=("--until", str(BRANCH_STEP))) == 0
    trunk = trunk_dir(out, trunk_resolved, mode) / f"trunk-step{BRANCH_STEP:08d}.pt"
    before = sha(trunk)
    assert run_branch(specs.trunk, out, synthetic.root, mode=mode) == 0
    assert run_train(specs.trunk, out, synthetic.root, mode=mode) == 0
    assert [info.step for info in resumes] == [0, BRANCH_STEP]  # the two trunk sessions
    assert sha(trunk) == before
    name = mf.run_name(trunk_resolved)
    m = manifest_of(out, mode, name)
    assert m["completed"] is True and m["resume_count"] == 1
    assert m["steps"]["attempted"] == TRUNK_STEPS and len(m["step_seconds"]) == TRUNK_STEPS
    assert_same_result(out, reference(mode), mode, trunk_resolved, RESUME_TOL[mode], name)
    branch = manifest_of(out, mode, mf.run_name(trunk_resolved, SHORT_LABEL))
    assert branch["completed"] is True


def test_a_trunk_killed_right_after_its_branch_point_resumes_from_the_branch_point(
    specs, synthetic, trunk_resolved, resumes, tmp_path
):
    """Only the branch-point file survives at that step: the trunk must continue from it, not behind it."""
    out = tmp_path / "killed"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", str(BRANCH_STEP))) == 0
    directory = trunk_dir(out, trunk_resolved)
    (directory / f"step{BRANCH_STEP:08d}.pt").unlink()
    trunk = directory / f"trunk-step{BRANCH_STEP:08d}.pt"
    before = sha(trunk)
    assert run_train(specs.trunk, out, synthetic.root) == 0
    assert resumes[-1].step == BRANCH_STEP and resumes[-1].path == trunk
    assert sha(trunk) == before


# --- G2: the environment travels with the checkpoint, and a resume under another is refused --------


def test_current_environment_names_the_fields_a_resume_compares():
    env = ck.current_environment()
    assert set(ENV_FIELDS) <= set(env)
    assert env["torch"] == str(torch.__version__)
    full = mf.collect_environment()
    assert {key: env[key] for key in ENV_FIELDS} == {key: full[key] for key in ENV_FIELDS}


def test_a_checkpoint_stores_the_environment_it_was_written_in(tmp_path, monkeypatch):
    marked = {**ck.current_environment(), "torch": "0.0.1-synthetic", "gpu_name": "Synthetic T4"}
    monkeypatch.setattr(ck, "current_environment", lambda: dict(marked))
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    trainer.train(until=5)
    for branch_point in (False, True):
        state = ck.load_checkpoint(ck.save_checkpoint(trainer, tmp_path, branch_point=branch_point))
        assert state["environment"] == marked


def test_a_checkpoint_without_an_environment_does_not_load(tmp_path):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    trainer.train(until=5)
    path = ck.save_checkpoint(trainer, tmp_path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["environment"]
    torch.save(state, path)
    with pytest.raises(ck.CheckpointError, match="environment"):
        ck.load_checkpoint(path)


@pytest.mark.parametrize("field", ENV_FIELDS)
def test_a_resume_under_another_environment_is_refused_and_changes_nothing(field, tmp_path, monkeypatch):
    cfg = resolved_for(steps=20)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, ck.checkpoint_dir(tmp_path, cfg))
    saved = ck.current_environment()
    other = {**saved, field: "another-synthetic-value"}
    monkeypatch.setattr(ck, "current_environment", lambda: dict(other))
    second = trainer_for(cfg)
    with pytest.raises(ck.CheckpointError) as caught:
        ck.resume_or_start(second, tmp_path)  # refused: not a silent start from step 0
    message = str(caught.value)
    assert field in message and "another-synthetic-value" in message and repr(saved[field]) in message
    assert second.step == 0 and second.records == []


def test_a_field_the_resume_does_not_compare_may_change(tmp_path, monkeypatch):
    cfg = resolved_for(steps=20)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, ck.checkpoint_dir(tmp_path, cfg))
    other = {**ck.current_environment(), "python": "9.9.9-synthetic"}
    monkeypatch.setattr(ck, "current_environment", lambda: dict(other))
    assert ck.resume_or_start(trainer_for(cfg), tmp_path).step == 5


def test_the_entry_point_refuses_a_resume_after_the_image_changed(
    specs, synthetic, trunk_resolved, reference, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "image"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "13")) == 0
    real = ck.current_environment
    monkeypatch.setattr(ck, "current_environment", lambda: {**real(), "torch": "9.9.9-synthetic"})
    kept = {key: value for key, value in snapshot(out).items() if key.startswith(("checkpoints", "manifests"))}
    capsys.readouterr()
    assert run_train(specs.trunk, out, synthetic.root) == 2
    err = capsys.readouterr().err
    assert "torch" in err and "9.9.9-synthetic" in err
    after = {key: value for key, value in snapshot(out).items() if key.startswith(("checkpoints", "manifests"))}
    assert after == kept
    # Back under the environment the run started in, the same command finishes it.
    monkeypatch.setattr(ck, "current_environment", real)
    assert run_train(specs.trunk, out, synthetic.root) == 0
    name = mf.run_name(trunk_resolved)
    assert_same_result(out, reference("opt"), "opt", trunk_resolved, RESUME_TOL["opt"], name)


def test_a_branch_is_refused_when_the_trunk_ran_under_another_environment(
    specs, synthetic, trunk_resolved, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "branch"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", str(BRANCH_STEP))) == 0
    real = ck.current_environment
    monkeypatch.setattr(ck, "current_environment", lambda: {**real(), "driver": "999.9-synthetic"})
    capsys.readouterr()
    assert run_branch(specs.trunk, out, synthetic.root) == 2
    err = capsys.readouterr().err
    assert "driver" in err and "999.9-synthetic" in err
    assert not (out / "manifests" / "opt" / f"{mf.run_name(trunk_resolved, SHORT_LABEL)}.json").exists()


# --- M2: a non-finite loss on the last step does not block the manifest ----------------------------


def test_a_non_finite_loss_on_the_last_step_still_gives_a_valid_manifest(
    specs, synthetic, trunk_resolved, tmp_path, monkeypatch
):
    """The step is skipped like any other with a non-finite gradient; the weights are intact."""
    real_step = trainer_module.Trainer.train_step
    real_ce = trainer_module.F.cross_entropy

    def step(self):
        if self.step == self.total_steps - 1:
            monkeypatch.setattr(trainer_module.F, "cross_entropy", lambda *a, **k: real_ce(*a, **k) * float("nan"))
        try:
            return real_step(self)
        finally:
            monkeypatch.setattr(trainer_module.F, "cross_entropy", real_ce)

    monkeypatch.setattr(trainer_module.Trainer, "train_step", step)
    out = tmp_path / "lastnan"
    # No in-run validation point on the last step (32 is not a multiple of 5): the patched loss
    # function is the trainer's alone.
    assert run_train(specs.trunk, out, synthetic.root, eval_every=5) == 0
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert mf.validate_manifest(m) == []
    assert m["completed"] is True and m["steps"]["skipped"] == 1 and m["steps"]["taken"] == TRUNK_STEPS - 1
    recorded = losses(final_state(out, trunk_resolved, "opt"))
    assert math.isnan(recorded[-1]) and math.isfinite(recorded[-2])
    # train_loss is the loss of the last step whose loss is finite.
    assert m["final_metrics"]["train_loss"] == recorded[-2]
    assert math.isfinite(m["final_metrics"]["val_loss"])
    # The same command again finds the run complete.
    assert run_train(specs.trunk, out, synthetic.root, eval_every=5) == 0


# --- B22, B45, B47: breaks of the review that no test caught ---------------------------------------


def test_gpu_shared_is_read_from_the_flag_on_cuda_and_refused_on_the_cpu():
    """B22. Only the device's type is read, so no CUDA device is needed here."""
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    assert rc.gpu_shared_for(cuda, "yes") is True
    assert rc.gpu_shared_for(cuda, "no") is False
    with pytest.raises(rc.EntryError):
        rc.gpu_shared_for(cuda, None)
    with pytest.raises(rc.EntryError):
        rc.gpu_shared_for(cpu, "yes")
    assert rc.gpu_shared_for(cpu, None) is False and rc.gpu_shared_for(cpu, "no") is False


def test_the_manifest_records_what_the_flag_said_about_sharing(specs, synthetic, trunk_resolved, tmp_path, monkeypatch):
    """B22, the other half: the value reaches the manifest. The CPU cannot say yes, so the answer is patched."""
    monkeypatch.setattr(rc, "gpu_shared_for", lambda device, value: True)
    out = tmp_path / "shared"
    assert run_train(specs.trunk, out, synthetic.root) == 0
    assert manifest_of(out, "opt", mf.run_name(trunk_resolved))["gpu_shared"] is True


def test_skipped_in_first_100_counts_steps_0_to_99():
    """B45: plan §7.3 reads the first 100 steps; the hundred-and-first is not one of them."""
    trainer = make_trainer(steps=102)
    for step in range(102):
        handle = poison_next_gradient(trainer, float("nan")) if step in (0, 99, 100, 101) else None
        trainer.train_step()
        if handle:
            handle.remove()
    summary = trainer.summary()
    assert summary["skipped_steps"] == 4
    assert summary["skipped_in_first_100"] == 2


def test_the_in_run_validation_reads_the_configured_number_of_windows(
    specs, synthetic, trunk_resolved, tmp_path, monkeypatch
):
    """B47: `--eval-curve-windows` bounds every in-run validation point; the final evaluation is unbounded."""
    seen: list = []
    real = rc.evaluate_split

    def spy(model, split, settings, **kwargs):
        seen.append(kwargs.get("max_windows"))
        return real(model, split, settings, **kwargs)

    monkeypatch.setattr(rc, "evaluate_split", spy)
    out = tmp_path / "windows"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--eval-curve-windows", "3")) == 0
    in_run = len(manifest_of(out, "opt", mf.run_name(trunk_resolved))["final_metrics"]["val_curve"])
    assert in_run > 0
    assert seen == [3] * in_run + [None, None]  # then the full validation and test splits
