"""T-16: checkpoint and resume (plan §5 "Resume fidelity", §12.3, §13.9). SYNTHETIC data only.

Written by the orchestrator before the implementation; the worker may not
change this file.

Plan §5: "Resume and branch fidelity are the tests people skip. On Kaggle you
will resume constantly ... A resume must restore data position, schedule
position, optimizer state, GradScaler state and RNG state. Most hand-rolled
trainers get the data position or the schedule step wrong first time, and the
curve looks plausible rather than broken."

Plan §12.3: resume automatically from the config hash, with no human decision;
keep the last two checkpoints, because one written during a crash can be
truncated; keep every branch-point checkpoint.

The contract for `src/training/checkpoint.py`
---------------------------------------------
    class CheckpointError(RuntimeError)

    SCHEMA_VERSION = 1

    def checkpoint_dir(root, resolved) -> Path
        root / config_hash(resolved). One directory per configuration.

    def save_checkpoint(trainer, directory, *, branch_point=False, keep=2, resume_count=0) -> Path
        Writes the trainer's whole state to `step{step:08d}.pt`, or to
        `trunk-step{step:08d}.pt` when `branch_point` is true, creating the
        directory. The write is atomic: a temporary file in the same directory,
        then os.replace. Afterwards only the newest `keep` files named
        `step*.pt` remain; older ones are deleted. `trunk-step*.pt` files are
        never deleted here, and do not count towards `keep`. keep < 1 raises
        ValueError. Leaves no temporary file behind.

    def list_checkpoints(directory) -> list[Path]
        The `step*.pt` files, oldest first. A missing directory gives [].

    def list_branch_points(directory) -> list[Path]
        The `trunk-step*.pt` files, oldest first.

    def load_checkpoint(path) -> dict
        The saved state. Raises CheckpointError if the file is missing,
        truncated, not a checkpoint, or of another schema version. The dict has
        at least: "schema_version", "config_hash", "resolved", "step",
        "steps_taken", "mode", "precision", "model", "optimizer", "scaler"
        (None when scaling is off), "records" (one entry per attempted step),
        "torch_rng_state", "resume_count", "branch_point".

    def latest_valid_checkpoint(directory) -> tuple[Path, dict] | None
        The newest `step*.pt` file that loads, with its state. A file that does
        not load is skipped, not deleted. None if there is none.

    def restore(trainer, state) -> None
        Puts a loaded state into a trainer built from the same resolved config:
        model weights, optimizer state, scaler state, `step`, `steps_taken`,
        `records` (as StepRecord objects) and the torch RNG state. Raises
        CheckpointError, changing nothing, if the state's config hash, mode or
        precision differ from the trainer's, or if the trainer has already
        taken a step.

    @dataclass(frozen=True)
    class ResumeInfo:
        step: int                 # 0 for a fresh start
        path: Path | None         # the checkpoint restored from
        resume_count: int         # how many times this run has been resumed, this time included

    def resume_or_start(trainer, root) -> ResumeInfo
        Looks in checkpoint_dir(root, trainer.resolved) and restores the latest
        valid checkpoint if there is one. No argument says which checkpoint: the
        configuration decides (plan §12.3, "no human decision").

    class Checkpointer:
        def __init__(self, root, *, every_steps=None, every_seconds=None,
                     branch_steps=(), keep=2, resume_count=0, clock=time.monotonic)
        def __call__(self, trainer, record) -> None      # an `on_step` hook
        saved: list[Path]                                 # what this object wrote, in order

        After each step it saves a checkpoint when `every_steps` steps have been
        attempted since the last save (trainer.step is a multiple of it), or
        when `every_seconds` have passed on `clock` since the last save, and
        always after the run's final step. When trainer.step equals one of
        `branch_steps` it also saves a branch-point checkpoint. At least one of
        `every_steps` and `every_seconds` must be given.

Nothing in the module asks a question, reads the global RNG to make a
decision, or keeps the position of the data stream anywhere but in
`trainer.step`: the stream and the schedule are functions of the step
(decisions 29, 54, 66).
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from src import config
from src.data.loader import TrainStream
from src.model import build_model
from src.training import checkpoint as ck
from src.training.schedule import decay_start, lr_at
from src.training.trainer import StepRecord, Trainer
from tests.fixtures.synthetic_tokens import rule_split
from tests.tolerances import RESUME_FIDELITY_FP16, RESUME_FIDELITY_FP32, scale_relative_error
from tests.variants import CASES, tiny_resolved

VOCAB, T, B = 128, 16, 4
SCALER = {"recipe.grad_scaler.init_scale": 65536.0, "recipe.grad_scaler.growth_interval": 2000}


def resolved_for(case=CASES[0], *, steps=60, init_seed=1, **overrides):
    extra = {
        "recipe.batch.sequences_per_step": B,
        "recipe.batch.sequence_length": T,
        "recipe.batch.tokens_per_step": B * T,
        "recipe.batch.micro_batch": 2,
        "run.steps": steps,
        "run.branches": [],
    }
    extra.update(overrides)
    return tiny_resolved(case, init_seed=init_seed, context_length=T, **extra)


def stream_for(steps):
    return TrainStream(rule_split(steps * B * T + 1, VOCAB), context_length=T, sequences_per_step=B, data_seed=1)


def trainer_for(cfg, *, stream=None, **kwargs):
    return Trainer(build_model(cfg), cfg, stream or stream_for(cfg["run"]["steps"]), **kwargs)


def scramble(trainer):
    """Make sure a restore has something to restore: move every weight."""
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(1.0)


def poison_next_gradient(trainer, value):
    weight = trainer.model.layers[0].mlp.gate.weight
    return weight.register_hook(lambda g: torch.full_like(g, value))


def assert_same_curve(resumed, reference, tolerance):
    assert [r.step for r in resumed] == [r.step for r in reference]
    for a, b in zip(resumed, reference):
        assert a.lr == b.lr, a.step
        assert a.skipped == b.skipped, a.step
        assert scale_relative_error(a.loss, b.loss) <= tolerance, a.step
        assert scale_relative_error(a.grad_norm, b.grad_norm) <= tolerance, a.step


class Killed(Exception):
    pass


# --- Resume fidelity (plan §5) --------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_save_and_restore_continue_the_uninterrupted_curve(case, tmp_path):
    cfg = resolved_for(case, steps=60)
    reference = trainer_for(cfg)
    reference.train()

    first = trainer_for(cfg)
    first.train(until=25)
    path = ck.save_checkpoint(first, tmp_path)

    second = trainer_for(cfg)
    scramble(second)
    ck.restore(second, ck.load_checkpoint(path))
    assert second.step == 25 and second.steps_taken == 25
    second.train()
    assert second.step == 60
    assert_same_curve(second.records, reference.records, RESUME_FIDELITY_FP32)
    for (name, a), (_, b) in zip(second.model.named_parameters(), reference.model.named_parameters()):
        assert scale_relative_error(a.detach(), b.detach()) <= RESUME_FIDELITY_FP32, name


def test_kill_at_step_500_and_resume_gives_the_uninterrupted_curve(tmp_path):
    """The plan's test, literally: a run of 600 steps killed at step 500, inside its decay,
    with checkpoints every 64 steps, resumed with no instruction but the config."""
    cfg = resolved_for(steps=600)
    assert decay_start(600, 0.2) == 480  # the kill and the resume are both inside the decay
    reference = trainer_for(cfg)
    reference.train()

    saver = ck.Checkpointer(tmp_path, every_steps=64)

    def on_step(trainer, record):
        saver(trainer, record)
        if trainer.step == 500:
            raise Killed

    doomed = trainer_for(cfg, on_step=on_step)
    with pytest.raises(Killed):
        doomed.train()
    assert doomed.step == 500

    survivor = trainer_for(cfg)
    scramble(survivor)
    info = ck.resume_or_start(survivor, tmp_path)
    assert info.step == 448 and survivor.step == 448  # the last multiple of 64 before the kill
    assert info.resume_count == 1
    assert info.path == ck.checkpoint_dir(tmp_path, cfg) / "step00000448.pt"
    survivor.train()
    assert survivor.step == 600 and survivor.steps_taken == 600
    assert_same_curve(survivor.records, reference.records, RESUME_FIDELITY_FP32)
    # The schedule position came back with the step: the resumed steps decay as scheduled.
    assert [r.lr for r in survivor.records[448:]] == [lr_at(s, cfg) for s in range(448, 600)]
    assert survivor.records[-1].lr < survivor.records[479].lr


def test_resume_reads_the_same_tokens_the_uninterrupted_run_would(tmp_path):
    """Data position (plan §13.9): after a restore the stream is asked for exactly the
    steps that were not yet attempted, each once."""
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner, self.max_steps = inner, inner.max_steps

        def batch(self, step):
            seen.append(step)
            return self.inner.batch(step)

    cfg = resolved_for(steps=30)
    first = trainer_for(cfg)
    first.train(until=11)
    path = ck.save_checkpoint(first, tmp_path)
    second = trainer_for(cfg, stream=Recording(stream_for(30)))
    ck.restore(second, ck.load_checkpoint(path))
    second.train()
    assert seen == list(range(11, 30))


def test_optimizer_state_is_restored(tmp_path):
    """Without AdamW's moments the next update differs although weights and step match."""
    cfg = resolved_for(steps=30)
    first = trainer_for(cfg)
    first.train(until=12)
    path = ck.save_checkpoint(first, tmp_path)
    second = trainer_for(cfg)
    ck.restore(second, ck.load_checkpoint(path))
    saved, restored = first.optimizer.state_dict()["state"], second.optimizer.state_dict()["state"]
    assert saved.keys() == restored.keys() and len(saved) > 0
    for key in saved:
        assert restored[key]["step"] == saved[key]["step"] == 12
        assert torch.equal(restored[key]["exp_avg"], saved[key]["exp_avg"])
        assert torch.equal(restored[key]["exp_avg_sq"], saved[key]["exp_avg_sq"])
    # The restored optimizer drives the restored model's parameters, not copies of them.
    owned = {id(p) for g in second.optimizer.param_groups for p in g["params"]}
    assert owned == {id(p) for p in second.model.parameters()}


def test_skipped_steps_survive_a_resume(tmp_path):
    """Plan §13.6: steps taken are compared across runs, so a resume must not lose a skip."""
    cfg = resolved_for(steps=20)
    first = trainer_for(cfg)
    first.train(until=3)
    handle = poison_next_gradient(first, float("inf"))
    assert first.train_step().skipped is True
    handle.remove()
    first.train(until=8)
    path = ck.save_checkpoint(first, tmp_path)

    second = trainer_for(cfg)
    ck.restore(second, ck.load_checkpoint(path))
    assert second.step == 8 and second.steps_taken == 7 and second.skipped_steps == 1
    second.train()
    summary = second.summary()
    assert summary["steps_attempted"] == 20 and summary["steps_taken"] == 19
    assert summary["skipped_steps"] == 1 and summary["skipped_in_first_100"] == 1
    assert [r.skipped for r in second.records] == [i == 3 for i in range(20)]
    assert all(isinstance(r, StepRecord) for r in second.records)
    assert second.records[:8] == first.records


def test_torch_rng_state_is_restored(tmp_path):
    cfg = resolved_for(steps=10)
    first = trainer_for(cfg)
    first.train(until=4)
    generator_state = torch.get_rng_state()
    path = ck.save_checkpoint(first, tmp_path)
    torch.rand(5)  # the global generator moves on
    assert not torch.equal(torch.get_rng_state(), generator_state)
    second = trainer_for(cfg)
    ck.restore(second, ck.load_checkpoint(path))
    assert torch.equal(torch.get_rng_state(), generator_state)


def test_a_loaded_state_is_a_value(tmp_path):
    """Added with T-17, which restores one loaded state more than once. Training a restored
    trainer must not change the state it was restored from, nor a second trainer restored
    from the same state."""
    cfg = resolved_for(steps=30)
    first = trainer_for(cfg)
    first.train(until=12)
    state = ck.load_checkpoint(ck.save_checkpoint(first, tmp_path))
    moments = {k: v["exp_avg"].clone() for k, v in state["optimizer"]["state"].items()}
    weights = {k: v.clone() for k, v in state["model"].items()}

    a, b = trainer_for(cfg), trainer_for(cfg)
    ck.restore(a, state)
    ck.restore(b, state)
    a.train(until=20)
    for key, value in moments.items():
        assert torch.equal(state["optimizer"]["state"][key]["exp_avg"], value)
    for key, value in weights.items():
        assert torch.equal(state["model"][key], value)
    b.train(until=20)
    assert [r.loss for r in a.records] == [r.loss for r in b.records]
    assert [r.loss for r in a.records[12:]] == [r.loss for r in first_continued(cfg, first)]


def first_continued(cfg, first):
    first.train(until=20)
    return first.records[12:]


def test_saving_changes_nothing_in_the_trainer(tmp_path):
    cfg = resolved_for(steps=30)
    plain, saving = trainer_for(cfg), trainer_for(cfg)
    plain.train()
    for stop in (7, 8, 20, 30):
        saving.train(until=stop)
        ck.save_checkpoint(saving, tmp_path)
    assert [r.loss for r in saving.records] == [r.loss for r in plain.records]


# --- What a checkpoint holds --------------------------------------------------------------------


def test_checkpoint_contents(tmp_path):
    cfg = resolved_for(steps=10)
    trainer = trainer_for(cfg)
    trainer.train(until=6)
    path = ck.save_checkpoint(trainer, tmp_path, resume_count=2)
    assert path == tmp_path / "step00000006.pt"
    state = ck.load_checkpoint(path)
    assert state["schema_version"] == ck.SCHEMA_VERSION == 1
    assert state["config_hash"] == config.config_hash(cfg)
    assert state["resolved"] == cfg  # the full config travels with every checkpoint (plan §11)
    assert state["step"] == 6 and state["steps_taken"] == 6
    assert state["mode"] == "opt" and state["precision"] == "fp32"
    assert state["scaler"] is None
    assert state["resume_count"] == 2 and state["branch_point"] is False
    assert len(state["records"]) == 6
    assert set(state["model"]) == set(trainer.model.state_dict())
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(state["model"][name], value), name
    assert state["model"]["embed.weight"].dtype == torch.float32  # fp32 master weights


def test_checkpoint_directory_is_keyed_by_the_config_hash(tmp_path):
    a, b = resolved_for(steps=10), resolved_for(steps=10, init_seed=2)
    assert ck.checkpoint_dir(tmp_path, a) == tmp_path / config.config_hash(a)
    assert ck.checkpoint_dir(tmp_path, a) != ck.checkpoint_dir(tmp_path, b)


# --- The last two checkpoints are kept (plan §12.3) -----------------------------------------------


def test_only_the_last_two_checkpoints_are_kept(tmp_path):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    for stop in (3, 6, 9, 12):
        trainer.train(until=stop)
        ck.save_checkpoint(trainer, tmp_path)
    assert [p.name for p in ck.list_checkpoints(tmp_path)] == ["step00000009.pt", "step00000012.pt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step00000009.pt", "step00000012.pt"]  # no temp files


def test_keep_is_respected_and_must_be_positive(tmp_path):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    for stop in (3, 6, 9, 12):
        trainer.train(until=stop)
        ck.save_checkpoint(trainer, tmp_path, keep=3)
    assert [p.name for p in ck.list_checkpoints(tmp_path)] == [
        "step00000006.pt", "step00000009.pt", "step00000012.pt",
    ]
    with pytest.raises(ValueError):
        ck.save_checkpoint(trainer, tmp_path, keep=0)


def test_branch_point_checkpoints_are_never_pruned(tmp_path):
    """Plan §12.3: every branch-point checkpoint stays until its branch is done and verified."""
    cfg = resolved_for(steps=30)
    trainer = trainer_for(cfg)
    trainer.train(until=5)
    trunk = ck.save_checkpoint(trainer, tmp_path, branch_point=True)
    assert trunk.name == "trunk-step00000005.pt"
    assert ck.load_checkpoint(trunk)["branch_point"] is True
    for stop in (6, 9, 12, 15, 18):
        trainer.train(until=stop)
        ck.save_checkpoint(trainer, tmp_path)
    assert [p.name for p in ck.list_branch_points(tmp_path)] == ["trunk-step00000005.pt"]
    assert [p.name for p in ck.list_checkpoints(tmp_path)] == ["step00000015.pt", "step00000018.pt"]


def test_listing_orders_by_step_and_tolerates_a_missing_directory(tmp_path):
    assert ck.list_checkpoints(tmp_path / "nothing") == []
    assert ck.list_branch_points(tmp_path / "nothing") == []
    assert ck.latest_valid_checkpoint(tmp_path / "nothing") is None
    cfg = resolved_for(steps=120)
    trainer = trainer_for(cfg)
    for stop in (9, 10, 101):
        trainer.train(until=stop)
        ck.save_checkpoint(trainer, tmp_path, keep=5)
    (tmp_path / "notes.txt").write_text("not a checkpoint")
    assert [p.name for p in ck.list_checkpoints(tmp_path)] == [
        "step00000009.pt", "step00000010.pt", "step00000101.pt",
    ]


# --- A checkpoint written during a crash (plan §12.3) ---------------------------------------------


@pytest.mark.parametrize("damage", ["truncate", "empty", "garbage"])
def test_a_damaged_newest_checkpoint_is_skipped(tmp_path, damage):
    cfg = resolved_for(steps=40)
    reference = trainer_for(cfg)
    reference.train()

    first = trainer_for(cfg)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    for stop in (10, 20):
        first.train(until=stop)
        ck.save_checkpoint(first, directory)
    newest = directory / "step00000020.pt"
    data = newest.read_bytes()
    newest.write_bytes({"truncate": data[: len(data) // 2], "empty": b"", "garbage": b"\x00" * 64}[damage])

    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(newest)
    path, state = ck.latest_valid_checkpoint(directory)
    assert path.name == "step00000010.pt" and state["step"] == 10
    assert newest.exists()  # skipped, not deleted

    second = trainer_for(cfg)
    info = ck.resume_or_start(second, tmp_path)
    assert info.step == 10 and info.path == path
    second.train()
    assert_same_curve(second.records, reference.records, RESUME_FIDELITY_FP32)


def test_loading_refuses_what_is_not_a_checkpoint(tmp_path):
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(tmp_path / "step00000001.pt")  # missing
    other = tmp_path / "step00000002.pt"
    torch.save({"hello": "world"}, other)
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(other)
    cfg = resolved_for(steps=10)
    trainer = trainer_for(cfg)
    trainer.train(until=3)
    path = ck.save_checkpoint(trainer, tmp_path)
    state = torch.load(path, weights_only=False)
    state["schema_version"] = 2
    torch.save(state, path)
    with pytest.raises(ck.CheckpointError):
        ck.load_checkpoint(path)


def test_a_failed_write_leaves_the_previous_checkpoints_alone(tmp_path, monkeypatch):
    cfg = resolved_for(steps=20)
    trainer = trainer_for(cfg)
    trainer.train(until=4)
    ck.save_checkpoint(trainer, tmp_path)
    trainer.train(until=8)

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        ck.save_checkpoint(trainer, tmp_path)
    monkeypatch.undo()
    assert [p.name for p in tmp_path.iterdir()] == ["step00000004.pt"]  # no partial file, no temp file
    assert ck.latest_valid_checkpoint(tmp_path)[1]["step"] == 4


# --- Restore refuses what does not match ----------------------------------------------------------


def test_restore_refuses_another_configuration(tmp_path):
    cfg = resolved_for(steps=20)
    first = trainer_for(cfg)
    first.train(until=5)
    state = ck.load_checkpoint(ck.save_checkpoint(first, tmp_path))
    for other in (resolved_for(steps=20, init_seed=2), resolved_for(steps=21), resolved_for(CASES[1], steps=20)):
        target = trainer_for(other)
        before = {n: p.detach().clone() for n, p in target.model.named_parameters()}
        with pytest.raises(ck.CheckpointError):
            ck.restore(target, state)
        assert target.step == 0 and target.records == []
        for name, p in target.model.named_parameters():
            assert torch.equal(p, before[name]), name


def test_restore_refuses_another_mode_and_a_trainer_that_has_started(tmp_path):
    cfg = resolved_for(steps=20)
    first = trainer_for(cfg)
    first.train(until=5)
    state = ck.load_checkpoint(ck.save_checkpoint(first, tmp_path))
    with pytest.raises(ck.CheckpointError):
        ck.restore(trainer_for(cfg, mode="ref"), state)
    started = trainer_for(cfg)
    started.train(until=1)
    with pytest.raises(ck.CheckpointError):
        ck.restore(started, state)


# --- Resume without a human (plan §12.3) ----------------------------------------------------------


def test_a_fresh_start_when_there_is_nothing_to_resume(tmp_path):
    cfg = resolved_for(steps=10)
    trainer = trainer_for(cfg)
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    info = ck.resume_or_start(trainer, tmp_path)
    assert info == ck.ResumeInfo(step=0, path=None, resume_count=0)
    assert trainer.step == 0
    for name, p in trainer.model.named_parameters():
        assert torch.equal(p, before[name]), name


def test_resume_picks_the_run_by_its_configuration(tmp_path):
    """Two runs share one root. Each resumes its own checkpoint, by config hash alone."""
    cfg_a, cfg_b = resolved_for(steps=20), resolved_for(steps=20, init_seed=2)
    a, b = trainer_for(cfg_a), trainer_for(cfg_b)
    a.train(until=6)
    b.train(until=9)
    ck.save_checkpoint(a, ck.checkpoint_dir(tmp_path, cfg_a))
    ck.save_checkpoint(b, ck.checkpoint_dir(tmp_path, cfg_b))
    again_a, again_b = trainer_for(cfg_a), trainer_for(cfg_b)
    assert ck.resume_or_start(again_a, tmp_path).step == 6
    assert ck.resume_or_start(again_b, tmp_path).step == 9
    for (name, p), (_, q) in zip(again_a.model.named_parameters(), a.model.named_parameters()):
        assert torch.equal(p, q), name
    import inspect

    assert list(inspect.signature(ck.resume_or_start).parameters) == ["trainer", "root"]


def test_resume_count_goes_up_with_every_resume(tmp_path):
    cfg = resolved_for(steps=30)
    counts = []
    stops = [8, 16, 24, 30]
    for stop in stops:
        trainer = trainer_for(cfg)
        info = ck.resume_or_start(trainer, tmp_path)
        counts.append(info.resume_count)
        trainer.on_step = ck.Checkpointer(tmp_path, every_steps=4, resume_count=info.resume_count)
        trainer.train(until=stop)
    assert counts == [0, 1, 2, 3]
    path, state = ck.latest_valid_checkpoint(ck.checkpoint_dir(tmp_path, cfg))
    assert state["step"] == 30 and state["resume_count"] == 3


def test_a_branch_point_alone_is_not_resumed_from(tmp_path):
    """Branch points are for branches (T-17). A resume uses the regular checkpoints."""
    cfg = resolved_for(steps=20)
    first = trainer_for(cfg)
    first.train(until=5)
    ck.save_checkpoint(first, ck.checkpoint_dir(tmp_path, cfg), branch_point=True)
    assert ck.resume_or_start(trainer_for(cfg), tmp_path).step == 0


# --- The Checkpointer hook --------------------------------------------------------------------------


def test_checkpointer_saves_every_n_steps_and_at_the_end(tmp_path):
    cfg = resolved_for(steps=22)
    saver = ck.Checkpointer(tmp_path, every_steps=5, keep=10)
    trainer = trainer_for(cfg, on_step=saver)
    trainer.train()
    directory = ck.checkpoint_dir(tmp_path, cfg)
    names = [f"step{s:08d}.pt" for s in (5, 10, 15, 20, 22)]
    assert [p.name for p in ck.list_checkpoints(directory)] == names
    assert saver.saved == [directory / n for n in names]
    assert ck.load_checkpoint(directory / names[-1])["step"] == 22


def test_checkpointer_saves_on_the_clock(tmp_path):
    """Plan §12.3: often enough to lose at most about ten minutes."""
    now = [0.0]
    cfg = resolved_for(steps=12)
    saver = ck.Checkpointer(tmp_path, every_seconds=600.0, keep=10, clock=lambda: now[0])

    def on_step(trainer, record):
        now[0] += 250.0  # every step "takes" 250 seconds
        saver(trainer, record)

    trainer = trainer_for(cfg, on_step=on_step)
    trainer.train()
    # 750 s have passed after steps 3, 6, 9; step 12 is the final step.
    assert [ck.load_checkpoint(p)["step"] for p in saver.saved] == [3, 6, 9, 12]


def test_checkpointer_saves_branch_points(tmp_path):
    cfg = resolved_for(steps=20)
    saver = ck.Checkpointer(tmp_path, every_steps=8, branch_steps=(6, 13))
    trainer_for(cfg, on_step=saver).train()
    directory = ck.checkpoint_dir(tmp_path, cfg)
    assert [p.name for p in ck.list_branch_points(directory)] == [
        "trunk-step00000006.pt", "trunk-step00000013.pt",
    ]
    assert [p.name for p in ck.list_checkpoints(directory)] == ["step00000016.pt", "step00000020.pt"]
    trunk = ck.load_checkpoint(directory / "trunk-step00000013.pt")
    assert trunk["step"] == 13 and trunk["branch_point"] is True
    assert len(trunk["records"]) == 13


def test_checkpointer_needs_a_cadence(tmp_path):
    with pytest.raises(ValueError):
        ck.Checkpointer(tmp_path)
    for bad in ({"every_steps": 0}, {"every_seconds": 0.0}, {"every_steps": 5, "keep": 0}):
        with pytest.raises(ValueError):
            ck.Checkpointer(tmp_path, **bad)


def test_checkpointing_does_not_touch_the_global_rng(tmp_path):
    cfg = resolved_for(steps=10)
    trainer = trainer_for(cfg, on_step=ck.Checkpointer(tmp_path, every_steps=2))
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    trainer.train()
    assert torch.equal(torch.rand(3), expected)


# --- gpu tier: GradScaler state (plan §5: "gpu for scaler state") ---------------------------------


@pytest.mark.gpu
def test_scaler_state_survives_a_resume_fp16(tmp_path):
    """A skip halves the scale. The halved scale and the growth tracker must come back."""
    cfg = resolved_for(steps=40, **SCALER)
    first = trainer_for(cfg, device="cuda")
    first.train(until=5)
    handle = poison_next_gradient(first, float("inf"))
    assert first.train_step().skipped is True
    handle.remove()
    first.train(until=12)
    assert first.scaler.get_scale() == 32768.0
    path = ck.save_checkpoint(first, tmp_path)

    state = ck.load_checkpoint(path)
    assert state["precision"] == "fp16" and state["scaler"] is not None
    second = trainer_for(cfg, device="cuda")
    assert second.scaler.get_scale() == 65536.0
    ck.restore(second, state)
    assert second.scaler.get_scale() == 32768.0
    assert second.scaler.state_dict() == first.scaler.state_dict()
    assert second.train_step().scale == 32768.0


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_resume_fidelity_fp16(case, tmp_path):
    cfg = resolved_for(case, steps=60, **SCALER)
    reference = trainer_for(cfg, device="cuda")
    reference.train()
    first = trainer_for(cfg, device="cuda")
    first.train(until=25)
    path = ck.save_checkpoint(first, tmp_path)
    second = trainer_for(cfg, device="cuda")
    scramble(second)
    ck.restore(second, ck.load_checkpoint(path))
    second.train()
    assert_same_curve(second.records, reference.records, RESUME_FIDELITY_FP16)
    assert math.isfinite(second.records[-1].loss)


@pytest.mark.gpu
def test_a_checkpoint_saved_on_cuda_restores_on_cuda_with_fp32_master_weights(tmp_path):
    cfg = resolved_for(steps=10, **SCALER)
    first = trainer_for(cfg, device="cuda")
    first.train(until=4)
    state = ck.load_checkpoint(ck.save_checkpoint(first, tmp_path))
    assert all(v.dtype == torch.float32 for v in state["model"].values())
    second = trainer_for(cfg, device="cuda")
    ck.restore(second, state)
    assert all(p.is_cuda and p.dtype == torch.float32 for p in second.model.parameters())
    assert all(
        v.is_cuda for s in second.optimizer.state_dict()["state"].values() for v in s.values() if torch.is_tensor(v)
    )
