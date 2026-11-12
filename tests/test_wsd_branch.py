"""T-17: WSD decay branches (plan §5 "WSD branch fidelity", §7.2, §13.9). SYNTHETIC data only.

Written by the orchestrator before the implementation; the worker may not
change this file.

Plan §7.2: the stable trunk of a warmup-stable-decay run is shared, so a short
decay branched from a trunk checkpoint gives the result for a shorter budget at
a fraction of the cost. "The 100M result decays from step 1,221 to 1,526."
Plan §5: "A decay branched from a trunk checkpoint matches an uninterrupted WSD
run of the same length." Plan §13.9: a branch that restores the data position
or the schedule step wrongly produces a plausible curve that is not comparable.

The contract for `src/training/branch.py`
-----------------------------------------
    class BranchError(RuntimeError)

    def branch_steps(resolved) -> dict[str, int]
        {budget label: trunk step at which that branch leaves} for every label in
        resolved["run"]["branches"], in that order. The step is
        decay_start(recipe.budgets[label], recipe.schedule.decay_fraction): the
        branch decays over the last 20% of ITS budget. These are the steps at
        which the trunk writes branch-point checkpoints
        (`Checkpointer(branch_steps=branch_steps(resolved).values())`).

    def branch_step(resolved, budget) -> int
        One entry of the above. BranchError if `budget` is not a branch of the run.

    def branch_resolved(parent_resolved, budget) -> dict
        The branch's own configuration: a deep copy of the parent's with
        run.steps = recipe.budgets[budget], run.budget = budget,
        run.branches = [], and run.parent = {"config_hash": <parent's hash>,
        "budget": <parent's run.budget>, "steps": <parent's run.steps>,
        "branch_step": <branch step>}. The schedule of this config is the WSD
        schedule of the short budget, which equals the trunk's schedule up to
        the branch step. Because of run.parent its config hash differs from the
        parent's AND from a standalone run of the same budget, so a branch has
        its own checkpoint directory and can never resume from, or be resumed
        by, a standalone run such as a reproducibility rerun.

    def start_branch(trunk_state, budget, stream, *, device="cpu", mode="opt",
                     precision=None, expected_backend=None, on_step=None) -> Trainer
        `trunk_state` is a loaded branch-point checkpoint of the parent run
        (`checkpoint.load_checkpoint(.../trunk-step*.pt)`). Builds the model and
        a Trainer for branch_resolved(trunk_state["resolved"], budget) and puts
        the trunk's state into it: weights, optimizer state, scaler state, step,
        steps taken, step records, RNG state. The trainer comes back at
        step == branch_step with total_steps == the budget's steps, ready to
        decay. BranchError, before anything is built, if the state is not a
        branch point, if its step is not this budget's branch step, or if
        `budget` is not a branch of the parent run.

    def start_or_resume_branch(parent_resolved, budget, stream, root, **trainer_kwargs)
            -> tuple[Trainer, ResumeInfo]
        What a branch job calls, with no human decision (plan §12.3). If the
        branch's own checkpoint directory (checkpoint_dir(root, branch config))
        holds a valid checkpoint, a fresh branch trainer resumes from it.
        Otherwise the branch starts from the parent's branch-point checkpoint
        `checkpoint_dir(root, parent_resolved) / trunk-step{branch_step:08d}.pt`
        and ResumeInfo is (step=branch_step, path=<that file>, resume_count=0).
        BranchError if neither exists.

    def branch_manifest_block(parent_resolved, budget, parent_checkpoint) -> dict
        The `branch` block of the branch's manifest (src/utils/manifest.py):
        {"budget", "steps", "branch_step", "parent_run", "parent_checkpoint"}.
        The manifest of a branch carries the PARENT's resolved config
        (decision 102); `parent_checkpoint` is stored as a string.

The trunk itself needs nothing new: it is an ordinary run whose Checkpointer
is given the branch steps. A trunk checkpoint is never a result (plan §7.2).
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from src import config
from src.data.loader import TrainStream
from src.model import build_model
from src.training import branch as br
from src.training import checkpoint as ck
from src.training.schedule import decay_start, lr_at
from src.training.trainer import Trainer
from src.utils import manifest as mf
from tests.fixtures.fake_manifests import SYNTHETIC_DATA, SYNTHETIC_GIT_SHA
from tests.fixtures.synthetic_tokens import rule_split
from tests.tolerances import BRANCH_FIDELITY_FP16, BRANCH_FIDELITY_FP32, scale_relative_error
from tests.variants import CASES, tiny_resolved

VOCAB, T, B = 128, 16, 4
TRUNK_STEPS, SHORT_STEPS = 120, 60  # the tiny "300M" and "100M"
WARMUP = 20  # shortened with the budgets; the real schedule is checked on the real configs below
AT = 48  # where the short branch leaves the trunk: decay_start(60, 0.2)
SHORT = "S"
SCALER = {"recipe.grad_scaler.init_scale": 65536.0, "recipe.grad_scaler.growth_interval": 2000}


def _tiny(case, *, steps, branches, budget, init_seed=1, **overrides):
    extra = {
        "recipe.batch.sequences_per_step": B,
        "recipe.batch.sequence_length": T,
        "recipe.batch.tokens_per_step": B * T,
        "recipe.batch.micro_batch": 2,
        "recipe.budgets": {SHORT: SHORT_STEPS, "L": TRUNK_STEPS, "M": 80},
        "recipe.schedule.warmup_steps": WARMUP,
        "run.steps": steps,
        "run.budget": budget,
        "run.branches": branches,
    }
    extra.update(overrides)
    return tiny_resolved(case, init_seed=init_seed, context_length=T, **extra)


def trunk_cfg(case=CASES[0], **kw):
    """The long run, with one branch at the short budget."""
    return _tiny(case, steps=TRUNK_STEPS, branches=[SHORT], budget="L", **kw)


def standalone_cfg(case=CASES[0], **kw):
    """An uninterrupted run of the short budget: what the branch must reproduce."""
    return _tiny(case, steps=SHORT_STEPS, branches=[], budget=SHORT, **kw)


def stream():
    return TrainStream(rule_split(TRUNK_STEPS * B * T + 1, VOCAB), context_length=T, sequences_per_step=B, data_seed=1)


def train_trunk_to_branch_point(cfg, root, **trainer_kwargs):
    """Run the trunk with its Checkpointer until just after the branch point, then stop."""
    saver = ck.Checkpointer(root, every_steps=20, branch_steps=br.branch_steps(cfg).values())
    trunk = Trainer(build_model(cfg), cfg, stream(), on_step=saver, **trainer_kwargs)
    trunk.train(until=br.branch_step(cfg, SHORT) + 5)
    return trunk


def trunk_state(cfg, root):
    path = ck.checkpoint_dir(root, cfg) / f"trunk-step{br.branch_step(cfg, SHORT):08d}.pt"
    return path, ck.load_checkpoint(path)


def assert_same_curve(got, reference, tolerance):
    assert [r.step for r in got] == [r.step for r in reference]
    for a, b in zip(got, reference):
        assert a.lr == b.lr, a.step
        assert a.skipped == b.skipped, a.step
        assert scale_relative_error(a.loss, b.loss) <= tolerance, a.step
        assert scale_relative_error(a.grad_norm, b.grad_norm) <= tolerance, a.step


# --- The numbers of plan §7.2 -----------------------------------------------------------------


def test_branch_steps_of_the_real_configs():
    arm_a = config.resolve("mha", init_seed=1, peak_lr=1e-3)
    assert br.branch_steps(arm_a) == {"100M": 1221}  # "decays from step 1,221 to 1,526"
    ladder = config.resolve("mha", ladder="mha_L6", init_seed=1, peak_lr=1e-3)
    assert br.branch_steps(ladder) == {"100M": 1221, "200M": 2442}
    assert list(br.branch_steps(ladder)) == ["100M", "200M"]
    assert br.branch_step(ladder, "200M") == 2442
    assert br.branch_steps(config.resolve("gqa4", "iso_backbone", init_seed=1, peak_lr=1e-3)) == {}


def test_the_plans_cost_of_a_branch():
    """Plan §7.2: a run with 100M and 300M results costs 4,883 steps instead of 4,578, +6.7%."""
    cfg = config.resolve("mha", init_seed=1, peak_lr=1e-3)
    branch = br.branch_resolved(cfg, "100M")
    extra = branch["run"]["steps"] - br.branch_step(cfg, "100M")
    assert extra == 305 and cfg["run"]["steps"] + extra == 4883
    assert round(100 * extra / cfg["run"]["steps"], 1) == 6.7


def test_a_budget_that_is_not_a_branch_is_refused():
    cfg = config.resolve("mha", init_seed=1, peak_lr=1e-3)
    for bad in ("200M", "300M", "50M", "nonsense"):
        with pytest.raises(br.BranchError):
            br.branch_step(cfg, bad)
        with pytest.raises(br.BranchError):
            br.branch_resolved(cfg, bad)


# --- The branch's own configuration --------------------------------------------------------------


def test_branch_config_is_the_short_schedule_with_its_parent_named():
    parent = config.resolve("mla192", init_seed=2, peak_lr=1.2e-3)
    before = copy.deepcopy(parent)
    branch = br.branch_resolved(parent, "100M")
    assert parent == before  # the parent's config is not modified
    assert branch["run"]["steps"] == 1526 and branch["run"]["budget"] == "100M"
    assert branch["run"]["branches"] == []
    assert branch["run"]["parent"] == {
        "config_hash": config.config_hash(parent),
        "budget": "300M",
        "steps": 4578,
        "branch_step": 1221,
    }
    for key in ("init_seed", "data_seed", "peak_lr"):
        assert branch["run"][key] == parent["run"][key]
    for key in ("variant", "arm", "ladder", "model", "init", "precision", "params", "recipe", "overrides"):
        assert branch[key] == parent[key], key


def test_trunk_and_branch_share_the_schedule_up_to_the_branch_point():
    parent = config.resolve("mha", init_seed=1, peak_lr=1e-3)
    branch = br.branch_resolved(parent, "100M")
    assert [lr_at(k, branch) for k in range(1221)] == [lr_at(k, parent) for k in range(1221)]
    # At the branch point the trunk is still at its peak; the branch starts to decay.
    assert lr_at(1221, parent) == lr_at(1220, parent) == 1e-3
    assert lr_at(1221, branch) == 1e-3 and lr_at(1222, branch) < 1e-3
    assert lr_at(1525, branch) < lr_at(1524, branch) and lr_at(1526, branch) == 0.0
    # The trunk's own decay starts much later.
    assert decay_start(4578, 0.2) == 3662 and lr_at(3662, parent) == 1e-3


def test_a_branch_never_shares_a_checkpoint_directory(tmp_path):
    """The 100M branch and a standalone 100M run (a reproducibility rerun, plan §8.1) have
    the same schedule, seeds and model. They must still be two runs."""
    parent = config.resolve("mha", init_seed=1, peak_lr=1e-3)
    standalone = config.resolve("mha", init_seed=1, peak_lr=1e-3, budget="100M", branches=[])
    branch = br.branch_resolved(parent, "100M")
    hashes = {config.config_hash(c) for c in (parent, standalone, branch)}
    assert len(hashes) == 3
    dirs = {ck.checkpoint_dir(tmp_path, c) for c in (parent, standalone, branch)}
    assert len(dirs) == 3
    # Apart from run.parent the branch IS the standalone run.
    stripped = copy.deepcopy(branch)
    del stripped["run"]["parent"]
    assert stripped == standalone


# --- Branch fidelity (plan §5) ---------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_a_branch_matches_the_uninterrupted_run_of_its_length(case, tmp_path):
    reference = Trainer(build_model(standalone_cfg(case)), standalone_cfg(case), stream())
    reference.train()

    cfg = trunk_cfg(case)
    assert br.branch_steps(cfg) == {SHORT: AT} and AT == decay_start(SHORT_STEPS, 0.2) > WARMUP
    train_trunk_to_branch_point(cfg, tmp_path)
    _, state = trunk_state(cfg, tmp_path)
    assert state["branch_point"] is True and state["step"] == AT

    branch = br.start_branch(state, SHORT, stream())
    assert branch.step == AT and branch.steps_taken == AT and branch.total_steps == SHORT_STEPS
    assert branch.resolved == br.branch_resolved(cfg, SHORT)
    branch.train()
    assert branch.step == SHORT_STEPS
    assert_same_curve(branch.records, reference.records, BRANCH_FIDELITY_FP32)
    for (name, a), (_, b) in zip(branch.model.named_parameters(), reference.model.named_parameters()):
        assert scale_relative_error(a.detach(), b.detach()) <= BRANCH_FIDELITY_FP32, name
    # It really decayed: the last learning rates fall, and the final loss is finite.
    lrs = [r.lr for r in branch.records[AT:]]
    assert lrs[0] == cfg["run"]["peak_lr"] and all(a > b for a, b in zip(lrs[1:], lrs[2:]))
    assert math.isfinite(branch.records[-1].loss)


def test_the_trunk_at_the_branch_point_is_not_the_branch_result(tmp_path):
    """Plan §7.2: never report a trunk checkpoint. The trunk goes on at peak learning rate
    and its weights differ from the decayed branch's."""
    cfg = trunk_cfg()
    trunk = train_trunk_to_branch_point(cfg, tmp_path)
    _, state = trunk_state(cfg, tmp_path)
    branch = br.start_branch(state, SHORT, stream())
    branch.train(until=AT + 5)
    peak = cfg["run"]["peak_lr"]
    assert [r.lr for r in trunk.records[AT + 1 : AT + 5]] == [peak] * 4
    assert all(r.lr < peak for r in branch.records[AT + 1 : AT + 5])
    assert trunk.records[:AT] == branch.records[:AT]  # the shared history, record for record
    # Step AT is the first each runs on its own: same weights, same tokens, same learning rate.
    assert trunk.records[AT].lr == branch.records[AT].lr == peak
    assert trunk.records[AT].loss == branch.records[AT].loss
    different = [
        name
        for (name, a), (_, b) in zip(trunk.model.named_parameters(), branch.model.named_parameters())
        if not torch.equal(a, b)
    ]
    assert different


def test_the_branch_reads_the_tokens_the_trunk_would_read_next(tmp_path):
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner, self.max_steps = inner, inner.max_steps

        def batch(self, step):
            seen.append(step)
            return self.inner.batch(step)

    cfg = trunk_cfg()
    train_trunk_to_branch_point(cfg, tmp_path)
    _, state = trunk_state(cfg, tmp_path)
    branch = br.start_branch(state, SHORT, Recording(stream()))
    branch.train()
    assert seen == list(range(AT, SHORT_STEPS))


def test_skips_and_records_of_the_trunk_carry_into_the_branch(tmp_path):
    """Plan §13.6: steps taken are compared across runs, branches included."""
    cfg = trunk_cfg()
    saver = ck.Checkpointer(tmp_path, every_steps=20, branch_steps=br.branch_steps(cfg).values())
    trunk = Trainer(build_model(cfg), cfg, stream(), on_step=saver)
    trunk.train(until=7)
    weight = trunk.model.layers[0].mlp.gate.weight
    handle = weight.register_hook(lambda g: torch.full_like(g, float("inf")))
    assert trunk.train_step().skipped is True
    handle.remove()
    trunk.train(until=AT + 1)
    _, state = trunk_state(cfg, tmp_path)
    branch = br.start_branch(state, SHORT, stream())
    assert branch.steps_taken == AT - 1 and branch.skipped_steps == 1
    branch.train()
    summary = branch.summary()
    assert summary["steps_attempted"] == SHORT_STEPS and summary["steps_taken"] == SHORT_STEPS - 1
    assert summary["skipped_in_first_100"] == 1 and summary["total_steps"] == SHORT_STEPS
    assert [r.step for r in branch.records] == list(range(SHORT_STEPS))


def test_starting_a_branch_does_not_touch_the_trunk_state(tmp_path):
    cfg = trunk_cfg()
    train_trunk_to_branch_point(cfg, tmp_path)
    path, state = trunk_state(cfg, tmp_path)
    before = path.read_bytes()
    kept = copy.deepcopy({k: state[k] for k in ("config_hash", "resolved", "step", "branch_point")})
    br.start_branch(state, SHORT, stream()).train()
    assert {k: state[k] for k in kept} == kept
    assert path.read_bytes() == before
    # Two branches from one state are identical: the state is a value.
    a = br.start_branch(state, SHORT, stream())
    b = br.start_branch(state, SHORT, stream())
    a.train(until=AT + 8)
    b.train(until=AT + 8)
    assert [r.loss for r in a.records] == [r.loss for r in b.records]
    assert a.records[-1].lr < a.records[AT].lr  # both decayed, independently


# --- Refusals --------------------------------------------------------------------------------------


def test_only_a_branch_point_of_the_right_step_starts_a_branch(tmp_path):
    cfg = trunk_cfg()
    trunk = train_trunk_to_branch_point(cfg, tmp_path)
    directory = ck.checkpoint_dir(tmp_path, cfg)
    _, good = trunk_state(cfg, tmp_path)

    regular = ck.load_checkpoint(ck.list_checkpoints(directory)[-1])  # a regular checkpoint
    assert regular["branch_point"] is False
    with pytest.raises(br.BranchError):
        br.start_branch(regular, SHORT, stream())

    wrong_step = ck.load_checkpoint(ck.save_checkpoint(trunk, tmp_path / "elsewhere", branch_point=True))
    assert wrong_step["branch_point"] is True and wrong_step["step"] == AT + 5
    with pytest.raises(br.BranchError):
        br.start_branch(wrong_step, SHORT, stream())

    with pytest.raises(br.BranchError):
        br.start_branch(good, "M", stream())  # a budget, but not a branch of this run

    with pytest.raises(ck.CheckpointError):
        br.start_branch(good, SHORT, stream(), mode="ref")  # the trunk ran in opt mode


def test_a_branch_of_a_branch_is_refused(tmp_path):
    cfg = trunk_cfg()
    train_trunk_to_branch_point(cfg, tmp_path)
    _, state = trunk_state(cfg, tmp_path)
    branch = br.start_branch(state, SHORT, stream())
    assert br.branch_steps(branch.resolved) == {}
    with pytest.raises(br.BranchError):
        br.branch_resolved(branch.resolved, SHORT)


# --- A branch job, without a human (plan §12.3) -------------------------------------------------------


def test_a_branch_job_starts_from_the_trunk_then_resumes_itself(tmp_path):
    reference = Trainer(build_model(standalone_cfg()), standalone_cfg(), stream())
    reference.train()
    cfg = trunk_cfg()
    train_trunk_to_branch_point(cfg, tmp_path)
    trunk_path, _ = trunk_state(cfg, tmp_path)

    # First session: nothing of the branch exists yet.
    first, info = br.start_or_resume_branch(cfg, SHORT, stream(), tmp_path)
    assert info == ck.ResumeInfo(step=AT, path=trunk_path, resume_count=0)
    branch_dir = ck.checkpoint_dir(tmp_path, br.branch_resolved(cfg, SHORT))
    assert branch_dir != ck.checkpoint_dir(tmp_path, cfg)
    first.on_step = ck.Checkpointer(tmp_path, every_steps=5, resume_count=info.resume_count)
    first.train(until=53)  # the session dies here; the checkpoints are at 50 and, before it, none
    assert [p.name for p in ck.list_checkpoints(branch_dir)] == ["step00000050.pt"]

    # Second session: the same call resumes the branch from its own checkpoint.
    second, info = br.start_or_resume_branch(cfg, SHORT, stream(), tmp_path)
    assert info.step == 50 and info.resume_count == 1
    assert info.path == branch_dir / "step00000050.pt"
    assert second.total_steps == SHORT_STEPS and second.resolved == br.branch_resolved(cfg, SHORT)
    second.train()
    assert_same_curve(second.records, reference.records, BRANCH_FIDELITY_FP32)
    # The trunk's directory was not written to by the branch.
    assert all(p.name.startswith(("step", "trunk-step")) for p in ck.checkpoint_dir(tmp_path, cfg).iterdir())
    assert ck.load_checkpoint(trunk_path)["step"] == AT


def test_a_branch_job_without_its_trunk_checkpoint_fails_loudly(tmp_path):
    cfg = trunk_cfg()
    with pytest.raises(br.BranchError):
        br.start_or_resume_branch(cfg, SHORT, stream(), tmp_path)
    # A trunk that has not reached the branch point has only regular checkpoints.
    trunk = Trainer(build_model(cfg), cfg, stream(), on_step=ck.Checkpointer(tmp_path, every_steps=20))
    trunk.train(until=40)
    with pytest.raises(br.BranchError):
        br.start_or_resume_branch(cfg, SHORT, stream(), tmp_path)


def test_a_standalone_run_never_resumes_a_branch(tmp_path):
    """The reproducibility rerun of plan §8.1 must start from scratch even if the branch with
    the same seeds and schedule has checkpoints under the same root."""
    cfg = trunk_cfg()
    train_trunk_to_branch_point(cfg, tmp_path)
    branch, info = br.start_or_resume_branch(cfg, SHORT, stream(), tmp_path)
    branch.on_step = ck.Checkpointer(tmp_path, every_steps=5)
    branch.train(until=56)
    rerun = Trainer(build_model(standalone_cfg()), standalone_cfg(), stream())
    assert ck.resume_or_start(rerun, tmp_path) == ck.ResumeInfo(step=0, path=None, resume_count=0)


# --- The manifest records parent and branch step (TASKS.md T-17) -----------------------------------


@pytest.mark.parametrize(
    "kwargs, budget, step, steps, name",
    [
        (dict(variant="mha"), "100M", 1221, 1526, "mha-isoparam-300M-lr1e3-s1-d1-b100M"),
        (dict(variant="mha", ladder="mha_L6"), "200M", 2442, 3052, "mha_L6-isoparam-300M-lr1e3-s1-d1-b200M"),
    ],
)
def test_the_branch_manifest_records_parent_and_branch_step(kwargs, budget, step, steps, name):
    parent = config.resolve(init_seed=1, peak_lr=1e-3, **kwargs)
    checkpoint_path = f"/kaggle/working/{config.config_hash(parent)}/trunk-step{step:08d}.pt"
    block = br.branch_manifest_block(parent, budget, checkpoint_path)
    assert block == {
        "budget": budget,
        "steps": steps,
        "branch_step": step,
        "parent_run": mf.run_name(parent),
        "parent_checkpoint": checkpoint_path,
    }
    manifest = mf.build_manifest(  # raises if the block does not fit the manifest rules
        kind="branch",
        resolved=parent,
        completed=True,
        git_sha=SYNTHETIC_GIT_SHA,
        data=dict(SYNTHETIC_DATA),
        environment={"python": "3.12.0", "torch": "2.15.0", "cuda": None, "driver": None,
                     "gpu_name": None, "compute_capability": None},
        backend=None,
        mode="opt",
        precision="fp32",
        steps={"total": steps, "attempted": steps, "taken": steps, "skipped": 0, "skipped_in_first_100": 0},
        final_scale=1.0,
        gpu_shared=False,
        session_ids=["SYNTHETIC-session"],
        step_seconds=[0.5] * steps,
        final_metrics={"val_loss": 3.0},  # SYNTHETIC
        branch=block,
    )
    assert manifest["name"] == name and manifest["kind"] == "branch"
    assert manifest["branch"]["parent_run"] == name.rsplit("-b", 1)[0]
    assert mf.is_result(manifest)


def test_manifest_block_accepts_a_path_object(tmp_path):
    parent = config.resolve("mha", init_seed=1, peak_lr=1e-3)
    block = br.branch_manifest_block(parent, "100M", tmp_path / "trunk-step00001221.pt")
    assert block["parent_checkpoint"] == str(tmp_path / "trunk-step00001221.pt")
    with pytest.raises(br.BranchError):
        br.branch_manifest_block(parent, "200M", "x.pt")


# --- gpu tier (plan §5: "gpu for fp16") -------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_branch_fidelity_fp16(case, tmp_path):
    reference_cfg = standalone_cfg(case, **SCALER)
    reference = Trainer(build_model(reference_cfg), reference_cfg, stream(), device="cuda")
    reference.train()
    cfg = trunk_cfg(case, **SCALER)
    train_trunk_to_branch_point(cfg, tmp_path, device="cuda")
    _, state = trunk_state(cfg, tmp_path)
    assert state["precision"] == "fp16" and state["scaler"] is not None
    branch = br.start_branch(state, SHORT, stream(), device="cuda")
    assert branch.scaler.state_dict() == state["scaler"]
    branch.train()
    assert_same_curve(branch.records, reference.records, BRANCH_FIDELITY_FP16)
