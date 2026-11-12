"""T-29: the entry points scripts/train.py, scripts/branch.py, scripts/evaluate.py and the Kaggle shim.

SYNTHETIC data only: a random-word corpus packed by the project's own packer, a tiny
model (2 layers, width 64) and budgets of 24 to 32 steps. Nothing here reads a corpus
or a result.

Acceptance (tasks list): "A cold start runs on CPU against a synthetic dataset and
resumes without human input." The resume tests below check that, and that the resumed
curve, the per-step losses and the final metrics equal those of an uninterrupted run.

Tolerances come from tests/tolerances.py only. The `gpu` test is written and skips
without CUDA (tests/conftest.py); it has not been run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import _run_common as rc
from scripts import branch as branch_script
from scripts import evaluate as evaluate_script
from scripts import train as train_script
from src import config
from src.data.loader import open_dataset
from src.data.packing import pack_dataset
from src.data.tokenizer import iter_documents, train_tokenizer
from src.eval.lm import ContextSpec, evaluate
from src.model import build_model
from src.training import branch as br
from src.training import checkpoint as ck
from src.utils import manifest as mf
from tests.tolerances import BRANCH_FIDELITY_FP32, FLOAT64_IDENTITY, RESUME_FIDELITY_FP32, scale_relative_error

ROOT = Path(__file__).resolve().parent.parent

T = 16                      # context length
B = 4                       # sequences per step
MICRO = B                   # micro-batch: one backward per step (accumulation is tested in test_trainer.py)
TRUNK_LABEL, TRUNK_STEPS = "32M", 32
SHORT_LABEL, SHORT_STEPS = "24M", 24
BRANCH_STEP = 19            # decay_start(24, 0.2): the trunk's branch point for the 24M branch
WARMUP = 4
EVAL_EVERY = 4              # in-run validation points at steps 4, 8, ..., 32
CKPT_EVERY = 8              # regular checkpoints at 8, 16, ...; keep = 2 leaves 24 and 32
PEAK_LR = 0.003
TARGETS = {"train": 4000, "val": 400, "test": 400}
MODES = ("ref", "opt")
# A relative tolerance per mode for comparing a resumed or branched run: None means exact.
RESUME_TOL = {"ref": None, "opt": RESUME_FIDELITY_FP32}
BRANCH_TOL = {"ref": None, "opt": BRANCH_FIDELITY_FP32}

WORDS = [
    "alpha", "beta", "gamma", "delta", "the", "of", "data", "model", "token",
    "naive", "cafe", "x", "yy", "zzz", "tab", "omega",
]


# --- Synthetic data and specs ----------------------------------------------------------------------


def _doc(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40)))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A SYNTHETIC dataset, packed once per module by the project's packer."""
    base = tmp_path_factory.mktemp("synthetic_entry_points")
    rng = random.Random(20261112)
    jsonl = base / "synthetic_corpus.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for i in range(1500):
            fh.write(json.dumps({"id": i, "text": _doc(rng)}, ensure_ascii=False) + "\n")
    tok = train_tokenizer(iter_documents([jsonl]), 300)
    root = base / "dataset"
    pack_dataset(iter_documents([jsonl]), tok, root, targets=TARGETS, context_length=T)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    return SimpleNamespace(root=root, meta=meta, vocab=meta["vocab_size"])


def _tiny(vocab: int, *, steps: int, label: str, branches: list[str]) -> dict:
    """Overrides that shrink the model and the recipe. The real values stay in the frozen files."""
    return {
        "model.vocab_size": vocab,
        "model.context_length": T,
        "model.d_model": 64,
        "model.n_heads": 8,
        "model.head_dim": 8,
        "model.n_layers": 2,
        "model.d_ff": 96,
        "recipe.batch.sequences_per_step": B,
        "recipe.batch.sequence_length": T,
        "recipe.batch.tokens_per_step": B * T,
        "recipe.batch.micro_batch": MICRO,
        "recipe.budgets": {TRUNK_LABEL: TRUNK_STEPS, SHORT_LABEL: SHORT_STEPS},
        "recipe.schedule.warmup_steps": WARMUP,
        "recipe.grad_scaler.init_scale": 65536.0,
        "recipe.grad_scaler.growth_interval": 2000,
        "run.steps": steps,
        "run.budget": label,
        "run.branches": branches,
    }


def _write_spec(path: Path, overrides: dict) -> Path:
    doc = {"variant": "mha", "arm": "iso_token", "init_seed": 1, "data_seed": 1,
           "peak_lr": PEAK_LR, "overrides": overrides}
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def specs(synthetic, tmp_path_factory):
    d = tmp_path_factory.mktemp("specs")
    return SimpleNamespace(
        trunk=_write_spec(d / "trunk.yaml", _tiny(synthetic.vocab, steps=TRUNK_STEPS, label=TRUNK_LABEL,
                                                  branches=[SHORT_LABEL])),
        short=_write_spec(d / "short.yaml", _tiny(synthetic.vocab, steps=SHORT_STEPS, label=SHORT_LABEL,
                                                  branches=[])),
    )


@pytest.fixture(scope="module")
def trunk_resolved(specs):
    return rc.resolve_spec(rc.load_run_spec(specs.trunk))


@pytest.fixture(scope="module")
def short_resolved(specs):
    return rc.resolve_spec(rc.load_run_spec(specs.short))


# --- Helpers ---------------------------------------------------------------------------------------


def train_argv(spec, out, data, *, mode="opt", eval_every=EVAL_EVERY, extra=()) -> list[str]:
    return [str(spec), "--data", str(data), "--out", str(out), "--mode", mode, "--device", "cpu",
            "--eval-every-steps", str(eval_every), "--checkpoint-every-steps", str(CKPT_EVERY), *extra]


def run_train(spec, out, data, *, mode="opt", eval_every=EVAL_EVERY, extra=()) -> int:
    return train_script.main(train_argv(spec, out, data, mode=mode, eval_every=eval_every, extra=extra))


def run_branch(spec, out, data, *, mode="opt", extra=()) -> int:
    argv = [str(spec), "--budget", SHORT_LABEL, "--data", str(data), "--out", str(out), "--mode", mode,
            "--device", "cpu", "--eval-every-steps", str(EVAL_EVERY), *extra]
    return branch_script.main(argv)


def manifest_of(out, mode, name) -> dict:
    return mf.read_manifest(Path(out) / "manifests" / mode / f"{name}.json")


def final_state(out, resolved, mode, step=TRUNK_STEPS) -> dict:
    directory = ck.checkpoint_dir(Path(out) / "checkpoints" / mode, resolved)
    return ck.load_checkpoint(directory / f"step{step:08d}.pt")


def losses(state) -> list[float]:
    return [row[2] for row in state["records"]]  # StepRecord: step, lr, loss, ...


def snapshot(root) -> dict[str, str]:
    root = Path(root)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _same(a, b, tol) -> bool:
    return a == b if tol is None else scale_relative_error(a, b) <= tol


def assert_same_losses(a: list[float], b: list[float], tol) -> None:
    assert len(a) == len(b) > 0
    for x, y in zip(a, b):
        assert _same(x, y, tol)


def assert_branch_matches_standalone(work, standalone, trunk_resolved, short_resolved, tol) -> None:
    """A branch's per-step losses and final metrics equal an uninterrupted run of its budget (plan §5)."""
    branch_cfg = br.branch_resolved(trunk_resolved, SHORT_LABEL)
    a = losses(final_state(work, branch_cfg, "opt", step=SHORT_STEPS))
    b = losses(final_state(standalone, short_resolved, "opt", step=SHORT_STEPS))
    assert_same_losses(a, b, tol)
    ma = manifest_of(work, "opt", mf.run_name(trunk_resolved, SHORT_LABEL))
    mb = manifest_of(standalone, "opt", mf.run_name(short_resolved))
    for key in ("val_loss", "test_loss", "train_loss"):
        assert _same(ma["final_metrics"][key], mb["final_metrics"][key], tol), key


def assert_same_result(out, other, mode, resolved, tol, name) -> None:
    """Per-step losses, the validation curve and the final metrics of two runs of one configuration."""
    assert_same_losses(losses(final_state(out, resolved, mode)), losses(final_state(other, resolved, mode)), tol)
    ma, mb = manifest_of(out, mode, name), manifest_of(other, mode, name)
    ca, cb = ma["final_metrics"]["val_curve"], mb["final_metrics"]["val_curve"]
    assert [(e["step"], e["tokens"]) for e in ca] == [(e["step"], e["tokens"]) for e in cb]
    for x, y in zip(ca, cb):
        assert _same(x["val_loss"], y["val_loss"], tol)
    for key in ("val_loss", "test_loss", "train_loss"):
        assert _same(ma["final_metrics"][key], mb["final_metrics"][key], tol), key


# --- Fixtures: the runs the tests compare against ---------------------------------------------------


@pytest.fixture(scope="module")
def reference(synthetic, specs, tmp_path_factory):
    """The uninterrupted trunk of the spec, one completed run per mode, each in an empty directory."""
    done: dict[str, Path] = {}

    def get(mode: str) -> Path:
        if mode not in done:
            out = tmp_path_factory.mktemp(f"reference_{mode}")
            assert run_train(specs.trunk, out, synthetic.root, mode=mode) == 0
            done[mode] = out
        return done[mode]

    return get


@pytest.fixture(scope="module")
def branch_source(synthetic, specs, tmp_path_factory):
    """A completed trunk to branch from. Branch tests copy it, so they never share a directory."""
    out = tmp_path_factory.mktemp("branch_source")
    assert run_train(specs.trunk, out, synthetic.root) == 0
    return out


@pytest.fixture(scope="module")
def standalone(synthetic, specs, tmp_path_factory):
    """An uninterrupted run of the short budget: what a branch must reproduce."""
    out = tmp_path_factory.mktemp("standalone")
    assert run_train(specs.short, out, synthetic.root) == 0
    return out


# --- Cold start ------------------------------------------------------------------------------------


def test_cold_start_writes_a_valid_completed_manifest(reference, trunk_resolved, synthetic):
    out = reference("opt")
    name = mf.run_name(trunk_resolved)
    m = manifest_of(out, "opt", name)
    assert mf.validate_manifest(m) == []
    assert m["name"] == name == "mha-isoparam-32M-lr3e3-s1-d1"
    assert m["kind"] == "run" and m["completed"] is True
    assert m["steps"] == {"total": TRUNK_STEPS, "attempted": TRUNK_STEPS, "taken": TRUNK_STEPS,
                          "skipped": 0, "skipped_in_first_100": 0}
    assert m["data"]["dataset_sha256"] == synthetic.meta["dataset_sha256"]
    assert m["data"]["tokenizer_sha256"] == synthetic.meta["tokenizer_sha256"]
    stream = rc.make_stream(rc.open_data(synthetic.root, trunk_resolved), trunk_resolved)
    assert m["data"]["order_sha256"] == stream.order_sha256
    assert set(m["final_metrics"]) == {"val_loss", "test_loss", "train_loss", "val", "test", "val_curve"}
    assert m["resume_count"] == 0 and len(m["session_ids"]) == 1
    assert m["backend"] is None and m["precision"] == "fp32" and m["mode"] == "opt"
    assert m["gpu_shared"] is False and m["branch"] is None and m["lr_source"] is None
    assert [e["step"] for e in m["final_metrics"]["val_curve"]] == list(range(EVAL_EVERY, TRUNK_STEPS + 1, EVAL_EVERY))


def test_final_val_loss_equals_an_independent_evaluation(reference, trunk_resolved, synthetic):
    out = reference("opt")
    state = final_state(out, trunk_resolved, "opt")
    model = build_model(trunk_resolved)
    model.load_state_dict(state["model"])
    split = open_dataset(synthetic.root, expected_dataset_sha256=None, expected_tokenizer_sha256=None)["val"]
    result = evaluate(
        model, split, context_length=T, batch_size=64,
        spec=ContextSpec.from_prereg(config.load_prereg()), device="cpu", mode="opt", precision="fp32",
    )
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert abs(result["loss"] - m["final_metrics"]["val_loss"]) <= FLOAT64_IDENTITY


def test_the_trunk_branch_point_survives_completion(reference, trunk_resolved):
    directory = ck.checkpoint_dir(reference("opt") / "checkpoints" / "opt", trunk_resolved)
    assert [p.name for p in ck.list_branch_points(directory)] == [f"trunk-step{BRANCH_STEP:08d}.pt"]
    assert [p.name for p in ck.list_checkpoints(directory)] == ["step00000024.pt", "step00000032.pt"]


def test_in_run_evaluation_does_not_change_training(reference, specs, synthetic, trunk_resolved, tmp_path):
    out = tmp_path / "no_eval"
    assert run_train(specs.trunk, out, synthetic.root, eval_every=10**9) == 0
    assert losses(final_state(out, trunk_resolved, "opt")) == losses(final_state(reference("opt"), trunk_resolved, "opt"))
    a = final_state(out, trunk_resolved, "opt")["model"]
    b = final_state(reference("opt"), trunk_resolved, "opt")["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_ref_and_opt_of_one_spec_do_not_collide(synthetic, specs, trunk_resolved, reference, tmp_path):
    out = tmp_path / "both"
    assert run_train(specs.trunk, out, synthetic.root, mode="ref") == 0
    assert run_train(specs.trunk, out, synthetic.root, mode="opt") == 0
    name = mf.run_name(trunk_resolved)
    for mode in MODES:
        m = manifest_of(out, mode, name)
        assert m["completed"] is True and m["mode"] == mode
        assert ck.list_checkpoints(ck.checkpoint_dir(out / "checkpoints" / mode, trunk_resolved))
    assert ck.checkpoint_dir(out / "checkpoints" / "ref", trunk_resolved) != ck.checkpoint_dir(out / "checkpoints" / "opt", trunk_resolved)
    assert losses(final_state(out, trunk_resolved, "opt")) == losses(final_state(reference("opt"), trunk_resolved, "opt"))
    assert losses(final_state(out, trunk_resolved, "ref")) == losses(final_state(reference("ref"), trunk_resolved, "ref"))


# --- Resume without human input (acceptance) ---------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_resume_after_an_interruption_needs_no_input_and_matches_the_uninterrupted_run(
    mode, reference, specs, synthetic, trunk_resolved, tmp_path
):
    out = tmp_path / "interrupted"
    name = mf.run_name(trunk_resolved)
    assert run_train(specs.trunk, out, synthetic.root, mode=mode, extra=("--until", "14")) == 0
    first = manifest_of(out, mode, name)
    assert first["completed"] is False and first["steps"]["attempted"] == 14
    # The identical command, with no flag: the second process resumes and finishes.
    assert run_train(specs.trunk, out, synthetic.root, mode=mode) == 0
    m = manifest_of(out, mode, name)
    assert m["completed"] is True and m["steps"]["attempted"] == TRUNK_STEPS
    assert m["resume_count"] == 1 and len(m["session_ids"]) == 2
    assert m["session_ids"][0] != m["session_ids"][1]
    assert_same_result(out, reference(mode), mode, trunk_resolved, RESUME_TOL[mode], name)


def test_resume_twice_with_two_interruptions_matches_the_uninterrupted_run(
    reference, specs, synthetic, trunk_resolved, tmp_path
):
    out = tmp_path / "twice"
    name = mf.run_name(trunk_resolved)
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "14")) == 0
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "30")) == 0
    assert run_train(specs.trunk, out, synthetic.root) == 0
    m = manifest_of(out, "opt", name)
    assert m["completed"] is True and m["resume_count"] == 2 and len(m["session_ids"]) == 3
    assert_same_result(out, reference("opt"), "opt", trunk_resolved, RESUME_TOL["opt"], name)


def test_the_manifest_at_an_interruption_is_valid_and_not_a_result(specs, synthetic, trunk_resolved, tmp_path):
    out = tmp_path / "stopped"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "9")) == 0
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert mf.validate_manifest(m) == []
    assert m["completed"] is False and m["final_metrics"] is None
    assert m["steps"]["attempted"] == 9 and m["steps"]["total"] == TRUNK_STEPS
    assert len(m["step_seconds"]) == 9
    assert mf.is_result(m) is False


def test_a_run_interrupted_before_any_checkpoint_starts_again_from_zero(specs, synthetic, trunk_resolved, tmp_path):
    out = tmp_path / "early"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "3")) == 0
    assert ck.list_checkpoints(ck.checkpoint_dir(out / "checkpoints" / "opt", trunk_resolved)) == []
    assert run_train(specs.trunk, out, synthetic.root) == 0
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert m["completed"] is True and m["steps"]["attempted"] == TRUNK_STEPS


def test_ops_do_not_change_the_config_hash(specs, synthetic, trunk_resolved, reference, tmp_path):
    out = tmp_path / "ops"
    name = mf.run_name(trunk_resolved)
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "9", "--checkpoint-every-seconds", "1000")) == 0
    assert run_train(specs.trunk, out, synthetic.root, extra=("--checkpoint-every-seconds", "1")) == 0
    m = manifest_of(out, "opt", name)
    assert m["resume_count"] == 1 and m["config_hash"] == rc.config.config_hash(trunk_resolved)
    assert len(list((out / "checkpoints" / "opt").iterdir())) == 1  # one configuration, one directory
    ops = json.loads((out / "manifests" / "opt" / f"{name}.ops.json").read_text(encoding="utf-8"))
    assert [entry["ops"]["checkpoint_every_seconds"] for entry in ops] == [1000.0, 1.0]
    assert_same_result(out, reference("opt"), "opt", trunk_resolved, RESUME_TOL["opt"], name)


# --- Idempotence, refusals ---------------------------------------------------------------------------


def test_a_third_invocation_after_completion_changes_no_file(reference, specs, synthetic):
    out = reference("opt")
    before = snapshot(out)
    assert run_train(specs.trunk, out, synthetic.root) == 0
    assert run_train(specs.trunk, out, synthetic.root, extra=("--checkpoint-every-seconds", "3")) == 0
    assert snapshot(out) == before


def _corrupted_copy(source: Path, dest: Path) -> Path:
    shutil.copytree(source, dest)
    path = dest / "train.bin"
    data = bytearray(path.read_bytes())
    data[20] ^= 0xFF  # one byte inside the training tokens
    path.write_bytes(bytes(data))
    return dest


def test_a_hash_mismatch_is_refused_by_train_branch_and_evaluate(
    synthetic, specs, trunk_resolved, branch_source, tmp_path, capsys
):
    bad = _corrupted_copy(synthetic.root, tmp_path / "bad_dataset")

    out = tmp_path / "train_out"
    assert run_train(specs.trunk, out, bad) == 2
    assert not out.exists()
    assert "does not match" in capsys.readouterr().err

    work = tmp_path / "branch_work"
    shutil.copytree(branch_source, work)
    before = snapshot(work)
    assert run_branch(specs.trunk, work, bad) == 2
    assert snapshot(work) == before

    checkpoint = ck.checkpoint_dir(branch_source / "checkpoints" / "opt", trunk_resolved) / f"step{TRUNK_STEPS:08d}.pt"
    result_path = tmp_path / "refused.json"
    assert evaluate_script.main([
        "--checkpoint", str(checkpoint), "--data", str(bad), "--split", "val",
        "--json", str(result_path), "--device", "cpu",
    ]) == 2
    assert not result_path.exists()


def test_an_expected_hash_that_differs_is_refused_and_the_matching_one_accepted(
    monkeypatch, synthetic, specs, tmp_path
):
    wrong = tmp_path / "data_wrong.yaml"
    wrong.write_text(yaml.safe_dump({"dataset_sha256": "0" * 64, "tokenizer": {"sha256": None}}), encoding="utf-8")
    monkeypatch.setattr(rc, "DATA_CONFIG", wrong)
    out = tmp_path / "wrong"
    assert run_train(specs.trunk, out, synthetic.root) == 2
    assert not out.exists()

    right = tmp_path / "data_right.yaml"
    right.write_text(yaml.safe_dump({
        "dataset_sha256": synthetic.meta["dataset_sha256"],
        "tokenizer": {"sha256": synthetic.meta["tokenizer_sha256"]},
    }), encoding="utf-8")
    monkeypatch.setattr(rc, "DATA_CONFIG", right)
    assert run_train(specs.trunk, tmp_path / "right", synthetic.root, extra=("--until", "1")) == 0


def test_an_incomplete_recipe_is_refused_before_any_work(specs, synthetic, tmp_path, capsys):
    doc = yaml.safe_load(specs.trunk.read_text(encoding="utf-8"))
    for key in ("recipe.batch.micro_batch", "recipe.grad_scaler.init_scale", "recipe.grad_scaler.growth_interval"):
        del doc["overrides"][key]
    bare = tmp_path / "bare.yaml"
    bare.write_text(yaml.safe_dump(doc), encoding="utf-8")
    out = tmp_path / "incomplete"
    assert run_train(bare, out, synthetic.root) == 2
    assert not out.exists()
    assert "unset recipe values" in capsys.readouterr().err


def test_cpu_refuses_cuda_only_flags(specs, synthetic, tmp_path):
    out = tmp_path / "flags"
    assert train_script.main([str(specs.trunk), "--data", str(synthetic.root), "--out", str(out),
                              "--device", "cpu", "--expected-backend", "efficient"]) == 2
    assert train_script.main([str(specs.trunk), "--data", str(synthetic.root), "--out", str(out),
                              "--device", "cpu", "--gpu-shared", "yes"]) == 2
    assert not out.exists()


# --- Branches ----------------------------------------------------------------------------------------


def test_a_branch_writes_a_branch_manifest_matching_the_standalone_run(
    branch_source, standalone, specs, synthetic, trunk_resolved, short_resolved, tmp_path
):
    work = tmp_path / "work"
    shutil.copytree(branch_source, work)
    assert run_branch(specs.trunk, work, synthetic.root) == 0
    name = mf.run_name(trunk_resolved, SHORT_LABEL)
    m = manifest_of(work, "opt", name)
    assert mf.validate_manifest(m) == []
    assert name.endswith("-b24M") and m["name"] == name
    assert m["kind"] == "branch" and m["completed"] is True
    assert m["config_hash"] == config.config_hash(trunk_resolved)
    assert m["branch"] == {
        "budget": SHORT_LABEL,
        "steps": SHORT_STEPS,
        "branch_step": BRANCH_STEP,
        "parent_run": mf.run_name(trunk_resolved),
        "parent_checkpoint": f"checkpoints/opt/{config.config_hash(trunk_resolved)}/trunk-step{BRANCH_STEP:08d}.pt",
    }
    assert m["steps"]["total"] == SHORT_STEPS and m["steps"]["attempted"] == SHORT_STEPS
    assert m["steps"]["taken"] == SHORT_STEPS
    assert_branch_matches_standalone(work, standalone, trunk_resolved, short_resolved, BRANCH_TOL["opt"])


def test_a_finished_branch_is_not_run_again(branch_source, specs, synthetic, trunk_resolved, tmp_path):
    work = tmp_path / "again"
    shutil.copytree(branch_source, work)
    assert run_branch(specs.trunk, work, synthetic.root) == 0
    before = snapshot(work)
    assert run_branch(specs.trunk, work, synthetic.root) == 0
    assert snapshot(work) == before


def test_a_branch_resumes_after_an_interruption_without_input(
    branch_source, standalone, specs, synthetic, trunk_resolved, short_resolved, tmp_path
):
    work = tmp_path / "resume_branch"
    shutil.copytree(branch_source, work)
    assert run_branch(specs.trunk, work, synthetic.root, extra=("--until", "22", "--checkpoint-every-steps", "1")) == 0
    name = mf.run_name(trunk_resolved, SHORT_LABEL)
    first = manifest_of(work, "opt", name)
    assert first["completed"] is False and first["steps"]["attempted"] == 22
    assert run_branch(specs.trunk, work, synthetic.root) == 0
    m = manifest_of(work, "opt", name)
    assert m["completed"] is True and m["resume_count"] == 1 and len(m["session_ids"]) == 2
    assert_branch_matches_standalone(work, standalone, trunk_resolved, short_resolved, BRANCH_TOL["opt"])


def test_a_branch_is_refused_when_the_trunk_has_no_branch_point(specs, synthetic, trunk_resolved, tmp_path, capsys):
    out = tmp_path / "early_trunk"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "15")) == 0
    before = snapshot(out)
    assert run_branch(specs.trunk, out, synthetic.root) == 2
    assert "trunk checkpoint" in capsys.readouterr().err
    assert snapshot(out) == before


def test_a_branch_that_is_not_one_of_the_trunks_branches_is_refused(specs, synthetic, tmp_path):
    assert branch_script.main([str(specs.trunk), "--budget", TRUNK_LABEL, "--data", str(synthetic.root),
                               "--out", str(tmp_path / "x"), "--device", "cpu"]) == 2


# --- Evaluation and the notebook --------------------------------------------------------------------


def test_evaluate_reproduces_the_manifest_losses_and_changes_nothing(reference, specs, synthetic, trunk_resolved, tmp_path):
    out = reference("opt")
    before = snapshot(out)
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    for split in ("val", "test"):
        result_path = tmp_path / f"{split}.json"
        assert evaluate_script.main([
            str(specs.trunk), "--out", str(out), "--data", str(synthetic.root), "--split", split,
            "--json", str(result_path), "--device", "cpu",
        ]) == 0
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert abs(result["loss"] - m["final_metrics"][f"{split}_loss"]) <= FLOAT64_IDENTITY
    assert snapshot(out) == before


def test_the_notebook_is_a_one_cell_shim_without_credentials():
    text = (ROOT / "notebooks" / "kaggle_shim.ipynb").read_text(encoding="utf-8")
    nb = json.loads(text)
    assert nb["nbformat"] == 4
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 1
    source = "".join(code[0]["source"])
    assert "scripts.train" in source and "--data" in source and "--out" in source
    assert source.count("\n") <= 10
    lowered = text.lower()
    for banned in ("kaggle_key", "kaggle.json", "kaggle datasets", "kaggle kernels", "kaggle competitions",
                   "password", "token"):
        assert banned not in lowered, banned


def test_the_example_run_spec_and_the_ops_settings_load():
    spec = rc.load_run_spec(ROOT / "configs" / "runs" / "example_mha_s1.yaml")
    resolved = rc.resolve_spec(spec)
    assert resolved["variant"] == "mha" and resolved["arm"] == "iso_token"
    assert resolved["run"]["peak_lr"] == spec["peak_lr"] > 0
    assert resolved["run"]["steps"] == config.load_recipe()["budgets"]["300M"]
    ops = rc.load_ops(argparse.Namespace())
    assert ops == {
        "checkpoint_every_seconds": 600, "checkpoint_every_steps": None, "eval_every_steps": 250,
        "eval_curve_windows": 256, "eval_batch_size": 64, "keep_checkpoints": 2,
    }
    assert not any("checkpoint" in key or "eval" in key for key in resolved)


# --- The module entry, in a subprocess ----------------------------------------------------------------


def test_the_module_entry_runs_in_a_subprocess(specs, synthetic, trunk_resolved, tmp_path):
    out = tmp_path / "subprocess"
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train", str(specs.trunk), "--data", str(synthetic.root),
         "--out", str(out), "--mode", "opt", "--device", "cpu", "--until", "5"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert m["completed"] is False and m["steps"]["attempted"] == 5


# --- The gpu tier: written, skipped without CUDA, not yet run -----------------------------------------


@pytest.mark.gpu
def test_cuda_fp16_cold_start_and_resume_record_the_enforced_backend_and_scale(specs, synthetic, trunk_resolved, tmp_path):
    out = tmp_path / "gpu"
    base = [str(specs.trunk), "--data", str(synthetic.root), "--out", str(out), "--mode", "opt",
            "--device", "cuda", "--gpu-shared", "no", "--expected-backend", "efficient",
            "--eval-every-steps", str(EVAL_EVERY), "--checkpoint-every-steps", str(CKPT_EVERY)]
    assert train_script.main(base + ["--until", "10"]) == 0
    assert train_script.main(base) == 0
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert m["completed"] is True and m["resume_count"] == 1
    assert m["backend"] == "efficient" and m["precision"] == "fp16"
    assert isinstance(m["scaler"]["final_scale"], float) and m["scaler"]["final_scale"] > 0
