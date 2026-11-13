"""T-41: the records a training run keeps per evaluation and per session (plan §10.2, §10.3, §12.3).

Two records: the context-bucketed loss at every in-run validation point, and the peak memory
of each session. SYNTHETIC data only: tiny models, budgets of 24 to 32 steps, the CPU. The
CUDA branch of peak_memory has not been run (no GPU here); the tests replace peak_memory
with known readings through monkeypatch.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from scripts import _run_common as rc
from scripts import train as train_script
from src import config
from src.data.loader import open_dataset
from src.eval.lm import ContextSpec, evaluate
from src.model import build_model
from src.training import checkpoint as ck
from src.utils import manifest as mf
from tests.test_entry_points import (  # noqa: F401  (fixtures are used by name)
    CKPT_EVERY,
    EVAL_EVERY,
    MODES,
    RESUME_TOL,
    ROOT,
    SHORT_LABEL,
    T,
    TRUNK_STEPS,
    final_state,
    manifest_of,
    reference,
    run_branch,
    run_train,
    specs,
    synthetic,
    train_argv,
    trunk_resolved,
)
from tests.tolerances import FLOAT64_IDENTITY, scale_relative_error

CURVE_ENTRY_KEYS = {"step", "tokens", "val_loss", "buckets", "late_context", "icl"}

# Known readings: a session's (allocated, reserved) pair. Each maximum below is reached by
# a different session, and the sum and the last session give other numbers.
FIRST = (100, 900)
SECOND = (700, 300)
THIRD = (500, 400)
MAX_OVER_THREE = {"allocated_bytes": 700, "reserved_bytes": 900}


def same_value(a, b, tol) -> bool:
    """Equal in structure; floats equal exactly when tol is None, else within the relative tolerance."""
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return a == b if tol is None else scale_relative_error(a, b) <= tol
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_value(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_value(x, y, tol) for x, y in zip(a, b))
    return a == b


def curve_of(out, mode, resolved) -> list[dict]:
    return manifest_of(out, mode, mf.run_name(resolved))["final_metrics"]["val_curve"]


def ckpt_dir_of(out, mode, resolved) -> Path:
    return ck.checkpoint_dir(Path(out) / "checkpoints" / mode, resolved)


@pytest.fixture
def readings(monkeypatch):
    """Replaces the peak reading: the session that runs reads the pair in state["pair"]."""
    state = {"pair": None, "resets": 0}

    def fake_peak(device):
        allocated, reserved = state["pair"]
        return {"allocated_bytes": allocated, "reserved_bytes": reserved}

    def counted_reset(device):
        state["resets"] += 1

    monkeypatch.setattr(rc, "peak_memory", fake_peak)
    monkeypatch.setattr(rc, "reset_peak_memory", counted_reset)
    return state


# --- 1. The context-bucketed loss at every validation point ----------------------------------------


def test_every_validation_point_carries_the_context_blocks(reference, trunk_resolved):
    curve = curve_of(reference("opt"), "opt", trunk_resolved)
    assert [e["step"] for e in curve] == list(range(EVAL_EVERY, TRUNK_STEPS + 1, EVAL_EVERY))
    prereg_buckets = list(ContextSpec.from_prereg(config.load_prereg()).buckets)
    for entry in curve:
        assert set(entry) == CURVE_ENTRY_KEYS
        assert [(b["low"], b["high"]) for b in entry["buckets"]] == prereg_buckets
        assert set(entry["late_context"]) == {"min_c", "n_tokens", "loss"}
        assert set(entry["icl"]) == {"late", "early", "score"}


def test_the_buckets_of_each_point_add_up_to_its_loss(reference, trunk_resolved):
    """The check the final evaluation's buckets pass (tests/test_eval_lm.py), applied to every point."""
    for entry in curve_of(reference("opt"), "opt", trunk_resolved):
        total = sum(b["n_tokens"] for b in entry["buckets"])
        assert total > 0
        weighted = sum(b["n_tokens"] * b["loss"] for b in entry["buckets"] if b["n_tokens"]) / total
        assert scale_relative_error(weighted, entry["val_loss"]) <= FLOAT64_IDENTITY


def test_the_last_point_equals_an_independent_evaluation_of_the_final_checkpoint(reference, trunk_resolved, synthetic):
    out = reference("opt")
    last = curve_of(out, "opt", trunk_resolved)[-1]
    assert last["step"] == TRUNK_STEPS
    model = build_model(trunk_resolved)
    model.load_state_dict(final_state(out, trunk_resolved, "opt")["model"])
    split = open_dataset(synthetic.root, expected_dataset_sha256=None, expected_tokenizer_sha256=None)["val"]
    windows = yaml.safe_load(rc.OPS_CONFIG.read_text(encoding="utf-8"))["eval_curve_windows"]
    result = evaluate(
        model, split, context_length=T, batch_size=64, spec=ContextSpec.from_prereg(config.load_prereg()),
        device="cpu", mode="opt", precision="fp32", max_windows=windows,
    )
    assert same_value(last["val_loss"], result["loss"], FLOAT64_IDENTITY)
    for key in ("buckets", "late_context", "icl"):
        assert same_value(last[key], result[key], FLOAT64_IDENTITY), key


@pytest.mark.parametrize("mode", MODES)
def test_a_stopped_and_resumed_run_has_the_same_curve_with_the_new_keys(
    mode, reference, specs, synthetic, trunk_resolved, tmp_path
):
    out = tmp_path / "resumed"
    assert run_train(specs.trunk, out, synthetic.root, mode=mode, extra=("--until", "14")) == 0
    assert run_train(specs.trunk, out, synthetic.root, mode=mode) == 0
    got = curve_of(out, mode, trunk_resolved)
    want = curve_of(reference(mode), mode, trunk_resolved)
    assert len(want) == TRUNK_STEPS // EVAL_EVERY
    assert same_value(got, want, RESUME_TOL[mode])


def test_a_curve_file_without_the_context_blocks_is_refused(tmp_path):
    path = tmp_path / "val_curve.json"
    path.write_text(json.dumps([{"step": 4, "tokens": 64, "val_loss": 3.0}]), encoding="utf-8")
    with pytest.raises(rc.EntryError):
        rc.ValCurve(path).load()


def test_a_branch_manifest_carries_the_same_records(specs, synthetic, trunk_resolved, tmp_path, readings):
    """A branch's own sessions are read by the branch: its points carry the blocks, its peak is its own."""
    out = tmp_path / "own_peaks"
    readings["pair"] = (1000, 2000)
    assert run_train(specs.trunk, out, synthetic.root) == 0
    readings["pair"] = (50, 60)
    assert run_branch(specs.trunk, out, synthetic.root) == 0
    name = mf.run_name(trunk_resolved, SHORT_LABEL)
    m = manifest_of(out, "opt", name)
    assert mf.validate_manifest(m) == []
    assert m["kind"] == "branch" and m["completed"] is True
    curve = m["final_metrics"]["val_curve"]
    assert [e["step"] for e in curve] == [20, 24]  # the branch's own steps: 19 is its start (BRANCH_STEP)
    for entry in curve:
        assert set(entry) == CURVE_ENTRY_KEYS
        assert scale_relative_error(
            sum(b["n_tokens"] * b["loss"] for b in entry["buckets"] if b["n_tokens"])
            / sum(b["n_tokens"] for b in entry["buckets"]),
            entry["val_loss"],
        ) <= FLOAT64_IDENTITY
    assert m["final_metrics"]["peak_memory"] == {"allocated_bytes": 50, "reserved_bytes": 60}
    trunk = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert trunk["final_metrics"]["peak_memory"] == {"allocated_bytes": 1000, "reserved_bytes": 2000}


# --- 2. Peak memory --------------------------------------------------------------------------------


def test_on_the_cpu_both_peaks_are_none(reference, trunk_resolved):
    m = manifest_of(reference("opt"), "opt", mf.run_name(trunk_resolved))
    assert m["final_metrics"]["peak_memory"] == {"allocated_bytes": None, "reserved_bytes": None}
    assert rc.peak_memory(torch.device("cpu")) == {"allocated_bytes": None, "reserved_bytes": None}
    rc.reset_peak_memory(torch.device("cpu"))  # a no-op here; must not raise


def test_the_peak_statistics_start_before_the_model_is_built(specs, synthetic, monkeypatch, tmp_path):
    order = []
    real_build = train_script.build_model
    monkeypatch.setattr(rc, "reset_peak_memory", lambda device: order.append("reset"))

    def build(resolved):
        order.append("build")
        return real_build(resolved)

    monkeypatch.setattr(train_script, "build_model", build)
    assert run_train(specs.trunk, tmp_path / "order", synthetic.root, extra=("--until", "3")) == 0
    assert order == ["reset", "build"]


def test_the_peak_is_the_maximum_over_three_sessions(specs, synthetic, trunk_resolved, readings, tmp_path):
    out = tmp_path / "three"
    readings["pair"] = FIRST
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "14")) == 0
    readings["pair"] = SECOND
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "30")) == 0
    readings["pair"] = THIRD
    assert run_train(specs.trunk, out, synthetic.root) == 0
    assert readings["resets"] == 3
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert len(m["session_ids"]) == 3
    assert m["final_metrics"]["peak_memory"] == MAX_OVER_THREE
    peaks = json.loads((ckpt_dir_of(out, "opt", trunk_resolved) / rc.PEAK_FILE).read_text(encoding="utf-8"))
    assert sorted(peaks) == sorted(m["session_ids"])
    readings_by_session = [
        {"allocated_bytes": a, "reserved_bytes": r} for a, r in (FIRST, SECOND, THIRD)
    ]
    assert [peaks[sid] for sid in m["session_ids"]] == readings_by_session


KILL_AFTER_CHECKPOINT = """
import json, os, signal, sys
from scripts import _run_common as rc
from scripts import train

args = json.loads(sys.argv[1])
rc.peak_memory = lambda device: {"allocated_bytes": %d, "reserved_bytes": %d}
original = rc.RunHook.__call__


def killed_after_checkpoint(self, trainer, record):
    original(self, trainer, record)
    if trainer.step == args["kill_step"]:
        os.kill(os.getpid(), signal.SIGKILL)


rc.RunHook.__call__ = killed_after_checkpoint
train.main(args["argv"])
""" % SECOND


def test_a_session_killed_after_a_checkpoint_still_counts(specs, synthetic, trunk_resolved, readings, tmp_path):
    """Session 2 is killed by SIGKILL right after its checkpoint at step 2 * CKPT_EVERY.

    Its reading is the only one that gives the maximum allocated, so the maximum is wrong
    if the killed session is not counted. Session 3 finishes the run.
    """
    out = tmp_path / "killed"
    readings["pair"] = FIRST
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "12")) == 0
    argv = train_argv(specs.trunk, out, synthetic.root)
    payload = json.dumps({"argv": argv, "kill_step": 2 * CKPT_EVERY})
    proc = subprocess.run(
        [sys.executable, "-c", KILL_AFTER_CHECKPOINT, payload],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == -signal.SIGKILL, proc.stderr
    directory = ckpt_dir_of(out, "opt", trunk_resolved)
    assert len(json.loads((directory / "sessions.json").read_text(encoding="utf-8"))) == 2
    readings["pair"] = THIRD
    assert run_train(specs.trunk, out, synthetic.root) == 0
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert len(m["session_ids"]) == 3 and m["completed"] is True
    assert m["final_metrics"]["peak_memory"] == MAX_OVER_THREE


def test_the_peak_file_is_not_a_checkpoint(reference, trunk_resolved):
    directory = ckpt_dir_of(reference("opt"), "opt", trunk_resolved)
    assert (directory / rc.PEAK_FILE).is_file()
    assert rc.PEAK_FILE not in [p.name for p in ck.list_checkpoints(directory)]
    assert rc.PEAK_FILE not in [p.name for p in ck.list_branch_points(directory)]


def test_a_malformed_peak_file_is_refused(tmp_path):
    path = tmp_path / rc.PEAK_FILE
    path.write_text(json.dumps({"s1": {"allocated_bytes": 1}}), encoding="utf-8")
    with pytest.raises(rc.EntryError):
        rc.PeakMemory(path, "s2", torch.device("cpu")).record()


def test_an_unfinished_manifest_keeps_its_shape(specs, synthetic, trunk_resolved, tmp_path, readings):
    readings["pair"] = FIRST
    out = tmp_path / "stopped"
    assert run_train(specs.trunk, out, synthetic.root, extra=("--until", "9")) == 0
    m = manifest_of(out, "opt", mf.run_name(trunk_resolved))
    assert m["completed"] is False and m["final_metrics"] is None
    assert mf.validate_manifest(m) == []
    assert (ckpt_dir_of(out, "opt", trunk_resolved) / rc.PEAK_FILE).is_file()
