"""T-33 Part C: a run whose loss is not finite leaves a record, and is never silently retried.

CPU, tiny model and SYNTHETIC data, with the fixtures and helpers of tests/test_entry_points.py
imported unchanged. No source file is edited to force a divergence. Two ways are used:

- A NaN model: every parameter of a tiny model is NaN, so its logits and loss are NaN.
  The evaluator must raise NonFiniteLoss there (test_evaluate_...).
- A forced divergence: `diverge_on` replaces `scripts._run_common.evaluate` (the name the
  entry points call) with a wrapper that raises NonFiniteLoss on one chosen call. In-run
  validation calls pass `max_windows`; the final evaluation does not, so the wrapper can
  tell them apart: the k-th validation point, or the k-th final evaluation (validation,
  then test).
"""

from __future__ import annotations

import copy
import shutil

import numpy as np
import pytest
import torch

from analysis import lr_sweep as sw
from scripts import _run_common as rc
from scripts import make_tables as mt
from src import config
from src.data.loader import TokenSplit
from src.eval.lm import ContextSpec, NonFiniteLoss, evaluate
from src.model import build_model
from src.utils import manifest as mf
from tests.test_entry_points import (  # noqa: F401  (fixtures, imported unchanged)
    EVAL_EVERY,
    PEAK_LR,
    SHORT_LABEL,
    TRUNK_LABEL,
    TRUNK_STEPS,
    branch_source,
    final_state,
    manifest_of,
    run_branch,
    run_train,
    snapshot,
    specs,
    synthetic,
    trunk_resolved,
)

MESSAGE = "non-finite loss in window 7 (forced by a test)"
# The first in-run validation point after the branch point (19 for the 24-step budget) is step 20.
BRANCH_DIVERGE_STEP = 20


def diverge_on(monkeypatch, *, when: str, call: int) -> None:
    """Make the `call`-th evaluation of `when` raise NonFiniteLoss; the others run as usual.

    `when` is "validation" (the in-run points, which pass max_windows) or "final" (the final
    validation split, then the test split).
    """
    real = rc.evaluate
    seen = {"validation": 0, "final": 0}

    def evaluate_or_diverge(model, split, **kwargs):
        kind = "validation" if kwargs["max_windows"] is not None else "final"
        seen[kind] += 1
        if kind == when and seen[kind] == call:
            raise NonFiniteLoss(MESSAGE)
        return real(model, split, **kwargs)

    monkeypatch.setattr(rc, "evaluate", evaluate_or_diverge)


# --- The evaluator --------------------------------------------------------------------------------


def test_a_non_finite_loss_is_a_value_error():
    assert issubclass(NonFiniteLoss, ValueError)


def test_evaluate_raises_non_finite_loss_for_a_model_with_nan_weights(synthetic, trunk_resolved):
    T = trunk_resolved["model"]["context_length"]
    spec = ContextSpec.from_prereg(config.load_prereg())
    tokens = np.zeros(4 * T + 1, dtype=np.uint16)  # SYNTHETIC: every token is id 0
    split = TokenSplit(tokens, np.array([0, len(tokens)], dtype=np.int64))

    healthy = evaluate(build_model(trunk_resolved), split, context_length=T, batch_size=2,
                       spec=spec, precision="fp32")
    assert np.isfinite(healthy["loss"])  # the control: a healthy model is not refused

    broken = build_model(trunk_resolved)
    with torch.no_grad():
        for parameter in broken.parameters():
            parameter.fill_(float("nan"))
    with pytest.raises(NonFiniteLoss, match="non-finite loss in window 0"):
        evaluate(broken, split, context_length=T, batch_size=2, spec=spec, precision="fp32")


# --- A training run ---------------------------------------------------------------------------------


def test_a_run_that_diverges_at_a_validation_point_is_recorded_and_exits_3(
    synthetic, specs, trunk_resolved, tmp_path, monkeypatch, capsys
):
    diverge_on(monkeypatch, when="validation", call=2)  # the second in-run point: step 8
    assert run_train(specs.trunk, tmp_path, synthetic.root) == rc.DIVERGED_EXIT

    name = mf.run_name(trunk_resolved)
    manifest = manifest_of(tmp_path, "opt", name)
    assert mf.validate_manifest(manifest) == []
    assert manifest["completed"] is False
    assert mf.is_result(manifest) is False
    assert manifest["kind"] == "run"
    assert manifest["final_metrics"] == {"diverged": True, "reason": MESSAGE, "step": 2 * EVAL_EVERY}
    assert manifest["steps"]["total"] == TRUNK_STEPS
    assert manifest["steps"]["attempted"] == 2 * EVAL_EVERY
    assert manifest["resume_count"] == 0
    assert "diverged at step 8 of 32" in capsys.readouterr().out

    results, skipped = mt.select_results([manifest], skip_unfinished=True)
    assert results == [] and skipped == [name]


def test_the_recorded_divergence_is_a_diverged_point_of_the_sweep(
    synthetic, specs, trunk_resolved, tmp_path, monkeypatch
):
    diverge_on(monkeypatch, when="validation", call=2)
    assert run_train(specs.trunk, tmp_path, synthetic.root) == rc.DIVERGED_EXIT
    manifest = manifest_of(tmp_path, "opt", mf.run_name(trunk_resolved))

    # The tiny runs are at the 32M budget, not the 100M sweep budget: the check reads that budget.
    prereg = copy.deepcopy(config.load_prereg())
    prereg["test_selection"]["sigma_init"]["budget"] = TRUNK_LABEL
    (point,) = sw.points_from_manifests([manifest], "mha", sweep_seed=1, prereg=prereg)
    assert point.diverged and point.loss is None
    assert point.reason == MESSAGE
    assert point.lr == PEAK_LR and point.run == manifest["name"]


def test_a_second_invocation_trains_nothing_changes_no_file_and_exits_3(
    synthetic, specs, trunk_resolved, tmp_path, monkeypatch, capsys
):
    diverge_on(monkeypatch, when="validation", call=2)
    assert run_train(specs.trunk, tmp_path, synthetic.root) == rc.DIVERGED_EXIT
    capsys.readouterr()
    before = snapshot(tmp_path)

    # No monkeypatch now: a retry that trained would find a healthy evaluator and change files.
    monkeypatch.undo()
    assert run_train(specs.trunk, tmp_path, synthetic.root) == rc.DIVERGED_EXIT
    assert snapshot(tmp_path) == before
    assert "recorded as diverged at step 8" in capsys.readouterr().out


def test_a_run_that_diverges_in_the_final_evaluation_is_recorded_and_keeps_its_checkpoints(
    synthetic, specs, trunk_resolved, tmp_path, monkeypatch
):
    diverge_on(monkeypatch, when="final", call=1)  # the final validation split
    assert run_train(specs.trunk, tmp_path, synthetic.root) == rc.DIVERGED_EXIT

    manifest = manifest_of(tmp_path, "opt", mf.run_name(trunk_resolved))
    assert mf.validate_manifest(manifest) == []
    assert manifest["completed"] is False and mf.is_result(manifest) is False
    assert manifest["final_metrics"] == {"diverged": True, "reason": MESSAGE, "step": TRUNK_STEPS}
    # Nothing was deleted: the final checkpoint the run had written is still there.
    assert final_state(tmp_path, trunk_resolved, "opt")["step"] == TRUNK_STEPS


# --- A branch ---------------------------------------------------------------------------------------


def test_a_branch_that_diverges_is_recorded_exits_3_and_is_refused_again(
    branch_source, specs, synthetic, trunk_resolved, tmp_path, monkeypatch, capsys
):
    work = tmp_path / "work"
    shutil.copytree(branch_source, work)
    diverge_on(monkeypatch, when="validation", call=1)
    assert run_branch(specs.trunk, work, synthetic.root) == rc.DIVERGED_EXIT

    name = mf.run_name(trunk_resolved, SHORT_LABEL)
    manifest = manifest_of(work, "opt", name)
    assert manifest["kind"] == "branch" and manifest["branch"]["budget"] == SHORT_LABEL
    assert mf.validate_manifest(manifest) == []
    assert manifest["completed"] is False and mf.is_result(manifest) is False
    assert manifest["final_metrics"] == {"diverged": True, "reason": MESSAGE, "step": BRANCH_DIVERGE_STEP}
    capsys.readouterr()

    before = snapshot(work)
    monkeypatch.undo()
    assert run_branch(specs.trunk, work, synthetic.root) == rc.DIVERGED_EXIT
    assert snapshot(work) == before
    assert "recorded as diverged" in capsys.readouterr().out
