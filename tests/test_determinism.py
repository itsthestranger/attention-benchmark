"""T-19: determinism in `ref` mode, and reproducibility measurement in `opt` mode (plan §5.1, §13).

Plan rows:
  * Determinism (`ref` only): same config and seeds give bit-identical loss for
    100 steps, twice. Tiers: cpu, gpu.
  * Reproducibility (`opt`): two identical runs land within a measured, recorded
    tolerance. Tiers: gpu, t4. The tolerance is measured and recorded, never
    judged here, so these tests check structure and finiteness only.

SYNTHETIC data only (tests/fixtures/synthetic_tokens.py). Bit-identity is checked
with ==: no tolerance is involved.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from src.data.loader import TrainStream
from src.model import build_model
from src.training.trainer import Trainer
from src.utils.reproducibility import (
    RunTrace,
    compare_runs,
    deterministic_algorithms,
    measure_reproducibility,
    reproducibility_record,
    run_losses,
)
from scripts.measure_reproducibility import main as measure_main
from tests.fixtures.synthetic_tokens import rule_split
from tests.variants import CASES, VOCAB, tiny_resolved

T, B, MICRO = 16, 4, 2  # context length, sequences per step, micro-batch
STEPS = 100  # plan §5.1: 100 steps
SCALER = {"recipe.grad_scaler.init_scale": 65536.0, "recipe.grad_scaler.growth_interval": 2000}


# --- Builders (synthetic tokens, tiny models) ---------------------------------------------------


def resolved_for(case, *, steps=STEPS, init_seed=1, **overrides):
    """The tiny model of tests/variants.py for this case, with a small batch and one run's steps."""
    extra = {
        "recipe.batch.sequences_per_step": B,
        "recipe.batch.sequence_length": T,
        "recipe.batch.tokens_per_step": B * T,
        "recipe.batch.micro_batch": MICRO,
        "run.steps": steps,
        "run.branches": [],
    }
    extra.update(overrides)
    return tiny_resolved(case, init_seed=init_seed, context_length=T, **extra)


def stream_for(steps=STEPS, data_seed=1):
    split = rule_split(steps * B * T + 1, VOCAB)
    return TrainStream(split, context_length=T, sequences_per_step=B, data_seed=data_seed)


def builder(case, *, steps=STEPS, init_seed=1, mode="ref", precision=None, device="cpu", stream=None, **overrides):
    """A build_run() callable: a fresh model and trainer on every call."""
    cfg = resolved_for(case, steps=steps, init_seed=init_seed, **overrides)
    stream = stream if stream is not None else stream_for(steps)

    def build_run():
        return Trainer(build_model(cfg), cfg, stream, device=device, mode=mode, precision=precision)

    return build_run


@pytest.fixture
def determinism_state():
    """Save and restore torch's deterministic-algorithms setting around a test."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


# --- cpu tier ------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_ref_mode_is_bit_identical_over_100_steps(case, determinism_state):
    """Plan §5.1: same config and seeds, two fresh runs, identical losses at every step."""
    stream = stream_for(STEPS)
    build = builder(case, mode="ref", stream=stream)
    with deterministic_algorithms():
        a = run_losses(build, STEPS)
        b = run_losses(build, STEPS)
    assert all(math.isfinite(x) for x in a.losses), "losses must be finite"
    assert a.losses == b.losses
    assert a.grad_norms == b.grad_norms
    result = compare_runs(a, b)
    assert result["bit_identical"] is True
    assert result["first_differing_step"] is None
    assert result["max_abs_loss_diff"] == 0.0
    assert result["n_steps"] == STEPS


def test_different_init_seeds_are_not_bit_identical(determinism_state):
    """The comparison is not vacuous: a changed seed changes the losses."""
    case = CASES[0]
    with deterministic_algorithms():
        a = run_losses(builder(case, init_seed=1), 20)
        b = run_losses(builder(case, init_seed=2), 20)
    result = compare_runs(a, b)
    assert result["bit_identical"] is False
    assert result["first_differing_step"] == 0
    assert result["max_abs_loss_diff"] > 0


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_determinism_does_not_depend_on_the_global_rng(case, determinism_state):
    """Different global seeds before each run give the same losses: nothing reads the global RNG."""
    build = builder(case, mode="ref")
    with deterministic_algorithms():
        torch.manual_seed(12345)
        a = run_losses(build, STEPS)
        torch.manual_seed(987654)
        b = run_losses(build, STEPS)
    assert compare_runs(a, b)["bit_identical"] is True
    assert a.losses == b.losses


@pytest.mark.parametrize("prev, prev_warn", [(False, False), (True, False), (True, True)])
def test_deterministic_algorithms_restores_previous_setting(prev, prev_warn, determinism_state):
    torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    with deterministic_algorithms():
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    assert torch.are_deterministic_algorithms_enabled() is prev
    assert torch.is_deterministic_algorithms_warn_only_enabled() is prev_warn


@pytest.mark.parametrize("prev", [False, True])
def test_deterministic_algorithms_restores_after_an_exception(prev, determinism_state):
    torch.use_deterministic_algorithms(prev)
    with pytest.raises(ValueError, match="boom"):
        with deterministic_algorithms():
            raise ValueError("boom")
    assert torch.are_deterministic_algorithms_enabled() is prev


def test_cuda_without_cublas_config_raises_before_changing_anything(monkeypatch, determinism_state):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        with deterministic_algorithms(device="cuda"):
            pass
    assert torch.are_deterministic_algorithms_enabled() is False


def test_cpu_needs_no_cublas_config(monkeypatch, determinism_state):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with deterministic_algorithms(device="cpu"):
        assert torch.are_deterministic_algorithms_enabled() is True


def _trace(losses, *, lrs=None, skipped=None, steps_taken=None):
    n = len(losses)
    return RunTrace(
        losses=list(losses),
        lrs=[0.1] * n if lrs is None else list(lrs),
        grad_norms=[1.0] * n,
        skipped=[False] * n if skipped is None else list(skipped),
        steps_taken=n if steps_taken is None else steps_taken,
    )


def test_compare_runs_numbers_on_hand_built_traces():
    a = _trace([1.0, 2.0])
    b = _trace([1.0, 2.5])
    result = compare_runs(a, b)
    assert result == {
        "n_steps": 2,
        "bit_identical": False,
        "max_abs_loss_diff": 0.5,
        "max_rel_loss_diff": 0.2,
        "final_abs_loss_diff": 0.5,
        "first_differing_step": 1,
    }
    # The reference is the second argument: |2.0 - 2.5| / 2.0.
    reversed_result = compare_runs(b, a)
    assert reversed_result["max_rel_loss_diff"] == 0.25


def test_compare_runs_identical_traces():
    a = _trace([1.0, 2.0, 3.0])
    result = compare_runs(a, _trace([1.0, 2.0, 3.0]))
    assert result["bit_identical"] is True
    assert result["first_differing_step"] is None
    assert result["max_abs_loss_diff"] == 0.0
    assert result["max_rel_loss_diff"] == 0.0


def test_compare_runs_rejects_different_lengths():
    with pytest.raises(ValueError):
        compare_runs(_trace([1.0, 2.0]), _trace([1.0]))


def test_compare_runs_rejects_different_learning_rates():
    with pytest.raises(ValueError):
        compare_runs(_trace([1.0, 2.0], lrs=[0.1, 0.2]), _trace([1.0, 2.0], lrs=[0.1, 0.3]))


def test_compare_runs_rejects_different_skipped_flags():
    with pytest.raises(ValueError):
        compare_runs(_trace([1.0, 2.0], skipped=[False, True]), _trace([1.0, 2.0], skipped=[False, False]))


def test_compare_runs_rejects_different_steps_taken():
    with pytest.raises(ValueError):
        compare_runs(_trace([1.0, 2.0], steps_taken=2), _trace([1.0, 2.0], steps_taken=1))


def test_measure_reproducibility_structure_on_cpu(determinism_state):
    """Structure and consistency only: CPU `opt` runs are not asserted to be bit-identical."""
    case = CASES[0]
    measurement = measure_reproducibility(builder(case, mode="opt", steps=5), 5, 2)
    assert measurement["n_steps"] == 5
    assert measurement["n_pairs"] == 2
    assert len(measurement["pairs"]) == 2
    assert measurement["max_abs_loss_diff"] == max(p["max_abs_loss_diff"] for p in measurement["pairs"])
    assert measurement["max_rel_loss_diff"] == max(p["max_rel_loss_diff"] for p in measurement["pairs"])
    assert measurement["bit_identical"] == all(p["bit_identical"] for p in measurement["pairs"])
    assert all(p["n_steps"] == 5 for p in measurement["pairs"])
    json.dumps(measurement)


class _ScriptedTrainer:
    """Stands in for a trainer whose losses are written out in the test."""

    def __init__(self, losses):
        self._losses = losses
        self.step = 0
        self.steps_taken = 0
        self.records = []

    def train(self, until):
        from src.training.trainer import StepRecord

        for i in range(until):
            self.records.append(StepRecord(i, 0.1, self._losses[i], 1.0, False, 1.0, 0.0))
        self.step = self.steps_taken = until


def test_measure_reproducibility_compares_every_run_with_the_first():
    """Added in review. Three scripted runs: the second differs from the first by 0.5
    at most, the third by 2.0. Both differences must appear, and the maximum is the larger."""
    scripts = iter([[1.0, 2.0], [1.0, 2.5], [1.0, 4.0]])
    measurement = measure_reproducibility(lambda: _ScriptedTrainer(next(scripts)), 2, 2)
    assert [p["max_abs_loss_diff"] for p in measurement["pairs"]] == [0.5, 2.0]
    assert [p["max_rel_loss_diff"] for p in measurement["pairs"]] == [0.25, 1.0]
    assert measurement["max_abs_loss_diff"] == 2.0
    assert measurement["max_rel_loss_diff"] == 1.0
    assert measurement["bit_identical"] is False
    assert [p["first_differing_step"] for p in measurement["pairs"]] == [1, 1]


@pytest.mark.parametrize("mode", ["opt", "ref"])
def test_script_runs_ref_deterministically_and_passes_its_mode_on(tmp_path, mode, monkeypatch, determinism_state):
    """Added in review: `ref` runs inside deterministic_algorithms(), `opt` does not,
    and the trainer is built in the mode asked for."""
    import scripts.measure_reproducibility as script

    seen = {}
    real_trainer = script.Trainer

    def recording_trainer(*args, **kwargs):
        seen.setdefault("deterministic", []).append(torch.are_deterministic_algorithms_enabled())
        seen.setdefault("mode", []).append(kwargs["mode"])
        return real_trainer(*args, **kwargs)

    torch.use_deterministic_algorithms(False)
    monkeypatch.setattr(script, "Trainer", recording_trainer)
    out = tmp_path / "r.json"
    assert script.main(["--steps", "3", "--pairs", "1", "--mode", mode, "--synthetic", "--out", str(out)]) == 0
    assert seen["mode"] == [mode, mode]
    assert seen["deterministic"] == [mode == "ref"] * 2
    assert torch.are_deterministic_algorithms_enabled() is False
    assert json.loads(out.read_text(encoding="utf-8"))["mode"] == mode


def test_record_carries_identity_and_is_json(determinism_state):
    case = CASES[0]
    resolved = resolved_for(case, steps=5)
    measurement = measure_reproducibility(builder(case, mode="opt", steps=5), 5, 1)
    record = reproducibility_record(measurement, resolved=resolved, mode="opt", precision="fp32", device="cpu")
    assert record["measurement"] == measurement
    assert record["mode"] == "opt"
    assert record["precision"] == "fp32"
    assert record["device"] == "cpu"
    assert record["cuda_version"] is None
    assert record["gpu_name"] is None
    assert len(record["config_hash"]) == 64
    assert record["torch_version"] == torch.__version__
    json.dumps(record)


@pytest.mark.parametrize("mode", ["opt", "ref"])
def test_script_end_to_end_on_synthetic_data(tmp_path, mode, determinism_state):
    out = tmp_path / "nested" / "r.json"
    code = measure_main(
        [
            "--variant", "mha", "--steps", "5", "--pairs", "1", "--mode", mode,
            "--synthetic", "--device", "cpu", "--out", str(out),
        ]
    )
    assert code == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["mode"] == mode
    assert record["precision"] == "fp32"
    assert len(record["config_hash"]) == 64
    assert record["torch_version"] == torch.__version__
    assert record["measurement"]["n_steps"] == 5
    assert record["measurement"]["n_pairs"] == 1
    assert len(record["measurement"]["pairs"]) == 1


# --- gpu tier (written, skipped without CUDA; not yet run) ---------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_ref_mode_is_bit_identical_over_100_steps_fp16_cuda(case, determinism_state):
    stream = stream_for(STEPS)
    build = builder(case, mode="ref", precision="fp16", device="cuda", stream=stream, **SCALER)
    with deterministic_algorithms(device="cuda"):
        a = run_losses(build, STEPS)
        b = run_losses(build, STEPS)
    assert a.losses == b.losses
    assert a.grad_norms == b.grad_norms
    result = compare_runs(a, b)
    assert result["bit_identical"] is True
    assert result["first_differing_step"] is None


def _reproducibility_check(case, n_steps=50, n_pairs=2, **kwargs):
    trainers: list[Trainer] = []
    base = builder(case, mode="opt", steps=n_steps, **kwargs)

    def build_run():
        trainer = base()
        trainers.append(trainer)
        return trainer

    measurement = measure_reproducibility(build_run, n_steps, n_pairs)
    assert measurement["n_pairs"] == n_pairs
    assert len(measurement["pairs"]) == n_pairs
    for trainer in trainers:
        assert all(math.isfinite(r.loss) for r in trainer.records)
    json.dumps(reproducibility_record(
        measurement,
        resolved=resolved_for(case, steps=n_steps),
        mode="opt",
        precision=trainers[0].precision,
        device=trainers[0].device.type,
    ))
    return measurement


@pytest.mark.gpu
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_opt_mode_reproducibility_fp16_cuda(case):
    """Measures and records run-to-run differences in opt mode. No threshold is applied here."""
    _reproducibility_check(case, precision="fp16", device="cuda", **SCALER)


@pytest.mark.t4
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_opt_mode_reproducibility_fp16_t4(case):
    """The same measurement on the Kaggle T4 (plan §5.1). No threshold is applied here."""
    _reproducibility_check(case, precision="fp16", device="cuda", **SCALER)
