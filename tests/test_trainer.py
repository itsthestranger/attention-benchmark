"""T-15: the trainer (plan §2.3, §7, §13.6). SYNTHETIC data only.

Written by the orchestrator before the implementation; the worker may not
change this file.

Plan §13.6, the highest-priority failure mode: GradScaler skips optimizer steps
on inf/nan gradients, so two variants can see identical tokens while taking
different numbers of steps, and no curve looks wrong. The trainer therefore
counts attempted and taken steps separately and never hides a skip.

The contract for `src/training/trainer.py`
------------------------------------------
    @dataclass(frozen=True)
    class StepRecord:
        step: int          # index of this step: steps attempted before it
        lr: float          # lr_at(step, resolved)
        loss: float        # mean cross-entropy of the step's tokens, in nats
        grad_norm: float   # global gradient norm before clipping, unscaled; inf or nan if not finite
        skipped: bool      # True if no optimizer update was applied
        scale: float       # GradScaler scale after the step; 1.0 when scaling is off
        seconds: float

    class Trainer:
        def __init__(self, model, resolved, stream, *, device="cpu", mode="opt",
                     precision=None, expected_backend=None, on_step=None)
        model, optimizer, resolved
        step: int             # steps attempted: the position in the data stream and the schedule
        steps_taken: int      # optimizer updates actually applied
        skipped_steps: int    # step - steps_taken
        total_steps: int      # resolved["run"]["steps"]
        records: list[StepRecord]
        def train_step(self) -> StepRecord
        def train(self, until: int | None = None) -> None   # until step == until; default total_steps
        def summary(self) -> dict

One step, in this order:
 1. `x, y = stream.batch(self.step)`; `set_lr(optimizer, lr_at(self.step, resolved))`;
    `optimizer.zero_grad(set_to_none=True)`.
 2. Split the step's sequences, in order, into micro-batches of
    `recipe.batch.micro_batch` sequences. For each: forward in `mode`, check the
    logits dtype, cross-entropy over its tokens, multiply by
    micro_batch / sequences_per_step, scale, backward. The gradients add up to
    the gradient of the mean loss over the whole step.
 3. `scaler.unscale_(optimizer)` once, then `clip_grad_norm_` at
    `recipe.optimizer.grad_clip_norm`. Its return value is `grad_norm`.
 4. The update is applied only if `grad_norm` is finite. With scaling on, that
    is GradScaler's own rule, and the trainer checks that the scaler agreed
    (the scale went down exactly when the norm was not finite), raising
    RuntimeError otherwise. With scaling off the trainer skips the update itself.
 5. `step` goes up by one whether or not the update was applied: a skipped
    step has still consumed its tokens and its place in the schedule.
 6. The record is appended and `on_step(trainer, record)` is called, if given.
    An exception from it stops the run at that step.
Gradients are left in place after a step (they are cleared at the start of the next).

- `precision`: "fp32" (no autocast, no scaling) or "fp16" (autocast to float16
  and `torch.amp.GradScaler` with `init_scale` and `growth_interval` from the
  recipe). Default: "fp16" on a CUDA device, "fp32" on the CPU. "fp16" on the
  CPU is a ValueError. The logits of every micro-batch must have the dtype the
  precision implies (float32 or float16), else RuntimeError: this is the
  in-loop dtype assertion of plan §3.1.
- `expected_backend`: None, or the name of the one fused attention backend the
  run may use ("flash", "efficient", "math", "cudnn"). When given, every forward
  runs inside `torch.nn.attention.sdpa_kernel` restricted to it, so a fallback
  fails inside the loop (plan §2.3). CUDA only: ValueError on the CPU.
- Constructor errors (ValueError): `micro_batch` unset (None) or not a divisor
  of `sequences_per_step`; "fp16" with `init_scale` or `growth_interval` unset;
  a stream whose `max_steps` is below `total_steps`; an unknown precision or mode.
- `train(until)`: ValueError if `until` is below `step` or above `total_steps`.
  `train_step()` when `step == total_steps`: ValueError.
- The trainer moves the model to `device` when constructed and each batch as
  it is used; the stream stays on the CPU.
- `summary()` returns at least: "steps_attempted", "steps_taken",
  "skipped_steps", "skipped_in_first_100", "final_scale", "precision",
  "micro_batch", "mode", "expected_backend".
- The trainer uses `build_optimizer`, `lr_at` and `set_lr`; it draws no random
  numbers and leaves the global RNG state alone.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src import config
from src.data.loader import TrainStream
from src.model import build_model
from src.training.schedule import lr_at
from src.training.trainer import StepRecord, Trainer
from tests.fixtures.synthetic_tokens import rule_split
from tests.tolerances import GRAD_ACCUMULATION_FP32, REF_OPT_FP16, REF_OPT_FP32, scale_relative_error

VOCAB, T, B = 64, 32, 8


def resolved_cfg(*, steps=40, peak_lr=1e-2, micro_batch=4, variant="mha", **overrides):
    dims = {
        "model.d_model": 64,
        "model.n_heads": 8,
        "model.head_dim": 8,
        "model.n_layers": 2,
        "model.d_ff": 96,
        "model.vocab_size": VOCAB,
        "model.context_length": T,
        "recipe.batch.sequences_per_step": B,
        "recipe.batch.sequence_length": T,
        "recipe.batch.tokens_per_step": B * T,
        "recipe.batch.micro_batch": micro_batch,
    }
    dims.update(overrides)
    return config.resolve(variant, init_seed=1, peak_lr=peak_lr, steps=steps, branches=[], overrides=dims)


def make_stream(steps, data_seed=1):
    split = rule_split(steps * B * T + 1, VOCAB)
    return TrainStream(split, context_length=T, sequences_per_step=B, data_seed=data_seed)


def make_trainer(*, steps=40, stream=None, **kwargs):
    trainer_kwargs = {k: kwargs.pop(k) for k in ("mode", "precision", "expected_backend", "on_step", "device") if k in kwargs}
    cfg = resolved_cfg(steps=steps, **kwargs)
    model = build_model(cfg)
    return Trainer(model, cfg, stream or make_stream(steps), **trainer_kwargs)


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def poison_next_gradient(trainer, value):
    """Make the gradient of one weight non-finite during the next backward pass only."""
    weight = trainer.model.layers[0].mlp.gate.weight
    handle = weight.register_hook(lambda g: torch.full_like(g, value))
    return handle


# --- A tiny model overfits on CPU ---------------------------------------------------------


def test_a_tiny_model_learns_the_synthetic_rule():
    trainer = make_trainer(steps=150)
    trainer.train()
    assert trainer.step == trainer.steps_taken == 150 and trainer.skipped_steps == 0
    losses = [r.loss for r in trainer.records]
    assert losses[-1] < losses[0]
    # Every next token is determined by the current one, so a trained model gets all of them.
    x, y = make_stream(150).batch(0)
    with torch.no_grad():
        logits, _ = trainer.model(x, mode="opt")
    assert torch.equal(logits.argmax(-1), y)


def test_records_follow_the_schedule_and_the_stream():
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner, self.max_steps = inner, inner.max_steps

        def batch(self, step):
            seen.append(step)
            return self.inner.batch(step)

    trainer = make_trainer(steps=30, stream=Recording(make_stream(30)))
    trainer.train(until=12)
    assert trainer.step == 12 and seen == list(range(12))
    trainer.train()
    assert trainer.step == 30 and seen == list(range(30))  # each batch once, in order
    assert [r.step for r in trainer.records] == list(range(30))
    for r in trainer.records:
        assert isinstance(r, StepRecord)
        assert r.lr == lr_at(r.step, trainer.resolved)
        assert math.isfinite(r.loss) and math.isfinite(r.grad_norm) and r.grad_norm > 0
        assert r.skipped is False and r.scale == 1.0 and r.seconds > 0


def test_the_optimizer_sees_the_scheduled_learning_rate():
    lrs = []
    trainer = make_trainer(
        steps=10, on_step=lambda tr, rec: lrs.append([g["lr"] for g in tr.optimizer.param_groups])
    )
    trainer.train()
    for step, groups in enumerate(lrs):
        assert groups == [lr_at(step, trainer.resolved)] * 2


def test_loss_is_the_mean_cross_entropy_of_the_step():
    trainer = make_trainer(steps=5)
    x, y = make_stream(5).batch(0)
    with torch.no_grad():
        logits, _ = trainer.model(x, mode="opt")
        expected = F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    record = trainer.train_step()
    assert scale_relative_error(record.loss, expected) <= REF_OPT_FP32


def test_training_does_not_touch_the_global_rng():
    trainer = make_trainer(steps=5)
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    trainer.train()
    assert torch.equal(torch.get_rng_state(), before)


def test_two_identical_runs_agree():
    a, b = make_trainer(steps=12), make_trainer(steps=12)
    a.train()
    b.train()
    assert [r.loss for r in a.records] == [r.loss for r in b.records]
    for (name, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters()):
        assert torch.equal(p, q), name


# --- Skipped steps (plan §13.6) ---------------------------------------------------------------


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_gradient_is_a_skipped_step(value):
    trainer = make_trainer(steps=10)
    trainer.train(until=3)
    before = snapshot(trainer.model)
    moments = [s["exp_avg"].clone() for s in trainer.optimizer.state.values()]

    handle = poison_next_gradient(trainer, value)
    record = trainer.train_step()
    handle.remove()

    assert record.skipped is True and record.step == 3
    assert not math.isfinite(record.grad_norm)
    assert trainer.step == 4 and trainer.steps_taken == 3 and trainer.skipped_steps == 1
    for name, p in trainer.model.named_parameters():
        assert torch.equal(p, before[name]), name  # no update, no NaN in the weights
    for old, state in zip(moments, trainer.optimizer.state.values()):
        assert torch.equal(state["exp_avg"], old)  # and none in the optimizer

    after = trainer.train_step()  # the run carries on, one step further along
    assert after.skipped is False and after.step == 4
    assert after.lr == lr_at(4, trainer.resolved)
    assert trainer.step == 5 and trainer.steps_taken == 4
    assert any(not torch.equal(p, before[n]) for n, p in trainer.model.named_parameters())


def test_a_skipped_step_still_consumes_its_batch():
    """Attempted steps index the data: the batch of a skipped step is not replayed."""
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner, self.max_steps = inner, inner.max_steps

        def batch(self, step):
            seen.append(step)
            return self.inner.batch(step)

    trainer = make_trainer(steps=6, stream=Recording(make_stream(6)))
    trainer.train(until=2)
    handle = poison_next_gradient(trainer, float("inf"))
    trainer.train_step()
    handle.remove()
    trainer.train()
    assert seen == [0, 1, 2, 3, 4, 5]
    assert trainer.step == 6 and trainer.steps_taken == 5


def test_summary_counts_attempted_taken_and_skipped():
    trainer = make_trainer(steps=8)
    for step in range(8):
        handle = poison_next_gradient(trainer, float("nan")) if step in (1, 5, 6) else None
        trainer.train_step()
        if handle:
            handle.remove()
    summary = trainer.summary()
    assert summary["steps_attempted"] == 8
    assert summary["steps_taken"] == 5
    assert summary["skipped_steps"] == 3
    assert summary["skipped_in_first_100"] == 3
    assert summary["final_scale"] == 1.0
    assert summary["precision"] == "fp32"
    assert summary["micro_batch"] == 4
    assert summary["mode"] == "opt"
    assert summary["expected_backend"] is None
    assert [r.skipped for r in trainer.records] == [s in (1, 5, 6) for s in range(8)]


# --- Accumulation matches the large batch ------------------------------------------------------


@pytest.mark.parametrize("micro_batch", [1, 2, 4])
def test_accumulation_matches_one_large_batch(micro_batch):
    whole = make_trainer(steps=6, micro_batch=B)
    parts = make_trainer(steps=6, micro_batch=micro_batch)
    first_whole, first_parts = whole.train_step(), parts.train_step()
    assert scale_relative_error(first_parts.loss, first_whole.loss) <= GRAD_ACCUMULATION_FP32
    assert scale_relative_error(first_parts.grad_norm, first_whole.grad_norm) <= GRAD_ACCUMULATION_FP32
    whole.train()
    parts.train()
    for (name, p), (_, q) in zip(parts.model.named_parameters(), whole.model.named_parameters()):
        assert scale_relative_error(p.detach(), q.detach()) <= GRAD_ACCUMULATION_FP32, name


def test_micro_batches_are_taken_in_order_and_cover_the_step():
    calls = []
    trainer = make_trainer(steps=2, micro_batch=2)
    original = trainer.model.forward

    def recording(*args, **kwargs):
        calls.append((args[0] if args else kwargs["idx"]).clone())
        return original(*args, **kwargs)

    trainer.model.forward = recording
    trainer.train_step()
    x, _ = make_stream(2).batch(0)
    assert [tuple(c.shape) for c in calls] == [(2, T)] * 4
    assert torch.equal(torch.cat(calls), x)


# --- Clipping -------------------------------------------------------------------------------------


def test_grad_norm_is_the_norm_before_clipping_and_gradients_are_clipped():
    trainer = make_trainer(steps=4, **{"recipe.optimizer.grad_clip_norm": 0.05})
    x, y = make_stream(4).batch(0)
    reference = build_model(trainer.resolved)
    logits, _ = reference(x, mode="opt")
    F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
    expected = torch.sqrt(sum(p.grad.pow(2).sum() for p in reference.parameters())).item()
    assert expected > 0.05  # the clip is active

    record = trainer.train_step()
    assert scale_relative_error(record.grad_norm, expected) <= REF_OPT_FP32
    left = torch.sqrt(sum(p.grad.pow(2).sum() for p in trainer.model.parameters())).item()
    assert scale_relative_error(left, 0.05) <= REF_OPT_FP32  # gradients stay in place, clipped


def test_a_gradient_below_the_clip_is_left_alone():
    trainer = make_trainer(steps=4, **{"recipe.optimizer.grad_clip_norm": 1.0e6})
    record = trainer.train_step()
    left = torch.sqrt(sum(p.grad.pow(2).sum() for p in trainer.model.parameters())).item()
    assert scale_relative_error(left, record.grad_norm) <= REF_OPT_FP32


# --- In-loop assertions (plan §2.3, §3.1) ---------------------------------------------------------


def test_on_step_runs_inside_the_loop_and_can_stop_the_run():
    def check(trainer, record):
        if record.step == 4:
            raise RuntimeError("backend changed")

    trainer = make_trainer(steps=10, on_step=check)
    with pytest.raises(RuntimeError, match="backend changed"):
        trainer.train()
    assert trainer.step == 5 and len(trainer.records) == 5  # not at startup: at the step itself


def test_wrong_logits_dtype_fails_inside_the_loop():
    trainer = make_trainer(steps=4)
    trainer.train_step()
    trainer.model.double()  # the run's precision is fp32
    with pytest.raises(RuntimeError, match="dtype"):
        trainer.train_step()


# --- Modes and construction ------------------------------------------------------------------------


def test_ref_and_opt_training_agree():
    ref, opt = make_trainer(steps=8, mode="ref"), make_trainer(steps=8, mode="opt")
    ref.train()
    opt.train()
    assert ref.summary()["mode"] == "ref"
    for r, o in zip(ref.records, opt.records):
        assert scale_relative_error(o.loss, r.loss) <= REF_OPT_FP32
    for (name, p), (_, q) in zip(opt.model.named_parameters(), ref.model.named_parameters()):
        assert scale_relative_error(p.detach(), q.detach()) <= REF_OPT_FP32, name


def test_precision_defaults_to_fp32_on_cpu_and_fp16_is_refused_there():
    assert make_trainer(steps=2).summary()["precision"] == "fp32"
    with pytest.raises(ValueError):
        make_trainer(steps=2, precision="fp16")
    with pytest.raises(ValueError):
        make_trainer(steps=2, precision="bf16")
    with pytest.raises(ValueError):
        make_trainer(steps=2, mode="fast")


def test_expected_backend_is_cuda_only():
    with pytest.raises(ValueError):
        make_trainer(steps=2, expected_backend="efficient")


def test_unset_micro_batch_is_refused():
    with pytest.raises(ValueError, match="micro_batch"):
        make_trainer(steps=2, micro_batch=None)


@pytest.mark.parametrize("micro_batch", [3, 16, 0])
def test_micro_batch_must_divide_the_step(micro_batch):
    with pytest.raises(ValueError, match="micro_batch"):
        make_trainer(steps=2, micro_batch=micro_batch)


def test_a_stream_shorter_than_the_run_is_refused():
    with pytest.raises(ValueError):
        make_trainer(steps=20, stream=make_stream(10))


def test_train_until_is_bounded():
    trainer = make_trainer(steps=6)
    trainer.train(until=3)
    with pytest.raises(ValueError):
        trainer.train(until=2)
    with pytest.raises(ValueError):
        trainer.train(until=7)
    trainer.train(until=3)  # already there: nothing to do
    assert trainer.step == 3
    trainer.train()
    assert trainer.step == trainer.total_steps == 6
    with pytest.raises(ValueError):
        trainer.train_step()  # the run is over


def test_the_recipe_as_frozen_cannot_start_a_run():
    """micro_batch is unset in recipe.yaml until the smoke runs (plan §7.3)."""
    cfg = config.resolve("mha", init_seed=1, peak_lr=1e-3, steps=2, branches=[], overrides={
        "model.d_model": 64, "model.n_heads": 8, "model.head_dim": 8, "model.n_layers": 1,
        "model.d_ff": 96, "model.vocab_size": VOCAB, "model.context_length": T,
        "recipe.batch.sequences_per_step": B,
    })
    assert cfg["recipe"]["batch"]["micro_batch"] is None
    with pytest.raises(ValueError, match="micro_batch"):
        Trainer(build_model(cfg), cfg, make_stream(2))


# --- gpu tier: fp16, GradScaler, backend (plan §5.1) -------------------------------------------------

SCALER = {"recipe.grad_scaler.init_scale": 65536.0, "recipe.grad_scaler.growth_interval": 2000}


@pytest.mark.gpu
def test_fp16_training_learns_and_reports_unscaled_gradient_norms():
    """`unscale_` before clipping: the recorded norm is the true one, not scale times it."""
    fp16 = make_trainer(steps=150, device="cuda", **SCALER)
    assert fp16.summary()["precision"] == "fp16"
    x, y = make_stream(150).batch(0)
    reference = build_model(fp16.resolved)
    logits, _ = reference(x, mode="opt")
    F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
    expected = torch.sqrt(sum(p.grad.pow(2).sum() for p in reference.parameters())).item()

    first = fp16.train_step()
    assert scale_relative_error(first.grad_norm, expected) <= REF_OPT_FP16
    assert first.scale == 65536.0
    fp16.train()
    assert fp16.skipped_steps == 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits, _ = fp16.model(x.cuda(), mode="opt")
    assert torch.equal(logits.argmax(-1).cpu(), y)


@pytest.mark.gpu
def test_fp16_skip_is_counted_and_the_scale_backs_off():
    trainer = make_trainer(steps=10, device="cuda", **SCALER)
    trainer.train(until=3)
    before = snapshot(trainer.model)
    handle = poison_next_gradient(trainer, float("inf"))
    record = trainer.train_step()
    handle.remove()
    assert record.skipped is True and record.scale == 32768.0  # backed off by half
    assert trainer.steps_taken == 3 and trainer.skipped_steps == 1
    for name, p in trainer.model.named_parameters():
        assert torch.equal(p, before[name]), name
    assert trainer.train_step().skipped is False
    assert trainer.summary()["final_scale"] == 32768.0


@pytest.mark.gpu
def test_fp16_needs_the_scaler_settings():
    with pytest.raises(ValueError, match="init_scale|growth_interval"):
        make_trainer(steps=2, device="cuda")  # both unset in the frozen recipe


@pytest.mark.gpu
def test_scaler_growth_follows_the_recipe():
    trainer = make_trainer(
        steps=12, device="cuda",
        **{"recipe.grad_scaler.init_scale": 1024.0, "recipe.grad_scaler.growth_interval": 5},
    )
    trainer.train()
    assert [r.scale for r in trainer.records[:5]] == [1024.0] * 4 + [2048.0]
    assert trainer.records[9].scale == 4096.0


@pytest.mark.gpu
def test_an_unusable_backend_fails_inside_the_loop():
    """Only the expected fused backend may run; a fallback is an error at that step."""
    fine = make_trainer(steps=3, device="cuda", expected_backend="math", **SCALER)
    fine.train()
    assert fine.summary()["expected_backend"] == "math"
    with pytest.raises(ValueError):
        make_trainer(steps=3, device="cuda", expected_backend="fastest", **SCALER)
