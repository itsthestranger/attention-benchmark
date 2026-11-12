"""The trainer enforces the expected fused kernel INSIDE the loop (plan §2.3 item 1, §5 "Backend selection").

Added after the handover audit (finding F4): the existing test of this could not fail,
because it never observed whether the restriction was active during a forward pass.

How it is observed: `torch.nn.attention.sdpa_kernel` switches the global
`torch.backends.cuda.*_sdp_enabled` flags for the duration of its block. A forward
hook on an attention module reads those flags while the trainer's forward runs.

cpu: the flags are plain process state, so this is observable without a GPU. The
trainer refuses `expected_backend` off CUDA at construction, so the cpu test sets
the attribute afterwards; that bypass is the test's, not the product's.
gpu / t4: the real path, constructed normally, for all seven variants.
"""

from __future__ import annotations

import pytest
import torch

from scripts.timing_pass import DEFAULT_VARIANTS, SYNTHETIC_DIMS, SYNTHETIC_VOCAB
from src import config
from src.data.loader import TrainStream
from src.model import build_model
from src.training.trainer import Trainer
from src.utils.reproducibility import synthetic_split

STEPS = 3
ALL_VARIANTS = DEFAULT_VARIANTS  # the seven variants of the study


def _flags() -> dict[str, bool]:
    cuda = torch.backends.cuda
    return {"flash": cuda.flash_sdp_enabled(), "efficient": cuda.mem_efficient_sdp_enabled(), "math": cuda.math_sdp_enabled()}


def _trainer(variant: str, device: str, expected_backend=None, mode: str = "opt") -> Trainer:
    resolved = config.resolve(
        variant, init_seed=1, peak_lr=1e-3, steps=STEPS, branches=[], overrides=dict(SYNTHETIC_DIMS)
    )
    batch = resolved["recipe"]["batch"]
    context = resolved["model"]["context_length"]
    stream = TrainStream(
        synthetic_split(STEPS * batch["sequences_per_step"] * context + 1, SYNTHETIC_VOCAB, seed=0),
        context_length=context, sequences_per_step=batch["sequences_per_step"], data_seed=1,
    )
    return Trainer(build_model(resolved), resolved, stream, device=device, mode=mode, expected_backend=expected_backend)


def _flags_seen_during_a_step(trainer: Trainer) -> list[dict[str, bool]]:
    seen: list[dict[str, bool]] = []
    handles = [layer.attn.register_forward_pre_hook(lambda module, args: seen.append(_flags())) for layer in trainer.model.layers]
    try:
        trainer.train_step()
    finally:
        for handle in handles:
            handle.remove()
    return seen


def test_without_an_expected_backend_nothing_is_restricted():
    before = _flags()
    seen = _flags_seen_during_a_step(_trainer("mha", "cpu"))
    assert seen and all(flags == before for flags in seen)


@pytest.mark.parametrize("backend", ["efficient", "math", "flash"])
def test_the_expected_backend_alone_is_enabled_during_every_forward(backend):
    before = _flags()
    trainer = _trainer("mha", "cpu")
    trainer.expected_backend = backend  # the constructor refuses this off CUDA; see the module docstring
    seen = _flags_seen_during_a_step(trainer)
    micro_batches = trainer.sequences_per_step // trainer.micro_batch
    assert len(seen) == micro_batches * len(trainer.model.layers)
    for flags in seen:
        assert flags == {name: name == backend for name in ("flash", "efficient", "math")}
    assert _flags() == before, "the restriction must end with the forward pass"


def test_the_restriction_is_applied_again_on_every_step():
    trainer = _trainer("gqa4", "cpu")
    trainer.expected_backend = "efficient"
    for _ in range(STEPS):
        seen = _flags_seen_during_a_step(trainer)
        assert seen and all(flags == {"flash": False, "efficient": True, "math": False} for flags in seen)


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_gpu_every_variant_trains_a_step_under_the_enforced_backend(variant):
    trainer = _trainer(variant, "cuda", expected_backend="efficient")
    seen = _flags_seen_during_a_step(trainer)
    assert seen and all(flags == {"flash": False, "efficient": True, "math": False} for flags in seen)
    assert trainer.steps_taken + trainer.skipped_steps == 1


@pytest.mark.t4
@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_t4_every_variant_trains_a_step_under_the_efficient_backend(variant):
    # Plan §2.3: on sm75 the memory-efficient kernel is the fused backend; a fallback must fail the step.
    trainer = _trainer(variant, "cuda", expected_backend="efficient")
    record = trainer.train_step()
    assert record.step == 0 and trainer.steps_taken + trainer.skipped_steps == 1
