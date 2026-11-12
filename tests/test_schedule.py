"""T-14: the WSD schedule and the optimizer's parameter groups (plan §7.1, §7.2).

Written by the orchestrator before the implementation. These tests define done;
the worker implementing T-14 may not change this file.

The contract
------------
`src/training/schedule.py`:

    decay_start(total_steps, decay_fraction) -> int
        round(total_steps * (1 - decay_fraction)): the step at which decay begins.

    wsd_lr(step, *, peak_lr, total_steps, warmup_steps, decay_fraction,
           decay_shape, final_lr=0.0) -> float
        The learning rate of optimizer step `step`, counted from 0: `step` is
        the number of updates already taken. A run of S steps uses
        wsd_lr(0) .. wsd_lr(S - 1); wsd_lr(S) is defined and is the end point.

            warm  = min(step / warmup_steps, 1)
            d     = decay_start(total_steps, decay_fraction)
            prog  = 0 if step <= d else (step - d) / (total_steps - d)
            decay = 1 - sqrt(prog)                       # decay_shape "one_minus_sqrt"
            lr    = final_lr + (peak_lr - final_lr) * warm * decay

        All arithmetic in Python floats. ValueError for a step outside
        [0, total_steps], for total_steps < 1, warmup_steps < 1, a decay
        fraction outside (0, 1), or an unknown decay shape.

    lr_at(step, resolved) -> float
        wsd_lr with peak_lr = resolved["run"]["peak_lr"], total_steps =
        resolved["run"]["steps"] and the rest from resolved["recipe"]["schedule"].

    set_lr(optimizer, lr) -> None
        Sets "lr" on every parameter group.

The schedule is a pure function of the step. Nothing keeps a position, so a
resume or a branch is on schedule by construction (plan §5, §13.9).

`src/training/optimizer.py`:

    param_groups(model, weight_decay) -> list[dict]
        Exactly two groups, in this order:
          {"params": [...], "weight_decay": weight_decay, "name": "decay"}
          {"params": [...], "weight_decay": 0.0, "name": "no_decay"}
        "decay" holds every parameter with 2 or more dimensions except the tied
        embedding `model.embed.weight`. "no_decay" holds everything else: norm
        gains and the embedding. Parameter order within a group follows
        model.named_parameters().

    build_optimizer(model, resolved) -> torch.optim.AdamW
        AdamW over param_groups(model, recipe.optimizer.weight_decay), with
        betas and eps from the recipe and lr = resolved["run"]["peak_lr"].
        ValueError if recipe.optimizer.name is not "adamw".
"""

import math

import pytest
import torch
import torch.nn as nn

from src import config
from src.attention.base import AttnBase, AttnConfig, register, unregister
from src.model import Transformer
from src.training.optimizer import build_optimizer, param_groups
from src.training.schedule import decay_start, lr_at, set_lr, wsd_lr
from tests.tolerances import FLOAT64_IDENTITY, REF_OPT_FP32, scale_relative_error

PEAK = 1.2e-3
WSD = dict(warmup_steps=100, decay_fraction=0.2, decay_shape="one_minus_sqrt")  # plan §7.2


def lr(step, total_steps, peak=PEAK, **overrides):
    return wsd_lr(step, peak_lr=peak, total_steps=total_steps, **{**WSD, **overrides})


def close(a, b) -> bool:
    return scale_relative_error(float(a), float(b)) <= FLOAT64_IDENTITY


# --- Where decay starts (plan §7.2) -----------------------------------------------


@pytest.mark.parametrize(
    "total_steps, start",
    [(763, 610), (1526, 1221), (3052, 2442), (4578, 3662)],
)
def test_decay_starts_at_80_percent_of_the_budget(total_steps, start):
    assert decay_start(total_steps, 0.2) == start
    assert isinstance(decay_start(total_steps, 0.2), int)


def test_branch_costs_match_the_plan():
    """A 100M branch adds 305 steps (+6.7%), a 200M branch 610 (+13%)."""
    assert 1526 - decay_start(1526, 0.2) == 305
    assert 3052 - decay_start(3052, 0.2) == 610
    assert 4578 + 305 == 4883


# --- The named steps (TASKS.md T-14) ------------------------------------------------


def test_lr_on_the_300m_run():
    S = 4578
    assert lr(0, S) == 0.0
    assert lr(100, S) == PEAK
    assert lr(1221, S) == PEAK  # the trunk is still stable where the 100M branch leaves
    assert lr(1526, S) == PEAK
    assert lr(3662, S) == PEAK  # decay begins here
    assert lr(4578, S) == 0.0
    assert 0.0 < lr(4577, S) < lr(3663, S) < PEAK


def test_lr_on_the_100m_branch():
    S = 1526
    assert lr(0, S) == 0.0
    assert lr(100, S) == PEAK
    assert lr(1220, S) == PEAK
    assert lr(1221, S) == PEAK  # decay begins here
    assert lr(1526, S) == 0.0
    assert 0.0 < lr(1525, S) < lr(1222, S) < PEAK


@pytest.mark.parametrize("total_steps", [763, 1526, 3052])
def test_a_branch_follows_the_trunk_until_its_decay_starts(total_steps):
    """Bit-identical, so a branch from a trunk checkpoint is on its own schedule."""
    for step in range(decay_start(total_steps, 0.2) + 1):
        assert lr(step, total_steps) == lr(step, 4578), step


# --- Shape ---------------------------------------------------------------------------


def test_warmup_is_linear_over_100_steps():
    for step in range(101):
        assert close(lr(step, 4578), PEAK * step / 100) or (step == 0 and lr(step, 4578) == 0.0)
    assert close(lr(1, 4578), PEAK / 100)
    assert close(lr(50, 4578), PEAK / 2)
    assert lr(101, 4578) == PEAK


@pytest.mark.parametrize("total_steps", [763, 1526, 3052, 4578])
def test_every_step_matches_the_formula(total_steps):
    start = round(total_steps * 0.8)
    for step in range(total_steps + 1):
        warm = min(step / 100, 1.0)
        prog = 0.0 if step <= start else (step - start) / (total_steps - start)
        expected = PEAK * warm * (1.0 - math.sqrt(prog))
        got = lr(step, total_steps)
        assert got == expected == 0.0 or close(got, expected), step


def test_decay_has_the_one_minus_sqrt_shape():
    S = 2000  # decay over steps 1600 .. 2000, so quarters land on whole steps
    assert decay_start(S, 0.2) == 1600
    assert close(lr(1700, S), PEAK * 0.5)  # progress 1/4 -> 1 - 1/2
    assert close(lr(1825, S), PEAK * 0.25)  # progress 9/16 -> 1 - 3/4
    assert close(lr(1616, S), PEAK * 0.8)  # progress 1/25 -> 1 - 1/5
    values = [lr(step, S) for step in range(1600, 2001)]
    assert all(a > b for a, b in zip(values, values[1:]))  # strictly decreasing
    # Below the straight line everywhere inside the decay: it drops fast first.
    for step in range(1601, 2000):
        assert lr(step, S) < PEAK * (1 - (step - 1600) / 400)
    # Convex: the drops get smaller.
    drops = [a - b for a, b in zip(values, values[1:])]
    assert all(a > b for a, b in zip(drops, drops[1:]))


def test_lr_never_exceeds_the_peak_and_is_never_negative():
    for total_steps in (76, 763, 4578):
        values = [lr(step, total_steps) for step in range(total_steps + 1)]
        assert min(values) >= 0.0 and max(values) <= PEAK


def test_lr_is_proportional_to_the_peak():
    for step in (0, 37, 100, 2000, 3662, 4000, 4577):
        assert close(lr(step, 4578, peak=2.4e-3), 2 * lr(step, 4578, peak=1.2e-3)) or lr(step, 4578) == 0.0


def test_final_lr_is_the_end_point():
    S = 2000
    assert wsd_lr(S, peak_lr=PEAK, total_steps=S, final_lr=1e-5, **WSD) == 1e-5
    assert wsd_lr(0, peak_lr=PEAK, total_steps=S, final_lr=1e-5, **WSD) == 1e-5
    assert wsd_lr(500, peak_lr=PEAK, total_steps=S, final_lr=1e-5, **WSD) == PEAK
    assert close(
        wsd_lr(1700, peak_lr=PEAK, total_steps=S, final_lr=1e-5, **WSD), 1e-5 + (PEAK - 1e-5) * 0.5
    )


def test_a_run_shorter_than_the_warmup_is_still_defined():
    """A 5M smoke run is 76 steps: warmup and decay overlap and multiply."""
    S = 76
    start = decay_start(S, 0.2)
    assert start == 61
    assert lr(0, S) == 0.0 and lr(S, S) == 0.0
    assert close(lr(61, S), PEAK * 0.61)
    assert close(lr(70, S), PEAK * 0.70 * (1 - math.sqrt(9 / 15)))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(step=-1),
        dict(step=4579),
        dict(total_steps=0, step=0),
        dict(warmup_steps=0),
        dict(decay_fraction=0.0),
        dict(decay_fraction=1.0),
        dict(decay_shape="cosine"),
        dict(decay_shape="linear"),
    ],
)
def test_invalid_arguments_raise(kwargs):
    args = dict(step=10, peak_lr=PEAK, total_steps=4578, **WSD)
    args.update(kwargs)
    with pytest.raises(ValueError):
        wsd_lr(**args)


def test_schedule_is_a_pure_function_of_the_step():
    forward = [lr(step, 1526) for step in range(1527)]
    backward = [lr(step, 1526) for step in reversed(range(1527))]
    assert forward == backward[::-1]


# --- From the resolved config -----------------------------------------------------------


def test_lr_at_reads_the_recipe_and_the_run():
    main = config.resolve("mha", init_seed=1, peak_lr=PEAK)
    assert main["run"]["steps"] == 4578
    for step in (0, 50, 100, 1221, 1526, 3662, 4000, 4578):
        assert lr_at(step, main) == lr(step, 4578), step
    branch = config.resolve("mha", init_seed=1, peak_lr=PEAK, budget="100M", branches=[])
    for step in (0, 100, 1221, 1400, 1526):
        assert lr_at(step, branch) == lr(step, 1526), step
    other_peak = config.resolve("mha", init_seed=1, peak_lr=2 * PEAK)
    assert lr_at(100, other_peak) == 2 * PEAK


def test_recipe_schedule_is_the_plans():
    schedule = config.load_recipe()["schedule"]
    assert schedule["warmup_steps"] == 100
    assert schedule["decay_fraction"] == 0.2
    assert schedule["decay_shape"] == "one_minus_sqrt"
    assert schedule["final_lr"] == 0.0


# --- Parameter groups (plan §7.1) -----------------------------------------------------------


class ToyWithLatentNorm(AttnBase):
    """Two matrices and a 1-D gain, like MLA's latent norm."""

    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.latent_norm = nn.Module()
        self.latent_norm.gain = nn.Parameter(torch.ones(cfg.d_model))

    @property
    def rope_dim(self) -> int:
        return self.cfg.head_dim

    def init_cache(self):
        return None

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        return self.o(self.q(x) * self.latent_norm.gain), cache


@pytest.fixture(scope="module", autouse=True)
def _toy_registered():
    register("toy_latent_norm")(ToyWithLatentNorm)
    yield
    unregister("toy_latent_norm")


def tiny_cfg(peak_lr: float = PEAK) -> dict:
    return config.resolve(
        "mha",
        init_seed=1,
        peak_lr=peak_lr,
        overrides={
            "model.attention": {"type": "toy_latent_norm"},
            "model.d_model": 16,
            "model.n_heads": 2,
            "model.head_dim": 8,
            "model.n_layers": 2,
            "model.vocab_size": 32,
            "model.d_ff": 24,
            "model.context_length": 8,
        },
    )


def tiny_model() -> Transformer:
    model = Transformer(tiny_cfg())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g))  # no zeros, so decay is visible
    return model


def names_in(group, model) -> list[str]:
    by_id = {id(p): n for n, p in model.named_parameters()}
    return [by_id[id(p)] for p in group["params"]]


def test_two_groups_cover_every_parameter_exactly_once():
    model = tiny_model()
    groups = param_groups(model, 0.1)
    assert [g["name"] for g in groups] == ["decay", "no_decay"]
    assert [g["weight_decay"] for g in groups] == [0.1, 0.0]
    decay, no_decay = names_in(groups[0], model), names_in(groups[1], model)
    assert sorted(decay + no_decay) == sorted(n for n, _ in model.named_parameters())
    assert not set(decay) & set(no_decay)


def test_weight_decay_covers_2d_matrices_but_not_the_tied_embedding():
    model = tiny_model()
    decay, no_decay = (names_in(g, model) for g in param_groups(model, 0.1))
    expected_decay = [
        n for n, p in model.named_parameters() if p.dim() >= 2 and n != "embed.weight"
    ]
    assert decay == expected_decay
    assert "embed.weight" in no_decay
    assert all(n == "embed.weight" or n.endswith(".gain") for n in no_decay)
    for expected in (
        "layers.0.mlp.gate.weight",
        "layers.0.mlp.up.weight",
        "layers.1.mlp.down.weight",
        "layers.0.attn.q.weight",
        "layers.1.attn.o.weight",
    ):
        assert expected in decay
    for expected in ("norm_f.gain", "layers.0.norm1.gain", "layers.1.norm2.gain", "layers.0.attn.latent_norm.gain"):
        assert expected in no_decay


def test_build_optimizer_uses_the_recipe():
    cfg = tiny_cfg()
    model = tiny_model()
    opt = build_optimizer(model, cfg)
    assert type(opt) is torch.optim.AdamW
    assert [g["name"] for g in opt.param_groups] == ["decay", "no_decay"]
    for group in opt.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8
        assert group["lr"] == PEAK
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert opt.param_groups[0]["amsgrad"] is False


def test_build_optimizer_refuses_another_optimizer():
    cfg = tiny_cfg()
    cfg["recipe"]["optimizer"]["name"] = "sgd"
    with pytest.raises(ValueError):
        build_optimizer(tiny_model(), cfg)


def test_one_step_with_zero_gradients_shrinks_only_the_decayed_matrices():
    """Decoupled weight decay: p <- p * (1 - lr * wd), and nothing else moves."""
    cfg = tiny_cfg(peak_lr=0.01)
    model = tiny_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer(model, cfg)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for n, p in model.named_parameters():
        if p.dim() >= 2 and n != "embed.weight":
            assert not torch.equal(p, before[n]), n
            assert scale_relative_error(p.detach(), before[n] * (1 - 0.01 * 0.1)) <= REF_OPT_FP32, n
        else:
            assert torch.equal(p, before[n]), n


def test_set_lr_sets_every_group():
    model = tiny_model()
    opt = build_optimizer(model, tiny_cfg())
    set_lr(opt, 3.3e-4)
    assert [g["lr"] for g in opt.param_groups] == [3.3e-4, 3.3e-4]
    # The step size follows: zero gradients, so only the decay term acts.
    before = model.layers[0].mlp.gate.weight.detach().clone()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    after = model.layers[0].mlp.gate.weight.detach()
    assert scale_relative_error(after, before * (1 - 3.3e-4 * 0.1)) <= REF_OPT_FP32
