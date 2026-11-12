"""T-03: per-parameter seeded initialisation and the init rule (plan §3.1, §4.5).

Written by the orchestrator before the implementation. These tests define done;
the worker implementing T-03 may not change this file.

The contract
------------
`src/utils/seeding.py`:

    param_seed(name, run_seed) -> int
        zlib.crc32(name.encode()) ^ run_seed, with `name` the full parameter
        name in the model, e.g. "layers.3.mlp.gate.weight".

    init_model(model, run_seed) -> None
        Overwrites EVERY parameter of a `Transformer`, in place:

        - a 1-D parameter whose name ends in ".gain" becomes ones;
        - a 2-D parameter becomes z * std, where
              g = torch.Generator(device="cpu"); g.manual_seed(param_seed(name, run_seed))
              z = torch.empty(shape, dtype=torch.float32).normal_(generator=g)
              std = model.init_std * multiplier * residual
          `multiplier` comes from the attention module's `init_std_multipliers()`
          (1 if the parameter is not listed), and `residual` is
          1 / sqrt(2 * model.n_layers) for residual output projections and 1 otherwise;
        - anything else is an error (ValueError): there is no rule for it.

        Residual output projections are every layer's `mlp.down.weight` and the
        attention parameters the module lists in `residual_output_params()`.
        The embedding (tied with the output head) is not one.

        It never reads or changes the global RNG, raises ValueError if two
        parameter names of the model share a crc32, and raises ValueError if an
        attention module lists a parameter it does not have. It calls
        `zlib.crc32` through the module (`import zlib`), not a copied name.

`src/attention/base.py`, two additions to `AttnBase`:

    residual_output_params(self) -> set[str]
        Local names (relative to the attention module, e.g. "o.weight") of the
        weights that write to the residual stream. The base class raises
        NotImplementedError, so no attention can forget to declare them.

    init_std_multipliers(self) -> dict[str, float]
        Local name -> multiplier on the base std. The base class returns {}.
        MLA uses it for its up-projections (plan §3.3).

`src/model.py`:

    Transformer stores `init_std`, taken from resolved["init"]["std"].

    build_model(resolved) -> Transformer
        Transformer(resolved) followed by init_model(model, resolved["run"]["init_seed"]).
        It raises ValueError unless the resolved `init` section says what the
        rule implements: distribution "normal", residual_output_scale
        "inverse_sqrt_two_layers", norm_gain 1.0, seeding "per_parameter".
        This is the only supported way to get a model for training or evaluation.
"""

import math
import zlib

import pytest
import torch
import torch.nn as nn

from src import config
from src.attention.base import AttnBase, AttnConfig, register, unregister
from src.model import Transformer, build_model
from src.utils import seeding
from src.utils.seeding import init_model, param_seed
from tests.tolerances import REF_OPT_FP32, scale_relative_error

BASE_STD = 0.02  # plan §3.1


# --- Toy attentions, registered for this module only ---------------------------


class _Toy(AttnBase):
    @property
    def rope_dim(self) -> int:
        return self.cfg.head_dim

    def init_cache(self):
        return None

    def forward(self, x, freqs, cache=None, *, mode="ref"):
        return torch.zeros_like(x), cache


class ToyDense(_Toy):
    """Two full-width matrices: `q` and the residual output `o`."""

    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        d = cfg.d_model
        self.q = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def residual_output_params(self):
        return {"o.weight"}


class ToyFactored(_Toy):
    """`q` and `o` as in ToyDense, plus a factor pair through a normalised latent."""

    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        d = cfg.d_model
        self.latent = cfg.options["latent"]
        self.q = nn.Linear(d, d, bias=False)
        self.down = nn.Linear(d, self.latent, bias=False)
        self.latent_gain = nn.Module()
        self.latent_gain.gain = nn.Parameter(torch.zeros(self.latent))
        self.up = nn.Linear(self.latent, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def residual_output_params(self):
        return {"o.weight"}

    def init_std_multipliers(self):
        return {"up.weight": math.sqrt(self.cfg.d_model / self.latent)}


class ToyUndeclared(_Toy):
    """Has a weight but does not say which parameters are residual outputs."""

    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)


class ToyWrongName(ToyDense):
    def residual_output_params(self):
        return {"out.weight"}  # no such parameter


class ToyWrongMultiplier(ToyDense):
    def init_std_multipliers(self):
        return {"v.weight": 2.0}  # no such parameter


class ToyBias(_Toy):
    """A 1-D parameter that is not a norm gain: there is no rule for it."""

    def __init__(self, cfg: AttnConfig):
        super().__init__(cfg)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def residual_output_params(self):
        return {"o.weight"}


TOYS = {
    "toy_dense": ToyDense,
    "toy_factored": ToyFactored,
    "toy_undeclared": ToyUndeclared,
    "toy_wrong_name": ToyWrongName,
    "toy_wrong_multiplier": ToyWrongMultiplier,
    "toy_bias": ToyBias,
}


@pytest.fixture(scope="module", autouse=True)
def _toys_registered():
    for name, cls in TOYS.items():
        register(name)(cls)
    yield
    for name in TOYS:
        unregister(name)


# --- Helpers -------------------------------------------------------------------

D, HEADS, HEAD_DIM, VOCAB = 32, 4, 8, 64


def resolved_cfg(attention: dict, *, n_layers: int = 3, d_ff: int = 48, init_seed: int = 1) -> dict:
    return config.resolve(
        "mha",
        init_seed=init_seed,
        peak_lr=1e-3,
        overrides={
            "model.attention": attention,
            "model.d_model": D,
            "model.n_heads": HEADS,
            "model.head_dim": HEAD_DIM,
            "model.n_layers": n_layers,
            "model.vocab_size": VOCAB,
            "model.d_ff": d_ff,
            "model.context_length": 16,
        },
    )


def model_for(attention: dict, **kwargs) -> Transformer:
    return build_model(resolved_cfg(attention, **kwargs))


def params(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def standard_draw(name: str, run_seed: int, shape) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(zlib.crc32(name.encode()) ^ run_seed)
    return torch.empty(tuple(shape), dtype=torch.float32).normal_(generator=g)


def expected_std(name: str, n_layers: int, multipliers: dict[str, float]) -> float:
    """The rule of plan §3.1 and §3.3, written out independently of the implementation."""
    residual = 1.0 / math.sqrt(2 * n_layers)
    local = name.split(".attn.", 1)[1] if ".attn." in name else None
    std = BASE_STD * multipliers.get(local, 1.0)
    if name.endswith(".mlp.down.weight") or local == "o.weight":
        std *= residual
    return std


def assert_follows_rule(model, run_seed: int, n_layers: int, multipliers: dict[str, float]):
    for name, p in model.named_parameters():
        if name.endswith(".gain"):
            assert torch.equal(p, torch.ones_like(p)), name
            continue
        assert p.dim() == 2, name
        expected = standard_draw(name, run_seed, p.shape) * expected_std(name, n_layers, multipliers)
        assert p.dtype == torch.float32, name
        assert scale_relative_error(p.detach(), expected) <= REF_OPT_FP32, name


# --- The seed -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["embed.weight", "layers.3.mlp.gate.weight", "norm_f.gain"])
@pytest.mark.parametrize("run_seed", [0, 1, 5, 15, 2**31])
def test_param_seed_is_crc32_xor_run_seed(name, run_seed):
    assert param_seed(name, run_seed) == zlib.crc32(name.encode()) ^ run_seed


# --- The rule -------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 3, 8])
@pytest.mark.parametrize("run_seed", [1, 7])
def test_dense_model_follows_the_rule(n_layers, run_seed):
    model = model_for({"type": "toy_dense"}, n_layers=n_layers, init_seed=run_seed)
    assert_follows_rule(model, run_seed, n_layers, {})


@pytest.mark.parametrize("latent", [8, 16])
def test_factored_model_follows_the_rule_with_its_multiplier(latent):
    model = model_for({"type": "toy_factored", "latent": latent}, n_layers=2, init_seed=3)
    assert_follows_rule(model, 3, 2, {"up.weight": math.sqrt(D / latent)})
    p = dict(model.named_parameters())
    # Down-projections keep the base std; the latent norm's gain is a norm gain.
    assert torch.equal(p["layers.0.attn.latent_gain.gain"], torch.ones(latent))


def test_std_values_of_the_rule():
    """The numbers of plan §3.1 and §3.3, at the plan's sizes."""
    assert BASE_STD == 0.02
    assert expected_std("layers.0.mlp.down.weight", 8, {}) == 0.02 / 4.0
    assert expected_std("layers.0.attn.o.weight", 8, {}) == 0.02 / 4.0
    assert expected_std("layers.0.mlp.gate.weight", 8, {}) == 0.02
    assert expected_std("embed.weight", 8, {}) == 0.02
    # MLA up-projections: 0.02 * sqrt(d / d_c) is 0.02 * sqrt(2) at d_c 192 and 0.04 at d_c 96.
    assert expected_std("layers.0.attn.up.weight", 8, {"up.weight": math.sqrt(384 / 96)}) == 0.04
    assert expected_std("layers.0.attn.up.weight", 8, {"up.weight": math.sqrt(384 / 192)}) == 0.02 * math.sqrt(2)


def test_every_parameter_is_overwritten():
    model = Transformer(resolved_cfg({"type": "toy_factored", "latent": 8}))
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))
    init_model(model, 4)
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name
    assert_follows_rule(model, 4, 3, {"up.weight": math.sqrt(D / 8)})


def test_norm_gains_are_exactly_one_even_if_they_were_not():
    model = Transformer(resolved_cfg({"type": "toy_dense"}))
    with torch.no_grad():
        model.norm_f.gain.fill_(3.0)
        model.layers[1].norm1.gain.fill_(-2.0)
    init_model(model, 1)
    for name, p in model.named_parameters():
        if name.endswith(".gain"):
            assert torch.equal(p, torch.ones_like(p)), name


def test_embedding_is_not_scaled_as_a_residual_output():
    one = params(model_for({"type": "toy_dense"}, n_layers=1))
    eight = params(model_for({"type": "toy_dense"}, n_layers=8))
    assert torch.equal(one["embed.weight"], eight["embed.weight"])
    assert torch.equal(one["layers.0.mlp.gate.weight"], eight["layers.0.mlp.gate.weight"])
    assert torch.equal(one["layers.0.attn.q.weight"], eight["layers.0.attn.q.weight"])


def test_depth_scaling_of_residual_outputs():
    """Same name, same shape, same seed: only the 1/sqrt(2 * layers) factor differs."""
    shallow = params(model_for({"type": "toy_dense"}, n_layers=2))
    deep = params(model_for({"type": "toy_dense"}, n_layers=8))
    for name in ("layers.0.mlp.down.weight", "layers.1.attn.o.weight"):
        assert not torch.equal(shallow[name], deep[name])
        # sqrt(2 * 2) = 2 and sqrt(2 * 8) = 4 are exact in floating point.
        assert scale_relative_error(shallow[name] * 2.0, deep[name] * 4.0) <= REF_OPT_FP32, name


# --- "Same init" across variants (plan §4.5) --------------------------------------


def test_same_name_and_shape_is_bit_identical_across_variants():
    dense = params(model_for({"type": "toy_dense"}, d_ff=48))
    factored = params(model_for({"type": "toy_factored", "latent": 8}, d_ff=64))
    shared = [n for n in dense if n in factored and dense[n].shape == factored[n].shape]
    # The embedding, the norm gains and the unchanged attention projections match;
    # the MLP does not, because d_ff differs (plan §4.5).
    assert "embed.weight" in shared
    assert "layers.2.attn.q.weight" in shared and "layers.2.attn.o.weight" in shared
    assert not any(".mlp." in n for n in shared)
    for name in shared:
        assert torch.equal(dense[name], factored[name]), name


def test_same_d_ff_makes_the_mlp_identical_too():
    dense = params(model_for({"type": "toy_dense"}, d_ff=48))
    factored = params(model_for({"type": "toy_factored", "latent": 8}, d_ff=48))
    for i in range(3):
        for w in ("gate", "up", "down"):
            name = f"layers.{i}.mlp.{w}.weight"
            assert torch.equal(dense[name], factored[name]), name


def test_values_do_not_depend_on_the_other_parameters():
    """A global RNG would shift every later tensor when one earlier shape changes."""
    small = params(model_for({"type": "toy_factored", "latent": 8}))
    large = params(model_for({"type": "toy_factored", "latent": 16}))
    assert small["layers.0.attn.down.weight"].shape != large["layers.0.attn.down.weight"].shape
    for name in ("layers.0.attn.o.weight", "layers.1.attn.q.weight", "layers.2.mlp.down.weight", "embed.weight"):
        assert torch.equal(small[name], large[name]), name


def test_layers_do_not_share_values():
    p = params(model_for({"type": "toy_dense"}))
    assert not torch.equal(p["layers.0.attn.q.weight"], p["layers.1.attn.q.weight"])
    assert not torch.equal(p["layers.0.mlp.gate.weight"], p["layers.0.mlp.up.weight"])
    assert not torch.equal(p["layers.0.mlp.gate.weight"], p["layers.2.mlp.gate.weight"])


# --- Seeds ----------------------------------------------------------------------


def test_different_seeds_differ_in_every_random_parameter():
    a = params(model_for({"type": "toy_dense"}, init_seed=1))
    b = params(model_for({"type": "toy_dense"}, init_seed=2))
    for name in a:
        if name.endswith(".gain"):
            continue
        assert not torch.equal(a[name], b[name]), name


def test_same_seed_is_bit_identical_whatever_the_global_rng_did():
    torch.manual_seed(123)
    a = params(model_for({"type": "toy_factored", "latent": 8}, init_seed=5))
    torch.manual_seed(999)
    torch.randn(1000)
    nn.Linear(7, 13)
    b = params(model_for({"type": "toy_factored", "latent": 8}, init_seed=5))
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_init_model_leaves_the_global_rng_alone():
    model = Transformer(resolved_cfg({"type": "toy_dense"}))
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    init_model(model, 9)
    assert torch.equal(torch.get_rng_state(), before)


def test_init_model_is_idempotent_and_reseedable():
    model = Transformer(resolved_cfg({"type": "toy_dense"}))
    init_model(model, 1)
    first = params(model)
    init_model(model, 2)
    assert not torch.equal(params(model)["embed.weight"], first["embed.weight"])
    init_model(model, 1)
    again = params(model)
    for name in first:
        assert torch.equal(first[name], again[name]), name


# --- build_model ------------------------------------------------------------------


def test_build_model_uses_the_run_seed_of_the_resolved_config():
    for seed in (1, 4):
        cfg = resolved_cfg({"type": "toy_dense"}, init_seed=seed)
        built = params(build_model(cfg))
        by_hand = Transformer(cfg)
        init_model(by_hand, seed)
        for name, p in by_hand.named_parameters():
            assert torch.equal(built[name], p), name


def test_build_model_takes_base_std_and_depth_from_the_config():
    cfg = resolved_cfg({"type": "toy_dense"}, n_layers=2)
    base = params(build_model(cfg))
    cfg["init"]["std"] = 0.04
    doubled = params(build_model(cfg))
    for name in ("embed.weight", "layers.0.mlp.down.weight", "layers.1.attn.o.weight"):
        assert scale_relative_error(doubled[name], base[name] * 2.0) <= REF_OPT_FP32, name
    assert torch.equal(doubled["norm_f.gain"], base["norm_f.gain"])


@pytest.mark.parametrize(
    "key, value",
    [
        ("distribution", "uniform"),
        ("residual_output_scale", "none"),
        ("norm_gain", 0.5),
        ("seeding", "global"),
    ],
)
def test_build_model_refuses_an_init_rule_it_does_not_implement(key, value):
    cfg = resolved_cfg({"type": "toy_dense"})
    cfg["init"][key] = value
    with pytest.raises(ValueError, match=key):
        build_model(cfg)


def test_transformer_stores_the_base_std():
    cfg = resolved_cfg({"type": "toy_dense"})
    assert Transformer(cfg).init_std == BASE_STD
    cfg["init"]["std"] = 0.01
    assert Transformer(cfg).init_std == 0.01


def test_full_size_backbone_follows_the_rule():
    """At the plan's sizes: 8 layers, so residual outputs get std 0.02 / 4."""
    cfg = config.resolve(
        "mha", init_seed=2, peak_lr=1e-3, overrides={"model.attention": {"type": "toy_dense"}}
    )
    model = build_model(cfg)
    for name in ("embed.weight", "layers.7.mlp.gate.weight", "layers.7.mlp.down.weight", "layers.0.attn.o.weight"):
        p = dict(model.named_parameters())[name]
        expected = standard_draw(name, 2, p.shape) * expected_std(name, 8, {})
        assert scale_relative_error(p.detach(), expected) <= REF_OPT_FP32, name
    assert torch.equal(model.norm_f.gain, torch.ones(384))


# --- Refusals ---------------------------------------------------------------------


def test_attention_must_declare_its_residual_outputs():
    undeclared = Transformer(resolved_cfg({"type": "toy_undeclared"}))
    with pytest.raises(NotImplementedError):
        undeclared.layers[0].attn.residual_output_params()
    with pytest.raises(NotImplementedError):
        build_model(resolved_cfg({"type": "toy_undeclared"}))


def test_base_class_has_no_multipliers():
    model = Transformer(resolved_cfg({"type": "toy_dense"}))
    assert model.layers[0].attn.init_std_multipliers() == {}
    assert AttnBase.init_std_multipliers(model.layers[0].attn) == {}


@pytest.mark.parametrize("attention", ["toy_wrong_name", "toy_wrong_multiplier"])
def test_declaring_a_parameter_that_does_not_exist_is_an_error(attention):
    with pytest.raises(ValueError):
        build_model(resolved_cfg({"type": attention}))


def test_a_parameter_without_a_rule_is_an_error():
    with pytest.raises(ValueError, match="bias"):
        build_model(resolved_cfg({"type": "toy_bias"}))


def test_a_crc32_collision_between_two_names_is_an_error(monkeypatch):
    model = Transformer(resolved_cfg({"type": "toy_dense"}))
    monkeypatch.setattr(seeding.zlib, "crc32", lambda data: 7)
    with pytest.raises(ValueError, match="crc32"):
        init_model(model, 1)


def test_backbone_parameter_names_have_distinct_seeds_at_full_size():
    cfg = config.resolve(
        "mha", ladder="mha_L10", init_seed=1, peak_lr=1e-3,
        overrides={"model.attention": {"type": "toy_dense"}},
    )
    names = [n for n, _ in Transformer(cfg).named_parameters()]
    assert len({zlib.crc32(n.encode()) for n in names}) == len(names)
