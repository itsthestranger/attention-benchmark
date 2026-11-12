"""T-21: synthetic memory probes (plan §9.2): generators, oracle, stream, accuracy, probe training.

Every number here is an exact count, an equality, or the exact 1.0 that plan §9.2
asks for. No tolerances, no skips. The data is synthetic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_probe import HELD_OUT_OFFSET, main, probe_resolved, run_probe
from src import config
from src.eval import synthetic as syn
from src.eval.synthetic import IGNORE, ProbeStream, accuracy, generate, make_spec, solve
from src.model import build_model

# The tiny MHA acceptance run (test 11). Settings chosen by measurement, see the report.
MHA_STEPS = 800
MHA_PEAK_LR = 3e-3
MHA_SEQUENCES_PER_STEP = 32
MHA_D_MODEL = 64
MHA_N_LAYERS = 2

SPECS = [
    make_spec("mqar", n_pairs=4),
    make_spec("mqar", n_pairs=64),
    make_spec("mqar", n_pairs=256),
    make_spec("mqar", n_pairs=16, n_keys=4096, n_values=4096),
    make_spec("multihop", n_chains=4),
    make_spec("multihop", n_chains=16),
    make_spec("fuzzy", n_pairs=8, key_len=3, n_noise=1),
    make_spec("fuzzy", n_pairs=8, key_len=3, n_noise=2),
    make_spec("fuzzy", n_pairs=8, key_len=4, n_noise=1),
    make_spec("fuzzy", n_pairs=8, key_len=4, n_noise=2),
    make_spec("selective_copy", n_data=8, n_blank=12),
    make_spec("induction", length=16),
]
SPEC_IDS = [f"{s.task}-{'-'.join(f'{k}{v}' for k, v in s.params)}" for s in SPECS]


def _expected_positions(spec):
    """The scored positions the plan documents, computed from the parameters alone."""
    p = spec.as_dict()
    if spec.task == "mqar":
        n = p["n_pairs"]
        return [2 * n + 1 + 2 * j for j in range(n)]
    if spec.task == "multihop":
        n = p["n_chains"]
        return [4 * n + 1 + 2 * j for j in range(n)]
    if spec.task == "fuzzy":
        n, k = p["n_pairs"], p["key_len"]
        s = n * (k + 1)
        return [s + 1 + j * (k + 1) + (k - 1) for j in range(n)]
    if spec.task == "selective_copy":
        c = p["n_data"] + p["n_blank"]
        return [c + i for i in range(p["n_data"])]
    if spec.task == "induction":
        length = p["length"]
        return list(range(length, 2 * length))
    raise AssertionError(spec.task)


def _batch(spec, n=64, seed=0):
    x, y = generate(spec, n, np.random.default_rng(seed))
    return x.numpy(), y.numpy()


# --- 1. shapes, dtype, ranges, documented formulas ----------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_shapes_dtype_and_vocabulary_range(spec):
    x, y = generate(spec, 32, np.random.default_rng(1))
    assert x.shape == (32, spec.seq_len) and y.shape == (32, spec.seq_len)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert int(x.min()) >= 0 and int(x.max()) < spec.vocab_size
    valid = (y == IGNORE) | ((y >= 0) & (y < spec.vocab_size))
    assert bool(valid.all())


def test_documented_lengths_and_vocabularies_literal():
    cases = [
        (make_spec("mqar", n_pairs=4), 17, 132),
        (make_spec("mqar", n_pairs=64), 257, 196),
        (make_spec("mqar", n_pairs=256), 1025, 4 + 512 + 64),
        (make_spec("mqar", n_pairs=16, n_keys=4096, n_values=4096), 65, 4 + 4096 + 4096),
        (make_spec("multihop", n_chains=4), 25, 4 + 64),
        (make_spec("fuzzy", n_pairs=4, key_len=3), 33, 4 + 64 + 64 + 16),
        (make_spec("fuzzy", n_pairs=4, key_len=4, n_noise=2), 41, 4 + 64 + 64 + 16),
        (make_spec("selective_copy", n_data=4, n_blank=6), 15, 4 + 16),
        (make_spec("induction", length=5), 11, 4 + 64),
    ]
    for spec, seq_len, vocab in cases:
        assert spec.seq_len == seq_len, spec
        assert spec.vocab_size == vocab, spec


# --- 2. the oracle reproduces every target from the input alone ---------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_oracle_reproduces_every_target_from_input(spec):
    x, y = _batch(spec, n=64, seed=2)
    for b in range(x.shape[0]):
        scored = [(t, int(y[b, t])) for t in range(spec.seq_len) if y[b, t] != IGNORE]
        assert solve(spec, x[b]) == scored, (spec, b)


# --- 3. teacher forcing ----------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_teacher_forcing_is_consistent(spec):
    x, y = _batch(spec, n=32, seed=3)
    checked = 0
    for b in range(x.shape[0]):
        for t in np.flatnonzero(y[b] != IGNORE):
            if t + 1 < spec.seq_len:
                assert y[b, t] == x[b, t + 1]
                checked += 1
    assert checked > 0


# --- 4. exactly the documented positions are scored ----------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_exactly_the_documented_positions_are_scored(spec):
    _, y = _batch(spec, n=32, seed=4)
    expected = _expected_positions(spec)
    for b in range(y.shape[0]):
        assert np.flatnonzero(y[b] != IGNORE).tolist() == expected


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_documented_scored_counts(spec):
    _, y = _batch(spec, n=8, seed=5)
    counts = (y != IGNORE).sum(axis=1)
    p = spec.as_dict()
    want = {
        "mqar": p.get("n_pairs"),
        "multihop": p.get("n_chains"),
        "fuzzy": p.get("n_pairs"),
        "selective_copy": p.get("n_data"),
        "induction": p.get("length"),
    }[spec.task]
    assert set(counts.tolist()) == {want}


# --- 5. structure of the sequences ---------------------------------------------------------


def test_mqar_keys_distinct_and_each_queried_once():
    spec = make_spec("mqar", n_pairs=64)
    n = spec.param("n_pairs")
    x, _ = _batch(spec, n=64, seed=10)
    for row in x:
        keys = row[0 : 2 * n : 2].tolist()
        values = row[1 : 2 * n : 2].tolist()
        assert len(set(keys)) == n
        assert row[2 * n] == syn.SEP
        queries = row[2 * n + 1 : 4 * n + 1 : 2].tolist()
        answers = row[2 * n + 2 : 4 * n + 1 : 2].tolist()
        assert sorted(queries) == sorted(keys)
        table = dict(zip(keys, values))
        assert answers == [table[q] for q in queries]


def test_multihop_symbols_distinct_and_chains_consistent():
    spec = make_spec("multihop", n_chains=8)
    n = spec.param("n_chains")
    x, _ = _batch(spec, n=64, seed=11)
    for row in x:
        ctx = row[: 4 * n].reshape(2 * n, 2).tolist()
        assert len(set(row[: 4 * n].tolist())) == 3 * n
        firsts = {a for a, _ in ctx}
        seconds = {b for _, b in ctx}
        starts = firsts - seconds
        assert len(starts) == n
        queried = row[4 * n + 1 : 6 * n + 1 : 2].tolist()
        assert sorted(queried) == sorted(starts)
        succ = dict(ctx)
        answers = row[4 * n + 2 : 6 * n + 1 : 2].tolist()
        assert answers == [succ[succ[a]] for a in queried]


def test_fuzzy_noise_positions_and_key_identity():
    for k, m in ((3, 1), (3, 2), (4, 1), (4, 2)):
        spec = make_spec("fuzzy", n_pairs=8, key_len=k, n_noise=m)
        n = spec.param("n_pairs")
        r = syn._ranges(spec)
        noise_lo, noise_hi = r["noise"]
        x, _ = _batch(spec, n=32, seed=12)
        s = n * (k + 1)
        for row in x:
            ctx = row[:s].reshape(n, k + 1)
            keys = ctx[:, :k]
            for j in range(k):
                assert len(set(keys[:, j].tolist())) == n
            identities = []
            for j in range(n):
                base = s + 1 + j * (k + 1)
                block = row[base : base + k + 1]
                noisy = [o for o in range(k) if noise_lo <= block[o] < noise_hi]
                assert len(noisy) == m
                clean = [o for o in range(k) if o not in noisy]
                matches = [
                    i for i in range(n) if all(keys[i, o] == block[o] for o in clean)
                ]
                assert len(matches) == 1
                identities.append(matches[0])
                assert block[k] == ctx[matches[0], k]
            assert sorted(identities) == list(range(n))


def test_selective_copy_copies_data_tokens_in_order():
    spec = make_spec("selective_copy", n_data=8, n_blank=12)
    n, m = spec.param("n_data"), spec.param("n_blank")
    x, _ = _batch(spec, n=64, seed=13)
    c = n + m
    for row in x:
        assert row[c] == syn.COPY
        assert (row[:c] == syn.BLANK).sum() == m
        assert row[c + 1 :].tolist() == [t for t in row[:c].tolist() if t != syn.BLANK]


def test_induction_second_half_repeats_first():
    spec = make_spec("induction", length=16)
    length = spec.param("length")
    x, _ = _batch(spec, n=64, seed=14)
    for row in x:
        assert row[length] == syn.SEP
        assert row[length + 1 :].tolist() == row[:length].tolist()
        assert len(set(row[:length].tolist())) == length


# --- 6. the answer cannot be read without the context --------------------------------------


def test_mqar_value_of_a_key_varies_across_sequences():
    spec = make_spec("mqar", n_pairs=4)
    n = spec.param("n_pairs")
    x, _ = _batch(spec, n=256, seed=20)
    seen: dict[int, set[int]] = {}
    for row in x:
        table = dict(zip(row[0 : 2 * n : 2].tolist(), row[1 : 2 * n : 2].tolist()))
        for key, value in table.items():
            seen.setdefault(key, set()).add(value)
    assert any(len(values) > 1 for values in seen.values())


def test_multihop_answer_for_a_symbol_varies_across_sequences():
    spec = make_spec("multihop", n_chains=4)
    n = spec.param("n_chains")
    x, _ = _batch(spec, n=256, seed=21)
    seen: dict[int, set[int]] = {}
    for row in x:
        succ = dict(row[: 4 * n].reshape(2 * n, 2).tolist())
        for a in {a for a in succ} - set(succ.values()):
            seen.setdefault(a, set()).add(succ[succ[a]])
    assert any(len(answers) > 1 for answers in seen.values())


# --- 7. determinism and the stream ---------------------------------------------------------


def test_same_rng_seed_gives_same_batch_and_different_seeds_differ():
    spec = make_spec("mqar", n_pairs=8)
    a = generate(spec, 16, np.random.default_rng(7))
    b = generate(spec, 16, np.random.default_rng(7))
    c = generate(spec, 16, np.random.default_rng(8))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_generators_do_not_read_the_global_rng():
    spec = make_spec("fuzzy", n_pairs=8, key_len=3, n_noise=1)
    np.random.seed(1)
    torch.manual_seed(1)
    first = generate(spec, 16, np.random.default_rng(3))
    np.random.seed(999)
    torch.manual_seed(999)
    second = generate(spec, 16, np.random.default_rng(3))
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_probe_stream_is_a_pure_function_of_step():
    spec = make_spec("induction", length=6)
    stream = ProbeStream(spec, sequences_per_step=8, seed=5, max_steps=10)
    first = stream.batch(5)
    stream.batch(0)
    stream.batch(9)
    again = stream.batch(5)
    assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])
    other_step = stream.batch(6)
    assert not torch.equal(first[0], other_step[0])
    other_seed = ProbeStream(spec, sequences_per_step=8, seed=6, max_steps=10).batch(5)
    assert not torch.equal(first[0], other_seed[0])


def test_probe_stream_refuses_steps_out_of_range():
    spec = make_spec("induction", length=6)
    stream = ProbeStream(spec, sequences_per_step=8, seed=5, max_steps=10)
    for bad in (-1, 10, 11):
        with pytest.raises(IndexError):
            stream.batch(bad)


def test_probe_stream_spec_is_hashable():
    spec = make_spec("mqar", n_pairs=4)
    assert hash(spec) == hash(make_spec("mqar", n_pairs=4))
    assert spec != make_spec("mqar", n_pairs=5)


# --- 8. make_spec refuses impossible parameters --------------------------------------------


@pytest.mark.parametrize(
    "task, params",
    [
        ("mqar", {}),
        ("mqar", {"n_pairs": 0}),
        ("mqar", {"n_pairs": 4, "n_keys": 3}),
        ("mqar", {"n_pairs": 4, "n_values": 0}),
        ("mqar", {"n_pairs": 4, "bogus": 1}),
        ("mqar", {"n_pairs": 4.0}),
        ("mqar", {"n_pairs": True}),
        ("multihop", {"n_chains": 4, "n_symbols": 11}),
        ("multihop", {"n_chains": -1}),
        ("fuzzy", {"n_pairs": 4, "key_len": 3, "n_noise": 3}),
        ("fuzzy", {"n_pairs": 4, "n_keys": 3}),
        ("fuzzy", {"n_pairs": 4, "n_noise_tokens": 0}),
        ("selective_copy", {"n_data": 0, "n_blank": 4}),
        ("selective_copy", {"n_data": 4, "n_blank": -1}),
        ("selective_copy", {"n_data": 4}),
        ("induction", {"length": 10, "n_symbols": 9}),
        ("induction", {"length": 0}),
        ("nonsense", {}),
    ],
)
def test_make_spec_refuses_impossible_parameters(task, params):
    with pytest.raises(ValueError):
        make_spec(task, **params)


def test_make_spec_fills_defaults():
    spec = make_spec("mqar", n_pairs=256)
    assert spec.as_dict() == {"n_pairs": 256, "n_keys": 512, "n_values": 64}
    assert make_spec("fuzzy", n_pairs=4).as_dict() == {
        "n_pairs": 4,
        "key_len": 3,
        "n_noise": 1,
        "n_keys": 64,
        "n_values": 64,
        "n_noise_tokens": 16,
    }


# --- 9. accuracy on a stub with hand-computed answers --------------------------------------


class _Stub(nn.Module):
    """Returns one-hot logits of a fixed prediction table."""

    def __init__(self, pred: torch.Tensor, vocab: int):
        super().__init__()
        self.pred = pred
        self.vocab = vocab
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, idx, mode="opt"):
        return F.one_hot(self.pred, self.vocab).float()


def test_accuracy_matches_hand_computed_values_and_restores_mode():
    y = torch.tensor([[5, IGNORE, 7, 9], [IGNORE, 3, 4, IGNORE], [1, 2, IGNORE, IGNORE]])
    pred = torch.tensor([[5, 0, 7, 0], [0, 3, 0, 0], [1, 2, 0, 0]])
    x = torch.zeros_like(y)
    model = _Stub(pred, vocab=10)
    model.train()
    model.child = nn.Dropout(0.5)  # a submodule in a different mode
    model.child.eval()
    result = accuracy(model, x, y)
    # Scored: 3 + 2 + 2 = 7. Correct: 2 + 1 + 2 = 5. Sequences with every scored position right: 1 of 3.
    assert result == {"token_accuracy": 5 / 7, "sequence_accuracy": 1 / 3, "n_scored": 7}
    assert model.training is True
    assert model.child.training is False


def test_accuracy_restores_eval_mode_too():
    y = torch.tensor([[1, IGNORE]])
    model = _Stub(torch.tensor([[1, 0]]), vocab=3)
    model.eval()
    accuracy(model, torch.zeros_like(y), y)
    assert model.training is False


def test_accuracy_refuses_no_scored_positions():
    y = torch.full((2, 3), IGNORE)
    with pytest.raises(ValueError):
        accuracy(_Stub(torch.zeros(2, 3, dtype=torch.long), vocab=4), torch.zeros_like(y), y)


# --- 10. probe_resolved --------------------------------------------------------------------

ALL_VARIANTS = config.variant_names() + ["recurrence"]


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_every_variant_builds_with_eight_layers_and_eight_heads(variant):
    spec = make_spec("mqar", n_pairs=4)
    resolved = probe_resolved(variant, spec, d_model=96, init_seed=1, steps=5, peak_lr=1e-3, n_layers=None)
    assert resolved["model"]["n_layers"] == 8
    assert resolved["model"]["n_heads"] == 8
    model = build_model(resolved)
    assert len(model.layers) == 8
    logits, _ = model(torch.zeros(2, spec.seq_len, dtype=torch.long), mode="opt")
    assert logits.shape == (2, spec.seq_len, spec.vocab_size)


def test_probe_widths_literal():
    spec = make_spec("mqar", n_pairs=4)

    def model_of(variant, d_model):
        return probe_resolved(variant, spec, d_model=d_model, init_seed=1, steps=5, peak_lr=1e-3)["model"]

    mha = model_of("mha", 96)
    assert (mha["head_dim"], mha["d_ff"], mha["d_model"]) == (12, 256, 96)
    mqa = model_of("mqa", 96)
    assert (mqa["head_dim"], mqa["d_ff"]) == (12, 312)
    mla = model_of("mla192", 96)
    assert (mla["attention"]["d_c"], mla["attention"]["d_c_q"], mla["attention"]["rope_dim"]) == (48, 48, 6)
    assert model_of("mha", 128)["head_dim"] == 16
    assert model_of("mha", 128)["d_ff"] == round(1024 * 128 / 384)
    assert model_of("recurrence", 96)["attention"] == {"type": "recurrence"}
    assert model_of("swa256", 96)["attention"] == {"type": "sliding_window", "n_kv_heads": 8, "window": 256}
    assert model_of("gqa4", 96)["attention"] == {"type": "grouped", "n_kv_heads": 4}


def test_probe_sets_task_shape_and_batch_entries():
    spec = make_spec("selective_copy", n_data=4, n_blank=6)
    resolved = probe_resolved("mha", spec, d_model=64, init_seed=3, steps=50, peak_lr=2e-3, sequences_per_step=16)
    assert resolved["model"]["vocab_size"] == spec.vocab_size
    assert resolved["model"]["context_length"] == spec.seq_len
    assert resolved["recipe"]["batch"]["micro_batch"] == 16
    assert resolved["recipe"]["batch"]["tokens_per_step"] == 16 * spec.seq_len
    assert resolved["run"]["steps"] == 50 and resolved["run"]["branches"] == []


@pytest.mark.parametrize(
    "variant, d_model",
    [("mha", 100), ("mha", 0), ("mla192", 112), ("mla96", 8)],
)
def test_probe_refuses_impossible_widths(variant, d_model):
    spec = make_spec("mqar", n_pairs=4)
    with pytest.raises(ValueError):
        probe_resolved(variant, spec, d_model=d_model, init_seed=1, steps=5, peak_lr=1e-3)


def test_probe_refuses_unknown_variant_and_bad_micro_batch():
    spec = make_spec("mqar", n_pairs=4)
    with pytest.raises(ValueError):
        probe_resolved("nope", spec, d_model=64, init_seed=1, steps=5, peak_lr=1e-3)
    with pytest.raises(ValueError):
        probe_resolved("mha", spec, d_model=64, init_seed=1, steps=5, peak_lr=1e-3, sequences_per_step=32, micro_batch=5)


# --- 11 and 13. the tiny runs ---------------------------------------------------------------


@pytest.fixture(scope="module")
def mha_probe_result():
    return run_probe(
        "mha",
        make_spec("mqar", n_pairs=4),
        d_model=MHA_D_MODEL,
        init_seed=1,
        steps=MHA_STEPS,
        peak_lr=MHA_PEAK_LR,
        n_layers=MHA_N_LAYERS,
        sequences_per_step=MHA_SEQUENCES_PER_STEP,
    )


def test_tiny_mha_solves_four_pair_mqar(mha_probe_result):
    assert mha_probe_result["sequence_accuracy"] == 1.0
    assert mha_probe_result["token_accuracy"] == 1.0
    assert mha_probe_result["skipped_steps"] == 0


def test_recurrence_probe_runs_and_reports_accuracies_in_range():
    result = run_probe(
        "recurrence",
        make_spec("mqar", n_pairs=4),
        d_model=MHA_D_MODEL,
        init_seed=1,
        steps=MHA_STEPS,
        peak_lr=MHA_PEAK_LR,
        n_layers=MHA_N_LAYERS,
        sequences_per_step=MHA_SEQUENCES_PER_STEP,
    )
    assert result["variant"] == "recurrence"
    assert 0.0 <= result["token_accuracy"] <= 1.0
    assert 0.0 <= result["sequence_accuracy"] <= 1.0


# --- 12. held-out data, result format, CLI -------------------------------------------------


def test_held_out_batches_differ_from_every_training_batch():
    spec = make_spec("mqar", n_pairs=4)
    train = ProbeStream(spec, sequences_per_step=32, seed=0, max_steps=60)
    held = ProbeStream(spec, sequences_per_step=32, seed=0 + HELD_OUT_OFFSET, max_steps=4)
    train_batches = [train.batch(i)[0] for i in range(60)]
    for i in range(4):
        h = held.batch(i)[0]
        assert not any(torch.equal(h, t) for t in train_batches)


def test_run_probe_result_is_json_serialisable(mha_probe_result):
    text = json.dumps(mha_probe_result)
    assert json.loads(text) == mha_probe_result
    for key in (
        "variant", "task", "spec", "d_model", "n_layers", "init_seed", "data_seed", "steps",
        "peak_lr", "final_train_loss", "token_accuracy", "sequence_accuracy", "n_scored", "config_hash",
    ):
        assert key in mha_probe_result
    assert mha_probe_result["task"] == "mqar"
    assert mha_probe_result["spec"]["params"] == {"n_keys": 64, "n_pairs": 4, "n_values": 64}


def test_cli_writes_the_result_as_json(tmp_path):
    out = tmp_path / "result.json"
    status = main(
        [
            "--variant", "mha",
            "--task", "mqar",
            "--param", "n_pairs=4",
            "--d-model", "64",
            "--seed", "1",
            "--steps", "3",
            "--peak-lr", "0.001",
            "--sequences-per-step", "4",
            "--eval-batches", "1",
            "--n-layers", "1",
            "--out", str(out),
        ]
    )
    assert status == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["variant"] == "mha" and written["task"] == "mqar"
    assert written["steps"] == 3 and written["n_layers"] == 1
    assert written["n_scored"] == 4 * 4
