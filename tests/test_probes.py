"""T-21: synthetic memory probes (plan §9.2): generators, oracle, stream, accuracy, probe training and records.

Every number here is an exact count, an equality, or the exact 1.0 that plan §9.2
asks for. No tolerances, no skips. The data is synthetic. The gpu test is written
and skips here; it has not run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import scripts.train_probe as tp
from scripts.train_probe import HELD_OUT_OFFSET, main, probe_path, probe_resolved, run_probe, run_to_dir
from src import config
from src.eval import synthetic as syn
from src.eval.synthetic import BLANK, IGNORE, SEP, ProbeStream, accuracy, counts, generate, make_spec, solve
from src.model import build_model
from src.training.trainer import Trainer
from tests.fixtures.unfilled import unfilled_recipe

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
RECALL_SPECS = [s for s in SPECS if s.task in ("mqar", "multihop", "fuzzy")]
COPY_SPECS = [s for s in SPECS if s.task in ("selective_copy", "induction")]
RECALL_IDS = [f"{s.task}-{'-'.join(f'{k}{v}' for k, v in s.params)}" for s in RECALL_SPECS]
COPY_IDS = [f"{s.task}-{'-'.join(f'{k}{v}' for k, v in s.params)}" for s in COPY_SPECS]

# The tiny memorisation run (acceptance, CPU). Measured with the layout of this commit, with
# the probe settings below, on the batch generate(spec, 16, default_rng(0)) for mqar 4 pairs,
# 16 keys, 16 values. Per init seed 1, 2, 3: the batch is first fully memorised after 39, 44
# and 42 steps, but not kept: the last step at which it is not fully memorised is 581, 607
# and 566. The budget is about twice the slowest durable point (607), and at 1220 steps all
# three seeds end at 64/64 tokens and 16/16 sequences with no skipped step.
MEMO_STEPS = 1220
MEMO_PEAK_LR = 3e-3
MEMO_SEQS = 16
MEMO_D_MODEL = 64
MEMO_N_LAYERS = 2
MEMO_BATCH_SEED = 0

# The held-out 4-pair MQAR run (gpu tier). 12,750 is three times 4,250, the slowest of three
# CPU solves measured earlier with these settings (first held-out sequence accuracy of 1.0 at
# steps 4,250, 2,250 and 3,000).
GPU_STEPS = 12_750
GPU_EVAL = 1024

MHA_D_MODEL = 64
MHA_N_LAYERS = 2


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
        (make_spec("mqar", n_pairs=64), 257, 4 + 128 + 128),
        (make_spec("mqar", n_pairs=256), 1025, 4 + 512 + 512),
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


# --- 3. teacher forcing, and its absence for the recall tasks ----------------------------


@pytest.mark.parametrize("spec", COPY_SPECS, ids=COPY_IDS)
def test_teacher_forcing_holds_for_copy_tasks(spec):
    x, y = _batch(spec, n=32, seed=3)
    checked = 0
    for b in range(x.shape[0]):
        for t in np.flatnonzero(y[b] != IGNORE):
            if t + 1 < spec.seq_len:
                assert y[b, t] == x[b, t + 1]
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("spec", RECALL_SPECS, ids=RECALL_IDS)
def test_recall_tasks_feed_blank_after_every_scored_position(spec):
    x, y = _batch(spec, n=32, seed=3)
    checked = 0
    for b in range(x.shape[0]):
        for t in np.flatnonzero(y[b] != IGNORE):
            if t + 1 < spec.seq_len:
                assert x[b, t + 1] == BLANK
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("spec", RECALL_SPECS, ids=RECALL_IDS)
def test_recall_answers_are_not_in_the_input_after_sep(spec):
    x, y = _batch(spec, n=64, seed=6)
    r = syn._ranges(spec)
    for b in range(x.shape[0]):
        s = int(np.flatnonzero(x[b] == SEP)[0])
        after = x[b, s + 1 :]
        for t in np.flatnonzero(y[b] != IGNORE):
            assert int(y[b, t]) not in after.tolist(), (spec, b, t)
        if spec.task in ("mqar", "fuzzy"):
            lo, hi = r["values"]
            assert not bool(((after >= lo) & (after < hi)).any()), (spec, b)


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
    counts_per_row = (y != IGNORE).sum(axis=1)
    p = spec.as_dict()
    want = {
        "mqar": p.get("n_pairs"),
        "multihop": p.get("n_chains"),
        "fuzzy": p.get("n_pairs"),
        "selective_copy": p.get("n_data"),
        "induction": p.get("length"),
    }[spec.task]
    assert set(counts_per_row.tolist()) == {want}


# --- 5. structure of the sequences ---------------------------------------------------------


def test_mqar_keys_distinct_each_queried_once_and_answers_are_the_values():
    spec = make_spec("mqar", n_pairs=64)
    n = spec.param("n_pairs")
    x, y = _batch(spec, n=64, seed=10)
    for b, row in enumerate(x):
        keys = row[0 : 2 * n : 2].tolist()
        values = row[1 : 2 * n : 2].tolist()
        assert len(set(keys)) == n
        assert row[2 * n] == syn.SEP
        queries = row[2 * n + 1 : 4 * n + 1 : 2].tolist()
        assert row[2 * n + 2 : 4 * n + 1 : 2].tolist() == [BLANK] * n
        assert sorted(queries) == sorted(keys)
        table = dict(zip(keys, values))
        answers = [int(y[b, 2 * n + 1 + 2 * j]) for j in range(n)]
        assert answers == [table[q] for q in queries]


def test_mqar_query_order_is_not_context_order():
    spec = make_spec("mqar", n_pairs=4, n_keys=16, n_values=16)
    n = spec.param("n_pairs")
    x, _ = _batch(spec, n=256, seed=30)
    first_query_slots = set()
    for row in x:
        keys = row[0 : 2 * n : 2].tolist()
        first_query_slots.add(keys.index(int(row[2 * n + 1])))
    assert first_query_slots == set(range(n))


def test_mqar_and_fuzzy_values_are_distinct_within_every_sequence():
    spec = make_spec("mqar", n_pairs=16)
    n = spec.param("n_pairs")
    x, _ = _batch(spec, n=64, seed=31)
    for row in x:
        assert len(set(row[1 : 2 * n : 2].tolist())) == n
    spec = make_spec("fuzzy", n_pairs=8, key_len=3, n_noise=1)
    n, k = spec.param("n_pairs"), spec.param("key_len")
    x, _ = _batch(spec, n=64, seed=31)
    for row in x:
        ctx = row[: n * (k + 1)].reshape(n, k + 1)
        assert len(set(ctx[:, k].tolist())) == n


def test_multihop_symbols_distinct_and_chains_consistent():
    spec = make_spec("multihop", n_chains=8)
    n = spec.param("n_chains")
    x, y = _batch(spec, n=64, seed=11)
    for b, row in enumerate(x):
        ctx = row[: 4 * n].reshape(2 * n, 2).tolist()
        assert len(set(row[: 4 * n].tolist())) == 3 * n
        firsts = {a for a, _ in ctx}
        seconds = {b_ for _, b_ in ctx}
        starts = firsts - seconds
        assert len(starts) == n
        queried = row[4 * n + 1 : 6 * n + 1 : 2].tolist()
        assert row[4 * n + 2 : 6 * n + 1 : 2].tolist() == [BLANK] * n
        assert sorted(queried) == sorted(starts)
        succ = dict(ctx)
        answers = [int(y[b, 4 * n + 1 + 2 * j]) for j in range(n)]
        assert answers == [succ[succ[a]] for a in queried]


def test_fuzzy_noise_positions_key_identity_and_answers():
    for k, m in ((3, 1), (3, 2), (4, 1), (4, 2)):
        spec = make_spec("fuzzy", n_pairs=8, key_len=k, n_noise=m)
        n = spec.param("n_pairs")
        r = syn._ranges(spec)
        noise_lo, noise_hi = r["noise"]
        x, y = _batch(spec, n=32, seed=12)
        s = n * (k + 1)
        for b, row in enumerate(x):
            ctx = row[:s].reshape(n, k + 1)
            keys = ctx[:, :k]
            for j in range(k):
                assert len(set(keys[:, j].tolist())) == n
            identities = []
            for j in range(n):
                base = s + 1 + j * (k + 1)
                block = row[base : base + k + 1]
                assert block[k] == BLANK
                noisy = [o for o in range(k) if noise_lo <= block[o] < noise_hi]
                assert len(noisy) == m
                clean = [o for o in range(k) if o not in noisy]
                matches = [i for i in range(n) if all(keys[i, o] == block[o] for o in clean)]
                assert len(matches) == 1
                identities.append(matches[0])
                assert int(y[b, base + k - 1]) == ctx[matches[0], k]
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


# --- 8. make_spec refuses impossible parameters and fills defaults -------------------------


@pytest.mark.parametrize(
    "task, params",
    [
        ("mqar", {}),
        ("mqar", {"n_pairs": 0}),
        ("mqar", {"n_pairs": 4, "n_keys": 3}),
        ("mqar", {"n_pairs": 4, "n_values": 0}),
        ("mqar", {"n_pairs": 4, "n_values": 3}),
        ("mqar", {"n_pairs": 4, "bogus": 1}),
        ("mqar", {"n_pairs": 4.0}),
        ("mqar", {"n_pairs": True}),
        ("multihop", {"n_chains": 4, "n_symbols": 11}),
        ("multihop", {"n_chains": -1}),
        ("fuzzy", {"n_pairs": 4, "key_len": 3, "n_noise": 3}),
        ("fuzzy", {"n_pairs": 4, "n_keys": 3}),
        ("fuzzy", {"n_pairs": 4, "n_values": 3}),
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
    assert spec.as_dict() == {"n_pairs": 256, "n_keys": 512, "n_values": 512}
    assert make_spec("mqar", n_pairs=4).as_dict() == {"n_pairs": 4, "n_keys": 64, "n_values": 64}
    assert make_spec("mqar", n_pairs=40).as_dict()["n_values"] == 80
    assert make_spec("fuzzy", n_pairs=4).as_dict() == {
        "n_pairs": 4,
        "key_len": 3,
        "n_noise": 1,
        "n_keys": 64,
        "n_values": 64,
        "n_noise_tokens": 16,
    }
    assert make_spec("fuzzy", n_pairs=40).as_dict()["n_values"] == 80


def test_difficulty_and_chance_level_literal():
    cases = [
        (make_spec("mqar", n_pairs=4), 4, 0.25),
        (make_spec("multihop", n_chains=4), 4, 0.25),
        (make_spec("fuzzy", n_pairs=8), 8, 0.125),
        (make_spec("selective_copy", n_data=8, n_blank=12), 8, None),
        (make_spec("induction", length=16), 16, None),
    ]
    for spec, level, chance in cases:
        assert syn.difficulty(spec) == level, spec
        assert syn.chance_level(spec) == chance, spec


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
    assert model_of("swa256", 96)["attention"] == {
        "type": "sliding_window",
        "n_kv_heads": 8,
        "window": 256,
        "opt_path": "chunked",
    }
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


# --- 11. acceptance: exact memorisation of one fixed batch (CPU) ----------------------------


class _FixedBatchStream:
    """The same batch at every step: the memorisation acceptance test."""

    def __init__(self, x, y, max_steps):
        self._x, self._y = x, y
        self.max_steps = max_steps

    def batch(self, step):
        return self._x, self._y


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tiny_mha_memorises_one_fixed_batch_exactly(seed):
    spec = make_spec("mqar", n_pairs=4, n_keys=16, n_values=16)
    x, y = generate(spec, MEMO_SEQS, np.random.default_rng(MEMO_BATCH_SEED))
    resolved = probe_resolved(
        "mha",
        spec,
        d_model=MEMO_D_MODEL,
        init_seed=seed,
        steps=MEMO_STEPS,
        peak_lr=MEMO_PEAK_LR,
        sequences_per_step=MEMO_SEQS,
        n_layers=MEMO_N_LAYERS,
    )
    trainer = Trainer(
        build_model(resolved), resolved, _FixedBatchStream(x, y, MEMO_STEPS), device="cpu", mode="opt"
    )
    trainer.train()
    assert trainer.skipped_steps == 0
    c = counts(trainer.model, x, y)
    assert c["n_correct_tokens"] / c["n_scored"] == 1.0
    assert c["n_correct_sequences"] / c["n_sequences"] == 1.0


def test_recurrence_probe_runs_and_reports_accuracies_in_range():
    result = run_probe(
        "recurrence",
        make_spec("mqar", n_pairs=4),
        d_model=MHA_D_MODEL,
        init_seed=1,
        steps=40,
        peak_lr=3e-3,
        n_layers=MHA_N_LAYERS,
        sequences_per_step=16,
        eval_sequences=64,
    )
    assert result["variant"] == "recurrence"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert 0.0 <= result["sequence_accuracy"] <= 1.0


# --- 12. held-out data, the record, pooling -------------------------------------------------


def test_held_out_batches_differ_from_every_training_batch():
    spec = make_spec("mqar", n_pairs=4)
    train = ProbeStream(spec, sequences_per_step=32, seed=0, max_steps=60)
    held = ProbeStream(spec, sequences_per_step=32, seed=0 + HELD_OUT_OFFSET, max_steps=4)
    train_batches = [train.batch(i)[0] for i in range(60)]
    for i in range(4):
        h = held.batch(i)[0]
        assert not any(torch.equal(h, t) for t in train_batches)


RECORD_KEYS = {
    "schema", "variant", "task", "generator", "difficulty", "init_seed", "accuracy", "sequence_accuracy",
    "chance", "n_scored", "n_sequences", "spec", "d_model", "n_layers", "data_seed", "steps", "steps_taken",
    "skipped_steps", "peak_lr", "sequences_per_step", "eval_sequences", "final_train_loss", "config_hash",
    "mode", "precision", "backend", "device", "environment", "git_sha",
}


def _tiny_kwargs(**overrides):
    kwargs = dict(
        d_model=MHA_D_MODEL,
        init_seed=1,
        steps=3,
        peak_lr=1e-3,
        sequences_per_step=4,
        eval_sequences=6,
        n_layers=1,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def probe_record():
    return run_probe("mha", make_spec("mqar", n_pairs=4), **_tiny_kwargs())


def test_record_has_exactly_the_schema_keys_with_the_right_types(probe_record):
    assert set(probe_record) == RECORD_KEYS
    r = probe_record
    assert r["schema"] == "probe/1"
    assert r["variant"] == "mha" and r["task"] == "mqar" and r["generator"] == "mqar"
    assert isinstance(r["difficulty"], int) and r["difficulty"] == 4
    assert isinstance(r["init_seed"], int)
    assert isinstance(r["accuracy"], float) and isinstance(r["sequence_accuracy"], float)
    assert r["chance"] == 0.25
    assert isinstance(r["n_scored"], int) and isinstance(r["n_sequences"], int)
    assert set(r["spec"]) == {"seq_len", "vocab_size", "params"}
    assert r["spec"]["params"] == {"n_keys": 64, "n_pairs": 4, "n_values": 64}
    assert r["d_model"] == MHA_D_MODEL and r["n_layers"] == 1
    assert r["steps"] == 3 and r["steps_taken"] == 3 and r["skipped_steps"] == 0
    assert r["sequences_per_step"] == 4 and r["eval_sequences"] == 6
    assert isinstance(r["final_train_loss"], float)
    assert isinstance(r["config_hash"], str) and len(r["config_hash"]) == 64
    assert r["mode"] == "opt" and r["precision"] == "fp32"
    assert r["backend"] is None and r["device"] == "cpu"
    assert isinstance(r["environment"], dict) and isinstance(r["git_sha"], str) and len(r["git_sha"]) == 40


def test_record_round_trips_through_json(probe_record):
    text = json.dumps(probe_record, allow_nan=False)
    assert json.loads(text) == probe_record


def test_record_chance_is_none_for_induction():
    record = run_probe("mha", make_spec("induction", length=4), **_tiny_kwargs())
    assert record["chance"] is None and record["generator"] == "induction"
    assert record["difficulty"] == 4


def test_record_accuracy_is_pooled_counts_over_held_out_batches(monkeypatch):
    """A stub count per batch: the record must pool them exactly, and the batches must be the sizes asked for."""
    shapes = []
    real_counts = tp.counts

    def stub(model, x, y, *, mode="opt"):
        shapes.append(int(x.shape[0]))
        real_counts(model, x, y, mode=mode)  # the real forward still runs
        n = int(x.shape[0])
        return {"n_scored": 4 * n, "n_correct_tokens": 3 * n + 1, "n_sequences": n, "n_correct_sequences": n - 1}

    monkeypatch.setattr(tp, "counts", stub)
    record = run_probe("mha", make_spec("mqar", n_pairs=4), **_tiny_kwargs(eval_sequences=10, sequences_per_step=4))
    assert shapes == [4, 4, 2]
    assert record["n_sequences"] == 10
    assert record["n_scored"] == 40
    # Pooled: tokens (3*10 + 3) = 33 of 40; sequences (10 - 3) = 7 of 10.
    assert record["accuracy"] == 33 / 40
    assert record["sequence_accuracy"] == 7 / 10


def test_eval_sequences_not_divisible_by_batch_gives_exactly_that_many():
    record = run_probe("mha", make_spec("mqar", n_pairs=4), **_tiny_kwargs(eval_sequences=10, sequences_per_step=4))
    assert record["n_sequences"] == 10
    assert record["n_scored"] == 40
    assert record["eval_sequences"] == 10


def test_panel_and_difficulty_param_change_task_and_difficulty_only():
    spec = make_spec("mqar", n_pairs=4, n_keys=16, n_values=16)
    base = run_probe("mha", spec, **_tiny_kwargs())
    panel = run_probe("mha", spec, **_tiny_kwargs(), panel="panelA", difficulty_param="n_keys")
    assert panel["task"] == "panelA" and panel["difficulty"] == 16
    assert base["task"] == "mqar" and base["difficulty"] == 4
    for key in RECORD_KEYS - {"task", "difficulty"}:
        assert panel[key] == base[key], key


def test_unknown_difficulty_param_and_bad_panel_are_refused():
    spec = make_spec("mqar", n_pairs=4)
    with pytest.raises(ValueError):
        run_probe("mha", spec, **_tiny_kwargs(), difficulty_param="n_chains")
    with pytest.raises(ValueError):
        run_probe("mha", spec, **_tiny_kwargs(), panel="a/b")


def test_cpu_refuses_expected_backend_before_training():
    spec = make_spec("mqar", n_pairs=4)
    with pytest.raises(ValueError, match="CUDA only"):
        run_probe("mha", spec, **_tiny_kwargs(), expected_backend="math")


# --- 13. the output file: name rule, resumption, atomicity, CLI ----------------------------


def test_output_file_name_follows_the_documented_rule(tmp_path):
    spec = make_spec("mqar", n_pairs=4)
    kwargs = _tiny_kwargs()
    path = probe_path(tmp_path, "mha", spec, **kwargs)
    match = re.fullmatch(r"mqar-mha-n4-s1-([0-9a-f]{8})\.json", path.name)
    assert match is not None, path.name
    resolved = probe_resolved("mha", spec, d_model=MHA_D_MODEL, init_seed=1, steps=3, peak_lr=1e-3,
                              sequences_per_step=4, n_layers=1)
    identity = {
        "config_hash": config.config_hash(resolved),
        "spec": {"seq_len": spec.seq_len, "vocab_size": spec.vocab_size, "params": spec.as_dict()},
        "data_seed": 0,
        "eval_sequences": 6,
        "mode": "opt",
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert match.group(1) == digest[:8]
    panel_path = probe_path(tmp_path, "mha", spec, **kwargs, panel="p1")
    assert panel_path.name == f"p1-mha-n4-s1-{digest[:8]}.json"


def test_a_second_call_does_not_train_again(tmp_path, monkeypatch):
    spec = make_spec("mqar", n_pairs=4)
    path, trained = run_to_dir(tmp_path, "mha", spec, **_tiny_kwargs())
    assert trained is True and path.exists()
    before = (path.stat().st_mtime_ns, path.read_bytes())

    def no_training(*args, **kwargs):
        raise AssertionError("run_probe was called for a run whose record exists")

    monkeypatch.setattr(tp, "run_probe", no_training)
    again, trained_again = run_to_dir(tmp_path, "mha", spec, **_tiny_kwargs())
    assert again == path and trained_again is False
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before


def test_force_retrains_and_replaces_the_file(tmp_path, monkeypatch):
    spec = make_spec("mqar", n_pairs=4)
    path, _ = run_to_dir(tmp_path, "mha", spec, **_tiny_kwargs())
    calls = []
    real = tp.run_probe

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(tp, "run_probe", spy)
    _, trained = run_to_dir(tmp_path, "mha", spec, force=True, **_tiny_kwargs())
    assert trained is True and calls == [1]
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "probe/1"


def test_a_training_error_or_interrupt_leaves_no_file(tmp_path, monkeypatch):
    spec = make_spec("mqar", n_pairs=4)

    class Boom(Trainer):
        def train(self, until=None):
            raise RuntimeError("injected failure")

    monkeypatch.setattr(tp, "Trainer", Boom)
    with pytest.raises(RuntimeError, match="injected"):
        run_to_dir(tmp_path, "mha", spec, **_tiny_kwargs())
    assert list(tmp_path.iterdir()) == []

    class Interrupted(Trainer):
        def train(self, until=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(tp, "Trainer", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_to_dir(tmp_path, "mha", spec, **_tiny_kwargs())
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_no_file_and_no_temporary(tmp_path, monkeypatch):
    spec = make_spec("mqar", n_pairs=4)

    def failing_replace(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected"):
        run_to_dir(tmp_path, "mha", spec, **_tiny_kwargs())
    assert list(tmp_path.iterdir()) == []


def test_cli_writes_one_record_then_skips(tmp_path, capsys):
    argv = [
        "--variant", "mha", "--task", "mqar", "--param", "n_pairs=4", "--d-model", "64", "--seed", "1",
        "--steps", "3", "--peak-lr", "0.001", "--sequences-per-step", "4", "--eval-sequences", "6",
        "--n-layers", "1", "--out", str(tmp_path),
    ]
    assert main(argv) == 0
    files = list(tmp_path.glob("mqar-mha-n4-s1-*.json"))
    assert len(files) == 1
    written = json.loads(files[0].read_text(encoding="utf-8"))
    assert written["schema"] == "probe/1" and written["n_sequences"] == 6 and written["n_scored"] == 24
    assert "wrote" in capsys.readouterr().out
    mtime = files[0].stat().st_mtime_ns
    assert main(argv) == 0
    assert "exists" in capsys.readouterr().out
    assert files[0].stat().st_mtime_ns == mtime


def test_cli_refuses_expected_backend_on_cpu(tmp_path, capsys):
    argv = [
        "--variant", "mha", "--task", "mqar", "--param", "n_pairs=4", "--d-model", "64", "--seed", "1",
        "--steps", "3", "--peak-lr", "0.001", "--sequences-per-step", "4", "--eval-sequences", "6",
        "--n-layers", "1", "--out", str(tmp_path), "--expected-backend", "math",
    ]
    assert main(argv) == 2
    assert "CUDA only" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# --- 14. the held-out 4-pair MQAR run: one helper, a CPU smoke test, and the gpu test -----


# The fp16 GradScaler values the recipe leaves unset (configs/recipe.yaml is frozen). These are
# the values of tests/test_trainer.py::SCALER; passing them as overrides is what that test does.
SCALER_OVERRIDES = {"recipe.grad_scaler.init_scale": 65536.0, "recipe.grad_scaler.growth_interval": 2000}


def mqar4_probe(seed, device, *, steps=GPU_STEPS, eval_sequences=GPU_EVAL):
    """The held-out 4-pair MQAR run of plan §9.2. On CUDA the trainer enforces the math kernel
    (as the gpu tests in test_trainer.py do); on CPU there is no kernel to enforce. The scaler
    overrides are passed on every device, so the CPU smoke test runs the same path."""
    return run_probe(
        "mha",
        make_spec("mqar", n_pairs=4, n_keys=16, n_values=16),
        d_model=MEMO_D_MODEL,
        n_layers=MEMO_N_LAYERS,
        init_seed=seed,
        steps=steps,
        peak_lr=MEMO_PEAK_LR,
        sequences_per_step=64,
        eval_sequences=eval_sequences,
        device=device,
        expected_backend="math" if device == "cuda" else None,
        overrides=SCALER_OVERRIDES,
    )


def test_mqar4_probe_helper_runs_on_cpu_and_returns_a_valid_record():
    record = mqar4_probe(1, "cpu", steps=3, eval_sequences=10)
    assert set(record) == RECORD_KEYS
    assert record["device"] == "cpu" and record["precision"] == "fp32" and record["backend"] is None
    assert record["steps_taken"] == 3 and record["n_sequences"] == 10 and record["eval_sequences"] == 10


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tiny_mha_solves_held_out_four_pair_mqar_on_cuda(seed):
    record = mqar4_probe(seed, "cuda")
    assert record["accuracy"] == 1.0
    assert record["sequence_accuracy"] == 1.0
    assert record["precision"] == "fp16"


# --- 15. review gaps: layout order, data seeds, skipped steps, mode, overrides ----------------


def test_multihop_context_pairs_are_shuffled_so_no_position_gives_the_second_pair():
    spec = make_spec("multihop", n_chains=4)
    n = spec.param("n_chains")
    x, _ = _batch(spec, n=256, seed=40)
    first_slots, second_slots = set(), set()
    for row in x:
        pairs = row[: 4 * n].reshape(2 * n, 2).tolist()
        query = int(row[4 * n + 1])
        i = next(i for i, (a, _) in enumerate(pairs) if a == query)
        j = next(j for j, (a, _) in enumerate(pairs) if a == pairs[i][1])
        first_slots.add(i)
        second_slots.add(j)
    assert first_slots == set(range(2 * n))
    assert second_slots == set(range(2 * n))


def test_fuzzy_first_query_refers_to_every_context_slot():
    spec = make_spec("fuzzy", n_pairs=4, key_len=3, n_noise=1)
    n, k = spec.param("n_pairs"), spec.param("key_len")
    noise_lo, noise_hi = syn._ranges(spec)["noise"]
    x, _ = _batch(spec, n=256, seed=41)
    s = n * (k + 1)
    slots = set()
    for row in x:
        ctx = row[:s].reshape(n, k + 1)
        block = row[s + 1 : s + 1 + k + 1]
        o = next(o for o in range(k) if not noise_lo <= block[o] < noise_hi)
        slots.add(int(np.flatnonzero(ctx[:, o] == block[o])[0]))
    assert slots == set(range(n))


def test_held_out_counts_come_from_the_held_out_stream(monkeypatch):
    seen = []
    real_counts = tp.counts

    def spy(model, x, y, *, mode="opt"):
        seen.append(x.clone())
        return real_counts(model, x, y, mode=mode)

    monkeypatch.setattr(tp, "counts", spy)
    spec = make_spec("mqar", n_pairs=4)
    run_probe("mha", spec, **_tiny_kwargs(data_seed=5, steps=3, eval_sequences=6, sequences_per_step=4))
    held = ProbeStream(spec, sequences_per_step=4, seed=5 + HELD_OUT_OFFSET, max_steps=2)
    expected = [held.batch(0)[0], held.batch(1)[0][:2]]
    assert len(seen) == 2
    for got, want in zip(seen, expected):
        assert torch.equal(got, want)
    train = ProbeStream(spec, sequences_per_step=4, seed=5, max_steps=3)
    for t in range(3):
        for got in seen:
            assert not torch.equal(got, train.batch(t)[0][: got.shape[0]])


def test_training_data_is_the_same_for_every_init_seed_and_variant(monkeypatch):
    made = []
    real_trainer = tp.Trainer

    class Recording(real_trainer):
        def __init__(self, model, resolved, stream, **kwargs):
            made.append(stream)
            super().__init__(model, resolved, stream, **kwargs)

    monkeypatch.setattr(tp, "Trainer", Recording)
    spec = make_spec("mqar", n_pairs=4)

    def first_training_batches(variant, init_seed, data_seed=0):
        made.clear()
        run_probe(variant, spec, **_tiny_kwargs(init_seed=init_seed, data_seed=data_seed, steps=2))
        stream = made[-1]
        return [stream.batch(i)[0] for i in range(2)]

    base = first_training_batches("mha", 1)
    for other in (first_training_batches("mha", 2), first_training_batches("mqa", 1)):
        assert all(torch.equal(a, b) for a, b in zip(base, other))
    different = first_training_batches("mha", 1, data_seed=1)
    assert not all(torch.equal(a, b) for a, b in zip(base, different))


def test_record_reports_the_trainers_skipped_and_taken_steps(monkeypatch):
    real_trainer = tp.Trainer

    class Stub(real_trainer):
        def train(self, until=None):
            super().train(until)
            self.steps_taken = 2

        @property
        def skipped_steps(self):
            return 1

    monkeypatch.setattr(tp, "Trainer", Stub)
    record = run_probe("mha", make_spec("mqar", n_pairs=4), **_tiny_kwargs(steps=3))
    assert record["steps_taken"] == 2 and record["skipped_steps"] == 1


def test_record_reports_the_trainers_precision(monkeypatch):
    """The trainer's precision, not a constant: the stub claims fp16, and the held-out
    counts are stubbed so that no CUDA autocast is opened on this CPU machine."""
    real_trainer = tp.Trainer

    class Stub(real_trainer):
        def train(self, until=None):
            super().train(until)  # trains in fp32 on this CPU, as the real trainer does
            self.precision = "fp16"  # what the record must report

    def fixed_counts(trainer, spec, **kwargs):
        return {"n_scored": 8, "n_correct_tokens": 5, "n_sequences": 2, "n_correct_sequences": 1}

    monkeypatch.setattr(tp, "Trainer", Stub)
    monkeypatch.setattr(tp, "_held_out_counts", fixed_counts)
    record = run_probe("mha", make_spec("mqar", n_pairs=4), **_tiny_kwargs(steps=2))
    assert record["precision"] == "fp16"
    assert record["accuracy"] == 5 / 8 and record["sequence_accuracy"] == 1 / 2


def test_record_n_layers_is_the_backbone_depth_when_not_given():
    """Without --n-layers the probe keeps the backbone's depth (8), and the record says so."""
    kwargs = _tiny_kwargs(steps=1, sequences_per_step=2, eval_sequences=2)
    del kwargs["n_layers"]
    record = run_probe("mha", make_spec("mqar", n_pairs=4), **kwargs)
    assert record["n_layers"] == 8 and isinstance(record["n_layers"], int)


@pytest.mark.parametrize("mode", ["ref", "opt"])
def test_held_out_counts_use_the_run_mode(monkeypatch, mode):
    modes_seen = []
    real_counts = tp.counts

    def spy(model, x, y, *, mode="opt"):
        modes_seen.append(mode)
        return real_counts(model, x, y, mode=mode)

    monkeypatch.setattr(tp, "counts", spy)
    run_probe("mha", make_spec("mqar", n_pairs=4), **_tiny_kwargs(mode=mode))
    assert modes_seen and set(modes_seen) == {mode}


def test_an_override_reaches_the_resolved_config_and_changes_hash_and_file_name(tmp_path, monkeypatch):
    unfilled_recipe(monkeypatch)  # so that the plain configuration has no init_scale of its own
    spec = make_spec("mqar", n_pairs=4)
    plain = probe_resolved("mha", spec, d_model=64, init_seed=1, steps=3, peak_lr=1e-3,
                           sequences_per_step=4, n_layers=1)
    tuned = probe_resolved("mha", spec, d_model=64, init_seed=1, steps=3, peak_lr=1e-3,
                           sequences_per_step=4, n_layers=1,
                           overrides={"recipe.grad_scaler.init_scale": 1024.0})
    assert tuned["recipe"]["grad_scaler"]["init_scale"] == 1024.0
    assert plain["recipe"]["grad_scaler"]["init_scale"] is None
    assert config.config_hash(tuned) != config.config_hash(plain)
    kwargs = _tiny_kwargs()
    plain_path = probe_path(tmp_path, "mha", spec, **kwargs)
    tuned_path = probe_path(tmp_path, "mha", spec, **kwargs, overrides={"recipe.grad_scaler.init_scale": 1024.0})
    assert plain_path.name != tuned_path.name


@pytest.mark.parametrize(
    "path, value",
    [
        ("model.d_model", 32),
        ("model.n_layers", 2),
        ("model.attention", {"type": "grouped", "n_kv_heads": 4}),
        ("recipe.batch.micro_batch", 2),
        ("recipe.batch.sequences_per_step", 8),
    ],
)
def test_an_override_of_a_key_the_probe_sets_is_refused(path, value):
    spec = make_spec("mqar", n_pairs=4)
    with pytest.raises(ValueError, match="set by the probe"):
        probe_resolved("mha", spec, d_model=64, init_seed=1, steps=3, peak_lr=1e-3,
                       sequences_per_step=4, n_layers=1, overrides={path: value})


def test_cli_override_flag_reaches_the_record_file(tmp_path):
    argv = [
        "--variant", "mha", "--task", "mqar", "--param", "n_pairs=4", "--d-model", "64", "--seed", "1",
        "--steps", "3", "--peak-lr", "0.001", "--sequences-per-step", "4", "--eval-sequences", "6",
        "--n-layers", "1", "--out", str(tmp_path), "--override", "recipe.grad_scaler.init_scale=1024.0",
    ]
    assert main(argv) == 0
    (path,) = tmp_path.glob("mqar-mha-n4-s1-*.json")
    written = json.loads(path.read_text(encoding="utf-8"))
    without = config.config_hash(probe_resolved(
        "mha", make_spec("mqar", n_pairs=4), d_model=64, init_seed=1, steps=3, peak_lr=0.001,
        sequences_per_step=4, n_layers=1,
    ))
    expected = config.config_hash(probe_resolved(
        "mha", make_spec("mqar", n_pairs=4), d_model=64, init_seed=1, steps=3, peak_lr=0.001,
        sequences_per_step=4, n_layers=1, overrides={"recipe.grad_scaler.init_scale": 1024.0},
    ))
    assert written["config_hash"] == expected
    assert written["config_hash"] != without


def test_cli_refuses_a_malformed_override(tmp_path, capsys):
    argv = [
        "--variant", "mha", "--task", "mqar", "--param", "n_pairs=4", "--d-model", "64", "--seed", "1",
        "--steps", "3", "--peak-lr", "0.001", "--out", str(tmp_path), "--override", "no-equals-sign",
    ]
    assert main(argv) == 2
    assert "PATH=VALUE" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []

