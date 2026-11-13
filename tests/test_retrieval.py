"""T-22: retrieval generators and scorer (plan §9.7). Synthetic data only; exact checks only.

The RoPE extension is open (decision 5), so the scorer must refuse anything longer than
the trained context. Test 10 pins that refusal.
"""

from __future__ import annotations

import inspect
import math
import random
import re

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.tokenizer import train_tokenizer
from src.eval import retrieval as R
from src.eval.retrieval import (
    ANSWER,
    FACT,
    FILLER_SENTENCES,
    PAD,
    QUERY,
    ContextTooLong,
    RetrievalExample,
    TokenLayout,
    _text,
    _zero_shot_from_seed,
    grid,
    needle_example,
    needle_text,
    passkey_example,
    passkey_text,
    score,
    token_layout,
    zero_shot_example,
)
from src.model import build_model
from tests.variants import Case, tiny_resolved

LAYOUT = token_layout(1024)  # keys 4..259, values 260..515, filler 516..1023
SMALL = token_layout(128, n_keys=32, n_values=32)  # for the tiny real model (vocab 128)
TINY = Case("mha", 8)
GEN_CASES = [
    (task, length, depth, nd)
    for task in ("passkey", "needle")
    for length in (1024, 2048, 8192)
    for depth in (0.0, 0.5, 1.0)
    for nd in (0, 4, 64)
]
KEY_LEN = {"passkey": 5, "needle": 1}


def _make(task, layout, *, length, depth, nd, rng, value_len=1):
    if task == "passkey":
        return passkey_example(layout, length=length, depth=depth, n_distractors=nd, rng=rng)
    return needle_example(layout, length=length, depth=depth, n_distractors=nd,
                          value_len=value_len, rng=rng)


def _payload_len(task, value_len=1):
    return 5 if task == "passkey" else 1 + value_len


def _oracle(prefix, layout, task, payload_len):
    """Reads only the prefix. Passkey: the one fact whose payload is all values. Needle: the fact
    whose key is the queried key, after `QUERY key ANSWER`."""
    facts = [tuple(prefix[i + 1: i + 1 + payload_len]) for i, t in enumerate(prefix) if t == FACT]
    if task == "passkey":
        assert prefix[-2:] == [QUERY, ANSWER]
        hits = [f for f in facts if all(t in layout.values for t in f)]
        assert len(hits) == 1
        return list(hits[0])
    assert prefix[-3] == QUERY and prefix[-1] == ANSWER
    key = prefix[-2]
    hits = [f[1:] for f in facts if f[0] == key]
    assert len(hits) == 1
    return list(hits[0])



# --- 1-4. Trained generators: format, oracle, position, distractors ---------

@pytest.mark.parametrize("task,length,depth,nd", GEN_CASES)
def test_format_and_specials(task, length, depth, nd):
    ex = _make(task, LAYOUT, length=length, depth=depth, nd=nd, rng=np.random.default_rng(1))
    tokens = list(ex.tokens)
    assert len(tokens) == length == ex.length
    assert ex.answer == ex.tokens[ex.answer_start:]
    assert all(0 <= t < LAYOUT.vocab_size for t in tokens)
    assert PAD not in tokens
    assert tokens.count(QUERY) == 1 and tokens.count(ANSWER) == 1
    assert tokens[ex.answer_start - 1] == ANSWER
    q_index = ex.answer_start - (2 if task == "passkey" else 3)
    assert tokens[q_index] == QUERY
    assert len(ex.answer) == KEY_LEN[task]
    assert (ex.task, ex.mode, ex.depth, ex.n_distractors) == (task, "trained", depth, nd)
    if task == "passkey":
        assert all(t in LAYOUT.values for t in ex.answer)
    else:
        assert tokens[ex.answer_start - 2] in LAYOUT.keys  # the queried key
        assert all(t in LAYOUT.values for t in ex.answer)


@pytest.mark.parametrize("task,length,depth,nd", GEN_CASES)
def test_prefix_oracle_recovers_answer(task, length, depth, nd):
    ex = _make(task, LAYOUT, length=length, depth=depth, nd=nd, rng=np.random.default_rng(2))
    prefix = list(ex.tokens[: ex.answer_start])
    assert _oracle(prefix, LAYOUT, task, _payload_len(task)) == list(ex.answer)


@pytest.mark.parametrize("task,length,depth,nd", GEN_CASES)
def test_target_position_follows_depth(task, length, depth, nd):
    ex = _make(task, LAYOUT, length=length, depth=depth, nd=nd, rng=np.random.default_rng(3))
    prefix = list(ex.tokens[: ex.answer_start])
    if task == "passkey":
        target = next(i for i, t in enumerate(prefix)
                      if t == FACT and all(u in LAYOUT.values for u in prefix[i + 1: i + 6]))
    else:
        key = prefix[-2]
        target = next(i for i, t in enumerate(prefix) if t == FACT and prefix[i + 1] == key)
    n_filler = sum(t in LAYOUT.filler for t in prefix)
    fillers_before = sum(t in LAYOUT.filler for t in prefix[:target])
    # Documented rule: floor(depth * filler + 1/2) filler tokens precede the target fact.
    assert fillers_before == math.floor(depth * n_filler + 0.5)
    if depth == 0.0:
        assert fillers_before == 0
    if depth == 1.0:
        assert fillers_before == n_filler


@pytest.mark.parametrize("task,length,depth,nd", GEN_CASES)
def test_distractors_are_exact_and_harmless(task, length, depth, nd):
    ex = _make(task, LAYOUT, length=length, depth=depth, nd=nd, rng=np.random.default_rng(4))
    prefix = list(ex.tokens[: ex.answer_start])
    assert prefix.count(FACT) == 1 + nd
    payload = _payload_len(task)
    facts = [(i, tuple(prefix[i + 1: i + 1 + payload])) for i, t in enumerate(prefix) if t == FACT]
    if task == "passkey":
        target_idx = [i for i, f in facts if all(t in LAYOUT.values for t in f)]
        assert len(target_idx) == 1
        decoys = [f for i, f in facts if i != target_idx[0]]
        assert len(decoys) == nd
        for d in decoys:
            assert all(t in LAYOUT.keys for t in d)  # never value-range tokens
            assert d != tuple(ex.answer)  # never the passkey
        # The answer occurs as a contiguous run only inside the target fact.
        answer = list(ex.answer)
        runs = [j for j in range(len(prefix) - len(answer) + 1) if prefix[j: j + len(answer)] == answer]
        assert runs == [target_idx[0] + 1]
    else:
        keys = [f[0] for _, f in facts]
        assert len(set(keys)) == 1 + nd
        assert prefix[-2] in keys
        values = [f[1:] for _, f in facts]
        assert all(all(t in LAYOUT.values for t in v) for v in values)


# --- 5. Determinism; no global RNG ------------------------------------------

@pytest.mark.parametrize("task", ["passkey", "needle"])
def test_same_seed_same_example_different_seed_differs(task):
    a = _make(task, LAYOUT, length=1024, depth=0.5, nd=4, rng=np.random.default_rng(11))
    b = _make(task, LAYOUT, length=1024, depth=0.5, nd=4, rng=np.random.default_rng(11))
    c = _make(task, LAYOUT, length=1024, depth=0.5, nd=4, rng=np.random.default_rng(12))
    assert a == b
    assert a.tokens != c.tokens


@pytest.mark.parametrize("task", ["passkey", "needle"])
def test_global_rng_is_neither_read_nor_advanced(task):
    np.random.seed(1)
    a = _make(task, LAYOUT, length=1024, depth=0.25, nd=4, rng=np.random.default_rng(7))
    np.random.seed(2)
    before = np.random.get_state()
    b = _make(task, LAYOUT, length=1024, depth=0.25, nd=4, rng=np.random.default_rng(7))
    after = np.random.get_state()
    assert a == b
    assert before[0] == after[0] and before[2] == after[2] and np.array_equal(before[1], after[1])


def test_needle_value_length_is_respected():
    ex = needle_example(LAYOUT, length=1024, depth=0.5, n_distractors=4, value_len=3,
                        rng=np.random.default_rng(5))
    assert len(ex.answer) == 3
    prefix = list(ex.tokens[: ex.answer_start])
    assert _oracle(prefix, LAYOUT, "needle", 4) == list(ex.answer)


# --- 6. Refusals -------------------------------------------------------------

def test_token_layout_fits_exactly():
    need = 4 + 256 + 256 + 1
    assert token_layout(need).filler == range(need - 1, need)
    with pytest.raises(ValueError):
        token_layout(need - 1)
    with pytest.raises(ValueError):
        token_layout(1024, n_keys=0)
    with pytest.raises(ValueError):
        token_layout(1024, n_values=-3)


def test_passkey_length_must_fit_facts_and_query():
    nd = 4
    fixed = (1 + nd) * 6 + 2 + 5  # 37 tokens with no filler
    ex = passkey_example(LAYOUT, length=fixed, depth=0.5, n_distractors=nd, rng=np.random.default_rng(6))
    assert len(ex.tokens) == fixed
    with pytest.raises(ValueError, match="too short"):
        passkey_example(LAYOUT, length=fixed - 1, depth=0.5, n_distractors=nd, rng=np.random.default_rng(6))


def test_needle_length_must_fit_facts_and_query():
    nd = 4
    fixed = (1 + nd) * 3 + 3 + 1  # 19 tokens with no filler
    ex = needle_example(LAYOUT, length=fixed, depth=1.0, n_distractors=nd, rng=np.random.default_rng(6))
    assert len(ex.tokens) == fixed
    with pytest.raises(ValueError, match="too short"):
        needle_example(LAYOUT, length=fixed - 1, depth=1.0, n_distractors=nd, rng=np.random.default_rng(6))


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), True, "0.5"])
def test_depth_outside_unit_interval_is_refused(bad):
    with pytest.raises(ValueError):
        passkey_example(LAYOUT, length=1024, depth=bad, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        needle_example(LAYOUT, length=1024, depth=bad, rng=np.random.default_rng(0))


def test_negative_distractors_are_refused():
    with pytest.raises(ValueError):
        passkey_example(LAYOUT, length=1024, depth=0.5, n_distractors=-1, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        needle_example(LAYOUT, length=1024, depth=0.5, n_distractors=-1, rng=np.random.default_rng(0))


def test_needle_needs_one_distinct_key_per_fact():
    small_keys = token_layout(1024, n_keys=4)
    needle_example(small_keys, length=1024, depth=0.5, n_distractors=3, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="distinct keys"):
        needle_example(small_keys, length=1024, depth=0.5, n_distractors=4, rng=np.random.default_rng(0))


# --- 7. Zero-shot ------------------------------------------------------------

def _word_encoder():
    vocab: dict[str, int] = {}

    def encode(text: str) -> list[int]:
        return [vocab.setdefault(w, len(vocab)) for w in text.split()]

    return encode


@pytest.fixture(scope="module")
def real_encode():
    """The project's byte-level BPE, trained on a tiny synthetic corpus as in tests/test_data.py."""
    rng = np.random.default_rng(22)
    words = ["alpha", "beta", "gamma", "delta", "token", "model", "data", "x", "yy", "café", "日本語"]
    texts = [" ".join(words[i] for i in rng.integers(0, len(words), size=30)) for _ in range(300)]
    tok = train_tokenizer(texts + list(FILLER_SENTENCES) * 20, 300)
    return lambda text: tok.encode(text, add_special_tokens=False).ids


ZS_CASES = [
    (task, length, depth, nd, enc_name)
    for task in ("passkey", "needle")
    for length in (1024, 8192)
    for depth in (0.0, 0.5, 1.0)
    for nd in (0, 4)
    for enc_name in ("stub", "real")
]


@pytest.mark.parametrize("task,length,depth,nd,enc_name", ZS_CASES)
def test_zero_shot_fills_length_and_text_is_right(task, length, depth, nd, enc_name, real_encode):
    encode = _word_encoder() if enc_name == "stub" else real_encode
    seed = 99
    ex, n = _zero_shot_from_seed(task, encode, length=length, depth=depth, n_distractors=nd, seed=seed)
    assert len(ex.tokens) <= length and ex.length == len(ex.tokens)

    # Adding one more filler sentence would exceed the length.
    nxt_prompt, nxt_answer = _text(task, depth=depth, n_filler_sentences=n + 1, n_distractors=nd,
                                   rng=np.random.default_rng(seed))
    assert len(encode(nxt_prompt)) + len(encode(nxt_answer)) > length

    prompt, answer = _text(task, depth=depth, n_filler_sentences=n, n_distractors=nd,
                           rng=np.random.default_rng(seed))
    assert list(ex.tokens[: ex.answer_start]) == list(encode(prompt))
    assert list(ex.answer) == list(encode(answer))
    assert ex.mode == "zero_shot"

    if task == "passkey":
        code = answer.strip()
        assert re.fullmatch(r"\d{5}", code) and answer == " " + code
        fact = f"The pass key is {code}. Remember it. {code} is the pass key."
        assert prompt.count(fact) == 1
        assert prompt.endswith("What is the pass key? The pass key is")
        door = re.findall(r"The door code is (\d{5})\.", prompt)
        assert len(door) == nd and code not in door
    else:
        word = answer.strip()
        assert answer == " " + word and word.isalpha()
        city = re.search(r"What is the secret word for (.+?)\? ", prompt).group(1)
        assert prompt.endswith(f"What is the secret word for {city}? The secret word for {city} is")
        facts = re.findall(r"The secret word for (.+?) is ([a-z]+)\.", prompt)
        assert [w for c, w in facts if c == city] == [word]
        others = [w for c, w in facts if c != city]
        assert len(others) == nd and word not in others


def test_zero_shot_same_seed_same_example(real_encode):
    a = zero_shot_example("needle", real_encode, length=1024, depth=0.5, n_distractors=4,
                          rng=np.random.default_rng(3))
    b = zero_shot_example("needle", real_encode, length=1024, depth=0.5, n_distractors=4,
                          rng=np.random.default_rng(3))
    assert a == b and a.length <= 1024
    assert passkey_text(depth=0.5, n_filler_sentences=10, rng=np.random.default_rng(4)) == \
        passkey_text(depth=0.5, n_filler_sentences=10, rng=np.random.default_rng(4))
    assert needle_text(depth=0.5, n_filler_sentences=10, rng=np.random.default_rng(4)) == \
        needle_text(depth=0.5, n_filler_sentences=10, rng=np.random.default_rng(4))


def test_zero_shot_refuses_length_too_short_for_prompt(real_encode):
    with pytest.raises(ValueError):
        zero_shot_example("passkey", real_encode, length=5, depth=0.5, rng=np.random.default_rng(0))


# --- 8. Grid -----------------------------------------------------------------

def test_grid_order_and_size():
    g = grid()
    assert len(g) == 4 * 5 * 4 == 80
    assert g[0] == {"length": 1024, "depth": 0.0, "n_distractors": 0}
    assert g[1] == {"length": 1024, "depth": 0.0, "n_distractors": 4}
    assert g[4] == {"length": 1024, "depth": 0.25, "n_distractors": 0}
    assert g[-1] == {"length": 8192, "depth": 1.0, "n_distractors": 64}
    assert len({tuple(sorted(d.items())) for d in g}) == 80


def test_grid_custom_arguments():
    assert grid(lengths=(16,), depths=(0.5,), distractors=(1, 2)) == [
        {"length": 16, "depth": 0.5, "n_distractors": 1},
        {"length": 16, "depth": 0.5, "n_distractors": 2},
    ]


# --- 9. Scorer with a stub model ---------------------------------------------

VOCAB_STUB = 128


class _Stub(nn.Module):
    """Logits whose argmax at each position is looked up from the input row."""

    def __init__(self, table):
        super().__init__()
        self.table = table
        self.drop = nn.Dropout(0.5)  # a submodule, so the mode restore is checked
        self.w = nn.Parameter(torch.zeros(1))
        self.grad_flags: list[bool] = []
        self.calls = 0

    def forward(self, idx, *, mode="opt"):
        self.calls += 1
        self.grad_flags.append(torch.is_grad_enabled())
        b_size, length = idx.shape
        logits = torch.zeros(b_size, length, VOCAB_STUB)
        for b in range(b_size):
            for t, p in enumerate(self.table[tuple(idx[b].tolist())]):
                logits[b, t, p] = 1.0
        return logits + self.w * 0.0, None


def _ex(tokens, answer_start, task="passkey"):
    return RetrievalExample(tokens=tuple(tokens), answer_start=answer_start,
                            answer=tuple(tokens[answer_start:]), task=task, mode="trained",
                            length=len(tokens), depth=0.0, n_distractors=0)


def _table_entry(ex, correct):
    """Prediction per position: the answer token where `correct`, a wrong id otherwise."""
    pred = [0] * len(ex.tokens)
    for i, t in enumerate(ex.answer):
        pred[ex.answer_start - 1 + i] = t if correct[i] else (t + 1) % VOCAB_STUB
    return tuple(ex.tokens), pred


def test_scorer_hand_computed_values_mixed_lengths():
    a = _ex([5, 6, 7, 8, 9, 10], 4)  # answer [9, 10]
    b = _ex([11, 12, 13, 14, 15], 3)  # answer [14, 15]
    c = _ex([20, 21, 22, 23], 3)  # answer [23]
    table = dict([_table_entry(a, [True, False]), _table_entry(b, [True, True]),
                  _table_entry(c, [False])])
    model = _Stub(table)
    out = score(model, [a, b, c], trained_context=8)
    assert out["n"] == 3
    assert out["per_example"] == [False, True, False]
    assert out["exact_match"] == 1 / 3
    assert out["token_accuracy"] == 3 / 5
    assert all(isinstance(v, bool) for v in out["per_example"])


def test_scorer_restores_mode_and_builds_no_graph():
    a = _ex([5, 6, 7, 8, 9, 10], 4)
    model = _Stub(dict([_table_entry(a, [True, True])]))
    model.train()
    flags = [m.training for m in model.modules()]
    score(model, [a], trained_context=8, batch_size=1)
    assert [m.training for m in model.modules()] == flags
    assert model.training is True
    model.eval()
    score(model, [a], trained_context=8)
    assert model.training is False
    assert model.grad_flags and not any(model.grad_flags)


def test_scorer_refuses_bad_arguments():
    a = _ex([5, 6, 7, 8], 2)
    model = _Stub(dict([_table_entry(a, [True, True])]))
    with pytest.raises(ValueError):
        score(model, [], trained_context=8)
    with pytest.raises(ValueError):
        score(model, [a], trained_context=8, mode="bogus")
    with pytest.raises(ValueError):
        score(model, [a], trained_context=8, batch_size=0)


# --- 10. The refusal: the RoPE extension stays open ---------------------------

def test_refusal_signature_has_no_bypass():
    params = inspect.signature(score).parameters
    assert [(name, p.kind) for name, p in params.items()] == [
        ("model", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("examples", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("trained_context", inspect.Parameter.KEYWORD_ONLY),
        ("mode", inspect.Parameter.KEYWORD_ONLY),
        ("batch_size", inspect.Parameter.KEYWORD_ONLY),
    ]


def test_refusal_beyond_trained_context_names_open_decision():
    ex = passkey_example(SMALL, length=2048, depth=0.5, n_distractors=4, rng=np.random.default_rng(8))
    model = _Stub({})
    with pytest.raises(ContextTooLong, match="open decision 5"):
        score(model, [ex], trained_context=1024)
    assert model.calls == 0  # refused before any forward pass


def test_length_2048_runs_with_a_model_trained_at_2048():
    resolved = tiny_resolved(TINY, context_length=2048)
    model = build_model(resolved)
    ex = passkey_example(SMALL, length=2048, depth=0.5, n_distractors=4, rng=np.random.default_rng(8))
    out = score(model, [ex], trained_context=resolved["model"]["context_length"])
    assert out["n"] == 1 and len(out["per_example"]) == 1


# --- 11. End to end on a tiny real model -------------------------------------

def test_end_to_end_tiny_model_returns_fractions():
    resolved = tiny_resolved(TINY, context_length=1024)
    model = build_model(resolved)
    trained_context = resolved["model"]["context_length"]
    examples = [
        _make("passkey", SMALL, length=1024, depth=d, nd=nd, rng=np.random.default_rng(20 + i))
        for i, (d, nd) in enumerate([(0.0, 0), (0.5, 4), (1.0, 4)])
    ]
    examples += [
        _make("needle", SMALL, length=1024, depth=d, nd=nd, rng=np.random.default_rng(30 + i))
        for i, (d, nd) in enumerate([(0.0, 0), (0.5, 4), (1.0, 4)])
    ]
    out = score(model, examples, trained_context=trained_context)
    assert out["n"] == len(examples) == len(out["per_example"])
    assert 0.0 <= out["exact_match"] <= 1.0
    assert 0.0 <= out["token_accuracy"] <= 1.0
    assert all(p.grad is None for p in model.parameters())
    single = score(model, examples, trained_context=trained_context, batch_size=1)
    assert single["per_example"] == out["per_example"]


class _RecordingStub(_Stub):
    """Also records how each forward was called."""

    def __init__(self, table):
        super().__init__(table)
        self.seen_training: list[bool] = []
        self.seen_mode: list[str] = []

    def forward(self, idx, *, mode="opt"):
        self.seen_training.append(self.training)
        self.seen_mode.append(mode)
        return super().forward(idx, mode=mode)


@pytest.mark.parametrize("start_training", [True, False])
@pytest.mark.parametrize("mode", ["ref", "opt"])
def test_scorer_runs_in_eval_mode_and_in_the_mode_asked_for(start_training, mode):
    """Added in review. `ref` and `opt` agree numerically, so only the call shows which ran."""
    a = _ex([5, 6, 7, 8, 9, 10], 4)
    model = _RecordingStub(dict([_table_entry(a, [True, True])]))
    model.train(start_training)
    out = score(model, [a], trained_context=8, mode=mode)
    assert out["exact_match"] == 1.0
    assert model.seen_training == [False] and model.seen_mode == [mode]
    assert model.training is start_training and model.drop.training is start_training
