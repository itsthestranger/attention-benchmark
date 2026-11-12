"""Long-context retrieval: task generators and scorer (plan §9.7). T-22.

Two tasks, "passkey" and "needle", in two modes:

- trained: token-level sequences over a synthetic `TokenLayout`, for models trained on them;
- zero_shot: text prompts, encoded by a caller-supplied `encode`, for LM-trained models.

Each example is one sequence with its answer at the end. The model is scored on
predicting every answer token from everything before it (teacher forcing).

`score` refuses any example longer than `trained_context`, the context the model was
trained at. Evaluating beyond it needs a RoPE extension method or a model trained at
that context. That choice is open decision 5 (plan §9.7) and is not made here, so there
is no flag to bypass the check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.attention.base import MODES

TASKS = ("passkey", "needle")
RETRIEVAL_MODES = ("trained", "zero_shot")

LENGTHS = (1024, 2048, 4096, 8192)  # plan §9.7: 1k -> 8k
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DISTRACTORS = (0, 4, 16, 64)

# Special token ids of the trained layout. Filler, keys and values follow them.
PAD, FACT, QUERY, ANSWER = 0, 1, 2, 3
N_SPECIAL = 4

PASSKEY_INSTRUCTION = (
    "There is a pass key hidden in the text below. Memorize it. "
    "I will ask you for it at the end."
)
NEEDLE_INSTRUCTION = (
    "Below is a long text with many facts. One of them is the secret word for a city. "
    "I will ask you for it at the end."
)
FILLER_SENTENCES = (
    "The grass is green.",
    "The sky is blue.",
    "The sun is yellow.",
    "Here we go.",
    "There and back again.",
)
# 79 distinct city names, no name a substring of another. Needle distractors need up to 64.
CITIES = (
    "Amsterdam", "Athens", "Austin", "Bangkok", "Barcelona", "Beijing", "Berlin", "Bogota",
    "Boston", "Brisbane", "Brussels", "Budapest", "Buenos Aires", "Cairo", "Calgary",
    "Canberra", "Caracas", "Chicago", "Copenhagen", "Dakar", "Delhi", "Denver", "Dublin",
    "Dubai", "Edinburgh", "Geneva", "Glasgow", "Hanoi", "Havana", "Helsinki", "Houston",
    "Istanbul", "Jakarta", "Kampala", "Karachi", "Kyoto", "Lagos", "Lima", "Lisbon", "Madrid",
    "Manila", "Melbourne", "Mexico City", "Miami", "Milan", "Montreal", "Moscow", "Mumbai",
    "Nairobi", "Naples", "Oslo", "Ottawa", "Oxford", "Panama", "Perth", "Prague", "Quito",
    "Reykjavik", "Riga", "Santiago", "Seattle", "Seoul", "Shanghai", "Singapore", "Sofia",
    "Stockholm", "Sydney", "Taipei", "Tallinn", "Tehran", "Tokyo", "Toronto", "Tunis",
    "Valencia", "Vienna", "Warsaw", "Yangon", "Zagreb", "Zurich",
)


@dataclass(frozen=True)
class TokenLayout:
    """Disjoint id ranges of a trained-mode vocabulary. Specials are the module constants."""

    vocab_size: int
    keys: range
    values: range
    filler: range


@dataclass(frozen=True)
class RetrievalExample:
    tokens: tuple[int, ...]  # the whole sequence, answer included at the end
    answer_start: int  # index of the first answer token
    answer: tuple[int, ...]  # tokens[answer_start:]
    task: str  # "passkey" | "needle"
    mode: str  # "trained" | "zero_shot"
    length: int  # len(tokens)
    depth: float  # where the target fact sits in the filler, 0.0 = start, 1.0 = end
    n_distractors: int


class ContextTooLong(RuntimeError):
    """An example is longer than the context the model was trained at."""


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _check_depth(depth) -> float:
    if isinstance(depth, bool) or not isinstance(depth, (int, float)) or not 0.0 <= depth <= 1.0:
        raise ValueError(f"depth must be a number in [0, 1], got {depth!r}")
    return float(depth)


def _check_layout(layout) -> None:
    if not isinstance(layout, TokenLayout):
        raise ValueError(f"layout must be a TokenLayout, got {type(layout).__name__}")


def _gap(fraction: float, n_filler: int) -> int:
    """Number of filler tokens (or sentences) before an item placed at `fraction`.

    floor(fraction * n + 1/2): 0.0 puts the item first, 1.0 puts it after all filler.
    """
    return int(math.floor(fraction * n_filler + 0.5))


def _assemble(filler: list, placed: list[tuple[int, list]]) -> list:
    """Insert each item's pieces before filler index `gap`. Items with equal gaps keep order."""
    out: list = []
    pos = 0
    for gap, piece in sorted(placed, key=lambda p: p[0]):
        out.extend(filler[pos:gap])
        pos = gap
        out.extend(piece)
    out.extend(filler[pos:])
    return out


def _example(tokens: list[int], answer_start: int, *, task: str, mode: str, depth: float,
             n_distractors: int) -> RetrievalExample:
    return RetrievalExample(
        tokens=tuple(tokens),
        answer_start=answer_start,
        answer=tuple(tokens[answer_start:]),
        task=task,
        mode=mode,
        length=len(tokens),
        depth=float(depth),
        n_distractors=n_distractors,
    )


def token_layout(vocab_size: int, *, n_keys: int = 256, n_values: int = 256) -> TokenLayout:
    """Specials, then keys, then values, then filler (the rest of the vocabulary)."""
    _check_int("vocab_size", vocab_size, 1)
    _check_int("n_keys", n_keys, 1)
    _check_int("n_values", n_values, 1)
    need = N_SPECIAL + n_keys + n_values + 1
    if vocab_size < need:
        raise ValueError(
            f"vocab_size {vocab_size} is too small: 4 specials, {n_keys} keys, {n_values} values "
            f"and at least one filler token need {need}"
        )
    keys_lo = N_SPECIAL
    values_lo = keys_lo + n_keys
    filler_lo = values_lo + n_values
    return TokenLayout(
        vocab_size=vocab_size,
        keys=range(keys_lo, values_lo),
        values=range(values_lo, filler_lo),
        filler=range(filler_lo, vocab_size),
    )


def passkey_example(layout: TokenLayout, *, length: int, depth: float, n_distractors: int = 0,
                    key_len: int = 5, rng: np.random.Generator) -> RetrievalExample:
    """Filler with one `FACT passkey` at `depth`, decoys `FACT key-range tokens`, then `QUERY ANSWER passkey`.

    The filler count is whatever makes the sequence exactly `length` tokens long. Draws come
    from `rng` only, in a fixed order.
    """
    _check_layout(layout)
    depth = _check_depth(depth)
    _check_int("length", length, 1)
    _check_int("n_distractors", n_distractors, 0)
    _check_int("key_len", key_len, 1)
    fixed = (1 + n_distractors) * (1 + key_len) + 2 + key_len
    if length < fixed:
        raise ValueError(
            f"length {length} is too short: {1 + n_distractors} facts and the query need at least {fixed}"
        )
    n_filler = length - fixed

    passkey = rng.integers(layout.values.start, layout.values.stop, size=key_len).tolist()
    decoys = [
        [FACT] + rng.integers(layout.keys.start, layout.keys.stop, size=key_len).tolist()
        for _ in range(n_distractors)
    ]
    filler = rng.integers(layout.filler.start, layout.filler.stop, size=n_filler).tolist()
    fractions = rng.random(n_distractors).tolist()
    placed = [(_gap(depth, n_filler), [FACT] + passkey)]
    placed += [(_gap(f, n_filler), d) for f, d in zip(fractions, decoys)]
    tokens = _assemble(filler, placed) + [QUERY, ANSWER] + passkey
    return _example(tokens, len(tokens) - key_len, task="passkey", mode="trained", depth=depth,
                    n_distractors=n_distractors)


def needle_example(layout: TokenLayout, *, length: int, depth: float, n_distractors: int = 0,
                   value_len: int = 1, rng: np.random.Generator) -> RetrievalExample:
    """Facts `FACT key value...` with distinct keys; the query repeats the target's key.

    The sequence ends `QUERY key ANSWER value...`; the answer is the target's value.
    """
    _check_layout(layout)
    depth = _check_depth(depth)
    _check_int("length", length, 1)
    _check_int("n_distractors", n_distractors, 0)
    _check_int("value_len", value_len, 1)
    n_facts = 1 + n_distractors
    if n_facts > len(layout.keys):
        raise ValueError(f"{n_facts} distinct keys needed, layout has {len(layout.keys)}")
    fact_len = 2 + value_len
    fixed = n_facts * fact_len + 3 + value_len
    if length < fixed:
        raise ValueError(f"length {length} is too short: {n_facts} facts and the query need at least {fixed}")
    n_filler = length - fixed

    key_index = rng.choice(len(layout.keys), size=n_facts, replace=False).tolist()
    keys = [layout.keys.start + k for k in key_index]
    values = rng.integers(layout.values.start, layout.values.stop, size=(n_facts, value_len)).tolist()
    facts = [[FACT, keys[i]] + values[i] for i in range(n_facts)]
    filler = rng.integers(layout.filler.start, layout.filler.stop, size=n_filler).tolist()
    fractions = rng.random(n_distractors).tolist()
    placed = [(_gap(depth, n_filler), facts[0])]
    placed += [(_gap(f, n_filler), facts[i + 1]) for i, f in enumerate(fractions)]
    tokens = _assemble(filler, placed) + [QUERY, keys[0], ANSWER] + values[0]
    return _example(tokens, len(tokens) - value_len, task="needle", mode="trained", depth=depth,
                    n_distractors=n_distractors)


def _distinct_words(rng: np.random.Generator, k: int) -> list[str]:
    words: list[str] = []
    while len(words) < k:
        word = "".join(chr(c) for c in rng.integers(ord("a"), ord("z") + 1, size=6))
        if word not in words:
            words.append(word)
    return words


def _text(task: str, *, depth: float, n_filler_sentences: int, n_distractors: int,
          rng: np.random.Generator) -> tuple[str, str]:
    """(prompt, answer) for one candidate filler count. Content is drawn before positions.

    Every draw depends only on `rng`, not on `n_filler_sentences`, so the same seed gives the
    same fact, distractors and fractional positions at every filler count.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    depth = _check_depth(depth)
    _check_int("n_filler_sentences", n_filler_sentences, 0)
    _check_int("n_distractors", n_distractors, 0)
    k = 1 + n_distractors

    if task == "passkey":
        codes = [f"{c:05d}" for c in rng.choice(100000, size=k, replace=False).tolist()]
        target_line = f"The pass key is {codes[0]}. Remember it. {codes[0]} is the pass key."
        other_lines = [f"The door code is {c}." for c in codes[1:]]
        question = "What is the pass key? The pass key is"
        answer = " " + codes[0]
        instruction = PASSKEY_INSTRUCTION
    else:
        cities = [CITIES[i] for i in rng.choice(len(CITIES), size=k, replace=False).tolist()]
        words = _distinct_words(rng, k)
        target_line = f"The secret word for {cities[0]} is {words[0]}."
        other_lines = [f"The secret word for {c} is {w}." for c, w in zip(cities[1:], words[1:])]
        question = f"What is the secret word for {cities[0]}? The secret word for {cities[0]} is"
        answer = " " + words[0]
        instruction = NEEDLE_INSTRUCTION

    fractions = rng.random(n_distractors).tolist()
    filler = [FILLER_SENTENCES[i % len(FILLER_SENTENCES)] for i in range(n_filler_sentences)]
    placed = [(_gap(depth, n_filler_sentences), [target_line])]
    placed += [(_gap(f, n_filler_sentences), [line]) for f, line in zip(fractions, other_lines)]
    prompt = " ".join([instruction, *_assemble(filler, placed), question])
    return prompt, answer


def passkey_text(*, depth: float, n_filler_sentences: int, n_distractors: int = 0,
                 rng: np.random.Generator) -> tuple[str, str]:
    """(prompt, answer) of the classic passkey prompt. The answer has a leading space."""
    return _text("passkey", depth=depth, n_filler_sentences=n_filler_sentences,
                 n_distractors=n_distractors, rng=rng)


def needle_text(*, depth: float, n_filler_sentences: int, n_distractors: int = 0,
                rng: np.random.Generator) -> tuple[str, str]:
    """(prompt, answer) of the secret-word prompt. The answer has a leading space."""
    return _text("needle", depth=depth, n_filler_sentences=n_filler_sentences,
                 n_distractors=n_distractors, rng=rng)


def _zero_shot_from_seed(task: str, encode, *, length: int, depth: float, n_distractors: int,
                         seed: int) -> tuple[RetrievalExample, int]:
    """Longest filler that keeps the encoded example within `length`, and that count.

    Every candidate replays the same seed. Binary search keeps the invariant that `lo`
    fits and `hi` does not, so the result satisfies: `lo` fits and `lo + 1` does not.
    """

    def build(n: int) -> tuple[list[int], list[int]]:
        prompt, answer = _text(task, depth=depth, n_filler_sentences=n, n_distractors=n_distractors,
                               rng=np.random.default_rng(seed))
        return list(encode(prompt)), list(encode(answer))

    def fits(n: int) -> bool:
        prompt_ids, answer_ids = build(n)
        return len(prompt_ids) + len(answer_ids) <= length

    if not fits(0):
        raise ValueError(f"length {length} is too short for the {task} prompt even with no filler")
    lo, hi = 0, 1
    while fits(hi):
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    prompt_ids, answer_ids = build(lo)
    example = _example(prompt_ids + answer_ids, len(prompt_ids), task=task, mode="zero_shot",
                       depth=depth, n_distractors=n_distractors)
    return example, lo


def zero_shot_example(task: str, encode, *, length: int, depth: float, n_distractors: int = 0,
                      rng: np.random.Generator) -> RetrievalExample:
    """Zero-shot example of at most `length` tokens, with as much filler as fits.

    `encode(text) -> list[int]` is supplied by the caller. Consumes exactly one draw from `rng`.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    _check_int("length", length, 1)
    _check_depth(depth)
    _check_int("n_distractors", n_distractors, 0)
    seed = int(rng.integers(0, 2**63 - 1))
    example, _ = _zero_shot_from_seed(task, encode, length=length, depth=depth,
                                      n_distractors=n_distractors, seed=seed)
    return example


def grid(lengths=LENGTHS, depths=DEPTHS, distractors=DISTRACTORS) -> list[dict]:
    """Every (length, depth, distractors) combination, lengths outermost."""
    return [
        {"length": length, "depth": depth, "n_distractors": n}
        for length in lengths
        for depth in depths
        for n in distractors
    ]


def _device(model: nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def score(model, examples: list[RetrievalExample], *, trained_context: int, mode: str = "opt",
          batch_size: int = 8) -> dict:
    """Exact match and token accuracy of `model` on `examples`.

    Raises ContextTooLong before running anything if an example is longer than
    `trained_context`. The model is put in eval mode under no_grad and its mode is restored.
    """
    _check_int("trained_context", trained_context, 1)
    _check_int("batch_size", batch_size, 1)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not examples:
        raise ValueError("examples is empty")
    for i, ex in enumerate(examples):
        if not 1 <= ex.answer_start < len(ex.tokens) or ex.answer != ex.tokens[ex.answer_start:]:
            raise ValueError(f"example {i} is inconsistent: answer must be tokens[answer_start:]")

    too_long = [len(ex.tokens) for ex in examples if len(ex.tokens) > trained_context]
    if too_long:
        raise ContextTooLong(
            f"example of length {max(too_long)} exceeds the trained context {trained_context}. "
            "Evaluating beyond the trained context needs a RoPE extension method or a model "
            "trained at that context; that choice is open decision 5 (plan §9.7) and has not "
            "been made. There is no flag to bypass this check."
        )

    # Batches hold examples of equal length only.
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].tokens))
    batches: list[list[int]] = []
    for i in order:
        if batches and len(examples[batches[-1][0]].tokens) == len(examples[i].tokens) \
                and len(batches[-1]) < batch_size:
            batches[-1].append(i)
        else:
            batches.append([i])

    results: list[list[bool]] = [[] for _ in examples]
    dev = _device(model)
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                idx = torch.tensor([examples[i].tokens for i in batch], dtype=torch.long, device=dev)
                logits, _ = model(idx, mode=mode)
                for row, i in enumerate(batch):
                    ex = examples[i]
                    # Position t predicts token t + 1: the answer is predicted from answer_start - 1 on.
                    pred = logits[row, ex.answer_start - 1: len(ex.tokens) - 1].argmax(dim=-1).tolist()
                    results[i] = [p == t for p, t in zip(pred, ex.answer)]
    finally:
        for m, flag in modes:
            m.training = flag

    per_example = [all(r) for r in results]
    token_hits = sum(sum(r) for r in results)
    token_total = sum(len(r) for r in results)
    n = len(examples)
    return {
        "n": n,
        "exact_match": sum(per_example) / n,
        "token_accuracy": token_hits / token_total,
        "per_example": per_example,
    }
