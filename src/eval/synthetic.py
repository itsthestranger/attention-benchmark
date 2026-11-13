"""Synthetic memory probes (plan §9.2): generators, the oracle, the batch stream, and accuracy.

Token layout shared by every probe: ids 0 .. N_SPECIAL-1 are special tokens; the
rest of the vocabulary is split into disjoint ranges per role (keys, values,
symbols, noise), in the order a task needs them.

Every generator returns (x, y), both int64 of shape (batch, T). y[b, t] is the
token the model must predict after reading x[b, :t+1], or IGNORE where nothing
is scored.

Recall tasks (mqar, multihop, fuzzy) never feed an answer back as an input. In
their query section the slot that holds the answer holds BLANK instead, and the
answer appears only in y. So for these tasks the input after every scored
position is BLANK, and no answer token appears in the input after SEP.

Teacher forcing: y[b, t] == x[b, t+1] at every scored t < T-1, for selective_copy
and induction only.

`solve` is the oracle. It reads only the input tokens of one sequence, parses
the sequence by the task's rules, and returns the scored positions with their
answers. It never looks at targets. Tests check that the two agree.

Values of mqar and fuzzy are drawn without replacement within a sequence, so a
sequence has n_pairs distinct values and n_values >= n_pairs is required.

A generator is a pure function of its numpy Generator. Nothing here reads or
seeds the global RNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

IGNORE = -100

PAD = 0
SEP = 1
COPY = 2
BLANK = 3
N_SPECIAL = 4

TASKS = ("mqar", "multihop", "fuzzy", "selective_copy", "induction")

_REQUIRED = object()


@dataclass(frozen=True)
class ProbeSpec:
    task: str
    seq_len: int
    vocab_size: int
    params: tuple  # sorted tuple of (name, value) pairs; hashable

    def param(self, name: str):
        return dict(self.params)[name]

    def as_dict(self) -> dict:
        return dict(self.params)


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value!r}")
    return value


# name -> default; a callable default receives the parameters resolved so far.
_DEFAULTS: dict[str, dict] = {
    "mqar": {
        "n_pairs": _REQUIRED,
        "n_keys": lambda p: max(64, 2 * p["n_pairs"]),
        "n_values": lambda p: max(64, 2 * p["n_pairs"]),
    },
    "multihop": {
        "n_chains": _REQUIRED,
        "n_symbols": lambda p: max(64, 3 * p["n_chains"]),
    },
    "fuzzy": {
        "n_pairs": _REQUIRED,
        "key_len": 3,
        "n_noise": 1,
        "n_keys": lambda p: max(64, 2 * p["n_pairs"]),
        "n_values": lambda p: max(64, 2 * p["n_pairs"]),
        "n_noise_tokens": 16,
    },
    "selective_copy": {
        "n_data": _REQUIRED,
        "n_blank": _REQUIRED,
        "n_symbols": 16,
    },
    "induction": {
        "length": _REQUIRED,
        "n_symbols": lambda p: max(64, p["length"]),
    },
}


def _resolve_params(task: str, params: Mapping) -> dict:
    if task not in _DEFAULTS:
        raise ValueError(f"unknown task {task!r}; known: {list(TASKS)}")
    schema = _DEFAULTS[task]
    unknown = sorted(set(params) - set(schema))
    if unknown:
        raise ValueError(f"unknown parameters for {task!r}: {unknown}; known: {sorted(schema)}")

    resolved: dict = {}
    for name, default in schema.items():
        if name in params:
            resolved[name] = params[name]
        elif default is _REQUIRED:
            raise ValueError(f"{task!r} needs the parameter {name!r}")
    for name, default in schema.items():
        if name in resolved:
            continue
        resolved[name] = default(resolved) if callable(default) else default

    minimums = {
        "mqar": {"n_pairs": 1, "n_keys": 1, "n_values": 1},
        "multihop": {"n_chains": 1, "n_symbols": 1},
        "fuzzy": {"n_pairs": 1, "key_len": 1, "n_noise": 0, "n_keys": 1, "n_values": 1, "n_noise_tokens": 1},
        "selective_copy": {"n_data": 1, "n_blank": 0, "n_symbols": 1},
        "induction": {"length": 1, "n_symbols": 1},
    }[task]
    for name, value in list(resolved.items()):
        resolved[name] = _check_int(name, value, minimums[name])

    if task in ("mqar", "fuzzy") and resolved["n_keys"] < resolved["n_pairs"]:
        raise ValueError(
            f"{task}: {resolved['n_pairs']} distinct keys do not fit in a key range of {resolved['n_keys']}"
        )
    if task in ("mqar", "fuzzy") and resolved["n_values"] < resolved["n_pairs"]:
        raise ValueError(
            f"{task}: {resolved['n_pairs']} distinct values do not fit in a value range of {resolved['n_values']}"
        )
    if task == "fuzzy" and resolved["n_noise"] >= resolved["key_len"]:
        raise ValueError(f"fuzzy: n_noise ({resolved['n_noise']}) must be less than key_len ({resolved['key_len']})")
    if task == "multihop" and resolved["n_symbols"] < 3 * resolved["n_chains"]:
        raise ValueError(
            f"multihop: {3 * resolved['n_chains']} distinct symbols needed, n_symbols is {resolved['n_symbols']}"
        )
    if task == "induction" and resolved["n_symbols"] < resolved["length"]:
        raise ValueError(
            f"induction: {resolved['length']} distinct symbols needed, n_symbols is {resolved['n_symbols']}"
        )
    return resolved


def _shape(task: str, p: dict) -> tuple[int, int]:
    """(seq_len, vocab_size) of a task with resolved parameters p."""
    if task == "mqar":
        n = p["n_pairs"]
        return 4 * n + 1, N_SPECIAL + p["n_keys"] + p["n_values"]
    if task == "multihop":
        return 6 * p["n_chains"] + 1, N_SPECIAL + p["n_symbols"]
    if task == "fuzzy":
        n, k = p["n_pairs"], p["key_len"]
        return 2 * n * (k + 1) + 1, N_SPECIAL + p["n_keys"] + p["n_values"] + p["n_noise_tokens"]
    if task == "selective_copy":
        return 2 * p["n_data"] + p["n_blank"] + 1, N_SPECIAL + p["n_symbols"]
    if task == "induction":
        return 2 * p["length"] + 1, N_SPECIAL + p["n_symbols"]
    raise ValueError(f"unknown task {task!r}")


def make_spec(task: str, **params) -> ProbeSpec:
    """Validate parameters, fill defaults, and compute seq_len and vocab_size."""
    resolved = _resolve_params(task, params)
    seq_len, vocab_size = _shape(task, resolved)
    return ProbeSpec(
        task=task,
        seq_len=seq_len,
        vocab_size=vocab_size,
        params=tuple(sorted(resolved.items())),
    )


def difficulty(spec: ProbeSpec) -> int:
    """The size of the task: the number of items a model must keep apart."""
    p = spec.as_dict()
    if spec.task in ("mqar", "fuzzy"):
        return p["n_pairs"]
    if spec.task == "multihop":
        return p["n_chains"]
    if spec.task == "selective_copy":
        return p["n_data"]
    if spec.task == "induction":
        return p["length"]
    raise ValueError(f"unknown task {spec.task!r}")


def chance_level(spec: ProbeSpec) -> float | None:
    """Token accuracy of a model that knows the context but cannot recall which answer belongs to the query.

    1/difficulty for the recall tasks, where the answer is one of `difficulty` items; None for the
    copy tasks, which have no such level.
    """
    if spec.task in ("mqar", "multihop", "fuzzy"):
        return 1 / difficulty(spec)
    if spec.task in ("selective_copy", "induction"):
        return None
    raise ValueError(f"unknown task {spec.task!r}")


def _ranges(spec: ProbeSpec) -> dict[str, tuple[int, int]]:
    """Half-open id ranges per role for a spec."""
    p = spec.as_dict()
    if spec.task == "mqar":
        keys = (N_SPECIAL, N_SPECIAL + p["n_keys"])
        values = (keys[1], keys[1] + p["n_values"])
        return {"keys": keys, "values": values}
    if spec.task == "fuzzy":
        keys = (N_SPECIAL, N_SPECIAL + p["n_keys"])
        values = (keys[1], keys[1] + p["n_values"])
        noise = (values[1], values[1] + p["n_noise_tokens"])
        return {"keys": keys, "values": values, "noise": noise}
    if spec.task in ("multihop", "selective_copy", "induction"):
        n = p["n_symbols"]
        return {"symbols": (N_SPECIAL, N_SPECIAL + n)}
    raise ValueError(f"unknown task {spec.task!r}")


# --- generators: one sequence each -------------------------------------------------------
# Each returns (x_row, scored): x_row is a list of ints of length seq_len, scored is a list
# of (position, answer) built from the generator's own knowledge of the sequence.


def _row_mqar(p: dict, r: dict, rng: np.random.Generator):
    n = p["n_pairs"]
    keys = rng.choice(p["n_keys"], n, replace=False) + r["keys"][0]
    vals = rng.choice(p["n_values"], n, replace=False) + r["values"][0]
    order = rng.permutation(n)
    ctx = np.stack([keys, vals], axis=1).reshape(-1)
    # Each query is a key followed by BLANK in the answer's slot; the answer is only in y.
    qa = np.stack([keys[order], np.full(n, BLANK, dtype=np.int64)], axis=1).reshape(-1)
    x = np.concatenate([ctx, [SEP], qa])
    scored = [(2 * n + 1 + 2 * j, int(vals[order[j]])) for j in range(n)]
    return x, scored


def _row_multihop(p: dict, r: dict, rng: np.random.Generator):
    n = p["n_chains"]
    sym = rng.choice(p["n_symbols"], 3 * n, replace=False) + r["symbols"][0]
    a, b, c = sym[0::3], sym[1::3], sym[2::3]
    pairs = np.concatenate([np.stack([a, b], axis=1), np.stack([b, c], axis=1)], axis=0)
    ctx = pairs[rng.permutation(2 * n)].reshape(-1)
    q = rng.permutation(n)
    qa = np.stack([a[q], np.full(n, BLANK, dtype=np.int64)], axis=1).reshape(-1)
    x = np.concatenate([ctx, [SEP], qa])
    scored = [(4 * n + 1 + 2 * j, int(c[q[j]])) for j in range(n)]
    return x, scored


def _row_fuzzy(p: dict, r: dict, rng: np.random.Generator):
    n, k, m = p["n_pairs"], p["key_len"], p["n_noise"]
    keytok = np.empty((n, k), dtype=np.int64)
    for j in range(k):
        # Distinct within each position: any single position identifies the key.
        keytok[:, j] = rng.choice(p["n_keys"], n, replace=False) + r["keys"][0]
    vals = rng.choice(p["n_values"], n, replace=False) + r["values"][0]
    ctx = np.concatenate([keytok, vals[:, None]], axis=1).reshape(-1)

    order = rng.permutation(n)
    blocks = []
    scored = []
    s = n * (k + 1)  # SEP position
    for j, i in enumerate(order):
        block = keytok[i].copy()
        noise_at = rng.choice(k, m, replace=False)
        block[noise_at] = r["noise"][0] + rng.integers(0, p["n_noise_tokens"], m)
        # The answer's slot holds BLANK; the value itself is only in y.
        blocks.append(np.concatenate([block, [BLANK]]))
        scored.append((s + 1 + j * (k + 1) + (k - 1), int(vals[i])))
    x = np.concatenate([ctx, [SEP], np.concatenate(blocks)])
    return x, scored


def _row_selective_copy(p: dict, r: dict, rng: np.random.Generator):
    n, m = p["n_data"], p["n_blank"]
    data = rng.integers(0, p["n_symbols"], n) + r["symbols"][0]
    slots = np.sort(rng.choice(n + m, n, replace=False))
    seq = np.full(n + m, BLANK, dtype=np.int64)
    seq[slots] = data
    x = np.concatenate([seq, [COPY], data])
    c = n + m
    scored = [(c + i, int(data[i])) for i in range(n)]
    return x, scored


def _row_induction(p: dict, r: dict, rng: np.random.Generator):
    length = p["length"]
    sym = rng.choice(p["n_symbols"], length, replace=False) + r["symbols"][0]
    x = np.concatenate([sym, [SEP], sym])
    scored = [(length, int(sym[0]))] + [(length + 1 + i, int(sym[i + 1])) for i in range(length - 1)]
    return x, scored


_ROW_BUILDERS = {
    "mqar": _row_mqar,
    "multihop": _row_multihop,
    "fuzzy": _row_fuzzy,
    "selective_copy": _row_selective_copy,
    "induction": _row_induction,
}


def generate(spec: ProbeSpec, batch: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch of sequences drawn from `rng`, as int64 tensors (batch, seq_len)."""
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch must be a positive int, got {batch!r}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
    p = spec.as_dict()
    r = _ranges(spec)
    build = _ROW_BUILDERS[spec.task]
    T = spec.seq_len
    x = np.full((batch, T), PAD, dtype=np.int64)
    y = np.full((batch, T), IGNORE, dtype=np.int64)
    for b in range(batch):
        row, scored = build(p, r, rng)
        x[b, : len(row)] = row
        for pos, answer in scored:
            y[b, pos] = answer
    return torch.from_numpy(x), torch.from_numpy(y)


# --- the oracle ---------------------------------------------------------------------------


def _content_length(x: list[int]) -> int:
    return x.index(PAD) if PAD in x else len(x)


def solve(spec: ProbeSpec, x_row) -> list[tuple[int, int]]:
    """The answers of one input sequence, from its input tokens alone.

    Returns [(position, answer token), ...] for every scored position, in order.
    """
    x = [int(v) for v in np.asarray(x_row).reshape(-1).tolist()]
    if len(x) != spec.seq_len:
        raise ValueError(f"expected a sequence of length {spec.seq_len}, got {len(x)}")
    L = _content_length(x)
    r = _ranges(spec)

    if spec.task == "mqar":
        s = x.index(SEP)
        table = {x[i]: x[i + 1] for i in range(0, s, 2)}
        return [(q, table[x[q]]) for q in range(s + 1, L, 2)]

    if spec.task == "multihop":
        s = x.index(SEP)
        succ = {x[i]: x[i + 1] for i in range(0, s, 2)}
        return [(q, succ[succ[x[q]]]) for q in range(s + 1, L, 2)]

    if spec.task == "fuzzy":
        p = spec.as_dict()
        k = p["key_len"]
        bs = k + 1
        s = x.index(SEP)
        n_ctx = s // bs
        ctx_keys = [x[i * bs : i * bs + k] for i in range(n_ctx)]
        ctx_vals = [x[i * bs + k] for i in range(n_ctx)]
        noise_lo, noise_hi = r["noise"]
        out = []
        n_queries = (L - s - 1) // bs
        for j in range(n_queries):
            base = s + 1 + j * bs
            block = x[base : base + bs]
            # n_noise < key_len, so some key position is not noise and identifies the key.
            o = next(o for o in range(k) if not noise_lo <= block[o] < noise_hi)
            i = next(i for i in range(n_ctx) if ctx_keys[i][o] == block[o])
            out.append((base + k - 1, ctx_vals[i]))
        return out

    if spec.task == "selective_copy":
        c = x.index(COPY)
        data = [t for t in x[:c] if t != BLANK]
        return [(q, data[q - c]) for q in range(c, L - 1)]

    if spec.task == "induction":
        s = x.index(SEP)
        first = x[:s]
        out = []
        for q in range(s, L - 1):
            if x[q] == SEP:
                out.append((q, first[0]))
            else:
                out.append((q, first[first.index(x[q]) + 1]))
        return out

    raise ValueError(f"unknown task {spec.task!r}")


# --- the batch stream ---------------------------------------------------------------------


class ProbeStream:
    """Batches of one probe. Stateless: batch(step) depends only on step and the constructor arguments."""

    def __init__(self, spec: ProbeSpec, *, sequences_per_step: int, seed: int, max_steps: int):
        self.spec = spec
        self._batch = _check_int("sequences_per_step", sequences_per_step, 1)
        self.seed = _check_int("seed", seed, 0)
        self.max_steps = _check_int("max_steps", max_steps, 1)

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < self.max_steps:
            raise IndexError(f"step {step!r} out of range [0, {self.max_steps})")
        rng = np.random.default_rng([self.seed, step])
        return generate(self.spec, self._batch, rng)


# --- accuracy -----------------------------------------------------------------------------


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    param = next(model.parameters(), None)
    return fallback if param is None else param.device


def counts(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, mode: str = "opt") -> dict:
    """Raw counts behind `accuracy`, so that callers can pool several batches exactly."""
    if x.shape != y.shape or x.dim() != 2:
        raise ValueError(f"x and y must be equal (batch, T) tensors, got {tuple(x.shape)} and {tuple(y.shape)}")
    modes = {m: m.training for m in model.modules()}
    device = _model_device(model, x.device)
    model.eval()
    try:
        with torch.no_grad():
            out = model(x.to(device), mode=mode)
            logits = out[0] if isinstance(out, tuple) else out
    finally:
        for m, flag in modes.items():
            m.training = flag
    target = y.to(logits.device)
    pred = logits.argmax(dim=-1)
    scored = target != IGNORE
    right = (pred == target) & scored
    row_scored = scored.sum(dim=1)
    row_ok = ((pred == target) | ~scored).all(dim=1) & (row_scored > 0)
    rows_with_scores = int((row_scored > 0).sum())
    return {
        "n_scored": int(scored.sum()),
        "n_correct_tokens": int(right.sum()),
        "n_sequences": rows_with_scores,
        "n_correct_sequences": int(row_ok.sum()),
    }


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, mode: str = "opt") -> dict:
    """token_accuracy, sequence_accuracy and n_scored, computed in eval mode under no_grad.

    The model's train/eval flags are restored afterwards, module by module.
    sequence_accuracy is over sequences with at least one scored position.
    """
    c = counts(model, x, y, mode=mode)
    if c["n_scored"] == 0:
        raise ValueError("no scored positions: accuracy is undefined")
    return {
        "token_accuracy": c["n_correct_tokens"] / c["n_scored"],
        "sequence_accuracy": c["n_correct_sequences"] / c["n_sequences"],
        "n_scored": c["n_scored"],
    }
