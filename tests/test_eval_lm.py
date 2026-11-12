"""T-20: evaluation with context-bucketed loss (plan §8.4, prereg `context_use`).

All token data here is SYNTHETIC (tests/fixtures/synthetic_documents.py) or
hand-built in this file. Hand-built document layouts are worked out by hand and
written out as literals, so they do not depend on the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.data.loader import TokenSplit
from src.eval.lm import ContextSpec, evaluate
from src.eval.positions import bucket_index, context_values
from tests.fixtures.synthetic_documents import constant_split, document_starts, synthetic_split
from tests.tolerances import FLOAT64_IDENTITY, REF_OPT_FP16, REF_OPT_FP32, scale_relative_error
from tests.variants import CASES, VOCAB, tiny_model

PREREG_BUCKETS = ((0, 64), (64, 256), (256, 512), (512, 1024))

# Tiny config: the model context is 64 (tests/variants.py), so the buckets are shrunk.
TINY_T = 64
TINY_SPEC = ContextSpec(
    buckets=((0, 4), (4, 16), (16, 32), (32, 64)),
    late_context_min=32,
    icl_late=(40, 50),
    icl_early=(4, 8),
)

# Hand-built layout for T = 8, 33 tokens, so 4 windows (w = 0..3). Documents
# (global offsets, EOS last):
#   A [0, 3)   tokens 0..2             starts at the split start
#   B [3, 5)   tokens 3..4             starts mid-window 0 (offset 3)
#   C [5, 6)   token 5 only            one-token document, mid-window 0
#   D [6, 7)   token 6 only            one-token document, mid-window 0
#   E [7, 8)   token 7 only            one-token document, mid-window 0
#   F [8, 25)  tokens 8..24            starts exactly at window 1's start; spans windows 1, 2, 3
#   G [25, 33) tokens 25..32           starts mid-window 3 at offset 1 (k = 1)
T8 = 8
HAND_DOC_STARTS = np.array([0, 3, 5, 6, 7, 8, 25, 33], dtype=np.int64)
HAND_N_TOKENS = 33
# c for the 4 windows x 8 targets, worked out by hand from the layout above.
#   window 0 (ws = 0):  targets g = 1..8.  g=1,2 in A: c = 1,2.  g=3,4 in B: c = 0,1.
#                       g=5 (C), 6 (D), 7 (E), 8 (first of F): c = 0.
#   window 1 (ws = 8):  g = 9..16 all in F, which starts at ws: c = t + 1.
#   window 2 (ws = 16): g = 17..24 in F, which started before ws: c = g - 16 = t + 1.
#   window 3 (ws = 24): g = 25 is the first token of G: c = 0; then 1..7.
HAND_C = np.array(
    [
        [1, 2, 0, 1, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ],
    dtype=np.int64,
)
HAND_SPEC = ContextSpec(
    buckets=((0, 2), (2, 4), (4, 8)),
    late_context_min=4,
    icl_late=(3, 5),  # both end points must count
    icl_early=(1, 2),  # both end points must count
)
TARGET = 5  # every token of the hand-built split is this class


def _split(seed: int = 3) -> TokenSplit:
    rng = np.random.default_rng(seed)
    starts = document_starts(641, 12, rng)  # 10 windows of 64, many boundaries
    return synthetic_split(starts, VOCAB, rng)


class PositionStub(nn.Module):
    """Logits depend only on the position t: the target class has logit 0.25 * t, all
    others 0. Every target is `target`, so each token's loss depends only on its position."""

    def __init__(self, T: int, vocab: int, target: int):
        super().__init__()
        table = torch.zeros(T, vocab)
        table[:, target] = 0.25 * torch.arange(T, dtype=torch.float32)
        self.table = table
        self.seen_grad_enabled: list[bool] = []

    def forward(self, idx, *, mode="opt"):
        self.seen_grad_enabled.append(torch.is_grad_enabled())
        b, t = idx.shape
        return self.table[:t].expand(b, t, -1).clone(), None


class NaNStub(PositionStub):
    """PositionStub whose logits are NaN in every window whose first input is `marker`."""

    def __init__(self, T: int, vocab: int, target: int, marker: int):
        super().__init__(T, vocab, target)
        self.marker = marker

    def forward(self, idx, *, mode="opt"):
        logits, state = super().forward(idx, mode=mode)
        logits = logits.clone()
        logits[idx[:, 0] == self.marker] = float("nan")
        return logits, state


def _stub_losses(stub: PositionStub, n_windows: int, T: int) -> np.ndarray:
    """Per-token losses of the stub on constant TARGET targets, from the same float32
    cross-entropy that evaluate() applies to the stub's logits."""
    logits, _ = stub(torch.zeros(n_windows, T, dtype=torch.long))
    targets = torch.full((n_windows, T), TARGET, dtype=torch.long)
    losses = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none")
    return losses.to(torch.float64).numpy().reshape(n_windows, T)


def _slow_context(doc_starts, first: int, n: int, T: int) -> np.ndarray:
    """Obviously correct per-token loop: find the document of each target by scanning."""
    rows = []
    for w in range(first, first + n):
        row = []
        ws = w * T
        for t in range(T):
            g = ws + t + 1
            doc = max(i for i in range(len(doc_starts) - 1) if doc_starts[i] <= g)
            row.append(g - max(int(doc_starts[doc]), ws))
        rows.append(row)
    return np.array(rows, dtype=np.int64).reshape(n, T)


# --- 1. Hand-built windows -------------------------------------------------


def test_hand_built_context_matrix_matches_literal_values():
    c = context_values(HAND_DOC_STARTS, 0, 4, T8)
    assert c.dtype == np.int64
    assert np.array_equal(c, HAND_C)


# --- 2. Agreement with a slow loop --------------------------------------------


@pytest.mark.parametrize("T", [1, 3, 8, 16])
@pytest.mark.parametrize("first", [0, 2])
def test_context_values_match_slow_loop_on_random_boundaries(T, first):
    rng = np.random.default_rng(100 + T)
    n_tokens = 150
    starts = document_starts(n_tokens, 5, rng)
    n_total = (n_tokens - 1) // T
    n = n_total - first
    assert n > 0
    fast = context_values(starts, first, n, T)
    assert np.array_equal(fast, _slow_context(starts, first, n, T))


# --- 3. Range and special values ---------------------------------------------


@pytest.mark.parametrize("T", [4, 8, 16])
def test_context_range_and_zero_exactly_at_document_starts(T):
    rng = np.random.default_rng(7 + T)
    n_tokens = 25 * T + 1
    starts = document_starts(n_tokens, 6, rng)
    n = (n_tokens - 1) // T
    c = context_values(starts, 0, n, T)
    assert c.min() >= 0
    assert c.max() <= T
    targets = np.arange(1, n * T + 1).reshape(n, T)  # global offsets g of the targets
    # c == 0 exactly where the target is the first token of a document.
    assert np.array_equal(c == 0, np.isin(targets, starts[:-1]))


@pytest.mark.parametrize("T", [4, 8])
def test_single_long_document_gives_c_equal_to_position_and_T_at_window_end(T):
    n_tokens = 10 * T + 1
    starts = np.array([0, n_tokens], dtype=np.int64)
    n = (n_tokens - 1) // T
    c = context_values(starts, 0, n, T)
    assert np.array_equal(c, np.tile(np.arange(1, T + 1), (n, 1)))
    assert np.all(c[:, -1] == T)


# --- 4. Buckets -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value, bucket",
    [(0, 0), (63, 0), (64, 1), (255, 1), (256, 2), (511, 2), (512, 3), (1023, 3), (1024, 3)],
)
def test_bucket_index_boundaries_with_prereg_buckets(value, bucket):
    assert int(bucket_index(np.array([value]), PREREG_BUCKETS)[0]) == bucket


def test_bucket_index_preserves_shape_and_rejects_values_in_no_bucket():
    c = np.array([[0, 1024], [64, 512]])
    assert bucket_index(c, PREREG_BUCKETS).tolist() == [[0, 3], [1, 3]]
    for bad in (1025, -1):
        with pytest.raises(ValueError):
            bucket_index(np.array([0, bad]), PREREG_BUCKETS)


@pytest.mark.parametrize(
    "buckets",
    [
        [],
        [(1, 64), (64, 256)],  # first low is not 0
        [(0, 64), (65, 256)],  # gap
        [(0, 64), (32, 256)],  # overlap
        [(0, 64), (64, 64)],  # empty bucket
        [(0, 64), (64, 32)],  # reversed
        [(0, 64, 9)],  # not a pair
    ],
)
def test_malformed_bucket_lists_raise(buckets):
    with pytest.raises(ValueError):
        bucket_index(np.array([0]), buckets)


# --- 5. Bucket-weighted loss equals aggregate ----------------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_bucket_weighted_loss_equals_aggregate(case):
    model = tiny_model(case)
    split = _split()
    res = evaluate(model, split, context_length=TINY_T, batch_size=3, spec=TINY_SPEC)
    counts = [b["n_tokens"] for b in res["buckets"]]
    assert sum(counts) == res["n_tokens"] == res["n_windows"] * TINY_T
    assert res["n_windows"] == split.n_windows(TINY_T) == 10
    weighted = sum(b["n_tokens"] * b["loss"] for b in res["buckets"] if b["n_tokens"]) / sum(counts)
    assert scale_relative_error(weighted, res["loss"]) <= FLOAT64_IDENTITY


# --- 6. Aggregate equals an independent computation ----------------------------


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_aggregate_loss_matches_direct_cross_entropy(case):
    model = tiny_model(case)
    split = _split()
    res = evaluate(model, split, context_length=TINY_T, batch_size=3, spec=TINY_SPEC)
    pairs = [split.window(w, TINY_T) for w in range(split.n_windows(TINY_T))]
    x = torch.from_numpy(np.stack([p[0] for p in pairs]))
    y = torch.from_numpy(np.stack([p[1] for p in pairs]))
    with torch.no_grad():
        logits, _ = model(x, mode="opt")
    direct = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1)).item()
    assert scale_relative_error(res["loss"], direct) <= REF_OPT_FP32


# --- 7. Known per-token losses ----------------------------------------------------


def test_known_per_token_losses_give_exact_bucket_late_and_icl_values():
    stub = PositionStub(T8, 8, TARGET)
    split = constant_split(HAND_DOC_STARTS, TARGET)
    res = evaluate(stub, split, context_length=T8, batch_size=4, spec=HAND_SPEC, precision="fp32")
    L = _stub_losses(stub, 4, T8)
    assert np.all(np.diff(L[0]) < 0)  # the loss really depends on the position

    c = HAND_C
    masks = [(c >= 0) & (c < 2), (c >= 2) & (c < 4), (c >= 4) & (c <= 8)]
    # Hand-counted: 11, 7 and 14 tokens; they sum to 32 = 4 windows x 8.
    assert [int(m.sum()) for m in masks] == [11, 7, 14]
    for block, mask in zip(res["buckets"], masks):
        assert block["n_tokens"] == int(mask.sum())
        assert scale_relative_error(block["loss"], L[mask].mean()) <= FLOAT64_IDENTITY

    late_mask = c >= 4
    assert res["late_context"]["min_c"] == 4
    assert res["late_context"]["n_tokens"] == 14 == int(late_mask.sum())
    assert scale_relative_error(res["late_context"]["loss"], L[late_mask].mean()) <= FLOAT64_IDENTITY

    # Both ICL windows include their end points: 3 and 5 for late, 1 and 2 for early.
    late_icl = (c >= 3) & (c <= 5)
    early_icl = (c >= 1) & (c <= 2)
    assert res["icl"]["late"]["n_tokens"] == 9 == int(late_icl.sum())
    assert res["icl"]["early"]["n_tokens"] == 9 == int(early_icl.sum())
    assert scale_relative_error(res["icl"]["late"]["loss"], L[late_icl].mean()) <= FLOAT64_IDENTITY
    assert scale_relative_error(res["icl"]["early"]["loss"], L[early_icl].mean()) <= FLOAT64_IDENTITY
    expected_score = L[late_icl].mean() - L[early_icl].mean()
    assert scale_relative_error(res["icl"]["score"], expected_score) <= FLOAT64_IDENTITY


def test_icl_end_points_are_inclusive_at_each_end():
    """A token at each end point counts; a token one outside does not. Tested by
    single-value windows: the window [c0, c0] must contain exactly the tokens with c == c0."""
    stub = PositionStub(T8, 8, TARGET)
    split = constant_split(HAND_DOC_STARTS, TARGET)
    L = _stub_losses(stub, 4, T8)
    for value in (1, 2, 3, 5):
        spec = ContextSpec(buckets=((0, 8),), late_context_min=8, icl_late=(value, value), icl_early=(0, 0))
        res = evaluate(stub, split, context_length=T8, batch_size=4, spec=spec, precision="fp32")
        assert res["icl"]["late"]["n_tokens"] == int((HAND_C == value).sum())
        assert scale_relative_error(res["icl"]["late"]["loss"], L[HAND_C == value].mean()) <= FLOAT64_IDENTITY


# --- 8. Empty buckets and empty windows --------------------------------------------


def test_empty_bucket_and_empty_icl_window_give_none():
    # Every document has 2 tokens, so every c is 0 or 1 and the bucket [2, 8] is empty.
    starts = np.array(list(range(0, 33, 2)) + [33], dtype=np.int64)
    stub = PositionStub(T8, 8, TARGET)
    spec = ContextSpec(buckets=((0, 1), (1, 2), (2, 8)), late_context_min=8, icl_late=(0, 1), icl_early=(5, 6))
    res = evaluate(stub, constant_split(starts, TARGET), context_length=T8, batch_size=4, spec=spec)
    assert res["buckets"][2] == {"low": 2, "high": 8, "n_tokens": 0, "loss": None}
    assert res["buckets"][0]["n_tokens"] > 0 and res["buckets"][0]["loss"] is not None
    assert res["late_context"] == {"min_c": 8, "n_tokens": 0, "loss": None}
    assert res["icl"]["early"] == {"low": 5, "high": 6, "n_tokens": 0, "loss": None}
    assert res["icl"]["score"] is None


# --- 9. Mode restoration and no gradient graph ----------------------------------------


@pytest.mark.parametrize("start_training", [True, False])
def test_restores_model_mode_and_builds_no_graph(start_training):
    model = tiny_model(CASES[0])
    model.train(start_training)
    evaluate(model, _split(), context_length=TINY_T, batch_size=4, spec=TINY_SPEC)
    assert model.training is start_training
    assert all(p.grad is None for p in model.parameters())


def test_restores_each_submodule_mode_and_no_grad_inside_forward():
    stub = PositionStub(T8, 8, TARGET)
    stub.train()
    child = nn.Linear(2, 2)
    stub.add_module("child", child)
    child.eval()
    evaluate(stub, constant_split(HAND_DOC_STARTS, TARGET), context_length=T8, batch_size=4, spec=HAND_SPEC)
    assert stub.training is True
    assert child.training is False
    assert stub.seen_grad_enabled and not any(stub.seen_grad_enabled)
    assert torch.is_grad_enabled()


class RecordingStub(PositionStub):
    """Also records how each forward was called."""

    def __init__(self, T: int, vocab: int, target: int):
        super().__init__(T, vocab, target)
        self.seen_training: list[bool] = []
        self.seen_mode: list[str] = []

    def forward(self, idx, *, mode="opt"):
        self.seen_training.append(self.training)
        self.seen_mode.append(mode)
        return super().forward(idx, mode=mode)


@pytest.mark.parametrize("start_training", [True, False])
def test_the_forward_runs_in_eval_mode(start_training):
    """Added in review: the model is evaluated in eval mode whatever state it arrives in."""
    stub = RecordingStub(8, 16, TARGET)
    stub.train(start_training)
    evaluate(stub, constant_split(HAND_DOC_STARTS, TARGET), context_length=8, batch_size=2, spec=HAND_SPEC)
    assert stub.seen_training and not any(stub.seen_training)
    assert stub.training is start_training


@pytest.mark.parametrize("mode", ["ref", "opt"])
def test_the_requested_mode_reaches_the_model(mode):
    """Added in review: `ref` and `opt` agree numerically, so only the call shows which ran."""
    stub = RecordingStub(8, 16, TARGET)
    evaluate(stub, constant_split(HAND_DOC_STARTS, TARGET), context_length=8, batch_size=2, spec=HAND_SPEC, mode=mode)
    assert stub.seen_mode and set(stub.seen_mode) == {mode}


def test_mode_restored_when_evaluation_raises():
    stub = NaNStub(T8, 8, TARGET, marker=7)
    stub.train()
    tokens = np.full(HAND_N_TOKENS, TARGET, dtype=np.uint16)
    tokens[2 * T8] = 7  # the first input of window 2
    split = TokenSplit(tokens, HAND_DOC_STARTS.copy())
    with pytest.raises(ValueError):
        evaluate(stub, split, context_length=T8, batch_size=4, spec=HAND_SPEC, precision="fp32")
    assert stub.training is True


# --- 10. Options and errors ------------------------------------------------------------


def test_max_windows_limits_the_windows_evaluated():
    stub = PositionStub(T8, 8, TARGET)
    split = constant_split(HAND_DOC_STARTS, TARGET)
    L = _stub_losses(stub, 4, T8)
    res = evaluate(stub, split, context_length=T8, batch_size=3, spec=HAND_SPEC, precision="fp32", max_windows=2)
    assert res["n_windows"] == 2
    assert res["n_tokens"] == 2 * T8
    assert scale_relative_error(res["loss"], L[:2].mean()) <= FLOAT64_IDENTITY
    res_all = evaluate(stub, split, context_length=T8, batch_size=3, spec=HAND_SPEC, precision="fp32", max_windows=10)
    assert res_all["n_windows"] == 4


@pytest.mark.parametrize("bad", [0, -1])
def test_max_windows_must_be_positive(bad):
    with pytest.raises(ValueError):
        evaluate(PositionStub(T8, 8, TARGET), constant_split(HAND_DOC_STARTS, TARGET),
                 context_length=T8, batch_size=4, spec=HAND_SPEC, precision="fp32", max_windows=bad)


def test_split_without_a_window_raises():
    split = constant_split(np.array([0, T8], dtype=np.int64), TARGET)  # 8 tokens: (8 - 1) // 8 = 0 windows
    with pytest.raises(ValueError):
        evaluate(PositionStub(T8, 8, TARGET), split, context_length=T8, batch_size=4, spec=HAND_SPEC, precision="fp32")


def test_non_finite_loss_names_the_window():
    tokens = np.full(HAND_N_TOKENS, TARGET, dtype=np.uint16)
    tokens[2 * T8] = 7  # window 2 starts with token 7, which makes the stub return NaN
    stub = NaNStub(T8, 8, TARGET, marker=7)
    with pytest.raises(ValueError, match="window 2"):
        evaluate(stub, TokenSplit(tokens, HAND_DOC_STARTS.copy()), context_length=T8, batch_size=4,
                 spec=HAND_SPEC, precision="fp32")


def test_fp16_on_cpu_raises():
    with pytest.raises(ValueError):
        evaluate(PositionStub(T8, 8, TARGET), constant_split(HAND_DOC_STARTS, TARGET),
                 context_length=T8, batch_size=4, spec=HAND_SPEC, device="cpu", precision="fp16")


def test_unknown_precision_raises():
    with pytest.raises(ValueError):
        evaluate(PositionStub(T8, 8, TARGET), constant_split(HAND_DOC_STARTS, TARGET),
                 context_length=T8, batch_size=4, spec=HAND_SPEC, precision="bf16")


# --- 11. The pre-registered spec -----------------------------------------------------------


def test_context_spec_from_prereg_has_the_registered_values():
    spec = ContextSpec.from_prereg(config.load_prereg())
    assert spec.buckets == ((0, 64), (64, 256), (256, 512), (512, 1024))
    assert spec.late_context_min == 512
    assert spec.icl_late == (450, 550)
    assert spec.icl_early == (40, 60)


# --- 12. GPU tier ------------------------------------------------------------------------


@pytest.mark.gpu
def test_fp16_on_cuda_agrees_with_fp32():
    model = tiny_model(CASES[0]).to("cuda")
    split = _split()
    r16 = evaluate(model, split, context_length=TINY_T, batch_size=4, spec=TINY_SPEC,
                   device="cuda", precision="fp16")
    r32 = evaluate(model, split, context_length=TINY_T, batch_size=4, spec=TINY_SPEC,
                   device="cuda", precision="fp32")
    assert scale_relative_error(r16["loss"], r32["loss"]) <= REF_OPT_FP16
