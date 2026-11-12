"""Plan §5, causal leak: perturb token t+1; output at position t is bit-identical.

Every variant, every mode. Written by the orchestrator before the
implementation; the worker may not change this file.
"""

import pytest
import torch

from tests.variants import CASES, MODES, VOCAB, tiny_model, tokens


@pytest.mark.parametrize("sharpen", [1.0, 6.0])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_future_tokens_do_not_change_earlier_logits(case, mode, sharpen):
    model = tiny_model(case, sharpen=sharpen)
    idx = tokens(3, 48)
    with torch.no_grad():
        base, _ = model(idx, mode=mode)
        for t in (0, 1, 17, 46):
            changed = idx.clone()
            changed[:, t + 1 :] = (changed[:, t + 1 :] + 1) % VOCAB
            out, _ = model(changed, mode=mode)
            assert torch.equal(out[:, : t + 1], base[:, : t + 1]), (case.name, mode, t)
            assert not torch.equal(out[:, t + 1 :], base[:, t + 1 :]), (case.name, mode, t)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_a_single_changed_token_only_changes_later_logits(case, mode):
    model = tiny_model(case, sharpen=6.0)
    idx = tokens(2, 40, seed=3)
    with torch.no_grad():
        base, _ = model(idx, mode=mode)
        changed = idx.clone()
        changed[:, 20] = (changed[:, 20] + 5) % VOCAB
        out, _ = model(changed, mode=mode)
    assert torch.equal(out[:, :20], base[:, :20])
    assert not torch.equal(out[:, 20], base[:, 20])
    assert not torch.equal(out[:, 39], base[:, 39])  # and it reaches the end through attention


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_logits_do_not_depend_on_the_other_sequences_in_the_batch(case, mode):
    model = tiny_model(case, sharpen=6.0)
    idx = tokens(4, 32, seed=5)
    with torch.no_grad():
        base, _ = model(idx, mode=mode)
        changed = idx.clone()
        changed[1:] = (changed[1:] + 7) % VOCAB
        out, _ = model(changed, mode=mode)
    assert torch.equal(out[0], base[0])


# --- gpu tier: the same under fp16 autocast (plan §5, "gpu for fp16") -------------------


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_future_tokens_do_not_change_earlier_logits_fp16(case, mode):
    model = tiny_model(case, sharpen=6.0).cuda()
    idx = tokens(3, 48).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        base, _ = model(idx, mode=mode)
        for t in (0, 17, 46):
            changed = idx.clone()
            changed[:, t + 1 :] = (changed[:, t + 1 :] + 1) % VOCAB
            out, _ = model(changed, mode=mode)
            assert torch.equal(out[:, : t + 1], base[:, : t + 1]), (case.name, mode, t)
