"""Training timing pass (plan §8.5, step 1): scripts/timing_pass.py.

cpu tier: round-robin order, warm-up discard, the SWA-below-MHA assertion, the readers, and
a tiny end-to-end pass. Timing is checked with a fake clock where the value matters.
gpu tier: the pass on CUDA with a tiny synthetic model; the SWA result is recorded, not
asserted, because a tiny synthetic model is not the study.
"""

from __future__ import annotations

import json

import pytest
import torch

from analysis import rqw_estimate as rq
from scripts import make_tables as mt
from scripts import timing_pass as tp
from src import config
from src.data.loader import TrainStream
from src.model import build_model
from src.training.trainer import Trainer
from src.utils.reproducibility import synthetic_split
from tests.fixtures.unfilled import unfilled_recipe
from tests.fixtures.fake_campaign import fake_campaign
from tests.fixtures.commitments import analyse_committed

CPU = torch.device("cpu")
ALL = tp.DEFAULT_VARIANTS
SMALL = ["--steps", "4", "--discarded", "1", "--repetitions", "2", "--peak-lr", "1e-3"]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# --- Round-robin order -----------------------------------------------------------------


def _fake_repetition(track, setup):
    return tp.Repetition(ms_per_step=1.0, peak_bytes=None)


def _fake_share(track, setup):
    return 0.25


def _setup_for_fake_runs():
    return tp.TimingSetup(stream=None, overrides={}, peak_lr=1e-3, steps=5, discarded=1, device=CPU)


def test_round_robin_is_repetition_major_for_every_track():
    seen = []

    def recording(track, setup):
        seen.append(track.key)
        return tp.Repetition(ms_per_step=1.0, peak_bytes=None)

    doc = tp.run_timing_pass(
        _setup_for_fake_runs(), variants=ALL, repetitions=3, with_ref=True,
        time_fn=recording, share_fn=_fake_share,
    )
    track_keys = list(ALL) + ["mla192/ref", "mla96/ref", "swa256/ref"]
    assert seen == [key for _ in range(3) for key in track_keys]
    assert doc["order"] == track_keys
    assert all(len(doc["variants"][key]["ms_per_step_by_repetition"]) == 3 for key in track_keys)


def test_round_robin_helper_never_runs_one_track_twice_in_a_row():
    schedule = tp.round_robin(tp.tracks(["mha", "mqa"], False), 3)
    assert [(t.key, r) for t, r in schedule] == [
        ("mha", 0), ("mqa", 0), ("mha", 1), ("mqa", 1), ("mha", 2), ("mqa", 2),
    ]


def test_ref_tracks_are_only_for_mla_and_swa_when_asked():
    assert [t.key for t in tp.tracks(ALL, False)] == list(ALL)
    assert [t.key for t in tp.tracks(ALL, True)] == list(ALL) + ["mla192/ref", "mla96/ref", "swa256/ref"]
    assert [t.key for t in tp.tracks(["mha", "mqa"], True)] == ["mha", "mqa"]


# --- Warm-up discard --------------------------------------------------------------------


def test_the_first_steps_of_each_repetition_are_discarded():
    clock = FakeClock()
    count = []

    def step():
        count.append(1)
        # Steps 0 and 1 are slow (warm-up), steps 2 to 5 take a quarter of a second.
        clock.now += 50.0 if len(count) <= 2 else 0.25

    durations = tp.time_steps(step, steps=6, discarded=2, device=CPU, clock=clock)
    assert len(count) == 6
    assert durations == [250.0, 250.0, 250.0, 250.0]


def test_discard_must_leave_measured_steps():
    with pytest.raises(ValueError):
        tp.time_steps(lambda: None, steps=3, discarded=3, device=CPU)


# --- SWA below MHA (plan §2.3 item 2) ---------------------------------------------------


def _entries(mha_share, mha_ms, swa_share, swa_ms):
    return {
        "mha": {"mode": "opt", "ms_per_step": mha_ms, "attention_share": mha_share},
        "swa256": {"mode": "opt", "ms_per_step": swa_ms, "attention_share": swa_share},
    }


def test_swa_below_mha_passes_when_its_attention_time_is_smaller():
    check = tp.check_swa_below_mha(_entries(mha_share=0.2, mha_ms=100.0, swa_share=0.1, swa_ms=100.0))
    assert check.holds is True
    assert check.mha_attention_ms == 0.2 * 100.0
    assert check.swa_attention_ms == 0.1 * 100.0
    tp.require_swa_below_mha(_entries(0.2, 100.0, 0.1, 100.0))


def test_swa_fails_when_its_attention_time_is_not_smaller():
    # SWA's share is larger, so its attention time is larger.
    entries = _entries(mha_share=0.2, mha_ms=100.0, swa_share=0.3, swa_ms=100.0)
    assert tp.check_swa_below_mha(entries).holds is False
    with pytest.raises(AssertionError, match="not below MHA"):
        tp.require_swa_below_mha(entries)


def test_swa_fails_when_a_smaller_share_is_offset_by_a_longer_step():
    # Share times step time: 0.1 x 300 = 30 is not below 0.2 x 100 = 20.
    entries = _entries(mha_share=0.2, mha_ms=100.0, swa_share=0.1, swa_ms=300.0)
    with pytest.raises(AssertionError):
        tp.require_swa_below_mha(entries)


def test_equal_attention_time_is_not_below():
    entries = _entries(mha_share=0.2, mha_ms=100.0, swa_share=0.2, swa_ms=100.0)
    assert tp.check_swa_below_mha(entries).holds is False


def test_the_swa_check_needs_both_entries():
    with pytest.raises(tp.TimingError, match="mha"):
        tp.check_swa_below_mha({"swa256": {"ms_per_step": 1.0, "attention_share": 0.1}})


# --- Readers and consumers --------------------------------------------------------------


def test_reader_returns_the_mapping_the_consumers_take(tmp_path):
    ms = {v: 10.0 + i for i, v in enumerate(ALL)}

    def timed(track, setup):
        return tp.Repetition(ms_per_step=ms[track.variant], peak_bytes=None)

    doc = tp.run_timing_pass(
        _setup_for_fake_runs(), variants=ALL, repetitions=2, with_ref=True,
        time_fn=timed, share_fn=_fake_share,
    )
    path = tp.write_timing_file(tmp_path / "timing_pass.json", doc)
    mapping = tp.read_ms_per_step(path)
    assert mapping == ms  # the ref tracks are not in the mapping
    assert set(mapping) == set(ALL)

    # Consumer 1: the wall-clock estimate takes {variant: ms per step}.
    estimate = rq.estimate([], mapping)
    assert estimate.reference_ms_per_step == ms["mha"]
    assert estimate.budget_ms == ms["mha"] * estimate.reference_steps
    assert [v.variant for v in estimate.variants] == ["swa256", "mla192"]
    expected_steps = rq.steps_afforded(ms["mha"], ms["swa256"], estimate.reference_steps)
    assert estimate.variants[0].steps == expected_steps

    # Consumer 2: the headline table takes the same mapping as train_ms_per_step.
    analysis = analyse_committed(fake_campaign())
    rows = {r["variant"]: r for r in mt.headline_rows(analysis, train_ms_per_step=mapping)}
    assert rows["mha"]["train ms/step"] == f"{ms['mha']:.1f}"
    assert rows["mla96"]["train ms/step"] == f"{ms['mla96']:.1f}"


def test_reader_refuses_another_schema(tmp_path):
    path = tmp_path / "timing_pass.json"
    path.write_text(json.dumps({"schema": "timing_pass/0", "variants": {}}), encoding="utf-8")
    with pytest.raises(tp.TimingError, match="schema"):
        tp.read_timing_pass(path)


def test_the_document_has_the_schema_keys(tmp_path):
    doc = tp.run_timing_pass(
        _setup_for_fake_runs(), variants=["mha"], repetitions=2,
        time_fn=_fake_repetition, share_fn=_fake_share,
    )
    assert set(doc) == {
        "schema", "timing_pass_id", "environment", "gpu_shared", "backend", "steps", "discarded",
        "repetitions", "order", "variants",
    }
    assert doc["schema"] == "timing_pass/1"
    assert doc["steps"] == 5 and doc["discarded"] == 1 and doc["repetitions"] == 2
    entry = doc["variants"]["mha"]
    assert set(entry) == {"mode", "ms_per_step", "ms_per_step_by_repetition", "attention_share", "peak_bytes"}
    assert entry["mode"] == "opt" and entry["peak_bytes"] is None


def test_timing_pass_ids_are_unique():
    ids = {
        tp.run_timing_pass(
            _setup_for_fake_runs(), variants=["mha"], repetitions=1,
            time_fn=_fake_repetition, share_fn=_fake_share,
        )["timing_pass_id"]
        for _ in range(3)
    }
    assert len(ids) == 3


# --- Attention timer ----------------------------------------------------------------------


def test_attention_timer_times_forward_and_backward_and_removes_its_hooks():
    resolved = config.resolve(
        "mha", init_seed=1, peak_lr=1e-3, steps=2, branches=[], overrides=dict(tp.SYNTHETIC_DIMS)
    )
    model = build_model(resolved)
    stream = TrainStream(synthetic_split(2 * 8 * 32 + 1, 128, seed=0), context_length=32,
                         sequences_per_step=8, data_seed=1)
    trainer = Trainer(model, resolved, stream, device="cpu", mode="opt")
    with tp.AttentionTimer(trainer.model, device=CPU) as timer:
        assert len(trainer.model.layers[0].attn._forward_hooks) == 1
        trainer.train_step()
        assert timer.total > 0.0
        timer.reset()
        assert timer.total == 0.0
    assert len(trainer.model.layers[0].attn._forward_hooks) == 0
    assert len(trainer.model.layers[0].attn._forward_pre_hooks) == 0


# --- End to end on CPU --------------------------------------------------------------------


def test_the_pass_runs_end_to_end_on_cpu_and_writes_the_schema(tmp_path):
    out = tmp_path / "timing_pass.json"
    code = tp.main(["--synthetic", "--variants", "mha", "swa256", "mla192", "--with-ref", *SMALL, "--out", str(out)])
    assert code == 0  # on CPU the SWA check is reported, not enforced
    doc = json.loads(out.read_text())
    assert doc["schema"] == "timing_pass/1"
    assert doc["backend"] is None  # no backend is enforced on CPU
    assert doc["timing_pass_id"].startswith("tp-")
    assert doc["gpu_shared"] is False
    assert (doc["steps"], doc["discarded"], doc["repetitions"]) == (4, 1, 2)
    assert doc["order"] == ["mha", "swa256", "mla192", "mla192/ref", "swa256/ref"]
    for key in doc["order"]:
        entry = doc["variants"][key]
        assert entry["mode"] == ("ref" if key.endswith("/ref") else "opt")
        assert isinstance(entry["ms_per_step"], float) and entry["ms_per_step"] > 0
        assert len(entry["ms_per_step_by_repetition"]) == 2
        assert 0.0 <= entry["attention_share"] <= 1.0
        assert entry["peak_bytes"] is None
    assert tp.read_ms_per_step(out) == {
        k: doc["variants"][k]["ms_per_step"] for k in ("mha", "swa256", "mla192")
    }


def test_the_pass_refuses_to_run_without_the_recipe_values(tmp_path, monkeypatch):
    # With micro_batch and the scaler values unset in the recipe: fail clearly, before any data is opened.
    unfilled_recipe(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        tp.main(["--variants", "mha", "--peak-lr", "1e-3", "--data-dir", str(tmp_path), "--out", str(tmp_path / "x.json")])
    assert exc.value.code == 2
    assert not (tmp_path / "x.json").exists()


def test_every_track_hands_the_expected_backend_to_its_trainer():
    # The Trainer refuses an expected backend off CUDA, so the refusal shows that the
    # name reaches it, on the opt and on the ref track. Added in review.
    steps = 2
    stream = TrainStream(
        synthetic_split(steps * 8 * 32 + 1, tp.SYNTHETIC_VOCAB, seed=0),
        context_length=32, sequences_per_step=8, data_seed=1,
    )
    setup = tp.TimingSetup(stream=stream, overrides=dict(tp.SYNTHETIC_DIMS), peak_lr=1e-3, steps=steps,
                           discarded=0, device=CPU, expected_backend="efficient")
    for track in (tp.Track("mha", "mha", "opt"), tp.Track("swa256/ref", "swa256", "ref")):
        with pytest.raises(ValueError, match="CUDA only"):
            tp.time_repetition(track, setup)
        with pytest.raises(ValueError, match="CUDA only"):
            tp.measure_share(track, setup)


def test_the_expected_backend_is_required_on_cuda_and_refused_on_cpu(tmp_path):
    from src.eval import inference as inf

    cuda = torch.device("cuda")
    with pytest.raises(inf.BenchError, match="required on CUDA"):
        inf.check_backend_flag(None, cuda)
    with pytest.raises(inf.BenchError, match="CUDA only"):
        inf.check_backend_flag("math", CPU)
    with pytest.raises(inf.BenchError, match="unknown SDPA backend"):
        inf.check_backend_flag("bogus", cuda)
    with pytest.raises(SystemExit):
        tp.main(["--synthetic", "--gpu-shared", "false", "--variants", "mha", *SMALL,
                 "--expected-backend", "math", "--out", str(tmp_path / "c.json")])
    with pytest.raises(SystemExit):
        tp.main(["--synthetic", "--gpu-shared", "false", "--device", "cuda", "--variants", "mha", *SMALL,
                 "--out", str(tmp_path / "d.json")])
    assert not (tmp_path / "c.json").exists() and not (tmp_path / "d.json").exists()


def test_the_pass_refuses_gpu_shared_true_and_requires_the_flag_on_cuda(tmp_path):
    with pytest.raises(SystemExit):
        tp.main(["--synthetic", "--gpu-shared", "true", "--variants", "mha", *SMALL, "--out", str(tmp_path / "a.json")])
    with pytest.raises(SystemExit):
        tp.main(["--synthetic", "--device", "cuda", "--variants", "mha", *SMALL, "--out", str(tmp_path / "b.json")])
    assert not (tmp_path / "a.json").exists() and not (tmp_path / "b.json").exists()


def test_the_null_variant_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        tp.main(["--synthetic", "--variants", "null", *SMALL, "--out", str(tmp_path / "n.json")])


# --- GPU tier ----------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_pass_runs_on_cuda_in_fp16_with_a_tiny_model():
    overrides = dict(tp.SYNTHETIC_DIMS)
    steps = 6
    stream = TrainStream(
        synthetic_split(steps * 8 * 32 + 1, tp.SYNTHETIC_VOCAB, seed=0),
        context_length=32, sequences_per_step=8, data_seed=1,
    )
    setup = tp.TimingSetup(stream=stream, overrides=overrides, peak_lr=1e-3, steps=steps, discarded=2,
                           device=torch.device("cuda"), expected_backend="efficient")
    doc = tp.run_timing_pass(setup, variants=["mha", "swa256"], repetitions=2, gpu_shared=False)
    assert doc["gpu_shared"] is False
    assert doc["backend"] == "efficient"
    for key in ("mha", "swa256"):
        entry = doc["variants"][key]
        assert isinstance(entry["peak_bytes"], int) and entry["peak_bytes"] > 0
        assert 0.0 <= entry["attention_share"] <= 1.0
    # Recorded, not asserted: a tiny synthetic model is not the study (plan §8.5).
    check = tp.check_swa_below_mha(doc["variants"])
    assert isinstance(check.holds, bool)
