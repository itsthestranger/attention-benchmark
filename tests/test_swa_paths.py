"""The SWA two-paths record (plan §2.3; decisions 244, 245, 247): scripts/swa_paths.py.

cpu tier: a tiny synthetic run, the statistics under a scripted clock, the round-robin order, the
opt path each track really runs, the recorded failure of the flex track, the refusals of the command
line, the checks of the document, and the printed lines. gpu tier: one run on CUDA, skipped here.
The documents are hand-written or synthetic; no value of any real measurement is asserted.
"""

from __future__ import annotations

import json
import re

import pytest
import torch

from scripts import swa_paths as sp
from scripts import timing_pass as tp
from src.attention.sliding_window import SlidingWindowAttention
from tests.tolerances import AGREEMENT_FP64
from tests.fixtures.fake_secondary import ENVIRONMENT

CPU = torch.device("cpu")
# A warning of FlexAttention outside torch.compile, filtered as in tests/test_sliding_window.py.
FLEX_WARNING = "ignore:flex_attention called without torch.compile"
TINY = ["--synthetic", "--batch", "4", "--repeats", "2", "--warmup", "1", "--repetitions", "2"]
JUDGEMENT_WORDS = re.compile(
    r"\b(pass|passes|passed|fail|fails|failed|ok|within|exceeds|exceed|above|below|holds|meets|"
    r"violates|verified|valid|refused|faster|slower)\b",
    re.IGNORECASE,
)


class ScriptedClock:
    """Returns values so that the k-th timed call lasts durations[k] seconds: each timed call reads twice."""

    def __init__(self, durations_seconds):
        self.values = []
        now = 0.0
        for seconds in durations_seconds:
            self.values += [now, now + seconds]
            now += seconds

    def __call__(self) -> float:
        return self.values.pop(0)


def setup_for(clock=None, **kwargs) -> sp.PathsSetup:
    """A tiny CPU setup: the synthetic dimensions of the timing pass, a few calls per repetition."""
    values = dict(overrides=dict(tp.SYNTHETIC_DIMS), device=CPU, batch=2, repeats=3, warmup=2, repetitions=2)
    values.update(kwargs)
    if clock is not None:
        values["clock"] = clock
    return sp.PathsSetup(**values)


def entry(values_ms) -> dict:
    """A hand-written available entry whose repetitions are the given medians, in milliseconds."""
    return {
        "attention_ms": sorted(values_ms)[len(values_ms) // 2],
        "attention_ms_by_repetition": list(values_ms),
        "attention_ms_min": min(values_ms),
        "attention_ms_max": max(values_ms),
        "error": None,
    }


def unavailable(error: str) -> dict:
    return {
        "attention_ms": None, "attention_ms_by_repetition": None,
        "attention_ms_min": None, "attention_ms_max": None, "error": error,
    }


def good_document() -> dict:
    """A synthetic swa_paths/1 document in which the flex track could not run."""
    return {
        "schema": "swa_paths/1",
        "environment": dict(ENVIRONMENT),
        "gpu_shared": False,
        "backend": "efficient",
        "device": "cuda",
        "precision": "fp16",
        "batch": 16,
        "context": 1024,
        "repetitions": 3,
        "repeats": 30,
        "warmup": 10,
        "order": ["mha", "swa256/chunked", "swa256/flex"],
        "tracks": {
            "mha": entry([10.0, 11.0, 12.0]),
            "swa256/chunked": entry([9.0, 9.5, 10.5]),
            "swa256/flex": unavailable("RuntimeError: synthetic"),
        },
    }


# --- The run, end to end on CPU -------------------------------------------------------------------


@pytest.mark.filterwarnings(FLEX_WARNING)
def test_the_script_runs_end_to_end_on_cpu_and_writes_a_checked_document(tmp_path, capsys):
    out = tmp_path / "swa_paths.json"
    assert sp.main([*TINY, "--out", str(out)]) == 0
    doc = sp.read_swa_paths(out)
    sp.check_document(doc)
    assert doc["schema"] == "swa_paths/1" and doc["gpu_shared"] is False
    assert (doc["device"], doc["precision"], doc["backend"]) == ("cpu", "fp32", None)
    assert doc["order"] == ["mha", "swa256/chunked", "swa256/flex"]
    for key in ("mha", "swa256/chunked"):
        assert doc["tracks"][key]["error"] is None
        assert doc["tracks"][key]["attention_ms"] > 0
        assert len(doc["tracks"][key]["attention_ms_by_repetition"]) == 2
    # On this torch (2.15.0, CPU) the uncompiled flex path has no backward, so the track is recorded
    # as unavailable with that error. Pinned here as the CPU behaviour; see the report of T-60.
    flex_error = doc["tracks"]["swa256/flex"]["error"]
    assert flex_error is not None and "backward" in flex_error
    assert doc["tracks"]["swa256/flex"]["attention_ms"] is None
    assert "wrote" in capsys.readouterr().out


# --- The statistics, under a scripted clock -----------------------------------------------------


def test_a_repetition_times_its_timed_calls_and_drops_the_warmup_calls():
    # Warm-up calls read no clock: a break that timed them would read the scripted values in the wrong place.
    clock = ScriptedClock([0.001, 0.002, 0.009])
    durations = sp.time_repetition(sp.TRACKS[0], setup_for(clock, repeats=3, warmup=2, repetitions=1))
    assert durations == pytest.approx([1.0, 2.0, 9.0], rel=AGREEMENT_FP64)
    assert clock.values == []  # every scripted reading was used, and none more


def test_each_repetition_value_is_the_median_of_its_timed_calls_and_the_entry_is_the_median_of_those():
    per_track = {
        sp.MHA_KEY: [[1.0, 2.0, 9.0], [4.0, 5.0, 6.0]],  # medians 2 and 5; the mean of the first would be 4
        sp.CHUNKED_KEY: [[3.0, 3.0, 3.0], [7.0, 8.0, 15.0]],  # medians 3 and 8
        sp.FLEX_KEY: [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],  # medians 20 and 50
    }
    seen = {key: iter(reps) for key, reps in per_track.items()}
    tracks = sp.run_tracks(setup_for(repetitions=2), time_fn=lambda track, _setup: next(seen[track.key]))
    mha = tracks[sp.MHA_KEY]
    assert mha["attention_ms_by_repetition"] == [2.0, 5.0]
    assert mha["attention_ms"] == pytest.approx(3.5, rel=AGREEMENT_FP64)
    assert (mha["attention_ms_min"], mha["attention_ms_max"], mha["error"]) == (2.0, 5.0, None)
    chunked = tracks[sp.CHUNKED_KEY]
    assert chunked["attention_ms_by_repetition"] == [3.0, 8.0]
    assert chunked["attention_ms"] == pytest.approx(5.5, rel=AGREEMENT_FP64)
    assert (chunked["attention_ms_min"], chunked["attention_ms_max"]) == (3.0, 8.0)
    flex = tracks[sp.FLEX_KEY]
    assert flex["attention_ms_by_repetition"] == [20.0, 50.0]
    assert flex["attention_ms"] == pytest.approx(35.0, rel=AGREEMENT_FP64)


def test_the_tracks_run_round_robin_in_track_order_repetition_major():
    assert [(t.key, r) for t, r in sp.round_robin(sp.TRACKS, 2)] == [
        (key, rep) for rep in (0, 1) for key in ("mha", "swa256/chunked", "swa256/flex")
    ]
    calls = []

    def record(track, _setup):
        calls.append(track.key)
        return [1.0]

    sp.run_tracks(setup_for(repetitions=3), time_fn=record)
    assert calls == ["mha", "swa256/chunked", "swa256/flex"] * 3


# --- The opt paths the tracks really run -------------------------------------------------------


def test_each_track_runs_the_opt_path_it_names(monkeypatch):
    seen = []  # the opt path of the module at each call; the call itself is replaced, so nothing runs
    monkeypatch.setattr(sp, "_step", lambda attn, x, freqs, device: seen.append(getattr(attn, "opt_path", None)))
    expected = {sp.MHA_KEY: None, sp.CHUNKED_KEY: "chunked", sp.FLEX_KEY: "flex"}
    for track in sp.TRACKS:
        seen.clear()
        sp.time_repetition(track, setup_for(repeats=1, warmup=1))
        assert seen and set(seen) == {expected[track.key]}, (track.key, seen)


# --- A failure of the flex track is a result; any other failure is not swallowed ----------------


@pytest.mark.filterwarnings(FLEX_WARNING)
def test_a_flex_track_that_raises_is_recorded_and_not_attempted_again(tmp_path, monkeypatch, capsys):
    calls = []

    def boom(self, x, q, k, v):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(SlidingWindowAttention, "_flex", boom)
    out = tmp_path / "swa_paths.json"
    assert sp.main([*TINY, "--repetitions", "3", "--out", str(out)]) == 0
    doc = sp.read_swa_paths(out)
    sp.check_document(doc)
    flex = doc["tracks"]["swa256/flex"]
    assert flex["error"] == "RuntimeError: boom"
    assert (flex["attention_ms"], flex["attention_ms_by_repetition"], flex["attention_ms_min"], flex["attention_ms_max"]) == (
        None, None, None, None,
    )
    assert calls == [1]  # the first call failed; the track was not tried in the later repetitions
    for key in ("mha", "swa256/chunked"):
        assert doc["tracks"][key]["error"] is None and doc["tracks"][key]["attention_ms"] > 0
    assert "swa256/flex: not available: RuntimeError: boom" in capsys.readouterr().out


def test_a_failure_of_the_chunked_track_is_not_swallowed(monkeypatch):
    def boom(self, x, *args):
        raise RuntimeError("boom")

    monkeypatch.setattr(SlidingWindowAttention, "_chunked", boom)
    with pytest.raises(RuntimeError, match="boom"):
        sp.run_tracks(setup_for())


# --- The command line refuses what it must --------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--device", "cuda"],  # --gpu-shared is missing on CUDA
        ["--device", "cuda", "--gpu-shared", "true", "--expected-backend", "efficient"],
        ["--device", "cuda", "--gpu-shared", "false"],  # --expected-backend is missing on CUDA
        ["--gpu-shared", "false", "--expected-backend", "efficient"],  # --expected-backend is refused on CPU
    ],
    ids=["no-gpu-shared", "gpu-shared-true", "no-backend-on-cuda", "backend-on-cpu"],
)
def test_the_command_line_refuses_the_flags_it_must(tmp_path, args):
    out = tmp_path / "refused.json"
    with pytest.raises(SystemExit) as exc:
        sp.main(["--synthetic", *args, "--out", str(out)])
    assert exc.value.code == 2
    assert not out.exists()


# --- The document check ---------------------------------------------------------------------------


def test_a_good_document_with_an_unavailable_flex_track_passes_the_check():
    sp.check_document(good_document())


def _mutation(label, mutate, problem):
    return pytest.param(mutate, problem, id=label)


def _pop(*path):
    def mutate(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return mutate


def _put(value, *path):
    def mutate(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    "mutate, problem",
    [
        _mutation("wrong-schema", _put("swa_paths/0", "schema"), "schema 'swa_paths/0'"),
        _mutation("gpu-shared-true", _put(True, "gpu_shared"), "gpu_shared is True"),
        _mutation("gpu-shared-missing", _pop("gpu_shared"), "gpu_shared is None"),
        _mutation("gpu-shared-null", _put(None, "gpu_shared"), "gpu_shared is None"),
        _mutation("missing-track", _pop("tracks", "swa256/flex"), "'swa256/flex' is missing"),
        _mutation("missing-key", _pop("tracks", "mha", "attention_ms_min"), "missing attention_ms_min"),
        _mutation("zero-time", _put(0.0, "tracks", "mha", "attention_ms"), "mha: attention_ms must be"),
        _mutation("negative-time", _put(-1.0, "tracks", "swa256/chunked", "attention_ms_min"), "attention_ms_min must be"),
        _mutation("text-time", _put("9.5", "tracks", "swa256/chunked", "attention_ms"), "attention_ms must be"),
        _mutation("null-time-without-error", _put(None, "tracks", "mha", "attention_ms"), "attention_ms must be"),
        _mutation("bool-time", _put(True, "tracks", "mha", "attention_ms_max"), "attention_ms_max must be"),
        _mutation(
            "mha-carries-error", _put("RuntimeError: x", "tracks", "mha", "error"), "mha: carries an error",
        ),
        _mutation(
            "chunked-carries-error", _put("RuntimeError: x", "tracks", "swa256/chunked", "error"),
            "swa256/chunked: carries an error",
        ),
        _mutation(
            "unavailable-with-numbers", _put(2.0, "tracks", "swa256/flex", "attention_ms"),
            "must be null",
        ),
    ],
)
def test_the_check_refuses_each_bad_document_and_names_the_problem(mutate, problem):
    doc = good_document()
    mutate(doc)
    with pytest.raises(sp.SwaPathsError, match=re.escape(problem)):
        sp.check_document(doc)


def test_the_check_lists_every_problem_at_once():
    doc = good_document()
    doc["schema"] = "swa_paths/0"
    doc["gpu_shared"] = True
    with pytest.raises(sp.SwaPathsError) as exc:
        sp.check_document(doc)
    assert "schema" in str(exc.value) and "gpu_shared" in str(exc.value)


def test_the_reader_refuses_another_schema(tmp_path):
    path = tmp_path / "swa_paths.json"
    doc = good_document()
    doc["schema"] = "timing_pass/1"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(sp.SwaPathsError, match="timing_pass/1"):
        sp.read_swa_paths(path)


# --- The printed lines --------------------------------------------------------------------------


def test_the_printed_lines_carry_no_comparison_or_verdict_word(tmp_path, monkeypatch, capsys):
    def boom(self, x, q, k, v):
        raise RuntimeError("boom")

    monkeypatch.setattr(SlidingWindowAttention, "_flex", boom)
    assert sp.main([*TINY, "--out", str(tmp_path / "printed.json")]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(("mha:", "swa256/"))]
    assert len(lines) == 3
    assert lines[0].startswith("mha: median ") and lines[1].startswith("swa256/chunked: median ")
    assert lines[2] == "swa256/flex: not available: RuntimeError: boom"
    for line in lines:
        assert JUDGEMENT_WORDS.search(line) is None, line


# --- The GPU tier -------------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_script_runs_on_cuda_in_fp16_and_records_the_backend(tmp_path):
    out = tmp_path / "swa_paths.json"
    assert sp.main([
        "--synthetic", "--device", "cuda", "--gpu-shared", "false", "--expected-backend", "efficient",
        "--batch", "4", "--repeats", "2", "--warmup", "1", "--repetitions", "1", "--out", str(out),
    ]) == 0
    doc = sp.read_swa_paths(out)
    sp.check_document(doc)
    assert doc["precision"] == "fp16"
    assert doc["backend"] == "efficient"
    assert doc["device"] == "cuda"
