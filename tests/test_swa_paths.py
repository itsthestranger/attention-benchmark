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
from src import config
from src.attention.sliding_window import SlidingWindowAttention
from tests.tolerances import AGREEMENT_FP64
from tests.fixtures.fake_secondary import ENVIRONMENT

CPU = torch.device("cpu")
# A warning of FlexAttention outside torch.compile, filtered as in tests/test_sliding_window.py.
FLEX_WARNING = "ignore:flex_attention called without torch.compile"
TINY = ["--synthetic", "--batch", "4", "--repeats", "2", "--warmup", "1", "--repetitions", "2"]
JUDGEMENT_WORDS = re.compile(
    r"\b(pass|passes|passed|fail|fails|failed|ok|within|exceeds|exceed|above|below|holds|meets|"
    r"violates|verified|valid|refused|faster|slower|fastest|slowest|quicker|better|worse|best|worst|wins|beats)\b",
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
        "synthetic": False,
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
    # The flex track is a result either way (decision 249): torch 2.15.0 has no backward for the uncompiled
    # FlexAttention on CPU, so here it is recorded with its error; a torch that has one gives it times.
    # Neither is pinned: no test depends on a message of torch.
    flex = doc["tracks"]["swa256/flex"]
    if flex["error"] is None:
        assert flex["attention_ms"] > 0 and len(flex["attention_ms_by_repetition"]) == 2
    else:
        assert isinstance(flex["error"], str) and flex["error"]
        assert flex["attention_ms"] is None and flex["attention_ms_by_repetition"] is None
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


def test_the_entry_is_the_median_of_three_repetitions_not_their_mean():
    order = []  # per track: repetition values 1, 2 and 9; the mean would be 4

    def time_fn(track, _setup):
        order.append(track.key)
        return [{0: 1.0, 1: 2.0, 2: 9.0}[order.count(track.key) - 1]]

    tracks = sp.run_tracks(setup_for(repetitions=3), time_fn=time_fn)
    for key in sp.ORDER:
        assert tracks[key]["attention_ms_by_repetition"] == [1.0, 2.0, 9.0]
        assert tracks[key]["attention_ms"] == 2.0
        assert (tracks[key]["attention_ms_min"], tracks[key]["attention_ms_max"]) == (1.0, 9.0)


def test_a_repetition_makes_its_warmup_calls_and_its_timed_calls_at_the_batch_of_the_setup(monkeypatch):
    real_step, shapes = sp._step, []

    def counting(attn, x, freqs, device):
        shapes.append(tuple(x.shape))
        real_step(attn, x, freqs, device)

    monkeypatch.setattr(sp, "_step", counting)
    setup = setup_for(batch=3, repeats=4, warmup=2)
    durations = sp.time_repetition(sp.TRACKS[0], setup)
    assert len(durations) == 4
    assert len(shapes) == 2 + 4  # the warm-up calls are made, and not timed
    context, d_model = tp.SYNTHETIC_DIMS["model.context_length"], tp.SYNTHETIC_DIMS["model.d_model"]
    assert set(shapes) == {(3, context, d_model)}


@pytest.mark.parametrize("track", sp.TRACKS[:2], ids=lambda track: track.key)
def test_a_call_is_a_forward_and_a_backward(track):
    attn, x, freqs = sp._build(track, setup_for())
    assert x.grad is None and all(p.grad is None for p in attn.parameters())
    sp._step(attn, x, freqs, CPU)
    assert x.grad is not None and x.grad.abs().sum() > 0  # the gradient reached the module's input
    assert any(p.grad is not None for p in attn.parameters())


def test_every_call_runs_inside_the_scope_of_the_expected_backend(monkeypatch):
    import contextlib

    events = []

    @contextlib.contextmanager
    def scope(name, device):
        events.append(("enter", name, device))
        yield
        events.append(("exit",))

    monkeypatch.setattr(sp, "backend_scope", scope)
    monkeypatch.setattr(sp, "_step", lambda attn, x, freqs, device: events.append(("call",)))
    sp.time_repetition(sp.TRACKS[0], setup_for(repeats=2, warmup=1, expected_backend="efficient"))
    assert events == [("enter", "efficient", CPU), ("call",), ("call",), ("call",), ("exit",)]


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
    assert flex["error"] == "RuntimeError: boom (repetition 1 of 3)"
    assert (flex["attention_ms"], flex["attention_ms_by_repetition"], flex["attention_ms_min"], flex["attention_ms_max"]) == (
        None, None, None, None,
    )
    assert calls == [1]  # the first call failed; the track was not tried in the later repetitions
    for key in ("mha", "swa256/chunked"):
        assert doc["tracks"][key]["error"] is None and doc["tracks"][key]["attention_ms"] > 0
    printed = capsys.readouterr()
    assert "swa256/flex: not available: RuntimeError: boom (repetition 1 of 3)" in printed.out
    assert "swa256/flex: failed in repetition 1 of 3" in printed.err and "Traceback" in printed.err


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
        ["--device", "cuda", "--expected-backend", "efficient"],  # only --gpu-shared is missing on CUDA
        ["--device", "cuda", "--gpu-shared", "true", "--expected-backend", "efficient"],
        ["--device", "cuda", "--gpu-shared", "false"],  # --expected-backend is missing on CUDA
        ["--gpu-shared", "false", "--expected-backend", "efficient"],  # --expected-backend is refused on CPU
    ],
    ids=["no-gpu-shared", "gpu-shared-true", "no-backend-on-cuda", "backend-on-cpu"],
)
def test_the_command_line_refuses_the_flags_it_must(tmp_path, capsys, args):
    out = tmp_path / "refused.json"
    with pytest.raises(SystemExit) as exc:
        sp.main(["--synthetic", *args, "--out", str(out)])
    assert exc.value.code == 2
    assert not out.exists()
    capsys.readouterr()


def test_each_refusal_of_the_command_line_names_its_own_reason(tmp_path, capsys):
    def refused(*args):
        with pytest.raises(SystemExit):
            sp.main(["--synthetic", *args, "--out", str(tmp_path / "refused.json")])
        return capsys.readouterr().err

    assert "--gpu-shared true|false is required on CUDA" in refused("--device", "cuda", "--expected-backend", "efficient")
    assert "gpu_shared is true" in refused("--device", "cuda", "--gpu-shared", "true", "--expected-backend", "efficient")
    assert "--expected-backend is required on CUDA" in refused("--device", "cuda", "--gpu-shared", "false")
    assert "CUDA only" in refused("--gpu-shared", "false", "--expected-backend", "efficient")


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


# --- Review S6 (docs/review_s6.md): the record cannot contradict itself or what ran ---------------


@pytest.mark.parametrize(
    "mutate, problem",
    [
        _mutation("median-outside-range", _put(99.0, "tracks", "mha", "attention_ms"), "mha: attention_ms is 99.0"),
        _mutation("min-not-of-list", _put(9.5, "tracks", "mha", "attention_ms_min"), "attention_ms_min is 9.5"),
        _mutation("max-not-of-list", _put(11.5, "tracks", "mha", "attention_ms_max"), "attention_ms_max is 11.5"),
        _mutation("min-above-max", _put(12.5, "tracks", "mha", "attention_ms_min"), "attention_ms_min is 12.5"),
        _mutation("wrong-repetition-count", _put(4, "repetitions"), "3 repetition values, the document says 4"),
        _mutation("nan-time", _put(float("nan"), "tracks", "mha", "attention_ms"), "mha: attention_ms must be"),
        _mutation("infinite-time", _put(float("inf"), "tracks", "mha", "attention_ms_max"), "attention_ms_max must be"),
        _mutation(
            "bad-value-in-list", _put([10.0, "11.0", 12.0], "tracks", "mha", "attention_ms_by_repetition"),
            "attention_ms_by_repetition must be",
        ),
        _mutation("empty-list", _put([], "tracks", "mha", "attention_ms_by_repetition"), "attention_ms_by_repetition must be"),
        _mutation("no-environment", _pop("environment"), "no environment object"),
        _mutation("no-synthetic-flag", _pop("synthetic"), "synthetic is None"),
        _mutation("no-batch", _pop("batch"), "batch is None"),
        _mutation("text-context", _put("1024", "context"), "context is '1024'"),
        _mutation("zero-repeats", _put(0, "repeats"), "repeats is 0"),
        _mutation("negative-warmup", _put(-1, "warmup"), "warmup is -1"),
        _mutation("other-device", _put("mps", "device"), "device is 'mps'"),
        _mutation("precision-not-of-device", _put("fp32", "precision"), "precision is 'fp32' on cuda"),
        _mutation("no-backend-on-cuda", _put(None, "backend"), "backend is None on cuda"),
        _mutation("other-order", _put(["mha", "swa256/flex", "swa256/chunked"], "order"), "order is"),
        _mutation("empty-error", _put(" ", "tracks", "swa256/flex", "error"), "error must be text or null"),
        _mutation("error-not-text", _put(7, "tracks", "swa256/flex", "error"), "error must be text or null"),
        _mutation("extra-track", _put({"error": "x"}, "tracks", "swa256/dense"), "unknown tracks: 'swa256/dense'"),
        _mutation(
            "unavailable-with-repetitions", _put([1.0], "tracks", "swa256/flex", "attention_ms_by_repetition"),
            "must be null",
        ),
        _mutation("unavailable-with-min", _put(1.0, "tracks", "swa256/flex", "attention_ms_min"), "must be null"),
        _mutation("unavailable-with-max", _put(1.0, "tracks", "swa256/flex", "attention_ms_max"), "must be null"),
    ],
)
def test_the_check_refuses_a_record_that_contradicts_itself(mutate, problem):
    doc = good_document()
    mutate(doc)
    with pytest.raises(sp.SwaPathsError, match=re.escape(problem)):
        sp.check_document(doc)


def test_a_cpu_record_has_no_backend_and_is_fp32():
    doc = good_document()
    doc.update(device="cpu", precision="fp32", backend=None)
    sp.check_document(doc)
    doc["backend"] = "efficient"
    with pytest.raises(sp.SwaPathsError, match="backend is 'efficient' on cpu"):
        sp.check_document(doc)


def test_the_reader_names_a_missing_file_and_a_file_that_is_not_json(tmp_path):
    with pytest.raises(sp.SwaPathsError, match="no_such.json.*scripts.swa_paths"):
        sp.read_swa_paths(tmp_path / "no_such.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(sp.SwaPathsError, match="broken.json: not JSON"):
        sp.read_swa_paths(broken)


def test_the_defaults_are_the_measurement_of_the_t4_test():
    # tests/test_sliding_window.py::test_swa_attention_time_at_ctx_1024_is_below_mha: batch 16, 30 calls after
    # 10 warm-up calls, three round-robin repetitions (decision 247).
    args = sp._parser().parse_args(["--out", "x.json"])
    assert (args.batch, args.repeats, args.warmup, args.repetitions) == (16, 30, 10, 3)
    assert args.synthetic is False and args.device == "cpu"
    assert (sp.BATCH, sp.REPEATS, sp.WARMUP, sp.REPETITIONS) == (16, 30, 10, 3)


def _captured_run(monkeypatch, tmp_path, *flags):
    """main() with the timing replaced: the setup it built and the document it wrote."""
    seen = {}

    def fake_run(setup, **kwargs):
        seen["setup"] = setup
        return {key: entry([1.0] * setup.repetitions) for key in sp.ORDER}

    monkeypatch.setattr(sp, "run_tracks", fake_run)
    out = tmp_path / "captured.json"
    assert sp.main([*flags, "--out", str(out)]) == 0
    return seen["setup"], sp.read_swa_paths(out)


def test_without_synthetic_the_real_model_is_timed_at_the_plans_context(monkeypatch, tmp_path):
    setup, doc = _captured_run(monkeypatch, tmp_path)
    assert dict(setup.overrides) == {}
    assert (doc["synthetic"], doc["context"], doc["batch"]) == (False, 1024, 16)
    # The module the setup builds is the plan's: d_model 384 at context 1,024.
    attn, x, freqs = sp._build(sp.TRACKS[1], sp.PathsSetup(overrides=setup.overrides, device=CPU, batch=1))
    assert tuple(x.shape) == (1, 1024, 384) and attn.opt_path == "chunked"


def test_the_document_records_what_was_run(monkeypatch, tmp_path):
    setup, doc = _captured_run(
        monkeypatch, tmp_path, "--synthetic", "--batch", "5", "--repeats", "7", "--warmup", "2", "--repetitions", "4"
    )
    assert (setup.batch, setup.repeats, setup.warmup, setup.repetitions) == (5, 7, 2, 4)
    assert (doc["batch"], doc["repeats"], doc["warmup"], doc["repetitions"]) == (5, 7, 2, 4)
    assert doc["synthetic"] is True and doc["context"] == tp.SYNTHETIC_DIMS["model.context_length"]
    assert dict(setup.overrides) == dict(tp.SYNTHETIC_DIMS)


def test_the_clock_starts_and_stops_after_a_synchronise(monkeypatch):
    events = []
    monkeypatch.setattr(sp, "_sync", lambda device: events.append("sync"))
    monkeypatch.setattr(sp, "_step", lambda attn, x, freqs, device: events.append("call"))

    def clock():
        events.append("clock")
        return 0.0

    sp.time_repetition(sp.TRACKS[0], setup_for(clock, repeats=2, warmup=1))
    timed = ["sync", "clock", "call", "sync", "clock"]
    assert events == ["call", *timed, *timed]  # the warm-up call, then two timed calls


def test_the_forward_runs_under_fp16_autocast_on_cuda_and_under_none_on_cpu(monkeypatch):
    import contextlib

    assert isinstance(sp._autocast(CPU), contextlib.nullcontext)
    with pytest.warns(UserWarning) if not torch.cuda.is_available() else contextlib.nullcontext():
        scope = sp._autocast(torch.device("cuda"))
    assert isinstance(scope, torch.autocast) and scope.device == "cuda" and scope.fast_dtype == torch.float16

    events = []

    @contextlib.contextmanager
    def recording(device):
        events.append("enter")
        yield
        events.append("exit")

    class Module:
        def __call__(self, x, freqs, mode):
            events.append(("forward", mode))
            return x * 1.0, None

    monkeypatch.setattr(sp, "_autocast", recording)
    sp._step(Module(), torch.ones(2, requires_grad=True), None, CPU)
    assert events == ["enter", ("forward", "opt"), "exit"]  # the forward inside, the backward outside


def test_a_missing_flex_attention_is_recorded_like_any_other_flex_failure(monkeypatch):
    import src.attention.sliding_window as swmod

    monkeypatch.setattr(swmod, "_flex_importable", lambda: False)  # set_opt_path("flex") then raises ValueError
    tracks = sp.run_tracks(setup_for(repeats=1, warmup=0, repetitions=2))
    flex = tracks[sp.FLEX_KEY]
    assert flex["error"].startswith("ValueError: ") and flex["error"].endswith("(repetition 1 of 2)")
    assert flex["attention_ms"] is None
    assert tracks[sp.MHA_KEY]["error"] is None and tracks[sp.CHUNKED_KEY]["error"] is None


@pytest.mark.parametrize("named", ["chunked", "flex"])
def test_the_record_does_not_depend_on_the_path_the_variant_file_names(monkeypatch, named):
    # Phil sets opt_path in configs/variants/swa256.yaml after the T4 check (RUNBOOK §1.2); the record times both either way.
    real = config.load_variant

    def load(name):
        variant = real(name)
        if name == "swa256":
            variant["attention"]["opt_path"] = named
        return variant

    monkeypatch.setattr(config, "load_variant", load)
    assert config.resolve("swa256", init_seed=1, peak_lr=1e-3, overrides=dict(tp.SYNTHETIC_DIMS))["model"]["attention"]["opt_path"] == named
    for track, path in ((sp.TRACKS[1], "chunked"), (sp.TRACKS[2], "flex")):
        attn, _, _ = sp._build(track, setup_for())
        assert attn.opt_path == path


def test_a_flex_failure_in_a_later_repetition_drops_the_earlier_values_and_names_the_repetition(capsys):
    calls = []

    def time_fn(track, _setup):
        calls.append(track.key)
        if track.key == sp.FLEX_KEY and calls.count(sp.FLEX_KEY) == 3:
            raise RuntimeError("out of\nmemory")
        return [1.0]

    tracks = sp.run_tracks(setup_for(repetitions=4), time_fn=time_fn)
    flex = tracks[sp.FLEX_KEY]
    assert flex["error"] == "RuntimeError: out of\nmemory (repetition 3 of 4)"
    assert (flex["attention_ms"], flex["attention_ms_by_repetition"]) == (None, None)
    assert calls.count(sp.FLEX_KEY) == 3 and calls.count(sp.MHA_KEY) == 4  # not tried in repetition 4
    assert "swa256/flex: failed in repetition 3 of 4" in capsys.readouterr().err
    assert sp.one_line(flex["error"]) == "RuntimeError: out of memory (repetition 3 of 4)"


# --- The printed lines --------------------------------------------------------------------------


def test_the_printed_lines_carry_no_comparison_or_verdict_word(tmp_path, monkeypatch, capsys):
    def boom(self, x, q, k, v):
        raise RuntimeError("boom")

    monkeypatch.setattr(SlidingWindowAttention, "_flex", boom)
    assert sp.main([*TINY, "--out", str(tmp_path / "printed.json")]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith(("mha:", "swa256/"))]
    assert len(lines) == 3
    assert lines[0].startswith("mha: median ") and lines[1].startswith("swa256/chunked: median ")
    assert lines[2] == "swa256/flex: not available: RuntimeError: boom (repetition 1 of 2)"
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
