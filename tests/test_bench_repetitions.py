"""T-49: the benchmark's repetitions (plan §8.6 as amended by decision 225), scripts/bench.py.

cpu tier: tiny synthetic models and grids. The measurement functions are the real ones unless a
test says it injects values: the injected values are SYNTHETIC and chosen so the statistics can be
checked by hand.
"""

from __future__ import annotations

import json

import pytest
import torch

from scripts import bench

CPU = torch.device("cpu")
SMALL = [
    "--synthetic", "--contexts", "8", "--decode-batches", "1", "--decode-caches", "8",
    "--warmup", "1", "--reps", "1", "--skip-max-batch",
]


# --- Round-robin order -----------------------------------------------------------------------


def test_repetitions_run_round_robin_over_variants_and_modes(tmp_path, monkeypatch):
    """Repetition 1 of every (variant, mode), then repetition 2 of every one, then repetition 3."""
    order: list[tuple[int, str, str]] = []
    current = {"variant": None}
    real_build = bench.build_model_for_benchmark

    def build(variant, **kwargs):
        current["variant"] = variant
        return real_build(variant, **kwargs)

    real_prefill = bench.measure_prefill

    def prefill(model, **kwargs):
        # The first prefill point of each track-repetition is the first call after its build.
        if kwargs["context"] == 8:
            order.append((current["variant"], kwargs["mode"]))
        return real_prefill(model, **kwargs)

    monkeypatch.setattr(bench, "build_model_for_benchmark", build)
    monkeypatch.setattr(bench, "measure_prefill", prefill)
    code = bench.main([*SMALL, "--variants", "mha", "mqa", "--with-ref", "--repetitions", "3",
                       "--out-dir", str(tmp_path)])
    assert code == 0
    tracks = [("mha", "opt"), ("mha", "ref"), ("mqa", "opt"), ("mqa", "ref")]
    expected = [track for _ in range(3) for track in tracks]
    assert order == expected


def test_each_track_is_built_once_per_repetition_in_round_robin_order(tmp_path, monkeypatch):
    builds: list[str] = []
    real_build = bench.build_model_for_benchmark

    def build(variant, **kwargs):
        builds.append(variant)
        return real_build(variant, **kwargs)

    monkeypatch.setattr(bench, "build_model_for_benchmark", build)
    bench.main([*SMALL, "--variants", "mha", "mqa", "--with-ref", "--repetitions", "2", "--out-dir", str(tmp_path)])
    assert builds == ["mha", "mha", "mqa", "mqa", "mha", "mha", "mqa", "mqa"]


# --- Once per (variant, mode): the KV check and the maximum batch ----------------------------


def test_the_kv_check_runs_once_per_variant_and_mode(tmp_path, monkeypatch):
    calls: list[str] = []
    real = bench.measure_kv_allocation

    def kv(model, **kwargs):
        calls.append("kv")
        return real(model, **kwargs)

    monkeypatch.setattr(bench, "measure_kv_allocation", kv)
    bench.main([*SMALL, "--variants", "mha", "mqa", "--with-ref", "--repetitions", "3", "--out-dir", str(tmp_path)])
    assert len(calls) == 4  # two variants, two modes, once each: not 4 x 3


def test_the_max_batch_search_runs_once_per_variant_and_mode_in_the_first_repetition(monkeypatch):
    """The search needs CUDA, so the track is put on its CUDA path by hand, with the search replaced.

    Only the flag that selects the search is set; the model is the CPU model, and nothing is timed on
    CUDA. The search is called in repetition 1 only.
    """
    searches: list[int] = []

    def fake_search(try_batch, **kwargs):
        searches.append(1)
        return 7

    monkeypatch.setattr(bench, "max_batch_before_oom", fake_search)
    track = bench._Track("mha", "opt", device=CPU, contexts=(8,), decode_batches=(1,), decode_caches=(8,),
                         warmup=0, reps=1, overrides=bench.SYNTHETIC_DIMS, with_max_batch=True)
    track.cuda = True  # the search's condition; see the docstring
    for _ in range(3):
        model = track.build()
        track.repetition(model)
    assert len(searches) == 1
    assert track.max_batch == 7
    doc = track.document()
    assert doc["max_batch_at_8k"] == 7
    assert doc["repetitions"] == 3


# --- Median, minimum, maximum and the list of repetitions -------------------------------------


def test_median_min_max_and_list_are_the_injected_repetition_values(tmp_path, monkeypatch):
    """Injected per-repetition values, SYNTHETIC: latency 3, 1, 2 ms and tok/s 30, 10, 20 in repetitions 1, 2, 3."""
    latency = [3.0, 1.0, 2.0]
    tok = [30.0, 10.0, 20.0]
    decode_ms = [0.5, 0.25, 0.75]
    decode_tok = [2000.0, 4000.0, 3000.0]
    calls = {"prefill": 0, "decode": 0}

    def prefill(model, **kwargs):
        i = calls["prefill"]
        calls["prefill"] += 1
        return {"context": kwargs["context"], "batch": kwargs["batch"], "latency_ms": latency[i],
                "tok_per_s": tok[i], "peak_bytes": None}

    def decode(model, **kwargs):
        i = calls["decode"]
        calls["decode"] += 1
        return {"batch": kwargs["batch"], "cache": kwargs["cache"], "ms_per_token": decode_ms[i],
                "tok_per_s": decode_tok[i], "peak_bytes": None}

    monkeypatch.setattr(bench, "measure_prefill", prefill)
    monkeypatch.setattr(bench, "measure_decode", decode)
    assert bench.main([*SMALL, "--variants", "mha", "--repetitions", "3", "--out-dir", str(tmp_path)]) == 0
    doc = json.loads((tmp_path / "bench_mha_opt.json").read_text(encoding="utf-8"))

    row = doc["prefill"][0]
    assert row["latency_ms"] == 2.0 and row["tok_per_s"] == 20.0
    assert row["latency_ms_by_repetition"] == [3.0, 1.0, 2.0]
    assert row["tok_per_s_by_repetition"] == [30.0, 10.0, 20.0]
    assert (row["latency_ms_min"], row["latency_ms_max"]) == (1.0, 3.0)
    assert (row["tok_per_s_min"], row["tok_per_s_max"]) == (10.0, 30.0)
    assert row["peak_bytes"] is None

    point = doc["decode"][0]
    assert point["ms_per_token"] == 0.5 and point["tok_per_s"] == 3000.0
    assert point["ms_per_token_by_repetition"] == [0.5, 0.25, 0.75]
    assert point["tok_per_s_by_repetition"] == [2000.0, 4000.0, 3000.0]
    assert (point["ms_per_token_min"], point["ms_per_token_max"]) == (0.25, 0.75)
    assert (point["tok_per_s_min"], point["tok_per_s_max"]) == (2000.0, 4000.0)


def test_the_aggregates_of_an_even_number_of_repetitions_are_the_mean_of_the_middle_two():
    rows = [
        {"context": 8, "batch": 1, "latency_ms": v, "tok_per_s": 8000.0 / v, "peak_bytes": p}
        for v, p in ((4.0, 10), (2.0, 30))
    ]
    out = bench.aggregate_prefill(rows)
    assert out["latency_ms"] == 3.0  # median of [4, 2]
    assert out["peak_bytes"] == 30  # the maximum over the repetitions
    assert out["latency_ms_min"] == 2.0 and out["latency_ms_max"] == 4.0


def test_a_peak_that_is_missing_in_any_repetition_is_none():
    rows = [
        {"batch": 1, "cache": 8, "ms_per_token": 1.0, "tok_per_s": 1000.0, "peak_bytes": 5},
        {"batch": 1, "cache": 8, "ms_per_token": 1.0, "tok_per_s": 1000.0, "peak_bytes": None},
    ]
    assert bench.aggregate_decode(rows)["peak_bytes"] is None


# --- --repetitions ----------------------------------------------------------------------------


def test_one_repetition_gives_min_median_and_max_equal(tmp_path):
    assert bench.main([*SMALL, "--variants", "mha", "--repetitions", "1", "--out-dir", str(tmp_path)]) == 0
    doc = json.loads((tmp_path / "bench_mha_opt.json").read_text(encoding="utf-8"))
    assert doc["repetitions"] == 1
    for row in doc["prefill"]:
        assert row["latency_ms"] == row["latency_ms_min"] == row["latency_ms_max"]
        assert row["tok_per_s"] == row["tok_per_s_min"] == row["tok_per_s_max"]
        assert len(row["latency_ms_by_repetition"]) == 1
    for row in doc["decode"]:
        assert row["ms_per_token"] == row["ms_per_token_min"] == row["ms_per_token_max"]
        assert row["tok_per_s"] == row["tok_per_s_min"] == row["tok_per_s_max"]
        assert len(row["tok_per_s_by_repetition"]) == 1


@pytest.mark.parametrize("value", ["0", "-1"])
def test_zero_or_negative_repetitions_are_refused_before_any_work(tmp_path, value):
    with pytest.raises(SystemExit) as exc:
        bench.main([*SMALL, "--variants", "mha", "--repetitions", value, "--out-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.glob("*.json"))


def test_the_default_number_of_repetitions_is_three():
    parser = bench._parser()
    args = parser.parse_args(["--out-dir", "x"])
    assert args.repetitions == 3 == bench.DEFAULT_REPETITIONS


def test_bench_variant_mode_runs_the_given_repetitions_for_one_pair():
    doc = bench.bench_variant_mode(
        "mha", "opt", device=CPU, contexts=(8,), decode_batches=(1,), decode_caches=(8,),
        overrides=bench.SYNTHETIC_DIMS, warmup=1, reps=1, repetitions=2,
    )
    assert doc["schema"] == "bench/2" and doc["repetitions"] == 2
    assert len(doc["prefill"][0]["latency_ms_by_repetition"]) == 2
    with pytest.raises(ValueError, match="repetitions"):
        bench.bench_variant_mode("mha", "opt", device=CPU, repetitions=0)


def test_a_track_holds_no_model_between_repetitions():
    """The caller builds and releases each repetition's model; the track keeps only numbers."""
    track = bench._Track("mha", "opt", device=CPU, contexts=(8,), decode_batches=(1,), decode_caches=(8,),
                         warmup=0, reps=1, overrides=bench.SYNTHETIC_DIMS, with_max_batch=False)
    model = track.build()
    track.repetition(model)
    assert all(not isinstance(v, torch.nn.Module) for v in vars(track).values())
