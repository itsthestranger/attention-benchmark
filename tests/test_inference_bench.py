"""Inference benchmark (plan §8.6, §10.4): src/eval/inference.py and scripts/bench.py.

cpu tier: tiny synthetic models, tiny grids, a fake clock where timing is checked. The
analytic KV bytes are checked against the plan's §3.2 table on the full-size models.
gpu tier: the measured KV allocation against the analytic value for every variant, and
bench on CUDA in fp16. Both skip cleanly without CUDA (tests/conftest.py).
"""

from __future__ import annotations

import json

import pytest
import torch

from scripts import bench
from scripts import make_tables as mt
from src.eval import inference as inf
from tests.fixtures.fake_campaign import fake_campaign
from tests.tolerances import KV_BYTES_MEASURED_REL
from tests.fixtures.commitments import analyse_committed

ALL = bench.ALL_VARIANTS
CPU = torch.device("cpu")

# Plan §3.2, KV bytes per token per layer at fp16. SWA is listed at its uncapped rate.
PLAN_KV_PER_LAYER = {
    "mha": 1536,
    "gqa4": 768,
    "gqa2": 384,
    "mqa": 192,
    "mla192": 432,
    "mla96": 240,
    "swa256": 1536,
}
PLAN_LAYERS = 8
PLAN_WINDOW = 256

SMALL_GRID = [
    "--contexts", "8", "16",
    "--decode-batches", "1", "2",
    "--decode-caches", "8", "16",
    "--warmup", "1",
    "--reps", "2",
    "--skip-max-batch",
]


class FakeClock:
    """A clock in seconds that only moves when told to."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _schema_keys(doc: dict) -> None:
    assert set(doc) == {
        "schema", "variant", "mode", "precision", "backend", "environment", "gpu_shared",
        "kv_bytes_per_token", "prefill", "decode", "max_batch_at_8k",
    }
    assert doc["schema"] == "bench/1"
    assert isinstance(doc["environment"], dict)
    assert set(doc["kv_bytes_per_token"]) == {"analytic", "measured", "measured_tokens"}
    for row in doc["prefill"]:
        assert set(row) == {"context", "batch", "latency_ms", "tok_per_s", "peak_bytes"}
        assert isinstance(row["context"], int) and isinstance(row["batch"], int)
        assert isinstance(row["latency_ms"], float) and row["latency_ms"] > 0
        assert isinstance(row["tok_per_s"], float) and row["tok_per_s"] > 0
    for row in doc["decode"]:
        assert set(row) == {"batch", "cache", "ms_per_token", "tok_per_s", "peak_bytes"}
        assert isinstance(row["batch"], int) and isinstance(row["cache"], int)
        assert isinstance(row["ms_per_token"], float) and row["ms_per_token"] > 0
        assert isinstance(row["tok_per_s"], float) and row["tok_per_s"] > 0


# --- Schema, end to end on CPU ----------------------------------------------------------


def test_bench_runs_end_to_end_for_every_variant_in_both_tracks(tmp_path):
    code = bench.main(["--synthetic", "--variants", *ALL, "--with-ref", "--out-dir", str(tmp_path), *SMALL_GRID])
    assert code == 0
    names = sorted(p.name for p in tmp_path.glob("bench_*.json"))
    assert names == sorted(f"bench_{v}_{m}.json" for v in ALL for m in ("opt", "ref"))
    for variant in ALL:
        for mode in ("opt", "ref"):
            doc = json.loads((tmp_path / f"bench_{variant}_{mode}.json").read_text())
            _schema_keys(doc)
            assert doc["variant"] == variant and doc["mode"] == mode
            assert doc["precision"] == "fp32"  # CPU
            assert doc["gpu_shared"] is False
            assert doc["backend"] is None  # no backend is enforced on CPU
            # CPU cannot measure these: they are null, not zero.
            assert doc["kv_bytes_per_token"]["measured"] is None
            assert doc["kv_bytes_per_token"]["measured_tokens"] is None
            assert doc["max_batch_at_8k"] is None
            assert all(row["peak_bytes"] is None for row in doc["prefill"] + doc["decode"])
            assert [(r["context"], r["batch"]) for r in doc["prefill"]] == [(8, 1), (16, 1)]
            assert [(r["batch"], r["cache"]) for r in doc["decode"]] == [(1, 8), (1, 16), (2, 8), (2, 16)]


def test_bench_writes_the_analytic_kv_bytes_of_the_synthetic_model(tmp_path):
    # Synthetic model: 2 layers, 8 query heads of dim 8. MHA: 2 x 8 x 8 x 2 bytes per token per layer.
    bench.main(["--synthetic", "--variants", "mha", "mqa", "swa256", "--out-dir", str(tmp_path), *SMALL_GRID])
    per_layer = {"mha": 256, "mqa": 32, "swa256": 256}
    for variant, bytes_per_layer in per_layer.items():
        doc = json.loads((tmp_path / f"bench_{variant}_opt.json").read_text())
        assert doc["kv_bytes_per_token"]["analytic"] == 2 * bytes_per_layer


def test_bench_refuses_gpu_shared_true(tmp_path):
    with pytest.raises(SystemExit) as exc:
        bench.main(["--synthetic", "--variants", "mha", "--gpu-shared", "true", "--out-dir", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.glob("*.json"))
    with pytest.raises(bench.BenchError, match="gpu_shared is true"):
        bench.check_gpu_shared(True, CPU)


def test_the_expected_backend_is_required_on_cuda_and_refused_on_cpu(tmp_path):
    cuda = torch.device("cuda")
    with pytest.raises(bench.BenchError, match="required on CUDA"):
        inf.check_backend_flag(None, cuda)
    assert inf.check_backend_flag("efficient", cuda) == "efficient"
    with pytest.raises(bench.BenchError, match="CUDA only"):
        inf.check_backend_flag("efficient", CPU)
    assert inf.check_backend_flag(None, CPU) is None
    # The command line: missing on CUDA, given on CPU, and unknown names, all refused before any work.
    with pytest.raises(SystemExit):
        bench.main(["--synthetic", "--device", "cuda", "--gpu-shared", "false", "--variants", "mha",
                    "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        bench.main(["--synthetic", "--variants", "mha", "--expected-backend", "efficient",
                    "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        bench.main(["--synthetic", "--variants", "mha", "--expected-backend", "bogus",
                    "--out-dir", str(tmp_path)])
    assert not list(tmp_path.glob("*.json"))


def test_unknown_backend_names_are_refused():
    with pytest.raises(inf.BenchError, match="unknown SDPA backend"):
        inf.check_backend_name("bogus")
    assert inf.check_backend_name(None) is None


def test_the_backend_mapping_covers_the_declared_names():
    from src.attention.base import SDPA_BACKENDS

    assert set(inf.SDPA_MEMBERS) == set(SDPA_BACKENDS)


def test_the_max_batch_fill_uses_the_small_chunk():
    model = inf.build_model_for_benchmark("mha", context_length=64, overrides=bench.SYNTHETIC_DIMS)
    lengths = []
    original = model.forward

    def recording(idx, **kwargs):
        lengths.append(idx.shape[1])
        return original(idx, **kwargs)

    model.forward = recording
    inf.try_decode_at(model, batch=2, cache=40, device=CPU)
    assert inf.MAX_BATCH_FILL_CHUNK == 8
    # Five fill forwards of at most the chunk size, then the single decode step.
    assert lengths == [8, 8, 8, 8, 8, 1]


def test_bench_on_cuda_requires_the_gpu_shared_flag(tmp_path):
    # Only the check is exercised: no CUDA device is touched.
    with pytest.raises(bench.BenchError, match="required on CUDA"):
        bench.check_gpu_shared(None, torch.device("cuda"))
    assert bench.check_gpu_shared(False, torch.device("cuda")) is False
    assert bench.check_gpu_shared(None, CPU) is False
    with pytest.raises(SystemExit):
        bench.main(["--synthetic", "--device", "cuda", "--variants", "mha", "--out-dir", str(tmp_path)])


def test_script_kv_tolerance_equals_the_tests_constant():
    assert bench.KV_BYTES_TOLERANCE_REL == KV_BYTES_MEASURED_REL


def test_the_script_does_not_import_from_tests():
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(bench.__file__).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not [n for n in names if n.split(".")[0] == "tests"]


# --- Analytic KV bytes against plan §3.2 -----------------------------------------------


@pytest.mark.parametrize("variant", ALL)
def test_analytic_kv_bytes_match_the_plan_table_at_full_size(variant):
    model = inf.build_model_for_benchmark(variant, context_length=1024)
    per_layer = PLAN_KV_PER_LAYER[variant]
    assert model.layers[0].attn.kv_bytes_per_token == per_layer
    assert inf.analytic_kv_bytes_per_token(model) == PLAN_LAYERS * per_layer
    if variant == "swa256":
        assert inf.max_cache_tokens(model) == PLAN_WINDOW
        assert inf.analytic_kv_cache_bytes(model, 10**6) == PLAN_LAYERS * per_layer * PLAN_WINDOW
        assert inf.analytic_kv_cache_bytes(model, 100) == PLAN_LAYERS * per_layer * 100
    else:
        assert inf.max_cache_tokens(model) is None
        assert inf.analytic_kv_cache_bytes(model, 1000) == PLAN_LAYERS * per_layer * 1000


def test_measured_kv_is_none_off_cuda():
    model = inf.build_model_for_benchmark("mha", context_length=32, overrides=bench.SYNTHETIC_DIMS)
    assert inf.measure_kv_allocation(model, device=CPU, n_tokens=16) is None


# --- Timing discipline -------------------------------------------------------------------


def test_warmup_measurements_are_discarded():
    clock = FakeClock()
    calls = []

    def fn():
        calls.append(1)
        # The first three calls are slow (warm-up); the four measured calls take half a second.
        clock.now += 100.0 if len(calls) <= 3 else 0.5

    timed = inf.time_calls(fn, warmup=3, reps=4, device=CPU, clock=clock)
    assert len(calls) == 7
    assert timed.ms == (500.0, 500.0, 500.0, 500.0)
    assert timed.mean_ms == 500.0
    assert timed.peak_bytes is None


def test_time_calls_validates_its_counts():
    with pytest.raises(ValueError):
        inf.time_calls(lambda: None, warmup=-1, reps=1, device=CPU)
    with pytest.raises(ValueError):
        inf.time_calls(lambda: None, warmup=0, reps=0, device=CPU)


def test_prefill_and_decode_report_the_requested_shapes():
    model = inf.build_model_for_benchmark("gqa2", context_length=32, overrides=bench.SYNTHETIC_DIMS)
    pre = inf.measure_prefill(model, mode="opt", context=16, batch=2, device=CPU, warmup=1, reps=2)
    dec = inf.measure_decode(model, mode="opt", batch=2, cache=12, device=CPU, warmup=1, reps=2)
    assert (pre["context"], pre["batch"]) == (16, 2)
    assert pre["tok_per_s"] == 2 * 16 * 1000.0 / pre["latency_ms"]
    assert (dec["batch"], dec["cache"]) == (2, 12)
    assert dec["tok_per_s"] == 2 * 1000.0 / dec["ms_per_token"]


def test_fill_state_holds_the_requested_length_capped_by_the_window():
    model = inf.build_model_for_benchmark("swa256", context_length=64, overrides=bench.SYNTHETIC_DIMS)
    window = model.layers[0].attn.window
    state = inf.fill_state(model, mode="opt", batch=1, length=40, device=CPU, chunk=16)
    assert state.pos == 40
    assert all(cache.length == min(40, window) for cache in state.caches)
    grouped = inf.build_model_for_benchmark("gqa2", context_length=64, overrides=bench.SYNTHETIC_DIMS)
    state = inf.fill_state(grouped, mode="opt", batch=1, length=40, device=CPU, chunk=16)
    assert all(cache.length == 40 for cache in state.caches)


# --- Maximum batch before OOM ------------------------------------------------------------


def _fits_up_to(k: int):
    def try_batch(b: int) -> None:
        if b > k:
            raise torch.OutOfMemoryError(f"fake out of memory at batch {b}")

    return try_batch


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 8, 100, 1000, 4095])
def test_max_batch_search_returns_the_largest_batch_that_fits(k):
    assert inf.max_batch_before_oom(_fits_up_to(k)) == k


def test_max_batch_is_zero_when_batch_one_does_not_fit():
    assert inf.max_batch_before_oom(_fits_up_to(0)) == 0


def test_max_batch_propagates_errors_that_are_not_out_of_memory():
    def broken(b: int) -> None:
        if b >= 4:
            raise RuntimeError("not an out-of-memory error")

    with pytest.raises(RuntimeError, match="not an out-of-memory"):
        inf.max_batch_before_oom(broken)


def test_max_batch_refuses_to_report_when_the_limit_still_fits():
    with pytest.raises(inf.BenchError, match="still fits"):
        inf.max_batch_before_oom(_fits_up_to(10**9), limit=64)


# --- Readers and the consumer ---------------------------------------------------------------


def _bench_doc(variant: str, mode: str, *, gpu_shared: bool = False, tok: float = 1234.5) -> dict:
    return {
        "schema": "bench/1",
        "variant": variant,
        "mode": mode,
        "precision": "fp16",
        "environment": {},
        "gpu_shared": gpu_shared,
        "kv_bytes_per_token": {"analytic": 1, "measured": None},
        "prefill": [],
        "decode": [
            {"batch": 1, "cache": 4096, "ms_per_token": 1.0, "tok_per_s": 1000.0, "peak_bytes": None},
            {"batch": 32, "cache": 4096, "ms_per_token": 2.0, "tok_per_s": tok, "peak_bytes": None},
        ],
        "max_batch_at_8k": None,
    }


def test_decode_reader_feeds_the_headline_column(tmp_path):
    for variant, tok in (("mha", 1000.0), ("mqa", 2500.0)):
        bench.write_bench_file(tmp_path, _bench_doc(variant, "opt", tok=tok))
    bench.write_bench_file(tmp_path, _bench_doc("gqa4", "ref", tok=9999.0))  # ref is not the headline

    mapping = bench.read_decode_tok_per_s(tmp_path)
    assert mapping == {"mha": 1000.0, "mqa": 2500.0}

    analysis = analyse_committed(fake_campaign())
    rows = {r["variant"]: r for r in mt.headline_rows(analysis, decode_tok_per_s=mapping)}
    column = "decode tok/s (batch 32, cache 4k)"
    assert rows["mha"][column] == "1000.0"
    assert rows["mqa"][column] == "2500.0"
    assert rows["gqa4"][column] == mt.BLANK


def test_decode_reader_refuses_a_file_recorded_with_a_shared_gpu(tmp_path):
    bench.write_bench_file(tmp_path, _bench_doc("mha", "opt", gpu_shared=True))
    with pytest.raises(bench.BenchError, match="gpu_shared true"):
        bench.read_decode_tok_per_s(tmp_path)


def test_decode_reader_refuses_a_wrong_schema(tmp_path):
    doc = _bench_doc("mha", "opt")
    doc["schema"] = "bench/0"
    bench.write_bench_file(tmp_path, doc)
    with pytest.raises(bench.BenchError, match="schema"):
        bench.read_decode_tok_per_s(tmp_path)


# --- GPU tier ------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("variant", ALL)
def test_measured_kv_bytes_agree_with_analytic_during_decode(variant):
    device = torch.device("cuda")
    model = inf.build_model_for_benchmark(variant, context_length=1024).to(device).half()
    total, n = inf.measure_kv_allocation(model, device=device, n_tokens=1024, backend="efficient")
    expected = inf.analytic_kv_cache_bytes(model, n)
    assert expected > 0
    assert abs(total - expected) / expected <= KV_BYTES_MEASURED_REL


@pytest.mark.gpu
def test_bench_runs_on_cuda_in_fp16(tmp_path):
    code = bench.main([
        "--synthetic", "--device", "cuda", "--gpu-shared", "false", "--expected-backend", "efficient",
        "--variants", "mha", "mqa",
        "--contexts", "16", "--decode-batches", "1", "--decode-caches", "16",
        "--max-batch-cache", "8192", "--warmup", "1", "--reps", "2",
        "--out-dir", str(tmp_path),
    ])
    assert code == 0
    for variant in ("mha", "mqa"):
        doc = json.loads((tmp_path / f"bench_{variant}_opt.json").read_text())
        _schema_keys(doc)
        assert doc["precision"] == "fp16"
        assert doc["gpu_shared"] is False
        assert doc["backend"] == "efficient"
        assert doc["kv_bytes_per_token"]["measured"] is not None
        assert doc["kv_bytes_per_token"]["measured_tokens"] == 1024
        assert isinstance(doc["max_batch_at_8k"], int) and 1 <= doc["max_batch_at_8k"] < inf.MAX_BATCH_LIMIT
        assert all(row["peak_bytes"] is not None for row in doc["prefill"] + doc["decode"])
