"""SYNTHETIC inputs for the plot tests. Nothing here comes from a real run.

Every number in this module is made up with seeded `numpy.random.default_rng`
generators (never the global RNG):

- `final_metrics["val"]` (the per-bucket validation block of
  `src/eval/lm.py`) and `final_metrics["val_curve"]` (the evaluations made
  during a run) added to completed results through `validate_manifest`;
- bench/2 records, probe records and retrieval records with the shapes the
  brief specifies;
- two constructed cases with hand-set numbers, for checks that need exact values.

Manifests are built with `fake_manifest`, so they pass the same validation as
real ones. Nothing here reads or writes results.
"""

from __future__ import annotations

import copy
import zlib

import numpy as np

from src.utils import manifest as mf
from tests.fixtures.fake_manifests import fake_manifest

# The Arm N runs (data-order runs and the null control) come from fake_campaign since T-32.
BUCKETS = ((0, 64), (64, 256), (256, 512), (512, 1024))  # plan §10.3
# Synthetic structure of the per-bucket losses: later context is a little harder.
BUCKET_OFFSETS = (0.12, 0.07, 0.04, 0.02)
# A synthetic variant-specific bump in the late buckets, so that the context plot has a shape.
CONTEXT_SHIFT = {"swa256": (0.0, 0.0, 0.03, 0.06), "mqa": (0.0, 0.0, 0.01, 0.01)}
BUCKET_NOISE = 0.002
CURVE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
CURVE_EXCESS = 0.5  # the first evaluation sits this far above the final loss, synthetic

BENCH_VARIANTS = ("mha", "gqa4", "mqa", "mla192", "gqa2", "mla96", "swa256")
PROBE_TASKS = ("mqar", "multihop")
PROBE_DIFFICULTIES = (4, 8, 16, 32, 64, 128, 256)
PROBE_SEEDS = tuple(range(1, 11))
RETRIEVAL_LENGTHS = (256, 512, 1024, 2048)
RETRIEVAL_DISTRACTORS = (0, 4)
RETRIEVAL_SEEDS = (1, 2, 3)
SYNTHETIC = {"synthetic": True}


def _key(text: str) -> int:
    """A stable integer for a name, so that each variant has its own noise stream."""
    return zlib.crc32(text.encode("utf-8"))


def _tokens(label: str) -> int:
    if not (label.endswith("M") and label[:-1].isdigit()):
        raise ValueError(f"synthetic fixtures place budgets as NM only, got {label!r}")
    return int(label[:-1]) * 1_000_000


def budget_tokens(name: str) -> int:
    """The tokens a result ends at: its branch budget, or its run budget."""
    parsed = mf.parse_run_name(name)
    return _tokens(parsed.branch if parsed.branch is not None else parsed.tokens)


# --- Per-bucket and per-curve fields --------------------------------------------------


def val_block(loss: float, bucket_losses) -> dict:
    """A `src.eval.lm.evaluate`-shaped block with the given per-bucket losses (synthetic)."""
    buckets = [
        {"low": lo, "high": hi, "n_tokens": 1024, "loss": float(value)}
        for (lo, hi), value in zip(BUCKETS, bucket_losses)
    ]
    window = {"n_tokens": 100, "loss": None}
    return {
        "loss": float(loss),
        "n_tokens": 4096,
        "n_windows": 4,
        "buckets": buckets,
        "late_context": {"min_c": 512, "n_tokens": 1024, "loss": float(bucket_losses[-1])},
        "icl": {
            "late": {"low": 450, "high": 550, **window, "loss": float(loss) + 0.1},
            "early": {"low": 40, "high": 60, **window, "loss": float(loss) + 0.2},
            "score": -0.1,
        },
    }


def curve_block(final_loss: float, budget: int, total_steps: int) -> list[dict]:
    """A `val_curve` ending exactly at the final loss, at the budget's last token (synthetic)."""
    return [
        {
            "step": round(total_steps * fraction),
            "tokens": round(budget * fraction),
            "val_loss": float(final_loss + CURVE_EXCESS * (1.0 - fraction)),
        }
        for fraction in CURVE_FRACTIONS
    ]


def eval_fields(manifest: dict) -> dict:
    """The `val` and `val_curve` fields of one completed result, synthetic but consistent with its loss."""
    final = float(manifest["final_metrics"]["val_loss"])
    parsed = mf.parse_run_name(manifest["name"])
    rng = np.random.default_rng([_key(parsed.variant), parsed.init_seed, parsed.data_seed, budget_tokens(manifest["name"])])
    shift = CONTEXT_SHIFT.get(parsed.variant, (0.0,) * len(BUCKETS))
    bucket_losses = [
        final + offset + extra + BUCKET_NOISE * float(rng.standard_normal())
        for offset, extra in zip(BUCKET_OFFSETS, shift)
    ]
    return {
        "val_loss": final,
        "val": val_block(final, bucket_losses),
        "val_curve": curve_block(final, budget_tokens(manifest["name"]), manifest["steps"]["total"]),
    }


def with_eval_fields(manifests) -> list[dict]:
    """Copies of the manifests with `val` and `val_curve` on every completed result, each validated."""
    out = []
    for manifest in manifests:
        edited = copy.deepcopy(manifest)
        if mf.is_result(edited):
            fields = eval_fields(edited)
            edited["final_metrics"] = {**edited["final_metrics"], "val": fields["val"], "val_curve": fields["val_curve"]}
            problems = mf.validate_manifest(edited)
            if problems:
                raise ValueError(f"{edited['name']}: {problems}")
        out.append(edited)
    return out


# --- Constructed cases with exact numbers -----------------------------------------------

CONSTRUCTED_BUCKETS = {
    ("mha", 1): (3.00, 3.10, 3.20, 3.30),
    ("mha", 2): (3.05, 3.12, 3.25, 3.28),
    ("mqa", 1): (3.02, 3.15, 3.22, 3.35),
    ("mqa", 2): (3.10, 3.12, 3.30, 3.27),
}


def bucket_manifests() -> list[dict]:
    """MHA and MQA at 300M, seeds 1 and 2, with hand-set per-bucket losses (constructed, not a result)."""
    manifests = []
    for (variant, seed), losses in CONSTRUCTED_BUCKETS.items():
        final = float(np.mean(losses))
        manifests.append(
            fake_manifest(
                variant,
                init_seed=seed,
                val_loss=final,
                final_metrics={"val_loss": final, "val": val_block(final, losses)},
            )
        )
    return manifests


CONSTRUCTED_CURVE_FINALS = {"mha_300": 3.2, "mha_100": 3.4, "mqa_300": 3.3}


def curve_manifests() -> list[dict]:
    """Three results with explicit curves (constructed): MHA at 300M, MHA's 100M branch, MQA at 300M."""
    curve = {
        "mha_300": [(100_000_000, 3.5), (200_000_000, 3.3), (300_000_000, 3.2)],
        "mha_100": [(50_000_000, 3.7), (100_000_000, 3.4)],
        "mqa_300": [(100_000_000, 3.6), (300_000_000, 3.3)],
    }
    manifests = [
        fake_manifest(
            "mha", init_seed=1,
            final_metrics={"val_loss": CONSTRUCTED_CURVE_FINALS["mha_300"], "val_curve": _curve_rows(curve["mha_300"], total=4578)},
        ),
        fake_manifest(
            "mha", init_seed=1, branch="100M",
            final_metrics={"val_loss": CONSTRUCTED_CURVE_FINALS["mha_100"], "val_curve": _curve_rows(curve["mha_100"], total=1526)},
        ),
        fake_manifest(
            "mqa", init_seed=1,
            final_metrics={"val_loss": CONSTRUCTED_CURVE_FINALS["mqa_300"], "val_curve": _curve_rows(curve["mqa_300"], total=4578)},
        ),
    ]
    return manifests


def _curve_rows(points, *, total: int) -> list[dict]:
    rows = []
    for tokens, loss in points:
        rows.append({"step": round(total * tokens / points[-1][0]), "tokens": tokens, "val_loss": loss})
    return rows


# --- bench/2, probe and retrieval records --------------------------------------------------

BENCH_REPETITIONS = 3  # plan §8.6 as amended by decision 225
BENCH_SPREAD = 0.02  # synthetic: the two outer repetitions lie 2% either side of the median


def repeated(name: str, median: float, *, decimals: int | None = None) -> dict:
    """The bench/2 fields of one timed quantity (synthetic): its median, its three repetitions with
    the median in the middle (so the median is exactly the statistic of the list), min and max.

    `median` must already have `decimals` decimal places when given; the outer repetitions are rounded
    the same way.
    """
    def fix(x: float) -> float:
        return x if decimals is None else round(x, decimals)

    low, high = fix(median * (1.0 - BENCH_SPREAD)), fix(median * (1.0 + BENCH_SPREAD))
    return {name: median, f"{name}_by_repetition": [high, median, low], f"{name}_min": low, f"{name}_max": high}


def bench_records() -> list[dict]:
    """bench/2 records for every headline variant, in ref and opt modes (synthetic numbers).

    Synthetic records. Their environment and backend are those of the synthetic campaign
    (fake_secondary.ENVIRONMENT; backend "efficient" on opt records, None on ref records), so that
    the timing-source checks (make_tables.check_timing_sources) accept them with it.
    """
    from tests.fixtures.fake_secondary import ENVIRONMENT  # deferred: fake_secondary imports this module

    records = []
    for index, variant in enumerate(BENCH_VARIANTS):
        for mode in ("ref", "opt"):
            rng = np.random.default_rng([_key("bench"), index, mode == "opt"])
            speed = (1.0 if mode == "opt" else 1.25) * (1.0 + 0.04 * index)
            prefill = []
            for context in (128, 512, 2048, 8192):
                latency = speed * (0.5 + context / 1500.0) * (1.0 + 0.01 * float(rng.standard_normal()))
                latency = round(latency, 3)
                tok = round(context / latency * 1000.0, 1)
                prefill.append({"context": context, "batch": 1, **repeated("latency_ms", latency, decimals=3),
                                **repeated("tok_per_s", tok, decimals=1), "peak_bytes": 10_000_000 + context})
            decode = []
            for batch in (1, 8, 32):
                for cache in (1024, 4096, 8192):
                    ms = round(speed * (1.0 + 0.4 * batch / 32.0 + cache / 8192.0) * (1.0 + 0.01 * float(rng.standard_normal())), 3)
                    tok = round(batch * 1000.0 / ms, 1)
                    decode.append({"batch": batch, "cache": cache, **repeated("ms_per_token", ms, decimals=3),
                                   **repeated("tok_per_s", tok, decimals=1), "peak_bytes": 20_000_000 + cache})
            records.append({
                "schema": "bench/2",
                "variant": variant,
                "mode": mode,
                "precision": "fp16",
                "environment": dict(ENVIRONMENT),
                "backend": "efficient" if mode == "opt" else None,
                "gpu_shared": False,
                "repetitions": BENCH_REPETITIONS,
                "kv_bytes_per_token": {"analytic": 1536, "measured": 1540.0, "measured_tokens": None},
                "prefill": prefill,
                "decode": decode,
                "max_batch_at_8k": 64,
            })
    return records


def probe_records() -> list[dict]:
    """Probe accuracies: softmax variants near saturation, the recurrent anchor degrading (synthetic)."""
    variants = (*BENCH_VARIANTS[:4], "recurrence")
    records = []
    for task_index, task in enumerate(PROBE_TASKS):
        for variant_index, variant in enumerate(variants):
            for difficulty in PROBE_DIFFICULTIES:
                for seed in PROBE_SEEDS:
                    rng = np.random.default_rng([_key("probe"), task_index, variant_index, difficulty, seed])
                    decay = 0.12 if variant == "recurrence" else 0.002
                    accuracy = 1.0 - decay * np.log2(difficulty) + 0.01 * float(rng.standard_normal())
                    clipped = float(np.clip(accuracy, 0.0, 1.0))
                    # Synthetic: the sequence accuracy is half the token accuracy, and the chance is 1/n.
                    records.append({"variant": variant, "task": task, "difficulty": difficulty,
                                    "init_seed": seed, "accuracy": clipped, "sequence_accuracy": 0.5 * clipped,
                                    "chance": 1 / difficulty})
    return records


def retrieval_records() -> list[dict]:
    """Retrieval accuracies for the headline variants, both tasks and modes (synthetic)."""
    records = []
    for variant_index, variant in enumerate(BENCH_VARIANTS[:4]):
        for task_index, task in enumerate(("passkey", "needle")):
            for mode_index, mode in enumerate(("zero_shot", "trained")):
                for length in RETRIEVAL_LENGTHS:
                    for distractors in RETRIEVAL_DISTRACTORS:
                        for seed in RETRIEVAL_SEEDS:
                            rng = np.random.default_rng([_key("retrieval"), variant_index, task_index, mode_index,
                                                         length, distractors, seed])
                            base = 0.55 if mode == "zero_shot" else 0.9
                            accuracy = base - 0.04 * distractors - 0.03 * np.log2(length / 256) \
                                + 0.02 * float(rng.standard_normal())
                            records.append({"variant": variant, "task": task, "mode": mode, "length": length,
                                            "n_distractors": distractors, "init_seed": seed,
                                            "accuracy": float(np.clip(accuracy, 0.0, 1.0))})
    return records
