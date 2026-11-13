"""SYNTHETIC inputs for the secondary-report tests (T-34). Nothing here comes from a real run.

- Arm B: standalone 100M runs of the iso-backbone arm (`arm_b_manifests`), built with `fake_manifest`.
- Arm A 100M branches of the exploratory variants, which `fake_campaign` leaves out
  (`secondary_campaign`). The real Arm A branches every variant (configs/arms/iso_token.yaml).
- A per-run ICL score (`with_context`). `with_eval_fields` gives every run the same score, so
  this module gives each run its own, which makes a gap exist.
- Timing-pass and bench documents with the environment of the fake manifests (`timing_document`,
  `bench_document`).

Every number is made up. Noise comes from `numpy.random.default_rng` with explicit seeds, never
the global RNG.
"""

from __future__ import annotations

import copy
import json
import zlib
from pathlib import Path

import numpy as np

from src.utils import manifest as mf
from tests.fixtures.fake_campaign import EXPLORATORY_VARIANTS, fake_campaign
from tests.fixtures.fake_manifests import fake_manifest
from tests.fixtures.fake_plot_inputs import BENCH_REPETITIONS, repeated, with_eval_fields

ARM_B_VARIANTS = ("gqa4", "gqa2", "mqa")
ARM_B_SEEDS = (1, 2, 3)
ARM_B_OFFSET = 0.05  # default Arm B loss minus Arm A's 100M branch of the same seed
ARM_B_SIGMA = 0.004
ICL_SIGMA = 0.02
ICL_BASE = -0.1
ICL_STREAM = 7  # the first entry of every ICL noise stream, so it never clashes with another fixture
EXPLORATORY_BRANCH_OFFSET = 0.3  # the 100M branch minus the 300M loss, as in fake_campaign

# The environment of the fake manifests (tests/fixtures/fake_manifests.py).
ENVIRONMENT = {
    "python": "3.11.9",
    "torch": "2.15.0",
    "cuda": "13.2",
    "driver": "999.0",
    "gpu_name": "Synthetic T4",
    "compute_capability": [7, 5],
}
BACKEND = "efficient"

# Synthetic step times (ms) and decode throughputs, for the opt track of each variant.
TIMING_MS = {"mha": 100.0, "gqa4": 100.5, "gqa2": 99.0, "mqa": 101.0, "mla192": 130.0, "mla96": 120.0, "swa256": 97.0}
DECODE_TOK_PER_S = {"mha": 1000.0, "gqa4": 1100.0, "gqa2": 1150.0, "mqa": 1200.0, "mla192": 900.0, "mla96": 950.0, "swa256": 1050.0}


def _key(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def find(manifests, *, variant: str, arm: str, init_seed: int, tokens: str, branch: str | None = None,
         data_seed: int = 1, peak_lr: float | None = None) -> dict | None:
    """The one manifest with these name fields, or None. Used by the fixtures and by the tests."""
    hits = []
    for manifest in manifests:
        parsed = mf.parse_run_name(manifest["name"])
        if (
            parsed.variant == variant
            and parsed.arm == arm
            and parsed.init_seed == init_seed
            and parsed.tokens == tokens
            and parsed.branch == branch
            and parsed.data_seed == data_seed
            and (peak_lr is None or parsed.peak_lr == peak_lr)
        ):
            hits.append(manifest)
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} manifests match {variant} {arm} {tokens} {branch} seed {init_seed}")
    return hits[0] if hits else None


def secondary_campaign(**kwargs) -> list[dict]:
    """fake_campaign plus 100M branches for the exploratory variants, where the 300M run exists."""
    manifests = fake_campaign(**kwargs)
    branches = []
    for variant in EXPLORATORY_VARIANTS:
        for seed in range(1, 4):
            run = find(manifests, variant=variant, arm="isoparam", init_seed=seed, tokens="300M")
            if run is None:
                continue
            loss = float(run["final_metrics"]["val_loss"]) + EXPLORATORY_BRANCH_OFFSET
            branches.append(fake_manifest(variant, init_seed=seed, branch="100M", val_loss=loss))
    return manifests + branches


def with_context(manifests: list[dict]) -> list[dict]:
    """Eval fields on every result, with a per-run ICL score (synthetic, seeded per run)."""
    out = with_eval_fields(manifests)
    for manifest in out:
        if not mf.is_result(manifest):
            continue
        parsed = mf.parse_run_name(manifest["name"])
        rng = np.random.default_rng([ICL_STREAM, _key(parsed.variant), parsed.init_seed, 1 if parsed.branch else 0])
        score = ICL_BASE + ICL_SIGMA * float(rng.standard_normal())
        manifest["final_metrics"]["val"]["icl"]["score"] = score
        problems = mf.validate_manifest(manifest)
        if problems:
            raise ValueError(f"{manifest['name']}: {problems}")
    return out


def arm_b_manifests(
    campaign: list[dict], *, offset: float = ARM_B_OFFSET, noise: float = ARM_B_SIGMA,
    peak_lr: float = 1e-3, variants=ARM_B_VARIANTS, seeds=ARM_B_SEEDS,
) -> list[dict]:
    """Standalone 100M Arm B runs. Each loss is the variant's Arm A 100M branch of the same seed, plus `offset`, plus noise."""
    manifests = []
    for variant in variants:
        for seed in seeds:
            branch = find(campaign, variant=variant, arm="isoparam", init_seed=seed, tokens="300M", branch="100M")
            if branch is None:
                raise ValueError(f"the campaign has no 100M branch of {variant} seed {seed}")
            rng = np.random.default_rng([ICL_STREAM + 1, _key(variant), seed])
            loss = float(branch["final_metrics"]["val_loss"]) + offset + noise * float(rng.standard_normal())
            manifests.append(
                fake_manifest(
                    variant, arm="iso_backbone", budget="100M", branches=[], init_seed=seed,
                    peak_lr=peak_lr, val_loss=loss,
                )
            )
    return manifests


def timing_document(*, shared: bool = False, environment: dict | None = None, backend: str | None = BACKEND,
                    ms: dict | None = None) -> dict:
    """A timing_pass/1 document with one opt entry per variant and one ref entry (not read as a timing)."""
    ms = dict(TIMING_MS if ms is None else ms)
    env = dict(ENVIRONMENT if environment is None else environment)
    variants = {
        key: {"mode": "opt", "ms_per_step": value, "ms_per_step_by_repetition": [value] * 3,
              "attention_share": 0.25, "peak_bytes": None}
        for key, value in ms.items()
    }
    variants["mla192/ref"] = {"mode": "ref", "ms_per_step": 999.0, "ms_per_step_by_repetition": [999.0],
                              "attention_share": 0.25, "peak_bytes": None}
    return {
        "schema": "timing_pass/1",
        "timing_pass_id": "SYNTHETIC-timing-pass",
        "environment": env,
        "gpu_shared": shared,
        "backend": backend,
        "steps": 220,
        "discarded": 20,
        "repetitions": 3,
        "order": list(variants),
        "variants": variants,
    }


def bench_document(variant: str, mode: str = "opt", *, shared: bool = False, environment: dict | None = None,
                   backend: str | None = BACKEND, tok_per_s: float | None = None) -> dict:
    """A synthetic bench/2 document with three repetitions. Only the opt document carries the headline
    decode point (batch 32, cache 4,096); its tok/s is the median, the repetitions lie 2% either side."""
    value = DECODE_TOK_PER_S[variant] if tok_per_s is None else tok_per_s
    env = dict(ENVIRONMENT if environment is None else environment)
    opt = mode == "opt"
    return {
        "schema": "bench/2",
        "variant": variant,
        "mode": mode,
        "precision": "fp16",
        "backend": backend if opt else None,
        "environment": env,
        "gpu_shared": shared,
        "repetitions": BENCH_REPETITIONS,
        "kv_bytes_per_token": {"analytic": 1536, "measured": None, "measured_tokens": None},
        "prefill": [],
        "decode": [
            {"batch": 1, "cache": 1024, **repeated("ms_per_token", 1.0), **repeated("tok_per_s", 1000.0),
             "peak_bytes": None},
            {"batch": 32, "cache": 4096, **repeated("ms_per_token", 32.0 / value), **repeated("tok_per_s", value),
             "peak_bytes": None},
        ],
        "max_batch_at_8k": None,
    }


def write_bench_documents(directory, documents) -> None:
    """bench_<variant>_<mode>.json files, as scripts/bench.py writes them."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        (out / f"bench_{doc['variant']}_{doc['mode']}.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def write_timing_document(path, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def with_val_removed(manifests: list[dict], *, variant: str, init_seed: int, tokens: str = "300M") -> list[dict]:
    """A copy in which one 300M result has no `val` block."""
    out = copy.deepcopy(manifests)
    target = find(out, variant=variant, arm="isoparam", init_seed=init_seed, tokens=tokens)
    if target is None:
        raise ValueError(f"no {variant} seed {init_seed} result to remove the val block from")
    target["final_metrics"].pop("val")
    assert mf.validate_manifest(target) == []
    return out
