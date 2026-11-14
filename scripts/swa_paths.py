"""The attention times of SWA's two opt paths, next to MHA's (plan §2.3; decisions 244, 245, 247).

Times one forward and one backward of the attention module alone, in opt mode, for three tracks in
this order: `mha`, `swa256/chunked` and `swa256/flex`. The tracks run round-robin: every track once
in repetition 0, then every track once in repetition 1, and so on. A repetition is `--warmup`
untimed calls and then `--repeats` timed calls, and its value is the median of the timed calls.
The document holds the median of the repetitions and their smallest and largest values.

Only the flex track may be unavailable: FlexAttention may be missing, or fail to compile on this GPU.
Its error is recorded and the script still exits 0. A failure of MHA or of the chunked path is an
error and stops the script.

Nothing here selects a path. SWA trains with `flex` if FlexAttention runs and agrees with the
reference on the T4, and with `chunked` otherwise (plan §2.3, decision 244). This script records
both times, as a disclosure (decision 245). The document is not cross-checked against the timing
pass (decision 247).

    python -m scripts.swa_paths --device cuda --gpu-shared false --expected-backend efficient --out FILE

On CUDA `--gpu-shared` and `--expected-backend` are required, and `--gpu-shared true` is refused.
`--expected-backend` is refused on CPU. `--synthetic` runs a tiny model for testing the script; its
numbers mean nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from scripts import timing_pass as tp
from src import config
from src.attention.base import AttnConfig, build_attention
from src.eval.inference import BenchError, backend_scope, check_backend_flag
from src.transformer import rope_freqs
from src.utils.manifest import collect_environment

SCHEMA = "swa_paths/1"
MHA_VARIANT = "mha"
SWA_VARIANT = "swa256"
MHA_KEY = "mha"
CHUNKED_KEY = "swa256/chunked"
FLEX_KEY = "swa256/flex"
INIT_SEED = 1  # the per-parameter init seed of every benchmark model (plan §4.5)
INPUT_SEED = 0  # seed of the explicit generator that draws the input
PEAK_LR = 1e-3  # config.resolve needs a learning rate; weights do not affect speed
BATCH = 16
REPEATS = 30
WARMUP = 10
REPETITIONS = 3
DEFAULT_CLOCK = time.perf_counter
NUMERIC_KEYS = ("attention_ms", "attention_ms_by_repetition", "attention_ms_min", "attention_ms_max")


class SwaPathsError(RuntimeError):
    """The SWA paths record cannot be read or reported, or a check it makes fails."""


@dataclass(frozen=True)
class Track:
    key: str
    variant: str
    opt_path: str | None  # None: the variant's own attention, with no path set
    optional: bool  # True: a failure is recorded as the track's result (the flex track only)


TRACKS = (
    Track(MHA_KEY, MHA_VARIANT, None, optional=False),
    Track(CHUNKED_KEY, SWA_VARIANT, "chunked", optional=False),
    Track(FLEX_KEY, SWA_VARIANT, "flex", optional=True),
)
ORDER = tuple(track.key for track in TRACKS)


@dataclass(frozen=True)
class PathsSetup:
    """Everything one repetition needs, fixed for the whole run."""

    overrides: Mapping
    device: torch.device
    batch: int = BATCH
    repeats: int = REPEATS
    warmup: int = WARMUP
    repetitions: int = REPETITIONS
    expected_backend: str | None = None  # the enforced SDPA backend; CUDA only
    clock: Callable[[], float] = DEFAULT_CLOCK


def round_robin(track_list: Sequence[Track], repetitions: int) -> list[tuple[Track, int]]:
    """Repetition-major, as timing_pass.round_robin: every track once in repetition 0, then once in repetition 1, ..."""
    return tp.round_robin(track_list, repetitions)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build(track: Track, setup: PathsSetup):
    """The attention module of `track` on the device, its input and its RoPE table. Seeded, so identical every repetition."""
    resolved = config.resolve(track.variant, init_seed=INIT_SEED, peak_lr=PEAK_LR, overrides=dict(setup.overrides))
    attn = build_attention(AttnConfig.from_resolved(resolved)).to(setup.device)
    if track.opt_path is not None:
        attn.set_opt_path(track.opt_path)
    model = resolved["model"]
    generator = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(setup.batch, model["context_length"], model["d_model"], generator=generator)
    x = x.to(setup.device).requires_grad_(True)
    freqs = rope_freqs(attn.rope_dim, 0, model["context_length"], model["rope_theta"], device=setup.device)
    return attn, x, freqs


def _step(attn, x, freqs, device: torch.device) -> None:
    """One call: forward in opt mode, then backward of the sum of the output."""
    with _autocast(device):
        y, _ = attn(x, freqs, mode="opt")
    y.float().sum().backward()


def time_repetition(track: Track, setup: PathsSetup) -> list[float]:
    """One repetition of one track: the milliseconds of each timed call, the warm-up calls dropped."""
    attn, x, freqs = _build(track, setup)
    device = setup.device
    with backend_scope(setup.expected_backend, device):
        for _ in range(setup.warmup):
            _step(attn, x, freqs, device)
            attn.zero_grad(set_to_none=True)
        durations = []
        for _ in range(setup.repeats):
            _sync(device)
            begin = setup.clock()
            _step(attn, x, freqs, device)
            _sync(device)
            durations.append((setup.clock() - begin) * 1000.0)
            attn.zero_grad(set_to_none=True)
    return durations


def _entry(repetitions: list[list[float]], error: str | None) -> dict:
    """The document entry of one track: per-repetition medians and their statistics, or the error with null numbers."""
    if error is not None:
        return {
            "attention_ms": None,
            "attention_ms_by_repetition": None,
            "attention_ms_min": None,
            "attention_ms_max": None,
            "error": error,
        }
    per_repetition = [statistics.median(values) for values in repetitions]
    return {
        "attention_ms": statistics.median(per_repetition),
        "attention_ms_by_repetition": per_repetition,
        "attention_ms_min": min(per_repetition),
        "attention_ms_max": max(per_repetition),
        "error": None,
    }


def run_tracks(setup: PathsSetup, *, time_fn: Callable[[Track, PathsSetup], list[float]] = time_repetition) -> dict:
    """The `tracks` mapping of the document. `time_fn` is injectable for tests.

    A track that fails in one repetition is not tried again, and its result is its error. Only an
    optional track's exception is caught; any other exception stops the run.
    """
    collected: dict[str, list[list[float]]] = {track.key: [] for track in TRACKS}
    errors: dict[str, str] = {}
    for track, _ in round_robin(TRACKS, setup.repetitions):
        if track.key in errors:
            continue
        if not track.optional:
            collected[track.key].append(time_fn(track, setup))
            continue
        try:
            collected[track.key].append(time_fn(track, setup))
        except Exception as exc:  # the flex track only (plan §2.3): recorded as its result
            errors[track.key] = f"{type(exc).__name__}: {exc}"
    return {track.key: _entry(collected[track.key], errors.get(track.key)) for track in TRACKS}


def build_document(tracks: dict, setup: PathsSetup, *, context: int, environment: dict) -> dict:
    return {
        "schema": SCHEMA,
        "environment": environment,
        "gpu_shared": False,
        "backend": setup.expected_backend,
        "device": setup.device.type,
        "precision": "fp16" if setup.device.type == "cuda" else "fp32",
        "batch": setup.batch,
        "context": context,
        "repetitions": setup.repetitions,
        "repeats": setup.repeats,
        "warmup": setup.warmup,
        "order": list(ORDER),
        "tracks": tracks,
    }


def write_swa_paths(path: str | Path, document: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return out


def read_swa_paths(path: str | Path) -> dict:
    """The document at `path`; SwaPathsError when its schema is not swa_paths/1."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = doc.get("schema") if isinstance(doc, dict) else None
    if schema != SCHEMA:
        raise SwaPathsError(f"{path}: schema {schema!r}, expected {SCHEMA!r}")
    return doc


def _is_time(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _track_problems(track: Track, entry) -> list[str]:
    key = track.key
    if not isinstance(entry, Mapping):
        return [f"{key}: the track is not an object"]
    missing = [name for name in NUMERIC_KEYS + ("error",) if name not in entry]
    if missing:
        return [f"{key}: missing {', '.join(missing)}"]
    error = entry["error"]
    if error is not None and not isinstance(error, str):
        return [f"{key}: error must be text or null, got {error!r}"]
    if error is not None:
        if not track.optional:
            return [f"{key}: carries an error ({error}); only the flex track may be unavailable (plan §2.3)"]
        numbers = [name for name in NUMERIC_KEYS if entry[name] is not None]
        return [f"{key}: unavailable, so {', '.join(numbers)} must be null"] if numbers else []
    problems = []
    for name in ("attention_ms", "attention_ms_min", "attention_ms_max"):
        if not _is_time(entry[name]):
            problems.append(f"{key}: {name} must be a positive number of milliseconds, got {entry[name]!r}")
    reps = entry["attention_ms_by_repetition"]
    if not isinstance(reps, list) or not reps or not all(_is_time(v) for v in reps):
        problems.append(f"{key}: attention_ms_by_repetition must be a non-empty list of positive numbers, got {reps!r}")
    return problems


def check_document(doc) -> None:
    """Raises SwaPathsError listing every problem of a swa_paths/1 document."""
    if not isinstance(doc, Mapping):
        raise SwaPathsError("the swa_paths document is not an object")
    problems = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    if doc.get("gpu_shared") is not False:
        problems.append(f"gpu_shared is {doc.get('gpu_shared')!r}, must be false: the paths run alone (plan §2.2)")
    tracks = doc.get("tracks")
    if not isinstance(tracks, Mapping):
        problems.append("no tracks object")
    else:
        for track in TRACKS:
            if track.key not in tracks:
                problems.append(f"track {track.key!r} is missing")
            else:
                problems.extend(_track_problems(track, tracks[track.key]))
    if problems:
        raise SwaPathsError("the swa_paths document cannot be reported:\n  " + "\n  ".join(problems))


# --- Command line -----------------------------------------------------------------------


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1"):
        return True
    if text.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gpu-shared", type=_bool, default=None, help="required on CUDA: true or false (plan §2.2)")
    p.add_argument(
        "--expected-backend", default=None,
        help="required on CUDA, refused on CPU: the SDPA backend the tracks run under "
             "(flash, efficient, math or cudnn; plan §2.3 item 1)",
    )
    p.add_argument("--batch", type=int, default=BATCH, help="sequences per call (default 16)")
    p.add_argument("--repeats", type=int, default=REPEATS, help="timed calls per repetition (default 30)")
    p.add_argument("--warmup", type=int, default=WARMUP, help="untimed calls before each repetition (default 10)")
    p.add_argument("--repetitions", type=int, default=REPETITIONS, help="round-robin repetitions (default 3)")
    p.add_argument(
        "--synthetic", action="store_true",
        help="a tiny synthetic model, for testing the script; its numbers mean nothing",
    )
    p.add_argument("--out", required=True, help="output JSON path")
    return p


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_shared is None:
        parser.error("--gpu-shared true|false is required on CUDA: state whether the GPU is shared (plan §2.2)")
    if args.gpu_shared:
        parser.error("refusing to run: gpu_shared is true; the SWA paths must run alone (plan §2.2)")
    try:
        expected_backend = check_backend_flag(args.expected_backend, device)
    except BenchError as exc:
        parser.error(str(exc))
    for name, value, minimum in (
        ("--batch", args.batch, 1), ("--repeats", args.repeats, 1),
        ("--warmup", args.warmup, 0), ("--repetitions", args.repetitions, 1),
    ):
        if value < minimum:
            parser.error(f"{name} must be at least {minimum}, got {value}")

    overrides = dict(tp.SYNTHETIC_DIMS) if args.synthetic else {}
    context = config.resolve(MHA_VARIANT, init_seed=INIT_SEED, peak_lr=PEAK_LR, overrides=overrides)["model"]["context_length"]
    setup = PathsSetup(
        overrides=overrides,
        device=device,
        batch=args.batch,
        repeats=args.repeats,
        warmup=args.warmup,
        repetitions=args.repetitions,
        expected_backend=expected_backend,
    )
    tracks = run_tracks(setup)
    document = build_document(tracks, setup, context=context, environment=collect_environment())
    check_document(document)
    path = write_swa_paths(args.out, document)
    print(f"wrote {path}")
    for track in TRACKS:
        entry = tracks[track.key]
        if entry["error"] is None:
            print(
                f"{track.key}: median {entry['attention_ms']:.3f} ms, "
                f"range {entry['attention_ms_min']:.3f} to {entry['attention_ms_max']:.3f} ms"
            )
        else:
            print(f"{track.key}: not available: {entry['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
