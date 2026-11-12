"""Training timing pass, plan §8.5 step 1.

"Dedicated GPU, 220 steps each with the first 20 discarded, three round-robin
repetitions. Each variant is timed on the path it trains with: `opt` for the grouped
variants and SWA, the naive path for MLA, whose absorbed path exists at inference only.
Records ms/step, attention's share of step time, and peak memory."

Every timed step is a real `Trainer.train_step()` on the resolved configuration.
Training runs in `opt` mode for every variant. Without a cache, `opt` MLA materialises
per-head keys and values (the naive path, decision 141), so the MLA training timing is
the naive path as the plan requires.

Round-robin: repetition 1 of every track in a fixed order, then repetition 2, then 3.
A track is one variant in its opt track, or `<variant>/ref` when `--with-ref` asks for
the ref track of MLA and SWA (plan §2.3 item 3).

Attention share (`attention_share`) is measured in a separate instrumented run, so the
hooks do not distort ms/step. It is (attention forward + attention backward wall time)
divided by the whole step's wall time, over the same measured steps. Attention's
backward is timed from the gradient of the module's output to the gradient of its input,
which contains the output projection and the attention kernels, and excludes the
surrounding norms and MLP. CUDA is synchronised at each boundary.

    python -m scripts.timing_pass --peak-lr 1e-3 --micro-batch 16 --init-scale 65536 \
        --growth-interval 2000 --data-dir data/prepared --device cuda --gpu-shared false \
        --with-ref --out results/timing/timing_pass.json

`--synthetic` replaces the model and data by a tiny synthetic setup. It exercises the
script; its numbers mean nothing about the study. Plan §8.5: the pass runs alone on the box.
On CUDA `--gpu-shared` is required, and the script refuses `true`. On CUDA `--expected-backend`
is required too (plan §2.3 item 1): every training step runs with only that SDPA kernel enabled,
and the name is written to the document's "backend" key. It is refused on CPU. On CUDA the SWA-below-MHA
assertion (plan §2.3 item 2) makes the script exit non-zero when it fails; on CPU it is only
reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import yaml

from src import config
from src.config import ConfigError
from src.data.loader import TrainStream, open_dataset
from src.model import build_model
from src.training.trainer import Trainer
from src.eval.inference import BenchError, check_backend_flag
from src.utils.manifest import collect_environment
from src.utils.reproducibility import synthetic_split

SCHEMA = "timing_pass/1"
STEPS = 220  # plan §8.5
DISCARDED = 20  # plan §8.5
REPETITIONS = 3  # plan §8.5
DEFAULT_VARIANTS = ("mha", "gqa4", "gqa2", "mqa", "mla192", "mla96", "swa256")
# Plan §2.3 item 3: the variants whose ref track is reported. A configuration list, not logic.
REF_TRACK_VARIANTS = ("mla192", "mla96", "swa256")
REFERENCE_VARIANT = "mha"  # plan §8.5, §2.3 item 2
SWA_VARIANT = "swa256"
INIT_SEED = 1
DEFAULT_CLOCK = time.perf_counter

SYNTHETIC_DIMS = {
    "model.d_model": 64,
    "model.n_heads": 8,
    "model.head_dim": 8,
    "model.n_layers": 2,
    "model.d_ff": 96,
    "model.vocab_size": 128,
    "model.context_length": 32,
    "recipe.batch.sequences_per_step": 8,
    "recipe.batch.sequence_length": 32,
    "recipe.batch.tokens_per_step": 8 * 32,
    "recipe.batch.micro_batch": 4,
    "recipe.grad_scaler.init_scale": 1024.0,  # synthetic only; the real values are set by Phil
    "recipe.grad_scaler.growth_interval": 2000,
}
SYNTHETIC_VOCAB = 128
SYNTHETIC_CONTEXT = 32


class TimingError(RuntimeError):
    """The timing pass cannot run, or a check it makes fails."""


@dataclass(frozen=True)
class Track:
    key: str  # "mha", or "mla192/ref"
    variant: str
    mode: str  # "opt" or "ref"


@dataclass(frozen=True)
class TimingSetup:
    """Everything one repetition needs, fixed for the whole pass."""

    stream: object  # a TrainStream (or any object with .batch(i) and .max_steps)
    overrides: dict
    peak_lr: float
    steps: int
    discarded: int
    device: torch.device
    clock: Callable[[], float] = DEFAULT_CLOCK
    init_seed: int = INIT_SEED
    expected_backend: str | None = None  # the enforced SDPA backend; CUDA only (plan §2.3 item 1)


@dataclass(frozen=True)
class Repetition:
    ms_per_step: float  # mean over the measured steps of this repetition
    peak_bytes: int | None  # None off CUDA


def tracks(variants: Sequence[str], with_ref: bool) -> tuple[Track, ...]:
    out = [Track(v, v, "opt") for v in variants]
    if with_ref:
        out += [Track(f"{v}/ref", v, "ref") for v in REF_TRACK_VARIANTS if v in variants]
    return tuple(out)


def round_robin(track_list: Sequence[Track], repetitions: int) -> list[tuple[Track, int]]:
    """Repetition-major: every track once in repetition 0, then every track once in repetition 1, ..."""
    return [(track, r) for r in range(repetitions) for track in track_list]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_steps(
    step: Callable[[], object],
    *,
    steps: int,
    discarded: int,
    device: torch.device,
    clock: Callable[[], float] = DEFAULT_CLOCK,
) -> list[float]:
    """Call `step` `steps` times, timing each. Returns the durations in ms of the calls after the first `discarded`."""
    if not 0 <= discarded < steps:
        raise ValueError(f"need 0 <= discarded < steps, got discarded={discarded}, steps={steps}")
    durations = []
    for _ in range(steps):
        _sync(device)
        t0 = clock()
        step()
        _sync(device)
        durations.append((clock() - t0) * 1000.0)
    return durations[discarded:]


def _trainer(track: Track, setup: TimingSetup) -> Trainer:
    resolved = config.resolve(
        track.variant,
        init_seed=setup.init_seed,
        peak_lr=setup.peak_lr,
        steps=setup.steps,
        branches=[],
        overrides=setup.overrides,
    )
    model = build_model(resolved)
    # Every track, the ref track included, runs under the enforced backend. A ref forward makes
    # no SDPA call, so the restriction changes nothing there.
    return Trainer(
        model,
        resolved,
        setup.stream,
        device=str(setup.device),
        mode=track.mode,
        expected_backend=setup.expected_backend,
    )


def time_repetition(track: Track, setup: TimingSetup) -> Repetition:
    """One repetition of one track: a fresh model and trainer, every step timed, the first `discarded` dropped."""
    trainer = _trainer(track, setup)
    cuda = setup.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(setup.device)
    ms = time_steps(
        trainer.train_step,
        steps=setup.steps,
        discarded=setup.discarded,
        device=setup.device,
        clock=setup.clock,
    )
    peak = torch.cuda.max_memory_allocated(setup.device) if cuda else None
    return Repetition(ms_per_step=sum(ms) / len(ms), peak_bytes=peak)


class AttentionTimer:
    """Wall time of every attention module's forward and backward, from forward and gradient hooks.

    Forward: pre-hook to forward-hook of the module. Backward: the gradient of the module's
    output becoming available to the gradient of its input becoming available. The total is
    read with `total` and reset with `reset()`. Use as a context manager: hooks are removed on exit.
    """

    def __init__(self, model, *, device: torch.device, clock: Callable[[], float] = DEFAULT_CLOCK):
        self.model = model
        self.device = device
        self.clock = clock
        self.total = 0.0
        self._handles: list = []
        self._fwd_start: dict[int, float] = {}
        self._bwd_start: dict[int, float] = {}

    def reset(self) -> None:
        self.total = 0.0

    def __enter__(self) -> "AttentionTimer":
        for index, layer in enumerate(self.model.layers):
            attn = layer.attn
            self._handles.append(attn.register_forward_pre_hook(self._pre(index)))
            self._handles.append(attn.register_forward_hook(self._post(index)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre(self, index: int):
        def hook(module, args):
            _sync(self.device)
            self._fwd_start[index] = self.clock()
            x = args[0]
            if torch.is_tensor(x) and x.requires_grad:
                x.register_hook(self._end_backward(index))

        return hook

    def _post(self, index: int):
        def hook(module, args, output):
            _sync(self.device)
            now = self.clock()
            self.total += now - self._fwd_start.pop(index)
            y = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(y) and y.requires_grad:
                y.register_hook(self._begin_backward(index))

        return hook

    def _begin_backward(self, index: int):
        def hook(grad):
            _sync(self.device)
            self._bwd_start[index] = self.clock()
            return None

        return hook

    def _end_backward(self, index: int):
        def hook(grad):
            _sync(self.device)
            start = self._bwd_start.pop(index, None)
            if start is not None:
                self.total += self.clock() - start
            return None

        return hook


def measure_share(track: Track, setup: TimingSetup) -> float:
    """Attention's share of step time over the measured steps, in a separate instrumented run."""
    trainer = _trainer(track, setup)
    attention = 0.0
    whole = 0.0
    with AttentionTimer(trainer.model, device=setup.device, clock=setup.clock) as timer:
        for i in range(setup.steps):
            timer.reset()
            _sync(setup.device)
            t0 = setup.clock()
            trainer.train_step()
            _sync(setup.device)
            elapsed = setup.clock() - t0
            if i >= setup.discarded:
                attention += timer.total
                whole += elapsed
    if whole <= 0:
        raise TimingError(f"{track.key}: the measured steps took no time; the clock is too coarse")
    return attention / whole


# --- The SWA assertion (plan §2.3 item 2) --------------------------------------------


@dataclass(frozen=True)
class SwaCheck:
    holds: bool
    mha_attention_ms: float  # attention_share x ms_per_step
    swa_attention_ms: float


def check_swa_below_mha(variants: dict) -> SwaCheck:
    """Whether SWA's measured attention time per step is below MHA's, from a pass's `variants` mapping.

    Attention time per step is attention_share x ms_per_step. Names two variants on
    purpose: plan §2.3 item 2 does.
    """
    for name in (REFERENCE_VARIANT, SWA_VARIANT):
        if name not in variants:
            raise TimingError(f"the SWA check needs the {name!r} entry")
    mha = variants[REFERENCE_VARIANT]
    swa = variants[SWA_VARIANT]
    mha_ms = float(mha["attention_share"]) * float(mha["ms_per_step"])
    swa_ms = float(swa["attention_share"]) * float(swa["ms_per_step"])
    return SwaCheck(holds=swa_ms < mha_ms, mha_attention_ms=mha_ms, swa_attention_ms=swa_ms)


def require_swa_below_mha(variants: dict) -> SwaCheck:
    """check_swa_below_mha, raising AssertionError when SWA's attention is not below MHA's."""
    check = check_swa_below_mha(variants)
    if not check.holds:
        raise AssertionError(
            f"SWA attention time {check.swa_attention_ms:.3f} ms/step is not below MHA's "
            f"{check.mha_attention_ms:.3f} ms/step (plan §2.3 item 2): a dense SWA would look like this"
        )
    return check


# --- The pass ---------------------------------------------------------------------------


def run_timing_pass(
    setup: TimingSetup,
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    repetitions: int = REPETITIONS,
    with_ref: bool = False,
    gpu_shared: bool = False,
    time_fn: Callable[[Track, TimingSetup], Repetition] = time_repetition,
    share_fn: Callable[[Track, TimingSetup], float] = measure_share,
) -> dict:
    """The schema-"timing_pass/1" document. `time_fn` and `share_fn` are injectable for tests."""
    track_list = tracks(variants, with_ref)
    collected: dict[str, list[Repetition]] = {t.key: [] for t in track_list}
    for track, _ in round_robin(track_list, repetitions):
        collected[track.key].append(time_fn(track, setup))

    entries = {}
    for track in track_list:
        reps = collected[track.key]
        peaks = [r.peak_bytes for r in reps if r.peak_bytes is not None]
        entries[track.key] = {
            "mode": track.mode,
            "ms_per_step": sum(r.ms_per_step for r in reps) / len(reps),
            "ms_per_step_by_repetition": [r.ms_per_step for r in reps],
            "attention_share": share_fn(track, setup),
            "peak_bytes": max(peaks) if peaks else None,
        }
    return {
        "schema": SCHEMA,
        "timing_pass_id": f"tp-{uuid.uuid4().hex}",
        "environment": collect_environment(),
        "gpu_shared": bool(gpu_shared),
        "backend": setup.expected_backend,
        "steps": setup.steps,
        "discarded": setup.discarded,
        "repetitions": repetitions,
        "order": [t.key for t in track_list],
        "variants": entries,
    }


def write_timing_file(path: str | Path, document: dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return out


def read_timing_pass(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA:
        raise TimingError(f"{path}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    return doc


def read_ms_per_step(path: str | Path) -> dict[str, float]:
    """{variant: ms per training step}, the opt track of every variant.

    The shape `analysis.rqw_estimate.estimate` and `make_tables.headline_rows(train_ms_per_step=...)` take.
    Ref-track entries are not included.
    """
    doc = read_timing_pass(path)
    return {
        key: float(entry["ms_per_step"])
        for key, entry in doc["variants"].items()
        if entry["mode"] == "opt"
    }


# --- Command line -----------------------------------------------------------------------


def _bool(text: str) -> bool:
    if text.lower() in ("true", "yes", "1"):
        return True
    if text.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--discarded", type=int, default=DISCARDED)
    p.add_argument("--repetitions", type=int, default=REPETITIONS)
    p.add_argument("--with-ref", action="store_true", help="also time the ref track of MLA and SWA")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gpu-shared", type=_bool, default=None, help="required on CUDA: true or false (plan §2.2)")
    p.add_argument(
        "--expected-backend", default=None,
        help="required on CUDA, refused on CPU: the SDPA backend every track must use "
             "(flash, efficient, math or cudnn; plan §2.3 item 1)",
    )
    p.add_argument("--peak-lr", type=float, required=True, help="peak learning rate; does not change speed")
    p.add_argument("--micro-batch", type=int, default=None, help="recipe.batch.micro_batch (unset in recipe.yaml)")
    p.add_argument("--init-scale", type=float, default=None, help="recipe.grad_scaler.init_scale (unset)")
    p.add_argument("--growth-interval", type=int, default=None, help="recipe.grad_scaler.growth_interval (unset)")
    p.add_argument("--data-dir", default=None, help="prepared dataset directory (not needed with --synthetic)")
    p.add_argument("--synthetic", action="store_true", help="tiny synthetic model and data, for testing the script")
    p.add_argument("--out", required=True, help="output JSON path")
    return p


def _overrides(args) -> dict:
    if args.synthetic:
        return dict(SYNTHETIC_DIMS)
    overrides = {}
    if args.micro_batch is not None:
        overrides["recipe.batch.micro_batch"] = args.micro_batch
    if args.init_scale is not None:
        overrides["recipe.grad_scaler.init_scale"] = args.init_scale
    if args.growth_interval is not None:
        overrides["recipe.grad_scaler.growth_interval"] = args.growth_interval
    return overrides


def _stream(args, resolved: dict):
    batch = resolved["recipe"]["batch"]
    context = resolved["model"]["context_length"]
    if args.synthetic:
        n_tokens = args.steps * batch["sequences_per_step"] * context + 1
        split = synthetic_split(n_tokens, SYNTHETIC_VOCAB, seed=0)
    else:
        data_cfg = yaml.safe_load((config.CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
        split = open_dataset(
            args.data_dir,
            expected_dataset_sha256=data_cfg["dataset_sha256"],
            expected_tokenizer_sha256=data_cfg["tokenizer"]["sha256"],
        )["train"]
    return TrainStream(
        split,
        context_length=context,
        sequences_per_step=batch["sequences_per_step"],
        data_seed=resolved["run"]["data_seed"],
    )


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    known = config.variant_names()
    for name in args.variants:
        if name == "null" or name not in known:
            parser.error(f"unknown or disallowed variant {name!r}")
    if not args.synthetic and args.data_dir is None:
        parser.error("--data-dir is required unless --synthetic is given")
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_shared is None:
        parser.error("--gpu-shared true|false is required on CUDA: state whether the GPU is shared (plan §2.2)")
    if args.gpu_shared:
        parser.error("refusing to run: gpu_shared is true; the timing pass must run alone (plan §2.2)")
    try:
        expected_backend = check_backend_flag(args.expected_backend, device)
    except BenchError as exc:
        parser.error(str(exc))

    overrides = _overrides(args)
    try:
        probe = config.resolve(
            args.variants[0], init_seed=INIT_SEED, peak_lr=args.peak_lr, steps=args.steps, branches=[],
            overrides=overrides,
        )
        config.require_complete(probe)  # fails clearly while micro_batch or the scaler values are unset
    except ConfigError as exc:
        parser.error(f"{exc}; pass them on the command line (see --help)")

    setup = TimingSetup(
        stream=_stream(args, probe),
        overrides=overrides,
        peak_lr=args.peak_lr,
        steps=args.steps,
        discarded=args.discarded,
        device=device,
        expected_backend=expected_backend,
    )
    document = run_timing_pass(
        setup,
        variants=args.variants,
        repetitions=args.repetitions,
        with_ref=args.with_ref,
        gpu_shared=bool(args.gpu_shared),  # a true value was refused above
    )
    path = write_timing_file(args.out, document)
    print(f"wrote {path}")

    variants = document["variants"]
    if REFERENCE_VARIANT in variants and SWA_VARIANT in variants:
        check = check_swa_below_mha(variants)
        print(
            f"SWA attention {check.swa_attention_ms:.3f} ms/step vs MHA {check.mha_attention_ms:.3f}: "
            f"{'below (holds)' if check.holds else 'NOT below'}"
        )
        if device.type == "cuda" and not check.holds:
            print("plan §2.3 item 2 fails on this GPU; the run is not a valid timing result", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
