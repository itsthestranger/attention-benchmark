"""The timing and inference tables (plan §8.5, §8.6, §10.4), built from timing_pass/1 and bench/2 documents.

    python -m scripts.timing_report --timing-pass results/timing/timing_pass.json \
        --swa-paths results/timing/swa_paths.json --bench results/bench --out results/tables
    python -m scripts.timing_report --blank --out results/tables/blank

Writes timing.md and inference.md into --out. Every number is read from a document, or computed
from one here. A blank form (no source for a table, or --blank) has the title, the column headers,
the captions and one row per expected track or variant, with every measured cell blank. timing.md
ends with the SWA two-paths section, built from the swa_paths/1 record of --swa-paths (plan §2.3,
decisions 244 and 245): blank when no record is given. The record is not cross-checked against the
timing pass (decision 247).

Thresholds of the plan appear only in captions and headers, as the plan states them. This report
decides no verdict: the words that would judge a row are never written, and no code outside
analysis/primary_test.py emits a verdict (plan §9.1, §11). The module does not import matplotlib.

Exit status: 0 when both files were written. It is 1 on a refusal, and then no file is written.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from scripts import bench as bd
from scripts import make_tables as mt
from scripts import swa_paths as sp
from scripts import timing_pass as tp
from src import config

TIMING_NAME = "timing.md"
INFERENCE_NAME = "inference.md"
BLANK = mt.BLANK
HEADLINE = mt.HEADLINE_VARIANTS
MHA = tp.REFERENCE_VARIANT
REF_SUFFIX = "/ref"
MODES = ("opt", "ref")
NULL_VARIANT = "null"

# Decimal places, one value per quantity. Stated here once; every column uses these.
MS_DECIMALS = 3
SPREAD_DECIMALS = 2
RATIO_DECIMALS = 3
SHARE_DECIMALS = 1
MEM_DECIMALS = 1
LATENCY_DECIMALS = 3
TOKS_DECIMALS = 1
MS_TOKEN_DECIMALS = 3
KV_DECIMALS = 1
KV_RATIO_DECIMALS = 3

PREDICTION_QUOTE = "Prediction: MHA, GQA-4, GQA-2 and MQA within ±3% of each other."  # plan §8.5, verbatim
KV_QUOTE = "analytic and measured bytes/token (assert within 5%)."  # plan §10.4, verbatim
MIB = 2**20
MAX_BATCH_CONTEXT = "8k"  # plan §8.6: the maximum batch is taken at a cache of 8,192

TIMING_TITLE = "# Training timing (plan §8.5)"
INFERENCE_TITLE = "# Inference benchmark (plan §8.6, §10.4)"
SPREAD_COLUMN = "spread over repetitions, % of mean"
RATIO_COLUMN = "ratio to MHA opt"
TIMING_COLUMNS = (
    "track",
    "ms/step, mean",
    "ms/step, each repetition",
    SPREAD_COLUMN,
    RATIO_COLUMN,
    "attention share, %",
    "peak allocated, MiB",
)
TIMING_CAPTIONS = (
    "Track `opt` is the variant on the path it trains with: opt mode for the grouped variants and SWA, "
    "and the naive path for MLA, whose absorbed path exists at inference only (plan §8.5).",
    "Track `ref` is the reference implementation of MLA and SWA: plain PyTorch ops, explicit masks, "
    "no fusion (plan §2.3 item 3, §4.4).",
    "MLA's and SWA's kernels are newer or hand-written, so their gap to MHA is partly implementation, "
    "not architecture (plan §13.3).",
    "Spread is (max - min) / mean over the track's own repetitions, in percent.",
    f'Plan §8.5 reads: "{PREDICTION_QUOTE}" Their step times are shown in the ms/step and ratio to MHA opt '
    "columns of the rows for MHA, gqa4, gqa2 and mqa.",
    "Ratio is each track's ms/step over MHA's opt track. Attention share is the measured attention time "
    "(forward and backward) over the whole step's time, in percent (plan §8.5).",
    "Peak memory is torch.cuda.max_memory_allocated; the timing document records no reserved figure.",
)
RANGE_SEP = "–"  # en dash: "min–max" in the range columns
SWA_PATHS_TITLE = "## SWA's two opt paths (plan §2.3)"
SWA_PATHS_COLUMNS = ("track", "attention, ms (median)", f"attention, ms (min{RANGE_SEP}max)", "ratio to MHA")
SWA_PATHS_CAPTIONS = (
    "Timed: one forward and one backward of the attention module alone, in opt mode, at the batch and "
    "context of the source line, with the SDPA backend of the source line enforced. It is not a training "
    "step, so these numbers are not the ms/step of the timing table.",
    "This table selected nothing: SWA trains on flex if FlexAttention runs on the T4 and agrees with the "
    "reference, else on chunked (plan §2.3, decision 244). The times are recorded so that the other path's "
    "time is known (decision 245). Ratio is each track's median over MHA's median.",
    "The swa256 rows of the timing table used the path named by opt_path in configs/variants/swa256.yaml "
    "at the commit of the timing pass; this section does not record which.",
    "The record has its own source line because it is made at the verification step and may come from "
    "another session than the timing pass.",
)
# Printed in the source line of a record made with --synthetic: such a record measures nothing.
SWA_PATHS_SYNTHETIC = " Synthetic test model: these numbers mean nothing."
# The SWA comparison under timing.md's table (plan §2.3 item 2): both attention times, no verdict word.
SWA_COMPARISON = "SWA attention time: {swa} ms/step; MHA: {mha} ms/step (attention share × ms/step, plan §2.3 item 2)."
PREFILL_COLUMNS = ("context", "batch", "latency, ms", "latency, ms (min–max)", "tokens/s", "peak allocated, MiB")
DECODE_COLUMNS = ("batch", "cache", "ms/token", "ms/token (min–max)", "tokens/s", "peak allocated, MiB")
KV_COLUMNS = (
    "analytic bytes/token",
    "measured bytes/token",
    "measured over tokens",
    "measured / analytic (plan §10.4)",
    f"max batch at cache {MAX_BATCH_CONTEXT}",
)
INFERENCE_CAPTIONS = (
    "Every number is the median of the repetitions named in each section's source line, measured round-robin "
    "over the variants; the range column gives the smallest and the largest repetition (decision 225).",
    "Prefill is one forward over batch x context tokens with no cache; latency is the mean over the "
    "timed calls of each repetition (plan §8.6).",
    "Decode is ms/token and tokens/s at each batch and cache length, the cache filled to the cache length "
    "before timing (plan §8.6).",
    "Track `opt` runs opt mode, with the SDPA backend named in its source line enforced (plan §2.3 item 1). "
    "Track `ref` runs the reference implementation, which makes no SDPA call (plan §4.4).",
    f"KV bytes: the analytic size is per token, from the model's shapes; the measured size is the allocation "
    f'growth of a cache holding the measured number of tokens, on CUDA. Plan §10.4 reads: "{KV_QUOTE}"',
    f"Maximum batch: the largest batch whose decode step at a cache of 8,192 does not run out of memory "
    f"(plan §8.6).",
    "Weights are random, and lengths past the trained context of 1,024 measure speed only (plan §8.6).",
)


class TimingReportError(RuntimeError):
    """A refusal: the report cannot be built from these documents. The message lists the problems."""


def _listing(head: str, problems: Sequence[str]) -> str:
    return "\n".join([f"{head}; {len(problems)} problem(s):"] + [f"  - {p}" for p in problems])


# --- Names and order -------------------------------------------------------------------------


def _is_ref(key: str) -> bool:
    return key.endswith(REF_SUFFIX)


def _variant_of(key: str) -> str:
    return key[: -len(REF_SUFFIX)] if _is_ref(key) else key


def _known(variant) -> bool:
    return bd.known_variant(variant)


def _rank(variant: str) -> int:
    """Headline variants first, in HEADLINE_VARIANTS order; any other known variant after them."""
    return HEADLINE.index(variant) if variant in HEADLINE else len(HEADLINE)


def _ordered_tracks(variants: Mapping[str, dict]) -> list[str]:
    """Each opt track by rank, its ref track directly under it. Ref tracks with no opt track come last."""
    keys = list(variants)
    opts = sorted((k for k in keys if not _is_ref(k)), key=lambda k: _rank(k))
    ordered: list[str] = []
    for key in opts:
        ordered.append(key)
        if f"{key}{REF_SUFFIX}" in variants:
            ordered.append(f"{key}{REF_SUFFIX}")
    ordered += [k for k in keys if _is_ref(k) and k not in ordered]
    return ordered


def expected_timing_tracks() -> list[str]:
    """Every track a timing pass with --with-ref writes, in report order (plan §2.3 item 3)."""
    out: list[str] = []
    for variant in HEADLINE:
        out.append(variant)
        if variant in tp.REF_TRACK_VARIANTS:
            out.append(f"{variant}{REF_SUFFIX}")
    return out


# --- Checks ------------------------------------------------------------------------------------


_is_number = bd.is_number
_is_positive = bd.is_positive


def _track_problems(key, entry) -> list[str]:
    name = f"track {key!r}"
    if not isinstance(key, str):
        return [f"track {key!r} is not a name"]
    variant = _variant_of(key)
    if not _known(variant):
        return [f"{name}: variant {variant!r} is not one the project knows"]
    if _is_ref(key) and variant not in tp.REF_TRACK_VARIANTS:
        return [f"{name}: a ref track exists only for {list(tp.REF_TRACK_VARIANTS)} (plan §2.3 item 3)"]
    if not isinstance(entry, Mapping):
        return [f"{name}: entry is not an object"]
    problems = []
    expected_mode = "ref" if _is_ref(key) else "opt"
    if entry.get("mode") != expected_mode:
        problems.append(f"{name}: mode {entry.get('mode')!r}, expected {expected_mode!r}")
    if not _is_positive(entry.get("ms_per_step")):
        problems.append(f"{name}: ms_per_step {entry.get('ms_per_step')!r}, expected a positive number")
    reps = entry.get("ms_per_step_by_repetition")
    if not isinstance(reps, list) or not reps or not all(_is_positive(v) for v in reps):
        problems.append(f"{name}: ms_per_step_by_repetition {reps!r}, expected a non-empty list of positive numbers")
    if not _is_number(entry.get("attention_share")):
        problems.append(f"{name}: attention_share {entry.get('attention_share')!r}, expected a number")
    peak = entry.get("peak_bytes")
    if peak is not None and not (isinstance(peak, int) and not isinstance(peak, bool) and peak >= 0):
        problems.append(f"{name}: peak_bytes {peak!r}, expected None or a non-negative integer")
    return problems


def check_timing_document(doc) -> None:
    """Refuse a timing document this report cannot read: another schema, an unknown or misplaced track."""
    if not isinstance(doc, Mapping):
        raise TimingReportError(f"the timing document is not an object: {doc!r}")
    problems = []
    if doc.get("schema") != tp.SCHEMA:
        problems.append(f"schema {doc.get('schema')!r}, expected {tp.SCHEMA!r}")
    variants = doc.get("variants")
    if not isinstance(variants, Mapping) or not variants:
        problems.append("the document has no variants")
    else:
        for key, entry in variants.items():
            problems += _track_problems(key, entry)
    if problems:
        raise TimingReportError(_listing("the timing document cannot be reported", problems))


def check_bench_documents(docs: Sequence) -> None:
    """Refuse bench documents this report cannot read (bench.check_documents): another schema, an
    unknown variant or mode, a missing field, a second document for one (variant, mode), or a point
    whose range is not the statistics of its repetitions."""
    try:
        bd.check_documents(docs)
    except bd.BenchError as exc:
        raise TimingReportError(str(exc)) from exc


def load_bench_documents(directory: str | Path) -> list[dict]:
    """Every bench_*.json in `directory`, schema-checked by file name. Refuses an empty directory."""
    paths = sorted(Path(directory).glob("bench_*.json"))
    if not paths:
        raise TimingReportError(f"{directory}: no bench documents (bench_*.json)")
    docs = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TimingReportError(f"{path.name}: not JSON: {exc}") from exc
        if not isinstance(doc, Mapping) or doc.get("schema") != bd.SCHEMA:
            found = doc.get("schema") if isinstance(doc, Mapping) else None
            raise TimingReportError(f"{path.name}: schema {found!r}, expected {bd.SCHEMA!r}")
        docs.append(doc)
    return docs


# --- Formatting --------------------------------------------------------------------------------


def _fmt(value, decimals: int) -> str:
    return BLANK if value is None else f"{value:.{decimals}f}"


def _mib(value) -> str:
    return BLANK if value is None else _fmt(value / MIB, MEM_DECIMALS)


def _text(value) -> str:
    return BLANK if value is None else str(value)


def _span(low, high, decimals: int) -> str:
    """The range of a repetition series, smallest to largest: "min–max"."""
    return f"{_fmt(low, decimals)}{RANGE_SEP}{_fmt(high, decimals)}"


def _md_table(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(row[c] for c in columns) + " |")
    return lines


def _environment(doc: Mapping | None) -> Mapping:
    env = None if doc is None else doc.get("environment")
    return env if isinstance(env, Mapping) else {}


# --- timing.md ---------------------------------------------------------------------------------


def _blank_timing_row(track: str) -> dict:
    row = {column: BLANK for column in TIMING_COLUMNS}
    row["track"] = track
    return row


def timing_rows(doc: Mapping) -> list[dict]:
    """One row per track of a timing document, cells as text, keyed by the TIMING_COLUMNS headers.

    When SWA's attention time is not below MHA's, SWA's opt row carries SWA_DENSE_MARK in its
    ms/step and ratio cells (plan §2.3 item 2, decision 226). No other row is marked.
    """
    check_timing_document(doc)
    variants = doc["variants"]
    mha = variants.get(MHA)
    mha_ms = None if mha is None else float(mha["ms_per_step"])
    rows = []
    for key in _ordered_tracks(variants):
        entry = variants[key]
        ms = float(entry["ms_per_step"])
        reps = [float(v) for v in entry["ms_per_step_by_repetition"]]
        spread = (max(reps) - min(reps)) / ms * 100.0
        rows.append({
            "track": key,
            "ms/step, mean": _fmt(ms, MS_DECIMALS),
            "ms/step, each repetition": ", ".join(_fmt(v, MS_DECIMALS) for v in reps),
            SPREAD_COLUMN: _fmt(spread, SPREAD_DECIMALS),
            RATIO_COLUMN: BLANK if mha_ms is None else _fmt(ms / mha_ms, RATIO_DECIMALS),
            "attention share, %": _fmt(float(entry["attention_share"]) * 100.0, SHARE_DECIMALS),
            "peak allocated, MiB": _mib(entry["peak_bytes"]),
        })
    if tp.swa_not_faster(tp.swa_check_of(doc)):
        for row in rows:
            if row["track"] == tp.SWA_VARIANT:
                row["ms/step, mean"] += tp.SWA_DENSE_MARK
                row[RATIO_COLUMN] += tp.SWA_DENSE_MARK
    return rows


def _swa_comparison(doc: Mapping | None) -> str | None:
    """The comparison line under the table: blanks for the blank form, None when either entry is absent."""
    if doc is None:
        return SWA_COMPARISON.format(swa=BLANK, mha=BLANK)
    check = tp.swa_check_of(doc)
    if check is None:
        return None
    return SWA_COMPARISON.format(
        swa=f"{check.swa_attention_ms:.{MS_DECIMALS}f}", mha=f"{check.mha_attention_ms:.{MS_DECIMALS}f}"
    )


def _timing_source(doc: Mapping | None) -> str:
    env = _environment(doc)
    return (
        f"Source: GPU {_text(env.get('gpu_name'))}; backend {_text(None if doc is None else doc.get('backend'))}; "
        f"torch {_text(env.get('torch'))}; CUDA {_text(env.get('cuda'))}; driver {_text(env.get('driver'))}; "
        f"timing pass {_text(None if doc is None else doc.get('timing_pass_id'))}; "
        f"{_text(None if doc is None else doc.get('steps'))} steps, first "
        f"{_text(None if doc is None else doc.get('discarded'))} discarded, "
        f"{_text(None if doc is None else doc.get('repetitions'))} repetitions."
    )


def render_timing(doc: Mapping | None, swa_paths: Mapping | None = None) -> str:
    """timing.md. `doc` is a timing_pass/1 document, or None for the blank form.

    `swa_paths` is a swa_paths/1 record for the last section; without one that section is blank.
    """
    if doc is None:
        rows = [_blank_timing_row(track) for track in expected_timing_tracks()]
    else:
        rows = timing_rows(doc)
    lines = [TIMING_TITLE, "", _timing_source(doc), ""]
    lines += _md_table(TIMING_COLUMNS, rows)
    lines += [""]
    comparison = _swa_comparison(doc)
    if comparison is not None:
        lines += [comparison, ""]
        check = None if doc is None else tp.swa_check_of(doc)
        if tp.swa_not_faster(check):
            lines += [f"{tp.SWA_DENSE_MARK} {tp.swa_dense_note(check)}", ""]
    lines += list(TIMING_CAPTIONS) + [""]
    lines += _swa_paths_section(swa_paths)
    return "\n".join(lines)


# --- the SWA two-paths section of timing.md (plan §2.3; decisions 244, 245) -----------------------


def _swa_paths_source(record: Mapping | None) -> str:
    env = _environment(record)

    def field(name: str) -> str:
        return _text(None if record is None else record.get(name))

    return (
        f"Source: GPU {_text(env.get('gpu_name'))}; backend {field('backend')}; torch {_text(env.get('torch'))}; "
        f"CUDA {_text(env.get('cuda'))}; driver {_text(env.get('driver'))}; {field('precision')}; "
        f"batch {field('batch')}, context {field('context')}; {field('repetitions')} repetitions of "
        f"{field('repeats')} calls after {field('warmup')} warm-up calls."
        + (SWA_PATHS_SYNTHETIC if record is not None and record.get("synthetic") else "")
    )


def _swa_paths_rows(record: Mapping | None) -> tuple[list[dict], list[str]]:
    """The rows of the SWA section, and the lines that say which tracks could not run. Blank for no record."""
    if record is None:
        blank = {column: BLANK for column in SWA_PATHS_COLUMNS}
        return [{**blank, "track": key} for key in sp.ORDER], []
    sp.check_document(record)
    tracks = record["tracks"]
    mha_ms = float(tracks[sp.MHA_KEY]["attention_ms"])
    rows, notes = [], []
    for key in sp.ORDER:
        entry = tracks[key]
        if entry["error"] is not None:
            rows.append({"track": key, SWA_PATHS_COLUMNS[1]: BLANK, SWA_PATHS_COLUMNS[2]: BLANK, SWA_PATHS_COLUMNS[3]: BLANK})
            notes.append(f"{key} could not run here: {sp.one_line(entry['error'])}")
            continue
        ms = float(entry["attention_ms"])
        rows.append({
            "track": key,
            SWA_PATHS_COLUMNS[1]: _fmt(ms, MS_DECIMALS),
            SWA_PATHS_COLUMNS[2]: _span(entry["attention_ms_min"], entry["attention_ms_max"], MS_DECIMALS),
            SWA_PATHS_COLUMNS[3]: _fmt(ms / mha_ms, RATIO_DECIMALS),
        })
    return rows, notes


def _swa_paths_section(record: Mapping | None) -> list[str]:
    rows, notes = _swa_paths_rows(record)
    lines = [SWA_PATHS_TITLE, "", _swa_paths_source(record), ""]
    lines += _md_table(SWA_PATHS_COLUMNS, rows) + [""]
    for note in notes:
        lines += [note, ""]
    lines += list(SWA_PATHS_CAPTIONS) + [""]
    return lines


# --- inference.md ------------------------------------------------------------------------------


@dataclass
class Section:
    variant: str
    mode: str
    source: str
    prefill: list[dict]
    decode: list[dict]
    kv: dict


def _bench_source(doc: Mapping) -> str:
    env = _environment(doc)
    return (
        f"Source: precision {_text(doc.get('precision'))}; backend {_text(doc.get('backend'))}; "
        f"GPU {_text(env.get('gpu_name'))}; torch {_text(env.get('torch'))}; CUDA {_text(env.get('cuda'))}; "
        f"driver {_text(env.get('driver'))}. Repetitions: {_text(doc.get('repetitions'))}."
    )


def _kv_cells(kv: Mapping, max_batch) -> dict:
    measured = kv["measured"]
    analytic = kv["analytic"]
    ratio = None if measured is None or not analytic else measured / analytic
    return {
        KV_COLUMNS[0]: _fmt(analytic, KV_DECIMALS),
        KV_COLUMNS[1]: _fmt(measured, KV_DECIMALS),
        KV_COLUMNS[2]: _text(kv["measured_tokens"]),
        KV_COLUMNS[3]: _fmt(ratio, KV_RATIO_DECIMALS),
        KV_COLUMNS[4]: _text(max_batch),
    }


def _section(doc: Mapping) -> Section:
    prefill = [
        {
            PREFILL_COLUMNS[0]: _text(row["context"]),
            PREFILL_COLUMNS[1]: _text(row["batch"]),
            PREFILL_COLUMNS[2]: _fmt(row["latency_ms"], LATENCY_DECIMALS),
            PREFILL_COLUMNS[3]: _span(row["latency_ms_min"], row["latency_ms_max"], LATENCY_DECIMALS),
            PREFILL_COLUMNS[4]: _fmt(row["tok_per_s"], TOKS_DECIMALS),
            PREFILL_COLUMNS[5]: _mib(row["peak_bytes"]),
        }
        for row in doc["prefill"]
    ]
    decode = [
        {
            DECODE_COLUMNS[0]: _text(row["batch"]),
            DECODE_COLUMNS[1]: _text(row["cache"]),
            DECODE_COLUMNS[2]: _fmt(row["ms_per_token"], MS_TOKEN_DECIMALS),
            DECODE_COLUMNS[3]: _span(row["ms_per_token_min"], row["ms_per_token_max"], MS_TOKEN_DECIMALS),
            DECODE_COLUMNS[4]: _fmt(row["tok_per_s"], TOKS_DECIMALS),
            DECODE_COLUMNS[5]: _mib(row["peak_bytes"]),
        }
        for row in doc["decode"]
    ]
    return Section(
        variant=doc["variant"],
        mode=doc["mode"],
        source=_bench_source(doc),
        prefill=prefill,
        decode=decode,
        kv=_kv_cells(doc["kv_bytes_per_token"], doc["max_batch_at_8k"]),
    )


def _blank_section(variant: str, mode: str) -> Section:
    prefill = [
        {**{c: BLANK for c in PREFILL_COLUMNS}, PREFILL_COLUMNS[0]: _text(context)}
        for context in bd.DEFAULT_CONTEXTS
    ]
    decode = [
        {**{c: BLANK for c in DECODE_COLUMNS}, DECODE_COLUMNS[0]: _text(batch), DECODE_COLUMNS[1]: _text(cache)}
        for batch in bd.DEFAULT_DECODE_BATCHES
        for cache in bd.DEFAULT_DECODE_CACHES
    ]
    kv = {c: BLANK for c in KV_COLUMNS}
    return Section(variant=variant, mode=mode, source="Source: no bench document given; measured cells are blank.",
                   prefill=prefill, decode=decode, kv=kv)


def inference_tables(docs: Sequence[Mapping] | None) -> list[Section]:
    """One section per (variant, mode): headline variants first, opt then ref. None or [] gives the blank form,
    which has every headline variant in both modes (the ref sections are what `bench.py --with-ref` writes)."""
    if not docs:
        return [_blank_section(variant, mode) for variant in HEADLINE for mode in MODES]
    check_bench_documents(docs)
    ordered = sorted(docs, key=lambda d: (_rank(d["variant"]), MODES.index(d["mode"])))
    return [_section(doc) for doc in ordered]


def _section_lines(section: Section) -> list[str]:
    lines = [f"## `{section.variant}`, {section.mode} mode", "", section.source, ""]
    lines += ["### Prefill", ""] + _md_table(PREFILL_COLUMNS, section.prefill) + [""]
    lines += ["### Decode", ""] + _md_table(DECODE_COLUMNS, section.decode) + [""]
    lines += ["### KV bytes and maximum batch", ""] + _md_table(KV_COLUMNS, [section.kv]) + [""]
    return lines


def render_inference(docs: Sequence[Mapping] | None) -> str:
    """inference.md from bench/2 documents, or the blank form for None or []."""
    lines = [INFERENCE_TITLE, ""] + list(INFERENCE_CAPTIONS) + [""]
    for section in inference_tables(docs):
        lines += _section_lines(section)
    return "\n".join(lines)


# --- Command line ------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write timing.md and inference.md from timing and bench documents.")
    parser.add_argument("--timing-pass", default=None, metavar="FILE", help="a timing_pass/1 document")
    parser.add_argument("--swa-paths", default=None, metavar="FILE", help="a swa_paths/1 record (scripts.swa_paths)")
    parser.add_argument("--bench", default=None, metavar="DIR", help="a directory of bench_*.json documents")
    parser.add_argument("--out", required=True, help="directory to write timing.md and inference.md to")
    parser.add_argument("--blank", action="store_true", help="ignore the sources; write the blank forms")
    args = parser.parse_args(argv)
    try:
        swa_doc = None
        if args.blank:
            timing_doc, bench_docs = None, []
        else:
            timing_doc = tp.read_timing_pass(args.timing_pass) if args.timing_pass else None
            bench_docs = load_bench_documents(args.bench) if args.bench else []
            swa_doc = sp.read_swa_paths(args.swa_paths) if args.swa_paths else None
        if timing_doc is not None:
            check_timing_document(timing_doc)
        check_bench_documents(bench_docs)
        mt.check_timing_sources([], timing_doc, bench_docs)  # the record is not cross-checked (decision 247)
        texts = {
            TIMING_NAME: render_timing(timing_doc, swa_paths=swa_doc),
            INFERENCE_NAME: render_inference(bench_docs),
        }
    except (
        TimingReportError, mt.TableError, tp.TimingError, bd.BenchError, sp.SwaPathsError, json.JSONDecodeError,
    ) as exc:
        print(exc, file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        path = out / name
        path.write_text(text, encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
