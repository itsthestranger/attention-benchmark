"""SYNTHETIC commitments for the fake campaigns (decision 220). Nothing here is a real declaration.

`configs/commitments.yaml` holds two declarations Phil makes during the campaign: the ladder
(`full`, or `l6_l8` when L10 is cut, plan §12.2 item 4) and the committed seed count (5 or 8,
plan §1.4). The analysis tests no primary while the seed count is missing or disagrees with what
the rule computes. A fake campaign stands for a campaign after its gate, so its tests commit what
the rule gives for those runs: `committed` computes that, and the wrappers below pass it on.

Tests of the commitment itself (a missing one, a wrong one) call the scripts directly. No test
depends on what the repository's own configs/commitments.yaml holds: Phil fills that file in
during the campaign (review S5, H1).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import make_plots, make_tables
from src.utils import manifest as mf

LADDER_FULL = "full"
LADDER_L6_L8 = "l6_l8"
UNCOMMITTED = {"ladder": LADDER_FULL, "seed_count": None}


def committed(results, prereg=None, *, ladder: str = LADDER_FULL) -> dict:
    """{"ladder", "seed_count"}: the declared ladder and the seed count the rule gives for these runs.

    The seed count is None when the rule has no decision yet, and when the runs are refused: the
    caller's own call then raises the same refusal.
    """
    try:
        analysis = make_tables.analyse(list(results), prereg, commitments={"ladder": ladder, "seed_count": None})
    except (make_tables.TableError, mf.ManifestError):
        return {"ladder": ladder, "seed_count": None}
    decision = analysis.decision
    return {"ladder": ladder, "seed_count": None if decision is None else decision.n_seeds}


def analyse_committed(results, *args, **kwargs):
    """make_tables.analyse with the rule's seed count committed, unless the caller passes commitments."""
    results = list(results)
    if "commitments" not in kwargs:
        prereg = args[0] if args else kwargs.get("prereg")
        kwargs["commitments"] = committed(results, prereg)
    return make_tables.analyse(results, *args, **kwargs)


def make_inputs_committed(manifests, *args, **kwargs):
    """make_plots.make_inputs with the rule's seed count committed, unless the caller passes commitments."""
    manifests = list(manifests)
    if "commitments" not in kwargs:
        kwargs["commitments"] = _committed_for_manifests(manifests, kwargs.get("prereg"))
    return make_plots.make_inputs(manifests, *args, **kwargs)


def _committed_for_manifests(manifests, prereg=None, *, ladder: str = LADDER_FULL) -> dict:
    try:
        results, _ = make_tables.select_results(list(manifests), skip_unfinished=True)
    except (make_tables.TableError, mf.ManifestError):
        return {"ladder": ladder, "seed_count": None}
    return committed(results, prereg, ladder=ladder)


def write_commitments(path: Path, commitments: dict) -> Path:
    path = Path(path)
    path.write_text(yaml.safe_dump(dict(commitments), sort_keys=False), encoding="utf-8")
    return path


def held_for(argv, *, ladder: str = LADDER_FULL) -> dict | None:
    """What the repository's file would hold after the gate, for a command line that reads --manifests DIR.

    None for --blank and for a command line without --manifests: the command then reads what it
    reads by itself. A directory that cannot be read gives an uncommitted declaration: the command
    then reports its own error.
    """
    argv = [str(a) for a in argv]
    if "--blank" in argv or "--manifests" not in argv:
        return None
    directory = Path(argv[argv.index("--manifests") + 1])
    try:
        manifests = make_tables.load_manifests(directory)
    except (make_tables.TableError, mf.ManifestError, json.JSONDecodeError, OSError):
        return dict(UNCOMMITTED, ladder=ladder)
    return _committed_for_manifests(manifests, ladder=ladder)


def tables_main(argv) -> int:
    """make_tables.main with the rule's seed count committed for the manifests it reads."""
    return make_tables.main(list(argv), commitments=held_for(argv))


def plots_main(argv) -> int:
    """make_plots.main with the rule's seed count committed for the manifests it reads."""
    return make_plots.main(list(argv), commitments=held_for(argv))
