"""Delete checkpoints that no later command needs (plan §12.3).

    python -m scripts.prune --out OUT [--mode opt|ref] [--final] [--apply]

Dry run by default: nothing is deleted unless --apply is given. The plan is built
completely before anything is deleted, and --apply deletes exactly the files the dry
run lists.

What may go, per manifest in OUT/manifests/MODE/:

1. A completed, valid run: its regular checkpoints except the one at its final step.
   Its branch point of a budget goes only when that budget's branch manifest exists
   and is a completed, valid branch of this run (same parent configuration, budget
   and parent name).
2. A completed, valid branch: the regular checkpoints of the branch's own directory
   except the one at its final step.
3. With --final, the final-step checkpoint of a completed, valid run or branch.

Never deleted: anything of an unfinished, diverged, unreadable or invalid run or
branch; any directory no manifest accounts for; any file that is not a step or
trunk-step checkpoint; any manifest.

The configuration directories and file names come from the project's own functions
(src/training/checkpoint.py, src/training/branch.py, src/utils/manifest.py).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from scripts import _run_common as rc
from src import config
from src.training import branch as br
from src.training import checkpoint as ck
from src.utils import manifest as mf

MB = 1_000_000  # bytes per reported megabyte
_CHECKPOINT_NAME = re.compile(r"(trunk-)?step[0-9]+\.pt")


@dataclass(frozen=True)
class Deletion:
    path: Path   # absolute or relative to the working directory, as OUT was given
    name: str    # run name of the manifest whose rule authorises the deletion
    size: int    # bytes


@dataclass(frozen=True)
class Kept:
    name: str    # run name of the entry, or "" when no manifest owns it
    path: Path   # the file or directory left alone
    reason: str


@dataclass
class Plan:
    out: Path
    mode: str
    final: bool
    deletions: list[Deletion] = field(default_factory=list)
    kept: list[Kept] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(d.size for d in self.deletions)


# --- Reading the manifests ---------------------------------------------------------------------


def _read_manifests(directory: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """(readable valid manifests by file name, unreadable or invalid ones by file name -> reason)."""
    valid: dict[str, dict] = {}
    bad: dict[str, str] = {}
    if not directory.is_dir():
        return valid, bad
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or not path.name.endswith(".json") or not path.is_file():
            continue
        try:
            valid[path.name] = mf.read_manifest(path)
        except (mf.ManifestError, OSError) as exc:
            lines = str(exc).splitlines() or [type(exc).__name__]
            bad[path.name] = lines[0]
    return valid, bad


def _own_dir(layout: rc.Layout, m: dict) -> Path:
    """The checkpoint directory of the configuration that a run or branch manifest describes."""
    if m["kind"] == "branch":
        return ck.checkpoint_dir(layout.checkpoints, br.branch_resolved(m["resolved"], m["branch"]["budget"]))
    return ck.checkpoint_dir(layout.checkpoints, m["resolved"])


def _diverged(m: dict) -> bool:
    metrics = m.get("final_metrics")
    return isinstance(metrics, dict) and metrics.get("diverged") is True


def _not_a_result(m: dict) -> str:
    if _diverged(m):
        return f"diverged at step {m['final_metrics']['step']} of {m['steps']['total']}: nothing deleted"
    return f"unfinished at step {m['steps']['attempted']} of {m['steps']['total']}: nothing deleted"


def _branch_blocker(manifests: dict, bad: dict, parent: dict, budget: str) -> str | None:
    """Why the branch point of `budget` must stay, or None when its branch is complete and valid."""
    fname = mf.manifest_filename({"kind": "branch", "name": mf.run_name(parent, budget)})
    if fname in bad:
        return f"branch {budget!r}: its manifest is unreadable or invalid"
    branch = manifests.get(fname)
    if branch is None:
        return f"branch {budget!r} has not been run (no manifest)"
    if not mf.is_result(branch):
        return f"branch {budget!r} is " + ("diverged" if _diverged(branch) else "unfinished")
    if (
        branch["kind"] != "branch"
        or branch["config_hash"] != config.config_hash(parent)
        or branch["branch"]["budget"] != budget
        or branch["branch"]["parent_run"] != mf.run_name(parent)
    ):
        return f"branch {budget!r}: its manifest does not belong to this run"
    return None


# --- Planning ----------------------------------------------------------------------------------


def _plan_checkpoints(result: Plan, name: str, directory: Path, total: int) -> None:
    """Regular checkpoints of one completed configuration, minus its final one (and the final with --final)."""
    files = ck.list_checkpoints(directory)
    final_name = ck._file_name(total, False)  # the project's own name for a regular checkpoint
    final = [f for f in files if f.name == final_name]
    regular = [f for f in files if f.name != final_name]
    if not final and regular:
        # The final checkpoint is gone; keep the newest regular one as the fallback.
        fallback = regular.pop()
        result.kept.append(Kept(name, fallback, "final checkpoint is missing; newest regular checkpoint kept"))
    for path in regular:
        result.deletions.append(Deletion(path, name, path.stat().st_size))
    for path in final:
        if result.final:
            result.deletions.append(Deletion(path, name, path.stat().st_size))
        else:
            result.kept.append(Kept(name, path, "final checkpoint kept (evaluation loads it; --final removes it)"))


def _plan_run(result: Plan, layout: rc.Layout, m: dict, manifests: dict, bad: dict) -> None:
    name, resolved = m["name"], m["resolved"]
    directory = _own_dir(layout, m)
    if not mf.is_result(m):
        result.kept.append(Kept(name, directory, _not_a_result(m)))
        return

    _plan_checkpoints(result, name, directory, resolved["run"]["steps"])

    budget_of = {
        ck._file_name(br.branch_step(resolved, budget), True): budget  # the project's own branch-point name
        for budget in resolved["run"]["branches"]
    }
    for path in ck.list_branch_points(directory):
        budget = budget_of.get(path.name)
        if budget is None:
            result.kept.append(Kept(name, path, "not a branch point of this run: left alone"))
            continue
        blocker = _branch_blocker(manifests, bad, resolved, budget)
        if blocker is None:
            result.deletions.append(Deletion(path, mf.run_name(resolved, budget), path.stat().st_size))
        else:
            result.kept.append(Kept(name, path, f"branch point kept: {blocker}"))


def _plan_branch(result: Plan, layout: rc.Layout, m: dict) -> None:
    name = m["name"]
    directory = _own_dir(layout, m)
    if not mf.is_result(m):
        result.kept.append(Kept(name, directory, _not_a_result(m)))
        return
    total = br.branch_resolved(m["resolved"], m["branch"]["budget"])["run"]["steps"]
    _plan_checkpoints(result, name, directory, total)


def plan(out, mode: str, *, final: bool = False) -> Plan:
    """What would be deleted and what is left alone, in OUT/checkpoints/MODE. Reads only; deletes nothing."""
    out = Path(out)
    if not out.is_dir():
        raise rc.EntryError(f"{out} is not a directory; nothing to prune")
    layout = rc.Layout(out, mode)
    result = Plan(out=out, mode=mode, final=final)

    manifests, bad = _read_manifests(layout.manifests)
    for fname, reason in bad.items():
        result.kept.append(Kept(
            Path(fname).stem, layout.manifests / fname,
            f"manifest unreadable or invalid; nothing of it is deleted: {reason}",
        ))

    owned: set[Path] = set()      # directories a run or branch manifest describes
    accounted: set[Path] = set()  # directories some readable manifest refers to
    for m in manifests.values():
        if m["kind"] in ("run", "branch"):
            owned.add(_own_dir(layout, m))
        accounted.add(ck.checkpoint_dir(layout.checkpoints, m["resolved"]))
    accounted |= owned

    for fname in sorted(manifests):
        m = manifests[fname]
        if m["kind"] == "run":
            _plan_run(result, layout, m, manifests, bad)
        elif m["kind"] == "branch":
            _plan_branch(result, layout, m)
        else:
            result.kept.append(Kept(m["name"], layout.manifests / fname,
                                    "trunk checkpoint manifest, not a result: nothing deleted for it"))

    if layout.checkpoints.is_dir():
        for entry in sorted(layout.checkpoints.iterdir()):
            if entry in owned:
                continue
            if not entry.is_dir():
                result.kept.append(Kept("", entry, "not a directory: left alone"))
            elif entry in accounted:
                result.kept.append(Kept("", entry, "no readable run manifest for this configuration: left alone"))
            else:
                result.kept.append(Kept("", entry, "unknown, no manifest accounts for it: left alone"))
    return result


# --- Deleting ----------------------------------------------------------------------------------


def apply(p: Plan) -> int:
    """Delete exactly the files of the plan. Every file is checked first; returns the bytes freed."""
    root = rc.Layout(p.out, p.mode).checkpoints
    for d in p.deletions:
        if d.path.parent.parent != root or not _CHECKPOINT_NAME.fullmatch(d.path.name):
            raise rc.EntryError(f"{d.path} is not a checkpoint of {root}; nothing deleted")
        if d.path.is_symlink() or not d.path.is_file():
            raise rc.EntryError(f"{d.path} is not a regular file; nothing deleted")
    freed = 0
    for d in p.deletions:
        try:
            size = d.path.stat().st_size
        except FileNotFoundError:
            continue  # already gone: a second apply finds nothing
        d.path.unlink()
        freed += size
    return freed


# --- Command line -----------------------------------------------------------------------------


def _rel(out: Path, path: Path) -> str:
    return path.relative_to(out).as_posix() if path.is_relative_to(out) else str(path)


def _lines(p: Plan, verb: str) -> list[str]:
    groups: dict[str, list[str]] = {}
    for d in sorted(p.deletions, key=lambda d: (d.name, str(d.path))):
        groups.setdefault(d.name, []).append(f"  {verb} {_rel(p.out, d.path)} ({d.size / MB:.1f} MB)")
    for k in sorted(p.kept, key=lambda k: (k.name, str(k.path))):
        groups.setdefault(k.name, []).append(f"  keep {_rel(p.out, k.path)}: {k.reason}")
    lines: list[str] = []
    for name in sorted(groups):
        lines.append(name if name else "(no manifest)")
        lines.extend(groups[name])
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Delete checkpoints no later command needs (plan §12.3). "
                                                 "Dry run unless --apply is given.")
    parser.add_argument("--out", required=True, metavar="OUT", help="the output directory of the runs")
    parser.add_argument("--mode", default="opt", help="opt or ref (default opt)")
    parser.add_argument("--final", action="store_true",
                        help="also delete the final-step checkpoints of completed runs and branches")
    parser.add_argument("--apply", action="store_true", help="delete the files the dry run lists")
    args = parser.parse_args(argv)

    def body() -> int:
        p = plan(args.out, args.mode, final=args.final)
        if not args.apply:
            for line in _lines(p, "would delete"):
                print(line)
            if p.deletions:
                print(f"total: {len(p.deletions)} files, {p.total_bytes / MB:.1f} MB would be freed; "
                      "dry run, nothing deleted (rerun with --apply)")
            else:
                print("nothing to delete")
            return 0
        freed = apply(p)
        for line in _lines(p, "deleted"):
            print(line)
        if p.deletions:
            print(f"total: {len(p.deletions)} files, {freed / MB:.1f} MB freed")
        else:
            print("nothing to delete")
        return 0

    return rc.guarded(body)


if __name__ == "__main__":
    raise SystemExit(main())
