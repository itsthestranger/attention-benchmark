"""T-00: the scaffold protects what the plan says it protects."""

import re
import tomllib
from pathlib import Path

import pytest
import torch

from tests import hardware
from tests.tolerances import scale_relative_error

ROOT = Path(__file__).resolve().parent.parent

# Plan §17.3. The pre-registration is a section of README.md; CODEOWNERS works
# per file, so the whole file is covered.
FROZEN_PATHS = [
    "/docs/plan_v5.md",
    "/configs/base.yaml",
    "/configs/recipe.yaml",
    "/configs/prereg.yaml",
    "/tests/tolerances.py",
    "/README.md",
]


def _codeowners() -> dict[str, list[str]]:
    rules = {}
    for line in (ROOT / ".github" / "CODEOWNERS").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            path, *owners = line.split()
            rules[path] = owners
    return rules


@pytest.mark.parametrize("path", FROZEN_PATHS)
def test_frozen_path_has_code_owner(path):
    rules = _codeowners()
    assert path in rules, f"{path} is frozen (plan §17.3) but not in CODEOWNERS"
    assert rules[path] and all(o.startswith("@") for o in rules[path])


def test_codeowners_protects_itself():
    assert "/.github/CODEOWNERS" in _codeowners()


def test_markers_registered_and_strict():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    names = {m.split(":", 1)[0] for m in cfg["markers"]}
    assert names == {hardware.GPU_MARKER, hardware.T4_MARKER}
    assert "--strict-markers" in cfg["addopts"]


def test_ci_runs_the_cpu_tier():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert 'pytest -m "not gpu and not t4"' in ci
    assert "pull_request" in ci


@pytest.mark.parametrize(
    "markers, cuda, name, runs",
    [
        (set(), False, None, True),
        (set(), True, "Tesla T4", True),
        ({"gpu"}, False, None, False),
        ({"gpu"}, True, "GeForce GTX 1070", True),
        ({"gpu"}, True, "Tesla T4", True),
        ({"t4"}, False, None, False),
        ({"t4"}, True, "GeForce GTX 1070", False),
        ({"t4"}, True, "Tesla P100-PCIE-16GB", False),
        ({"t4"}, True, "Tesla T4", True),
        ({"gpu", "t4"}, True, "GeForce GTX 1070", False),
    ],
)
def test_hardware_tiers(markers, cuda, name, runs):
    reason = hardware.skip_reason(markers, cuda, name)
    assert (reason is None) == runs
    if not runs:
        assert "not passed" in reason


@pytest.mark.gpu
def test_gpu_tier_canary():
    """Runs only where the gpu tier can run; shows up as skipped elsewhere."""
    assert torch.cuda.is_available()


@pytest.mark.t4
def test_t4_tier_canary():
    """Runs only on the Kaggle T4, which is sm75 (plan §2.3)."""
    assert hardware.T4_NAME_FRAGMENT in torch.cuda.get_device_name(0)
    assert torch.cuda.get_device_capability(0) == (7, 5)


# Skipping is decided in one place, from hardware alone. Anything else that
# skips or expects failure is a weakened test (CLAUDE.md, hard rules).
_FORBIDDEN = re.compile(r"\bxfail\b|mark\.skip|pytest\.skip\(|importorskip|\bskipif\b|unittest\.skip")
_SKIP_MACHINERY = {"conftest.py", "test_scaffold.py"}


def test_no_test_skips_or_expects_failure():
    offenders = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if path.name in _SKIP_MACHINERY:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _FORBIDDEN.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_scale_relative_error_definition():
    b = torch.tensor([0.0, 2.0, -4.0])
    assert scale_relative_error(b, b) == 0.0
    assert scale_relative_error(b + torch.tensor([0.5, 0.0, 0.0]), b) == 0.125
    assert scale_relative_error(3.0, 4.0) == 0.25
    zeros = torch.zeros(3)
    assert scale_relative_error(zeros, zeros) == 0.0
    assert scale_relative_error(zeros + 1.0, zeros) == float("inf")
