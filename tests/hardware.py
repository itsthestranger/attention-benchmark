"""Hardware tiers (plan §5.1): which marked tests can run on this machine.

`gpu` tests need any CUDA card; `t4` tests need the Kaggle T4. Without the
hardware they skip, and a skip is never a pass: the handover lists every test
that has not run.
"""

from __future__ import annotations

GPU_MARKER = "gpu"
T4_MARKER = "t4"
T4_NAME_FRAGMENT = "T4"


def detect() -> tuple[bool, str | None]:
    """(CUDA available, name of device 0 or None)."""
    import torch

    if not torch.cuda.is_available():
        return False, None
    return True, torch.cuda.get_device_name(0)


def skip_reason(markers: set[str], cuda_available: bool, gpu_name: str | None) -> str | None:
    """Why a test with these markers cannot run here, or None if it can."""
    if T4_MARKER in markers:
        if not cuda_available:
            return "t4 tier: no CUDA device (plan §5.1); not run, not passed"
        if gpu_name is None or T4_NAME_FRAGMENT not in gpu_name:
            return f"t4 tier: device is {gpu_name!r}, not a T4 (plan §5.1); not run, not passed"
    if GPU_MARKER in markers and not cuda_available:
        return "gpu tier: no CUDA device (plan §5.1); not run, not passed"
    return None
