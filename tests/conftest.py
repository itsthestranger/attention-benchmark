"""Applies the hardware tiers of plan §5.1 to every collected test."""

import pytest

from tests import hardware


def pytest_collection_modifyitems(config, items):
    cuda_available, gpu_name = hardware.detect()
    for item in items:
        markers = {m.name for m in item.iter_markers()}
        reason = hardware.skip_reason(markers, cuda_available, gpu_name)
        if reason is not None:
            item.add_marker(pytest.mark.skip(reason=reason))
