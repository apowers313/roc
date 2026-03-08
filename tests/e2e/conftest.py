# mypy: disable-error-code="no-untyped-def"

"""Conftest for e2e tests -- inherits fixtures from root tests/conftest.py.
Only adds the all_components fixture for full pipeline testing.
"""

import gc
from typing import Generator

import pytest

from roc.component import Component


@pytest.fixture
def all_components() -> Generator[None, None, None]:
    """Load all components (perception, attention, objects, sequencer, etc.)."""
    Component.reset()
    gc.collect(2)
    assert Component.get_component_count() == 0

    Component.init()

    yield

    Component.reset()
    gc.collect(2)
