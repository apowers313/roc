# mypy: disable-error-code="no-untyped-def"

"""Lightweight conftest for unit tests -- no DB, no observability, no logger init."""

from typing import Generator

import pytest

from roc.config import Config


@pytest.fixture(autouse=True)
def unit_config_init() -> Generator[None, None, None]:
    """Minimal config init for unit tests. No logger, no observability beyond defaults."""
    Config.reset()
    Config.init()
    settings = Config.get()
    settings.observation_shape = (21, 79)
    settings.gym_actions = (
        ord("j"),
        ord("k"),
        ord("h"),
        ord("l"),
        ord("y"),
        ord("u"),
        ord("b"),
        ord("n"),
        ord("e"),
        ord("."),
    )

    yield

    Config.reset()
    Config.init()


@pytest.fixture
def reset_observability() -> Generator[None, None, None]:
    """Reset observability singleton for tests that need it."""
    from roc.reporting.observability import Observability, ObservabilityBase

    orig_instances = ObservabilityBase._instances.copy()
    orig_remote_log = getattr(Observability, "_remote_log_configured", False)

    yield

    ObservabilityBase._instances = orig_instances
    Observability._remote_log_configured = orig_remote_log
