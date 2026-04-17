# mypy: disable-error-code="no-untyped-def"

"""Lightweight conftest for unit tests -- no DB, no observability, no logger init."""

from typing import Generator

import pytest

from roc.framework.config import Config
from roc.framework.expmod import ExpMod, expmod_modtype_current, expmod_registry


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


def _restore_expmod_registry(
    registry: dict[str, dict[str, ExpMod]], snapshot: dict[str, set[str]]
) -> None:
    """Remove entries from registry that were added after snapshot was taken."""
    for modtype in list(registry.keys()):
        if modtype not in snapshot:
            del registry[modtype]
            continue
        for name in [*registry[modtype].keys()]:
            if name not in snapshot[modtype]:
                del registry[modtype][name]


@pytest.fixture(autouse=True)
def clean_expmod_state():
    """Save and restore expmod global state between tests.

    Only removes entries added during the test -- does not blow away
    pre-existing entries that other modules may have registered.
    """
    orig_registry_snapshot = {k: set(v.keys()) for k, v in expmod_registry.items()}
    orig_current = dict(expmod_modtype_current)

    yield

    _restore_expmod_registry(expmod_registry, orig_registry_snapshot)

    expmod_modtype_current.clear()
    expmod_modtype_current.update(orig_current)
