# ruff: noqa: F401
"""Framework infrastructure -- component system, events, config, expmod."""

from roc.framework.component import (
    Component,
    ComponentId,
    ComponentKey,
    ComponentName,
    ComponentType,
)
from roc.framework.config import Config
from roc.framework.event import BusConnection, Event, EventBus
from roc.framework.expmod import ExpMod
# Note: do NOT re-export 'logger' here -- it shadows the logger module
# and breaks 'import roc.framework.logger' (resolves to the Logger object
# instead of the module). Use 'from roc.framework.logger import logger' directly.
