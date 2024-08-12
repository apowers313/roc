# ruff: noqa: F401 E402
"""Reinforcement Learning of Concepts"""

from threading import Thread
from typing import Any

# not used here, but the files have to be loaded in order for the components to
# be registered
import roc.feature_extractors  # noqa: F401
import roc.logger as roc_logger
from roc.jupyter import is_jupyter

from .action import ActionData, action_bus
from .component import Component
from .config import Config
from .gymnasium import NethackGym
from .jupyter import RocJupyterMagics
from .perception import Perception, PerceptionData

__all__ = [
    # Component Exports
    "Component",
    # Gym Exports
    "GymComponent",
    # Perception Interface Exports
    "Perception",
    "PerceptionData",
    # Action Exports
    "action_bus",
    "ActionData",
]


def init(config: dict[str, Any] | None = None) -> None:
    """Initializes the agent before starting the agent."""

    Config.init(config)
    roc_logger.init()
    Component.init()
    # Gym.init()
    RocJupyterMagics.init()


def start() -> None:
    """Starts the agent."""

    g = NethackGym()

    if is_jupyter():
        # if running in Jupyter, start in a thread so that we can still inspect
        # or debug from the iPython shell
        roc_logger.logger.debug("Starting ROC: running in thread")
        t = Thread(target=g.start)
        t.start()
    else:
        roc_logger.logger.debug("Starting ROC: NOT running in thread")
        g.start()
