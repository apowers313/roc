# ruff: noqa: F401 E402
"""Reinforcement Learning of Concepts"""

from typing import Any

# not used here, but the files have to be loaded in order for the components to
# be registered
import roc.feature_extractors  # noqa: F401
import roc.logger as logger

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
    logger.init()
    Component.init()
    # Gym.init()
    RocJupyterMagics.init()


def start() -> None:
    """Starts the agent."""

    g = NethackGym()
    g.start()
