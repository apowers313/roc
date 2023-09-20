# ruff: noqa: F401 E402
"""Reinforcement Learning of Concepts"""

# from icecream import install as install_icecream

# install_icecream()

import roc.logger as logger

from .action import ActionData, action_bus
from .component import Component
from .config import Config
from .gymnasium import NethackGym
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


def init() -> None:
    """Initializes the agent before starting the agent."""

    Config.init()
    logger.init()
    Component.init()
    # Gym.init()


def start() -> None:
    """Starts the agent."""

    g = NethackGym()
    g.start()
