# ruff: noqa: F401
"""Reinforcement Learning of Concepts"""

import roc.logger as logger

from .action import ActionData, action_bus
from .component import Component
from .config import Config
from .gymnasium import NethackGym
from .perception import PerceptionData, perception_bus

__all__ = [
    # Component Exports
    "Component",
    # Gym Exports
    "GymComponent",
    # Perception Interface Exports
    "perception_bus",
    "PerceptionData",
    # Action Exports
    "action_bus",
    "ActionData",
]


def init() -> None:
    Config.init()
    logger.init()
    Component.init()
    # Gym.init()


def start() -> None:
    g = NethackGym()
    g.start()
