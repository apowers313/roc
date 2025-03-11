# ruff: noqa: F401 E402
"""Reinforcement Learning of Concepts"""

from threading import Thread
from typing import Any

# not used here, but the files have to be loaded in order for the components to
# be registered
import roc.feature_extractors
import roc.logger as roc_logger

from .action import Action, ActionData
from .component import Component
from .config import Config
from .expmod import ExpMod
from .gymnasium import NethackGym
from .jupyter import RocJupyterMagics, is_jupyter
from .perception import Perception, PerceptionData
from .reporting.observability import Observability
from .reporting.state import State

__all__ = [
    # Component Exports
    "Component",
    # Gym Exports
    "GymComponent",
    # Perception Interface Exports
    "Perception",
    "PerceptionData",
    # Action Exports
    "ActionData",
    "Action",
    # Experiment Module Support
    "ExpMod",
]


ng: NethackGym | None = None


def init(config: dict[str, Any] | None = None) -> None:
    """Initializes the agent before starting the agent."""
    Config.init(config)
    roc_logger.init()
    Observability.init()
    global ng
    ng = NethackGym()
    Component.init()
    ExpMod.init()
    RocJupyterMagics.init()
    State.init()


def start() -> None:
    """Starts the agent."""
    global ng
    if ng is None:
        raise Exception("Call .init() before .start()")

    if is_jupyter():
        # if running in Jupyter, start in a thread so that we can still inspect
        # or debug from the iPython shell
        roc_logger.logger.debug("Starting ROC: running in thread")
        t = Thread(target=ng.start)
        t.start()
    else:
        roc_logger.logger.debug("Starting ROC: NOT running in thread")
        ng.start()
