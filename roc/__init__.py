# ruff: noqa: F401
"""Reinforcement Learning of Concepts"""

from .action import ActionData, action_bus
from .component import Component
from .environment import EnvData, environment_bus
from .gymnasium import GymComponent

__all__ = [
    # Component Exports
    "Component",
    # Gym Exports
    "GymComponent",
    # Environment Exports
    "environment_bus",
    "EnvData",
    # Action Exports
    "action_bus",
    "ActionData",
]


def foo() -> None:
    pass


# import roc.event
# import roc.graphdb
# import roc.perception

# from importlib import metadata as importlib_metadata

# def get_version() -> str:
#     """Gets the version of this package

#     Returns:
#         str: The version string of the package in MAJOR.MINOR.REVISION format, or unknown if the \
#             version wasn't set.
#     """
#     try:
#         return importlib_metadata.version(__name__)
#     except importlib_metadata.PackageNotFoundError:  # pragma: no cover
#         return "unknown"


# version: str = get_version()
