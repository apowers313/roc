"""Reinforcement Learning of Concepts"""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    """Gets the version of this package

    Returns:
        str: The version string of the package in MAJOR.MINOR.REVISION format, or unknown if the version wasn't set.
    """
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
