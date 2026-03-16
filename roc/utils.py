"""Shared utility functions."""

from datetime import datetime


def _timestamp_str() -> str:
    """Returns a formatted timestamp string for use in filenames."""
    return datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
