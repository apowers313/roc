"""Reusable Panel components for the ROC debug dashboard."""

from roc.reporting.components.grid_viewer import GridViewer
from roc.reporting.components.resolution_inspector import ResolutionInspector
from roc.reporting.components.theme import COMPACT_CELL_CSS

__all__ = [
    "COMPACT_CELL_CSS",
    "GridViewer",
    "ResolutionInspector",
]
