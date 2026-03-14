"""Reusable Panel components for the ROC debug dashboard."""

from roc.reporting.components.charts import event_bar_chart
from roc.reporting.components.grid_viewer import GridViewer
from roc.reporting.components.resolution_inspector import ResolutionInspector
from roc.reporting.components.status_bar import compact_status_bar
from roc.reporting.components.tables import compact_kv_table, compact_log_table

__all__ = [
    "compact_kv_table",
    "compact_log_table",
    "compact_status_bar",
    "event_bar_chart",
    "GridViewer",
    "ResolutionInspector",
]
