"""Shared compact stylesheets for ROC Panel components.

Most styling comes from Panel's built-in theme system and Tabulator's ``fast``
theme. This module only defines the minimal scoped CSS that cannot be achieved
through component parameters alone (cell padding).
"""

# Tight cell padding for compact KV tables. The ``fast`` Tabulator theme handles
# colors and dark mode; this only adjusts spacing.
# Apply via: Tabulator(..., stylesheets=[COMPACT_CELL_CSS])
COMPACT_CELL_CSS = """
:host .tabulator-cell {
    padding: 1px 4px;
}
"""
