"""Design tokens for ROC Panel components (compact-mantine dark palette)."""

import html as html_mod

BG = "#0d1117"
SURFACE = "#161b22"
SURFACE_EL = "#1f2428"
INPUT_BG = "#2a3035"
BORDER = "#30363d"
TEXT = "#c9d1d9"
TEXT_DIM = "#8b949e"
TEXT_MUTED = "#484f58"
ACCENT = "#58a6ff"
SUCCESS = "#3fb950"
WARNING = "#d29922"
ERROR = "#f85149"
FONT = "'JetBrains Mono','Fira Code',Consolas,'Liberation Mono',monospace"
FONT_SIZE = "11px"
FONT_SIZE_SMALL = "10px"
FONT_SIZE_GRID = "9px"

# Shared CSS for compact Tabulator tables
COMPACT_TABLE_CSS = f"""
:host .tabulator {{
    font-family: {FONT} !important;
    font-size: {FONT_SIZE} !important;
    background: transparent !important;
    border: none !important;
}}
:host .tabulator-header {{
    display: none !important;
}}
:host .tabulator-tableholder {{
    background: transparent !important;
}}
:host .tabulator-row {{
    background: transparent !important;
    border: none !important;
    min-height: 0 !important;
}}
:host .tabulator-row.tabulator-row-even {{
    background: transparent !important;
}}
:host .tabulator-cell {{
    padding: 1px 4px !important;
    border: none !important;
    color: {TEXT} !important;
    background: transparent !important;
}}
:host .tabulator-cell:first-child {{
    color: {TEXT_DIM} !important;
}}
"""

# Shared CSS for compact Number indicators
COMPACT_NUMBER_CSS = f"""
:host {{
    font-family: {FONT};
    --design-primary-text-color: {TEXT};
    --design-background-text-color: {TEXT};
}}
"""


def no_data_html(what: str) -> str:
    """Render a compact no-data placeholder."""
    return f'<div style="color:{TEXT_MUTED};font-size:11px;padding:4px 0;">No {what} data</div>'


def title_html(title: str) -> str:
    """Render a compact sub-panel title."""
    return (
        f'<div style="font-size:10px;color:{TEXT_DIM};font-family:{FONT};'
        f"font-weight:500;padding:0 0 2px 0;text-transform:uppercase;"
        f'letter-spacing:0.5px;">{html_mod.escape(title)}</div>'
    )
