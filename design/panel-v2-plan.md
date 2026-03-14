# Panel Dashboard v2 -- Fresh Build Plan

Rewrite the Panel debug dashboard from scratch using Panel's built-in components and
theming instead of custom CSS/HTML. The data layer (RunStore, StepData, ParquetExporter)
is unchanged -- only the UI layer is rewritten.

## Guiding Principles

- Use Panel components as intended -- no global CSS, no `!important`, no raw HTML for
  things Panel provides components for
- Let Panel's theme system handle light/dark mode (FastListTemplate theme toggle)
- Bokeh instead of Vega-Lite for charts (already a dependency, theme-aware)
- `pn.indicators.Number` for metric badges, `pn.pane.Str`/`pn.pane.Markdown` for text
- Raw HTML only where genuinely necessary (screen renderer -- per-character colored spans)
- Follow design/panel-best-practices.md

## What Stays Unchanged

- `roc/reporting/run_store.py` -- data API
- `roc/reporting/parquet_exporter.py` -- data export
- `roc/reporting/screen_renderer.py` -- `render_grid_pane()` genuinely needs HTML for
  per-character coloring. Terminal screen background stays black in both light/dark mode
  (it is a terminal emulator, black background is expected).
- Entry point `panel-debug = "roc.reporting.panel_debug:main"`

## What Gets Deleted

- `roc/reporting/components/tokens.py` -- hardcoded colors, HTML helpers
- `roc/reporting/components/status_bar.py` -- 100% raw HTML
- `roc/reporting/components/charts.py` -- Vega-Lite dependency
- `roc/reporting/components/tables.py` -- rebuild without `!important` CSS
- `roc/reporting/components/grid_viewer.py` -- rebuild without HTML helpers
- `roc/reporting/components/resolution_inspector.py` -- rebuild without HTML helpers
- `roc/reporting/components/__init__.py` -- rebuild exports
- `roc/reporting/panel_debug.py` -- rebuild from scratch

## New File Structure

```
roc/reporting/
    panel_debug.py          # Dashboard Viewer + main() entry point
    run_store.py            # (unchanged)
    parquet_exporter.py     # (unchanged)
    screen_renderer.py      # (unchanged)
    components/
        __init__.py         # Re-exports
        grid_viewer.py      # Viewer wrapping screen_renderer (pn.pane.HTML, justified)
        resolution_inspector.py  # Viewer using Tabulator + pn.pane.Str
        theme.py            # Shared compact stylesheet constants (scoped :host CSS)
```

## Build Phases

### Phase 1: Theme + skeleton

**Files:** `theme.py`, `panel_debug.py` (skeleton)

`theme.py`:
- Define a small set of scoped `:host` stylesheet strings for making Tabulator compact
  (hide headers on KV tables, tight padding)
- No color constants -- let Panel's theme handle colors
- No `!important` -- use specificity or scoped selectors

`panel_debug.py` skeleton:
- Minimal `PanelDashboard` Viewer
- `FastListTemplate(theme="dark")` with theme toggle enabled
- Player widget, run/game/speed selectors
- Empty Card layout with all sections
- Verify it serves and the light/dark toggle works

### Phase 2: Data panels using Panel components

**Status bar:**
- `pn.Row` of `pn.indicators.Number` with `colors=` for HP conditional coloring
- Shows HP, Score, Depth, Gold, Energy, Hunger
- Falls back to Step/Game display when no metrics available

**Info line:**
- `pn.pane.Str` for "Step N | Game N | timestamp"

**KV tables:**
- `pn.widgets.Tabulator` with `theme="midnight"` + compact scoped stylesheet from
  `theme.py`
- Used for: features, game metrics, graph summary, attenuation

**Log table:**
- `pn.widgets.Tabulator` with `theme="midnight"`
- `HTMLTemplateFormatter` for severity coloring (proper Tabulator API)
- Severity filtering via RadioButtonGroup

**No-data states:**
- `pn.pane.Str("No data")` instead of HTML helper functions

**Section titles:**
- Card titles handle section labeling
- Sub-section labels use `pn.pane.Markdown` if needed

### Phase 3: Charts + grid viewer

**Event bar chart:**
- Replace Vega-Lite with `bokeh.plotting.figure` + `hbar`
- Use `pn.pane.Bokeh(fig, theme="dark_minimal")` -- adapts to light/dark automatically

**GridViewer:**
- Viewer wrapping `render_grid_pane()` in `pn.pane.HTML`
- This is the one justified use of raw HTML
- Placeholder uses `pn.pane.Str`
- Title via Card, not HTML helper

### Phase 4: Resolution inspector

**Outcome badge:**
- `pn.pane.Str` with `styles={"color": ..., "font-weight": "bold"}`

**Summary:**
- Reuse KV Tabulator pattern from Phase 2

**Candidates:**
- `pn.widgets.Tabulator` with column headers shown (not hidden like KV tables)

**Features:**
- `pn.pane.Str` for comma-separated feature list

### Phase 5: Wiring + tests

**Wiring:**
- Wire all `param.watch` callbacks (same Viewer pattern -- widget -> param -> business
  logic -> component update)
- Run/game/speed/log-level selectors
- Step player with interval-based playback

**Tests:**
- Rewrite `tests/unit/test_panel_debug.py` to test the new component structure
- Test dashboard instantiation, widget interactions, data rendering
- No tests that depend on specific CSS classes or HTML content

**Cleanup:**
- Remove `vegalite` / `vega` from dependencies if not used elsewhere
- Delete backup files (`panel_debug.py.bak`, `components.bak/`)
- Delete old `test_panel_debug.py.bak`

## Layout Structure

```
FastListTemplate (theme="dark", theme toggle enabled)
+-- Transport Bar (sticky)
|   +-- Row: Run selector, Game selector, Speed selector
|   +-- Player widget (step slider with playback controls)
|   +-- Row: Info line (Str) | Status bar (Row of indicators.Number)
|
+-- Card: "Game State" (expanded)
|   +-- Row: Screen GridViewer | Column: Vitals table, Graph DB table, Events chart
|
+-- Card: "Perception" (collapsed)
|   +-- Features table
|
+-- Card: "Attention" (collapsed)
|   +-- Row: Saliency GridViewer | Column: Focus points table, Attenuation table
|
+-- Card: "Object Resolution" (collapsed)
|   +-- Row: Object info table | Resolution inspector
|
+-- Card: "Log Messages" (collapsed)
|   +-- Log level RadioButtonGroup
|   +-- Log table
```

## Data Flow

```
User changes widget (step/run/game/speed/log_level)
    |
    v
Widget change -> param.watch handler -> param sync
    |
    v
Business logic (_on_step_change, _on_run_change, etc.)
    |
    v
RunStore.get_step_data(step) queries Parquet via DuckDB
    |
    v
Update component .object/.value properties (no recreating Panel objects)
    |
    v
Panel reactivity updates browser view
```
