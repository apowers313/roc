# Panel Component Plan

## Motivation

The current `PanelDashboard` renders all data as raw HTML strings via helper
functions (`_render_kv_table_html`, `_stat_row`, etc.). This bypasses Panel's
component system entirely -- we're using `pn.pane.HTML` as a dumb container for
hand-built HTML.

Panel already has components designed for the visualizations we need:

| Our HTML hack | Panel component | What it gives us |
|---|---|---|
| `_render_kv_table_html()` | `pn.widgets.Tabulator` | Sortable, filterable, themed data table |
| `_stat_badge()` inline HTML | `pn.indicators.Number` | Colored metric with thresholds |
| Event counts as text | `pn.pane.Vega` | Bar chart, sparkline |
| Log lines as colored divs | `pn.widgets.Tabulator` | Scrollable, filterable log table |
| `render_grid_pane()` | `pn.pane.HTML` (keep) | Genuinely needs custom HTML |

By building ON TOP of these components and styling them compact, we get:
- **Built-in interactivity** -- sorting, filtering, hover tooltips for free
- **Consistent theming** -- `theme='midnight'`, `theme_classes=['table-sm']`
- **Smaller code** -- replace 50 lines of HTML string building with 5 lines of Tabulator config
- **Better performance** -- Tabulator uses virtual rendering for large datasets

The only custom HTML we keep is `render_grid_pane()` for the character grids
(screen and saliency), because no Panel component renders colored monospace
character grids.

## Architecture

### Compact Styling Strategy

Instead of building custom Viewer components that generate HTML, we:

1. **Use Panel's built-in components** (Tabulator, Number, Vega, Card)
2. **Style them compact** via `stylesheets` parameter on each component
3. **Create factory functions** that produce pre-configured compact instances
4. **Share design tokens** via a `tokens.py` module

This means our "components" are really **factory functions** that return standard
Panel components with compact styling applied:

```python
# Instead of:
class CompactTable(Viewer):
    def __panel__(self):
        return pn.pane.HTML(self._build_html_string())

# We do:
def compact_tabulator(df, **kwargs):
    return pn.widgets.Tabulator(
        df,
        theme="midnight",
        theme_classes=["table-sm"],
        stylesheets=[COMPACT_TABLE_CSS],
        **kwargs,
    )
```

### When to use a Viewer subclass

Only when the component needs **reactive state that isn't just data display**:

- `ResolutionInspector` -- needs to parse a complex decision dict and lay out
  multiple sub-panels (outcome badge + candidate table + feature list)
- `GridViewer` -- wraps `render_grid_pane()` with param-based data updates

Simple data displays (tables, metrics, charts) use factory functions, not Viewers.

## Shared Design Tokens

**File**: `roc/reporting/components/tokens.py`

```python
"""Design tokens for ROC Panel components (compact-mantine dark palette)."""

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
    font-family: {FONT};
    font-size: {FONT_SIZE};
    background: transparent;
    border: none;
}}
:host .tabulator-header {{
    display: none;
}}
:host .tabulator-row {{
    background: transparent;
    border: none;
    min-height: 0;
}}
:host .tabulator-cell {{
    padding: 1px 4px;
    border: none;
    color: {TEXT};
}}
:host .tabulator-cell:first-child {{
    color: {TEXT_DIM};
}}
"""

# Shared CSS for compact Number indicators
COMPACT_NUMBER_CSS = f"""
:host {{
    font-family: {FONT};
    --design-primary-text-color: {TEXT};
    --design-background-text-color: {TEXT};
}}
:host .bk-panel-models-widgets-indicators-Number {{
    font-size: {FONT_SIZE};
}}
"""
```

## Components and Factory Functions

### 1. `compact_kv_table(data, title)` -- Factory Function

**File**: `roc/reporting/components/tables.py`

**Purpose**: Replace `_render_kv_table_html`. Creates a Tabulator widget configured
for compact key-value display.

**Signature**:
```python
def compact_kv_table(
    data: dict[str, Any] | None,
    title: str = "",
    max_width: int = 320,
) -> pn.widgets.Tabulator | pn.pane.HTML:
```

**Behavior**:
- If `data` is None, return `pn.pane.HTML` with "No {title} data" placeholder
- Convert dict to a 2-column DataFrame (key, value)
- Truncate values > 80 chars
- Return `pn.widgets.Tabulator` with:
  - `theme="midnight"`, `theme_classes=["table-sm"]`
  - `show_index=False`, `header_filters=False`
  - `stylesheets=[COMPACT_TABLE_CSS]` (hides header, compact rows)
  - `width=max_width`, `height` auto-sized
  - No pagination

**Replaces**: `_render_kv_table_html()` (used 6 times), `_render_dict_html()`,
`_stat_row()` in key-value contexts.

**Example**:
```python
# Before:
self._metrics_pane.object = _render_kv_table_html(data.game_metrics, "game metrics")

# After:
self._metrics_table = compact_kv_table(data.game_metrics, "game metrics")
```

---

### 2. `compact_log_table(logs, min_level)` -- Factory Function

**File**: `roc/reporting/components/tables.py`

**Purpose**: Replace `_update_logs` HTML string building. Creates a Tabulator
for log display with severity coloring.

**Signature**:
```python
def compact_log_table(
    logs: list[dict[str, Any]] | None,
    min_level: str = "DEBUG",
) -> pn.widgets.Tabulator | pn.pane.HTML:
```

**Behavior**:
- Filter logs by severity
- Convert to DataFrame with columns: level, message
- Use Tabulator with row formatters for severity coloring:
  - `text_align` and `formatters` to color level column
- `theme="midnight"`, `theme_classes=["table-sm"]`
- `height=200`, scrollable
- Header hidden (levels self-evident from color)

**Replaces**: `_update_logs()` HTML string building.

---

### 3. `event_bar_chart(event_data)` -- Factory Function

**File**: `roc/reporting/components/charts.py`

**Purpose**: Replace text dump of event bus counts with a horizontal bar chart.
Uses Vega-Lite via `pn.pane.Vega`.

**Signature**:
```python
def event_bar_chart(
    event_data: dict[str, Any] | None,
) -> pn.pane.Vega | pn.pane.HTML:
```

**Behavior**:
- If None, return placeholder
- Build Vega-Lite spec:
  ```json
  {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "width": 280,
    "height": 100,
    "data": {"values": [{"bus": "perception", "count": 178}, ...]},
    "mark": "bar",
    "encoding": {
      "y": {"field": "bus", "type": "nominal", "sort": "-x",
            "axis": {"labelColor": "#8b949e", "labelFont": "...", "labelFontSize": 10}},
      "x": {"field": "count", "type": "quantitative",
            "axis": {"labelColor": "#8b949e", "grid": false}},
      "color": {"value": "#58a6ff"}
    },
    "config": {
      "view": {"stroke": "transparent"},
      "background": "transparent"
    }
  }
  ```
- Return `pn.pane.Vega(spec, sizing_mode="stretch_width")`

**Replaces**: `_update_events()` flat text/table rendering.

---

### 4. `GridViewer` -- Viewer Subclass

**File**: `roc/reporting/components/grid_viewer.py`

**Purpose**: Wraps `render_grid_pane()` with reactive param updates. This stays
as a Viewer because the grid rendering genuinely needs custom HTML (colored
monospace character spans) that no Panel component provides.

**Params**:
```python
class GridViewer(Viewer):
    grid_data = param.Dict(default=None, allow_None=True)
    title = param.String(default="")
```

**`__panel__`**:
```python
def __panel__(self):
    return pn.Column(
        pn.pane.HTML(tokens._title_html(self.title)) if self.title else pn.Spacer(height=0),
        self._html_pane,
        sizing_mode="stretch_width",
        styles={"gap": "0px"},
    )
```

**Reactive update**: `@param.depends('grid_data', watch=True)` method calls
`render_grid_pane()` and sets `self._html_pane.object`.

**Replaces**: Direct `render_grid_pane()` calls + `pn.pane.HTML` wiring for
screen and saliency panes.

---

### 5. `ResolutionInspector` -- Viewer Subclass

**File**: `roc/reporting/components/resolution_inspector.py`

**Purpose**: Visualizes object resolution decisions -- answers "why did the agent
match (or not match) this object?" This is a Viewer because it composes multiple
sub-panels (outcome badge + candidate table + feature list) from a single complex
data structure.

**Params**:
```python
class ResolutionInspector(Viewer):
    decision = param.Dict(default=None, allow_None=True)
```

**`__panel__`**: Composes:

1. **Outcome badge** -- `pn.pane.HTML` with colored text:
   - "MATCHED" (green), "NEW OBJECT" (blue), "LOW CONFIDENCE" (yellow)
2. **Summary** -- `compact_kv_table` with algorithm, location, num_candidates
3. **Candidates** -- `pn.widgets.Tabulator` showing candidate list:
   - For symmetric-difference: columns (object_id, distance), sorted ascending
   - For Dirichlet: columns (object_id, probability), sorted descending
   - Matched row highlighted
4. **Features** -- comma-separated feature list as compact text

**Expected decision dict** (from `object.py` `_log_decision` methods):

```python
# SymmetricDifferenceResolution:
{
    "event": "resolution_decision",
    "algorithm": "symmetric-difference",
    "outcome": "match" | "new_object",
    "tick": int,
    "x": int, "y": int,
    "features": ["FeatureStr1", ...],
    "num_candidates": int,
    "matched_object_id": int | None,
    "best_distance": float | None,
    "candidate_distances": [("obj_id", distance), ...],
}

# DirichletCategoricalResolution:
{
    "event": "resolution_decision",
    "algorithm": "dirichlet-categorical",
    "outcome": "match" | "new_object" | "low_confidence",
    "tick": int,
    "x": int, "y": int,
    "features": ["FeatureStr1", ...],
    "num_candidates": int,
    "matched_object_id": int | None,
    "posteriors": [("obj_id", probability), ...],
    "vocab_size": int,
    "total_objects_tracked": int,
    "matched_alpha_sum": float,
    "matched_alpha_count": int,
}
```

**Note**: The resolution system is actively evolving. The data format above reflects
the current `_log_decision` output from both resolution algorithms. If the format
changes, update this section and the component together.

---

### 6. `StatusBar` -- Factory Function

**File**: `roc/reporting/components/status_bar.py`

**Purpose**: Replace `_update_status_bar` with `pn.indicators.Number` widgets
styled compact.

**Signature**:
```python
def compact_status_bar(
    metrics: dict[str, Any] | None,
    step: int = 0,
    game_number: int = 0,
) -> pn.Row:
```

**Behavior**:
- Return a `pn.Row` of `pn.indicators.Number` widgets
- If metrics available: HP (with color thresholds), Score, Depth, Gold, Energy
- If no metrics: Step, Game
- Each Number: `font_size="11pt"`, `title_size="9pt"`, compact stylesheets
- HP uses `colors=[(0.25, ERROR), (0.5, WARNING), (1.0, SUCCESS)]`

**Note**: `pn.indicators.Number` may be too large at default sizing. If it
can't be made compact enough with stylesheets, fall back to `pn.pane.HTML`
with `_stat_badge()` (current approach). Test first.

---

## Implementation Order

### Phase 1: Foundation

1. Create `roc/reporting/components/__init__.py`
2. Create `roc/reporting/components/tokens.py` (extract design tokens)
3. Update `panel_debug.py` to import tokens from `components.tokens`
4. All existing tests pass (no behavior change)

### Phase 2: Tables

5. Create `roc/reporting/components/tables.py` with `compact_kv_table()`
6. Replace `_render_kv_table_html()` calls in `_update_features`, `_update_metrics`,
   `_update_graph`, `_update_events`, `_update_attenuation`
7. Create `compact_log_table()` in same file
8. Replace `_update_logs()` with `compact_log_table()`
9. Tests for table functions
10. Remove unused HTML helper functions

### Phase 3: Grid and Charts

11. Create `roc/reporting/components/grid_viewer.py` (GridViewer)
12. Replace screen and saliency `pn.pane.HTML` with `GridViewer` instances
13. Create `roc/reporting/components/charts.py` with `event_bar_chart()`
14. Replace `_update_events` table with bar chart
15. Tests for GridViewer and charts

### Phase 4: Resolution Inspector

16. Create `roc/reporting/components/resolution_inspector.py`
17. Add to Object Resolution card in dashboard
18. Tests for ResolutionInspector

### Phase 5: Status Bar (if feasible)

19. Test `pn.indicators.Number` with compact stylesheets
20. If compact enough, create `compact_status_bar()` factory
21. If not compact enough, keep current `_stat_badge()` HTML approach
22. Replace in dashboard if using Number indicators

## Testing Strategy

**Component tests** go in `tests/unit/test_components_<name>.py`.

Each test:
- Creates the component/calls the factory with test data
- Checks the returned Panel object type (Tabulator, Vega, HTML, Row)
- Checks that data is correctly displayed (Tabulator.value DataFrame has expected rows)
- Checks None/empty handling

**Integration tests** stay in `test_panel_debug.py`:
- Verify dashboard correctly wires components
- Existing 50 tests continue to pass

## File Structure

```
roc/reporting/
    components/
        __init__.py                  # Exports all public functions and classes
        tokens.py                    # Design tokens
        tables.py                    # compact_kv_table(), compact_log_table()
        charts.py                    # event_bar_chart()
        grid_viewer.py               # GridViewer (Viewer subclass)
        resolution_inspector.py      # ResolutionInspector (Viewer subclass)
        status_bar.py                # compact_status_bar()
    panel_debug.py                   # Dashboard (uses components)
    run_store.py                     # Unchanged
    screen_renderer.py               # Unchanged

tests/unit/
    test_components_tables.py
    test_components_charts.py
    test_components_grid.py
    test_components_resolution.py
    test_components_status.py
    test_panel_debug.py              # Existing (unchanged)
```

## References

- [Tabulator](https://panel.holoviz.org/reference/widgets/Tabulator.html) --
  theme="midnight", theme_classes=["table-sm"], stylesheets for compact sizing
- [Number indicator](https://panel.holoviz.org/reference/indicators/Number.html) --
  colors parameter for threshold-based coloring
- [Vega pane](https://panel.holoviz.org/reference/panes/Vega.html) --
  inline Vega-Lite specs, transparent background
- [Viewer](https://panel.holoviz.org/reference/custom_components/Viewer.html) --
  subclass pattern for custom components
- [Apply CSS](https://panel.holoviz.org/how_to/styling/apply_css.html) --
  stylesheets parameter, :host selector
- [Design Variables](https://panel.holoviz.org/how_to/styling/design_variables.html) --
  CSS variable overrides
