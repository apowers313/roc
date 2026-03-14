# Panel Debug Dashboard -- UI/UX Redesign Plan

## Problem Statement

The current dashboard has poor usability:
- Black text on white background with no visual grouping
- Layout too spread out, especially on tablet screens (iPad)
- No way to hide/show sections -- everything visible at once
- No visual hierarchy -- all data given equal weight
- Saliency map rendered identically to screen (should be a heatmap)
- Does not match the Grafana visual language the project already uses

## Design Principles

Adapted from the @graphty/compact-mantine package design philosophy:

1. **Compact by default**: Optimize for dense, information-rich display. Every pixel
   should earn its place. Standard component heights of 24px, tight spacing, small fonts.
2. **Progressive disclosure**: Most general info at top, details on demand via
   collapsible sections. Users should find what they need within 5 seconds.
3. **Group by meaning**: Related data in cards with subtle borders, not scattered
   across the page. One concept per card.
4. **Dark mode**: Reduce eye strain during long debug sessions. Match Grafana's
   dark theme and the compact-mantine dark color palette.
5. **No visual noise**: No borders on inputs (use semantic background colors instead),
   no heavy outlines on cards (use 1px subtle borders or box-shadow), no focus rings
   in this professional tool context.

## Design Tokens

Aligned with @graphty/compact-mantine's dark palette (`compactDarkColors`):

```
Background:       #0d1117   (page -- compactDarkColors[9])
Surface:          #161b22   (cards -- compactDarkColors[8])
Surface elevated: #1f2428   (card headers -- compactDarkColors[7])
Input bg:         #2a3035   (input fields -- compactDarkColors[6])
Border:           #30363d   (card edges, separators)
Text primary:     #c9d1d9   (main text)
Text secondary:   #8b949e   (labels, dimmed text)
Text muted:       #484f58   (disabled, hints)
Accent:           #58a6ff   (links, active states)
Success:          #3fb950   (HP healthy, good states)
Warning:          #d29922   (HP low, warnings)
Error:            #f85149   (HP critical, errors)
```

### Compact Sizing Constants (from compact-mantine)

```
Component height:  24px     (inputs, buttons, icons, indicators)
Font size primary: 11px     (body text, labels, values)
Font size small:   10px     (badges, secondary labels, section headers)
Font size tiny:     9px     (badges, timestamps)
Spacing xs:         4px     (gap between items in sections)
Spacing sm:         6px     (gap between related groups)
Spacing md:         8px     (card body padding, control padding)
Spacing lg:        12px     (gap between cards/sections)
Border radius:      4px     (cards, inputs)
Font family:       'JetBrains Mono', 'Fira Code', Consolas, 'Liberation Mono', monospace
```

### Key Compact Patterns

- **No borders on inputs**: Use semantic background color (`#2a3035`) instead of borders
- **No focus rings**: `focusRing: "never"` -- appropriate for dense professional UIs
- **Dropdowns**: No border, use `box-shadow` for elevation
- **Labels**: 11px, dimmed color, marginBottom 1px, lineHeight 1.2
- **Key-value pairs**: Label left (dimmed), value right (primary, fw=500), 4px vertical padding
  (modeled on compact-mantine's `StatRow` component)

## Layout Structure

Use `pn.template.FastListTemplate(theme="dark")` with sidebar for controls.

```
+---------------------------------------------------------------------+
| ROC Debug Dashboard                [Run: v] [Game: v] [Speed: v]    |
+---------------------------------------------------------------------+
| [<< < || > >> >>] Step [====o====================] 42/200  Game 1   |
+---------------------------------------------------------------------+
| HP 48/50  Score 16  Depth 1  Gold 16  Energy 9/10  Hunger: OK       |
+---------------------------------------------------------------------+
|                                                                      |
| v Perception                                                         |
| +-----------------------------+------------------------------------+ |
| | Screen                      | Features                           | |
| | [21x79 colored game grid]   | total_features: 146                | |
| |                             | unique_chars: 15    monsters: 3    | |
| |                             | items: 1   walls: 31  floor: 168   | |
| +-----------------------------+------------------------------------+ |
|                                                                      |
| v Attention                                                          |
| +-----------------------------+------------------------------------+ |
| | Saliency Heatmap            | Focus (21,9) score=0.93            | |
| | [blue-to-red heatmap        |   radius=4, points_in_focus=14     | |
| |  over game characters]      | Attenuation                       | |
| | [Blue] Low ---- High [Red]  |   penalty=0.21, history=5, max=0.5| |
| +-----------------------------+------------------------------------+ |
|                                                                      |
| > Object Resolution                       [matched obj-6, dist=1.06]|
| > Game State                                   [nodes=58, edges=92] |
| > Log Messages                              [Level: v] [4 messages] |
+---------------------------------------------------------------------+
```

### Collapsed Section Headers

Collapsed sections show a **summary in the header** so users can decide
whether to expand without clicking. This follows the "smell test" pattern --
the header alone tells you if something needs attention.

Examples:
- `> Object Resolution  [matched obj-6, dist=1.06, 2 candidates]`
- `> Game State  [HP 48/50, nodes=58, events=104]`
- `> Log Messages  [Level: DEBUG] [2 errors, 4 total]`

## Section Details

### Status Bar (always visible, not collapsible)

Compact row of key metrics using `pn.pane.HTML` styled as inline stat badges.
24px height, 11px font. Each metric is a compact `<span>` with label (dimmed)
and value (primary). HP uses conditional coloring:
- Green (#3fb950): HP > 50% of max
- Yellow (#d29922): HP 25-50% of max
- Red (#f85149): HP < 25% of max

Fields: HP, Score, Depth, Gold, Energy, Hunger, Step, Game.

### Perception Section (expanded by default)

`pn.Card(collapsible=True, collapsed=False)` with 8px body padding.

- **Screen**: `pn.pane.HTML` with `render_grid_pane()`. 9px monospace font,
  black background, inline-block. No changes needed.
- **Features**: `pn.pane.HTML` styled as a compact key-value list.
  Two-column layout within the features card using CSS grid or flexbox
  for denser packing: `total_features: 146  unique_chars: 15`

Side-by-side in a `pn.Row` with 4px gap.

### Attention Section (expanded by default)

`pn.Card(collapsible=True, collapsed=False)`.

- **Saliency Heatmap**: `pn.pane.HTML` using `render_grid_pane()`.
  The saliency map uses a blue-to-red heatmap encoding in the **background
  color** of each cell:
  - Blue (#0000bb) = lowest saliency (no visual features)
  - Cyan -> Green -> Yellow = increasing saliency
  - Red (#bb0000) = highest saliency (strongest attention)
  Game characters are overlaid in white/gray foreground text so the user sees
  both the game layout and where the agent is paying attention.

  Below the heatmap, a compact legend bar:
  ```html
  <div style="display:flex; align-items:center; gap:4px; font-size:10px; color:#8b949e;">
    <span>Low</span>
    <div style="height:8px; flex:1; background:linear-gradient(to right, #0000bb, #00bbbb, #00bb00, #bbbb00, #bb0000); border-radius:2px;"></div>
    <span>High</span>
  </div>
  ```

- **Focus & Attenuation**: Combined into one compact panel.
  Focus point coordinates + saliency score as a StatRow-style layout.
  Attenuation details below with 4px gap. Saves vertical space.

### Object Resolution Section (collapsed by default)

`pn.Card(collapsible=True, collapsed=True)`.
Header shows summary: outcome, object ID, distance.

- **Decision + Metrics**: Single compact key-value display.
  Algorithm, outcome, matched object, candidates, distance, threshold,
  features compared -- all as StatRow-style label: value pairs.

### Game State Section (collapsed by default)

`pn.Card(collapsible=True, collapsed=True)`.
Header shows summary: key vitals.

Three sub-groups using compact-mantine ControlGroup-style headers:

- **Vitals**: HP, HP max, score, depth, gold, energy, hunger as key-value pairs
- **Graph DB**: total_nodes, total_edges, object_nodes, frame_nodes
- **Events**: per-bus event counts (perception, attention, sequencer, etc.)

### Log Messages Section (collapsed by default)

`pn.Card(collapsible=True, collapsed=True)`.
Header shows: log level filter dropdown + message count + error count (if any).

- Log level filter as a compact 24px-height select widget in the card header
- Scrollable content area with `max-height: 200px; overflow-y: auto`
- Each line rendered as HTML with severity-based coloring:
  - DEBUG: `#484f58` (muted)
  - INFO: `#c9d1d9` (primary)
  - WARN: `#d29922` (warning yellow)
  - ERROR: `#f85149` (error red)
- 11px monospace font, 4px vertical gap between lines
- Format: `[LEVEL] message` with level in fixed-width span

## Global CSS

```css
/* Applied via pn.extension(raw_css=[...]) */

body {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'JetBrains Mono', 'Fira Code', Consolas, 'Liberation Mono', monospace;
    font-size: 11px;
    line-height: 1.4;
}

/* Card styling */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    margin-bottom: 4px;
}

.card-header {
    background: #1f2428;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 500;
    border-bottom: 1px solid #30363d;
    cursor: pointer;
}

.card-body {
    padding: 8px;
}

/* Stat row (label-value pair) */
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 11px;
}

.stat-label {
    color: #8b949e;
}

.stat-value {
    color: #c9d1d9;
    font-weight: 500;
}

/* Status bar */
.status-bar {
    display: flex;
    gap: 12px;
    padding: 4px 8px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    font-size: 11px;
}

.status-item .label {
    color: #8b949e;
    margin-right: 4px;
}

.status-item .value {
    color: #c9d1d9;
    font-weight: 500;
}

/* Log line colors */
.log-debug { color: #484f58; }
.log-info  { color: #c9d1d9; }
.log-warn  { color: #d29922; }
.log-error { color: #f85149; }

/* Scrollable log container */
.log-container {
    max-height: 200px;
    overflow-y: auto;
    font-size: 11px;
    line-height: 1.6;
}

/* Select widgets -- compact, no border */
select {
    height: 24px;
    font-size: 11px;
    background: #2a3035;
    color: #c9d1d9;
    border: none;
    border-radius: 4px;
    padding: 0 8px;
}
```

## Implementation Steps

### Step 1: Switch to FastListTemplate with dark theme
- Replace `pn.Column` return from `servable()` with `FastListTemplate`
- Move Run/Game/Speed selectors to template sidebar
- Apply design tokens via `raw_css` on `pn.extension()`
- Set `theme="dark"`, `accent_base_color="#58a6ff"`, `header_background="#161b22"`

### Step 2: Add compact status bar
- Render key metrics as inline HTML stat badges (not pn.indicators.Number -- too large)
- HP with conditional color class
- Always visible at top of main content area, 24px height

### Step 3: Convert sections to collapsible Cards
- Wrap each section in `pn.Card(collapsible=True)`
- Perception + Attention: `collapsed=False`
- Object Resolution, Game State, Logs: `collapsed=True`
- Style cards with compact padding (8px body, 4px 8px header)
- Add summary text to collapsed section headers

### Step 4: Apply compact spacing throughout
- All gaps: 4px between items, 4-8px between sections
- Card body padding: 8px
- Font size: 11px everywhere except grids (9px) and badges (9-10px)
- `sizing_mode="stretch_width"` on all cards
- Remove all unnecessary margin/padding

### Step 5: Render data as compact HTML
- Replace `pn.pane.Str` with `pn.pane.HTML` for all data panels
- Use StatRow-style HTML (flex row, label dimmed, value bold)
- Features in two-column grid layout
- Logs as colored HTML spans in scrollable container

### Step 6: Add saliency legend
- Static HTML gradient bar below saliency heatmap
- Labels: "Low" (left) / "High" (right)
- 8px height, linear-gradient from blue to red
- 10px font, secondary color

### Step 7: Polish and test
- Test on iPad-width viewport (1024px)
- Test collapse/expand behavior
- Verify all data still displays correctly
- Check that screen and saliency grids don't overflow
- Verify log level filter works with new HTML rendering
- Screenshot with Playwright and verify with nanobanana

## Completed Fixes (v3)

- [x] Saliency map: now uses real game chars with blue-to-red HSV heatmap bg
- [x] Log dropdown: wired up `_on_log_level_change` that re-renders with stored `_last_data`
- [x] Compact tables: `_render_kv_table_html` with `max-width:320px` replaces `_stat_row`
- [x] Events: rendered as compact key-value table instead of inline text
- [x] Features/Graph/Metrics: all use `_render_kv_table_html` for tight label-value display

## Future: Charts and Visualizations

Some data is better displayed as charts than tables. Panel supports HoloViews,
Bokeh, and Vega/Altair charts natively. These should be added iteratively.

### Where Charts Are Better Than Tables

| Data | Current | Better As | Why |
|------|---------|-----------|-----|
| Events per bus | KV table | Horizontal bar chart | Relative proportions visible at a glance |
| HP / Energy | Status badge | Mini gauge or progress bar | Shows ratio vs max visually |
| Saliency score | Text | Small radial gauge | 0-1 value maps naturally to a gauge |
| Object resolution candidates | Text | Small bar showing distance vs threshold | Visual "did it match?" indicator |
| Graph node/edge counts over steps | Single value | Sparkline (mini line chart) | Shows trend, not just current value |
| Feature counts | KV table | Stacked bar or treemap | Shows composition and relative proportions |

### Implementation Approach

Use `pn.pane.Vega` with inline Vega-Lite specs for lightweight charts.
Vega-Lite is already a Panel dependency and renders client-side.

Example: events bar chart
```python
spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "width": 280, "height": 120,
    "data": {"values": [{"bus": k, "count": v} for k, v in events.items()]},
    "mark": "bar",
    "encoding": {
        "y": {"field": "bus", "type": "nominal", "sort": "-x"},
        "x": {"field": "count", "type": "quantitative"},
        "color": {"value": "#58a6ff"}
    },
    "config": {"view": {"stroke": "transparent"}, "axis": {"labelColor": "#8b949e"}}
}
pn.pane.Vega(spec)
```

Example: HP progress bar (pure HTML, no chart library)
```html
<div style="display:flex;align-items:center;gap:4px;">
  <span style="color:#8b949e;width:20px;">HP</span>
  <div style="flex:1;height:8px;background:#30363d;border-radius:4px;">
    <div style="width:96%;height:100%;background:#3fb950;border-radius:4px;"></div>
  </div>
  <span style="color:#c9d1d9;font-weight:500;">48/50</span>
</div>
```

### Priority Order for Chart Additions

1. HP/Energy as progress bars in the status bar (HTML only, no deps)
2. Events per bus as horizontal bar chart (Vega-Lite)
3. Graph node/edge sparklines across steps (requires multi-step query, deferred)
4. Feature composition as stacked bar (Vega-Lite)
5. Resolution distance vs threshold as visual indicator (HTML)

### Grafana Patterns to Adopt

From the ROC Grafana dashboard:
- **Stat panels**: Single big number with optional sparkline (Games, Nodes/sec, Edges/sec)
- **Time series**: Line charts for CPU, Memory, Threads over time
- **Flame graph**: Pyroscope profiling data
- **Log table**: Structured log viewer with timestamp + message columns
- **Gauge**: Percentage displays (CPU %)

For the Panel dashboard, the most applicable patterns are:
- **Stat-style display** for single metrics (already done via status bar)
- **Log table** for structured log viewing (could replace plain text with Tabulator)
- **Bar charts** for event bus activity and feature composition
