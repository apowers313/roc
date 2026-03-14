# Panel Best Practices

This document defines how to build Panel UI in the ROC project. Follow these rules when
creating or modifying any code in `roc/reporting/`.

## Core Principle: Work With the Framework

Panel has a theming system, a component library, and a styling API. Use them. Do not
reimplement what Panel already provides. Raw HTML and global CSS overrides are a last resort,
not a default.

---

## 1. Theming

### Use Panel's Design System

Panel separates **Designs** (visual frameworks: Bootstrap, Material, Native) from **Themes**
(color palettes: default, dark). Set these globally:

```python
pn.extension(design="bootstrap", theme="dark")
```

Or on the template:

```python
template = pn.template.FastListTemplate(
    theme="dark",
    accent_base_color="#58a6ff",
    header_background="#0d1117",
    background_color="#0d1117",
)
```

### Use Design Variables (CSS Custom Properties)

Panel exposes design tokens as CSS custom properties. Override these instead of writing
raw CSS:

```python
# Global override
pn.extension(global_css=[":root { --design-primary-color: #58a6ff; }"])

# Per-component override
widget = pn.widgets.TextInput(
    stylesheets=[":host { --design-primary-color: #58a6ff; }"]
)
```

Key design variables:

| Variable | Purpose |
|---|---|
| `--design-primary-color` | Primary brand/accent color |
| `--design-primary-text-color` | Text on primary color |
| `--design-secondary-color` | Secondary brand color |
| `--design-background-color` | Background layer |
| `--design-surface-color` | Surface/card layer |

### Do NOT

- Write a global CSS block that overrides `.bk-root`, `.bk-input`, `.bk-menu`, `.bk-btn`,
  `.bk-slider-title`, `fast-card`, or other Panel/Bokeh internal class names. These are
  implementation details that change between Panel versions.
- Use `!important` to override Panel's own styles. If you need `!important`, you are
  fighting the framework.
- Duplicate Panel's dark theme colors in Python constants and then bake them into inline
  `style=` attributes. Use CSS custom properties instead.

---

## 2. Components: Use What Panel Provides

### Indicators (for metrics/KPIs)

Use `pn.indicators.*` instead of building HTML badge strings:

```python
# Good -- uses Panel's Number indicator
hp_display = pn.indicators.Number(
    name="HP",
    value=42,
    format="{value}/100",
    colors=[(25, "danger"), (50, "warning"), (100, "success")],
    font_size="11pt",
    title_size="9pt",
)

# Bad -- raw HTML string
html = f'<span style="color:green;font-size:11px">HP: 42/100</span>'
pn.pane.HTML(html)
```

Available indicators:
- **`Number`** -- KPI with conditional coloring via `colors=[(threshold, color), ...]`
- **`Trend`** -- KPI card with sparkline
- **`Gauge`** -- Speedometer arc with `bounds`, `colors`
- **`LinearGauge`** -- Compact horizontal/vertical scale
- **`Progress`** -- Progress bar
- **`BooleanStatus`** -- Colored circle for boolean states
- **`String`** -- Styled text display

All accept semantic color names: `"primary"`, `"secondary"`, `"success"`, `"info"`,
`"danger"`, `"warning"`, `"light"`, `"dark"`.

### Text Display

Use Panel text panes instead of `pn.pane.HTML` with raw HTML strings:

```python
# Good
pn.pane.Markdown("## Section Title", styles={"color": "#c9d1d9"})
pn.pane.Str("Step 42 | Game 1", styles={"font-size": "11px"})
pn.widgets.StaticText(value="No data available")

# Bad
pn.pane.HTML('<div style="font-size:11px;color:#8b949e;text-transform:uppercase">TITLE</div>')
```

### Tables

Use Tabulator with built-in themes and scoped stylesheets:

```python
table = pn.widgets.Tabulator(
    df,
    theme="midnight",            # Built-in dark theme
    theme_classes=["table-sm"],  # Compact rows
    show_index=False,
    stylesheets=[custom_css],    # Scoped overrides via :host
)
```

Built-in Tabulator themes: `"default"`, `"simple"`, `"midnight"`, `"modern"`,
`"bootstrap5"`, `"materialize"`, `"bulma"`, `"fast"`.

Use `stylesheets=` for scoped CSS when a built-in theme is close but needs tweaks.
Target `:host .tabulator*` selectors. This is the correct approach for Tabulator
customization.

### Cards and Layouts

Use Card parameters instead of inline styles:

```python
# Good -- uses Card's built-in parameters
card = pn.Card(
    content,
    title="Game State",
    collapsible=True,
    collapsed=False,
    header_background="#161b22",
    header_color="#c9d1d9",
    active_header_background="#1f2428",
    styles={"background": "#0d1117"},  # styles= is fine for the outer container
)

# Bad -- ignoring Card parameters and overriding with CSS
card = pn.Card(content, title="Game State")
# then in _GLOBAL_CSS: .card-header { background: #161b22 !important; }
```

### Charts (Vega-Lite)

Vega-Lite handles its own theming via the spec. Pass colors from design tokens into the
Vega spec directly:

```python
spec = {
    "config": {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "axis": {"labelColor": "#8b949e", "titleColor": "#c9d1d9"},
    },
    # ...
}
pn.pane.Vega(spec)
```

This is the correct approach -- Vega manages its own rendering and does not use Panel's
CSS.

---

## 3. CSS: When and How

### The `stylesheets` Parameter (Preferred)

Scoped to the component's shadow DOM. Styles do not leak:

```python
custom_css = """
:host .tabulator-header { display: none; }
:host .tabulator-row { background: transparent; }
"""
table = pn.widgets.Tabulator(df, stylesheets=[custom_css])
```

### The `styles` Parameter (Container Only)

Dictionary applied to the outer `<div>`. Cannot reach internal elements:

```python
pn.Column(content, styles={"gap": "4px", "padding": "8px"})
```

This is fine for layout properties (gap, padding, margin, background). Do not scatter
theme colors here -- prefer centralizing them in design variables or a shared stylesheet
constant.

### When Raw CSS Is Appropriate

- Tabulator internal elements (header, rows, cells) via `stylesheets=`
- Scrollbar styling (no Panel API for this)
- Truly custom layouts that Panel's FlexBox/GridSpec cannot express

### When Raw CSS Is NOT Appropriate

- Overriding Panel widget appearances globally (use design variables)
- Changing input/button/slider styles (use `design=` and theme)
- Changing Card/template chrome (use template and Card parameters)

---

## 4. When Raw HTML Is Acceptable

Some things genuinely require `pn.pane.HTML`:

- **Colored character grids** (screen_renderer.py) -- per-character `<span>` elements with
  individual fg/bg colors. No Panel widget exists for this.
- **Tabulator cell formatters** -- `HTMLTemplateFormatter` is the standard Tabulator API
  for custom cell rendering. This is expected.

Everything else should use Panel components. If you find yourself writing `f'<div style=...'`,
stop and check whether a Panel component exists first.

---

## 5. Architecture Patterns

### Viewer Pattern

Use `pn.viewable.Viewer` for components with reactive state:

```python
class MyComponent(pn.viewable.Viewer):
    data = param.Dict(default={})

    def __init__(self, **params):
        super().__init__(**params)
        self._content = pn.pane.Str("placeholder")

    @param.depends("data", watch=True)
    def _update(self):
        self._content.object = format_data(self.data)

    def __panel__(self):
        return pn.Column(self._content)
```

### Factory Functions

Use plain functions for stateless component creation:

```python
def compact_kv_table(data: dict, title: str = "") -> pn.widgets.Tabulator:
    df = pd.DataFrame(...)
    return pn.widgets.Tabulator(df, theme="midnight", ...)
```

### When to Use Which

- **Viewer**: component has internal state that changes after creation (GridViewer,
  ResolutionInspector, PanelDashboard)
- **Factory function**: component is created once from data and does not update itself
  (compact_kv_table, event_bar_chart)

### Reactivity

- Use `pn.bind(fn, widget)` or `@param.depends("param_name")` for reactive updates
- Reserve `watch=True` for side effects (updating another component's `.object`/`.value`)
- Use `throttled=True` on sliders to avoid excessive callback firings
- Update existing component `.object`/`.value` properties instead of recreating Panel
  objects (avoids flicker)

### Performance

- Use `defer_load=True` or `loading_indicator=True` for slow-loading sections
- Use `@pn.cache` for expensive computations
- Move repeated inline CSS to external stylesheet files or shared constants for production

---

## 6. Checklist for New Panel Code

Before submitting Panel UI code, verify:

- [ ] No `!important` in CSS (if present, you are fighting the framework)
- [ ] No overrides of Panel/Bokeh internal classes (`.bk-root`, `.bk-input`, `.bk-btn`, etc.)
- [ ] No raw HTML for text display, badges, or KPIs (use Markdown, Str, indicators)
- [ ] Tabulator uses a built-in theme + scoped `stylesheets=` for tweaks
- [ ] Colors come from design variables or shared constants, not inline `style=` attributes
- [ ] Viewer pattern used only for stateful components; factory functions for stateless ones
- [ ] `watch=True` used only for side effects, not for returning display objects
- [ ] Sliders use `throttled=True`
