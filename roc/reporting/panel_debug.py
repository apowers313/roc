"""Panel debug dashboard for stepping through ROC game data.

Displays game screen, saliency heatmap, and all agent data panels organized
by pipeline stage in collapsible cards. Data loaded from Parquet via RunStore/DuckDB.

Uses Panel's built-in components and theming -- no global CSS overrides.
Components are created once and updated in place to avoid flicker during playback.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import param
import panel as pn
from bokeh.models import FactorRange, Range1d
from bokeh.models.widgets.tables import HTMLTemplateFormatter
from bokeh.plotting import figure
from panel.viewable import Viewer

from roc.reporting.components.grid_viewer import GridViewer
from roc.reporting.components.resolution_inspector import ResolutionInspector
from roc.reporting.components.theme import COMPACT_CELL_CSS
from roc.reporting.run_store import RunStore, StepData
from roc.reporting.step_buffer import StepBuffer

#: Log severity levels ordered for filtering.
_LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]

#: Severity level numbers for log filtering.
_LEVEL_NUMBERS: dict[str, int] = {
    "DEBUG": 5,
    "INFO": 9,
    "WARN": 13,
    "WARNING": 13,
    "ERROR": 17,
}

_EMPTY_KV_DF = pd.DataFrame({"key": ["--"], "value": ["No data"]})
_EMPTY_LOG_DF = pd.DataFrame({"level": ["--"], "message": ["No log data"]})


def _format_timestamp(ts: int | None) -> str:
    """Format a nanosecond OTel timestamp as local time."""
    if ts is None:
        return "N/A"
    dt = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _parse_features(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse raw feature data into a clean dict for display."""
    merged: dict[str, Any] = {}
    for feat in features:
        if "raw" in feat:
            for line in str(feat["raw"]).strip().split("\n"):
                line = line.strip()
                if ": " in line:
                    k, v = line.split(": ", 1)
                    merged[k.strip()] = v.strip()
        else:
            merged.update(feat)
    return merged


def _parse_events(event_summary: list[dict[str, Any]]) -> dict[str, int]:
    """Parse event summary list into a single dict for charting."""
    merged: dict[str, int] = {}
    for item in event_summary:
        for k, v in item.items():
            if k not in ("step", "game_number") and isinstance(v, (int, float)):
                merged[k] = int(v)
    return merged


def _dict_to_df(data: dict[str, Any] | None) -> pd.DataFrame:
    """Convert a dict to a key-value DataFrame for Tabulator display."""
    if not data:
        return _EMPTY_KV_DF

    rows = []
    for k, v in data.items():
        if k in ("raw", "step", "game_number"):
            continue
        val_str = str(v)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        rows.append({"key": str(k), "value": val_str})

    return pd.DataFrame(rows) if rows else _EMPTY_KV_DF


def _make_kv_tabulator() -> pn.widgets.Tabulator:
    """Create a reusable compact key-value Tabulator."""
    return pn.widgets.Tabulator(
        _EMPTY_KV_DF,
        theme="fast",
        show_index=False,
        header_filters=False,
        configuration={"headerVisible": False},
        stylesheets=[COMPACT_CELL_CSS],
        sizing_mode="stretch_width",
        disabled=True,
        pagination=None,
    )


def _filter_logs(
    logs: list[dict[str, Any]] | None,
    min_level: str = "DEBUG",
) -> pd.DataFrame:
    """Filter logs by severity and return a DataFrame."""
    if not logs:
        return _EMPTY_LOG_DF

    min_num = _LEVEL_NUMBERS.get(min_level, 0)
    rows = []
    for log in logs:
        severity = log.get("severity_number")
        if severity is not None and severity < min_num:
            continue
        level = log.get("severity_text", "?")
        body = log.get("body", "")
        rows.append({"level": level, "message": str(body)})

    return pd.DataFrame(rows) if rows else _EMPTY_LOG_DF


def _make_log_tabulator() -> pn.widgets.Tabulator:
    """Create a reusable log Tabulator with severity formatting."""
    level_fmt = HTMLTemplateFormatter(
        template=(
            '<% if (value === "ERROR") { %>'
            '<span style="color:var(--panel-danger-color, #e74c3c);font-weight:600">'
            "[<%= value %>]</span>"
            '<% } else if (value === "WARN" || value === "WARNING") { %>'
            '<span style="color:var(--panel-warning-color, #f39c12);font-weight:600">'
            "[<%= value %>]</span>"
            "<% } else { %>"
            "<span>[<%= value %>]</span>"
            "<% } %>"
        )
    )
    msg_fmt = HTMLTemplateFormatter(
        template=(
            '<% if (level === "ERROR") { %>'
            '<span style="color:var(--panel-danger-color, #e74c3c)"><%= value %></span>'
            '<% } else if (level === "WARN" || level === "WARNING") { %>'
            '<span style="color:var(--panel-warning-color, #f39c12)"><%= value %></span>'
            "<% } else { %>"
            "<span><%= value %></span>"
            "<% } %>"
        )
    )

    return pn.widgets.Tabulator(
        _EMPTY_LOG_DF,
        theme="fast",
        show_index=False,
        header_filters=False,
        configuration={"headerVisible": False},
        stylesheets=[COMPACT_CELL_CSS],
        sizing_mode="stretch_width",
        height=200,
        disabled=True,
        pagination=None,
        formatters={"level": level_fmt, "message": msg_fmt},
    )


def _make_event_chart() -> tuple[pn.pane.Bokeh, Any]:
    """Create a reusable Bokeh horizontal bar chart.

    Returns the pn.pane.Bokeh wrapper and the Bokeh figure for data updates.
    """
    # Start with a placeholder -- will be replaced on first data update
    placeholder_names = ["--"]
    fig = figure(
        y_range=FactorRange(*placeholder_names),
        height=200,
        width=300,
        toolbar_location=None,
    )
    renderer = fig.hbar(y=placeholder_names, right=[0], height=0.6)
    fig.xgrid.visible = False  # type: ignore[attr-defined]
    fig.x_range = Range1d(start=0, end=1)
    fig.min_border_left = 80
    fig.min_border_right = 10
    fig.min_border_top = 5
    fig.min_border_bottom = 5

    pane = pn.pane.Bokeh(fig, sizing_mode="stretch_width", height=200, theme="dark_minimal")
    return pane, (fig, renderer)


_HELP_TEXT = """\
| Key | Action |
|-----|--------|
| **Right** | Next step |
| **Left** | Previous step |
| **Home** | First step |
| **End** | Last step |
| **Space** | Play / pause |
| **+** / **=** | Faster playback |
| **-** | Slower playback |
| **G** | Next game |
| **B** | Toggle bookmark |
| **N** / **]** | Next bookmark |
| **P** / **[** | Previous bookmark |
| **?** / **H** | Toggle this help |
"""

#: Keys handled by the keyboard listener (prevents default browser behavior).
_HANDLED_KEYS = [
    "ArrowRight",
    "ArrowLeft",
    "Home",
    "End",
    " ",
    "+",
    "-",
    "=",
    "b",
    "g",
    "n",
    "p",
    "[",
    "]",
    "?",
    "h",
]


class KeyboardShortcuts(pn.custom.ReactComponent):
    """Invisible component that captures keyboard shortcuts via React useEffect.

    Renders nothing.  Attaches a ``keydown`` listener on ``window`` and sends
    matched key names to Python via ``model.send_msg()``.  Uses React lifecycle
    for proper cleanup.
    """

    shortcuts = param.List(
        default=[{"name": k, "key": k} for k in _HANDLED_KEYS],
        doc="List of shortcut dicts with 'name' and 'key' fields.",
    )

    _esm = """
    function hashShortcut({ key, altKey, ctrlKey, metaKey, shiftKey }) {
      return `${key}.${+!!altKey}.${+!!ctrlKey}.${+!!metaKey}.${+!!shiftKey}`;
    }

    export function render({ model }) {
      const [shortcuts] = model.useState("shortcuts");
      const keyedShortcuts = {};
      for (const shortcut of shortcuts) {
        keyedShortcuts[hashShortcut(shortcut)] = shortcut.name;
      }

      function onKeyDown(e) {
        const tag = e.target.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        const name = keyedShortcuts[hashShortcut(e)];
        if (name) {
          e.preventDefault();
          e.stopPropagation();
          model.send_msg(name);
        }
      }

      React.useEffect(() => {
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
      });

      return <></>;
    }
    """


class BookmarkManager:
    """Manages step bookmarks with JSON persistence in the run directory."""

    def __init__(self, run_dir: Path) -> None:
        self._file = run_dir / "bookmarks.json"
        self._bookmarks: list[dict[str, Any]] = []
        self.load()

    def toggle(self, step: int, game: int, annotation: str = "") -> bool:
        """Toggle bookmark at *step*. Returns ``True`` if added, ``False`` if removed."""
        for i, bm in enumerate(self._bookmarks):
            if bm["step"] == step:
                self._bookmarks.pop(i)
                self.save()
                return False
        self._bookmarks.append({"step": step, "game": game, "annotation": annotation})
        self._bookmarks.sort(key=lambda b: b["step"])
        self.save()
        return True

    def next_bookmark(self, current_step: int) -> int | None:
        """Return the next bookmarked step after *current_step*, or ``None``."""
        for bm in self._bookmarks:
            if bm["step"] > current_step:
                return int(bm["step"])
        return None

    def prev_bookmark(self, current_step: int) -> int | None:
        """Return the previous bookmarked step before *current_step*, or ``None``."""
        for bm in reversed(self._bookmarks):
            if bm["step"] < current_step:
                return int(bm["step"])
        return None

    def is_bookmarked(self, step: int) -> bool:
        """Return whether *step* is bookmarked."""
        return any(bm["step"] == step for bm in self._bookmarks)

    def save(self) -> None:
        """Persist bookmarks to ``bookmarks.json``."""
        self._file.write_text(json.dumps(self._bookmarks, indent=2))

    def load(self) -> None:
        """Load bookmarks from ``bookmarks.json`` if it exists."""
        if self._file.exists():
            try:
                self._bookmarks = json.loads(self._file.read_text())
            except (json.JSONDecodeError, TypeError):
                self._bookmarks = []

    def as_list(self) -> list[dict[str, Any]]:
        """Return a copy of the bookmark list."""
        return list(self._bookmarks)

    def as_df(self) -> pd.DataFrame:
        """Return bookmarks as a DataFrame for Tabulator display."""
        if not self._bookmarks:
            return pd.DataFrame(
                {
                    "step": pd.Series(dtype=int),
                    "game": pd.Series(dtype=int),
                    "annotation": pd.Series(dtype=str),
                }
            )
        return pd.DataFrame(self._bookmarks)


class PanelDashboard(Viewer):
    """Debug dashboard for ROC game state inspection.

    Components are created once in __init__ and updated in place via
    .value/.object property changes to avoid flicker during playback.
    """

    SPEEDS: list[tuple[str, int]] = [
        ("0.5x", 2000),
        ("1x", 1000),
        ("2x", 500),
        ("5x", 200),
        ("10x", 100),
        ("20x", 50),
    ]
    _DEFAULT_SPEED = "5x"

    step = param.Integer(default=1, doc="Current step number")
    run_name = param.String(default="", doc="Active run directory name")
    game = param.String(default="0", doc="Active game number (as string)")
    speed = param.String(default="5x", doc="Playback speed label")
    log_level = param.String(default="DEBUG", doc="Minimum log severity to display")

    def __init__(
        self,
        store: RunStore,
        data_dir: Path | None = None,
        step_buffer: StepBuffer | None = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._store = store
        self._data_dir = data_dir or store.run_dir.parent
        self._step_buffer = step_buffer
        self._speed_to_interval = dict(self.SPEEDS)
        self._updating_game = False
        self._last_data: StepData | None = None
        self._last_seen_step: int = 0
        self._live_mode = step_buffer is not None
        self._user_paused = False

        min_step, max_step = store.step_range()
        self.run_name = store.run_dir.name

        # -- Widgets --
        self._step_widget = pn.widgets.Player(
            name="Step",
            start=max(min_step, 1),
            end=max(max_step, 1),
            value=max(min_step, 1),
            step=1,
            interval=self._speed_to_interval[self._DEFAULT_SPEED],
            width=400,
            show_loop_controls=False,
            loop_policy="once",
            visible_buttons=["first", "previous", "pause", "play", "next", "last"],
        )

        self._speed_selector = pn.widgets.Select(
            name="Speed",
            options=[label for label, _ in self.SPEEDS],
            value=self._DEFAULT_SPEED,
            width=80,
        )

        runs = RunStore.list_runs(self._data_dir)
        current_run = store.run_dir.name
        self._run_selector = pn.widgets.AutocompleteInput(
            name="Run",
            options=runs if runs else [current_run],
            value=current_run,
            restrict=True,
            min_characters=0,
        )

        games_df = store.list_games()
        game_options: list[int] = list(games_df["game_number"]) if len(games_df) > 0 else [1]
        self._game_selector = pn.widgets.Select(
            name="Game",
            options=[str(g) for g in game_options],
            value=str(game_options[0]) if game_options else "1",
        )

        self._log_level_selector = pn.widgets.RadioButtonGroup(
            name="Log Level",
            options=_LOG_LEVELS,
            value="DEBUG",
            button_type="default",
            button_style="outline",
        )

        # -- Live mode --
        self._live_badge = pn.pane.Markdown(
            "**LIVE**",
            visible=False,
            styles={"color": "var(--panel-success-color, #2ecc71)", "font-size": "14px"},
            sizing_mode="fixed",
            width=60,
        )
        self._new_data_badge = pn.pane.Markdown(
            "*New data available*",
            visible=False,
            styles={"color": "var(--panel-warning-color, #f39c12)", "font-size": "12px"},
            sizing_mode="fixed",
            width=150,
        )

        # -- Bookmarks, keyboard, help --
        self._bookmarks = BookmarkManager(store.run_dir)

        self._kb_shortcuts = KeyboardShortcuts()
        self._kb_shortcuts.on_msg(self._on_keypress_msg)

        self._help_pane = pn.pane.Markdown(_HELP_TEXT, visible=False, sizing_mode="stretch_width")

        self._bookmark_table = pn.widgets.Tabulator(
            self._bookmarks.as_df(),
            theme="fast",
            show_index=False,
            header_filters=False,
            stylesheets=[COMPACT_CELL_CSS],
            sizing_mode="stretch_width",
            disabled=True,
            pagination=None,
            height=150,
        )
        self._bookmark_table.on_click(self._on_bookmark_click)

        # -- Persistent component instances (created once, updated in place) --
        self._screen_viewer = GridViewer()
        self._saliency_viewer = GridViewer()
        self._info_pane = pn.pane.Str("", sizing_mode="stretch_width")

        # Status indicators (created once, updated via .value)
        self._hp_indicator = pn.indicators.Number(
            name="HP",
            value=0,
            format="{value}",
            font_size="14pt",
            title_size="8pt",
        )
        self._score_indicator = pn.indicators.Number(
            name="Score",
            value=0,
            font_size="14pt",
            title_size="8pt",
        )
        self._depth_indicator = pn.indicators.Number(
            name="Depth",
            value=0,
            font_size="14pt",
            title_size="8pt",
        )
        self._gold_indicator = pn.indicators.Number(
            name="Gold",
            value=0,
            font_size="14pt",
            title_size="8pt",
        )
        self._energy_indicator = pn.indicators.Number(
            name="Energy",
            value=0,
            format="{value}",
            font_size="14pt",
            title_size="8pt",
        )
        self._hunger_indicator = pn.indicators.Number(
            name="Hunger",
            value=0,
            font_size="14pt",
            title_size="8pt",
        )

        # KV tabulators (created once, updated via .value = new_df)
        self._metrics_table = _make_kv_tabulator()
        self._graph_table = _make_kv_tabulator()
        self._features_table = _make_kv_tabulator()
        self._attenuation_table = _make_kv_tabulator()
        self._resolution_inspector = ResolutionInspector()

        # Text panes for raw/variable data (updated via .object)
        self._object_pane = pn.pane.Str("No object data", sizing_mode="stretch_width")
        self._focus_pane = pn.pane.Str("No focus data", sizing_mode="stretch_width")

        # Event bar chart (persistent Bokeh figure, updated via data source)
        self._events_pane, self._events_chart_state = _make_event_chart()

        # Log table (created once, updated via .value = new_df)
        self._log_table = _make_log_tabulator()

        # -- Wire widgets to params --
        self._step_widget.param.watch(self._handle_step_widget, "value")
        self._step_widget.param.watch(self._handle_direction_widget, "direction")
        self._speed_selector.param.watch(self._handle_speed_widget, "value")
        self._run_selector.param.watch(self._handle_run_widget, "value")
        self._game_selector.param.watch(self._handle_game_widget, "value")
        self._log_level_selector.param.watch(self._handle_log_level_widget, "value")

        self.param.watch(self._handle_step, ["step"])
        self.param.watch(self._handle_speed, ["speed"])
        self.param.watch(self._handle_run, ["run_name"])
        self.param.watch(self._handle_game, ["game"])
        self.param.watch(self._handle_log_level, ["log_level"])

        # In live mode, show "playing" state so the pause button works.
        # Disable auto-advance timer -- live data comes from push callbacks.
        if step_buffer is not None:
            self._step_widget.interval = 2**31 - 1
            self._step_widget.direction = 1

        # Initial render
        self.step = max(min_step, 1)
        self._on_step_change(self.step)

    # -- Widget -> param sync --

    def _handle_step_widget(self, event: param.parameterized.Event) -> None:
        self.step = event.new

    def _handle_direction_widget(self, event: param.parameterized.Event) -> None:
        self._user_paused = event.new == 0
        if self._user_paused and self._step_buffer is not None:
            # Restore normal playback interval for review mode
            label = self._speed_selector.value
            self._step_widget.interval = self._speed_to_interval.get(
                label, self._speed_to_interval[self._DEFAULT_SPEED]
            )

    def _handle_speed_widget(self, event: param.parameterized.Event) -> None:
        self.speed = event.new

    def _handle_run_widget(self, event: param.parameterized.Event) -> None:
        self.run_name = event.new

    def _handle_game_widget(self, event: param.parameterized.Event) -> None:
        if event.new and event.new.isdigit():
            self.game = event.new

    def _handle_log_level_widget(self, event: param.parameterized.Event) -> None:
        self.log_level = event.new

    # -- Param -> business logic --

    def _handle_step(self, event: param.parameterized.Event) -> None:
        self._on_step_change(event.new)

    def _handle_speed(self, event: param.parameterized.Event) -> None:
        self._on_speed_change(event)

    def _handle_run(self, event: param.parameterized.Event) -> None:
        self._on_run_change(event.new)

    def _handle_game(self, event: param.parameterized.Event) -> None:
        if str(event.new).isdigit():
            self._on_game_change(int(event.new))

    def _handle_log_level(self, event: param.parameterized.Event) -> None:
        self._on_log_level_change()

    # -- Live mode --

    def _is_following(self) -> bool:
        """Return True if the player is at or near the latest step."""
        return bool(self._step_widget.value >= self._step_widget.end)

    def _on_new_data(self) -> None:
        """Handle a push notification from the step buffer.

        Called on the Tornado thread via ``call_soon_threadsafe``.
        All widget mutations happen here -- single-threaded, no guards needed.
        """
        if self._step_buffer is None:
            return
        latest = self._step_buffer.get_latest()
        if latest is None or latest.step <= self._last_seen_step:
            return
        # On the first notification after init from Parquet data, jump to live.
        # The dashboard may init with end >> value (historical data in RunStore),
        # making _is_following() False even though the user just opened the page.
        first_notification = self._last_seen_step == 0 and self._step_widget.value <= 1
        self._last_seen_step = latest.step

        was_following = (self._is_following() and not self._user_paused) or first_notification

        # Always update slider end and game options (lightweight)
        if latest.step > self._step_widget.end:
            self._step_widget.end = latest.step

        # Update game selector options from buffer
        for g in self._step_buffer.game_numbers:
            game_str = str(g)
            if game_str not in (self._game_selector.options or []):
                opts = list(self._game_selector.options or [])
                opts.append(game_str)
                self._game_selector.options = opts

        if was_following:
            # Following: advance slider, auto-switch game, let watcher chain update data
            self._updating_game = True
            try:
                self._game_selector.value = str(latest.game_number)
            finally:
                self._updating_game = False
            old_value = self._step_widget.value
            self._step_widget.end = latest.step
            self._step_widget.value = latest.step
            # Keep auto-advance timer disabled while following live
            if self._step_widget.interval < 2**31 - 1:
                self._step_widget.interval = 2**31 - 1
            # If step didn't change (e.g. first push at startup), the watcher
            # won't fire so we must call _on_step_change directly.
            if old_value == latest.step:
                self._on_step_change(latest.step)

    # -- Business logic --

    def _on_step_change(self, step: int) -> None:
        """Fetch data for step and update all components in place."""
        # In live mode, try the step buffer first for recent steps
        data: StepData | None = None
        if self._step_buffer is not None:
            data = self._step_buffer.get_step(step)
        if data is None:
            data = self._store.get_step_data(step)

        self._apply_step_data(data)

    def _apply_step_data(self, data: StepData) -> None:
        """Update all dashboard widgets from a StepData instance."""
        # Update live/new-data badges
        following = self._is_following()
        self._live_badge.visible = following
        # Show badge when not at latest, clear when following or advancing
        if not following and self._step_widget.value < self._step_widget.end:
            self._new_data_badge.visible = True
        elif following or (self._last_data is not None and data.step > self._last_data.step):
            self._new_data_badge.visible = False
        self._last_data = data

        # Grid viewers (reactive via param)
        self._screen_viewer.grid_data = data.screen
        self._saliency_viewer.grid_data = data.saliency

        # Info line (with bookmark indicator)
        mark = " [*]" if self._bookmarks.is_bookmarked(data.step) else ""
        self._info_pane.object = (
            f"Step {data.step} | Game {data.game_number} | "
            f"{_format_timestamp(data.timestamp)}{mark}"
        )

        # Status indicators (update .value in place)
        metrics = data.game_metrics
        if metrics:
            hp = metrics.get("hp", 0)
            hp_max = metrics.get("hp_max", 1)
            self._hp_indicator.name = "HP"
            self._hp_indicator.value = int(hp) if isinstance(hp, (int, float)) else 0
            self._hp_indicator.format = f"{{value}}/{hp_max}"
            self._hp_indicator.colors = [
                (int(hp_max * 0.25), "danger"),
                (int(hp_max * 0.5), "warning"),
                (int(hp_max) + 1, "success"),
            ]
            self._score_indicator.name = "Score"
            self._score_indicator.value = int(metrics.get("score", 0))
            self._depth_indicator.name = "Depth"
            self._depth_indicator.value = int(metrics.get("depth", 0))
            self._gold_indicator.name = "Gold"
            self._gold_indicator.value = int(metrics.get("gold", 0))
            energy = metrics.get("energy", 0)
            energy_max = metrics.get("energy_max", 1)
            self._energy_indicator.name = "Energy"
            self._energy_indicator.value = int(energy) if isinstance(energy, (int, float)) else 0
            self._energy_indicator.format = f"{{value}}/{energy_max}"
            hunger = metrics.get("hunger", 0)
            self._hunger_indicator.value = int(hunger) if isinstance(hunger, (int, float)) else 0
        # else: keep previous indicator values (metrics arrive slightly after screen)

        # KV tables (update DataFrame in place)
        if data.game_metrics:
            self._metrics_table.value = _dict_to_df(data.game_metrics)
        self._graph_table.value = _dict_to_df(data.graph_summary)
        self._features_table.value = _dict_to_df(
            _parse_features(data.features) if data.features else None
        )
        self._attenuation_table.value = _dict_to_df(data.attenuation)
        self._resolution_inspector.decision = data.resolution_metrics

        # Object info (text, update .object)
        if data.object_info:
            parts = []
            for item in data.object_info:
                if isinstance(item.get("raw"), str):
                    parts.append(str(item["raw"]).strip())
                else:
                    for k, v in item.items():
                        if k not in ("step", "game_number"):
                            parts.append(f"{k}: {v}")
            self._object_pane.object = "\n".join(parts) if parts else "No object data"
        else:
            self._object_pane.object = "No object data"

        # Focus points (text, update .object)
        if data.focus_points:
            parts = []
            for item in data.focus_points:
                if isinstance(item.get("raw"), str):
                    parts.append(str(item["raw"]).strip())
                else:
                    for k, v in item.items():
                        if k not in ("step", "game_number"):
                            parts.append(f"{k}: {v}")
            self._focus_pane.object = "\n".join(parts) if parts else "No focus data"
        else:
            self._focus_pane.object = "No focus data"

        # Event bar chart (update Bokeh data source in place)
        if data.event_summary:
            event_data = _parse_events(data.event_summary)
            if event_data:
                self._update_event_chart(event_data)

        # Log table (update DataFrame in place)
        self._log_table.value = _filter_logs(data.logs, self._log_level_selector.value)

    def _update_event_chart(self, event_data: dict[str, int]) -> None:
        """Update the Bokeh bar chart data source in place."""
        fig, renderer = self._events_chart_state
        names = list(event_data.keys())
        counts = list(event_data.values())

        if not names or not counts:
            return

        # Update the data source
        renderer.data_source.data = {"y": names, "right": counts}

        # Update the y_range factors
        fig.y_range.factors = names

        # Update x_range in place (replacing the object breaks Bokeh sync)
        max_count = max(counts) if counts else 1
        fig.x_range.start = 0
        fig.x_range.end = max_count * 1.1

        # Update chart height
        chart_height = max(len(names) * 25, 80)
        fig.height = chart_height
        self._events_pane.height = chart_height

    def _on_log_level_change(self) -> None:
        """Re-filter logs with new severity level."""
        if self._last_data is not None:
            self._log_table.value = _filter_logs(
                self._last_data.logs,
                self._log_level_selector.value,
            )

    def _on_run_change(self, run_name: str) -> None:
        """Switch to a different run directory."""
        run_dir = self._data_dir / run_name
        self._store = RunStore(run_dir)
        self._bookmarks = BookmarkManager(run_dir)
        self._update_bookmark_list()
        # Reset so new run's steps aren't skipped
        self._last_seen_step = 0
        games_df = self._store.list_games()
        game_options = [str(g) for g in games_df["game_number"]] if len(games_df) > 0 else ["0"]
        self._game_selector.options = game_options
        self._game_selector.value = game_options[0] if game_options else "0"
        min_step, max_step = self._store.step_range()
        self._step_widget.start = max(min_step, 1)
        self._step_widget.end = max(max_step, 1)
        self._step_widget.value = max(min_step, 1)

    def _on_speed_change(self, event: object) -> None:
        """Update player interval from speed label."""
        label = getattr(event, "new", self._DEFAULT_SPEED)
        # Don't override disabled timer when following live
        if self._step_buffer is not None and not self._user_paused and self._is_following():
            return
        self._step_widget.interval = self._speed_to_interval.get(
            label, self._speed_to_interval[self._DEFAULT_SPEED]
        )

    def _on_game_change(self, game_number: int) -> None:
        """Jump to the first step of a game."""
        if self._updating_game:
            return
        try:
            min_step, max_step = self._store.step_range(game_number=game_number)
            if min_step > 0:
                if min_step < self._step_widget.start:
                    self._step_widget.start = min_step
                if max_step > self._step_widget.end:
                    self._step_widget.end = max_step
                self._step_widget.value = min_step
        except Exception:
            pass

    # -- Keyboard shortcuts --

    def _on_keypress_msg(self, event: Any) -> None:
        """Handle a keyboard shortcut message from ``KeyboardShortcuts``."""
        key = event.data if hasattr(event, "data") else str(event)
        if key:
            self._dispatch_key(key)

    def _on_keypress(self, event: param.parameterized.Event) -> None:
        """Dispatch a keyboard event from a param watcher (legacy/test path)."""
        key_str = event.new
        if not key_str or key_str == "init":
            return
        key = key_str.split(":")[0]
        self._dispatch_key(key)

    def _dispatch_key(self, key: str) -> None:
        """Route a key name to the appropriate handler."""
        if key == "ArrowRight":
            self._step_widget.value = min(self._step_widget.value + 1, self._step_widget.end)
        elif key == "ArrowLeft":
            self._step_widget.value = max(self._step_widget.value - 1, self._step_widget.start)
        elif key == "Home":
            self._step_widget.value = self._step_widget.start
            if self._step_buffer is not None and self._step_widget.direction != 0:
                self._step_widget.direction = 0
        elif key == "End":
            self._step_widget.value = self._step_widget.end
            self._user_paused = False
            if self._step_buffer is not None:
                self._step_widget.interval = 2**31 - 1
                if self._step_widget.direction != 1:
                    self._step_widget.direction = 1
        elif key == " ":
            self._toggle_play()
        elif key in ("+", "="):
            self._increase_speed()
        elif key == "-":
            self._decrease_speed()
        elif key == "g":
            self._cycle_game()
        elif key == "b":
            self._toggle_bookmark()
        elif key in ("n", "]"):
            self._jump_next_bookmark()
        elif key in ("p", "["):
            self._jump_prev_bookmark()
        elif key in ("?", "h"):
            self._toggle_help()

    def _increase_speed(self) -> None:
        """Switch to the next faster speed option."""
        options = [label for label, _ in self.SPEEDS]
        try:
            idx = options.index(self._speed_selector.value)
        except ValueError:
            return
        if idx < len(options) - 1:
            self._speed_selector.value = options[idx + 1]

    def _decrease_speed(self) -> None:
        """Switch to the next slower speed option."""
        options = [label for label, _ in self.SPEEDS]
        try:
            idx = options.index(self._speed_selector.value)
        except ValueError:
            return
        if idx > 0:
            self._speed_selector.value = options[idx - 1]

    def _toggle_play(self) -> None:
        """Toggle play/pause via the Player widget's direction param."""
        if self._step_widget.direction == 0:
            self._step_widget.direction = 1
        else:
            self._step_widget.direction = 0

    def _cycle_game(self) -> None:
        """Jump to the next game (wraps around)."""
        options = self._game_selector.options or []
        if not options:
            return
        try:
            idx = options.index(self._game_selector.value)
        except ValueError:
            return
        next_idx = (idx + 1) % len(options)
        self._game_selector.value = options[next_idx]

    # -- Bookmarks --

    def _toggle_bookmark(self, step: int | None = None, annotation: str = "") -> None:
        """Add or remove a bookmark at *step* (defaults to current step)."""
        if step is None:
            step = self._step_widget.value
        # Use the actual game_number from step data, not the selector value
        game = self._last_data.game_number if self._last_data is not None else int(self.game)
        self._bookmarks.toggle(step, game, annotation)
        self._update_bookmark_list()
        self._update_bookmark_indicator()

    def _jump_next_bookmark(self) -> None:
        """Jump to the next bookmarked step."""
        target = self._bookmarks.next_bookmark(self._step_widget.value)
        if target is not None:
            self._step_widget.value = target

    def _jump_prev_bookmark(self) -> None:
        """Jump to the previous bookmarked step."""
        target = self._bookmarks.prev_bookmark(self._step_widget.value)
        if target is not None:
            self._step_widget.value = target

    def _update_bookmark_indicator(self) -> None:
        """Update the info pane to reflect bookmark state for current step."""
        data = self._last_data
        if data is None:
            return
        mark = " [*]" if self._bookmarks.is_bookmarked(data.step) else ""
        self._info_pane.object = (
            f"Step {data.step} | Game {data.game_number} | "
            f"{_format_timestamp(data.timestamp)}{mark}"
        )

    def _update_bookmark_list(self) -> None:
        """Refresh the bookmark Tabulator from the bookmark manager."""
        self._bookmark_table.value = self._bookmarks.as_df()

    def _on_bookmark_click(self, event: Any) -> None:
        """Jump to a bookmarked step when clicked in the bookmark table."""
        row_idx = event.row
        df = self._bookmark_table.value
        if row_idx < len(df):
            step = int(df.iloc[row_idx]["step"])
            self._step_widget.value = step

    # -- Help overlay --

    def _toggle_help(self) -> None:
        """Show or hide the keyboard shortcut help overlay."""
        self._help_pane.visible = not self._help_pane.visible

    # -- Layout --

    def __panel__(self) -> pn.Column:
        """Build the dashboard layout."""
        controls = pn.Row(
            self._run_selector,
            self._game_selector,
            self._speed_selector,
            sizing_mode="stretch_width",
        )

        status_row = pn.Row(
            self._hp_indicator,
            self._score_indicator,
            self._depth_indicator,
            self._gold_indicator,
            self._energy_indicator,
            self._hunger_indicator,
            sizing_mode="stretch_width",
        )

        transport_bar = pn.Column(
            controls,
            self._step_widget,
            pn.Row(
                self._info_pane,
                self._live_badge,
                self._new_data_badge,
                status_row,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

        game_state_card = pn.Card(
            pn.Row(
                self._screen_viewer,
                pn.Column(
                    pn.pane.Markdown("**Vitals**"),
                    self._metrics_table,
                    pn.pane.Markdown("**Graph DB**"),
                    self._graph_table,
                    pn.pane.Markdown("**Events**"),
                    self._events_pane,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            title="Game State",
            collapsible=True,
            collapsed=False,
            sizing_mode="stretch_width",
        )

        perception_card = pn.Card(
            self._features_table,
            title="Perception",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )

        attention_card = pn.Card(
            pn.Row(
                self._saliency_viewer,
                pn.Column(
                    pn.pane.Markdown("**Focus Points**"),
                    self._focus_pane,
                    pn.pane.Markdown("**Attenuation**"),
                    self._attenuation_table,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            title="Attention",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )

        object_card = pn.Card(
            pn.Row(
                pn.Column(
                    pn.pane.Markdown("**Object Info**"),
                    self._object_pane,
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    pn.pane.Markdown("**Resolution**"),
                    self._resolution_inspector,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            title="Object Resolution",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )

        log_card = pn.Card(
            pn.Row(self._log_level_selector),
            self._log_table,
            title="Log Messages",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )

        bookmarks_card = pn.Card(
            self._bookmark_table,
            title="Bookmarks",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
        )

        return pn.Column(
            self._kb_shortcuts,
            self._help_pane,
            transport_bar,
            game_state_card,
            perception_card,
            attention_card,
            object_card,
            log_card,
            bookmarks_card,
            sizing_mode="stretch_width",
        )


def main() -> None:
    """Entry point for the panel-debug command."""
    pn.extension("tabulator")

    parser = argparse.ArgumentParser(description="ROC Panel Debug Dashboard")
    parser.add_argument("--run", type=str, default=None, help="Run directory name")
    parser.add_argument("--data-dir", type=str, default=None, help="Parent data directory")
    parser.add_argument("--port", type=int, default=9042, help="Port (default: 9042)")
    args = parser.parse_args()

    from roc.config import Config

    Config.init()
    cfg = Config.get()

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg.data_dir)
    if args.run:
        run_dir = data_dir / args.run
    else:
        runs = RunStore.list_runs(data_dir)
        if not runs:
            print(f"No runs found in {data_dir}")
            return
        run_dir = data_dir / runs[-1]

    store = RunStore(run_dir)
    dashboard = PanelDashboard(store, data_dir=data_dir)

    template = pn.template.FastListTemplate(
        title="ROC Debug Dashboard",
        theme="dark",
        main=[dashboard],
    )

    serve_kwargs: dict[str, object] = {
        "port": args.port,
        "address": "0.0.0.0",
        "websocket_origin": "*",
        "show": False,
        "title": "ROC Debug Dashboard",
    }
    if cfg.ssl_certfile and cfg.ssl_keyfile:
        serve_kwargs["ssl_certfile"] = cfg.ssl_certfile
        serve_kwargs["ssl_keyfile"] = cfg.ssl_keyfile

    pn.serve(template, **serve_kwargs)


if __name__ == "__main__":
    main()
