"""Compact dark-mode Panel debug dashboard for stepping through ROC game data.

Displays game screen, saliency heatmap, and all agent data panels organized
by pipeline stage in collapsible cards. Data loaded from Parquet via RunStore/DuckDB.

Uses Panel's Viewer pattern with param.Parameter fields for reactive state.
Delegates rendering to reusable components in ``roc.reporting.components``.
"""

from __future__ import annotations

import argparse
import html as html_mod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import param
import panel as pn
from panel.viewable import Viewer

from roc.reporting.components.charts import event_bar_chart
from roc.reporting.components.grid_viewer import GridViewer
from roc.reporting.components.resolution_inspector import ResolutionInspector
from roc.reporting.components.status_bar import compact_status_bar
from roc.reporting.components.tables import compact_kv_table, compact_log_table
from roc.reporting.components.tokens import (
    ACCENT,
    BG,
    BORDER,
    FONT,
    INPUT_BG,
    SURFACE,
    SURFACE_EL,
    TEXT,
    TEXT_DIM,
    TEXT_MUTED,
    no_data_html,
    title_html,
)
from roc.reporting.run_store import RunStore, StepData

# Re-export tokens for test access
_BG = BG
_SURFACE = SURFACE
_SURFACE_EL = SURFACE_EL
_INPUT_BG = INPUT_BG
_BORDER = BORDER
_TEXT = TEXT
_TEXT_DIM = TEXT_DIM
_TEXT_MUTED = TEXT_MUTED
_ACCENT = ACCENT
_SUCCESS = "#3fb950"
_ERROR = "#f85149"
_WARNING = "#d29922"
_FONT = FONT

#: Log severity levels ordered by number for filtering.
_LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR"]

# -- Global CSS for dark compact theme --
_GLOBAL_CSS = f"""
body {{
    background: {BG} !important;
    color: {TEXT};
    font-family: {FONT};
    font-size: 11px;
    line-height: 1.4;
}}
.bk-root, .bk, .pn-container {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: {FONT} !important;
}}
.card {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    margin-bottom: 4px !important;
}}
.card-header {{
    background: {SURFACE_EL} !important;
    padding: 2px 8px !important;
    font-size: 11px !important;
    border-bottom: 1px solid {BORDER} !important;
    min-height: 0 !important;
}}
.card-body, .card .bk-panel-models-layout-Card {{
    padding: 8px !important;
}}
select, .bk-input, input[type="text"] {{
    height: 24px !important;
    font-size: 11px !important;
    background: {INPUT_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    padding: 0 8px !important;
    font-family: {FONT} !important;
}}
.bk-menu {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    font-family: {FONT} !important;
    font-size: 11px !important;
    max-height: 200px !important;
    overflow-y: auto !important;
}}
.bk-menu > div {{
    padding: 4px 8px !important;
    color: {TEXT} !important;
    cursor: pointer !important;
}}
.bk-menu > div.bk-active, .bk-menu > div:hover {{
    background: {INPUT_BG} !important;
    color: {TEXT} !important;
}}
.bk-btn {{
    background: {INPUT_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
}}
.bk-slider-title {{
    color: {TEXT_DIM} !important;
    font-size: 11px !important;
}}
label, .bk-label {{
    color: {TEXT_DIM} !important;
    font-size: 11px !important;
    font-family: {FONT} !important;
}}
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
fast-card.pn-wrapper {{
    overflow-x: clip !important;
}}
/* RadioButtonGroup compact styling */
.bk-btn-group .bk-btn {{
    font-size: 10px !important;
    padding: 2px 8px !important;
    min-height: 0 !important;
    line-height: 1.2 !important;
}}
.bk-btn-group .bk-btn.bk-active {{
    background: {ACCENT} !important;
    color: {BG} !important;
    border-color: {ACCENT} !important;
}}
"""


def _format_timestamp(ts: int | None) -> str:
    """Format a nanosecond OTel timestamp as local time."""
    if ts is None:
        return "N/A"
    dt = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _preformatted_html(text: str) -> str:
    """Render preformatted text as a compact HTML block."""
    lines = text.strip().split("\n")
    parts = [
        f'<div style="white-space:pre;color:{TEXT};font-size:10px;'
        f'font-family:{FONT};line-height:1.3;">{html_mod.escape(line)}</div>'
        for line in lines
    ]
    return "".join(parts)


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


def _parse_events(event_summary: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse event summary list into a single dict for charting."""
    merged: dict[str, Any] = {}
    for item in event_summary:
        for k, v in item.items():
            if k not in ("step", "game_number"):
                merged[k] = v
    return merged


# -- Dashboard Viewer --


class PanelDashboard(Viewer):
    """Compact dark-mode debug dashboard for ROC game state inspection.

    Uses Panel's Viewer pattern with param.Parameter fields for reactive state.
    Delegates rendering to components in ``roc.reporting.components``.
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

    def __init__(self, store: RunStore, data_dir: Path | None = None, **params: Any) -> None:
        super().__init__(**params)
        self._store = store
        self._data_dir = data_dir or store.run_dir.parent
        self._speed_to_interval = dict(self.SPEEDS)
        self._updating_game = False
        self._last_data: StepData | None = None

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

        self._speed_selector = pn.widgets.AutocompleteInput(
            name="Speed",
            options=[label for label, _ in self.SPEEDS],
            value=self._DEFAULT_SPEED,
            restrict=True,
            min_characters=0,
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
        game_options: list[int] = list(games_df["game_number"]) if len(games_df) > 0 else [0]
        self._game_selector = pn.widgets.AutocompleteInput(
            name="Game",
            options=[str(g) for g in game_options],
            value=str(game_options[0]) if game_options else "0",
            restrict=True,
            min_characters=0,
        )

        self._log_level_selector = pn.widgets.RadioButtonGroup(
            name="Log Level",
            options=_LOG_LEVELS,
            value="DEBUG",
            button_type="default",
            button_style="outline",
        )

        # -- Component instances --
        self._screen_viewer = GridViewer(title="Screen")
        self._saliency_viewer = GridViewer(title="Saliency")
        self._resolution_inspector = ResolutionInspector()

        # -- Simple panes (for data that updates via .object or component replacement) --
        self._status_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._info_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._features_container = pn.Column(sizing_mode="stretch_width")
        self._metrics_container = pn.Column(sizing_mode="stretch_width")
        self._graph_container = pn.Column(sizing_mode="stretch_width")
        self._events_container = pn.Column(sizing_mode="stretch_width")
        self._logs_container = pn.Column(sizing_mode="stretch_width")
        self._object_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._focus_pane = pn.pane.HTML("", sizing_mode="stretch_width")
        self._attenuation_container = pn.Column(sizing_mode="stretch_width")

        # -- Wire widgets to params --
        self._step_widget.param.watch(self._handle_step_widget, "value")
        self._speed_selector.param.watch(self._handle_speed_widget, "value")
        self._run_selector.param.watch(self._handle_run_widget, "value")
        self._game_selector.param.watch(self._handle_game_widget, "value")
        self._log_level_selector.param.watch(self._handle_log_level_widget, "value")

        self.param.watch(self._handle_step, ["step"])
        self.param.watch(self._handle_speed, ["speed"])
        self.param.watch(self._handle_run, ["run_name"])
        self.param.watch(self._handle_game, ["game"])
        self.param.watch(self._handle_log_level, ["log_level"])

        # Initial render
        self.step = max(min_step, 1)
        self._on_step_change(self.step)

    # -- Widget -> param sync --

    def _handle_step_widget(self, event: param.parameterized.Event) -> None:
        self.step = event.new

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

    # -- Business logic --

    def _on_step_change(self, step: int) -> None:
        """Fetch data for step and update all components."""
        data = self._store.get_step_data(step)
        self._last_data = data

        # Grid viewers (reactive via param)
        self._screen_viewer.grid_data = data.screen
        self._saliency_viewer.grid_data = data.saliency

        # Resolution inspector (reactive via param)
        self._resolution_inspector.decision = data.resolution_metrics

        # Status bar (factory returns new pane)
        self._status_pane.object = compact_status_bar(
            data.game_metrics, step=data.step, game_number=data.game_number
        ).object

        # Info line
        self._info_pane.object = (
            f'<div style="font-size:11px;color:{TEXT_DIM};padding:2px 0;'
            f'font-family:{FONT};">'
            f"Step {data.step} | Game {data.game_number} | "
            f"{_format_timestamp(data.timestamp)}</div>"
        )

        # Tables and charts (replace container contents)
        self._update_container(
            self._features_container,
            compact_kv_table(_parse_features(data.features), "feature")
            if data.features
            else pn.pane.HTML(no_data_html("feature")),
        )
        self._update_container(
            self._metrics_container,
            compact_kv_table(data.game_metrics, "game metrics"),
        )
        self._update_container(
            self._graph_container,
            compact_kv_table(data.graph_summary, "graph DB"),
        )
        self._update_container(
            self._events_container,
            event_bar_chart(_parse_events(data.event_summary))
            if data.event_summary
            else pn.pane.HTML(no_data_html("event")),
        )
        self._update_container(
            self._attenuation_container,
            compact_kv_table(data.attenuation, "attenuation"),
        )

        # Object info (still HTML -- mixed raw/dict format)
        self._object_pane.object = self._render_object(data)

        # Focus points (still HTML -- DataFrame string format)
        self._focus_pane.object = self._render_focus(data)

        # Logs
        self._update_container(
            self._logs_container,
            compact_log_table(data.logs, self._log_level_selector.value),
        )

        # Sync game selector
        game_str = str(data.game_number)
        if game_str in (self._game_selector.options or []):
            self._updating_game = True
            self._game_selector.value = game_str
            self._updating_game = False

    def _on_log_level_change(self) -> None:
        if self._last_data is not None:
            self._update_container(
                self._logs_container,
                compact_log_table(self._last_data.logs, self._log_level_selector.value),
            )

    def _on_run_change(self, run_name: str) -> None:
        run_dir = self._data_dir / run_name
        self._store = RunStore(run_dir)
        games_df = self._store.list_games()
        game_options = [str(g) for g in games_df["game_number"]] if len(games_df) > 0 else ["0"]
        self._game_selector.options = game_options
        self._game_selector.value = game_options[0] if game_options else "0"
        min_step, max_step = self._store.step_range()
        self._step_widget.start = max(min_step, 1)
        self._step_widget.end = max(max_step, 1)
        self._step_widget.value = max(min_step, 1)

    def _on_speed_change(self, event: object) -> None:
        label = getattr(event, "new", self._DEFAULT_SPEED)
        self._step_widget.interval = self._speed_to_interval.get(
            label, self._speed_to_interval[self._DEFAULT_SPEED]
        )

    def _on_game_change(self, game_number: int) -> None:
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

    # -- Render helpers for data that still needs custom HTML --

    @staticmethod
    def _render_object(data: StepData) -> str:
        if data.object_info is None:
            return no_data_html("object resolution")
        parts: list[str] = []
        for item in data.object_info:
            if "raw" in item:
                parts.append(_preformatted_html(str(item["raw"])))
            else:
                for k, v in item.items():
                    if k not in ("step", "game_number"):
                        val = str(v)[:77] + "..." if len(str(v)) > 80 else str(v)
                        parts.append(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:1px 0;font-size:11px;font-family:{FONT};max-width:320px;">'
                            f'<span style="color:{TEXT_DIM};margin-right:8px;">'
                            f"{html_mod.escape(str(k))}</span>"
                            f'<span style="color:{TEXT};font-weight:500;">'
                            f"{html_mod.escape(val)}</span></div>"
                        )
        return (
            f'<div style="font-family:{FONT};font-size:11px;color:{TEXT};'
            f'line-height:1.5;">{"".join(parts)}</div>'
            if parts
            else no_data_html("object resolution")
        )

    @staticmethod
    def _render_focus(data: StepData) -> str:
        if data.focus_points is None:
            return no_data_html("focus point")
        parts: list[str] = []
        for item in data.focus_points:
            if "raw" in item:
                parts.append(_preformatted_html(str(item["raw"])))
            else:
                for k, v in item.items():
                    if k not in ("step", "game_number"):
                        val = str(v)[:77] + "..." if len(str(v)) > 80 else str(v)
                        parts.append(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:1px 0;font-size:11px;font-family:{FONT};max-width:320px;">'
                            f'<span style="color:{TEXT_DIM};margin-right:8px;">'
                            f"{html_mod.escape(str(k))}</span>"
                            f'<span style="color:{TEXT};font-weight:500;">'
                            f"{html_mod.escape(val)}</span></div>"
                        )
        return (
            f'<div style="font-family:{FONT};font-size:11px;color:{TEXT};'
            f'line-height:1.5;">{"".join(parts)}</div>'
            if parts
            else no_data_html("focus point")
        )

    @staticmethod
    def _update_container(container: pn.Column, component: Any) -> None:
        """Replace a container's contents with a new component."""
        container.clear()
        container.append(component)

    # -- Layout --

    @staticmethod
    def _card(
        title: str,
        *objects: Any,
        collapsed: bool = False,
    ) -> pn.Card:
        return pn.Card(
            *objects,
            title=title,
            collapsible=True,
            collapsed=collapsed,
            sizing_mode="stretch_width",
            header_background=SURFACE_EL,
            header_color=TEXT,
            active_header_background=SURFACE_EL,
            styles={
                "background": SURFACE,
                "border": f"1px solid {BORDER}",
                "border-radius": "4px",
                "margin-bottom": "2px",
                "padding": "4px",
            },
        )

    def __panel__(self) -> pn.Column:
        controls = pn.Row(
            self._run_selector,
            self._game_selector,
            self._speed_selector,
            sizing_mode="stretch_width",
            styles={"gap": "8px"},
        )

        def _titled(title: str, content: Any) -> pn.Column:
            return pn.Column(
                pn.pane.HTML(title_html(title)),
                content,
                sizing_mode="stretch_width",
                styles={"gap": "0px"},
            )

        # Game State (expanded)
        game_state_card = self._card(
            "Game State",
            pn.Row(
                self._screen_viewer,
                pn.Column(
                    _titled("Vitals", self._metrics_container),
                    _titled("Graph DB", self._graph_container),
                    _titled("Events", self._events_container),
                    sizing_mode="stretch_width",
                    styles={"gap": "2px"},
                ),
                sizing_mode="stretch_width",
                styles={"gap": "4px"},
            ),
            collapsed=False,
        )

        # Perception (collapsed)
        perception_card = self._card(
            "Perception",
            _titled("Features", self._features_container),
            collapsed=True,
        )

        # Attention (collapsed)
        attention_card = self._card(
            "Attention",
            pn.Row(
                self._saliency_viewer,
                pn.Column(
                    _titled("Focus Points", self._focus_pane),
                    _titled("Attenuation", self._attenuation_container),
                    sizing_mode="stretch_width",
                    styles={"gap": "2px"},
                ),
                sizing_mode="stretch_width",
                styles={"gap": "4px"},
            ),
            collapsed=True,
        )

        # Object Resolution (collapsed)
        object_card = self._card(
            "Object Resolution",
            pn.Row(
                _titled("Decision", self._object_pane),
                _titled("Resolution", self._resolution_inspector),
                sizing_mode="stretch_width",
                styles={"gap": "4px"},
            ),
            collapsed=True,
        )

        # Log Messages (collapsed)
        log_card = self._card(
            "Log Messages",
            pn.Row(self._log_level_selector),
            self._logs_container,
            collapsed=True,
        )

        # Sticky transport bar
        transport_bar = pn.Column(
            controls,
            self._step_widget,
            pn.Row(
                self._info_pane,
                self._status_pane,
                sizing_mode="stretch_width",
                styles={"gap": "4px"},
            ),
            sizing_mode="stretch_width",
            styles={
                "position": "sticky",
                "top": "0",
                "z-index": "100",
                "background": BG,
                "padding": "4px",
                "border-bottom": f"1px solid {BORDER}",
            },
        )

        return pn.Column(
            transport_bar,
            game_state_card,
            perception_card,
            attention_card,
            object_card,
            log_card,
            sizing_mode="stretch_width",
            styles={"gap": "2px", "background": BG, "padding": "0 4px 40px 4px"},
        )


def main() -> None:
    """Entry point for the panel-debug command."""
    pn.extension(raw_css=[_GLOBAL_CSS])

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
        accent_base_color=ACCENT,
        header_background=SURFACE_EL,
        background_color=BG,
        main=[dashboard],
        raw_css=[_GLOBAL_CSS],
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
