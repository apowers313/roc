# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/panel_debug.py.

Tests the v2 dashboard built on Panel's components and theming.
No tests depend on specific CSS classes or HTML content.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import panel as pn
import pytest
from helpers.otel import make_log_record
from opentelemetry._logs import SeverityNumber

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.parquet_exporter import ParquetExporter
from roc.reporting.run_store import RunStore


@pytest.fixture()
def populated_run_dir(tmp_path: Path) -> Path:
    """Create a run directory with known test data using ParquetExporter."""
    store = DuckLakeStore(tmp_path)
    exporter = ParquetExporter(store=store, background=False)

    exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
    for i in range(10):
        screen_body = json.dumps(
            {
                "chars": [[65 + i, 66 + i]],
                "fg": [["ffffff", "ff0000"]],
                "bg": [["000000", "000000"]],
            }
        )
        exporter.export([make_log_record(event_name="roc.screen", body=screen_body)])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.saliency",
                    body=json.dumps(
                        {
                            "chars": [[46, 64]],
                            "fg": [["ffffff", "ffff55"]],
                            "bg": [["0000ff", "ff0000"]],
                        }
                    ),
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.features",
                    body=f"\t\twall: {3 + i}\n\t\tfloor: {7 + i}\n",
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.object",
                    body=f"Object: wall at ({i}, {i + 1})",
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.attention.focus_points",
                    body=f"[({i}, {i + 1}, 0.{i + 1})]",
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.saliency_attenuation",
                    body=json.dumps(
                        {
                            "penalty_applied": True,
                            "max_penalty": 0.5,
                            "history_size": i + 1,
                        }
                    ),
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.game_metrics",
                    body=json.dumps(
                        {
                            "hp": 16 - i,
                            "hp_max": 16,
                            "score": i * 10,
                            "depth": 1,
                            "gold": i * 5,
                            "energy": 4,
                            "energy_max": 4,
                            "hunger": 0,
                        }
                    ),
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.graphdb.summary",
                    body=json.dumps({"node_count": 10 + i, "edge_count": 20 + i}),
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.event.summary",
                    body=json.dumps({"bus": "perception", "event_count": 5 + i}),
                )
            ]
        )
        exporter.export(
            [
                make_log_record(
                    event_name="roc.resolution.decision",
                    body=json.dumps(
                        {
                            "outcome": "match" if i > 0 else "new_object",
                            "tick": i + 1,
                            "x": i,
                            "y": i + 1,
                            "features": ["ShapeNode(.)", "ColorNode(GREY)"],
                            "num_candidates": i + 1,
                            "posteriors": [["obj1", 0.8], ["new", 0.2]],
                        }
                    ),
                )
            ]
        )
        # Log messages with different severity levels
        exporter.export([make_log_record(body=f"debug message {i}", severity=SeverityNumber.DEBUG)])
        exporter.export([make_log_record(body=f"info message {i}", severity=SeverityNumber.INFO)])
        exporter.export([make_log_record(body=f"error message {i}", severity=SeverityNumber.ERROR)])

    exporter.export([make_log_record(event_name="roc.game_start", body="game 2")])
    for i in range(5):
        screen_body = json.dumps(
            {
                "chars": [[75 + i, 76 + i]],
                "fg": [["ffffff", "00ff00"]],
                "bg": [["000000", "000000"]],
            }
        )
        exporter.export([make_log_record(event_name="roc.screen", body=screen_body)])

    exporter.shutdown()
    return tmp_path


@pytest.fixture()
def mock_store(populated_run_dir: Path) -> RunStore:
    """Create a RunStore backed by test data."""
    return RunStore(populated_run_dir)


class TestDashboardCreation:
    def test_dashboard_creates_without_error(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard is not None

    def test_dashboard_has_player(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        players = list(layout.select(pn.widgets.Player))
        assert len(players) >= 1

    def test_dashboard_has_run_selector(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        autos = list(layout.select(pn.widgets.AutocompleteInput))
        assert len(autos) >= 1

    def test_dashboard_has_game_selector(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        selects = list(layout.select(pn.widgets.Select))
        assert len(selects) >= 1


class TestScreenRendering:
    def test_screen_viewer_has_data(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._screen_viewer.grid_data is not None
        html = dashboard._screen_viewer._render()
        assert "<span" in html

    def test_screen_viewer_handles_none(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._screen_viewer.grid_data = None
        html = dashboard._screen_viewer._render()
        assert html == ""


class TestStepChange:
    def test_step_change_triggers_query(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        with patch.object(mock_store, "get_step_data", wraps=mock_store.get_step_data) as spy:
            dashboard._on_step_change(5)
            spy.assert_called_once_with(5)

    def test_game_selector_jumps_to_game(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._step_widget.start == 1
        assert dashboard._step_widget.end == 15

        dashboard._on_game_change(2)
        assert dashboard._step_widget.value == 11

        dashboard._on_game_change(1)
        assert dashboard._step_widget.value == 1

    def test_player_has_no_loop_controls(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._step_widget.show_loop_controls is False

    def test_player_shows_standard_transport_buttons(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        expected = ["first", "previous", "pause", "play", "next", "last"]
        assert dashboard._step_widget.visible_buttons == expected

    def test_speed_selector_changes_interval(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._step_widget.interval == 200

        event = MagicMock()
        event.new = "1x"
        dashboard._on_speed_change(event)
        assert dashboard._step_widget.interval == 1000

        event.new = "20x"
        dashboard._on_speed_change(event)
        assert dashboard._step_widget.interval == 50

    def test_speed_selector_has_labeled_options(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        options = dashboard._speed_selector.options
        assert "1x" in options
        assert "5x" in options
        assert "10x" in options

    def test_info_line_updates(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        info = dashboard._info_pane.object
        assert "Step 1" in info
        assert "Game 1" in info


class TestStatusIndicators:
    def test_hp_indicator_updates(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        assert dashboard._hp_indicator.value == 16
        assert "16" in dashboard._hp_indicator.format

    def test_score_indicator_updates(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(5)
        assert dashboard._score_indicator.value == 40

    def test_hunger_indicator_present(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        assert dashboard._hunger_indicator.name == "Hunger"

    def test_keeps_previous_metrics_when_missing(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        # Step 5 has metrics (HP=12, score=40)
        dashboard._on_step_change(5)
        assert dashboard._hp_indicator.name == "HP"
        prev_hp = dashboard._hp_indicator.value
        # Step 12 is in game 2 which has no metrics -- indicators keep previous values
        dashboard._on_step_change(12)
        assert dashboard._hp_indicator.name == "HP"
        assert dashboard._hp_indicator.value == prev_hp


class TestDataPanels:
    def test_metrics_table_populated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        df = dashboard._metrics_table.value
        assert len(df) > 0
        assert "hp" in df["key"].values

    def test_graph_table_populated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        df = dashboard._graph_table.value
        assert len(df) > 0
        assert "node_count" in df["key"].values

    def test_features_table_populated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        df = dashboard._features_table.value
        assert len(df) > 0

    def test_attenuation_table_populated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        df = dashboard._attenuation_table.value
        assert len(df) > 0

    def test_object_info_updated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        text = dashboard._object_pane.object
        assert isinstance(text, str)
        assert len(text) > 0

    def test_focus_pane_updated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        text = dashboard._focus_pane.object
        assert isinstance(text, str)
        assert len(text) > 0

    def test_resolution_inspector_updated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        assert dashboard._resolution_inspector.decision is not None
        assert dashboard._resolution_inspector._outcome_pane.object == "NEW OBJECT"

        dashboard._on_step_change(5)
        assert dashboard._resolution_inspector._outcome_pane.object == "MATCHED"

    def test_log_table_populated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        df = dashboard._log_table.value
        assert len(df) > 0

    def test_object_info_dict_format(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        # Mock get_step_data to return dict-format object_info (not raw)
        from roc.reporting.run_store import StepData

        mock_data = StepData(
            step=1,
            game_number=1,
            object_info=[{"type": "wall", "x": 5, "y": 3}],
            focus_points=[{"x": 5, "y": 3, "strength": 0.9}],
        )
        with patch.object(mock_store, "get_step_data", return_value=mock_data):
            dashboard._on_step_change(1)
        assert "type: wall" in dashboard._object_pane.object
        assert "x: 5" in dashboard._focus_pane.object

    def test_log_level_filter(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        all_count = len(dashboard._log_table.value)

        dashboard._log_level_selector.value = "ERROR"
        dashboard._on_log_level_change()
        error_count = len(dashboard._log_table.value)
        assert error_count < all_count


class TestHelperFunctions:
    def test_format_timestamp_none(self):
        from roc.reporting.panel_debug import _format_timestamp

        assert _format_timestamp(None) == "N/A"

    def test_format_timestamp_valid(self):
        from roc.reporting.panel_debug import _format_timestamp

        # 1 second in nanoseconds -- just check it produces a time-like string
        result = _format_timestamp(1_000_000_000)
        assert ":" in result
        assert "." in result

    def test_parse_features_raw(self):
        from roc.reporting.panel_debug import _parse_features

        features = [{"raw": "\t\twall: 3\n\t\tfloor: 7\n"}]
        result = _parse_features(features)
        assert result["wall"] == "3"
        assert result["floor"] == "7"

    def test_parse_features_dict(self):
        from roc.reporting.panel_debug import _parse_features

        features = [{"wall": 3, "floor": 7}]
        result = _parse_features(features)
        assert result["wall"] == 3

    def test_parse_events(self):
        from roc.reporting.panel_debug import _parse_events

        events = [{"perception": 100, "step": 1, "game_number": 1}]
        result = _parse_events(events)
        assert result == {"perception": 100}
        assert "step" not in result

    def test_dict_to_df_none(self):
        from roc.reporting.panel_debug import _dict_to_df, _EMPTY_KV_DF

        result = _dict_to_df(None)
        assert result is _EMPTY_KV_DF

    def test_dict_to_df_skips_raw(self):
        from roc.reporting.panel_debug import _dict_to_df

        result = _dict_to_df({"raw": "blob", "hp": 16})
        assert len(result) == 1
        assert "hp" in result["key"].values

    def test_dict_to_df_truncates(self):
        from roc.reporting.panel_debug import _dict_to_df

        result = _dict_to_df({"key": "x" * 200})
        val = result["value"].iloc[0]
        assert val.endswith("...")
        assert len(val) == 80

    def test_filter_logs_none(self):
        from roc.reporting.panel_debug import _filter_logs, _EMPTY_LOG_DF

        result = _filter_logs(None)
        assert result is _EMPTY_LOG_DF

    def test_filter_logs_by_level(self):
        from roc.reporting.panel_debug import _filter_logs

        logs = [
            {"severity_text": "DEBUG", "severity_number": 5, "body": "debug"},
            {"severity_text": "ERROR", "severity_number": 17, "body": "error"},
        ]
        result = _filter_logs(logs, min_level="ERROR")
        assert len(result) == 1
        assert result["level"].iloc[0] == "ERROR"


class TestWidgetHandlers:
    def test_handle_speed_widget(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        event = MagicMock()
        event.new = "10x"
        dashboard._handle_speed_widget(event)
        assert dashboard.speed == "10x"

    def test_handle_run_widget(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        event = MagicMock()
        event.new = "some-run"
        dashboard._handle_run_widget(event)
        assert dashboard.run_name == "some-run"

    def test_handle_game_widget_valid(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        event = MagicMock()
        event.new = "1"
        dashboard._handle_game_widget(event)
        assert dashboard.game == "1"

    def test_handle_game_widget_non_digit(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        original = dashboard.game
        event = MagicMock()
        event.new = "abc"
        dashboard._handle_game_widget(event)
        assert dashboard.game == original

    def test_handle_speed_param(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        event = MagicMock()
        event.new = "1x"
        dashboard._handle_speed(event)
        assert dashboard._step_widget.interval == 1000

    def test_handle_run_param(self, populated_run_dir: Path, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store, data_dir=populated_run_dir.parent)
        event = MagicMock()
        event.new = populated_run_dir.name
        dashboard._handle_run(event)

    def test_handle_game_param(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        event = MagicMock()
        event.new = "1"
        dashboard._handle_game(event)


class TestRunChange:
    def test_run_change_reloads_store(self, populated_run_dir: Path, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store, data_dir=populated_run_dir.parent)
        old_store = dashboard._store
        dashboard._on_run_change(populated_run_dir.name)
        # Store should be replaced
        assert dashboard._store is not old_store


class TestSaliencyPanel:
    def test_saliency_viewer_has_data(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._saliency_viewer.grid_data is not None
        html = dashboard._saliency_viewer._render()
        assert "<span" in html


class TestFullLayout:
    def test_layout_has_all_cards(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = list(layout.select(pn.Card))
        card_titles = [c.title for c in cards if hasattr(c, "title")]

        assert "Game State" in card_titles
        assert "Perception" in card_titles
        assert "Attention" in card_titles
        assert "Object Resolution" in card_titles
        assert "Log Messages" in card_titles

    def test_game_state_expanded(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        assert not cards["Game State"].collapsed

    def test_detail_sections_collapsed(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        assert cards["Perception"].collapsed
        assert cards["Attention"].collapsed
        assert cards["Object Resolution"].collapsed
        assert cards["Log Messages"].collapsed

    def test_player_widget_present(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        players = list(layout.select(pn.widgets.Player))
        assert len(players) >= 1

    def test_screen_in_game_state_card(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        game_html = list(cards["Game State"].select(pn.pane.HTML))
        assert len(game_html) >= 1

    def test_no_global_css(self, mock_store: RunStore):
        """Verify we are not injecting global CSS overrides."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        # No styles dict with background colors on the top-level Column
        styles = layout.styles or {}
        assert "background" not in styles

    def test_uses_panel_indicators(self, mock_store: RunStore):
        """Verify status bar uses pn.indicators.Number, not HTML."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        indicators = list(layout.select(pn.indicators.Number))
        assert len(indicators) >= 5  # HP, Score, Depth, Gold, Energy, Hunger

    def test_layout_has_bookmarks_card(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = list(layout.select(pn.Card))
        card_titles = [c.title for c in cards if hasattr(c, "title")]
        assert "Bookmarks" in card_titles

    def test_bookmarks_card_collapsed(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        assert cards["Bookmarks"].collapsed


class TestBookmarkManager:
    def test_toggle_adds_bookmark(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        added = mgr.toggle(step=5, game=1)
        assert added is True
        assert mgr.is_bookmarked(5)

    def test_toggle_removes_existing(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        mgr.toggle(step=5, game=1)
        removed = mgr.toggle(step=5, game=1)
        assert removed is False
        assert not mgr.is_bookmarked(5)

    def test_persists_to_json(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        mgr.toggle(step=42, game=1, annotation="spike here")
        bookmarks_file = populated_run_dir / "bookmarks.json"
        assert bookmarks_file.exists()
        data = json.loads(bookmarks_file.read_text())
        assert len(data) == 1
        assert data[0]["step"] == 42
        assert data[0]["annotation"] == "spike here"

    def test_loads_from_json(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr1 = BookmarkManager(populated_run_dir)
        mgr1.toggle(step=3, game=1)
        mgr1.toggle(step=7, game=1)

        mgr2 = BookmarkManager(populated_run_dir)
        assert mgr2.is_bookmarked(3)
        assert mgr2.is_bookmarked(7)

    def test_navigation_next(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        mgr.toggle(step=3, game=1)
        mgr.toggle(step=7, game=1)
        mgr.toggle(step=10, game=1)
        assert mgr.next_bookmark(1) == 3
        assert mgr.next_bookmark(3) == 7
        assert mgr.next_bookmark(10) is None

    def test_navigation_prev(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        mgr.toggle(step=3, game=1)
        mgr.toggle(step=7, game=1)
        mgr.toggle(step=10, game=1)
        assert mgr.prev_bookmark(10) == 7
        assert mgr.prev_bookmark(7) == 3
        assert mgr.prev_bookmark(3) is None

    def test_bookmarks_sorted(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        mgr.toggle(step=10, game=1)
        mgr.toggle(step=3, game=1)
        mgr.toggle(step=7, game=1)
        steps = [bm["step"] for bm in mgr.as_list()]
        assert steps == [3, 7, 10]

    def test_as_df_empty(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        df = mgr.as_df()
        assert len(df) == 0
        assert "step" in df.columns
        assert "annotation" in df.columns

    def test_as_df_with_data(self, populated_run_dir: Path):
        from roc.reporting.panel_debug import BookmarkManager

        mgr = BookmarkManager(populated_run_dir)
        mgr.toggle(step=5, game=1, annotation="test")
        df = mgr.as_df()
        assert len(df) == 1
        assert df.iloc[0]["step"] == 5


class TestBookmarkIntegration:
    def test_toggle_bookmark_adds_and_removes(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 5
        dashboard._toggle_bookmark()
        assert dashboard._bookmarks.is_bookmarked(5)
        assert "[*]" in dashboard._info_pane.object

        dashboard._toggle_bookmark()
        assert not dashboard._bookmarks.is_bookmarked(5)
        assert "[*]" not in dashboard._info_pane.object

    def test_toggle_bookmark_with_annotation(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._toggle_bookmark(step=7, annotation="interesting")
        assert dashboard._bookmarks.is_bookmarked(7)
        bms = dashboard._bookmarks.as_list()
        assert bms[0]["annotation"] == "interesting"

    def test_jump_next_bookmark(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._toggle_bookmark(step=5)
        dashboard._toggle_bookmark(step=10)
        dashboard._step_widget.value = 1
        dashboard._jump_next_bookmark()
        assert dashboard._step_widget.value == 5

    def test_jump_prev_bookmark(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._toggle_bookmark(step=3)
        dashboard._toggle_bookmark(step=8)
        dashboard._step_widget.value = 10
        dashboard._jump_prev_bookmark()
        assert dashboard._step_widget.value == 8

    def test_no_jump_when_no_bookmarks(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 5
        dashboard._jump_next_bookmark()
        assert dashboard._step_widget.value == 5

    def test_bookmark_indicator_on_step_change(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._toggle_bookmark(step=3)
        dashboard._on_step_change(3)
        assert "[*]" in dashboard._info_pane.object
        dashboard._on_step_change(4)
        assert "[*]" not in dashboard._info_pane.object

    def test_bookmark_list_updates(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert len(dashboard._bookmark_table.value) == 0
        dashboard._toggle_bookmark(step=5)
        assert len(dashboard._bookmark_table.value) == 1
        dashboard._toggle_bookmark(step=5)
        assert len(dashboard._bookmark_table.value) == 0

    def test_bookmark_uses_step_data_game_number(self, mock_store: RunStore):
        """Regression: bookmark should use the game_number from step data, not the selector."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        # Navigate to step 5 which is in game 1
        dashboard._step_widget.value = 5
        dashboard._on_step_change(5)
        # Note: game selector value doesn't affect bookmark game_number;
        # bookmarks always use the game_number from the current step data.
        # Bookmark should use the step data's game_number, not the selector
        dashboard._toggle_bookmark()
        bms = dashboard._bookmarks.as_list()
        assert len(bms) == 1
        # Step 5 is in game 1, so bookmark should say game 1
        assert dashboard._last_data is not None
        assert bms[0]["game"] == dashboard._last_data.game_number

    def test_run_change_reloads_bookmarks(self, populated_run_dir: Path, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store, data_dir=populated_run_dir.parent)
        dashboard._toggle_bookmark(step=5)
        assert dashboard._bookmarks.is_bookmarked(5)
        # Reload same run -- bookmarks should persist
        dashboard._on_run_change(populated_run_dir.name)
        assert dashboard._bookmarks.is_bookmarked(5)


class TestKeyboardShortcuts:
    def test_increase_speed(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._speed_selector.value == "5x"
        dashboard._increase_speed()
        assert dashboard._speed_selector.value == "10x"

    def test_decrease_speed(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._speed_selector.value == "5x"
        dashboard._decrease_speed()
        assert dashboard._speed_selector.value == "2x"

    def test_increase_speed_at_max(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._speed_selector.value = "20x"
        dashboard._increase_speed()
        assert dashboard._speed_selector.value == "20x"

    def test_decrease_speed_at_min(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._speed_selector.value = "0.5x"
        dashboard._decrease_speed()
        assert dashboard._speed_selector.value == "0.5x"

    def test_keypress_arrow_right(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 5
        event = MagicMock()
        event.new = "ArrowRight:123"
        dashboard._on_keypress(event)
        assert dashboard._step_widget.value == 6

    def test_keypress_arrow_left(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 5
        event = MagicMock()
        event.new = "ArrowLeft:123"
        dashboard._on_keypress(event)
        assert dashboard._step_widget.value == 4

    def test_keypress_home(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 10
        event = MagicMock()
        event.new = "Home:123"
        dashboard._on_keypress(event)
        assert dashboard._step_widget.value == dashboard._step_widget.start

    def test_keypress_end(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 1
        event = MagicMock()
        event.new = "End:123"
        dashboard._on_keypress(event)
        assert dashboard._step_widget.value == dashboard._step_widget.end

    def test_keypress_bookmark_toggle(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 5
        event = MagicMock()
        event.new = "b:123"
        dashboard._on_keypress(event)
        assert dashboard._bookmarks.is_bookmarked(5)

    def test_keypress_help_toggle(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert not dashboard._help_pane.visible
        event = MagicMock()
        event.new = "?:123"
        dashboard._on_keypress(event)
        assert dashboard._help_pane.visible
        dashboard._on_keypress(event)
        assert not dashboard._help_pane.visible

    def test_keypress_speed_plus(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._speed_selector.value == "5x"
        event = MagicMock()
        event.new = "+:123"
        dashboard._on_keypress(event)
        assert dashboard._speed_selector.value == "10x"

    def test_keypress_speed_minus(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._speed_selector.value == "5x"
        event = MagicMock()
        event.new = "-:123"
        dashboard._on_keypress(event)
        assert dashboard._speed_selector.value == "2x"

    def test_keypress_space_toggles_play(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._step_widget.direction == 0
        event = MagicMock()
        event.new = " :123"
        dashboard._on_keypress(event)
        assert dashboard._step_widget.direction == 1
        dashboard._on_keypress(event)
        assert dashboard._step_widget.direction == 0

    def test_keypress_g_cycles_game(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        initial_game = dashboard._game_selector.value
        event = MagicMock()
        event.new = "g:123"
        dashboard._on_keypress(event)
        assert dashboard._game_selector.value != initial_game

    def test_keypress_g_wraps_around(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        options = dashboard._game_selector.options
        # Cycle through all games and back to start
        event = MagicMock()
        event.new = "g:123"
        for _ in range(len(options)):
            dashboard._on_keypress(event)
        assert dashboard._game_selector.value == options[0]

    def test_keypress_empty_ignored(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        event = MagicMock()
        event.new = ""
        dashboard._on_keypress(event)  # Should not raise


class TestLiveMode:
    def test_live_mode_detects_new_steps(self, tmp_path: Path):
        """Step count should increase after new data is written."""
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)
        exporter.export([make_log_record(event_name="roc.game_start", body="game 1")])
        for _ in range(10):
            exporter.export(
                [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
            )
        exporter.force_flush()
        store = RunStore(dl_store)
        assert store.step_count() == 10
        # Write 5 more steps
        for _ in range(5):
            exporter.export(
                [make_log_record(event_name="roc.screen", body='{"chars":[],"fg":[],"bg":[]}')]
            )
        exporter.force_flush()
        assert store.step_count() == 15

    def test_poll_updates_dashboard_when_following(self, mock_store: RunStore):
        """Poll-based live update should advance step when following."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        # Move to last step -- following mode
        dashboard._step_widget.value = dashboard._step_widget.end
        assert dashboard._playback.live_following.is_active

        # Simulate a push from the game loop, then poll
        new_step = dashboard._step_widget.end + 1
        buf.push(StepData(step=new_step, game_number=1))
        dashboard._on_new_data()

        assert dashboard._step_widget.end == new_step
        assert dashboard._last_data.step == new_step

    def test_live_mode_auto_advances_when_following(self, mock_store: RunStore):
        """When in live_following, push should auto-advance with new data."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        buf.push(StepData(step=1, game_number=1))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active
        # Push more data -- should auto-advance display
        buf.push(StepData(step=2, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 2

    def test_live_mode_does_not_advance_when_paused(self, mock_store: RunStore):
        """When paused in live mode, push should not advance the display."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        buf.push(StepData(step=1, game_number=1))
        dashboard._on_new_data()
        dashboard._playback.send("pause")
        buf.push(StepData(step=2, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 1  # display stayed at step 1

    def test_historical_mode_never_following(self, mock_store: RunStore):
        """Historical (no step buffer) mode is never in live_following."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = dashboard._step_widget.end
        assert not dashboard._playback.live_following.is_active
        assert dashboard._playback.historical.is_active

    def test_live_badge_visible_when_following(self, mock_store: RunStore):
        """LIVE indicator should show when in live_following state."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        # Push data so we're in live following mode
        buf.push(StepData(step=1, game_number=1))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active
        assert dashboard._live_badge.visible

    def test_live_badge_hidden_when_reviewing(self, mock_store: RunStore):
        """LIVE indicator should hide when not following."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        buf.push(StepData(step=1, game_number=1))
        dashboard._on_new_data()
        # Pause to exit following
        dashboard._playback.send("pause")
        assert not dashboard._live_badge.visible

    def test_live_badge_hidden_in_historical_mode(self, mock_store: RunStore):
        """LIVE indicator should never show in historical (non-live) mode."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = dashboard._step_widget.end
        dashboard._on_step_change(dashboard._step_widget.end)
        assert not dashboard._live_badge.visible

    def test_game_selector_shows_games(self, mock_store: RunStore):
        """Game dropdown should be populated with available games."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        options = dashboard._game_selector.options
        assert len(options) >= 2
        assert "1" in options
        assert "2" in options

    def test_game_change_updates_step_range(self, mock_store: RunStore):
        """Switching game should adjust the slider range."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_game_change(2)
        # Step widget should now show game 2's first step
        assert dashboard._step_widget.value == 11

    def test_game_selector_shows_summary(self, mock_store: RunStore):
        """list_games should include step counts."""
        games_df = mock_store.list_games()
        assert len(games_df) >= 2
        assert "steps" in games_df.columns
        game1 = games_df[games_df["game_number"] == 1]
        assert game1.iloc[0]["steps"] == 10

    def test_poll_shows_new_data_badge_when_reviewing(self, mock_store: RunStore):
        """Poll should show 'new data' badge when user is reviewing (not following)."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        # Pause and step back -- not following
        dashboard._playback.send("pause")
        dashboard._step_widget.value = 5
        assert not dashboard._playback.live_following.is_active

        # Push new data and poll
        new_step = dashboard._step_widget.end + 1
        buf.push(StepData(step=new_step, game_number=1))
        dashboard._on_new_data()

        # Slider end expands so user can play forward, value stays put
        assert dashboard._step_widget.end == new_step
        assert dashboard._step_widget.value == 5
        # Badge was set by _on_new_data when not following
        assert dashboard._new_data_badge.visible

    def test_poll_adds_new_game_to_selector(self, mock_store: RunStore):
        """Poll with a new game_number should add it to the game selector."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        new_step = dashboard._step_widget.end + 1
        buf.push(StepData(step=new_step, game_number=99))
        dashboard._on_new_data()

        assert "99" in dashboard._game_selector.options

    def test_poll_while_reviewing_updates_slider_end(self, mock_store: RunStore):
        """When not following, polls should still update step_widget.end so Play works."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._playback.send("pause")
        dashboard._step_widget.value = 5
        assert not dashboard._playback.live_following.is_active
        original_end = dashboard._step_widget.end

        # Push 10 steps while not following, then poll
        for i in range(10):
            buf.push(StepData(step=original_end + 1 + i, game_number=1))
        dashboard._on_new_data()

        # Slider end should advance so Play can reach new steps
        assert dashboard._step_widget.end == original_end + 10
        # But value stays where the user left it
        assert dashboard._step_widget.value == 5

    def test_end_key_jumps_to_latest(self, mock_store: RunStore):
        """Pressing End should jump to latest step."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = 5  # not following

        # Push data while not following, then poll
        for i in range(10):
            buf.push(StepData(step=20 + i, game_number=1))
        dashboard._on_new_data()

        # Press End via keyboard dispatch
        dashboard._dispatch_key("End")

        # Should jump to latest
        assert dashboard._step_widget.end >= 29
        assert dashboard._step_widget.value == dashboard._step_widget.end

    def test_game_selector_not_auto_switched_during_review(self, mock_store: RunStore):
        """Reviewing game 1 while game 3 is live should NOT auto-switch to game 3."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        # Simulate: user pauses, selects game 1 and navigates to step 5
        dashboard._playback.send("pause")
        dashboard._game_selector.value = "1"
        dashboard._step_widget.value = 5

        # Push game 3 data while not following, then poll
        for i in range(5):
            buf.push(StepData(step=100 + i, game_number=3))
        dashboard._on_new_data()

        # Game selector should still show "1" (user's choice)
        assert dashboard._game_selector.value == "1"
        # But game 3 should be in the options
        assert "3" in dashboard._game_selector.options

    def test_no_phantom_game_zero(self, mock_store: RunStore):
        """Game selector should not contain a phantom '0' game."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        options = dashboard._game_selector.options
        assert "0" not in options

    def test_poll_new_game_preserves_existing_games(self, mock_store: RunStore):
        """When a new game starts, previous games must stay in the selector."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push game 1, poll
        buf.push(StepData(step=100, game_number=1))
        dashboard._on_new_data()
        assert "1" in dashboard._game_selector.options

        # Push game 2, poll -- game 1 must NOT disappear
        buf.push(StepData(step=101, game_number=2))
        dashboard._on_new_data()
        assert "1" in dashboard._game_selector.options
        assert "2" in dashboard._game_selector.options

        # Push game 3, poll -- games 1 and 2 must still be there
        buf.push(StepData(step=102, game_number=3))
        dashboard._on_new_data()
        assert "1" in dashboard._game_selector.options
        assert "2" in dashboard._game_selector.options
        assert "3" in dashboard._game_selector.options

    def test_poll_buffer_noop_when_no_new_data(self, mock_store: RunStore):
        """Poll should do nothing if no new data has arrived."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        original_end = dashboard._step_widget.end
        original_value = dashboard._step_widget.value

        dashboard._on_new_data()

        assert dashboard._step_widget.end == original_end
        assert dashboard._step_widget.value == original_value

    def test_poll_coalesces_multiple_pushes(self, mock_store: RunStore):
        """Multiple pushes between polls should coalesce to a single update."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        call_count = 0
        orig_apply = dashboard._apply_step_data

        def counting_apply(data: StepData) -> None:
            nonlocal call_count
            call_count += 1
            orig_apply(data)

        dashboard._apply_step_data = counting_apply  # type: ignore[method-assign]

        # Push 5 steps without polling
        base = dashboard._step_widget.end
        for i in range(5):
            buf.push(
                StepData(
                    step=base + 1 + i,
                    game_number=1,
                    game_metrics={"hp": 10, "hp_max": 16},
                )
            )

        # Single poll should only trigger _apply_step_data once
        dashboard._on_new_data()
        assert call_count == 1
        # Should be displaying the latest step
        assert dashboard._last_data.step == base + 5

    def test_on_new_data_idempotent_for_same_step(self, mock_store: RunStore):
        """Calling _on_new_data twice with same buffer state should only apply once."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        call_count = 0
        orig_apply = dashboard._apply_step_data

        def counting_apply(data: StepData) -> None:
            nonlocal call_count
            call_count += 1
            orig_apply(data)

        dashboard._apply_step_data = counting_apply  # type: ignore[method-assign]

        base = dashboard._step_widget.end
        buf.push(StepData(step=base + 1, game_number=1))
        dashboard._on_new_data()
        dashboard._on_new_data()  # same step, should be no-op
        dashboard._on_new_data()  # same step, should be no-op
        assert call_count == 1

    def test_step_buffer_data_available_during_replay(self, mock_store: RunStore):
        """Steps in the buffer should be retrievable when user navigates back."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push several steps with screen data
        base = dashboard._step_widget.end
        pushed_steps = []
        for i in range(5):
            step = base + 1 + i
            pushed_steps.append(step)
            buf.push(
                StepData(
                    step=step,
                    game_number=1,
                    screen={"chars": [[65 + i]], "fg": [["ffffff"]], "bg": [["000000"]]},
                    game_metrics={"hp": 10, "hp_max": 16},
                )
            )
        dashboard._on_new_data()

        # Navigate back to a buffered step -- should still have screen data
        target_step = pushed_steps[1]  # second pushed step
        dashboard._on_step_change(target_step)
        assert dashboard._last_data is not None
        assert dashboard._last_data.step == target_step
        assert dashboard._last_data.screen is not None

    def test_empty_store_defaults_game_one(self, tmp_path: Path):
        """With no Parquet data, game selector should default to '1', not '0'."""
        from roc.reporting.panel_debug import PanelDashboard

        store = RunStore(tmp_path)
        dashboard = PanelDashboard(store)
        assert dashboard._game_selector.value == "1"
        assert "0" not in dashboard._game_selector.options


class TestHelpOverlay:
    def test_help_starts_hidden(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert not dashboard._help_pane.visible

    def test_toggle_help_shows_and_hides(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._toggle_help()
        assert dashboard._help_pane.visible
        dashboard._toggle_help()
        assert not dashboard._help_pane.visible

    def test_help_content_has_shortcuts(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        content = dashboard._help_pane.object
        assert "Next step" in content
        assert "Previous step" in content
        assert "Play / pause" in content
        assert "Next game" in content
        assert "Toggle bookmark" in content
        assert "Toggle this help" in content

    def test_help_in_layout(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        md_panes = list(layout.select(pn.pane.Markdown))
        help_panes = [p for p in md_panes if "Next step" in str(p.object)]
        assert len(help_panes) == 1


class TestPollEdgeCases:
    """Regression tests for pull-based polling edge cases."""

    def test_first_poll_same_step_as_init(self, tmp_path: Path):
        """Bug regression: first poll where step==1 matches initial slider value.

        If _step_widget.value is already 1 and poll sets it to 1, the param
        watcher won't fire. _on_new_data must detect this and call
        _on_step_change directly.
        """
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        # Empty store: slider starts at value=1, end=1
        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)
        assert dashboard._step_widget.value == 1
        assert dashboard._step_widget.end == 1

        # Push step 1 (same as slider value) and poll
        buf.push(
            StepData(
                step=1,
                game_number=1,
                game_metrics={"hp": 16, "hp_max": 16},
                screen={"chars": [[65]], "fg": [["ffffff"]], "bg": [["000000"]]},
            )
        )
        dashboard._on_new_data()

        # _apply_step_data must have been called despite value not changing
        assert dashboard._last_data is not None
        assert dashboard._last_data.step == 1
        assert dashboard._hp_indicator.value == 16

    def test_poll_same_step_not_retriggered(self, mock_store: RunStore):
        """Second poll with no new data should be a no-op."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        buf.push(StepData(step=100, game_number=1, game_metrics={"hp": 10, "hp_max": 16}))
        dashboard._on_new_data()
        assert dashboard._last_seen_step == 100

        # Second poll with same data -- should return early
        call_count = 0
        orig = dashboard._apply_step_data

        def counting(data):
            nonlocal call_count
            call_count += 1
            orig(data)

        dashboard._apply_step_data = counting  # type: ignore[method-assign]
        dashboard._on_new_data()
        assert call_count == 0

    def test_poll_empty_buffer(self, mock_store: RunStore):
        """Poll on an empty buffer should do nothing."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        original_step = dashboard._step_widget.value
        dashboard._on_new_data()
        assert dashboard._step_widget.value == original_step

    def test_poll_no_buffer(self, mock_store: RunStore):
        """Poll with step_buffer=None should do nothing."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store, step_buffer=None)
        dashboard._on_new_data()  # Should not raise

    def test_live_following_ignores_game_change(self, mock_store: RunStore):
        """Game changes from _on_new_data don't re-enter _on_game_change in live following."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push data with game change -- should update selector without triggering
        # _on_game_change (which would jump to the game's first step)
        buf.push(StepData(step=100, game_number=1))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active
        assert dashboard._last_data.step == 100

        # Push another game change
        buf.push(StepData(step=101, game_number=99))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active
        assert dashboard._last_data.step == 101

    def test_poll_rapid_game_transitions(self, mock_store: RunStore):
        """Multiple game transitions between polls should all appear in selector."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push steps spanning 5 games between polls
        base = dashboard._step_widget.end
        for i in range(50):
            buf.push(StepData(step=base + 1 + i, game_number=10 + (i // 10)))

        dashboard._on_new_data()

        # All 5 games should be in selector
        options = dashboard._game_selector.options
        for g in [10, 11, 12, 13, 14]:
            assert str(g) in options, f"Game {g} missing from options {options}"

    def test_poll_following_then_review_then_following(self, mock_store: RunStore):
        """Full lifecycle: follow -> review -> follow should work cleanly."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end
        assert dashboard._playback.live_following.is_active

        # Phase 1: Follow mode -- push and poll
        base = dashboard._step_widget.end
        for i in range(5):
            buf.push(StepData(step=base + 1 + i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == base + 5
        assert dashboard._playback.live_following.is_active
        assert dashboard._live_badge.visible

        # Phase 2: User pauses and reviews -- go back
        dashboard._playback.send("pause")
        dashboard._step_widget.value = base + 2
        assert not dashboard._playback.live_following.is_active

        # Push more while reviewing
        for i in range(5):
            buf.push(StepData(step=base + 6 + i, game_number=1))
        dashboard._on_new_data()
        # Slider end should advance, but value stays
        assert dashboard._step_widget.end == base + 10
        assert dashboard._step_widget.value == base + 2

        # Phase 3: User presses End to return to live
        dashboard._dispatch_key("End")
        assert dashboard._step_widget.value == dashboard._step_widget.end
        assert dashboard._playback.live_following.is_active

        # New data should auto-advance display again
        for i in range(3):
            buf.push(StepData(step=base + 11 + i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == base + 13

    def test_poll_game_switch_during_review_no_jump(self, mock_store: RunStore):
        """Switching games manually while reviewing should not be overridden by poll."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        # Push data for games 1 and 2
        for i in range(10):
            buf.push(StepData(step=100 + i, game_number=1))
        for i in range(10):
            buf.push(StepData(step=110 + i, game_number=2))
        dashboard._on_new_data()

        # User pauses, switches to game 1 and goes to step 105
        dashboard._playback.send("pause")
        dashboard._game_selector.value = "1"
        dashboard._step_widget.value = 105
        assert not dashboard._playback.live_following.is_active

        # More game 2 data arrives
        for i in range(5):
            buf.push(StepData(step=120 + i, game_number=2))
        dashboard._on_new_data()

        # User's game selection and step must be preserved
        assert dashboard._game_selector.value == "1"
        assert dashboard._step_widget.value == 105

    def test_buffer_game_numbers_property(self):
        """game_numbers should return sorted unique game numbers."""
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        assert buf.game_numbers == []

        buf.push(StepData(step=1, game_number=3))
        buf.push(StepData(step=2, game_number=1))
        buf.push(StepData(step=3, game_number=2))
        buf.push(StepData(step=4, game_number=1))

        assert buf.game_numbers == [1, 2, 3]

    def test_buffer_capacity_eviction(self):
        """Ring buffer should evict oldest entries when full."""
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=5)
        for i in range(10):
            buf.push(StepData(step=i + 1, game_number=1))

        assert len(buf) == 5
        assert buf.min_step == 6
        assert buf.max_step == 10
        # Evicted steps should return None
        assert buf.get_step(1) is None
        assert buf.get_step(5) is None
        # Recent steps should be present
        assert buf.get_step(6) is not None
        assert buf.get_step(10) is not None

    def test_poll_slider_play_catches_up(self, mock_store: RunStore):
        """After pausing in review, pressing Play should be able to catch up."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=1000)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        # Push many steps
        base = dashboard._step_widget.end
        for i in range(100):
            buf.push(StepData(step=base + 1 + i, game_number=1))
        dashboard._on_new_data()

        # User pauses and goes to step 5 (way behind)
        dashboard._playback.send("pause")
        dashboard._step_widget.value = 5
        assert not dashboard._playback.live_following.is_active

        # Slider end should be far ahead
        assert dashboard._step_widget.end == base + 100

        # User can step forward to catch up
        for _ in range(10):
            dashboard._step_widget.value = min(
                dashboard._step_widget.value + 1, dashboard._step_widget.end
            )
        assert dashboard._step_widget.value == 15

    def test_on_new_data_tracks_last_seen(self, mock_store: RunStore):
        """_on_new_data should track the last seen step to skip duplicates."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        buf.push(StepData(step=100, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_seen_step == 100

        # Same step again should be skipped
        dashboard._on_new_data()
        assert dashboard._last_seen_step == 100

    def test_poll_with_only_game_metrics_change(self, mock_store: RunStore):
        """Poll should update even when only metrics change, not step number."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push step with HP=16
        base = dashboard._step_widget.end
        buf.push(StepData(step=base + 1, game_number=1, game_metrics={"hp": 16, "hp_max": 16}))
        dashboard._on_new_data()
        assert dashboard._hp_indicator.value == 16

        # Push next step with HP=10
        buf.push(StepData(step=base + 2, game_number=1, game_metrics={"hp": 10, "hp_max": 16}))
        dashboard._on_new_data()
        assert dashboard._hp_indicator.value == 10

    def test_pause_stops_live_following(self, tmp_path: Path):
        """Regression: pause button must stop live follow via _on_new_data.

        In LIVE mode, step advancement comes from push callbacks, not the
        Player's auto-advance timer.  The playback state machine tracks
        the mode so _on_new_data respects the paused state.
        """
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        # Push a few steps to get into live mode
        for i in range(1, 4):
            buf.push(StepData(step=i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 3
        assert dashboard._playback.live_following.is_active

        # User clicks the pause button
        dashboard._playback.send("pause")
        assert dashboard._playback.live_paused.is_active

        # Push more data -- should NOT advance because user paused
        buf.push(StepData(step=4, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 3  # display stayed at 3
        assert dashboard._step_widget.end == 4  # slider end grew

        buf.push(StepData(step=5, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 3  # still at 3

        # User presses End to return to live
        dashboard._dispatch_key("End")
        assert dashboard._playback.live_following.is_active
        assert dashboard._step_widget.value == 5

        # Now pushes should advance display again
        buf.push(StepData(step=6, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 6  # following again

    def test_pause_then_play_resumes_following(self, tmp_path: Path):
        """After pause+play, live following should resume."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        for i in range(1, 4):
            buf.push(StepData(step=i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 3

        # User clicks pause button
        dashboard._playback.send("pause")
        assert dashboard._playback.live_paused.is_active

        # Push -- should not advance display
        buf.push(StepData(step=4, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 3

        # User clicks play button -- enters catchup (behind live edge)
        dashboard._playback.send("resume")
        assert dashboard._playback.live_catchup.is_active

        # Go to end so we're following again
        dashboard._dispatch_key("End")
        assert dashboard._step_widget.value == 4

        # Now pushes should advance display
        buf.push(StepData(step=5, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_data.step == 5


    def test_step_back_while_following_pauses(self, tmp_path: Path):
        """Regression: clicking prev/slider while live should auto-pause.

        Without this, _on_new_data snaps the user back to the live edge
        within ~200ms of any navigation, making it impossible to review
        historical steps during a live session.
        """
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        # Push enough steps so navigating back triggers user_navigate
        # (needs to be > 30 steps from end to be outside the near_end threshold)
        for i in range(1, 51):
            buf.push(StepData(step=i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active

        # Simulate user clicking to a step far from the end (>30 away)
        dashboard._step_widget.value = 10
        assert dashboard._playback.live_paused.is_active

        # Push more data -- should NOT snap back
        buf.push(StepData(step=51, game_number=1))
        dashboard._on_new_data()
        assert dashboard._step_widget.value == 10  # stayed
        assert dashboard._step_widget.end == 51  # end grew

    def test_game_change_while_following_pauses(self, tmp_path: Path):
        """Regression: changing game selector while live should auto-pause."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        for i in range(1, 4):
            buf.push(StepData(step=i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active

        # User changes game -- should pause
        dashboard._playback.send("user_navigate")
        assert dashboard._playback.live_paused.is_active

    def test_catchup_timer_does_not_reenter_pause(self, tmp_path: Path):
        """Regression: timer-driven step advancement in catchup must not
        trigger user_navigate which would immediately re-pause.

        The timer increments _step_widget.value by 1 each tick. Since
        new < end, _handle_step_widget must NOT send user_navigate in
        catchup mode.
        """
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        for i in range(1, 20):
            buf.push(StepData(step=i, game_number=1))
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active

        # Pause, then resume -> catchup
        dashboard._playback.send("pause")
        dashboard._playback.send("resume")
        assert dashboard._playback.live_catchup.is_active

        # Simulate timer ticks incrementing the slider
        for step in range(dashboard._step_widget.value + 1, 15):
            dashboard._step_widget.value = step
            # Must stay in catchup -- NOT transition to paused
            assert dashboard._playback.live_catchup.is_active, (
                f"Step {step}: expected catchup, got "
                f"{dashboard._playback.current_state}"
            )

    def test_first_button_updates_display(self, tmp_path: Path):
        """Regression: clicking 'first' while paused must update display
        data, even if self.step is already 1 (param watcher no-op)."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        for i in range(1, 10):
            buf.push(StepData(step=i, game_number=1))
        dashboard._on_new_data()

        # Pause and navigate to step 5
        dashboard._playback.send("pause")
        dashboard._step_widget.value = 5
        assert dashboard._last_data is not None
        step_5_data = dashboard._last_data.step

        # Now click "first" -- value goes to 1
        dashboard._step_widget.value = 1
        assert dashboard._last_data is not None
        assert dashboard._last_data.step != step_5_data  # data actually changed

    def test_push_does_not_update_slider_value(self, tmp_path: Path):
        """Push updates must not set _step_widget.value while following.

        Server-side value updates race with client-side button clicks,
        causing user interactions (prev/next) to be swallowed.
        """
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        store = RunStore(tmp_path)
        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(store, step_buffer=buf)

        buf.push(StepData(step=1, game_number=1))
        dashboard._on_new_data()
        initial_value = dashboard._step_widget.value

        # Push many more steps
        for i in range(2, 50):
            buf.push(StepData(step=i, game_number=1))
            dashboard._on_new_data()

        # Slider end should have grown but value should not have been
        # updated by pushes (only by on_enter_live_following on first entry)
        assert dashboard._step_widget.end == 49
        # Value stays at wherever on_enter_live_following set it (1)
        assert dashboard._step_widget.value == initial_value


class TestRealWorldSimulations:
    """Simulate realistic debugging workflows end-to-end."""

    def test_investigate_death_across_games(self, mock_store: RunStore):
        """User investigates a death: switch game, step backward, bookmark, return to live."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=1000)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        # Move to end so we're in following mode
        dashboard._step_widget.value = dashboard._step_widget.end

        # Game is live in game 3
        base = dashboard._step_widget.end
        for i in range(50):
            game = 1 + i // 20  # games 1, 2, 3
            buf.push(
                StepData(
                    step=base + 1 + i,
                    game_number=game,
                    game_metrics={"hp": max(16 - i % 20, 0), "hp_max": 16},
                )
            )
        dashboard._on_new_data()
        assert dashboard._playback.live_following.is_active
        assert dashboard._last_data.step == base + 50

        # User pauses to investigate, then switches to game 1
        dashboard._playback.send("pause")
        dashboard._game_selector.value = "1"
        # _on_game_change should jump to game 1's first step
        assert not dashboard._playback.live_following.is_active

        # User steps through looking for HP drop
        for _ in range(5):
            dashboard._dispatch_key("ArrowRight")

        # Bookmark the interesting step
        current_step = dashboard._step_widget.value
        dashboard._dispatch_key("b")
        assert dashboard._bookmarks.is_bookmarked(current_step)

        # Step forward a few more
        for _ in range(3):
            dashboard._dispatch_key("ArrowRight")

        # Jump back to bookmark
        dashboard._dispatch_key("[")
        assert dashboard._step_widget.value == current_step

        # Meanwhile, more live data arrives
        for i in range(10):
            buf.push(StepData(step=base + 51 + i, game_number=3))
        dashboard._on_new_data()

        # User's review position must be preserved
        assert dashboard._step_widget.value == current_step
        assert not dashboard._playback.live_following.is_active

        # Return to live
        dashboard._dispatch_key("End")
        assert dashboard._playback.live_following.is_active

    def test_rapid_home_end_toggling(self, mock_store: RunStore):
        """Rapidly toggle Home/End while live data arrives."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=1000)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        base = dashboard._step_widget.end
        for i in range(20):
            buf.push(StepData(step=base + 1 + i, game_number=1))
        dashboard._on_new_data()

        # Rapid Home/End/Home/End
        dashboard._dispatch_key("Home")
        assert dashboard._step_widget.value == dashboard._step_widget.start
        assert not dashboard._playback.live_following.is_active

        dashboard._dispatch_key("End")
        assert dashboard._step_widget.value == dashboard._step_widget.end
        assert dashboard._playback.live_following.is_active

        dashboard._dispatch_key("Home")
        assert not dashboard._playback.live_following.is_active

        # Push more data while at Home
        for i in range(5):
            buf.push(StepData(step=base + 21 + i, game_number=1))
        dashboard._on_new_data()

        # Value should stay at Home
        assert dashboard._step_widget.value == dashboard._step_widget.start
        # But slider end should have advanced
        assert dashboard._step_widget.end == base + 25

        dashboard._dispatch_key("End")
        assert dashboard._step_widget.value == base + 25
        assert dashboard._playback.live_following.is_active

    def test_play_from_review_position(self, mock_store: RunStore):
        """Press Play while reviewing mid-game, verify slider advances."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=1000)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        base = dashboard._step_widget.end
        for i in range(50):
            buf.push(StepData(step=base + 1 + i, game_number=1))
        dashboard._on_new_data()

        # User presses Home then navigates to step 5 to review
        dashboard._dispatch_key("Home")
        dashboard._step_widget.value = 5
        assert not dashboard._playback.live_following.is_active

        # User presses Space to play
        dashboard._dispatch_key(" ")
        assert dashboard._step_widget.direction == 1

        # Simulate player advancing a few steps (Player widget auto-increments)
        for _ in range(3):
            dashboard._step_widget.value += 1

        assert dashboard._step_widget.value == 8

        # Pause
        dashboard._dispatch_key(" ")
        assert dashboard._step_widget.direction == 0
        assert dashboard._step_widget.value == 8

    def test_game_boundary_stepping(self, mock_store: RunStore):
        """Step through the exact boundary where games change."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push game 1 (steps 100-109) and game 2 (steps 110-119)
        base = dashboard._step_widget.end
        for i in range(20):
            game = 1 if i < 10 else 2
            buf.push(
                StepData(
                    step=base + 1 + i,
                    game_number=game,
                    game_metrics={"hp": 16 - i, "hp_max": 16},
                )
            )
        dashboard._on_new_data()

        # Navigate to step just before game boundary
        boundary_step = base + 10  # last step of game 1
        dashboard._step_widget.value = boundary_step
        dashboard._on_step_change(boundary_step)
        assert dashboard._last_data is not None
        assert dashboard._last_data.game_number == 1

        # Step forward into game 2
        dashboard._dispatch_key("ArrowRight")
        dashboard._on_step_change(boundary_step + 1)
        assert dashboard._last_data is not None
        assert dashboard._last_data.game_number == 2

        # Step backward back to game 1
        dashboard._dispatch_key("ArrowLeft")
        dashboard._on_step_change(boundary_step)
        assert dashboard._last_data is not None
        assert dashboard._last_data.game_number == 1

    def test_speed_change_during_playback(self, mock_store: RunStore):
        """Change speed while Player is actively playing."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)

        # Start playing
        dashboard._dispatch_key(" ")
        assert dashboard._step_widget.direction == 1
        assert dashboard._step_widget.interval == 200  # 5x default

        # Increase speed twice
        dashboard._dispatch_key("+")
        assert dashboard._step_widget.interval == 100  # 10x
        dashboard._dispatch_key("+")
        assert dashboard._step_widget.interval == 50  # 20x

        # Decrease speed
        dashboard._dispatch_key("-")
        assert dashboard._step_widget.interval == 100  # 10x

        # Still playing
        assert dashboard._step_widget.direction == 1

        # Pause
        dashboard._dispatch_key(" ")
        assert dashboard._step_widget.direction == 0

    def test_rapid_game_cycling_with_g_key(self, mock_store: RunStore):
        """Rapidly press G to cycle through all games."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        initial = dashboard._game_selector.value
        num_games = len(dashboard._game_selector.options)

        # Cycle through all games
        visited = [initial]
        for _ in range(num_games * 2):  # go around twice
            dashboard._dispatch_key("g")
            visited.append(dashboard._game_selector.value)

        # Should wrap around and visit all games
        unique_visited = set(visited)
        assert len(unique_visited) == num_games

        # After 2 full cycles, should be back at initial
        assert dashboard._game_selector.value == initial

    def test_run_change_resets_poll_state(self, populated_run_dir: Path, mock_store: RunStore):
        """Bug regression: _last_seen_step must reset when switching runs."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf, data_dir=populated_run_dir.parent)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Advance poll state to a high step
        buf.push(StepData(step=500, game_number=1))
        dashboard._on_new_data()
        assert dashboard._last_seen_step == 500

        # Switch runs -- _last_seen_step must reset
        dashboard._on_run_change(populated_run_dir.name)
        assert dashboard._last_seen_step == 0

    def test_multiple_bookmarks_navigation(self, mock_store: RunStore):
        """Create several bookmarks, navigate forward and backward between them."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)

        # Create bookmarks at steps 3, 7, 12
        dashboard._toggle_bookmark(step=3)
        dashboard._toggle_bookmark(step=7)
        dashboard._toggle_bookmark(step=12)

        # Start at step 1
        dashboard._step_widget.value = 1

        # Navigate forward through bookmarks
        dashboard._dispatch_key("]")
        assert dashboard._step_widget.value == 3

        dashboard._dispatch_key("]")
        assert dashboard._step_widget.value == 7

        dashboard._dispatch_key("]")
        assert dashboard._step_widget.value == 12

        # No more bookmarks forward
        dashboard._dispatch_key("]")
        assert dashboard._step_widget.value == 12

        # Navigate backward
        dashboard._dispatch_key("[")
        assert dashboard._step_widget.value == 7

        dashboard._dispatch_key("[")
        assert dashboard._step_widget.value == 3

        # No more bookmarks backward
        dashboard._dispatch_key("[")
        assert dashboard._step_widget.value == 3

    def test_help_toggle_during_interaction(self, mock_store: RunStore):
        """Toggle help overlay, ensure other shortcuts still work."""
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._step_widget.value = 5

        # Show help
        dashboard._dispatch_key("?")
        assert dashboard._help_pane.visible

        # Other shortcuts should still work while help is shown
        dashboard._dispatch_key("ArrowRight")
        assert dashboard._step_widget.value == 6

        # Hide help
        dashboard._dispatch_key("h")
        assert not dashboard._help_pane.visible

    def test_poll_during_game_switch(self, mock_store: RunStore):
        """Poll arrives while user is mid-game-switch (game selector open)."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        # User pauses and reviews game 1, step 5
        dashboard._playback.send("pause")
        dashboard._game_selector.value = "1"
        dashboard._step_widget.value = 5
        assert not dashboard._playback.live_following.is_active

        # Poll arrives with game 3 data
        buf.push(StepData(step=200, game_number=3))
        dashboard._on_new_data()

        # User's review position must be preserved
        assert dashboard._game_selector.value == "1"
        assert dashboard._step_widget.value == 5

        # Game 3 should be in options
        assert "3" in dashboard._game_selector.options

    def test_evicted_step_falls_back_to_store(self, mock_store: RunStore):
        """When buffer evicts a step, navigation should fall back to RunStore."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        # Tiny buffer that evicts quickly
        buf = StepBuffer(capacity=5)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)
        dashboard._step_widget.value = dashboard._step_widget.end

        # Push 10 steps -- first 5 get evicted
        base = dashboard._step_widget.end
        for i in range(10):
            buf.push(StepData(step=base + 1 + i, game_number=1))
        dashboard._on_new_data()

        # Navigate to step 3 (which is in the RunStore, not buffer)
        dashboard._on_step_change(3)
        assert dashboard._last_data is not None
        assert dashboard._last_data.step == 3

        # Navigate to a buffered step
        dashboard._on_step_change(base + 8)
        assert dashboard._last_data is not None
        assert dashboard._last_data.step == base + 8

    def test_live_badge_and_new_data_badge_mutual_exclusion(self, mock_store: RunStore):
        """LIVE and 'New data available' badges should never both be visible."""
        from roc.reporting.panel_debug import PanelDashboard
        from roc.reporting.run_store import StepData
        from roc.reporting.step_buffer import StepBuffer

        buf = StepBuffer(capacity=100)
        dashboard = PanelDashboard(mock_store, step_buffer=buf)

        # Following -- LIVE should show (state machine enters live_following on init)
        assert dashboard._playback.live_following.is_active
        assert dashboard._live_badge.visible
        assert not dashboard._new_data_badge.visible

        # Push new data
        base = dashboard._step_widget.end
        buf.push(StepData(step=base + 1, game_number=1))
        dashboard._on_new_data()

        # Still following -- should still be LIVE
        assert dashboard._live_badge.visible

        # Pause and go back to review
        dashboard._playback.send("pause")
        dashboard._step_widget.value = 5
        assert not dashboard._live_badge.visible

        # Push more data while reviewing
        buf.push(StepData(step=base + 2, game_number=1))
        dashboard._on_new_data()
        assert dashboard._new_data_badge.visible
        assert not dashboard._live_badge.visible
