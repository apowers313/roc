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

from roc.reporting.parquet_exporter import ParquetExporter
from roc.reporting.run_store import RunStore


@pytest.fixture()
def populated_run_dir(tmp_path: Path) -> Path:
    """Create a run directory with known test data using ParquetExporter."""
    exporter = ParquetExporter(run_dir=tmp_path, flush_interval=100)

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

    def test_step_change_updates_game_selector(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(5)
        assert dashboard._game_selector.value == "1"

        dashboard._on_step_change(12)
        assert dashboard._game_selector.value == "2"

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

    def test_fallback_when_no_metrics(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        # Step 12 is in game 2 which has no metrics
        dashboard._on_step_change(12)
        assert dashboard._hp_indicator.name == "Step"
        assert dashboard._score_indicator.name == "Game"


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
