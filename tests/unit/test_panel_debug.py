# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/panel_debug.py."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from helpers.otel import make_log_record
from opentelemetry._logs import SeverityNumber

from roc.reporting.parquet_exporter import ParquetExporter
from roc.reporting.run_store import RunStore, StepData


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


@pytest.fixture()
def sample_step_data(mock_store: RunStore) -> StepData:
    """Get a StepData instance for step 1."""
    return mock_store.get_step_data(1)


class TestDashboardCreation:
    def test_dashboard_creates_without_error(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard is not None

    def test_dashboard_has_step_slider(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        players = [w for w in layout.select(pn.widgets.Player) if isinstance(w, pn.widgets.Player)]
        assert len(players) >= 1, "Dashboard should contain a Player widget"

    def test_dashboard_has_run_selector(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        autos = [
            w
            for w in layout.select(pn.widgets.AutocompleteInput)
            if isinstance(w, pn.widgets.AutocompleteInput)
        ]
        assert len(autos) >= 1, "Dashboard should contain an AutocompleteInput widget"


class TestScreenRendering:
    def test_screen_viewer_has_data(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._screen_viewer.grid_data is not None
        html = dashboard._screen_viewer._render()
        assert "<span" in html
        assert "monospace" in html

    def test_screen_viewer_handles_none(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._screen_viewer.grid_data = None
        html = dashboard._screen_viewer._render()
        assert "No data" in html


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

    def test_step_change_updates_game_indicator(self, mock_store: RunStore):
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
        from unittest.mock import MagicMock

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

    def test_timestamp_formatted_as_local_time(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        info = dashboard._info_pane.object
        assert "202" in info
        assert ":" in info


class TestSaliencyPanel:
    def test_saliency_viewer_has_data(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        assert dashboard._saliency_viewer.grid_data is not None
        html = dashboard._saliency_viewer._render()
        assert "<span" in html

    def test_saliency_viewer_handles_none(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._saliency_viewer.grid_data = None
        html = dashboard._saliency_viewer._render()
        assert "No data" in html


class TestComponentIntegration:
    """Test that the dashboard correctly wires data to components."""

    def test_features_populated_on_step(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        children = list(dashboard._features_container)
        assert len(children) > 0

    def test_metrics_populated_on_step(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        children = list(dashboard._metrics_container)
        assert len(children) > 0

    def test_graph_populated_on_step(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        children = list(dashboard._graph_container)
        assert len(children) > 0

    def test_events_populated_on_step(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        children = list(dashboard._events_container)
        assert len(children) > 0

    def test_logs_populated_on_step(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        children = list(dashboard._logs_container)
        assert len(children) > 0

    def test_object_info_rendered(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        html = dashboard._object_pane.object
        assert isinstance(html, str)
        assert len(html) > 0

    def test_focus_points_rendered(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        html = dashboard._focus_pane.object
        assert isinstance(html, str)
        assert len(html) > 0

    def test_attenuation_populated(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        children = list(dashboard._attenuation_container)
        assert len(children) > 0

    def test_log_level_filter_updates(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        dashboard._on_step_change(1)
        # Change to ERROR and re-render
        dashboard._log_level_selector.value = "ERROR"
        dashboard._on_log_level_change()
        children = list(dashboard._logs_container)
        assert len(children) > 0


class TestFullLayout:
    def test_layout_has_all_sections(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = list(layout.select(pn.Card))
        card_titles = [c.title for c in cards if hasattr(c, "title")]

        assert "Perception" in card_titles
        assert "Attention" in card_titles
        assert "Object Resolution" in card_titles
        assert "Game State" in card_titles
        assert "Log Messages" in card_titles

    def test_game_state_expanded(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        assert not cards["Game State"].collapsed

    def test_detail_sections_collapsed(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        assert cards["Perception"].collapsed
        assert cards["Attention"].collapsed
        assert cards["Object Resolution"].collapsed
        assert cards["Log Messages"].collapsed

    def test_status_bar_present(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        status = dashboard._status_pane.object
        assert "HP" in status
        assert "Score" in status

    def test_player_widget_always_accessible(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        players = list(layout.select(pn.widgets.Player))
        assert len(players) >= 1

    def test_screen_in_game_state_card(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = {c.title: c for c in layout.select(pn.Card) if hasattr(c, "title")}
        # Screen viewer should be inside Game State card (search for HTML panes with grid content)
        game_html = list(cards["Game State"].select(pn.pane.HTML))
        assert len(game_html) >= 1


class TestDesignRegression:
    """Lock in design decisions to prevent regressions."""

    def test_dark_mode_colors_used(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard, _BG

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        styles = layout.styles or {}
        assert styles.get("background") == _BG

    def test_saliency_no_legend(self, mock_store: RunStore):
        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        html = dashboard._saliency_viewer._render()
        assert "Low" not in html
        assert "High" not in html

    def test_compact_card_margins(self, mock_store: RunStore):
        import panel as pn

        from roc.reporting.panel_debug import PanelDashboard

        dashboard = PanelDashboard(mock_store)
        layout = dashboard.servable()
        cards = list(layout.select(pn.Card))
        for card in cards:
            margin = (card.styles or {}).get("margin-bottom", "0px")
            px = int(margin.replace("px", ""))
            assert px <= 4, f"Card margin {margin} > 4px"
