# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/state.py -- additional coverage for State.init, State.print, gauges."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    mock.node_counter = MagicMock()
    mock.edge_counter = MagicMock()
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestStateInit:
    @patch("roc.reporting.state.State.print_startup_info")
    @patch("roc.reporting.state.ObjectResolver")
    @patch("roc.reporting.state.Attention")
    def test_init_sets_up_listeners(self, mock_attention, mock_obj_resolver, mock_print_startup):
        from roc.reporting.state import State, _state_init_done

        import roc.reporting.state as state_mod

        # Reset to allow init to run
        original = state_mod._state_init_done
        state_mod._state_init_done = False

        try:
            State.init()
            assert state_mod._state_init_done is True
            mock_print_startup.assert_called_once()
        finally:
            state_mod._state_init_done = original

    @patch("roc.reporting.state.State.print_startup_info")
    def test_init_skips_if_already_done(self, mock_print_startup):
        import roc.reporting.state as state_mod

        from roc.reporting.state import State

        original = state_mod._state_init_done
        state_mod._state_init_done = True

        try:
            State.init()
            # Should not call print_startup_info since already initialized
            mock_print_startup.assert_not_called()
        finally:
            state_mod._state_init_done = original


class TestStatePrint:
    @patch("roc.reporting.state.State.init")
    @patch("builtins.print")
    def test_print_outputs_sections(self, mock_print, mock_init):
        from roc.reporting.state import State, states

        State.print()

        # Verify init was called
        mock_init.assert_called_once()

        # Collect all print calls
        printed = [str(call[0][0]) for call in mock_print.call_args_list]
        all_output = "\n".join(printed)

        # Should have three section headers
        assert "ENVIRONMENT" in all_output
        assert "GRAPH DB" in all_output
        assert "AGENT" in all_output


class TestNodeCacheGauge:
    def test_yields_observation(self):
        from roc.reporting.state import node_cache_gague

        mock_cache = MagicMock()
        mock_cache.currsize = 10
        mock_cache.maxsize = 100

        with (
            patch("roc.reporting.state.Node.get_cache", return_value=mock_cache),
            patch("roc.reporting.state.State.send_events"),
            patch("roc.reporting.state.State.print"),
        ):
            results = list(node_cache_gague())

        assert len(results) == 1
        assert results[0].value == 10
        assert results[0].attributes == {"max": 100}


class TestEdgeCacheGauge:
    def test_yields_observation(self):
        from roc.reporting.state import edge_cache_gague

        mock_cache = MagicMock()
        mock_cache.currsize = 5
        mock_cache.maxsize = 50

        with patch("roc.reporting.state.Edge.get_cache", return_value=mock_cache):
            results = list(edge_cache_gague())

        assert len(results) == 1
        assert results[0].value == 5
        assert results[0].attributes == {"max": 50}


class TestStateSendEventsExtra:
    @patch("roc.reporting.state.Observability.event")
    def test_send_events_with_saliency(self, mock_event):
        from roc.reporting.state import State, states

        mock_sal = MagicMock()
        mock_sal.to_html_vals.return_value = "<html>"
        mock_sal.feature_report.return_value = {"feat1": 0.5}

        states.screen.val = None
        states.salency.val = mock_sal
        states.object.val = None
        states.attention.val = None

        State.send_events()

        # Should send SaliencyObsEvent and FeatureObsEvent
        assert mock_event.call_count == 2
        states.salency.val = None

    @patch("roc.reporting.state.Observability.event")
    def test_send_events_with_object(self, mock_event):
        from roc.reporting.state import State, states

        mock_obj = MagicMock()
        states.screen.val = None
        states.salency.val = None
        states.object.val = mock_obj
        states.attention.val = None

        State.send_events()
        assert mock_event.call_count == 1
        states.object.val = None

    @patch("roc.reporting.state.Observability.event")
    def test_send_events_with_attention(self, mock_event):
        from roc.reporting.state import State, states

        mock_att = MagicMock()
        mock_att.focus_points = "data"
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = mock_att

        State.send_events()
        assert mock_event.call_count == 1
        states.attention.val = None
