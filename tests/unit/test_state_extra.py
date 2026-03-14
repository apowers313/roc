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
        from roc.reporting.state import State

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
        from roc.reporting.state import State

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

        with patch("roc.reporting.state.Node.get_cache", return_value=mock_cache):
            results = list(node_cache_gague())

        assert len(results) == 1
        assert results[0].value == 10
        assert results[0].attributes == {"max": 100}

    def test_gauge_does_not_call_print_or_emit(self):
        """node_cache_gauge callback should not call State.print() or emit_state_logs()."""
        from roc.reporting.state import node_cache_gague

        mock_cache = MagicMock()
        mock_cache.currsize = 10
        mock_cache.maxsize = 100

        with (
            patch("roc.reporting.state.Node.get_cache", return_value=mock_cache),
            patch("roc.reporting.state.State.print") as mock_print,
            patch("roc.reporting.state.State.emit_state_logs") as mock_emit,
        ):
            list(node_cache_gague())

        mock_print.assert_not_called()
        mock_emit.assert_not_called()


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


class TestEmitStateLogsExtra:
    def test_emit_state_logs_with_saliency(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_sal = MagicMock()
        mock_sal.feature_report.return_value = {"feat1": 0.5}
        mock_sal.to_html_vals.return_value = {
            "chars": [[46]],
            "fg": [["ffffff"]],
            "bg": [["0000ff"]],
        }

        states.screen.val = None
        states.salency.val = mock_sal
        states.object.val = None
        states.attention.val = None

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # Should emit saliency, features, and graphdb.summary records
            assert mock_logger.return_value.emit.call_count == 3
        states.salency.val = None

    def test_emit_state_logs_with_object(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_obj = MagicMock()
        states.screen.val = None
        states.salency.val = None
        states.object.val = mock_obj
        states.attention.val = None

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # object + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
        states.object.val = None

    def test_emit_state_logs_with_attention(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_att = MagicMock()
        mock_att.focus_points = "data"
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = mock_att

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # focus_points + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
        states.attention.val = None
