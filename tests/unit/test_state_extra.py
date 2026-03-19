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
        from roc.config import Config
        from roc.event import Event
        from roc.reporting.state import State, states

        cfg = Config.get()
        cfg.emit_state_saliency = True
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

    def test_emit_state_logs_with_intrinsic(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_intr = MagicMock()
        mock_intr.intrinsics = {"hp": 10}
        mock_intr.normalized_intrinsics = {"hp": 0.5}
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = mock_intr

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # intrinsics + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            record = calls[0][0][0]
            assert record.attributes["event.name"] == "roc.intrinsics"
        states.intrinsic.val = None

    def test_emit_state_logs_with_significance(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_sig = MagicMock()
        mock_sig.significance = 0.75
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = mock_sig

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # significance + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            record = calls[0][0][0]
            assert record.attributes["event.name"] == "roc.significance"
        states.significance.val = None

    def test_emit_state_logs_with_action(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_action = MagicMock()
        mock_action.action = 42
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = None
        states.action.val = mock_action

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # action + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            record = calls[0][0][0]
            assert record.attributes["event.name"] == "roc.action"
        states.action.val = None

    def test_emit_state_logs_with_transform(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_edge = MagicMock()
        mock_edge.dst = "some_change"
        mock_transform = MagicMock()
        mock_transform.transform.src_edges = [mock_edge]
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = None
        states.action.val = None
        states.transform.val = mock_transform

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # transform_summary + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            record = calls[0][0][0]
            assert record.attributes["event.name"] == "roc.transform_summary"
        states.transform.val = None

    def test_emit_state_logs_with_predict(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        mock_pred = MagicMock()
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = None
        states.action.val = None
        states.transform.val = None
        states.predict.val = mock_pred

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # prediction + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            record = calls[0][0][0]
            assert record.attributes["event.name"] == "roc.prediction"
        states.predict.val = None

    def test_emit_state_logs_with_message(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = None
        states.action.val = None
        states.transform.val = None
        states.predict.val = None
        states.message.val = "Hello adventurer!"

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # message + graphdb.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            record = calls[0][0][0]
            assert record.attributes["event.name"] == "roc.message"
            assert record.body == "Hello adventurer!"
        states.message.val = None

    def test_emit_state_logs_with_empty_message(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = None
        states.action.val = None
        states.transform.val = None
        states.predict.val = None
        states.message.val = "   "

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # empty message should NOT emit, only graphdb.summary
            assert mock_logger.return_value.emit.call_count == 1
        states.message.val = None

    def test_emit_state_logs_with_event_bus_counts(self):
        from roc.event import Event
        from roc.reporting.state import State, states

        Event._step_counts.clear()
        Event._step_counts["TestBus"] = 5
        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None
        states.intrinsic.val = None
        states.significance.val = None
        states.action.val = None
        states.transform.val = None
        states.predict.val = None
        states.message.val = None

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.emit_state_logs()
            # graphdb.summary + event.summary
            assert mock_logger.return_value.emit.call_count == 2
            calls = mock_logger.return_value.emit.call_args_list
            event_names = [c[0][0].attributes["event.name"] for c in calls]
            assert "roc.event.summary" in event_names
        Event._step_counts.clear()


class TestNewStateClasses:
    """Tests for the new state classes added to support additional pipeline stages."""

    def test_current_intrinsic_state_init(self):
        from roc.reporting.state import CurrentIntrinsicState

        s = CurrentIntrinsicState()
        assert s.name == "curr-intrinsic"
        assert s.display_name == "Current Intrinsics"
        assert s.val is None

    def test_current_intrinsic_state_set_and_str(self):
        from roc.reporting.state import CurrentIntrinsicState

        s = CurrentIntrinsicState()
        mock_data = MagicMock()
        mock_data.configure_mock(**{"__repr__": MagicMock(return_value="IntrinsicData(hp=10)")})
        s.set(mock_data)
        assert s.val is mock_data
        result = str(s)
        assert "Current Intrinsics:" in result

    def test_current_intrinsic_state_str_none(self):
        from roc.reporting.state import CurrentIntrinsicState

        s = CurrentIntrinsicState()
        assert str(s) == "Current Intrinsics: None"

    def test_current_significance_state_init(self):
        from roc.reporting.state import CurrentSignificanceState

        s = CurrentSignificanceState()
        assert s.name == "curr-significance"
        assert s.display_name == "Current Significance"
        assert s.val is None

    def test_current_significance_state_set_and_str(self):
        from roc.reporting.state import CurrentSignificanceState

        s = CurrentSignificanceState()
        mock_data = MagicMock()
        mock_data.significance = 0.75
        s.set(mock_data)
        assert s.val is mock_data
        result = str(s)
        assert "Current Significance: 0.75" in result

    def test_current_significance_state_str_none(self):
        from roc.reporting.state import CurrentSignificanceState

        s = CurrentSignificanceState()
        assert str(s) == "Current Significance: None"

    def test_current_action_state_init(self):
        from roc.reporting.state import CurrentActionState

        s = CurrentActionState()
        assert s.name == "curr-action"
        assert s.display_name == "Current Action"
        assert s.val is None

    def test_current_action_state_set_and_str(self):
        from roc.reporting.state import CurrentActionState

        s = CurrentActionState()
        mock_data = MagicMock()
        mock_data.action = 42
        s.set(mock_data)
        assert s.val is mock_data
        result = str(s)
        assert "Current Action: 42" in result

    def test_current_action_state_str_none(self):
        from roc.reporting.state import CurrentActionState

        s = CurrentActionState()
        assert str(s) == "Current Action: None"

    def test_current_transform_state_init(self):
        from roc.reporting.state import CurrentTransformState

        s = CurrentTransformState()
        assert s.name == "curr-transform"
        assert s.display_name == "Current Transform"
        assert s.val is None

    def test_current_transform_state_set_and_str(self):
        from roc.reporting.state import CurrentTransformState

        s = CurrentTransformState()
        mock_data = MagicMock()
        mock_data.transform.__str__ = MagicMock(return_value="Transform(changes=3)")
        s.set(mock_data)
        assert s.val is mock_data
        result = str(s)
        assert "Current Transform:" in result

    def test_current_transform_state_str_none(self):
        from roc.reporting.state import CurrentTransformState

        s = CurrentTransformState()
        assert str(s) == "Current Transform: None"

    def test_current_predict_state_init(self):
        from roc.reporting.state import CurrentPredictState

        s = CurrentPredictState()
        assert s.name == "curr-predict"
        assert s.display_name == "Current Prediction"
        assert s.val is None

    def test_current_predict_state_set_and_str(self):
        from roc.reporting.state import CurrentPredictState

        s = CurrentPredictState()
        mock_data = MagicMock()
        type(mock_data).__name__ = "MockPrediction"
        s.set(mock_data)
        assert s.val is mock_data
        result = str(s)
        assert "Current Prediction: MockPrediction" in result

    def test_current_predict_state_str_none(self):
        from roc.reporting.state import CurrentPredictState

        s = CurrentPredictState()
        assert str(s) == "Current Prediction: None"

    def test_current_message_state_init(self):
        from roc.reporting.state import CurrentMessageState

        s = CurrentMessageState()
        assert s.name == "curr-message"
        assert s.display_name == "Current Message"
        assert s.val is None

    def test_current_message_state_set_and_str(self):
        from roc.reporting.state import CurrentMessageState

        s = CurrentMessageState()
        s.set("You hit the goblin!")
        assert s.val == "You hit the goblin!"
        result = str(s)
        assert "Current Message: You hit the goblin!" in result

    def test_current_message_state_str_none(self):
        from roc.reporting.state import CurrentMessageState

        s = CurrentMessageState()
        assert str(s) == "Current Message: None"

    def test_current_resolution_state_init(self):
        from roc.reporting.state import CurrentResolutionState

        s = CurrentResolutionState()
        assert s.name == "curr-resolution"
        assert s.display_name == "Current Resolution"
        assert s.val is None

    def test_current_resolution_state_set_and_str(self):
        from roc.reporting.state import CurrentResolutionState

        s = CurrentResolutionState()
        data = {"outcome": "matched", "object_id": 5}
        s.set(data)
        assert s.val is data
        result = str(s)
        assert "Current Resolution: matched" in result

    def test_current_resolution_state_str_no_outcome(self):
        from roc.reporting.state import CurrentResolutionState

        s = CurrentResolutionState()
        s.set({"object_id": 5})
        result = str(s)
        assert "Current Resolution: unknown" in result

    def test_current_resolution_state_str_none(self):
        from roc.reporting.state import CurrentResolutionState

        s = CurrentResolutionState()
        assert str(s) == "Current Resolution: None"

    def test_current_attenuation_state_init(self):
        from roc.reporting.state import CurrentAttenuationState

        s = CurrentAttenuationState()
        assert s.name == "curr-attenuation"
        assert s.display_name == "Current Attenuation"
        assert s.val is None

    def test_current_attenuation_state_set_and_str(self):
        from roc.reporting.state import CurrentAttenuationState

        s = CurrentAttenuationState()
        data = {"flavor": "exponential", "factor": 0.9}
        s.set(data)
        assert s.val is data
        result = str(s)
        assert "Current Attenuation: exponential" in result

    def test_current_attenuation_state_str_no_flavor(self):
        from roc.reporting.state import CurrentAttenuationState

        s = CurrentAttenuationState()
        s.set({"factor": 0.9})
        result = str(s)
        assert "Current Attenuation: unknown" in result

    def test_current_attenuation_state_str_none(self):
        from roc.reporting.state import CurrentAttenuationState

        s = CurrentAttenuationState()
        assert str(s) == "Current Attenuation: None"


class TestStateListNewFields:
    """Verify the new state classes are present in StateList."""

    def test_statelist_has_new_fields(self):
        import dataclasses

        from roc.reporting.state import StateList

        sl = StateList()
        field_names = [f.name for f in dataclasses.fields(sl)]
        assert "intrinsic" in field_names
        assert "significance" in field_names
        assert "action" in field_names
        assert "transform" in field_names
        assert "predict" in field_names
        assert "message" in field_names
        assert "resolution" in field_names
        assert "attenuation_data" in field_names

    def test_statelist_new_fields_are_correct_types(self):
        from roc.reporting.state import (
            CurrentActionState,
            CurrentAttenuationState,
            CurrentIntrinsicState,
            CurrentMessageState,
            CurrentPredictState,
            CurrentResolutionState,
            CurrentSignificanceState,
            CurrentTransformState,
            StateList,
        )

        sl = StateList()
        assert isinstance(sl.intrinsic, CurrentIntrinsicState)
        assert isinstance(sl.significance, CurrentSignificanceState)
        assert isinstance(sl.action, CurrentActionState)
        assert isinstance(sl.transform, CurrentTransformState)
        assert isinstance(sl.predict, CurrentPredictState)
        assert isinstance(sl.message, CurrentMessageState)
        assert isinstance(sl.resolution, CurrentResolutionState)
        assert isinstance(sl.attenuation_data, CurrentAttenuationState)

    def test_get_state_names_includes_new_states(self):
        from roc.reporting.state import State

        names = State.get_state_names()
        assert "intrinsic" in names
        assert "significance" in names
        assert "action" in names
        assert "transform" in names
        assert "predict" in names
        assert "message" in names
        assert "resolution" in names
        assert "attenuation_data" in names
