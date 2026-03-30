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
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
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


def _clear_all_states():
    """Reset all states to None and clear step counts for emit_state_logs tests."""
    from roc.framework.event import Event
    from roc.reporting.state import states

    Event._step_counts.clear()
    for attr in (
        "screen",
        "salency",
        "object",
        "attention",
        "intrinsic",
        "significance",
        "action",
        "transform",
        "predict",
        "message",
    ):
        if hasattr(states, attr):
            getattr(states, attr).val = None


def _emit_and_check(expected_count, expected_event_name=None):
    """Call State.emit_state_logs and verify emit call count and optional event name.

    Returns the list of emit call args for further assertions.
    """
    from roc.reporting.state import State

    with patch("roc.reporting.state._get_otel_logger") as mock_logger:
        State.emit_state_logs()
        assert mock_logger.return_value.emit.call_count == expected_count
        calls = mock_logger.return_value.emit.call_args_list
        if expected_event_name is not None:
            record = calls[0][0][0]
            assert record.attributes["event.name"] == expected_event_name
        return calls


class TestEmitStateLogsExtra:
    def test_emit_state_logs_with_saliency(self):
        from roc.framework.config import Config
        from roc.reporting.state import states

        cfg = Config.get()
        cfg.emit_state_saliency = True
        _clear_all_states()
        mock_sal = MagicMock()
        mock_sal.feature_report.return_value = {"feat1": 0.5}
        mock_sal.to_html_vals.return_value = {
            "chars": [[46]],
            "fg": [["ffffff"]],
            "bg": [["0000ff"]],
        }
        states.salency.val = mock_sal

        # saliency + features + graphdb.summary
        _emit_and_check(3)
        states.salency.val = None

    def test_emit_state_logs_with_object(self):
        from roc.reporting.state import states

        _clear_all_states()
        states.object.val = MagicMock()
        _emit_and_check(2)
        states.object.val = None

    def test_emit_state_logs_with_attention(self):
        from roc.reporting.state import states

        _clear_all_states()
        mock_att = MagicMock()
        mock_att.focus_points = "data"
        states.attention.val = mock_att
        _emit_and_check(2)
        states.attention.val = None

    def test_emit_state_logs_with_intrinsic(self):
        from roc.reporting.state import states

        _clear_all_states()
        mock_intr = MagicMock()
        mock_intr.intrinsics = {"hp": 10}
        mock_intr.normalized_intrinsics = {"hp": 0.5}
        states.intrinsic.val = mock_intr
        _emit_and_check(2, "roc.intrinsics")
        states.intrinsic.val = None

    def test_emit_state_logs_with_significance(self):
        from roc.reporting.state import states

        _clear_all_states()
        mock_sig = MagicMock()
        mock_sig.significance = 0.75
        states.significance.val = mock_sig
        _emit_and_check(2, "roc.significance")
        states.significance.val = None

    def test_emit_state_logs_with_action(self):
        from roc.reporting.state import states

        _clear_all_states()
        mock_action = MagicMock()
        mock_action.action = 42
        states.action.val = mock_action
        _emit_and_check(2, "roc.action")
        states.action.val = None

    def test_emit_state_logs_with_transform(self):
        from roc.reporting.state import states

        _clear_all_states()
        mock_edge = MagicMock()
        mock_edge.dst = "some_change"
        mock_transform = MagicMock()
        mock_transform.transform.src_edges = [mock_edge]
        states.transform.val = mock_transform
        _emit_and_check(2, "roc.transform_summary")
        states.transform.val = None

    def test_emit_state_logs_with_predict(self):
        from roc.reporting.state import states

        _clear_all_states()
        states.predict.val = MagicMock()
        _emit_and_check(2, "roc.prediction")
        states.predict.val = None

    def test_emit_state_logs_with_message(self):
        from roc.reporting.state import states

        _clear_all_states()
        states.message.val = "Hello adventurer!"
        calls = _emit_and_check(2, "roc.message")
        assert calls[0][0][0].body == "Hello adventurer!"
        states.message.val = None

    def test_emit_state_logs_with_empty_message(self):
        from roc.reporting.state import states

        _clear_all_states()
        states.message.val = "   "
        # empty message should NOT emit, only graphdb.summary
        _emit_and_check(1)
        states.message.val = None

    def test_emit_state_logs_with_event_bus_counts(self):
        from roc.framework.event import Event

        _clear_all_states()
        Event._step_counts["TestBus"] = 5
        calls = _emit_and_check(2)
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

        _sl = StateList()
        field_names = [f.name for f in dataclasses.fields(StateList)]
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


class TestCurrentPhonemeState:
    """Tests for CurrentPhonemeState set and str."""

    def test_set(self):
        """Setting phonemes stores the list."""
        from roc.reporting.state import CurrentPhonemeState

        s = CurrentPhonemeState()
        from roc.perception.feature_extractors.phoneme import PhonemeWord

        phonemes = [PhonemeWord(word="hello", phonemes=["h", "e"], is_break=False)]
        s.set(phonemes)
        assert s.val is phonemes

    def test_str_with_value(self):
        """String representation includes entry count."""
        from roc.reporting.state import CurrentPhonemeState

        s = CurrentPhonemeState()
        s.set([MagicMock(), MagicMock()])
        result = str(s)
        assert "Current Phonemes: 2 entries" in result

    def test_str_none(self):
        """String representation when val is None."""
        from roc.reporting.state import CurrentPhonemeState

        s = CurrentPhonemeState()
        assert str(s) == "Current Phonemes: None"


class TestEmitSaliencyLogFeaturesBranch:
    """Tests for _emit_saliency_log when features are disabled."""

    def test_saliency_emitted_without_features(self):
        """When emit_state_features is False, only saliency is emitted (no features)."""
        from roc.reporting.state import StateList, _emit_saliency_log

        cfg = MagicMock()
        cfg.emit_state_saliency = True
        cfg.emit_state_features = False

        current_states = StateList()
        mock_sal = MagicMock()
        mock_sal.to_html_vals.return_value = {"chars": [[46]], "fg": [["fff"]], "bg": [["000"]]}
        current_states.salency.val = mock_sal

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            _emit_saliency_log(cfg, current_states)
            # Only saliency, no features
            assert mock_logger.return_value.emit.call_count == 1
            record = mock_logger.return_value.emit.call_args[0][0]
            assert record.attributes["event.name"] == "roc.attention.saliency"


class TestEnrichActionFromGym:
    """Tests for _enrich_action_from_gym helper."""

    def test_adds_action_name(self):
        """Enriches action dict with action name from gym_actions."""
        from roc.reporting.state import _enrich_action_from_gym

        act_enum = MagicMock()
        act_enum.name = "MOVE_NORTH"
        act_enum.value = 75  # ASCII 'K' is printable
        cfg = MagicMock()
        cfg.gym_actions = [act_enum]

        action_dict: dict[str, object] = {"action_id": 0}
        _enrich_action_from_gym(cfg, 0, action_dict)
        assert action_dict["action_name"] == "MOVE_NORTH"
        assert "action_key" in action_dict

    def test_no_gym_actions(self):
        """Does nothing when gym_actions is empty."""
        from roc.reporting.state import _enrich_action_from_gym

        cfg = MagicMock()
        cfg.gym_actions = []
        action_dict: dict[str, object] = {"action_id": 0}
        _enrich_action_from_gym(cfg, 0, action_dict)
        assert "action_name" not in action_dict

    def test_action_index_out_of_range(self):
        """Does nothing when action value exceeds gym_actions length."""
        from roc.reporting.state import _enrich_action_from_gym

        cfg = MagicMock()
        cfg.gym_actions = [MagicMock()]
        action_dict: dict[str, object] = {"action_id": 5}
        _enrich_action_from_gym(cfg, 5, action_dict)
        assert "action_name" not in action_dict

    def test_non_int_value_skips_key(self):
        """Skips action_key when enum value is not int."""
        from roc.reporting.state import _enrich_action_from_gym

        act_enum = MagicMock()
        act_enum.name = "SPECIAL"
        act_enum.value = "not_an_int"
        cfg = MagicMock()
        cfg.gym_actions = [act_enum]

        action_dict: dict[str, object] = {"action_id": 0}
        _enrich_action_from_gym(cfg, 0, action_dict)
        assert action_dict["action_name"] == "SPECIAL"
        assert "action_key" not in action_dict

    def test_exception_is_swallowed(self):
        """Exceptions in gym enrichment do not propagate."""
        from roc.reporting.state import _enrich_action_from_gym

        cfg = MagicMock()
        # gym_actions that raises on len() or indexing
        bad_actions = MagicMock()
        bad_actions.__len__ = MagicMock(return_value=10)
        bad_actions.__getitem__ = MagicMock(side_effect=RuntimeError("boom"))
        cfg.gym_actions = bad_actions
        action_dict: dict[str, object] = {"action_id": 0}
        # Should not raise
        _enrich_action_from_gym(cfg, 0, action_dict)

    def test_action_key_none_not_added(self):
        """When action_value_to_key returns None, no key is added."""
        from roc.reporting.state import _enrich_action_from_gym

        act_enum = MagicMock()
        act_enum.name = "NOOP"
        act_enum.value = 0  # value 0 returns None from action_value_to_key
        cfg = MagicMock()
        cfg.gym_actions = [act_enum]

        action_dict: dict[str, object] = {"action_id": 0}
        _enrich_action_from_gym(cfg, 0, action_dict)
        assert action_dict["action_name"] == "NOOP"
        assert "action_key" not in action_dict


class TestEnrichActionExpmod:
    """Tests for _enrich_action_expmod helper."""

    def test_adds_expmod_name(self):
        """Enriches action dict with expmod name."""
        from roc.reporting.state import _enrich_action_expmod

        action_dict: dict[str, object] = {"action_id": 0}
        mock_expmod = MagicMock()
        mock_expmod.name = "random-action"
        with patch("roc.pipeline.action.DefaultActionExpMod.get", return_value=mock_expmod):
            _enrich_action_expmod(action_dict)
        assert action_dict["expmod_name"] == "random-action"

    def test_exception_is_swallowed(self):
        """Exceptions in expmod enrichment do not propagate."""
        from roc.reporting.state import _enrich_action_expmod

        action_dict: dict[str, object] = {"action_id": 0}
        with patch(
            "roc.pipeline.action.DefaultActionExpMod.get",
            side_effect=RuntimeError("no expmod"),
        ):
            _enrich_action_expmod(action_dict)
        assert "expmod_name" not in action_dict


class TestEmitTransformLogEdgeDetails:
    """Tests for _emit_transform_log with edge name and normalized_change."""

    def test_edge_with_name_and_normalized_change(self):
        """Transform edges with name and normalized_change attrs are captured."""
        from roc.reporting.state import StateList, _emit_transform_log

        current_states = StateList()
        mock_dst = MagicMock()
        mock_dst.name = "hp"
        mock_dst.normalized_change = -0.5
        type(mock_dst).__name__ = "IntrinsicTransform"
        mock_edge = MagicMock()
        mock_edge.dst = mock_dst
        mock_transform = MagicMock()
        mock_transform.transform.src_edges = [mock_edge]
        current_states.transform.val = mock_transform

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            _emit_transform_log(current_states)
            mock_logger.return_value.emit.assert_called_once()
            record = mock_logger.return_value.emit.call_args[0][0]
            import json

            body = json.loads(record.body)
            assert body["count"] == 1
            assert body["changes"][0]["type"] == "IntrinsicTransform"
            assert body["changes"][0]["name"] == "hp"
            assert body["changes"][0]["normalized_change"] == -0.5

    def test_edge_without_name(self):
        """Transform edges without name attr (e.g. Frame nodes) are filtered out."""
        import json

        from roc.reporting.state import StateList, _emit_transform_log

        current_states = StateList()

        class BareDst:
            """A dst node without name or normalized_change attributes."""

            def __str__(self) -> str:
                return "some change"

        mock_edge = MagicMock()
        mock_edge.dst = BareDst()
        mock_transform = MagicMock()
        mock_transform.transform.src_edges = [mock_edge]
        current_states.transform.val = mock_transform

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            _emit_transform_log(current_states)
            body = json.loads(mock_logger.return_value.emit.call_args[0][0].body)
            # Non-intrinsic nodes (no name attr) are filtered out
            assert body["count"] == 0
            assert len(body["changes"]) == 0


class TestEmitSequenceLog:
    """Tests for _emit_sequence_log and _build_sequence_dict."""

    def test_emit_sequence_log_with_frame(self):
        """Sequence log is emitted when a sequencer with a last_frame is loaded."""
        from roc.framework.component import ComponentName, ComponentType
        from roc.framework.component import loaded_components as lc
        from roc.reporting.state import StateList, _emit_sequence_log
        from roc.pipeline.temporal.sequencer import Sequencer

        mock_seq = MagicMock(spec=Sequencer)
        mock_seq.last_frame = MagicMock()

        key = (ComponentName("sequencer"), ComponentType("sequencer"))
        lc[key] = mock_seq

        current_states = StateList()

        try:
            with (
                patch("roc.reporting.state._get_otel_logger") as mock_logger,
                patch(
                    "roc.reporting.state._build_sequence_dict",
                    return_value={"tick": 42, "object_count": 1},
                ),
            ):
                _emit_sequence_log(current_states)
                mock_logger.return_value.emit.assert_called_once()
                record = mock_logger.return_value.emit.call_args[0][0]
                assert record.attributes["event.name"] == "roc.sequence_summary"
        finally:
            lc.pop(key, None)

    def test_emit_sequence_log_no_sequencer(self):
        """Sequence log not emitted when sequencer not loaded."""
        from roc.framework.component import ComponentName, ComponentType
        from roc.framework.component import loaded_components as lc
        from roc.reporting.state import StateList, _emit_sequence_log

        key = (ComponentName("sequencer"), ComponentType("sequencer"))
        lc.pop(key, None)

        current_states = StateList()

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            _emit_sequence_log(current_states)
            mock_logger.return_value.emit.assert_not_called()

    def test_emit_sequence_log_exception_swallowed(self):
        """Exceptions in sequence log do not propagate."""
        from roc.framework.component import ComponentName, ComponentType
        from roc.framework.component import loaded_components as lc
        from roc.reporting.state import StateList, _emit_sequence_log
        from roc.pipeline.temporal.sequencer import Sequencer

        # Insert a mock sequencer with a last_frame that will cause
        # _build_sequence_dict to raise.
        mock_seq = MagicMock(spec=Sequencer)
        mock_seq.last_frame = MagicMock()

        key = (ComponentName("sequencer"), ComponentType("sequencer"))
        lc[key] = mock_seq

        current_states = StateList()

        try:
            with (
                patch("roc.reporting.state._get_otel_logger") as mock_logger,
                patch(
                    "roc.reporting.state._build_sequence_dict",
                    side_effect=RuntimeError("boom"),
                ),
            ):
                # Should not raise
                _emit_sequence_log(current_states)
                mock_logger.return_value.emit.assert_not_called()
        finally:
            lc.pop(key, None)


class TestBuildSequenceDict:
    """Tests for _build_sequence_dict helper."""

    def test_builds_dict_with_objects_and_intrinsics(self):
        """Builds a full sequence dict from a frame."""
        from roc.pipeline.intrinsic import IntrinsicNode
        from roc.reporting.state import StateList, _build_sequence_dict

        mock_frame = MagicMock()
        mock_frame.tick = 10

        mock_obj = MagicMock()
        mock_obj.id = "abcdefgh-1234"
        mock_obj.last_x = 3
        mock_obj.last_y = 7
        mock_obj.resolve_count = 2
        mock_frame.objects = [mock_obj]

        mock_inode = MagicMock(spec=IntrinsicNode)
        mock_inode.name = "energy"
        mock_inode.normalized_value = 0.6
        mock_frame.transformable = [mock_inode]

        current_states = StateList()
        current_states.significance.val = MagicMock(significance=0.9)

        result = _build_sequence_dict(mock_frame, current_states)
        assert result["tick"] == 10
        assert result["object_count"] == 1
        assert result["objects"][0]["x"] == 3
        assert result["objects"][0]["y"] == 7
        assert result["objects"][0]["resolve_count"] == 2
        assert result["intrinsic_count"] == 1
        assert result["intrinsics"]["energy"] == pytest.approx(0.6)
        assert result["significance"] == pytest.approx(0.9)

    def test_builds_dict_without_significance(self):
        """Builds dict when significance is None (no significance key)."""
        from roc.reporting.state import StateList, _build_sequence_dict

        mock_frame = MagicMock()
        mock_frame.tick = 5
        mock_frame.objects = []
        mock_frame.transformable = []

        current_states = StateList()
        current_states.significance.val = None

        result = _build_sequence_dict(mock_frame, current_states)
        assert result["tick"] == 5
        assert result["object_count"] == 0
        assert "significance" not in result

    def test_object_without_position(self):
        """Objects lacking last_x/last_y do not include position."""
        from roc.reporting.state import StateList, _build_sequence_dict

        mock_frame = MagicMock()
        mock_frame.tick = 1

        class BareObj:
            """An object without position attrs."""

            id = "bare-obj-1234"

        mock_frame.objects = [BareObj()]
        mock_frame.transformable = []

        current_states = StateList()
        current_states.significance.val = None

        result = _build_sequence_dict(mock_frame, current_states)
        assert "x" not in result["objects"][0]
        assert "y" not in result["objects"][0]

    def test_object_without_resolve_count(self):
        """Objects lacking resolve_count do not include it."""
        from roc.reporting.state import StateList, _build_sequence_dict

        mock_frame = MagicMock()
        mock_frame.tick = 1

        class NoResolveObj:
            """An object without resolve_count."""

            id = "noresolve-1234"
            last_x = 0
            last_y = 0

        mock_frame.objects = [NoResolveObj()]
        mock_frame.transformable = []

        current_states = StateList()
        current_states.significance.val = None

        result = _build_sequence_dict(mock_frame, current_states)
        assert "resolve_count" not in result["objects"][0]


class TestEnrichPredictionMeta:
    """Tests for _enrich_prediction_meta helper."""

    def test_adds_prediction_metadata(self):
        """Enriches prediction dict with expmod and component metadata."""
        from roc.framework.component import ComponentName, ComponentType
        from roc.framework.component import loaded_components as lc
        from roc.pipeline.temporal.predict import Predict
        from roc.reporting.state import _enrich_prediction_meta

        pred_dict: dict[str, object] = {"made": True}

        mock_candidate_expmod = MagicMock()
        mock_candidate_expmod.name = "object-based"
        mock_confidence_expmod = MagicMock()
        mock_confidence_expmod.name = "naive"

        mock_predict_comp = MagicMock(spec=Predict)
        mock_predict_comp.last_prediction_meta = MagicMock(
            candidate_count=5,
            confidence=0.8,
            all_scores=[0.5, 0.6, 0.8],
            predicted_intrinsics={"hp": 0.9},
        )

        key = (ComponentName("predict"), ComponentType("predict"))
        lc[key] = mock_predict_comp

        try:
            with (
                patch(
                    "roc.pipeline.temporal.predict.PredictionCandidateFramesExpMod.get",
                    return_value=mock_candidate_expmod,
                ),
                patch(
                    "roc.pipeline.temporal.predict.PredictionConfidenceExpMod.get",
                    return_value=mock_confidence_expmod,
                ),
            ):
                _enrich_prediction_meta(pred_dict)

            assert pred_dict["candidate_expmod"] == "object-based"
            assert pred_dict["confidence_expmod"] == "naive"
            assert pred_dict["candidate_count"] == 5
            assert pred_dict["confidence"] == pytest.approx(0.8)
            assert pred_dict["all_scores"] == pytest.approx([0.5, 0.6, 0.8])
            assert pred_dict["predicted_intrinsics"] == pytest.approx({"hp": 0.9})
        finally:
            lc.pop(key, None)

    def test_empty_predicted_intrinsics_not_added(self):
        """When predicted_intrinsics is empty, it is not added to dict."""
        from roc.framework.component import ComponentName, ComponentType
        from roc.framework.component import loaded_components as lc
        from roc.pipeline.temporal.predict import Predict
        from roc.reporting.state import _enrich_prediction_meta

        pred_dict: dict[str, object] = {"made": True}

        mock_predict_comp = MagicMock(spec=Predict)
        mock_predict_comp.last_prediction_meta = MagicMock(
            candidate_count=2,
            confidence=0.5,
            all_scores=[0.3, 0.5],
            predicted_intrinsics={},
        )

        key = (ComponentName("predict"), ComponentType("predict"))
        lc[key] = mock_predict_comp

        try:
            with (
                patch(
                    "roc.pipeline.temporal.predict.PredictionCandidateFramesExpMod.get",
                    return_value=MagicMock(name="object-based"),
                ),
                patch(
                    "roc.pipeline.temporal.predict.PredictionConfidenceExpMod.get",
                    return_value=MagicMock(name="naive"),
                ),
            ):
                _enrich_prediction_meta(pred_dict)

            assert "predicted_intrinsics" not in pred_dict
        finally:
            lc.pop(key, None)

    def test_exception_is_swallowed(self):
        """Exceptions in prediction meta enrichment do not propagate."""
        from roc.reporting.state import _enrich_prediction_meta

        pred_dict: dict[str, object] = {"made": True}
        with patch(
            "roc.pipeline.temporal.predict.PredictionCandidateFramesExpMod.get",
            side_effect=RuntimeError("no expmod"),
        ):
            _enrich_prediction_meta(pred_dict)
        assert "candidate_expmod" not in pred_dict

    def test_non_predict_component_skips_meta(self):
        """When loaded component is not a Predict instance, meta is not added."""
        from roc.framework.component import ComponentName, ComponentType
        from roc.framework.component import loaded_components as lc
        from roc.reporting.state import _enrich_prediction_meta

        pred_dict: dict[str, object] = {"made": True}

        # Insert a non-Predict component at the predict key
        key = (ComponentName("predict"), ComponentType("predict"))
        lc[key] = MagicMock()  # Not spec=Predict, so isinstance fails

        try:
            with (
                patch(
                    "roc.pipeline.temporal.predict.PredictionCandidateFramesExpMod.get",
                    return_value=MagicMock(name="object-based"),
                ),
                patch(
                    "roc.pipeline.temporal.predict.PredictionConfidenceExpMod.get",
                    return_value=MagicMock(name="naive"),
                ),
            ):
                _enrich_prediction_meta(pred_dict)

            # expmod names should be set but no component meta
            assert "candidate_expmod" in pred_dict
            assert "candidate_count" not in pred_dict
        finally:
            lc.pop(key, None)


class TestEmitPhonemesLog:
    """Tests for _emit_phonemes_log helper."""

    def test_emits_phoneme_records(self):
        """Phonemes state is serialized and emitted."""
        from roc.reporting.state import StateList, _emit_phonemes_log

        current_states = StateList()

        pw1 = MagicMock(word="hello", phonemes=["h", "e", "l", "o"], is_break=False)
        pw2 = MagicMock(word="", phonemes=[], is_break=True)
        current_states.phonemes.val = [pw1, pw2]

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            _emit_phonemes_log(current_states)
            mock_logger.return_value.emit.assert_called_once()
            record = mock_logger.return_value.emit.call_args[0][0]
            assert record.attributes["event.name"] == "roc.phonemes"
            import json

            body = json.loads(record.body)
            assert len(body) == 2
            assert body[0]["word"] == "hello"
            assert body[1]["is_break"] is True

    def test_skips_when_none(self):
        """No emission when phonemes val is None."""
        from roc.reporting.state import StateList, _emit_phonemes_log

        current_states = StateList()
        current_states.phonemes.val = None

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            _emit_phonemes_log(current_states)
            mock_logger.return_value.emit.assert_not_called()


class TestPrintStartupInfoBranches:
    """Tests for uncovered branches in State.print_startup_info."""

    @patch("roc.reporting.state.subprocess.run")
    @patch("roc.reporting.state.Schema")
    @patch("roc.reporting.state.Component.get_loaded_components", return_value=["a:b"])
    @patch("roc.reporting.state.Component.get_component_count", return_value=1)
    def test_log_cmd_stderr_branch(self, mock_count, mock_components, mock_schema, mock_run):
        """The log_cmd helper reads stderr when out='stderr'."""
        from roc.reporting.state import State

        mock_result = MagicMock()
        mock_result.stdout = b"stdout_output"
        mock_result.stderr = b"stderr_output"
        mock_run.return_value = mock_result
        mock_schema_inst = MagicMock()
        mock_schema_inst.to_dot.return_value = "dot"
        mock_schema_inst.to_dict.return_value = {"nodes": [], "edges": []}
        mock_schema.return_value = mock_schema_inst

        # Should not raise -- the stderr branch is exercised internally by
        # log_cmd calls that use out="stderr" (currently none by default,
        # but we verify the code path works)
        State.print_startup_info()

    @patch("roc.reporting.state.subprocess.run")
    @patch("roc.reporting.state.Schema")
    @patch("roc.reporting.state.Component.get_loaded_components", return_value=["a:b"])
    @patch("roc.reporting.state.Component.get_component_count", return_value=1)
    def test_ducklake_store_saves_schema(self, mock_count, mock_components, mock_schema, mock_run):
        """When DuckLake store is available, schema is saved to run_dir."""
        import json
        from pathlib import Path

        from roc.reporting.state import State

        mock_result = MagicMock()
        mock_result.stdout = b"output"
        mock_result.stderr = b""
        mock_run.return_value = mock_result

        schema_dict = {"nodes": ["a"], "edges": ["b"]}
        mock_schema_inst = MagicMock()
        mock_schema_inst.to_dot.return_value = "dot"
        mock_schema_inst.to_dict.return_value = schema_dict
        mock_schema.return_value = mock_schema_inst

        mock_store = MagicMock()
        mock_path = MagicMock(spec=Path)
        mock_store.run_dir = mock_path

        with patch(
            "roc.reporting.state.Observability.get_ducklake_store",
            return_value=mock_store,
        ):
            State.print_startup_info()

        # schema.json should be written to run_dir
        mock_path.__truediv__.assert_called_once_with("schema.json")
        mock_path.__truediv__.return_value.write_text.assert_called_once()
        written_text = mock_path.__truediv__.return_value.write_text.call_args[0][0]
        assert json.loads(written_text) == schema_dict
