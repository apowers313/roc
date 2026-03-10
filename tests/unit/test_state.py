# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/state.py."""

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
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


class TestState:
    def test_init(self):
        from roc.reporting.state import State

        class ConcreteState(State[int]):
            pass

        s = ConcreteState("test_name")
        assert s.name == "test_name"
        assert s.display_name == "test_name"
        assert s.val is None

    def test_init_with_display_name(self):
        from roc.reporting.state import State

        class ConcreteState(State[int]):
            pass

        s = ConcreteState("test", display_name="Test Display")
        assert s.display_name == "Test Display"

    def test_str(self):
        from roc.reporting.state import State

        class ConcreteState(State[int]):
            pass

        s = ConcreteState("test", display_name="My State")
        s.val = 42
        assert str(s) == "My State: 42"

    def test_get_raises_when_none(self):
        from roc.reporting.state import State

        class ConcreteState(State[int]):
            pass

        s = ConcreteState("test")
        with pytest.raises(Exception, match="Trying to get state value before it is set"):
            s.get()

    def test_get_returns_value(self):
        from roc.reporting.state import State

        class ConcreteState(State[int]):
            pass

        s = ConcreteState("test")
        s.val = 42
        assert s.get() == 42

    def test_set(self):
        from roc.reporting.state import State

        class ConcreteState(State[int]):
            pass

        s = ConcreteState("test")
        s.set(99)
        assert s.val == 99

    def test_get_states(self):
        from roc.reporting.state import State, StateList

        result = State.get_states()
        assert isinstance(result, StateList)

    def test_get_state_names(self):
        from roc.reporting.state import State

        names = State.get_state_names()
        assert isinstance(names, list)
        assert "loop" in names
        assert "node_cache" in names
        assert "edge_cache" in names
        assert "screen" in names


class TestLoopState:
    def test_init(self):
        from roc.reporting.state import LoopState

        ls = LoopState()
        assert ls.val == 0
        assert ls.name == "loop"
        assert ls.display_name == "Loop Number"

    def test_incr(self):
        from roc.reporting.state import LoopState

        ls = LoopState()
        assert ls.val == 0
        ls.incr()
        assert ls.val == 1
        ls.incr()
        assert ls.val == 2


class TestNodeCacheState:
    def test_get(self):
        from roc.reporting.state import NodeCacheState

        mock_cache = MagicMock()
        mock_cache.currsize = 10
        mock_cache.maxsize = 100
        with patch("roc.reporting.state.Node.get_cache", return_value=mock_cache):
            ncs = NodeCacheState()
            assert ncs.get() == 0.1

    def test_str(self):
        from roc.reporting.state import NodeCacheState

        mock_cache = MagicMock()
        mock_cache.currsize = 10
        mock_cache.maxsize = 100
        with patch("roc.reporting.state.Node.get_cache", return_value=mock_cache):
            ncs = NodeCacheState()
            result = str(ncs)
            assert "Node Cache: 10 / 100" in result


class TestEdgeCacheState:
    def test_get(self):
        from roc.reporting.state import EdgeCacheState

        mock_cache = MagicMock()
        mock_cache.currsize = 5
        mock_cache.maxsize = 50
        with patch("roc.reporting.state.Edge.get_cache", return_value=mock_cache):
            ecs = EdgeCacheState()
            assert ecs.get() == 0.1

    def test_str(self):
        from roc.reporting.state import EdgeCacheState

        mock_cache = MagicMock()
        mock_cache.currsize = 5
        mock_cache.maxsize = 50
        with patch("roc.reporting.state.Edge.get_cache", return_value=mock_cache):
            ecs = EdgeCacheState()
            result = str(ecs)
            assert "Edge Cache: 5 / 50" in result


class TestCurrentScreenState:
    def test_str_none(self):
        from roc.reporting.state import CurrentScreenState

        css = CurrentScreenState()
        assert str(css) == "Current Screen: None"

    def test_str_with_value(self):
        from roc.reporting.state import CurrentScreenState

        css = CurrentScreenState()
        mock_screen = {"chars": "abc", "colors": "def", "cursor": (0, 0)}
        css.set(mock_screen)
        with patch("nle.nethack.tty_render", return_value="rendered_screen"):
            result = str(css)
            assert "rendered_screen" in result
            assert "Current Screen:" in result


class TestCurrentSaliencyMapState:
    def test_str_none(self):
        from roc.reporting.state import CurrentSaliencyMapState

        csms = CurrentSaliencyMapState()
        assert str(csms) == "Current Saliency Map: None"

    def test_str_with_value(self):
        from roc.reporting.state import CurrentSaliencyMapState

        csms = CurrentSaliencyMapState()
        mock_sal = MagicMock()
        mock_sal.feature_report.return_value = {"feat1": 0.5}
        csms.set(mock_sal)
        result = str(csms)
        assert "Current Saliency Map:" in result
        assert "feat1: 0.5" in result


class TestCurrentAttentionState:
    def test_str_none(self):
        from roc.reporting.state import CurrentAttentionState

        cas = CurrentAttentionState()
        assert str(cas) == "Current Attention: None"

    def test_str_with_value(self):
        from roc.reporting.state import CurrentAttentionState

        cas = CurrentAttentionState()
        mock_att = MagicMock(spec=str)
        cas.set(mock_att)
        result = str(cas)
        assert "Current Attention:" in result


class TestCurrentObjectState:
    def test_str_none(self):
        from roc.reporting.state import CurrentObjectState

        cos = CurrentObjectState()
        assert str(cos) == "Current Object: None"

    def test_str_with_value(self):
        from roc.reporting.state import CurrentObjectState

        cos = CurrentObjectState()
        mock_obj = MagicMock(spec=str)
        cos.set(mock_obj)
        result = str(cos)
        assert "Current Object:" in result


class TestComponentsState:
    def test_get(self):
        from roc.reporting.state import ComponentsState

        cs = ComponentsState()
        with patch(
            "roc.reporting.state.Component.get_loaded_components", return_value=["a:b", "c:d"]
        ):
            result = cs.get()
            assert result == ["a:b", "c:d"]

    def test_str(self):
        from roc.reporting.state import ComponentsState

        cs = ComponentsState()
        with patch("roc.reporting.state.Component.get_loaded_components", return_value=["a:b"]):
            with patch("roc.reporting.state.Component.get_component_count", return_value=1):
                result = str(cs)
                assert "1 components loaded" in result
                assert "a:b" in result


class TestBlstatsState:
    def test_instantiate(self):
        from roc.reporting.state import BlstatsState

        bs = BlstatsState("blstats")
        assert bs.name == "blstats"


class TestStateList:
    def test_default_fields(self):
        from roc.reporting.state import StateList

        sl = StateList()
        fields = [f.name for f in dataclasses.fields(sl)]
        assert "loop" in fields
        assert "node_cache" in fields
        assert "edge_cache" in fields
        assert "screen" in fields
        assert "salency" in fields
        assert "attention" in fields
        assert "object" in fields
        assert "components" in fields


class TestBytes2Human:
    def test_zero(self):
        from roc.reporting.state import bytes2human

        assert bytes2human(0) == "0B"

    def test_kilobyte(self):
        from roc.reporting.state import bytes2human

        assert bytes2human(1024) == "1.0KB"

    def test_megabyte(self):
        from roc.reporting.state import bytes2human

        assert bytes2human(1048576) == "1.0MB"

    def test_gigabyte(self):
        from roc.reporting.state import bytes2human

        result = bytes2human(1073741824)
        assert result == "1.0GB"

    def test_small_value(self):
        from roc.reporting.state import bytes2human

        assert bytes2human(500) == "500B"


class TestEmitStateRecord:
    def test_emit_state_record_creates_log_record(self):
        from roc.reporting.state import _emit_state_record

        with patch("roc.reporting.state.otel_logger") as mock_logger:
            _emit_state_record("roc.screen", "test body")
            mock_logger.emit.assert_called_once()
            log_record = mock_logger.emit.call_args[0][0]
            assert log_record.body == "test body"
            assert log_record.attributes["event.name"] == "roc.screen"

    def test_emit_state_record_has_severity_info(self):
        from opentelemetry._logs import SeverityNumber

        from roc.reporting.state import _emit_state_record

        with patch("roc.reporting.state.otel_logger") as mock_logger:
            _emit_state_record("roc.test", "body")
            log_record = mock_logger.emit.call_args[0][0]
            assert log_record.severity_number == SeverityNumber.INFO
            assert log_record.severity_text == "INFO"


class TestStatePrintStartupInfo:
    @patch("roc.reporting.state.subprocess.run")
    @patch("roc.reporting.state.Schema")
    @patch("roc.reporting.state.Component.get_loaded_components", return_value=["a:b"])
    @patch("roc.reporting.state.Component.get_component_count", return_value=1)
    def test_print_startup_info(self, mock_count, mock_components, mock_schema, mock_run):
        from roc.reporting.state import State

        mock_result = MagicMock()
        mock_result.stdout = b"output"
        mock_result.stderr = b""
        mock_run.return_value = mock_result
        mock_schema_inst = MagicMock()
        mock_schema_inst.to_dot.return_value = "dot"
        mock_schema.return_value = mock_schema_inst

        # Should not raise
        State.print_startup_info()


class TestEmitStateLogs:
    def test_emit_state_logs_all_none(self):
        from roc.reporting.state import State, states

        states.screen.val = None
        states.salency.val = None
        states.object.val = None
        states.attention.val = None

        with patch("roc.reporting.state.otel_logger") as mock_logger:
            State.emit_state_logs()
            mock_logger.emit.assert_not_called()

    def test_emit_state_logs_with_screen(self):
        from roc.reporting.state import State, states

        states.screen.val = {"chars": np.array([[65, 66]])}
        states.salency.val = None
        states.object.val = None
        states.attention.val = None

        with patch("roc.reporting.state.otel_logger") as mock_logger:
            State.emit_state_logs()
            assert mock_logger.emit.call_count == 1
            log_record = mock_logger.emit.call_args[0][0]
            assert log_record.attributes["event.name"] == "roc.screen"
            assert "AB" in log_record.body
        states.screen.val = None
