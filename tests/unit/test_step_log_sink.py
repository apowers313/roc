"""Tests for the step_log_sink ring buffer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from roc.reporting.step_log_sink import (
    _buffer,
    _lock,
    drain_step_logs,
    set_current_step,
    step_log_sink,
)


@pytest.fixture(autouse=True)
def _clear_buffer() -> Any:
    """Reset the sink state between tests."""
    import roc.reporting.step_log_sink as mod

    with _lock:
        _buffer.clear()
    mod._current_step = 0
    yield
    with _lock:
        _buffer.clear()
    mod._current_step = 0


def _make_message(text: str, level: str = "INFO") -> MagicMock:
    """Build a fake loguru message object with a .record attribute."""
    from datetime import datetime, timezone

    msg = MagicMock(spec=str)
    msg.strip.return_value = text
    msg.record = {
        "time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "level": MagicMock(name=level),
        "module": "test_mod",
        "function": "test_fn",
        "line": 42,
    }
    msg.record["level"].name = level
    return msg


class TestStepLogSink:
    def test_captures_log_for_current_step(self) -> None:
        set_current_step(5)
        step_log_sink(_make_message("hello"))

        logs = drain_step_logs(5)
        assert logs is not None
        assert len(logs) == 1
        assert logs[0]["step"] == 5
        assert logs[0]["body"] == "hello"
        assert logs[0]["severity_text"] == "INFO"
        assert logs[0]["module"] == "test_mod"
        assert logs[0]["function"] == "test_fn"
        assert logs[0]["line"] == 42

    def test_multiple_logs_per_step(self) -> None:
        set_current_step(3)
        step_log_sink(_make_message("first"))
        step_log_sink(_make_message("second"))
        step_log_sink(_make_message("third"))

        logs = drain_step_logs(3)
        assert logs is not None
        assert len(logs) == 3
        assert [l["body"] for l in logs] == ["first", "second", "third"]

    def test_drain_returns_none_for_empty_step(self) -> None:
        set_current_step(1)
        step_log_sink(_make_message("log for step 1"))

        result = drain_step_logs(99)
        assert result is None

    def test_drain_filters_by_step(self) -> None:
        set_current_step(1)
        step_log_sink(_make_message("step 1 log"))

        set_current_step(2)
        step_log_sink(_make_message("step 2 log"))

        logs = drain_step_logs(1)
        assert logs is not None
        assert len(logs) == 1
        assert logs[0]["body"] == "step 1 log"

        logs = drain_step_logs(2)
        assert logs is not None
        assert len(logs) == 1
        assert logs[0]["body"] == "step 2 log"

    def test_timestamp_is_epoch_ms(self) -> None:
        set_current_step(0)
        step_log_sink(_make_message("ts test"))

        logs = drain_step_logs(0)
        assert logs is not None
        # 2026-01-01T00:00:00Z in epoch ms
        assert logs[0]["timestamp"] == 1767225600000

    def test_ring_buffer_evicts_old_entries(self) -> None:
        import roc.reporting.step_log_sink as mod

        # Temporarily shrink the buffer to test eviction
        old_buffer = mod._buffer
        mod._buffer = type(old_buffer)(maxlen=5)
        try:
            set_current_step(1)
            for i in range(10):
                step_log_sink(_make_message(f"msg-{i}"))

            # Only last 5 should remain
            with mod._lock:
                assert len(mod._buffer) == 5

            logs = drain_step_logs(1)
            assert logs is not None
            assert len(logs) == 5
            assert logs[0]["body"] == "msg-5"
            assert logs[-1]["body"] == "msg-9"
        finally:
            mod._buffer = old_buffer

    def test_drain_with_no_logs_at_all(self) -> None:
        result = drain_step_logs(0)
        assert result is None

    def test_gymnasium_step_numbering_pattern(self) -> None:
        """Regression: gymnasium increments loop_num between set_current_step
        and drain_step_logs.  set_current_step must use loop_num+1 so that
        drain_step_logs(loop_num) matches after the increment."""
        loop_num = 0

        # Simulate gymnasium loop: set step BEFORE processing, drain AFTER increment
        set_current_step(loop_num + 1)  # tags logs as step 1
        step_log_sink(_make_message("processing log"))
        loop_num += 1  # now 1

        logs = drain_step_logs(loop_num)  # drain step 1
        assert logs is not None, (
            "drain_step_logs must find logs when set_current_step uses loop_num+1"
        )
        assert len(logs) == 1
        assert logs[0]["step"] == 1
