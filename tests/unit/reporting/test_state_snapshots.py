# mypy: disable-error-code="no-untyped-def"

"""Unit tests for tick-level state snapshots in roc/reporting/state.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roc.framework.config import Config


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    mock.node_counter = MagicMock()
    mock.edge_counter = MagicMock()
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestSnapshotConfig:
    def test_debug_snapshot_interval_default_zero(self):
        assert Config.get().debug_snapshot_interval == 0


class TestSnapshotEmission:
    def test_snapshot_emitted_at_interval(self):
        """Snapshot is emitted every N ticks when interval > 0."""
        from roc.reporting.state import State

        settings = Config.get()
        settings.debug_snapshot_interval = 5

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            for i in range(1, 11):
                State.maybe_emit_snapshot(i)
            # Should have 2 snapshots (tick 5 and tick 10)
            assert mock_logger.return_value.emit.call_count == 2

    def test_snapshot_not_emitted_when_disabled(self):
        """No snapshots when interval is 0."""
        from roc.reporting.state import State

        settings = Config.get()
        settings.debug_snapshot_interval = 0

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            for i in range(1, 11):
                State.maybe_emit_snapshot(i)
            mock_logger.return_value.emit.assert_not_called()

    def test_snapshot_not_emitted_on_non_interval_ticks(self):
        """No snapshot on ticks that aren't multiples of the interval."""
        from roc.reporting.state import State

        settings = Config.get()
        settings.debug_snapshot_interval = 5

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            for i in [1, 2, 3, 4, 6, 7, 8, 9]:
                State.maybe_emit_snapshot(i)
            mock_logger.return_value.emit.assert_not_called()

    def test_snapshot_emitted_on_interval_tick(self):
        """Snapshot emitted exactly on interval tick."""
        from roc.reporting.state import State

        settings = Config.get()
        settings.debug_snapshot_interval = 3

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.maybe_emit_snapshot(3)
            mock_logger.return_value.emit.assert_called_once()

    def test_snapshot_contains_expected_fields(self):
        """Snapshot record should include screen, objects, and tick info."""
        from roc.reporting.state import State, states

        settings = Config.get()
        settings.debug_snapshot_interval = 1

        # Set up some state
        states.screen.val = {"chars": np.array([[65, 66], [67, 68]])}
        mock_obj = MagicMock()
        mock_obj.configure_mock(**{"__str__": MagicMock(return_value="test object")})
        states.object.val = mock_obj

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.maybe_emit_snapshot(1)
            mock_logger.return_value.emit.assert_called_once()

            log_record = mock_logger.return_value.emit.call_args[0][0]
            body = log_record.body
            assert "tick" in body
            assert "screen" in body

        # Clean up
        states.screen.val = None
        states.object.val = None

    def test_snapshot_with_no_state_still_emits(self):
        """Snapshot emits even when state values are None."""
        from roc.reporting.state import State, states

        settings = Config.get()
        settings.debug_snapshot_interval = 1

        states.screen.val = None
        states.object.val = None

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.maybe_emit_snapshot(1)
            mock_logger.return_value.emit.assert_called_once()

            log_record = mock_logger.return_value.emit.call_args[0][0]
            body = log_record.body
            assert "tick" in body

    def test_snapshot_has_correct_event_name(self):
        """Snapshot record should have event name in attributes."""
        from roc.reporting.state import State

        settings = Config.get()
        settings.debug_snapshot_interval = 1

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.maybe_emit_snapshot(1)
            log_record = mock_logger.return_value.emit.call_args[0][0]
            assert log_record.attributes is not None
            assert log_record.attributes.get("event.name") == "roc.state.snapshot"

    def test_snapshot_tick_zero_not_emitted(self):
        """Tick 0 should not trigger a snapshot (avoid division edge case)."""
        from roc.reporting.state import State

        settings = Config.get()
        settings.debug_snapshot_interval = 5

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            State.maybe_emit_snapshot(0)
            mock_logger.return_value.emit.assert_not_called()
