# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/game_manager.py (thread-only mode)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from roc.game.game_manager import GameManager, _THREAD_JOIN_TIMEOUT


class TestInitialState:
    def test_state_is_idle(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm.state == "idle"

    def test_no_run_name(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._current_run_name is None

    def test_no_game_thread_initially(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._game_thread is None

    def test_no_error_message(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._error_message is None

    def test_stop_event_exists(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert isinstance(gm._stop_event, threading.Event)
        assert not gm._stop_event.is_set()


class TestGetStatus:
    def test_idle_status_shape(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        status = gm.get_status()
        assert status == {"state": "idle", "run_name": None}

    def test_includes_error_when_set(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._error_message = "something broke"
        status = gm.get_status()
        assert status["error"] == "something broke"

    def test_excludes_error_when_none(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        status = gm.get_status()
        assert "error" not in status


class TestSetState:
    def test_calls_on_state_change_callback(self, tmp_path: Path) -> None:
        callback = MagicMock()
        gm = GameManager(data_dir=tmp_path, on_state_change=callback)
        gm._set_state("initializing")
        assert gm.state == "initializing"
        callback.assert_called_once()
        call_arg = callback.call_args[0][0]
        assert call_arg["state"] == "initializing"

    def test_handles_callback_exception_gracefully(self, tmp_path: Path) -> None:
        callback = MagicMock(side_effect=ValueError("boom"))
        gm = GameManager(data_dir=tmp_path, on_state_change=callback)
        # Should not raise
        gm._set_state("running")
        assert gm.state == "running"
        callback.assert_called_once()

    def test_works_without_callback(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path, on_state_change=None)
        gm._set_state("stopping")
        assert gm.state == "stopping"


class TestStartGame:
    def test_raises_when_not_idle(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        with pytest.raises(RuntimeError, match="Cannot start game"):
            gm.start_game()

    def test_raises_when_initializing(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "initializing"
        with pytest.raises(RuntimeError, match="Cannot start game"):
            gm.start_game()

    def test_raises_when_stopping(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "stopping"
        with pytest.raises(RuntimeError, match="Cannot start game"):
            gm.start_game()

    def test_spawns_daemon_thread(self, tmp_path: Path) -> None:
        """start_game() creates a daemon thread named 'game-worker'."""
        gm = GameManager(data_dir=tmp_path)

        with patch.object(gm, "_run_game") as mock_run:
            gm.start_game(num_games=3)

            assert gm._game_thread is not None
            assert gm._game_thread.daemon is True
            assert gm._game_thread.name == "game-worker"
            # Wait for thread to start and call _run_game
            gm._game_thread.join(timeout=2)
            mock_run.assert_called_once_with(3)

    def test_sets_state_to_initializing(self, tmp_path: Path) -> None:
        callback = MagicMock()
        gm = GameManager(data_dir=tmp_path, on_state_change=callback)

        with patch.object(gm, "_run_game"):
            gm.start_game()

        first_call_arg = callback.call_args_list[0][0][0]
        assert first_call_arg["state"] == "initializing"

    def test_clears_stop_event(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._stop_event.set()

        with patch.object(gm, "_run_game"):
            gm.start_game()

        assert not gm._stop_event.is_set()

    def test_clears_previous_error(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._error_message = "old error"

        with patch.object(gm, "_run_game"):
            gm.start_game()

        assert gm._error_message is None

    def test_returns_starting(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)

        with patch.object(gm, "_run_game"):
            result = gm.start_game()

        assert result == "starting"


class TestStopGame:
    def test_raises_when_idle(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Cannot stop game"):
            gm.stop_game()

    def test_raises_when_stopping(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "stopping"
        with pytest.raises(RuntimeError, match="Cannot stop game"):
            gm.stop_game()

    def test_sets_stop_event(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._game_thread = MagicMock(spec=threading.Thread)
        gm._game_thread.is_alive.return_value = False

        gm.stop_game()

        assert gm._stop_event.is_set()
        # Thread already exited, so state transitions stopping -> idle
        assert gm.state == "idle"

    def test_stays_stopping_if_thread_alive(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = True  # thread did not exit
        gm._game_thread = mock_thread

        gm.stop_game()

        assert gm._stop_event.is_set()
        # Thread still alive, state stays stopping
        assert gm.state == "stopping"

    def test_joins_thread_with_timeout(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = False
        gm._game_thread = mock_thread

        gm.stop_game()

        mock_thread.join.assert_called_once_with(timeout=_THREAD_JOIN_TIMEOUT)

    def test_logs_warning_if_thread_still_alive(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        mock_thread = MagicMock(spec=threading.Thread)
        mock_thread.is_alive.return_value = True  # thread did not exit
        gm._game_thread = mock_thread

        with patch("roc.game.game_manager.logger") as mock_logger:
            gm.stop_game()
            mock_logger.warning.assert_called()

    def test_sets_stop_requested(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._game_thread = MagicMock(spec=threading.Thread)
        gm._game_thread.is_alive.return_value = False

        assert not gm.is_stop_requested()
        gm.stop_game()
        assert gm.is_stop_requested()


class TestRunGame:
    """Verify _run_game() thread entry point behavior."""

    def test_calls_game_entry_function(self, tmp_path: Path) -> None:
        """_run_game() calls the game_entry callable."""
        game_entry = MagicMock()
        gm = GameManager(data_dir=tmp_path, game_entry=game_entry)
        gm._state = "initializing"

        gm._run_game(num_games=3)

        game_entry.assert_called_once()
        kwargs = game_entry.call_args[1]
        assert kwargs["num_games"] == 3
        assert kwargs["stop_event"] is gm._stop_event

    def test_raises_error_when_no_game_entry(self, tmp_path: Path) -> None:
        """_run_game fails if no game_entry callable was provided."""
        gm = GameManager(data_dir=tmp_path)
        gm._state = "initializing"

        gm._run_game(num_games=1)

        assert gm._error_message == "No game_entry callable provided for thread mode"
        assert gm.state == "idle"

    def test_passes_on_run_name_callback(self, tmp_path: Path) -> None:
        """game_entry receives on_run_name callback that sets run name."""
        game_entry = MagicMock()
        gm = GameManager(data_dir=tmp_path, game_entry=game_entry)
        gm._state = "initializing"

        gm._run_game(num_games=2)

        kwargs = game_entry.call_args[1]
        # Call the on_run_name callback and verify it sets the run name
        kwargs["on_run_name"]("my-test-run")
        assert gm._current_run_name == "my-test-run"

    def test_transitions_to_idle_on_success(self, tmp_path: Path) -> None:
        game_entry = MagicMock()
        callback = MagicMock()
        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=callback,
            game_entry=game_entry,
        )
        gm._state = "initializing"

        gm._run_game(num_games=1)

        assert gm.state == "idle"

    def test_captures_error_on_exception(self, tmp_path: Path) -> None:
        game_entry = MagicMock(side_effect=RuntimeError("NLE crashed"))
        gm = GameManager(data_dir=tmp_path, game_entry=game_entry)
        gm._state = "initializing"

        gm._run_game(num_games=1)

        assert gm._error_message == "NLE crashed"
        assert gm.state == "idle"

    def test_transitions_to_running_before_game_entry(self, tmp_path: Path) -> None:
        """State should be 'running' when game_entry is called."""
        observed_state = {}

        def capture_state(**kwargs):
            observed_state["state"] = kwargs["stop_event"]  # just to use kwargs
            observed_state["game_state"] = gm.state

        game_entry = MagicMock(side_effect=capture_state)
        gm = GameManager(data_dir=tmp_path, game_entry=game_entry)
        gm._state = "initializing"

        gm._run_game(num_games=1)

        assert observed_state["game_state"] == "running"

    def test_notifies_state_change_on_running(self, tmp_path: Path) -> None:
        game_entry = MagicMock()
        callback = MagicMock()
        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=callback,
            game_entry=game_entry,
        )
        gm._state = "initializing"

        gm._run_game(num_games=1)

        # Should have at least "running" and "idle" state changes
        states = [c[0][0]["state"] for c in callback.call_args_list]
        assert "running" in states
        assert "idle" in states

    def test_notifies_state_change_on_error(self, tmp_path: Path) -> None:
        game_entry = MagicMock(side_effect=ValueError("boom"))
        callback = MagicMock()
        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=callback,
            game_entry=game_entry,
        )
        gm._state = "initializing"

        gm._run_game(num_games=1)

        # Final state should be idle
        last_call = callback.call_args_list[-1][0][0]
        assert last_call["state"] == "idle"

    def test_sets_run_name_from_game_entry(self, tmp_path: Path) -> None:
        """game_entry can set run_name via the on_run_name callback."""

        def fake_game_entry(*, num_games, stop_event, on_run_name):
            on_run_name("2026-04-05_test-run")

        gm = GameManager(data_dir=tmp_path, game_entry=fake_game_entry)
        gm._state = "initializing"

        gm._run_game(num_games=1)

        assert gm._current_run_name == "2026-04-05_test-run"


class TestIsStopRequested:
    def test_false_initially(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert not gm.is_stop_requested()

    def test_cleared_on_start_game(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._stop_requested.set()
        assert gm.is_stop_requested()

        with patch.object(gm, "_run_game"):
            gm.start_game()
        assert not gm.is_stop_requested()


class TestIntegration:
    """Integration tests: thread lifecycle with real threading."""

    def test_start_and_stop_with_real_thread(self, tmp_path: Path) -> None:
        """Full lifecycle: start -> game runs -> stop -> idle."""
        game_started = threading.Event()
        game_can_finish = threading.Event()

        def fake_game_entry(*, num_games, stop_event, on_run_name):
            on_run_name("test-run")
            game_started.set()
            # Simulate game loop that checks stop_event
            while not stop_event.is_set() and not game_can_finish.is_set():
                time.sleep(0.01)

        callback = MagicMock()
        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=callback,
            game_entry=fake_game_entry,
        )

        gm.start_game(num_games=1)
        assert game_started.wait(timeout=2), "Game should have started"
        assert gm.state == "running"
        assert gm._current_run_name == "test-run"

        gm.stop_game()
        assert gm.state == "idle"
        assert gm._game_thread is not None
        assert not gm._game_thread.is_alive()

    def test_game_completes_naturally(self, tmp_path: Path) -> None:
        """Game exits on its own without stop_game() being called."""

        def fast_game(*, num_games, stop_event, on_run_name):
            on_run_name("fast-run")
            # Game completes immediately

        callback = MagicMock()
        gm = GameManager(
            data_dir=tmp_path,
            on_state_change=callback,
            game_entry=fast_game,
        )

        gm.start_game()
        # Wait for thread to finish
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=2)

        assert gm.state == "idle"
        assert gm._error_message is None

    def test_game_crash_captured(self, tmp_path: Path) -> None:
        """Unhandled exception in game_entry is captured as error."""

        def crashing_game(*, num_games, stop_event, on_run_name):
            raise RuntimeError("Segfault simulation")

        gm = GameManager(
            data_dir=tmp_path,
            game_entry=crashing_game,
        )

        gm.start_game()
        assert gm._game_thread is not None
        gm._game_thread.join(timeout=2)

        assert gm.state == "idle"
        assert gm._error_message == "Segfault simulation"

    def test_is_stop_requested_works(self, tmp_path: Path) -> None:
        """is_stop_requested() reflects the stop_event state."""
        game_started = threading.Event()

        def blocking_game(*, num_games, stop_event, on_run_name):
            game_started.set()
            stop_event.wait(timeout=5)

        gm = GameManager(
            data_dir=tmp_path,
            game_entry=blocking_game,
        )

        gm.start_game()
        game_started.wait(timeout=2)

        assert not gm.is_stop_requested()
        gm.stop_game()
        assert gm.is_stop_requested()
