# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/game_manager.py."""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from roc.game_manager import GameManager, _SIGKILL_TIMEOUT


class TestInitialState:
    def test_state_is_idle(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm.state == "idle"

    def test_no_run_name(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._current_run_name is None

    def test_no_process(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._process is None

    def test_no_exit_code(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._exit_code is None

    def test_no_error_message(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        assert gm._error_message is None


class TestGetStatus:
    def test_idle_status_shape(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        status = gm.get_status()
        assert status == {"state": "idle", "run_name": None}

    def test_includes_exit_code_when_set(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._exit_code = 42
        status = gm.get_status()
        assert status["exit_code"] == 42

    def test_includes_error_when_set(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._error_message = "something broke"
        status = gm.get_status()
        assert status["error"] == "something broke"

    def test_excludes_exit_code_when_none(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        status = gm.get_status()
        assert "exit_code" not in status

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

    @patch("roc.game_manager.threading.Thread")
    @patch("roc.game_manager.subprocess.Popen")
    def test_succeeds_with_mocked_subprocess(
        self, mock_popen_cls: MagicMock, mock_thread_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_proc = MagicMock()
        mock_popen_cls.return_value = mock_proc
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        gm = GameManager(data_dir=tmp_path)
        result = gm.start_game(num_games=3)

        assert result == "starting"
        mock_popen_cls.assert_called_once()
        # Two threads: log thread and monitor thread
        assert mock_thread_cls.call_count == 2
        assert mock_thread.start.call_count == 2

    @patch("roc.game_manager.threading.Thread")
    @patch("roc.game_manager.subprocess.Popen")
    def test_sets_state_to_initializing(
        self, mock_popen_cls: MagicMock, mock_thread_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_popen_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        callback = MagicMock()
        gm = GameManager(data_dir=tmp_path, on_state_change=callback)
        gm.start_game()

        # The first state change should be to "initializing"
        first_call_arg = callback.call_args_list[0][0][0]
        assert first_call_arg["state"] == "initializing"

    @patch("roc.game_manager.subprocess.Popen", side_effect=FileNotFoundError("uv not found"))
    def test_failed_popen_raises_runtime_error(
        self, mock_popen_cls: MagicMock, tmp_path: Path
    ) -> None:
        gm = GameManager(data_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Failed to start game"):
            gm.start_game()
        # State should revert to idle
        assert gm.state == "idle"
        assert gm._error_message is not None

    @patch("roc.game_manager.threading.Thread")
    @patch("roc.game_manager.subprocess.Popen")
    def test_passes_server_url_in_command(
        self, mock_popen_cls: MagicMock, mock_thread_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_popen_cls.return_value = MagicMock()
        mock_thread_cls.return_value = MagicMock()

        gm = GameManager(data_dir=tmp_path, server_url="http://localhost:9043")
        gm.start_game(num_games=2)

        cmd = mock_popen_cls.call_args[0][0]
        assert "--dashboard-callback-url=http://localhost:9043/api/internal/step" in cmd
        assert "--num-games=2" in cmd


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

    @patch("roc.game_manager.threading.Thread")
    def test_sends_sigterm_to_process(self, mock_thread_cls: MagicMock, tmp_path: Path) -> None:
        mock_thread_cls.return_value = MagicMock()
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345

        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._process = mock_proc

        gm.stop_game()

        mock_proc.send_signal.assert_called_once_with(signal.SIGTERM)
        assert gm.state == "stopping"

    @patch("roc.game_manager.threading.Thread")
    def test_handles_already_exited_process(
        self, mock_thread_cls: MagicMock, tmp_path: Path
    ) -> None:
        mock_thread_cls.return_value = MagicMock()
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        mock_proc.send_signal.side_effect = ProcessLookupError()

        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._process = mock_proc

        # Should not raise
        gm.stop_game()
        assert gm.state == "stopping"


class TestMonitorProcess:
    def test_detects_new_run_directory(self, tmp_path: Path) -> None:
        """When a new directory appears in data_dir, monitor sets run name and state."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        # poll() returns None first (still running), then 0 (exited)
        mock_proc.poll.side_effect = [None, None, None]
        mock_proc.wait.return_value = 0

        callback = MagicMock()
        gm = GameManager(data_dir=tmp_path, on_state_change=callback)
        gm._state = "initializing"
        gm._process = mock_proc

        existing_runs: set[str] = set()

        # Create a new run directory to be discovered
        run_dir = tmp_path / "2026-03-22_run001"
        run_dir.mkdir()

        with patch("roc.game_manager.time.sleep"):
            gm._monitor_process(existing_runs)

        assert gm._current_run_name == "2026-03-22_run001"
        assert gm._exit_code == 0
        # Final state should be idle (after process exits)
        assert gm.state == "idle"

    def test_handles_process_exit_before_run_dir_found(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        # Process already exited (poll returns exit code immediately)
        mock_proc.poll.return_value = 1
        mock_proc.wait.return_value = 1

        gm = GameManager(data_dir=tmp_path)
        gm._state = "initializing"
        gm._process = mock_proc

        with patch("roc.game_manager.time.sleep"):
            gm._monitor_process(set())

        assert gm._current_run_name is None
        assert gm._exit_code == 1
        assert gm.state == "idle"

    def test_handles_clean_exit(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0

        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._process = mock_proc

        with patch("roc.game_manager.time.sleep"):
            gm._monitor_process(set())

        assert gm._exit_code == 0
        assert gm._error_message is None
        assert gm.state == "idle"

    def test_handles_crash_exit(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 1
        mock_proc.wait.return_value = 42

        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._process = mock_proc

        with patch("roc.game_manager.time.sleep"):
            gm._monitor_process(set())

        assert gm._exit_code == 42
        assert gm._error_message == "Exited with code 42"
        assert gm.state == "idle"

    def test_handles_signal_death(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = -signal.SIGTERM
        mock_proc.wait.return_value = -signal.SIGTERM

        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._process = mock_proc

        with patch("roc.game_manager.time.sleep"):
            gm._monitor_process(set())

        assert gm._exit_code == -signal.SIGTERM
        assert gm._error_message is not None
        assert "signal" in gm._error_message.lower() or "SIGTERM" in gm._error_message
        assert gm.state == "idle"

    def test_cleans_up_process_reference(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0

        gm = GameManager(data_dir=tmp_path)
        gm._state = "running"
        gm._process = mock_proc

        with patch("roc.game_manager.time.sleep"):
            gm._monitor_process(set())

        assert gm._process is None


class TestStreamLogs:
    def test_reads_lines_from_stdout(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        lines = [b"hello world\n", b"second line\n"]
        mock_proc.stdout = iter(lines)

        gm = GameManager(data_dir=tmp_path)
        gm._process = mock_proc

        # Should run without error and consume all lines
        gm._stream_logs()

    def test_handles_empty_stdout(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.stdout = iter([])

        gm = GameManager(data_dir=tmp_path)
        gm._process = mock_proc

        gm._stream_logs()


class TestSigkillWatchdog:
    def test_returns_immediately_when_process_is_none(self, tmp_path: Path) -> None:
        gm = GameManager(data_dir=tmp_path)
        gm._process = None
        # Should return without error
        gm._sigkill_watchdog()

    def test_does_not_kill_if_process_exits_before_timeout(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.wait.return_value = 0  # exits cleanly

        gm = GameManager(data_dir=tmp_path)
        gm._process = mock_proc

        gm._sigkill_watchdog()

        mock_proc.wait.assert_called_once_with(timeout=_SIGKILL_TIMEOUT)
        mock_proc.kill.assert_not_called()

    def test_escalates_to_sigkill_on_timeout(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="play", timeout=10)

        gm = GameManager(data_dir=tmp_path)
        gm._process = mock_proc

        gm._sigkill_watchdog()

        mock_proc.kill.assert_called_once()

    def test_handles_already_gone_on_kill(self, tmp_path: Path) -> None:
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="play", timeout=10)
        mock_proc.kill.side_effect = ProcessLookupError()

        gm = GameManager(data_dir=tmp_path)
        gm._process = mock_proc

        # Should not raise
        gm._sigkill_watchdog()
