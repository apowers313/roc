"""Game lifecycle management via subprocess isolation.

Spawns `uv run play` as a subprocess so each game gets a clean Python
interpreter. No singleton cleanup needed between runs.
"""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Literal

from loguru import logger

# Seconds to wait for cooperative shutdown (game checks REST response) before SIGTERM
_GRACEFUL_TIMEOUT = 5
# Seconds to wait after SIGTERM before escalating to SIGKILL
_SIGKILL_TIMEOUT = 10


class GameManager:
    """Manages game lifecycle via subprocess."""

    def __init__(
        self,
        data_dir: Path,
        on_state_change: Callable[[dict[str, Any]], None] | None = None,
        server_url: str | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._on_state_change = on_state_change
        self._server_url = server_url
        self._process: subprocess.Popen[bytes] | None = None
        self._state: Literal["idle", "initializing", "running", "stopping"] = "idle"
        self._current_run_name: str | None = None
        self._exit_code: int | None = None
        self._error_message: str | None = None
        self._monitor_thread: threading.Thread | None = None
        self._log_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()

    @property
    def state(self) -> str:
        return self._state

    def is_stop_requested(self) -> bool:
        """Check whether a cooperative stop has been requested."""
        return self._stop_requested.is_set()

    def start_game(self, num_games: int = 5) -> str:
        """Spawn a game subprocess.

        Returns a placeholder message. The actual run name is detected
        asynchronously by the monitor thread.

        Raises RuntimeError if a game is already running.
        """
        with self._lock:
            if self._state != "idle":
                raise RuntimeError(f"Cannot start game: state is {self._state}")
            self._set_state("initializing")
            self._exit_code = None
            self._error_message = None
            self._stop_requested.clear()

        # Snapshot existing run directories before spawning
        existing_runs = set()
        if self._data_dir.is_dir():
            existing_runs = {d.name for d in self._data_dir.iterdir() if d.is_dir()}

        # Build the command. Disable the subprocess's own dashboard server
        # since we are the dashboard server. Pass callback URL so the game
        # pushes step data back to us via HTTP.
        cmd = [
            "uv",
            "run",
            "play",
            "--no-dashboard-enabled",
            f"--num-games={num_games}",
        ]
        if self._server_url is not None:
            cmd.append(f"--dashboard-callback-url={self._server_url}/api/internal/step")

        logger.info("Starting game subprocess: {}", " ".join(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent),
            )
        except Exception as e:
            logger.opt(exception=True).error("Failed to spawn game subprocess")
            with self._lock:
                self._error_message = str(e)
                self._set_state("idle")
            raise RuntimeError(f"Failed to start game: {e}")

        # Start log streaming thread (reads subprocess stdout)
        self._log_thread = threading.Thread(
            target=self._stream_logs,
            daemon=True,
            name="game-logs",
        )
        self._log_thread.start()

        # Start monitor thread to detect run name and wait for exit
        self._monitor_thread = threading.Thread(
            target=self._monitor_process,
            args=(existing_runs,),
            daemon=True,
            name="game-monitor",
        )
        self._monitor_thread.start()

        return "starting"

    def stop_game(self) -> None:
        """Request cooperative shutdown, escalate to SIGTERM then SIGKILL."""
        with self._lock:
            if self._state not in ("initializing", "running"):
                raise RuntimeError(f"Cannot stop game: state is {self._state}")
            self._set_state("stopping")

        self._stop_requested.set()
        logger.info("Cooperative stop requested for game subprocess")

        if self._process is not None:
            # Start a watchdog that escalates: graceful -> SIGTERM -> SIGKILL
            threading.Thread(
                target=self._shutdown_watchdog,
                daemon=True,
                name="game-shutdown-watchdog",
            ).start()

    def get_status(self) -> dict[str, Any]:
        """Return current game state."""
        status: dict[str, Any] = {
            "state": self._state,
            "run_name": self._current_run_name,
        }
        if self._exit_code is not None:
            status["exit_code"] = self._exit_code
        if self._error_message is not None:
            status["error"] = self._error_message
        return status

    def _set_state(self, new_state: Literal["idle", "initializing", "running", "stopping"]) -> None:
        """Update state and notify callback."""
        self._state = new_state
        logger.info("Game state: {} (run={})", new_state, self._current_run_name)
        if self._on_state_change is not None:
            try:
                self._on_state_change(self.get_status())
            except Exception:
                logger.opt(exception=True).warning("on_state_change callback failed")

    def _stream_logs(self) -> None:
        """Read subprocess stdout/stderr and forward to loguru."""
        assert self._process is not None
        assert self._process.stdout is not None
        try:
            for line in self._process.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.info("[game] {}", text)
        except Exception:
            pass  # process exited, pipe closed

    def _monitor_process(self, existing_runs: set[str]) -> None:
        """Monitor the game subprocess: detect run name, wait for exit."""
        assert self._process is not None

        # Poll for the new run directory to appear (up to 30s)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                # Process exited before we found the run dir
                break
            if self._data_dir.is_dir():
                current_runs = {d.name for d in self._data_dir.iterdir() if d.is_dir()}
                new_runs = current_runs - existing_runs
                if new_runs:
                    # Pick the newest (lexicographically largest since names are timestamp-prefixed)
                    self._current_run_name = sorted(new_runs)[-1]
                    with self._lock:
                        if self._state == "initializing":
                            self._set_state("running")
                    break
            time.sleep(0.5)

        # Wait for the subprocess to exit
        try:
            exit_code = self._process.wait(timeout=None)
            self._exit_code = exit_code
            was_stopping = self._state == "stopping"
            if exit_code == 0:
                logger.info("Game subprocess exited cleanly (code 0)")
            elif was_stopping:
                # User-initiated stop: non-zero exit is expected (SIGTERM -> exit 1)
                logger.info("Game subprocess stopped (code {})", exit_code)
            elif exit_code < 0:
                sig = -exit_code
                sig_name = (
                    signal.Signals(sig).name
                    if sig in signal.Signals._value2member_map_
                    else str(sig)
                )
                logger.warning("Game subprocess killed by signal {} ({})", sig_name, sig)
                self._error_message = f"Killed by signal {sig_name}"
            else:
                logger.error("Game subprocess crashed with exit code {}", exit_code)
                self._error_message = f"Exited with code {exit_code}"
        except Exception:
            logger.opt(exception=True).warning("Error waiting for game subprocess")
            self._error_message = "Monitor error"

        # Clean up
        with self._lock:
            self._process = None
            self._set_state("idle")
            # Keep _current_run_name so the UI can still show the last run

    def _shutdown_watchdog(self) -> None:
        """Escalate shutdown: wait for cooperative exit, then SIGTERM, then SIGKILL."""
        proc = self._process
        if proc is None:
            return

        # Tier 1: wait for the game to exit cooperatively (via REST stop response)
        try:
            proc.wait(timeout=_GRACEFUL_TIMEOUT)
            return  # game exited cleanly
        except subprocess.TimeoutExpired:
            pass

        # Tier 2: send SIGTERM
        logger.warning(
            "Game did not exit within {}s of stop request, sending SIGTERM (pid={})",
            _GRACEFUL_TIMEOUT,
            proc.pid,
        )
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return  # already gone

        # Tier 3: wait for SIGTERM, then escalate to SIGKILL
        try:
            proc.wait(timeout=_SIGKILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Game subprocess did not exit within {}s after SIGTERM, sending SIGKILL",
                _SIGKILL_TIMEOUT,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass  # already gone
