"""Game lifecycle management via daemon thread.

Runs the game as a daemon thread within the server process. This enables
shared memory access to GraphCache and other Python objects -- no serialization,
no HTTP callbacks, no IPC.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from loguru import logger

# Seconds to wait for thread to exit after stop_event is set
_THREAD_JOIN_TIMEOUT = 10


class GameEntryFn(Protocol):
    """Protocol for the game entry point callable used in thread mode."""

    def __call__(
        self,
        *,
        num_games: int,
        stop_event: threading.Event,
        on_run_name: Callable[[str], None],
    ) -> None: ...


class GameManager:
    """Manages game lifecycle via daemon thread."""

    def __init__(
        self,
        data_dir: Path,
        on_state_change: Callable[[dict[str, Any]], None] | None = None,
        server_url: str | None = None,
        game_entry: GameEntryFn | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._on_state_change = on_state_change
        self._server_url = server_url
        self._game_entry = game_entry
        # Thread state
        self._game_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Shared state
        self._state: Literal["idle", "initializing", "running", "stopping"] = "idle"
        self._current_run_name: str | None = None
        self._error_message: str | None = None
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()

    @property
    def state(self) -> str:
        return self._state

    def is_stop_requested(self) -> bool:
        """Check whether a cooperative stop has been requested."""
        return self._stop_requested.is_set()

    def start_game(self, num_games: int = 5) -> str:
        """Start a game as a daemon thread.

        Returns a placeholder message. The actual run name is reported
        asynchronously via the on_run_name callback.

        Raises RuntimeError if a game is already running.
        """
        with self._lock:
            if self._state != "idle":
                raise RuntimeError(f"Cannot start game: state is {self._state}")
            self._set_state("initializing")
            self._error_message = None
            self._stop_requested.clear()
            self._stop_event.clear()

        return self._start_thread(num_games)

    def stop_game(self) -> None:
        """Request game shutdown via stop event."""
        with self._lock:
            if self._state not in ("initializing", "running"):
                raise RuntimeError(f"Cannot stop game: state is {self._state}")
            self._set_state("stopping")

        self._stop_requested.set()
        self._stop_event.set()
        self._stop_thread()

    def get_status(self) -> dict[str, Any]:
        """Return current game state."""
        status: dict[str, Any] = {
            "state": self._state,
            "run_name": self._current_run_name,
        }
        if self._error_message is not None:
            status["error"] = self._error_message
        return status

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def _start_thread(self, num_games: int) -> str:
        """Spawn a daemon thread to run the game."""
        logger.info("Starting game in thread mode (num_games={})", num_games)
        self._game_thread = threading.Thread(
            target=self._run_game,
            args=(num_games,),
            name="game-worker",
            daemon=True,
        )
        self._game_thread.start()
        return "starting"

    def _run_game(self, num_games: int) -> None:
        """Game entry point -- runs on the game thread."""
        try:
            with self._lock:
                self._set_state("running")

            if self._game_entry is None:
                raise RuntimeError("No game_entry callable provided for thread mode")

            self._game_entry(
                num_games=num_games,
                stop_event=self._stop_event,
                on_run_name=self._set_run_name,
            )
        except Exception as e:
            self._error_message = str(e)
            logger.opt(exception=True).error("Game thread failed")
        finally:
            with self._lock:
                self._set_state("idle")

    def _set_run_name(self, run_name: str) -> None:
        """Callback for the game thread to report its run name.

        Re-triggers the state change notification so the server can set up
        the live session (which requires the run name).
        """
        self._current_run_name = run_name
        logger.info("Game thread reported run name: {}", run_name)
        # Re-notify so _emit_game_state_changed can call _start_live_session
        # now that the run_name is known.
        if self._on_state_change is not None:
            try:
                self._on_state_change(self.get_status())
            except Exception:
                logger.opt(exception=True).warning("on_state_change callback failed")

    def _stop_thread(self) -> None:
        """Stop the game thread via Event and join."""
        logger.info("Stop requested for game thread")
        if self._game_thread is not None:
            self._game_thread.join(timeout=_THREAD_JOIN_TIMEOUT)
            if self._game_thread.is_alive():
                logger.warning(
                    "Game thread did not exit within {}s (daemon thread will die on process exit)",
                    _THREAD_JOIN_TIMEOUT,
                )
            else:
                with self._lock:
                    self._set_state("idle")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_state(self, new_state: Literal["idle", "initializing", "running", "stopping"]) -> None:
        """Update state and notify callback."""
        self._state = new_state
        logger.info("Game state: {} (run={})", new_state, self._current_run_name)
        if self._on_state_change is not None:
            try:
                self._on_state_change(self.get_status())
            except Exception:
                logger.opt(exception=True).warning("on_state_change callback failed")
