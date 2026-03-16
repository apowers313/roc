"""Thread-safe ring buffer for pushing StepData from the game loop to the dashboard."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from roc.reporting.run_store import StepData

# Module-level singleton so gymnasium.py can push without importing the dashboard.
_step_buffer: StepBuffer | None = None
_buffer_lock = threading.Lock()


class StepBuffer:
    """Ring buffer that holds recent StepData for the live dashboard.

    The game loop calls ``push()`` to add data; the dashboard reads via
    ``get_latest()`` or ``get_step()``.  Listeners registered via
    ``add_listener()`` are called after each push so the dashboard can
    schedule a UI update on the Tornado thread.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self._buf: deque[StepData] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._listeners: list[Callable[[], None]] = []
        self._listener_lock = threading.Lock()

    def push(self, data: StepData) -> None:
        """Append a step to the buffer and notify listeners."""
        with self._lock:
            self._buf.append(data)
        # Notify outside the data lock to avoid deadlock
        with self._listener_lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn()
            except Exception:
                pass  # listener errors must not break the game loop

    def add_listener(self, fn: Callable[[], None]) -> None:
        """Register a callback invoked (from the game thread) after each push."""
        with self._listener_lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[], None]) -> None:
        """Unregister a push listener."""
        with self._listener_lock:
            try:
                self._listeners.remove(fn)
            except ValueError:
                pass

    def get_latest(self) -> StepData | None:
        """Return the most recent StepData, or None if empty."""
        with self._lock:
            if self._buf:
                return self._buf[-1]
            return None

    def get_step(self, step: int) -> StepData | None:
        """Return a specific step if still in the buffer."""
        with self._lock:
            for data in self._buf:
                if data.step == step:
                    return data
            return None

    @property
    def max_step(self) -> int:
        """Return the highest step number in the buffer, or 0."""
        with self._lock:
            if self._buf:
                return self._buf[-1].step
            return 0

    @property
    def min_step(self) -> int:
        """Return the lowest step number in the buffer, or 0."""
        with self._lock:
            if self._buf:
                return self._buf[0].step
            return 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def game_numbers(self) -> list[int]:
        """Return sorted unique game numbers present in the buffer."""
        with self._lock:
            return sorted({d.game_number for d in self._buf})


def register_step_buffer(buf: StepBuffer) -> None:
    """Register the global step buffer (called by dashboard_server)."""
    global _step_buffer
    with _buffer_lock:
        _step_buffer = buf


def get_step_buffer() -> StepBuffer | None:
    """Return the global step buffer, or None if no dashboard is running."""
    with _buffer_lock:
        return _step_buffer


def clear_step_buffer() -> None:
    """Clear the global step buffer reference (for shutdown/testing)."""
    global _step_buffer
    with _buffer_lock:
        _step_buffer = None
