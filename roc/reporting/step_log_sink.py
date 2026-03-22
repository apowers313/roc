"""Loguru sink that captures log records for live dashboard display."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

# Ring buffer capacity -- one game step produces 0-5 log lines typically.
# 2000 entries covers ~400-2000 steps of history.
_MAX_ENTRIES = 2000

_buffer: deque[dict[str, Any]] = deque(maxlen=_MAX_ENTRIES)
_lock = threading.Lock()
_current_step: int = 0


def set_current_step(step: int) -> None:
    """Called by the game loop before each step's processing."""
    global _current_step
    _current_step = step


def step_log_sink(message: str) -> None:
    """Loguru sink: capture log records tagged with the current step."""
    record = message.record  # type: ignore
    entry = {
        "step": _current_step,
        "timestamp": int(record["time"].timestamp() * 1000),
        "severity_text": record["level"].name,
        "body": message.strip(),
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    with _lock:
        _buffer.append(entry)


def drain_step_logs(step: int) -> list[dict[str, Any]] | None:
    """Return all captured logs for a step, removing them from the buffer.

    Returns None if no logs were captured (keeps StepData.logs sparse).
    """
    with _lock:
        logs = [e for e in _buffer if e["step"] == step]
    return logs if logs else None
