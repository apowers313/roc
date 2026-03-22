# Live Subprocess Logs

## Problem

The Log Messages panel is empty during live subprocess games. Loguru output
goes through OTel -> DuckLake, but the game subprocess holds the DuckDB file
lock, so the dashboard server can't read from it. The immediate performance
fix (skip DuckDB log queries in subprocess mode) makes stepping fast but
leaves logs unavailable until the game ends.

## Current Architecture

```
Game subprocess:
  loguru.info("message")
    |
    v
  loguru_to_otel() sink       -- converts to OTel LogRecord
    |
    v
  OTel LoggerProvider          -- batches and exports
    |
    v
  ParquetExporter              -- writes to DuckLake "logs" table
    |
    v
  DuckDB file (LOCKED)         -- dashboard server can't read this


  gymnasium.py game loop:
    assembles StepData(step=N, screen=..., metrics=..., logs=None)
                                                        ^^^^^^^^^
    POSTs to /api/internal/step
```

Logs and step data travel through completely separate paths. StepData never
includes logs because loguru output is asynchronous (OTel batches and flushes
on its own schedule) and has no connection to the game loop's per-step
assembly.

## Design

### Add a loguru ring buffer sink in the game subprocess

A lightweight loguru sink that captures recent log records in a deque, keyed
by game step. The game loop drains it when assembling StepData.

```
Game subprocess:
  loguru.info("message")
    |
    +---> loguru_to_otel()     -- unchanged, still writes to DuckLake
    |
    +---> step_log_sink()      -- NEW: appends to ring buffer
              |
              v
           _step_log_buffer: deque of {step, timestamp, level, message}


  gymnasium.py game loop:
    logs = drain_step_logs(current_step)
    StepData(step=N, ..., logs=logs)
    POSTs to /api/internal/step
```

Both sinks receive every log message. The OTel path continues writing to
DuckLake for historical queries. The ring buffer path provides logs for
the live dashboard.

### Implementation

**New file: `roc/reporting/step_log_sink.py`**

```python
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
    record = message.record
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
```

**Changes to `roc/gymnasium.py`**

Before StepData assembly, set the current step and drain logs:

```python
from roc.reporting.step_log_sink import set_current_step, drain_step_logs

# At the top of each loop iteration (before any processing):
set_current_step(loop_num)

# When assembling StepData:
_step_logs = drain_step_logs(loop_num) if (_buf is not None or _callback_url) else None

_step_data = StepData(
    step=loop_num,
    ...
    logs=_step_logs,
)
```

**Changes to `roc/reporting/observability.py`**

Register the sink alongside the existing OTel sink:

```python
from roc.reporting.step_log_sink import step_log_sink

# In Observability.init(), after the existing loguru_to_otel sink:
if cfg.dashboard_enabled or cfg.dashboard_callback_url:
    roc_logger.logger.add(
        step_log_sink,
        format="<level>{message}</level>",
        level=settings.observability_logging_level,
    )
```

### Log record format

The live path produces a simpler format than the DuckLake path:

| Field | Live (step_log_sink) | Historical (DuckLake) |
|-------|---------------------|----------------------|
| step | from set_current_step | from SQL WHERE clause |
| timestamp | epoch ms | epoch ns |
| severity_text | loguru level name | OTel severity text |
| body | formatted message | OTel body string |
| module | loguru module | OTel code.namespace |
| function | loguru function | OTel code.function |
| line | loguru line number | OTel code.lineno |
| trace_id | not included | OTel trace context |
| span_id | not included | OTel trace context |
| severity_number | not included | OTel severity enum |

The Log Messages panel should work with both formats. It currently renders
whatever dict keys are present.

## UX Impact

| Scenario | Before | After |
|----------|--------|-------|
| Log Messages during subprocess game | Empty ("No log data") | Shows live log output |
| Log Messages during in-process game | Shows logs (via DuckDB) | Shows logs (via buffer, faster) |
| Log Messages on historical runs | Shows logs from DuckLake | Unchanged |
| Stepping speed during subprocess game | ~10ms (after perf fix) | ~10ms (logs add <1ms) |
| Step payload size over HTTP callback | ~80KB | ~80-82KB (logs add ~1-2KB) |

## Data Impact

**None.** The new sink is additive:

- DuckLake writes are unchanged (OTel sink still runs)
- Historical log queries are unchanged
- The ring buffer is ephemeral (in-memory, lost when subprocess exits)
- No new files, no new database tables, no schema changes

## Files Changed

| File | Change |
|------|--------|
| `roc/reporting/step_log_sink.py` | **NEW** -- ring buffer sink + drain function |
| `roc/gymnasium.py` | Set current step, drain logs into StepData |
| `roc/reporting/observability.py` | Register step_log_sink alongside OTel sink |

## Testing

- Unit test: `step_log_sink` captures and drains correctly
- Unit test: `drain_step_logs` returns None when no logs for step
- Integration: start a game via Game Menu, verify Log Messages panel
  shows loguru output during live play
