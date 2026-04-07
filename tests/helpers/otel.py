"""Test helpers for creating OTel LogRecord objects."""

from time import time_ns
from typing import Any

from opentelemetry._logs import SeverityNumber
from opentelemetry._logs import LogRecord

# Sentinel: when ``tick`` is not passed we auto-stamp ``Clock.get()`` at
# record-creation time, mirroring production's ``_emit_state_record``.
# Callers can pass ``tick=None`` explicitly to produce a tickless record,
# which is useful for exercising the exporter's Clock fallback path.
_AUTO_TICK: Any = object()


def make_log_record(
    event_name: str | None = None,
    body: str | None = None,
    severity: SeverityNumber = SeverityNumber.INFO,
    timestamp: int | None = None,
    attributes: dict[str, Any] | None = None,
    tick: int | None | Any = _AUTO_TICK,
) -> LogRecord:
    """Create an OTel LogRecord for testing.

    Args:
        event_name: Optional event.name attribute (e.g. "roc.screen").
        body: The log record body string.
        severity: OTel severity number.
        timestamp: Nanosecond timestamp. Defaults to current time.
        attributes: Additional attributes to include.
        tick: Controls the ``tick`` attribute on the record:

            * **Omitted** (default): stamps ``Clock.get()`` at call time,
              mirroring production's ``_emit_state_record``. Tests that
              want per-step data can just call ``Clock.set(N)`` before
              emitting.
            * **int**: stamps the explicit integer.
            * **None**: emits the record without a ``tick`` attribute --
              used to exercise the exporter's Clock fallback path.

    Returns:
        A LogRecord instance ready for export.
    """
    attrs: dict[str, Any] = {}
    if event_name is not None:
        attrs["event.name"] = event_name
    if tick is _AUTO_TICK:
        from roc.framework.clock import Clock

        attrs["tick"] = Clock.get()
    elif tick is not None:
        attrs["tick"] = tick
    if attributes is not None:
        attrs.update(attributes)

    return LogRecord(
        timestamp=timestamp or time_ns(),
        severity_number=severity,
        severity_text=severity.name if severity else "INFO",
        body=body,
        attributes=attrs if attrs else None,
    )
