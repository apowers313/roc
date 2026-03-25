"""Test helpers for creating OTel LogRecord objects."""

from time import time_ns
from typing import Any

from opentelemetry._logs import SeverityNumber
from opentelemetry._logs import LogRecord


def make_log_record(
    event_name: str | None = None,
    body: str | None = None,
    severity: SeverityNumber = SeverityNumber.INFO,
    timestamp: int | None = None,
    attributes: dict[str, Any] | None = None,
) -> LogRecord:
    """Create an OTel LogRecord for testing.

    Args:
        event_name: Optional event.name attribute (e.g. "roc.screen").
        body: The log record body string.
        severity: OTel severity number.
        timestamp: Nanosecond timestamp. Defaults to current time.
        attributes: Additional attributes to include.

    Returns:
        A LogRecord instance ready for export.
    """
    attrs: dict[str, Any] = {}
    if event_name is not None:
        attrs["event.name"] = event_name
    if attributes is not None:
        attrs.update(attributes)

    return LogRecord(
        timestamp=timestamp or time_ns(),
        severity_number=severity,
        severity_text=severity.name if severity else "INFO",
        body=body,
        attributes=attrs if attrs else None,
    )
