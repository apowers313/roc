"""OTel log exporter that POSTs records to the Remote Logger MCP server."""

from __future__ import annotations

import json
import ssl
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from opentelemetry.sdk._logs.export import LogExporter, LogExportResult

_SEVERITY_MAP = {
    1: "TRACE",
    2: "TRACE",
    3: "TRACE",
    4: "TRACE",
    5: "DEBUG",
    6: "DEBUG",
    7: "DEBUG",
    8: "DEBUG",
    9: "INFO",
    10: "INFO",
    11: "INFO",
    12: "INFO",
    13: "WARN",
    14: "WARN",
    15: "WARN",
    16: "WARN",
    17: "ERROR",
    18: "ERROR",
    19: "ERROR",
    20: "ERROR",
    21: "FATAL",
    22: "FATAL",
    23: "FATAL",
    24: "FATAL",
}


class RemoteLoggerExporter(LogExporter):
    """OTel log exporter that POSTs records to the Remote Logger MCP server.

    Converts OTel LogRecords to the Remote Logger's expected JSON format and
    sends them via HTTP POST. Connection errors are silently caught to avoid
    crashing the game.
    """

    def __init__(self, url: str, session_id: str, timeout: float = 2.0) -> None:
        self._url = url
        self._session_id = session_id
        self._timeout = timeout
        # Support self-signed certs on dev servers
        self._ssl_ctx: ssl.SSLContext | None = None
        if url.startswith("https://"):
            self._ssl_ctx = ssl.create_default_context()
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE

    def export(self, batch: Sequence[Any]) -> LogExportResult:
        """Export log records to the Remote Logger endpoint.

        Args:
            batch: Sequence of LogData objects from OTel.

        Returns:
            LogExportResult.SUCCESS or LogExportResult.FAILURE.
        """
        try:
            logs = []
            for log_data in batch:
                record = log_data.log_record
                # Convert timestamp (nanoseconds) to ISO format
                ts = record.timestamp
                if ts:
                    time_str = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).isoformat()
                else:
                    time_str = datetime.now(tz=timezone.utc).isoformat()

                # Map severity number to level string
                level = "INFO"
                if record.severity_number:
                    level = _SEVERITY_MAP.get(record.severity_number.value, "INFO")

                body_str = str(record.body) if record.body is not None else ""

                logs.append(
                    {
                        "time": time_str,
                        "level": level,
                        "message": body_str,
                    }
                )

            payload = {
                "sessionId": self._session_id,
                "logs": logs,
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=self._timeout, context=self._ssl_ctx)
            return LogExportResult.SUCCESS
        except Exception:
            return LogExportResult.FAILURE

    def shutdown(self) -> None:
        """No resources to clean up."""

    def force_flush(self, timeout_millis: int = 0) -> bool:
        """Nothing to flush -- records are sent synchronously."""
        return True
