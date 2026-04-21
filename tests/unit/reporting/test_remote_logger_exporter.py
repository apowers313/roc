# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/remote_logger_exporter.py."""

import json
from time import time_ns
from unittest.mock import MagicMock

from opentelemetry._logs import SeverityNumber
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import LogRecordExportResult, SimpleLogRecordProcessor

from roc.reporting.remote_logger_exporter import RemoteLoggerExporter


def _make_log_data(body: str, severity: SeverityNumber = SeverityNumber.INFO) -> list[MagicMock]:
    """Create a list with one mock LogData matching OTel's export interface."""
    mock = MagicMock()
    mock.log_record.body = body
    mock.log_record.timestamp = time_ns()
    mock.log_record.severity_number = severity
    mock.log_record.severity_text = severity.name if severity else "INFO"
    mock.log_record.attributes = {}
    mock.log_record.trace_id = None
    mock.log_record.span_id = None
    return [mock]


class TestRemoteLoggerExporterPost:
    def test_exporter_posts_to_endpoint(self, httpserver):
        """RemoteLoggerExporter POSTs JSON to the configured URL."""
        httpserver.expect_request("/log", method="POST").respond_with_data("ok")
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="test-session",
        )
        batch = _make_log_data("test message")
        result = exporter.export(batch)
        assert result == LogRecordExportResult.SUCCESS

    def test_exporter_handles_connection_error(self):
        """Exporter should not raise on connection failure."""
        exporter = RemoteLoggerExporter(
            url="http://localhost:1/log",
            session_id="test-session",
            timeout=0.5,
        )
        batch = _make_log_data("test message")
        result = exporter.export(batch)
        assert result == LogRecordExportResult.FAILURE

    def test_exporter_formats_records_as_remote_logger_json(self, httpserver):
        """Records should match the Remote Logger's expected format."""
        received = []

        def handler(request):
            received.append(json.loads(request.data))
            return "ok"

        httpserver.expect_request("/log", method="POST").respond_with_handler(handler)
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="test-session-123",
        )
        batch = _make_log_data("hello world", SeverityNumber.WARN)
        exporter.export(batch)

        assert len(received) == 1
        payload = received[0]
        assert payload["sessionId"] == "test-session-123"
        assert len(payload["logs"]) == 1
        log_entry = payload["logs"][0]
        assert "time" in log_entry
        assert log_entry["level"] == "WARN"
        assert log_entry["message"] == "hello world"

    def test_exporter_sends_multiple_records(self, httpserver):
        """Multiple log records in one batch should all appear in the POST."""
        received = []

        def handler(request):
            received.append(json.loads(request.data))
            return "ok"

        httpserver.expect_request("/log", method="POST").respond_with_handler(handler)
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="multi-test",
        )
        batch = _make_log_data("msg1") + _make_log_data("msg2")
        exporter.export(batch)

        assert len(received) == 1
        assert len(received[0]["logs"]) == 2
        messages = [l["message"] for l in received[0]["logs"]]
        assert "msg1" in messages
        assert "msg2" in messages


class TestRemoteLoggerExporterWithOTel:
    def test_exporter_works_with_simple_processor(self, httpserver, reset_observability):
        """Exporter integrates correctly with SimpleLogRecordProcessor."""
        received = []

        def handler(request):
            received.append(json.loads(request.data))
            return "ok"

        httpserver.expect_request("/log", method="POST").respond_with_handler(handler)
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="otel-test",
        )

        lp = LoggerProvider()
        lp.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        logger = lp.get_logger("test")
        logger.emit(LogRecord(body="otel message", timestamp=time_ns()))

        assert len(received) == 1
        assert received[0]["logs"][0]["message"] == "otel message"


class TestRemoteLoggerExporterHealthCheck:
    def test_logs_warning_when_server_unreachable(self, caplog):
        """Init should log a warning when the remote logger server is unreachable."""
        import logging

        with caplog.at_level(logging.WARNING, logger="roc.reporting.remote_logger_exporter"):
            RemoteLoggerExporter(
                url="http://localhost:1/log",
                session_id="health-check-test",
                timeout=0.5,
            )
        assert any("unreachable" in r.message for r in caplog.records)

    def test_logs_info_when_server_reachable(self, httpserver, caplog):
        """Init should log info when the remote logger server responds to /status."""
        import logging

        httpserver.expect_request("/status", method="GET").respond_with_data("ok")
        httpserver.expect_request("/log", method="POST").respond_with_data("ok")
        with caplog.at_level(logging.INFO, logger="roc.reporting.remote_logger_exporter"):
            RemoteLoggerExporter(
                url=httpserver.url_for("/log"),
                session_id="health-check-ok",
            )
        assert any("reachable" in r.message.lower() for r in caplog.records)


class TestRemoteLoggerExporterEdgeCases:
    def test_none_body_becomes_empty_string(self, httpserver):
        """A record with None body should produce an empty message string."""
        received = []

        def handler(request):
            received.append(json.loads(request.data))
            return "ok"

        httpserver.expect_request("/log", method="POST").respond_with_handler(handler)
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="none-body",
        )
        batch = _make_log_data("placeholder")
        batch[0].log_record.body = None
        exporter.export(batch)

        assert received[0]["logs"][0]["message"] == ""

    def test_none_severity_defaults_to_info(self, httpserver):
        """A record with no severity should default to INFO level."""
        received = []

        def handler(request):
            received.append(json.loads(request.data))
            return "ok"

        httpserver.expect_request("/log", method="POST").respond_with_handler(handler)
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="no-sev",
        )
        batch = _make_log_data("no severity")
        batch[0].log_record.severity_number = None
        exporter.export(batch)

        assert received[0]["logs"][0]["level"] == "INFO"

    def test_shutdown_is_safe(self):
        """shutdown() should not raise."""
        exporter = RemoteLoggerExporter(
            url="http://localhost:1/log",
            session_id="shutdown-test",
        )
        exporter.shutdown()

    def test_force_flush_returns_true(self):
        """force_flush() should return True."""
        exporter = RemoteLoggerExporter(
            url="http://localhost:1/log",
            session_id="flush-test",
        )
        assert exporter.force_flush() is True

    def test_http_error_returns_failure(self, httpserver):
        """HTTP 500 should return FAILURE."""
        httpserver.expect_request("/log", method="POST").respond_with_data("error", status=500)
        exporter = RemoteLoggerExporter(
            url=httpserver.url_for("/log"),
            session_id="error-test",
        )
        batch = _make_log_data("test")
        result = exporter.export(batch)
        assert result == LogRecordExportResult.FAILURE
