# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/observability.py -- loguru_to_otel coverage."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from opentelemetry._logs import SeverityNumber


class TestLoguruToOtel:
    """Test the loguru_to_otel function (lines 307-361)."""

    def _make_mock_msg(self, level_no=20, exception=None):
        """Create a mock loguru message string with a .record attribute."""
        from datetime import datetime, timezone

        record = {
            "extra": {"custom_key": "custom_val"},
            "file": SimpleNamespace(path="/test/file.py"),
            "function": "test_func",
            "line": 42,
            "module": "test_module",
            "process": SimpleNamespace(id=1234),
            "thread": SimpleNamespace(id=5678, name="MainThread"),
            "time": datetime.now(tz=timezone.utc),
            "level": SimpleNamespace(no=level_no),
            "exception": exception,
            "name": "test_logger",
        }

        # loguru passes a string with a .record attribute
        msg = MagicMock(spec=str)
        msg.strip.return_value = "test log message"
        msg.record = record
        return msg

    def test_basic_log_message(self):
        from roc.reporting.observability import loguru_to_otel

        msg = self._make_mock_msg(level_no=20)

        mock_logger = MagicMock()
        mock_logger_provider = MagicMock()
        mock_logger_provider.get_logger.return_value = mock_logger

        with patch(
            "roc.reporting.observability.otel_logs.get_logger_provider",
            return_value=mock_logger_provider,
        ):
            loguru_to_otel(msg)

        mock_logger.emit.assert_called_once()
        log_record = mock_logger.emit.call_args[0][0]
        assert log_record.body == "test log message"
        assert log_record.severity_text == "INFO"

    def test_noop_logger_skips_emit(self):
        from opentelemetry._logs import NoOpLogger
        from roc.reporting.observability import loguru_to_otel

        msg = self._make_mock_msg(level_no=20)

        mock_noop = MagicMock(spec=NoOpLogger)
        mock_logger_provider = MagicMock()
        mock_logger_provider.get_logger.return_value = mock_noop

        with patch(
            "roc.reporting.observability.otel_logs.get_logger_provider",
            return_value=mock_logger_provider,
        ):
            # Should not raise even though it's NoOp
            loguru_to_otel(msg)

        mock_noop.emit.assert_not_called()

    def test_exception_with_type(self):
        from roc.reporting.observability import loguru_to_otel

        exc = SimpleNamespace(
            type=ValueError,
            value="bad value",
            traceback=None,
        )
        msg = self._make_mock_msg(level_no=40, exception=exc)

        mock_logger = MagicMock()
        mock_logger_provider = MagicMock()
        mock_logger_provider.get_logger.return_value = mock_logger

        with patch(
            "roc.reporting.observability.otel_logs.get_logger_provider",
            return_value=mock_logger_provider,
        ):
            loguru_to_otel(msg)

        log_record = mock_logger.emit.call_args[0][0]
        assert "exception.type" in log_record.attributes
        assert log_record.attributes["exception.type"] == "ValueError"
        assert "exception.message" in log_record.attributes

    def test_exception_with_traceback(self):
        from roc.reporting.observability import loguru_to_otel

        # record["exception"] is unpacked via * in traceback.format_exception,
        # so we need an iterable object that also has .type, .value, .traceback attrs
        exc_type = RuntimeError
        exc_value = RuntimeError("runtime error")
        exc_tb = MagicMock()  # fake traceback object

        class IterableExc:
            type = exc_type
            value = exc_value
            traceback = exc_tb

            def __iter__(self):
                return iter((self.type, self.value, self.traceback))

        exc = IterableExc()
        msg = self._make_mock_msg(level_no=40, exception=exc)

        mock_logger = MagicMock()
        mock_logger_provider = MagicMock()
        mock_logger_provider.get_logger.return_value = mock_logger

        with (
            patch(
                "roc.reporting.observability.otel_logs.get_logger_provider",
                return_value=mock_logger_provider,
            ),
            patch(
                "roc.reporting.observability.traceback.format_exception",
                return_value=["Traceback...\n", "RuntimeError: runtime error\n"],
            ),
        ):
            loguru_to_otel(msg)

        log_record = mock_logger.emit.call_args[0][0]
        assert "exception.stacktrace" in log_record.attributes

    def test_exception_none_fields(self):
        """Exception with all None fields should not add exception attributes."""
        from roc.reporting.observability import loguru_to_otel

        exc = SimpleNamespace(
            type=None,
            value=None,
            traceback=None,
        )
        msg = self._make_mock_msg(level_no=30, exception=exc)

        mock_logger = MagicMock()
        mock_logger_provider = MagicMock()
        mock_logger_provider.get_logger.return_value = mock_logger

        with patch(
            "roc.reporting.observability.otel_logs.get_logger_provider",
            return_value=mock_logger_provider,
        ):
            loguru_to_otel(msg)

        log_record = mock_logger.emit.call_args[0][0]
        assert "exception.type" not in log_record.attributes


class TestLgToOtelSeverityExtra:
    """Additional severity level tests not covered by the existing test file."""

    def test_level_59_max_valid(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(59)
        assert name == "FATAL"
        assert isinstance(sev, SeverityNumber)

    def test_level_1(self):
        from roc.reporting.observability import _lg_to_otel_severity

        _sev, name = _lg_to_otel_severity(1)
        assert name == "TRACE"

    def test_level_9(self):
        from roc.reporting.observability import _lg_to_otel_severity

        _sev, name = _lg_to_otel_severity(9)
        assert name == "TRACE"

    def test_level_35(self):
        from roc.reporting.observability import _lg_to_otel_severity

        _sev, name = _lg_to_otel_severity(35)
        assert name == "WARN"

    def test_level_45(self):
        from roc.reporting.observability import _lg_to_otel_severity

        _sev, name = _lg_to_otel_severity(45)
        assert name == "ERROR"
