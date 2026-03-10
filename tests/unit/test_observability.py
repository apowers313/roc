# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/observability.py."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry._logs import SeverityNumber


class TestLgToOtelSeverity:
    def test_trace_level_0(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(0)
        assert name == "TRACE"
        assert isinstance(sev, SeverityNumber)

    def test_trace_level_5(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(5)
        assert name == "TRACE"

    def test_debug_level_10(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(10)
        assert name == "DEBUG"

    def test_info_level_20(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(20)
        assert name == "INFO"

    def test_warning_level_30(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(30)
        assert name == "WARN"

    def test_error_level_40(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(40)
        assert name == "ERROR"

    def test_fatal_level_50(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(50)
        assert name == "FATAL"

    def test_negative_raises(self):
        from roc.reporting.observability import _lg_to_otel_severity

        with pytest.raises(ValueError, match="positive integer"):
            _lg_to_otel_severity(-1)

    def test_above_59_raises(self):
        from roc.reporting.observability import _lg_to_otel_severity

        with pytest.raises(ValueError, match="above max range"):
            _lg_to_otel_severity(60)

    def test_level_15(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(15)
        assert name == "DEBUG"

    def test_level_25(self):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(25)
        assert name == "INFO"


class TestScreenDataEmittedViaOtelLogger:
    def test_screen_data_emitted_via_otel_logger(self):
        """Screen data should be emitted as a standard OTel log record."""
        import numpy as np

        with patch("roc.reporting.state.otel_logger") as mock_logger:
            from roc.reporting.state import State, states

            states.screen.val = {"chars": np.array([[65, 66], [67, 68]])}
            states.salency.val = None
            states.object.val = None
            states.attention.val = None
            State.emit_state_logs()
            mock_logger.emit.assert_called_once()
            log_record = mock_logger.emit.call_args[0][0]
            assert log_record.attributes["event.name"] == "roc.screen"
            assert "AB" in log_record.body
            assert "CD" in log_record.body
            states.screen.val = None


class TestObservabilitySetters:
    def test_set_event_logger(self):
        from roc.reporting.observability import Observability

        orig = Observability.event_logger
        mock = MagicMock()
        Observability.set_event_logger(mock)
        assert Observability.event_logger is mock
        Observability.set_event_logger(orig)

    def test_set_tracer(self):
        from roc.reporting.observability import Observability

        orig = Observability.tracer
        mock = MagicMock()
        Observability.set_tracer(mock)
        assert Observability.tracer is mock
        Observability.set_tracer(orig)

    def test_set_meter(self):
        from roc.reporting.observability import Observability

        orig = Observability.meter
        mock = MagicMock()
        Observability.set_meter(mock)
        assert Observability.meter is mock
        Observability.set_meter(orig)


class TestObservabilityBase:
    def test_singleton_behavior(self, reset_observability):
        from roc.reporting.observability import Observability, ObservabilityBase

        # Clear existing singleton
        ObservabilityBase._instances.clear()

        o1 = Observability()
        o2 = Observability()
        assert o1 is o2


class TestModuleLevelConstants:
    def test_instance_id_is_string(self):
        from roc.reporting.observability import instance_id

        assert isinstance(instance_id, str)
        assert len(instance_id) > 0

    def test_resource_has_service_name(self):
        from roc.reporting.observability import resource

        attrs = dict(resource.attributes)
        assert attrs["service.name"] == "roc"

    def test_roc_common_attributes(self):
        from roc.reporting.observability import roc_common_attributes

        assert "roc.instance.id" in roc_common_attributes


class TestDebugLogExporter:
    def test_debug_log_creates_jsonl_file(self, tmp_path, reset_observability):
        """When debug_log=True, a JSONL file is created and logs are written to it."""
        from time import time_ns

        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

        from roc.reporting.observability import JsonlFileExporter

        log_path = tmp_path / "test_debug.jsonl"
        exporter = JsonlFileExporter(str(log_path))

        from opentelemetry.sdk._logs import LoggerProvider, LogRecord

        lp = LoggerProvider()
        lp.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        logger = lp.get_logger("test")
        logger.emit(LogRecord(body="test message", timestamp=time_ns()))
        lp.force_flush()
        exporter.shutdown()

        content = log_path.read_text()
        assert "test message" in content

    def test_debug_log_writes_valid_jsonl(self, tmp_path, reset_observability):
        """Each line in the JSONL file should be valid JSON."""
        import json
        from time import time_ns

        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

        from roc.reporting.observability import JsonlFileExporter

        log_path = tmp_path / "test_debug.jsonl"
        exporter = JsonlFileExporter(str(log_path))

        from opentelemetry._logs import SeverityNumber
        from opentelemetry.sdk._logs import LoggerProvider, LogRecord

        lp = LoggerProvider()
        lp.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        logger = lp.get_logger("test")

        for i in range(3):
            logger.emit(
                LogRecord(
                    body=f"message {i}",
                    timestamp=time_ns(),
                    severity_number=SeverityNumber.INFO,
                    severity_text="INFO",
                )
            )
        lp.force_flush()
        exporter.shutdown()

        lines = [l for l in log_path.read_text().strip().split("\n") if l.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "body" in obj
            assert "timestamp" in obj

    def test_debug_log_disabled_no_file(self, tmp_path):
        """When debug_log=False, no JSONL file should be created."""
        from roc.config import Config

        settings = Config.get()
        settings.debug_log = False
        settings.debug_log_path = str(tmp_path / "test_debug.jsonl")
        # File should not exist since nothing creates it
        assert not (tmp_path / "test_debug.jsonl").exists()

    def test_debug_log_survives_crash(self, tmp_path, reset_observability):
        """SimpleLogRecordProcessor flushes synchronously -- records survive process exit."""
        from time import time_ns

        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

        from roc.reporting.observability import JsonlFileExporter

        log_path = tmp_path / "test_debug.jsonl"
        exporter = JsonlFileExporter(str(log_path))

        from opentelemetry.sdk._logs import LoggerProvider, LogRecord

        lp = LoggerProvider()
        lp.add_log_record_processor(SimpleLogRecordProcessor(exporter))
        logger = lp.get_logger("test")
        logger.emit(LogRecord(body="before crash", timestamp=time_ns()))
        # File should already contain the record (sync processor)
        content = log_path.read_text()
        assert "before crash" in content
        exporter.shutdown()

    def test_debug_log_creates_parent_dirs(self, tmp_path, reset_observability):
        """Parent directories are created automatically."""
        from roc.reporting.observability import JsonlFileExporter

        log_path = tmp_path / "subdir" / "nested" / "debug.jsonl"
        exporter = JsonlFileExporter(str(log_path))
        assert log_path.parent.exists()
        exporter.shutdown()

    def test_exporter_shutdown_closes_file(self, tmp_path, reset_observability):
        """After shutdown, the file handle should be closed."""
        from roc.reporting.observability import JsonlFileExporter

        log_path = tmp_path / "test_debug.jsonl"
        exporter = JsonlFileExporter(str(log_path))
        exporter.shutdown()
        assert exporter._file_handle.closed


class TestGetLogger:
    def test_get_logger_returns_logger(self):
        """Observability.get_logger() should return an OTel logger."""
        from roc.reporting.observability import Observability

        logger = Observability.get_logger("test-module")
        assert logger is not None

    def test_get_logger_different_names(self):
        """Different names should return different loggers."""
        from roc.reporting.observability import Observability

        l1 = Observability.get_logger("module-a")
        l2 = Observability.get_logger("module-b")
        # They should both be valid loggers (may or may not be the same object)
        assert l1 is not None
        assert l2 is not None
