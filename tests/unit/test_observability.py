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


class TestObservabilityEvent:
    def test_constructor_merges_attributes(self):
        from roc.reporting.observability import ObservabilityEvent, roc_common_attributes

        evt = ObservabilityEvent("test.event", body="body", attributes={"custom": "val"})
        assert evt.name == "test.event"
        # Merged attributes should contain both custom and common
        assert "custom" in evt.attributes
        for key in roc_common_attributes:
            assert key in evt.attributes

    def test_constructor_default_attributes(self):
        from roc.reporting.observability import ObservabilityEvent, roc_common_attributes

        evt = ObservabilityEvent("test.event")
        for key in roc_common_attributes:
            assert key in evt.attributes


class TestObservabilityEvent2:
    def test_event_calls_emit(self):
        from roc.reporting.observability import Observability

        mock_logger = MagicMock()
        orig_logger = Observability.event_logger
        Observability.set_event_logger(mock_logger)
        try:
            mock_evt = MagicMock()
            Observability.event(mock_evt)
            mock_logger.emit.assert_called_once_with(mock_evt)
        finally:
            Observability.set_event_logger(orig_logger)


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
