# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/reporting/observability.py."""

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry._logs import SeverityNumber


class TestLgToOtelSeverity:
    @pytest.mark.parametrize(
        ("level", "expected_name"),
        [
            (0, "TRACE"),
            (5, "TRACE"),
            (10, "DEBUG"),
            (15, "DEBUG"),
            (20, "INFO"),
            (25, "INFO"),
            (30, "WARN"),
            (40, "ERROR"),
            (50, "FATAL"),
        ],
    )
    def test_severity_mapping(self, level, expected_name):
        from roc.reporting.observability import _lg_to_otel_severity

        sev, name = _lg_to_otel_severity(level)
        assert name == expected_name
        assert isinstance(sev, SeverityNumber)

    def test_negative_raises(self):
        from roc.reporting.observability import _lg_to_otel_severity

        with pytest.raises(ValueError, match="positive integer"):
            _lg_to_otel_severity(-1)

    def test_above_59_raises(self):
        from roc.reporting.observability import _lg_to_otel_severity

        with pytest.raises(ValueError, match="above max range"):
            _lg_to_otel_severity(60)


class TestScreenDataEmittedViaOtelLogger:
    def test_screen_data_emitted_via_otel_logger(self):
        """Screen data should be emitted as a standard OTel log record in JSON format."""
        import json

        import numpy as np

        from roc.framework.config import Config

        cfg = Config.get()
        cfg.emit_state_screen = True

        with patch("roc.reporting.state._get_otel_logger") as mock_logger:
            from roc.framework.event import Event
            from roc.reporting.state import State, states

            Event._step_counts.clear()
            states.screen.val = {
                "chars": np.array([[65, 66], [67, 68]]),
                "colors": np.array([[7, 7], [7, 7]]),
            }
            states.salency.val = None
            states.object.val = None
            states.attention.val = None
            State.emit_state_logs()
            # screen + graphdb.summary (no event.summary since step_counts cleared)
            assert mock_logger.return_value.emit.call_count == 2
            log_record = mock_logger.return_value.emit.call_args_list[0][0][0]
            assert log_record.attributes["event.name"] == "roc.screen"
            body = json.loads(log_record.body)
            assert body["chars"] == [[65, 66], [67, 68]]
            assert "fg" in body
            assert "bg" in body
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


class TestInitOtlpLogging:
    """Tests for _init_otlp_logging (lines 144-153)."""

    def test_creates_logger_provider_with_processors(self, reset_observability):
        """_init_otlp_logging should create a LoggerProvider and add batch processor."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()
        settings = Config.get()
        settings.debug_remote_log = False

        with (
            patch("roc.reporting.observability.OTLPLogExporter") as mock_exporter_cls,
            patch("roc.reporting.observability.BatchLogRecordProcessor") as mock_batch,
            patch("roc.reporting.observability.LoggerProvider") as mock_provider_cls,
        ):
            mock_provider = MagicMock()
            mock_provider_cls.return_value = mock_provider

            obs = object.__new__(Observability)
            result = obs._init_otlp_logging(settings)

            mock_exporter_cls.assert_called_once_with(
                endpoint=settings.observability_host, insecure=True
            )
            mock_batch.assert_called_once()
            mock_provider.add_log_record_processor.assert_called()
            assert result is mock_provider


class TestInitFallbackLogging:
    """Tests for _init_fallback_logging (lines 155-162)."""

    def test_fallback_logging_not_pytest(self, reset_observability):
        """When not in pytest scanning mode, fallback creates LoggerProvider."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = False
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            with (
                patch("roc.reporting.observability.LoggerProvider") as mock_provider_cls,
                patch("roc.reporting.observability.otel_logs") as mock_otel_logs,
            ):
                mock_provider = MagicMock()
                mock_provider_cls.return_value = mock_provider

                obs = object.__new__(Observability)
                obs._init_fallback_logging(settings)

                mock_provider_cls.assert_called_once()
                mock_otel_logs.set_logger_provider.assert_called_once_with(
                    logger_provider=mock_provider
                )
        finally:
            obs_mod._disable_for_pytest_scanning = orig

    def test_fallback_logging_sets_noop_event_logger(self, reset_observability):
        """Fallback logging always sets a NoOpEventLogger."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()

        with patch.object(Observability, "set_event_logger") as mock_set:
            obs = object.__new__(Observability)
            obs._init_fallback_logging(settings)

            mock_set.assert_called_once()
            from opentelemetry._events import NoOpEventLogger

            assert isinstance(mock_set.call_args[0][0], NoOpEventLogger)


class TestAttachRemoteLogExporter:
    """Tests for _attach_remote_log_exporter (lines 164-179)."""

    def test_skips_when_debug_remote_log_false(self, reset_observability):
        """Should return early when debug_remote_log is False."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = False

        mock_provider = MagicMock()
        obs = object.__new__(Observability)
        obs._attach_remote_log_exporter(settings, mock_provider)

        mock_provider.add_log_record_processor.assert_not_called()

    def test_attaches_when_debug_remote_log_true(self, reset_observability):
        """Should create RemoteLoggerExporter and add processor when enabled."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = True
        settings.debug_remote_log_url = "http://test:9080/log"

        mock_provider = MagicMock()
        obs = object.__new__(Observability)
        Observability._remote_log_configured = False

        with (
            patch(
                "roc.reporting.observability.RemoteLoggerExporter",
                create=True,
            ) as mock_exporter_cls,
            patch("roc.reporting.observability.SimpleLogRecordProcessor") as _mock_proc,
        ):
            # Patch the import inside the method
            with patch.dict(
                "sys.modules",
                {
                    "roc.reporting.remote_logger_exporter": MagicMock(
                        RemoteLoggerExporter=mock_exporter_cls
                    )
                },
            ):
                obs._attach_remote_log_exporter(settings, mock_provider)

            mock_provider.add_log_record_processor.assert_called_once()
            assert Observability._remote_log_configured is True


class TestAttachParquetExporter:
    """Tests for _attach_parquet_exporter (lines 181-193)."""

    def test_skips_when_allow_parquet_false(self, reset_observability):
        """Should return early when _allow_parquet is False."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        Observability._allow_parquet = False

        mock_provider = MagicMock()
        obs = object.__new__(Observability)
        obs._attach_parquet_exporter(settings, mock_provider)

        mock_provider.add_log_record_processor.assert_not_called()

    def test_attaches_when_allow_parquet_true(self, reset_observability):
        """Should create DuckLakeStore and ParquetExporter when _allow_parquet is True."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        Observability._allow_parquet = True
        Observability._parquet_configured = False

        mock_provider = MagicMock()
        obs = object.__new__(Observability)

        with (
            patch("roc.reporting.observability.SimpleLogRecordProcessor") as _mock_proc,
            patch(
                "roc.reporting.ducklake_store.DuckLakeStore",
            ) as _mock_store_cls,
            patch(
                "roc.reporting.observability.ParquetExporter",
            ) as _mock_parquet_cls,
        ):
            obs._attach_parquet_exporter(settings, mock_provider)

            mock_provider.add_log_record_processor.assert_called_once()
            assert Observability._parquet_configured is True
            assert hasattr(obs, "_ducklake_store")
            assert hasattr(obs, "_parquet_exporter")

        # Cleanup
        Observability._allow_parquet = False
        Observability._parquet_configured = False


class TestAddLoguruSinks:
    """Tests for _add_loguru_sinks (lines 195-209)."""

    def test_adds_loguru_sink(self, reset_observability):
        """Should add loguru_to_otel sink."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.dashboard_enabled = False
        settings.dashboard_callback_url = None

        obs = object.__new__(Observability)

        with patch("roc.reporting.observability.roc_logger") as mock_logger:
            obs._add_loguru_sinks(settings)

            mock_logger.logger.add.assert_called_once()

    def test_adds_dashboard_sink_when_enabled(self, reset_observability):
        """Should add step_log_sink when dashboard_enabled is True."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.dashboard_enabled = True

        obs = object.__new__(Observability)

        with (
            patch("roc.reporting.observability.roc_logger") as mock_logger,
            patch(
                "roc.reporting.step_log_sink.step_log_sink",
                create=True,
            ),
        ):
            obs._add_loguru_sinks(settings)

            # Should be called twice: once for loguru_to_otel, once for step_log_sink
            assert mock_logger.logger.add.call_count == 2

    def test_adds_dashboard_sink_when_callback_url_set(self, reset_observability):
        """Should add step_log_sink when dashboard_callback_url is set."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.dashboard_enabled = False
        settings.dashboard_callback_url = "http://localhost:8000/callback"

        obs = object.__new__(Observability)

        with (
            patch("roc.reporting.observability.roc_logger") as mock_logger,
            patch(
                "roc.reporting.step_log_sink.step_log_sink",
                create=True,
            ),
        ):
            obs._add_loguru_sinks(settings)

            assert mock_logger.logger.add.call_count == 2


class TestInitEventLogger:
    """Tests for _init_event_logger (lines 211-222)."""

    def test_initializes_event_logger(self, reset_observability):
        """Should create EventLoggerProvider and set the event logger."""
        from roc.reporting.observability import Observability

        mock_provider = MagicMock()
        obs = object.__new__(Observability)

        with (
            patch("roc.reporting.observability.EventLoggerProvider") as mock_elp_cls,
            patch("roc.reporting.observability.otel_events") as mock_events,
            patch("roc.reporting.observability.roc_logger"),
            patch.object(Observability, "set_event_logger") as mock_set,
        ):
            mock_elp = MagicMock()
            mock_elp_cls.return_value = mock_elp

            obs._init_event_logger(mock_provider)

            mock_elp_cls.assert_called_once_with(logger_provider=mock_provider)
            mock_elp.get_event_logger.assert_called_once()
            mock_set.assert_called_once()
            mock_events.set_event_logger_provider.assert_called_once()


class TestInitMetrics:
    """Tests for _init_metrics (lines 224-249)."""

    def test_initializes_metrics_when_enabled(self, reset_observability):
        """Should create MeterProvider and set it when observability_metrics is True."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_metrics = True
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            obs = object.__new__(Observability)

            with (
                patch("roc.reporting.observability.OTLPMetricExporter") as mock_exp_cls,
                patch(
                    "roc.reporting.observability.PeriodicExportingMetricReader"
                ) as mock_reader_cls,
                patch("roc.reporting.observability.MeterProvider") as mock_mp_cls,
                patch("roc.reporting.observability.otel_metrics") as mock_metrics,
                patch("roc.reporting.observability.SystemMetricsInstrumentor") as mock_smi,
                patch("roc.reporting.observability.roc_logger"),
            ):
                mock_mp = MagicMock()
                mock_mp_cls.return_value = mock_mp
                mock_metrics.get_meter_provider.return_value = mock_mp

                obs._init_metrics(settings)

                mock_exp_cls.assert_called_once()
                mock_reader_cls.assert_called_once()
                mock_mp_cls.assert_called_once()
                mock_metrics.set_meter_provider.assert_called_once_with(mock_mp)
                mock_smi.assert_called_once()
        finally:
            obs_mod._disable_for_pytest_scanning = orig

    def test_skips_metrics_when_disabled(self, reset_observability):
        """Should use NoOp meter when observability_metrics is False."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_metrics = False

        obs = object.__new__(Observability)

        with (
            patch("roc.reporting.observability.otel_metrics") as mock_metrics,
            patch.object(Observability, "set_meter") as mock_set,
        ):
            mock_mp = MagicMock()
            mock_metrics.get_meter_provider.return_value = mock_mp

            obs._init_metrics(settings)

            mock_mp.get_meter.assert_called_once()
            mock_set.assert_called_once()


class TestInitTracing:
    """Tests for _init_tracing (lines 251-270)."""

    def test_initializes_tracing_when_enabled(self, reset_observability):
        """Should create TracerProvider and set it when observability_tracing is True."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_tracing = True
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            obs = object.__new__(Observability)

            with (
                patch("roc.reporting.observability.OTLPSpanExporter") as mock_exp_cls,
                patch("roc.reporting.observability.TracerProvider") as mock_tp_cls,
                patch("roc.reporting.observability.BatchSpanProcessor") as mock_bsp,
                patch("roc.reporting.observability.otel_trace") as mock_trace,
                patch("roc.reporting.observability.roc_logger"),
            ):
                mock_tp = MagicMock()
                mock_tp_cls.return_value = mock_tp
                mock_trace.get_tracer_provider.return_value = mock_tp

                obs._init_tracing(settings)

                mock_exp_cls.assert_called_once()
                mock_tp_cls.assert_called_once()
                mock_bsp.assert_called_once()
                mock_tp.add_span_processor.assert_called_once()
                mock_trace.set_tracer_provider.assert_called_once_with(mock_tp)
        finally:
            obs_mod._disable_for_pytest_scanning = orig


class TestInitProfiling:
    """Tests for _init_profiling (lines 272-287)."""

    def test_initializes_profiling_when_enabled(self, reset_observability):
        """Should call pyroscope.configure when profiling is enabled."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_profiling = True
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            with (
                patch("roc.reporting.observability.pyroscope") as mock_pyroscope,
                patch("roc.reporting.observability.roc_logger"),
            ):
                Observability._init_profiling(settings)

                mock_pyroscope.configure.assert_called_once()
                call_kwargs = mock_pyroscope.configure.call_args[1]
                assert call_kwargs["application_name"] == "roc"
                assert call_kwargs["server_address"] == settings.observability_profiling_host
        finally:
            obs_mod._disable_for_pytest_scanning = orig

    def test_skips_profiling_when_disabled(self, reset_observability):
        """Should not call pyroscope.configure when profiling is disabled."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_profiling = False

        with patch("roc.reporting.observability.pyroscope") as mock_pyroscope:
            Observability._init_profiling(settings)

            mock_pyroscope.configure.assert_not_called()


class TestObservabilityInit:
    """Tests for Observability.__init__ (lines 124-132)."""

    def test_init_calls_all_init_methods(self, reset_observability):
        """__init__ should call all four initialization methods."""
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()

        with (
            patch.object(Observability, "_init_logging") as mock_log,
            patch.object(Observability, "_init_metrics") as mock_met,
            patch.object(Observability, "_init_tracing") as mock_trace,
            patch.object(Observability, "_init_profiling") as mock_prof,
            patch("roc.reporting.observability.roc_logger"),
        ):
            Observability()

            mock_log.assert_called_once()
            mock_met.assert_called_once()
            mock_trace.assert_called_once()
            mock_prof.assert_called_once()


class TestInitLogging:
    """Tests for _init_logging (lines 134-142)."""

    def test_init_logging_otlp_path(self, reset_observability):
        """When observability_logging is True and not pytest, should use OTLP path."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_logging = True
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            obs = object.__new__(Observability)

            with (
                patch.object(obs, "_init_otlp_logging") as mock_otlp,
                patch("roc.reporting.observability.otel_logs") as mock_logs,
                patch.object(obs, "_add_loguru_sinks") as mock_sinks,
                patch.object(obs, "_init_event_logger") as mock_event,
            ):
                mock_provider = MagicMock()
                mock_otlp.return_value = mock_provider

                obs._init_logging(settings)

                mock_otlp.assert_called_once_with(settings)
                mock_logs.set_logger_provider.assert_called_once_with(logger_provider=mock_provider)
                mock_sinks.assert_called_once_with(settings)
                mock_event.assert_called_once_with(mock_provider)
        finally:
            obs_mod._disable_for_pytest_scanning = orig

    def test_init_logging_fallback_path(self, reset_observability):
        """When observability_logging is False, should use fallback path."""
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.observability_logging = False

        obs = object.__new__(Observability)

        with patch.object(obs, "_init_fallback_logging") as mock_fallback:
            obs._init_logging(settings)

            mock_fallback.assert_called_once_with(settings)


class TestObservabilityStaticInit:
    """Tests for Observability.init() static method (lines 290-306)."""

    def test_init_with_enable_parquet(self, reset_observability):
        """Observability.init(enable_parquet=True) should set _allow_parquet and call _init_parquet."""
        from roc.reporting.observability import Observability, ObservabilityBase

        Observability._allow_parquet = False
        Observability._parquet_configured = False

        with (
            patch.object(ObservabilityBase, "__call__", return_value=MagicMock()),
            patch.object(Observability, "_init_parquet") as mock_parquet,
            patch.object(Observability, "_configure_debug_exporters"),
        ):
            Observability.init(enable_parquet=True)

            assert Observability._allow_parquet is True
            mock_parquet.assert_called_once()

        # Cleanup
        Observability._allow_parquet = False

    def test_init_without_enable_parquet(self, reset_observability):
        """Observability.init() without enable_parquet should not call _init_parquet."""
        from roc.reporting.observability import Observability, ObservabilityBase

        with (
            patch.object(ObservabilityBase, "__call__", return_value=MagicMock()),
            patch.object(Observability, "_init_parquet") as mock_parquet,
            patch.object(Observability, "_configure_debug_exporters"),
        ):
            Observability.init(enable_parquet=False)

            mock_parquet.assert_not_called()

    def test_init_skips_parquet_when_already_configured(self, reset_observability):
        """Should not call _init_parquet if _parquet_configured is already True."""
        from roc.reporting.observability import Observability, ObservabilityBase

        Observability._allow_parquet = False
        Observability._parquet_configured = True

        with (
            patch.object(ObservabilityBase, "__call__", return_value=MagicMock()),
            patch.object(Observability, "_init_parquet") as mock_parquet,
            patch.object(Observability, "_configure_debug_exporters"),
        ):
            Observability.init(enable_parquet=True)

            mock_parquet.assert_not_called()

        # Cleanup
        Observability._allow_parquet = False
        Observability._parquet_configured = False


class TestInitParquet:
    """Tests for Observability._init_parquet (lines 308-331)."""

    def test_init_parquet_creates_store_and_exporter(self, reset_observability):
        """Should create DuckLakeStore, ParquetExporter, and add processor."""
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_instance = MagicMock()
        ObservabilityBase._instances[Observability] = mock_instance
        Observability._parquet_configured = False

        mock_provider = MagicMock()
        mock_provider.add_log_record_processor = MagicMock()

        with (
            patch("roc.reporting.ducklake_store.DuckLakeStore") as mock_store_cls,
            patch("roc.reporting.observability.ParquetExporter") as mock_parquet_cls,
            patch(
                "roc.reporting.observability.otel_logs.get_logger_provider",
                return_value=mock_provider,
            ),
        ):
            Observability._init_parquet()

            mock_store_cls.assert_called_once()
            mock_parquet_cls.assert_called_once()
            mock_provider.add_log_record_processor.assert_called_once()
            assert Observability._parquet_configured is True

        # Cleanup
        Observability._parquet_configured = False

    def test_init_parquet_returns_early_if_no_instance(self, reset_observability):
        """Should return early if no singleton instance exists."""
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()
        Observability._parquet_configured = False

        with patch("roc.reporting.ducklake_store.DuckLakeStore") as mock_store_cls:
            Observability._init_parquet()

            mock_store_cls.assert_not_called()
            assert Observability._parquet_configured is False

    def test_init_parquet_skips_processor_if_no_add_method(self, reset_observability):
        """Should skip adding processor if provider lacks add_log_record_processor."""
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_instance = MagicMock()
        ObservabilityBase._instances[Observability] = mock_instance
        Observability._parquet_configured = False

        # Provider without add_log_record_processor attribute
        mock_provider = MagicMock(spec=[])

        with (
            patch("roc.reporting.ducklake_store.DuckLakeStore") as mock_store_cls,
            patch("roc.reporting.observability.ParquetExporter") as mock_parquet_cls,
            patch(
                "roc.reporting.observability.otel_logs.get_logger_provider",
                return_value=mock_provider,
            ),
        ):
            Observability._init_parquet()

            mock_store_cls.assert_called_once()
            mock_parquet_cls.assert_called_once()
            # Provider does not have add_log_record_processor, so no call
            assert not hasattr(mock_provider, "add_log_record_processor")
            assert Observability._parquet_configured is True

        # Cleanup
        Observability._parquet_configured = False


class TestGetDucklakeStore:
    """Tests for Observability.get_ducklake_store (lines 333-339)."""

    def test_returns_store_when_configured(self, reset_observability):
        """Should return the DuckLakeStore if it exists on the instance."""
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_instance = MagicMock()
        mock_store = MagicMock()
        mock_instance._ducklake_store = mock_store
        ObservabilityBase._instances[Observability] = mock_instance

        result = Observability.get_ducklake_store()
        assert result is mock_store

    def test_returns_none_when_no_instance(self, reset_observability):
        """Should return None if no singleton instance exists."""
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()

        result = Observability.get_ducklake_store()
        assert result is None

    def test_returns_none_when_no_ducklake_store(self, reset_observability):
        """Should return None if instance has no _ducklake_store attribute."""
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_instance = MagicMock(spec=[])
        ObservabilityBase._instances[Observability] = mock_instance

        result = Observability.get_ducklake_store()
        assert result is None


class TestShutdown:
    """Tests for Observability.shutdown (lines 341-353)."""

    def test_shutdown_with_parquet_exporter_and_store(self, reset_observability):
        """Should call shutdown on parquet exporter and close on ducklake store."""
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_instance = MagicMock()
        mock_instance._parquet_exporter = MagicMock()
        mock_instance._ducklake_store = MagicMock()
        ObservabilityBase._instances[Observability] = mock_instance

        with patch("roc.reporting.observability.pyroscope") as mock_pyroscope:
            Observability.shutdown()

            mock_instance._parquet_exporter.shutdown.assert_called_once()
            mock_instance._ducklake_store.close.assert_called_once()
            mock_pyroscope.shutdown.assert_called_once()

    def test_shutdown_without_parquet_or_store(self, reset_observability):
        """Should handle instance without parquet exporter or ducklake store."""
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_instance = MagicMock(spec=[])
        ObservabilityBase._instances[Observability] = mock_instance

        with patch("roc.reporting.observability.pyroscope") as mock_pyroscope:
            Observability.shutdown()

            mock_pyroscope.shutdown.assert_called_once()

    def test_shutdown_no_instance(self, reset_observability):
        """Should handle missing instance gracefully."""
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()

        with patch("roc.reporting.observability.pyroscope") as mock_pyroscope:
            Observability.shutdown()

            mock_pyroscope.shutdown.assert_called_once()

    def test_shutdown_pyroscope_exception(self, reset_observability):
        """Should swallow pyroscope.shutdown() exceptions."""
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()

        with patch("roc.reporting.observability.pyroscope") as mock_pyroscope:
            mock_pyroscope.shutdown.side_effect = RuntimeError("pyroscope error")
            # Should not raise
            Observability.shutdown()


class TestReset:
    """Tests for Observability.reset() per-game cleanup.

    Regression coverage for the unified server's back-to-back game flow:
    without reset(), the second game hits `ConnectionException: Connection
    already closed` when the old parquet exporter tries to write to the
    first game's closed DuckLakeStore, and the second game also reuses
    the first game's instance_id as its run directory name.
    """

    @pytest.fixture
    def full_state_reset(self):
        """Save/restore all state that Observability.reset() touches."""
        import roc.reporting.observability as obs_mod
        from roc.reporting.observability import Observability, ObservabilityBase

        orig_instances = ObservabilityBase._instances.copy()
        orig_parquet = Observability._parquet_configured
        orig_remote = Observability._remote_log_configured
        orig_allow = Observability._allow_parquet
        orig_instance_id = obs_mod.instance_id
        orig_resource = obs_mod.resource
        orig_common = obs_mod.roc_common_attributes
        yield
        ObservabilityBase._instances = orig_instances
        Observability._parquet_configured = orig_parquet
        Observability._remote_log_configured = orig_remote
        Observability._allow_parquet = orig_allow
        obs_mod.instance_id = orig_instance_id
        obs_mod.resource = orig_resource
        obs_mod.roc_common_attributes = orig_common

    def test_reset_regenerates_instance_id(self, full_state_reset):
        """reset() must produce a fresh instance_id so the next game has
        its own run directory.
        """
        import roc.reporting.observability as obs_mod
        from roc.reporting.observability import Observability

        before = obs_mod.instance_id
        Observability.reset()
        after = obs_mod.instance_id
        assert after != before
        assert isinstance(after, str)
        assert len(after) > 0

    def test_reset_rebuilds_resource_and_common_attributes(self, full_state_reset):
        """The OTel resource and roc_common_attributes must reflect the
        new instance_id -- stale values would tag future logs with the
        previous run's ID.
        """
        import roc.reporting.observability as obs_mod
        from roc.reporting.observability import Observability

        Observability.reset()
        resource_attrs = dict(obs_mod.resource.attributes)
        assert resource_attrs["service.instance.id"] == obs_mod.instance_id
        assert obs_mod.roc_common_attributes["roc.instance.id"] == obs_mod.instance_id

    def test_reset_preserves_singleton_and_clears_parquet_flags(self, full_state_reset):
        """After reset(), the singleton is preserved but per-game flags are
        cleared so the next ``init`` calls ``_init_parquet`` to attach a
        fresh exporter to the shared provider.

        The singleton intentionally survives because OTel's
        ``set_logger_provider`` is ``Once``-protected: any replacement
        provider installed after ``reset`` would be silently ignored,
        leaving the second game's records flowing into the shut-down
        first-game provider.
        """
        from roc.reporting.observability import Observability, ObservabilityBase

        # Seed a fake singleton and flip the flags
        mock_instance = MagicMock()
        mock_instance._parquet_exporter = MagicMock()
        mock_instance._ducklake_store = MagicMock()
        ObservabilityBase._instances[Observability] = mock_instance
        Observability._parquet_configured = True
        Observability._remote_log_configured = True
        Observability._allow_parquet = True

        Observability.reset()

        # Singleton is preserved across reset()
        assert Observability in ObservabilityBase._instances
        # Parquet flags are cleared so next init() rebuilds the exporter
        assert Observability._parquet_configured is False
        assert Observability._allow_parquet is False
        # Remote-log flag stays set because the exporter is server-wide
        assert Observability._remote_log_configured is True

    def test_reset_closes_parquet_exporter_and_store(self, full_state_reset):
        """reset() must shut down the parquet exporter and close the
        DuckLakeStore so the next game opens a fresh connection.

        Uses a real ``Observability`` instance via ``object.__new__``
        (bypassing the heavy ``__init__`` side effects) so that
        attribute-deletion semantics are real -- ``MagicMock.hasattr``
        always returns True, which would mask the ``del instance._x``
        cleanup that ``reset()`` performs.
        """
        from roc.reporting.observability import Observability, ObservabilityBase

        mock_exporter = MagicMock()
        mock_store = MagicMock()
        mock_processor = MagicMock()

        instance = object.__new__(Observability)
        instance._parquet_exporter = mock_exporter
        instance._ducklake_store = mock_store
        instance._parquet_processor = mock_processor
        ObservabilityBase._instances[Observability] = instance

        Observability.reset()

        mock_exporter.shutdown.assert_called_once()
        mock_store.close.assert_called_once()
        mock_processor.shutdown.assert_called_once()
        # Attributes are deleted after cleanup so the next _init_parquet
        # starts from a clean slate. We rely on a real object here
        # because MagicMock.hasattr always returns True.
        assert not hasattr(instance, "_parquet_exporter")
        assert not hasattr(instance, "_ducklake_store")
        assert not hasattr(instance, "_parquet_processor")

    def test_reset_does_not_shut_down_logger_provider(self, full_state_reset):
        """reset() must NOT shut down the active logger provider.

        OTel's ``set_logger_provider`` is ``Once``-protected. Shutting down
        the provider and trying to install a replacement would leave the
        process with a dead provider because the second call to
        ``set_logger_provider`` is silently ignored. Instead, reset()
        leaves the provider running and only detaches the per-game
        parquet processor.
        """
        from roc.reporting.observability import Observability

        mock_provider = MagicMock()
        with patch("roc.reporting.observability.otel_logs") as mock_logs:
            mock_logs.get_logger_provider.return_value = mock_provider
            Observability.reset()
            mock_provider.shutdown.assert_not_called()

    def test_reset_removes_parquet_processor_from_provider(self, full_state_reset):
        """reset() removes the per-game parquet processor from the shared
        provider so the next game can attach a fresh one without
        accumulating dead processors.

        Uses a real ``LoggerProvider`` (from the OTel SDK) with real
        processors so the removal path exercises the actual
        ``_multi_log_record_processor._log_record_processors`` tuple
        manipulation, not a mocked stand-in that could pass even if the
        real internal structure changed.
        """
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import (
            ConsoleLogExporter,
            SimpleLogRecordProcessor,
        )

        from roc.reporting.observability import Observability, ObservabilityBase

        # Real LoggerProvider with two real processors: one "server-wide"
        # (to preserve) and one "per-game parquet" (to remove).
        logger_provider = LoggerProvider()
        other_processor = SimpleLogRecordProcessor(ConsoleLogExporter())
        parquet_processor = SimpleLogRecordProcessor(ConsoleLogExporter())
        logger_provider.add_log_record_processor(other_processor)
        logger_provider.add_log_record_processor(parquet_processor)
        # Sanity check: both processors are attached before reset.
        assert other_processor in logger_provider._multi_log_record_processor._log_record_processors
        assert (
            parquet_processor in logger_provider._multi_log_record_processor._log_record_processors
        )

        instance = object.__new__(Observability)
        instance._parquet_exporter = MagicMock()
        instance._ducklake_store = MagicMock()
        instance._parquet_processor = parquet_processor
        ObservabilityBase._instances[Observability] = instance

        with patch("roc.reporting.observability.otel_logs") as mock_logs:
            mock_logs.get_logger_provider.return_value = logger_provider
            Observability.reset()

        # The parquet processor is gone; the unrelated processor stays.
        remaining = logger_provider._multi_log_record_processor._log_record_processors
        assert parquet_processor not in remaining
        assert other_processor in remaining

    def test_reset_swallows_exporter_shutdown_errors(self, full_state_reset):
        """A broken exporter must not prevent reset() from completing."""
        from roc.reporting.observability import Observability, ObservabilityBase

        instance = object.__new__(Observability)
        instance._parquet_exporter = MagicMock()
        instance._parquet_exporter.shutdown.side_effect = RuntimeError("boom")
        instance._ducklake_store = MagicMock()
        instance._ducklake_store.close.side_effect = RuntimeError("boom2")
        instance._parquet_processor = MagicMock()
        instance._parquet_processor.shutdown.side_effect = RuntimeError("boom3")
        ObservabilityBase._instances[Observability] = instance

        # Must not raise
        Observability.reset()
        # Singleton survives despite errors; flags are cleared
        assert Observability in ObservabilityBase._instances
        assert Observability._parquet_configured is False
        assert Observability._allow_parquet is False

    def test_reset_handles_missing_instance(self, full_state_reset):
        """reset() with no singleton present is a valid no-op path."""
        from roc.reporting.observability import Observability, ObservabilityBase

        ObservabilityBase._instances.clear()
        # Must not raise
        Observability.reset()

    def test_sequential_games_attach_fresh_parquet_to_shared_provider(
        self, full_state_reset, tmp_path
    ):
        """Regression for the UAT sequential-game failure.

        The original bug: after the first game's ``reset()``, the second
        game's ``Observability.init(enable_parquet=True)`` created a new
        ``LoggerProvider`` and tried to install it via
        ``otel_logs.set_logger_provider``. OTel's ``Once`` guard silently
        ignored the replacement, so the second game's parquet exporter
        was wired to a provider that ``get_logger_provider`` never
        returns. Every log record kept flowing into the shut-down
        first-game provider and the second game's DuckLake catalog
        stayed empty -- the "0g, N steps" inconsistency the UAT report
        flagged as Critical/D-01.

        The fix: keep the singleton and shared provider alive across
        games; only recycle the per-game parquet store + exporter +
        processor via ``reset`` / ``_init_parquet``. This test walks
        that sequence using REAL OTel ``LoggerProvider``, REAL
        ``DuckLakeStore``, and REAL ``ParquetExporter`` instances --
        only ``settings.data_dir`` is redirected to ``tmp_path`` so the
        test does not touch the real data directory. Every assertion
        fires against real object state. A mock-based version would
        pass even if the LoggerProvider internals changed underneath us;
        this one fails loudly if they do.
        """
        import roc.reporting.observability as obs_mod
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import (
            ConsoleLogExporter,
            SimpleLogRecordProcessor,
        )

        from roc.framework.config import Config
        from roc.reporting.ducklake_store import DuckLakeStore
        from roc.reporting.observability import Observability, ObservabilityBase
        from roc.reporting.parquet_exporter import ParquetExporter

        # Redirect data_dir to tmp_path so we do not pollute the real
        # run directory while using real DuckLakeStore instances.
        settings = Config.get()
        original_data_dir = settings.data_dir
        settings.data_dir = str(tmp_path)

        # Real shared LoggerProvider with a "server-wide" processor
        # that must survive both game resets (modeling the OTLP batch
        # processor in production).
        logger_provider = LoggerProvider(shutdown_on_exit=False)
        server_wide = SimpleLogRecordProcessor(ConsoleLogExporter())
        logger_provider.add_log_record_processor(server_wide)

        instance = object.__new__(Observability)
        ObservabilityBase._instances[Observability] = instance

        try:
            with patch("roc.reporting.observability.otel_logs") as mock_logs:
                mock_logs.get_logger_provider.return_value = logger_provider

                # --- Game 1: real DuckLakeStore, real ParquetExporter,
                # real SimpleLogRecordProcessor attached to the shared
                # provider via _init_parquet.
                Observability._allow_parquet = True
                game1_instance_id = obs_mod.instance_id
                Observability._init_parquet()
                game1_store = instance._ducklake_store
                game1_exporter = instance._parquet_exporter
                game1_processor = instance._parquet_processor

                assert isinstance(game1_store, DuckLakeStore)
                assert isinstance(game1_exporter, ParquetExporter)
                attached = logger_provider._multi_log_record_processor._log_record_processors
                assert game1_processor in attached
                assert server_wide in attached
                assert str(tmp_path) in str(game1_store.run_dir)
                assert game1_instance_id in str(game1_store.run_dir)

                # --- End of game 1.
                Observability.reset()

                # Singleton survived reset -- critical for the fix.
                assert Observability in ObservabilityBase._instances
                attached = logger_provider._multi_log_record_processor._log_record_processors
                # Game 1's per-game processor is gone.
                assert game1_processor not in attached
                # Server-wide processor is preserved.
                assert server_wide in attached
                # Per-game attributes were deleted from the instance.
                assert not hasattr(instance, "_ducklake_store")
                assert not hasattr(instance, "_parquet_exporter")
                assert not hasattr(instance, "_parquet_processor")
                # Parquet flags cleared so the next init recreates them.
                assert Observability._parquet_configured is False
                assert Observability._allow_parquet is False
                # A new run identity was generated for game 2.
                assert obs_mod.instance_id != game1_instance_id

                # --- Game 2: real _init_parquet against the same
                # shared provider. The new processor must land on the
                # same provider that log emitters see -- this is the
                # load-bearing check. If the fix regressed, the new
                # processor would land on a throw-away provider.
                Observability._allow_parquet = True
                Observability._init_parquet()
                game2_store = instance._ducklake_store
                game2_exporter = instance._parquet_exporter
                game2_processor = instance._parquet_processor

                assert isinstance(game2_store, DuckLakeStore)
                assert isinstance(game2_exporter, ParquetExporter)
                assert game2_store is not game1_store
                assert game2_exporter is not game1_exporter
                assert game2_processor is not game1_processor
                attached = logger_provider._multi_log_record_processor._log_record_processors
                assert game2_processor in attached
                assert server_wide in attached
                # Game 2's run directory uses the new instance_id.
                assert obs_mod.instance_id in str(game2_store.run_dir)
                assert game1_instance_id not in str(game2_store.run_dir)
                assert game2_store.run_dir != game1_store.run_dir

                # Clean up game 2 so the fixture does not leak an open
                # DuckLake connection.
                Observability.reset()
        finally:
            settings.data_dir = original_data_dir


class TestConfigureDebugExporters:
    """Tests for Observability._configure_debug_exporters (lines 355-401)."""

    def test_skips_when_pytest_scanning(self, reset_observability):
        """Should return early when _disable_for_pytest_scanning is True."""
        import roc.reporting.observability as obs_mod
        from roc.reporting.observability import Observability

        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = True

        try:
            with patch("roc.reporting.observability.otel_logs") as mock_logs:
                Observability._configure_debug_exporters()

                mock_logs.get_logger_provider.assert_not_called()
        finally:
            obs_mod._disable_for_pytest_scanning = orig

    def test_skips_when_nothing_needed(self, reset_observability):
        """Should return early when neither remote log nor parquet is needed."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = False
        Observability._allow_parquet = False
        Observability._remote_log_configured = False
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            with patch("roc.reporting.observability.otel_logs") as mock_logs:
                Observability._configure_debug_exporters()

                mock_logs.get_logger_provider.assert_not_called()
        finally:
            obs_mod._disable_for_pytest_scanning = orig
            Observability._allow_parquet = False

    def test_creates_logger_provider_if_missing(self, reset_observability):
        """Should create a LoggerProvider if the current one is not an SDK provider."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = True
        settings.debug_remote_log_url = "http://test:9080/log"
        Observability._remote_log_configured = False
        Observability._allow_parquet = False
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        try:
            # Return a non-LoggerProvider object so isinstance check fails
            mock_noop_provider = MagicMock(spec=[])

            with (
                patch(
                    "roc.reporting.observability.otel_logs.get_logger_provider",
                    return_value=mock_noop_provider,
                ),
                patch("roc.reporting.observability.otel_logs.set_logger_provider") as mock_set,
                patch("roc.reporting.observability.SimpleLogRecordProcessor"),
                patch.dict(
                    "sys.modules",
                    {
                        "roc.reporting.remote_logger_exporter": MagicMock(),
                    },
                ),
            ):
                Observability._configure_debug_exporters()

                # A new LoggerProvider was created and set
                mock_set.assert_called_once()
                assert Observability._remote_log_configured is True
        finally:
            obs_mod._disable_for_pytest_scanning = orig
            Observability._remote_log_configured = False
            Observability._allow_parquet = False

    def test_adds_remote_log_exporter(self, reset_observability):
        """Should add RemoteLoggerExporter when debug_remote_log is True."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = True
        settings.debug_remote_log_url = "http://test:9080/log"
        Observability._remote_log_configured = False
        Observability._allow_parquet = False
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        # Use spec=LoggerProvider so isinstance check passes
        from opentelemetry.sdk._logs import LoggerProvider

        mock_provider = MagicMock(spec=LoggerProvider)

        try:
            with (
                patch(
                    "roc.reporting.observability.otel_logs.get_logger_provider",
                    return_value=mock_provider,
                ),
                patch("roc.reporting.observability.SimpleLogRecordProcessor"),
                patch.dict(
                    "sys.modules",
                    {
                        "roc.reporting.remote_logger_exporter": MagicMock(),
                    },
                ),
            ):
                Observability._configure_debug_exporters()

                mock_provider.add_log_record_processor.assert_called()
                assert Observability._remote_log_configured is True
        finally:
            obs_mod._disable_for_pytest_scanning = orig
            Observability._remote_log_configured = False
            Observability._allow_parquet = False

    def test_adds_parquet_exporter(self, reset_observability):
        """Should add ParquetExporter when _allow_parquet is True."""
        import roc.reporting.observability as obs_mod
        from roc.framework.config import Config
        from roc.reporting.observability import Observability

        settings = Config.get()
        settings.debug_remote_log = False
        Observability._allow_parquet = True
        Observability._parquet_configured = False
        Observability._remote_log_configured = True  # skip remote log branch
        orig = obs_mod._disable_for_pytest_scanning
        obs_mod._disable_for_pytest_scanning = False

        # Use spec=LoggerProvider so isinstance check passes
        from opentelemetry.sdk._logs import LoggerProvider

        mock_provider = MagicMock(spec=LoggerProvider)

        try:
            with (
                patch(
                    "roc.reporting.observability.otel_logs.get_logger_provider",
                    return_value=mock_provider,
                ),
                patch("roc.reporting.observability.SimpleLogRecordProcessor"),
                patch("roc.reporting.ducklake_store.DuckLakeStore"),
                patch("roc.reporting.observability.ParquetExporter"),
            ):
                Observability._configure_debug_exporters()

                mock_provider.add_log_record_processor.assert_called()
                assert Observability._parquet_configured is True
        finally:
            obs_mod._disable_for_pytest_scanning = orig
            Observability._allow_parquet = False
            Observability._parquet_configured = False
            Observability._remote_log_configured = False
