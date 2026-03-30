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
