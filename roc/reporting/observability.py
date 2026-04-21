"""OpenTelemetry-based observability: logging, metrics, tracing, and profiling."""

from __future__ import annotations

import importlib
import math
import os
import sys
import traceback
from datetime import datetime
from time import time_ns
from typing import Any

import pyroscope
from flexihumanhash import FlexiHumanHash
from opentelemetry import _events as otel_events
from opentelemetry import _logs as otel_logs
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry._events import EventLogger, NoOpEventLogger
from opentelemetry._logs import NoOpLogger, SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.metrics import Meter, Observation
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Tracer

import roc.framework.logger as roc_logger
from roc.framework.config import Config
from roc.reporting.parquet_exporter import ParquetExporter

__all__ = [
    "Observability",
    "Observation",
]

_disable_for_pytest_scanning = "pytest" in sys.modules and "PYTEST_VERSION" in os.environ

# TODO: replace with metrics config
os.environ["OTEL_METRIC_EXPORT_INTERVAL"] = "5000"  # so we don't have to wait 60s for metrics
os.environ["OTEL_PYTHON_LOG_CORRELATION"] = "true"
system_metrics_config = {
    "system.cpu.time": ["idle", "user", "system", "irq"],
    "system.cpu.utilization": ["idle", "user", "system", "irq"],
    "system.memory.usage": ["used", "free", "cached"],
    "system.memory.utilization": ["used", "free", "cached"],
    "system.swap.usage": ["used", "free"],
    "system.swap.utilization": ["used", "free"],
    "system.disk.io": ["read", "write"],
    "system.disk.operations": ["read", "write"],
    "system.disk.time": ["read", "write"],
    "system.network.dropped.packets": ["transmit", "receive"],
    "system.network.packets": ["transmit", "receive"],
    "system.network.errors": ["transmit", "receive"],
    "system.network.io": ["transmit", "receive"],
    "system.network.connections": ["family", "type"],
    "system.thread_count": None,
    "process.runtime.memory": ["rss", "vms"],
    "process.runtime.cpu.time": ["user", "system"],
    "process.runtime.gc_count": None,
    "process.runtime.thread_count": None,
    "process.runtime.cpu.utilization": None,
    "process.runtime.context_switches": ["involuntary", "voluntary"],
    "process.open_file_descriptor.count": None,
}

roc_version = importlib.metadata.version("roc")


def _generate_instance_id() -> str:
    """Create a new timestamped, human-readable run identifier.

    The format is ``YYYYMMDDHHMMSS-<adj>-<firstname>-<lastname>``. Called
    once at module import and again from ``Observability.reset()`` between
    game runs so each run gets a distinct directory and dashboard entry.
    """
    t = datetime.now().strftime("%Y%m%d%H%M%S")
    fmt = t + "-{{adj|lower}}-{{firstname|lower}}-{{lastname|lower}}"
    return str(FlexiHumanHash(fmt).rand())


instance_id = _generate_instance_id()

resource = Resource.create(
    {
        "service.name": "roc",
        "service.instance.id": instance_id,
    }
)

roc_common_attributes = {
    # "roc.version": roc_version,
    "roc.instance.id": instance_id,
}


def _remove_log_record_processor(logger_provider: Any, processor: LogRecordProcessor) -> None:
    """Remove a specific ``LogRecordProcessor`` from a ``LoggerProvider``.

    OTel does not expose a public remove API on ``LoggerProvider``. We
    reach into ``_multi_log_record_processor._log_record_processors`` --
    a tuple the SDK exposes specifically so concurrent ``on_emit``
    iterations can copy-on-modify -- and filter out the target
    processor. This is the only way to swap per-game parquet exporters
    in and out of a shared ``LoggerProvider`` that must live for the
    full server process because ``set_logger_provider`` is
    ``Once``-protected.
    """
    multi = getattr(logger_provider, "_multi_log_record_processor", None)
    if multi is None:
        return
    processors = getattr(multi, "_log_record_processors", None)
    if processors is None:
        return
    new_processors = tuple(p for p in processors if p is not processor)
    lock = getattr(multi, "_lock", None)
    if lock is not None:
        with lock:
            multi._log_record_processors = new_processors
    else:
        multi._log_record_processors = new_processors


class ObservabilityBase(type):
    """Metaclass that makes Observability a singleton."""

    _instances: dict[type, Observability] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Observability:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.meter: Meter
        self.tracer: Tracer
        self.event_logger: EventLogger


class Observability(metaclass=ObservabilityBase):
    """Singleton that initializes and provides access to OpenTelemetry instrumentation."""

    _remote_log_configured: bool = False
    _parquet_configured: bool = False
    # DuckLake/parquet store creation is deferred until roc.init() sets this
    # to True. This prevents the module-level Observability.init() (needed
    # for decorator tracers) from creating empty run directories.
    _allow_parquet: bool = False

    def __init__(self) -> None:
        settings = Config.get()
        roc_logger.init()
        global _disable_for_pytest_scanning

        if settings.observability_logging and not _disable_for_pytest_scanning:
            self._check_otel_endpoint(settings.observability_host)

        self._init_logging(settings)
        self._init_metrics(settings)
        self._init_tracing(settings)
        self._init_profiling(settings)

    @staticmethod
    def _check_otel_endpoint(host: str) -> None:
        """Log a warning if the OTel collector endpoint is unreachable."""
        import socket as _socket

        try:
            parts = host.rsplit(":", 1)
            addr = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 4317
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((addr, port))
            sock.close()
        except Exception:
            roc_logger.logger.warning(
                "OTel collector unreachable at {} -- telemetry will be lost", host
            )

    def _init_logging(self, settings: Config) -> None:
        """Initialize OTel logging, remote log, parquet, and event logger."""
        if settings.observability_logging and not _disable_for_pytest_scanning:
            logger_provider = self._init_otlp_logging(settings)
            otel_logs.set_logger_provider(logger_provider=logger_provider)
            self._add_loguru_sinks(settings)
            self._init_event_logger(logger_provider)
        else:
            self._init_fallback_logging(settings)

    def _init_otlp_logging(self, settings: Config) -> LoggerProvider:
        """Set up OTLP log export with optional remote log and parquet exporters."""
        logger_provider = LoggerProvider(resource=resource)
        otlp_log_exporter = OTLPLogExporter(endpoint=settings.observability_host, insecure=True)
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(otlp_log_exporter, max_export_batch_size=32)
        )
        self._attach_remote_log_exporter(settings, logger_provider)
        self._attach_parquet_exporter(settings, logger_provider)
        return logger_provider

    def _init_fallback_logging(self, settings: Config) -> None:
        """Set up local-only logging when OTLP is disabled."""
        if not _disable_for_pytest_scanning:
            logger_provider = LoggerProvider(resource=resource)
            self._attach_remote_log_exporter(settings, logger_provider)
            self._attach_parquet_exporter(settings, logger_provider)
            otel_logs.set_logger_provider(logger_provider=logger_provider)
        self.set_event_logger(NoOpEventLogger("roc"))

    def _attach_remote_log_exporter(
        self, settings: Config, logger_provider: LoggerProvider
    ) -> None:
        """Attach the remote logger exporter if configured."""
        if not settings.debug_remote_log:
            return
        from roc.reporting.remote_logger_exporter import RemoteLoggerExporter

        self._remote_log_exporter = RemoteLoggerExporter(
            url=settings.debug_remote_log_url,
            session_id=instance_id,
        )
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self._remote_log_exporter)
        )
        Observability._remote_log_configured = True

    def _attach_parquet_exporter(self, settings: Config, logger_provider: LoggerProvider) -> None:
        """Attach the DuckLake/parquet exporter if _allow_parquet is set.

        The ``SimpleLogRecordProcessor`` wrapper is stored on the instance so
        ``Observability.reset`` can remove it from the shared LoggerProvider
        between game runs. Without tracking the processor we would have no
        way to clean up the per-game parquet exporter, because OTel does not
        expose a public remove API on LoggerProvider.
        """
        if not Observability._allow_parquet:
            return
        from pathlib import Path

        from roc.reporting.ducklake_store import DuckLakeStore

        run_dir = Path(settings.data_dir) / instance_id
        self._ducklake_store = DuckLakeStore(run_dir)
        self._parquet_exporter = ParquetExporter(store=self._ducklake_store)
        self._parquet_processor = SimpleLogRecordProcessor(self._parquet_exporter)
        logger_provider.add_log_record_processor(self._parquet_processor)
        Observability._parquet_configured = True

    def _add_loguru_sinks(self, settings: Config) -> None:
        """Connect loguru to OTel and optionally add live dashboard log sink."""
        roc_logger.logger.add(
            loguru_to_otel,
            format="<level>{message}</level>",
            level=settings.observability_logging_level,
        )
        if settings.dashboard_enabled or settings.dashboard_callback_url:
            from roc.reporting.step_log_sink import step_log_sink

            roc_logger.logger.add(
                step_log_sink,
                format="<level>{message}</level>",
                level=settings.observability_logging_level,
            )

    def _init_event_logger(self, logger_provider: LoggerProvider) -> None:
        """Initialize the OTel event logger from the logger provider."""
        event_logger_provider = EventLoggerProvider(logger_provider=logger_provider)
        self.set_event_logger(
            event_logger_provider.get_event_logger(
                "roc",
                version=roc_version,
                attributes=roc_common_attributes,
            )
        )
        otel_events.set_event_logger_provider(event_logger_provider=event_logger_provider)
        roc_logger.logger.debug(f"OpenTelemetry log initialized, instance ID {instance_id}.")

    def _init_metrics(self, settings: Config) -> None:
        """Initialize OTel metrics if enabled."""
        if settings.observability_metrics and not _disable_for_pytest_scanning:
            roc_logger.logger.debug("initializing OpenTelemetry metrics...")
            otlp_metrics_exporter = OTLPMetricExporter(
                endpoint=settings.observability_host, insecure=True
            )
            metric_reader = PeriodicExportingMetricReader(
                otlp_metrics_exporter,
                export_interval_millis=settings.observability_metrics_interval,
            )
            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            otel_metrics.set_meter_provider(meter_provider)
            SystemMetricsInstrumentor(
                labels=roc_common_attributes,
                config=system_metrics_config,
            ).instrument()
        # NOTE: this will be a NoOpMeterProvider if the block above wasn't executed
        mp = otel_metrics.get_meter_provider()
        self.set_meter(
            mp.get_meter(
                "roc",
                version=roc_version,
                attributes=roc_common_attributes,
            )
        )

    def _init_tracing(self, settings: Config) -> None:
        """Initialize OTel tracing if enabled."""
        if settings.observability_tracing and not _disable_for_pytest_scanning:
            roc_logger.logger.debug("initializing OpenTelemetry trace...")
            otlp_trace_exporter = OTLPSpanExporter(
                endpoint=settings.observability_host, insecure=True
            )
            tracer_provider = TracerProvider(resource=resource)
            span_processor = BatchSpanProcessor(otlp_trace_exporter)
            tracer_provider.add_span_processor(span_processor)
            otel_trace.set_tracer_provider(tracer_provider)
        # NOTE: this will be a NoOpTracerProvider if the block above wasn't executed
        tp = otel_trace.get_tracer_provider()
        self.set_tracer(
            tp.get_tracer(
                "roc",
                instrumenting_library_version=roc_version,
                attributes=roc_common_attributes,
            )
        )

    @staticmethod
    def _init_profiling(settings: Config) -> None:
        """Initialize Pyroscope profiling if enabled."""
        if not settings.observability_profiling or _disable_for_pytest_scanning:
            return
        roc_logger.logger.debug("initializing Pyroscope profiling...")
        pyroscope.configure(
            application_name="roc",
            server_address=settings.observability_profiling_host,
            sample_rate=100,  # default is 100
            oncpu=False,  # report cpu time only; default is True
            tags=roc_common_attributes,
        )

    @staticmethod
    def init(enable_parquet: bool = False) -> None:
        """Initializes the Observability singleton and configures debug exporters.

        When ``enable_parquet`` is True (set by ``roc.init()``), the DuckLake
        store and parquet exporter are created. Without this flag (module-level
        init), only the tracer/meter are set up -- no run directories are created.
        """
        if enable_parquet:
            Observability._allow_parquet = True
        Observability()
        # If parquet was just enabled and not yet configured, add the exporter
        # to the existing logger provider. This handles the case where the
        # module-level init created the singleton without parquet, and now
        # roc.init() wants to add it.
        if enable_parquet and not Observability._parquet_configured:
            Observability._init_parquet()
        Observability._configure_debug_exporters()

    @classmethod
    def _init_parquet(cls) -> None:
        """Add the DuckLake/parquet exporter to the existing logger provider.

        Called after roc.init() sets _allow_parquet=True, if the module-level
        init already created the singleton without parquet. Also called
        after ``Observability.reset`` clears ``_parquet_configured`` so the
        next game run can install a fresh exporter without re-running
        ``__init__`` (which would duplicate the server-wide OTLP, remote
        log, and loguru sinks).

        The ``SimpleLogRecordProcessor`` wrapper is stored on the instance
        so ``reset`` can remove it from the shared LoggerProvider between
        game runs.
        """
        from pathlib import Path

        from roc.framework.config import Config
        from roc.reporting.ducklake_store import DuckLakeStore

        instance = ObservabilityBase._instances.get(cls)
        if instance is None:
            return
        settings = Config.get()
        run_dir = Path(settings.data_dir) / instance_id
        instance._ducklake_store = DuckLakeStore(run_dir)
        instance._parquet_exporter = ParquetExporter(store=instance._ducklake_store)
        instance._parquet_processor = SimpleLogRecordProcessor(instance._parquet_exporter)
        # Add to existing logger provider
        provider = otel_logs.get_logger_provider()
        if hasattr(provider, "add_log_record_processor"):
            provider.add_log_record_processor(instance._parquet_processor)
        cls._parquet_configured = True

    @classmethod
    def get_ducklake_store(cls) -> Any:
        """Return the DuckLakeStore if Parquet export is configured, else None."""
        instance = ObservabilityBase._instances.get(cls)
        if instance is not None and hasattr(instance, "_ducklake_store"):
            return instance._ducklake_store
        return None

    @classmethod
    def shutdown(cls) -> None:
        """Flush exporters, close DuckLake store, and shut down profiling."""
        instance = ObservabilityBase._instances.get(cls)
        if instance is not None:
            if hasattr(instance, "_parquet_exporter"):
                instance._parquet_exporter.shutdown()
            if hasattr(instance, "_ducklake_store"):
                instance._ducklake_store.close()
        try:
            pyroscope.shutdown()
        except Exception:
            pass

    @classmethod
    def reset(cls) -> None:
        """Reset per-game state so the next game run can install fresh DuckLake.

        Called at the end of a game in thread mode (unified server). Removes
        the per-game parquet exporter processor from the shared
        ``LoggerProvider``, closes the DuckLake store, and regenerates
        ``instance_id`` / ``resource`` / ``roc_common_attributes`` so the
        next call to ``Observability.init()`` creates a new run directory.

        **Important**: this does NOT pop the ``Observability`` singleton or
        shut down the shared ``LoggerProvider``. Three reasons:

        1. OTel's ``set_logger_provider`` is ``Once``-protected. A new
           ``LoggerProvider`` installed after the global has already been
           set is silently ignored, which means any new per-game processors
           added to it would never receive records. This manifested as
           "second game produces steps but catalog stays empty" -- the
           symptom that drove this investigation.
        2. The OTLP batch processor and the remote logger exporter are
           server-wide sinks that should persist across game runs.
           Shutting down the provider would kill them too.
        3. Re-running ``__init__`` on a fresh instance would re-add those
           server-wide processors, leaving the existing provider with
           duplicate OTLP/remote log processors every time a game starts.

        Instead, we leave the singleton in place and only recycle the
        per-game DuckLake store + parquet exporter. The next game's
        ``Observability.init(enable_parquet=True)`` sees the existing
        instance, sees ``_parquet_configured=False`` (cleared here), and
        calls ``_init_parquet`` to attach a fresh exporter bound to the
        new run directory.
        """
        instance = ObservabilityBase._instances.get(cls)
        if instance is not None:
            # Remove the parquet processor from the shared provider first
            # so no new records land in the store we are about to close.
            provider = otel_logs.get_logger_provider()
            if hasattr(instance, "_parquet_processor"):
                try:
                    _remove_log_record_processor(provider, instance._parquet_processor)
                except Exception:
                    pass
                try:
                    instance._parquet_processor.shutdown()
                except Exception:
                    pass
                try:
                    del instance._parquet_processor
                except Exception:
                    pass
            if hasattr(instance, "_parquet_exporter"):
                try:
                    instance._parquet_exporter.shutdown()
                except Exception:
                    pass
                try:
                    del instance._parquet_exporter
                except Exception:
                    pass
            if hasattr(instance, "_ducklake_store"):
                try:
                    instance._ducklake_store.close()
                except Exception:
                    pass
                try:
                    del instance._ducklake_store
                except Exception:
                    pass

        # Clear the per-game flags so the next ``Observability.init`` call
        # triggers ``_init_parquet`` to attach a fresh exporter.
        cls._parquet_configured = False
        cls._allow_parquet = False
        # Leave ``_remote_log_configured`` alone -- the remote log exporter
        # stays attached to the provider across game runs.

        # Regenerate the module-level run identity. New imports of
        # ``instance_id`` will see the fresh value; existing callers must
        # access it via the module attribute (``observability.instance_id``)
        # rather than a snapshot from ``from ... import instance_id``.
        global instance_id, resource, roc_common_attributes
        instance_id = _generate_instance_id()
        resource = Resource.create(
            {
                "service.name": "roc",
                "service.instance.id": instance_id,
            }
        )
        roc_common_attributes = {
            "roc.instance.id": instance_id,
        }

    @classmethod
    def _configure_debug_exporters(cls) -> None:
        """Add debug exporters to the existing logger provider if config requires them.

        This is called after singleton creation so that CLI flags (which update
        Config after the module-level init) can still enable debug exporters.
        """
        settings = Config.get()
        global _disable_for_pytest_scanning
        if _disable_for_pytest_scanning:
            return

        needs_remote_log = settings.debug_remote_log and not getattr(
            cls, "_remote_log_configured", False
        )
        needs_parquet = cls._allow_parquet and not getattr(cls, "_parquet_configured", False)
        if not needs_remote_log and not needs_parquet:
            return

        provider = otel_logs.get_logger_provider()
        if not isinstance(provider, LoggerProvider):
            # No LoggerProvider was set up during __init__ (e.g. OTLP logging
            # was disabled and no debug exporters were needed at that time).
            # Create one now so we have somewhere to attach debug exporters.
            provider = LoggerProvider(resource=resource)
            otel_logs.set_logger_provider(logger_provider=provider)

        if needs_remote_log:
            from roc.reporting.remote_logger_exporter import RemoteLoggerExporter

            remote_exporter = RemoteLoggerExporter(
                url=settings.debug_remote_log_url,
                session_id=instance_id,
            )
            provider.add_log_record_processor(SimpleLogRecordProcessor(remote_exporter))
            cls._remote_log_configured = True

        if needs_parquet:
            from pathlib import Path

            from roc.reporting.ducklake_store import DuckLakeStore

            run_dir = Path(settings.data_dir) / instance_id
            ducklake_store = DuckLakeStore(run_dir)
            parquet_exporter = ParquetExporter(store=ducklake_store)
            provider.add_log_record_processor(SimpleLogRecordProcessor(parquet_exporter))
            cls._parquet_configured = True

    @classmethod
    def set_event_logger(cls, event_logger: EventLogger) -> None:
        """Sets the OpenTelemetry event logger."""
        cls.event_logger = event_logger

    @classmethod
    def set_tracer(cls, tracer: Tracer) -> None:
        """Sets the OpenTelemetry tracer."""
        cls.tracer = tracer

    @classmethod
    def set_meter(cls, meter: Meter) -> None:
        """Sets the OpenTelemetry meter."""
        cls.meter = meter

    @staticmethod
    def get_logger(name: str) -> Any:
        """Returns a named OTel logger for emitting structured log records.

        Args:
            name: The logger name (typically a module name).

        Returns:
            An OTel Logger instance.
        """
        return otel_logs.get_logger_provider().get_logger(
            name,
            version=roc_version,
            attributes=roc_common_attributes,
        )


# skip natural LogRecord attributes
# http://docs.python.org/library/logging.html#logrecord-attributes
_RESERVED_ATTRS = frozenset(
    (
        "asctime",
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "getMessage",
        "message",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    )
)


def _lg_to_otel_severity(loguru_sev: int) -> tuple[SeverityNumber, str]:
    """Converts loguru default log levels to OpenTelemetry severity numbers

    loguru log level docs:
    https://loguru.readthedocs.io/en/stable/api/logger.html#levels

    OpenTelemetry severity number docs:
    https://opentelemetry.io/docs/specs/otel/logs/data-model/#field-severitynumber

    Args:
        loguru_sev (int): the loguru severity number

    Raises:
        ValueError: if loguru severity number is less than 0 or more than 59

    Returns:
        tuple[SeverityNumber, str]: the OpenTelemetry severity number and
        severity range string (e.g. "DEBUG", "ERROR")
    """
    if loguru_sev > 59:
        raise ValueError("loguru log severity above max range")

    if loguru_sev < 0:
        raise ValueError("loguru log severity must be a positive integer")

    # loguru severity is every 10 levels with some weirdness mixed in
    # anything below trace gets mapped to "trace", and "success" gets mapped to info
    loguru_range_sev, loguru_sub_sev = divmod(loguru_sev, 10)
    # otel severity is every 4 levels starting at 1
    otel_range_sev = min(loguru_range_sev * 4 + 1, 24)
    # map the severity range to a name
    otel_range_name = ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL")[loguru_range_sev]
    # find the sub-number in the range
    otel_sub_sev = math.floor(loguru_sub_sev / 10 * 4)
    # merge the range with it's sub-number
    otel_sev = otel_range_sev + otel_sub_sev

    return (SeverityNumber(otel_sev), otel_range_name)


# Largely copied from:
# https://github.com/open-telemetry/opentelemetry-python/blob/a7fe4f8bac7fa36291c6acf86982bbb356e3ae6d/opentelemetry-sdk/src/opentelemetry/sdk/_logs/_internal/__init__.py#L558
def loguru_to_otel(msg: str) -> None:
    """Loguru sink that forwards log records to OpenTelemetry."""
    body = msg.strip()

    # loguru hides all the log information on a "record" attribute on the string
    record = msg.record  # type: ignore

    # basic log attributes
    attrs = {k: v for k, v in record["extra"].items() if k not in _RESERVED_ATTRS}
    attrs[SpanAttributes.CODE_FILEPATH] = record["file"].path
    attrs[SpanAttributes.CODE_FUNCTION] = record["function"]
    attrs[SpanAttributes.CODE_LINENO] = record["line"]
    attrs[SpanAttributes.CODE_NAMESPACE] = record["module"]
    attrs["process.pid"] = record["process"].id
    attrs["thread.id"] = record["thread"].id
    attrs["thread.name"] = record["thread"].name
    timestamp = int(record["time"].timestamp() * 10**9)
    observered_timestamp = time_ns()
    span_context = otel_trace.get_current_span().get_span_context()
    severity_number, level_name = _lg_to_otel_severity(record["level"].no)

    # handle exceptions
    if record["exception"] is not None:
        if record["exception"].type is not None:
            attrs[SpanAttributes.EXCEPTION_TYPE] = record["exception"].type.__name__
        if record["exception"].value is not None:
            attrs[SpanAttributes.EXCEPTION_MESSAGE] = record["exception"].value
        if record["exception"].traceback is not None:
            attrs[SpanAttributes.EXCEPTION_STACKTRACE] = msg.strip()
            body = traceback.format_exception(*record["exception"])[-1]

    # TODO:
    # record["name"]  # noqa: ERA001
    # are all the resource attributes correct?

    log_record = LogRecord(
        timestamp=timestamp,
        observed_timestamp=observered_timestamp,
        trace_id=span_context.trace_id,
        span_id=span_context.span_id,
        trace_flags=span_context.trace_flags,
        severity_text=level_name,
        severity_number=severity_number,
        body=body,
        attributes=attrs,
    )

    # TODO: if self.dropped_attributes: warn

    otel_logger = otel_logs.get_logger_provider().get_logger(
        record["name"],
        version=roc_version,
        attributes=roc_common_attributes,
    )
    if not isinstance(otel_logger, NoOpLogger):
        otel_logger.emit(log_record)


# NOTE: Observability.trace gets called as a decorator, which requires
# initializing here
Observability.init()
