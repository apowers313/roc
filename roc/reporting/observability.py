from __future__ import annotations

import importlib
import math
import os
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
from opentelemetry._events import Event, EventLogger, NoOpEventLogger
from opentelemetry._logs import NoOpLogger, SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.metrics import Meter, Observation
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider, LogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import Tracer

from ..config import Config
from ..logger import logger

__all__ = [
    "Observability",
    "Observation",
]

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
t = datetime.now().strftime("%Y%m%d%H%M%S")
format_str = t + "-{{adj|lower}}-{{firstname|lower}}-{{lastname|lower}}"
instance_id = str(FlexiHumanHash(format_str).rand())

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


class ObservabilityEvent(Event):
    def __init__(
        self,
        name: str,
        body: Any | None = None,
        attributes: dict[str, int | float | str] = dict(),
    ):
        merged_attrs = attributes | roc_common_attributes
        super().__init__(name, body=body, attributes=merged_attrs)


class ObservabilityBase(type):
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
    def __init__(self) -> None:
        settings = Config.get()

        if settings.observability_logging:
            # log init
            logger_provider = LoggerProvider(resource=resource)
            otlp_log_exporter = OTLPLogExporter(endpoint=settings.observability_host, insecure=True)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_log_exporter))
            otel_logs.set_logger_provider(logger_provider=logger_provider)

            # connect logs to loguru
            logger.add(
                loguru_to_otel,
                format="<level>{message}</level>",
                level=settings.observability_logging_level,
            )
            # events init
            event_logger_provider = EventLoggerProvider(logger_provider=logger_provider)

            self.set_event_logger(
                event_logger_provider.get_event_logger(
                    "roc",
                    version=roc_version,
                    attributes=roc_common_attributes,
                )
            )
            otel_events.set_event_logger_provider(event_logger_provider=event_logger_provider)
            logger.debug(f"OpenTelemetry log initialized, instance ID {instance_id}.")
        else:
            self.set_event_logger(NoOpEventLogger("roc"))

        if settings.observability_metrics:
            logger.debug("initializing OpenTelemetry metrics...")
            # metrics init
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
        # NOTE: this will be a NoOpMeterProvider if the block of code above wasn't executed
        mp = otel_metrics.get_meter_provider()
        self.set_meter(
            mp.get_meter(
                "roc",
                version=roc_version,
                attributes=roc_common_attributes,
            )
        )

        if settings.observability_tracing:
            # trace init
            logger.debug("initializing OpenTelemetry trace...")
            otlp_trace_exporter = OTLPSpanExporter(
                endpoint=settings.observability_host, insecure=True
            )
            tracer_provider = TracerProvider(resource=resource)
            span_processor = BatchSpanProcessor(otlp_trace_exporter)
            tracer_provider.add_span_processor(span_processor)
            otel_trace.set_tracer_provider(tracer_provider)
        # NOTE: this will be a NoOpTracerProvider if the block of code above wasn't executed
        tp = otel_trace.get_tracer_provider()
        self.set_tracer(
            tp.get_tracer(
                "roc",
                instrumenting_library_version=roc_version,
                attributes=roc_common_attributes,
            )
        )

        if settings.observability_profiling:
            # profiling init
            logger.debug("initializing Pyroscope profiling...")
            pyroscope.configure(
                application_name="roc",
                # TODO: profiling in otel is current unstable, switch this to
                # otel when it stabilizes
                server_address=settings.observability_profiling_host,
                sample_rate=100,  # default is 100
                # detect_subprocesses=True,
                oncpu=False,  # report cpu time only; default is True
                tags=roc_common_attributes,
            )

    @staticmethod
    def init() -> None:
        Observability()

    @classmethod
    def event(cls, evt: Event) -> None:
        Observability.event_logger.emit(evt)

    @classmethod
    def set_event_logger(cls, event_logger: EventLogger) -> None:
        cls.event_logger = event_logger

    @classmethod
    def set_tracer(cls, tracer: Tracer) -> None:
        cls.tracer = tracer

    @classmethod
    def set_meter(cls, meter: Meter) -> None:
        cls.meter = meter


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
        resource=resource,
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
