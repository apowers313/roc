from __future__ import annotations

import math
import os
import traceback
from time import time_ns

from opentelemetry._logs import NoOpLogger, SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace import get_current_span

from .config import Config
from .logger import logger

os.environ["OTEL_METRIC_EXPORT_INTERVAL"] = "5000"  # so we don't have to wait 60s for metrics
os.environ["OTEL_RESOURCE_ATTRIBUTES"] = "service.name=rolldice,service.instance.id=localhost:8082"

logger_provider: LoggerProvider | None = None


resource = Resource.create(
    {
        "service.name": "roc",
        "service.instance.id": "roc1",
    }
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
    span_context = get_current_span().get_span_context()
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
    # record["name"]
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

    # TODO: otel_logger: get_logger_provider?
    # loki = get_logger(record["name"], logger_provider=logger_provider)
    assert logger_provider is not None
    otel_logger = logger_provider.get_logger(record["name"])
    if not isinstance(otel_logger, NoOpLogger):
        otel_logger.emit(log_record)


class Observability:
    @staticmethod
    def init() -> None:
        settings = Config.get()

        if settings.observability_logging:
            # log init
            otlp_exporter = OTLPLogExporter(endpoint=settings.observability_host, insecure=True)
            global logger_provider
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(otlp_exporter))

            #####################
            # built in logging
            # import logging as python_logging

            # handler = LoggingHandler(level=python_logging.DEBUG, logger_provider=logger_provider)
            # logger2 = python_logging.getLogger("myapp.area2")
            # logger2.addHandler(handler)
            # logger2.setLevel(python_logging.DEBUG)
            # logger2.info("Service starting...")
            #####################

            # connect to loguru
            logger.add(loguru_to_otel, format="<level>{message}</level>")

        # TODO: trace init
        # TODO: metrics init

    @staticmethod
    def trace() -> None:
        pass

    @staticmethod
    def create_metric() -> None:
        pass

    @staticmethod
    def get_metric() -> None:
        pass
