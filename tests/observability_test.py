import pytest
from opentelemetry._logs import SeverityNumber

from roc.logger import logger
from roc.observability import Observability, _lg_to_otel_severity


class TestObservability:
    def test_exists(self) -> None:
        Observability()

    def test_logger(self) -> None:
        Observability.init()
        logger.info("this is a little test message")

    def test_logger_extra_fields(self) -> None:
        Observability.init()
        logger.debug("asdfquer", foo="bar", answer=42)

    def test_logger_exception(self) -> None:
        Observability.init()

        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("That was dumb")

    def test_logger_caught_exception(self) -> None:
        Observability.init()

        @logger.catch
        def my_exception() -> None:
            raise ValueError("SOMETHING BAD HAPPENED!!!")

        my_exception()

    def test_logger_severity_translation_boundaries(self) -> None:
        with pytest.raises(ValueError, match="loguru log severity above max range"):
            _lg_to_otel_severity(60)

        with pytest.raises(ValueError, match="loguru log severity must be a positive integer"):
            _lg_to_otel_severity(-1)

    def test_logger_severity_translation(self) -> None:
        res = [_lg_to_otel_severity(sev) for sev in range(60)]

        assert res == [
            (SeverityNumber(1), "TRACE"),
            (SeverityNumber(1), "TRACE"),
            (SeverityNumber(1), "TRACE"),
            (SeverityNumber(2), "TRACE"),
            (SeverityNumber(2), "TRACE"),
            (SeverityNumber(3), "TRACE"),
            (SeverityNumber(3), "TRACE"),
            (SeverityNumber(3), "TRACE"),
            (SeverityNumber(4), "TRACE"),
            (SeverityNumber(4), "TRACE"),
            (SeverityNumber(5), "DEBUG"),
            (SeverityNumber(5), "DEBUG"),
            (SeverityNumber(5), "DEBUG"),
            (SeverityNumber(6), "DEBUG"),
            (SeverityNumber(6), "DEBUG"),
            (SeverityNumber(7), "DEBUG"),
            (SeverityNumber(7), "DEBUG"),
            (SeverityNumber(7), "DEBUG"),
            (SeverityNumber(8), "DEBUG"),
            (SeverityNumber(8), "DEBUG"),
            (SeverityNumber(9), "INFO"),
            (SeverityNumber(9), "INFO"),
            (SeverityNumber(9), "INFO"),
            (SeverityNumber(10), "INFO"),
            (SeverityNumber(10), "INFO"),
            (SeverityNumber(11), "INFO"),
            (SeverityNumber(11), "INFO"),
            (SeverityNumber(11), "INFO"),
            (SeverityNumber(12), "INFO"),
            (SeverityNumber(12), "INFO"),
            (SeverityNumber(13), "WARN"),
            (SeverityNumber(13), "WARN"),
            (SeverityNumber(13), "WARN"),
            (SeverityNumber(14), "WARN"),
            (SeverityNumber(14), "WARN"),
            (SeverityNumber(15), "WARN"),
            (SeverityNumber(15), "WARN"),
            (SeverityNumber(15), "WARN"),
            (SeverityNumber(16), "WARN"),
            (SeverityNumber(16), "WARN"),
            (SeverityNumber(17), "ERROR"),
            (SeverityNumber(17), "ERROR"),
            (SeverityNumber(17), "ERROR"),
            (SeverityNumber(18), "ERROR"),
            (SeverityNumber(18), "ERROR"),
            (SeverityNumber(19), "ERROR"),
            (SeverityNumber(19), "ERROR"),
            (SeverityNumber(19), "ERROR"),
            (SeverityNumber(20), "ERROR"),
            (SeverityNumber(20), "ERROR"),
            (SeverityNumber(21), "FATAL"),
            (SeverityNumber(21), "FATAL"),
            (SeverityNumber(21), "FATAL"),
            (SeverityNumber(22), "FATAL"),
            (SeverityNumber(22), "FATAL"),
            (SeverityNumber(23), "FATAL"),
            (SeverityNumber(23), "FATAL"),
            (SeverityNumber(23), "FATAL"),
            (SeverityNumber(24), "FATAL"),
            (SeverityNumber(24), "FATAL"),
        ]
