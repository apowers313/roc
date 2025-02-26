import pytest
from opentelemetry._events import Event
from opentelemetry._logs import SeverityNumber

from roc.logger import logger
from roc.reporting.observability import Observability, _lg_to_otel_severity, roc_common_attributes


@pytest.mark.requires_observability
class TestObservability:
    @pytest.fixture(scope="function", autouse=True)
    def obs_init(self) -> None:
        Observability.init()

    def test_exists(self) -> None:
        Observability()

    def test_system_metrics(self) -> None:
        """logs automatic system metrics like CPU and memory"""

        import time

        for i in range(120):
            print("loop", i)
            time.sleep(1)

    def test_metrics_counter(self) -> None:
        # Observability.get_counter()
        pass

    def test_logger(self) -> None:
        # mocking
        # https://pytest-mock.readthedocs.io/en/latest/usage.html
        logger.info("this is a little test message")

    def test_logger_extra_fields(self) -> None:
        logger.debug("asdfquer", foo="bar", answer=42)

    def test_logger_exception(self) -> None:
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("That was dumb")

    def test_logger_caught_exception(self) -> None:
        @logger.catch
        def my_exception() -> None:
            raise ValueError("SOMETHING BAD HAPPENED!!!")

        my_exception()

    def test_logger_severity_translation_boundaries(self) -> None:
        with pytest.raises(ValueError, match="loguru log severity above max range"):
            _lg_to_otel_severity(60)

        with pytest.raises(ValueError, match="loguru log severity must be a positive integer"):
            _lg_to_otel_severity(-1)

    def test_event(self) -> None:
        Observability.init()

        class TestEvent(Event):
            def __init__(self) -> None:
                super().__init__(
                    "test_event",
                    # body=screens[0],
                    body={
                        "foo": "bar",
                        "arr": [1, 2, 3],
                        "thing": True,
                    },
                    severity_number=SeverityNumber.INFO,
                    attributes=roc_common_attributes,
                )

        # logger.info(f"{Back.yellow}{Fore.red}sending test event{Style.reset}")
        # logger.info("""<div style="background-color:powderblue;">this is a log message</div>""")
        event_logger = Observability.get_event_logger()
        event_logger.emit(TestEvent())

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
