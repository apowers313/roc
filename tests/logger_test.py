# mypy: disable-error-code="no-untyped-def"

import pytest
from pydantic import ValidationError

import roc.logger as roc_logger
from roc.config import Config
from roc.logger import LogFilter, logger


class FakeRecord:
    def __init__(self, level: str, module: str):
        self.level = type("Level", (object,), dict(no=logger.level(level).no))
        self.module = module

    def __getitem__(self, key):
        return self.__dict__[key]


class TestLogger:
    def test_default_log_level(self):
        settings = Config.get()
        assert settings.log_level == "INFO"
        roc_logger.init()
        assert roc_logger.default_log_filter.level == "INFO"  # type: ignore

    # def test_logging(self):
    #     logger.info("THIS IS LOGGING")

    def test_log(self, capfd):
        logger.info("asdf1234")
        logger.trace("qwer5678")
        captured = capfd.readouterr()

        assert "asdf1234" in captured.err
        assert "qwer5678" not in captured.err


class TestModuleParsing:
    def test_parse_module_str_empty(self):
        ret = LogFilter.parse_module_str("")

        assert ret == []

    def test_parse_module_str_one(self):
        ret = LogFilter.parse_module_str("config:INFO")

        assert len(ret) == 1
        assert ret[0].module_name == "config"
        assert ret[0].log_level == "INFO"

    def test_parse_module_str_many(self):
        ret = LogFilter.parse_module_str("logger:INFO;config:TRACE;component:CRITICAL")

        assert len(ret) == 3
        assert ret[0].module_name == "logger"
        assert ret[0].log_level == "INFO"
        assert ret[1].module_name == "config"
        assert ret[1].log_level == "TRACE"
        assert ret[2].module_name == "component"
        assert ret[2].log_level == "CRITICAL"

    def test_parse_module_str_bad_level(self):
        with pytest.raises(ValidationError):
            LogFilter.parse_module_str("config:blah")

    def test_parse_module_str_bad_module(self):
        with pytest.raises(ValidationError):
            LogFilter.parse_module_str("foo:INFO")


class TestLogFilter:
    def test_fake_record(self):
        r = FakeRecord("INFO", "event")

        assert r["level"].no == 20
        assert r["module"] == "event"

    def test_filter_defaults(self):
        filter = LogFilter(use_module_settings=False)

        assert filter.level == "INFO"
        assert filter.level_num == 20
        assert filter.module_levels == {}
        assert filter(FakeRecord("INFO", "event"))
        assert not filter(FakeRecord("TRACE", "event"))
        assert not filter(FakeRecord("TRACE", "config"))
        assert filter(FakeRecord("CRITICAL", "event"))

    def test_filter_level(self):
        filter = LogFilter(level="TRACE")

        assert filter.level == "TRACE"
        assert filter.level_num == 5
        assert filter(FakeRecord("INFO", "event"))
        assert filter(FakeRecord("TRACE", "event"))
        assert filter(FakeRecord("TRACE", "config"))
        assert filter(FakeRecord("CRITICAL", "event"))

    def test_filter_module_levels(self):
        filter = LogFilter(log_modules="logger:INFO;config:TRACE;component:CRITICAL")

        # doesn't mess up default level
        assert filter.level == "INFO"
        assert filter.level_num == 20
        # sets up per-module levels
        assert len(filter.module_levels) == 3
        assert filter.module_levels["logger"] == "INFO"
        assert filter.module_levels["config"] == "TRACE"
        assert filter.module_levels["component"] == "CRITICAL"
        # filters correctly
        assert filter(FakeRecord("INFO", "logger"))
        assert not filter(FakeRecord("DEBUG", "logger"))
        assert filter(FakeRecord("CRITICAL", "logger"))
        assert filter(FakeRecord("TRACE", "config"))
        assert filter(FakeRecord("CRITICAL", "config"))
        assert not filter(FakeRecord("TRACE", "component"))
        assert not filter(FakeRecord("ERROR", "component"))
        assert filter(FakeRecord("CRITICAL", "component"))
