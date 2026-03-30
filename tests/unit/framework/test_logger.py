# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/logger.py."""

from unittest.mock import MagicMock

import pytest

from roc.framework.logger import DebugModuleLevel, LogFilter, init, module_names


class TestDebugModuleLevel:
    def test_valid_module_name(self):
        # Use a name from the actual module_names list
        if module_names:
            name = module_names[0]
            dml = DebugModuleLevel(module_name=name, log_level="DEBUG")
            assert dml.module_name == name
            assert dml.log_level == "DEBUG"

    def test_invalid_module_name_raises(self):
        with pytest.raises(Exception):
            DebugModuleLevel(module_name="nonexistent_fake_module_xyz", log_level="DEBUG")

    def test_valid_log_levels(self):
        if module_names:
            name = module_names[0]
            for level in ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
                dml = DebugModuleLevel(module_name=name, log_level=level)
                assert dml.log_level == level


class TestLogFilterConstructor:
    def test_defaults(self):
        lf = LogFilter()
        assert lf.level is not None
        assert lf.level_num is not None

    def test_custom_level(self):
        lf = LogFilter(level="ERROR")
        assert lf.level == "ERROR"

    def test_custom_log_modules(self):
        if module_names:
            name = module_names[0]
            lf = LogFilter(log_modules=f"{name}:DEBUG")
            assert name in lf.module_levels
            assert lf.module_levels[name] == "DEBUG"

    def test_empty_log_modules(self):
        lf = LogFilter(log_modules="", use_module_settings=False)
        assert lf.module_levels == {}

    def test_use_module_settings_false(self):
        lf = LogFilter(use_module_settings=False)
        assert lf.module_levels == {}


class TestLogFilterCall:
    def test_record_at_level_passes(self):
        lf = LogFilter(level="WARNING")
        record = MagicMock()
        record.__getitem__ = lambda self, key: {
            "module": "some_module",
            "level": MagicMock(no=30),
        }[key]
        assert lf(record) is True

    def test_record_above_level_passes(self):
        lf = LogFilter(level="WARNING")
        record = MagicMock()
        record.__getitem__ = lambda self, key: {
            "module": "some_module",
            "level": MagicMock(no=40),
        }[key]
        assert lf(record) is True

    def test_record_below_level_fails(self):
        lf = LogFilter(level="ERROR")
        record = MagicMock()
        record.__getitem__ = lambda self, key: {
            "module": "some_module",
            "level": MagicMock(no=10),
        }[key]
        assert lf(record) is False

    def test_module_specific_level_passes(self):
        if not module_names:
            pytest.skip("No module names available")
        name = module_names[0]
        lf = LogFilter(level="ERROR", log_modules=f"{name}:DEBUG")
        record = MagicMock()
        record.__getitem__ = lambda self, key: {
            "module": name,
            "level": MagicMock(no=10),
        }[key]
        assert lf(record) is True

    def test_module_specific_level_fails(self):
        if not module_names:
            pytest.skip("No module names available")
        name = module_names[0]
        lf = LogFilter(level="ERROR", log_modules=f"{name}:WARNING")
        record = MagicMock()
        record.__getitem__ = lambda self, key: {
            "module": name,
            "level": MagicMock(no=10),
        }[key]
        assert lf(record) is False


class TestLogFilterParseModuleStr:
    def test_empty_string(self):
        result = LogFilter.parse_module_str("")
        assert result == []

    def test_whitespace_string(self):
        result = LogFilter.parse_module_str("   ")
        assert result == []

    def test_single_module(self):
        if not module_names:
            pytest.skip("No module names available")
        name = module_names[0]
        result = LogFilter.parse_module_str(f"{name}:DEBUG")
        assert len(result) == 1
        assert result[0].module_name == name
        assert result[0].log_level == "DEBUG"

    def test_multiple_modules(self):
        if len(module_names) < 2:
            pytest.skip("Need at least 2 module names")
        name1, name2 = module_names[0], module_names[1]
        result = LogFilter.parse_module_str(f"{name1}:DEBUG;{name2}:ERROR")
        assert len(result) == 2


class TestInit:
    def test_sets_initialized(self):
        import roc.framework.logger as logger_mod

        # Reset state
        orig = logger_mod._initialized
        logger_mod._initialized = False
        try:
            init()
            assert logger_mod._initialized is True
        finally:
            logger_mod._initialized = orig

    def test_doesnt_reinitialize(self):
        import roc.framework.logger as logger_mod

        logger_mod._initialized = True
        # Should return early without error
        init()
        assert logger_mod._initialized is True
