# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/tables.py."""

import panel as pn

from roc.reporting.components.tables import compact_kv_table, compact_log_table


class TestCompactKvTable:
    def test_renders_dict_as_tabulator(self):
        result = compact_kv_table({"hp": 16, "score": 42})
        assert isinstance(result, pn.widgets.Tabulator)
        assert len(result.value) == 2
        assert "hp" in result.value["key"].values
        assert "42" in result.value["value"].values

    def test_none_returns_placeholder(self):
        result = compact_kv_table(None, title="test")
        assert isinstance(result, pn.pane.HTML)
        assert "No test data" in result.object

    def test_empty_dict_returns_placeholder(self):
        result = compact_kv_table({}, title="test")
        assert isinstance(result, pn.pane.HTML)

    def test_skips_raw_key(self):
        result = compact_kv_table({"raw": "blob", "hp": 16})
        assert isinstance(result, pn.widgets.Tabulator)
        assert len(result.value) == 1
        assert "raw" not in result.value["key"].values

    def test_truncates_long_values(self):
        result = compact_kv_table({"key": "x" * 200})
        assert isinstance(result, pn.widgets.Tabulator)
        val = result.value["value"].iloc[0]
        assert val.endswith("...")
        assert len(val) == 80

    def test_uses_midnight_theme(self):
        result = compact_kv_table({"a": 1})
        assert isinstance(result, pn.widgets.Tabulator)
        assert result.theme == "simple"


class TestCompactLogTable:
    def test_renders_logs_as_tabulator(self):
        logs = [
            {"severity_text": "INFO", "severity_number": 9, "body": "hello"},
            {"severity_text": "ERROR", "severity_number": 17, "body": "fail"},
        ]
        result = compact_log_table(logs)
        assert isinstance(result, pn.widgets.Tabulator)
        assert len(result.value) == 2

    def test_none_returns_placeholder(self):
        result = compact_log_table(None)
        assert isinstance(result, pn.pane.HTML)

    def test_filters_by_level(self):
        logs = [
            {"severity_text": "DEBUG", "severity_number": 5, "body": "debug msg"},
            {"severity_text": "ERROR", "severity_number": 17, "body": "error msg"},
        ]
        result = compact_log_table(logs, min_level="ERROR")
        assert isinstance(result, pn.widgets.Tabulator)
        assert len(result.value) == 1
        assert result.value["level"].iloc[0] == "ERROR"

    def test_all_filtered_returns_placeholder(self):
        logs = [
            {"severity_text": "DEBUG", "severity_number": 5, "body": "debug msg"},
        ]
        result = compact_log_table(logs, min_level="ERROR")
        assert isinstance(result, pn.pane.HTML)

    def test_has_hidden_color_column(self):
        logs = [{"severity_text": "INFO", "severity_number": 9, "body": "msg"}]
        result = compact_log_table(logs)
        assert isinstance(result, pn.widgets.Tabulator)
        assert "_color" in result.hidden_columns
