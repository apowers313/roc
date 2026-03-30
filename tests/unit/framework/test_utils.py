# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/utils.py."""

import re

from roc.framework.utils import _timestamp_str


class TestTimestampStr:
    def test_returns_string(self):
        result = _timestamp_str()
        assert isinstance(result, str)

    def test_format(self):
        result = _timestamp_str()
        # Expected format: YYYY.MM.DD-HH.MM.SS
        pattern = r"^\d{4}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}$"
        assert re.match(pattern, result), f"Timestamp '{result}' does not match expected format"
