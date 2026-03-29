# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/feature_extractors/color.py -- ColorNode.attr_strs coverage."""

from unittest.mock import MagicMock, patch

import pytest

from roc.feature_extractors.color import ColorNode


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestColorNodeAttrStrs:
    """Test all 17 color type values (0-16) plus the error case."""

    EXPECTED_COLORS = {
        0: "BLACK",
        1: "RED",
        2: "GREEN",
        3: "BROWN",
        4: "BLUE",
        5: "MAGENTA",
        6: "CYAN",
        7: "GREY",
        8: "NO COLOR",
        9: "ORANGE",
        10: "BRIGHT GREEN",
        11: "YELLOW",
        12: "BRIGHT BLUE",
        13: "BRIGHT MAGENTA",
        14: "BRIGHT CYAN",
        15: "WHITE",
        16: "MAX",
    }

    @pytest.mark.parametrize("type_val,expected", EXPECTED_COLORS.items())
    def test_color_type(self, type_val, expected):
        node = ColorNode(type=type_val)
        assert node.attr_strs == [expected]

    def test_impossible_color_raises(self):
        node = ColorNode(type=17)
        with pytest.raises(Exception, match="impossible color"):
            _ = node.attr_strs

    def test_impossible_color_large_value(self):
        node = ColorNode(type=999)
        with pytest.raises(Exception, match="impossible color"):
            _ = node.attr_strs
