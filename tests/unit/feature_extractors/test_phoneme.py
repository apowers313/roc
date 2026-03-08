# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/feature_extractors/phoneme.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.feature_extractors.phoneme import PhonemeFeature, PhonemeNode


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestPhonemeNode:
    def test_attr_strs(self):
        """Line 22: attr_strs returns [str(self.type)]."""
        node = PhonemeNode(type=42)
        assert node.attr_strs == ["42"]

    def test_attr_strs_zero(self):
        node = PhonemeNode(type=0)
        assert node.attr_strs == ["0"]


class TestPhonemeFeature:
    def test_create_nodes(self):
        """Line 33: _create_nodes returns a PhonemeNode with type=42."""
        feature = PhonemeFeature(
            origin_id=MagicMock(),
            phonemes=[["h", "eh", "l", "ow"]],
        )
        node = feature._create_nodes()
        assert isinstance(node, PhonemeNode)
        assert node.type == 42

    def test_dbfetch_nodes(self):
        """Line 36: _dbfetch_nodes calls PhonemeNode.find_one."""
        feature = PhonemeFeature(
            origin_id=MagicMock(),
            phonemes=[["h", "eh", "l", "ow"]],
        )
        with patch.object(PhonemeNode, "find_one", return_value=None) as mock_find:
            result = feature._dbfetch_nodes()
            mock_find.assert_called_once_with("src.type = $type", params={"type": 42})
            assert result is None

    def test_dbfetch_nodes_found(self):
        """_dbfetch_nodes returns the node when found."""
        feature = PhonemeFeature(
            origin_id=MagicMock(),
            phonemes=[],
        )
        mock_node = PhonemeNode(type=42)
        with patch.object(PhonemeNode, "find_one", return_value=mock_node):
            result = feature._dbfetch_nodes()
            assert result is mock_node
