# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/transformable.py."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from roc.pipeline.temporal.transformable import Transform, Transformable


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestTransform:
    def test_src_frame(self):
        t = Transform()
        mock_frame = MagicMock()
        mock_frame.labels = {"Frame"}

        mock_edge = MagicMock()
        mock_edge.src = mock_frame

        mock_edge_list = MagicMock()
        mock_edge_list.select.return_value = [mock_edge]

        with patch.object(
            type(t), "dst_edges", new_callable=PropertyMock, return_value=mock_edge_list
        ):
            result = t.src_frame
            assert result is mock_frame
            mock_edge_list.select.assert_called_once_with(type="Change")

    def test_dst_frame(self):
        t = Transform()
        mock_frame = MagicMock()
        mock_frame.labels = {"Frame"}

        mock_edge = MagicMock()
        mock_edge.dst = mock_frame

        mock_edge_list = MagicMock()
        mock_edge_list.select.return_value = [mock_edge]

        with patch.object(
            type(t), "src_edges", new_callable=PropertyMock, return_value=mock_edge_list
        ):
            result = t.dst_frame
            assert result is mock_frame
            mock_edge_list.select.assert_called_once_with(type="Change")


class TestTransformableABC:
    def test_is_abstract(self):
        assert Transformable.__abstractmethods__ == {
            "same_transform_type",
            "compatible_transform",
            "create_transform",
            "apply_transform",
        }
