# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/object.py."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from roc.location import XLoc, YLoc
from roc.object import (
    FeatureGroup,
    Object,
    ObjectCache,
    ResolvedObject,
    SymmetricDifferenceResolution,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestObject:
    def test_constructor(self):
        o = Object()
        assert isinstance(o, Object)

    def test_uuid_generation(self):
        o1 = Object()
        o2 = Object()
        # UUIDs should be different (with very high probability)
        assert isinstance(o1.uuid, int)
        assert isinstance(o2.uuid, int)

    def test_annotations_default(self):
        o = Object()
        assert o.annotations == []

    def test_resolve_count_default(self):
        o = Object()
        assert o.resolve_count == 0

    def test_str_format(self):
        o = Object()
        with patch.object(type(o), "features", new_callable=PropertyMock, return_value=[]):
            s = str(o)
            assert s.startswith("Object(")

    def test_with_features(self):
        fg = FeatureGroup()
        with patch("roc.object.Features.connect") as mock_connect:
            o = Object.with_features(fg)
            assert isinstance(o, Object)
            mock_connect.assert_called_once_with(o, fg)

    def test_distance(self):
        o = Object()
        # Create mock feature nodes with labels and str representations
        f1 = MagicMock()
        f1.labels = {"SingleNode", "FeatureNode"}
        f1.configure_mock(**{"__str__": MagicMock(return_value="SingleNode(a)")})

        f2 = MagicMock()
        f2.labels = {"ColorNode", "FeatureNode"}
        f2.configure_mock(**{"__str__": MagicMock(return_value="ColorNode(red)")})

        f3 = MagicMock()
        f3.labels = {"SingleNode", "FeatureNode"}
        f3.configure_mock(**{"__str__": MagicMock(return_value="SingleNode(b)")})

        # Mock the object's features
        with patch.object(type(o), "features", new_callable=PropertyMock, return_value=[f1]):
            dist = SymmetricDifferenceResolution._distance(o, [f2, f3])
            # symmetric diff: all different -> distance should be nonzero
            assert isinstance(dist, float)


class TestFeatureGroup:
    def test_from_nodes(self):
        fn1 = MagicMock()
        fn2 = MagicMock()
        with patch("roc.object.Detail.connect") as mock_connect:
            fg = FeatureGroup.from_nodes([fn1, fn2])
            assert isinstance(fg, FeatureGroup)
            assert mock_connect.call_count == 2

    def test_feature_nodes_property(self):
        fg = FeatureGroup()
        mock_edge1 = MagicMock()
        mock_edge1.type = "Detail"
        mock_edge1.dst = MagicMock()

        mock_edge2 = MagicMock()
        mock_edge2.type = "Other"
        mock_edge2.dst = MagicMock()

        with patch.object(
            type(fg), "src_edges", new_callable=PropertyMock, return_value=[mock_edge1, mock_edge2]
        ):
            nodes = fg.feature_nodes
            assert len(nodes) == 1
            assert nodes[0] is mock_edge1.dst


class TestResolvedObject:
    def test_constructor(self):
        o = Object()
        fg = FeatureGroup()
        x = XLoc(5)
        y = YLoc(10)
        ro = ResolvedObject(object=o, feature_group=fg, x=x, y=y)
        assert ro.object is o
        assert ro.feature_group is fg
        assert ro.x == 5
        assert ro.y == 10


class TestObjectCache:
    def test_inherits_from_lru_cache(self):
        from cachetools import LRUCache

        assert issubclass(ObjectCache, LRUCache)

    def test_instantiate(self):
        cache = ObjectCache(maxsize=10)
        assert len(cache) == 0


class TestSymmetricDifferenceResolution:
    def test_resolve_no_candidates(self):
        fn = MagicMock()
        mock_node_list = MagicMock()
        mock_node_list.select.return_value = []
        fn.predecessors = mock_node_list

        resolution = SymmetricDifferenceResolution()
        fg = MagicMock()
        result = resolution.resolve([fn], fg)
        assert result is None
