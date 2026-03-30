# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/perception.py."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from roc.perception.location import XLoc, YLoc
from roc.perception.base import (
    AuditoryData,
    Direction,
    Feature,
    PointFeature,
    ProprioceptiveData,
    Settled,
    VisionData,
    _to_numpy,
    cache_registry,
)


class TestVisionData:
    def test_constructor(self):
        glyphs = np.array([[1, 2], [3, 4]])
        chars = np.array([[65, 66], [67, 68]])
        colors = np.array([[0, 1], [2, 3]])
        vd = VisionData(glyphs, chars, colors)
        assert np.array_equal(vd.glyphs, glyphs)
        assert np.array_equal(vd.chars, chars)
        assert np.array_equal(vd.colors, colors)

    def test_from_dict_valid(self):
        d = {
            "glyphs": np.array([[1, 2]]),
            "chars": np.array([[65, 66]]),
            "colors": np.array([[0, 1]]),
        }
        vd = VisionData.from_dict(d)
        assert np.array_equal(vd.glyphs, d["glyphs"])

    def test_from_dict_missing_key(self):
        d = {"glyphs": np.array([[1]]), "chars": np.array([[65]])}
        with pytest.raises(Exception, match="colors"):
            VisionData.from_dict(d)

    def test_from_dict_converts_lists(self):
        d = {
            "glyphs": [[1, 2]],
            "chars": [[65, 66]],
            "colors": [[0, 1]],
        }
        vd = VisionData.from_dict(d)
        assert isinstance(vd.glyphs, np.ndarray)

    def test_for_test(self):
        data = [[65, 66], [67, 68]]
        vd = VisionData.for_test(data)
        assert np.array_equal(vd.glyphs, np.array(data))
        assert np.array_equal(vd.chars, np.array(data))
        assert np.array_equal(vd.colors, np.array(data))
        # Ensure they are independent copies
        vd.glyphs[0, 0] = 999
        assert vd.chars[0, 0] != 999


class TestAuditoryData:
    def test_constructor(self):
        ad = AuditoryData("You hear a door open.")
        assert ad.msg == "You hear a door open."


class TestProprioceptiveData:
    def test_constructor(self):
        inv_strs = np.array(["sword", "shield"])
        inv_letters = np.array([97, 98])
        inv_glyphs = np.array([100, 200])
        inv_oclasses = np.array([1, 2])
        pd = ProprioceptiveData(inv_strs, inv_letters, inv_glyphs, inv_oclasses)
        assert np.array_equal(pd.inv_strs, inv_strs)

    def test_from_dict(self):
        d = {
            "inv_strs": np.array(["a"]),
            "inv_letters": np.array([97]),
            "inv_glyphs": np.array([1]),
            "inv_oclasses": np.array([0]),
        }
        pd = ProprioceptiveData.from_dict(d)
        assert np.array_equal(pd.inv_strs, d["inv_strs"])

    def test_from_dict_missing_key(self):
        d = {"inv_strs": np.array(["a"])}
        with pytest.raises(Exception, match="inv_letters"):
            ProprioceptiveData.from_dict(d)


class TestDirection:
    def test_all_values(self):
        assert Direction.up.value == "UP"
        assert Direction.down.value == "DOWN"
        assert Direction.left.value == "LEFT"
        assert Direction.right.value == "RIGHT"
        assert Direction.up_right.value == "UP_RIGHT"
        assert Direction.up_left.value == "UP_LEFT"
        assert Direction.down_right.value == "DOWN_RIGHT"
        assert Direction.down_left.value == "DOWN_LEFT"

    def test_str(self):
        assert str(Direction.up) == "UP"
        assert str(Direction.down_left) == "DOWN_LEFT"


class TestSettled:
    def test_can_instantiate(self):
        s = Settled()
        assert isinstance(s, Settled)


class TestFeature:
    def test_dataclass(self):
        f = Feature(feature_name="test_feat", origin_id=("comp", "type"))
        assert f.feature_name == "test_feat"
        assert f.origin_id == ("comp", "type")


class TestPointFeatureAbstract:
    """Test PointFeature's concrete methods. Since it's abstract, we create a concrete subclass."""

    def _make_concrete_point_feature(self, type_val=65, point=(XLoc(1), YLoc(2))):
        @dataclass(kw_only=True)
        class ConcretePointFeature(PointFeature[MagicMock]):
            def _create_nodes(self):
                return MagicMock()

            def _dbfetch_nodes(self):
                return None

        return ConcretePointFeature(
            feature_name="test",
            origin_id=("c", "t"),
            type=type_val,
            point=point,
        )

    def test_get_points(self):
        f = self._make_concrete_point_feature()
        pts = f.get_points()
        assert pts == {(XLoc(1), YLoc(2))}

    def test_node_hash(self):
        f = self._make_concrete_point_feature(type_val=65)
        assert f.node_hash() == 65


class TestAreaFeatureAbstract:
    """Test AreaFeature's concrete methods via a concrete subclass."""

    def _make_concrete_area_feature(self, type_val=65, points=None, size=3):
        from roc.perception.base import AreaFeature

        if points is None:
            points = {(XLoc(0), YLoc(0)), (XLoc(1), YLoc(0)), (XLoc(2), YLoc(0))}

        @dataclass(kw_only=True)
        class ConcreteAreaFeature(AreaFeature[MagicMock]):
            def _create_nodes(self):
                return MagicMock()

            def _dbfetch_nodes(self):
                return None

        return ConcreteAreaFeature(
            feature_name="area_test",
            origin_id=("c", "t"),
            type=type_val,
            points=points,
            size=size,
        )

    def test_get_points(self):
        pts = {(XLoc(0), YLoc(0)), (XLoc(1), YLoc(1))}
        f = self._make_concrete_area_feature(points=pts, size=2)
        assert f.get_points() == pts

    def test_node_hash(self):
        f = self._make_concrete_area_feature(type_val=42, size=5)
        assert f.node_hash() == hash((42, 5))


class TestToNumpy:
    def test_valid_key(self):
        d = {"foo": np.array([1, 2, 3])}
        result = _to_numpy(d, "foo")
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_missing_key_raises(self):
        d = {"bar": np.array([1])}
        with pytest.raises(Exception, match="Expected 'foo' to exist"):
            _to_numpy(d, "foo")

    def test_non_array_converts(self):
        d = {"foo": [1, 2, 3]}
        result = _to_numpy(d, "foo")
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_already_numpy_returns_same(self):
        arr = np.array([10, 20])
        d = {"x": arr}
        result = _to_numpy(d, "x")
        assert result is arr


class TestCacheRegistry:
    def test_cache_registry_is_defaultdict(self):
        # Accessing a new key should create a WeakValueDictionary
        from weakref import WeakValueDictionary

        key = "unit_test_cache_key_xyz"
        cache = cache_registry[key]
        assert isinstance(cache, WeakValueDictionary)
        # Cleanup
        del cache_registry[key]
