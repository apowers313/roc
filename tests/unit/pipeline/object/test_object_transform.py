# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/object_transform.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.perception.location import XLoc, YLoc
from roc.pipeline.object.object import ObjectId
from roc.pipeline.object.object_instance import ObjectInstance
from roc.pipeline.object.object_transform import (
    DeltaChange,
    DiscreteChange,
    DistanceChange,
    MotionChange,
    ObjectHistory,
    ObjectTransform,
    PositionChange,
    PropertyChange,
    PropertyTransformNode,
    SizeChange,
    TransformDetail,
    _compute_property_changes,
)
from roc.pipeline.temporal.transformable import Transform


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestPositionChange:
    def test_position_change_computed(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(7), y=YLoc(3), tick=2)
        changes = _compute_property_changes(curr, prev)
        assert len(changes) == 1
        assert isinstance(changes[0], PositionChange)
        assert changes[0].dx == 2 and changes[0].dy == 0

    def test_no_movement_returns_empty(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=2)
        changes = _compute_property_changes(curr, prev)
        assert changes == []

    def test_diagonal_movement(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(6), y=YLoc(4), tick=2)
        changes = _compute_property_changes(curr, prev)
        assert len(changes) == 1
        assert isinstance(changes[0], PositionChange)
        assert changes[0].dx == 1 and changes[0].dy == 1

    def test_negative_movement(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(3), y=YLoc(1), tick=2)
        changes = _compute_property_changes(curr, prev)
        assert isinstance(changes[0], PositionChange)
        assert changes[0].dx == -2 and changes[0].dy == -2

    def test_y_only_movement(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(7), tick=2)
        changes = _compute_property_changes(curr, prev)
        assert isinstance(changes[0], PositionChange)
        assert changes[0].dx == 0 and changes[0].dy == 4


class TestObjectTransform:
    def test_is_transform_subclass(self):
        ot = ObjectTransform(
            object_uuid=ObjectId(42),
            num_discrete_changes=0,
            num_continuous_changes=1,
        )
        assert isinstance(ot, Transform)

    def test_from_changes_creates_transform_with_position_nodes(self):
        """PositionChange produces 2 PropertyTransformNodes (x and y)."""
        changes = [PositionChange(dx=2, dy=-1)]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        assert ot.object_uuid == 42
        assert ot.num_continuous_changes == 1
        assert ot.num_discrete_changes == 0
        children: list[PropertyTransformNode] = [
            e.dst  # type: ignore[misc]
            for e in ot.src_edges
            if isinstance(e, TransformDetail)
        ]
        assert len(children) == 2
        names = {c.property_name for c in children}
        assert names == {"x", "y"}
        x_node = next(c for c in children if c.property_name == "x")
        assert x_node.delta == pytest.approx(2.0)
        y_node = next(c for c in children if c.property_name == "y")
        assert y_node.delta == pytest.approx(-1.0)

    def test_from_changes_empty_returns_none(self):
        """No changes -> create_transform returns None."""
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=2)
        t = curr.create_transform(prev)
        assert t is None

    def test_create_transform_returns_object_transform(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(7), y=YLoc(3), tick=2)
        t = curr.create_transform(prev)
        assert isinstance(t, ObjectTransform)
        assert t.object_uuid == 1

    def test_object_transform_str(self):
        ot = ObjectTransform(
            object_uuid=ObjectId(42),
            num_discrete_changes=0,
            num_continuous_changes=1,
        )
        s = str(ot)
        assert "ObjectTransform" in s


class TestPropertyTransformNode:
    def test_create_continuous_node(self):
        ptn = PropertyTransformNode(
            property_name="x",
            change_type="continuous",
            old_value=5,
            new_value=7,
            delta=2.0,
        )
        assert ptn.property_name == "x"
        assert ptn.change_type == "continuous"
        assert ptn.delta == pytest.approx(2.0)

    def test_create_with_none_values(self):
        ptn = PropertyTransformNode(
            property_name="y",
            change_type="continuous",
            old_value=None,
            new_value=None,
            delta=0.0,
        )
        assert ptn.old_value is None
        assert ptn.new_value is None

    def test_is_node_subclass(self):
        ptn = PropertyTransformNode(
            property_name="x",
            change_type="continuous",
            old_value=None,
            new_value=None,
            delta=1.0,
        )
        from roc.db.graphdb import Node

        assert isinstance(ptn, Node)


class TestTransformDetail:
    def test_transform_detail_edge(self):
        ot = ObjectTransform(
            object_uuid=ObjectId(42),
            num_discrete_changes=0,
            num_continuous_changes=1,
        )
        ptn = PropertyTransformNode(
            property_name="x",
            change_type="continuous",
            old_value=None,
            new_value=None,
            delta=2.0,
        )
        edge = TransformDetail.connect(ot, ptn)
        assert edge.src_id == ot.id

    def test_allowed_connections(self):
        assert TransformDetail.model_fields["allowed_connections"].default == [
            ("ObjectTransform", "PropertyTransformNode")
        ]


class TestAllPropertyChanges:
    def test_discrete_change_glyph(self):
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            glyph_type=111,
            color_type=7,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            glyph_type=111,
            color_type=3,
        )
        changes = _compute_property_changes(curr, prev)
        assert any(
            isinstance(c, DiscreteChange)
            and c.property_name == "color_type"
            and c.old_value == 7
            and c.new_value == 3
            for c in changes
        )

    def test_size_change_flood(self):
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            flood_size=5,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            flood_size=8,
        )
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, SizeChange) and c.delta == 3 for c in changes)

    def test_distance_change(self):
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            distance=10,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            distance=8,
        )
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, DistanceChange) and c.delta == -2 for c in changes)

    def test_motion_change(self):
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            motion_direction="LEFT",
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            motion_direction="RIGHT",
        )
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, MotionChange) for c in changes)

    def test_none_values_produce_no_change_for_discrete(self):
        """Discrete properties (glyph, color, shape): if EITHER side is None, no change."""
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            glyph_type=111,
            color_type=None,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            glyph_type=111,
            color_type=7,
        )
        changes = _compute_property_changes(curr, prev)
        assert not any(
            isinstance(c, DiscreteChange) and c.property_name == "color_type" for c in changes
        )

    def test_motion_direction_none_to_value_is_a_change(self):
        """Motion direction: None->value IS a change (unlike discrete properties)."""
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            motion_direction=None,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            motion_direction="LEFT",
        )
        changes = _compute_property_changes(curr, prev)
        assert any(
            isinstance(c, MotionChange) and c.old_direction is None and c.new_direction == "LEFT"
            for c in changes
        )

    def test_delta_none_to_pair_is_a_change(self):
        """Delta feature: None->pair IS a change (same semantics as motion_direction)."""
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            delta_old=None,
            delta_new=None,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            delta_old=100,
            delta_new=111,
        )
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, DeltaChange) for c in changes)

    def test_motion_direction_both_none_is_not_a_change(self):
        """None == None, so no MotionChange."""
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            motion_direction=None,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            motion_direction=None,
        )
        changes = _compute_property_changes(curr, prev)
        assert not any(isinstance(c, MotionChange) for c in changes)

    def test_multiple_simultaneous_changes(self):
        """Position + discrete + continuous changes all detected."""
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(5),
            y=YLoc(3),
            tick=1,
            color_type=7,
            distance=10,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(7),
            y=YLoc(4),
            tick=2,
            color_type=3,
            distance=8,
        )
        changes = _compute_property_changes(curr, prev)
        types = {type(c) for c in changes}
        assert PositionChange in types
        assert DiscreteChange in types
        assert DistanceChange in types

    def test_size_change_line(self):
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            line_size=4,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            line_size=6,
        )
        changes = _compute_property_changes(curr, prev)
        assert any(
            isinstance(c, SizeChange) and c.property_name == "line_size" and c.delta == 2
            for c in changes
        )

    def test_delta_both_none_is_not_a_change(self):
        """Both old/new delta pairs None means no DeltaChange."""
        prev = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            delta_old=None,
            delta_new=None,
        )
        curr = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=2,
            delta_old=None,
            delta_new=None,
        )
        changes = _compute_property_changes(curr, prev)
        assert not any(isinstance(c, DeltaChange) for c in changes)


class TestFromChangesAllTypes:
    def _get_children(self, ot: ObjectTransform) -> list[PropertyTransformNode]:
        return [
            e.dst  # type: ignore[misc]
            for e in ot.src_edges
            if isinstance(e, TransformDetail)
        ]

    def test_discrete_change_creates_node(self):
        changes: list[PropertyChange] = [
            DiscreteChange(property_name="color_type", old_value=7, new_value=3),
        ]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        assert ot.num_discrete_changes == 1
        assert ot.num_continuous_changes == 0
        children = self._get_children(ot)
        assert len(children) == 1
        assert children[0].property_name == "color_type"
        assert children[0].change_type == "discrete"
        assert children[0].old_value == 7
        assert children[0].new_value == 3
        assert children[0].delta is None

    def test_size_change_creates_node(self):
        changes: list[PropertyChange] = [
            SizeChange(property_name="flood_size", old_value=5, new_value=8, delta=3),
        ]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        assert ot.num_continuous_changes == 1
        children = self._get_children(ot)
        assert len(children) == 1
        assert children[0].property_name == "flood_size"
        assert children[0].change_type == "continuous"
        assert children[0].delta == pytest.approx(3.0)

    def test_distance_change_creates_node(self):
        changes: list[PropertyChange] = [
            DistanceChange(old_value=10, new_value=8, delta=-2),
        ]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        children = self._get_children(ot)
        assert len(children) == 1
        assert children[0].property_name == "distance"
        assert children[0].change_type == "continuous"
        assert children[0].delta == -2.0

    def test_motion_change_creates_node(self):
        changes: list[PropertyChange] = [
            MotionChange(old_direction=None, new_direction="LEFT"),
        ]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        assert ot.num_discrete_changes == 1
        children = self._get_children(ot)
        assert len(children) == 1
        assert children[0].property_name == "motion_direction"
        assert children[0].change_type == "discrete"
        assert children[0].old_value is None
        assert children[0].new_value == "LEFT"

    def test_delta_change_creates_node(self):
        changes: list[PropertyChange] = [
            DeltaChange(old_pair=(100, 111), new_pair=(200, 222)),
        ]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        assert ot.num_discrete_changes == 1
        children = self._get_children(ot)
        assert len(children) == 1
        assert children[0].property_name == "delta"
        assert children[0].change_type == "discrete"
        assert children[0].old_value == (100, 111)
        assert children[0].new_value == (200, 222)

    def test_mixed_changes(self):
        changes: list[PropertyChange] = [
            PositionChange(dx=2, dy=-1),
            DiscreteChange(property_name="color_type", old_value=7, new_value=3),
            DistanceChange(old_value=10, new_value=8, delta=-2),
        ]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        # Position is continuous, distance is continuous, color is discrete
        assert ot.num_continuous_changes == 2
        assert ot.num_discrete_changes == 1
        children = self._get_children(ot)
        # Position creates 2 nodes (x + y), color creates 1, distance creates 1 = 4 total
        assert len(children) == 4


class TestObjectHistory:
    def test_object_history_edge(self):
        from roc.pipeline.object.object import Object

        obj = Object(uuid=ObjectId(42))
        ot = ObjectTransform(
            object_uuid=ObjectId(42),
            num_discrete_changes=1,
            num_continuous_changes=1,
        )
        edge = ObjectHistory.connect(obj, ot)
        assert edge.src_id == obj.id
        assert edge.dst_id == ot.id

    def test_object_history_allowed_connections(self):
        assert ObjectHistory.model_fields["allowed_connections"].default == [
            ("Object", "ObjectTransform")
        ]
