# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/object_instance.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.db.graphdb import Node
from roc.perception.location import XLoc, YLoc
from roc.pipeline.object.object import FeatureGroup, Features, Object, ObjectId, ResolutionContext
from roc.pipeline.object.object_instance import (
    FrameFeatures,
    ObjectInstance,
    ObservedAs,
    RelationshipGroup,
    Relationships,
    SituatedObjectInstance,
)
from roc.pipeline.temporal.sequencer import Frame


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestObjectInstance:
    def test_create_with_position_and_features(self):
        """ObjectInstance stores position, tick, and feature values."""
        oi = ObjectInstance(
            object_uuid=ObjectId(42),
            x=XLoc(5),
            y=YLoc(3),
            tick=7,
            glyph_type=111,
            color_type=7,
            shape_type=64,
        )
        assert oi.object_uuid == 42
        assert oi.x == 5 and oi.y == 3
        assert oi.tick == 7
        assert oi.glyph_type == 111
        assert oi.color_type == 7
        assert oi.shape_type == 64

    def test_default_feature_types_are_none(self):
        """Feature types default to None when not specified."""
        oi = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1)
        assert oi.glyph_type is None
        assert oi.color_type is None
        assert oi.shape_type is None

    def test_from_resolution_extracts_physical_features(self):
        """Factory extracts glyph, color, shape from FeatureGroup."""
        from roc.perception.feature_extractors.color import ColorNode
        from roc.perception.feature_extractors.shape import ShapeNode
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111), ColorNode(type=7), ShapeNode(type=64)])
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(5), y=YLoc(3), tick=7)
        )
        assert oi.object_uuid == obj.uuid
        assert oi.glyph_type == 111
        assert oi.color_type == 7
        assert oi.shape_type == 64
        assert oi.x == 5 and oi.y == 3
        assert oi.tick == 7

    def test_from_resolution_handles_missing_features(self):
        """Missing features produce None, not errors."""
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111)])
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1)
        )
        assert oi.glyph_type == 111
        assert oi.color_type is None
        assert oi.shape_type is None

    def test_from_resolution_handles_empty_feature_group(self):
        """Empty FeatureGroup produces all-None feature types."""
        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup()
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1)
        )
        assert oi.glyph_type is None
        assert oi.color_type is None
        assert oi.shape_type is None

    def test_same_transform_type_matches_by_uuid(self):
        oi1 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        oi2 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(5), y=YLoc(3), tick=2)
        assert oi1.same_transform_type(oi2)

    def test_same_transform_type_rejects_different_uuid(self):
        oi1 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        oi2 = ObjectInstance(object_uuid=ObjectId(99), x=XLoc(0), y=YLoc(0), tick=2)
        assert not oi1.same_transform_type(oi2)

    def test_same_transform_type_rejects_intrinsic(self):
        from roc.pipeline.intrinsic import IntrinsicNode

        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        intrinsic = IntrinsicNode(name="hp", raw_value=15, normalized_value=0.8)
        assert not oi.same_transform_type(intrinsic)

    def test_compatible_transform(self):
        from roc.pipeline.object.object_transform import ObjectTransform

        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        ot = ObjectTransform(
            object_uuid=ObjectId(42),
            num_discrete_changes=0,
            num_continuous_changes=1,
        )
        assert oi.compatible_transform(ot)

    def test_compatible_transform_rejects_intrinsic_transform(self):
        from roc.pipeline.intrinsic import IntrinsicTransform

        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        it = IntrinsicTransform(name="hp", normalized_change=0.1)
        assert not oi.compatible_transform(it)

    def test_create_transform_returns_object_transform(self):
        from roc.pipeline.object.object_transform import ObjectTransform

        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(7), y=YLoc(3), tick=2)
        t = curr.create_transform(prev)
        assert isinstance(t, ObjectTransform)
        assert t.object_uuid == 1

    def test_create_transform_returns_none_when_unchanged(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=2)
        t = curr.create_transform(prev)
        assert t is None

    def test_apply_transform_raises_not_implemented(self):
        """Stub for this phase."""
        from roc.pipeline.object.object_transform import ObjectTransform

        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        ot = ObjectTransform(
            object_uuid=ObjectId(42),
            num_discrete_changes=0,
            num_continuous_changes=0,
        )
        with pytest.raises(NotImplementedError):
            oi.apply_transform(ot)

    def test_is_node_subclass(self):
        """ObjectInstance is a graph Node."""
        oi = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1)
        assert isinstance(oi, Node)

    def test_labels_include_object_instance(self):
        """ObjectInstance has its own label in the graph."""
        oi = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1)
        assert "ObjectInstance" in oi.labels


class TestEdgeSchemas:
    def test_observed_as_connects_instance_to_object(self):
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        obj = Object(uuid=ObjectId(42))
        edge = ObservedAs.connect(oi, obj)
        assert edge.src_id == oi.id

    def test_situated_object_instance_edge(self):
        frame = Frame(tick=1)
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        edge = SituatedObjectInstance.connect(frame, oi)
        assert edge.src_id == frame.id

    def test_frame_features_edge(self):
        frame = Frame(tick=1)
        fg = FeatureGroup()
        edge = FrameFeatures.connect(frame, fg)
        assert edge.src_id == frame.id

    def test_features_allows_object_instance(self):
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        fg = FeatureGroup()
        edge = Features.connect(oi, fg)
        assert edge.src_id == oi.id

    def test_observed_as_allowed_connections(self):
        assert ObservedAs.model_fields["allowed_connections"].default == [
            ("ObjectInstance", "Object")
        ]

    def test_situated_object_instance_allowed_connections(self):
        assert SituatedObjectInstance.model_fields["allowed_connections"].default == [
            ("Frame", "ObjectInstance")
        ]

    def test_frame_features_allowed_connections(self):
        assert FrameFeatures.model_fields["allowed_connections"].default == [
            ("Frame", "FeatureGroup")
        ]


class TestFeaturesConnectExistingMatch:
    def test_existing_object_gets_feature_group_linked(self):
        """Prerequisite fix: Features.connect works for ObjectInstance -> FeatureGroup."""
        from roc.perception.feature_extractors.single import SingleNode

        fg1 = FeatureGroup.from_nodes([SingleNode(type=111)])
        obj = Object.with_features(fg1)
        fg2 = FeatureGroup.from_nodes([SingleNode(type=111)])
        Features.connect(obj, fg2)
        assert len(obj.feature_groups) == 2


class TestRelationalFields:
    def test_relational_field_defaults_are_none(self):
        """All relational fields default to None."""
        oi = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1)
        assert oi.flood_size is None
        assert oi.line_size is None
        assert oi.delta_old is None
        assert oi.delta_new is None
        assert oi.motion_direction is None
        assert oi.distance is None

    def test_create_with_relational_fields(self):
        oi = ObjectInstance(
            object_uuid=ObjectId(1),
            x=XLoc(0),
            y=YLoc(0),
            tick=1,
            flood_size=5,
            line_size=4,
            distance=10,
            motion_direction="LEFT",
            delta_old=100,
            delta_new=111,
        )
        assert oi.flood_size == 5
        assert oi.line_size == 4
        assert oi.distance == 10
        assert oi.motion_direction == "LEFT"
        assert oi.delta_old == 100
        assert oi.delta_new == 111


class TestRelationshipGroup:
    def test_from_nodes_connects_detail_edges(self):
        from roc.perception.feature_extractors.distance import DistanceNode
        from roc.perception.feature_extractors.motion import MotionNode
        from roc.perception.base import Direction

        dn = DistanceNode(size=5)
        mn = MotionNode(type=1, direction=Direction.left)
        rg = RelationshipGroup.from_nodes([dn, mn])
        assert len(rg.feature_nodes) == 2

    def test_from_nodes_empty(self):
        rg = RelationshipGroup.from_nodes([])
        assert len(rg.feature_nodes) == 0

    def test_relationships_edge(self):
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        rg = RelationshipGroup()
        edge = Relationships.connect(oi, rg)
        assert edge.src_id == oi.id

    def test_relationships_allowed_connections(self):
        assert Relationships.model_fields["allowed_connections"].default == [
            ("ObjectInstance", "RelationshipGroup")
        ]

    def test_is_node_subclass(self):
        rg = RelationshipGroup()
        assert isinstance(rg, Node)


class TestFromResolutionWithRelationshipGroup:
    def test_extracts_distance_from_relationship_group(self):
        from roc.perception.feature_extractors.distance import DistanceNode
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111)])
        rg = RelationshipGroup.from_nodes([DistanceNode(size=5)])
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1), rg=rg
        )
        assert oi.distance == 5

    def test_extracts_motion_from_relationship_group(self):
        from roc.perception.feature_extractors.motion import MotionNode
        from roc.perception.feature_extractors.single import SingleNode
        from roc.perception.base import Direction

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111)])
        rg = RelationshipGroup.from_nodes([MotionNode(type=1, direction=Direction.left)])
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1), rg=rg
        )
        assert oi.motion_direction == "LEFT"

    def test_extracts_delta_from_relationship_group(self):
        from roc.perception.feature_extractors.delta import DeltaNode
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111)])
        rg = RelationshipGroup.from_nodes([DeltaNode(old_val=100, new_val=111)])
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1), rg=rg
        )
        assert oi.delta_old == 100
        assert oi.delta_new == 111

    def test_extracts_flood_size_from_feature_group(self):
        from roc.perception.feature_extractors.flood import FloodNode
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes(
            [
                SingleNode(type=111),
                FloodNode(type=1, size=8, color=7, shape=64),
            ]
        )
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1)
        )
        assert oi.flood_size == 8

    def test_extracts_line_size_from_feature_group(self):
        from roc.perception.feature_extractors.line import LineNode
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes(
            [
                SingleNode(type=111),
                LineNode(type=1, size=4, color=7, shape=64),
            ]
        )
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1)
        )
        assert oi.line_size == 4

    def test_from_resolution_without_rg_still_works(self):
        """Backward compatible: no rg parameter means no relational features."""
        from roc.perception.feature_extractors.single import SingleNode

        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111)])
        oi = ObjectInstance.from_resolution(
            obj, fg, ResolutionContext(x=XLoc(0), y=YLoc(0), tick=1)
        )
        assert oi.distance is None
        assert oi.motion_direction is None
        assert oi.delta_old is None
