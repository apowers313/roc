"""Per-observation object instance tracking for frame-level object state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from .graphdb import Edge, EdgeConnectionsList, Node
from .location import XLoc, YLoc
from .object import FeatureGroup, Object, ObjectId, ResolutionContext
from .perception import Detail, FeatureNode
from .transformable import Transform, Transformable

if TYPE_CHECKING:
    from .object_transform import ObjectTransform


def _extract_physical_features(
    fg: FeatureGroup,
) -> tuple[int | None, int | None, int | None, int | None, int | None]:
    """Extract physical feature values from a FeatureGroup.

    Returns:
        Tuple of (glyph_type, color_type, shape_type, flood_size, line_size).
    """
    from .feature_extractors.color import ColorNode
    from .feature_extractors.flood import FloodNode
    from .feature_extractors.line import LineNode
    from .feature_extractors.shape import ShapeNode
    from .feature_extractors.single import SingleNode

    glyph_type: int | None = None
    color_type: int | None = None
    shape_type: int | None = None
    flood_size: int | None = None
    line_size: int | None = None

    for fn in fg.feature_nodes:
        if isinstance(fn, SingleNode):
            glyph_type = fn.type
        elif isinstance(fn, ColorNode):
            color_type = fn.type
        elif isinstance(fn, ShapeNode):
            shape_type = fn.type
        elif isinstance(fn, FloodNode):
            flood_size = fn.size
        elif isinstance(fn, LineNode):
            line_size = fn.size

    return glyph_type, color_type, shape_type, flood_size, line_size


def _extract_relational_features(
    rg: RelationshipGroup,
) -> tuple[int | None, int | None, str | None, int | None]:
    """Extract relational feature values from a RelationshipGroup.

    Returns:
        Tuple of (delta_old, delta_new, motion_direction, distance).
    """
    from .feature_extractors.delta import DeltaNode
    from .feature_extractors.distance import DistanceNode
    from .feature_extractors.motion import MotionNode

    delta_old: int | None = None
    delta_new: int | None = None
    motion_direction: str | None = None
    distance: int | None = None

    for fn in rg.feature_nodes:
        if isinstance(fn, DistanceNode):
            distance = fn.size
        elif isinstance(fn, MotionNode):
            motion_direction = fn.direction.value
        elif isinstance(fn, DeltaNode):
            delta_old = fn.old_val
            delta_new = fn.new_val

    return delta_old, delta_new, motion_direction, distance


class ObjectInstance(Node, Transformable):
    """Per-observation record: object type X at position (x,y) at tick t.

    Multiple ObjectInstances can reference the same Object (type) within a single frame.
    Three orcs on screen = three ObjectInstances, one Object. No de-duplication is performed.
    Position is authoritative on ObjectInstance, not on Object.
    Object.last_x/last_y remain for backward compatibility but are not authoritative.
    """

    object_uuid: ObjectId
    x: XLoc
    y: YLoc
    tick: int
    # Physical features
    glyph_type: int | None = Field(default=None)
    color_type: int | None = Field(default=None)
    shape_type: int | None = Field(default=None)
    flood_size: int | None = Field(default=None)
    line_size: int | None = Field(default=None)
    # Relational features (extracted from RelationshipGroup)
    delta_old: int | None = Field(default=None)
    delta_new: int | None = Field(default=None)
    motion_direction: str | None = Field(default=None)
    distance: int | None = Field(default=None)

    @staticmethod
    def from_resolution(
        obj: Object,
        fg: FeatureGroup,
        ctx: ResolutionContext,
        rg: RelationshipGroup | None = None,
    ) -> ObjectInstance:
        """Extract features from FeatureGroup and optional RelationshipGroup."""
        glyph_type, color_type, shape_type, flood_size, line_size = _extract_physical_features(fg)

        delta_old: int | None = None
        delta_new: int | None = None
        motion_direction: str | None = None
        distance: int | None = None
        if rg is not None:
            delta_old, delta_new, motion_direction, distance = _extract_relational_features(rg)

        return ObjectInstance(
            object_uuid=obj.uuid,
            x=ctx.x,
            y=ctx.y,
            tick=ctx.tick,
            glyph_type=glyph_type,
            color_type=color_type,
            shape_type=shape_type,
            flood_size=flood_size,
            line_size=line_size,
            delta_old=delta_old,
            delta_new=delta_new,
            motion_direction=motion_direction,
            distance=distance,
        )

    def same_transform_type(self, other: Any) -> bool:
        """True if other is an ObjectInstance tracking the same Object (by uuid)."""
        return isinstance(other, ObjectInstance) and other.object_uuid == self.object_uuid

    def compatible_transform(self, t: Transform) -> bool:
        """True if the transform is an ObjectTransform."""
        from .object_transform import ObjectTransform

        return isinstance(t, ObjectTransform)

    def create_transform(self, previous: Any) -> ObjectTransform | None:
        """Compute property changes between this and a previous ObjectInstance.

        Returns None if nothing changed.
        """
        from .object_transform import ObjectTransform, _compute_property_changes

        changes = _compute_property_changes(current=self, previous=previous)
        if not changes:
            return None
        return ObjectTransform.from_changes(self.object_uuid, changes)

    def apply_transform(self, t: Transform) -> Node:
        """Apply a transform to produce a new ObjectInstance. Future work."""
        raise NotImplementedError("ObjectInstance.apply_transform is future work")

    def __str__(self) -> str:
        """Human-readable representation."""
        return f"ObjectInstance(uuid={self.object_uuid}, pos=({self.x},{self.y}), tick={self.tick})"


class ObservedAs(Edge):
    """Links an ObjectInstance to the persistent Object it observes."""

    allowed_connections: EdgeConnectionsList = [("ObjectInstance", "Object")]


class SituatedObjectInstance(Edge):
    """Links a Frame to an ObjectInstance situated within it."""

    allowed_connections: EdgeConnectionsList = [("Frame", "ObjectInstance")]


class FrameFeatures(Edge):
    """Links a Frame to a FeatureGroup (replaces FrameAttribute for FeatureGroups)."""

    allowed_connections: EdgeConnectionsList = [("Frame", "FeatureGroup")]


class RelationshipGroup(Node):
    """Collection of relational feature nodes for one observation."""

    @staticmethod
    def from_nodes(nodes: list[FeatureNode]) -> RelationshipGroup:
        """Create a RelationshipGroup and connect feature nodes via Detail edges."""
        rg = RelationshipGroup()
        for fn in nodes:
            Detail.connect(rg, fn)
        return rg

    @property
    def feature_nodes(self) -> list[FeatureNode]:
        """All FeatureNodes connected to this group via Detail edges."""
        return [
            e.dst for e in self.src_edges.select(type="Detail") if isinstance(e.dst, FeatureNode)
        ]


class Relationships(Edge):
    """Links an ObjectInstance to its RelationshipGroup."""

    allowed_connections: EdgeConnectionsList = [("ObjectInstance", "RelationshipGroup")]
