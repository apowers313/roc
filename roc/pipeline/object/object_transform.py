"""Object-level transforms tracking property changes between frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .graphdb import Edge, EdgeConnectionsList, Node
from .object import ObjectId
from .transformable import Transform

if TYPE_CHECKING:
    from .object_instance import ObjectInstance


@dataclass
class PositionChange:
    """Records a position delta between two ObjectInstances."""

    dx: int
    dy: int


@dataclass
class DiscreteChange:
    """Records a discrete property change (old value -> new value)."""

    property_name: str
    old_value: Any
    new_value: Any


@dataclass
class SizeChange:
    """Records a size property change (flood_size, line_size)."""

    property_name: str
    old_value: int
    new_value: int
    delta: int


@dataclass
class DistanceChange:
    """Records a distance property change."""

    old_value: int
    new_value: int
    delta: int


@dataclass
class MotionChange:
    """Records a motion direction change."""

    old_direction: str | None
    new_direction: str | None


@dataclass
class DeltaChange:
    """Records a delta feature change (old_val/new_val pair)."""

    old_pair: tuple[int, int] | None
    new_pair: tuple[int, int] | None


PropertyChange = (
    PositionChange | DiscreteChange | SizeChange | DistanceChange | MotionChange | DeltaChange
)


class ObjectTransform(Transform):
    """Transform representing property changes for a single tracked object between frames."""

    object_uuid: ObjectId
    num_discrete_changes: int = Field(default=0)
    num_continuous_changes: int = Field(default=0)

    @staticmethod
    def from_changes(object_uuid: ObjectId, changes: Sequence[PropertyChange]) -> ObjectTransform:
        """Create an ObjectTransform with PropertyTransformNode children from a list of changes."""
        num_continuous = 0
        num_discrete = 0

        ot = ObjectTransform(
            object_uuid=object_uuid,
            num_discrete_changes=0,
            num_continuous_changes=0,
        )

        for change in changes:
            nodes = _change_to_nodes(change)
            for node in nodes:
                TransformDetail.connect(ot, node)
            # Count per-change, not per-node (PositionChange -> 2 nodes but 1 change)
            if isinstance(change, (PositionChange, SizeChange, DistanceChange)):
                num_continuous += 1
            else:
                num_discrete += 1

        ot.num_continuous_changes = num_continuous
        ot.num_discrete_changes = num_discrete

        return ot

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ObjectTransform(uuid={self.object_uuid}, "
            f"continuous={self.num_continuous_changes}, "
            f"discrete={self.num_discrete_changes})"
        )


class PropertyTransformNode(Node):
    """A single property change within an ObjectTransform."""

    property_name: str
    change_type: str  # "continuous" or "discrete"
    old_value: Any = Field(default=None)
    new_value: Any = Field(default=None)
    delta: float | None = Field(default=None)


class TransformDetail(Edge):
    """Links an ObjectTransform to its PropertyTransformNode children."""

    allowed_connections: EdgeConnectionsList = [("ObjectTransform", "PropertyTransformNode")]


class ObjectHistory(Edge):
    """Links an Object to the ObjectTransforms computed from its observations."""

    allowed_connections: EdgeConnectionsList = [("Object", "ObjectTransform")]


def _change_to_nodes(change: PropertyChange) -> list[PropertyTransformNode]:
    """Convert a PropertyChange into one or more PropertyTransformNodes."""
    if isinstance(change, PositionChange):
        return [
            PropertyTransformNode(
                property_name="x",
                change_type="continuous",
                old_value=None,
                new_value=None,
                delta=float(change.dx),
            ),
            PropertyTransformNode(
                property_name="y",
                change_type="continuous",
                old_value=None,
                new_value=None,
                delta=float(change.dy),
            ),
        ]
    if isinstance(change, DiscreteChange):
        return [
            PropertyTransformNode(
                property_name=change.property_name,
                change_type="discrete",
                old_value=change.old_value,
                new_value=change.new_value,
                delta=None,
            ),
        ]
    if isinstance(change, SizeChange):
        return [
            PropertyTransformNode(
                property_name=change.property_name,
                change_type="continuous",
                old_value=change.old_value,
                new_value=change.new_value,
                delta=float(change.delta),
            ),
        ]
    if isinstance(change, DistanceChange):
        return [
            PropertyTransformNode(
                property_name="distance",
                change_type="continuous",
                old_value=change.old_value,
                new_value=change.new_value,
                delta=float(change.delta),
            ),
        ]
    if isinstance(change, MotionChange):
        return [
            PropertyTransformNode(
                property_name="motion_direction",
                change_type="discrete",
                old_value=change.old_direction,
                new_value=change.new_direction,
                delta=None,
            ),
        ]
    # DeltaChange (must be last -- all other types checked above)
    return [
        PropertyTransformNode(
            property_name="delta",
            change_type="discrete",
            old_value=change.old_pair,
            new_value=change.new_pair,
            delta=None,
        ),
    ]


def _position_change(current: ObjectInstance, previous: ObjectInstance) -> PositionChange | None:
    """Return a PositionChange if the object moved, else None."""
    dx = int(current.x) - int(previous.x)
    dy = int(current.y) - int(previous.y)
    if dx != 0 or dy != 0:
        return PositionChange(dx=dx, dy=dy)
    return None


def _discrete_changes(current: ObjectInstance, previous: ObjectInstance) -> list[DiscreteChange]:
    """Return DiscreteChange entries for glyph_type, color_type, shape_type."""
    changes: list[DiscreteChange] = []
    for prop in ("glyph_type", "color_type", "shape_type"):
        old_val = getattr(previous, prop)
        new_val = getattr(current, prop)
        if old_val is not None and new_val is not None and old_val != new_val:
            changes.append(DiscreteChange(property_name=prop, old_value=old_val, new_value=new_val))
    return changes


def _size_changes(current: ObjectInstance, previous: ObjectInstance) -> list[SizeChange]:
    """Return SizeChange entries for flood_size and line_size."""
    changes: list[SizeChange] = []
    for prop in ("flood_size", "line_size"):
        old_val = getattr(previous, prop)
        new_val = getattr(current, prop)
        if old_val is not None and new_val is not None and old_val != new_val:
            changes.append(
                SizeChange(
                    property_name=prop,
                    old_value=old_val,
                    new_value=new_val,
                    delta=new_val - old_val,
                )
            )
    return changes


def _distance_change(current: ObjectInstance, previous: ObjectInstance) -> DistanceChange | None:
    """Return a DistanceChange if both distances are known and differ, else None."""
    if previous.distance is not None and current.distance is not None:
        d = current.distance - previous.distance
        if d != 0:
            return DistanceChange(old_value=previous.distance, new_value=current.distance, delta=d)
    return None


def _motion_change(current: ObjectInstance, previous: ObjectInstance) -> MotionChange | None:
    """Return a MotionChange if motion_direction changed, else None."""
    if previous.motion_direction != current.motion_direction:
        return MotionChange(
            old_direction=previous.motion_direction, new_direction=current.motion_direction
        )
    return None


def _delta_pair_change(current: ObjectInstance, previous: ObjectInstance) -> DeltaChange | None:
    """Return a DeltaChange if the delta pair changed, else None."""
    prev_pair: tuple[int, int] | None = None
    if previous.delta_old is not None and previous.delta_new is not None:
        prev_pair = (previous.delta_old, previous.delta_new)
    curr_pair: tuple[int, int] | None = None
    if current.delta_old is not None and current.delta_new is not None:
        curr_pair = (current.delta_old, current.delta_new)
    if prev_pair != curr_pair:
        return DeltaChange(old_pair=prev_pair, new_pair=curr_pair)
    return None


def _compute_property_changes(
    current: ObjectInstance, previous: ObjectInstance
) -> list[PropertyChange]:
    """Compute all property changes between two ObjectInstances."""
    changes: list[PropertyChange] = []

    pos = _position_change(current, previous)
    if pos is not None:
        changes.append(pos)

    changes.extend(_discrete_changes(current, previous))
    changes.extend(_size_changes(current, previous))

    dist = _distance_change(current, previous)
    if dist is not None:
        changes.append(dist)

    motion = _motion_change(current, previous)
    if motion is not None:
        changes.append(motion)

    delta = _delta_pair_change(current, previous)
    if delta is not None:
        changes.append(delta)

    return changes
