# Implementation Plan: Object Transforms & Multi-Cycle Attention

## Overview

This plan implements two coupled changes described in `design/object-transforms.md`:

1. **Multi-cycle attention**: The attention system runs N serial cycles per game step (default 4), resolving multiple objects per frame instead of one.
2. **Object transforms**: A new `ObjectInstance` hub node tracks per-observation object state and implements `Transformable` to compute property deltas between frames.

These are coupled because object transforms require the same object in consecutive frames, which rarely happens with single-object attention + IOR. Multi-cycle attention populates each frame with multiple objects, making same-object matches across frames common.

### Phase Structure

Each phase is a **vertical slice** -- it cuts through the full stack (graph types, pipeline logic, backend emission, dashboard UI) to deliver one user-visible feature end-to-end. Every phase produces something you can see in the running dashboard.

| Phase | What You Can See | Duration |
|-------|-----------------|----------|
| 1 | Position deltas for "@" in Transition panel | 2-3 days |
| 2 | All property changes + object history API | 2-3 days |
| 3 | 4 objects per frame, attention cycle stepper | 2-3 days |
| 4 | Click any object to see its full history modal | 2-3 days |

---

## Phase 1: Single-Object Position Tracking (End-to-End)

**What this phase accomplishes**: The thinnest possible vertical slice through the entire stack. With single-object attention (no multi-cycle yet), the system creates an ObjectInstance per observation, computes position deltas between frames, and shows them in the dashboard. You can run a game and see the "@" character's dx/dy in the Transition panel.

**Duration**: 2-3 days

### New Files

| File | Contents |
|------|----------|
| `roc/object_instance.py` | ObjectInstance node, ObservedAs edge, SituatedObjectInstance edge, FrameFeatures edge |
| `roc/object_transform.py` | ObjectTransform, PropertyTransformNode, TransformDetail edge, PositionChange, _compute_property_changes (position-only) |
| `tests/unit/test_object_instance.py` | ObjectInstance creation, edge schemas, Transformable methods |
| `tests/unit/test_object_transform.py` | Position delta computation, ObjectTransform.from_changes, PropertyTransformNode |

### Modified Files

| File | Change |
|------|--------|
| `roc/object.py` | Add `Features.connect(o, fg)` for existing matches; update `Features.allowed_connections` to include `(ObjectInstance, FeatureGroup)` |
| `roc/sequencer.py` | Create ObjectInstance in `handle_object_resolution_event`, attach via FrameFeatures + SituatedObjectInstance; update `Frame.transformable` to traverse SituatedObjectInstance edges; update `Frame.objects` to also traverse `SituatedObjectInstance -> ObjectInstance -> ObservedAs -> Object` (not just the old `FrameAttribute -> FeatureGroup -> Features -> Object` path). **Phase 1 Sequencer behavior**: use the original FeatureGroup as-is (no feature splitting -- that's Phase 2). Replace `FrameAttribute.connect(frame, fg)` with `FrameFeatures.connect(frame, fg)` for FeatureGroup attachment. Link ObjectInstance to the unsplit FeatureGroup via `Features.connect(oi, fg)`. Remove `("Frame", "FeatureGroup")` from `FrameAttribute.allowed_connections` since `FrameFeatures` now handles this -- or if needed for backward compat with historical data replay, document explicitly and remove in Phase 2. |
| `roc/transformer.py` | **Critical**: Update `_select_transformable_edges` to also select `SituatedObjectInstance` edges (not just `FrameAttribute`), so ObjectInstances are found as Transformable nodes. The current function filters by `e.type == "FrameAttribute"` which will miss ObjectInstances entirely. Either add `SituatedObjectInstance` to the filter, or rewrite to check `isinstance(e.dst, Transformable)` instead of filtering by edge type. Also verify that `Change.connect(transform, objectTransform)` works with the existing `Change.allowed_connections` -- ObjectTransform extends Transform, so `("Transform", "Transform")` should cover it via label hierarchy, but if label matching is exact-only, add `("Transform", "ObjectTransform")` to `Change.allowed_connections`. No ambiguity detection needed yet (deferred to Phase 3 since single-object frames have no ambiguity). |
| `roc/reporting/state.py` | Emit `object_transforms` in `transform_summary` (list of {uuid, changes}). Also enhance `sequence_summary.objects` to include per-object fields: `glyph`, `color`, `shape`, `x`, `y`, `matched_previous` (bool -- whether this object had a transform computed), `resolve_count` (requires traversal: ObjectInstance -> ObservedAs -> Object -> Object.resolve_count). Phase 3 adds `cycle_number`. |
| `dashboard-ui/src/components/panels/TransitionPanel.tsx` | Add object transform rows below intrinsic rows (simple table: object glyph, dx, dy) |
| `dashboard-ui/src/components/panels/SequencePanel.tsx` | Show object glyph + position alongside existing intrinsic display |

### Tests to Write First

- `tests/unit/test_object_instance.py`:

```python
class TestObjectInstance:
    def test_create_with_position_and_features(self):
        """ObjectInstance stores position, tick, and feature values."""
        oi = ObjectInstance(
            object_uuid=ObjectId(42), x=XLoc(5), y=YLoc(3), tick=7,
            glyph_type=111, color_type=7, shape_type=64,
        )
        assert oi.object_uuid == 42
        assert oi.x == 5 and oi.y == 3
        assert oi.glyph_type == 111

    def test_from_resolution_extracts_physical_features(self):
        """Factory extracts glyph, color, shape from FeatureGroup."""
        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111), ColorNode(type=7), ShapeNode(type=64)])
        # Phase 1: no RelationshipGroup yet, pass empty or None
        oi = ObjectInstance.from_resolution(obj, fg, x=XLoc(5), y=YLoc(3), tick=7)
        assert oi.glyph_type == 111 and oi.color_type == 7

    def test_from_resolution_handles_missing_features(self):
        """Missing features produce None, not errors."""
        obj = Object(uuid=ObjectId(1))
        fg = FeatureGroup.from_nodes([SingleNode(type=111)])
        oi = ObjectInstance.from_resolution(obj, fg, x=XLoc(0), y=YLoc(0), tick=1)
        assert oi.glyph_type == 111
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
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        intrinsic = IntrinsicNode(name="hp", raw_value=15, normalized_value=0.8)
        assert not oi.same_transform_type(intrinsic)

    def test_compatible_transform(self):
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        ot = ObjectTransform(object_uuid=ObjectId(42),
                             num_discrete_changes=0, num_continuous_changes=1)
        assert oi.compatible_transform(ot)

    def test_apply_transform_raises_not_implemented(self):
        """Stub for this phase."""
        oi = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(0), y=YLoc(0), tick=1)
        ot = ObjectTransform(object_uuid=ObjectId(42),
                             num_discrete_changes=0, num_continuous_changes=0)
        with pytest.raises(NotImplementedError):
            oi.apply_transform(ot)


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
```

- `tests/unit/test_object_transform.py`:

```python
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
        assert changes[0].dx == 1 and changes[0].dy == 1


class TestObjectTransform:
    def test_from_changes_creates_transform_with_position_nodes(self):
        """PositionChange produces 2 PropertyTransformNodes (x and y)."""
        changes = [PositionChange(dx=2, dy=-1)]
        ot = ObjectTransform.from_changes(ObjectId(42), changes)
        assert ot.object_uuid == 42
        assert ot.num_continuous_changes == 1
        children = [e.dst for e in ot.src_edges if isinstance(e, TransformDetail)]
        assert len(children) == 2
        names = {c.property_name for c in children}
        assert names == {"x", "y"}
        x_node = next(c for c in children if c.property_name == "x")
        assert x_node.delta == 2.0

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

    def test_transform_detail_edge(self):
        ot = ObjectTransform(object_uuid=ObjectId(42),
                             num_discrete_changes=0, num_continuous_changes=1)
        ptn = PropertyTransformNode(property_name="x", change_type="continuous",
                                    old_value=None, new_value=None, delta=2.0)
        edge = TransformDetail.connect(ot, ptn)
        assert edge.src_id == ot.id


class TestFeaturesConnectExistingMatch:
    def test_existing_object_gets_feature_group_linked(self):
        """Prerequisite fix: Features.connect for existing matches."""
        fg1 = FeatureGroup.from_nodes([SingleNode(type=111)])
        obj = Object.with_features(fg1)
        fg2 = FeatureGroup.from_nodes([SingleNode(type=111)])
        Features.connect(obj, fg2)
        assert len(obj.feature_groups) == 2
```

- `tests/unit/test_sequencer.py` (additions):

```python
class TestSequencerObjectInstance:
    def test_creates_object_instance_on_resolution(self):
        """Sequencer creates ObjectInstance when it receives ResolvedObject."""
        # Send ResolvedObject event to sequencer
        # Verify: current frame has SituatedObjectInstance edge

    def test_object_instance_connected_to_object(self):
        """ObjectInstance -> ObservedAs -> Object."""
        # Send ResolvedObject, check ObservedAs edge exists

    def test_frame_transformable_includes_object_instance(self):
        """Frame.transformable returns ObjectInstances alongside IntrinsicNodes."""
        # Create frame with both
        # Verify both types returned
```

- `tests/unit/test_transformer.py` (additions):

```python
class TestTransformerObjectTransforms:
    def test_computes_position_transform_for_matching_object(self):
        """Same Object in both frames produces an ObjectTransform with position delta."""
        # Create two frames with ObjectInstance(uuid=X) at different positions
        # Connect with NextFrame
        # Run _compute_transforms
        # Verify: Transform has Change edge to ObjectTransform

    def test_intrinsic_transforms_still_work(self):
        """ObjectInstance transforms coexist with IntrinsicTransforms."""
        # Create frames with both ObjectInstances and IntrinsicNodes
        # Verify both transform types present

    def test_unmatched_objects_produce_no_transform(self):
        """Object in frame1 but not frame2 -> no ObjectTransform."""
        # Frame1: ObjectInstance(uuid=X), Frame2: ObjectInstance(uuid=Y)
        # Verify: no ObjectTransform
```

- `dashboard-ui/src/components/panels/TransitionPanel.test.tsx` (additions):

```typescript
describe("TransitionPanel object transforms", () => {
  it("shows object position delta when present", () => {
    const data = {
      transform_summary: {
        object_transforms: [
          { uuid: 42, changes: [{ property: "x", delta: 2 }, { property: "y", delta: 0 }] }
        ],
        // ... existing intrinsic transforms
      }
    };
    render(<TransitionPanel data={data} />);
    expect(screen.getByText(/x: \+2/)).toBeInTheDocument();
  });

  it("handles empty object transforms gracefully", () => {
    // No object_transforms field (old data) -> renders without error
  });
});
```

### Implementation Details

**`roc/object_instance.py`** (NEW):

```python
class ObjectInstance(Node, Transformable):
    """Per-observation record: object type X at position (x,y) at tick t.

    Multiple ObjectInstances can reference the same Object (type) within a single frame.
    Three orcs on screen = three ObjectInstances, one Object. No de-duplication is performed
    (design decision D13). Position is authoritative on ObjectInstance, not on Object.
    Object.last_x/last_y remain for backward compatibility but are not authoritative.
    """
    object_uuid: ObjectId
    x: XLoc
    y: YLoc
    tick: int
    # Physical features (Phase 1: extracted from FeatureGroup)
    glyph_type: int | None = None
    color_type: int | None = None
    shape_type: int | None = None
    # Remaining fields (flood_size, line_size, distance, etc.) added in Phase 2

    @staticmethod
    def from_resolution(obj, fg, x, y, tick) -> ObjectInstance:
        """Extract physical features from FeatureGroup. Phase 2 adds RelationshipGroup."""
        # Walk fg.feature_nodes, match on SingleNode/ColorNode/ShapeNode
        ...

    def same_transform_type(self, other) -> bool:
        return isinstance(other, ObjectInstance) and other.object_uuid == self.object_uuid

    def compatible_transform(self, t) -> bool:
        return isinstance(t, ObjectTransform)

    def create_transform(self, previous) -> ObjectTransform | None:
        changes = _compute_property_changes(current=self, previous=previous)
        if not changes:
            return None
        return ObjectTransform.from_changes(self.object_uuid, changes)

    def apply_transform(self, t) -> ObjectInstance:
        raise NotImplementedError("ObjectInstance.apply_transform is future work")

class ObservedAs(Edge):
    allowed_connections = [("ObjectInstance", "Object")]

class SituatedObjectInstance(Edge):
    allowed_connections = [("Frame", "ObjectInstance")]

class FrameFeatures(Edge):
    allowed_connections = [("Frame", "FeatureGroup")]
```

**`roc/object_transform.py`** (NEW):

```python
@dataclass
class PositionChange:
    dx: int
    dy: int
# Phase 2 adds: SizeChange, DistanceChange, DiscreteChange, MotionChange, DeltaChange

PropertyChange = PositionChange  # Phase 2: union of all change types

class ObjectTransform(Transform):
    object_uuid: ObjectId
    num_discrete_changes: int
    num_continuous_changes: int

    @staticmethod
    def from_changes(object_uuid, changes) -> ObjectTransform: ...

class PropertyTransformNode(Node):
    property_name: str
    change_type: str  # "continuous" or "discrete"
    old_value: Any
    new_value: Any
    delta: float | None

class TransformDetail(Edge):
    allowed_connections = [("ObjectTransform", "PropertyTransformNode")]

def _compute_property_changes(current, previous) -> list[PropertyChange]:
    """Phase 1: position only. Phase 2: all properties."""
    changes = []
    dx = int(current.x) - int(previous.x)
    dy = int(current.y) - int(previous.y)
    if dx != 0 or dy != 0:
        changes.append(PositionChange(dx=dx, dy=dy))
    return changes
```

**`roc/object.py`** (MODIFY):
- `Features.allowed_connections`: add `("ObjectInstance", "FeatureGroup")`
- `do_object_resolution`: add `Features.connect(o, fg)` after matching existing object

**`roc/sequencer.py`** (MODIFY):
- `handle_object_resolution_event`: create ObjectInstance from resolved object + feature group, connect ObservedAs to Object, attach to frame via FrameFeatures + SituatedObjectInstance
- `Frame.transformable` property: also traverse SituatedObjectInstance edges

**`roc/reporting/state.py`** (MODIFY):
- In transform summary emission: include `object_transforms` list with uuid + position changes

**`dashboard-ui/src/components/panels/TransitionPanel.tsx`** (MODIFY):
- Below existing intrinsic transform rows, render object transform rows showing glyph + dx/dy

**`dashboard-ui/src/components/panels/SequencePanel.tsx`** (MODIFY):
- Show the resolved object's glyph and position alongside existing intrinsic display

### Dependencies

- External: None
- Internal: Existing Transformable ABC, Node/Edge system, FeatureGroup, perception FeatureNodes

### Verification

1. Run: `make test` -- all new and existing tests pass
2. Run: `make lint` -- passes
3. **Start a game and check the dashboard**:
   ```bash
   make run
   # Get dashboard URL:
   npx servherd info roc-ui
   ```
   Start a game via the dashboard. Navigate between steps. In the **Transition panel**, you should see object position deltas like `@ x: +1, y: 0` when the player character moves. In the **Sequence panel**, you should see the resolved object's glyph and position.

4. Verify via API:
   ```bash
   # Fetch a step where the player moved:
   curl -s http://localhost:PORT/api/live/step/5 | python -m json.tool | grep object_transforms
   ```
   Expected: `object_transforms` array with position changes.

---

## Phase 2: Full Property Tracking + Object History

**What this phase accomplishes**: Extends Phase 1 from position-only to all properties (glyph, color, shape, size, distance, motion, delta). Adds RelationshipGroup for relational features with feature splitting in the Sequencer. Adds ObjectHistory edges connecting transforms to Objects, plus an API endpoint for querying object history. You can see all property changes in the Transition panel and query any object's full history.

**Duration**: 2-3 days

### New Files

| File | Contents |
|------|----------|
| `tests/unit/test_object_transform.py` (extend) | All property change types, full delta computation |
| `tests/unit/test_api_object_history.py` | Object history endpoint |

### Modified Files

| File | Change |
|------|--------|
| `roc/object_instance.py` | Add remaining fields (flood_size, line_size, delta_old, delta_new, motion_direction, distance); add RelationshipGroup node + Relationships edge; update from_resolution to accept RelationshipGroup and extract relational features |
| `roc/object_transform.py` | Add all PropertyChange types (SizeChange, DistanceChange, DiscreteChange, MotionChange, DeltaChange); extend _compute_property_changes and _change_to_nodes for full property coverage; add ObjectHistory edge |
| `roc/sequencer.py` | Split features by FeatureKind into FeatureGroup (physical) + RelationshipGroup (relational) before creating ObjectInstance |
| `roc/transformer.py` | After creating ObjectTransform, connect Object -> ObjectTransform via ObjectHistory edge |
| `roc/object.py` | Update Detail.allowed_connections for (RelationshipGroup, FeatureNode) |
| `roc/reporting/state.py` | Emit full property changes in transform_summary (not just position) |
| `roc/reporting/api_server.py` | New endpoint: GET /api/runs/{run}/object/{id}/history |
| `dashboard-ui/src/components/panels/TransitionPanel.tsx` | Show all property changes (discrete: old->new, continuous: delta) |
| `dashboard-ui/src/components/panels/SequencePanel.tsx` | Show full object details (glyph, color, shape, position) |
| `dashboard-ui/src/api/queries.ts` | Add useObjectHistory hook |

### Tests to Write First

- `tests/unit/test_object_transform.py` (extend Phase 1 tests):

```python
class TestAllPropertyChanges:
    def test_discrete_change_glyph(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              glyph_type=111, color_type=7)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              glyph_type=111, color_type=3)
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, DiscreteChange) and c.property_name == "color_type"
                   and c.old_value == 7 and c.new_value == 3 for c in changes)

    def test_size_change_flood(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              flood_size=5)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              flood_size=8)
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, SizeChange) and c.delta == 3 for c in changes)

    def test_distance_change(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              distance=10)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              distance=8)
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, DistanceChange) and c.delta == -2 for c in changes)

    def test_motion_change(self):
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              motion_direction="LEFT")
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              motion_direction="RIGHT")
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, MotionChange) for c in changes)

    def test_none_values_produce_no_change_for_discrete(self):
        """Discrete properties (glyph, color, shape): if EITHER side is None, no change."""
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              glyph_type=111, color_type=None)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              glyph_type=111, color_type=7)
        changes = _compute_property_changes(curr, prev)
        assert not any(isinstance(c, DiscreteChange) and c.property_name == "color_type"
                       for c in changes)

    def test_motion_direction_none_to_value_is_a_change(self):
        """Motion direction: None->value IS a change (unlike discrete properties).

        Design Section 9: motion_direction comparison does NOT guard with None checks.
        `None != "LEFT"` is True, so this produces a MotionChange. This differs from
        discrete properties (glyph, color, shape) which require both sides non-None.
        """
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              motion_direction=None)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              motion_direction="LEFT")
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, MotionChange) and c.old_direction is None
                   and c.new_direction == "LEFT" for c in changes)

    def test_delta_none_to_pair_is_a_change(self):
        """Delta feature: None->pair IS a change (same semantics as motion_direction)."""
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              delta_old=None, delta_new=None)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              delta_old=100, delta_new=111)
        changes = _compute_property_changes(curr, prev)
        assert any(isinstance(c, DeltaChange) for c in changes)

    def test_motion_direction_both_none_is_not_a_change(self):
        """None == None, so no MotionChange."""
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=1,
                              motion_direction=None)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(0), y=YLoc(0), tick=2,
                              motion_direction=None)
        changes = _compute_property_changes(curr, prev)
        assert not any(isinstance(c, MotionChange) for c in changes)

    def test_multiple_simultaneous_changes(self):
        """Position + discrete + continuous changes all detected."""
        prev = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(5), y=YLoc(3), tick=1,
                              color_type=7, distance=10)
        curr = ObjectInstance(object_uuid=ObjectId(1), x=XLoc(7), y=YLoc(4), tick=2,
                              color_type=3, distance=8)
        changes = _compute_property_changes(curr, prev)
        types = {type(c) for c in changes}
        assert PositionChange in types
        assert DiscreteChange in types
        assert DistanceChange in types


class TestObjectHistory:
    def test_object_history_edge(self):
        obj = Object(uuid=ObjectId(42))
        ot = ObjectTransform(object_uuid=ObjectId(42),
                             num_discrete_changes=1, num_continuous_changes=1)
        edge = ObjectHistory.connect(obj, ot)
        assert edge.src_id == obj.id
        assert edge.dst_id == ot.id

    def test_transformer_connects_object_history(self):
        """After computing ObjectTransform, Transformer links it to Object."""
        # Set up two frames with matching ObjectInstances
        # Run _compute_transforms
        # Verify: Object has ObjectHistory edge to the created ObjectTransform


class TestRelationshipGroup:
    def test_from_nodes_connects_detail_edges(self):
        dn = DistanceNode(size=5)
        mn = MotionNode(type=1, direction=Direction.LEFT)
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


class TestSequencerFeatureSplitting:
    def test_splits_physical_and_relational_features(self):
        """Sequencer creates FeatureGroup with PHYSICAL, RelationshipGroup with RELATIONAL."""
        # Send ResolvedObject with mixed features (SingleNode + DistanceNode)
        # Verify: ObjectInstance -> Features -> FeatureGroup (physical only)
        # Verify: ObjectInstance -> Relationships -> RelationshipGroup (relational only)

    def test_object_instance_extracts_relational_features(self):
        """ObjectInstance.distance, motion_direction populated from RelationshipGroup."""
        # Trigger sequencer with features including DistanceNode(size=5)
        # Verify: ObjectInstance.distance == 5
```

- API tests:

```python
class TestObjectHistoryEndpoint:
    def test_returns_states_and_transforms(self, client):
        """GET /api/runs/{run}/object/{id}/history returns states + transforms."""
        # Pre-populate run data with an object that has 3 observations
        resp = client.get(f"/api/runs/{run}/object/{obj_id}/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "states" in data and len(data["states"]) == 3
        assert "transforms" in data
        assert "info" in data

    def test_unknown_object_returns_404(self, client):
        resp = client.get(f"/api/runs/{run}/object/99999/history")
        assert resp.status_code == 404
```

- Dashboard tests:

```typescript
describe("TransitionPanel full property changes", () => {
  it("shows discrete changes as old->new", () => {
    const data = {
      transform_summary: {
        object_transforms: [{
          uuid: 42,
          changes: [
            { property: "color_type", type: "discrete", old_value: 7, new_value: 3 },
            { property: "x", type: "continuous", delta: 2 },
          ]
        }]
      }
    };
    render(<TransitionPanel data={data} />);
    expect(screen.getByText(/color_type: 7 -> 3/)).toBeInTheDocument();
    expect(screen.getByText(/x: \+2/)).toBeInTheDocument();
  });
});

describe("useObjectHistory", () => {
  it("fetches object history on demand", async () => {
    // Mock API, verify hook returns states and transforms
  });
});
```

### Implementation Details

**`roc/object_instance.py`** (EXTEND):
- Add fields: `flood_size`, `line_size`, `delta_old`, `delta_new`, `motion_direction`, `distance`
- Add `RelationshipGroup` node with `from_nodes()` and `feature_nodes` property
- Add `Relationships` edge
- Update `from_resolution()` to accept optional `rg: RelationshipGroup` parameter and extract relational features

**`roc/object_transform.py`** (EXTEND):
- Add dataclasses: `SizeChange`, `DistanceChange`, `DiscreteChange`, `MotionChange`, `DeltaChange`
- Update `PropertyChange` union type to: `PositionChange | SizeChange | DistanceChange | DiscreteChange | MotionChange | DeltaChange`
- Extend `_compute_property_changes` with discrete, size, distance, motion, delta comparisons
- Extend `_change_to_nodes` to handle all change types. Specifically:
  - `MotionChange` -> one PropertyTransformNode with `property_name="motion_direction"`, `change_type="discrete"`, `old_value=old_direction`, `new_value=new_direction`, `delta=None`
  - `DeltaChange` -> one PropertyTransformNode with `property_name="delta"`, `change_type="discrete"`, `old_value=old_pair` (tuple or None), `new_value=new_pair` (tuple or None), `delta=None`
  - `DiscreteChange` -> one PropertyTransformNode with `property_name=change.property_name`, `change_type="discrete"`, `old_value`, `new_value`, `delta=None`
  - `SizeChange` -> one PropertyTransformNode with `property_name=change.property_name`, `change_type="continuous"`, `old_value`, `new_value`, `delta=float(change.delta)`
  - `DistanceChange` -> one PropertyTransformNode with `property_name="distance"`, `change_type="continuous"`, `old_value`, `new_value`, `delta=float(change.delta)`
  - `PositionChange` -> two PropertyTransformNodes: `("x", "continuous", None, None, float(dx))` and `("y", "continuous", None, None, float(dy))`
- Add `ObjectHistory` edge

**`roc/object.py`** (MODIFY):
- Update `Detail.allowed_connections` to add `("RelationshipGroup", "FeatureNode")` to the existing list. This enables `RelationshipGroup.from_nodes()` to connect relational FeatureNodes via Detail edges, mirroring how FeatureGroup uses Detail.

**`roc/sequencer.py`** (MODIFY):
- In `handle_object_resolution_event`: split `fg.feature_nodes` by `FeatureKind`, create separate `FeatureGroup.from_nodes(physical)` and `RelationshipGroup.from_nodes(relational)`, pass both to `ObjectInstance.from_resolution`
- Connect: `Features(oi, fg)`, `Relationships(oi, rg)`

**`roc/transformer.py`** (MODIFY):
- After `Change.connect(ret, t)` for ObjectTransform, add:
  ```python
  if isinstance(t, ObjectTransform):
      obj = Object.find_one(uuid=t.object_uuid)
      if obj is not None:
          ObjectHistory.connect(obj, t)
  ```

**`roc/reporting/api_server.py`** (MODIFY):
- Add `GET /api/runs/{run}/object/{object_id}/history` endpoint

### Dependencies

- External: None
- Internal: Phase 1 (ObjectInstance, ObjectTransform, pipeline integration)

### Verification

1. Run: `make test` -- all tests pass
2. Run: `make lint` -- passes
3. **Dashboard**: Start a game. In the **Transition panel**, you should now see all property changes -- discrete changes shown as `color_type: 7 -> 3`, continuous changes as `distance: -2`. The **Sequence panel** shows full object details (glyph, color, shape, position).
4. **Object history API**:
   ```bash
   # Find an object that has been seen multiple times:
   curl -s http://localhost:PORT/api/live/all-objects | python -m json.tool | head -20
   # Query its history:
   curl -s http://localhost:PORT/api/live/object/NODE_ID/history | python -m json.tool
   ```
   Expected: `states` array with multiple observations (tick, x, y, properties), `transforms` array with property changes between consecutive observations.

---

## Phase 3: Multi-Cycle Attention + Multi-Object Dashboard

**What this phase accomplishes**: VisionAttention runs N cycles (default 4) with IOR between cycles, producing multiple objects per frame. The dashboard shows all N objects in the Sequence panel, a cycle stepper in the Attention panel, and per-cycle resolution details. The Transformer adds ambiguity detection (skips transforms for Objects with multiple instances per frame). Old historical runs display correctly via backward compatibility wrapping.

**Duration**: 2-3 days

### New Files

| File | Contents |
|------|----------|
| `tests/unit/test_multi_cycle_attention.py` | Attention loop, IOR, AttentionSettled (note: design Section 16 lists a separate `test_attention_settled.py` but we consolidate AttentionSettled tests here since they are small and tightly coupled to the multi-cycle loop) |
| `tests/integration/test_object_transform_pipeline.py` | End-to-end pipeline: multi-cycle attention -> resolution -> sequencer -> transformer -> transforms (50+ steps) |
| `dashboard-ui/src/components/panels/AttentionCycleSummary.tsx` | Cycle summary table + stepper |
| `dashboard-ui/src/components/panels/AttentionCycleSummary.test.tsx` | Tests |

### Modified Files

| File | Change |
|------|--------|
| `roc/attention.py` | Add AttentionSettled class; rewrite do_attention with N-cycle loop |
| `roc/config.py` | Add `attention_cycles: int = 4` |
| `roc/object.py` | ObjectResolver event filter to skip AttentionSettled |
| `roc/transformer.py` | Ambiguity detection: skip transforms when same uuid has multiple instances per frame |
| `roc/reporting/state.py` | Accumulate per-cycle saliency/resolution data; emit as lists |
| `roc/reporting/data_store.py` | Backward compatibility: wrap old single-field data into list format |
| `dashboard-ui/src/components/panels/SaliencyMap.tsx` | Accept cycle index, render selected cycle |
| `dashboard-ui/src/components/panels/AttenuationPanel.tsx` | Accept per-cycle data |
| `dashboard-ui/src/components/panels/ResolutionInspector.tsx` | Add summary table + cycle stepper |
| `dashboard-ui/src/components/panels/SequencePanel.tsx` | Show all N objects in table |
| `dashboard-ui/src/types.ts` | Update StepData: saliency_cycles[], resolution_cycles[] |

### Tests to Write First

- `tests/unit/test_multi_cycle_attention.py`:

```python
class TestAttentionSettled:
    def test_is_distinct_from_attention_data(self):
        settled = AttentionSettled()
        assert not isinstance(settled, VisionAttentionData)


class TestMultiCycleAttention:
    def test_emits_n_attention_events(self):
        """4+ saliency peaks + N=4 -> 4 AttentionData + 1 AttentionSettled."""
        # Set up VisionAttention with synthetic saliency map (5 peaks)
        # Config.attention_cycles = 4
        # Trigger attention
        # Verify: 4 AttentionData events + 1 AttentionSettled

    def test_fewer_peaks_than_cycles_breaks_early(self):
        """Only 2 peaks + N=4 -> 2 AttentionData + 1 AttentionSettled."""

    def test_zero_peaks_emits_only_settled(self):
        """No peaks -> 0 AttentionData + 1 AttentionSettled."""

    def test_ior_shifts_focus_between_cycles(self):
        """Each cycle attends a different location."""
        # Verify: focus points in cycle 1,2,3,4 are all different

    def test_default_cycles_is_four(self):
        assert Config.get().attention_cycles == 4


class TestObjectResolverMultiCycle:
    def test_resolves_each_cycle_independently(self):
        """3 AttentionData events -> 3 ResolvedObject events."""

    def test_ignores_attention_settled(self):
        """AttentionSettled does not trigger resolution."""


class TestTransformerAmbiguity:
    def test_skips_multi_instance_same_uuid(self):
        """2 ObjectInstance(uuid=X) in frame1 -> no transform for X."""
        # Frame1: 2 instances of uuid X (two orcs)
        # Frame2: 1 instance of uuid X
        # Verify: no ObjectTransform for X

    def test_unique_instance_still_gets_transform(self):
        """uuid=X unique in both frames -> ObjectTransform computed."""
        # Frame1: 1 instance X, 2 instances Y
        # Frame2: 1 instance X, 1 instance Y
        # Verify: ObjectTransform for X, NOT for Y
```

- Dashboard tests:

```typescript
describe("AttentionCycleSummary", () => {
  it("renders summary row per cycle with three columns", () => {
    // Design Section 11.2: three columns per cycle:
    //   Pre-IOR Peak (what would have been selected without attenuation)
    //   Post-IOR Peak (what the attenuated map selected)
    //   Focused Point (the actual location attended)
    const cycles = [
      {
        preIorPeak: { x: 10, y: 5, strength: 0.95 },
        postIorPeak: { x: 10, y: 5, strength: 0.95 },
        focusedPoint: { x: 10, y: 5, strength: 0.95 },
      },
      {
        preIorPeak: { x: 10, y: 5, strength: 0.95 },
        postIorPeak: { x: 15, y: 8, strength: 0.82 },
        focusedPoint: { x: 15, y: 8, strength: 0.82 },
      },
      {
        preIorPeak: { x: 10, y: 5, strength: 0.95 },
        postIorPeak: { x: 22, y: 1, strength: 0.71 },
        focusedPoint: { x: 22, y: 1, strength: 0.71 },
      },
    ];
    render(<AttentionCycleSummary cycles={cycles} />);
    expect(screen.getAllByRole("row")).toHaveLength(4); // header + 3
    // Verify all three columns present in header
    expect(screen.getByText("Pre-IOR Peak")).toBeInTheDocument();
    expect(screen.getByText("Post-IOR Peak")).toBeInTheDocument();
    expect(screen.getByText("Focused Point")).toBeInTheDocument();
  });

  it("stepper switches saliency map display", async () => {
    // Click cycle 2 button, verify detail shows cycle 2 data
  });

  it("single cycle shows no stepper (backward compat)", () => {
    render(<AttentionCycleSummary cycles={[singleCycle]} />);
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });
});

describe("ResolutionInspector multi-cycle", () => {
  it("renders summary table with correct columns", () => {
    // Design Section 11.3 wireframe: # | Outcome | Object | Location | Candidates
    // resolution_cycles array with 4 entries
    // Verify: 4 summary rows + header
    expect(screen.getByText("Outcome")).toBeInTheDocument();
    expect(screen.getByText("Object")).toBeInTheDocument();
    expect(screen.getByText("Location")).toBeInTheDocument();
    expect(screen.getByText("Candidates")).toBeInTheDocument();
    // Object column shows GlyphBadge with color
  });

  it("stepper switches resolution detail", async () => {
    // Click cycle 2, verify detail shows cycle 2 candidates
  });
});

describe("SequencePanel multi-object", () => {
  it("displays all objects in frame as table rows", () => {
    const data = {
      sequence_summary: {
        tick: 7,
        objects: [
          { glyph: "@", color: "WHITE", x: 10, y: 5, matched_previous: true, cycle_number: 1, resolve_count: 42 },
          { glyph: "d", color: "RED", x: 15, y: 8, matched_previous: true, cycle_number: 2, resolve_count: 7 },
          { glyph: ".", color: "GREY", x: 22, y: 1, matched_previous: false, cycle_number: 3, resolve_count: 1 },
        ],
        // Intrinsics are textual rows (no bar graphs -- design Section 11.4)
        intrinsics: { hp: { raw: 15, normalized: 0.8, matched: true } }
      }
    };
    render(<SequencePanel data={data} />);
    expect(screen.getAllByRole("row")).toHaveLength(4); // header + 3 objects
  });

  it("shows resolve count column for objects", () => {
    // Design Section 11.4: "Resolves" column shows resolve_count
    // Verify: "42" appears in the resolves column for "@"
  });

  it("renders position as combined (x,y) format", () => {
    // Design Section 11.4 wireframe shows Pos column as "(10,5)"
    // Verify: combined format rendered, not separate x and y columns
    expect(screen.getByText("(10,5)")).toBeInTheDocument();
  });

  it("does not display shape as a separate column", () => {
    // Design wireframe (Section 11.4) shows columns: #, Glyph, Pos, Color, Matched, Resolves
    // Shape is NOT a visible column (available in data for ObjectModal, not in table).
    // Note: shape is included in sequence_summary.objects data but only displayed in ObjectModal.
  });

  it("shows intrinsics as textual rows not bar graphs", () => {
    // Design Section 11.4: "No bar graphs; intrinsics are textual rows"
    // Verify: intrinsic rows show Name, Raw, Normalized, Matched columns
    // Verify: no progress bar elements rendered
  });

  it("shows frame counts in header", () => {
    // "Frame: tick=7 | 3 objects | 4 intrinsics"
  });
});

describe("backward compatibility", () => {
  it("wraps single saliency as saliency_cycles[0]", () => {
    // Old-format StepData with saliency (not saliency_cycles)
    // Verify: renders correctly with 1 cycle, no stepper
  });

  it("wraps single resolution_metrics as resolution_cycles[0]", () => {
    // Old-format StepData with resolution_metrics (not resolution_cycles)
  });

  it("wraps single-object sequence_summary with defaults", () => {
    // Old-format single-object sequence_summary gets wrapped as a 1-element list
    // with default fields: matched_previous: false, cycle_number: 0
  });
});
```

### Implementation Details

**`roc/attention.py`** (MODIFY):
```python
class AttentionSettled:
    """Sentinel: all attention cycles for this frame are complete."""

# In VisionAttention.do_attention(), replace single get_focus with loop:
attenuation = SaliencyAttenuationExpMod.get(default="none")
cycles = Config.get().attention_cycles
cycle_metadata = []  # Per-cycle data for dashboard (design Section 11.2)

for cycle in range(cycles):
    # Capture pre-IOR peak: find the peak on the saliency map BEFORE attenuation
    # is applied for this cycle. This shows what would have been selected without IOR.
    pre_ior_peak = self.saliency_map.get_raw_peak()  # or equivalent unattenuated peak

    focus_points = self.saliency_map.get_focus()  # applies attenuation, finds peaks
    if len(focus_points) == 0:
        break

    # post_ior_peak = focus_points top peak (the attenuated result)
    # focused_point = final selected point (same as post_ior_peak for now)
    cycle_metadata.append({
        "pre_ior_peak": pre_ior_peak,
        "post_ior_peak": focus_points.iloc[0],
        "focused_point": focus_points.iloc[0],
    })

    self.att_conn.send(VisionAttentionData(focus_points=..., saliency_map=...))
    attenuation.notify_focus(focus_points)

self.att_conn.send(AttentionSettled(cycle_metadata=cycle_metadata))
# AttentionSettled carries the accumulated cycle_metadata for the whole frame.
# This is the cleanest transport mechanism: it fires once per frame when all metadata
# is complete, and state.py already needs to listen for it. Storing metadata on the
# component instance would require direct attribute access (violating invariant #1).
```

**Attention bus type update**: The attention bus is currently typed as `EventBus[AttentionData]`. Since `AttentionSettled` is also emitted on this bus, the type must be updated to `EventBus[AttentionData | AttentionSettled]` in `roc/attention.py` where the bus is declared on the `Attention` class. This is a one-line change to the class-level bus attribute.

**Note on pre-IOR peak capture**: The exact mechanism depends on how `SaliencyMap.get_focus()` applies attenuation internally. If attenuation is applied inside `get_focus()`, a separate `get_raw_peak()` method (or calling `get_focus()` with attenuation disabled) is needed to capture the pre-IOR peak. If attenuation is applied externally before `get_focus()`, the pre-IOR peak can be read from the unattenuated map. Determine the right approach during implementation.

**`roc/object.py`** (MODIFY): ObjectResolver skips AttentionSettled events (filter or isinstance check).

**`roc/transformer.py`** (MODIFY): Add Counter-based ambiguity detection before computing ObjectInstance transforms.

**`roc/reporting/state.py`** (MODIFY): Accumulate `saliency_cycles` and `resolution_cycles` lists. Reset per step.

**`roc/reporting/data_store.py`** (MODIFY): When loading old data, wrap `saliency -> saliency_cycles: [saliency]`, `resolution_metrics -> resolution_cycles: [resolution_metrics]`.

**`roc/action.py`** (NO CHANGES): Action synchronizes via the predict bus, not via AttentionSettled. The design document's Section 10 pipeline flow diagram (line 1171) says "Action gated on ActionRequest AND AttentionSettled" but this contradicts Sections 4.5 and D7 which explicitly say "No AttentionSettled sentinel, no barriers, no custom synchronization between attention and action." The correct behavior per D7: Action waits for ActionRequest + prediction from predict bus. Do NOT add AttentionSettled gating to Action.

**`roc/reporting/state.py`** (MODIFY): Per-cycle data must include pre-IOR peak, post-IOR peak, and focused point for each cycle (not just focused point). This requires VisionAttention to record all three values per cycle and pass them through the state emission pipeline.

**Dashboard changes**: AttentionCycleSummary (new component with 3-column summary: Pre-IOR Peak, Post-IOR Peak, Focused Point), SaliencyMap (accept cycle prop), ResolutionInspector (add summary with GlyphBadge per resolved object + stepper), SequencePanel (table of N objects with resolve_count column, intrinsics as textual rows with no bar graphs), StepData types updated.

**Integration test**: `tests/integration/test_object_transform_pipeline.py` (NEW):

```python
class TestObjectTransformPipeline:
    def test_full_pipeline_produces_object_transforms(self):
        """Multi-cycle attention -> resolution -> sequencer -> transformer -> transforms.

        Runs 50+ steps to verify pipeline stability with multi-cycle attention.
        """
        # Load full pipeline (all_components fixture)
        # Set Config.attention_cycles = 4
        # Run 50 game steps
        # Verify: TransformResult events contain ObjectTransforms for matching objects
        # Verify: No crashes, no deadlocks

    def test_multi_cycle_multiple_objects_per_frame(self):
        """Frames contain multiple ObjectInstances after multi-cycle attention."""
        # Run 10 game steps
        # Verify: frames have N ObjectInstances (where N <= attention_cycles)

    def test_transform_object_history_connected(self):
        """After pipeline, Object nodes have ObjectHistory edges."""
        # Run 10 steps
        # Find an Object present in consecutive frames
        # Verify: Object --ObjectHistory--> ObjectTransform exists

    def test_ambiguity_skips_multi_instance_transforms(self):
        """Objects with multiple instances per frame get no transforms."""
        # Run until a frame has 2 instances of same Object uuid
        # Verify: no ObjectTransform for that uuid
```

### Dependencies

- External: None
- Internal: Phase 2 (full ObjectInstance with all properties, feature splitting, ObjectHistory)

### Verification

1. Run: `make test` -- all tests pass
2. Run: `make lint` -- passes
3. **Dashboard -- multi-cycle attention**:
   - Start a game. Navigate to any step.
   - **Sequence panel**: Should now show 4 (or fewer) objects per frame in a table with glyph, position, color, matched status.
   - **Attention panel**: Cycle summary table at top shows all 4 cycles with focus points. Click cycle buttons to switch the saliency map and attenuation details below.
   - **Resolution panel**: Summary table shows 4 resolution outcomes. Click stepper to see each cycle's candidates.
   - **Transition panel**: Should show transforms for objects that matched across frames (likely just "@" and a few others). Objects with multiple instances (e.g., two 'd' monsters) should have no transforms (ambiguity detection).

4. **Backward compatibility**: Load an old historical run. Verify single-cycle data renders correctly -- no stepper, just the single cycle's data. No errors.

5. Verify cycle count in logs:
   ```bash
   roc_log_modules=attention roc_log_level=debug uv run play 2>&1 | head -100
   ```
   Expected: 4 attention cycle log entries per game step.

---

## Phase 4: Object Modal + Transition Redesign + Highlights

**What this phase accomplishes**: Click any object anywhere in the dashboard to open an Object Modal showing its full history (observations + transforms). The Transition panel is redesigned with three-column prev/current/delta alignment. The highlight system supports multi-color highlights across all panels. A shared ObjectLink component provides consistent object references everywhere.

**Duration**: 2-3 days

### New Files

| File | Contents |
|------|----------|
| `dashboard-ui/src/components/common/ObjectLink.tsx` | Shared clickable object reference |
| `dashboard-ui/src/components/common/ObjectLink.test.tsx` | Tests |
| `dashboard-ui/src/components/panels/ObjectModal.tsx` | Object detail modal |
| `dashboard-ui/src/components/panels/ObjectModal.test.tsx` | Tests |

### Modified Files

| File | Change |
|------|--------|
| `dashboard-ui/src/components/panels/TransitionPanel.tsx` | Three-column redesign: prev, current, delta with row alignment |
| `dashboard-ui/src/components/panels/TransitionPanel.test.tsx` | Tests for new layout |
| `dashboard-ui/src/state/highlight.tsx` | Extend from HighlightPoint[] to HighlightEntry[] with colors |
| `dashboard-ui/src/state/highlight.test.tsx` | Tests |
| `dashboard-ui/src/components/common/GameScreen.tsx` | Render colored borders from HighlightEntry |
| `dashboard-ui/src/components/panels/SaliencyMap.tsx` | Also render colored cell borders from HighlightEntry (design Section 11.7: "same color appears on the game screen cell border, the saliency map cell border") |
| `dashboard-ui/src/components/panels/SequencePanel.tsx` | Object rows use ObjectLink |
| `dashboard-ui/src/components/panels/AllObjects.tsx` | Object rows use ObjectLink |
| `dashboard-ui/src/components/panels/ResolutionInspector.tsx` | Matched object uses ObjectLink |

### Tests to Write First

```typescript
// ObjectLink.test.tsx
describe("ObjectLink", () => {
  it("renders glyph badge with color", () => {
    render(<ObjectLink objectId={42} glyph="@" color="WHITE" />);
    expect(screen.getByText("@")).toBeInTheDocument();
  });

  it("opens Object Modal on click", async () => {
    render(<ObjectLink objectId={42} glyph="@" color="WHITE" />);
    await userEvent.click(screen.getByText("@"));
    expect(screen.getByText(/Object:/)).toBeInTheDocument();
  });
});

// ObjectModal.test.tsx
describe("ObjectModal", () => {
  it("renders object info header", () => {
    // Mock useObjectHistory
    // Verify: uuid, glyph, color, match count

  it("renders state history table with diff highlighting", () => {
    // 3 observations, color changed in #2
    // Verify: 3 rows, color cell in row 2 highlighted

  it("renders transform history table", () => {
    // 2 transforms between 3 observations
    // Verify: rows show from_tick -> to_tick + changes

  it("tick references are clickable for navigation", () => {
    // Click tick 5 -> step navigation triggered
  });
});

// TransitionPanel.test.tsx (rewrite)
describe("TransitionPanel three-column", () => {
  it("aligns same object across prev/current columns", () => {
    // Object X in both frames at different positions
    // Verify: same row with prev, current, delta columns

  it("shows new object with -- in previous column", () => {
    // Object Z only in current
    // Verify: prev shows "--", delta shows "(new)"

  it("shows gone object with -- in current column", () => {
    // Object W only in previous
    // Verify: current shows "--", delta shows "(gone)"

  it("shows intrinsic deltas in same three-column layout", () => {
    // hp changed, energy unchanged
    // Verify: hp row has delta, energy row shows "(no change)"

  it("object cells use ObjectLink", () => {
    // Click object cell -> modal opens
  });

  it("step references in header are clickable", () => {
    // Design Section 11.5: tick in header is clickable for step navigation
    // Click "tick=6" in Previous Frame header -> navigates to step 6
  });
});

// highlight.test.tsx
describe("HighlightContext multi-color", () => {
  it("assigns distinct colors from palette", () => {
    // Toggle 3 points
    // Verify: each gets different color from palette

  it("toggles off on second click", () => {
    // Toggle, then toggle same point
    // Verify: removed

  it("clears on step change", () => {
    // Add highlights, trigger step change
    // Verify: empty
  });
});
```

### Implementation Details

**`ObjectLink.tsx`** (NEW):
```typescript
// Renders GlyphBadge with click handler
// On click: sets selected objectId in context, opens ObjectModal
// Props: objectId, glyph, color, label?
// Used across: SequencePanel, TransitionPanel, AllObjects, ResolutionInspector
```

**`ObjectModal.tsx`** (NEW):
```typescript
// Mantine Modal, fetches useObjectHistory(run, objectId)
// Sections:
//   1. Header: glyph badge + uuid + debug info (node_id, resolve_count, first_seen)
//      Note: use `resolve_count` consistently (matches Object model field name).
//      The design's Object Modal wireframe labels this "Matches: 42" in the UI,
//      and the API response spec uses "match_count" -- unify on `resolve_count`
//      as the data field name, display as "Matches" in the UI label.
//   2. Observation stepper: [1] [2] [3] ... pages through observations when many
//      (design Section 11.6: "The history section uses a stepper to page through
//       observations when there are many")
//   3. State history table: tick, x, y, glyph, color, shape, distance
//      - Changed cells get diff highlighting between consecutive rows
//   4. Transform history table: from_tick -> to_tick, property changes
//   5. Clickable ticks for step navigation
```

**`TransitionPanel.tsx`** (REWRITE):
```typescript
// Three-column layout: Previous | Current | Delta
// Build row data:
//   1. Match objects by uuid across prev/current frames
//   2. Matched: show both + delta
//   3. New (current only): "--" in prev, "(new)" in delta
//   4. Gone (previous only): "--" in current, "(gone)" in delta
// Same layout for intrinsics (matched by name)
// Diff highlighting on changed cells
// All object cells use ObjectLink
```

**`highlight.tsx`** (MODIFY):
```typescript
type HighlightEntry = {
  x: number; y: number;
  color: string;       // from rotating palette
  label?: string;
  source?: string;     // originating panel
};
// Palette: ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
// togglePoint: assigns next color, or removes if already highlighted
// Clear on step change
```

**`GameScreen.tsx`** (MODIFY): Render colored cell borders from HighlightEntry (instead of always yellow).

**Integrate ObjectLink**: Replace direct object rendering in SequencePanel, AllObjects, ResolutionInspector with ObjectLink component.

### Dependencies

- External: None (Mantine Modal already in project)
- Internal: Phase 3 (multi-object data in all panels), Phase 2 (useObjectHistory hook + API endpoint)

### Verification

1. Run: `cd dashboard-ui && npx vitest run` -- all tests pass
2. Run: `make lint` -- passes
3. **Dashboard -- Object Modal**:
   - Start a game, navigate to a step with resolved objects.
   - Click any object glyph in the **Sequence panel** -> Object Modal opens.
   - Modal shows: object info header, state history table with diff highlighting, transform history.
   - Click a tick number in the modal -> dashboard navigates to that step.
   - Close modal. Click an object in **All Objects** table -> same modal opens.
   - Click matched object in **Resolution Inspector** -> same modal opens.

4. **Dashboard -- Transition panel**:
   - Three columns visible: Previous, Current, Delta.
   - "@" character appears on same row in both columns with position delta.
   - New objects show "--" in Previous column.
   - Objects that disappeared show "--" in Current column.
   - Changed cells have colored backgrounds.

5. **Dashboard -- Highlights**:
   - Click a focus point in **Attention panel** -> colored highlight on game screen.
   - Click an object in **Sequence panel** -> different color highlight on its position.
   - Multiple highlights visible simultaneously with distinct colors.
   - Navigate to next step -> highlights clear.

6. **Cross-panel consistency**: Verify ObjectLink works identically from every panel (Sequence, Transition, AllObjects, Resolution).

---

## Common Utilities

- **`RelationshipGroup.from_nodes()`**: Mirrors `FeatureGroup.from_nodes()`. Shares the `Detail` edge for connecting child FeatureNodes. Introduced in Phase 2.
- **`split_features_by_kind(nodes)`**: Utility `(physical, relational) = split_features_by_kind(fg.feature_nodes)`. Used by Sequencer (Phase 2). Lives in `object_instance.py` or `perception.py`.
- **`_compute_property_changes()`**: Delta computation between two ObjectInstances. Phase 1 = position-only, Phase 2 = all properties. Lives in `object_transform.py`.
- **`ObjectLink` component**: Shared clickable object reference (Phase 4). Renders glyph badge, opens ObjectModal on click. Used across all panels.
- **Cycle stepper pattern**: Both Attention and Resolution panels use "summary table + SegmentedControl + detail view" (Phase 3). Consider a shared `useCycleSelection(count)` hook or `CycleStepper` wrapper.

## External Libraries Assessment

- **No new Python libraries needed.** All functionality uses existing stdlib and project infrastructure.
- **No new JavaScript libraries needed.** Mantine (Modal, SegmentedControl, Table), TanStack Query, and Recharts are already in the project.

## Risk Mitigation

- **Risk: FrameAttribute edge migration breaks existing code.**
  Mitigation: Phase 1 adds new `FrameFeatures` and `SituatedObjectInstance` edges alongside existing `FrameAttribute`. Keep `FrameAttribute` for TakeAction and IntrinsicNode. Update `Frame.transformable` and `Frame.objects` to traverse both old and new edge types. Existing code paths continue to work.

- **Risk: Multi-cycle attention changes event timing, causing race conditions.**
  Mitigation: Phase 3 tests verify events are emitted sequentially (not concurrently). The RxPY `observe_on(ThreadPoolScheduler)` serializes events within a single observer. The predict bus synchronization point is unchanged. Add integration test running 50+ steps.

- **Risk: ObjectResolver does not properly skip AttentionSettled.**
  Mitigation: Phase 3 includes explicit test. Use `isinstance(e.data, AttentionSettled)` check. If bus typing requires it, update the bus type parameter to `AttentionData | AttentionSettled`.

- **Risk: Backward compatibility wrapping misses edge cases in old StepData.**
  Mitigation: Phase 3 tests explicitly load old-format data and verify wrapping. Manual verification by loading an old historical run.

- **Risk: PropertyTransformNode graph overhead (many small nodes per step).**
  Mitigation: With 1 object per frame (Phases 1-2) this is ~2-5 PropertyTransformNodes per step. With 4 objects (Phase 3), ~12-20 per step. At 1 step/s this is well within Memgraph capacity. Monitor during integration testing.

- **Risk: `Frame.objects` property traversal breaks with new edge topology.**
  Mitigation: Phase 1 updates `Frame.objects` to traverse both `FrameAttribute -> FeatureGroup -> Features -> Object` (old path) and `SituatedObjectInstance -> ObjectInstance -> ObservedAs -> Object` (new path). Test both paths.

- **Risk: Phase 1 changes dashboard but only shows 1 object (pre-multi-cycle).**
  Mitigation: This is expected and acceptable -- the "@" character is usually re-resolved every frame, so you see its position deltas. The Transition panel works with 1 object per frame. Phase 3 then expands to N objects, and the dashboard already handles it because Phase 1-2 built the list-based data structures.

- **Risk: `_select_transformable_edges` in transformer.py filters by edge type `"FrameAttribute"`, which will miss ObjectInstances attached via `SituatedObjectInstance`.**
  Mitigation: Phase 1 explicitly updates this function (see Modified Files table for transformer.py). Either add `SituatedObjectInstance` to the edge type filter, or rewrite the filter to check `isinstance(e.dst, Transformable)` instead of filtering by edge type string. The latter is more robust and forward-compatible. This is a critical path -- without this fix, ObjectInstance transforms will never be computed. A Phase 1 unit test (`test_computes_position_transform_for_matching_object`) will catch this immediately.

- **Risk: `Change.connect(transform, objectTransform)` may fail if label matching is exact-only and does not recognize ObjectTransform as a Transform.**
  Mitigation: Phase 1 includes a verification step (see Modified Files table for transformer.py). ObjectTransform extends Transform, so its labels should include "Transform" via the node label hierarchy. If `Change.allowed_connections = [("Transform", "Transform"), ...]` does not match because label matching is exact-only, add `("Transform", "ObjectTransform")` and `("ObjectTransform", "PropertyTransformNode")` entries. Test this early in Phase 1.

- **Risk: Design document Section 10 pipeline flow contradicts Sections 4.5 and D7 about Action gating.**
  Mitigation: Section 10 line 1171 says "Action gated on ActionRequest AND AttentionSettled" but this is incorrect per the design's own Sections 4.5 and D7. The plan follows D7: Action synchronizes via the predict bus only. Do NOT add AttentionSettled gating to Action. Phase 3 explicitly notes this (see `roc/action.py` NO CHANGES note).
