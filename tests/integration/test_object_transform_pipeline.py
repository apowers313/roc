# mypy: disable-error-code="no-untyped-def"

"""Integration tests for the multi-cycle attention -> resolution -> sequencer ->
transformer pipeline (Phase 3).

Requires Memgraph to be running.
"""

from unittest.mock import patch

from helpers.nethack_screens2 import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.config import Config
from roc.object import Object
from roc.object_instance import ObjectInstance, SituatedObjectInstance
from roc.object_transform import ObjectTransform
from roc.perception import VisionData
from roc.sequencer import Frame, NextFrame
from roc.transformer import _compute_transforms, _get_ambiguous_uuids


def _load_perception_and_resolution():
    """Load perception + attention + resolution pipeline, return (delta, resolver)."""
    object_resolver = Component.get("resolver", "object")
    Component.get("vision", "attention")
    delta = Component.get("delta", "perception")
    Component.get("distance", "perception")
    Component.get("flood", "perception")
    Component.get("motion", "perception")
    Component.get("single", "perception")
    Component.get("line", "perception")
    Component.get("color", "perception")
    Component.get("shape", "perception")
    return delta, object_resolver


class TestMultiCycleAttentionPipeline:
    """Full pipeline: multi-cycle attention -> resolution -> sequencer -> transformer."""

    def test_multi_cycle_produces_multiple_resolved_objects(self, empty_components):
        """With attention_cycles=4, each screen produces multiple ResolvedObject events."""
        Config.get().attention_cycles = 4
        delta, object_resolver = _load_perception_and_resolution()

        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=object_resolver.obj_res_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))
        s.input_conn.send(VisionData.from_dict(screens[2]))

        # With 4 cycles * 3 screens, should get more than 3 objects
        assert s.output.call_count > 3, (
            f"Expected more than 3 resolved objects with 4 cycles, got {s.output.call_count}"
        )
        s.shutdown()

    def test_single_cycle_gives_one_per_screen(self, empty_components):
        """With attention_cycles=1, each screen produces exactly 1 ResolvedObject."""
        Config.get().attention_cycles = 1
        delta, object_resolver = _load_perception_and_resolution()

        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=object_resolver.obj_res_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))
        s.input_conn.send(VisionData.from_dict(screens[2]))

        assert s.output.call_count == 3
        s.shutdown()


class TestAmbiguityIntegration:
    """Verify ambiguity detection with real graph nodes."""

    def test_ambiguous_uuids_detected(self):
        """Multiple ObjectInstances with same uuid are detected as ambiguous."""
        frame = Frame()
        o = Object()

        SituatedObjectInstance.connect(frame, ObjectInstance(object_uuid=o.uuid, x=1, y=1, tick=1))
        SituatedObjectInstance.connect(frame, ObjectInstance(object_uuid=o.uuid, x=5, y=5, tick=1))

        assert o.uuid in _get_ambiguous_uuids(frame)

    def test_unique_uuids_not_ambiguous(self):
        """Single ObjectInstance per uuid is not ambiguous."""
        frame = Frame()
        o1, o2 = Object(), Object()

        SituatedObjectInstance.connect(frame, ObjectInstance(object_uuid=o1.uuid, x=1, y=1, tick=1))
        SituatedObjectInstance.connect(frame, ObjectInstance(object_uuid=o2.uuid, x=5, y=5, tick=1))

        assert len(_get_ambiguous_uuids(frame)) == 0

    def test_ambiguous_objects_skipped_in_transforms(self):
        """Objects with multiple instances in a frame get no ObjectTransforms."""
        frame1, frame2 = Frame(), Frame()
        NextFrame.connect(frame1, frame2)
        o = Object()

        SituatedObjectInstance.connect(frame1, ObjectInstance(object_uuid=o.uuid, x=1, y=1, tick=1))
        SituatedObjectInstance.connect(frame1, ObjectInstance(object_uuid=o.uuid, x=5, y=5, tick=1))
        SituatedObjectInstance.connect(frame2, ObjectInstance(object_uuid=o.uuid, x=3, y=3, tick=2))

        with patch("roc.object.Object.find_one", return_value=None):
            transform = _compute_transforms(frame2, frame1)

        ot = [
            e.dst
            for e in transform.src_edges.select(type="Change")
            if isinstance(e.dst, ObjectTransform)
        ]
        assert len(ot) == 0

    def test_unique_object_gets_transform_and_history(self):
        """Non-ambiguous moved object gets ObjectTransform + ObjectHistory."""
        frame1, frame2 = Frame(), Frame()
        NextFrame.connect(frame1, frame2)
        o = Object()

        SituatedObjectInstance.connect(frame1, ObjectInstance(object_uuid=o.uuid, x=1, y=1, tick=1))
        SituatedObjectInstance.connect(frame2, ObjectInstance(object_uuid=o.uuid, x=3, y=3, tick=2))

        with patch("roc.object.Object.find_one", return_value=o):
            transform = _compute_transforms(frame2, frame1)

        ot = [
            e.dst
            for e in transform.src_edges.select(type="Change")
            if isinstance(e.dst, ObjectTransform)
        ]
        assert len(ot) == 1
        assert ot[0].object_uuid == o.uuid
        assert len(o.src_edges.select(type="ObjectHistory")) == 1

    def test_mixed_ambiguous_and_unique(self):
        """Only unique objects get transforms; ambiguous ones are skipped."""
        frame1, frame2 = Frame(), Frame()
        NextFrame.connect(frame1, frame2)
        unique_obj, ambig_obj = Object(), Object()

        SituatedObjectInstance.connect(
            frame1, ObjectInstance(object_uuid=unique_obj.uuid, x=1, y=1, tick=1)
        )
        SituatedObjectInstance.connect(
            frame1, ObjectInstance(object_uuid=ambig_obj.uuid, x=5, y=5, tick=1)
        )
        SituatedObjectInstance.connect(
            frame1, ObjectInstance(object_uuid=ambig_obj.uuid, x=8, y=8, tick=1)
        )
        SituatedObjectInstance.connect(
            frame2, ObjectInstance(object_uuid=unique_obj.uuid, x=3, y=3, tick=2)
        )
        SituatedObjectInstance.connect(
            frame2, ObjectInstance(object_uuid=ambig_obj.uuid, x=6, y=6, tick=2)
        )

        with patch("roc.object.Object.find_one", return_value=unique_obj):
            transform = _compute_transforms(frame2, frame1)

        ot = [
            e.dst
            for e in transform.src_edges.select(type="Change")
            if isinstance(e.dst, ObjectTransform)
        ]
        assert len(ot) == 1
        assert ot[0].object_uuid == unique_obj.uuid
