# mypy: disable-error-code="no-untyped-def"

"""Unit tests for multi-cycle attention, AttentionSettled, ObjectResolver filtering,
and Transformer ambiguity detection (Phase 3)."""

from unittest.mock import MagicMock, patch

import pytest

from roc.pipeline.attention.attention import AttentionSettled, VisionAttentionData
from roc.framework.config import Config
from roc.perception.location import XLoc, YLoc
from roc.pipeline.object.object import ObjectId
from roc.pipeline.object.object_instance import ObjectInstance, SituatedObjectInstance
from roc.pipeline.temporal.transformer import _compute_transforms, _get_ambiguous_uuids


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.db.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestAttentionSettled:
    def test_is_distinct_from_attention_data(self):
        settled: object = AttentionSettled(cycle_metadata=[])
        assert not isinstance(settled, VisionAttentionData)

    def test_carries_cycle_metadata(self):
        meta = [{"focused_point": {"x": 10, "y": 5, "strength": 0.95}}]
        settled = AttentionSettled(cycle_metadata=meta)
        assert settled.cycle_metadata == meta
        assert len(settled.cycle_metadata) == 1

    def test_empty_metadata_for_zero_cycles(self):
        settled = AttentionSettled(cycle_metadata=[])
        assert settled.cycle_metadata == []


class TestAttentionCyclesConfig:
    def test_default_cycles_is_four(self):
        assert Config.get().attention_cycles == 4

    def test_cycles_can_be_overridden(self):
        settings = Config.get()
        settings.attention_cycles = 2
        assert settings.attention_cycles == 2


class TestObjectResolverSkipsSettled:
    def test_event_filter_behavior(self):
        """ObjectResolver filters out AttentionSettled but accepts VisionAttentionData."""
        from roc.pipeline.object.object import ObjectResolver

        resolver = ObjectResolver()
        try:
            # AttentionSettled should be filtered out
            settled = AttentionSettled(cycle_metadata=[])
            settled_event = MagicMock()
            settled_event.data = settled
            settled_event.src_id.name = "vision"
            settled_event.src_id.type = "attention"
            assert resolver.event_filter(settled_event) is False

            # VisionAttentionData from vision/attention should pass
            attn_event = MagicMock()
            attn_event.data = MagicMock(spec=VisionAttentionData)
            attn_event.src_id.name = "vision"
            attn_event.src_id.type = "attention"
            assert resolver.event_filter(attn_event) is True
        finally:
            resolver.shutdown()


class TestTransformerAmbiguity:
    @pytest.fixture(autouse=True)
    def reset_tick(self):
        import roc.pipeline.temporal.sequencer as seq

        original = seq.tick
        seq.tick = 0
        yield
        seq.tick = original

    def test_get_ambiguous_uuids_detects_duplicates(self):
        """Two ObjectInstances with the same uuid in a frame -> ambiguous."""
        from roc.pipeline.temporal.sequencer import Frame

        frame = Frame()
        oi1 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(1), y=YLoc(1), tick=1)
        oi2 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(5), y=YLoc(5), tick=1)
        SituatedObjectInstance.connect(frame, oi1)
        SituatedObjectInstance.connect(frame, oi2)

        ambiguous = _get_ambiguous_uuids(frame)
        assert ObjectId(42) in ambiguous

    def test_get_ambiguous_uuids_unique_not_flagged(self):
        """Single ObjectInstance per uuid -> not ambiguous."""
        from roc.pipeline.temporal.sequencer import Frame

        frame = Frame()
        oi1 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(1), y=YLoc(1), tick=1)
        oi2 = ObjectInstance(object_uuid=ObjectId(99), x=XLoc(5), y=YLoc(5), tick=1)
        SituatedObjectInstance.connect(frame, oi1)
        SituatedObjectInstance.connect(frame, oi2)

        ambiguous = _get_ambiguous_uuids(frame)
        assert len(ambiguous) == 0

    def test_skips_multi_instance_same_uuid(self):
        """2 ObjectInstance(uuid=X) in frame1 -> no transform for X."""
        from roc.pipeline.temporal.sequencer import Frame, NextFrame

        frame1 = Frame()
        frame2 = Frame()
        NextFrame.connect(frame1, frame2)

        # Frame1: 2 instances of uuid 42
        oi1a = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(1), y=YLoc(1), tick=1)
        oi1b = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(5), y=YLoc(5), tick=1)
        SituatedObjectInstance.connect(frame1, oi1a)
        SituatedObjectInstance.connect(frame1, oi1b)

        # Frame2: 1 instance of uuid 42
        oi2 = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(3), y=YLoc(3), tick=2)
        SituatedObjectInstance.connect(frame2, oi2)

        transform = _compute_transforms(frame2, frame1)
        # No child transforms should exist for the ambiguous uuid
        child_edges = transform.src_edges.select(type="Change")
        assert len(child_edges) == 0

    def test_unique_instance_still_gets_transform(self):
        """uuid=X unique in both frames -> ObjectTransform computed."""
        from roc.pipeline.temporal.sequencer import Frame, NextFrame

        frame1 = Frame()
        frame2 = Frame()
        NextFrame.connect(frame1, frame2)

        # Frame1: 1 instance of uuid 42, 2 instances of uuid 99 (ambiguous)
        oi1_a = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(1), y=YLoc(1), tick=1)
        oi1_b = ObjectInstance(object_uuid=ObjectId(99), x=XLoc(5), y=YLoc(5), tick=1)
        oi1_c = ObjectInstance(object_uuid=ObjectId(99), x=XLoc(8), y=YLoc(8), tick=1)
        SituatedObjectInstance.connect(frame1, oi1_a)
        SituatedObjectInstance.connect(frame1, oi1_b)
        SituatedObjectInstance.connect(frame1, oi1_c)

        # Frame2: 1 instance of uuid 42 (moved), 1 instance of uuid 99
        oi2_a = ObjectInstance(object_uuid=ObjectId(42), x=XLoc(3), y=YLoc(3), tick=2)
        oi2_b = ObjectInstance(object_uuid=ObjectId(99), x=XLoc(6), y=YLoc(6), tick=2)
        SituatedObjectInstance.connect(frame2, oi2_a)
        SituatedObjectInstance.connect(frame2, oi2_b)

        transform = _compute_transforms(frame2, frame1)
        # Should have a transform for uuid 42 (moved from (1,1) to (3,3))
        # but NOT for uuid 99 (ambiguous in frame1)
        from roc.pipeline.object.object_transform import ObjectTransform

        child_edges = transform.src_edges.select(type="Change")
        object_transforms = [e.dst for e in child_edges if isinstance(e.dst, ObjectTransform)]
        # uuid 42 should have a transform (position changed)
        assert len(object_transforms) == 1
        assert object_transforms[0].object_uuid == ObjectId(42)
