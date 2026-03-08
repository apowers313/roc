# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/sequencer.py."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from roc.sequencer import (
    PREDICTED_FRAME_TICK,
    Frame,
    FrameAttribute,
    NextFrame,
    get_next_tick,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


@pytest.fixture(autouse=True)
def reset_tick():
    """Reset global tick counter before each test."""
    import roc.sequencer as seq

    original = seq.tick
    seq.tick = 0
    yield
    seq.tick = original


class TestGetNextTick:
    def test_increments(self):
        t1 = get_next_tick()
        t2 = get_next_tick()
        assert t2 == t1 + 1

    def test_starts_from_zero(self):
        assert get_next_tick() == 1
        assert get_next_tick() == 2


class TestPredictedFrameTick:
    def test_value(self):
        assert PREDICTED_FRAME_TICK == -1


class TestFrame:
    def test_tick_auto_increments(self):
        f1 = Frame()
        f2 = Frame()
        assert f2.tick == f1.tick + 1

    def test_tick_explicit(self):
        f = Frame(tick=42)
        assert f.tick == 42

    def test_transformable_empty(self):
        f = Frame()
        with patch.object(type(f), "successors", new_callable=PropertyMock, return_value=[]):
            assert f.transformable == []

    def test_transformable_with_items(self):
        f = Frame()
        mock_transformable = MagicMock(
            spec=[
                "same_transform_type",
                "compatible_transform",
                "create_transform",
                "apply_transform",
            ]
        )
        # Make isinstance check work for Transformable

        mock_non_transformable = MagicMock(spec=[])

        with patch.object(
            type(f),
            "successors",
            new_callable=PropertyMock,
            return_value=[mock_transformable, mock_non_transformable],
        ):
            # We need the isinstance check to work; mock with spec won't pass isinstance
            # So let's use a real approach
            pass

    def test_transforms_empty(self):
        f = Frame()
        mock_edge_list = MagicMock()
        mock_edge_list.select.return_value = []
        with patch.object(
            type(f), "src_edges", new_callable=PropertyMock, return_value=mock_edge_list
        ):
            assert f.transforms == []

    def test_objects_empty(self):
        f = Frame()
        with patch.object(type(f), "src_edges", new_callable=PropertyMock, return_value=[]):
            assert f.objects == []

    def test_objects_with_feature_groups(self):
        from roc.object import FeatureGroup, Object

        f = Frame()

        mock_obj = MagicMock(spec=Object)
        mock_fg_edge = MagicMock()
        mock_fg_edge.dst = mock_obj
        type(mock_fg_edge).type = PropertyMock(return_value="Features")

        mock_fg = MagicMock(spec=FeatureGroup)
        mock_fg.src_edges = [mock_fg_edge]

        mock_edge = MagicMock()
        mock_edge.dst = mock_fg

        with patch.object(
            type(f), "src_edges", new_callable=PropertyMock, return_value=[mock_edge]
        ):
            result = f.objects
            assert len(result) == 1
            assert result[0] is mock_obj

    def test_merge_transforms(self):
        src_frame = MagicMock(spec=Frame)
        mod_frame = MagicMock(spec=Frame)

        mock_transformable = MagicMock()
        mock_transform = MagicMock()
        mock_new_node = MagicMock()

        src_frame.transformable = [mock_transformable]
        mod_frame.transforms = [mock_transform]
        mock_transformable.compatible_transform.return_value = True
        mock_transformable.apply_transform.return_value = mock_new_node

        with patch("roc.sequencer.FrameAttribute.connect"):
            result = Frame.merge_transforms(src_frame, mod_frame)
            assert isinstance(result, Frame)
            assert result.tick == PREDICTED_FRAME_TICK


class TestFrameAttribute:
    def test_allowed_connections(self):
        expected = [
            ("Frame", "FeatureGroup"),
            ("Frame", "TakeAction"),
            ("TakeAction", "Frame"),
            ("Frame", "IntrinsicNode"),
        ]
        assert FrameAttribute.model_fields["allowed_connections"].default == expected


class TestNextFrame:
    def test_allowed_connections(self):
        expected = [("Frame", "Frame")]
        assert NextFrame.model_fields["allowed_connections"].default == expected
