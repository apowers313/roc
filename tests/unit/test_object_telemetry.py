# mypy: disable-error-code="no-untyped-def"

"""Unit tests for object resolution telemetry (Phase 2)."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from roc.location import XLoc, YLoc
from roc.object import (
    Object,
    ObjectResolver,
    ResolutionContext,
    SymmetricDifferenceResolution,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


def make_feature_node(label: str, str_repr: str) -> MagicMock:
    """Create a mock FeatureNode with label and string representation."""
    fn = MagicMock()
    fn.labels = {label, "FeatureNode"}
    fn.configure_mock(**{"__str__": MagicMock(return_value=str_repr)})
    return fn


class TestSymmetricDifferenceDecisionCounter:
    def test_resolution_decision_counter_match(self):
        """Verify decision counter records 'match' outcome."""
        resolution = SymmetricDifferenceResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        # Create a candidate object
        obj = Object()
        with patch.object(type(obj), "features", new_callable=PropertyMock, return_value=[fn1]):
            # Mock graph walk: fn1 -> FeatureGroup -> Object
            mock_fg = MagicMock()
            mock_fg_nl = MagicMock()
            mock_fg_nl.select.return_value = [mock_fg]
            fn1.predecessors = mock_fg_nl

            mock_obj_nl = MagicMock()
            mock_obj_nl.select.return_value = [obj]
            mock_fg.predecessors = mock_obj_nl

            fg = MagicMock()
            ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

            with patch.object(resolution.decision_counter, "add") as mock_add:
                result = resolution.resolve([fn1], fg, ctx)
                assert result is obj
                mock_add.assert_called_once_with(1, attributes={"outcome": "match"})

    def test_resolution_decision_counter_new(self):
        """Verify decision counter records 'new_object' outcome."""
        resolution = SymmetricDifferenceResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")
        fn2 = make_feature_node("SingleNode", "SingleNode(b)")

        # Create a candidate object with very different features
        obj = Object()
        obj_feature = make_feature_node("SingleNode", "SingleNode(z)")
        with patch.object(
            type(obj),
            "features",
            new_callable=PropertyMock,
            return_value=[obj_feature],
        ):
            # Mock graph walk to return candidate with high distance
            mock_fg = MagicMock()
            mock_fg_nl = MagicMock()
            mock_fg_nl.select.return_value = [mock_fg]
            fn1.predecessors = mock_fg_nl

            fn2_nl = MagicMock()
            fn2_nl.select.return_value = []
            fn2.predecessors = fn2_nl

            mock_obj_nl = MagicMock()
            mock_obj_nl.select.return_value = [obj]
            mock_fg.predecessors = mock_obj_nl

            fg = MagicMock()
            ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

            with patch.object(resolution.decision_counter, "add") as mock_add:
                result = resolution.resolve([fn1, fn2], fg, ctx)
                assert result is None
                mock_add.assert_called_once_with(1, attributes={"outcome": "new_object"})

    def test_candidates_histogram_recorded(self):
        """Verify candidate count is recorded as histogram."""
        resolution = SymmetricDifferenceResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        obj = Object()
        with patch.object(type(obj), "features", new_callable=PropertyMock, return_value=[fn1]):
            mock_fg = MagicMock()
            mock_fg_nl = MagicMock()
            mock_fg_nl.select.return_value = [mock_fg]
            fn1.predecessors = mock_fg_nl

            mock_obj_nl = MagicMock()
            mock_obj_nl.select.return_value = [obj]
            mock_fg.predecessors = mock_obj_nl

            fg = MagicMock()
            ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

            with patch.object(resolution.candidates_histogram, "record") as mock_record:
                resolution.resolve([fn1], fg, ctx)
                mock_record.assert_called_once_with(1)

    def test_candidates_histogram_zero(self):
        """Verify candidate count of zero is recorded."""
        resolution = SymmetricDifferenceResolution()
        fn1 = make_feature_node("SingleNode", "SingleNode(a)")

        mock_nl = MagicMock()
        mock_nl.select.return_value = []
        fn1.predecessors = mock_nl

        fg = MagicMock()
        ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

        with patch.object(resolution.candidates_histogram, "record") as mock_record:
            resolution.resolve([fn1], fg, ctx)
            mock_record.assert_called_once_with(0)


def _make_attention_event(x: int, y: int):
    """Build a minimal mock AttentionEvent for do_object_resolution."""
    import pandas as pd

    focus_points = pd.DataFrame([{"x": x, "y": y, "strength": 1.0, "label": 0}])
    mock_feature = MagicMock()
    saliency_map = MagicMock()
    saliency_map.get_val.return_value = [mock_feature]

    event = MagicMock()
    event.data.focus_points = focus_points
    event.data.saliency_map = saliency_map
    return event


def _make_resolver():
    """Create an ObjectResolver without triggering bus/component setup."""
    resolver = ObjectResolver.__new__(ObjectResolver)
    resolver.resolved_object_counter = MagicMock()
    resolver.obj_res_conn = MagicMock()
    return resolver


class TestResolutionTelemetry:
    def test_spatial_distance_recorded_on_match(self):
        """When an object is matched, spatial distance is recorded."""
        resolver = _make_resolver()
        event = _make_attention_event(x=7, y=8)

        matched_obj = Object()
        matched_obj.last_x = XLoc(5)
        matched_obj.last_y = YLoc(5)
        matched_obj.last_tick = 10

        mock_resolution = MagicMock()
        mock_resolution.resolve.return_value = matched_obj

        with (
            patch("roc.object.FeatureGroup.with_features"),
            patch("roc.object.ObjectResolutionExpMod.get", return_value=mock_resolution),
            patch("roc.sequencer.tick", 11),
            patch.object(ObjectResolver, "spatial_distance_histogram") as mock_spatial,
            patch.object(ObjectResolver, "temporal_gap_histogram"),
        ):
            resolver.do_object_resolution(event)
            # Manhattan distance: |7-5| + |8-5| = 2 + 3 = 5
            mock_spatial.record.assert_called_once_with(5)

    def test_temporal_gap_recorded_on_match(self):
        """When an object is matched, temporal gap is recorded."""
        resolver = _make_resolver()
        event = _make_attention_event(x=5, y=5)

        matched_obj = Object()
        matched_obj.last_x = XLoc(5)
        matched_obj.last_y = YLoc(5)
        matched_obj.last_tick = 10

        mock_resolution = MagicMock()
        mock_resolution.resolve.return_value = matched_obj

        with (
            patch("roc.object.FeatureGroup.with_features"),
            patch("roc.object.ObjectResolutionExpMod.get", return_value=mock_resolution),
            patch("roc.sequencer.tick", 42),
            patch.object(ObjectResolver, "spatial_distance_histogram"),
            patch.object(ObjectResolver, "temporal_gap_histogram") as mock_temporal,
        ):
            resolver.do_object_resolution(event)
            # Gap: 42 - 10 = 32
            mock_temporal.record.assert_called_once_with(32)

    def test_no_spatial_distance_for_new_object(self):
        """New objects have no last position, so no spatial distance recorded."""
        resolver = _make_resolver()
        event = _make_attention_event(x=5, y=5)

        mock_resolution = MagicMock()
        mock_resolution.resolve.return_value = None  # No match -> new object

        with (
            patch("roc.object.FeatureGroup.with_features"),
            patch("roc.object.ObjectResolutionExpMod.get", return_value=mock_resolution),
            patch("roc.object.Object.with_features", return_value=Object()),
            patch("roc.sequencer.tick", 1),
            patch.object(ObjectResolver, "spatial_distance_histogram") as mock_spatial,
            patch.object(ObjectResolver, "temporal_gap_histogram") as mock_temporal,
        ):
            resolver.do_object_resolution(event)
            mock_spatial.record.assert_not_called()
            mock_temporal.record.assert_not_called()

    def test_no_temporal_gap_for_new_object(self):
        """Matched object with last_tick=0 does not record temporal gap."""
        resolver = _make_resolver()
        event = _make_attention_event(x=5, y=5)

        matched_obj = Object()
        matched_obj.last_x = XLoc(5)
        matched_obj.last_y = YLoc(5)
        matched_obj.last_tick = 0  # Never seen before

        mock_resolution = MagicMock()
        mock_resolution.resolve.return_value = matched_obj

        with (
            patch("roc.object.FeatureGroup.with_features"),
            patch("roc.object.ObjectResolutionExpMod.get", return_value=mock_resolution),
            patch("roc.sequencer.tick", 10),
            patch.object(ObjectResolver, "spatial_distance_histogram"),
            patch.object(ObjectResolver, "temporal_gap_histogram") as mock_temporal,
        ):
            resolver.do_object_resolution(event)
            mock_temporal.record.assert_not_called()
