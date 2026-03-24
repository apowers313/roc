# mypy: disable-error-code="no-untyped-def"


from typing import Any

import pytest
from helpers.nethack_screens2 import screens
from helpers.util import StubComponent

from roc.attention import VisionAttention
from roc.component import Component
from roc.event import Event
from roc.feature_extractors.color import Color, ColorFeature
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.distance import Distance
from roc.feature_extractors.flood import Flood
from roc.feature_extractors.line import Line
from roc.feature_extractors.motion import Motion
from roc.feature_extractors.shape import Shape, ShapeFeature
from roc.feature_extractors.single import Single, SingleFeature
from roc.graphdb import Node
from roc.location import XLoc, YLoc
from roc.object import (
    FeatureGroup,
    Object,
    ObjectResolver,
    ResolutionContext,
    ResolvedObject,
    SymmetricDifferenceResolution,
)
from roc.perception import FeatureNode, VisionData


@pytest.fixture
def features() -> Any:
    # colors
    color1: FeatureNode = ColorFeature(
        origin_id=("foo", "bar"), type=2, point=(XLoc(1), YLoc(2))
    ).to_nodes()
    color2: FeatureNode = ColorFeature(
        origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2))
    ).to_nodes()
    color3: FeatureNode = ColorFeature(
        origin_id=("foo", "bar"), type=4, point=(XLoc(1), YLoc(2))
    ).to_nodes()

    # shapes
    shape1: FeatureNode = ShapeFeature(
        origin_id=("foo", "bar"), type=64, point=(XLoc(1), YLoc(2))
    ).to_nodes()
    shape2: FeatureNode = ShapeFeature(
        origin_id=("foo", "bar"), type=106, point=(XLoc(1), YLoc(2))
    ).to_nodes()
    shape3: FeatureNode = ShapeFeature(
        origin_id=("foo", "bar"), type=100, point=(XLoc(1), YLoc(2))
    ).to_nodes()

    # singles
    single1: FeatureNode = SingleFeature(
        origin_id=("foo", "bar"), type=1, point=(XLoc(1), YLoc(2))
    ).to_nodes()
    single2: FeatureNode = SingleFeature(
        origin_id=("foo", "bar"), type=2, point=(XLoc(1), YLoc(2))
    ).to_nodes()
    single3: FeatureNode = SingleFeature(
        origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2))
    ).to_nodes()

    ret: dict[str, Any] = {
        "color1": color1,
        "color2": color2,
        "color3": color3,
        "shape1": shape1,
        "shape2": shape2,
        "shape3": shape3,
        "single1": single1,
        "single2": single2,
        "single3": single3,
    }

    return ret


class TestObject:
    def test_basic(self) -> None:
        o = Object()
        assert o.uuid > 0
        assert isinstance(o, Node)

    def test_object_features(self, features) -> None:
        fg = FeatureGroup.from_nodes([features["shape1"], features["color1"], features["single1"]])
        obj = Object.with_features(fg)

        assert len(obj.features) == 3
        assert features["shape1"] in obj.features
        assert features["color1"] in obj.features
        assert features["single1"] in obj.features

    def test_distance_zero(self, features) -> None:
        fg = FeatureGroup.from_nodes([features["shape1"], features["color1"], features["single1"]])
        obj = Object.with_features(fg)

        dist = SymmetricDifferenceResolution._distance(
            obj, [features["shape1"], features["color1"], features["single1"]]
        )
        assert dist == 0

    def test_distance_one(self, features) -> None:
        fg = FeatureGroup.from_nodes([features["shape1"], features["color1"]])
        obj = Object.with_features(fg)

        dist = SymmetricDifferenceResolution._distance(
            obj, [features["shape1"], features["color1"], features["single1"]]
        )
        assert dist == 1

    def test_distance_two_exclusive(self, features) -> None:
        fg = FeatureGroup.from_nodes([features["shape1"], features["color1"]])
        obj = Object.with_features(fg)

        dist = SymmetricDifferenceResolution._distance(
            obj, [features["color1"], features["single1"]]
        )
        assert dist == 2


class TestSymmetricDifferenceResolution:
    def setup_method(self) -> None:
        self.resolution = SymmetricDifferenceResolution()

    def test_empty(self) -> None:
        f = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2)))
        nodes = f.to_nodes()
        candidates = self.resolution._find_candidates([nodes])
        assert len(candidates) == 0

    def test_one_object(self) -> None:
        fn = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2))).to_nodes()
        fg = FeatureGroup.from_nodes([fn])
        o = Object.with_features(fg)
        candidates = self.resolution._find_candidates([fn])
        assert len(candidates) == 1
        o2, dist = candidates[0]
        assert o2 is o
        assert dist == 0

    def test_two_objects(self, features) -> None:
        fg1 = FeatureGroup.from_nodes([features["single1"], features["single2"]])
        o1 = Object.with_features(fg1)
        fg2 = FeatureGroup.from_nodes([features["single2"], features["single3"]])
        o2 = Object.with_features(fg2)
        assert o1 is not o2

        # feature common to both objects
        candidates = self.resolution._find_candidates([features["single2"]])
        assert len(candidates) == 2
        candidate_ids = [c[0].id for c in candidates]
        assert o1.id in candidate_ids
        assert o2.id in candidate_ids

        # feature only in object 1
        candidates = self.resolution._find_candidates([features["single1"]])
        assert len(candidates) == 1
        candidate_ids = [c[0].id for c in candidates]
        assert o1.id in candidate_ids
        assert o2.id not in candidate_ids

        # feature only in object 2
        candidates = self.resolution._find_candidates([features["single3"]])
        assert len(candidates) == 1
        candidate_ids = [c[0].id for c in candidates]
        assert o1.id not in candidate_ids
        assert o2.id in candidate_ids

        # overlapping features, object 1 is stronger
        candidates = self.resolution._find_candidates([features["single1"], features["single2"]])
        assert len(candidates) == 2
        ret_obj, dist = candidates[0]
        assert ret_obj is o1
        assert dist == 0
        ret_obj, dist = candidates[1]
        assert ret_obj is o2
        assert dist == 2

        # overlapping features, object 2 is stronger
        candidates = self.resolution._find_candidates([features["single2"], features["single3"]])
        assert len(candidates) == 2
        ret_obj, dist = candidates[0]
        assert ret_obj is o2
        assert dist == 0
        ret_obj, dist = candidates[1]
        assert ret_obj is o1
        assert dist == 2


class TestSymmetricDifferenceRegression:
    """Regression tests for symmetric-difference object resolution bugs."""

    def setup_method(self) -> None:
        self.resolution = SymmetricDifferenceResolution()

    def test_distinct_features_produce_distinct_objects(self, features) -> None:
        """Objects with completely different features must not collapse into one.

        Regression: the resolver was matching all observations to the first object
        because shared FeatureNodes created graph paths to the original object,
        and the distance check compared against stale features (only the first
        FeatureGroup was ever attached).

        In production this caused 15,407 resolutions with only 1 unique object.
        """
        ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

        # Observation 1: single1 + color1 + shape1
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        result1 = self.resolution.resolve(fg1.feature_nodes, fg1, ctx)
        assert result1 is None, "first observation should create a new object"
        Object.with_features(fg1)

        # Observation 2: single2 + color2 + shape2 (completely different features)
        fg2 = FeatureGroup.from_nodes([features["single2"], features["color2"], features["shape2"]])
        ctx2 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=2)
        result2 = self.resolution.resolve(fg2.feature_nodes, fg2, ctx2)
        assert result2 is None, (
            "observation with completely different features should create a new object, "
            f"but matched existing object {result2}"
        )

    def test_shared_feature_does_not_cause_false_match(self, features) -> None:
        """Two observations sharing one feature but differing in two should not match.

        The distance threshold is <= 1, so a symmetric difference of 2 or more
        should produce a new object rather than matching.
        """
        ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

        # Observation 1: single1 + color1 + shape1
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        result1 = self.resolution.resolve(fg1.feature_nodes, fg1, ctx)
        assert result1 is None
        Object.with_features(fg1)

        # Observation 2: single1 + color2 + shape2 (shares single1, differs in 2)
        fg2 = FeatureGroup.from_nodes([features["single1"], features["color2"], features["shape2"]])
        ctx2 = ResolutionContext(x=XLoc(5), y=YLoc(5), tick=2)
        result2 = self.resolution.resolve(fg2.feature_nodes, fg2, ctx2)
        # symmetric diff: {color1, shape1} ^ {color2, shape2} = 4, so should NOT match
        assert result2 is None, (
            f"observations differing in 2+ features should not match, but matched object {result2}"
        )

    def test_identical_features_should_match(self, features) -> None:
        """Observations with the same features should match the existing object."""
        ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)

        # Observation 1: single1 + color1 + shape1
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        result1 = self.resolution.resolve(fg1.feature_nodes, fg1, ctx)
        assert result1 is None
        obj1 = Object.with_features(fg1)

        # Observation 2: same features
        fg2 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        ctx2 = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=2)
        result2 = self.resolution.resolve(fg2.feature_nodes, fg2, ctx2)
        assert result2 is obj1, "identical features should match the existing object"

    def test_many_observations_do_not_collapse(self, features) -> None:
        """Simulates multiple ticks: distinct feature sets must produce distinct objects.

        This is the core regression scenario from production where 15k+ resolutions
        all collapsed into a single object.
        """
        objects: list[Object] = []

        # Three observations with completely distinct feature sets
        feature_sets = [
            [features["single1"], features["color1"], features["shape1"]],
            [features["single2"], features["color2"], features["shape2"]],
            [features["single3"], features["color3"], features["shape3"]],
        ]

        for i, feat_set in enumerate(feature_sets):
            fg = FeatureGroup.from_nodes(feat_set)
            ctx_i = ResolutionContext(x=XLoc(i), y=YLoc(i), tick=i + 1)
            result = self.resolution.resolve(fg.feature_nodes, fg, ctx_i)
            if result is None:
                obj = Object.with_features(fg)
                objects.append(obj)
            else:
                objects.append(result)

        unique_ids = {o.uuid for o in objects}
        assert len(unique_ids) == 3, (
            f"3 distinct feature sets should produce 3 distinct objects, got {len(unique_ids)}"
        )

    def test_reused_features_cause_false_match(self, features) -> None:
        """When a new observation shares a feature node with an existing object but
        differs in other features, the resolver should NOT match if distance > 1.

        Regression: in production, FeatureNodes are cached and reused. A new
        observation sharing one SingleNode with an existing object would find that
        object as a candidate via graph walk. The bug is that with distance <= 1,
        observations sharing 2 of 3 features incorrectly match, and over many ticks
        all observations collapse to a single object.
        """
        # Observation 1: single1 + color1 + shape1 -> new object
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        ctx1 = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)
        result1 = self.resolution.resolve(fg1.feature_nodes, fg1, ctx1)
        assert result1 is None
        Object.with_features(fg1)

        # Observation 2: single1 + color1 + shape2
        # Shares single1 and color1 with obj1 (distance should be 2: shape1 vs shape2)
        fg2 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape2"]])
        ctx2 = ResolutionContext(x=XLoc(2), y=YLoc(2), tick=2)
        result2 = self.resolution.resolve(fg2.feature_nodes, fg2, ctx2)
        assert result2 is None, (
            "observation differing in shape should not match (distance=2), "
            f"but matched object {result2}"
        )

    def test_repeated_same_location_collapses_to_one_object(self, features) -> None:
        """Observing the same features repeatedly should always match the same object.

        Regression: simulates the production loop where do_object_resolution
        creates a new FeatureGroup each tick (via FeatureGroup.with_features)
        but never attaches it to the matched object. Each tick's FeatureGroup
        adds more predecessor links from shared FeatureNodes, but the distance
        check should still work correctly.
        """
        objects_created: list[Object] = []

        # Tick 1: first observation -> new object
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        ctx1 = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)
        result = self.resolution.resolve(fg1.feature_nodes, fg1, ctx1)
        assert result is None
        obj1 = Object.with_features(fg1)
        objects_created.append(obj1)

        # Ticks 2-10: same features, should match obj1 each time
        for tick in range(2, 11):
            fg = FeatureGroup.from_nodes(
                [features["single1"], features["color1"], features["shape1"]]
            )
            ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=tick)
            result = self.resolution.resolve(fg.feature_nodes, fg, ctx)
            assert result is obj1, (
                f"tick {tick}: identical features should match obj1, got {result}"
            )

    def test_different_location_produces_new_object(self, features) -> None:
        """Observing completely different features should produce a new object,
        even after many ticks of observing the first object.

        Regression: the production bug showed 15k+ resolutions collapsing to 1
        object. This test ensures that after repeated matches to obj1, a truly
        different observation still creates a new object.
        """
        # Create first object
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        ctx1 = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)
        result = self.resolution.resolve(fg1.feature_nodes, fg1, ctx1)
        assert result is None
        obj1 = Object.with_features(fg1)

        # Match obj1 a few times (simulating production loop)
        for tick in range(2, 6):
            fg = FeatureGroup.from_nodes(
                [features["single1"], features["color1"], features["shape1"]]
            )
            ctx = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=tick)
            result = self.resolution.resolve(fg.feature_nodes, fg, ctx)
            assert result is obj1

        # Now observe completely different features
        fg_new = FeatureGroup.from_nodes(
            [features["single3"], features["color3"], features["shape3"]]
        )
        ctx_new = ResolutionContext(x=XLoc(10), y=YLoc(10), tick=10)
        result_new = self.resolution.resolve(fg_new.feature_nodes, fg_new, ctx_new)
        assert result_new is None, (
            f"completely different features should create new object, but matched {result_new}"
        )

    def test_distance_inflation_from_multiple_graph_paths(self, features) -> None:
        """Bug: _find_candidates uses += to accumulate distance, inflating it
        when the same object is found via multiple FeatureNode graph paths.

        If an observation shares N feature nodes with an object, _distance()
        is called N times and summed. True distance of 1 becomes N, causing
        near-matches to be rejected. This makes the resolver only match objects
        with EXACTLY identical features (distance=0), which in NetHack means
        all similar-looking cells collapse to one object per game.
        """
        # Object 1: single1 + color1 + shape1
        fg1 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape1"]])
        ctx1 = ResolutionContext(x=XLoc(1), y=YLoc(1), tick=1)
        result1 = self.resolution.resolve(fg1.feature_nodes, fg1, ctx1)
        assert result1 is None
        Object.with_features(fg1)

        # Observation 2: single1 + color1 + shape2
        # Shares single1 and color1 -> obj1 found via 2 graph paths
        # True distance = 2 (shape1 vs shape2 symmetric diff)
        # Inflated distance = 2 * 2 = 4 (distance computed and summed twice)
        fg2 = FeatureGroup.from_nodes([features["single1"], features["color1"], features["shape2"]])

        candidates = self.resolution._find_candidates(fg2.feature_nodes)
        assert len(candidates) >= 1, "should find obj1 as candidate"
        _, dist = candidates[0]
        # BUG: distance is inflated because _distance is called once per shared
        # feature node and summed via +=. True distance is 2, but gets doubled.
        assert dist == pytest.approx(2.0), (
            f"distance should be 2 (true symmetric diff), got {dist} "
            f"(inflated by += accumulation if > 2)"
        )

    def test_no_physical_features_shared_node_false_match(self) -> None:
        """Bug: when two observations share a non-physical feature node and have
        no physical features, _distance returns 0 (empty set ^ empty set).

        This causes completely different events to match because the distance
        check only considers SingleNode/ColorNode/ShapeNode.
        """
        from roc.feature_extractors.delta import DeltaFeature

        # Create a shared DeltaNode (same old_val/new_val = same hash = same node)
        shared_delta = DeltaFeature(
            origin_id=("foo", "bar"),
            old_val=100,
            new_val=200,
            point=(XLoc(10), YLoc(5)),
        ).to_nodes()

        # A different DeltaNode
        other_delta = DeltaFeature(
            origin_id=("foo", "bar"),
            old_val=300,
            new_val=400,
            point=(XLoc(30), YLoc(15)),
        ).to_nodes()

        # Observation 1: shared_delta + other features at (10,5)
        fg1 = FeatureGroup.from_nodes([shared_delta])
        ctx1 = ResolutionContext(x=XLoc(10), y=YLoc(5), tick=1)
        result1 = self.resolution.resolve(fg1.feature_nodes, fg1, ctx1)
        assert result1 is None
        Object.with_features(fg1)

        # Observation 2: shared_delta + different delta at (30,15)
        # Found as candidate via shared_delta -> fg1 -> obj1
        # Both have no physical features, so distance = 0 -> false match
        fg2 = FeatureGroup.from_nodes([shared_delta, other_delta])
        ctx2 = ResolutionContext(x=XLoc(30), y=YLoc(15), tick=2)
        result2 = self.resolution.resolve(fg2.feature_nodes, fg2, ctx2)
        # BUG: distance is 0 because _distance ignores non-physical features
        assert result2 is None, (
            "observations with no physical features but different non-physical "
            "features should not match, but _distance returns 0 for empty sets"
        )


class TestObjectResolver:
    def test_exists(self, empty_components) -> None:
        ObjectResolver()

    def test_basic(self, empty_components) -> None:
        object_resolver = Component.get("resolver", "object")
        assert isinstance(object_resolver, ObjectResolver)
        attention = Component.get("vision", "attention")
        assert isinstance(attention, VisionAttention)
        delta = Component.get("delta", "perception")
        assert isinstance(delta, Delta)
        flood = Component.get("flood", "perception")
        assert isinstance(flood, Flood)
        line = Component.get("line", "perception")
        assert isinstance(line, Line)
        motion = Component.get("motion", "perception")
        assert isinstance(motion, Motion)
        single = Component.get("single", "perception")
        assert isinstance(single, Single)
        distance = Component.get("distance", "perception")
        assert isinstance(distance, Distance)
        color = Component.get("color", "perception")
        assert isinstance(color, Color)
        shape = Component.get("shape", "perception")
        assert isinstance(shape, Shape)
        s = StubComponent(
            input_bus=delta.pb_conn.attached_bus,
            output_bus=object_resolver.obj_res_conn.attached_bus,
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))
        s.input_conn.send(VisionData.from_dict(screens[2]))

        assert s.output.call_count == 3

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ResolvedObject)
        o = e.data.object
        first_uuid = o.uuid

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ResolvedObject)
        o = e.data.object
        second_uuid = o.uuid
        assert second_uuid != first_uuid

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ResolvedObject)
        o = e.data.object
        third_uuid = o.uuid
        assert third_uuid != first_uuid
        assert third_uuid == second_uuid
        assert o.resolve_count == 1
