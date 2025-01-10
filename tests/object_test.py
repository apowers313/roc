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
from roc.object import CandidateObjects, FeatureGroup, Object, ObjectResolver
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

        dist = Object.distance(obj, [features["shape1"], features["color1"], features["single1"]])
        assert dist == 0

    def test_distance_one(self, features) -> None:
        fg = FeatureGroup.from_nodes([features["shape1"], features["color1"]])
        obj = Object.with_features(fg)

        dist = Object.distance(obj, [features["shape1"], features["color1"], features["single1"]])
        assert dist == 1

    def test_distance_two_exclusive(self, features) -> None:
        fg = FeatureGroup.from_nodes([features["shape1"], features["color1"]])
        obj = Object.with_features(fg)

        dist = Object.distance(obj, [features["color1"], features["single1"]])
        assert dist == 2


class TestCandidateObjects:
    def test_empty(self) -> None:
        f = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2)))
        nodes = f.to_nodes()
        objs = CandidateObjects([nodes])
        assert len(objs) == 0

    def test_one_object(self) -> None:
        o = Object()
        fn = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2))).to_nodes()
        fg = FeatureGroup.from_nodes([fn])
        Node.connect(o, fg, "Features")
        objs = CandidateObjects([fn])
        assert len(objs) == 1
        o2, dist = objs[0]
        assert o2 is o
        assert dist == 0

    def test_two_objects(self, features) -> None:
        fg1 = FeatureGroup.from_nodes([features["single1"], features["single2"]])
        o1 = Object.with_features(fg1)
        fg2 = FeatureGroup.from_nodes([features["single2"], features["single3"]])
        o2 = Object.with_features(fg2)
        assert o1 is not o2

        # feature common to both objects
        objs = CandidateObjects([features["single2"]])
        assert len(objs) == 2
        assert o1.id in objs.order
        assert o2.id in objs.order

        # feature only in object 1
        objs = CandidateObjects([features["single1"]])
        assert len(objs) == 1
        assert o1.id in objs.order
        assert o2.id not in objs.order

        # feature only in object 2
        objs = CandidateObjects([features["single3"]])
        assert len(objs) == 1
        assert o1.id not in objs.order
        assert o2.id in objs.order

        # overlapping features, object 1 is stronger
        objs = CandidateObjects([features["single1"], features["single2"]])
        assert len(objs) == 2
        assert o1.id in objs.order
        assert o2.id in objs.order
        ret_obj, dist = objs[0]
        assert ret_obj is o1
        assert dist == 0
        ret_obj, dist = objs[1]
        assert ret_obj is o2
        assert dist == 2

        # overlapping features, object 2 is stronger
        objs = CandidateObjects([features["single2"], features["single3"]])
        assert len(objs) == 2
        assert o1.id in objs.order
        assert o2.id in objs.order
        ret_obj, dist = objs[0]
        assert ret_obj is o2
        assert dist == 0
        ret_obj, dist = objs[1]
        assert ret_obj is o1
        assert dist == 2


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
        assert isinstance(e.data, Object)
        o = e.data
        # assert o.resolve_count == 0
        first_uuid = o.uuid

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Object)
        o = e.data
        # assert o.resolve_count == 1
        second_uuid = o.uuid
        assert second_uuid != first_uuid

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Object)
        o = e.data
        third_uuid = o.uuid
        assert third_uuid != first_uuid
        assert third_uuid == second_uuid
        assert o.resolve_count == 1
