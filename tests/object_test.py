# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens2 import screens
from helpers.util import StubComponent

from roc.attention import VisionAttention
from roc.component import Component
from roc.event import Event
from roc.feature_extractors.color import Color
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.distance import Distance
from roc.feature_extractors.flood import Flood
from roc.feature_extractors.line import Line
from roc.feature_extractors.motion import Motion
from roc.feature_extractors.shape import Shape
from roc.feature_extractors.single import Single, SingleFeature
from roc.graphdb import Node
from roc.location import XLoc, YLoc
from roc.object import CandidateObjects, Object, ObjectResolver
from roc.perception import VisionData


class TestObject:
    def test_basic(self) -> None:
        o = Object()
        assert o.uuid > 0
        assert isinstance(o, Node)


class TestCandidateObjects:
    def test_empty(self) -> None:
        f = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2)))
        nodes = f.to_nodes()
        objs = CandidateObjects([nodes])
        assert len(objs) == 0

    def test_one_object(self) -> None:
        o = Object()
        fn = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2))).to_nodes()
        Node.connect(o, fn, "Feature")
        objs = CandidateObjects([fn])
        assert len(objs) == 1
        o2, str = objs[0]
        assert o2 is o
        assert str == 1.0

    def test_two_objects(self) -> None:
        f1 = SingleFeature(origin_id=("foo", "bar"), type=1, point=(XLoc(1), YLoc(2))).to_nodes()
        f2 = SingleFeature(origin_id=("foo", "bar"), type=2, point=(XLoc(1), YLoc(2))).to_nodes()
        f3 = SingleFeature(origin_id=("foo", "bar"), type=3, point=(XLoc(1), YLoc(2))).to_nodes()
        assert f1 is not f2
        assert f2 is not f3
        assert f1 is not f3
        o1 = Object.with_features([f1, f2])
        o2 = Object.with_features([f2, f3])
        assert o1 is not o2

        # feature common to both objects
        objs = CandidateObjects([f2])
        assert len(objs) == 2
        assert o1.id in objs.order
        assert o2.id in objs.order

        # feature only in object 1
        objs = CandidateObjects([f1])
        assert len(objs) == 1
        assert o1.id in objs.order
        assert o2.id not in objs.order

        # feature only in object 2
        objs = CandidateObjects([f3])
        assert len(objs) == 1
        assert o1.id not in objs.order
        assert o2.id in objs.order

        # overlapping features, object 1 is stronger
        objs = CandidateObjects([f1, f2])
        assert len(objs) == 2
        assert o1.id in objs.order
        assert o2.id in objs.order
        ret_obj, str = objs[0]
        assert ret_obj is o1
        assert str == 2.0
        ret_obj, str = objs[1]
        assert ret_obj is o2
        assert str == 1.0

        # overlapping features, object 2 is stronger
        objs = CandidateObjects([f2, f3])
        assert len(objs) == 2
        assert o1.id in objs.order
        assert o2.id in objs.order
        ret_obj, str = objs[0]
        assert ret_obj is o2
        assert str == 2.0
        ret_obj, str = objs[1]
        assert ret_obj is o1
        assert str == 1.0


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
        uuid = o.uuid

        # second event
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Object)
        o = e.data
        # assert o.resolve_count == 1
        assert o.uuid == uuid

        # third event
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Object)
        o = e.data
        assert o.resolve_count == 2
        assert o.uuid == uuid
