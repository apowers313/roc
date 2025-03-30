# mypy: disable-error-code="no-untyped-def"

from helpers.util import StubComponent

from roc.action import Action, TakeAction
from roc.component import Component
from roc.event import Event
from roc.intrinsic import Intrinsic, IntrinsicData, IntrinsicNode, IntrinsicTransform
from roc.location import XLoc, YLoc
from roc.object import FeatureGroup, Object, ObjectResolver, ResolvedObject
from roc.sequencer import PREDICTED_FRAME_TICK, Frame, FrameAttribute, Sequencer
from roc.transformable import Transform
from roc.transformer import Change


class TestFrame:
    def test_get_transforms(self) -> None:
        t = Transform()
        t1 = Transform()
        f = Frame(tick=-1)

        Change.connect(f, t)
        Change.connect(t, t1)

        assert len(f.transforms) == 1
        assert f.transforms[0] is t1

    def test_get_transformable(self) -> None:
        i = IntrinsicNode(name="hp", raw_value=7, normalized_value=0.5)
        f = Frame(tick=-1)
        assert len(f.src_edges) == 0
        assert len(f.transformable) == 0

        FrameAttribute.connect(f, i)

        assert len(f.src_edges) == 1
        assert len(f.transformable) == 1
        assert f.transformable[0] is i

    def test_merge_transforms(self) -> None:
        # the original frame
        src = Frame()
        i = IntrinsicNode(name="hp", raw_value=7, normalized_value=0.5)
        FrameAttribute.connect(src, i)
        # the frame for prediction
        mod = Frame(tick=-1)
        t = Transform()
        it = IntrinsicTransform(normalized_change=0.05)
        Change.connect(mod, t)
        Change.connect(t, it)

        res = Frame.merge_transforms(src, mod)

        assert isinstance(res, Frame)
        assert res.tick == PREDICTED_FRAME_TICK
        assert len(res.edges) == 1
        assert len(res.src_edges) == 1
        assert res.src_edges[0].type == "FrameAttribute"
        n = res.src_edges[0].dst
        assert n.labels == {"IntrinsicNode"}
        assert isinstance(n, IntrinsicNode)
        assert n.normalized_value == 0.55


class TestSequencer:
    def test_exists(self, empty_components) -> None:
        Sequencer()

    def test_action(self, empty_components) -> None:
        sequencer = Component.get("sequencer", "sequencer")
        assert isinstance(sequencer, Sequencer)
        action = Component.get("action", "action")
        assert isinstance(action, Action)

        s = StubComponent(
            input_bus=action.action_bus_conn.attached_bus,
            output_bus=sequencer.sequencer_conn.attached_bus,
        )

        take_action = TakeAction(action=20)
        s.input_conn.send(take_action)
        assert s.output.call_count == 1

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Frame)
        frame = e.data
        assert len(frame.edges) == 2
        assert len(frame.src_edges) == 2
        assert frame.src_edges[0].type == "FrameAttribute"
        assert frame.src_edges[0].dst is take_action
        assert frame.src_edges[1].type == "NextFrame"
        assert isinstance(frame.src_edges[1].dst, Frame)

    def test_basic(self, empty_components) -> None:
        sequencer = Component.get("sequencer", "sequencer")
        assert isinstance(sequencer, Sequencer)
        action = Component.get("action", "action")
        assert isinstance(action, Action)
        object_resolver = Component.get("resolver", "object")
        assert isinstance(object_resolver, ObjectResolver)
        intrinsic = Component.get("intrinsic", "intrinsic")
        assert isinstance(intrinsic, Intrinsic)

        s = StubComponent(
            input_bus=action.action_bus_conn.attached_bus,
            output_bus=sequencer.sequencer_conn.attached_bus,
        )

        o = Object()
        fg = FeatureGroup()
        object_resolver.obj_res_conn.send(
            ResolvedObject(object=o, feature_group=fg, x=XLoc(0), y=YLoc(0))
        )
        assert s.output.call_count == 0

        intrinsic.int_conn.send(IntrinsicData({"hunger": 1, "hp": 7, "hpmax": 14}))
        assert s.output.call_count == 0

        a = TakeAction(action=20)
        s.input_conn.send(a)
        assert s.output.call_count == 1

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Frame)
        frame = e.data
        assert len(frame.edges) == 5
        assert len(frame.src_edges) == 5
        # object
        assert frame.src_edges[0].type == "FrameAttribute"
        assert frame.src_edges[0].dst is fg
        # hp intrinsic
        assert frame.src_edges[1].type == "FrameAttribute"
        assert isinstance(frame.src_edges[1].dst, IntrinsicNode)
        assert frame.src_edges[1].dst.name in {"hunger", "hp"}
        # hunger intrinsic
        assert frame.src_edges[2].type == "FrameAttribute"
        assert isinstance(frame.src_edges[2].dst, IntrinsicNode)
        assert frame.src_edges[2].dst.name in {"hunger", "hp"}
        # action
        assert frame.src_edges[3].type == "FrameAttribute"
        assert frame.src_edges[3].dst is a
        # next frame
        assert frame.src_edges[4].type == "NextFrame"
        assert isinstance(frame.src_edges[4].dst, Frame)
