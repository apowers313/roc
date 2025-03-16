# mypy: disable-error-code="no-untyped-def"

from helpers.util import StubComponent

from roc.action import Action, TakeAction
from roc.component import Component
from roc.event import Event
from roc.intrinsic import Intrinsic, IntrinsicData, IntrinsicNode
from roc.object import Object, ObjectResolver
from roc.sequencer import Frame, Sequencer


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
        assert frame.src_edges[0].type == "FrameAttributes"
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
        object_resolver.obj_res_conn.send(o)
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
        print(frame.src_edges[3].dst)
        assert len(frame.edges) == 5
        assert len(frame.src_edges) == 5
        # object
        assert frame.src_edges[0].type == "FrameAttributes"
        assert frame.src_edges[0].dst is o
        # hp intrinsic
        assert frame.src_edges[1].type == "FrameAttributes"
        assert isinstance(frame.src_edges[1].dst, IntrinsicNode)
        assert frame.src_edges[1].dst.name in {"hunger", "hp"}
        # hunger intrinsic
        assert frame.src_edges[2].type == "FrameAttributes"
        assert isinstance(frame.src_edges[2].dst, IntrinsicNode)
        assert frame.src_edges[2].dst.name in {"hunger", "hp"}
        # action
        assert frame.src_edges[3].type == "FrameAttributes"
        assert frame.src_edges[3].dst is a
        # next frame
        assert frame.src_edges[4].type == "NextFrame"
        assert isinstance(frame.src_edges[4].dst, Frame)
