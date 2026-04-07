# mypy: disable-error-code="no-untyped-def"

import pytest
from helpers.util import StubComponent

from roc.pipeline.action import Action, TakeAction
from roc.framework.component import Component
from roc.framework.event import Event
from roc.pipeline.intrinsic import Intrinsic, IntrinsicData, IntrinsicNode, IntrinsicTransform
from roc.perception.location import XLoc, YLoc
from roc.pipeline.object.object import FeatureGroup, Object, ObjectResolver, ResolvedObject
from roc.pipeline.temporal.sequencer import PREDICTED_FRAME_TICK, Frame, FrameAttribute, Sequencer
from roc.pipeline.temporal.transformable import Transform
from roc.pipeline.temporal.transformer import Change


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
        it = IntrinsicTransform(name="hp", normalized_change=0.05)
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
        assert n.normalized_value == pytest.approx(0.55)


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

        # A TakeAction on an otherwise-empty Sequencer closes a brand-new
        # Frame with just the FrameAttribute -> TakeAction edge attached.
        # The "next" Frame is only created when the next cycle's first
        # data event arrives (lazy creation), so at the moment this frame
        # is emitted there is no NextFrame edge yet.
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Frame)
        frame = e.data
        assert len(frame.src_edges) == 1
        assert frame.src_edges[0].type == "FrameAttribute"
        assert frame.src_edges[0].dst is take_action

    def test_basic(self, empty_components, tmp_path) -> None:
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

        # Frame for cycle 1 is emitted: has feature/object edges, intrinsic
        # FrameAttribute edges, and the closing FrameAttribute to the action.
        # Sequencer now defers creating the next Frame until data for cycle 2
        # arrives, so there is no NextFrame edge yet -- see the multi-cycle
        # test below for that case.
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Frame)
        frame = e.data
        frame.render(depth=3, file_directory=tmp_path)
        # Frame1 should have: feature/object edges + 2 intrinsic FrameAttribute
        # edges + 1 action FrameAttribute edge. No NextFrame yet.
        edge_types = [edge.type for edge in frame.src_edges]
        assert "FrameAttribute" in edge_types
        assert "NextFrame" not in edge_types
        # Intrinsic nodes present
        intrinsic_nodes = [
            edge.dst for edge in frame.src_edges if isinstance(edge.dst, IntrinsicNode)
        ]
        assert len(intrinsic_nodes) == 2
        intrinsic_names = {n.name for n in intrinsic_nodes}
        assert intrinsic_names == {"hunger", "hp"}
        # Action present
        action_edges = [edge for edge in frame.src_edges if edge.dst is a]
        assert len(action_edges) == 1

    def test_multi_cycle_creates_next_frame(self, empty_components) -> None:
        """Across two action cycles, a NextFrame edge should link the frames
        and the pending TakeAction should attach to the new frame's
        FrameAttribute edges."""
        sequencer = Component.get("sequencer", "sequencer")
        assert isinstance(sequencer, Sequencer)
        action = Component.get("action", "action")
        assert isinstance(action, Action)
        intrinsic = Component.get("intrinsic", "intrinsic")
        assert isinstance(intrinsic, Intrinsic)

        s = StubComponent(
            input_bus=action.action_bus_conn.attached_bus,
            output_bus=sequencer.sequencer_conn.attached_bus,
        )

        # cycle 1: intrinsic data -> action
        intrinsic.int_conn.send(IntrinsicData({"hunger": 1, "hp": 7, "hpmax": 14}))
        a1 = TakeAction(action=20)
        s.input_conn.send(a1)
        assert s.output.call_count == 1
        frame1 = s.output.call_args_list[0].args[0].data
        assert isinstance(frame1, Frame)

        # cycle 2: intrinsic data (lazily creates frame2) -> action
        intrinsic.int_conn.send(IntrinsicData({"hunger": 1, "hp": 7, "hpmax": 14}))
        a2 = TakeAction(action=21)
        s.input_conn.send(a2)
        assert s.output.call_count == 2
        frame2 = s.output.call_args_list[1].args[0].data
        assert isinstance(frame2, Frame)
        assert frame2 is not frame1

        # frame1 should now have a NextFrame edge pointing at frame2
        next_frame_edges = frame1.src_edges.select(type="NextFrame")
        assert len(next_frame_edges) == 1
        assert next_frame_edges[0].dst is frame2

        # frame2 should have a FrameAttribute incoming from a1 (the action
        # that led into it) AND a FrameAttribute outgoing to a2 (its closing
        # action).
        incoming_action_edges = [edge for edge in frame2.dst_edges if edge.src is a1]
        assert len(incoming_action_edges) == 1
        outgoing_action_edges = [edge for edge in frame2.src_edges if edge.dst is a2]
        assert len(outgoing_action_edges) == 1
