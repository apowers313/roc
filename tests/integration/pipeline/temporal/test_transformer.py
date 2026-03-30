# mypy: disable-error-code="no-untyped-def"

from helpers.util import StubComponent

from roc.pipeline.action import Action, TakeAction
from roc.framework.component import Component
from roc.pipeline.intrinsic import Intrinsic, IntrinsicData, IntrinsicTransform
from roc.perception.location import XLoc, YLoc
from roc.pipeline.object.object import FeatureGroup, Object, ObjectResolver, ResolvedObject
from roc.pipeline.temporal.sequencer import Frame, Sequencer
from roc.pipeline.temporal.transformable import Transform
from roc.pipeline.temporal.transformer import Change, Transformer, TransformResult


class TestTransform:
    def test_src_frame(self) -> None:
        src = Frame(tick=-1)
        dst = Frame(tick=-1)
        t = Transform()
        Change.connect(src, t)
        Change.connect(t, dst)

        ret = t.src_frame

        assert ret is src

    def test_dst_frame(self) -> None:
        src = Frame(tick=-1)
        dst = Frame(tick=-1)
        t = Transform()
        Change.connect(src, t)
        Change.connect(t, dst)

        ret = t.dst_frame

        assert ret is dst


class TestTransformer:
    def test_exists(self, empty_components) -> None:
        Transformer()

    def test_basic(self, empty_components) -> None:
        transformer = Component.get("transformer", "transformer")
        assert isinstance(transformer, Transformer)
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
            output_bus=transformer.transformer_conn.attached_bus,
        )

        # first frame
        o = Object()
        fg = FeatureGroup()
        object_resolver.obj_res_conn.send(
            ResolvedObject(object=o, feature_group=fg, x=XLoc(0), y=YLoc(0))
        )
        intrinsic.int_conn.send(IntrinsicData({"hunger": 1, "hp": 14, "hpmax": 14}))
        s.input_conn.send(TakeAction(action=20))

        # second frame
        object_resolver.obj_res_conn.send(
            ResolvedObject(object=o, feature_group=fg, x=XLoc(0), y=YLoc(0))
        )
        intrinsic.int_conn.send(IntrinsicData({"hunger": 2, "hp": 7, "hpmax": 14}))
        s.input_conn.send(TakeAction(action=20))

        assert s.output.call_count == 1

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e.data, TransformResult)
        t = e.data.transform
        assert len(t.src_edges) == 3
        assert len(t.dst_edges) == 1
        transform_nodes = t.successors.select(labels={"Transform", "IntrinsicTransform"})
        assert len(transform_nodes) == 2
        assert isinstance(transform_nodes[0], IntrinsicTransform)
        assert transform_nodes[0].name == "hp"
        assert transform_nodes[0].normalized_change == -0.5
        assert isinstance(transform_nodes[1], IntrinsicTransform)
        assert transform_nodes[1].name == "hunger"
        assert transform_nodes[1].normalized_change == -0.25
