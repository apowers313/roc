# mypy: disable-error-code="no-untyped-def"

from helpers.util import StubComponent

from roc.action import Action, TakeAction
from roc.component import Component
from roc.intrinsic import Intrinsic, IntrinsicData, IntrinsicTransform
from roc.object import Object, ObjectResolver
from roc.sequencer import Sequencer
from roc.transformable import Transform
from roc.transformer import Transformer


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
        object_resolver.obj_res_conn.send(Object())
        intrinsic.int_conn.send(IntrinsicData({"hunger": 1, "hp": 14, "hpmax": 14}))
        s.input_conn.send(TakeAction(action=20))

        # second frame
        object_resolver.obj_res_conn.send(Object())
        intrinsic.int_conn.send(IntrinsicData({"hunger": 2, "hp": 7, "hpmax": 14}))
        s.input_conn.send(TakeAction(action=20))

        assert s.output.call_count == 1

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e.data, Transform)
        t = e.data
        assert len(t.src_edges) == 3
        assert len(t.dst_edges) == 1
        transform_nodes = t.successors.select(labels={"Transform", "IntrinsicTransform"})
        assert len(transform_nodes) == 2
        assert isinstance(transform_nodes[0], IntrinsicTransform)
        assert transform_nodes[0].normalized_change == -0.5
        assert isinstance(transform_nodes[1], IntrinsicTransform)
        assert transform_nodes[1].normalized_change == -0.25
