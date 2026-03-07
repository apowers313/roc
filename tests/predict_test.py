# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens3 import screens as screens3
from helpers.util import StubComponent

from roc.action import Action, TakeAction
from roc.attention import VisionAttention
from roc.component import Component
from roc.config import Config
from roc.feature_extractors.color import Color
from roc.feature_extractors.delta import Delta
from roc.feature_extractors.distance import Distance
from roc.feature_extractors.flood import Flood
from roc.feature_extractors.line import Line
from roc.feature_extractors.motion import Motion
from roc.feature_extractors.shape import Shape
from roc.feature_extractors.single import Single
from roc.intrinsic import Intrinsic
from roc.object import ObjectResolver
from roc.perception import VisionData
from roc.predict import Predict
from roc.sequencer import Sequencer
from roc.transformer import Transformer


class TestPredict:
    def test_exists(self, empty_components) -> None:
        Predict()

    def test_basic(self, empty_components) -> None:
        predict = Component.get("predict", "predict")
        assert isinstance(predict, Predict)
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
        delta = Component.get("delta", "perception")
        assert isinstance(delta, Delta)
        attention = Component.get("vision", "attention")
        assert isinstance(attention, VisionAttention)
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
            input_bus=delta.pb_conn.attached_bus,  # perception bus
            output_bus=predict.predict_conn.attached_bus,
        )

        settings = Config.get()
        if settings.gym_actions is None:
            raise ValueError("Trying to get action before actions have been configured")

        action_idx = settings.gym_actions.index(108)

        s.input_conn.send(VisionData.from_dict(screens3[0]))
        action.action_bus_conn.send(TakeAction(action=action_idx))

        s.input_conn.send(VisionData.from_dict(screens3[1]))
        action.action_bus_conn.send(TakeAction(action=action_idx))

        s.input_conn.send(VisionData.from_dict(screens3[2]))
        action.action_bus_conn.send(TakeAction(action=action_idx))

        print("s.output.call_count", s.output.call_count)
        print("first", s.output.call_args_list[0].args[0].data)

        # # first frame
        # o = Object()
        # fg = FeatureGroup()
        # object_resolver.obj_res_conn.send(
        #     ResolvedObject(object=o, feature_group=fg, x=XLoc(0), y=YLoc(0))
        # )
        # intrinsic.int_conn.send(IntrinsicData({"hunger": 1, "hp": 14, "hpmax": 14}))
        # s.input_conn.send(TakeAction(action=20))

        # # second frame
        # object_resolver.obj_res_conn.send(
        #     ResolvedObject(object=o, feature_group=fg, x=XLoc(0), y=YLoc(0))
        # )
        # intrinsic.int_conn.send(IntrinsicData({"hunger": 2, "hp": 7, "hpmax": 14}))
        # s.input_conn.send(TakeAction(action=20))

        # assert s.output.call_count == 1
        # print("s.output.call_count", s.output.call_count)
        # print("first", s.output.call_args_list[0].args[0])

        # # first event
        # e = s.output.call_args_list[0].args[0]
        # assert isinstance(e.data, Transform)
        # t = e.data
        # assert len(t.src_edges) == 3
        # assert len(t.dst_edges) == 1
        # transform_nodes = t.successors.select(labels={"Transform", "IntrinsicTransform"})
        # assert len(transform_nodes) == 2
        # assert isinstance(transform_nodes[0], IntrinsicTransform)
        # assert transform_nodes[0].name == "hp"
        # assert transform_nodes[0].normalized_change == 0.5
        # assert isinstance(transform_nodes[1], IntrinsicTransform)
        # assert transform_nodes[0].name == "hunger"
        # assert transform_nodes[1].normalized_change == -0.25
