# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens3 import screens as screens3
from helpers.util import StubComponent

from roc.pipeline.action import Action, TakeAction
from roc.pipeline.attention.attention import VisionAttention
from roc.framework.component import Component
from roc.framework.config import Config
from roc.perception.feature_extractors.color import Color
from roc.perception.feature_extractors.delta import Delta
from roc.perception.feature_extractors.distance import Distance
from roc.perception.feature_extractors.flood import Flood
from roc.perception.feature_extractors.line import Line
from roc.perception.feature_extractors.motion import Motion
from roc.perception.feature_extractors.shape import Shape
from roc.perception.feature_extractors.single import Single
from roc.pipeline.intrinsic import Intrinsic
from roc.pipeline.object.object import ObjectResolver
from roc.perception.base import VisionData
from roc.pipeline.temporal.predict import Predict
from roc.pipeline.temporal.sequencer import Sequencer
from roc.pipeline.temporal.transformer import Transformer


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
