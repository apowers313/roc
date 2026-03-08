# mypy: disable-error-code="no-untyped-def"

"""End-to-end tests that exercise the full agent pipeline:
Environment -> Perception -> Attention -> ObjectResolver -> Sequencer -> Transformer
"""

import time
from unittest.mock import MagicMock

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.action import Action, ActionRequest, TakeAction
from roc.attention import Attention, VisionAttentionData
from roc.component import Component
from roc.event import Event
from roc.intrinsic import Intrinsic, IntrinsicData
from roc.object import ObjectResolver, ResolvedObject
from roc.perception import Perception, VisionData
from roc.sequencer import Frame, Sequencer
from roc.transformer import TransformResult, Transformer


class TestFullPipeline:
    def test_single_frame(self, all_components) -> None:
        """Feed intrinsics and an action through the pipeline; verify a frame is produced."""
        # Create a stub to observe sequencer output
        s = StubComponent(
            input_bus=Action.bus,
            output_bus=Sequencer.bus,
        )

        # Send intrinsic data
        intrinsic = Intrinsic.bus.connect(s)
        intrinsic.send(IntrinsicData({"hunger": 1, "hp": 14, "hpmax": 14}))

        # Send an action to trigger frame creation
        s.input_conn.send(ActionRequest())

        # Allow events to propagate
        time.sleep(0.5)

        # The pipeline should have produced a frame event
        assert s.output.call_count >= 1
        frame_event = s.output.call_args_list[-1].args[0]
        assert isinstance(frame_event, Event)
        assert isinstance(frame_event.data, Frame)

    def test_two_frames_produce_transform(self, all_components) -> None:
        """Feed two rounds of data; verify transforms are detected between frames."""
        # Create a stub to observe transformer output
        s = StubComponent(
            input_bus=Action.bus,
            output_bus=Transformer.bus,
        )

        # Connect to intrinsic bus
        intrinsic = Intrinsic.bus.connect(s)

        # Frame 1: hp at 100%
        intrinsic.send(IntrinsicData({"hunger": 1, "hp": 14, "hpmax": 14}))
        s.input_conn.send(ActionRequest())
        time.sleep(0.3)

        # Frame 2: hp dropped
        intrinsic.send(IntrinsicData({"hunger": 1, "hp": 10, "hpmax": 14}))
        s.input_conn.send(ActionRequest())
        time.sleep(0.3)

        # Transformer should have detected changes between frames
        assert s.output.call_count >= 1
        last_event = s.output.call_args_list[-1].args[0]
        assert isinstance(last_event, Event)
        assert isinstance(last_event.data, TransformResult)
