# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.color import Color, ColorFeature, ColorNode
from roc.feature_extractors.single import Single
from roc.location import XLoc, YLoc
from roc.perception import Settled, VisionData


class TestColor:
    def test_color_exists(self) -> None:
        Color()

    def test_to_nodes(self, fake_component) -> None:
        f = ColorFeature(origin_id=("foo", "bar"), type=31337, point=(XLoc(1), YLoc(2)))
        n = f.to_nodes()
        assert isinstance(n, ColorNode)
        assert n.labels == {"FeatureNode", "ColorNode"}
        assert n.type == 31337

    def test_screen0(self, empty_components) -> None:
        c = Component.get("color", "perception")
        assert isinstance(c, Color)
        single = Component.get("single", "perception")
        assert isinstance(single, Single)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
            filter=lambda e: e.src_id.name == "color",
        )

        s.input_conn.send(VisionData.from_dict(screens[0]))

        assert s.output.call_count == 14

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.feature_name == "Color"
        assert e.data.type == 7  # -
        assert e.data.point == (15, 3)

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 7  # -
        assert e.data.point == (19, 3)

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 7  # |
        assert e.data.point == (15, 4)

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 7  # |
        assert e.data.point == (19, 4)

        # event 5
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 7  # .
        assert e.data.point == (15, 5)

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 6  # [
        assert e.data.point == (16, 5)

        # event 7
        e = s.output.call_args_list[6].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 15  # @
        assert e.data.point == (17, 5)

        # event 8
        e = s.output.call_args_list[7].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 5  # x
        assert e.data.point == (18, 5)

        # event 9
        e = s.output.call_args_list[8].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 3  # +
        assert e.data.point == (19, 5)

        # event 10
        e = s.output.call_args_list[9].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 15  # f
        assert e.data.point == (16, 6)

        # event 11
        e = s.output.call_args_list[10].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 7  # -
        assert e.data.point == (15, 8)

        # event 12
        e = s.output.call_args_list[11].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 3  # |
        assert e.data.point == (18, 8)

        # event 13
        e = s.output.call_args_list[12].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, ColorFeature)
        assert e.data.type == 7  # -
        assert e.data.point == (19, 8)

        # event 14
        e = s.output.call_args_list[13].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
