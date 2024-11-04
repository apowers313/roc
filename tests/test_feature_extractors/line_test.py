# mypy: disable-error-code="no-untyped-def"

from helpers.util import (
    StubComponent,
)

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.line import Line, LineFeature
from roc.perception import Settled, VisionData


class TestLine:
    def test_line_exists(self) -> None:
        Line()

    def test_horizontal(self, empty_components) -> None:
        c = Component.get("line", "perception")
        assert isinstance(c, Line)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 6

        # event 1
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.feature_name == "Line"
        assert e.data.size == 5
        assert e.data.type == 0
        assert e.data.points == {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}

        # event 2
        e = s.output.call_args_list[1].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 0
        assert e.data.points == {(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)}

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)}

        # event 4
        e = s.output.call_args_list[3].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 0
        assert e.data.points == {(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)}

        # event 5
        e = s.output.call_args_list[4].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 0
        assert e.data.points == {(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)}

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_horizontal_partial(self, empty_components) -> None:
        c = Component.get("line", "perception")
        assert isinstance(c, Line)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [2, 0, 0, 0, 0, 0, 2, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0, 0, 2, 2],
                ]
            )
        )

        assert s.output.call_count == 6

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)}

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_vertical(self, empty_components) -> None:
        c = Component.get("line", "perception")
        assert isinstance(c, Line)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                ]
            )
        )

        assert s.output.call_count == 6

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)}

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)

    def test_vertical_partial(self, empty_components) -> None:
        c = Component.get("line", "perception")
        assert isinstance(c, Line)
        s = StubComponent(
            input_bus=c.pb_conn.attached_bus,
            output_bus=c.pb_conn.attached_bus,
        )

        s.input_conn.send(
            VisionData.for_test(
                [
                    [2, 0, 0, 0, 2],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [2, 0, 0, 0, 2],
                    [2, 0, 0, 0, 2],
                ]
            )
        )

        assert s.output.call_count == 6

        # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert e.data.size == 5
        assert e.data.type == 1
        assert e.data.points == {(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)}

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
