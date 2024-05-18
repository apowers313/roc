# mypy: disable-error-code="no-untyped-def"

from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.feature_extractors.line import Line, LineFeature
from roc.perception import Settled, VisionData
from roc.point import TypedPointCollection


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
            VisionData(
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

        # # event 1
        # e = s.output.call_args_list[0].args[0]
        # assert isinstance(e, Event)
        # assert isinstance(e.data, LineFeature)
        # assert isinstance(e.data.feature, TypedPointCollection)
        # ln = e.data.feature
        # assert ln.size == 5
        # assert ln.type == 1
        # p = ln.points[0]
        # assert p.x == 0 and p.y == 2

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
            VisionData(
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

        # # event 3
        e = s.output.call_args_list[2].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, LineFeature)
        assert isinstance(e.data.feature, TypedPointCollection)
        ln = e.data.feature
        assert ln.size == 5
        assert ln.type == 1
        p = ln.points[0]
        assert p.x == 1 and p.y == 2

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
            VisionData(
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
        assert isinstance(e.data.feature, TypedPointCollection)
        ln = e.data.feature
        assert ln.size == 5
        assert ln.type == 1
        p = ln.points[0]
        assert p.x == 2 and p.y == 0

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
            VisionData(
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
        assert isinstance(e.data.feature, TypedPointCollection)
        ln = e.data.feature
        assert ln.size == 5
        assert ln.type == 1
        p = ln.points[0]
        assert p.x == 2 and p.y == 1

        # event 6
        e = s.output.call_args_list[5].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, Settled)
