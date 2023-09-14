# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from helpers.nethack_screens import screens
from helpers.util import component_response_args

from roc.event import Event
from roc.perception import DeltaData, VisionData

screen0 = VisionData(screen=screens[0]["chars"])
screen1 = VisionData(screen=screens[1]["chars"])


class TestDelta:
    @component_response_args("delta", "perception", "pb_conn", [screen0, screen1])
    def test_basic(self, empty_components, component_response) -> None:
        assert isinstance(component_response, MagicMock)
        assert component_response.call_count == 1
        assert len(component_response.call_args.args) == 1
        e = component_response.call_args.args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, DeltaData)
        diff_list = e.data.diff_list
        assert len(diff_list) == 2
        # Diff(x=6, y=16, val1=102, val2=46), Diff(x=6, y=17, val1=46, val2=102)])
        assert diff_list[0].x == 6
        assert diff_list[0].y == 16
        assert diff_list[0].val1 == 102  # f
        assert diff_list[0].val2 == 46  # .
        assert diff_list[1].x == 6
        assert diff_list[1].y == 17
        assert diff_list[1].val1 == 46  # .
        assert diff_list[1].val2 == 102  # f
        # kitten moved right!
