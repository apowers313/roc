# mypy: disable-error-code="no-untyped-def"

from unittest.mock import MagicMock

from helpers.nethack_screens import screens
from helpers.util import component_response_args

from roc.perception import VisionData

screen0 = VisionData(screen=screens[0]["chars"])


class TestDelta:
    @component_response_args("delta", "perception", "pb_conn", screen0)
    def test_basic(self, component_response) -> None:
        print("component_response result:", component_response)
        assert isinstance(component_response, MagicMock)
        assert component_response.call_count == 1
        print(component_response.call_args)
