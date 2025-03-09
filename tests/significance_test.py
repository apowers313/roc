# mypy: disable-error-code="no-untyped-def"

import math

from helpers.nethack_blstats import blstat0 as test_blstat
from helpers.util import StubComponent

from roc.component import Component
from roc.event import Event
from roc.intrinsic import Intrinsic, RawIntrinsicData
from roc.significance import Significance, SignificanceData


class TestSignificance:
    def test_exists(self, empty_components) -> None:
        Significance()

    def test_basic(self, empty_components) -> None:
        significance = Component.get("significance", "significance")
        assert isinstance(significance, Significance)
        intrinsic = Component.get("intrinsic", "intrinsic")
        assert isinstance(intrinsic, Intrinsic)
        s = StubComponent(
            input_bus=intrinsic.int_conn.attached_bus,
            output_bus=significance.significance_conn.attached_bus,
        )

        s.input_conn.send(RawIntrinsicData(test_blstat))

        assert s.output.call_count == 1

        # first event
        e = s.output.call_args_list[0].args[0]
        assert isinstance(e, Event)
        assert isinstance(e.data, SignificanceData)
        sig = e.data
        assert math.isclose(sig.significance, 1.4)
