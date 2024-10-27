# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_blstats import blstat0 as test_blstat
from helpers.util import StubComponent

from roc.component import Component
from roc.intrinsic import Intrinsic, IntrinsicBoolOp, IntrinsicIntOp


class TestIntrinsicInt:
    def test_validate(self) -> None:
        op = IntrinsicIntOp("test", -2, 8)
        assert not op.validate(-3)
        assert op.validate(-2)
        assert op.validate(3)
        assert op.validate(8)
        assert not op.validate(9)

    def test_normalize(self) -> None:
        op = IntrinsicIntOp("test", -2, 8)
        assert op.range == 10
        assert op.normalize(-2) == 0
        assert op.normalize(3) == 0.5
        assert op.normalize(8) == 1


class TestIntrinsicBool:
    def test_validate(self) -> None:
        op = IntrinsicBoolOp("test")
        assert op.validate(True)
        assert op.validate(False)

    def test_normalize(self) -> None:
        op = IntrinsicBoolOp("test")
        assert op.normalize(False) == 0
        assert op.normalize(True) == 1


class TestIntrinsic:
    def test_exists(self) -> None:
        Intrinsic()

    def test_basic(self) -> None:
        intrinsic = Component.get("intrinsic", "intrinsic")
        assert isinstance(intrinsic, Intrinsic)
        s = StubComponent(
            input_bus=intrinsic.int_conn.attached_bus,
            output_bus=intrinsic.int_conn.attached_bus,
        )

        s.input_conn.send(test_blstat)

        # assert s.output.call_count == 3
