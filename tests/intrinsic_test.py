# mypy: disable-error-code="no-untyped-def"

from helpers.nethack_blstats import blstat0 as test_blstat
from helpers.util import StubComponent

from roc.intrinsic import (
    Intrinsic,
    IntrinsicBoolOp,
    IntrinsicData,
    IntrinsicIntOp,
    config_intrinsics,
)


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

        s.input_conn.send(IntrinsicData(test_blstat))

        # assert s.output.call_count == 3

    def test_config_int(self) -> None:
        ret = config_intrinsics([("hp", "int:-3:7")])

        assert len(ret) == 1
        assert isinstance(ret["hp"], IntrinsicIntOp)
        assert ret["hp"].name == "hp"
        assert ret["hp"].min == -3
        assert ret["hp"].max == 7
        assert ret["hp"].range == 10

    def test_config_bool(self) -> None:
        ret = config_intrinsics([("foo", "bool")])

        assert len(ret) == 1
        assert isinstance(ret["foo"], IntrinsicBoolOp)
        assert ret["foo"].name == "foo"
