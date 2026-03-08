# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/intrinsic.py."""

from unittest.mock import MagicMock, patch

import pytest

from roc.config import (
    ConfigBoolIntrinsic,
    ConfigIntIntrinsic,
    ConfigIntrinsicType,
    ConfigMapIntrinsic,
    ConfigPercentIntrinsic,
)


@pytest.fixture(autouse=True)
def mock_db():
    mock = MagicMock()
    mock.strict_schema = False
    mock.strict_schema_warns = False
    mock.node_counter = MagicMock()
    mock.edge_counter = MagicMock()
    with patch("roc.graphdb.GraphDB.singleton", return_value=mock):
        yield mock


class TestIntrinsicIntOp:
    def test_constructor(self):
        from roc.intrinsic import IntrinsicIntOp

        op = IntrinsicIntOp("test_hp", config=(0, 100))
        assert op.name == "test_hp"
        assert op.min == 0
        assert op.max == 100
        assert op.range == 100

    def test_constructor_negative_range(self):
        from roc.intrinsic import IntrinsicIntOp

        op = IntrinsicIntOp("test", config=(-50, 50))
        assert op.min == -50
        assert op.max == 50
        assert op.range == 100

    def test_validate_in_range(self):
        from roc.intrinsic import IntrinsicIntOp

        op = IntrinsicIntOp("test", config=(0, 100))
        assert op.validate(50) is True
        assert op.validate(0) is True
        assert op.validate(100) is True

    def test_validate_out_of_range(self):
        from roc.intrinsic import IntrinsicIntOp

        op = IntrinsicIntOp("test", config=(0, 100))
        assert op.validate(-1) is False
        assert op.validate(101) is False

    def test_normalize(self):
        from roc.intrinsic import IntrinsicIntOp

        op = IntrinsicIntOp("test", config=(0, 100))
        assert op.normalize(0) == 0.0
        assert op.normalize(100) == 1.0
        assert op.normalize(50) == 0.5

    def test_normalize_negative_range(self):
        from roc.intrinsic import IntrinsicIntOp

        op = IntrinsicIntOp("test", config=(-50, 50))
        assert op.normalize(0) == 0.5
        assert op.normalize(-50) == 0.0
        assert op.normalize(50) == 1.0


class TestIntrinsicPercentOp:
    def test_constructor(self):
        from roc.intrinsic import IntrinsicPercentOp

        op = IntrinsicPercentOp("hp", config="maxhp")
        assert op.name == "hp"
        assert op.base == "maxhp"

    def test_validate_positive_int(self):
        from roc.intrinsic import IntrinsicPercentOp

        op = IntrinsicPercentOp("hp", config="maxhp")
        assert op.validate(10) is True

    def test_validate_zero(self):
        from roc.intrinsic import IntrinsicPercentOp

        op = IntrinsicPercentOp("hp", config="maxhp")
        assert op.validate(0) is False

    def test_validate_negative(self):
        from roc.intrinsic import IntrinsicPercentOp

        op = IntrinsicPercentOp("hp", config="maxhp")
        assert op.validate(-1) is False

    def test_normalize(self):
        from roc.intrinsic import IntrinsicPercentOp

        op = IntrinsicPercentOp("hp", config="maxhp")
        result = op.normalize(50, raw_intrinsics={"maxhp": 100})
        assert result == 0.5

    def test_normalize_full(self):
        from roc.intrinsic import IntrinsicPercentOp

        op = IntrinsicPercentOp("hp", config="maxhp")
        result = op.normalize(100, raw_intrinsics={"maxhp": 100})
        assert result == 1.0


class TestIntrinsicMapOp:
    def test_constructor(self):
        from roc.intrinsic import IntrinsicMapOp

        mapping = {0: 0.0, 1: 0.5, 2: 1.0}
        op = IntrinsicMapOp("hunger", config=mapping)
        assert op.name == "hunger"
        assert op.map == mapping

    def test_validate_in_map(self):
        from roc.intrinsic import IntrinsicMapOp

        op = IntrinsicMapOp("hunger", config={0: 0.0, 1: 0.5})
        assert op.validate(0) is True
        assert op.validate(1) is True

    def test_validate_not_in_map(self):
        from roc.intrinsic import IntrinsicMapOp

        op = IntrinsicMapOp("hunger", config={0: 0.0, 1: 0.5})
        assert op.validate(99) is False

    def test_normalize(self):
        from roc.intrinsic import IntrinsicMapOp

        op = IntrinsicMapOp("hunger", config={0: 0.0, 1: 0.5, 2: 1.0})
        assert op.normalize(0) == 0.0
        assert op.normalize(1) == 0.5
        assert op.normalize(2) == 1.0


class TestIntrinsicBoolOp:
    def test_validate_always_true(self):
        from roc.intrinsic import IntrinsicBoolOp

        op = IntrinsicBoolOp("blind")
        assert op.validate(True) is True
        assert op.validate(False) is True

    def test_normalize_true(self):
        from roc.intrinsic import IntrinsicBoolOp

        op = IntrinsicBoolOp("blind")
        assert op.normalize(True) == 1.0

    def test_normalize_false(self):
        from roc.intrinsic import IntrinsicBoolOp

        op = IntrinsicBoolOp("blind")
        assert op.normalize(False) == 0.0


class TestIntrinsicOpInitSubclass:
    def test_raises_if_intrinsic_type_not_set(self):
        from roc.intrinsic import IntrinsicOp

        with pytest.raises(TypeError, match="must set intrinsic_type"):

            class BadOp(IntrinsicOp[int]):
                def validate(self, val):
                    return True

                def normalize(self, val, **kwargs):
                    return 0.0

    def test_raises_on_duplicate_type(self):
        from roc.intrinsic import IntrinsicOp

        with pytest.raises(TypeError, match="already registered"):

            class DuplicateOp(IntrinsicOp[int]):
                intrinsic_type = "int"  # already registered

                def validate(self, val):
                    return True

                def normalize(self, val, **kwargs):
                    return 0.0


class TestIntrinsicNode:
    def test_str(self):
        from roc.intrinsic import IntrinsicNode

        node = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        assert str(node) == "IntrinsicNode('hp', 50(0.5))"

    def test_same_transform_type_true(self):
        from roc.intrinsic import IntrinsicNode

        node1 = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        node2 = IntrinsicNode(name="hp", raw_value=40, normalized_value=0.4)
        assert node1.same_transform_type(node2) is True

    def test_same_transform_type_false_different_name(self):
        from roc.intrinsic import IntrinsicNode

        node1 = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        node2 = IntrinsicNode(name="energy", raw_value=40, normalized_value=0.4)
        assert node1.same_transform_type(node2) is False

    def test_same_transform_type_false_different_type(self):
        from roc.intrinsic import IntrinsicNode

        node1 = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        other = MagicMock()
        assert node1.same_transform_type(other) is False

    def test_compatible_transform(self):
        from roc.intrinsic import IntrinsicNode, IntrinsicTransform

        node = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        t = IntrinsicTransform(name="hp", normalized_change=-0.1)
        assert node.compatible_transform(t) is True

    def test_compatible_transform_false(self):
        from roc.intrinsic import IntrinsicNode

        node = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        other = MagicMock()
        assert node.compatible_transform(other) is False

    def test_create_transform_same_value_returns_none(self):
        from roc.intrinsic import IntrinsicNode

        node1 = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        node2 = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        assert node2.create_transform(node1) is None

    def test_create_transform_different_value(self):
        from roc.intrinsic import IntrinsicNode, IntrinsicTransform

        prev = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        curr = IntrinsicNode(name="hp", raw_value=40, normalized_value=0.4)
        t = curr.create_transform(prev)
        assert t is not None
        assert isinstance(t, IntrinsicTransform)
        assert t.name == "hp"
        assert pytest.approx(t.normalized_change) == -0.1

    def test_apply_transform(self):
        from roc.intrinsic import IntrinsicNode, IntrinsicTransform

        node = IntrinsicNode(name="hp", raw_value=50, normalized_value=0.5)
        t = IntrinsicTransform(name="hp", normalized_change=-0.1)
        result = node.apply_transform(t)
        assert isinstance(result, IntrinsicNode)
        assert result.name == "hp"
        assert pytest.approx(result.normalized_value) == 0.4
        assert result.raw_value is None


class TestIntrinsicTransform:
    def test_str(self):
        from roc.intrinsic import IntrinsicTransform

        t = IntrinsicTransform(name="hp", normalized_change=-0.1)
        assert str(t) == "IntrinsicTransform('hp', -0.1)"


class TestIntrinsicData:
    def test_constructor_normalizes(self):
        from roc.intrinsic import Intrinsic, IntrinsicData, IntrinsicIntOp

        # Set up intrinsic spec manually
        Intrinsic.intrinsic_spec = {"hp": IntrinsicIntOp("hp", config=(0, 100))}
        data = IntrinsicData({"hp": 50})
        assert data.intrinsics == {"hp": 50}
        assert pytest.approx(data.normalized_intrinsics["hp"]) == 0.5

    def test_repr(self):
        from roc.intrinsic import Intrinsic, IntrinsicData, IntrinsicIntOp

        Intrinsic.intrinsic_spec = {"hp": IntrinsicIntOp("hp", config=(0, 100))}
        data = IntrinsicData({"hp": 50})
        r = repr(data)
        assert "hp: 50" in r

    def test_to_nodes(self):
        from roc.intrinsic import Intrinsic, IntrinsicData, IntrinsicIntOp, IntrinsicNode

        Intrinsic.intrinsic_spec = {"hp": IntrinsicIntOp("hp", config=(0, 100))}
        data = IntrinsicData({"hp": 50})
        nodes = data.to_nodes()
        assert len(nodes) == 1
        assert isinstance(nodes[0], IntrinsicNode)
        assert nodes[0].name == "hp"
        assert nodes[0].raw_value == 50
        assert pytest.approx(nodes[0].normalized_value) == 0.5


class TestConfigIntrinsics:
    def test_creates_int_op(self):
        from roc.intrinsic import _config_intrinsics

        specs: list[ConfigIntrinsicType] = [ConfigIntIntrinsic(name="hp", config=(0, 100))]
        result = _config_intrinsics(specs)
        assert "hp" in result
        assert result["hp"].name == "hp"

    def test_creates_percent_op(self):
        from roc.intrinsic import _config_intrinsics

        specs: list[ConfigIntrinsicType] = [ConfigPercentIntrinsic(name="hp_pct", config="maxhp")]
        result = _config_intrinsics(specs)
        assert "hp_pct" in result

    def test_creates_map_op(self):
        from roc.intrinsic import _config_intrinsics

        specs: list[ConfigIntrinsicType] = [
            ConfigMapIntrinsic(name="hunger", config={0: 0.0, 1: 0.5})
        ]
        result = _config_intrinsics(specs)
        assert "hunger" in result

    def test_creates_bool_op(self):
        from roc.intrinsic import _config_intrinsics

        specs: list[ConfigIntrinsicType] = [ConfigBoolIntrinsic(name="blind")]
        result = _config_intrinsics(specs)
        assert "blind" in result

    def test_creates_multiple(self):
        from roc.intrinsic import _config_intrinsics

        specs: list[ConfigIntrinsicType] = [
            ConfigIntIntrinsic(name="hp", config=(0, 100)),
            ConfigBoolIntrinsic(name="blind"),
        ]
        result = _config_intrinsics(specs)
        assert len(result) == 2
