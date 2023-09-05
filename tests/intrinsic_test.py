# mypy: disable-error-code="no-untyped-def"

from roc.intrinsic import intrinsic_registry


class TestIntrinsicRegistry:
    def test_registered(self):
        assert "no-op" in intrinsic_registry
