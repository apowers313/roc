# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/significance.py."""

from roc.significance import SignificanceData


class TestSignificanceData:
    def test_constructor(self):
        sd = SignificanceData(significance=0.75)
        assert sd.significance == 0.75

    def test_significance_field(self):
        sd = SignificanceData(significance=0.0)
        assert sd.significance == 0.0

        sd2 = SignificanceData(significance=1.0)
        assert sd2.significance == 1.0
