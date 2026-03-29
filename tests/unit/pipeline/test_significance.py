# mypy: disable-error-code="no-untyped-def"

"""Unit tests for roc/significance.py."""

import pytest

from roc.significance import SignificanceData


class TestSignificanceData:
    def test_constructor(self):
        sd = SignificanceData(significance=0.75)
        assert sd.significance == pytest.approx(0.75)

    def test_significance_field(self):
        sd = SignificanceData(significance=0.0)
        assert sd.significance == pytest.approx(0.0)

        sd2 = SignificanceData(significance=1.0)
        assert sd2.significance == pytest.approx(1.0)
