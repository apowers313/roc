# mypy: disable-error-code="no-untyped-def"

"""Unit tests for SaliencyMap pure logic (no DB, no component wiring)."""

from copy import deepcopy
from unittest.mock import MagicMock

from roc.attention import SaliencyMap
from roc.location import IntGrid


def _make_grid():
    return IntGrid(
        [
            [32, 32, 32],
            [49, 50, 51],
            [97, 98, 99],
        ]
    )


def _make_feature(name="Test"):
    """Create a mock VisualFeature with the required feature_name attribute."""
    f = MagicMock()
    f.feature_name = name
    return f


class TestSaliencyMap:
    def test_create(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        assert sm.shape == (3, 3)

    def test_get_empty_cell(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        val = sm.get_val(0, 0)
        assert isinstance(val, list)
        assert len(val) == 0

    def test_get_all_cells_empty(self) -> None:
        g = IntGrid(
            [
                [32, 32, 32],
                [49, 50, 51],
                [97, 98, 99],
                [97, 98, 99],
            ]
        )
        sm = SaliencyMap(g)
        val = sm.get_val(0, 0)
        assert len(val) == 0
        val = sm.get_val(2, 3)
        assert len(val) == 0

    def test_add_val(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature()
        sm.add_val(1, 2, f)

        assert len(sm.get_val(0, 0)) == 0
        val = sm.get_val(1, 2)
        assert len(val) == 1
        assert val[0] is f

    def test_add_multiple(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f1 = _make_feature()
        f2 = _make_feature()
        sm.add_val(2, 2, f1)
        sm.add_val(2, 2, f2)

        assert len(sm.get_val(0, 0)) == 0
        val = sm.get_val(2, 2)
        assert len(val) == 2
        assert f1 in val
        assert f2 in val

    def test_clear(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature()
        sm.add_val(1, 2, f)
        assert len(sm.get_val(1, 2)) == 1

        sm.clear()

        assert len(sm.get_val(1, 2)) == 0

    def test_get_strength_empty(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        assert sm.get_strength(0, 1) == 0

    def test_get_strength_one_feature(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature()
        sm.add_val(0, 0, f)
        assert sm.get_strength(0, 0) == 1

    def test_get_strength_multiple(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature()
        sm.add_val(1, 1, f)
        sm.add_val(1, 1, f)
        assert sm.get_strength(1, 1) == 2

    def test_get_strength_single_feature_bonus(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature("Single")
        sm.add_val(0, 0, f)
        # 1 (base) + 10 (Single bonus) = 11
        assert sm.get_strength(0, 0) == 11

    def test_get_strength_delta_feature_bonus(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature("Delta")
        sm.add_val(0, 0, f)
        # 1 (base) + 15 (Delta bonus) = 16
        assert sm.get_strength(0, 0) == 16

    def test_get_strength_motion_feature_bonus(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature("Motion")
        sm.add_val(0, 0, f)
        # 1 (base) + 20 (Motion bonus) = 21
        assert sm.get_strength(0, 0) == 21

    def test_get_max_strength(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        f = _make_feature()

        sm.add_val(0, 0, f)
        assert sm.get_max_strength() == 1

        sm.add_val(1, 1, f)
        sm.add_val(1, 1, f)
        assert sm.get_max_strength() == 2

        sm.add_val(2, 2, f)
        sm.add_val(2, 2, f)
        sm.add_val(2, 2, f)
        assert sm.get_max_strength() == 3

    def test_get_max_strength_empty(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        assert sm.get_max_strength() == 0

    def test_deepcopy(self) -> None:
        g = _make_grid()
        sm1 = SaliencyMap(g)
        f = _make_feature()
        sm1.add_val(0, 0, f)

        sm2 = deepcopy(sm1)

        assert sm1 is not sm2
        assert sm1.shape == sm2.shape
        assert sm2.grid is not None
        assert sm1.grid is not sm2.grid
        assert sm2.grid[0, 0] == 32
        assert sm2.grid[2, 2] == 99
        assert isinstance(sm2[0, 0], list)
        assert sm1[0, 0] is not sm2[0, 0]
        assert sm2[0, 0][0] is f

    def test_feature_report(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)

        f1 = _make_feature("Flood")
        f2 = _make_feature("Line")
        f3 = _make_feature("Line")

        sm.add_val(0, 0, f1)
        sm.add_val(1, 1, f2)
        sm.add_val(2, 2, f3)

        report = sm.feature_report()
        assert report["Flood"] == 1
        assert report["Line"] == 2

    def test_feature_report_empty(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        report = sm.feature_report()
        assert report == {}

    def test_feature_report_same_feature_multiple_cells(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)

        f = _make_feature("Flood")
        # Same feature object added to multiple cells
        sm.add_val(0, 0, f)
        sm.add_val(1, 1, f)

        report = sm.feature_report()
        # Same id(f) so it's counted as 1 unique feature
        assert report["Flood"] == 1

    def test_grid_reference(self) -> None:
        g = _make_grid()
        sm = SaliencyMap(g)
        assert sm.grid is g
