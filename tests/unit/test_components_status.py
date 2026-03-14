# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/status_bar.py."""

import panel as pn

from roc.reporting.components.status_bar import compact_status_bar
from roc.reporting.components.tokens import SUCCESS, ERROR, WARNING


class TestCompactStatusBar:
    def test_with_metrics_shows_hp(self):
        result = compact_status_bar({"hp": 16, "hp_max": 16, "score": 42})
        assert isinstance(result, pn.pane.HTML)
        assert "HP" in result.object
        assert "16/16" in result.object

    def test_hp_green_when_healthy(self):
        result = compact_status_bar({"hp": 16, "hp_max": 16, "score": 0})
        assert SUCCESS in result.object

    def test_hp_red_when_critical(self):
        result = compact_status_bar({"hp": 2, "hp_max": 16, "score": 0})
        assert ERROR in result.object

    def test_hp_yellow_when_low(self):
        result = compact_status_bar({"hp": 6, "hp_max": 16, "score": 0})
        assert WARNING in result.object

    def test_no_metrics_shows_step(self):
        result = compact_status_bar(None, step=42, game_number=2)
        assert isinstance(result, pn.pane.HTML)
        assert "Step" in result.object
        assert "42" in result.object
        assert "Game" in result.object
