# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/charts.py."""

import panel as pn

from roc.reporting.components.charts import event_bar_chart


class TestEventBarChart:
    def test_renders_dict_as_vega(self):
        data = {"perception_bus": 178, "attention_bus": 48, "action_bus": 1}
        result = event_bar_chart(data)
        assert isinstance(result, pn.pane.Vega)

    def test_none_returns_placeholder(self):
        result = event_bar_chart(None)
        assert isinstance(result, pn.pane.HTML)
        assert "No event data" in result.object

    def test_empty_dict_returns_placeholder(self):
        result = event_bar_chart({})
        assert isinstance(result, pn.pane.HTML)

    def test_skips_non_numeric_values(self):
        data = {"bus": "not_a_number", "real_bus": 42}
        result = event_bar_chart(data)
        assert isinstance(result, pn.pane.Vega)
        # Vega spec should have only the numeric value
        values = result.object["data"]["values"]
        assert len(values) == 1
        assert values[0]["bus"] == "real_bus"

    def test_skips_step_and_game_number(self):
        data = {"step": 5, "game_number": 1, "perception": 100}
        result = event_bar_chart(data)
        assert isinstance(result, pn.pane.Vega)
        values = result.object["data"]["values"]
        assert len(values) == 1
