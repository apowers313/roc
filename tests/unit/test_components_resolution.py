# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/resolution_inspector.py."""

import panel as pn

from roc.reporting.components.resolution_inspector import ResolutionInspector


class TestResolutionInspector:
    def test_none_shows_placeholder(self):
        inspector = ResolutionInspector(decision=None)
        assert inspector._outcome_pane.object == "No resolution data"
        assert inspector._summary_table.visible is False
        assert inspector._candidates_table.visible is False

    def test_match_outcome(self):
        decision = {
            "outcome": "match",
            "tick": 42,
            "x": 10,
            "y": 5,
            "features": ["ShapeNode(-)", "ColorNode(GREY)"],
            "num_candidates": 3,
            "matched_object_id": 12345,
            "posteriors": [("12345", 0.85), ("67890", 0.15)],
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._outcome_pane.object == "MATCHED"
        assert inspector._summary_table.visible is True

    def test_new_object_outcome(self):
        decision = {
            "outcome": "new_object",
            "features": [],
            "num_candidates": 0,
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._outcome_pane.object == "NEW OBJECT"

    def test_low_confidence_outcome(self):
        decision = {
            "outcome": "low_confidence",
            "features": [],
            "num_candidates": 1,
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._outcome_pane.object == "LOW CONFIDENCE"

    def test_candidates_table_with_posteriors(self):
        decision = {
            "outcome": "match",
            "features": [],
            "num_candidates": 2,
            "posteriors": [("obj1", 0.85), ("obj2", 0.15)],
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._candidates_table.visible is True
        df = inspector._candidates_table.value
        assert len(df) == 2
        assert "probability" in df.columns

    def test_candidates_table_with_distances(self):
        decision = {
            "outcome": "match",
            "features": [],
            "num_candidates": 2,
            "candidate_distances": [("obj1", 0.3), ("obj2", 1.5)],
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._candidates_table.visible is True
        df = inspector._candidates_table.value
        assert len(df) == 2
        assert "distance" in df.columns

    def test_features_displayed(self):
        decision = {
            "outcome": "match",
            "features": ["ShapeNode(-)", "ColorNode(GREY)", "SingleNode(42)"],
            "num_candidates": 1,
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._features_pane.visible is True
        assert "ShapeNode" in inspector._features_pane.object
        assert "ColorNode" in inspector._features_pane.object

    def test_summary_table_populated(self):
        decision = {
            "outcome": "match",
            "tick": 42,
            "x": 10,
            "y": 5,
            "num_candidates": 3,
            "vocab_size": 64,
            "total_objects_tracked": 10,
            "features": [],
        }
        inspector = ResolutionInspector(decision=decision)
        assert inspector._summary_table.visible is True
        df = inspector._summary_table.value
        keys = list(df["key"])
        assert "location" in keys
        assert "tick" in keys
        assert "candidates" in keys

    def test_reactive_update(self):
        inspector = ResolutionInspector(decision=None)
        assert inspector._outcome_pane.object == "No resolution data"

        inspector.decision = {"outcome": "match", "features": []}
        assert inspector._outcome_pane.object == "MATCHED"

    def test_panel_returns_column(self):
        inspector = ResolutionInspector(decision=None)
        panel = inspector.__panel__()
        assert isinstance(panel, pn.Column)
