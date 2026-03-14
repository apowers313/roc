# mypy: disable-error-code="no-untyped-def"

"""Tests for roc/reporting/components/resolution_inspector.py."""

import panel as pn

from roc.reporting.components.resolution_inspector import ResolutionInspector
from roc.reporting.components.tokens import SUCCESS, ACCENT


class TestResolutionInspector:
    def test_none_shows_placeholder(self):
        inspector = ResolutionInspector(decision=None)
        panel = inspector.__panel__()
        html_panes = [c for c in panel if isinstance(c, pn.pane.HTML)]
        assert any("No resolution data" in p.object for p in html_panes)

    def test_match_outcome_shows_green(self):
        decision = {
            "event": "resolution_decision",
            "algorithm": "symmetric-difference",
            "outcome": "match",
            "tick": 42,
            "x": 10,
            "y": 5,
            "features": ["ShapeNode(-)", "ColorNode(GREY)"],
            "num_candidates": 3,
            "matched_object_id": 12345,
            "best_distance": 0.5,
            "candidate_distances": [("12345", 0.5), ("67890", 1.2)],
        }
        inspector = ResolutionInspector(decision=decision)
        panel = inspector.__panel__()
        html_panes = [c for c in panel if isinstance(c, pn.pane.HTML)]
        badge_html = html_panes[0].object if html_panes else ""
        assert SUCCESS in badge_html
        assert "MATCHED" in badge_html

    def test_new_object_outcome_shows_accent(self):
        decision = {
            "outcome": "new_object",
            "algorithm": "symmetric-difference",
            "features": [],
            "num_candidates": 0,
        }
        inspector = ResolutionInspector(decision=decision)
        panel = inspector.__panel__()
        html_panes = [c for c in panel if isinstance(c, pn.pane.HTML)]
        badge_html = html_panes[0].object if html_panes else ""
        assert ACCENT in badge_html
        assert "NEW OBJECT" in badge_html

    def test_candidates_table_rendered(self):
        decision = {
            "outcome": "match",
            "algorithm": "symmetric-difference",
            "features": [],
            "num_candidates": 2,
            "candidate_distances": [("obj1", 0.3), ("obj2", 1.5)],
        }
        inspector = ResolutionInspector(decision=decision)
        panel = inspector.__panel__()
        tabulators = [c for c in panel if isinstance(c, pn.widgets.Tabulator)]
        assert len(tabulators) >= 1
        # Should have the candidate data
        df = tabulators[-1].value
        assert len(df) == 2

    def test_features_listed(self):
        decision = {
            "outcome": "match",
            "algorithm": "dirichlet-categorical",
            "features": ["ShapeNode(-)", "ColorNode(GREY)", "SingleNode(42)"],
            "num_candidates": 1,
        }
        inspector = ResolutionInspector(decision=decision)
        panel = inspector.__panel__()
        html_panes = [c for c in panel if isinstance(c, pn.pane.HTML)]
        all_html = " ".join(p.object for p in html_panes if p.object)
        assert "ShapeNode" in all_html
        assert "ColorNode" in all_html

    def test_dirichlet_posteriors_rendered(self):
        decision = {
            "outcome": "match",
            "algorithm": "dirichlet-categorical",
            "features": [],
            "num_candidates": 2,
            "posteriors": [("obj1", 0.85), ("obj2", 0.15)],
            "vocab_size": 64,
            "total_objects_tracked": 10,
        }
        inspector = ResolutionInspector(decision=decision)
        panel = inspector.__panel__()
        tabulators = [c for c in panel if isinstance(c, pn.widgets.Tabulator)]
        # Should have posteriors table
        assert len(tabulators) >= 1

    def test_reactive_update(self):
        inspector = ResolutionInspector(decision=None)
        panel = inspector.__panel__()
        html_panes = [c for c in panel if isinstance(c, pn.pane.HTML)]
        assert any("No resolution data" in p.object for p in html_panes)

        inspector.decision = {"outcome": "match", "algorithm": "test", "features": []}
        html_panes = [c for c in panel if isinstance(c, pn.pane.HTML)]
        assert any("MATCHED" in p.object for p in html_panes)
