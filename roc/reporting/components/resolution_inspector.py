"""ResolutionInspector -- composite Viewer for object resolution decisions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import param
import panel as pn
from panel.viewable import Viewer

from roc.reporting.components.theme import COMPACT_CELL_CSS


class ResolutionInspector(Viewer):
    """Visualizes an object resolution decision.

    Shows the outcome badge, summary stats, candidate table, and feature list
    from a single ``roc.resolution.decision`` event dict.

    All sub-components are created once and updated in place to avoid flicker.
    """

    decision = param.Dict(
        default=None,
        allow_None=True,
        doc="Resolution decision dict from roc.resolution.decision event",
    )

    _OUTCOME_LABELS: dict[str, str] = {
        "match": "MATCHED",
        "new_object": "NEW OBJECT",
        "low_confidence": "LOW CONFIDENCE",
    }

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        # Persistent components -- updated in place
        self._outcome_pane = pn.pane.Str("No resolution data", sizing_mode="stretch_width")
        self._summary_table = pn.widgets.Tabulator(
            pd.DataFrame({"key": ["--"], "value": ["--"]}),
            theme="fast",
            show_index=False,
            header_filters=False,
            configuration={"headerVisible": False},
            stylesheets=[COMPACT_CELL_CSS],
            sizing_mode="stretch_width",
            disabled=True,
            pagination=None,
            visible=False,
        )
        self._candidates_table = pn.widgets.Tabulator(
            pd.DataFrame({"object": [], "probability": []}),
            theme="fast",
            show_index=False,
            header_filters=False,
            stylesheets=[COMPACT_CELL_CSS],
            sizing_mode="stretch_width",
            height=150,
            disabled=True,
            pagination=None,
            visible=False,
        )
        self._features_pane = pn.pane.Str("", sizing_mode="stretch_width", visible=False)

        self._render()

    @param.depends("decision", watch=True)
    def _render(self) -> None:
        d = self.decision
        if d is None:
            self._outcome_pane.object = "No resolution data"
            self._summary_table.visible = False
            self._candidates_table.visible = False
            self._features_pane.visible = False
            return

        # 1. Outcome badge
        outcome = d.get("outcome", "unknown")
        label = self._OUTCOME_LABELS.get(outcome, outcome.upper())
        self._outcome_pane.object = label

        # 2. Summary table
        summary: dict[str, Any] = {}
        if "algorithm" in d:
            summary["algorithm"] = d["algorithm"]
        if "x" in d and "y" in d:
            summary["location"] = f"({d['x']}, {d['y']})"
        if "tick" in d:
            summary["tick"] = d["tick"]
        if "num_candidates" in d:
            summary["candidates"] = d["num_candidates"]
        if "matched_object_id" in d:
            summary["matched"] = d["matched_object_id"]
        if "vocab_size" in d:
            summary["vocab_size"] = d["vocab_size"]
        if "total_objects_tracked" in d:
            summary["objects_tracked"] = d["total_objects_tracked"]

        if summary:
            rows = [{"key": str(k), "value": str(v)} for k, v in summary.items()]
            self._summary_table.value = pd.DataFrame(rows)
            self._summary_table.visible = True
        else:
            self._summary_table.visible = False

        # 3. Candidates table (posteriors or distances)
        candidates_df = self._build_candidates_df(d)
        if candidates_df is not None and len(candidates_df) > 0:
            self._candidates_table.value = candidates_df
            self._candidates_table.height = min(len(candidates_df) * 25 + 30, 150)
            self._candidates_table.visible = True
        else:
            self._candidates_table.visible = False

        # 4. Features
        features = d.get("features")
        if features and isinstance(features, list):
            feat_str = ", ".join(str(f) for f in features[:20])
            if len(features) > 20:
                feat_str += f" ... (+{len(features) - 20})"
            self._features_pane.object = feat_str
            self._features_pane.visible = True
        else:
            self._features_pane.visible = False

    @staticmethod
    def _build_candidates_df(d: dict[str, Any]) -> pd.DataFrame | None:
        """Build a DataFrame from candidate distances or posteriors."""
        dists = d.get("candidate_distances")
        if dists and isinstance(dists, list):
            rows = [
                {"object": str(obj_id), "distance": round(float(dist), 4)}
                for obj_id, dist in dists
            ]
            if rows:
                return pd.DataFrame(rows)

        posts = d.get("posteriors")
        if posts and isinstance(posts, list):
            rows = [
                {"object": str(obj_id), "probability": round(float(prob), 6)}
                for obj_id, prob in posts
            ]
            if rows:
                return pd.DataFrame(rows)

        return None

    def __panel__(self) -> pn.Column:
        return pn.Column(
            self._outcome_pane,
            self._summary_table,
            pn.pane.Markdown("**Candidates**"),
            self._candidates_table,
            pn.pane.Markdown("**Features**"),
            self._features_pane,
            sizing_mode="stretch_width",
        )
