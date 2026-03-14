"""ResolutionInspector -- composite Viewer for object resolution decisions."""

from __future__ import annotations

import html as html_mod
from typing import Any

import pandas as pd
import param
import panel as pn
from panel.viewable import Viewer

from roc.reporting.components.tables import compact_kv_table
from roc.reporting.components.tokens import (
    ACCENT,
    COMPACT_TABLE_CSS,
    FONT,
    SUCCESS,
    TEXT_DIM,
    TEXT_MUTED,
    WARNING,
    no_data_html,
)

_OUTCOME_COLORS: dict[str, str] = {
    "match": SUCCESS,
    "new_object": ACCENT,
    "low_confidence": WARNING,
}

_OUTCOME_LABELS: dict[str, str] = {
    "match": "MATCHED",
    "new_object": "NEW OBJECT",
    "low_confidence": "LOW CONFIDENCE",
}


class ResolutionInspector(Viewer):
    """Visualizes an object resolution decision.

    Shows the outcome badge, summary stats, candidate table, and feature list
    from a single ``roc.resolution.decision`` event dict.
    """

    decision = param.Dict(
        default=None,
        allow_None=True,
        doc="Resolution decision dict from roc.resolution.decision event",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._content = pn.Column(sizing_mode="stretch_width")
        self._render()

    @param.depends("decision", watch=True)
    def _render(self) -> None:
        self._content.clear()
        d = self.decision
        if d is None:
            self._content.append(
                pn.pane.HTML(no_data_html("resolution"), sizing_mode="stretch_width")
            )
            return

        # 1. Outcome badge
        outcome = d.get("outcome", "unknown")
        color = _OUTCOME_COLORS.get(outcome, TEXT_MUTED)
        label = _OUTCOME_LABELS.get(outcome, outcome.upper())
        self._content.append(
            pn.pane.HTML(
                f'<span style="color:{color};font-weight:700;font-size:12px;'
                f'font-family:{FONT};letter-spacing:0.5px;">{html_mod.escape(label)}</span>',
                sizing_mode="stretch_width",
            )
        )

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
        if "best_distance" in d:
            summary["distance"] = round(d["best_distance"], 4) if d["best_distance"] else None
        if "vocab_size" in d:
            summary["vocab_size"] = d["vocab_size"]
        if "total_objects_tracked" in d:
            summary["objects_tracked"] = d["total_objects_tracked"]

        if summary:
            self._content.append(compact_kv_table(summary, "resolution"))

        # 3. Candidates table
        candidates_df = self._build_candidates_df(d)
        if candidates_df is not None and len(candidates_df) > 0:
            cand_height = min(len(candidates_df) * 20 + 4, 200)
            self._content.append(
                pn.widgets.Tabulator(
                    candidates_df,
                    theme="simple",
                    theme_classes=["table-sm"],
                    show_index=False,
                    header_filters=False,
                    stylesheets=[COMPACT_TABLE_CSS],
                    sizing_mode="fixed",
                    width=320,
                    height=cand_height,
                    disabled=True,
                    pagination=None,
                )
            )

        # 4. Features
        features = d.get("features")
        if features and isinstance(features, list):
            feat_str = ", ".join(str(f) for f in features[:20])
            if len(features) > 20:
                feat_str += f" ... (+{len(features) - 20})"
            self._content.append(
                pn.pane.HTML(
                    f'<div style="font-size:10px;color:{TEXT_DIM};font-family:{FONT};'
                    f'padding:2px 0;">{html_mod.escape(feat_str)}</div>',
                    sizing_mode="stretch_width",
                )
            )

    @staticmethod
    def _build_candidates_df(d: dict[str, Any]) -> pd.DataFrame | None:
        """Build a DataFrame from candidate distances or posteriors."""
        # Symmetric difference: candidate_distances is [(obj_id, distance), ...]
        dists = d.get("candidate_distances")
        if dists and isinstance(dists, list):
            rows = [
                {"object": str(obj_id), "distance": round(float(dist), 4)} for obj_id, dist in dists
            ]
            if rows:
                return pd.DataFrame(rows)

        # Dirichlet: posteriors is [(obj_id, probability), ...]
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
        return self._content
