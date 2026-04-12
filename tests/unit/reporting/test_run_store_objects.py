# mypy: disable-error-code="no-untyped-def"

"""Object identity invariance regression tests (BUG-H4).

Architectural Invariant #9 says PHYSICAL features (shape, glyph, color) are
object identity invariants. They must NOT change based on which game's events
the dashboard happens to be filtering by.

The bug: ``RunStore.get_all_objects(game_number=N)`` rebuilt objects by
replaying ONLY game N's resolution events. If an object was first observed
in game 1 (full PHYSICAL features captured) and then matched in game 2,
the game-2-filtered query would not see the original ``new_object`` event
and would synthesize a placeholder entry from the match's ``matched_attrs``
(which only contains whatever the resolution algorithm compared on, not
the canonical features). The same node_id would therefore have different
``shape``/``glyph``/``color`` depending on the game filter.

The interim fix: canonical PHYSICAL features come from an UNFILTERED events
query; ``match_count`` is the only thing that varies by game filter.

These tests pin the new contract:

* Same node_id observed in two games yields the same shape/glyph/color
  regardless of game filter.
* match_count is per-game when filtering, total when unfiltered.
* step_added stays at the canonical first-observation step.
* Objects with no events in the filtered game are excluded from the result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from helpers.otel import make_log_record
from opentelemetry._logs import LogRecord

from roc.framework.clock import Clock
from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.parquet_exporter import ParquetExporter
from roc.reporting.run_store import RunStore


@pytest.fixture(autouse=True)
def _default_clock_tick() -> Generator[None, None, None]:
    Clock.set(1)
    yield
    Clock.reset()


def _new_object_event(features: list[str]) -> LogRecord:
    return make_log_record(
        event_name="roc.resolution.decision",
        body=json.dumps({"outcome": "new_object", "features": features}),
    )


def _new_object_id_event(node_id: int) -> LogRecord:
    return make_log_record(
        event_name="roc.resolution.new_object_id",
        body=json.dumps({"new_object_id": node_id}),
    )


def _match_event(node_id: int, char: str = "@", glyph: int = 333) -> LogRecord:
    return make_log_record(
        event_name="roc.resolution.decision",
        body=json.dumps(
            {
                "outcome": "match",
                "matched_object_id": node_id,
                "features": [f"ShapeNode({char})"],
                "matched_attrs": {"char": char, "glyph": glyph},
            }
        ),
    )


def _screen_event() -> LogRecord:
    return make_log_record(event_name="roc.screen", body='{"chars":[]}')


class TestObjectIdentityInvariance:
    """An object's PHYSICAL features must not change based on game filter."""

    def test_physical_features_invariant_across_game_filters(self, tmp_path: Path) -> None:
        """Object first observed in game 1 with full canonical features must
        report the same features when queried with ``game_number=2``."""
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)

        # Game 1: new object with FULL features
        Clock.set(1)
        exporter.export([make_log_record(event_name="roc.game_start", body="g1")])
        exporter.export([_screen_event()])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.resolution.decision",
                    body=json.dumps(
                        {
                            "outcome": "new_object",
                            "features": [
                                "ShapeNode(@)",
                                "ColorNode(white)",
                                "SingleNode(333)",
                            ],
                        }
                    ),
                )
            ]
        )
        exporter.export([_new_object_id_event(87)])

        # Game 2: same object matched -- with PARTIAL matched_attrs
        # (no color, no glyph). This is the bug-trigger: the old code
        # would synthesize a placeholder with shape='@' but color=None,
        # glyph=None.
        Clock.set(2)
        exporter.export([make_log_record(event_name="roc.game_start", body="g2")])
        exporter.export([_screen_event()])
        exporter.export(
            [
                make_log_record(
                    event_name="roc.resolution.decision",
                    body=json.dumps(
                        {
                            "outcome": "match",
                            "matched_object_id": 87,
                            "features": ["ShapeNode(@)"],
                            "matched_attrs": {"char": "@"},  # no color, no glyph
                        }
                    ),
                )
            ]
        )

        exporter.shutdown()
        store = RunStore(dl_store)

        # Unfiltered: canonical features
        all_objs = store.get_all_objects()
        all_by_id = {o["node_id"]: o for o in all_objs}
        assert "87" in all_by_id
        canonical = all_by_id["87"]
        assert canonical["shape"] == "@"
        assert canonical["color"] == "white"
        assert canonical["glyph"] == "333"

        # Filtered to game 2: SAME canonical features. The bug used to
        # produce shape='@' but color=None, glyph=None.
        g2_objs = store.get_all_objects(game_number=2)
        g2_by_id = {o["node_id"]: o for o in g2_objs}
        assert "87" in g2_by_id
        assert g2_by_id["87"]["shape"] == "@"
        assert g2_by_id["87"]["color"] == "white"
        assert g2_by_id["87"]["glyph"] == "333"

        # Filtered to game 1: SAME canonical features.
        g1_objs = store.get_all_objects(game_number=1)
        g1_by_id = {o["node_id"]: o for o in g1_objs}
        assert g1_by_id["87"]["shape"] == "@"
        assert g1_by_id["87"]["color"] == "white"
        assert g1_by_id["87"]["glyph"] == "333"


class TestMatchCountFilteredByGame:
    """match_count must reflect the game filter; identity must not."""

    def test_match_count_total_when_unfiltered(self, tmp_path: Path) -> None:
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)

        Clock.set(1)
        exporter.export([make_log_record(event_name="roc.game_start", body="g1")])
        exporter.export([_screen_event()])
        exporter.export(
            [_new_object_event(["ShapeNode(@)", "ColorNode(white)", "SingleNode(333)"])]
        )
        exporter.export([_new_object_id_event(50)])

        # 3 matches in game 1
        for tick in range(2, 5):
            Clock.set(tick)
            exporter.export([_screen_event()])
            exporter.export([_match_event(50)])

        # 2 matches in game 2
        Clock.set(5)
        exporter.export([make_log_record(event_name="roc.game_start", body="g2")])
        for tick in range(5, 7):
            Clock.set(tick)
            exporter.export([_screen_event()])
            exporter.export([_match_event(50)])

        exporter.shutdown()
        store = RunStore(dl_store)

        all_objs = store.get_all_objects()
        all_by_id = {o["node_id"]: o for o in all_objs}
        assert all_by_id["50"]["match_count"] == 5  # 3 + 2

    def test_match_count_per_game_when_filtered(self, tmp_path: Path) -> None:
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)

        Clock.set(1)
        exporter.export([make_log_record(event_name="roc.game_start", body="g1")])
        exporter.export([_screen_event()])
        exporter.export(
            [_new_object_event(["ShapeNode(@)", "ColorNode(white)", "SingleNode(333)"])]
        )
        exporter.export([_new_object_id_event(50)])

        for tick in range(2, 5):  # 3 matches in game 1
            Clock.set(tick)
            exporter.export([_screen_event()])
            exporter.export([_match_event(50)])

        Clock.set(5)
        exporter.export([make_log_record(event_name="roc.game_start", body="g2")])
        for tick in range(5, 7):  # 2 matches in game 2
            Clock.set(tick)
            exporter.export([_screen_event()])
            exporter.export([_match_event(50)])

        exporter.shutdown()
        store = RunStore(dl_store)

        g1_objs = {o["node_id"]: o for o in store.get_all_objects(game_number=1)}
        g2_objs = {o["node_id"]: o for o in store.get_all_objects(game_number=2)}

        # Game 1: the new_object event itself + 3 matches = match_count=3
        # (new_object outcome doesn't increment match_count)
        assert g1_objs["50"]["match_count"] == 3
        # Game 2: 2 matches
        assert g2_objs["50"]["match_count"] == 2


class TestStepAddedCanonical:
    """step_added must be the canonical first-observation step regardless of filter."""

    def test_step_added_is_canonical_step_when_filtering_later_game(self, tmp_path: Path) -> None:
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)

        Clock.set(1)
        exporter.export([make_log_record(event_name="roc.game_start", body="g1")])
        exporter.export([_screen_event()])
        exporter.export([_new_object_event(["ShapeNode(@)", "SingleNode(333)"])])
        exporter.export([_new_object_id_event(99)])

        Clock.set(5)
        exporter.export([make_log_record(event_name="roc.game_start", body="g2")])
        exporter.export([_screen_event()])
        exporter.export([_match_event(99)])

        exporter.shutdown()
        store = RunStore(dl_store)

        # Game 2 filter: object exists, step_added points back to game 1's
        # canonical step (1), NOT None (which was the BUG-M2 placeholder).
        g2_objs = {o["node_id"]: o for o in store.get_all_objects(game_number=2)}
        assert "99" in g2_objs
        assert g2_objs["99"]["step_added"] == 1


class TestExclusionWhenGameHasNoEvents:
    """Objects with zero events in the filtered game must be excluded."""

    def test_object_only_in_game_1_excluded_from_game_2_filter(self, tmp_path: Path) -> None:
        dl_store = DuckLakeStore(tmp_path)
        exporter = ParquetExporter(store=dl_store, background=False)

        Clock.set(1)
        exporter.export([make_log_record(event_name="roc.game_start", body="g1")])
        exporter.export([_screen_event()])
        exporter.export([_new_object_event(["ShapeNode(@)"])])
        exporter.export([_new_object_id_event(11)])

        # Game 2 has its OWN distinct object
        Clock.set(2)
        exporter.export([make_log_record(event_name="roc.game_start", body="g2")])
        exporter.export([_screen_event()])
        exporter.export([_new_object_event(["ShapeNode(d)"])])
        exporter.export([_new_object_id_event(22)])

        exporter.shutdown()
        store = RunStore(dl_store)

        # Game 2 filter: only object 22 should appear, not object 11
        g2_objs = {o["node_id"]: o for o in store.get_all_objects(game_number=2)}
        assert "22" in g2_objs
        assert "11" not in g2_objs

        # Unfiltered: both visible
        all_objs = {o["node_id"]: o for o in store.get_all_objects()}
        assert "11" in all_objs
        assert "22" in all_objs
