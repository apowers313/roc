"""Unit tests for the DataStore class."""

from __future__ import annotations

from pathlib import Path

import pytest

from roc.reporting.data_store import DataStore
from roc.reporting.run_store import StepData
from roc.reporting.step_buffer import StepBuffer


class TestIndexStep:
    """Test that _on_step_pushed correctly accumulates history indices."""

    def test_graph_history_accumulated(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                graph_summary={"node_count": 10, "node_max": 100, "edge_count": 5, "edge_max": 50},
            )
        )
        buf.push(
            StepData(
                step=2,
                game_number=1,
                graph_summary={"node_count": 12, "node_max": 100, "edge_count": 6, "edge_max": 50},
            )
        )

        history = ds.get_graph_history("test-run", game=1)
        assert len(history) == 2
        assert history[0]["step"] == 1
        assert history[0]["node_count"] == 10
        assert history[1]["step"] == 2
        assert history[1]["node_count"] == 12

    def test_event_history_accumulated(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                event_summary=[{"roc.perception": 3, "roc.attention": 2}],
            )
        )

        history = ds.get_event_history("test-run", game=1)
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["roc.perception"] == 3

    def test_intrinsics_history_accumulated(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(StepData(step=1, game_number=1, intrinsics={"hp": 10, "max_hp": 16}))

        history = ds.get_intrinsics_history("test-run", game=1)
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["hp"] == 10

    def test_metrics_history_accumulated(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(StepData(step=1, game_number=1, game_metrics={"score": 0, "level": 1}))

        history = ds.get_metrics_history("test-run", game=1)
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["score"] == 0

    def test_metrics_history_field_filter(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(StepData(step=1, game_number=1, game_metrics={"score": 0, "level": 1, "gold": 5}))

        history = ds.get_metrics_history("test-run", game=1, fields=["score", "gold"])
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["score"] == 0
        assert history[0]["gold"] == 5
        assert "level" not in history[0]

    def test_action_history_accumulated(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(step=1, game_number=1, action_taken={"action_id": 3, "action_name": "north"})
        )

        history = ds.get_action_history("test-run", game=1)
        assert len(history) == 1
        assert history[0]["action_id"] == 3

    def test_resolution_events_accumulated(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "match",
                    "matched_object_id": 42,
                    "matched_attrs": {"char": "@", "color": "white", "glyph": "64"},
                    "features": ["ShapeNode(@)", "ColorNode(white)", "SingleNode(64)"],
                },
            )
        )

        history = ds.get_resolution_history("test-run", game=1)
        assert len(history) == 1
        assert history[0]["step"] == 1
        assert history[0]["outcome"] == "match"
        assert history[0]["correct"] is True

    def test_null_fields_not_indexed(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(StepData(step=1, game_number=1))

        assert ds.get_graph_history("test-run", game=1) == []
        assert ds.get_event_history("test-run", game=1) == []
        assert ds.get_intrinsics_history("test-run", game=1) == []
        assert ds.get_metrics_history("test-run", game=1) == []
        assert ds.get_action_history("test-run", game=1) == []
        assert ds.get_resolution_history("test-run", game=1) == []

    def test_multi_game_indexing(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(StepData(step=1, game_number=1, graph_summary={"node_count": 1}))
        buf.push(StepData(step=2, game_number=2, graph_summary={"node_count": 2}))

        h1 = ds.get_graph_history("test-run", game=1)
        h2 = ds.get_graph_history("test-run", game=2)
        h_all = ds.get_graph_history("test-run", game=None)

        assert len(h1) == 1
        assert h1[0]["node_count"] == 1
        assert len(h2) == 1
        assert h2[0]["node_count"] == 2
        assert len(h_all) == 2


class TestSessionLifecycle:
    def test_set_and_clear_live_session(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        assert ds.live_run_name == "test-run"
        assert ds.live_buffer is buf

        ds.clear_live_session()

        name_after: str | None = ds.live_run_name
        buf_after: StepBuffer | None = ds.live_buffer
        assert name_after is None
        assert buf_after is None

    def test_clear_session_clears_indices(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)
        buf.push(StepData(step=1, game_number=1, graph_summary={"node_count": 1}))

        assert len(ds.get_graph_history("test-run", game=1)) == 1

        ds.clear_live_session()

        # After clearing, querying uses RunStore path (would fail for non-existent dir)
        with pytest.raises(FileNotFoundError):
            ds.get_graph_history("test-run", game=1)


class TestResolutionHistory:
    def test_match_correctness_true(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "match",
                    "matched_attrs": {"char": "@", "color": "white", "glyph": "64"},
                    "features": ["ShapeNode(@)", "ColorNode(white)", "SingleNode(64)"],
                },
            )
        )

        history = ds.get_resolution_history("test-run", game=1)
        assert history[0]["correct"] is True

    def test_match_correctness_false(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "match",
                    "matched_attrs": {"char": "d", "color": "red", "glyph": "100"},
                    "features": ["ShapeNode(@)", "ColorNode(white)", "SingleNode(64)"],
                },
            )
        )

        history = ds.get_resolution_history("test-run", game=1)
        assert history[0]["correct"] is False

    def test_match_correctness_none_when_no_attrs(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={"outcome": "match"},
            )
        )

        history = ds.get_resolution_history("test-run", game=1)
        assert history[0]["correct"] is None

    def test_new_object_has_no_correct_field(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={"outcome": "new_object", "features": []},
            )
        )

        history = ds.get_resolution_history("test-run", game=1)
        assert "correct" not in history[0]


class TestAllObjects:
    def test_new_object_with_id(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "new_object",
                    "features": ["ShapeNode(@)", "ColorNode(white)", "SingleNode(64)"],
                    "new_object_id": 100,
                },
            )
        )

        objects = ds.get_all_objects("test-run", game=1)
        assert len(objects) == 1
        assert objects[0]["node_id"] == "100"
        assert objects[0]["shape"] == "@"
        assert objects[0]["color"] == "white"
        assert objects[0]["glyph"] == "64"
        assert objects[0]["step_added"] == 1
        assert objects[0]["match_count"] == 0

    def test_new_object_without_id(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "new_object",
                    "features": ["ShapeNode(d)"],
                },
            )
        )

        objects = ds.get_all_objects("test-run", game=1)
        assert len(objects) == 1
        assert objects[0]["node_id"] is None
        assert objects[0]["shape"] == "d"

    def test_match_increments_count(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "new_object",
                    "features": ["ShapeNode(@)"],
                    "new_object_id": 42,
                },
            )
        )
        buf.push(
            StepData(
                step=2,
                game_number=1,
                resolution_metrics={
                    "outcome": "match",
                    "matched_object_id": 42,
                    "matched_attrs": {"char": "@"},
                    "features": ["ShapeNode(@)"],
                },
            )
        )
        buf.push(
            StepData(
                step=3,
                game_number=1,
                resolution_metrics={
                    "outcome": "match",
                    "matched_object_id": 42,
                    "matched_attrs": {"char": "@"},
                    "features": ["ShapeNode(@)"],
                },
            )
        )

        objects = ds.get_all_objects("test-run", game=1)
        assert len(objects) == 1
        assert objects[0]["match_count"] == 2

    def test_match_creates_object_if_not_seen(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        buf.push(
            StepData(
                step=1,
                game_number=1,
                resolution_metrics={
                    "outcome": "match",
                    "matched_object_id": 99,
                    "matched_attrs": {"char": "d", "color": "red", "glyph": "100"},
                    "features": ["ShapeNode(d)", "ColorNode(red)"],
                },
            )
        )

        objects = ds.get_all_objects("test-run", game=1)
        assert len(objects) == 1
        assert objects[0]["node_id"] == "99"
        assert objects[0]["match_count"] == 1
        assert objects[0]["step_added"] is None


class TestLiveStatus:
    def test_no_live_session(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        status = ds.get_live_status()
        assert status["active"] is False

    def test_active_with_data(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)
        buf.push(StepData(step=1, game_number=1))

        status = ds.get_live_status()
        assert status["active"] is True
        assert status["run_name"] == "test-run"
        assert status["step"] == 1

    def test_session_without_data(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        status = ds.get_live_status()
        assert status["active"] is False


class TestPushLiveStep:
    def test_push_indexes_step(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)

        ds.push_live_step(StepData(step=1, game_number=1, graph_summary={"node_count": 5}))

        assert len(buf) == 1
        history = ds.get_graph_history("test-run", game=1)
        assert len(history) == 1
        assert history[0]["node_count"] == 5


class TestHistoricalFallback:
    def test_non_live_run_raises_for_missing_dir(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            ds.get_graph_history("nonexistent-run", game=1)

    def test_list_runs_empty_dir(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        assert ds.list_runs(min_steps=0) == []

    def test_get_live_step_no_session(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        assert ds.get_live_step(1) is None

    def test_get_live_step_from_buffer(self, tmp_path: Path) -> None:
        ds = DataStore(tmp_path)
        buf = StepBuffer(capacity=100)
        ds.set_live_session("test-run", buf)
        buf.push(StepData(step=5, game_number=1))

        result = ds.get_live_step(5)
        assert result is not None
        assert result.step == 5
