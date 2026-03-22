"""Unit tests for the FastAPI dashboard API server."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app
from roc.reporting.data_store import DataStore, RunSummary
from roc.reporting.run_store import StepData
from roc.reporting.step_buffer import StepBuffer


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _setup_data_store(tmp_path: Path) -> Generator[None, None, None]:
    """Point the API at a DataStore backed by a temp directory."""
    import roc.reporting.api_server as mod

    orig = mod._data_store
    mod._data_store = DataStore(tmp_path)
    yield
    mod._data_store = orig


@pytest.fixture()
def live_buffer() -> StepBuffer:
    """Set up a StepBuffer with multi-game test data as the live run."""
    import roc.reporting.api_server as mod

    assert mod._data_store is not None
    buf = StepBuffer(capacity=100)
    mod._data_store.set_live_session("test-live-run", buf)
    for i in range(1, 11):
        buf.push(StepData(step=i, game_number=1))
    for i in range(11, 16):
        buf.push(StepData(step=i, game_number=2))
    return buf


class TestListRuns:
    def test_returns_empty_when_no_runs(self, client: TestClient) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_runs_when_present(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "list_runs") as mock_list:
            mock_list.return_value = [RunSummary(name="test-run", games=2, steps=100)]
            resp = client.get("/api/runs")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["name"] == "test-run"
            assert data[0]["steps"] == 100


class TestStepRange:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step-range")
        assert resp.status_code == 404


class TestGraphHistory:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/graph-history")
        assert resp.status_code == 404

    def test_returns_data_from_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [
            {"step": 1, "node_count": 10, "node_max": 100, "edge_count": 20, "edge_max": 200},
        ]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_graph_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/graph-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_graph_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/graph-history?game=2")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 2)


class TestEventHistory:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/event-history")
        assert resp.status_code == 404

    def test_returns_data_from_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [
            {"step": 1, "roc.perception": 5, "roc.attention": 3},
        ]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_event_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/event-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_event_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/event-history?game=1")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 1)


class TestGetStep:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step/1")
        assert resp.status_code == 404


class TestBookmarks:
    def test_returns_empty_when_no_bookmarks(self, client: TestClient, tmp_path: Path) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        resp = client.get("/api/runs/test-run/bookmarks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_save_and_load_bookmarks(self, client: TestClient, tmp_path: Path) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        bookmarks = [
            {
                "step": 42,
                "game": 1,
                "annotation": "interesting",
                "created": "2026-03-16T00:00:00",
            }
        ]
        resp = client.post("/api/runs/test-run/bookmarks", json=bookmarks)
        assert resp.status_code == 200

        resp = client.get("/api/runs/test-run/bookmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["step"] == 42
        assert data[0]["annotation"] == "interesting"


# ---------------------------------------------------------------------------
# Regression tests for bugs found during dashboard development
# ---------------------------------------------------------------------------


class TestLiveGamesStepCounts:
    """Regression: games endpoint showed (0 steps) for all live games."""

    def test_games_include_step_counts(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/games")
        assert resp.status_code == 200
        games = resp.json()
        assert len(games) == 2
        assert games[0]["game_number"] == 1
        assert games[0]["steps"] == 10
        assert games[1]["game_number"] == 2
        assert games[1]["steps"] == 5


class TestLiveStepRangeGameFilter:
    """Regression: step-range endpoint ignored game filter for live run."""

    def test_global_range(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/step-range")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 15

    def test_game1_range(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/step-range?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 10

    def test_game2_range(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/step-range?game=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 11
        assert data["max"] == 15

    def test_nonexistent_game_returns_zeros(
        self, client: TestClient, live_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-live-run/step-range?game=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 0
        assert data["max"] == 0


class TestLiveStatus:
    def test_no_live_session(self, client: TestClient) -> None:
        resp = client.get("/api/live/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False
        assert data["run_name"] is None

    def test_active_live_session(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/live/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["run_name"] == "test-live-run"
        assert data["step"] == 15
        assert data["game_numbers"] == [1, 2]


class TestLiveStepEndpoint:
    def test_get_step_from_buffer(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/live/step/5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 5
        assert data["game_number"] == 1

    def test_step_not_in_buffer_returns_404(
        self, client: TestClient, live_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/live/step/999")
        assert resp.status_code == 404

    def test_no_live_session_returns_404(self, client: TestClient) -> None:
        resp = client.get("/api/live/step/1")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Regression: subprocess mode buffer was not consulted by REST endpoints,
# causing panels to show no data after browser refresh during a live game
# started via the Game Menu.
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_subprocess_buffer() -> StepBuffer:
    """Set up a StepBuffer in subprocess/HTTP callback mode."""
    import roc.reporting.api_server as mod

    assert mod._data_store is not None
    buf = StepBuffer(capacity=100)
    mod._data_store.set_live_session("test-subprocess-run", buf)
    for i in range(1, 11):
        buf.push(StepData(step=i, game_number=1))
    for i in range(11, 16):
        buf.push(StepData(step=i, game_number=2))
    return buf


class TestSubprocessBufferInRunsList:
    """Regression: live runs started via subprocess didn't appear in the
    runs dropdown because _get_run_summary tried DuckLake (locked by subprocess)
    and fell back to 0 steps, which was filtered by min_steps."""

    def test_live_run_appears_in_runs_list(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs?min_steps=0")
        assert resp.status_code == 200
        runs = resp.json()
        names = [r["name"] for r in runs]
        assert "test-subprocess-run" in names

    def test_live_run_has_correct_step_count(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs?min_steps=0")
        assert resp.status_code == 200
        runs = resp.json()
        live = next(r for r in runs if r["name"] == "test-subprocess-run")
        assert live["steps"] == 15
        assert live["games"] == 2

    def test_live_run_not_filtered_by_min_steps(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs?min_steps=10")
        assert resp.status_code == 200
        runs = resp.json()
        names = [r["name"] for r in runs]
        assert "test-subprocess-run" in names


class TestSubprocessBufferStepFetch:
    """Regression: GET /api/runs/{run}/step/{n} returned 404 for live runs
    started via subprocess because _get_step_data didn't check the live buffer."""

    def test_step_from_subprocess_buffer(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step/5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 5
        assert data["game_number"] == 1

    def test_step_not_in_subprocess_buffer_returns_404(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step/999")
        assert resp.status_code == 404


class TestSubprocessBufferGames:
    """Regression: GET /api/runs/{run}/games returned 404 for subprocess live runs."""

    def test_games_from_subprocess_buffer(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-subprocess-run/games")
        assert resp.status_code == 200
        games = resp.json()
        assert len(games) == 2
        assert games[0]["game_number"] == 1
        assert games[0]["steps"] == 10
        assert games[1]["game_number"] == 2
        assert games[1]["steps"] == 5


class TestSubprocessBufferStepRange:
    """Regression: GET /api/runs/{run}/step-range returned 404 for subprocess live runs."""

    def test_global_range(self, client: TestClient, live_subprocess_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step-range")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 15

    def test_game_filtered_range(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step-range?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 10


class TestSubprocessBufferLiveStatus:
    """Regression: /api/live/status didn't report subprocess buffer sessions."""

    def test_active_subprocess_session(
        self, client: TestClient, live_subprocess_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/live/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["run_name"] == "test-subprocess-run"
        assert data["step"] == 15
        assert data["game_numbers"] == [1, 2]


# ---------------------------------------------------------------------------
# Live history endpoints -- previously broken in subprocess mode because
# they went straight to DuckLake which was file-locked by the subprocess.
# Now served from DataStore's in-memory indices.
# ---------------------------------------------------------------------------


class TestLiveHistoryEndpoints:
    """Test that history endpoints return data from the live buffer."""

    @pytest.fixture()
    def live_history_buffer(self) -> StepBuffer:
        """Set up a buffer with history-relevant StepData fields."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("test-history-run", buf)
        buf.push(
            StepData(
                step=1,
                game_number=1,
                graph_summary={"node_count": 10, "edge_count": 5},
                event_summary=[{"roc.perception": 3}],
                intrinsics={"hp": 16, "max_hp": 16},
                game_metrics={"score": 0, "level": 1},
                action_taken={"action_id": 7, "action_name": "north"},
                resolution_metrics={
                    "outcome": "new_object",
                    "features": ["ShapeNode(@)", "ColorNode(white)"],
                    "new_object_id": 42,
                },
            )
        )
        buf.push(
            StepData(
                step=2,
                game_number=1,
                graph_summary={"node_count": 12, "edge_count": 6},
                game_metrics={"score": 1, "level": 1},
                resolution_metrics={
                    "outcome": "match",
                    "matched_object_id": 42,
                    "matched_attrs": {"char": "@", "color": "white", "glyph": "64"},
                    "features": ["ShapeNode(@)", "ColorNode(white)", "SingleNode(64)"],
                },
            )
        )
        return buf

    def test_graph_history(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/graph-history?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["step"] == 1
        assert data[0]["node_count"] == 10

    def test_event_history(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/event-history?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["roc.perception"] == 3

    def test_intrinsics_history(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/intrinsics-history?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["hp"] == 16

    def test_metrics_history(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/metrics-history?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["score"] == 0
        assert data[1]["score"] == 1

    def test_metrics_history_field_filter(
        self, client: TestClient, live_history_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-history-run/metrics-history?game=1&fields=score")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert "score" in data[0]
        assert "level" not in data[0]

    def test_action_history(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/action-history?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["action_id"] == 7

    def test_resolution_history(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/resolution-history?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["outcome"] == "new_object"
        assert data[1]["outcome"] == "match"
        assert data[1]["correct"] is True

    def test_all_objects(self, client: TestClient, live_history_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-history-run/all-objects?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["node_id"] == "42"
        assert data[0]["match_count"] == 1
        assert data[0]["shape"] == "@"
