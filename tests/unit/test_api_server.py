"""Unit tests for the FastAPI dashboard API server."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app
from roc.reporting.run_store import StepData
from roc.reporting.step_buffer import StepBuffer


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _setup_data_dir(tmp_path: Path) -> Generator[None, None, None]:
    """Point the API at a temp directory and reset server state."""
    import roc.reporting.api_server as mod

    orig_dir = mod._data_dir
    orig_buf = mod._step_buffer
    orig_run = mod._live_run_name
    mod._data_dir = tmp_path
    mod._step_buffer = None
    mod._live_run_name = None
    mod._run_stores.clear()
    yield
    mod._data_dir = orig_dir
    mod._step_buffer = orig_buf
    mod._live_run_name = orig_run
    mod._run_stores.clear()


@pytest.fixture()
def live_buffer() -> StepBuffer:
    """Set up a StepBuffer with multi-game test data as the live run."""
    import roc.reporting.api_server as mod

    buf = StepBuffer(capacity=100)
    for i in range(1, 11):
        buf.push(StepData(step=i, game_number=1))
    for i in range(11, 16):
        buf.push(StepData(step=i, game_number=2))
    mod._step_buffer = buf
    mod._live_run_name = "test-live-run"
    return buf


class TestListRuns:
    def test_returns_empty_when_no_runs(self, client: TestClient) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_runs_when_present(self, client: TestClient, tmp_path: Path) -> None:
        # Create a mock run directory with the structure RunStore.list_runs expects
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        # RunStore.list_runs checks for DuckLakeStore.is_valid_run
        # which we need to mock
        with patch("roc.reporting.run_store.RunStore.list_runs", return_value=["test-run"]):
            with patch("roc.reporting.api_server._get_store") as mock_store:
                store = MagicMock()
                store.list_games.return_value = MagicMock(__len__=lambda s: 2)
                store.list_games.return_value.__len__ = lambda s: 2
                store.step_count.return_value = 100
                mock_store.return_value = store
                resp = client.get("/api/runs")
                assert resp.status_code == 200


class TestStepRange:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step-range")
        assert resp.status_code == 404


class TestGraphHistory:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/graph-history")
        assert resp.status_code == 404

    def test_returns_data_from_store(self, client: TestClient) -> None:
        mock_data = [
            {"step": 1, "node_count": 10, "node_max": 100, "edge_count": 20, "edge_max": 200},
        ]
        with patch("roc.reporting.api_server._get_store") as mock_store:
            store = MagicMock()
            store.get_graph_history.return_value = mock_data
            mock_store.return_value = store
            resp = client.get("/api/runs/test-run/graph-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            store.get_graph_history.assert_called_once_with(None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        with patch("roc.reporting.api_server._get_store") as mock_store:
            store = MagicMock()
            store.get_graph_history.return_value = []
            mock_store.return_value = store
            resp = client.get("/api/runs/test-run/graph-history?game=2")
            assert resp.status_code == 200
            store.get_graph_history.assert_called_once_with(2)


class TestEventHistory:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/event-history")
        assert resp.status_code == 404

    def test_returns_data_from_store(self, client: TestClient) -> None:
        mock_data = [
            {"step": 1, "roc.perception": 5, "roc.attention": 3},
        ]
        with patch("roc.reporting.api_server._get_store") as mock_store:
            store = MagicMock()
            store.get_event_history.return_value = mock_data
            mock_store.return_value = store
            resp = client.get("/api/runs/test-run/event-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            store.get_event_history.assert_called_once_with(None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        with patch("roc.reporting.api_server._get_store") as mock_store:
            store = MagicMock()
            store.get_event_history.return_value = []
            mock_store.return_value = store
            resp = client.get("/api/runs/test-run/event-history?game=1")
            assert resp.status_code == 200
            store.get_event_history.assert_called_once_with(1)


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
