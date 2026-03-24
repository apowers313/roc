"""Unit tests for the FastAPI dashboard API server."""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
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


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestConvertNumpy:
    """Test _convert_numpy with various numpy types."""

    def test_converts_numpy_integer(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        result = _convert_numpy(np.int64(42))
        assert result == 42
        assert type(result) is int

    def test_converts_numpy_floating(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        result = _convert_numpy(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert type(result) is float

    def test_converts_numpy_ndarray(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        result = _convert_numpy(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_converts_numpy_bool(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        result = _convert_numpy(np.bool_(True))
        assert result is True
        assert type(result) is bool

    def test_converts_nested_dict_with_numpy(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        data = {"count": np.int64(5), "values": [np.float64(1.0), np.float64(2.0)]}
        result = _convert_numpy(data)
        assert result == {"count": 5, "values": [1.0, 2.0]}
        assert type(result["count"]) is int

    def test_passes_through_plain_python_types(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        assert _convert_numpy("hello") == "hello"
        assert _convert_numpy(42) == 42
        assert _convert_numpy(None) is None


class TestGetStepWithData:
    """Test GET /api/runs/{run}/step/{step} returning actual data."""

    def test_returns_step_data(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/step/3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 3
        assert data["game_number"] == 1

    def test_returns_step_with_numpy_values(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("numpy-run", buf)
        buf.push(
            StepData(
                step=1,
                game_number=1,
                game_metrics={"score": np.int64(100), "ratio": np.float64(0.5)},
            )
        )
        resp = client.get("/api/runs/numpy-run/step/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_metrics"]["score"] == 100
        assert type(data["game_metrics"]["score"]) is int


class TestGetStepsBatch:
    """Test GET /api/runs/{run}/steps?steps=1,2,3 batch endpoint."""

    def test_returns_batch_data(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/steps?steps=1,3,5")
        assert resp.status_code == 200
        data = resp.json()
        assert "1" in data
        assert "3" in data
        assert "5" in data
        assert data["1"]["step"] == 1
        assert data["5"]["game_number"] == 1

    def test_missing_steps_are_skipped(self, client: TestClient, live_buffer: StepBuffer) -> None:
        resp = client.get("/api/runs/test-live-run/steps?steps=1,999")
        assert resp.status_code == 200
        data = resp.json()
        assert "1" in data
        assert "999" not in data

    def test_returns_503_when_no_data_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            resp = client.get("/api/runs/test-run/steps?steps=1,2")
            assert resp.status_code == 503
        finally:
            mod._data_store = orig


class TestGetStepRangeWithData:
    """Test GET /api/runs/{run}/step-range returning actual data (not 404)."""

    def test_returns_step_range_from_live(
        self, client: TestClient, live_buffer: StepBuffer
    ) -> None:
        resp = client.get("/api/runs/test-live-run/step-range")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 15

    def test_returns_503_when_no_data_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            resp = client.get("/api/runs/test-run/step-range")
            assert resp.status_code == 503
        finally:
            mod._data_store = orig


class TestMetricsHistoryStandalone:
    """Test GET /api/runs/{run}/metrics-history standalone."""

    def test_returns_metrics_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "score": 10}, {"step": 2, "score": 20}]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_metrics_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/metrics-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None, None)

    def test_passes_game_and_fields(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_metrics_history") as mock_method:
            mock_method.return_value = [{"step": 1, "score": 10}]
            resp = client.get("/api/runs/test-run/metrics-history?game=1&fields=score")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 1, ["score"])

    def test_returns_503_when_no_data_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            resp = client.get("/api/runs/test-run/metrics-history")
            assert resp.status_code == 503
        finally:
            mod._data_store = orig


class TestIntrinsicsHistoryStandalone:
    """Test GET /api/runs/{run}/intrinsics-history standalone."""

    def test_returns_intrinsics_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "hp": 16}, {"step": 2, "hp": 14}]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_intrinsics_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/intrinsics-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_intrinsics_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/intrinsics-history?game=2")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 2)


class TestActionHistoryStandalone:
    """Test GET /api/runs/{run}/action-history standalone."""

    def test_returns_actions_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "action_id": 7, "action_name": "north"}]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_action_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/action-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_action_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/action-history?game=1")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 1)


class TestResolutionHistoryStandalone:
    """Test GET /api/runs/{run}/resolution-history standalone."""

    def test_returns_resolution_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "outcome": "new_object"}]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_resolution_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/resolution-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_resolution_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/resolution-history?game=3")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 3)


class TestAllObjectsStandalone:
    """Test GET /api/runs/{run}/all-objects standalone."""

    def test_returns_objects_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"node_id": "1", "shape": "@", "match_count": 5}]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_all_objects") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/all-objects")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_all_objects") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/all-objects?game=1")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 1)


class TestGameStatus:
    """Test GET /api/game/status endpoint."""

    def test_returns_idle_when_no_manager(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig = mod._game_manager
        mod._game_manager = None
        try:
            resp = client.get("/api/game/status")
            assert resp.status_code == 200
            assert resp.json()["state"] == "idle"
        finally:
            mod._game_manager = orig

    def test_returns_status_from_manager(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        mock_mgr = MagicMock()
        mock_mgr.get_status.return_value = {
            "state": "running",
            "run_name": "my-run",
            "exit_code": None,
            "error": None,
        }
        orig = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.get("/api/game/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["state"] == "running"
            assert data["run_name"] == "my-run"
        finally:
            mod._game_manager = orig


class TestGameStart:
    """Test POST /api/game/start endpoint."""

    def test_returns_503_when_no_manager(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig = mod._game_manager
        mod._game_manager = None
        try:
            resp = client.post("/api/game/start")
            assert resp.status_code == 503
        finally:
            mod._game_manager = orig

    def test_starts_game_successfully(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        mock_mgr = MagicMock()
        mock_mgr.start_game.return_value = "started"
        orig = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.post("/api/game/start?num_games=3")
            assert resp.status_code == 200
            assert resp.json() == {"status": "started"}
            mock_mgr.start_game.assert_called_once_with(num_games=3)
        finally:
            mod._game_manager = orig

    def test_returns_409_on_runtime_error(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        mock_mgr = MagicMock()
        mock_mgr.start_game.side_effect = RuntimeError("Already running")
        orig = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.post("/api/game/start")
            assert resp.status_code == 409
        finally:
            mod._game_manager = orig


class TestGameStop:
    """Test POST /api/game/stop endpoint."""

    def test_returns_503_when_no_manager(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig = mod._game_manager
        mod._game_manager = None
        try:
            resp = client.post("/api/game/stop")
            assert resp.status_code == 503
        finally:
            mod._game_manager = orig

    def test_stops_game_successfully(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        mock_mgr = MagicMock()
        orig = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.post("/api/game/stop")
            assert resp.status_code == 200
            assert resp.json() == {"status": "stopping"}
            mock_mgr.stop_game.assert_called_once()
        finally:
            mod._game_manager = orig

    def test_returns_409_on_runtime_error(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        mock_mgr = MagicMock()
        mock_mgr.stop_game.side_effect = RuntimeError("Not running")
        orig = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.post("/api/game/stop")
            assert resp.status_code == 409
        finally:
            mod._game_manager = orig


class TestReceiveStep:
    """Test POST /api/internal/step endpoint."""

    def test_receives_and_stores_step(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("recv-test-run", buf)
        resp = client.post(
            "/api/internal/step",
            json={"step": 1, "game_number": 1, "game_metrics": {"score": 10}},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert len(buf) == 1
        stored = buf.get_step(1)
        assert stored is not None
        assert stored.step == 1
        assert stored.game_number == 1

    def test_ignores_unknown_fields(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("recv-test-run2", buf)
        resp = client.post(
            "/api/internal/step",
            json={"step": 1, "game_number": 1, "unknown_field": "whatever"},
        )
        assert resp.status_code == 200
        assert len(buf) == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestEmitGameStateChanged:
    """Test _emit_game_state_changed function."""

    def test_starts_live_session_on_running(self) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        orig_loop = mod._sio_loop
        mod._sio_loop = None  # disable socket.io emission
        try:
            mod._emit_game_state_changed({"state": "running", "run_name": "emit-test-run"})
            assert mod._data_store.live_run_name == "emit-test-run"
            assert mod._data_store.live_buffer is not None
        finally:
            mod._data_store.clear_live_session()
            mod._sio_loop = orig_loop

    def test_stops_live_session_on_idle(self) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("emit-stop-run", buf)
        orig_loop = mod._sio_loop
        mod._sio_loop = None
        try:
            mod._emit_game_state_changed({"state": "idle"})
            assert mod._data_store.live_run_name is None
            assert mod._data_store.live_buffer is None
        finally:
            mod._sio_loop = orig_loop

    def test_emits_socket_event_when_loop_running(self) -> None:
        from unittest.mock import MagicMock

        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        orig_loop = mod._sio_loop
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        mock_loop.call_soon_threadsafe = MagicMock()
        mod._sio_loop = mock_loop
        try:
            with patch("asyncio.run_coroutine_threadsafe") as mock_run:
                mod._emit_game_state_changed({"state": "idle"})
                mock_run.assert_called_once()
                # Close the unawaited coroutine to avoid RuntimeWarning
                coro = mock_run.call_args[0][0]
                coro.close()
        finally:
            mod._sio_loop = orig_loop


class TestStartStopLiveSession:
    """Test _start_live_session and _stop_live_session functions."""

    def test_start_creates_buffer(self) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        mod._start_live_session("session-test-run")
        try:
            assert mod._data_store.live_run_name == "session-test-run"
            assert mod._data_store.live_buffer is not None
        finally:
            mod._data_store.clear_live_session()

    def test_stop_clears_session(self) -> None:
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        mod._start_live_session("session-stop-run")
        mod._stop_live_session()
        assert mod._data_store.live_run_name is None
        assert mod._data_store.live_buffer is None

    def test_start_does_nothing_when_no_data_store(self) -> None:
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            mod._start_live_session("no-store-run")
        finally:
            mod._data_store = orig

    def test_stop_does_nothing_when_no_data_store(self) -> None:
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            mod._stop_live_session()
        finally:
            mod._data_store = orig


class TestLiveStatusInitializing:
    """Test live_status when game_manager is initializing but no steps yet."""

    def test_returns_active_when_manager_initializing(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        assert mod._data_store is not None
        # Set up a live session with an empty buffer
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("init-run", buf)
        # Don't push any steps -- buffer is empty
        mock_mgr = MagicMock()
        mock_mgr.state = "initializing"
        orig_mgr = mod._game_manager
        try:
            mod._game_manager = mock_mgr
            resp = client.get("/api/live/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["active"] is True
            assert data["run_name"] == "init-run"
            assert data["step"] == 0
            assert data["game_numbers"] == []
        finally:
            mod._game_manager = orig_mgr

    def test_returns_active_when_manager_running_no_steps(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        from unittest.mock import MagicMock

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("running-empty-run", buf)
        mock_mgr = MagicMock()
        mock_mgr.state = "running"
        orig_mgr = mod._game_manager
        try:
            mod._game_manager = mock_mgr
            resp = client.get("/api/live/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["active"] is True
            assert data["run_name"] == "running-empty-run"
        finally:
            mod._game_manager = orig_mgr


# ---------------------------------------------------------------------------
# Tests for 503 responses when _data_store is None
# ---------------------------------------------------------------------------


class TestDataStoreNoneReturns503:
    """Test that endpoints return 503 when _data_store is None."""

    @pytest.fixture(autouse=True)
    def _nullify_data_store(self) -> Generator[None, None, None]:
        """Temporarily set _data_store to None for these tests."""
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        yield
        mod._data_store = orig

    def test_list_runs_returns_empty(self, client: TestClient) -> None:
        """list_runs returns empty list when _data_store is None."""
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_games_returns_503(self, client: TestClient) -> None:
        """list_games returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/games")
        assert resp.status_code == 503

    def test_get_step_returns_503(self, client: TestClient) -> None:
        """get_step returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/step/1")
        assert resp.status_code == 503

    def test_graph_history_returns_503(self, client: TestClient) -> None:
        """get_graph_history returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/graph-history")
        assert resp.status_code == 503

    def test_event_history_returns_503(self, client: TestClient) -> None:
        """get_event_history returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/event-history")
        assert resp.status_code == 503

    def test_intrinsics_history_returns_503(self, client: TestClient) -> None:
        """get_intrinsics_history returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/intrinsics-history")
        assert resp.status_code == 503

    def test_action_history_returns_503(self, client: TestClient) -> None:
        """get_action_history returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/action-history")
        assert resp.status_code == 503

    def test_resolution_history_returns_503(self, client: TestClient) -> None:
        """get_resolution_history returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/resolution-history")
        assert resp.status_code == 503

    def test_all_objects_returns_503(self, client: TestClient) -> None:
        """get_all_objects returns 503 when _data_store is None."""
        resp = client.get("/api/runs/test-run/all-objects")
        assert resp.status_code == 503

    def test_save_bookmarks_returns_503(self, client: TestClient) -> None:
        """save_bookmarks returns 503 when _data_store is None."""
        bookmarks = [{"step": 1, "game": 1, "annotation": "test", "created": "2026-01-01T00:00:00"}]
        resp = client.post("/api/runs/test-run/bookmarks", json=bookmarks)
        assert resp.status_code == 503

    def test_get_bookmarks_returns_empty(self, client: TestClient) -> None:
        """get_bookmarks returns empty list when _data_store is None."""
        resp = client.get("/api/runs/test-run/bookmarks")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Schema endpoint tests
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Test GET /api/runs/{run}/schema endpoint."""

    def test_returns_schema_from_store(self, client: TestClient) -> None:
        """get_schema returns schema data from the store."""
        import roc.reporting.api_server as mod

        mock_schema = {"nodes": [{"label": "Object"}], "edges": [{"type": "HAS"}]}
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_schema") as mock_method:
            mock_method.return_value = mock_schema
            resp = client.get("/api/runs/test-run/schema")
            assert resp.status_code == 200
            assert resp.json() == mock_schema
            mock_method.assert_called_once_with("test-run")

    def test_returns_404_when_schema_not_found(self, client: TestClient) -> None:
        """get_schema returns 404 when no schema exists for the run."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_schema") as mock_method:
            mock_method.return_value = None
            resp = client.get("/api/runs/test-run/schema")
            assert resp.status_code == 404

    def test_returns_503_when_no_data_store(self, client: TestClient) -> None:
        """get_schema returns 503 when _data_store is None."""
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            resp = client.get("/api/runs/test-run/schema")
            assert resp.status_code == 503
        finally:
            mod._data_store = orig


# ---------------------------------------------------------------------------
# Action map endpoint tests
# ---------------------------------------------------------------------------


class TestGetActionMap:
    """Test GET /api/runs/{run}/action-map endpoint."""

    def test_returns_action_map_from_store(self, client: TestClient) -> None:
        """get_action_map returns action map data from the store."""
        import roc.reporting.api_server as mod

        mock_map = [{"id": 0, "name": "wait"}, {"id": 1, "name": "north"}]
        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_action_map") as mock_method:
            mock_method.return_value = mock_map
            resp = client.get("/api/runs/test-run/action-map")
            assert resp.status_code == 200
            assert resp.json() == mock_map
            mock_method.assert_called_once_with("test-run")

    def test_returns_empty_list_when_no_action_map(self, client: TestClient) -> None:
        """get_action_map returns empty list when action_map is None."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_action_map") as mock_method:
            mock_method.return_value = None
            resp = client.get("/api/runs/test-run/action-map")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_returns_503_when_no_data_store(self, client: TestClient) -> None:
        """get_action_map returns 503 when _data_store is None."""
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            resp = client.get("/api/runs/test-run/action-map")
            assert resp.status_code == 503
        finally:
            mod._data_store = orig


# ---------------------------------------------------------------------------
# Bookmark edge cases
# ---------------------------------------------------------------------------


class TestBookmarkEdgeCases:
    """Test bookmark endpoint edge cases."""

    def test_returns_empty_on_malformed_bookmarks_file(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """get_bookmarks returns empty list when bookmarks file is malformed."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        run_dir = tmp_path / "bad-run"
        run_dir.mkdir()
        bookmarks_file = run_dir / "bookmarks.json"
        bookmarks_file.write_text("not valid json {{{")
        resp = client.get("/api/runs/bad-run/bookmarks")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# get_steps_batch exception handling
# ---------------------------------------------------------------------------


class TestGetStepsBatchExceptionPath:
    """Test get_steps_batch exception fallback."""

    def test_returns_empty_on_data_store_exception(self, client: TestClient) -> None:
        """get_steps_batch returns empty dict when data_store raises."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "get_steps_batch") as mock_method:
            mock_method.side_effect = RuntimeError("DB error")
            resp = client.get("/api/runs/test-run/steps?steps=1,2,3")
            assert resp.status_code == 200
            assert resp.json() == {}


# ---------------------------------------------------------------------------
# receive_step stop response
# ---------------------------------------------------------------------------


class TestReceiveStepStopResponse:
    """Test POST /api/internal/step returns stop flag when requested."""

    def test_returns_stop_when_manager_requests_stop(self, client: TestClient) -> None:
        """receive_step includes stop:true when game_manager requests stop."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("stop-test-run", buf)

        mock_mgr = MagicMock()
        mock_mgr.is_stop_requested.return_value = True
        orig_mgr = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.post(
                "/api/internal/step",
                json={"step": 1, "game_number": 1},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["stop"] is True
        finally:
            mod._game_manager = orig_mgr

    def test_no_stop_when_manager_does_not_request_stop(self, client: TestClient) -> None:
        """receive_step omits stop key when game_manager does not request stop."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("no-stop-test-run", buf)

        mock_mgr = MagicMock()
        mock_mgr.is_stop_requested.return_value = False
        orig_mgr = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.post(
                "/api/internal/step",
                json={"step": 1, "game_number": 1},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "stop" not in data
        finally:
            mod._game_manager = orig_mgr


# ---------------------------------------------------------------------------
# receive_action_map endpoint
# ---------------------------------------------------------------------------


class TestReceiveActionMap:
    """Test POST /api/internal/action-map endpoint."""

    def test_stores_action_map(self, client: TestClient) -> None:
        """receive_action_map stores the action map in data_store."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "set_action_map") as mock_method:
            action_map = [{"id": 0, "name": "wait"}, {"id": 1, "name": "north"}]
            resp = client.post(
                "/api/internal/action-map",
                json={"run_name": "map-test-run", "action_map": action_map},
            )
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
            mock_method.assert_called_once_with("map-test-run", action_map)

    def test_ignores_missing_run_name(self, client: TestClient) -> None:
        """receive_action_map does not store when run_name is empty."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "set_action_map") as mock_method:
            resp = client.post(
                "/api/internal/action-map",
                json={"run_name": "", "action_map": [{"id": 0, "name": "wait"}]},
            )
            assert resp.status_code == 200
            mock_method.assert_not_called()

    def test_ignores_non_list_action_map(self, client: TestClient) -> None:
        """receive_action_map does not store when action_map is not a list."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        with patch.object(mod._data_store, "set_action_map") as mock_method:
            resp = client.post(
                "/api/internal/action-map",
                json={"run_name": "test-run", "action_map": "not-a-list"},
            )
            assert resp.status_code == 200
            mock_method.assert_not_called()

    def test_handles_no_data_store(self, client: TestClient) -> None:
        """receive_action_map returns ok even when _data_store is None."""
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            resp = client.post(
                "/api/internal/action-map",
                json={"run_name": "test-run", "action_map": [{"id": 0}]},
            )
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
        finally:
            mod._data_store = orig


# ---------------------------------------------------------------------------
# _notify_new_step function
# ---------------------------------------------------------------------------


class TestNotifyNewStep:
    """Test _notify_new_step function."""

    def test_emits_step_via_socket_when_loop_running(self) -> None:
        """_notify_new_step emits socket event when loop is running."""
        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        mod._sio_loop = mock_loop
        try:
            with patch("asyncio.run_coroutine_threadsafe") as mock_run:
                step_data = StepData(step=1, game_number=1)
                mod._notify_new_step(step_data)
                mock_run.assert_called_once()
                # Close the unawaited coroutine to avoid RuntimeWarning
                coro = mock_run.call_args[0][0]
                coro.close()
        finally:
            mod._sio_loop = orig_loop

    def test_does_nothing_when_loop_is_none(self) -> None:
        """_notify_new_step does nothing when _sio_loop is None."""
        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        mod._sio_loop = None
        try:
            with patch("asyncio.run_coroutine_threadsafe") as mock_run:
                step_data = StepData(step=1, game_number=1)
                mod._notify_new_step(step_data)
                mock_run.assert_not_called()
        finally:
            mod._sio_loop = orig_loop

    def test_does_nothing_when_loop_not_running(self) -> None:
        """_notify_new_step does nothing when the event loop is not running."""
        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        mod._sio_loop = mock_loop
        try:
            with patch("asyncio.run_coroutine_threadsafe") as mock_run:
                step_data = StepData(step=1, game_number=1)
                mod._notify_new_step(step_data)
                mock_run.assert_not_called()
        finally:
            mod._sio_loop = orig_loop

    def test_swallows_exceptions(self) -> None:
        """_notify_new_step swallows exceptions to avoid breaking the game loop."""
        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        mod._sio_loop = mock_loop
        try:
            with patch("asyncio.run_coroutine_threadsafe", side_effect=RuntimeError("boom")):
                step_data = StepData(step=1, game_number=1)
                # Should not raise
                mod._notify_new_step(step_data)
        finally:
            mod._sio_loop = orig_loop


# ---------------------------------------------------------------------------
# _emit_game_state_changed edge cases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# live_status with manager but no live_run_name
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestLiveStatusManagerNoRunName:
    """Test live_status when manager is running but no live_run_name set."""

    def test_returns_inactive_when_no_run_name(self, client: TestClient) -> None:
        """live_status returns inactive when manager is running but no run name."""
        import roc.reporting.api_server as mod

        mock_mgr = MagicMock()
        mock_mgr.state = "initializing"
        orig_mgr = mod._game_manager
        # Clear any live session so live_run_name is None
        assert mod._data_store is not None
        mod._data_store.clear_live_session()
        mod._game_manager = mock_mgr
        try:
            resp = client.get("/api/live/status")
            assert resp.status_code == 200
            data = resp.json()
            # No live_run_name means the fallback returns inactive
            assert data["active"] is False
        finally:
            mod._game_manager = orig_mgr

    def test_returns_inactive_when_manager_idle(self, client: TestClient) -> None:
        """live_status returns inactive when manager state is idle."""
        import roc.reporting.api_server as mod

        mock_mgr = MagicMock()
        mock_mgr.state = "idle"
        orig_mgr = mod._game_manager
        mod._game_manager = mock_mgr
        try:
            resp = client.get("/api/live/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["active"] is False
        finally:
            mod._game_manager = orig_mgr


# ---------------------------------------------------------------------------
# _capture_event_loop (startup handler)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestCaptureEventLoop:
    """Test the _capture_event_loop startup handler."""

    def test_sets_sio_loop_and_signals_ready(self) -> None:
        """_capture_event_loop sets _sio_loop and signals _server_ready."""
        import asyncio

        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        orig_ready = mod._server_ready
        orig_summary_thread = mod._summary_thread
        mod._server_ready = __import__("threading").Event()
        mod._summary_thread = None
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(mod._capture_event_loop())
            assert mod._sio_loop is loop
            assert mod._server_ready.is_set()
            loop.close()
        finally:
            mod._sio_loop = orig_loop
            mod._server_ready = orig_ready
            mod._summary_thread = orig_summary_thread

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
    def test_starts_summary_thread_when_data_store_present(self) -> None:
        """_capture_event_loop starts a summary thread when _data_store exists."""
        import asyncio
        import threading

        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        orig_ready = mod._server_ready
        orig_summary_thread = mod._summary_thread
        mod._server_ready = threading.Event()
        mod._summary_thread = None
        try:
            assert mod._data_store is not None
            with patch.object(mod._data_store, "populate_run_summaries"):
                loop = asyncio.new_event_loop()
                loop.run_until_complete(mod._capture_event_loop())
                # _capture_event_loop sets _summary_thread via global assignment;
                # use getattr to bypass mypy type narrowing from the None assignment above
                thread = getattr(mod, "_summary_thread")
                assert isinstance(thread, threading.Thread)
                assert thread.daemon is True
                # Wait for the thread to finish (it's mocked so it returns immediately)
                thread.join(timeout=2)
                loop.close()
        finally:
            mod._sio_loop = orig_loop
            mod._server_ready = orig_ready
            mod._summary_thread = orig_summary_thread

    def test_does_not_restart_summary_thread(self) -> None:
        """_capture_event_loop does not restart summary thread if already set."""
        import asyncio

        import roc.reporting.api_server as mod

        orig_loop = mod._sio_loop
        orig_ready = mod._server_ready
        orig_summary_thread = mod._summary_thread
        mod._server_ready = __import__("threading").Event()
        existing_thread = MagicMock()
        mod._summary_thread = existing_thread
        try:
            with patch.object(mod.sio, "emit", return_value=MagicMock()):
                loop = asyncio.new_event_loop()
                loop.run_until_complete(mod._capture_event_loop())
                # Should not have been replaced
                assert mod._summary_thread is existing_thread
                loop.close()
        finally:
            mod._sio_loop = orig_loop
            mod._server_ready = orig_ready
            mod._summary_thread = orig_summary_thread


# ---------------------------------------------------------------------------
# stop_dashboard
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestStopDashboard:
    """Test stop_dashboard function."""

    def test_clears_buffer_and_session(self) -> None:
        """stop_dashboard clears step buffer and live session."""
        import roc.reporting.api_server as mod

        assert mod._data_store is not None
        buf = StepBuffer(capacity=100)
        mod._data_store.set_live_session("stop-test", buf)
        with patch("roc.reporting.step_buffer.clear_step_buffer") as mock_clear:
            mod.stop_dashboard()
            mock_clear.assert_called_once()
        assert mod._data_store.live_run_name is None

    def test_handles_no_data_store(self) -> None:
        """stop_dashboard does not crash when _data_store is None."""
        import roc.reporting.api_server as mod

        orig = mod._data_store
        mod._data_store = None
        try:
            with patch("roc.reporting.step_buffer.clear_step_buffer"):
                mod.stop_dashboard()
        finally:
            mod._data_store = orig


# ---------------------------------------------------------------------------
# start_dashboard
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestStartDashboard:
    """Test start_dashboard function."""

    def test_returns_early_when_already_started(self) -> None:
        """start_dashboard does nothing if already started."""
        import roc.reporting.api_server as mod

        orig = mod._started
        mod._started = True
        try:
            # Should return immediately without doing anything
            mod.start_dashboard()
        finally:
            mod._started = orig

    def test_returns_early_when_dashboard_disabled(self) -> None:
        """start_dashboard returns early when dashboard is disabled in config."""
        import roc.reporting.api_server as mod

        orig_started = mod._started
        mod._started = False
        try:
            mock_cfg = MagicMock()
            mock_cfg.dashboard_enabled = False
            with patch("roc.config.Config.get", return_value=mock_cfg):
                mod.start_dashboard()
            assert mod._started is False
        finally:
            mod._started = orig_started

    def test_returns_early_when_no_ducklake_store(self) -> None:
        """start_dashboard returns early when DuckLakeStore is not available."""
        import roc.reporting.api_server as mod

        orig_started = mod._started
        mod._started = False
        try:
            mock_cfg = MagicMock()
            mock_cfg.dashboard_enabled = True
            with (
                patch("roc.config.Config.get", return_value=mock_cfg),
                patch(
                    "roc.reporting.observability.Observability.get_ducklake_store",
                    return_value=None,
                ),
                patch.object(mod.sio, "emit", return_value=MagicMock()),
            ):
                mod.start_dashboard()
            assert mod._started is False
        finally:
            mod._started = orig_started
