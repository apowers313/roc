"""Unit tests for the FastAPI dashboard API server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app
from roc.reporting.run_store import StepData
from roc.reporting.run_writer import RunWriter
from roc.reporting.types import RunSummary


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _setup_data_store(tmp_path: Path) -> Generator[None, None, None]:
    """Point the API at a RunRegistry backed by a temp directory.

    Resets the unified-run singletons (reader, registry, cache) so the
    next ``_get_reader()`` call rebuilds them against the new registry.
    """
    import roc.reporting.api_server as mod
    from roc.reporting.step_cache import StepCache

    orig_reader = mod._run_reader
    orig_registry = mod._run_registry
    orig_cache = mod._step_cache
    mod.init_data_dir(tmp_path)
    mod._run_reader = None
    mod._step_cache = StepCache(capacity=5000)
    yield
    mod._run_reader = orig_reader
    mod._run_registry = orig_registry
    mod._step_cache = orig_cache


@pytest.fixture()
def live_buffer(tmp_path: Path) -> Generator[RunWriter, None, None]:
    """Set up a RunWriter with multi-game test data as the live run.

    Seeds DuckLake with screen records for game 1 (steps 1-10) and game 2
    (steps 11-15) so that list_games(), step-range, and step fetches all
    work correctly. Also pushes steps via RunWriter so the in-memory cache
    and registry step-range are updated.
    """
    import roc.reporting.api_server as mod
    from roc.reporting.ducklake_store import DuckLakeStore

    run_dir = tmp_path / "test-live-run"
    store = DuckLakeStore(run_dir, read_only=False)
    # Seed DuckLake with screen records so list_games() and step lookups work
    records = [
        {"step": i, "game_number": 1, "timestamp": i * 1000, "body": "{}"} for i in range(1, 11)
    ] + [{"step": i, "game_number": 2, "timestamp": i * 1000, "body": "{}"} for i in range(11, 16)]
    store.insert("screens", records)
    registry = mod._get_registry()
    assert registry is not None
    writer = RunWriter("test-live-run", registry, mod._step_cache, None, store)
    for i in range(1, 11):
        writer.push_step(StepData(step=i, game_number=1))
    for i in range(11, 16):
        writer.push_step(StepData(step=i, game_number=2))
    try:
        yield writer
    finally:
        writer.close()
        store.close()


class TestListRuns:
    def test_returns_empty_when_no_runs(self, client: TestClient) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_runs_when_present(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "list_runs") as mock_list:
            mock_list.return_value = [RunSummary(name="test-run", games=2, steps=100)]
            resp = client.get("/api/runs")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["name"] == "test-run"
            assert data[0]["steps"] == 100

    # ----------------------------------------------------------------
    # /api/runs include_all parameter (regression: missing-runs bug)
    # ----------------------------------------------------------------

    def test_default_does_not_pass_include_all(self, client: TestClient) -> None:
        """Default behavior must call list_runs(min_steps=10, include_all=False)."""
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "list_runs") as mock_list:
            mock_list.return_value = []
            client.get("/api/runs")
            mock_list.assert_called_once_with(min_steps=10, include_all=False)

    def test_include_all_flag_propagates(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "list_runs") as mock_list:
            mock_list.return_value = []
            client.get("/api/runs?include_all=true")
            mock_list.assert_called_once_with(min_steps=10, include_all=True)

    def test_min_steps_param_overrides_default(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "list_runs") as mock_list:
            mock_list.return_value = []
            client.get("/api/runs?min_steps=0&include_all=true")
            mock_list.assert_called_once_with(min_steps=0, include_all=True)

    def test_response_includes_status_field(self, client: TestClient) -> None:
        """Each run in the response carries a status (regression for the
        bug class where the dropdown couldn't tell ok/short/empty/corrupt
        runs apart)."""
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "list_runs") as mock_list:
            mock_list.return_value = [
                RunSummary(name="ok-run", games=1, steps=100, status="ok"),
                RunSummary(
                    name="empty-run",
                    games=0,
                    steps=0,
                    status="empty",
                ),
                RunSummary(
                    name="bad-run",
                    games=0,
                    steps=0,
                    status="corrupt",
                    error="DuckLake catalog open failed",
                ),
            ]
            resp = client.get("/api/runs?include_all=true")
            assert resp.status_code == 200
            data = resp.json()
            statuses = {r["name"]: r["status"] for r in data}
            assert statuses == {
                "ok-run": "ok",
                "empty-run": "empty",
                "bad-run": "corrupt",
            }
            bad = next(r for r in data if r["name"] == "bad-run")
            assert bad["error"] == "DuckLake catalog open failed"


class TestStepRange:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step-range")
        assert resp.status_code == 404

    def test_step_range_includes_tail_growing_false_for_closed_run(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Phase 3: closed runs return tail_growing=False in step-range."""
        from roc.reporting.ducklake_store import DuckLakeStore

        run_dir = tmp_path / "closed-run"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            records = [
                {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                for s in range(1, 4)
            ]
            store.insert("screens", records)
        finally:
            store.close()

        resp = client.get("/api/runs/closed-run/step-range")
        assert resp.status_code == 200
        body = resp.json()
        assert body["min"] == 1
        assert body["max"] == 3
        assert body["tail_growing"] is False

    def test_step_range_endpoint_returns_tail_growing_true_for_active_writer(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Phase 3: an active writer flips tail_growing=True for step-range."""
        import roc.reporting.api_server as mod
        from roc.reporting.ducklake_store import DuckLakeStore

        run_dir = tmp_path / "live-run"
        # Seed some baseline rows so the registry can list_games() etc.
        seed = DuckLakeStore(run_dir, read_only=False)
        try:
            seed.insert(
                "screens",
                [
                    {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                    for s in range(1, 3)
                ],
            )
        finally:
            seed.close()

        # Open a writer and attach via the registry (mimics RunWriter init).
        reader = mod._get_reader()
        assert reader is not None
        registry = mod._run_registry
        assert registry is not None
        writer_store = DuckLakeStore(run_dir, read_only=False)
        try:
            registry.attach_writer_store("live-run", writer_store)
            resp = client.get("/api/runs/live-run/step-range")
            assert resp.status_code == 200
            body = resp.json()
            assert body["tail_growing"] is True
        finally:
            registry.detach_writer_store("live-run")
            writer_store.close()

    def test_step_range_after_writer_close_returns_false(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Phase 3: after detach, step-range tail_growing flips back to False."""
        import roc.reporting.api_server as mod
        from roc.reporting.ducklake_store import DuckLakeStore

        run_dir = tmp_path / "lifecycle-run"
        seed = DuckLakeStore(run_dir, read_only=False)
        try:
            seed.insert(
                "screens",
                [
                    {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                    for s in range(1, 3)
                ],
            )
        finally:
            seed.close()

        reader = mod._get_reader()
        assert reader is not None
        registry = mod._run_registry
        assert registry is not None

        writer_store = DuckLakeStore(run_dir, read_only=False)
        registry.attach_writer_store("lifecycle-run", writer_store)
        writer_store.close()
        registry.detach_writer_store("lifecycle-run")

        resp = client.get("/api/runs/lifecycle-run/step-range")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tail_growing"] is False


class TestGraphHistory:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/graph-history")
        assert resp.status_code == 404

    def test_returns_data_from_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [
            {"step": 1, "node_count": 10, "node_max": 100, "edge_count": 20, "edge_max": 200},
        ]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/graph-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", "graph", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/graph-history?game=2")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", "graph", 2)


class TestEventHistory:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/event-history")
        assert resp.status_code == 404

    def test_returns_data_from_store(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [
            {"step": 1, "roc.perception": 5, "roc.attention": 3},
        ]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/event-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", "event", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/event-history?game=1")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", "event", 1)


class TestGetStep:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step/1")
        assert resp.status_code == 404

    def test_step_endpoint_returns_typed_envelope_on_404(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """A step that's out of range for an existing run returns a 404
        with the typed StepResponse envelope, not an empty body."""
        from roc.reporting.ducklake_store import DuckLakeStore

        # Create a real DuckLake run with 3 steps
        run_dir = tmp_path / "envelope-run"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            records = [
                {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                for s in range(1, 4)
            ]
            store.insert("screens", records)
        finally:
            store.close()

        # Asking for step 999 should yield 404 with the envelope
        resp = client.get("/api/runs/envelope-run/step/999")
        assert resp.status_code == 404
        body = resp.json()
        # The body is the StepResponse envelope, not an empty dict.
        assert body.get("status") == "out_of_range"
        assert body.get("data") is None
        assert body.get("range") is not None
        assert body["range"]["max"] == 3

    def test_step_endpoint_returns_typed_envelope_on_500(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """A backend error returns a 500 with the typed StepResponse envelope."""
        import roc.reporting.api_server as mod
        from roc.reporting.ducklake_store import DuckLakeStore

        # Create a real DuckLake run with 3 steps
        run_dir = tmp_path / "boom-run"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            records = [
                {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                for s in range(1, 4)
            ]
            store.insert("screens", records)
        finally:
            store.close()

        # Force the registry entry to load, then patch its run_store
        reader = mod._get_reader()
        assert reader is not None
        registry = mod._run_registry
        assert registry is not None
        entry = registry.get("boom-run")
        assert entry is not None
        with patch.object(entry.run_store, "get_step_data", side_effect=RuntimeError("kaboom")):
            resp = client.get("/api/runs/boom-run/step/2")
        assert resp.status_code == 500
        body = resp.json()
        assert body.get("status") == "error"
        assert "RuntimeError" in body.get("error", "")
        assert "kaboom" in body.get("error", "")


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

    def test_games_include_step_counts(self, client: TestClient, live_buffer: RunWriter) -> None:
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

    def test_global_range(self, client: TestClient, live_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-live-run/step-range")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 15

    def test_game1_range(self, client: TestClient, live_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-live-run/step-range?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 10

    def test_game2_range(self, client: TestClient, live_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-live-run/step-range?game=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 11
        assert data["max"] == 15

    def test_nonexistent_game_returns_zeros(
        self, client: TestClient, live_buffer: RunWriter
    ) -> None:
        resp = client.get("/api/runs/test-live-run/step-range?game=99")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 0
        assert data["max"] == 0


# ---------------------------------------------------------------------------
# Regression: subprocess mode buffer was not consulted by REST endpoints,
# causing panels to show no data after browser refresh during a live game
# started via the Game Menu.
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_subprocess_buffer(tmp_path: Path) -> Generator[RunWriter, None, None]:
    """Set up a RunWriter in subprocess/HTTP callback mode.

    Seeds DuckLake with screen records for game 1 (steps 1-10) and game 2
    (steps 11-15) so that list_games(), step-range, and step fetches all
    work correctly. Also pushes steps via RunWriter so the in-memory cache
    and registry step-range are updated.
    """
    import roc.reporting.api_server as mod
    from roc.reporting.ducklake_store import DuckLakeStore

    run_dir = tmp_path / "test-subprocess-run"
    store = DuckLakeStore(run_dir, read_only=False)
    # Seed DuckLake with screen records so list_games() and step lookups work
    records = [
        {"step": i, "game_number": 1, "timestamp": i * 1000, "body": "{}"} for i in range(1, 11)
    ] + [{"step": i, "game_number": 2, "timestamp": i * 1000, "body": "{}"} for i in range(11, 16)]
    store.insert("screens", records)
    registry = mod._get_registry()
    assert registry is not None
    writer = RunWriter("test-subprocess-run", registry, mod._step_cache, None, store)
    for i in range(1, 11):
        writer.push_step(StepData(step=i, game_number=1))
    for i in range(11, 16):
        writer.push_step(StepData(step=i, game_number=2))
    try:
        yield writer
    finally:
        writer.close()
        store.close()


class TestSubprocessBufferInRunsList:
    """Regression: live runs started via subprocess didn't appear in the
    runs dropdown because _get_run_summary tried DuckLake (locked by subprocess)
    and fell back to 0 steps, which was filtered by min_steps."""

    def test_live_run_appears_in_runs_list(
        self, client: TestClient, live_subprocess_buffer: RunWriter
    ) -> None:
        resp = client.get("/api/runs?min_steps=0")
        assert resp.status_code == 200
        runs = resp.json()
        names = [r["name"] for r in runs]
        assert "test-subprocess-run" in names

    def test_live_run_has_correct_step_count(
        self, client: TestClient, live_subprocess_buffer: RunWriter
    ) -> None:
        resp = client.get("/api/runs?min_steps=0")
        assert resp.status_code == 200
        runs = resp.json()
        live = next(r for r in runs if r["name"] == "test-subprocess-run")
        assert live["steps"] == 15
        assert live["games"] == 2

    def test_live_run_not_filtered_by_min_steps(
        self, client: TestClient, live_subprocess_buffer: RunWriter
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
        self, client: TestClient, live_subprocess_buffer: RunWriter
    ) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step/5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 5
        assert data["game_number"] == 1

    def test_step_not_in_subprocess_buffer_falls_through_to_store(
        self, client: TestClient, live_subprocess_buffer: RunWriter
    ) -> None:
        """When the step isn't in the buffer, the live read path falls
        through to the writer's RunStore. The registry has a real
        backing store (via attach_writer_store), so the request returns
        an out_of_range response for steps beyond the range."""
        resp = client.get("/api/runs/test-subprocess-run/step/999")
        assert resp.status_code == 404
        data = resp.json()
        assert data.get("status") == "out_of_range"


class TestSubprocessBufferGames:
    """Regression: GET /api/runs/{run}/games returned 404 for subprocess live runs."""

    def test_games_from_subprocess_buffer(
        self, client: TestClient, live_subprocess_buffer: RunWriter
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

    def test_global_range(self, client: TestClient, live_subprocess_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step-range")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 15

    def test_game_filtered_range(
        self, client: TestClient, live_subprocess_buffer: RunWriter
    ) -> None:
        resp = client.get("/api/runs/test-subprocess-run/step-range?game=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 10


# ---------------------------------------------------------------------------
# Live history endpoints -- previously broken in subprocess mode because
# they went straight to DuckLake which was file-locked by the subprocess.
# Now served from DataStore's in-memory indices.
# ---------------------------------------------------------------------------


# Phase 2 of the unified-run architecture deletes the in-memory live history
# indexing path that ``TestLiveHistoryEndpoints`` exercised here. The same
# behavior shapes are now covered against a real DuckLake catalog by
# ``tests/integration/reporting/test_history_via_ducklake.py`` -- live runs
# read through the writer's ``DuckLakeStore`` (installed via
# ``RunRegistry.attach_writer_store``) and historical runs read through a
# fresh ``read_only=True`` instance, so a pure in-memory unit test for the
# old buffer-driven path no longer maps to a real code path.


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

    # ----------------------------------------------------------------
    # Regression: NAType / NaN serialization (the recurring "step
    # endpoint returns 500 and dashboard renders nothing" failure)
    # ----------------------------------------------------------------

    def test_pandas_NA_becomes_none(self) -> None:
        """Pandas NA must serialize to None, not raise TypeError.

        Hit during dashboard probe at 2026-04-08: GET /step/1 returned
        500 because DuckDB returned a nullable Int column whose missing
        values are pd.NA, and json.dumps cannot encode NAType. The UI
        showed "no data" with no visible error -- a textbook example
        of the "errors not visible in the UI" failure mode.
        """
        import json

        import pandas as pd

        from roc.reporting.api_server import _convert_numpy

        result = _convert_numpy(pd.NA)
        assert result is None
        # And it must round-trip through json.dumps without raising.
        assert json.dumps(_convert_numpy({"x": pd.NA})) == '{"x": null}'

    def test_pandas_NaT_becomes_none(self) -> None:
        import pandas as pd

        from roc.reporting.api_server import _convert_numpy

        assert _convert_numpy(pd.NaT) is None

    def test_numpy_nan_becomes_none(self) -> None:
        """numpy NaN cannot be encoded as valid JSON -- map to None."""
        import json

        from roc.reporting.api_server import _convert_numpy

        result = _convert_numpy(np.float64("nan"))
        assert result is None
        # Without the fix, json.dumps would emit "NaN" which is invalid JSON.
        assert json.dumps(_convert_numpy({"x": np.float64("nan")})) == '{"x": null}'

    def test_numpy_inf_becomes_none(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        assert _convert_numpy(np.float64("inf")) is None
        assert _convert_numpy(np.float64("-inf")) is None

    def test_python_nan_becomes_none(self) -> None:
        from roc.reporting.api_server import _convert_numpy

        assert _convert_numpy(float("nan")) is None

    def test_nested_NA_in_list(self) -> None:
        """A list with NA elements (e.g. logs from DuckLake) must serialize."""
        import json

        import pandas as pd

        from roc.reporting.api_server import _convert_numpy

        data = {"rows": [{"col": pd.NA}, {"col": 5}]}
        result = _convert_numpy(data)
        # The whole tree must be JSON-serializable now.
        out = json.loads(json.dumps(result))
        assert out["rows"][0]["col"] is None
        assert out["rows"][1]["col"] == 5


class TestGetStepWithData:
    """Test GET /api/runs/{run}/step/{step} returning actual data."""

    def test_returns_step_data(self, client: TestClient, live_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-live-run/step/3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 3
        assert data["game_number"] == 1

    def test_returns_step_with_numpy_values(self, client: TestClient, tmp_path: Path) -> None:
        import roc.reporting.api_server as mod
        from roc.reporting.ducklake_store import DuckLakeStore

        run_dir = tmp_path / "numpy-run"
        store = DuckLakeStore(run_dir, read_only=False)
        registry = mod._get_registry()
        assert registry is not None
        writer = RunWriter("numpy-run", registry, mod._step_cache, None, store)
        try:
            writer.push_step(
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
        finally:
            writer.close()
            store.close()


class TestGetStepsBatch:
    """Test GET /api/runs/{run}/steps?steps=1,2,3 batch endpoint."""

    def test_returns_batch_data(self, client: TestClient, live_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-live-run/steps?steps=1,3,5")
        assert resp.status_code == 200
        data = resp.json()
        assert "1" in data
        assert "3" in data
        assert "5" in data
        assert data["1"]["step"] == 1
        assert data["5"]["game_number"] == 1

    def test_missing_steps_fall_through_to_store(
        self, client: TestClient, live_buffer: RunWriter
    ) -> None:
        """Missing steps fall through to the store and return empty StepData (game_number=0)."""
        resp = client.get("/api/runs/test-live-run/steps?steps=1,999")
        assert resp.status_code == 200
        data = resp.json()
        assert "1" in data
        assert "999" in data
        assert data["1"]["game_number"] == 1
        assert data["999"]["game_number"] == 0

    def test_returns_503_when_no_registry(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig_registry = mod._run_registry
        orig_reader = mod._run_reader
        mod._run_registry = None
        mod._run_reader = None
        try:
            resp = client.get("/api/runs/test-run/steps?steps=1,2")
            assert resp.status_code == 503
        finally:
            mod._run_registry = orig_registry
            mod._run_reader = orig_reader

    def test_returns_500_on_internal_error(
        self, client: TestClient, live_buffer: RunWriter
    ) -> None:
        with patch(
            "roc.reporting.api_server._get_reader",
            return_value=MagicMock(get_steps_batch=MagicMock(side_effect=RuntimeError("db crash"))),
        ):
            resp = client.get("/api/runs/test-live-run/steps?steps=1,2")
        assert resp.status_code == 500
        assert "failed" in resp.json()["detail"].lower()


class TestGetStepRangeWithData:
    """Test GET /api/runs/{run}/step-range returning actual data (not 404)."""

    def test_returns_step_range_from_live(self, client: TestClient, live_buffer: RunWriter) -> None:
        resp = client.get("/api/runs/test-live-run/step-range")
        assert resp.status_code == 200
        data = resp.json()
        assert data["min"] == 1
        assert data["max"] == 15

    def test_returns_503_when_no_registry(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig_registry = mod._run_registry
        orig_reader = mod._run_reader
        mod._run_registry = None
        mod._run_reader = None
        try:
            resp = client.get("/api/runs/test-run/step-range")
            assert resp.status_code == 503
        finally:
            mod._run_registry = orig_registry
            mod._run_reader = orig_reader


class TestMetricsHistoryStandalone:
    """Test GET /api/runs/{run}/metrics-history standalone."""

    def test_returns_metrics_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "score": 10}, {"step": 2, "score": 20}]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/metrics-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", "metrics", None, fields=None)

    def test_passes_game_and_fields(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = [{"step": 1, "score": 10}]
            resp = client.get("/api/runs/test-run/metrics-history?game=1&fields=score")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", "metrics", 1, fields=["score"])

    def test_returns_503_when_no_registry(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        orig_reader = mod._run_reader
        orig_registry = mod._run_registry
        mod._run_reader = None
        mod._run_registry = None
        try:
            resp = client.get("/api/runs/test-run/metrics-history")
            assert resp.status_code == 503
        finally:
            mod._run_reader = orig_reader
            mod._run_registry = orig_registry


class TestIntrinsicsHistoryStandalone:
    """Test GET /api/runs/{run}/intrinsics-history standalone."""

    def test_returns_intrinsics_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "hp": 16}, {"step": 2, "hp": 14}]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/intrinsics-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", "intrinsics", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/intrinsics-history?game=2")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", "intrinsics", 2)


class TestActionHistoryStandalone:
    """Test GET /api/runs/{run}/action-history standalone."""

    def test_returns_actions_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "action_id": 7, "action_name": "north"}]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/action-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", "action", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/action-history?game=1")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", "action", 1)


class TestResolutionHistoryStandalone:
    """Test GET /api/runs/{run}/resolution-history standalone."""

    def test_returns_resolution_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"step": 1, "outcome": "new_object"}]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/resolution-history")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", "resolution", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_history") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/resolution-history?game=3")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", "resolution", 3)


class TestAllObjectsStandalone:
    """Test GET /api/runs/{run}/all-objects standalone.

    Phase 1 bug-fix migration (BUG-C1): the endpoint reads through
    ``RunReader`` instead of ``DataStore``. The patches target the reader.
    """

    def test_returns_objects_from_mock(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        mock_data = [{"node_id": "1", "shape": "@", "match_count": 5}]
        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_all_objects") as mock_method:
            mock_method.return_value = mock_data
            resp = client.get("/api/runs/test-run/all-objects")
            assert resp.status_code == 200
            assert resp.json() == mock_data
            mock_method.assert_called_once_with("test-run", None)

    def test_passes_game_filter(self, client: TestClient) -> None:
        import roc.reporting.api_server as mod

        reader = mod._get_reader()
        assert reader is not None
        with patch.object(reader, "get_all_objects") as mock_method:
            mock_method.return_value = []
            resp = client.get("/api/runs/test-run/all-objects?game=1")
            assert resp.status_code == 200
            mock_method.assert_called_once_with("test-run", 1)


class TestGameManagerSingleton:
    """Test the ``set_game_manager``/``get_game_manager`` interface.

    The old code path had ``server_cli.py`` writing directly to
    ``api_server._game_manager``, a module attribute hop that made
    single-ownership of the GameManager unenforceable. All production
    code now goes through ``set_game_manager``, which rejects a
    silent overwrite with a different instance.
    """

    def test_set_and_get_roundtrip(self) -> None:
        import roc.reporting.api_server as mod
        from unittest.mock import MagicMock

        orig = mod._game_manager
        try:
            mod.set_game_manager(None)
            assert mod.get_game_manager() is None

            mgr = MagicMock()
            mod.set_game_manager(mgr)
            assert mod.get_game_manager() is mgr
        finally:
            mod._game_manager = orig

    def test_set_same_manager_twice_is_noop(self) -> None:
        """Re-installing the same instance is allowed (idempotent)."""
        import roc.reporting.api_server as mod
        from unittest.mock import MagicMock

        orig = mod._game_manager
        try:
            mod.set_game_manager(None)
            mgr = MagicMock()
            mod.set_game_manager(mgr)
            mod.set_game_manager(mgr)  # must not raise
            assert mod.get_game_manager() is mgr
        finally:
            mod._game_manager = orig

    def test_silent_overwrite_with_different_instance_raises(self) -> None:
        """Installing a second DIFFERENT GameManager is a programming error."""
        import roc.reporting.api_server as mod
        from unittest.mock import MagicMock

        orig = mod._game_manager
        try:
            mod.set_game_manager(None)
            mgr1 = MagicMock()
            mgr2 = MagicMock()
            mod.set_game_manager(mgr1)
            with pytest.raises(RuntimeError, match="already installed"):
                mod.set_game_manager(mgr2)
        finally:
            mod._game_manager = orig

    def test_clearing_to_none_always_allowed(self) -> None:
        """``set_game_manager(None)`` clears the slot and unblocks reinstall."""
        import roc.reporting.api_server as mod
        from unittest.mock import MagicMock

        orig = mod._game_manager
        try:
            mod.set_game_manager(None)
            mgr1 = MagicMock()
            mod.set_game_manager(mgr1)
            mod.set_game_manager(None)  # clear
            mgr2 = MagicMock()
            mod.set_game_manager(mgr2)  # must not raise
            assert mod.get_game_manager() is mgr2
        finally:
            mod._game_manager = orig


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


# ---------------------------------------------------------------------------
# Tests for 503 responses when _run_registry is None
# ---------------------------------------------------------------------------


class TestDataStoreNoneReturns503:
    """Test that endpoints return 503 when _run_registry is None."""

    @pytest.fixture(autouse=True)
    def _nullify_registry(self) -> Generator[None, None, None]:
        """Temporarily set _run_registry and _run_reader to None for these tests."""
        import roc.reporting.api_server as mod

        orig_registry = mod._run_registry
        orig_reader = mod._run_reader
        mod._run_registry = None
        mod._run_reader = None
        yield
        mod._run_registry = orig_registry
        mod._run_reader = orig_reader

    def test_list_runs_returns_empty(self, client: TestClient) -> None:
        """list_runs returns empty list when _run_registry is None."""
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_games_returns_503(self, client: TestClient) -> None:
        """list_games returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/games")
        assert resp.status_code == 503

    def test_get_step_returns_503(self, client: TestClient) -> None:
        """get_step returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/step/1")
        assert resp.status_code == 503

    def test_graph_history_returns_503(self, client: TestClient) -> None:
        """get_graph_history returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/graph-history")
        assert resp.status_code == 503

    def test_event_history_returns_503(self, client: TestClient) -> None:
        """get_event_history returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/event-history")
        assert resp.status_code == 503

    def test_intrinsics_history_returns_503(self, client: TestClient) -> None:
        """get_intrinsics_history returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/intrinsics-history")
        assert resp.status_code == 503

    def test_action_history_returns_503(self, client: TestClient) -> None:
        """get_action_history returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/action-history")
        assert resp.status_code == 503

    def test_resolution_history_returns_503(self, client: TestClient) -> None:
        """get_resolution_history returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/resolution-history")
        assert resp.status_code == 503

    def test_all_objects_returns_503(self, client: TestClient) -> None:
        """get_all_objects returns 503 when _run_registry is None."""
        resp = client.get("/api/runs/test-run/all-objects")
        assert resp.status_code == 503

    def test_save_bookmarks_returns_503(self, client: TestClient) -> None:
        """save_bookmarks returns 503 when _run_registry is None."""
        bookmarks = [{"step": 1, "game": 1, "annotation": "test", "created": "2026-01-01T00:00:00"}]
        resp = client.post("/api/runs/test-run/bookmarks", json=bookmarks)
        assert resp.status_code == 503

    def test_get_bookmarks_returns_empty(self, client: TestClient) -> None:
        """get_bookmarks returns empty list when _run_registry is None."""
        resp = client.get("/api/runs/test-run/bookmarks")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Schema endpoint tests
# ---------------------------------------------------------------------------


class TestGetSchema:
    """Test GET /api/runs/{run}/schema endpoint.

    Phase 1 bug-fix migration (BUG-C1): the endpoint reads through
    ``RunReader`` instead of ``DataStore``. The reader reads ``schema.json``
    directly from the run directory; tests now seed real files instead of
    mocking the store.
    """

    def test_returns_schema_from_store(self, client: TestClient, tmp_path: Path) -> None:
        """get_schema returns schema data from the store."""
        import json

        mock_schema = {"nodes": [{"label": "Object"}], "edges": [{"type": "HAS"}]}
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        (run_dir / "schema.json").write_text(json.dumps(mock_schema))
        resp = client.get("/api/runs/test-run/schema")
        assert resp.status_code == 200
        assert resp.json() == mock_schema

    def test_returns_404_when_schema_not_found(self, client: TestClient) -> None:
        """get_schema returns 404 when no schema exists for the run."""
        resp = client.get("/api/runs/test-run/schema")
        assert resp.status_code == 404

    def test_returns_503_when_no_registry(self, client: TestClient) -> None:
        """get_schema returns 503 when _run_registry is None."""
        import roc.reporting.api_server as mod

        orig_registry = mod._run_registry
        orig_reader = mod._run_reader
        mod._run_registry = None
        mod._run_reader = None
        try:
            resp = client.get("/api/runs/test-run/schema")
            assert resp.status_code == 503
        finally:
            mod._run_registry = orig_registry
            mod._run_reader = orig_reader


# ---------------------------------------------------------------------------
# Action map endpoint tests
# ---------------------------------------------------------------------------


class TestGetActionMap:
    """Test GET /api/runs/{run}/action-map endpoint.

    Phase 1 bug-fix migration (BUG-C1): the endpoint reads through
    ``RunReader`` instead of ``DataStore``. The reader reads
    ``action_map.json`` directly from the run directory.
    """

    def test_returns_action_map_from_store(self, client: TestClient, tmp_path: Path) -> None:
        """get_action_map returns action map data from the store."""
        import json

        mock_map = [{"id": 0, "name": "wait"}, {"id": 1, "name": "north"}]
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        (run_dir / "action_map.json").write_text(json.dumps(mock_map))
        resp = client.get("/api/runs/test-run/action-map")
        assert resp.status_code == 200
        assert resp.json() == mock_map

    def test_returns_empty_list_when_no_action_map(self, client: TestClient) -> None:
        """get_action_map returns empty list when action_map is None."""
        resp = client.get("/api/runs/test-run/action-map")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_503_when_no_registry(self, client: TestClient) -> None:
        """get_action_map returns 503 when _run_registry is None."""
        import roc.reporting.api_server as mod

        orig_registry = mod._run_registry
        orig_reader = mod._run_reader
        mod._run_registry = None
        mod._run_reader = None
        try:
            resp = client.get("/api/runs/test-run/action-map")
            assert resp.status_code == 503
        finally:
            mod._run_registry = orig_registry
            mod._run_reader = orig_reader


# ---------------------------------------------------------------------------
# Bookmark edge cases
# ---------------------------------------------------------------------------


class TestBookmarkEdgeCases:
    """Test bookmark endpoint edge cases."""

    def test_returns_empty_on_malformed_bookmarks_file(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """get_bookmarks returns empty list when bookmarks file is malformed."""
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
        mod._server_ready = __import__("threading").Event()
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(mod._capture_event_loop())
            assert mod._sio_loop is loop
            assert mod._server_ready.is_set()
            loop.close()
        finally:
            mod._sio_loop = orig_loop
            mod._server_ready = orig_ready


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
            with patch("roc.framework.config.Config.get", return_value=mock_cfg):
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
                patch("roc.framework.config.Config.get", return_value=mock_cfg),
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

    def test_does_not_add_middleware_in_server_mode(self) -> None:
        """Regression: start_dashboard must not touch the running FastAPI app
        when invoked from the game thread under server_cli.py.

        The game thread calls start_dashboard() from _game_main() for
        standalone-mode compatibility. Under server_cli.py, uvicorn is
        already serving `app`, so app.add_middleware() would raise
        `RuntimeError: Cannot add middleware after an application has started`
        and crash the game thread before gym.start() runs -- meaning no
        frames ever flow into the StepBuffer.
        """
        import roc.reporting.api_server as mod

        orig_started = mod._started
        orig_mgr = mod._game_manager
        mod._started = False
        mod._game_manager = MagicMock()
        try:
            mock_cfg = MagicMock()
            mock_cfg.dashboard_enabled = True
            with (
                patch("roc.framework.config.Config.get", return_value=mock_cfg),
                patch.object(mod.app, "add_middleware") as mock_add_mw,
            ):
                mod.start_dashboard()
            assert mock_add_mw.call_count == 0
            assert mod._started is True
        finally:
            mod._started = orig_started
            mod._game_manager = orig_mgr


# ---------------------------------------------------------------------------
# Socket.io subscribe_run / unsubscribe_run (Phase 4)
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
class TestSubscribeRunSocketHandlers:
    """Phase 4: Socket.io is now an invalidation channel.

    ``subscribe_run`` registers a per-sid callback on ``RunRegistry`` so
    every step push fans out as a tiny ``{run, step}`` ``step_added``
    payload to the originating client only. The handler must clean up
    the prior subscription on rebind and on disconnect.
    """

    def _setup_run(self, tmp_path: Path, run_name: str = "sub-run") -> tuple[Any, Any]:
        """Seed a registry entry by attaching a writer store."""
        import roc.reporting.api_server as mod
        from roc.reporting.ducklake_store import DuckLakeStore

        run_dir = tmp_path / run_name
        store = DuckLakeStore(run_dir, read_only=False)
        registry = mod._get_registry()
        assert registry is not None
        registry.attach_writer_store(run_name, store)
        return store, registry

    def test_subscribe_run_registers_callback_with_registry(self, tmp_path: Path) -> None:
        import asyncio

        import roc.reporting.api_server as mod

        store, registry = self._setup_run(tmp_path)
        try:
            mod._sio_subscriptions.clear()
            orig_loop = mod._sio_loop
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mod._sio_loop = mock_loop
            try:
                with patch("asyncio.run_coroutine_threadsafe") as mock_run:
                    asyncio.new_event_loop().run_until_complete(
                        mod.subscribe_run("sid-A", "sub-run")
                    )
                    # The subscription is recorded on the sid
                    assert "sid-A" in mod._sio_subscriptions
                    # Trigger a notification: should hop the loop and emit
                    registry.notify_subscribers("sub-run", 42)
                    mock_run.assert_called_once()
                    # Close the unawaited coroutine to avoid RuntimeWarning
                    coro = mock_run.call_args[0][0]
                    coro.close()
            finally:
                mod._sio_loop = orig_loop
                mod._sio_subscriptions.clear()
        finally:
            registry.detach_writer_store("sub-run")
            store.close()

    def test_subscribe_run_replaces_prior_subscription(self, tmp_path: Path) -> None:
        """Subscribing the same sid to a new run drops the old subscription."""
        import asyncio

        import roc.reporting.api_server as mod

        store, registry = self._setup_run(tmp_path, run_name="sub-r1")
        store2, _ = self._setup_run(tmp_path, run_name="sub-r2")
        try:
            mod._sio_subscriptions.clear()
            mod._sio_loop = MagicMock(is_running=MagicMock(return_value=True))
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(mod.subscribe_run("sid-X", "sub-r1"))
                first = mod._sio_subscriptions["sid-X"]
                loop.run_until_complete(mod.subscribe_run("sid-X", "sub-r2"))
                second = mod._sio_subscriptions["sid-X"]
                assert second is not first
            finally:
                mod._sio_subscriptions.clear()
        finally:
            registry.detach_writer_store("sub-r1")
            registry.detach_writer_store("sub-r2")
            store.close()
            store2.close()

    def test_unsubscribe_run_removes_sid_subscription(self, tmp_path: Path) -> None:
        import asyncio

        import roc.reporting.api_server as mod

        store, registry = self._setup_run(tmp_path)
        try:
            mod._sio_subscriptions.clear()
            mod._sio_loop = MagicMock(is_running=MagicMock(return_value=True))
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(mod.subscribe_run("sid-Y", "sub-run"))
                assert "sid-Y" in mod._sio_subscriptions
                loop.run_until_complete(mod.unsubscribe_run("sid-Y"))
                assert "sid-Y" not in mod._sio_subscriptions
            finally:
                mod._sio_subscriptions.clear()
        finally:
            registry.detach_writer_store("sub-run")
            store.close()

    def test_unsubscribe_run_unknown_sid_is_noop(self) -> None:
        import asyncio

        import roc.reporting.api_server as mod

        mod._sio_subscriptions.clear()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(mod.unsubscribe_run("never-subscribed"))
        # No exception, no state change
        assert mod._sio_subscriptions == {}

    def test_disconnect_drops_subscription(self, tmp_path: Path) -> None:
        """Disconnecting a client drops their per-sid run subscription."""
        import asyncio

        import roc.reporting.api_server as mod

        store, registry = self._setup_run(tmp_path)
        try:
            mod._sio_subscriptions.clear()
            mod._sio_loop = MagicMock(is_running=MagicMock(return_value=True))
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(mod.subscribe_run("sid-D", "sub-run"))
                assert "sid-D" in mod._sio_subscriptions
                loop.run_until_complete(mod.disconnect("sid-D"))
                assert "sid-D" not in mod._sio_subscriptions
            finally:
                mod._sio_subscriptions.clear()
        finally:
            registry.detach_writer_store("sub-run")
            store.close()

    def test_subscribe_run_ignores_empty_run_name(self) -> None:
        import asyncio

        import roc.reporting.api_server as mod

        mod._sio_subscriptions.clear()
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(mod.subscribe_run("sid-E", ""))
            assert "sid-E" not in mod._sio_subscriptions
        finally:
            mod._sio_subscriptions.clear()

    def test_step_added_payload_is_minimal_run_step_only(self, tmp_path: Path) -> None:
        """The new step_added event must contain only {run, step} -- no full StepData."""
        import asyncio

        import roc.reporting.api_server as mod

        store, registry = self._setup_run(tmp_path)
        try:
            mod._sio_subscriptions.clear()
            mod._sio_loop = MagicMock(is_running=MagicMock(return_value=True))
            try:
                with patch.object(mod.sio, "emit", return_value=MagicMock()) as mock_emit:

                    def fake_run_coro(coro: Any, _loop: Any) -> Any:
                        # Drive the coroutine far enough to capture the
                        # ``sio.emit`` call argument, then close it cleanly.
                        try:
                            coro.send(None)
                        except StopIteration:
                            pass
                        coro.close()
                        return MagicMock()

                    with patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(mod.subscribe_run("sid-P", "sub-run"))
                        registry.notify_subscribers("sub-run", 9)
                        # mock_emit should have been called with the minimal payload
                        assert mock_emit.call_count >= 1
                        args, kwargs = mock_emit.call_args
                        assert args[0] == "step_added"
                        payload = args[1]
                        assert payload == {"run": "sub-run", "step": 9}
                        # Targeted at the originating sid
                        assert kwargs.get("to") == "sid-P"
            finally:
                mod._sio_loop = None
                mod._sio_subscriptions.clear()
        finally:
            registry.detach_writer_store("sub-run")
            store.close()
