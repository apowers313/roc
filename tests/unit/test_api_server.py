"""Unit tests for the FastAPI dashboard API server."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from roc.reporting.api_server import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _setup_data_dir(tmp_path: Path) -> Any:
    """Point the API at a temp directory."""
    import roc.reporting.api_server as mod

    orig = mod._data_dir
    mod._data_dir = tmp_path
    mod._run_stores.clear()
    yield
    mod._data_dir = orig
    mod._run_stores.clear()


class TestListRuns:
    def test_returns_empty_when_no_runs(self, client: TestClient) -> None:
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_runs_when_present(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # Create a mock run directory with the structure RunStore.list_runs expects
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        # RunStore.list_runs checks for DuckLakeStore.is_valid_run
        # which we need to mock
        with patch("roc.reporting.run_store.RunStore.list_runs", return_value=["test-run"]):
            with patch(
                "roc.reporting.api_server._get_store"
            ) as mock_store:
                store = MagicMock()
                store.list_games.return_value = MagicMock(__len__=lambda s: 2)
                store.list_games.return_value.__len__ = lambda s: 2  # type: ignore[attr-defined]
                store.step_count.return_value = 100
                mock_store.return_value = store
                resp = client.get("/api/runs")
                assert resp.status_code == 200


class TestStepRange:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step-range")
        assert resp.status_code == 404


class TestGetStep:
    def test_returns_404_for_missing_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent/step/1")
        assert resp.status_code == 404


class TestBookmarks:
    def test_returns_empty_when_no_bookmarks(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        run_dir = tmp_path / "test-run"
        run_dir.mkdir()
        resp = client.get("/api/runs/test-run/bookmarks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_save_and_load_bookmarks(
        self, client: TestClient, tmp_path: Path
    ) -> None:
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
