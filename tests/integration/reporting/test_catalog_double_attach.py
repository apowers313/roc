# mypy: disable-error-code="no-untyped-def"

"""Regression tests for the unified-run RunReader methods that replaced the
legacy DataStore catalog double-attach path.

Background (BUG-C1 in design/dashboard-bugfix-plan.md):

The legacy code path opened a separate ``DuckLakeStore(read_only=True)``
inside ``DataStore._get_run_store`` even after ``RunRegistry._load`` had
already opened one (or after ``RunRegistry.attach_writer_store`` installed
the writer's store). DuckDB rejects the second attach with
``BinderException: Unique file handle conflict`` (Phase 0 spike confirmed
this for the writer-vs-reader case in ``tmp/ducklake_concurrency_spike.py``).

The fix is to route the four affected endpoints (``/games``, ``/all-objects``,
``/schema``, ``/action-map``) through ``RunReader``, which reads from the
single shared store owned by ``RunRegistry``. These tests confirm that:

1. The new ``RunReader`` methods exist and read the right data.
2. They never open a second ``DuckLakeStore`` against the same catalog file
   (whether the run is closed-via-registry or active-via-writer).
3. ``get_schema`` and ``get_action_map`` are pure filesystem reads -- they
   never touch DuckLake at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_reader import RunReader
from roc.reporting.run_registry import RunRegistry
from roc.reporting.step_cache import StepCache


def _seed_run_via_writer(data_dir: Path, name: str, *, steps: int) -> None:
    """Create a run by opening a writer, inserting screen rows, and closing.

    Mirrors the helper in ``test_run_reader_integration.py``.
    """
    run_dir = data_dir / name
    store = DuckLakeStore(run_dir, read_only=False)
    try:
        records = [
            {
                "step": s,
                "game_number": 1,
                "timestamp": s * 1000,
                "body": "{}",
            }
            for s in range(1, steps + 1)
        ]
        store.insert("screens", records)
    finally:
        store.close()


def _make_reader(tmp_path: Path) -> tuple[RunReader, RunRegistry]:
    registry = RunRegistry(tmp_path)
    reader = RunReader(registry, StepCache(capacity=100))
    return reader, registry


# ---------------------------------------------------------------------------
# list_games
# ---------------------------------------------------------------------------


class TestRunReaderListGames:
    def test_list_games_for_closed_run_returns_typed_dicts(self, tmp_path: Path) -> None:
        """Closed run: registry opens read_only=True, RunReader reads from
        the shared entry.run_store. Must NOT open a second store."""
        _seed_run_via_writer(tmp_path, "closed-run", steps=5)
        reader, registry = _make_reader(tmp_path)
        # Force the registry to open the closed-run store first
        registry.get("closed-run")

        games = reader.list_games("closed-run")
        assert isinstance(games, list)
        assert len(games) >= 1
        first = games[0]
        # The shape must match the GameSummary fields used by the API
        assert "game_number" in first
        assert "steps" in first
        assert "start_ts" in first
        assert "end_ts" in first
        assert int(first["game_number"]) == 1
        assert int(first["steps"]) == 5

    def test_list_games_for_active_run_uses_writer_store(self, tmp_path: Path) -> None:
        """Active run: writer is open, registry has writer's store via
        ``attach_writer_store``. RunReader.list_games must read through the
        writer's instance -- the only valid second-attach is no second attach
        at all."""
        _seed_run_via_writer(tmp_path, "active-run", steps=3)
        reader, registry = _make_reader(tmp_path)

        store = DuckLakeStore(tmp_path / "active-run", read_only=False)
        try:
            registry.attach_writer_store("active-run", store)
            games = reader.list_games("active-run")
            assert isinstance(games, list)
            assert len(games) >= 1
            assert int(games[0]["game_number"]) == 1
            assert int(games[0]["steps"]) == 3
        finally:
            registry.detach_writer_store("active-run")
            store.close()

    def test_list_games_unknown_run_raises_file_not_found(self, tmp_path: Path) -> None:
        reader, _ = _make_reader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.list_games("does-not-exist")


# ---------------------------------------------------------------------------
# get_all_objects
# ---------------------------------------------------------------------------


class TestRunReaderGetAllObjects:
    def test_get_all_objects_for_closed_run_returns_list(self, tmp_path: Path) -> None:
        """Closed run with no events table: must return ``[]`` rather than
        raising. The legacy DataStore path was opening a second
        DuckLakeStore here -- the fix routes through the registry's shared
        store."""
        _seed_run_via_writer(tmp_path, "closed-run", steps=4)
        reader, registry = _make_reader(tmp_path)
        registry.get("closed-run")

        objects = reader.get_all_objects("closed-run")
        assert isinstance(objects, list)

    def test_get_all_objects_for_active_run_uses_writer_store(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "active-run", steps=3)
        reader, registry = _make_reader(tmp_path)

        store = DuckLakeStore(tmp_path / "active-run", read_only=False)
        try:
            registry.attach_writer_store("active-run", store)
            objects = reader.get_all_objects("active-run")
            assert isinstance(objects, list)
        finally:
            registry.detach_writer_store("active-run")
            store.close()

    def test_get_all_objects_unknown_run_raises_file_not_found(self, tmp_path: Path) -> None:
        reader, _ = _make_reader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.get_all_objects("does-not-exist")


# ---------------------------------------------------------------------------
# get_schema (filesystem-only)
# ---------------------------------------------------------------------------


class TestRunReaderGetSchema:
    def test_get_schema_returns_dict_when_file_exists(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "schema-run", steps=2)
        schema_dict = {"nodes": [{"label": "Object"}], "edges": []}
        (tmp_path / "schema-run" / "schema.json").write_text(json.dumps(schema_dict))

        reader, _ = _make_reader(tmp_path)
        result = reader.get_schema("schema-run")
        assert result == schema_dict

    def test_get_schema_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "schemaless-run", steps=2)
        reader, _ = _make_reader(tmp_path)
        assert reader.get_schema("schemaless-run") is None

    def test_get_schema_does_not_open_ducklake(self, tmp_path: Path) -> None:
        """``get_schema`` is filesystem-only. It must not require any
        ``RunRegistry`` entry to exist (which would open a DuckLake store)."""
        run_dir = tmp_path / "isolated-run"
        run_dir.mkdir()
        schema_dict = {"version": 1}
        (run_dir / "schema.json").write_text(json.dumps(schema_dict))

        reader, registry = _make_reader(tmp_path)
        result = reader.get_schema("isolated-run")
        assert result == schema_dict
        # No registry entry should have been created as a side effect
        assert "isolated-run" not in registry._entries

    def test_get_schema_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "broken-schema"
        run_dir.mkdir()
        (run_dir / "schema.json").write_text("{not valid json}")
        reader, _ = _make_reader(tmp_path)
        assert reader.get_schema("broken-schema") is None


# ---------------------------------------------------------------------------
# get_action_map (filesystem-only)
# ---------------------------------------------------------------------------


class TestRunReaderGetActionMap:
    def test_get_action_map_returns_list_when_file_exists(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "action-run", steps=2)
        action_map = [{"id": 0, "name": "north"}, {"id": 1, "name": "south"}]
        (tmp_path / "action-run" / "action_map.json").write_text(json.dumps(action_map))

        reader, _ = _make_reader(tmp_path)
        result = reader.get_action_map("action-run")
        assert result == action_map

    def test_get_action_map_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "no-action-map", steps=2)
        reader, _ = _make_reader(tmp_path)
        assert reader.get_action_map("no-action-map") is None

    def test_get_action_map_does_not_open_ducklake(self, tmp_path: Path) -> None:
        """``get_action_map`` is filesystem-only -- no DuckLake required."""
        run_dir = tmp_path / "actionmap-isolated"
        run_dir.mkdir()
        action_map = [{"id": 5, "name": "wait"}]
        (run_dir / "action_map.json").write_text(json.dumps(action_map))

        reader, registry = _make_reader(tmp_path)
        result = reader.get_action_map("actionmap-isolated")
        assert result == action_map
        assert "actionmap-isolated" not in registry._entries

    def test_get_action_map_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "broken-action-map"
        run_dir.mkdir()
        (run_dir / "action_map.json").write_text("{not valid json}")
        reader, _ = _make_reader(tmp_path)
        assert reader.get_action_map("broken-action-map") is None


# ---------------------------------------------------------------------------
# Catalog double-attach regression
# ---------------------------------------------------------------------------


class TestNoCatalogDoubleAttach:
    """The load-bearing tests: after the registry has opened (or attached)
    the run's store, the new RunReader paths must NOT open a second store
    against the same catalog file. We don't catch the exception name; we
    just verify the calls succeed and return real data."""

    def test_list_games_after_registry_load_does_not_double_attach(self, tmp_path: Path) -> None:
        _seed_run_via_writer(tmp_path, "double-attach-list", steps=5)
        reader, registry = _make_reader(tmp_path)
        # Force the registry to open the closed-run store
        entry = registry.get("double-attach-list")
        assert entry is not None
        # Same call repeated -- the second one is the one that used to crash
        for _ in range(2):
            games = reader.list_games("double-attach-list")
            assert len(games) >= 1

    def test_get_all_objects_after_registry_load_does_not_double_attach(
        self, tmp_path: Path
    ) -> None:
        _seed_run_via_writer(tmp_path, "double-attach-objects", steps=5)
        reader, registry = _make_reader(tmp_path)
        entry = registry.get("double-attach-objects")
        assert entry is not None
        for _ in range(2):
            objects = reader.get_all_objects("double-attach-objects")
            assert isinstance(objects, list)

    def test_list_games_during_active_writer_does_not_double_attach(self, tmp_path: Path) -> None:
        """The Phase 0 finding scenario: a writer is open, then a read
        path tries to open a second DuckLakeStore. The new RunReader path
        must reuse the writer's instance via the registry."""
        _seed_run_via_writer(tmp_path, "writer-list", steps=4)
        reader, registry = _make_reader(tmp_path)
        store = DuckLakeStore(tmp_path / "writer-list", read_only=False)
        try:
            registry.attach_writer_store("writer-list", store)
            for _ in range(2):
                games = reader.list_games("writer-list")
                assert len(games) >= 1
        finally:
            registry.detach_writer_store("writer-list")
            store.close()
