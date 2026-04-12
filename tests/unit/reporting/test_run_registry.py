"""Unit tests for the RunRegistry."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_registry import RunRegistry


def _seed_run(data_dir: Path, name: str, *, steps: int) -> None:
    """Create a real DuckLake run with ``steps`` rows in the screens table."""
    run_dir = data_dir / name
    store = DuckLakeStore(run_dir, read_only=False)
    try:
        records = [
            {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
            for s in range(1, steps + 1)
        ]
        store.insert("screens", records)
    finally:
        store.close()


class TestRunRegistryGet:
    def test_get_returns_none_for_unknown_run(self, tmp_path: Path) -> None:
        reg = RunRegistry(tmp_path)
        assert reg.get("does-not-exist") is None

    def test_get_lazy_loads_entry_from_data_dir(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=10)
        reg = RunRegistry(tmp_path)
        entry = reg.get("run-1")
        assert entry is not None
        assert entry.name == "run-1"
        assert entry.range.max == 10
        assert entry.range.min == 1
        assert entry.range.tail_growing is False

    def test_second_get_returns_cached_entry(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=5)
        reg = RunRegistry(tmp_path)
        e1 = reg.get("run-1")
        e2 = reg.get("run-1")
        assert e1 is e2  # same instance from cache


class TestRunRegistryMarkGrowing:
    def test_mark_growing_flips_tail_growing(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=3)
        reg = RunRegistry(tmp_path)
        reg.mark_growing("run-1", growing=True)
        entry = reg.get("run-1")
        assert entry is not None
        assert entry.range.tail_growing is True
        reg.mark_growing("run-1", growing=False)
        entry = reg.get("run-1")
        assert entry is not None
        assert entry.range.tail_growing is False

    def test_mark_growing_unknown_run_does_not_raise(self, tmp_path: Path) -> None:
        reg = RunRegistry(tmp_path)
        reg.mark_growing("nonexistent", growing=True)  # must not raise


class TestRunRegistryUpdateMaxStep:
    def test_update_max_step_advances_max(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=3)
        reg = RunRegistry(tmp_path)
        reg.get("run-1")  # warm cache
        reg.update_max_step("run-1", 7)
        entry = reg.get("run-1")
        assert entry is not None
        assert entry.range.max == 7

    def test_update_max_step_updates_summary_steps_and_status(self, tmp_path: Path) -> None:
        """Bug regression: update_max_step must keep entry.summary in sync.

        When a writer attaches to a brand-new run with zero rows, the summary
        gets ``steps=0, status="empty"``. As steps are pushed, the summary
        must reflect the growing step count and flip status to ``"ok"`` so
        that ``/api/runs`` does not label the run as "[empty]" while it is
        actively receiving data.
        """
        run_dir = tmp_path / "live-run"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            reg.attach_writer_store("live-run", store)
            entry = reg.get("live-run")
            assert entry is not None
            # Freshly attached with empty catalog: summary starts at 0/empty
            assert entry.summary.steps == 0
            assert entry.summary.status == "empty"

            # Simulate push_step advancing the max
            reg.update_max_step("live-run", 1)
            entry = reg.get("live-run")
            assert entry is not None
            assert entry.summary.steps == 1
            assert entry.summary.status == "ok"

            reg.update_max_step("live-run", 50)
            entry = reg.get("live-run")
            assert entry is not None
            assert entry.summary.steps == 50
            assert entry.summary.status == "ok"
        finally:
            reg.detach_writer_store("live-run")
            store.close()

    def test_update_max_step_unknown_run_is_noop(self, tmp_path: Path) -> None:
        reg = RunRegistry(tmp_path)
        reg.update_max_step("nonexistent", 5)  # must not raise

    def test_update_max_step_bumps_games_count_on_first_step(self, tmp_path: Path) -> None:
        """Regression for the UAT "0g, N steps" run label.

        ``attach_writer_store`` runs before any game_start event has been
        written to the catalog, so the initial ``summary.games`` count
        is 0. As steps start flowing, the UI label was stuck on "0g"
        while the step count climbed -- visible as e.g. "0g, 237 steps"
        in the run selector dropdown, which the UAT flagged as
        inconsistent. Fix: bump ``games`` to at least 1 on the first
        step push so the label reflects reality while waiting for the
        parquet exporter to flush the real row count.
        """
        run_dir = tmp_path / "live-games-run"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            reg.attach_writer_store("live-games-run", store)
            entry = reg.get("live-games-run")
            assert entry is not None
            # Freshly attached with empty catalog
            assert entry.summary.games == 0

            reg.update_max_step("live-games-run", 1)
            entry = reg.get("live-games-run")
            assert entry is not None
            assert entry.summary.games == 1

            # Subsequent pushes must not increment beyond 1 (we rely on
            # ``list_games`` queries for the real count once data has
            # been flushed to the catalog).
            reg.update_max_step("live-games-run", 50)
            entry = reg.get("live-games-run")
            assert entry is not None
            assert entry.summary.games == 1
        finally:
            reg.detach_writer_store("live-games-run")
            store.close()


class TestRunRegistrySubscribe:
    def test_subscribe_returns_no_op_unsubscribe_for_unknown_run(self, tmp_path: Path) -> None:
        reg = RunRegistry(tmp_path)
        unsub = reg.subscribe("does-not-exist", lambda step: None)
        unsub()  # must not raise

    def test_subscribe_callback_invoked_on_notify(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=1)
        reg = RunRegistry(tmp_path)
        received: list[int] = []
        reg.subscribe("run-1", lambda step: received.append(step))
        reg.notify_subscribers("run-1", 5)
        assert received == [5]

    def test_unsubscribe_stops_callbacks(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=1)
        reg = RunRegistry(tmp_path)
        received: list[int] = []
        unsub = reg.subscribe("run-1", lambda step: received.append(step))
        unsub()
        reg.notify_subscribers("run-1", 5)
        assert received == []

    def test_subscriber_exception_does_not_break_other_subscribers(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=1)
        reg = RunRegistry(tmp_path)

        def boom(step: int) -> None:
            raise RuntimeError("boom")

        reg.subscribe("run-1", boom)
        good: list[int] = []
        reg.subscribe("run-1", good.append)
        reg.notify_subscribers("run-1", 7)
        assert good == [7]

    def test_notify_subscribers_dispatches_outside_lock(self, tmp_path: Path) -> None:
        """A re-entrant callback that calls reg.get(...) must not deadlock."""
        _seed_run(tmp_path, "run-1", steps=1)
        reg = RunRegistry(tmp_path)
        observed: list[Any] = []

        def cb(step: int) -> None:
            observed.append(reg.get("run-1"))

        reg.subscribe("run-1", cb)
        reg.notify_subscribers("run-1", 1)
        assert len(observed) == 1
        assert observed[0] is not None


class TestRunRegistryAttachDetach:
    def test_attach_writer_store_flips_tail_growing(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=3)
        reg = RunRegistry(tmp_path)
        store = DuckLakeStore(tmp_path / "run-1", read_only=False)
        try:
            reg.attach_writer_store("run-1", store)
            entry = reg.get("run-1")
            assert entry is not None
            assert entry.range.tail_growing is True
            assert entry.store is store
        finally:
            reg.detach_writer_store("run-1")
            store.close()

    def test_detach_writer_store_returns_to_read_only(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=3)
        reg = RunRegistry(tmp_path)
        store = DuckLakeStore(tmp_path / "run-1", read_only=False)
        reg.attach_writer_store("run-1", store)
        store.close()  # writer is done
        reg.detach_writer_store("run-1")
        entry = reg.get("run-1")
        assert entry is not None
        assert entry.range.tail_growing is False
        # Confirm we can still read steps via the new entry's store
        assert entry.range.max == 3

    def test_attach_then_detach_unknown_run_does_not_raise(self, tmp_path: Path) -> None:
        reg = RunRegistry(tmp_path)
        # detach before any attach is a no-op
        reg.detach_writer_store("nope")

    def test_post_detach_placeholder_with_subscribers_reloads_on_get(self, tmp_path: Path) -> None:
        """Subscriber-preserving placeholder must NOT serve stale 0/0 forever.

        ``detach_writer_store`` deliberately leaves a stub entry behind
        when the run has live subscribers, so that subscriber bindings
        survive the brief window between writer teardown and the next
        ``get`` call. The stub has ``store=None`` and ``range=0/0``.

        The contract: the *next* ``get`` for that name must lazily reload
        from disk and replace the placeholder with a fresh entry that
        reflects the persisted catalog state. Without this, every
        ``/api/runs/<run>/step-range`` call returns ``0/0`` for the rest
        of the process lifetime, and ``/step/N`` returns "run store
        unavailable" 500 -- the failure mode caught during Phase 7
        validation when the dashboard auto-selected a just-stopped
        live run as the default landing target.
        """
        _seed_run(tmp_path, "live-then-historical", steps=2)
        reg = RunRegistry(tmp_path)

        # Open the writer store BEFORE subscribing -- we cannot lazy-load
        # a read-only store first because the writer's store and the
        # read-only store would collide on the catalog file handle in
        # the same Python process. In production this is fine because
        # the writer lives in the game subprocess.
        store = DuckLakeStore(tmp_path / "live-then-historical", read_only=False)
        observed: list[int] = []
        try:
            reg.attach_writer_store("live-then-historical", store)
            # Subscribe AFTER attach so the registry already has the
            # writer's store as the entry's store -- subscribe() reuses
            # the existing entry instead of calling _load.
            reg.subscribe("live-then-historical", observed.append)
            store.insert(
                "screens",
                [
                    {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                    for s in range(3, 11)
                ],
            )
            store.checkpoint()
        finally:
            reg.detach_writer_store("live-then-historical")
            store.close()

        # Sanity check: the stub IS in place with the stale 0/0 range.
        stub = reg._entries.get("live-then-historical")
        assert stub is not None
        assert stub.store is None
        assert stub.range.max == 0
        assert stub.subscribers, "subscriber must be preserved across detach"

        # The next get() must lazily reload and return a fresh entry
        # whose range reflects the persisted catalog state.
        fresh = reg.get("live-then-historical")
        assert fresh is not None
        assert fresh.store is not None
        assert fresh.run_store is not None
        assert fresh.range.max == 10
        assert fresh.range.min == 1
        assert fresh.range.tail_growing is False

        # Subscribers must survive the reload so callbacks still fire.
        assert fresh.subscribers, "subscribers must be preserved across reload"
        reg.notify_subscribers("live-then-historical", 10)
        assert observed == [10]


class TestRunRegistryList:
    def test_list_returns_runs_in_directory(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-a", steps=20)
        _seed_run(tmp_path, "run-b", steps=20)
        reg = RunRegistry(tmp_path)
        summaries = reg.list()
        names = {s.name for s in summaries}
        assert "run-a" in names
        assert "run-b" in names

    def test_list_empty_data_dir(self, tmp_path: Path) -> None:
        reg = RunRegistry(tmp_path)
        assert reg.list() == []


class TestRunRegistryConcurrency:
    def test_concurrent_get_does_not_deadlock(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "run-1", steps=10)
        reg = RunRegistry(tmp_path)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    entry = reg.get("run-1")
                    assert entry is not None
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestRunRegistryCorruptCatalog:
    def test_get_returns_corrupt_entry_for_invalid_catalog(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "broken-run"
        run_dir.mkdir()
        # Create a catalog.duckdb file that is not a valid DuckDB database
        (run_dir / "catalog.duckdb").write_bytes(b"not a real ducklake catalog")
        reg = RunRegistry(tmp_path)
        entry = reg.get("broken-run")
        assert entry is not None
        assert entry.summary.status == "corrupt"
        assert entry.summary.error is not None
        # The store and run_store should be None for a corrupt entry
        assert entry.store is None
        assert entry.run_store is None

    def test_get_returns_none_for_directory_without_catalog(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no-catalog-run"
        run_dir.mkdir()
        # No catalog.duckdb at all
        reg = RunRegistry(tmp_path)
        entry = reg.get("no-catalog-run")
        assert entry is None


class TestRunRegistryStaleEntry:
    def test_stale_entry_reloaded_on_mtime_change(self, tmp_path: Path) -> None:
        """Touch the catalog file; next get() returns refreshed range.

        Models a writer that takes over the catalog, mutates it, releases
        it, and the next get() reloads the closed-run entry from disk.
        """
        _seed_run(tmp_path, "stale-run", steps=3)
        reg = RunRegistry(tmp_path)
        entry = reg.get("stale-run")
        assert entry is not None
        assert entry.range.max == 3

        # Force the registry to drop its read-only store so the writer
        # can take over the catalog file. (In production this happens
        # implicitly because attach_writer_store disposes the prior store.)
        prior_store = entry.store
        if prior_store is not None:
            prior_store.close()
        reg._entries.clear()

        writer_store = DuckLakeStore(tmp_path / "stale-run", read_only=False)
        try:
            reg.attach_writer_store("stale-run", writer_store)
            records = [
                {"step": s, "game_number": 1, "timestamp": s * 1000, "body": "{}"}
                for s in range(4, 11)
            ]
            writer_store.insert("screens", records)
            writer_store.checkpoint()
        finally:
            reg.detach_writer_store("stale-run")
            writer_store.close()

        # Next get() should reload from disk and observe the new max.
        entry2 = reg.get("stale-run")
        assert entry2 is not None
        assert entry2.range.max == 10


class TestRunRegistryTailGrowing:
    """Phase 3: ``StepRange.tail_growing`` is the only liveness signal."""

    def test_step_range_includes_tail_growing_false_by_default(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r", steps=3)
        reg = RunRegistry(tmp_path)
        entry = reg.get("r")
        assert entry is not None
        assert entry.range.tail_growing is False

    def test_mark_growing_propagates_to_subsequent_get(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r", steps=3)
        reg = RunRegistry(tmp_path)
        reg.mark_growing("r", growing=True)
        entry = reg.get("r")
        assert entry is not None
        assert entry.range.tail_growing is True

    def test_attach_writer_store_sets_tail_growing_true(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r", steps=2)
        reg = RunRegistry(tmp_path)
        store = DuckLakeStore(tmp_path / "r", read_only=False)
        try:
            reg.attach_writer_store("r", store)
            entry = reg.get("r")
            assert entry is not None
            assert entry.range.tail_growing is True
        finally:
            reg.detach_writer_store("r")
            store.close()

    def test_detach_writer_store_resets_tail_growing_false(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "r", steps=2)
        reg = RunRegistry(tmp_path)
        store = DuckLakeStore(tmp_path / "r", read_only=False)
        reg.attach_writer_store("r", store)
        store.close()
        reg.detach_writer_store("r")
        entry = reg.get("r")
        assert entry is not None
        assert entry.range.tail_growing is False


class TestRunRegistryListFiltering:
    def test_list_filters_by_status_default(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "ok-run", steps=20)
        # Create an empty catalog (status="empty")
        (tmp_path / "empty-run").mkdir()
        store = DuckLakeStore(tmp_path / "empty-run", read_only=False)
        store.close()
        reg = RunRegistry(tmp_path)
        # Default: only ok runs
        results = reg.list()
        names = {r.name for r in results}
        assert "ok-run" in names
        # Empty runs filtered out by default
        assert "empty-run" not in names

    def test_list_include_all_returns_all(self, tmp_path: Path) -> None:
        _seed_run(tmp_path, "ok-run", steps=20)
        (tmp_path / "empty-run").mkdir()
        store = DuckLakeStore(tmp_path / "empty-run", read_only=False)
        store.close()
        reg = RunRegistry(tmp_path)
        results = reg.list(include_all=True)
        names = {r.name for r in results}
        assert "ok-run" in names
        assert "empty-run" in names
