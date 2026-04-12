"""Unit tests for RunWriter (Phase 3 of unified-run architecture)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_registry import RunRegistry
from roc.reporting.run_store import StepData
from roc.reporting.run_writer import RunWriter
from roc.reporting.step_cache import StepCache


def _make_step(step: int, *, game_number: int = 1) -> StepData:
    return StepData(step=step, game_number=game_number)


class TestRunWriterInit:
    def test_init_attaches_writer_store_to_registry(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            try:
                entry = reg.get("run-1")
                assert entry is not None
                assert entry.range.tail_growing is True
                assert entry.store is store
            finally:
                writer.close()
        finally:
            store.close()


class TestRunWriterPushStep:
    def test_push_step_writes_through_to_cache(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            try:
                data = _make_step(1)
                writer.push_step(data)
                assert cache.get("run-1", 1) is data
            finally:
                writer.close()
        finally:
            store.close()

    def test_push_step_advances_registry_max_step(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            try:
                writer.push_step(_make_step(1))
                writer.push_step(_make_step(7))
                entry = reg.get("run-1")
                assert entry is not None
                assert entry.range.max == 7
                assert entry.range.tail_growing is True
            finally:
                writer.close()
        finally:
            store.close()

    def test_push_step_does_not_call_exporter_for_pure_writethrough(self, tmp_path: Path) -> None:
        """The exporter is the canonical write path; push_step does not
        directly call it (the OTel pipeline already handles that). The
        writer just keeps the cache and registry warm."""
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            try:
                writer.push_step(_make_step(1))
                exporter.queue.assert_not_called()
            finally:
                writer.close()
        finally:
            store.close()

    def test_push_step_notifies_subscribers(self, tmp_path: Path) -> None:
        """Phase 4: push_step must call notify_subscribers so the api_server's
        Socket.io subscribe_run handler can broadcast a {run, step} payload."""
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            try:
                received: list[int] = []
                reg.subscribe("run-1", received.append)
                writer.push_step(_make_step(1))
                writer.push_step(_make_step(2))
                writer.push_step(_make_step(3))
                assert received == [1, 2, 3]
            finally:
                writer.close()
        finally:
            store.close()

    def test_push_step_notify_swallows_subscriber_errors(self, tmp_path: Path) -> None:
        """A failing subscriber must not break the push (game loop must keep running)."""
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            try:

                def boom(step: int) -> None:
                    raise RuntimeError("subscriber boom")

                reg.subscribe("run-1", boom)
                # Must not raise
                writer.push_step(_make_step(1))
                # Cache and registry max should still be advanced
                assert cache.get("run-1", 1) is not None
                entry = reg.get("run-1")
                assert entry is not None
                assert entry.range.max == 1
            finally:
                writer.close()
        finally:
            store.close()


class TestRunWriterClose:
    def test_close_detaches_writer_store_from_registry(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            writer.push_step(_make_step(1))
            writer.close()
            store.close()
            entry = reg.get("run-1")
            assert entry is not None
            assert entry.range.tail_growing is False
        finally:
            try:
                store.close()
            except Exception:
                pass

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run-1"
        store = DuckLakeStore(run_dir, read_only=False)
        try:
            reg = RunRegistry(tmp_path)
            cache = StepCache()
            exporter = MagicMock()
            writer = RunWriter("run-1", reg, cache, exporter, store)
            writer.close()
            writer.close()  # must not raise
        finally:
            store.close()
