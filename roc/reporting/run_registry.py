"""Single owner of all per-run state for the unified-run dashboard.

The registry indexes per-run entries by name and owns the SINGLE
``DuckLakeStore`` for each run. Phase 0 of the unified-run architecture
proved that we cannot open a separate ``read_only=True`` ``DuckLakeStore``
against the same catalog file in the same process while the writer is
active -- DuckDB rejects it as ``BinderException: Unique file handle
conflict``. The registry is therefore the place that owns the SINGLE
``DuckLakeStore`` for each run:

- ``tail_growing == True`` -> the writer's instance, installed via
  ``attach_writer_store``. ``RunReader`` reads from this instance under its
  existing ``_lock`` on cache miss.
- ``tail_growing == False`` -> a fresh ``read_only=True`` ``DuckLakeStore``
  opened lazily by ``_load``.

Never construct a separate ``read_only=True`` ``DuckLakeStore`` for an
active run. See ``roc/reporting/CLAUDE.md`` for the load-bearing invariant.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from roc.framework.logger import logger
from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_store import RunStore
from roc.reporting.types import RunSummary, StepRange

Unsubscribe = Callable[[], None]


@dataclass
class _RunEntry:
    """Per-run state owned by ``RunRegistry``.

    The ``store`` field is the SINGLE ``DuckLakeStore`` for this run. Its
    provenance depends on ``range.tail_growing`` (Phase 0 enforces this):

    - ``tail_growing == True``: ``store`` is the same ``DuckLakeStore``
      instance held by the active ``RunWriter``, installed via
      ``attach_writer_store``. Do NOT construct a separate
      ``read_only=True`` instance for the same catalog file -- DuckDB
      rejects the second attach with ``Unique file handle conflict``.
    - ``tail_growing == False``: ``store`` is a fresh ``read_only=True``
      ``DuckLakeStore`` opened lazily by ``_load`` and owned by this entry.
      The next call to ``attach_writer_store`` replaces it with the
      writer's instance.
    """

    name: str
    summary: RunSummary
    store: DuckLakeStore | None
    run_store: RunStore | None
    range: StepRange
    mtime: float
    owns_store: bool = False
    subscribers: list[Callable[[int], None]] = field(default_factory=list)


class RunRegistry:
    """Single point of ownership for all per-run state.

    One ``threading.RLock`` covers everything. Notifications snapshot the
    subscribers under the lock and dispatch outside the lock so a re-entrant
    callback (e.g. one that calls ``get`` again) cannot deadlock.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._entries: dict[str, _RunEntry] = {}
        self._lock = threading.RLock()
        self._summary_alias_counter = 0

    @property
    def data_dir(self) -> Path:
        """The root data directory containing per-run subdirectories."""
        return self._data_dir

    # -------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------

    def get(self, name: str) -> _RunEntry | None:
        """Return the registry entry for ``name`` or ``None`` if missing.

        On a cache miss this lazily opens a ``read_only=True``
        ``DuckLakeStore`` (the closed-run path validated by Phase 0). On a
        cache hit, the entry's ``mtime`` is compared against the on-disk
        catalog: if the catalog has been touched since the cached read, the
        entry is reloaded so the range is fresh.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is not None:
                # When tail_growing is True, the writer owns the catalog
                # file. We MUST NOT try to reopen ``read_only=True`` here
                # -- DuckDB rejects the second attach with a unique file
                # handle conflict (Phase 0 finding). The range is kept
                # fresh by ``update_max_step`` from the writer side.
                if entry.range.tail_growing:
                    return entry
                # Post-detach placeholder: ``detach_writer_store`` leaves
                # behind a stub entry with ``store=None`` and ``range=0/0``
                # so subscriber bindings survive the writer teardown
                # window. By the time the next ``get`` arrives the writer
                # has fully released the catalog, so we can lazily open a
                # fresh ``read_only=True`` store and refresh the range.
                # Without this, the entry is stuck at 0/0 forever and
                # ``/step-range`` / ``/step/N`` return stale values
                # ("run store unavailable") for the rest of the process
                # lifetime.
                if entry.store is None and not entry.range.tail_growing:
                    fresh = self._load(name)
                    if fresh is not None:
                        fresh.subscribers = entry.subscribers
                        self._entries[name] = fresh
                        return fresh
                    return entry
                if entry.owns_store and self._stale(entry):
                    self._dispose_entry_store(entry)
                    fresh = self._load(name)
                    if fresh is not None:
                        # Preserve subscribers across reload
                        fresh.subscribers = entry.subscribers
                        self._entries[name] = fresh
                    else:
                        del self._entries[name]
                        return None
                    return fresh
                return entry
            entry = self._load(name)
            if entry is not None:
                self._entries[name] = entry
            return entry

    def list(
        self,
        *,
        min_steps: int = 10,
        include_all: bool = False,
    ) -> list[RunSummary]:
        """Return summaries for every run in the data directory.

        Every run is returned with a ``status`` field
        (``ok``/``empty``/``short``/``corrupt``). By default only
        ``status="ok"`` runs that clear the ``min_steps`` bar are returned.
        Pass ``include_all=True`` to receive every run including
        ``empty``, ``short``, and ``corrupt`` -- this is what the
        dashboard's "Show all runs" toggle uses to surface runs that
        would otherwise vanish silently.

        ``ok`` runs whose ``steps`` are below ``min_steps`` are demoted
        to ``status="short"``. Tail-growing (live) runs are exempt from
        the demotion -- they can have 0 steps the instant a game starts
        and we still want them visible.
        """
        names = RunStore.list_runs(self._data_dir)
        names.reverse()
        results: list[RunSummary] = []
        for name in names:
            entry = self.get(name)
            if entry is None:
                continue
            summary = entry.summary
            tail_growing = entry.range.tail_growing
            if not tail_growing and summary.status == "ok" and summary.steps < min_steps:
                summary = summary.model_copy(update={"status": "short"})
            if include_all or summary.status == "ok" or tail_growing:
                results.append(summary)
        return results

    # -------------------------------------------------------------------
    # State mutation
    # -------------------------------------------------------------------

    def mark_growing(self, name: str, *, growing: bool) -> None:
        """Flip the entry's ``tail_growing`` flag without touching the store.

        Used by tests and by code that needs to advertise a run as live
        without installing a writer's store. The store ownership rule
        (writer's instance for ``tail_growing == True``) is enforced by
        ``attach_writer_store`` / ``detach_writer_store``, not this method.
        """
        with self._lock:
            entry = self._entries.get(name) or self._load(name)
            if entry is None:
                return
            entry.range = StepRange(
                min=entry.range.min,
                max=entry.range.max,
                tail_growing=growing,
            )
            self._entries[name] = entry

    def update_max_step(self, name: str, step: int) -> None:
        """Advance the entry's max step (called by the writer's push).

        Also keeps ``entry.summary.steps``, ``entry.summary.status``, and
        ``entry.summary.games`` in sync so ``/api/runs`` returns accurate
        counts while the run is actively receiving data (not stale
        "empty" from attach time). ``attach_writer_store`` runs before
        any game_start event has been written to the catalog, so the
        initial ``games`` count is almost always 0 -- we bump it to at
        least 1 as soon as the first step arrives, matching the reality
        that step data implies at least one game.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry.range = StepRange(
                min=entry.range.min if entry.range.min > 0 else 1,
                max=step,
                tail_growing=entry.range.tail_growing,
            )
            updates: dict[str, Any] = {}
            if entry.summary.steps != step:
                updates["steps"] = step
                if step > 0:
                    updates["status"] = "ok"
            # Games count starts at 0 because attach runs before the
            # parquet exporter has flushed any rows. Once steps start
            # flowing, we know there is at least one game live.
            if step > 0 and entry.summary.games < 1:
                updates["games"] = 1
            if updates:
                entry.summary = entry.summary.model_copy(update=updates)

    def attach_writer_store(self, name: str, store: DuckLakeStore) -> None:
        """Install the writer's ``DuckLakeStore`` as the entry's read store.

        Replaces any existing read-only store for the run. The writer's
        ``_lock`` serializes all reads against concurrent writes for the
        lifetime of the writer. Flips ``tail_growing`` on.

        Phase 0 invariant: this method does NOT call ``_load`` because the
        writer is already attached to the catalog and a second
        ``read_only=True`` ``DuckLakeStore`` against the same catalog file
        would fail with ``Unique file handle conflict``. Instead, the
        registry queries the writer's store directly to populate the
        entry's range and summary.
        """
        with self._lock:
            existing = self._entries.get(name)
            if existing is not None:
                # Drop the prior entry's read-only store (we're switching
                # to the writer's store as the read store).
                self._dispose_entry_store(existing)
                preserved_subscribers = existing.subscribers
            else:
                preserved_subscribers = []

            run_store = RunStore(store)
            try:
                games_df = run_store.list_games()
                games = int(len(games_df))
                steps = int(games_df["steps"].sum()) if games > 0 else int(run_store.step_count())
                min_step, max_step = run_store.step_range(None)
                min_step = int(min_step)
                max_step = int(max_step)
            except Exception as exc:
                logger.warning("failed to read writer store for run {}: {}", name, exc)
                games, steps, min_step, max_step = 0, 0, 0, 0

            summary = RunSummary(
                name=name,
                games=games,
                steps=steps,
                status="ok" if steps > 0 else "empty",
            )
            run_dir = self._data_dir / name
            entry = _RunEntry(
                name=name,
                summary=summary,
                store=store,
                run_store=run_store,
                range=StepRange(
                    min=min_step,
                    max=max_step,
                    tail_growing=True,
                ),
                mtime=self._catalog_mtime(run_dir),
                owns_store=False,
                subscribers=preserved_subscribers,
            )
            self._entries[name] = entry

    def detach_writer_store(self, name: str) -> None:
        """Drop the writer's store from the registry.

        Called from ``RunWriter.close``. The writer owns the store and is
        responsible for closing it via ``RunWriter.close``. We just drop
        our reference here. The next ``get`` for this run lazily opens a
        fresh ``read_only=True`` ``DuckLakeStore`` against the now-quiescent
        catalog (the closed-run path validated by Phase 0).
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            preserved_subscribers = entry.subscribers
            del self._entries[name]
            # Don't eagerly call _load -- the writer's store may still
            # be flushing or even still open at this point. The next get()
            # will open it lazily once the writer fully releases the file.
            # We preserve subscribers via a placeholder if any exist.
            if preserved_subscribers:
                # Keep a minimal entry alive so subscriber bindings survive
                # the brief window between detach and the next get().
                placeholder = _RunEntry(
                    name=name,
                    summary=RunSummary(name=name, games=0, steps=0, status="empty"),
                    store=None,
                    run_store=None,
                    range=StepRange(min=0, max=0, tail_growing=False),
                    mtime=0.0,
                    owns_store=False,
                    subscribers=preserved_subscribers,
                )
                self._entries[name] = placeholder

    # -------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------

    def subscribe(self, name: str, callback: Callable[[int], None]) -> Unsubscribe:
        """Register a callback for ``step_added`` notifications.

        For unknown runs returns a no-op unsubscribe so the caller can
        bind without checking. For known runs the callback is invoked
        once per ``notify_subscribers`` call.
        """
        with self._lock:
            entry = self._entries.get(name) or self._load(name)
            if entry is None:
                return _noop_unsubscribe
            entry.subscribers.append(callback)
            self._entries[name] = entry

        def _unsub() -> None:
            with self._lock:
                e = self._entries.get(name)
                if e is None:
                    return
                try:
                    e.subscribers.remove(callback)
                except ValueError:
                    pass

        return _unsub

    def notify_subscribers(self, name: str, step: int) -> None:
        """Snapshot subscribers under the lock; dispatch outside the lock."""
        with self._lock:
            entry = self._entries.get(name)
            subs = list(entry.subscribers) if entry else []
        for cb in subs:
            try:
                cb(step)
            except Exception as exc:
                logger.warning("subscriber error for run {}: {}", name, exc)

    # -------------------------------------------------------------------
    # Internal: load and stale-check
    # -------------------------------------------------------------------

    def _load(self, name: str) -> _RunEntry | None:
        """Open a fresh ``read_only=True`` store for a closed run.

        This is the closed-run path validated by Phase 0's closed-run
        control. Returns ``None`` if the run directory does not exist or
        is not a valid DuckLake catalog. On open failure (corrupt
        catalog), still returns an entry whose ``summary.status ==
        "corrupt"`` so the dashboard can show the run with the right
        marker instead of silently dropping it.
        """
        run_dir = self._data_dir / name
        if not run_dir.is_dir():
            return None
        if not DuckLakeStore.is_valid_run(run_dir):
            return None
        self._summary_alias_counter += 1
        alias = (
            "reg_" + name.replace("-", "_").replace(".", "_") + f"_{self._summary_alias_counter}"
        )
        try:
            store = DuckLakeStore(run_dir, read_only=True, alias=alias)
        except Exception as exc:
            logger.warning("failed to open catalog for run {}: {}", name, exc)
            summary = RunSummary(
                name=name,
                games=0,
                steps=0,
                status="corrupt",
                error=f"{type(exc).__name__}: {exc}",
            )
            placeholder_range = StepRange(min=0, max=0, tail_growing=False)
            # Use the data-dir mtime as a stable, non-zero seed.
            mtime = self._catalog_mtime(run_dir)
            # We can't construct a RunStore without a store. Return a
            # corrupted entry whose store is None -- callers must check.
            return _RunEntry(
                name=name,
                summary=summary,
                store=None,
                run_store=None,
                range=placeholder_range,
                mtime=mtime,
                owns_store=False,
            )

        try:
            run_store = RunStore(store)
            games_df = run_store.list_games()
            games = len(games_df)
            steps = int(games_df["steps"].sum()) if games > 0 else run_store.step_count()
            min_step, max_step = run_store.step_range(None)
        except Exception as exc:
            logger.warning("failed to read run {} summary: {}", name, exc)
            try:
                store.close()
            except Exception:
                pass
            summary = RunSummary(
                name=name,
                games=0,
                steps=0,
                status="corrupt",
                error=f"{type(exc).__name__}: {exc}",
            )
            return _RunEntry(
                name=name,
                summary=summary,
                store=None,
                run_store=None,
                range=StepRange(min=0, max=0, tail_growing=False),
                mtime=self._catalog_mtime(run_dir),
                owns_store=False,
            )

        if steps == 0:
            status = "empty"
        else:
            status = "ok"
        summary = RunSummary(name=name, games=games, steps=steps, status=status)
        return _RunEntry(
            name=name,
            summary=summary,
            store=store,
            run_store=run_store,
            range=StepRange(min=min_step, max=max_step, tail_growing=False),
            mtime=self._catalog_mtime(run_dir),
            owns_store=True,
        )

    def _catalog_mtime(self, run_dir: Path) -> float:
        """Return the mtime of the catalog file or 0.0 if missing."""
        catalog = run_dir / "catalog.duckdb"
        try:
            return catalog.stat().st_mtime
        except OSError:
            return 0.0

    def _stale(self, entry: _RunEntry) -> bool:
        """Return True if the on-disk catalog is newer than the cached entry."""
        run_dir = self._data_dir / entry.name
        return self._catalog_mtime(run_dir) > entry.mtime

    def _dispose_entry_store(self, entry: _RunEntry) -> None:
        """Close the entry's owned store, if any."""
        if entry.owns_store and entry.store is not None:
            try:
                entry.store.close()
            except Exception:
                pass


def _noop_unsubscribe() -> None:
    """No-op unsubscribe returned for unknown runs."""
    return None
