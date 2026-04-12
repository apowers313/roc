"""Single write facade for an active ROC run.

``RunWriter`` is the only public write path for run data in the unified-run
architecture. It owns the writer-side coordination between three pieces:

1. ``StepCache`` (write-through): every pushed step is mirrored into the
   process-wide LRU cache so subsequent ``RunReader`` reads of an active
   run almost never need to take the writer's ``DuckLakeStore._lock``.
   Phase 0 of the unified-run architecture proved that we cannot open a
   separate ``read_only=True`` ``DuckLakeStore`` against an active
   writer's catalog file in the same process; the cache is the only thing
   that keeps reads cheap during a live game.

2. ``ParquetExporter`` (canonical durable write path): the exporter is
   driven by OpenTelemetry log records emitted from the game pipeline.
   ``RunWriter`` does not call ``exporter.queue`` directly -- the OTel
   pipeline owns that side of the contract. The exporter is held here
   so ``close()`` can request a final flush.

3. ``RunRegistry`` (liveness signal + range advancement): on init the
   writer's ``DuckLakeStore`` is installed as the registry entry's read
   store via ``attach_writer_store`` (this also flips
   ``StepRange.tail_growing`` to ``True``). Each ``push_step`` call
   advances the entry's max step. On ``close`` the writer's store is
   detached and ``tail_growing`` flips back to ``False``.

Phase 4 wires ``push_step`` through ``RunRegistry.notify_subscribers`` so
Socket.io invalidation flows naturally through the same call site:
subscribers register via ``RunRegistry.subscribe`` (typically from the
api_server's ``subscribe_run`` Socket.io handler) and receive a callback
on every step push. The callback invalidates the relevant TanStack Query
keys on the browser, which then refetches via the unified ``RunReader``
path.
"""

from __future__ import annotations

from typing import Any

from roc.reporting.ducklake_store import DuckLakeStore
from roc.reporting.run_registry import RunRegistry
from roc.reporting.run_store import StepData
from roc.reporting.step_cache import StepCache


class RunWriter:
    """Single owner of writer-side state for one active run.

    The writer's ``DuckLakeStore`` is the SINGLE store for this run --
    it doubles as the read store for in-process reads via
    ``RunRegistry.attach_writer_store``. Do not construct a separate
    ``read_only=True`` ``DuckLakeStore`` for the same catalog file
    while a writer is open; DuckDB rejects it as
    ``BinderException: Unique file handle conflict``.

    Args:
        run: The run name (directory name under the data dir).
        registry: The shared ``RunRegistry`` singleton.
        cache: The shared ``StepCache`` singleton.
        exporter: The ``ParquetExporter`` driving the OTel write path.
            Held only so ``close()`` can request a final flush; the
            writer does not enqueue records directly.
        store: The writer-owned ``DuckLakeStore``. Owned by the caller;
            ``close()`` does NOT close it (the caller is responsible
            for the underlying lifetime, mirroring how
            ``Observability.reset()`` already manages this in production).
    """

    def __init__(
        self,
        run: str,
        registry: RunRegistry,
        cache: StepCache,
        exporter: Any,
        store: DuckLakeStore,
    ) -> None:
        self._run = run
        self._registry = registry
        self._cache = cache
        self._exporter = exporter
        self._store = store
        self._closed = False
        # Phase 0 pivot: install our store as the registry's read store
        # for this run. This flips ``tail_growing`` on and ensures any
        # ``RunReader`` for the same run reads through the writer's
        # store on cache miss.
        registry.attach_writer_store(run, store)

    @property
    def run(self) -> str:
        """The run name owned by this writer."""
        return self._run

    def push_step(self, data: StepData) -> None:
        """Mirror a step into the cache, advance the registry's max, and notify.

        Called from the game thread (or wherever ``StepData`` is
        produced). Cheap and non-blocking; does not touch the
        ``DuckLakeStore``. The OTel pipeline is the canonical durable
        write path -- this method is purely for keeping the in-memory
        view warm for ``RunReader`` and the ``RunRegistry`` step
        range fresh.

        Phase 4: after the cache and range are updated, subscribers are
        notified so the api_server's Socket.io ``subscribe_run`` handlers
        can broadcast a tiny ``{run, step}`` invalidation event to the
        browser. ``notify_subscribers`` snapshots subscribers under the
        registry lock and dispatches outside it, so a slow callback can
        never block the game loop's push.
        """
        self._cache.put(self._run, data.step, data)
        self._registry.update_max_step(self._run, data.step)
        self._registry.notify_subscribers(self._run, data.step)

    def close(self) -> None:
        """Detach from the registry. Idempotent.

        After ``close()`` returns, the registry entry's
        ``tail_growing`` is ``False`` and the next ``RunReader`` read
        for this run will lazily reopen a fresh ``read_only=True``
        ``DuckLakeStore`` against the now-quiescent catalog file (the
        closed-run path validated by Phase 0's closed-run control).

        Does NOT close the underlying ``DuckLakeStore`` -- the caller
        owns that. In production, ``Observability.reset()`` is the
        single owner of the writer's store lifetime.
        """
        if self._closed:
            return
        self._closed = True
        self._registry.detach_writer_store(self._run)
