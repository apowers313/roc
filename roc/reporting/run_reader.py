"""Single read facade for any run, live or historical.

``RunReader`` is the only public read path for run data in the unified-run
architecture. Callers do not see the live/historical distinction; the reader
routes internally to the hot in-memory ``StepCache`` first and falls back to
the entry's ``DuckLakeStore`` on cache miss. The store comes from
``RunRegistry``: the writer's instance for active runs (the only way Phase 0
allows reads against an active catalog), and a fresh ``read_only=True``
instance for closed runs.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Callable

from roc.framework.logger import logger
from roc.reporting.run_registry import RunRegistry, Unsubscribe
from roc.reporting.run_store import StepData
from roc.reporting.step_cache import StepCache
from roc.reporting.types import HistoryKind, RunSummary, StepRange, StepResponse


def _step_data_to_dict(data: Any) -> dict[str, Any]:
    """Convert a StepData (dataclass) to a plain dict."""
    return dataclasses.asdict(data)


class RunReader:
    """Public read facade. See module docstring for details.

    The cache is mandatory: Phase 0 proved that we cannot open a separate
    ``read_only=True`` ``DuckLakeStore`` against the same catalog file in
    the same process while the writer is active. The cache absorbs the
    common case so reads of an active run almost never need to take the
    writer's ``_lock``.
    """

    def __init__(
        self,
        registry: RunRegistry,
        cache: StepCache,
    ) -> None:
        self._registry = registry
        self._cache = cache

    def get_step(self, run: str, step: int) -> StepResponse:
        """Single step lookup with a typed envelope.

        Returns ``StepResponse`` with one of:

        - ``status="ok"``: ``data`` is the StepData as a dict.
        - ``status="run_not_found"``: no entry for ``run``.
        - ``status="out_of_range"``: ``step`` is below 1 or above the run's
          current max.
        - ``status="not_emitted"``: the step is in range but the store
          returned no rows.
        - ``status="error"``: the store raised; ``error`` carries the
          exception class and message.
        """
        entry = self._registry.get(run)

        if entry is None:
            return StepResponse(status="run_not_found", data=None)
        if entry.run_store is None:
            # Corrupt entry from RunRegistry._load.
            return StepResponse(
                status="error",
                error=entry.summary.error or "run store unavailable",
                range=entry.range,
            )
        if step < 1 or step > entry.range.max:
            return StepResponse(
                status="out_of_range",
                data=None,
                range=entry.range,
            )

        cached = self._cache.get(run, step)
        if cached is not None:
            return StepResponse(
                status="ok",
                data=_step_data_to_dict(cached),
                range=entry.range,
            )

        try:
            data = entry.run_store.get_step_data(step)
        except Exception as exc:
            logger.warning("step read failed run={} step={}: {}", run, step, exc)
            return StepResponse(
                status="error",
                data=None,
                error=f"{type(exc).__name__}: {exc}",
                range=entry.range,
            )
        self._cache.put(run, step, data)
        return StepResponse(
            status="ok",
            data=_step_data_to_dict(data),
            range=entry.range,
        )

    def get_steps_batch(self, run: str, steps: list[int]) -> dict[int, StepData]:
        """Get data for multiple steps in a single registry round trip.

        Cache hits are returned directly; misses are resolved against the
        registry entry's ``RunStore`` and inserted into the cache. Steps
        that fail to load are silently skipped, matching the prefetch
        path's "best-effort batch" semantics.

        Raises ``FileNotFoundError`` for unknown runs so the API layer can
        translate to a 404.
        """
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        if entry.run_store is None:
            return {}
        result: dict[int, StepData] = {}
        uncached: list[int] = []
        for s in steps:
            cached = self._cache.get(run, s)
            if cached is not None:
                result[s] = cached
            else:
                uncached.append(s)
        if uncached:
            try:
                fetched = entry.run_store.get_steps_data(uncached)
            except Exception as exc:
                logger.warning(
                    "batch step read failed run={} count={}: {}",
                    run,
                    len(uncached),
                    exc,
                )
                fetched = {}
            for s, data in fetched.items():
                result[s] = data
                self._cache.put(run, s, data)
        return result

    def get_step_range(self, run: str, game: int | None = None) -> StepRange:
        """Return the run's ``{min, max, tail_growing}`` range.

        Raises ``FileNotFoundError`` for unknown runs so the API layer can
        translate to a 404. ``tail_growing`` reflects whether a writer is
        currently attached to the run via ``RunRegistry``.

        BUG-H2 fix: per-game queries route through
        ``entry.run_store.step_range(game)`` so the SQL filter is honored.
        ``entry.range`` is the global range cached at load time and cannot
        answer per-game questions. ``game=None`` still uses the cached
        range to keep the common case allocation-free.
        """
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        if game is not None and entry.run_store is not None:
            min_step, max_step = entry.run_store.step_range(game)
            return StepRange(
                min=int(min_step),
                max=int(max_step),
                tail_growing=entry.range.tail_growing,
            )
        return entry.range

    def get_history(
        self,
        run: str,
        kind: HistoryKind,
        game: int | None = None,
        fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Dispatch to the run's ``RunStore`` for the requested history kind.

        Routes every history query through the registry's ``RunStore``.
        Active runs read from the writer's ``DuckLakeStore`` (installed
        via ``RunRegistry.attach_writer_store``); closed runs read from a
        fresh ``read_only=True`` instance opened lazily by the registry.
        """
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        if entry.run_store is None:
            return []
        if kind == "metrics":
            return entry.run_store.get_metrics_history(game, fields)
        method_map: dict[HistoryKind, Callable[..., list[dict[str, Any]]]] = {
            "graph": entry.run_store.get_graph_history,
            "event": entry.run_store.get_event_history,
            "intrinsics": entry.run_store.get_intrinsics_history,
            "action": entry.run_store.get_action_history,
            "resolution": entry.run_store.get_resolution_history,
        }
        method = method_map[kind]
        return method(game)

    def list_games(self, run: str) -> list[dict[str, Any]]:
        """Return per-game summaries for a run.

        Each entry has ``game_number``, ``steps``, ``start_ts``, ``end_ts``
        (matching the ``GameSummary`` model in ``api_server.py``). Reads
        through ``entry.run_store`` so the call shares the registry's single
        ``DuckLakeStore`` instance -- the writer's store for active runs,
        the lazy ``read_only=True`` instance for closed runs.

        Raises ``FileNotFoundError`` for unknown runs so the API layer can
        translate it to a 404.
        """
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        if entry.run_store is None:
            return []
        df = entry.run_store.list_games()
        return [
            {
                "game_number": int(row["game_number"]),
                "steps": int(row["steps"]),
                "start_ts": int(row["start_ts"]) if row.get("start_ts") is not None else None,
                "end_ts": int(row["end_ts"]) if row.get("end_ts") is not None else None,
            }
            for _, row in df.iterrows()
        ]

    def get_all_objects(self, run: str, game: int | None = None) -> list[dict[str, Any]]:
        """Return all resolved objects with attributes, match counts, and
        creation step.

        Reads through ``entry.run_store`` for the same reason as
        ``list_games`` -- to avoid the catalog double-attach.
        """
        entry = self._registry.get(run)
        if entry is None:
            raise FileNotFoundError(run)
        if entry.run_store is None:
            return []
        return entry.run_store.get_all_objects(game)

    def get_schema(self, run: str) -> dict[str, Any] | None:
        """Return the graph database schema for a run.

        Reads ``schema.json`` directly from the run directory -- no DuckLake
        involved. Returns ``None`` if the file is missing or unparseable so
        the API layer can return a 404 / empty payload.
        """
        schema_path = self._registry.data_dir / run / "schema.json"
        if not schema_path.exists():
            return None
        try:
            result: dict[str, Any] = json.loads(schema_path.read_text())
            return result
        except Exception:
            logger.warning("schema.json parse failed for run=%s", run)
            return None

    def get_action_map(self, run: str) -> list[dict[str, Any]] | None:
        """Return the full action-id-to-name mapping for a run.

        Reads ``action_map.json`` directly from the run directory -- no
        DuckLake involved. Returns ``None`` if the file is missing or
        unparseable.
        """
        action_map_path = self._registry.data_dir / run / "action_map.json"
        if not action_map_path.exists():
            return None
        try:
            result: list[dict[str, Any]] = json.loads(action_map_path.read_text())
            return result
        except Exception:
            logger.warning("action_map.json parse failed for run=%s", run)
            return None

    def list_runs(
        self,
        *,
        min_steps: int = 10,
        include_all: bool = False,
    ) -> list[RunSummary]:
        """List runs in the data directory via the registry.

        ``min_steps`` defines the minimum step count for an "ok" run --
        runs below the bar are demoted to ``status="short"``. Tail-growing
        runs are exempt so a freshly-started game with 0 steps still
        appears in the dropdown. ``include_all`` returns runs of every
        status (``empty``, ``short``, ``corrupt``).
        """
        return self._registry.list(min_steps=min_steps, include_all=include_all)

    def subscribe(
        self,
        run: str,
        callback: Callable[[int], None],
    ) -> Unsubscribe:
        """Subscribe to ``step_added`` notifications for a run.

        Returns a no-op unsubscribe for unknown runs (so callers can
        bind without checking). Phase 4 wires ``RunWriter.push_step``
        through ``RunRegistry.notify_subscribers``, so subscribers
        registered here are invoked once per step push from the active
        writer. The api_server's Socket.io ``subscribe_run`` handler is
        the canonical caller; the callback emits a tiny ``{run, step}``
        ``step_added`` payload to the connected browser, which then
        invalidates its TanStack Query keys and refetches via the
        unified ``RunReader`` path.
        """
        return self._registry.subscribe(run, callback)
