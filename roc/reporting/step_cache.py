"""Process-wide LRU cache of recently-touched StepData.

Used by ``RunReader`` (cache-first reads) and ``RunWriter`` (write-through).
The cache is private to those classes and not exposed in any public API.

Phase 0 of the unified-run architecture proved that we cannot open a separate
``read_only=True`` ``DuckLakeStore`` against the same catalog file in the same
process while a writer is active. The cache absorbs the common case so reads
of an active run almost never need to take the writer's lock; on cache miss
``RunReader`` falls back to the writer's own ``DuckLakeStore`` instance.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from roc.reporting.run_store import StepData


class StepCache:
    """Process-wide LRU keyed by ``(run, step)``.

    Backed by ``collections.OrderedDict`` and a ``threading.Lock`` so it is
    safe for concurrent use from FastAPI request threads and the game
    writer thread.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self._capacity = capacity
        self._cache: OrderedDict[tuple[str, int], StepData] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, run: str, step: int) -> StepData | None:
        """Return the cached StepData for ``(run, step)`` or ``None``.

        A cache hit promotes the entry to most-recently-used.
        """
        key = (run, step)
        with self._lock:
            data = self._cache.get(key)
            if data is not None:
                self._cache.move_to_end(key)
            return data

    def put(self, run: str, step: int, data: StepData) -> None:
        """Insert or update an entry, evicting the LRU if over capacity."""
        key = (run, step)
        with self._lock:
            self._cache[key] = data
            self._cache.move_to_end(key)
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def invalidate_run(self, run: str) -> None:
        """Drop every cached entry for the given run.

        Used when the writer for a run closes and the next read should fall
        through to a freshly-opened ``read_only=True`` ``DuckLakeStore``.
        """
        with self._lock:
            keys = [k for k in self._cache if k[0] == run]
            for k in keys:
                del self._cache[k]

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
