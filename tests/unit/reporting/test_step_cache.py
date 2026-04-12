"""Unit tests for the StepCache LRU cache."""

from __future__ import annotations

import threading

from roc.reporting.run_store import StepData
from roc.reporting.step_cache import StepCache


def _sd(step: int, game: int = 1) -> StepData:
    """Create a minimal StepData instance for testing."""
    return StepData(step=step, game_number=game)


class TestStepCacheBasics:
    def test_get_returns_none_for_missing_key(self) -> None:
        cache = StepCache(capacity=10)
        assert cache.get("run-a", 1) is None

    def test_put_then_get_returns_value(self) -> None:
        cache = StepCache(capacity=10)
        sd = _sd(1)
        cache.put("run-a", 1, sd)
        assert cache.get("run-a", 1) is sd

    def test_keys_are_namespaced_by_run(self) -> None:
        cache = StepCache(capacity=10)
        cache.put("run-a", 1, _sd(1))
        cache.put("run-b", 1, _sd(1, game=2))
        a = cache.get("run-a", 1)
        b = cache.get("run-b", 1)
        assert a is not None
        assert b is not None
        assert a.game_number == 1
        assert b.game_number == 2


class TestStepCacheEviction:
    def test_lru_eviction_oldest_first(self) -> None:
        cache = StepCache(capacity=3)
        cache.put("run-a", 1, _sd(1))
        cache.put("run-a", 2, _sd(2))
        cache.put("run-a", 3, _sd(3))
        cache.put("run-a", 4, _sd(4))
        # Adding 4 should evict 1 (the LRU entry)
        assert cache.get("run-a", 1) is None
        assert cache.get("run-a", 2) is not None
        assert cache.get("run-a", 3) is not None
        four = cache.get("run-a", 4)
        assert four is not None
        assert four.step == 4

    def test_get_promotes_to_most_recent(self) -> None:
        cache = StepCache(capacity=3)
        cache.put("r", 1, _sd(1))
        cache.put("r", 2, _sd(2))
        cache.put("r", 3, _sd(3))
        # Touch step 1 -> promotes it to MRU
        cache.get("r", 1)
        # Adding step 4 should evict step 2 (now LRU), not step 1
        cache.put("r", 4, _sd(4))
        assert cache.get("r", 1) is not None
        assert cache.get("r", 2) is None
        assert cache.get("r", 3) is not None
        assert cache.get("r", 4) is not None

    def test_put_updates_existing_key(self) -> None:
        cache = StepCache(capacity=3)
        cache.put("r", 1, _sd(1))
        new_sd = _sd(1, game=99)
        cache.put("r", 1, new_sd)
        result = cache.get("r", 1)
        assert result is new_sd
        assert result.game_number == 99


class TestStepCacheInvalidation:
    def test_invalidate_run_clears_only_that_run(self) -> None:
        cache = StepCache(capacity=10)
        cache.put("r1", 1, _sd(1))
        cache.put("r1", 2, _sd(2))
        cache.put("r2", 1, _sd(1))
        cache.put("r2", 2, _sd(2))
        cache.invalidate_run("r1")
        assert cache.get("r1", 1) is None
        assert cache.get("r1", 2) is None
        assert cache.get("r2", 1) is not None
        assert cache.get("r2", 2) is not None

    def test_invalidate_unknown_run_is_noop(self) -> None:
        cache = StepCache(capacity=10)
        cache.put("r1", 1, _sd(1))
        cache.invalidate_run("nonexistent")
        assert cache.get("r1", 1) is not None


class TestStepCacheThreadSafety:
    def test_thread_safe_concurrent_put_get(self) -> None:
        """8 threads doing 1000 puts/gets each, no exceptions, no torn data."""
        cache = StepCache(capacity=2000)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(1000):
                    run = f"run-{thread_id}"
                    cache.put(run, i, _sd(i, game=thread_id))
                    got = cache.get(run, i)
                    if got is not None and got.step != i:
                        raise AssertionError(
                            f"torn data: thread={thread_id} expected step={i} got step={got.step}"
                        )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
