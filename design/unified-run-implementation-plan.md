# Implementation Plan for Unified Run Architecture

## Overview

This plan executes the design in `design/unified-run-architecture.md`: collapse
the dashboard's parallel "live" and "historical" data paths into a single
read/write path through `RunReader` / `RunWriter` / `RunRegistry` / `StepCache`,
with `tail_growing: bool` as the only signal of liveness at the API boundary.
On the frontend, TanStack Query becomes the only client store; Socket.io is
demoted to invalidation-only; and the four-state playback machine collapses to
two booleans.

The work is split into seven phases. **Phase 0** is a half-day measurement
spike that de-risks the central performance assumption before any production
code changes. **Phase 1** is the MVP wedge: it introduces the new facade as a
pass-through wrapper over the existing `DataStore` so that every API endpoint
flows through one place, with no behavior change. **Phases 2-5** then move
each load-bearing concept (history indices, `tail_growing`, Socket.io
invalidation, the playback machine) onto the new shape one slice at a time.
**Phase 6** is a deletion-only cleanup pass.

Key targets:
- Net deletion: ~800-1000 lines of code (~400 backend, ~400 frontend).
- Zero new external dependencies (LRU comes from `collections.OrderedDict`,
  invalidation comes from existing `socket.io` + TanStack Query).
- Each phase ends with a working dashboard that the user can manually verify
  end-to-end.
- Three CLAUDE.md "rules" (`Stale Closure Pattern`, `Live/historical feature
  parity`, `URL parameter sovereignty`) become unenforceable or disappear by
  the end.

## Phase Breakdown

### Phase 0: DuckLake Concurrency Spike (COMPLETED 2026-04-08 -- pivot taken)

**De-risks the central assumption** that DuckLake reads stay fast under
concurrent writes once data inlining is on and a read-only secondary
connection is used. This is pure measurement; no production code is touched.

**Outcome**: The "read-only secondary connection" leg of the assumption
is **wrong**. DuckDB rejects opening a second `DuckLakeStore(read_only=True)`
against the catalog of an active writer in the same process. The pivot
(StepCache mandatory; active runs share the writer's `DuckLakeStore`) is
detailed in the "Phase 0 outcome" block below. Phases 1-6 keep the same
shape; the rule is hidden inside `RunRegistry`.

**Duration**: 0.5 day (actual: ~1 hour)

**Tests to Write First**:

- `tmp/ducklake_concurrency_spike.py`: standalone benchmark script (deleted
  after the phase). Not part of the test suite -- it's a one-off
  measurement.
  ```python
  # Pseudocode of the spike
  def main() -> None:
      with TemporaryDirectory() as tmpdir:
          writer_store = DuckLakeStore(Path(tmpdir), read_only=False)
          reader_store = DuckLakeStore(Path(tmpdir), read_only=True)
          stop = threading.Event()
          writer = threading.Thread(target=write_loop, args=(writer_store, stop))
          reader = threading.Thread(target=read_loop, args=(reader_store, stop))
          writer.start(); reader.start()
          time.sleep(30)
          stop.set()
          writer.join(); reader.join()
          report_latencies(reader.latencies)  # p50, p95, p99
  ```

**Implementation**:

- `tmp/ducklake_concurrency_spike.py`: writer at 60 Hz, reader continuously
  querying random recent steps via the existing `DuckLakeStore` API. Run
  twice: once with `data_inlining_row_limit=500` (current default), once
  with `0`.
- Record results in a comment at the top of the script (or paste into the
  decision log in the design doc).

**Dependencies**:
- External: none (uses existing `DuckLakeStore` and stdlib threading).
- Internal: none.

**Verification**:
1. Run: `uv run python tmp/ducklake_concurrency_spike.py`
2. Decision criteria (in priority order):
   - Reader can open a `read_only=True` connection while a writer is active
     without raising "unique file handle conflict" (criterion 1).
   - p99 read latency < 50ms under load (criterion 2).
   - No torn rows; reader sees consistent step records (criterion 3).
   - No reader exceptions over the 30-second run (criterion 4).

**Phase 0 outcome (2026-04-08)**: COMPLETED. Pivot taken.

- Criterion 1 **FAILED**. DuckDB 1.5.0 rejects opening a second
  `DuckLakeStore(read_only=True)` against the catalog file of an active
  writer in the same Python process with
  `BinderException: Unique file handle conflict`. Reproduced with both
  `alias="lake_reader"` and `alias="lake"`, so the failure is at the
  catalog-file level, not the DuckLake alias level. Reproduced with both
  `data_inlining_row_limit=500` and `=0`, so it is independent of inlining.
- Criteria 2, 3, 4 are not measurable while criterion 1 fails (the
  reader thread never opens its connection).
- **Closed-run control PASSED**: opening `read_only=True` *after*
  `writer.close()` succeeds and reads back data cleanly. Closed runs
  continue to use a separate `read_only=True` instance as the design
  assumed.

**Pivot adopted (matches the original "Criterion 1 or 2 fails" branch
but with a stronger constraint -- the reader cannot open at all, not
merely slowly):**

1. `StepCache` is **mandatory**, not an optimization. `RunWriter.push_step`
   write-throughs into the cache so almost all hot reads of an active
   run avoid the writer's `DuckLakeStore._lock` entirely.
2. For active runs (`tail_growing=True`), `RunRegistry._RunEntry.store`
   MUST be the same `DuckLakeStore` instance held by the active
   `RunWriter`. `RunReader` reads from that shared instance under its
   existing `_lock` on cache miss. Do NOT construct a separate
   `read_only=True` `DuckLakeStore` for an active run.
3. For closed runs (`tail_growing=False`), `RunRegistry._RunEntry.store`
   is a fresh `read_only=True` instance opened lazily. The closed-run
   path is unchanged from the original design.
4. `RunWriter.__init__` takes the writer's `DuckLakeStore` as a
   parameter and calls `RunRegistry.attach_writer_store(name, store)`
   to install it. `RunWriter.close()` calls
   `RunRegistry.detach_writer_store(name)` so the next read lazily
   reopens `read_only=True`.

Phases 1-6 are unchanged in shape; the active-vs-closed store ownership
rule is hidden inside `RunRegistry`. See
`design/unified-run-architecture.md` "Risk #2" and the `_RunEntry`
docstring for the full rationale, and `tmp/ducklake_concurrency_spike.py`
for the measurement. **Phase 1 must add a new invariant to
`roc/reporting/CLAUDE.md`** documenting that active-writer reads share
the writer's `DuckLakeStore` instance; this is the only way the rule
survives future maintenance.

(Originally: "Criterion 3 fails -> stop. Re-evaluate the storage layer
entirely." Did not happen -- closed-run reads work, so the storage
layer is not at fault; the limit is in-process attach uniqueness.)

---

### Phase 1: RunReader / RunRegistry / StepCache Wedge (No Behavior Change)

**Goal**: Introduce the unified facade as a pass-through wrapper over the
existing `DataStore`, `StepBuffer`, and `RunStore` internals. Every API
endpoint reads through `RunReader`. The internal cache/index code stays in
place. This is the MVP -- it ships nothing visible but creates the seam every
later phase pivots on.

**Duration**: 1.5-2 days

**Tests to Write First**:

- `tests/unit/reporting/test_step_cache.py`: new
  ```python
  def test_lru_eviction_oldest_first():
      cache = StepCache(capacity=3)
      cache.put("run-a", 1, sd(1)); cache.put("run-a", 2, sd(2))
      cache.put("run-a", 3, sd(3)); cache.put("run-a", 4, sd(4))
      assert cache.get("run-a", 1) is None  # evicted
      assert cache.get("run-a", 4).step == 4

  def test_get_promotes_to_most_recent():
      cache = StepCache(capacity=3)
      cache.put("r", 1, sd(1)); cache.put("r", 2, sd(2)); cache.put("r", 3, sd(3))
      cache.get("r", 1)                  # promotes 1 to MRU
      cache.put("r", 4, sd(4))           # evicts 2, not 1
      assert cache.get("r", 1) is not None
      assert cache.get("r", 2) is None

  def test_invalidate_run_clears_only_that_run():
      cache = StepCache(capacity=10)
      cache.put("r1", 1, sd(1)); cache.put("r2", 1, sd(1))
      cache.invalidate_run("r1")
      assert cache.get("r1", 1) is None
      assert cache.get("r2", 1) is not None

  def test_thread_safe_concurrent_put_get():
      # 8 threads, each doing 1000 puts/gets, no exceptions, no torn data
  ```

- `tests/unit/reporting/test_run_registry.py`: new
  ```python
  def test_get_lazy_loads_entry_from_data_dir(tmp_path):
      _seed_run(tmp_path, "run-1", steps=10)
      reg = RunRegistry(tmp_path)
      entry = reg.get("run-1")
      assert entry.range.max == 10
      assert entry.range.tail_growing is False

  def test_mark_growing_flips_tail_growing():
      reg = RunRegistry(tmp_path)
      reg.mark_growing("run-1", growing=True)
      assert reg.get("run-1").range.tail_growing is True
      reg.mark_growing("run-1", growing=False)
      assert reg.get("run-1").range.tail_growing is False

  def test_subscribe_returns_no_op_unsubscribe_for_unknown_run():
      reg = RunRegistry(tmp_path)
      unsub = reg.subscribe("does-not-exist", lambda step: None)
      unsub()  # must not raise

  def test_notify_subscribers_dispatches_outside_lock():
      # Verify a re-entrant callback that calls reg.get(...) doesn't deadlock.

  def test_stale_entry_reloaded_on_mtime_change():
      # Touch the catalog file; next get() returns refreshed range.
  ```

- `tests/unit/reporting/test_run_reader.py`: new
  ```python
  def test_get_step_returns_ok_envelope_for_valid_step():
      reader = make_reader_with_steps(["r1"], steps=5)
      resp = reader.get_step("r1", 3)
      assert resp.status == "ok"
      assert resp.data.step == 3
      assert resp.range.max == 5

  def test_get_step_returns_run_not_found_for_unknown_run():
      reader = make_reader()
      resp = reader.get_step("nope", 1)
      assert resp.status == "run_not_found"
      assert resp.data is None

  def test_get_step_returns_out_of_range():
      reader = make_reader_with_steps(["r1"], steps=5)
      resp = reader.get_step("r1", 99)
      assert resp.status == "out_of_range"
      assert resp.range.max == 5

  def test_get_step_returns_error_envelope_on_store_exception():
      reader = make_reader_with_failing_store()
      resp = reader.get_step("r1", 1)
      assert resp.status == "error"
      assert "BoomError" in resp.error

  def test_get_step_cache_hit_skips_store():
      cache = StepCache(); cache.put("r1", 1, sd(1))
      reader = make_reader_with_failing_store(cache=cache)
      resp = reader.get_step("r1", 1)
      assert resp.status == "ok"  # never touched the store

  def test_get_step_populates_cache_on_miss():
      cache = StepCache()
      reader = make_reader_with_steps(["r1"], steps=2, cache=cache)
      reader.get_step("r1", 1)
      assert cache.get("r1", 1) is not None
  ```

- `tests/unit/reporting/test_api_server.py`: modify
  - Replace direct `DataStore` mocks with `RunReader` mocks.
  - Add: `test_step_endpoint_returns_typed_envelope_on_404` (asserts the
    response body matches `StepResponse(status="run_not_found")`, not an
    empty `{}`).
  - Add: `test_step_endpoint_returns_typed_envelope_on_500` (simulate
    store exception, assert `status="error"` body, not silent empty data).

- `tests/integration/reporting/test_run_reader_integration.py`: new
  ```python
  def test_inprocess_read_after_writer_flush(tmp_path):
      # Spin up a real DuckLakeStore + RunWriter, push 5 steps, close,
      # verify RunReader can read all 5 from a fresh registry.

  def test_cache_hit_during_active_write(tmp_path):
      # Push step 1, do not flush; reader.get_step("r", 1) hits the cache
      # (verified via timing or a counter on the underlying store).
  ```

**Implementation**:

- `roc/reporting/step_cache.py`: new. ~80 lines.
  ```python
  class StepCache:
      """Process-wide LRU keyed by (run, step). Private to RunReader/RunWriter."""
      def __init__(self, capacity: int = 5000) -> None: ...
      def get(self, run: str, step: int) -> StepData | None: ...
      def put(self, run: str, step: int, data: StepData) -> None: ...
      def invalidate_run(self, run: str) -> None: ...
  ```

- `roc/reporting/run_registry.py`: new. ~250 lines.
  ```python
  @dataclass
  class _RunEntry:
      name: str
      summary: RunSummary
      # The single DuckLakeStore for this run. Provenance depends on
      # tail_growing (enforced by Phase 0; see Risk #1 in the arch doc):
      #   tail_growing == True  -> writer's instance, installed via
      #                            attach_writer_store()
      #   tail_growing == False -> fresh read_only=True instance opened
      #                            lazily by _load()
      # Never construct a separate read_only=True DuckLakeStore for an
      # active run -- DuckDB will reject the second attach with
      # "Unique file handle conflict".
      store: RunStore
      range: StepRange
      mtime: float
      subscribers: list[Callable[[int], None]] = field(default_factory=list)

  class RunRegistry:
      def __init__(self, data_dir: Path) -> None: ...
      def get(self, name: str) -> _RunEntry | None: ...
      def list(self, *, include_all: bool = False) -> list[RunSummary]: ...
      def mark_growing(self, name: str, *, growing: bool) -> None: ...
      def update_max_step(self, name: str, step: int) -> None: ...
      def notify_subscribers(self, name: str, step: int) -> None: ...
      def subscribe(self, name: str, cb: Callable[[int], None]) -> Unsubscribe: ...
      # Phase 0 pivot: writer install/uninstall hooks.
      def attach_writer_store(self, name: str, store: DuckLakeStore) -> None:
          """Replace the entry's store with the writer's instance and
          flip tail_growing on. Called from RunWriter.__init__."""
          ...
      def detach_writer_store(self, name: str) -> None:
          """Drop the writer's store from the entry and flip
          tail_growing off. The next get() lazily reopens
          read_only=True. Called from RunWriter.close()."""
          ...
  ```
  Internally borrows `_load_run_summary`, `_get_run_store`, and the
  filesystem-walking logic from today's `DataStore`. The `_load`
  helper opens a fresh `read_only=True DuckLakeStore` -- this is the
  closed-run path validated by Phase 0's closed-run control.

- `roc/reporting/run_reader.py`: new. ~150 lines.
  ```python
  class RunReader:
      def __init__(self, registry: RunRegistry, cache: StepCache) -> None: ...
      def get_step(self, run: str, step: int) -> StepResponse: ...
      def get_step_range(self, run: str) -> StepRange: ...
      def get_history(self, run: str, kind: HistoryKind, game: int | None = None) -> list[dict]: ...
      def list_runs(self, *, include_all: bool = False) -> list[RunSummary]: ...
      def subscribe(self, run: str, cb: Callable[[int], None]) -> Unsubscribe: ...  # no-op until Phase 4
  ```
  `get_step` MUST check `StepCache` first. The cache is mandatory --
  Phase 0 proved that for an active run we cannot open a separate
  `read_only=True` DuckLakeStore in the same process, so the only
  fallback is the writer's own DuckLakeStore (installed by
  `RunRegistry.attach_writer_store`) accessed under its `_lock`.
  Hot reads must hit the cache to avoid serializing on writer activity.

  In Phase 1, `get_history` delegates to `DataStore`'s existing live or
  historical method (whichever the legacy code chose). In Phase 2 it
  switches to direct `RunStore` calls.

- `roc/reporting/types.py` or `roc/reporting/api_models.py`: add the
  `StepResponse`, `StepRange`, `HistoryKind` Pydantic types if not already
  present.
  ```python
  class StepResponse(BaseModel):
      status: Literal["ok", "run_not_found", "out_of_range", "not_emitted", "error"]
      data: StepData | None = None
      range: StepRange | None = None
      error: str | None = None

  class StepRange(BaseModel):
      min: int
      max: int
      tail_growing: bool = False  # field added; default false until Phase 3
  ```

- `roc/reporting/api_server.py`: modify ~10 endpoints.
  - Add module-level singletons: `_step_cache = StepCache()`,
    `_registry = RunRegistry(...)`, `_reader = RunReader(_registry, _step_cache)`.
  - Replace the body of each step/range/history endpoint with a 1-2 line call
    to `_reader`.
  - The legacy `DataStore` instance stays alive only because the existing
    history methods still need its `_indices`. Phase 2 removes it.

- `roc/reporting/data_store.py`: modify (not deleted yet).
  - Add an internal pointer to the new `_registry` so legacy methods can
    delegate. No fields removed yet.

**Dependencies**:
- External: none new.
- Internal: existing `DuckLakeStore`, `StepBuffer`, `RunStore`, `StepData`.

**Verification**:
1. Run: `make test`
2. Run: `pnpm -C dashboard-ui exec vitest run`
3. Run: `make lint`
4. Manual:
   - `make run`
   - Open the dashboard at the URL from `npx servherd info roc-ui`.
   - Browse a historical run; step through 5-10 frames; the data should
     display correctly (no blank panels, no console errors).
   - Click "Game" → "Start"; a game subprocess spawns; the new run appears
     in the dropdown within ~5s; live data streams.
   - Switch to a different historical run while the game runs; both
     should work.
5. Grep check: `grep -n "is_live\|_live_buffer" roc/reporting/api_server.py`
   should return zero hits (all live branches now sit inside `DataStore`,
   to be removed in Phase 6).

**Reversible**: Yes. Delete the new files; revert `api_server.py`.

---

### Phase 2: Delete `_indices`; History Reads via DuckLake

**Goal**: Remove `DataStore._indices` and `_GameIndex`. All history queries
hit DuckLake via `RunStore`. This validates Risk #3 (history performance) on
real data before the frontend surgery starts.

**Duration**: 1 day

**Tests to Write First**:

- `tests/integration/reporting/test_history_via_ducklake.py`: new
  ```python
  def test_graph_history_for_10k_step_run(tmp_path, benchmark):
      _seed_run(tmp_path, "big", steps=10_000)
      reader = make_reader(tmp_path)
      result = benchmark(reader.get_history, "big", "graph", game=1)
      assert len(result) > 0
      assert benchmark.stats["mean"] < 0.5  # 500ms target

  def test_metrics_history_returns_same_shape_as_legacy(tmp_path):
      # Snapshot test against the old DataStore output for the same run.

  def test_event_history_dispatch_by_kind(tmp_path):
      reader = make_reader(tmp_path)
      for kind in ("graph", "event", "metrics", "intrinsics", "action", "resolution"):
          result = reader.get_history("r", kind, game=1)
          assert isinstance(result, list)
  ```

- `tests/unit/reporting/test_data_store.py`: modify
  - Delete every test that asserts on `_indices`, `_GameIndex`,
    `_index_step`, `_get_live_history`, `_on_step_pushed`,
    `_index_event_summary`. (~10-12 tests removed.)
  - Keep behavior tests; rewrite a few to assert on `RunReader` output.

**Implementation**:

- `roc/reporting/run_store.py`: ensure each `get_*_history` method exists
  on `RunStore`. Most are already there; add the missing ones (likely
  `get_resolution_history` and any Phase-specific shape helpers that only
  lived in `DataStore._get_live_history`).

- `roc/reporting/run_reader.py`: replace the Phase 1 stub for `get_history`
  with a direct dispatch into the entry's `RunStore`.
  ```python
  def get_history(self, run: str, kind: HistoryKind, game: int | None = None) -> list[dict]:
      entry = self._registry.get(run)
      if entry is None:
          raise FileNotFoundError(run)
      method = {
          "graph": entry.store.get_graph_history,
          "event": entry.store.get_event_history,
          "metrics": entry.store.get_metrics_history,
          "intrinsics": entry.store.get_intrinsics_history,
          "action": entry.store.get_action_history,
          "resolution": entry.store.get_resolution_history,
      }[kind]
      return method(game=game)
  ```

- `roc/reporting/data_store.py`: delete
  - `_GameIndex` dataclass (~14 lines)
  - `_indices: dict[int, _GameIndex]` field
  - `_on_step_pushed`, `_index_step`, `_append_if_present`,
    `_index_event_summary` (~50 lines)
  - `_get_live_history` (~10 lines)
  - The listener registration that called `_on_step_pushed`.
  - The `if self._is_live(run_name):` branches in each `get_*_history`
    method.

**Dependencies**:
- External: none.
- Internal: Phase 1 (`RunReader`, `RunRegistry`).

**Verification**:
1. Run: `make test` (unit + integration)
2. Manual:
   - `make run`
   - Start a game; open the Graph & Events panel; verify the graph
     history chart renders for the live run.
   - Stop the game; navigate to a historical run; verify the same chart
     renders for it.
   - Open the Intrinsics panel; verify the time-series chart renders.
   - Open the Metrics panel; verify the latency histograms render.
3. Performance check: in the dashboard, hit
   `GET /api/runs/<10k-step-run>/graph-history` and confirm the response
   completes in < 500ms. If the manual click is too imprecise, use:
   `time uv run python tmp/measure_history.py <run-name>`.
4. Grep check: `grep -n "_indices\|_GameIndex\|_get_live_history" roc/reporting/data_store.py`
   should return zero hits.

**Reversible**: Yes. Restore `_indices` from git; re-attach the listener.

---

### Phase 3: `tail_growing` in API; Delete Frontend Live-Status State

**Goal**: Add `tail_growing: bool` to the step-range response, sourced from
`RunRegistry`. Delete `liveRunName`, `liveGameNumber`, `liveGameActive` from
`DashboardContext`. The "GO LIVE" badge reads its condition from
`useStepRange().data.tail_growing`.

**Duration**: 1.5 days

**Tests to Write First**:

- `tests/unit/reporting/test_run_registry.py`: extend
  ```python
  def test_step_range_includes_tail_growing_false_by_default(tmp_path):
      reg = RunRegistry(tmp_path)
      _seed_run(tmp_path, "r", steps=3)
      assert reg.get("r").range.tail_growing is False

  def test_mark_growing_propagates_to_subsequent_get(tmp_path):
      reg = RunRegistry(tmp_path)
      _seed_run(tmp_path, "r", steps=3)
      reg.mark_growing("r", growing=True)
      assert reg.get("r").range.tail_growing is True
  ```

- `tests/unit/reporting/test_api_server.py`: extend
  ```python
  def test_step_range_endpoint_returns_tail_growing(client):
      _start_writer("r")
      response = client.get("/api/runs/r/step-range")
      assert response.json()["tail_growing"] is True

  def test_step_range_after_writer_close_returns_false(client):
      writer = _start_writer("r")
      writer.close()
      assert client.get("/api/runs/r/step-range").json()["tail_growing"] is False
  ```

- `dashboard-ui/src/api/queries.test.tsx`: extend
  ```typescript
  it("useStepRange returns tail_growing in the response", async () => {
      mockApi.get("/api/runs/r/step-range").reply(200, {
          min: 1, max: 10, tail_growing: true,
      });
      const { result } = renderHook(() => useStepRange("r"), { wrapper });
      await waitFor(() => expect(result.current.data?.tail_growing).toBe(true));
  });
  ```

- `dashboard-ui/src/components/status/StatusBar.test.tsx`: rewrite the
  GO LIVE badge tests to drive off `tail_growing` instead of
  `liveGameActive`.
  ```typescript
  it("shows GO LIVE badge when current run tail_growing is true", () => {
      mockUseStepRange.mockReturnValue({ data: { min: 1, max: 5, tail_growing: true }});
      render(<StatusBar />);
      expect(screen.getByText("GO LIVE")).toBeInTheDocument();
  });

  it("hides GO LIVE badge when tail_growing is false", () => {
      mockUseStepRange.mockReturnValue({ data: { min: 1, max: 5, tail_growing: false }});
      render(<StatusBar />);
      expect(screen.queryByText("GO LIVE")).not.toBeInTheDocument();
  });
  ```

**Implementation**:

- `roc/reporting/types.py`: ensure `StepRange.tail_growing` is wired
  through (was added in Phase 1 but defaulted to false).

- `roc/reporting/run_writer.py`: new. ~50 lines.
  ```python
  class RunWriter:
      def __init__(self, run: str, registry: RunRegistry, cache: StepCache,
                   exporter: ParquetExporter, store: DuckLakeStore) -> None:
          self._run = run; self._registry = registry
          self._cache = cache; self._exporter = exporter
          self._store = store
          # Phase 0 pivot: install our store as the read store and flip
          # tail_growing on. RunReader for this run now reads from the
          # writer's DuckLakeStore (under its _lock) on cache miss; a
          # separate read_only=True instance is impossible while we're open.
          registry.attach_writer_store(run, store)

      def push_step(self, data: StepData) -> None:
          self._cache.put(self._run, data.step, data)
          self._exporter.queue(data)
          self._registry.update_max_step(self._run, data.step)
          # notify_subscribers is wired in Phase 4

      def close(self) -> None:
          self._exporter.flush()
          # Drop our store from the registry; the next read for this
          # run lazily reopens read_only=True against the now-quiescent
          # catalog file (validated by Phase 0's closed-run control).
          self._registry.detach_writer_store(self._run)
  ```

- `roc/reporting/api_server.py`: modify the `/api/internal/step` handler
  to look up (or lazily create) a `RunWriter` for the run and call
  `push_step`. Hook `GameManager` shutdown to call `close()`.
  ```python
  _writers: dict[str, RunWriter] = {}

  @app.post("/api/internal/step")
  async def receive_step(step: StepData) -> None:
      writer = _writers.get(step.run_name)
      if writer is None:
          # Phase 0 pivot: the writer owns the SINGLE DuckLakeStore for
          # this run (read+write). RunRegistry shares this instance with
          # RunReader via attach_writer_store. Do not construct a separate
          # read_only=True DuckLakeStore here -- DuckDB rejects it as
          # "Unique file handle conflict".
          run_dir = _data_dir / step.run_name
          store = DuckLakeStore(run_dir, read_only=False)
          writer = RunWriter(
              step.run_name, _registry, _step_cache, _exporter, store
          )
          _writers[step.run_name] = writer
      writer.push_step(step)
  ```

- `dashboard-ui/src/types/api.ts`: add `tail_growing: boolean` to the
  `StepRange` interface.

- `dashboard-ui/src/state/context.tsx`: delete (~25 lines)
  - `liveRunName`, `setLiveRunName`
  - `liveGameNumber`, `setLiveGameNumber`
  - `liveGameActive`, `setLiveGameActive`
  - The corresponding context-value entries.

- `dashboard-ui/src/components/status/StatusBar.tsx`: replace the
  `liveGameActive` read with `useStepRange(run).data?.tail_growing ?? false`.

- `dashboard-ui/src/hooks/useLiveUpdates.ts`: delete `liveStatus` polling
  and the `pollLiveStatusRef` initialization. (The whole hook will be
  deleted in Phase 6; in Phase 3 we remove just the live-status path.)

**Dependencies**:
- External: none.
- Internal: Phase 1 (`RunRegistry`, `RunReader`), Phase 2 (`_indices` gone).

**Verification**:
1. Run: `make test`, `pnpm -C dashboard-ui exec vitest run`
2. Manual:
   - `make run`
   - Open the dashboard with no game running. The GO LIVE badge is
     **not** visible. Use the React DevTools or the network tab: every
     `/api/runs/<r>/step-range` response has `"tail_growing": false`.
   - Click "Game" → "Start". Within ~1s, navigate to the new run; the
     GO LIVE badge appears. The step-range response now has
     `"tail_growing": true`.
   - Stop the game. The badge disappears. The next step-range response
     has `"tail_growing": false`.
3. Grep check:
   `grep -rn "liveRunName\|liveGameActive" dashboard-ui/src/` should
   return zero hits.

**Reversible**: Mostly. Context fields can be added back if needed.

---

### Phase 4: Socket.io as Invalidation; Delete `liveData` and Refs

**Goal**: Change the Socket.io payload from full `StepData` to `{run, step}`
notifications. Frontend handler calls `queryClient.invalidateQueries`. Delete
`liveData` state, the `isFollowing ?` ternary, and the entire ref pile in
`App.tsx`.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/reporting/test_run_registry.py`: extend
  ```python
  def test_subscribe_callback_invoked_on_update_max_step(tmp_path):
      reg = RunRegistry(tmp_path)
      _seed_run(tmp_path, "r", steps=0)
      received = []
      reg.subscribe("r", lambda step: received.append(step))
      reg.update_max_step("r", 1)
      reg.notify_subscribers("r", 1)
      assert received == [1]

  def test_unsubscribe_stops_callbacks():
      reg = RunRegistry(tmp_path)
      received = []
      unsub = reg.subscribe("r", received.append)
      unsub()
      reg.notify_subscribers("r", 1)
      assert received == []

  def test_subscriber_exception_does_not_break_other_subscribers():
      reg = RunRegistry(tmp_path)
      reg.subscribe("r", lambda step: 1/0)
      good = []
      reg.subscribe("r", good.append)
      reg.notify_subscribers("r", 1)
      assert good == [1]
  ```

- `tests/unit/reporting/test_api_server.py`: extend
  ```python
  async def test_subscribe_run_event_emits_step_added_to_caller(sio_client):
      await sio_client.emit("subscribe_run", "r")
      _push_step("r", 1)
      msg = await sio_client.wait_for("step_added", timeout=1)
      assert msg == {"run": "r", "step": 1}

  async def test_socket_payload_does_not_contain_full_step_data(sio_client):
      await sio_client.emit("subscribe_run", "r")
      _push_step("r", 1)
      msg = await sio_client.wait_for("step_added")
      assert "screen" not in msg and "metrics" not in msg
  ```

- `dashboard-ui/src/hooks/useRunSubscription.test.tsx`: new
  ```typescript
  it("emits subscribe_run on mount and unsubscribe on unmount", () => {
      const { unmount } = renderHook(() => useRunSubscription("r"), { wrapper });
      expect(socket.emit).toHaveBeenCalledWith("subscribe_run", "r");
      unmount();
      expect(socket.emit).toHaveBeenCalledWith("unsubscribe_run", "r");
  });

  it("invalidates step-range query on step_added", async () => {
      const queryClient = new QueryClient();
      renderHook(() => useRunSubscription("r"), { wrapper: wrap(queryClient) });
      const spy = vi.spyOn(queryClient, "invalidateQueries");
      socket.emit("step_added", { run: "r", step: 5 });
      await waitFor(() => {
          expect(spy).toHaveBeenCalledWith({ queryKey: ["step-range", "r"] });
      });
  });

  it("ignores step_added events for other runs", () => {
      const queryClient = new QueryClient();
      renderHook(() => useRunSubscription("r1"), { wrapper: wrap(queryClient) });
      const spy = vi.spyOn(queryClient, "invalidateQueries");
      socket.emit("step_added", { run: "r2", step: 5 });
      expect(spy).not.toHaveBeenCalled();
  });
  ```

- `tests/e2e/test_live_update_via_invalidation.py` (Playwright): new
  ```python
  async def test_dashboard_advances_as_steps_arrive(page):
      await page.goto(DASHBOARD_URL)
      await page.click('button:has-text("Game")')
      await page.click('button:has-text("Start")')
      # Auto-follow on by default for new runs.
      step_label = page.locator('[data-testid="current-step"]')
      await expect(step_label).to_have_text("1", timeout=10_000)
      await expect(step_label).not_to_have_text("1", timeout=15_000)
  ```

**Implementation**:

- `roc/reporting/api_server.py`:
  - Add `subscribe_run` / `unsubscribe_run` Socket.io event handlers.
    Each binds `RunReader.subscribe(run, on_step)` and stores the
    unsubscribe by `sid`. On disconnect, run all unsubscribes.
  - Wire `RunWriter.push_step` to call
    `self._registry.notify_subscribers(self._run, data.step)`.
  - Change the existing `new_step` broadcast from full `StepData` to
    `{run, step}`. The legacy broadcast stays for one phase to allow a
    rollback.

- `dashboard-ui/src/hooks/useRunSubscription.ts`: new. ~40 lines.
  ```typescript
  export function useRunSubscription(run: string): void {
      const queryClient = useQueryClient();
      useEffect(() => {
          if (!run) return;
          socket.emit("subscribe_run", run);
          const onStepAdded = ({ run: r, step }: { run: string; step: number }) => {
              if (r !== run) return;
              queryClient.invalidateQueries({ queryKey: ["step-range", run] });
          };
          socket.on("step_added", onStepAdded);
          return () => {
              socket.off("step_added", onStepAdded);
              socket.emit("unsubscribe_run", run);
          };
      }, [run, queryClient]);
  }
  ```

- `dashboard-ui/src/App.tsx`: delete (~80 lines)
  - `liveData` state.
  - `liveRunSelected`, `initialUrlRun` refs.
  - The whole ref pile: `runRef`, `gameRef`, `stepRef`, `stepMinRef`,
    `stepMaxRef`, `isFollowingRef`, `liveRunNameRef`.
  - The `onNewStep` callback.
  - The `data = isFollowing ? liveData ?? restData : restData` ternary.
    Replace with `const data = restData;`.
  - Add a single `useRunSubscription(run)` call near the top of the
    component.
  - Add a temporary inline auto-follow effect that watches
    `useStepRange.data.max` and advances `step` when `autoFollow &&
    tail_growing && step === prevMax`. (Phase 5 promotes this to a
    proper hook.)

- `dashboard-ui/src/hooks/useLiveUpdates.ts`: delete the Socket.io
  `new_step` listener body. Leave the file alive for now if other code
  imports it; full deletion is in Phase 6.

**Dependencies**:
- External: none.
- Internal: Phase 3 (`tail_growing`).

**Verification**:
1. Run: `make test`, `pnpm -C dashboard-ui exec vitest run`,
   `pnpm -C dashboard-ui exec playwright test`.
2. Manual:
   - `make run`
   - Start a game. The dashboard shows live data updating in real time.
   - Click the slider to scrub backward. Auto-follow drops; navigation
     works; the live game continues writing in the background.
   - Click the GO LIVE badge. The dashboard catches up to the latest
     step and resumes following.
   - Set playback speed to 10x. The dashboard does not get "stuck" or
     fall behind by more than a frame.
3. Grep check:
   `grep -n "liveData\|onNewStep\|runRef\|gameRef\|isFollowingRef" dashboard-ui/src/App.tsx`
   should return zero hits.
4. Network tab: confirm Socket.io `step_added` payloads contain only
   `{run, step}` -- no `screen`, `metrics`, etc.

**Reversible**: Hard. Mitigated by Playwright coverage and a phase
boundary so Phase 4 can be reverted as a unit.

---

### Phase 5: Collapse Playback State Machine to `autoFollow`

**Goal**: Replace the four-state `playbackReducer` with two booleans
(`playing`, `autoFollow`). Update all consumers.

**Duration**: 1 day

**Tests to Write First**:

- `dashboard-ui/src/state/playback.test.ts`: rewrite (replaces the old
  state-machine transition tests).
  ```typescript
  describe("playback (boolean model)", () => {
      it("autoFollow defaults to true on a new live run", () => {
          const { result } = renderHook(() => usePlayback(), { wrapper });
          expect(result.current.autoFollow).toBe(true);
      });

      it("explicit navigation drops autoFollow to false", () => {
          const { result } = renderHook(() => usePlayback(), { wrapper });
          act(() => result.current.navigate({ step: 5 }));
          expect(result.current.autoFollow).toBe(false);
      });

      it("GO LIVE sets autoFollow=true and snaps to head", () => {
          const range = { min: 1, max: 100, tail_growing: true };
          const { result } = renderHook(() => usePlayback(range), { wrapper });
          act(() => result.current.goLive());
          expect(result.current.autoFollow).toBe(true);
          expect(result.current.step).toBe(100);
      });

      it("autoFollow + tail_growing pulls step forward as max grows", () => {
          // simulate range.data.max increment via TanStack Query refetch
      });

      it("autoFollow does nothing when tail_growing is false", () => {
          // verify the auto-follow effect bails out for closed runs
      });

      it("playing is independent of autoFollow", () => {
          const { result } = renderHook(() => usePlayback(), { wrapper });
          act(() => result.current.setPlaying(true));
          expect(result.current.playing).toBe(true);
          expect(result.current.autoFollow).toBe(true); // unchanged
      });
  });
  ```

- `dashboard-ui/src/components/transport/TransportBar.test.tsx`: modify
  - Replace assertions on `playback === "live_following"` with checks on
    `autoFollow`.

**Implementation**:

- `dashboard-ui/src/state/playback.ts`: replace contents.
  ```typescript
  export interface PlaybackState {
      playing: boolean;
      autoFollow: boolean;
  }

  export const initialPlayback: PlaybackState = {
      playing: false,
      autoFollow: true,  // new live runs default to following
  };
  ```
  Delete the reducer, all action types, and all transition functions.

- `dashboard-ui/src/state/context.tsx`: replace the `playback` field with
  `playing`, `setPlaying`, `autoFollow`, `setAutoFollow`.

- `dashboard-ui/src/App.tsx`:
  - Promote the inline auto-follow effect from Phase 4 into a proper
    `useAutoFollow(run, step)` hook (or keep inline -- the design notes
    that hooks are optional here).
  - Replace `dispatchPlayback({ type: "GO_LIVE" })` with
    `setAutoFollow(true); navigate({ step: range.data.max })`.
  - Replace `dispatchPlayback({ type: "USER_NAVIGATE" })` with
    `setAutoFollow(false)`.

- `dashboard-ui/src/components/transport/TransportBar.tsx`: replace
  `playback === "live_following"` with `autoFollow`.

- `dashboard-ui/src/components/status/StatusBar.tsx`: replace
  `playback === "live_following"` with `autoFollow`.

**Dependencies**:
- External: none.
- Internal: Phase 4 (`useRunSubscription`, `tail_growing` already in the
  query result).

**Verification**:
1. Run: `pnpm -C dashboard-ui exec vitest run`
2. Manual:
   - `make run`
   - Start a game. Dashboard auto-follows; new steps appear as they
     arrive.
   - Click on a previous step in the slider. Auto-follow drops; the
     viewer stays on that step; new pushes do **not** advance.
   - Click GO LIVE. Dashboard snaps to the latest step and resumes
     following.
   - Pause playback while live. Steps still arrive (the data is fetched)
     but the slider does not advance. Resume; auto-follow continues.
   - Switch games via the game dropdown while following. Auto-follow
     resets to the new game's head.
3. Grep check:
   `grep -rn "playback ===\|live_following\|live_paused\|live_catchup\|playbackReducer" dashboard-ui/src/`
   should return zero hits.

**Reversible**: Yes -- small frontend change, isolated to a few files.

---

### Phase 6: Final Cleanup; Delete Dead Code

**Goal**: Remove every method and field that exists only to support the old
live-vs-historical split. Update CLAUDE.md files. No new behavior; this is a
deletion-only pass.

**Duration**: 1 day

**Tests to Write First**:

- No new tests. Existing tests should keep passing as code is removed.

**Implementation**:

- `roc/reporting/data_store.py`: delete (~400 lines)
  - `_live_run_name`, `_live_buffer`, `_live_store` fields.
  - `_run_stores`, `_run_summary_cache` fields (replaced by
    `RunRegistry._entries`).
  - `set_live_session()`, `clear_live_session()`.
  - `_is_live()`, `is_live()`.
  - `_supplement_logs()`.
  - All `if self._is_live(...)` branches in remaining methods.
  - The class itself becomes a thin shim or is deleted entirely; if a
    few callers still exist, point them at `RunReader` directly.

- `roc/reporting/step_buffer.py`: delete the file. Replace any remaining
  imports with `roc/reporting/step_cache.py`.

- `roc/reporting/graph_api.py`: remove the live-cache vs historical-archive
  routing in graph endpoints; route everything through `RunReader.get_history`.

- `roc/reporting/api_server.py`: remove any leftover branches that
  reference `is_live()` or check whether a run is "live" outside the
  `tail_growing` field.

- `dashboard-ui/src/hooks/useLiveUpdates.ts`: delete the file. Update
  imports across `dashboard-ui/src/` to remove references.

- `dashboard-ui/src/state/context.tsx`: delete the URL-sync effect (the
  search params hook does this directly now).

- `CLAUDE.md` files (per the design's "CLAUDE.md Updates" section):
  - `/CLAUDE.md`: add Architectural Invariant #11 ("Run data has one read
    path through `RunReader`").
  - `/CLAUDE.md`: rewrite the React Dashboard data-flow paragraph.
  - `roc/reporting/CLAUDE.md`: rewrite Key Decisions, Invariants,
    Non-Obvious Behavior, Anti-Patterns, Interfaces sections.
  - `dashboard-ui/CLAUDE.md`: delete the "Stale Closure Pattern" section,
    delete "Live/historical feature parity" invariant, replace the
    "Playback State Machine" section with the two-boolean model.
  - `roc/game/CLAUDE.md`: add a small note about `RunWriter` usage.

**Dependencies**:
- External: none.
- Internal: Phases 1-5 (all functionality on the new path).

**Verification**:
1. Run: `make test`
2. Run: `make lint`
3. Run: `pnpm -C dashboard-ui exec vitest run`
4. Run: `pnpm -C dashboard-ui exec tsc --noEmit`
5. Run: `pnpm -C dashboard-ui exec playwright test`
6. Manual smoke test (full lifecycle):
   - Start a game; play to step ~50; pause; scrub back to step 10;
     scrub forward to step 30; click GO LIVE; let it catch up; stop
     the game; navigate to a different historical run; come back to
     the original; verify all panels still load.
7. Grep checks (must return zero hits):
   - `grep -rn "is_live\|_live_buffer\|_live_run_name\|_indices\|_GameIndex\|_supplement_logs" roc/reporting/`
   - `grep -rn "liveData\|liveRunName\|liveGameActive\|runRef\|gameRef\|playbackReducer" dashboard-ui/src/`
8. Sanity check: open `roc/reporting/data_store.py`. It should be ~80
   lines or fewer (or deleted).

**Reversible**: Doesn't need to be -- this is a deletion-only pass and
the code being removed is dead by this point.

---

## Common Utilities Needed

- **`StepCache` (`roc/reporting/step_cache.py`)** -- Process-wide LRU
  keyed by `(run, step)`. Backed by `collections.OrderedDict` + a
  `threading.Lock`. No new dependency. Used by `RunReader` (cache-first
  reads) and `RunWriter` (write-through). Replaces the role of today's
  `StepBuffer` but is multi-run and not exposed in any public API.

- **`StepResponse` envelope (`roc/reporting/types.py` or
  `api_models.py`)** -- Typed response that explicitly distinguishes
  `ok`, `run_not_found`, `out_of_range`, `not_emitted`, and `error`.
  Eliminates the "blank panel = ambiguous failure" class of bug. Used by
  every step-fetching endpoint.

- **`RunRegistry` lock pattern** -- Single `threading.RLock` covering
  all per-run state. Notifications snapshot subscribers under the lock
  and dispatch outside the lock to prevent deadlock. Reused for
  registry mutations from `RunWriter` and `RunReader`.

- **`useRunSubscription` hook (`dashboard-ui/src/hooks/useRunSubscription.ts`)**
  -- The single point where Socket.io meets TanStack Query. All future
  invalidation flows go through this hook; new code should not add a
  parallel subscription.

- **`useDashboardLocation` hook** (introduced incrementally in Phase 3
  and finalized in Phase 5) -- Wraps `useSearchParams` so the URL is the
  direct input for `run`/`game`/`step` instead of being mediated through
  `DashboardContext`.

## External Libraries Assessment

- **LRU cache**: Use `collections.OrderedDict` (stdlib). Considered
  `cachetools.LRUCache` but stdlib is sufficient and avoids a new
  dependency. The existing `StepBuffer` already uses this pattern; we
  generalize it.

- **Socket.io invalidation hook**: Use existing `socket.io-client` plus
  TanStack Query's `invalidateQueries`. No new library needed.
  Considered the `react-query` Socket.io plugins but they target a
  different use case (subscription-as-query) and would not simplify
  the contract here.

- **Concurrent benchmark (Phase 0)**: Use `pytest-benchmark` if Phase 2
  uses a benchmark fixture; otherwise stdlib `time.perf_counter`. No
  new dependency required.

- **Playwright tests (Phase 4)**: Already in the project under
  `tests/e2e/`. Reuse existing harness; no new dependency.

## Risk Mitigation

- **Risk: DuckLake reads slow under concurrent writes** (HIGH -- RESOLVED 2026-04-08).
  Phase 0 measured this. Outcome: stronger than the original risk
  predicted. A second `DuckLakeStore(read_only=True)` cannot be opened
  in the same process while a writer's `DuckLakeStore` is attached to
  the same catalog file -- DuckDB rejects it as
  `BinderException: Unique file handle conflict`, regardless of the
  DuckLake alias. The closed-run path (read after `writer.close()`)
  works fine; the constraint is purely on the in-process active-writer
  case. **Resulting rule (now load-bearing for Phase 1+)**: `StepCache`
  is mandatory; for active runs `RunRegistry._RunEntry.store` MUST be
  the same `DuckLakeStore` instance held by the active `RunWriter`,
  installed via `attach_writer_store` and dropped via
  `detach_writer_store`. Phase 1 must add a corresponding invariant to
  `roc/reporting/CLAUDE.md`. See `tmp/ducklake_concurrency_spike.py`
  for the measurement and `design/unified-run-architecture.md` Risk #2
  for the full rationale.

- **Risk: History query performance via DuckLake** (MEDIUM).
  Mitigation: Phase 2 measures this on a 10K-step run. If it's too
  slow, add a per-run history cache inside `RunReader.get_history`
  keyed by run + kind + game, evicted on `mtime` change. The public
  API does not change.

- **Risk: Frontend playback regression** (LOW-MEDIUM).
  Mitigation: Phase 5 rewrites the tests as user-perspective scenarios
  ("GO LIVE → new step arrives → step advances", "user clicks at step
  50 while following → autoFollow drops → next push does not advance")
  before the refactor. Playwright covers the visible behaviors end-to-end
  in Phase 4.

- **Risk: Race condition between `RunWriter.close()` and in-flight
  reads** (LOW).
  Mitigation: `RunRegistry`'s single `RLock` serializes
  `detach_writer_store()` against `get()`. A reader holding a
  reference to the writer's store from before `close()` will block on
  the writer's `_lock` for the duration of the final flush; this is
  acceptable because the closed-run path (which the next `get()` will
  open) cannot start until the writer's store is fully closed.
  Subscribers are notified outside the lock; a callback that arrives
  after `close()` is harmless because the queryClient invalidation is
  idempotent.

- **Risk: Multiple `_writers` for the same run name** (LOW).
  Mitigation: `_writers` dict in `api_server.py` ensures one writer per
  run. `GameManager` is the only thing that creates writers, and it
  enforces one game at a time. If a name collision did occur, the
  existing entry would be reused -- documented as "do not reuse run
  names" (Open Question #3 in the design).

- **Risk: Test churn** (LOW).
  Mitigation: Most coupled tests get deleted along with the code they
  test. Behavioral tests translate directly. Phase 2 absorbs the
  largest test rewrite (~10-12 tests in `test_data_store.py`).

- **Risk: Phase 4 is hard to revert** (MEDIUM).
  Mitigation: Phase 4 is broken into a clear sequence (add new code,
  flip the switch, delete old code) so a partial revert is possible.
  Playwright tests in Phase 4 catch regressions before the cleanup of
  the legacy `new_step` broadcast.

- **Risk: In-process dashboard mode (`uv run play`) regresses** (LOW).
  Mitigation: Phase 1 explicitly verifies that `RunReader` /
  `RunWriter` work when the game thread and the dashboard server share
  the same process (Open Question #6). If anything diverges, the fix
  is contained to the `_writers` initialization in `api_server.py`.
