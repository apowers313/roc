# Dashboard Bug-fix Plan

**Created**: 2026-04-09
**Source**: Comprehensive Playwright validation by general-purpose subagent on
2026-04-09 (after Phase 5 completion). Found 14 bugs in current dashboard /
backend behavior — most cascade from a partial Phase 2 RunRegistry migration
that left half the REST endpoints on the legacy `DataStore._get_run_store`
path.

## Background

The unified-run architecture (see `design/unified-run-architecture.md` and
`design/unified-run-implementation-plan.md`) introduced `RunRegistry`,
`RunReader`, `RunWriter`, and `StepCache` to consolidate the dashboard's
read/write paths. Phase 2 was supposed to migrate every endpoint through
`RunReader`, but four endpoints were missed:

- `GET /api/runs/{run}/games`         → still uses `DataStore.list_games`
- `GET /api/runs/{run}/all-objects`   → still uses `DataStore.get_all_objects`
- `GET /api/runs/{run}/schema`        → still uses `DataStore.get_schema`
- `GET /api/runs/{run}/action-map`    → still uses `DataStore.get_action_map`

`DataStore._get_run_store` lazily opens a `DuckLakeStore(read_only=True,
alias="r_<run>")` against the run's catalog. When the same run has already
been opened by `RunRegistry._load` (alias `reg_<run>_<n>`), DuckDB rejects the
second attach with `BinderException: Unique file handle conflict`. The
endpoints return HTTP 500 and the dashboard cascades into a broken state.

The frontend then makes things worse: when the explicit `?run=<name>` URL's
`/games` call 500s, the run-selection effect falls back to `runs[0]` and
silently rewrites the URL — violating the documented "URL parameter
sovereignty" invariant in `dashboard-ui/CLAUDE.md`.

The default landing page also picks the wrong step: it uses the run's total
step count instead of the per-game range, so a run with 256 total steps but
only 6 in game 2 ends up requesting step 256 of game 2 → 500 → "no data
everywhere".

These three issues combine to make the dashboard appear completely broken on
first load.

## Bug List

| ID  | Severity | Title                                                                  |
| --- | -------- | ---------------------------------------------------------------------- |
| C1  | Critical | DuckLake catalog double-attach 500s on /games, /all-objects, /schema, /action-map |
| C2  | Critical | URL parameter sovereignty violated -- explicit ?run=X silently redirects |
| H1  | High     | Default landing computes step from run total instead of per-game range |
| H2  | High     | /step-range without game param returns 0/0 for some runs               |
| H3  | High     | AllObjects popout silently filters by current game; no indicator       |
| H4  | High     | Same node_id returns different PHYSICAL features per game filter (Invariant 9 violation) |
| M1  | Medium   | Default run auto-selection picks newest by name without checking renderability |
| M2  | Medium   | step_added=null shows "--" with no tooltip explanation                 |
| M3  | Medium   | 13 panels show empty states cascading from C1/H1                       |
| M4  | Medium   | "B" key (bookmark) silently no-ops in ERROR state                      |
| M5  | Medium   | Graph & Events popout shows "no history" while inline GraphViz works   |
| M6  | Medium   | Action Histogram popout empty despite 256-step run                     |
| L1  | Low      | Cytoscape `wheelSensitivity` warning fires twice on every page load    |
| L2  | Low      | GraphViz emits 25+ `elements:rebuild` debug messages per page load     |

---

## Phased Plan

The phases are ordered by **dependency**, not severity. Fixing C1 first will
mechanically resolve M3 (the cascading panel emptiness) and unblock testing of
M5 / M6. Fixing the URL sovereignty + default-step bugs together is the
correct unit because they share the App.tsx auto-select effect.

### Phase 1 — Backend unification (fix C1, the cascade root)

**Goal**: Stop `DataStore` and `RunRegistry` from racing for the same DuckDB
catalog. After this phase the four broken endpoints return 200 for every
historical run.

**Approach (preferred)**: Migrate `list_games`, `get_all_objects`,
`get_schema`, `get_action_map` from `DataStore` to `RunReader`, mirroring the
existing Phase 2 pattern used by `get_step` and the history endpoints. This
removes `DataStore._get_run_store` from the hot path entirely. The closed-run
read path (lazy `read_only=True` open) lives in `RunRegistry._load`; the
active-run read path (writer's instance shared via `attach_writer_store`)
lives in `RunWriter` / `RunRegistry.attach_writer_store`. Both are already
implemented and tested — we just need to use them.

**Files to modify**:

- `roc/reporting/run_reader.py`
  - Add `list_games(run, *, include_buffer_for_live=True) -> list[GameSummary]`.
    Read from `entry.run_store.list_games()`. For active runs the live buffer
    has the freshest counts; check `entry.range.tail_growing` to decide
    whether to consult `_data_store.list_games(run)` for the live path until
    Phase 6 deletes the live buffer entirely.
  - Add `get_all_objects(run, game=None) -> list[dict]`. Delegate to
    `entry.run_store.get_all_objects(game)`.
  - Add `get_schema(run) -> dict | None`. Read `schema.json` directly from
    `self._registry._data_dir / run / "schema.json"` (no DuckLake involved).
  - Add `get_action_map(run) -> list[dict] | None`. Read `action_map.json`
    directly. Both files are written by the writer; reading is filesystem-only.
- `roc/reporting/api_server.py`
  - `list_games`: replace `_data_store.list_games` with `_get_reader().list_games`.
    Translate `FileNotFoundError` to 404.
  - `get_all_objects`: replace `_data_store.get_all_objects` with
    `_get_reader().get_all_objects`.
  - `get_schema`: replace `_data_store.get_schema` with `_get_reader().get_schema`.
  - `get_action_map`: replace `_data_store.get_action_map` with
    `_get_reader().get_action_map`.
- `roc/reporting/data_store.py`
  - Mark `_get_run_store`, `list_games`, `get_all_objects`, `get_schema`,
    `get_action_map`, `set_action_map` as **legacy**. Phase 6 deletes them.
  - Do NOT delete in this phase — `set_action_map` is still called by the
    game writer at run start. Add a TODO with the Phase 6 reference.

**Tests to write FIRST** (TDD):

- `tests/integration/reporting/test_catalog_double_attach.py` — new file
  ```python
  def test_list_games_after_registry_load_does_not_double_attach(tmp_path):
      # Seed a run, force RunRegistry to open read_only, then call
      # list_games via RunReader. Must NOT raise BinderException.
      _seed_run(tmp_path, "double-attach", steps=5)
      cache = StepCache()
      registry = RunRegistry(tmp_path)
      _ = registry.get("double-attach")  # opens reg_double_attach
      reader = RunReader(registry, cache)
      games = reader.list_games("double-attach")
      assert len(games) >= 1

  def test_get_all_objects_after_registry_load_does_not_double_attach(tmp_path):
      # Same pattern for get_all_objects
      ...

  def test_get_schema_filesystem_only(tmp_path):
      # Should not touch DuckLake at all
      ...
  ```
- `tests/unit/reporting/test_api_server.py` — extend
  ```python
  def test_list_games_uses_reader_not_data_store(client):
      # Mock the reader, verify endpoint calls reader.list_games
      ...

  def test_list_games_returns_404_for_unknown_run(client):
      ...
  ```

**Verification**:

1. `make test`
2. Manual: `curl -sk https://dev.ato.ms:9043/api/runs/<HISTORICAL_RUN>/games`
   should return 200 for any run that previously 500'd.
3. Server log: zero `BinderException: Unique file handle conflict` errors
   during normal browsing.
4. Browse to several historical runs in the dashboard; all sections render.

**Expected impact**: BUG-C1 fixed. BUG-M3 mostly resolves (panels start
showing data). BUG-M5 / BUG-M6 likely resolved or surfaced as different bugs
to investigate in Phase 3.

---

### Phase 2 — Frontend URL sovereignty + default selection (C2, H1, M1)

**Goal**: Restore the documented "URL parameter sovereignty" invariant. When
`?run=X` is in the URL, NEVER overwrite it with a different run, even if X's
endpoints fail. Default landing must compute the correct step for the
selected game, not the run total.

**Files to modify**:

- `dashboard-ui/src/App.tsx`
  - The auto-select-live-run effect (around lines 396-423 in current HEAD)
    must check `initialUrlRun.current` first. If an explicit URL run is set
    AND it differs from `liveStatus.run_name`, do NOT auto-navigate.
  - The fallback "first run" effect in `TransportBar.tsx` (around lines
    174-188) sets `setRun(name)` whenever `runs[0]` exists and `run` is
    empty. This is fine on a totally empty URL but must NOT clobber an
    explicit URL whose `/games` call failed. Add a guard:
    `if (initialUrlRun.current) return;`
- `dashboard-ui/src/state/context.tsx`
  - The default-step selection happens implicitly via `setStepRange` clamp
    in the `setRun` callback chain. The actual choice of "what step to land
    on for a new run" is in `TransportBar.tsx`'s effect that fetches
    `step-range?game=1`. The bug: this always passes `game=1` even if the
    selected game is N. Fix: pass the current `game` from context.
- `dashboard-ui/src/components/transport/TransportBar.tsx`
  - The auto-select-first-run effect should use the per-game step-range to
    compute the default step, not the global `min`.
- New error UI in StatusBar or App when an explicit URL fails to load:
  - Show a red banner: "Run '<name>' could not be loaded: <error>". Provide a
    "Browse runs" button that clears the URL and navigates to the home view.

**Tests to write FIRST**:

- `dashboard-ui/src/App.url-sovereignty.test.tsx` — new file
  ```typescript
  it("does not override explicit ?run=X when X's /games call returns 500", async () => {
      // Set window.location.search to ?run=broken-run&game=1&step=10
      // Mock /api/runs/broken-run/games to 500
      // Mock /api/live/status to return active=true, run_name=other-run
      // Render App
      // Wait for fetches
      // Assert: window.location.search still contains run=broken-run
      // Assert: an error banner is visible
  });

  it("does not auto-navigate to liveStatus run when URL has explicit run", () => {
      // Set ?run=specific-run
      // Mock liveStatus active for different-run
      // Render App
      // Assert: setRun was never called with "different-run"
  });
  ```
- `dashboard-ui/src/components/transport/TransportBar.default-step.test.tsx`
  ```typescript
  it("computes default step from per-game step-range, not run total", () => {
      // Run has 256 total steps; game 2 only has steps 251-256
      // useGames returns [{game_number:1, steps:250}, {game_number:2, steps:6}]
      // useStepRange(run, 2) returns {min:251, max:256}
      // Render
      // Slider should show 1/6 (game 2 has 6 steps), not 256/256
  });
  ```

**Verification**:

1. `pnpm -C dashboard-ui exec vitest run`
2. Manual:
   - Navigate to a known-broken run (until C1 is fixed, any historical run);
     verify URL is preserved and an error message appears.
   - Navigate to `?run=<good-run>&game=2&step=<game2-max>`; verify the right
     game and step load.
3. Network tab: confirm `/api/runs/<run>/step-range?game=<n>` is called
   instead of without the game param when the URL specifies a game.

---

### Phase 3 — Cross-game step range + history endpoints (H2, M5, M6)

**Goal**: Fix the broken cross-game `step_range` query that returns 0/0 for
some runs. Investigate whether the history endpoint emptiness in M5 / M6 is a
separate bug or cascades from H2.

**Investigation first**: After Phase 1, re-test the popouts on a known-good
historical run. If they now show data, M5 / M6 are resolved by Phase 1 and
this phase is shorter.

**Files to investigate**:

- `roc/reporting/run_store.py:108-123` — `step_range(game_number)`. The query
  is `SELECT MIN(step), MAX(step) FROM <table>` (no game filter when
  `game_number is None`). Verify that it actually returns the correct min/max
  across all games. If `screens` table has rows for game 1 but not game 2 due
  to a writer bug, the unfiltered query may still work — but the test against
  the `jittery-eliot-oringa` run shows 0/0 instead of 256. Check whether the
  table is empty or whether there's a cache issue.
- `roc/reporting/run_registry.py:198-207` — when `attach_writer_store`
  populates the initial range from `run_store.step_range(None)`, it can
  catch the same 0/0 issue and never get updated by `update_max_step` if the
  writer never pushed steps.
- `roc/reporting/run_registry.py:161-171` — `update_max_step` keeps
  `range.min` at 1 if previously 0; but if the run is closed and reopened
  (closed-run path via `_load`), the range should reflect the persisted min.
- `roc/reporting/run_store.py:_query_table` — make sure it actually scans the
  Parquet files for closed runs and not just the catalog inlining.

**Hypothesis to verify** (most likely cause of H2): The `jittery-eliot-oringa`
run was the run I started during the silent-idle bug investigation. The
writer wrote steps to the StepBuffer / via the OTel exporter, but the
exporter may not have flushed (no CHECKPOINT) before the writer was detached.
After detach, `RunRegistry._load` opens read-only and `RunStore.step_range`
queries the catalog, but the data lives in inlined catalog rows that the
read-only handle can or cannot see depending on the inlining state.

**Files to modify** (TBD after investigation):

- Likely `roc/reporting/run_writer.py` — add a `flush()` call before
  `detach_writer_store()` so the parquet files are merged before the closed-
  run path opens read-only.
- Likely `roc/reporting/run_store.py` — the `step_range(None)` query may need
  to UNION across `screens` and `metrics` tables if the writer used different
  emit modes for different games.

**Tests to write FIRST**:

- `tests/integration/reporting/test_run_store_step_range.py` — new
  ```python
  def test_step_range_after_writer_detach_returns_persisted_max(tmp_path):
      # Seed run with writer, push 10 steps, detach (close), reopen via _load
      # step_range(None) must return (1, 10), not (0, 0)
      ...

  def test_step_range_with_multiple_games_returns_global_max(tmp_path):
      # Seed run with game 1 (steps 1-50) and game 2 (steps 51-56)
      # step_range(None) must return (1, 56)
      ...
  ```

**Verification**:

1. `make test`
2. Manual: `curl -sk .../runs/<RUN>/step-range` returns the correct
   `{min, max}` for every historical run.
3. Manual: open the Graph & Events popout for a multi-game run and verify
   data renders.

---

### Phase 4 — Object data integrity (H4, H3)

**Goal**: Fix the architectural invariant violation where the same object's
PHYSICAL features change depending on game filter. Update the AllObjects
popout UX to make filtering visible.

**Root cause** (BUG-H4): `RunStore.get_all_objects` (run_store.py:314-344)
reconstructs objects from `roc.resolution.decision` events. When filtered by
`game_number`, it only sees that game's events, and the latest reported
`shape`/`glyph`/`color` becomes the "object's" attributes. This is wrong:
PHYSICAL features (shape, glyph, color) are object identity invariants, not
event-derived.

**Fix approach (preferred)**: Persist a separate `objects` table during run
archival that records the canonical PHYSICAL attributes for each
`object.uuid` exactly once (at first observation). The query then becomes a
simple JOIN:

```sql
SELECT o.uuid, o.shape, o.glyph, o.color, o.first_step,
       COUNT(d.id) AS match_count
FROM lake.objects o
LEFT JOIN lake.events d
  ON d.object_uuid = o.uuid
  AND d.event_name = 'roc.resolution.decision'
  AND (? IS NULL OR d.game_number = ?)
GROUP BY o.uuid;
```

**Fix approach (interim, less work)**: Continue reconstructing from events
but query ALL events (no game filter) to get the canonical features, then
overlay match counts from the filtered query. This preserves identity while
allowing per-game match-count views.

**Files to modify**:

- `roc/reporting/run_store.py`
  - Rewrite `get_all_objects` and helpers `_collect_resolution_decisions`,
    `_link_object_node_ids`, `_apply_match_events`. The unfiltered query
    populates the canonical PHYSICAL features; the filtered query only
    contributes match counts.
- `dashboard-ui/src/components/panels/AllObjects.tsx` (BUG-H3)
  - Add a "Show all games" toggle that defaults to OFF (current behavior).
  - When the toggle is OFF, change the popout title to "Objects in Game N".
  - Show "<filtered>/<total>" count.
  - Add a tooltip on `step_added: --` (BUG-M2): "created in an earlier game"
    when the user is filtering by game.

**Tests to write FIRST**:

- `tests/unit/reporting/test_run_store_objects.py`
  ```python
  def test_get_all_objects_physical_features_invariant_across_game_filters(tmp_path):
      # Seed a run with object node_id=-87 first observed in game 1 as @/333,
      # then observed in game 2 (still @/333). 
      # get_all_objects() and get_all_objects(game=2) must return the same
      # shape/glyph/color for node_id -87.
      ...

  def test_get_all_objects_match_count_filtered_by_game(tmp_path):
      # Same object, 100 matches in game 1, 4 matches in game 2.
      # get_all_objects() returns match_count=104.
      # get_all_objects(game=2) returns match_count=4.
      ...
  ```
- `dashboard-ui/src/components/panels/AllObjects.test.tsx`
  ```typescript
  it("shows filtered/total count when game filter is active", () => {
      // Render with 35 total, 10 in current game
      // Expect "10 / 35" text
  });
  it("toggle to 'Show all games' refetches without game filter", () => {
      ...
  });
  ```

**Verification**:

1. `make test`
2. Manual: open AllObjects on a multi-game run; verify same object node_id
   shows the same shape/glyph/color regardless of game filter.
3. Manual: verify the count display and toggle work.

---

### Phase 5 — UX papercuts (M2, M4)

**Goal**: Small frontend fixes to silence confusion.

**Files to modify**:

- `dashboard-ui/src/components/panels/AllObjects.tsx` (BUG-M2)
  - Already covered in Phase 4: tooltip on `step_added: --`.
- `dashboard-ui/src/App.tsx` or wherever `toggleBookmark` is wired (BUG-M4)
  - When called with no current step data (e.g., the dashboard is in ERROR
    state), show a Mantine notification: "Cannot bookmark: no step loaded".
  - Use Mantine's `notifications.show` or an existing toast pattern.

**Tests**:

- Add to `dashboard-ui/src/hooks/useBookmarks.test.ts`:
  ```typescript
  it("toggleBookmark in ERROR state shows a notification and does not add", () => {
      ...
  });
  ```

**Verification**: Manual.

---

### Phase 6 — Cleanup (L1, L2)

**Goal**: Eliminate console warnings and log spam.

**Files to modify**:

- `dashboard-ui/src/components/panels/GraphVisualization.tsx`
  - **BUG-L1**: Find the `wheelSensitivity` option in the cytoscape init.
    Either remove it (use cytoscape default) or, if it's intentional, wrap
    the cytoscape constructor with `console.warn` filtering. Default removal
    is preferred — the warning exists for a reason and the custom value
    likely isn't load-bearing.
  - **BUG-L2**: Find the `[GraphViz] elements:rebuild` debug log (line 2334
    per the bug report). Gate it behind a `localStorage.getItem('graphviz-debug')`
    check or a config flag. Currently fires 25+ times per page load.

**Tests**: Visual / console inspection only.

**Verification**: Open browser console; should be quiet during a normal page
load.

---

### Phase 7 — Regression coverage and smoke tests

**Goal**: Add tests that would have caught these bugs and prevent
re-introduction.

**Tests to add**:

1. **Default-page smoke test** (would have caught the C1+H1 cascade
   immediately):
   ```typescript
   // dashboard-ui/e2e/dashboard-default-loads.spec.ts (Playwright)
   test('default landing page loads without ERROR within 3 seconds', async ({page}) => {
       await page.goto(DASHBOARD_URL);
       // Wait for the StatusBar to render
       await expect(page.locator('[data-testid="status-bar"]')).toBeVisible();
       // Ensure no ERROR badge after 3 seconds
       await page.waitForTimeout(3000);
       const errorBadge = page.locator('text=ERROR');
       await expect(errorBadge).not.toBeVisible();
   });
   ```

2. **URL sovereignty regression** (would have caught BUG-C2):
   ```typescript
   // dashboard-ui/e2e/url-sovereignty.spec.ts
   test('?run=X URL is preserved even when /games returns 500', async ({page, context}) => {
       await context.route('**/api/runs/test-run/games', r => r.fulfill({status: 500, body: '{}'}));
       await page.goto(`${DASHBOARD_URL}/?run=test-run&game=1&step=10`);
       await page.waitForTimeout(2000);
       expect(page.url()).toContain('run=test-run');
   });
   ```

3. **Catalog-sharing regression** (would have caught BUG-C1):
   - Already in Phase 1's test list, repeated here for emphasis.
   - `tests/integration/reporting/test_catalog_double_attach.py`

4. **Object identity regression** (would have caught BUG-H4):
   - Already in Phase 4's test list.

**File**: `dashboard-ui/e2e/` directory exists already; add the new specs
there. Run via `pnpm -C dashboard-ui exec playwright test`.

---

## Task List

Tasks are ordered by **execution order** (dependencies first). Each task
should be a single PR or commit unit.

### Phase 1 tasks

- [ ] **T1.1**: Read `roc/reporting/run_reader.py` and `run_registry.py` to
      confirm the existing API surface and understand how `get_step` /
      `get_history` route through them. No code changes; this is the
      orientation pass.
- [ ] **T1.2**: Write failing test `test_catalog_double_attach.py` (Phase 1
      tests above).
- [ ] **T1.3**: Add `RunReader.list_games` and migrate the `list_games`
      endpoint. Make T1.2 pass.
- [ ] **T1.4**: Add `RunReader.get_all_objects` and migrate the `all-objects`
      endpoint.
- [ ] **T1.5**: Add `RunReader.get_schema` and migrate the `schema` endpoint.
      (Filesystem-only — does not touch DuckLake.)
- [ ] **T1.6**: Add `RunReader.get_action_map` and migrate the `action-map`
      endpoint.
- [ ] **T1.7**: Mark `DataStore._get_run_store`, `list_games`, `get_all_objects`,
      `get_schema`, `get_action_map` as legacy (TODO comments referencing
      Phase 6 deletion).
- [ ] **T1.8**: Run `make test` and `make lint`. Verify all green.
- [ ] **T1.9**: Manual verification: restart `roc-server`; navigate dashboard
      to a historical run; confirm no `BinderException` in server logs.

### Phase 2 tasks

- [ ] **T2.1**: Read `dashboard-ui/src/App.tsx` (the auto-select-live-run
      effect) and `dashboard-ui/src/components/transport/TransportBar.tsx`
      (the auto-select-first-run effect) to understand the current selection
      logic.
- [ ] **T2.2**: Write failing test `App.url-sovereignty.test.tsx`.
- [ ] **T2.3**: Modify the auto-select effects to respect `initialUrlRun`.
      Make T2.2 pass.
- [ ] **T2.4**: Write failing test `TransportBar.default-step.test.tsx`.
- [ ] **T2.5**: Fix the default-step calculation to use per-game range. Make
      T2.4 pass.
- [ ] **T2.6**: Add an error banner UI (StatusBar or App-level) when an
      explicit URL run fails to load. Show "Run X could not be loaded" with a
      "Browse runs" link that clears the URL.
- [ ] **T2.7**: Run vitest, build, lint. Verify all green.
- [ ] **T2.8**: Manual verification (with Phase 1 deployed): navigate to a
      specific run via URL; verify URL is preserved.

### Phase 3 tasks

- [ ] **T3.1**: Read `run_store.py:step_range`, `run_writer.py`,
      `run_registry.py:attach_writer_store` to find why `step_range(None)`
      returns 0/0 for `jittery-eliot-oringa`.
- [ ] **T3.2**: Write failing test
      `test_run_store_step_range_after_writer_detach`.
- [ ] **T3.3**: Implement the fix (likely: `RunWriter.close` flushes the
      exporter / triggers a CHECKPOINT before detaching, so `_load` sees
      persisted data).
- [ ] **T3.4**: Re-test the popouts (Graph & Events, Action Histogram). If
      they still show empty data on a multi-game run, dig into the
      `*-history` endpoints in `RunStore`. Write failing tests as needed.
- [ ] **T3.5**: Run `make test`. Verify all green.

### Phase 4 tasks

- [ ] **T4.1**: Read `run_store.py:get_all_objects` and helpers. Decide
      between "interim fix" (overlay match counts) and "proper fix" (persist
      objects table).
- [ ] **T4.2**: Write failing test
      `test_get_all_objects_physical_features_invariant`.
- [ ] **T4.3**: Implement the chosen fix. Make T4.2 pass.
- [ ] **T4.4**: Write failing test for AllObjects popout: filter indicator,
      counts, "Show all games" toggle.
- [ ] **T4.5**: Implement the AllObjects.tsx UX changes.
- [ ] **T4.6**: Run vitest + pytest. Verify all green.

### Phase 5 tasks

- [ ] **T5.1**: Add `step_added: --` tooltip in `AllObjects.tsx`.
- [ ] **T5.2**: Add bookmark notification for ERROR state in App.tsx /
      useBookmarks. Test included.

### Phase 6 tasks

- [ ] **T6.1**: Remove or guard `wheelSensitivity` in
      `GraphVisualization.tsx`. Verify console warning is gone.
- [ ] **T6.2**: Gate `[GraphViz] elements:rebuild` log behind a debug flag.
      Verify console is quiet on normal page loads.

### Phase 7 tasks

- [ ] **T7.1**: Add `dashboard-default-loads.spec.ts` Playwright e2e test.
- [ ] **T7.2**: Add `url-sovereignty.spec.ts` Playwright e2e test.
- [ ] **T7.3**: Run the full e2e suite via
      `pnpm -C dashboard-ui exec playwright test`. Verify all green.

### Final validation

- [ ] **TF.1**: Run `make test` (full Python suite).
- [ ] **TF.2**: Run `make lint`.
- [ ] **TF.3**: Run `pnpm -C dashboard-ui exec vitest run`.
- [ ] **TF.4**: Run `pnpm -C dashboard-ui run build`.
- [ ] **TF.5**: Run `pnpm -C dashboard-ui exec playwright test`.
- [ ] **TF.6**: Manual smoke via Playwright MCP: load `/`, start a game,
      verify LIVE → GO LIVE → GO LIVE click → LIVE flow, stop game, browse
      historical, open every panel and popout. **Re-run the same kind of
      comprehensive validation that found these bugs** to confirm no
      regressions.
- [ ] **TF.7**: Spawn a fresh general-purpose subagent and have it repeat
      the validation pass to find any remaining bugs.

---

## Risk and Reversibility Notes

- **Phase 1 is reversible** but invasive: it adds new methods to RunReader
  and removes uses of DataStore methods. Reverting requires re-pointing the
  endpoints back at DataStore. Keep the old DataStore methods alive (marked
  legacy) until Phase 6 cleanup.
- **Phase 2 is reversible** and isolated to two frontend files. Low risk.
- **Phase 3 may be a rabbit hole** if the root cause is in the writer's
  flush behavior. Time-box: if T3.1 takes more than 90 minutes, escalate
  for design discussion.
- **Phase 4 is the largest behavioral change**. The "proper fix" (persist
  objects table) is a schema migration; the "interim fix" (event overlay)
  is a self-contained change in `run_store.py`. Recommend interim first to
  unblock the user, then schema migration as a follow-up.
- **Phases 5, 6, 7 are independent and low-risk**.

## Files Likely Touched

**Backend (Python)**:
- `roc/reporting/run_reader.py` (Phase 1)
- `roc/reporting/run_registry.py` (Phase 3)
- `roc/reporting/run_writer.py` (Phase 3)
- `roc/reporting/run_store.py` (Phase 3, 4)
- `roc/reporting/api_server.py` (Phase 1)
- `roc/reporting/data_store.py` (Phase 1, mark legacy only)
- `tests/integration/reporting/test_catalog_double_attach.py` (new, Phase 1)
- `tests/integration/reporting/test_run_store_step_range.py` (new, Phase 3)
- `tests/unit/reporting/test_run_store_objects.py` (new, Phase 4)
- `tests/unit/reporting/test_api_server.py` (extend, Phase 1)

**Frontend (TypeScript)**:
- `dashboard-ui/src/App.tsx` (Phase 2, 5)
- `dashboard-ui/src/components/transport/TransportBar.tsx` (Phase 2)
- `dashboard-ui/src/components/panels/AllObjects.tsx` (Phase 4, 5)
- `dashboard-ui/src/components/panels/GraphVisualization.tsx` (Phase 6)
- `dashboard-ui/src/App.url-sovereignty.test.tsx` (new, Phase 2)
- `dashboard-ui/src/components/transport/TransportBar.default-step.test.tsx` (new, Phase 2)
- `dashboard-ui/src/components/panels/AllObjects.test.tsx` (extend, Phase 4)
- `dashboard-ui/src/hooks/useBookmarks.test.ts` (extend, Phase 5)
- `dashboard-ui/e2e/dashboard-default-loads.spec.ts` (new, Phase 7)
- `dashboard-ui/e2e/url-sovereignty.spec.ts` (new, Phase 7)

## What This Plan Does NOT Cover

- The 14 untested areas the validation agent flagged (bookmark CRUD via
  REST, copy link, schema popout interaction, pipeline live indicators,
  responsive layouts, browser back/forward, most keyboard shortcuts beyond
  B/Escape, etc.). These should be covered by the second validation pass
  in TF.7 after the bug fixes land.
- A full schema migration to persist a canonical `objects` table (Phase 4
  proper fix). The interim fix is sufficient to satisfy the invariant; the
  schema migration is a separate follow-up.
- Phase 6 of the unified-run plan (delete `_indices`, `_GameIndex`,
  `_get_live_history`, etc. from `data_store.py`). The current bug-fix plan
  treats those as legacy; they should be removed in a separate cleanup PR
  after this plan completes.

## How to Resume After Context Compact

Read this file. Start with Phase 1 task T1.1. Each task has explicit file
paths and test names. The task list above is canonical; re-read it each time
you complete a task to find the next one. Mark completed tasks with [x] in a
separate scratch buffer or via the TaskCreate / TaskUpdate tools.

The bugs are documented at the top of this file; refer back to the Bug List
table for severity and the prose description above each phase for context.
