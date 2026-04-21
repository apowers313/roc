# ROC Dashboard UAT Test Plan

## Environment

- **Dashboard UI**: https://dev.ato.ms:9044 (Vite dev server, proxies /api and /socket.io to backend)
- **API Server**: https://localhost:9043 (FastAPI + Socket.io, HTTPS self-signed cert)
- **Screenshots**: Save to `./tmp/uat/` with test case ID prefix (e.g., `tmp/uat/TC-NAV-001-step-forward.png`)
- **Tools**: Use Playwright MCP (browser_navigate, browser_click, browser_snapshot, browser_take_screenshot, browser_press_key, browser_evaluate, browser_console_messages, browser_wait_for)
- **Visual verification**: Use Nanobanana MCP for objective yes/no questions about screenshots. Do NOT ask leading questions.

### Entry Criteria

Before starting, verify ALL of the following. If any fail, STOP and report:
1. `curl -sk https://localhost:9043/api/game/status` returns JSON containing `"state":"idle"` (server is up)
2. `curl -sk https://localhost:9043/api/runs` returns a non-empty JSON array (run data exists)
3. Navigate to https://dev.ato.ms:9044 -- page loads without error
4. `npx servherd list` shows both `roc-server` and `roc-ui` as online

### Pre-flight: Live Game Smoke Test

After entry criteria pass, attempt a quick game start/stop to verify the game backend:
1. `curl -sk -X POST https://localhost:9043/api/game/start -H 'Content-Type: application/json' -d '{"num_games": 1}'`
2. Wait 15 seconds
3. `curl -sk https://localhost:9043/api/game/status` -- check state and whether steps were produced
4. `curl -sk -X POST https://localhost:9043/api/game/stop`
5. **Verify game progression** (not just step production):
   - Fetch steps 1 through 10 from the pre-flight run via the API
   - Extract the turn counter (T:N from the screen status line) and position (x, y from game_metrics)
   - **Expected**: Turn counter reaches at least T:2 within 10 steps. Position (x, y) changes at least once.
   - If the turn counter is stuck at T:1 or T:2 for all 10 steps AND position never changes, the game engine is not processing actions. This is a critical backend defect that blocks all live-game tests.

If the game fails to produce steps OR fails the progression check, immediately report:
- The error from game status
- The turn counter and position values for steps 1-10
- How many tests are blocked (all tests tagged `Requires: live-game`)
- Ask the user whether to proceed with partial testing or abort entirely

Do NOT silently skip blocked tests and claim the run passed.

### How to Interact with the Dashboard

- **Mantine Select dropdowns** (Run, Game, Speed): Click the input field to open, then click the option or type to filter
- **Accordion sections**: Click the section title text or chevron to expand/collapse
- **Popout buttons**: Small icon buttons in section toolbars or the top popout bar
- **Step slider**: Click and drag, or use keyboard arrows when focused
- **Keyboard shortcuts**: Use browser_press_key with the key name (e.g., "ArrowRight", "Space", "b", "Shift+/")
- **Game Menu**: Click the gamepad icon (top-left area of menu bar)

---

## Test Cases

## Category 1: Page Load and Initial State (TC-INIT)

### TC-INIT-001: Dashboard loads without errors
**Priority**: Critical
**Preconditions**: Servers running, at least one run with data exists
**Steps**:
1. Navigate to https://dev.ato.ms:9044
   **Expected**: Page loads with title "ROC"
   **Verify**: Take screenshot. Check browser_console_messages for zero errors.
2. Verify the run selector dropdown is populated
   **Expected**: Run selector shows a run name in `[MM/DD HH:MM] name (Ng, N steps)` format
   **Verify**: Evaluate JS: `document.querySelector('input[placeholder="Run"]').value` is non-empty
3. Verify connection indicator
   **Expected**: Green dot visible in the transport bar area
   **Verify**: Evaluate JS: `document.querySelector('[title="Connected"]') !== null`

### TC-INIT-002: Auto-select first run populates all panels
**Priority**: Critical
**Preconditions**: TC-INIT-001 passed
**Steps**:
1. Verify GameScreen renders NetHack terminal
   **Expected**: Game State section shows colored text characters (NetHack dungeon display)
   **Verify**: Take screenshot of Game State section. Check for non-empty content.
2. Verify StatusBar shows metrics
   **Expected**: HP bar and at least Score, Depth, Gold, Energy, Hunger labels visible
   **Verify**: Snapshot shows metric labels with numeric values
3. Verify step counter shows valid position
   **Expected**: Format "N / M" where both N and M are positive integers
   **Verify**: Evaluate JS to read step counter text

### TC-INIT-003: URL parameters restore state on reload
**Priority**: High
**Regression for**: B1-003 (URL override by live game), B5-002 (URL sovereignty)
**Preconditions**: Dashboard loaded with a run selected
**Steps**:
1. Note current URL parameters (?run=X&game=Y&step=Z)
   **Verify**: Evaluate JS: `window.location.search`
2. Navigate to step 50 using the slider or arrow keys
3. Reload the page (navigate to the same URL)
   **Expected**: Same run, game, and step are restored
   **Verify**: Evaluate JS after reload: `window.location.search` contains the same run and step=50

## Category 2: Step Navigation (TC-NAV)

### TC-NAV-001: Step forward and backward buttons
**Priority**: Critical
**Preconditions**: A run with 50+ steps is selected
**Steps**:
1. Click "First step" button to go to step 1
2. Click "Next step" button 3 times
   **Expected**: Step counter shows "4 / N"
   **Verify**: Evaluate JS for step counter value
3. Click "Previous step" button once
   **Expected**: Step counter shows "3 / N"
4. Take screenshot showing step 3 game state

### TC-NAV-002: Jump to start and end
**Priority**: High
**Steps**:
1. Click "Last step" button
   **Expected**: Step counter shows "N / N" (at maximum)
   **Verify**: Evaluate JS confirming step equals max
2. Click "First step" button
   **Expected**: Step counter shows "1 / N"
3. Take screenshot at first step

### TC-NAV-003: Slider navigation
**Priority**: High
**Steps**:
1. Click on the step slider at approximately the middle position
   **Expected**: Step jumps to approximately N/2
   **Verify**: Step counter updates to reflect new position
2. Take screenshot showing mid-run game state

### TC-NAV-004: Step clamping on out-of-range URL
**Priority**: High
**Regression for**: B4-001 (out-of-range shows blank), B4-012 (Game 0 display)
**Steps**:
1. Navigate to URL with step=99999 for the current run
   **Expected**: Step is clamped to the run's maximum step
   **Verify**: Step counter shows "N / N" (not "99999")
2. Verify StatusBar does NOT show "Game 0"
   **Expected**: Shows valid metrics or "No data at step N" -- never "Game 0"
   **Verify**: Take screenshot of status bar

### TC-NAV-005: Keyboard navigation
**Priority**: Medium
**Regression for**: B5-025 (broken keyboard shortcuts)
**Steps**:
1. Press ArrowRight
   **Expected**: Step advances by 1
2. Press ArrowLeft
   **Expected**: Step goes back by 1
3. Press Home
   **Expected**: Jumps to first step
4. Press End
   **Expected**: Jumps to last step
5. Press Shift+ArrowRight
   **Expected**: Advances by 10 steps
6. Press "-" (minus key)
   **Expected**: Speed selector changes to a slower speed
   **Verify**: Evaluate JS for speed hidden input value
7. Press Shift+= (plus key)
   **Expected**: Speed selector changes back to a faster speed
8. Press Shift+/ (question mark)
   **Expected**: Keyboard Shortcuts help dialog opens
   **Verify**: Take screenshot of help dialog

### TC-NAV-006: Boundary clamping -- cannot step past limits
**Priority**: Medium
**Steps**:
1. Jump to first step
2. Press ArrowLeft 3 times
   **Expected**: Step stays at 1 (does not go below minimum)
3. Jump to last step
4. Press ArrowRight 3 times
   **Expected**: Step stays at maximum (does not go above)

## Category 3: Run and Game Selection (TC-SEL)

### TC-SEL-001: Switch between runs
**Priority**: Critical
**Steps**:
1. Note current run name and step data
2. Click the Run selector dropdown
3. Select a different run
   **Expected**: All panels update to the new run's data. Step resets to 1.
   **Verify**: Take screenshot. Verify game screen shows different content.
4. Check console for errors
   **Expected**: Zero errors during run switch

### TC-SEL-002: Multi-game run -- game switching
**Priority**: High
**Regression for**: B5-003 (wrong step range for game), B5-004 (global range ignoring game)
**Preconditions**: Select a run with 2+ games (e.g., jittery-eliot-oringa)
**Steps**:
1. Verify Game dropdown shows multiple games
   **Expected**: At least "Game 1 (N steps)" and "Game 2 (M steps)"
2. Select Game 1 -- note step range
3. Select Game 2
   **Expected**: Step range changes to Game 2's range (different from Game 1)
   **Verify**: Evaluate JS for step max. Take screenshot.
4. Verify GameScreen shows different data for Game 2
5. Switch back to Game 1
   **Expected**: Step range restores to Game 1's range

### TC-SEL-003: Show All runs toggle
**Priority**: Medium
**Regression for**: B4-008 (runs silently disappearing)
**Steps**:
1. Note number of runs visible in dropdown (count options)
2. Click "Show all" checkbox
   **Expected**: More runs appear in the dropdown (including short/empty/corrupt)
   **Verify**: Evaluate JS for run count before and after toggle
3. Uncheck "Show all"
   **Expected**: Run list returns to filtered state

### TC-SEL-004: Run selector label format
**Priority**: Low
**Regression for**: B1-019 (hard to scan format)
**Steps**:
1. Open the run selector dropdown
   **Expected**: Each entry follows format `[MM/DD HH:MM] name (Ng, N steps)`
   **Verify**: Take screenshot of dropdown options

### TC-SEL-005: Cross-run graph 404 prevention
**Priority**: Medium
**Regression for**: B5-021 (stale debounced step causes 404)
**Steps**:
1. Select a run with many steps (400+)
2. Navigate to a high step (e.g., 400)
3. Open the Graph Visualization section and wait for it to load
4. Switch to a run with few steps (e.g., 13)
   **Expected**: Zero 404 errors in console. Graph loads for step 1 of new run.
   **Verify**: Check browser_console_messages for errors

## Category 4: Playback (TC-PLAY)

### TC-PLAY-001: Play and pause
**Priority**: High
**Steps**:
1. Jump to step 1
2. Set speed to 5x (200ms) via speed dropdown
3. Click Play button
   **Expected**: Steps advance automatically. Step counter increments.
   **Verify**: Wait 3 seconds, then check step counter is > 10
4. Click Pause button
   **Expected**: Steps stop advancing. Step counter holds steady.
   **Verify**: Record step, wait 2 seconds, verify step unchanged

### TC-PLAY-002: Speed control
**Priority**: Medium
**Steps**:
1. Set speed to 1x (1000ms)
2. Play for 3 seconds
   **Expected**: Approximately 3 steps advance
3. Pause, set speed to 10x (100ms)
4. Play for 3 seconds
   **Expected**: Approximately 30 steps advance
5. Pause

### TC-PLAY-003: Playback data-ready gating
**Priority**: High
**Regression for**: B1-002 (slow step navigation), request pileup
**Steps**:
1. Set speed to 20x (50ms)
2. Play for 5 seconds
   **Expected**: Steps advance smoothly without request pileup errors
   **Verify**: Check console for zero errors after playback

## Category 5: Game Lifecycle (TC-GAME)

### TC-GAME-001: Game menu initial state
**Priority**: Medium
**Steps**:
1. Click the gamepad icon to open Game Menu
   **Expected**: Shows "No game running" text
   **Verify**: Take screenshot of game menu

### TC-GAME-002: Start game
**Priority**: Critical
**Requires**: live-game
**Regression for**: B5-015 (silent crash), B3-014 (frames not flowing), B2-009 (live broken)
**Steps**:
1. Open Game Menu, set number of games to 1
2. Click "Start Game"
   **Expected**: Button shows loading state
3. Wait 15 seconds for game initialization
4. Verify game is running
   **Expected**: LIVE badge appears in status bar. Step counter advances.
   **Verify**: Evaluate JS: `document.querySelector('[title="Connected"]') !== null`. Take screenshot.
5. Verify GameScreen shows live game data (NetHack dungeon with @ character)
6. Check console for errors
   **Expected**: Zero errors (transient 404s during first 2-3 steps are acceptable)

### TC-GAME-003: Run selector label during live game
**Priority**: Medium
**Requires**: live-game
**Regression for**: B5-020 (shows "[empty]" for live run), B1-006 (live run not in dropdown)
**Preconditions**: Game is running (from TC-GAME-002)
**Steps**:
1. Open the Run selector dropdown
   **Expected**: The active live run appears with step count > 0, NOT "[empty]"
   **Verify**: Evaluate JS for the run option label text

### TC-GAME-004: Live mode auto-follow
**Priority**: High
**Requires**: live-game
**Regression for**: B4-011 (stale closure missed updates), B4-013 (GO LIVE not updating)
**Preconditions**: Game is running with LIVE badge visible
**Steps**:
1. Verify step counter is advancing (auto-follow active)
2. Click "Previous step" button to break auto-follow
   **Expected**: "GO LIVE" badge appears (yellow). Steps stop advancing.
   **Verify**: Take screenshot showing GO LIVE badge
3. Click the "GO LIVE" badge
   **Expected**: Dashboard snaps to latest step. LIVE badge reappears. Steps resume advancing.
   **Verify**: Take screenshot showing LIVE badge

### TC-GAME-005: Stop game
**Priority**: Critical
**Requires**: live-game
**Regression for**: B5-019 (GO LIVE persists after stop), B5-018 (stale 0/0 range)
**Preconditions**: Game is running
**Steps**:
1. Open Game Menu, click "Stop Game"
2. Wait 10 seconds
3. Verify LIVE/GO LIVE badges disappear
   **Expected**: No LIVE or GO LIVE badges visible
   **Verify**: Take screenshot of status bar area
4. Verify run data is still navigable
   **Expected**: Can step forward and backward through the completed run
5. Verify step-range API returns tail_growing=false
   **Verify**: Evaluate JS: `fetch('/api/runs/'+encodeURIComponent(currentRun)+'/step-range').then(r=>r.json())` and check tail_growing

### TC-GAME-006: Sequential game starts -- game progression
**Priority**: Critical
**Requires**: live-game
**Regression for**: B3-019 (component already exists), B3-020 (ExpMod duplicate), B3-013 (stale cache), B6-001 (dead EventBus Subject after first game kills sequential games)
**Preconditions**: Previous game stopped (from TC-GAME-005)
**Steps**:
1. Start a new game (1 game)
2. Wait 15 seconds
   **Expected**: New run appears. Dashboard auto-navigates. LIVE badge shows.
3. Stop the game
4. Verify the second game actually progressed -- this is the critical regression check:
   - Fetch step data for steps 1 through 15 of the new run via the API
   - Check the turn counter (T:N in the screen status line) across those steps
   **Expected**: Turn counter advances past T:3 within 15 steps. Position (x, y) changes at least once.
   **Verify**: `curl -sk https://localhost:9043/api/runs/{RUN}/step/{N}?game=1` for steps 1-15. Extract game_metrics.x, game_metrics.y, and the T:N value from the screen status line. At least 3 distinct turn values must appear and at least 2 distinct (x, y) positions.
   **If this fails**: The EventBus Subjects were killed by Component.shutdown() after the first game. The action bus cache contains stale TakeActions from game 1, and the Action component's handler never fires for game 2. Every step replays the same stale action.
5. Switch back to the first run (from TC-GAME-002/005)
   **Expected**: All data intact and navigable

### TC-GAME-007: Game crash error display
**Priority**: Medium
**Regression for**: B5-015 (silent crash shows "idle, no details")
**Steps**:
1. Open Game Menu after a failed game (if one exists in game state)
   **Expected**: If error field is non-null, error message is displayed in red text
   **Verify**: Take screenshot of game menu showing error state

## Category 6: Status Bar (TC-STATUS)

### TC-STATUS-001: Metrics display
**Priority**: High
**Steps**:
1. Navigate to a step with game data
   **Expected**: HP bar shows colored progress (green/yellow/red based on ratio). Score, Depth, Gold, Energy, Hunger show numeric values.
   **Verify**: Take screenshot of status bar

### TC-STATUS-002: Error badge on fetch failure
**Priority**: High
**Regression for**: B4-009 (500 errors invisible), B5-027 (LIVE + ERROR simultaneously)
**Steps**:
1. Navigate to a URL with a non-existent run: `?run=does-not-exist-run`
   **Expected**: ERROR badge appears in status bar with tooltip showing error details
   **Verify**: Take screenshot showing ERROR badge

### TC-STATUS-003: Missing data display
**Priority**: Medium
**Regression for**: B4-001 (Game 0 display)
**Steps**:
1. After TC-STATUS-002, check status bar text
   **Expected**: Shows meaningful text, NOT "Step N | Game 0"
   **Verify**: Snapshot text content

### TC-STATUS-004: Browse runs escape from broken run
**Priority**: High
**Regression for**: B5-016 (invisible button), B5-017 (Browse runs trap)
**Steps**:
1. Navigate to `?run=does-not-exist-run`
   **Expected**: Red error banner appears with visible "Browse runs" button
   **Verify**: Take screenshot showing the banner and button (button text must be readable)
2. Click "Browse runs" button
   **Expected**: Dashboard navigates to a working run. Data renders.
   **Verify**: Panels show real data. URL no longer contains the broken run name.

## Category 7: All Accordion Sections (TC-SECT)

### TC-SECT-001: Pipeline Status
**Priority**: Medium
**Steps**:
1. Expand "Pipeline Status" section
   **Expected**: Shows pipeline stage data (timing, counts)
   **Verify**: Take screenshot. Content area is not empty.

### TC-SECT-002: Game State
**Priority**: Critical
**Regression for**: B1-013 (screen/saliency different steps)
**Steps**:
1. Expand "Game State" section (should be open by default)
   **Expected**: GameScreen shows colored NetHack terminal. GameMetrics table shows rows with values.
   **Verify**: Take screenshot showing both GameScreen and GameMetrics side by side

### TC-SECT-003: Log Messages
**Priority**: Medium
**Steps**:
1. Expand "Log Messages" section
   **Expected**: Shows log entries for the current step (may be empty for some steps)
   **Verify**: Take screenshot

### TC-SECT-004: Intrinsics and Significance
**Priority**: High
**Steps**:
1. Expand "Intrinsics & Significance" section
   **Expected**: Left panel shows key-value pairs (hp, hunger, etc.). Right panel shows a line chart with data points.
   **Verify**: Take screenshot showing both panels

### TC-SECT-005: Inventory
**Priority**: Medium
**Steps**:
1. Expand "Inventory" section
   **Expected**: Shows inventory items or "No inventory data" placeholder
   **Verify**: Take screenshot

### TC-SECT-006: Visual Perception
**Priority**: High
**Regression for**: B2-003 (glyph shows "--"), B2-011 (numeric IDs not characters)
**Steps**:
1. Expand "Visual Perception" section
   **Expected**: FeatureTable shows feature entries. ObjectInfo shows object details.
   **Verify**: Take screenshot. Verify glyph column shows actual characters (e.g., @, d, .) not numeric IDs or "--"

### TC-SECT-007: Aural Perception
**Priority**: Medium
**Regression for**: B1-012 (crashes on old runs)
**Steps**:
1. Expand "Aural Perception" section
   **Expected**: Shows phoneme data or "No auditory data" -- must NOT crash
   **Verify**: Take screenshot. Check console for zero errors.

### TC-SECT-008: Visual Attention
**Priority**: High
**Regression for**: B2-001 (saliency 1 step behind), B2-014/B2-015 (identical saliency maps)
**Steps**:
1. Expand "Visual Attention" section
   **Expected**: AttentionSpread, SaliencyMap (colored grid), FocusPoints, AttenuationPanel all render
   **Verify**: Take screenshot showing saliency map with colored regions
2. If cycle stepper is present (>1 cycle), click through cycles
   **Expected**: Each cycle shows a visually different saliency map
   **Verify**: Take screenshots of cycle 1 and last cycle -- they must differ

### TC-SECT-008a: Attention cycle count matches configuration
**Priority**: Critical
**Regression for**: B7-001 (TOCTOU race in _handle_settled triples attention cycles)
**Steps**:
1. Query the API for saliency_cycles at several steps:
   `curl -sk https://localhost:9043/api/runs/{RUN}/step/{STEP}` for steps 1, 5, 10, 20
2. Count the number of entries in the `saliency_cycles` array for each step
   **Expected**: Each step has exactly 4 saliency cycles (matching `attention_cycles` config default of 4). Must NOT be 8, 12, or any other multiple of 4.
   **Verify**: `len(step_data.saliency_cycles) == 4` for all checked steps
3. Verify all cycles have complete metadata (focused_point, pre_ior_peak, post_ior_peak)
   **Expected**: Every cycle entry has focused_point, pre_ior_peak, and post_ior_peak keys -- not just the first 4
   **Verify**: Check that no cycle entry is missing these keys
   **If this fails**: The _settled_lock in VisionAttention._handle_settled is missing or broken. Multiple ThreadPoolScheduler threads are passing the "all extractors settled" check concurrently, each triggering a full round of attention cycles. See roc/pipeline/attention/attention.py.

### TC-SECT-009: Object Resolution
**Priority**: High
**Regression for**: B1-001 (off-by-one), B2-019 (glyph shows "-")
**Steps**:
1. Expand "Object Resolution" section
   **Expected**: ResolutionInspector shows resolution details. EventSummary shows events.
   **Verify**: Take screenshot. Verify matched objects show glyph characters, not "-"

### TC-SECT-010: Graph Visualization
**Priority**: High
**Regression for**: B3-005 (blank on Safari), B3-011 (click oscillation), B3-010 (live flicker)
**Steps**:
1. Expand "Graph Visualization" section
2. Wait for graph to load
   **Expected**: Nodes and edges are visible. Frame timeline nodes appear.
   **Verify**: Take screenshot showing graph with labeled nodes
3. Click a node to expand it
   **Expected**: Node expands to show children. No oscillation/flicker.
   **Verify**: Take screenshot of expanded node
4. Click "Fit" button
   **Expected**: Graph zooms to fit all visible nodes

### TC-SECT-011: Sequences
**Priority**: Medium
**Steps**:
1. Expand "Sequences" section
   **Expected**: Shows frame sequence data
   **Verify**: Take screenshot

### TC-SECT-012: Transitions
**Priority**: Medium
**Regression for**: B2-018 (raw Frame strings), B3-004 (wrong transform data)
**Steps**:
1. Expand "Transitions" section
   **Expected**: Shows structured transform data with glyph characters and deltas -- NOT raw node strings
   **Verify**: Take screenshot. Verify content shows property names/values, not "Node(-755, labels={'Frame'})"
2. If step links exist, click one
   **Expected**: Dashboard navigates to the clicked step

### TC-SECT-013: Prediction
**Priority**: Low
**Steps**:
1. Expand "Prediction" section
   **Expected**: Shows prediction data or "No prediction data" placeholder
   **Verify**: Take screenshot

### TC-SECT-014: Actions
**Priority**: Medium
**Steps**:
1. Expand "Actions" section
   **Expected**: Shows action taken with human-readable name
   **Verify**: Take screenshot

### TC-SECT-015: Section height stability during navigation
**Priority**: Medium
**Regression for**: B2-004 (accordion bouncing)
**Steps**:
1. Open Pipeline Status, Game State, and Visual Perception sections
2. Rapidly step forward 10 times
   **Expected**: No visible section height changes or bouncing
   **Verify**: Take screenshots at step 1 and step 10 -- section layout should be stable

## Category 8: Popout Panels (TC-POP)

### TC-POP-001: All Objects popout
**Priority**: High
**Regression for**: B5-005 (silent game filter), B1-017 (glyph shows "--")
**Steps**:
1. Click "Open All Objects panel" button in the top popout toolbar
   **Expected**: Right-side drawer opens with sortable object table
   **Verify**: Take screenshot of drawer content
2. Verify table has data rows with glyph characters (not "--" for objects with matches)
3. Close the drawer by clicking outside or the X button
   **Expected**: Drawer closes. Page scroll position preserved.
   **Verify**: Check page is still scrolled to same position (not reset to top)

### TC-POP-002: Graph and Events popout
**Priority**: High
**Steps**:
1. Click "Open Graph & Events panel" button
   **Expected**: Drawer opens with GraphHistory and EventHistory charts
   **Verify**: Take screenshot showing at least one chart with data points
2. Click a data point on the chart
   **Expected**: Dashboard navigates to the clicked step
   **Verify**: Step counter updates to match clicked step
3. Close the drawer

### TC-POP-003: Schema popout
**Priority**: Medium
**Regression for**: B2-005 (iOS viewport zoom)
**Steps**:
1. In the Game State section toolbar, click the Schema popout button
   **Expected**: Drawer opens showing schema data (Mermaid diagram or JSON)
   **Verify**: Take screenshot
2. Close the drawer

### TC-POP-004: Resolution Error Rate popout
**Priority**: Medium
**Steps**:
1. In the Object Resolution section toolbar, click "Resolution Error Rate" popout
   **Expected**: Drawer opens with resolution error chart
   **Verify**: Take screenshot
2. Close the drawer

### TC-POP-005: Action Histogram popout
**Priority**: Medium
**Regression for**: B2-007 (x-axis growing), B5-012 (empty)
**Steps**:
1. In the Actions section toolbar, click "Action Histogram" popout
   **Expected**: Drawer opens with bar chart. Bars have human-readable action names.
   **Verify**: Take screenshot. Verify axis labels are action names (e.g., "kick"), not "Action #N"
2. Close the drawer

### TC-POP-006: Scroll position preserved after drawer close
**Priority**: Medium
**Regression for**: B1-015 (scroll position resets)
**Steps**:
1. Scroll down to the Object Resolution section
2. Open "All Objects" popout from the section toolbar
3. Close the drawer
   **Expected**: Page remains scrolled to Object Resolution section
   **Verify**: Object Resolution section is still visible without scrolling

## Category 9: Bookmarks (TC-BM)

### TC-BM-001: Add and remove bookmark
**Priority**: Medium
**Regression for**: B5-010 (silent no-op in error state)
**Steps**:
1. Navigate to step 20
2. Press "b" to toggle bookmark
   **Expected**: Bookmark indicator appears. Bookmarks section auto-opens.
   **Verify**: Take screenshot showing bookmark bar and Bookmarks section
3. Press "b" again to remove bookmark
   **Expected**: Bookmark removed. Bookmarks section auto-closes (if no other bookmarks).

### TC-BM-002: Navigate between bookmarks
**Priority**: Medium
**Steps**:
1. Add bookmarks at step 10 and step 30
2. Navigate to step 1
3. Press "]" (right bracket)
   **Expected**: Navigates to step 10
4. Press "]" again
   **Expected**: Navigates to step 30
5. Press "[" (left bracket)
   **Expected**: Navigates back to step 10
6. Clean up: remove both bookmarks

### TC-BM-003: Bookmark guard when no data loaded
**Priority**: Medium
**Regression for**: B5-010 (phantom bookmarks)
**Steps**:
1. Navigate to `?run=does-not-exist-run`
2. Press "b" to try adding a bookmark
   **Expected**: Yellow alert "Cannot bookmark: no step loaded" appears. No bookmark created.
   **Verify**: Take screenshot showing the alert notification

## Category 10: Graph Visualization Deep Test (TC-GRAPH)

### TC-GRAPH-001: Node expand and detail panel
**Priority**: High
**Regression for**: B3-015 (empty detail rows), B4-003 (transform no info), B4-006 (missing parent)
**Steps**:
1. Open Graph Visualization section
2. Wait for graph to load
3. Click a Frame node
   **Expected**: Detail panel appears with frame info (tick, action name)
   **Verify**: Take screenshot of detail panel
4. If Object nodes are visible, click one
   **Expected**: Detail panel shows object UUID, human name, features
   **Verify**: UUID does not end in "000" (B4-002 regression)

### TC-GRAPH-002: Node type filter
**Priority**: Medium
**Regression for**: B3-006 (intrinsic nodes cluttering)
**Steps**:
1. Click "Show Nodes" button to open filter
   **Expected**: Dropdown shows node type checkboxes
2. Uncheck "Object Instance"
   **Expected**: ObjectInstance nodes disappear from graph
3. Re-check "Object Instance"
   **Expected**: ObjectInstance nodes reappear
   **Verify**: Take screenshot after re-enabling

### TC-GRAPH-003: Fit and Reset controls
**Priority**: Low
**Steps**:
1. Zoom into the graph (scroll wheel or pinch)
2. Click "Fit" button
   **Expected**: Graph zooms to fit all nodes in viewport
3. Click "Reset" (Unpin All) button
   **Expected**: Graph resets to initial layout state

### TC-GRAPH-004: Graph data for stopped game
**Priority**: High
**Requires**: live-game
**Regression for**: B3-012 (graph.json not written on REST stop)
**Preconditions**: A game was started and stopped via REST API (TC-GAME-005)
**Steps**:
1. Navigate to the run from the stopped game
2. Open Graph Visualization section
   **Expected**: Graph data loads and nodes render -- NOT "No graph data"
   **Verify**: Take screenshot showing graph nodes

## Category 11: Data Accuracy (TC-DATA)

### TC-DATA-001: Step data changes between steps
**Priority**: Critical
**Steps**:
1. Navigate to step 1, take screenshot
2. Navigate to step 5, take screenshot
3. Navigate to step 20, take screenshot
   **Expected**: All three screenshots show different game states (different screen content, different metrics)
   **Verify**: Use Nanobanana to compare: "Are these three screenshots showing different game states? Answer YES or NO."

### TC-DATA-002: Metrics consistency across panels
**Priority**: High
**Steps**:
1. Navigate to a step with data
2. Note HP value from StatusBar
3. Expand Intrinsics section
   **Expected**: HP value in Intrinsics panel matches StatusBar HP
   **Verify**: Compare values visually or via JS evaluation

### TC-DATA-003: Chart click-to-navigate accuracy
**Priority**: Medium
**Regression for**: all click-to-navigate charts
**Steps**:
1. Open Intrinsics & Significance section
2. Click a data point on the IntrinsicsChart
   **Expected**: Step counter updates to the clicked step. Data panels reflect that step.
   **Verify**: Step counter matches the clicked chart position

## Category 12: Error Handling (TC-ERR)

### TC-ERR-001: Non-existent run handling
**Priority**: High
**Regression for**: B5-009 (13 panels empty), B5-016/017 (Browse runs)
**Steps**:
1. Navigate to `?run=this-run-does-not-exist`
   **Expected**: Error banner appears: "Run ... could not be loaded" with visible "Browse runs" button
   **Verify**: Take screenshot. Button text is readable (not white-on-white).
2. Click "Browse runs"
   **Expected**: Dashboard selects a working run. All panels render data.

### TC-ERR-002: Empty catalog run handling
**Priority**: Medium
**Regression for**: B5-023 (unresponsive Game dropdown)
**Steps**:
1. If a run with empty catalog exists (check "Show all" toggle for [empty] runs):
   Select it
   **Expected**: Banner shows "This run has no recorded steps (empty catalog)" with "Browse runs" button
   **Verify**: Take screenshot

### TC-ERR-003: Fetch error visibility
**Priority**: High
**Regression for**: B4-009 (HTTP 500 invisible)
**Steps**:
1. If any step returns an error, verify ERROR badge appears in StatusBar
   **Expected**: Red ERROR badge with tooltip showing error message
   **Verify**: Take screenshot of ERROR badge with tooltip

## Category 13: Highlight and Click-to-Navigate (TC-CLICK)

### TC-CLICK-001: Object highlights on game screen
**Priority**: Medium
**Regression for**: B1-014 (highlight offset), B2-002 (row offset), B2-016/017 (not clickable)
**Steps**:
1. Expand Visual Attention section
2. If FocusPoints shows coordinates, click on a coordinate
   **Expected**: Highlight appears on GameScreen at the correct position
   **Verify**: Take screenshot showing highlighted point on screen

### TC-CLICK-002: Transition step links
**Priority**: Medium
**Steps**:
1. Expand Transitions section
2. If clickable step/tick links exist, click one
   **Expected**: Dashboard navigates to the referenced step
   **Verify**: Step counter matches clicked link value

## Category 14: Live Mode Specific (TC-LIVE)

### TC-LIVE-001: URL sovereignty during live game
**Priority**: High
**Requires**: live-game
**Regression for**: B1-003 (URL hijack), B5-002 (sovereignty violated)
**Preconditions**: No game running
**Steps**:
1. Navigate to explicit URL with a historical run: `?run=HISTORICAL_RUN_NAME&step=50`
2. Start a new game via Game Menu
3. Wait 10 seconds
   **Expected**: Dashboard stays on the historical run (URL sovereignty preserved)
   **Verify**: URL still contains the historical run name. Dashboard still shows historical data.
4. Stop the game

### TC-LIVE-002: Auto-navigation to new game (no URL)
**Priority**: High
**Requires**: live-game
**Steps**:
1. Navigate to dashboard with NO run parameter (just https://dev.ato.ms:9044)
2. Start a new game
3. Wait 10 seconds
   **Expected**: Dashboard auto-navigates to the new live run. LIVE badge appears.
   **Verify**: URL contains the new run name. Step counter is advancing.
4. Stop the game

## Category 15: Performance (TC-PERF)

### TC-PERF-001: Step navigation speed from cache
**Priority**: High
**Regression for**: B1-002 (1 second per step), B8-001 (prefetch centered on range midpoint)
**Preconditions**: A run with 1000+ steps exists (e.g., snuffly-sandy-natale with 4767 steps in game 1)
**Steps**:
1. Enable CDP network throttling to simulate real device conditions:
   ```js
   const client = await page.context().newCDPSession(page);
   await client.send('Network.emulateNetworkConditions', {
     offline: false,
     downloadThroughput: 1.5 * 1024 * 1024 / 8,
     uploadThroughput: 750 * 1024 / 8,
     latency: 40,
   });
   ```
2. Navigate to step 50 of the large run and wait 5 seconds (allow prefetch to fill the +-100 window)
3. Measure time to navigate 10 steps forward:
   **Verify**: Evaluate JS: record `performance.now()` before each step advance, record after render settles.
   **Expected**: Each step completes under 500ms (cache hit even with 40ms network RTT). If any step takes over 1000ms, the prefetch window is likely not centered on the current step.
4. Disable throttling:
   ```js
   await client.send('Network.emulateNetworkConditions', {
     offline: false,
     downloadThroughput: -1,
     uploadThroughput: -1,
     latency: 0,
   });
   ```
5. Check console for zero errors during throttled navigation

### TC-PERF-002: No console errors during rapid navigation
**Priority**: Medium
**Steps**:
1. Rapidly click "Next step" 20 times in quick succession
2. Check console for errors
   **Expected**: Zero errors in console
   **Verify**: browser_console_messages shows no errors

## Category 16: Screen-Action Consistency (TC-ACTION)

### TC-ACTION-001: Screen changes after movement actions
**Priority**: Critical
**Steps**:
1. Navigate to a run with 50+ steps
2. Find a step where the action is a directional movement (N, S, E, W, NE, NW, SE, SW)
3. Compare the screen content at that step with the NEXT step
4. If the action was NOT blocked (i.e., the message line does not say "It's a wall" or similar), the @ position should have changed
**Expected**: At least some movement actions cause the @ character to change position on the screen
**Verify**: Compare x,y coordinates from metrics between consecutive steps. At least 30% of movement actions should result in a position change.

### TC-ACTION-002: Non-movement actions leave screen recognizable
**Priority**: High
**Steps**:
1. Find steps with non-movement actions (SEARCH, PRAY, WAIT, EAT, etc.)
2. Verify the screen still renders valid NetHack content (has @ character, has borders, etc.)
**Expected**: Non-movement actions may not change position but screen remains valid
**Verify**: Screen contains @ character and status line

### TC-ACTION-003: Action sequence produces game progression
**Priority**: High
**Steps**:
1. Check the first step and last step of a run
2. Compare metrics: at least one of (score, xp_level, experience, depth, gold) should have changed
**Expected**: Over the course of a full run, the game state progresses (not stuck)
**Verify**: At least one metric differs between first and last step
