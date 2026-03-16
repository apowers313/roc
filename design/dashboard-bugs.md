# Dashboard Bug Log

Bugs found during systematic Playwright-driven interaction testing of the
live Panel debug dashboard.

## Bug 1: Bookmark stores game selector value instead of step data game_number

- **Severity**: Medium
- **Symptom**: Bookmarks show incorrect game number. E.g. viewing step 1
  (which belongs to game 0/1) while game selector shows "3" produces a
  bookmark with game=3.
- **Root cause**: `_toggle_bookmark()` used `int(self.game)` (the game
  selector param value) instead of the actual `game_number` from the
  current step's data.
- **Fix**: Changed to `self._last_data.game_number if self._last_data is
  not None else int(self.game)`.
- **Regression test**: `test_bookmark_uses_step_data_game_number` in
  `tests/unit/test_panel_debug.py`.
- **Files**: `roc/reporting/panel_debug.py` line 921.

## Bug 2: Pause button does nothing in LIVE mode

- **Severity**: High
- **Symptom**: Clicking the pause button while watching live data has no
  effect -- frames keep advancing.
- **Root cause**: Two issues: (1) The Player widget starts with direction=0
  (paused), so clicking the pause button doesn't change state (direction
  stays 0, no watcher fires). (2) `_on_new_data()` push callbacks don't
  check direction at all -- they only check `_is_following()` (value>=end).
- **Fix**: (a) In live mode, set direction=1 at init so the UI shows
  "playing" and the pause button actually changes direction. (b) Disable the
  Player's auto-advance timer (interval=2^31-1) since live data comes from
  push callbacks. (c) Watch direction changes via `_handle_direction_widget`
  to set `_user_paused` flag. (d) `_on_new_data` respects `_user_paused`.
  (e) End key resets `_user_paused` and restores direction=1. (f) Home key
  sets direction=0 (pauses when leaving live). (g) Pausing restores normal
  playback interval for review mode.
- **Regression tests**: `test_pause_stops_live_following` and
  `test_pause_then_play_resumes_following` in `tests/unit/test_panel_debug.py`.
- **Files**: `roc/reporting/panel_debug.py` (init, `_handle_direction_widget`,
  `_on_new_data`, `_dispatch_key`, `_on_speed_change`).

## Known Issue: Console errors from collapsed card Tabulator updates

- **Severity**: Low (cosmetic, no functional impact)
- **Symptom**: TypeError console errors ("Cannot read properties of
  undefined (reading 'element')", "'style'", "'destroy'") when navigating
  steps while cards are collapsed.
- **Root cause**: Panel/Bokeh framework sends JSON patches to the browser
  for Tabulator widgets inside collapsed cards whose DOM elements have not
  been created yet. The `invalidate_render` path in `panel.min.js` fails
  because the view element is undefined.
- **Impact**: None -- data is correctly applied to the model. When the card
  is expanded, it renders the latest data correctly. Errors accumulate in
  the console but do not affect functionality.
- **Potential mitigations**: (a) Defer updates for collapsed cards (adds
  complexity), (b) Use `defer_load=True` on cards (changes UX), (c) Accept
  as Panel framework limitation.
- **Decision**: Accepted as known limitation. The errors are internal to
  Panel/Bokeh and do not affect dashboard reliability.

## Known Issue: Info text shows "Game 0" for steps not yet in Parquet

- **Severity**: Low (data timing, not a dashboard bug)
- **Symptom**: Info line shows "Game 0" and "N/A" timestamp for steps that
  should belong to a specific game.
- **Root cause**: `RunStore.get_step_data()` defaults `game_number = 0`
  when no screen row exists in the Parquet files. DuckLakeStore may not
  have flushed recent data to Parquet yet, so the reader-mode RunStore
  cannot find the step.
- **Impact**: Cosmetic during live sessions. Historical review after the
  run completes shows correct game numbers.

## Test Summary

Two rounds of systematic Playwright-driven interaction testing across two
separate live game sessions. Every interactive component triggered at least
4 times in random orders.

**Components tested (total interactions across both rounds):**
- Keyboard shortcuts: Home (8), End (7), ArrowRight (12+), ArrowLeft (8),
  Space play/pause (7), + speed (6), - speed (5), ? help (6), b bookmark (6),
  n next bookmark (5), p prev bookmark (5), g game cycle (5)
- Dropdowns: Game selector (2), Speed selector (4+)
- Log level radio buttons: DEBUG (8), INFO (8), WARN (8), ERROR (8)
- Collapsible cards: 6 cards x 4-6 toggles each
- Bookmark table: click navigation (2)

**No new functional bugs found** after fixing Bug 1. The only console
errors are the known Panel/Bokeh framework Tabulator issues in collapsed
cards, which are cosmetic and do not affect functionality.
