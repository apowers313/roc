# Implementation Plan for W&B Integration

## Overview

Integrate Weights & Biases experiment tracking into ROC to enable per-step game review (screens + saliency maps), run comparison, hyperparameter tracking, and sweep-based optimization. W&B operates as a peer to the existing OTel pipeline -- both receive the same metrics, but W&B adds rich media, permanent per-run storage, and built-in comparison tools.

## Phase Breakdown

### Phase 1: Core Integration (Config + WandbReporter lifecycle)

**What this phase accomplishes**: Runs appear in W&B with correct config, ExpMod parameters, game boundaries, and summary metrics. No per-step data yet.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_wandb_reporter.py`: WandbReporter lifecycle tests
  - `test_init_disabled`: When `wandb_enabled=False`, `WandbReporter.init()` does nothing and all subsequent calls are no-ops
  - `test_init_creates_run`: With `wandb_mode="disabled"` (W&B's built-in no-network mode), verify `init()` sets `_run` state
  - `test_start_game_increments_game_num`: Call `start_game(1)`, verify `_game_num == 1`
  - `test_end_game_logs_boundary`: Call `end_game()`, verify the boundary event dict contains `game_end`, `outcome`, `final_score`
  - `test_finish_sets_summary`: After 2 games with known scores, verify `run.summary` contains `total_games=2`, `mean_score`, `max_score`
  - `test_run_name_uses_instance_id`: Verify the run name matches the observability `instance_id`
  - `test_sweep_detection`: When `wandb.run.sweep_id` is set, reporter adjusts naming

- `tests/unit/test_expmod_params.py`: ExpMod.params_dict() tests
  - `test_params_dict_returns_public_attrs`: A test ExpMod with `threshold=0.5` and `name="test"` returns `{"threshold": 0.5}`
  - `test_params_dict_excludes_private`: Attributes starting with `_` are excluded
  - `test_params_dict_excludes_callables`: Methods are excluded
  - `test_concrete_expmod_params`: `LinearDeclineAttenuation.params_dict()` returns expected keys (`capacity`, `radius`, etc.)

- `tests/unit/test_config_wandb.py`: Config W&B field tests
  - `test_wandb_defaults`: All W&B config fields have expected defaults (`wandb_enabled=False`, `wandb_mode="online"`, etc.)
  - `test_wandb_env_vars`: Environment variables with `roc_` prefix override W&B fields

**Implementation**:

- `roc/config.py`: Add W&B config fields
  ```python
  # W&B integration
  wandb_enabled: bool = Field(default=False)
  wandb_project: str = Field(default="ROC")
  wandb_entity: str = Field(default="")
  wandb_host: str = Field(default="")
  wandb_api_key: str = Field(default="")
  wandb_tags: list[str] = Field(default=[])
  wandb_log_screens: bool = Field(default=True)
  wandb_log_saliency: bool = Field(default=True)
  wandb_log_interval: int = Field(default=1)
  wandb_artifacts: list[str] = Field(default=[])
  wandb_mode: str = Field(default="online")
  ```

- `roc/expmod.py`: Add `params_dict()` to `ExpMod` base class
  ```python
  def params_dict(self) -> dict[str, Any]:
      return {
          k: v for k, v in self.__dict__.items()
          if not k.startswith("_") and not callable(v)
      }
  ```

- `roc/reporting/wandb_reporter.py`: Create `WandbReporter` singleton
  - `init(config)` -- calls `wandb.login()` + `wandb.init()` with full config dump + ExpMod params, uses `instance_id` as run name
  - `start_game(game_num)` -- logs game_start boundary, resets game_tick
  - `end_game(outcome, summary)` -- logs game_end boundary with final stats
  - `log_step(data)` -- increments step counter, calls `wandb.log()` (stub for Phase 2)
  - `log_media(key, content)` -- logs `wandb.Html` content (stub for Phase 3)
  - `finish()` -- sets summary metrics, calls `run.finish()`
  - All methods are no-ops when `wandb_enabled=False`

- `roc/__init__.py`: Hook `WandbReporter.init()` after `Observability.init()`
- `roc/gymnasium.py`: Hook `start_game()` / `end_game()` at game boundaries, `finish()` at session end

**Dependencies**:
- External: `wandb` (add to pyproject.toml)
- Internal: `roc/config.py`, `roc/reporting/observability.py` (for `instance_id`, `roc_version`)

**Verification**:
1. Run: `roc_wandb_enabled=true roc_wandb_mode=disabled uv run play` -- game completes without errors, W&B disabled-mode logs appear in stderr
2. Run: `roc_wandb_enabled=true roc_wandb_host=http://hal.ato.ms:8080 roc_wandb_entity=apowers uv run play` -- run appears in W&B UI at hal.ato.ms:8080 with correct config, game boundaries, and summary (no per-step charts yet)
3. Run: `make test` -- all existing tests pass, new tests pass
4. Run: `make lint` -- no regressions

---

### Phase 2: Per-Step Numeric Metrics

**What this phase accomplishes**: Numeric metrics (HP, score, depth, position, saliency stats, object counts, action) appear in W&B charts per step, grouped by game number. OTel continues receiving the same data unchanged.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_wandb_reporter.py` (extend):
  - `test_log_step_increments_counter`: After 3 calls to `log_step()`, internal step counter is 3
  - `test_log_step_includes_game_num`: Every `log_step()` call includes `game_num` in the data dict
  - `test_log_step_includes_game_tick`: Every call includes `game_tick` that resets per game
  - `test_log_step_disabled_noop`: When `wandb_enabled=False`, `log_step()` does not raise

- `tests/unit/test_metrics.py`: RocMetrics abstraction tests
  - `test_record_histogram_calls_otel`: Mock OTel histogram, verify `record_histogram()` calls it
  - `test_record_histogram_calls_wandb`: Mock WandbReporter, verify `record_histogram()` calls `log_step()`
  - `test_increment_counter_calls_both`: Both OTel counter and W&B receive the increment
  - `test_wandb_disabled_otel_still_works`: When W&B is off, OTel calls still fire
  - `test_log_step_wandb_only`: `log_step()` only goes to W&B, not OTel

- `tests/integration/test_wandb_metrics.py`:
  - `test_full_tick_metrics_logged`: Run a single tick through the pipeline with mock environment, verify `WandbReporter.log_step()` received expected metric keys

**Implementation**:

- `roc/reporting/metrics.py`: Create `RocMetrics` unified abstraction
  ```python
  class RocMetrics:
      @staticmethod
      def record_histogram(name: str, value: float, attributes: dict | None = None) -> None: ...
      @staticmethod
      def increment_counter(name: str, amount: int = 1, attributes: dict | None = None) -> None: ...
      @staticmethod
      def log_step(data: dict) -> None: ...
      @staticmethod
      def log_media(key: str, html: str) -> None: ...
  ```

- `roc/reporting/wandb_reporter.py`: Flesh out `log_step()` -- accept data dict, prepend `game_num` and `game_tick`, call `wandb.log()` with global step counter

- `roc/gymnasium.py`: After each tick, collect core state metrics (score, hp, energy, hunger, depth, position) and call `WandbReporter.log_step()` or `RocMetrics.log_step()`

- `roc/attention.py`: After `get_focus()`, emit saliency metrics (peak_count, top_peak_strength, top_peak_shifted, entropy, omega) via `RocMetrics`

- `roc/saliency_attenuation.py`: Migrate existing OTel histogram/counter calls to use `RocMetrics` equivalents for dual-emission (keep existing OTel calls working during transition)

- `roc/object.py` (if applicable): Emit object resolution counts via `RocMetrics`

**Dependencies**:
- External: None new (wandb already added in Phase 1)
- Internal: Phase 1 (`WandbReporter` singleton, config fields)

**Verification**:
1. Run: `roc_wandb_enabled=true roc_wandb_host=http://hal.ato.ms:8080 roc_wandb_entity=apowers uv run play`
2. Open run in W&B UI -- verify line charts for `score`, `hp`, `depth`, `peak_count`, `top_peak_strength` have data points per step
3. Verify charts can be filtered by `game_num`
4. Verify OTel/Grafana dashboards still show the same metrics (no regression)
5. Run: `make test` -- all tests pass

---

### Phase 3: Rich Media (Screens + Saliency Maps)

**What this phase accomplishes**: Game screens and saliency heatmaps appear in W&B as scrubbable HTML panels. Per-game W&B Tables enable SQL-like filtering of step data.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_screen_renderer.py`: HTML rendering tests
  - `test_render_screen_html_structure`: Output contains `<pre>`, `<span>` tags with color styles
  - `test_render_screen_html_escapes_special_chars`: Characters like `<`, `>`, `&` are HTML-escaped
  - `test_render_screen_html_color_mapping`: Color index 1 (red) maps to correct RGB
  - `test_render_saliency_html_structure`: Saliency grid renders with focus point highlighting

- `tests/unit/test_wandb_reporter.py` (extend):
  - `test_log_media_respects_interval`: With `wandb_log_interval=3`, media is only logged on ticks 0, 3, 6...
  - `test_log_media_respects_screen_toggle`: When `wandb_log_screens=False`, screen HTML is not logged
  - `test_log_media_respects_saliency_toggle`: When `wandb_log_saliency=False`, saliency HTML is not logged
  - `test_game_table_accumulated`: After a 3-tick game, the W&B Table has 3 rows with expected columns
  - `test_game_table_logged_at_game_end`: Table is logged during `end_game()`, not during `log_step()`

- `tests/unit/test_screen_renderer.py` (extend with test helper data):
  - `test_render_known_screen`: Render a screen from `tests/helpers/nethack_screens.py` and verify it produces valid HTML with expected character count

**Implementation**:

- `roc/reporting/screen_renderer.py`: Port `render_screen_html()` from `tmp/log_screens_wandb.py`
  - Move `CURSES_COLORS` palette and `render_screen_html(screen, step)` function
  - Add `render_saliency_html(saliency_grid, focus_points)` using existing `SaliencyMap.to_debug_grid().to_html_vals()` infrastructure
  - Both return self-contained HTML strings

- `roc/reporting/wandb_reporter.py`: Extend with media and table support
  - `log_media(key, html_str)` -- wraps in `wandb.Html()`, respects `wandb_log_interval`
  - `_accumulate_step(data, screen_html, saliency_html)` -- adds row to in-progress `wandb.Table`
  - `end_game()` -- logs accumulated table as `game_{N}_steps`
  - `start_game()` -- resets table accumulator

- `roc/gymnasium.py`: Capture current screen state and pass to `WandbReporter.log_media()`
- `roc/attention.py`: After saliency map computation, render saliency HTML and pass to `WandbReporter.log_media()`

**Dependencies**:
- External: None new
- Internal: Phase 2 (per-step metric logging), `roc/reporting/state.py` (for `CurrentScreenState`), `tests/helpers/nethack_screens.py` (test data)

**Verification**:
1. Run: `roc_wandb_enabled=true roc_wandb_host=http://hal.ato.ms:8080 roc_wandb_entity=apowers uv run play`
2. Open run in W&B UI -- navigate to Media section, verify `screen` panel shows colored NetHack game screens
3. Use the step slider to scrub through screens like a video playback
4. Verify `saliency_map` panel shows heatmaps with highlighted focus points (yellow=top, orange=rank 2-3, cyan=other)
5. Open the `game_1_steps` table artifact -- verify it has all metric columns plus screen/saliency HTML columns
6. Filter the table by a condition (e.g., `objects_new > 0`) to verify SQL-like querying works
7. Run: `make test` -- all tests pass

---

### Phase 4: Artifacts & Sweeps

**What this phase accomplishes**: Config snapshots and optional graph/debug-log artifacts are uploaded at session end. Example sweep configs enable automated hyperparameter search.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_wandb_reporter.py` (extend):
  - `test_config_artifact_always_uploaded`: After `finish()`, a config artifact exists
  - `test_graph_artifact_uploaded_when_configured`: With `wandb_artifacts=["graph"]` and a graph file present, artifact is uploaded
  - `test_debug_log_artifact_uploaded_when_configured`: With `wandb_artifacts=["debug_log"]` and a debug log file present, artifact is uploaded
  - `test_artifact_missing_file_skipped`: If graph export file doesn't exist, no crash -- just a warning log
  - `test_sweep_run_uses_sweep_name`: When `wandb.run.sweep_id` is truthy, run name is not overridden

- `tests/unit/test_sweep_config.py`: Validate sweep YAML files
  - `test_sweep_yaml_valid`: Each YAML file in `experiments/sweeps/` parses without errors
  - `test_sweep_yaml_has_required_fields`: Each sweep config has `program`, `method`, `metric`, `parameters`
  - `test_sweep_parameters_match_config_fields`: All parameter names in sweep YAML correspond to valid `roc_*` env var names

**Implementation**:

- `roc/reporting/wandb_reporter.py`: Extend `finish()` with artifact uploads
  - Always upload config snapshot as `wandb.Artifact("config-{run_id}", type="config")`
  - Conditionally upload graph GML file when `"graph" in wandb_artifacts` and file exists
  - Conditionally upload debug log JSONL when `"debug_log" in wandb_artifacts` and file exists
  - Sweep detection: if `wandb.run.sweep_id` is set, skip overriding run name

- `experiments/sweeps/sweep_saliency_linear.yaml`: Example sweep for linear-decline parameters
  ```yaml
  program: .venv/bin/play
  method: bayes
  metric:
    name: mean_score
    goal: maximize
  parameters:
    roc_wandb_enabled:
      value: "true"
    roc_expmods_use:
      value: '[["saliency-attenuation", "linear-decline"]]'
    roc_saliency_attenuation_capacity:
      values: [3, 5, 7, 10]
    roc_saliency_attenuation_radius:
      values: [2, 3, 5]
    roc_saliency_attenuation_max_penalty:
      min: 0.5
      max: 1.0
    roc_num_games:
      value: 3
  ```

- `experiments/sweeps/sweep_saliency_ai.yaml`: Example sweep for active-inference parameters
- `experiments/sweeps/sweep_cross_expmod.yaml`: Example sweep comparing across ExpMod types

**Dependencies**:
- External: `pyyaml` (for sweep config validation tests -- likely already available)
- Internal: Phase 1-3 (full WandbReporter functionality)

**Verification**:
1. Run: `roc_wandb_enabled=true roc_wandb_artifacts='["graph"]' roc_graphdb_export=true roc_wandb_host=http://hal.ato.ms:8080 roc_wandb_entity=apowers uv run play`
2. Open run in W&B UI -- navigate to Artifacts tab, verify config snapshot is present and downloadable
3. Verify graph artifact is present (when graph export was enabled)
4. Run a sweep:
   ```bash
   wandb sweep experiments/sweeps/sweep_saliency_linear.yaml
   wandb agent <sweep_id>
   ```
5. Open sweep in W&B UI -- verify multiple runs appear with varied parameters, parallel coordinates plot shows parameter-to-score relationships
6. Run: `make test` -- all tests pass

---

### Phase 5: Reporting Abstraction Migration

**What this phase accomplishes**: All relevant instrumentation sites use `RocMetrics` for unified dual-emission to OTel + W&B. No duplicate instrumentation remains. OTel-only data (spans, system metrics) stays untouched.

**Duration**: 2-3 days

**Tests to Write First**:

- `tests/unit/test_metrics.py` (extend):
  - `test_all_histograms_use_roc_metrics`: Grep-style audit test -- verify no direct `Observability.meter.create_histogram` calls remain in production code (except in `metrics.py` itself)
  - `test_all_counters_use_roc_metrics`: Same for counters
  - `test_otel_only_metrics_documented`: System metrics and tracing spans are explicitly excluded from migration

- `tests/integration/test_dual_emission.py`:
  - `test_saliency_metrics_reach_both_backends`: Run attention pipeline, mock both OTel and W&B backends, verify both receive `peak_count` and `top_peak_strength`
  - `test_object_metrics_reach_both_backends`: Same for object resolution counters
  - `test_wandb_off_otel_unaffected`: With `wandb_enabled=False`, run pipeline, verify OTel metrics are unchanged from baseline

**Implementation**:

- Audit all files for direct OTel metric calls:
  - `roc/saliency_attenuation.py` -- migrate `histogram.record()` and `counter.add()` to `RocMetrics`
  - `roc/attention.py` -- migrate any direct metric calls
  - `roc/object.py` -- migrate resolution counters
  - `roc/gymnasium.py` -- migrate game/observation counters
  - `roc/action.py` -- migrate action logging (if applicable)

- `roc/reporting/metrics.py`: Ensure all metric names used across the codebase are routed through `RocMetrics` with consistent naming

- Keep OTel-only for:
  - Tracing spans (`Observability.tracer.start_as_current_span`)
  - System metrics (CPU, memory via OTel instrumentation)
  - Profiling (Pyroscope)
  - Structured log records (these are OTel-specific format)

**Dependencies**:
- External: None new
- Internal: Phase 2 (`RocMetrics` abstraction)

**Verification**:
1. Run: `roc_wandb_enabled=true roc_wandb_host=http://hal.ato.ms:8080 roc_wandb_entity=apowers uv run play`
2. Open W&B UI -- verify all expected metrics appear (complete list from design doc)
3. Open Grafana -- verify OTel dashboards still show the same metrics (no regression or missing data)
4. Run: `uv run ruff check roc/ --select E501` (or grep) -- confirm no direct histogram/counter creation outside `metrics.py`
5. Run: `make test` and `make lint` -- all pass

---

## Common Utilities Needed

- **`roc/reporting/screen_renderer.py`**: Shared HTML rendering for game screens and saliency maps. Used by `wandb_reporter.py` for media logging. Could also be used by future debug tools or web UIs.

- **`RocMetrics` (in `roc/reporting/metrics.py`)**: Central dispatch for numeric metrics. Eliminates duplicate instrumentation. Used throughout the codebase wherever metrics are recorded.

- **`ExpMod.params_dict()`**: Generic parameter extraction for any ExpMod. Used by `WandbReporter` at init and potentially by future config comparison tools.

## External Libraries Assessment

- **`wandb`**: Required. The W&B Python SDK. Well-maintained, supports offline mode and self-hosted instances. Already validated in `tmp/log_screens_wandb.py` prototype. Add to `pyproject.toml` dependencies.

- **`pyyaml`**: Likely already a transitive dependency. Used only in sweep config validation tests. If not present, the test can use `json` format for sweep configs instead.

## Risk Mitigation

- **W&B SDK import time / performance**: The `wandb` package is heavy. Mitigate by lazy-importing it only inside `WandbReporter.init()` and guarding all calls behind `wandb_enabled` checks. When disabled, zero import overhead.

- **W&B network failures during runs**: Use `wandb.init(settings=wandb.Settings(silent=True))` to suppress noisy warnings. W&B SDK already handles network failures gracefully with local buffering and retry. Document that `wandb_mode="offline"` is available for fully disconnected runs.

- **Test isolation**: All tests use `wandb_mode="disabled"` via the existing test config isolation (pytest overrides env vars). No network calls in tests. The W&B SDK's disabled mode is specifically designed for this.

- **Per-step media storage size**: Each HTML screen is ~5-10KB. At 1000 steps/game x 5 games = ~50-100MB per run. This is manageable for W&B. The `wandb_log_interval` escape hatch exists for very long runs. Document expected storage in the sweep configs.

- **Breaking existing OTel pipeline**: Phase 5 (migration) is explicitly last and gradual. Each migration swaps one call site at a time. Integration tests verify dual-emission. OTel-only features (spans, system metrics) are never touched.

- **ExpMod `params_dict()` exposing sensitive data**: The default implementation only returns public instance attributes. If an ExpMod stores sensitive state, it can override `params_dict()` to exclude it. Review existing ExpMods during Phase 1 implementation.
