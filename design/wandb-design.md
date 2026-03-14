# W&B Integration Design for ROC

## Goals

1. **Performance understanding** -- review each run's behavior step-by-step, including game screens and saliency maps, to understand what worked and what didn't
2. **Replicability** -- capture full configuration (Config + active ExpMods + parameters) so any run can be reproduced
3. **Future improvements** -- store data in queryable, portable formats that support offline analysis and model training
4. **Trend analysis** -- compare metrics across runs over time to detect regressions or improvements
5. **Parameter comparison** -- compare different ExpMod configurations and hyperparameters, including automated sweep support

## Architecture

### Relationship to Existing Observability

W&B is a **peer** to the existing OTel pipeline, not downstream of it. Both systems receive the same data but serve different purposes:

| Concern | OTel (Grafana/Prometheus/Loki) | W&B |
|---------|-------------------------------|-----|
| Primary use | Live monitoring, alerting, system health | Experiment tracking, comparison, review |
| Granularity | Continuous time-series | Per-run, per-step |
| Retention | Rolling window | Permanent per-run |
| Rich media | No | Screens, saliency maps, artifacts |
| Hyperparameters | No | Yes (config, sweeps) |
| Comparison | Ad-hoc PromQL | Built-in run comparison, tables |

To avoid duplicate instrumentation, introduce a thin **reporting abstraction** that emits to both systems from a single call site.

### Reporting Abstraction

Add `roc/reporting/metrics.py` with a unified interface:

```python
class RocMetrics:
    """Unified metrics reporting to OTel and W&B."""

    @staticmethod
    def record_histogram(name: str, value: float, attributes: dict | None = None) -> None:
        """Record a value to both OTel histogram and W&B step log."""
        ...

    @staticmethod
    def increment_counter(name: str, amount: int = 1, attributes: dict | None = None) -> None:
        """Increment both OTel counter and W&B step log."""
        ...

    @staticmethod
    def log_step(data: dict) -> None:
        """Log a dict of values as a W&B step. OTel equivalent is structured log."""
        ...

    @staticmethod
    def log_media(key: str, html: str) -> None:
        """Log HTML media (screens, heatmaps) to W&B only."""
        ...
```

Existing call sites that currently call OTel directly (e.g., `histogram.record(value)`) would migrate to `RocMetrics.record_histogram()` which fans out to both backends. This is a gradual migration -- existing OTel calls continue to work, and new instrumentation uses the unified interface.

**When not to unify**: Some data is backend-specific. OTel tracing spans and system metrics (CPU, memory) stay OTel-only. Rich media (screens, saliency maps) and hyperparameter config are W&B-only. The abstraction handles the common case (numeric metrics and structured decisions) and each backend retains its own direct API for specialized data.

### W&B Module

Add `roc/reporting/wandb_reporter.py` as the W&B integration point:

```
roc/reporting/
  observability.py    # Existing OTel setup (unchanged)
  remote_logger_exporter.py  # Existing remote logger (unchanged)
  state.py            # Existing state tracking (unchanged)
  metrics.py          # NEW: unified reporting abstraction
  wandb_reporter.py   # NEW: W&B lifecycle management
```

`WandbReporter` is a singleton (matching the `Observability` pattern) that manages the W&B run lifecycle:

```python
class WandbReporter:
    _instance: WandbReporter | None = None
    _run: wandb.Run | None = None
    _step: int = 0
    _game_num: int = 0

    @classmethod
    def init(cls, config: Config) -> None: ...

    @classmethod
    def start_game(cls, game_num: int) -> None: ...

    @classmethod
    def end_game(cls, outcome: str, summary: dict) -> None: ...

    @classmethod
    def log_step(cls, data: dict) -> None: ...

    @classmethod
    def log_media(cls, key: str, content: wandb.Html | wandb.Image) -> None: ...

    @classmethod
    def finish(cls) -> None: ...
```

## Run Structure

### One W&B Run Per Session

Each `uv run play` invocation creates **one W&B run**. This run may contain multiple games (episodes) controlled by `num_games`.

**Rationale**: A session is the natural unit of an experiment. You set config, choose ExpMods, and run N games to evaluate that configuration. Sweeps also operate at the session level.

### Game Tagging

Games within a run are distinguished by:

1. **Step-level `game_num` field** -- logged with every `wandb.log()` call, enabling filtering by game
2. **Game boundary markers** -- `wandb.log({"game_start": game_num})` and `wandb.log({"game_end": game_num, "outcome": "...", "final_score": ...})`
3. **W&B Tags** -- the run itself gets tags like `games:5` for easy filtering in the project view

This allows filtering any chart or table by game number while keeping all games in a single comparable run.

### Step Counter

W&B uses a global step counter across the entire run (not reset per game). The step maps to game ticks:

```
Game 1: steps 0..N
Game 2: steps N+1..M
Game 3: steps M+1..P
```

Each step log includes `game_num` and `game_tick` (tick within the current game, reset per game) for filtering.

## Configuration

### Config Fields

Add to `roc/config.py`:

```python
# W&B integration
wandb_enabled: bool = False
wandb_project: str = "ROC"
wandb_entity: str = ""           # W&B username or team
wandb_host: str = ""             # Empty = wandb.ai cloud; set for self-hosted
wandb_api_key: str = ""          # API key (or use WANDB_API_KEY env var)
wandb_tags: list[str] = []       # Additional run tags
wandb_log_screens: bool = True   # Log game screens per step
wandb_log_saliency: bool = True  # Log saliency heatmaps per step
wandb_log_interval: int = 1      # Log media every N steps (1 = every step, default)
wandb_artifacts: list[str] = []  # Artifact types to save: "graph", "debug_log"
wandb_mode: str = "online"       # "online", "offline", "disabled"
```

**Environment variable examples**:
```bash
roc_wandb_enabled=true
roc_wandb_entity=apowers
roc_wandb_host=http://hal.ato.ms:8080
roc_wandb_api_key=local-wandb_v1_...
```

**Self-hosted vs cloud**: When `wandb_host` is empty, W&B defaults to cloud (wandb.ai). When set, it points to the self-hosted instance. No other configuration differences -- the SDK handles the rest. This adds zero complexity to the user-facing config.

### Run Config (Hyperparameters)

At run init, log the full experiment configuration to `wandb.config`. Use the existing `instance_id` from `observability.py` (FlexiHumanHash format: `YYYYMMDDHHMMSS-adj-firstname-lastname`) as the W&B run name, keeping naming consistent across all observability systems:

```python
from roc.reporting.observability import instance_id, roc_version

wandb.init(
    name=instance_id,  # Reuse existing ROC instance ID
    config={
        # ROC Config fields (all of them, for full replicability)
        "roc_config": config.model_dump(),

        # Active ExpMods with their parameters
        "expmods": {
            modtype: {
                "name": name,
                "params": expmod_instance.params_dict(),
            }
            for modtype, name in active_expmods
        },

        # Environment metadata
        "python_version": sys.version,
        "roc_version": roc_version,
        "instance_id": instance_id,
    }
)
```

This enables:
- Filtering runs by any config value in the W&B UI
- Comparing hyperparameters across runs in parallel coordinates plots
- Full replication: given the config dump, you can reconstruct the exact command

### ExpMod Parameter Tracking

Add a `params_dict()` method to the `ExpMod` base class:

```python
class ExpMod:
    def params_dict(self) -> dict[str, Any]:
        """Return tunable parameters for experiment tracking.

        Override in subclasses that have configurable parameters.
        Default implementation returns all non-private instance attributes.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
```

Each ExpMod's parameters are automatically captured at run init. When comparing runs, you can see exactly which parameters differed (e.g., `expmods.saliency-attenuation.params.max_attenuation: 0.9 vs 0.7`).

## Per-Step Data

### Numeric Metrics (logged every step or every `wandb_log_interval` steps)

These are the metrics logged via `wandb.log()` at each game tick:

**Core state:**
| Metric | Source | Description |
|--------|--------|-------------|
| `game_num` | gymnasium.py | Current game number |
| `game_tick` | gymnasium.py | Tick within current game |
| `hp` | intrinsic | Current HP (raw) |
| `hp_normalized` | intrinsic | HP as fraction of max |
| `energy` | intrinsic | Current energy |
| `hunger` | intrinsic | Hunger state (normalized) |
| `score` | blstats | Game score |
| `gold` | blstats | Gold carried |
| `depth` | blstats | Dungeon depth |
| `x`, `y` | blstats | Agent position |

**Attention & saliency:**
| Metric | Source | Description |
|--------|--------|-------------|
| `peak_count` | saliency_attenuation | Number of saliency peaks |
| `top_peak_strength` | saliency_attenuation | Strength of strongest peak |
| `top_peak_shifted` | saliency_attenuation | Whether IOR shifted the peak (0/1) |
| `focus_x`, `focus_y` | attention | Primary focus point location |

**Active inference (when flavor=active-inference):**
| Metric | Source | Description |
|--------|--------|-------------|
| `entropy_at_focus` | saliency_attenuation | Entropy at attended location |
| `entropy_min` | saliency_attenuation | Min entropy across beliefs |
| `entropy_max` | saliency_attenuation | Max entropy across beliefs |
| `omega` | saliency_attenuation | Global volatility |
| `beliefs_tracked` | saliency_attenuation | Number of location beliefs |
| `vocab_size` | saliency_attenuation | State vocabulary size |

**Object resolution:**
| Metric | Source | Description |
|--------|--------|-------------|
| `objects_resolved` | object.py | Objects resolved this tick |
| `objects_new` | object.py | New objects created |
| `objects_matched` | object.py | Objects matched to existing |
| `resolution_confidence` | object.py | Mean posterior confidence |

**Action:**
| Metric | Source | Description |
|--------|--------|-------------|
| `action` | action.py | Action ID taken |

### Rich Media (logged every step, or at `wandb_log_interval`)

| Key | Type | Description |
|-----|------|-------------|
| `screen` | `wandb.Html` | Game screen rendered as colored HTML (21x79 characters with curses colors) |
| `saliency_map` | `wandb.Html` | Saliency heatmap rendered as colored HTML with focus points highlighted (yellow=top, orange=rank 2-3, cyan=other) |

**Implementation**: Reuse the existing `SaliencyMap.to_debug_grid().to_html_vals()` infrastructure and the HTML rendering from `attention.py`'s structured log. The `render_screen_html()` function from our prototype (`tmp/log_screens_wandb.py`) becomes the production renderer.

**Interval control**: By default, screens and saliency maps are logged every step (`wandb_log_interval=1`) since this is the primary way to review games. The `wandb_log_interval` setting exists as an escape hatch for very long runs -- setting it to 5 means media is logged every 5th tick. Numeric metrics are always logged every tick regardless of this setting.

### Game Boundary Events

At game start:
```python
wandb.log({
    "game_start": game_num,
    "game_tick": 0,
})
```

At game end:
```python
wandb.log({
    "game_end": game_num,
    "outcome": "death" | "win" | "truncated",
    "final_score": score,
    "final_depth": depth,
    "total_ticks": tick_count,
    "game_duration_seconds": elapsed,
})
```

### Run Summary

At session end, `wandb.run.summary` gets aggregate stats:

```python
run.summary["total_games"] = game_count
run.summary["total_steps"] = total_steps
run.summary["mean_score"] = mean(scores)
run.summary["max_score"] = max(scores)
run.summary["mean_ticks_survived"] = mean(ticks_per_game)
run.summary["win_rate"] = wins / game_count
```

These summary metrics appear in the W&B project table, enabling quick comparison across runs without drilling into step data.

### W&B Tables

In addition to `wandb.log()` for real-time charting, log a `wandb.Table` per game for richer querying. Tables support SQL-like filtering, sorting, and grouping in the W&B UI.

**Game step table** (one per game, logged at game end):

```python
columns = [
    "game_tick", "hp", "hp_normalized", "energy", "hunger",
    "score", "gold", "depth", "x", "y",
    "peak_count", "top_peak_strength", "top_peak_shifted",
    "entropy_at_focus", "omega", "beliefs_tracked",
    "objects_resolved", "objects_new", "objects_matched",
    "action",
    "screen",       # wandb.Html
    "saliency_map", # wandb.Html
]
table = wandb.Table(columns=columns)

# Accumulate rows during game, then log at game end:
wandb.log({f"game_{game_num}_steps": table})
```

**Object resolution table** (one per game, logged at game end):

```python
columns = [
    "game_tick", "outcome", "candidate_count",
    "matched_obj_id", "match_score", "location_x", "location_y",
    "features",
]
```

This enables filtering like "show me all ticks where a new object was created" or "find ticks where resolution confidence was below 0.5" without writing code.

## Artifacts

### Configurable via `wandb_artifacts`

| Artifact Type | Config Value | Contents | Notes |
|---------------|-------------|----------|-------|
| Graph export | `"graph"` | GML file from `graphdb_export` | Can take several minutes; off by default |
| Debug log | `"debug_log"` | JSONL file from `debug_log` | Duplicative of W&B step data; include only if you want a portable single-file record |
| Config snapshot | (always) | `config.json` with full Config dump | Small, always included for replicability |

**Implementation**: At `WandbReporter.finish()`, check `wandb_artifacts` list and upload matching files:

```python
if "graph" in config.wandb_artifacts:
    artifact = wandb.Artifact(f"graph-{run.id}", type="graph")
    artifact.add_file(graph_export_path)
    run.log_artifact(artifact)
```

**Other artifacts to consider in the future:**
- **Trained models / learned parameters**: If ROC develops learned components (e.g., trained action policies, learned feature weights), these would be natural artifacts
- **Replay data**: A compact binary format for replaying games without the full environment, useful for training offline
- **Object graph snapshots**: Serialized graph state at interesting moments (e.g., when a new object type is first encountered)

## Dashboards and Visualizations

### Default W&B Workspace Panels

When a user opens the ROC project in W&B, configure these default panels:

**Section 1: Game Overview**
- Line plot: `score` vs step (grouped by `game_num`)
- Line plot: `depth` vs step (grouped by `game_num`)
- Line plot: `hp` vs step (grouped by `game_num`)
- Scalar: `total_games`, `mean_score`, `max_score` (from summary)

**Section 2: Screen Review**
- MediaBrowser: `screen` (step slider for game playback)
- MediaBrowser: `saliency_map` (step slider, side-by-side with screen)

**Section 3: Attention & Saliency**
- Line plot: `peak_count` vs step
- Line plot: `top_peak_strength` vs step
- Line plot: `entropy_at_focus` vs step (active-inference runs)
- Line plot: `omega` vs step (active-inference runs)
- Bar chart: `top_peak_shifted` cumulative count

**Section 4: Object Resolution**
- Line plot: `objects_resolved` cumulative vs step
- Line plot: `objects_new` vs `objects_matched` per step
- Line plot: `resolution_confidence` vs step

**Section 5: Agent State**
- Line plot: `energy` vs step
- Line plot: `hunger` vs step
- Scatter plot: `x` vs `y` colored by step (agent trajectory)

### Cross-Run Comparison Views

The W&B project table automatically shows all runs with summary metrics. Key comparison workflows:

1. **ExpMod comparison**: Filter by `config.expmods.saliency-attenuation.name`, compare `mean_score` across `linear-decline` vs `active-inference` vs `no-attenuation`
2. **Hyperparameter impact**: Parallel coordinates plot with `config.expmods.saliency-attenuation.params.max_attenuation` on one axis and `mean_score` on the other
3. **Regression detection**: Sort runs by date, plot `mean_score` over time to detect regressions after code changes
4. **Per-game breakdown**: Within a run, filter step data by `game_num` to compare game-to-game variance

## Sweep Support

### W&B Sweeps for Hyperparameter Search

Define sweep configurations that explore ExpMod parameters:

```yaml
# sweep_saliency.yaml
program: .venv/bin/play
method: bayes  # or grid, random
metric:
  name: mean_score
  goal: maximize
parameters:
  roc_expmods_use:
    value: '[["saliency-attenuation", "active-inference"]]'
  roc_saliency_attenuation_ai_max_attenuation:
    min: 0.5
    max: 1.0
  roc_saliency_attenuation_ai_saliency_weight:
    min: 0.0
    max: 1.0
  roc_saliency_attenuation_ai_omega_alpha_prior:
    values: [1.0, 2.0, 5.0, 10.0]
  roc_saliency_attenuation_ai_omega_beta_prior:
    values: [0.5, 1.0, 2.0]
  roc_num_games:
    value: 5
```

**How it works**: W&B sweep agent launches `play` with environment variables overriding Config fields. Since Config reads from env vars with `roc_` prefix, no code changes are needed for sweep parameter injection -- Config's pydantic-settings integration handles it automatically.

**Sweep initialization**: `WandbReporter.init()` detects if it's running inside a sweep (via `wandb.run.sweep_id`) and adjusts behavior accordingly (e.g., uses the sweep-assigned run name instead of generating one).

**Multi-ExpMod sweeps**: To compare across ExpMod types, use a categorical parameter:

```yaml
parameters:
  roc_expmods_use:
    values:
      - '[["saliency-attenuation", "no-attenuation"]]'
      - '[["saliency-attenuation", "linear-decline"]]'
      - '[["saliency-attenuation", "active-inference"]]'
```

### Sweep Metrics

The sweep optimizer needs a single scalar metric to optimize. Use `mean_score` from run summary as the default. Other candidates:
- `mean_ticks_survived` -- for survival-focused optimization
- `max_depth` -- for exploration-focused optimization
- Custom composite metrics defined in the sweep config

## Implementation Plan

### Phase 1: Core Integration

1. Add W&B config fields to `config.py`
2. Create `roc/reporting/wandb_reporter.py` with `WandbReporter` singleton
3. Add `params_dict()` to `ExpMod` base class
4. Hook `WandbReporter.init()` into `roc.init()` (after Config and Observability)
5. Hook `WandbReporter.start_game()` / `end_game()` into `gymnasium.py` game loop
6. Hook `WandbReporter.finish()` into session teardown
7. Log full config + ExpMod params at run init
8. Log run summary at session end

**Deliverable**: Runs appear in W&B with correct config, game boundaries, and summary metrics. No per-step data yet.

### Phase 2: Per-Step Metrics

1. Create `roc/reporting/metrics.py` with `RocMetrics` abstraction
2. Migrate key instrumentation sites to use `RocMetrics`:
   - `gymnasium.py`: core state (hp, score, depth, position)
   - `saliency_attenuation.py`: peak count, strength, entropy, omega
   - `object.py`: resolution decisions, object counts
   - `action.py`: action taken
3. Existing OTel-only sites continue unchanged; new unified sites emit to both

**Deliverable**: Numeric metrics appear in W&B charts per step. OTel continues to receive the same data.

### Phase 3: Rich Media

1. Add screen HTML renderer to `wandb_reporter.py` (port from prototype)
2. Add saliency map HTML renderer (reuse `to_debug_grid().to_html_vals()`)
3. Hook into `attention.py` / `gymnasium.py` to capture and log media every step (respecting `wandb_log_interval` as escape hatch)
4. Respect `wandb_log_screens` and `wandb_log_saliency` toggles
5. Accumulate step data into `wandb.Table` per game; log table at game end

**Deliverable**: Game screens and saliency heatmaps visible in W&B MediaBrowser, scrubbable by step.

### Phase 4: Artifacts & Sweeps

1. Implement artifact upload at session end (graph export, config snapshot)
2. Create example sweep YAML configs in `experiments/sweeps/`
3. Document sweep workflow (create, launch agent, analyze results)
4. Test sweep agent with self-hosted instance

**Deliverable**: Full experiment workflow -- launch sweeps, compare results, download artifacts.

### Phase 5: Reporting Abstraction Migration

1. Audit all existing OTel instrumentation sites
2. Identify which should also emit to W&B
3. Gradually migrate to `RocMetrics` where dual-emission is needed
4. Keep OTel-only for system metrics, tracing spans, profiling

**Deliverable**: All relevant metrics flow to both OTel and W&B from unified call sites. No duplicate instrumentation.

## Testing Strategy

- **Unit tests**: `WandbReporter` with `wandb_mode="disabled"` -- verify no crashes when W&B is off
- **Unit tests**: `RocMetrics` abstraction with mocked backends
- **Unit tests**: `ExpMod.params_dict()` returns expected parameters
- **Integration test**: Full game run with `wandb_mode="offline"` -- verify offline run directory contains expected data
- **Manual test**: Run against self-hosted instance, verify screens and metrics appear in UI

Tests should not require a W&B server. Use `wandb_mode="disabled"` in the test config fixture to ensure W&B code paths are exercised without network calls.

## Known Limitations

### Media Panel Step Synchronization

W&B's "Sync slider by key (Step)" feature only works for image, video, audio, and point
cloud panels. It does NOT work for `wandb.Html` panels. Since ROC logs game screens and
saliency maps as `wandb.Html`, the two media panels have independent step sliders -- moving
one does not update the other.

The sync feature is found in: Section settings > Media settings > Sync tab. The "Media
settings" menu item only appears when the section contains image/video/audio panels. It is
completely absent for sections containing only HTML panels.

**Investigated alternatives:**
- Switching to `wandb.Image` (PIL-rendered PNGs) would enable sync but: (a) requires
  rendering every frame as an image, (b) loses text interactivity, (c) cannot sync with
  text/log panels anyway since sync only works between supported media types
- Inter-panel communication via JavaScript is blocked because W&B renders HTML panels in
  iframes with `sandbox="allow-scripts"` (no `allow-same-origin`), preventing
  `BroadcastChannel`, `localStorage`, or `postMessage` relay between panels
- Combining multiple media keys into one panel (e.g., screen + saliency in the same panel)
  shares a single slider but produces an interleaved view that is hard to read, and older
  runs that lack saliency data show error placeholders

**Impact:** This is the primary limitation that motivated the Panel dashboard design
(see `design/panel-design.md`). Per-step game state review -- the main use case for the
media panels -- requires synchronized stepping across screen, saliency, objects, logs, and
metrics, which W&B cannot provide.

### No Text/Log Step Slider

There is no way to associate text or log data with a specific W&B step and scrub through
it with a slider. `wandb.Html` panels have step sliders, but general text content,
`wandb.Table` rows, and log lines do not. GitHub issue #6286 (step slider for
`wandb.Table`) has 30+ upvotes and remains open.

This means even if screen and saliency panels were synced, there is no W&B mechanism to
show "objects resolved at step 42" or "log messages at step 42" alongside the visual
panels.

### Self-Hosted Version Feature Gaps

The self-hosted W&B instance (hal.ato.ms:8080) may lag behind the cloud version in feature
availability. The media sync feature was found to be available for image panels but the
documentation does not clearly state which self-hosted server versions include it. Feature
parity with wandb.ai cloud is not guaranteed.

### Duplicate Data Pipeline

W&B runs as a peer to the OTel pipeline, meaning all per-step data is emitted twice --
once to OTel (Prometheus/Loki/Grafana) and once to W&B. The `RocMetrics` abstraction
mitigates the code duplication but the data itself is stored in two places. This increases
storage usage and means dashboards/queries must be maintained in both systems.

### Recommendation

Given these limitations, W&B's primary remaining value is:
- **Sweep orchestration** (replaceable by Optuna or Ray Tune)
- **Cross-run comparison** tables (available in Grafana via Prometheus labels)
- **Artifact versioning** (not currently used heavily)

Per-step game state debugging -- the most important debugging workflow -- should move to
a dedicated Panel dashboard that reads from the JSONL debug log. See
`design/panel-design.md` for that design. W&B may be removed entirely once the Panel
dashboard and an alternative sweep solution (e.g., Optuna) are in place.

## Decisions

1. **W&B Tables**: Yes -- log per-step data as `wandb.Table` in addition to `wandb.log()`. Tables enable richer querying (SQL-like filters, joins across columns). Particularly useful for object resolution decisions where you want to filter by outcome type. Log a table per game with columns for all step metrics plus media references.

2. **Run naming**: Use ROC's existing `instance_id` (FlexiHumanHash format: `YYYYMMDDHHMMSS-adj-firstname-lastname`) as the W&B run name. This keeps naming consistent across OTel, debug logs, remote logger, and W&B.

3. **Multi-agent future**: Deferred -- will address when the need arises. W&B supports run groups for this pattern.

4. **Media logging frequency**: Default to logging screens and saliency maps at **every step** (`wandb_log_interval=1`). This is the primary way to review games and understand agent behavior. The interval config exists as an escape hatch for very long runs where storage becomes a concern, but the default prioritizes full observability.
