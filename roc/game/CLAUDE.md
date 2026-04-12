# Game Subsystem

## Why This Design

The game subsystem exists to decouple the ROC agent from any specific environment.
Currently ROC targets NetHack via NLE/Gymnasium, but the long-term intent is one-shot
reinforcement learning against arbitrary environments -- other Gymnasium envs (MuJoCo,
Isaac Gym), and eventually ROS2 with real robotics. The stable contract is not the Gym
class hierarchy but the three buses it writes to: perception, action, and intrinsic.
When the environment changes, this entire directory gets replaced; the rest of the
pipeline stays the same.

This code is deliberately shim-quality. It formats environment observations for the
feature extractors and drives the synchronous turn loop. Maintenance and polish are
not primary concerns -- the code is disposable by design.

## Key Decisions

- **Thread mode** -- GameManager runs the game as a daemon thread within the server
  process. All Python objects (GraphCache, StepCache, DataStore) are shared via the
  process heap. No serialization, no HTTP callbacks, no IPC. The game pushes StepData
  via `push_step_from_game`, which invokes `RunWriter.push_step` to fan out to the cache,
  the async DuckLake exporter, and any subscribers. The game does not need to know whether
  anyone is watching.

- **Stop signaling via threading.Event** -- The server sets a `threading.Event` and
  the game thread checks `stop_event.is_set()` each step. The thread is marked as
  `daemon=True` so it dies when the server process exits. If the thread hangs, the
  join timeout expires and the server logs a warning but remains responsive.

- **Action retrieved from bus cache, not a blocking subscribe** -- `get_action()` scans
  the action bus cache for the last TakeAction event. This is historical -- a blocking
  subscribe would work because the full pipeline runs synchronously within `send_obs()`
  before `get_action()` executes. The cache approach remains because it works and this
  code is disposable.

- **Breakpoints are largely obsolete** -- The `breakpoint.py` module was a Jupyter-era
  debugging tool for pausing the game loop. The DAP MCP debugger and the dashboard's
  full historical traceability have superseded it. Treat as legacy infrastructure.

## Invariants

**The game subsystem communicates ONLY via perception, action, and intrinsic buses.**
Gym connects to exactly three buses: `Perception.bus`, `Action.bus`, and `Intrinsic.bus`.
It must not listen on or send to any other bus (object, sequencer, transformer, predict,
significance, attention). If it did, downstream components would have an invisible
dependency on the game layer, breaking environment portability. When the Gym is swapped
for a different environment driver, only the bus-writing code needs to change.

**Reporting data must never flow back into the agent.** Over half of gymnasium.py
collects dashboard/archival data (screen rendering, transform summaries, graph stats,
inventory, metrics). This data exists for human debugging and scientific review ONLY.
Pipeline components must never import or consume reporting structures. If a component
needs data, it must arrive via an EventBus from another component -- not from the
reporting/display collection layer. Violating this creates a hidden feedback loop where
the agent's behavior depends on display-layer state.

**NLE numpy buffers must be copied before storage.** NLE reuses internal numpy arrays
between steps. Any code that holds a reference to `obs["tty_chars"]`, `obs["tty_colors"]`,
etc. across steps must `.copy()` them first. Without the copy, stored data silently
mutates when the next step overwrites the buffer.

## Non-Obvious Behavior

**The synchronous loop contract**: The game loop calls `send_obs()` which publishes
VisionData, AuditoryData, ProprioceptiveData, and IntrinsicData on the perception and
intrinsic buses. The entire pipeline (attention -> object -> sequencer -> transformer ->
predict -> action) runs to completion synchronously within those bus dispatches before
control returns to `get_action()`. This works because RxPY's ThreadPoolScheduler
processes all downstream handlers before the send returns. The pipeline's natural
endpoint (Predict) is what Action waits for -- do not add custom synchronization
barriers between pipeline stages and Action.

**Dashboard data push goes through RunWriter.** `_push_dashboard_data()` invokes
`push_step_from_game()` which calls `RunWriter.push_step`. This fans out to the
in-process `StepCache`, the async `ParquetExporter`, and any registered subscribers.
In standalone mode (`uv run play`), the RunWriter is created by `start_dashboard()`.
If no RunWriter is registered, the push is skipped silently.

**GameManager receives run names via callback.** The game thread calls the `on_run_name`
callback to report its run name to the GameManager. This replaces the old filesystem
polling approach used in subprocess mode.

**Per-step state reset timing matters.** Cycle accumulators (saliency_cycles,
resolution_cycles, attenuation_cycles) are reset AFTER dashboard data has been read
from them. Moving the reset before the dashboard push loses that step's cycle data.

## Anti-Patterns

- **Do not import from roc.game in pipeline components.** The game layer depends on
  the pipeline (it imports Action, Perception, Intrinsic to access their buses), not
  the other way around. Any reverse dependency breaks environment portability.

- **Do not add NLE-specific logic outside this directory and feature_extractors/.** The
  only exception is `roc/reporting/state.py` for screen rendering in the display layer.
  If you need NLE-specific data downstream, extract it as an abstract feature first.

- **Do not add new buses to Gym.** The three-bus contract (perception, action, intrinsic)
  is the stable interface. If new data needs to flow from the environment, it should go
  through one of these existing buses as a new data type.

- **Clean up NLE state between runs.** Call `env.close()` and `del env` to fully
  release NLE's C extensions between game runs. Prototype testing confirmed this works
  reliably for sequential and concurrent runs in a single process.

- **Do not add synchronization between pipeline stages and Action.** The predict bus
  result is the natural synchronization point. See the root CLAUDE.md pipeline
  synchronization section.

## Interfaces

- **Writes to**: `Perception.bus` (VisionData, AuditoryData, ProprioceptiveData),
  `Intrinsic.bus` (IntrinsicData), `Action.bus` (ActionRequest)
- **Reads from**: `Action.bus` cache (TakeAction -- the pipeline's response)
- **Step reporting**: Calls `push_step_from_game()` -> `RunWriter.push_step` ->
  `StepCache` + `ParquetExporter` + subscribers. No direct StepBuffer access.
- **GameManager**: Manages game lifecycle via daemon thread, controlled via REST
  endpoints (`/api/game/start`, `/api/game/stop`). Stop signaling via
  `threading.Event`.
