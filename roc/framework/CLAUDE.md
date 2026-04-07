# Framework

## Why This Design

The framework provides the structural skeleton for ROC's component-based, event-driven
architecture. The core design motivation is **future distribution**: components may
eventually run on separate machines (e.g., visual feature extraction on GPU hardware,
prediction on high-memory nodes). The EventBus abstraction exists specifically to become
a network boundary -- today it uses RxPY in-process subjects, but it is designed to be
swapped for inter-process transports (e.g., ROC2 DDS) without changing any component
code. This is why direct method calls between components are forbidden -- they would
create coupling that prevents distribution.

ExpMods exist for **scientific experimentation**, not just code modularity. ROC explores
different RL approaches, and being able to swap algorithms, record which was active, and
compare results through the dashboard is essential for learning what works. Config
supports **scientific reproducibility** -- capturing full configuration state so
experiments can be replicated exactly.

## Key Decisions

- **RxPY is a transitional transport** -- the EventBus API (send/listen) is the stable
  interface. RxPY subjects are an implementation detail that will be replaced (likely by
  ROC2 DDS for inter-process communication). Do not depend on RxPY-specific features
  (operators, schedulers, subject internals) outside of event.py itself.

- **EventBus names are globally unique** -- enforced at creation time in a module-level
  set. This makes the bus topology static and discoverable, required for debugging/tracing
  and for mapping bus names to future transport channels.

- **EventBuses must be class-level attributes on Components** -- no ad-hoc bus creation
  in functions, methods, or module scope. The topology must be knowable at class
  definition time, before any instances exist.

- **ExpMod instances are created at import time** via `__init_subclass__`, not lazily.
  Config may not be initialized yet when an ExpMod's `__init__` runs. ExpMods that read
  Config must use try/except around `Config.get()` and fall back to class-attribute
  defaults. This pattern is documented in expmod.py's module docstring.

- **Config.init() runs at module import time** (bottom of config.py). Logger and other
  framework modules need Config available when they initialize, so the first import of
  anything in `roc.framework` triggers config loading from env vars and .env.

- **Pytest deliberately ignores env vars and .env** -- config.py detects `pytest` in
  `sys.modules` and mangles the env prefix to a garbage string. This ensures tests are
  deterministic and never polluted by local .env files. Tests configure via
  `Config.init(config={...}, force=True)`.

- **Config contains secrets** (db_password, SSL cert/key paths). Any reproducibility
  feature that captures or exports config state must not leak these values.

## Invariants

- **One live instance per Component name+type pair.** The WeakSet check in `__init__`
  raises ValueError on duplicates. Violating this creates ambiguous event routing -- two
  instances would both listen and send on the same buses with the same ComponentId,
  making events impossible to trace.

- **EventBus names must be unique.** Duplicate names would silently merge unrelated event
  streams into a single RxPY subject, causing components to receive events with data
  types they cannot handle.

- **Config must be accessed via `Config.get()`, never `Config()`.** Constructing Config
  directly bypasses the singleton and creates a second instance with potentially different
  values. Code reading the "other" Config would see stale or default settings.

- **ExpMod selection must go through `ExpMod.get(default="name")`.** Hardcoding a
  specific implementation bypasses config-driven selection, making it invisible to the
  dashboard and breaking experiment tracking.

## Non-Obvious Behavior

### Threading Model

All EventBus listeners execute on a shared `ThreadPoolScheduler` (cpu_count * 2 threads).
Multiple listeners on the same bus can execute concurrently. **There are no ordering
guarantees** between events. There is pending work to add thread mutexes and/or prove
why races are not an issue in practice. Do not assume thread safety when adding listeners
that share mutable state with other listeners.

### Self-Send Filtering

`Component.event_filter()` drops events where `src_id` matches the receiving component.
This prevents infinite loops when a component both sends and listens on the same bus.
Override only if you specifically need to receive your own events.

### Shutdown Tears Down All Observers

`Component.shutdown()` calls `on_completed()` on all observers of every bus the component
is connected to -- not just its own subscribers. This is a known limitation: shutting down
one component terminates listeners from other components on shared buses. In practice this
is not a problem because `Component.reset()` shuts down all components together. Do not
call `shutdown()` on individual components and expect others to keep working.

### Initialization Order

1. `config.py` import triggers `Config.init()` (singleton from env/.env)
2. `logger.py` reads Config on first `LogFilter()` creation (during `logger.init()`)
3. ExpMod subclass definitions trigger `__init_subclass__`, which instantiates them
   (Config may not exist yet -- hence the try/except pattern)
4. `ExpMod.init()` loads external module files and activates config-selected implementations
5. `Component.init()` instantiates auto-loaded and perception components

## Anti-Patterns

- **Do not import RxPY in component code.** Components interact with the bus through
  `BusConnection.send()` and `BusConnection.listen()`. Using RxPY operators or subjects
  directly couples to the current transport, which will be replaced.

- **Do not block inside event listeners.** Listeners run on the shared ThreadPoolScheduler.
  A blocking call (sleep, synchronous HTTP, lock wait) starves event processing across the
  entire pipeline. Use the reporting layer's threading patterns for blocking work.

- **Do not call `Config.init()` in component code.** Use `Config.get()`. Calling init
  again after the singleton exists logs a warning and is ignored (unless `force=True`,
  which is for tests only).

- **Do not hardcode algorithm implementations.** Even with only one implementation today,
  wire through ExpMod so alternatives can be swapped via config without code changes.
