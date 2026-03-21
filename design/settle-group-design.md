# SettleGroup Design

## Problem

Multiple components in the ROC pipeline need to wait for a set of sources to
finish before proceeding:

- **VisionAttention** waits for all feature extractors to signal `Settled`
  before emitting focus points. This is implemented inline with a `set[str]`
  and manual bookkeeping (attention.py lines 378-414).
- **Sequencer** needs to wait for object resolution and intrinsics before
  emitting a Frame. Currently it uses TakeAction as the frame-closure trigger,
  which creates a circular dependency that prevents Action from waiting on
  downstream pipeline stages (Predict).

Both components implement the same pattern: track which sources have reported
in, fire a callback when all have, reset for the next step. There is no
watchdog for stalled sources, no warning for unexpected events, and no reuse
between the two.

## Design

A reusable `SettleGroup` utility that encapsulates the settle-and-fire pattern.

### Location

`roc/event.py`, alongside `EventBus` and `BusConnection`. It is event-system
infrastructure, not a component.

### API

```python
class SettleGroup:
    def __init__(
        self,
        name: str,
        sources: set[str],
        on_settled: Callable[[], None],
        ignored: set[str] | None = None,
        timeout: float = 5.0,
    ) -> None: ...

    def notify(self, source: str) -> None: ...
    def reset(self) -> None: ...
    def add_source(self, source: str) -> None: ...
    def remove_source(self, source: str) -> None: ...
```

### Parameters

- **name**: Human-readable label for log messages (e.g. `"VisionAttention"`,
  `"Sequencer"`).
- **sources**: Set of source names that must all report before the callback
  fires. Can be dynamic -- `add_source()` / `remove_source()` modify it at
  runtime.
- **on_settled**: Callback invoked (on the calling thread) when all sources
  have reported. SettleGroup automatically resets after firing.
- **ignored**: Set of source names that are known but not required for
  settling. Calls to `notify()` with an ignored name are silently dropped.
  This prevents warnings for events the component listens to but does not
  wait on (e.g. TakeAction on the Action bus).
- **timeout**: Watchdog interval in seconds. If all sources have not settled
  within this time after the first `notify()` call (or after a `reset()`),
  a warning is logged.

### `notify(source)`

The single entry point for handlers. Every handler calls `notify(name)` and
SettleGroup decides what to do:

1. If `source` is in **sources**: mark it as settled. If all sources are now
   settled, cancel watchdog, invoke `on_settled`, and reset.
2. If `source` is in **ignored**: no-op.
3. If `source` is in **neither**: log a warning:
   `"{name} saw '{source}' which is not in its settle watch list: {sources | ignored}"`.
   This catches cases where a new bus source is added but nobody updated the
   SettleGroup configuration.

### `reset()`

Clears all settled state and restarts the watchdog. Called automatically after
`on_settled` fires. Can also be called manually if a step is abandoned.

### `add_source(source)` / `remove_source(source)`

Modify the required sources set at runtime. Needed for VisionAttention where
the set of feature extractors is dynamic (loaded from config, tracked via
weakrefs). Adding a source also adds it to the pending set. Removing a source
discards it from pending and may trigger settling if it was the last pending
source.

### Watchdog

A `threading.Timer` (daemon thread) started on the first `notify()` call
after a reset. If `on_settled` has not fired within `timeout` seconds, logs:

    "{name} has not settled in {timeout}s, still waiting on: {pending}"

The watchdog does not raise or abort -- it is diagnostic only. It resets
along with the rest of the state when `on_settled` fires or `reset()` is
called.

### Thread Safety

All internal state is guarded by a `threading.Lock`. Bus listeners run on the
ThreadPoolScheduler and may call `notify()` concurrently from different worker
threads. RxPY's ThreadPoolScheduler does not serialize callbacks across
different subscriptions -- each subscription's handler runs independently on
whatever pool thread is available. A component like Sequencer that listens on
three buses can have three handlers running concurrently, all calling
`notify()` on the same SettleGroup instance.

Note: VisionAttention's current inline settled-set has a latent race condition
for the same reason (no lock on `self.settled`). SettleGroup fixes this.

## Usage

### VisionAttention (replaces inline settled-set)

```python
class VisionAttention(Component):
    def __init__(self):
        super().__init__()
        self.pb_conn = self.connect_bus(Perception.bus)
        self.pb_conn.listen(self.do_attention)
        self.att_conn = self.connect_bus(Attention.bus)
        self.saliency_map = SaliencyMap()

        self._settle = SettleGroup(
            name="VisionAttention",
            sources=set(FeatureExtractor.list()),
            on_settled=self._all_settled,
            ignored={"vision-data"},
        )

    def do_attention(self, e):
        if isinstance(e.data, VisionData):
            self.saliency_map.grid = IntGrid(e.data.chars)
            self._settle.notify("vision-data")
            return

        if isinstance(e.data, Settled):
            self._settle.notify(str(e.src_id))
            return

        assert isinstance(e.data, VisualFeature)
        for p in e.data.get_points():
            self.saliency_map.add_val(p[0], p[1], e.data)

    def _all_settled(self):
        self.att_conn.send(
            VisionAttentionData(
                focus_points=self.saliency_map.get_focus(),
                saliency_map=self.saliency_map,
            )
        )
        self.saliency_map = SaliencyMap()
```

Note: VisualFeature events are not passed through `notify()` because they
arrive many times per extractor per step -- they are data, not signals. Only
Settled and VisionData are signals with settle/ignore semantics.

### Sequencer (new settling behavior)

```python
class Sequencer(Component):
    def __init__(self):
        super().__init__()
        self.sequencer_conn = self.connect_bus(Sequencer.bus)
        self.obj_res_conn = self.connect_bus(ObjectResolver.bus)
        self.obj_res_conn.listen(self.handle_object_resolution_event)
        self.action_conn = self.connect_bus(Action.bus)
        self.action_conn.listen(self.handle_action_event)
        self.intrinsic_conn = self.connect_bus(Intrinsic.bus)
        self.intrinsic_conn.listen(self.handle_intrinsic_event)
        self.last_frame: Frame | None = None
        self.current_frame: Frame = Frame()

        self._settle = SettleGroup(
            name="Sequencer",
            sources={"object", "intrinsic"},
            on_settled=self._emit_frame,
            ignored={"action"},
        )

    def handle_object_resolution_event(self, e):
        FrameAttribute.connect(self.current_frame, e.data.feature_group)
        self._settle.notify("object")

    def handle_intrinsic_event(self, e):
        for node in e.data.to_nodes():
            FrameAttribute.connect(self.current_frame, node)
        self._settle.notify("intrinsic")

    def handle_action_event(self, e):
        if isinstance(e.data, TakeAction):
            self._settle.notify("action")
            # Attach to already-emitted frame retroactively.
            # Frame is a graph Node -- edges can be added anytime.
            if self.last_frame is not None:
                FrameAttribute.connect(self.last_frame, e.data)
                FrameAttribute.connect(e.data, self.current_frame)

    def _emit_frame(self):
        self.last_frame = self.current_frame
        self.current_frame = Frame()
        NextFrame.connect(self.last_frame, self.current_frame)
        self.sequencer_conn.send(self.last_frame)
```

### Resulting Pipeline Flow

Decoupling frame emission from TakeAction breaks the circular dependency:

```
Gym.send_obs() -> Perception -> Attention -> ObjectResolver -+
                                                              |
Gym.send_intrinsics() -> IntrinsicData ----------------------+
                                                              |
                                                    Sequencer settles
                                                              |
                                                         emit Frame
                                                              |
                                                        Transformer
                                                              |
                                                          Predict
                                                              |
                                    Action (waits for Predict + ActionRequest)
                                                              |
                                                         TakeAction
                                                              |
                                              Gym.get_action() returns
                                              Gym.env.step()
                                              Gym.emit_state_logs()
```

Action waits for both an ActionRequest (from Gym) and a PredictResult (from
Predict) before emitting TakeAction. Since get_action() blocks until
TakeAction is ready, the entire pipeline -- including all OTel logging from
ObjectResolver -- is guaranteed to complete before emit_state_logs()
increments the step counter. This fixes the off-by-one logging bug.

## Edge Cases

### First step (no previous frame)

Transformer returns early when there is no previous frame (no NextFrame edge).
It still needs to emit something so Predict fires and Action can proceed.
Options:

1. Transformer emits a sentinel TransformResult with an empty Transform.
2. Predict listens on Sequencer bus too and handles first-frame directly.
3. Action has a fallback: if no PredictResult arrives within a timeout, emit
   a default action.

Option 3 is the most robust -- it handles both the first-step case and any
future case where Predict has no candidates. Action's timeout should be
shorter than the SettleGroup watchdog (e.g. 2s vs 5s) so it fires before the
pipeline appears stalled.

### Dynamic feature extractors

VisionAttention's source set comes from `FeatureExtractor.list()`, which is
a weakref list that changes as extractors are created/destroyed. The
SettleGroup's `add_source()` / `remove_source()` methods support this. The
component is responsible for keeping the source set in sync (same as today).

### Multiple objects per step (future)

If ObjectResolver is changed to resolve multiple focus points per step, the
Sequencer would need to know how many ResolvedObjects to expect. Options:

1. ObjectResolver sends a Settled signal after all focus points are processed.
   Sequencer waits for this signal instead of the first ResolvedObject.
2. SettleGroup gains a `mark_count` mode: settle after N notifications instead
   of one.

Option 1 is simpler and matches the existing FeatureExtractor pattern.

### Watchdog fires during slow resolution

The 5-second timeout is diagnostic, not fatal. If Dirichlet resolution takes
a long time on a complex scene, the warning helps distinguish "slow" from
"hung." The timeout is configurable per SettleGroup instance.
