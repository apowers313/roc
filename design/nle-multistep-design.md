# NLE Multi-Step Observation Design

## Problem Statement

The NetHack Learning Environment (NLE) does not expose intermediate game
states when the player is helpless (fainted, paralyzed, sleeping, etc.).
When the agent is helpless, NetHack's internal game loop processes multiple
turns -- monsters move, attack, the player takes damage -- but NLE only
returns the final observation when the player regains control (or dies).

This means the ROC dashboard cannot show the frames where a grid bug is
attacking the fainted player, or where HP drops from 14 to 0 across
multiple monster attacks. The agent sees HP=14 on one step and HP=0 (death
screen) on the next, with no visibility into what happened in between.

### Concrete Example

Run `20260317084832-bladed-yasmin-shoifet`, game 2, steps 8752-8757:

| Step | HP | Message |
|------|-----|---------|
| 8752 | 14/14 | You faint from lack of food. You regain consciousness. |
| 8753 | 14/14 | You feel fragile! You haven't been watching your health. |
| 8756 | 14/14 | It's solid stone. |
| 8757 | 0/0 | (death screen: "Killed by a grid bug, while fainted from lack of food") |

The grid bug attacked multiple times during the faint, but none of those
frames are visible to the agent or the dashboard.

## Root Cause Analysis

### NetHack's `multi` Variable

NetHack uses the global variable `multi` to control helplessness. When
`multi < 0`, the player cannot act. The moveloop in `allmain.c` (lines
317-324) processes each helpless turn:

```c
if (multi < 0) {
    if (++multi == 0) {
        unmul((char *) 0);
        if (u.utotype)
            deferred_goto();
    }
}
```

During this loop, the game **never calls `rhack()`** (the player input
function). NLE's yield mechanism (`nle_yield` in `nle.c`) only fires when
`rhack()` calls `getch_method()` to request input. So when `multi < 0`,
the entire multi-turn skip happens inside a single `nle_step()` call with
no opportunity for the Python agent to observe intermediate states.

### All Situations That Cause Multi-Turn Skips

The `multi` variable is set negative by many game events:

| Situation | Duration (turns) | Worst Case |
|-----------|-----------------|------------|
| Sleep ray (wand/monster) | `d(nd,25)` scales with level | ~250 turns |
| Frozen by gaze (medusa etc.) | varies | 127 turns |
| Sleep trap | 1-25 | 25 |
| Sleep potion | 13-35 | 35 |
| Paralysis (monster attack) | 1-10 | 10 |
| Paralysis (monster casting) | 1-dmg | varies |
| Frozen by trap | 5-30 | 30 |
| Frozen by potion | 13-35 | 35 |
| Fainting from hunger | up to 10 | 10 |
| Fainting from rotten food | varies | varies |
| Gazing into mirror | varies by level | ~30 |
| Gazing into crystal ball | 1-10 | 10 |
| Vomiting | 2 | 2 |
| Stoning | 3 | 3 |
| Being frightened | 3 | 3 |
| Praying | 3 | 3 |
| Turning undead | 1-5 | 5 |
| Dragging iron ball | 2 | 2 |
| Stuck in spider web | 2-4 | 4 |
| Jumping | 1 | 1 |
| Taking off armor (stolen) | varies | varies |
| Opening containers | 1 | 1 |

Additionally, **occupation callbacks** (eating, reading spellbooks, opening
tins, putting on armor) also skip the yield via a similar mechanism in the
moveloop (lines 381-408 of `allmain.c`).

**Note:** `--More--` prompts during wand effects (bouncing lightning, etc.)
DO yield to the agent via `xwaitforspace()` -> `nhgetch()` -> `nle_yield()`.
The `rl_delay_output()` function is intentionally a no-op in the RL window
port, so visual animation effects are instantaneous.

### NLE's Yield Mechanism

NLE uses `fcontext` cooperative coroutines. The game runs on a separate
stack (`generatorcontext`). The yield cycle:

```
Python: nle_step(action)
  -> jump_fcontext into NH
  -> NH game loop runs
  -> rhack() -> getch_method() -> fill_obs() -> nle_yield(TRUE)
  -> jump_fcontext back to Python
  -> Python gets observation
```

When `multi < 0`, `rhack()` is never called, so the entire cycle above
never executes. The game loop runs internally until `multi` reaches 0.

## Design Decision

Implement a three-mode configuration option `helpless_obs` that controls
how NLE handles observations during helpless turns:

### Mode 0: `off` (default)

Current behavior. Helpless turns are invisible to the agent. Backwards
compatible, zero performance impact.

### Mode 1: `callback-and-ignore-actions`

Yield to the agent on every game turn during helplessness. The observation
includes a `passive` flag indicating the agent cannot act. Whatever action
the agent returns is ignored -- the game continues processing the next
helpless turn.

**Flow:**
```
Agent: step(action)
  -> NH processes action -> fainting starts (multi = -10)
  -> turn 1: monsters move, HP drops
  -> fill_obs(), set obs.passive=1 -> nle_yield() -> agent gets obs
Agent: step(anything)  <- action is IGNORED
  -> turn 2: monsters move
  -> fill_obs(), set obs.passive=1 -> nle_yield() -> agent gets obs
  ... repeat for each helpless turn ...
Agent: step(anything)  <- action is IGNORED
  -> multi reaches 0, player regains control
  -> rhack() called -> fill_obs(), set obs.passive=0 -> nle_yield()
Agent: step(real_action)  <- this action matters
```

**Pros:**
- Simple implementation (~40-50 lines of C)
- No buffer management or memory overhead
- Agent sees each intermediate frame in real-time
- Better for dashboard visualization (frames appear in step timeline)
- Reuses existing coroutine mechanism identically

**Cons:**
- Changes Gym API semantics (step() returns obs that aren't decision points)
- Each intermediate turn requires a Python<->C context switch (~microseconds, negligible)
- Step counter inflates (10-turn faint = 10 extra steps in trajectory)
- RL training loops need to handle passive flag (skip in replay buffer)

### Mode 2: `batch-multistep-screens`

Buffer all intermediate frames during helplessness, return them as an array
alongside the final observation.

**Flow:**
```
Agent: step(action)
  -> NH processes action -> fainting starts (multi = -10)
  -> turn 1: capture tty_chars/blstats/message to ring buffer
  -> turn 2: capture to ring buffer
  ... all helpless turns captured ...
  -> multi reaches 0, rhack() called
  -> fill_obs() with normal obs + intermediate_frames array
  -> nle_yield() -> agent gets obs with N buffered frames
```

**Pros:**
- Clean Gym API (one step = one observation, just with extra data)
- No extra context switches during helplessness
- Better for RL training (no passive frames to handle)

**Cons:**
- More complex implementation (~70-115 lines of C)
- Memory overhead: ~0.5-1 MB buffer always allocated when enabled
- Need to handle buffer overflow (worst case ~250 turns from sleep ray)
- Agent only sees intermediate frames after helplessness ends (post-hoc)

### Recommendation

Implement all three modes. Mode 1 is better for ROC's dashboard
visualization use case (frames appear in the step timeline naturally).
Mode 2 is better for RL training. Both are opt-in and backwards compatible.

## NLE Configuration System

### Current Architecture

Configuration flows from Python to C through this chain:

```
Python (NLE.__init__)
  -> nle/env/base.py accepts parameters
  -> nle/nethack/nethack.py Nethack class
  -> pybind11 C++ wrapper (pynethack.cc)
  -> nle_settings C struct (nleobs.h)
  -> nle_start() copies to global (nle.c)
  -> allmain.c reads global settings
```

### The `nle_settings` Struct (nleobs.h)

```c
typedef struct nle_settings {
    char hackdir[4096];
    char scoreprefix[4096];
    char options[32768];
    char wizkit[4096];
    int spawn_monsters;        // only existing int/bool option
    char ttyrecname[4096];
} nle_settings;
```

### Observation Key Pattern

Optional observations use a null-pointer pattern:

1. Python allocates numpy buffers only for requested observation keys
2. C++ wrapper passes NULL for unrequested observations
3. C code checks `if (obs->tty_chars)` before writing

This is the pattern we follow for the `intermediate_frames` buffer in
mode 2.

## Implementation Plan

### Files to Modify

#### C/C++ Layer

1. **`include/nleobs.h`**
   - Add `int helpless_obs;` to `nle_settings` struct (0/1/2)
   - Add `int passive;` to `nle_obs` struct (flag for mode 1)
   - For mode 2: define `nle_intermediate_frame` struct and add
     `nle_intermediate_frame *intermediate_frames;` +
     `int num_intermediate_frames;` to `nle_obs`

2. **`src/allmain.c`**
   - In the `multi < 0` block (lines 317-324):
     - Mode 1: call `fill_obs(); obs->passive = 1; nle_yield(TRUE);`
     - Mode 2: call `capture_intermediate_frame();`
   - In the occupation block (lines 381-408): same treatment
   - After `multi` reaches 0 or occupation ends: reset passive flag,
     set `num_intermediate_frames` for mode 2

3. **`src/nle.c`**
   - For mode 1: when returning from `nle_yield` during passive,
     skip reading the action (or read and discard it)
   - For mode 2: allocate/free intermediate frame buffer in
     `nle_start()` / cleanup
   - Add `capture_intermediate_frame()` function that snapshots
     current tty_chars, blstats, message into the ring buffer

4. **`win/rl/winrl.cc`**
   - For mode 1: add `yield_passive()` that calls `fill_obs()` +
     `nle_yield()` with passive flag set
   - For mode 2: add `capture_intermediate()` that copies current
     display state to the ring buffer without yielding

5. **`win/rl/pynethack.cc`**
   - Accept `helpless_obs` parameter in constructor
   - Store in `settings_.helpless_obs`
   - For mode 2: accept optional `intermediate_frames` buffer via
     `set_buffers()` (following existing pattern)

#### Python Layer

6. **`nle/nethack/nethack.py`**
   - Accept `helpless_obs=0` in `Nethack.__init__`
   - Pass through to C++ constructor
   - For mode 2: add `"intermediate_frames"` to `OBSERVATION_DESC`
     with shape `(MAX_INTERMEDIATE_FRAMES, 24, 80)` and dtype `uint8`
   - Add `"num_intermediate_frames"` scalar observation

7. **`nle/env/base.py`**
   - Accept `helpless_obs=0` in `NLE.__init__`
   - Pass through to `Nethack` class
   - Document the three modes in docstring

### Intermediate Frame Buffer (Mode 2)

```c
#define NLE_MAX_INTERMEDIATE_FRAMES 256

typedef struct nle_intermediate_frame {
    unsigned char tty_chars[NLE_TERM_LI * NLE_TERM_CO];  // 24*80 = 1920
    signed char tty_colors[NLE_TERM_LI * NLE_TERM_CO];   // 1920
    unsigned char message[NLE_MESSAGE_SIZE];               // 256
    long blstats[NLE_BLSTATS_SIZE];                       // 27*8 = 216
} nle_intermediate_frame;
// Per frame: ~4312 bytes
// Max buffer: 256 * 4312 = ~1.1 MB
```

256 frames handles the worst case (sleep ray `d(10,25)` = up to 250 turns).

### Testing Strategy

1. **Unit test for mode 1**: Create a scenario where the agent faints
   (eat rotten food or starve). Verify that:
   - Multiple `step()` calls return obs with `passive=True`
   - HP changes are visible in intermediate frames
   - The final frame has `passive=False`
   - Action values during passive frames are ignored

2. **Unit test for mode 2**: Same scenario, verify that:
   - Single `step()` call returns obs with `intermediate_frames` array
   - `num_intermediate_frames` matches the number of helpless turns
   - Each frame shows a valid game state with progressing time

3. **Backwards compatibility test**: Verify mode 0 (default) produces
   identical behavior to current NLE.

4. **Stress test**: Trigger worst-case sleep ray. Verify no buffer
   overflow, no crashes, correct frame count.

### Estimated Effort

| Component | Mode 1 | Mode 2 | Shared |
|-----------|--------|--------|--------|
| nleobs.h | 5 lines | 15 lines | -- |
| allmain.c | 15 lines | 20 lines | -- |
| nle.c | 10 lines | 15 lines | -- |
| winrl.cc | 15 lines | 30 lines | -- |
| pynethack.cc | -- | -- | 10 lines |
| nethack.py | -- | -- | 15 lines |
| base.py | -- | -- | 5 lines |
| Tests | 30 lines | 30 lines | 20 lines |
| **Total** | **~75 lines C** | **~110 lines C** | **~50 lines Python** |

### Risk Assessment

- **Mode 1 risk: LOW-MEDIUM.** Adds yield points in the game loop, but
  structurally identical to existing yields. Main risk is subtle state
  corruption if an occupation callback expects no yield mid-execution.
  Mitigated by testing each helplessness type.

- **Mode 2 risk: LOW.** Purely additive data capture. No game loop control
  flow changes. Only risk is memory overhead when enabled.

- **Backwards compatibility: GUARANTEED.** Default `helpless_obs=0` produces
  identical behavior. Both modes are opt-in.

## ROC Integration

Once NLE exposes intermediate frames, ROC needs minimal changes:

### Mode 1 Integration (Preferred for ROC)

The gymnasium.py main loop calls `env.step(action)` and processes the
observation. With mode 1:

- Each passive yield becomes a regular step in the ROC pipeline
- `StepData` gets populated normally (screen, metrics, etc.)
- The dashboard shows each intermediate frame in the step timeline
- The `passive` flag could be stored in StepData for the dashboard to
  display (e.g., dim overlay showing "HELPLESS" on the game screen)
- Action selection is skipped during passive frames (action is ignored
  by NLE anyway, but ROC should not run the full perception/attention/
  resolution/action pipeline for passive frames)

### Mode 2 Integration

- After each `env.step()`, check `obs["num_intermediate_frames"]`
- If > 0, iterate through `obs["intermediate_frames"]` and create
  StepData entries for each
- Dashboard shows them as sub-steps (e.g., step 8752.1, 8752.2, etc.)

### Dashboard Changes

- Add a visual indicator for passive/helpless frames (e.g., red border
  on game screen, "FAINTED" badge in pipeline status)
- The transport bar step counter should distinguish passive steps
  (mode 1) or show sub-steps (mode 2)

## NLE Repository Status

The official NLE repository (`facebookresearch/nle`) has been archived
(read-only) since May 2024. This change would need to be implemented in
a fork. ROC already uses a local NLE build at `/home/apowers/Projects/nle/`.

## Open Questions

1. Should occupation callbacks (eating, reading) also yield/capture
   intermediate frames? They are interruptible by monsters, so the agent
   might want to see the game state each turn during a long eat.

2. For mode 1, should the step counter in ROC treat passive frames as
   real steps or sub-steps? Real steps is simpler but inflates counts.

3. For mode 2, should the buffer include glyphs (dungeon map) in addition
   to tty_chars? Glyphs add 3318 bytes/frame but are more useful for RL.

4. Should the `passive` flag distinguish between helplessness types
   (fainted vs paralyzed vs sleeping)? The `multi_reason` string is
   available in NetHack's C code.

5. Is 256 frames sufficient for the mode 2 buffer? Sleep rays can
   theoretically exceed this with high-level monsters. Options: larger
   buffer, dynamic allocation, or truncation with a flag.
