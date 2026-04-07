"""Monotonic tick counter owned by the game loop.

The Clock represents the N-th observation cycle of the current ROC run.
The game loop calls ``Clock.set(loop_num + 1)`` at the start of each
observation cycle; downstream components (Sequencer, ObjectResolver,
saliency attenuation, ParquetExporter) read the current tick via
``Clock.get()``.

Why one clock: historically the Sequencer had its own module-level
``tick`` global (incremented via ``Frame.tick`` default_factory) and the
ParquetExporter had a separate ``_step_counter`` keyed off
``roc.screen`` events. These were supposed to mean the same thing but
drifted due to an eager Frame created in ``Sequencer.__init__``, a
Pydantic schema-introspection call that invoked the default_factory as
a side effect, and an "unclosed trailing frame" left over at game end.
Collapsing both into ``Clock`` eliminates that class of off-by-one bug
by making the game loop the single source of truth.

``Clock.get()`` is pure -- calling it has no side effects -- so it is
safe for ``Frame.tick`` to use it as a Pydantic ``default_factory``
without triggering counter advancement during schema introspection.
"""

from __future__ import annotations


class Clock:
    """Single source of truth for the current observation cycle number."""

    _tick: int = 0

    @classmethod
    def get(cls) -> int:
        """Return the current tick (N-th observation cycle, 1-indexed once set)."""
        return cls._tick

    @classmethod
    def set(cls, tick: int) -> None:
        """Set the current tick.

        Called by the game loop at the start of each observation cycle.
        ``tick`` should be monotonically increasing but the class does
        not enforce this -- tests may set arbitrary values.
        """
        cls._tick = tick

    @classmethod
    def reset(cls) -> None:
        """Reset the clock to 0. Called between runs and by test fixtures."""
        cls._tick = 0
