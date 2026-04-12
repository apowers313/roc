"""Pydantic response types for the unified-run dashboard API.

These types are the public contract for ``RunReader`` and the REST endpoints
that wrap it. They explicitly distinguish ``ok`` from ``run_not_found``,
``out_of_range``, ``not_emitted``, and ``error`` so the dashboard never has
to guess why a step came back empty.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

HistoryKind = Literal[
    "graph",
    "event",
    "metrics",
    "intrinsics",
    "action",
    "resolution",
]


class RunSummary(BaseModel):
    """Summary metadata for a single run.

    The ``status`` field exists so the dashboard can SHOW runs that
    would otherwise be silently filtered, and explain why. The four
    states are:

    - ``ok``: catalog readable, has at least one game with step data.
    - ``empty``: catalog readable, but no games or steps were ever
      written. Usually means the game crashed before the first step,
      or that ``Observability.init()`` ran without a real game.
    - ``short``: catalog readable, has data, but step count is below
      the dropdown's "interesting run" threshold (default 10).
    - ``corrupt``: opening or querying the catalog raised an exception.
      ``error`` carries the exception message for diagnostics.

    Returning all four (instead of dropping non-ok ones) is what
    prevents the recurring "I made a run, where is it in the dropdown?"
    bug class. The frontend filters by status with a visible toggle.
    """

    name: str
    games: int
    steps: int
    status: str = "ok"
    error: str | None = None


class StepRange(BaseModel):
    """Step range for a run, with the ``tail_growing`` liveness signal.

    ``tail_growing`` is the only signal of liveness at the API boundary.
    Phase 1 wires the field through with a default of ``False``; Phase 3
    flips it on whenever a ``RunWriter`` is attached to the run.
    """

    min: int
    max: int
    tail_growing: bool = False


StepResponseStatus = Literal[
    "ok",
    "run_not_found",
    "out_of_range",
    "not_emitted",
    "error",
]


class StepResponse(BaseModel):
    """Typed envelope for a single-step lookup.

    The dashboard uses ``status`` to render the right message instead of
    silently dropping the step. ``data`` is populated only when ``status
    == "ok"``; ``error`` is populated only when ``status == "error"``.
    """

    status: StepResponseStatus
    data: dict[str, Any] | None = None
    range: StepRange | None = None
    error: str | None = None
