# mypy: disable-error-code="no-untyped-def"

"""Regression test: exactly attention_cycles VisionAttentionData events per step.

The settled check in VisionAttention._handle_settled is susceptible to a
TOCTOU race when multiple ThreadPoolScheduler threads deliver Settled events
concurrently.  Without the _settled_lock, N threads can all pass the
"all extractors settled" check before any thread clears the set, each
triggering a full round of attention cycles.  In production this manifested
as 12 cycles per step (3 threads x 4 configured cycles) instead of 4.
"""

from __future__ import annotations

from typing import Any

from helpers.nethack_screens import screens
from helpers.util import StubComponent

from roc.framework.component import Component
from roc.framework.config import Config
from roc.framework.event import Event
from roc.perception.base import VisionData
from roc.perception.feature_extractors.delta import Delta
from roc.pipeline.attention.attention import (
    AttentionSettled,
    VisionAttention,
    VisionAttentionData,
)


def _wire_pipeline(empty_components: None, cycles: int) -> StubComponent[Any, Any]:
    """Set up all perception components + VisionAttention and return a StubComponent."""
    Config.get().attention_cycles = cycles
    delta = Component.get("delta", "perception")
    assert isinstance(delta, Delta)
    attention = Component.get("vision", "attention")
    assert isinstance(attention, VisionAttention)
    for name in ("flood", "line", "motion", "single", "distance", "color", "shape"):
        assert Component.get(name, "perception") is not None
    return StubComponent(
        input_bus=delta.pb_conn.attached_bus,
        output_bus=attention.att_conn.attached_bus,
    )


class TestAttentionCycleCount:
    """Verify that each step produces exactly attention_cycles VisionAttentionData events."""

    def test_four_cycles_per_step(self, empty_components) -> None:
        """With attention_cycles=4, one screen must produce exactly 4 data + 1 settled."""
        s = _wire_pipeline(empty_components, cycles=4)

        s.input_conn.send(VisionData.from_dict(screens[0]))

        data_events = [
            c.args[0]
            for c in s.output.call_args_list
            if isinstance(c.args[0], Event) and isinstance(c.args[0].data, VisionAttentionData)
        ]
        settled_events = [
            c.args[0]
            for c in s.output.call_args_list
            if isinstance(c.args[0], Event) and isinstance(c.args[0].data, AttentionSettled)
        ]

        assert len(data_events) == 4, (
            f"Expected exactly 4 VisionAttentionData events, got {len(data_events)}. "
            f"If >4, the _settled_lock race condition has regressed."
        )
        assert len(settled_events) == 1

    def test_two_cycles_per_step(self, empty_components) -> None:
        """With attention_cycles=2, one screen must produce exactly 2 data + 1 settled."""
        s = _wire_pipeline(empty_components, cycles=2)

        s.input_conn.send(VisionData.from_dict(screens[0]))

        data_events = [
            c.args[0]
            for c in s.output.call_args_list
            if isinstance(c.args[0], Event) and isinstance(c.args[0].data, VisionAttentionData)
        ]
        settled_events = [
            c.args[0]
            for c in s.output.call_args_list
            if isinstance(c.args[0], Event) and isinstance(c.args[0].data, AttentionSettled)
        ]

        assert len(data_events) == 2
        assert len(settled_events) == 1

    def test_cycle_count_stable_across_steps(self, empty_components) -> None:
        """Multiple screens must each produce exactly attention_cycles events."""
        s = _wire_pipeline(empty_components, cycles=4)

        s.input_conn.send(VisionData.from_dict(screens[0]))
        s.input_conn.send(VisionData.from_dict(screens[1]))

        data_events = [
            c.args[0]
            for c in s.output.call_args_list
            if isinstance(c.args[0], Event) and isinstance(c.args[0].data, VisionAttentionData)
        ]
        settled_events = [
            c.args[0]
            for c in s.output.call_args_list
            if isinstance(c.args[0], Event) and isinstance(c.args[0].data, AttentionSettled)
        ]

        assert len(data_events) == 8, (
            f"Expected 8 VisionAttentionData events (4 per step x 2 steps), "
            f"got {len(data_events)}"
        )
        assert len(settled_events) == 2

    def test_cycle_count_stable_across_sequential_resets(self, empty_components) -> None:
        """Cycle count must not grow after Component.reset + EventBus.reset_all_buses.

        Regression: EventBus.reset_all_buses used vars() instead of walking
        the MRO, so buses defined on ABC parents (Perception.bus, Attention.bus)
        were never reset. Old subscriptions accumulated on the same Subject,
        causing N*4 cycles on the Nth game.
        """
        from roc.framework.event import EventBus

        for game_num in range(1, 4):
            s = _wire_pipeline(empty_components, cycles=4)

            s.input_conn.send(VisionData.from_dict(screens[0]))

            data_events = [
                c.args[0]
                for c in s.output.call_args_list
                if isinstance(c.args[0], Event)
                and isinstance(c.args[0].data, VisionAttentionData)
            ]

            assert len(data_events) == 4, (
                f"Game {game_num}: expected 4 VisionAttentionData, got {len(data_events)}. "
                f"Bus reset likely missed ABC-parent buses."
            )

            Component.reset()
            EventBus.reset_all_buses()
